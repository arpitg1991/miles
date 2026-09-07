"""Multi-segment episodes behind ``--arena-train-segments`` (ADR-0011).

The Harbor gym may ship a trajectory whose ``steps`` holds several
self-contained compaction segments (AREnATasks ADR-0048): archived segments
carry ``segment_end`` and no ``stop_reason``; the final segment is last. The
default ``final`` mode trains ``steps[-1]`` only (byte-identical to today).
``all`` trains one Sample per segment; every segment of an episode shares one
``rollout_id`` (= today's ``index``) and the episode reward, ``index`` stays
unique, truncation stays an episode property, and telemetry counts episodes.

Run: python -m pytest tests/fast/plugins/arena/test_multi_segment_episodes.py -v
"""

from __future__ import annotations

import logging
import queue
import types as pytypes

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.fast.plugins.arena.test_group_identity import (
    _TRAIN_PARALLEL_CONFIG,
    _expected_group_normalized,
    _make_args,
    _StubMaskGenerator,
)

import miles.utils.mask_utils as mask_utils
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.utils.types import Sample
from miles_plugins.arena.nats_arena import nats_rollout
from miles_plugins.arena.nats_arena.nats_rollout import (
    _SEGMENT_INDEX_STRIDE,
    NATSRolloutWorker,
    _batch_telemetry,
    _result_to_episodes_full_trajectory,
    _result_to_samples_full_trajectory,
    _training_steps,
    generate_rollout,
)

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

N = 4  # n_samples_per_prompt
_LOGGER = "miles_plugins.arena.nats_arena.nats_rollout"

# Cumulative per-segment loss masks. Response window = from the first 1 on.
_SEG_A = [0, 0, 1, 0, 1, 1]  # window [1, 0, 1, 1]
_SEG_B = [0, 1, 1, 1]  # window [1, 1, 1]
_FINAL = [0, 0, 0, 1, 1]  # window [1, 1]
# Final segment with an earlier clean turn and a clipped trailing run.
_FINAL_MULTI_TURN = [0, 1, 1, 0, 1, 1, 1]  # window [1, 1, 0, 1, 1, 1]


# ---------------------------------------------------------------------------
# Helpers — wire-shaped dicts in the style of test_mask_clipped_final_turn.py
# ---------------------------------------------------------------------------


def _arrays(loss_mask: list[int], tag: int) -> dict:
    # token_ids are tagged per step so a test can tell segments apart;
    # log_probs aligned 1:1 with token_ids keeps the rollout-logprob check green.
    return {
        "token_ids": [tag] * len(loss_mask),
        "loss_mask": list(loss_mask),
        "log_probs": [-0.1] * len(loss_mask),
        "has_generate_tokens": True,
    }


def _segment(loss_mask: list[int], end: str = "compaction", tag: int = 1) -> dict:
    """Archived segment: ``segment_end`` marker, NO stop_reason (ADR-0048)."""
    return {**_arrays(loss_mask, tag), "truncated_generates": 0, "segment_end": end}


def _step(loss_mask: list[int], stop_reason: str = "stop", tag: int = 9) -> dict:
    """Final (top-level) segment: carries the episode's stop_reason."""
    return {**_arrays(loss_mask, tag), "stop_reason": stop_reason}


def _traj(steps: list[dict], reward: float = 1.0, **extra) -> dict:
    return {"reward": reward, "steps": steps, "weight_versions": [3], **extra}


def _result(trajectories: list[dict], task_id: str = "t1.g0.") -> dict:
    return {
        "task_id": task_id,
        "gym_name": "g",
        "status": "success",
        "trajectories": trajectories,
        "group_metrics": {"num_turns_mean": 2.0},
    }


def _three_segments(stop_reason: str = "stop", final: list[int] | None = None) -> list[dict]:
    return [
        _segment(_SEG_A, tag=1),
        _segment(_SEG_B, tag=2),
        _step(final or _FINAL, stop_reason=stop_reason, tag=9),
    ]


def _args(mode: str = "final", **overrides) -> pytypes.SimpleNamespace:
    fields = {
        "arena_train_segments": mode,
        "arena_mask_clipped_final_turn": False,
        "arena_keep_timeout_trajectories": False,
        "arena_keep_context_error_trajectories": False,
    }
    fields.update(overrides)
    return pytypes.SimpleNamespace(**fields)


def _make_worker(args) -> NATSRolloutWorker:
    """Bare worker — no NATS/thread machinery, just what _process_group reads."""
    worker = object.__new__(NATSRolloutWorker)
    worker.args = args
    worker.n_per_prompt = args.n_samples_per_prompt
    worker.output_queue = queue.Queue()
    worker._output_group_counter = 0
    worker._tokenizer = None  # the fast path never touches it
    worker._train_segments = args.arena_train_segments
    return worker


def _drain(worker: NATSRolloutWorker) -> list[list[Sample]]:
    groups = []
    while True:
        try:
            groups.append(worker.output_queue.get_nowait())
        except queue.Empty:
            break
    return groups


def _episodes_with_segments(counts: list[int], rewards: list[float] | None = None) -> list[dict]:
    """One trajectory per count: ``k`` segments = ``k - 1`` archived + the final."""
    rewards = rewards or [1.0] * len(counts)
    trajs = []
    for k, reward in zip(counts, rewards, strict=True):
        steps = [_segment(_SEG_A, tag=10 + i) for i in range(k - 1)] + [_step(_FINAL, tag=9)]
        trajs.append(_traj(steps, reward=reward))
    return trajs


# ===========================================================================
# 1. Step selection and per-segment conversion
# ===========================================================================


def test_default_final_mode_reads_last_step_only():
    episodes = _result_to_episodes_full_trajectory(
        _result([_traj(_three_segments())]), tokenizer=None, args=_args("final")
    )
    assert len(episodes) == 1
    (s,) = episodes[0]
    assert s.tokens == [9] * len(_FINAL)
    assert s.loss_mask == [1, 1]
    assert s.metadata["segment"] == 0
    assert s.metadata["n_segments"] == 1
    assert s.metadata["group_metrics"] == {"num_turns_mean": 2.0}
    # The flat wrapper returns the same single sample.
    assert (
        len(
            _result_to_samples_full_trajectory(
                _result([_traj(_three_segments())]), tokenizer=None, args=_args("final")
            )
        )
        == 1
    )


def test_all_mode_one_sample_per_segment_shared_reward():
    episodes = _result_to_episodes_full_trajectory(
        _result([_traj(_three_segments())]), tokenizer=None, args=_args("all")
    )
    assert len(episodes) == 1
    a, b, f = episodes[0]
    assert [s.reward for s in (a, b, f)] == [1.0, 1.0, 1.0]
    assert [s.metadata["segment"] for s in (a, b, f)] == [0, 1, 2]
    assert all(s.metadata["n_segments"] == 3 for s in (a, b, f))
    # Each response window comes from its OWN cumulative loss_mask.
    assert a.tokens == [1] * len(_SEG_A) and a.loss_mask == [1, 0, 1, 1] and a.response_length == 4
    assert b.tokens == [2] * len(_SEG_B) and b.loss_mask == [1, 1, 1] and b.response_length == 3
    assert f.tokens == [9] * len(_FINAL) and f.loss_mask == [1, 1] and f.response_length == 2
    assert all(len(s.rollout_log_probs) == s.response_length for s in (a, b, f))
    assert all(s.status == Sample.Status.COMPLETED for s in (a, b, f))
    assert all(not s.remove_sample for s in (a, b, f))
    # Trajectory-level fields land on every segment; group_metrics on the first only.
    assert all(s.weight_versions == ["3"] for s in (a, b, f))
    assert "group_metrics" in a.metadata
    assert "group_metrics" not in b.metadata and "group_metrics" not in f.metadata


def test_all_mode_without_marker_keeps_last_step():
    # Inspect sglang_perstep shape: cumulative steps, has_generate_tokens on
    # every step, no segment_end marker -> steps[-1] path in BOTH modes.
    steps = [_step(_SEG_A, tag=1), _step(_SEG_B, tag=2), _step(_FINAL, tag=9)]
    for step in steps:
        assert "segment_end" not in step
    assert _training_steps(steps, _args("all")) == steps[-1:]
    assert _training_steps(steps, _args("final")) == steps[-1:]
    assert _training_steps([], _args("all")) == []
    (s,) = _result_to_samples_full_trajectory(_result([_traj(steps)]), tokenizer=None, args=_args("all"))
    assert s.tokens == [9] * len(_FINAL)


def test_closed_segment_has_no_stop_reason_so_no_truncation():
    steps = _three_segments(stop_reason="stop")
    assert all("stop_reason" not in st for st in steps[:-1])
    samples = _result_to_samples_full_trajectory(_result([_traj(steps)]), tokenizer=None, args=_args("all"))
    assert [s.status for s in samples] == [Sample.Status.COMPLETED] * 3
    assert all(not s.remove_sample for s in samples)


def test_final_length_truncates_every_segment_flag_off():
    samples = _result_to_samples_full_trajectory(
        _result([_traj(_three_segments(stop_reason="length"))]),
        tokenizer=None,
        args=_args("all", arena_mask_clipped_final_turn=False),
    )
    assert len(samples) == 3
    assert all(s.status == Sample.Status.TRUNCATED for s in samples)
    assert all(s.remove_sample is True for s in samples)
    assert all(s.metadata["removal_reason"] == "length" for s in samples)


def test_final_length_mask_flag_salvages_earlier_segments():
    a, b, f = _result_to_samples_full_trajectory(
        _result([_traj(_three_segments(stop_reason="length", final=_FINAL_MULTI_TURN))]),
        tokenizer=None,
        args=_args("all", arena_mask_clipped_final_turn=True),
    )
    # Truncation is an episode property: status TRUNCATED everywhere ...
    assert all(s.status == Sample.Status.TRUNCATED for s in (a, b, f))
    # ... but only the FINAL segment's trailing run is zeroed.
    assert f.loss_mask == [1, 1, 0, 0, 0, 0]
    assert f.response_length == len(f.loss_mask) == len(f.rollout_log_probs)
    assert a.loss_mask == [1, 0, 1, 1]
    assert b.loss_mask == [1, 1, 1]
    assert all(not s.remove_sample for s in (a, b, f))
    assert all("removal_reason" not in s.metadata for s in (a, b, f))


def test_context_error_removes_all_segments_unless_kept():
    result = _result([_traj(_three_segments(), agent_stop_reason="context_error")])
    removed = _result_to_samples_full_trajectory(result, tokenizer=None, args=_args("all"))
    assert len(removed) == 3
    assert all(s.status == Sample.Status.TRUNCATED for s in removed)
    assert all(s.remove_sample is True for s in removed)
    assert all(s.metadata["removal_reason"] == "context_error" for s in removed)

    kept = _result_to_samples_full_trajectory(
        result, tokenizer=None, args=_args("all", arena_keep_context_error_trajectories=True)
    )
    assert all(s.status == Sample.Status.TRUNCATED for s in kept)
    assert all(not s.remove_sample for s in kept)
    assert all(s.metadata["kept_context_error"] is True for s in kept)
    assert all(s.metadata["agent_stop_reason"] == "context_error" for s in kept)
    assert all("removal_reason" not in s.metadata for s in kept)


def test_timeout_salvage_keeps_every_segment():
    result = _result([_traj(_three_segments(), agent_stop_reason="timeout")])
    removed = _result_to_samples_full_trajectory(result, tokenizer=None, args=_args("all"))
    assert all(s.remove_sample is True for s in removed)
    assert all(s.metadata["removal_reason"] == "timeout" for s in removed)

    kept = _result_to_samples_full_trajectory(
        result, tokenizer=None, args=_args("all", arena_keep_timeout_trajectories=True)
    )
    assert len(kept) == 3
    assert all(s.status == Sample.Status.TRUNCATED for s in kept)
    assert all(not s.remove_sample for s in kept)
    assert all(s.metadata["kept_timeout"] is True for s in kept)
    assert [s.loss_mask for s in kept] == [[1, 0, 1, 1], [1, 1, 1], [1, 1]]


def test_slow_path_sample_is_segment_zero_of_one(monkeypatch):
    # A messages-only trajectory carries the same segment stamp as a one-step
    # fast-path episode, so a consumer can index metadata["segment"] everywhere.
    monkeypatch.setattr(mask_utils, "MultiTurnLossMaskGenerator", _StubMaskGenerator)
    (s,) = _result_to_samples_full_trajectory(
        _result([{"reward": 1.0, "messages": [{"role": "user", "content": "x"}]}]),
        tokenizer=None,
        args=_args("all"),
    )
    assert s.response_length == 6  # stub mask: 10 tokens, 6-token response
    assert (s.metadata["segment"], s.metadata["n_segments"]) == (0, 1)


# ===========================================================================
# 2. _process_group: pad / trim / stamp per EPISODE
# ===========================================================================


def test_process_group_pads_and_trims_by_episode():
    worker = _make_worker(_make_args(arena_train_segments="all"))
    # 3 real episodes of 1, 2, 3 segments; n_per_prompt=4 -> one pad episode.
    worker._process_group(
        "t.g0.",
        [_result(_episodes_with_segments([1, 2, 3]))],
        instance_id="t",
        dedup_key="t#e0",
    )
    (flat,) = _drain(worker)
    assert len(flat) == 1 + 2 + 3 + 1
    pad = flat[-1]
    src = flat[0]  # episodes[0][-1]: the first episode is one final segment
    assert pad.status == Sample.Status.FAILED
    assert pad.remove_sample is True
    assert pad.reward == 0.0
    assert pad.tokens == src.tokens
    assert pad.response_length == src.response_length
    assert pad.loss_mask == [0] * src.response_length
    assert pad.rollout_log_probs == [0.0] * pad.response_length
    assert pad.metadata["mode"] == "failed"
    assert (pad.metadata["segment"], pad.metadata["n_segments"]) == (0, 1)
    assert pad.metadata["instance_id"] == "t" and pad.metadata["dedup_key"] == "t#e0"
    assert all(s.metadata["instance_id"] == "t" for s in flat)

    # 5 one-segment episodes -> trimmed to n_per_prompt episodes.
    worker._process_group("t.g1.", [_result(_episodes_with_segments([1] * 5))])
    (flat,) = _drain(worker)
    assert len(flat) == N


def test_process_group_stamps_shared_rollout_id_and_unique_index_all_mode():
    worker = _make_worker(_make_args(arena_train_segments="all"))
    for task in ("a.g0.", "b.g1."):
        worker._process_group(task, [_result(_episodes_with_segments([1, 2, 3]), task_id=task)])
    groups = _drain(worker)
    assert len(groups) == 2
    all_indices: list[int] = []
    for gid, flat in enumerate(groups):
        assert all(s.group_index == gid for s in flat)
        # Walk episodes by (segment k restarts at 0) and check the stamp.
        e = -1
        for s in flat:
            k = s.metadata["segment"]
            if k == 0:
                e += 1
            base = gid * N + e
            assert s.rollout_id == base, (s.rollout_id, base, k)
            assert s.index == base + k * _SEGMENT_INDEX_STRIDE
            all_indices.append(s.index)
        assert e == N - 1  # 3 real episodes + 1 pad
        assert sorted({s.rollout_id for s in flat}) == [gid * N + e for e in range(N)]
    assert len(set(all_indices)) == len(all_indices)
    # int64-safe: the largest stamped index stays far below 2**63.
    assert max(all_indices) < 2**63


def test_process_group_final_mode_matches_adr_0003():
    """Mirror test_group_identity: rollout_id None, index == gid*n+i."""
    worker = _make_worker(_make_args(arena_train_segments="final"))
    for task in ("a.g0.", "b.g1."):
        worker._process_group(task, [_result(_episodes_with_segments([1, 2, 3]), task_id=task)])
    group_a, group_b = _drain(worker)
    for gid, group in enumerate((group_a, group_b)):
        assert len(group) == N  # one sample per episode; the v2 steps collapse to steps[-1]
        assert [s.group_index for s in group] == [gid] * N
        assert [s.index for s in group] == [gid * N + i for i in range(N)]
        assert all(s.rollout_id is None for s in group)
        assert all(not hasattr(s, "group_id") for s in group)


# ===========================================================================
# 3. Real miles conversion path in ``all`` mode
# ===========================================================================


def test_grpo_normalizes_per_episode_and_mask_sums_span_segments():
    args = _make_args(rollout_batch_size=1, global_batch_size=4, arena_train_segments="all")
    worker = _make_worker(args)
    counts = [1, 2, 1, 3]
    rewards = [1.0, 0.0, 1.0, 0.0]
    worker._process_group("t.g0.", [_result(_episodes_with_segments(counts, rewards))])
    (flat,) = _drain(worker)
    assert len(flat) == sum(counts)

    data, metadata = postprocess_rollout_data(args, [flat], _TRAIN_PARALLEL_CONFIG)
    # Compact mode (rollout_id set): no sample-count trim to global_batch_size.
    assert len(data) == sum(counts)
    train_data = convert_samples_to_train_data(args, data, metadata, None, None)

    # One GRPO entry per EPISODE, broadcast to its segments.
    per_episode = _expected_group_normalized(rewards)
    expected = [adv for adv, k in zip(per_episode, counts, strict=True) for _ in range(k)]
    assert train_data["rewards"] == pytest.approx(expected)
    assert train_data["raw_reward"] == [r for r, k in zip(rewards, counts, strict=True) for _ in range(k)]

    rollout_ids = train_data["rollout_ids"]
    assert rollout_ids == [e for e, k in enumerate(counts) for _ in range(k)]
    assert len(set(rollout_ids)) == 4
    assert train_data["sample_indices"] == [
        e + k * _SEGMENT_INDEX_STRIDE for e, n in enumerate(counts) for k in range(n)
    ]

    # rollout_mask_sums: equal across an episode's segments, equal to the sum of its masks.
    sums_by_episode: dict[int, int] = {}
    for rid, mask in zip(rollout_ids, train_data["loss_masks"], strict=True):
        sums_by_episode[rid] = sums_by_episode.get(rid, 0) + sum(mask)
    assert train_data["rollout_mask_sums"] == [sums_by_episode[rid] for rid in rollout_ids]
    assert all(v > 0 for v in sums_by_episode.values())
    # The 3-segment episode's shared sum spans all three segments.
    assert sums_by_episode[3] == 3 + 3 + 2  # two _SEG_A windows (3 ones each) + _FINAL (2 ones)

    assert all(
        len(lp) == rl for lp, rl in zip(train_data["rollout_log_probs"], train_data["response_lengths"], strict=True)
    )


@pytest.mark.parametrize(
    ("final_step", "mask_flag", "reason", "truncated_ratio"),
    [
        # The final segment is ONE clipped turn: nothing to salvage there, so it
        # is removed alone while the earlier segments keep their masks.
        (_step(_FINAL, stop_reason="length", tag=9), True, "length", 1.0),
        # A rejected final generate: log_probs short on the final segment only.
        ({**_step(_FINAL, tag=9), "log_probs": [-0.1] * (len(_FINAL) - 1)}, False, "bad_logprobs", 0.0),
    ],
    ids=["clipped_only_final", "bad_logprobs_final"],
)
def test_per_segment_removal_counts_the_episode_once(final_step, mask_flag, reason, truncated_ratio):
    telemetry = {}
    for mode in ("all", "final"):
        worker = _make_worker(
            _make_args(n_samples_per_prompt=1, arena_train_segments=mode, arena_mask_clipped_final_turn=mask_flag)
        )
        steps = [_segment(_SEG_A, tag=1), _segment(_SEG_B, tag=2), dict(final_step)]
        worker._process_group("t.g0.", [_result([_traj(steps)])])
        (flat,) = _drain(worker)
        telemetry[mode] = _batch_telemetry([flat], [flat])
        if mode == "all":
            a, b, f = flat
            assert [s.remove_sample for s in (a, b, f)] == [False, False, True]
            assert f.metadata["removal_reason"] == reason
            assert all("removal_reason" not in s.metadata for s in (a, b))
            assert a.loss_mask == [1, 0, 1, 1] and b.loss_mask == [1, 1, 1]
    # The episode is removed once, exactly like its single sample under ``final``.
    assert telemetry["all"]["removed_frac"] == telemetry["final"]["removed_frac"] == 1.0
    assert telemetry["all"]["truncated_ratio"] == telemetry["final"]["truncated_ratio"] == truncated_ratio
    assert telemetry["all"]["total_episodes"] == telemetry["final"]["total_episodes"] == 1


# ===========================================================================
# 4. generate_rollout: partial-batch guard and per-episode telemetry
# ===========================================================================


def _sample(
    *,
    group_index: int,
    index: int,
    rollout_id: int | None,
    reward: float,
    response_length: int,
    status=Sample.Status.COMPLETED,
    remove: bool = False,
    mode: str = "full_trajectory",
    **meta,
) -> Sample:
    s = Sample()
    s.group_index = group_index
    s.index = index
    s.rollout_id = rollout_id
    s.reward = reward
    s.response_length = response_length
    s.status = status
    s.remove_sample = remove
    s.tokens = [1] * (response_length + 1)
    # A None-valued key is absent, like a segment that carries no removal_reason.
    s.metadata = {"mode": mode, "task_id": f"t{group_index}", "gym_name": "g"}
    s.metadata.update({k: v for k, v in meta.items() if v is not None})
    s.weight_versions = ["3"]
    return s


def _segments(rollout_id: int, n: int, **kwargs) -> list[Sample]:
    """``n`` segments of one episode; a list-valued kwarg gives one value per segment."""
    per_segment = {k: v for k, v in kwargs.items() if isinstance(v, list)}
    shared = {k: v for k, v in kwargs.items() if k not in per_segment}
    return [
        _sample(
            group_index=rollout_id,
            index=rollout_id + k * _SEGMENT_INDEX_STRIDE,
            rollout_id=rollout_id,
            reward=1.0,
            response_length=2,
            **shared,
            **{key: values[k] for key, values in per_segment.items()},
        )
        for k in range(n)
    ]


class _FakeWorker:
    """What generate_rollout reads from the global worker, without NATS."""

    def __init__(self, groups: list[list[Sample]], mode: str):
        self.output_queue: queue.Queue = queue.Queue()
        for g in groups:
            self.output_queue.put(g)
        self._train_segments = mode
        self.worker_thread = pytypes.SimpleNamespace(is_alive=lambda: False)

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()


def _rollout_args(mode: str, gbs: int = 4) -> pytypes.SimpleNamespace:
    return pytypes.SimpleNamespace(
        rollout_global_dataset=True,
        global_batch_size=gbs,
        n_samples_per_prompt=1,
        dynamic_sampling_filter_path=None,
        use_wandb=False,
        arena_train_segments=mode,
    )


def test_partial_batch_guard_raises_in_all_mode(monkeypatch):
    # 3 one-episode groups but global_batch_size 4; the worker thread is dead
    # so collection returns a partial batch.
    groups = [[_sample(group_index=i, index=i, rollout_id=i, reward=1.0, response_length=2)] for i in range(3)]
    monkeypatch.setattr(nats_rollout, "get_global_worker", lambda args, ds: _FakeWorker(groups, "all"))
    with pytest.raises(RuntimeError, match="global_batch_size"):
        generate_rollout(_rollout_args("all"), 0, data_source=pytypes.SimpleNamespace())

    # ``final`` keeps today's behaviour: the partial batch is returned and the
    # downstream trim/schedule raises instead.
    groups = [[_sample(group_index=i, index=i, rollout_id=None, reward=1.0, response_length=2)] for i in range(3)]
    monkeypatch.setattr(nats_rollout, "get_global_worker", lambda args, ds: _FakeWorker(groups, "final"))
    data = generate_rollout(_rollout_args("final"), 0, data_source=pytypes.SimpleNamespace())
    assert len(data) == 3


def test_generate_rollout_removal_breakdown_is_per_episode(monkeypatch, caplog):
    truncated = Sample.Status.TRUNCATED
    groups = [
        # Whole episode removed (flag off): "length" on both segments.
        _segments(0, 2, status=truncated, remove=True, removal_reason="length"),
        # Only the FINAL segment removed (one clipped turn under the mask flag).
        _segments(1, 3, status=truncated, remove=[False, False, True], removal_reason=[None, None, "length"]),
        # Gym-side pad.
        [_sample(group_index=2, index=2, rollout_id=2, reward=0.0, response_length=2, remove=True, mode="failed")],
        # Salvaged timeout episode: kept_timeout on every segment, none removed.
        _segments(3, 2, status=truncated, kept_timeout=True),
    ]
    monkeypatch.setattr(nats_rollout, "get_global_worker", lambda args, ds: _FakeWorker(groups, "all"))
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        data = generate_rollout(_rollout_args("all"), 0, data_source=pytypes.SimpleNamespace())
    assert len(data) == 4
    lines = [r.getMessage() for r in caplog.records]
    assert "Failed-sample distribution: failed=1/4 (0.250), removed_total=3/4 (0.750)" in lines
    assert "Removal reasons: length=2, failed=1 (kept_timeout=1, kept_context_error=0)" in lines
    assert "Segments: 8 samples over 4 episodes (compaction_segments_mean=2.00)" in lines


def test_telemetry_counts_episodes_not_segments():
    # One TRUNCATED episode with reward 1.0 split into two segments (rollout_id 0)
    # versus the same episode as a single sample.
    seg0 = _sample(group_index=0, index=0, rollout_id=0, reward=1.0, response_length=4, status=Sample.Status.TRUNCATED)
    seg1 = _sample(
        group_index=0,
        index=_SEGMENT_INDEX_STRIDE,
        rollout_id=0,
        reward=1.0,
        response_length=3,
        status=Sample.Status.TRUNCATED,
    )
    single = _sample(
        group_index=0, index=0, rollout_id=None, reward=1.0, response_length=7, status=Sample.Status.TRUNCATED
    )
    split = _batch_telemetry([[seg0, seg1]], [[seg0, seg1]])
    whole = _batch_telemetry([[single]], [[single]])
    for key in (
        "total_episodes",
        "avg_reward",
        "truncated_ratio",
        "reward_nonzero_frac",
        "avg_response_length",
        "failed_frac",
        "removed_frac",
        "nonzero_count",
    ):
        assert split[key] == whole[key], key
    assert split["total_episodes"] == 1
    assert split["avg_response_length"] == 7  # summed over the episode's segments
    assert split["compaction_segments_mean"] == 2.0
    assert whole["compaction_segments_mean"] == 1.0

    # Mixed group: a 2-segment reward-1 episode and a 1-segment reward-0 pad
    # episode -> avg_reward 0.5 (per episode), NOT 2/3 (per sample).
    pad = _sample(
        group_index=0,
        index=1,
        rollout_id=1,
        reward=0.0,
        response_length=4,
        status=Sample.Status.FAILED,
        remove=True,
        mode="failed",
    )
    mixed = _batch_telemetry([[seg0, seg1, pad]], [[seg0, seg1, pad]])
    assert mixed["total_episodes"] == 2 and mixed["total_samples"] == 3
    assert mixed["avg_reward"] == pytest.approx(0.5)
    assert mixed["reward_nonzero_frac"] == pytest.approx(0.5)
    assert mixed["failed_frac"] == pytest.approx(0.5)
    assert mixed["removed_frac"] == pytest.approx(0.5)
    assert mixed["truncated_ratio"] == pytest.approx(0.5)
    assert mixed["compaction_segments_mean"] == pytest.approx(1.5)
    assert mixed["mean_group_std"] == pytest.approx(0.5)  # std over episode rewards [1, 0]
