"""Regression tests for the port's one deliberate semantic adaptation:
group-identity stamping in ``NATSRolloutWorker._process_group``.

The AGISlime original stamped a shared ``Sample.group_id`` per prompt-group;
miles has no ``group_id`` and its ``rollout_id`` means something different
(compact siblings of ONE rollout execution that must share one reward). The
port therefore stamps a shared ``group_index`` (keys miles' GRPO reward
normalization segments), a unique ``index == gid * n_samples_per_prompt + i``
(int64-packable, per-trajectory loss denominators), and deliberately leaves
``rollout_id`` None. ADR-0011 amends this for ``--arena-train-segments all``:
the EPISODE id (never the group id) is stamped on ``rollout_id`` so compaction
segments of one episode share one reward; the default ``final`` mode keeps the
assertions below byte-identical. A regression to ``rollout_id=gid`` would
crash the 27B job at step 1 with "all samples in rollout N must share one
reward"; a dropped ``index`` stamp would crash int64 packing — both reproduced
during verification, neither caught by the wire-format suite.

Also covers the slow-path (messages-only) rollout_log_probs zero-fill under
``use_rollout_logprobs``: a group mixing fast- and slow-path trajectories must
not emit a None row into train_data["rollout_log_probs"].

These tests exercise the real miles conversion path (torch + miles imports),
unlike the import-light wire-format tests in test_nats_arena.py.

Run: python -m pytest tests/fast/plugins/arena/test_group_identity.py -v
"""

from __future__ import annotations

import queue
import types as pytypes

import pytest
import torch

import miles.utils.mask_utils as mask_utils
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.utils.types import Sample
from miles_plugins.arena.nats_arena.nats_rollout import NATSRolloutWorker
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

N_SAMPLES_PER_PROMPT = 4

_TRAIN_PARALLEL_CONFIG = {
    "dp_size": 1,
    "cp_size": 1,
    "vpp_size": 1,
    "microbatch_group_size_per_vp_stage": 1,
}


# ---------------------------------------------------------------------------
# Helpers — smoke-shaped args (grpo, rewards_normalization on) and
# wire-contract-shaped results, mirroring the adversarial-verification repro.
# ---------------------------------------------------------------------------


def _make_args(**overrides) -> pytypes.SimpleNamespace:
    args = pytypes.SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        rollout_batch_size=2,
        global_batch_size=8,
        reward_key=None,
        use_rollout_logprobs=True,
        disable_rollout_trim_samples=False,
        use_dynamic_global_batch_size=False,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=65536,
        micro_batch_size=None,
        balance_data=True,
        loss_mask_type="qwen3_5",
        rollout_max_context_len=131072,
        sglang_context_length=131072,
        lora_configs=None,
        multi_lora_n_adapters=0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _fast_traj(reward: float, tok_n: int = 12, prompt_n: int = 4, synthetic: bool = False) -> dict:
    """Fast-path trajectory: GenerateClient token-level data (has_generate_tokens)."""
    if synthetic:
        return {"synthetic": True, "stop_reason": "errored", "messages": [], "steps": []}
    return {
        "reward": reward,
        "messages": [{"role": "user", "content": "x"}],
        "steps": [
            {
                "has_generate_tokens": True,
                "token_ids": list(range(tok_n)),
                "loss_mask": [0] * prompt_n + [1] * (tok_n - prompt_n),
                "log_probs": [-0.1] * tok_n,
                "stop_reason": "stop",
                "weight_version": 3,
            }
        ],
        "agent_stop_reason": "completed",
        "weight_versions": [3],
    }


def _slow_traj(reward: float) -> dict:
    """Slow-path trajectory: messages only, no token-level data (re-tokenize path)."""
    return {
        "reward": reward,
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ],
        "steps": [{"stop_reason": "stop"}],
        "agent_stop_reason": "completed",
        "weight_versions": [3],
    }


def _result(task_id: str, trajectories: list[dict], gym: str = "financeagent") -> dict:
    return {
        "task_id": task_id,
        "gym_name": gym,
        "status": "success",
        "trajectories": trajectories,
        "group_metrics": {"num_turns_mean": 2.0},
    }


def _make_worker(args) -> NATSRolloutWorker:
    """Bare worker — no NATS/thread machinery, just what _process_group reads."""
    worker = object.__new__(NATSRolloutWorker)
    worker.args = args
    worker.n_per_prompt = args.n_samples_per_prompt
    worker.output_queue = queue.Queue()
    worker._output_group_counter = 0
    worker._tokenizer = None  # the fast path never touches it
    return worker


def _drain(worker: NATSRolloutWorker) -> list[list[Sample]]:
    groups = []
    while True:
        try:
            groups.append(worker.output_queue.get_nowait())
        except queue.Empty:
            break
    return groups


def _expected_group_normalized(rewards: list[float]) -> list[float]:
    """Hand-computed miles GRPO normalization: (r - mean) / (unbiased std + 1e-6)."""
    t = torch.tensor(rewards, dtype=torch.float)
    centered = t - t.mean()
    if len(rewards) > 1 and t.std() > 0:
        centered = centered / (t.std() + 1e-6)
    return centered.tolist()


class _StubMaskGenerator:
    """Stands in for MultiTurnLossMaskGenerator so the slow path runs without a
    real HF tokenizer: 10 tokens, 4-token prompt, 6-token response."""

    def __init__(self, tokenizer, tokenizer_type=None):
        pass

    def get_loss_mask(self, messages):
        return list(range(10)), [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    def get_response_lengths(self, loss_masks):
        return [len(mask[mask.index(1):]) if 1 in mask else 0 for mask in loss_masks]


# ===========================================================================
# 1. Group-identity stamping in _process_group
# ===========================================================================


class TestGroupIdentityStamping:
    def test_stamping_with_synthetic_pad(self):
        """Shared group_index, unique index == gid*n+i, rollout_id None, no group_id."""
        worker = _make_worker(_make_args())
        # 3 real trajectories + 1 synthetic -> _process_group pads the slot.
        trajs = [_fast_traj(1.0), _fast_traj(0.0), _fast_traj(0.5), _fast_traj(0.0, synthetic=True)]
        worker._process_group(
            "task-A.g0.", [_result("task-A.g0.", trajs)],
            instance_id="task-A", dedup_key="task-A#e0",
        )
        groups = _drain(worker)
        assert len(groups) == 1
        group = groups[0]
        assert len(group) == N_SAMPLES_PER_PROMPT

        assert [s.group_index for s in group] == [0] * N_SAMPLES_PER_PROMPT
        assert [s.index for s in group] == list(range(N_SAMPLES_PER_PROMPT))
        assert all(s.rollout_id is None for s in group), (
            "rollout_id must stay None: stamping the shared group id onto it makes "
            "miles' _normalize_rewards_by_rollout demand one reward per group"
        )
        # The AGISlime overlay's Sample.group_id does not exist on miles Sample
        # and must not be re-introduced as a dead instance attribute.
        assert all(not hasattr(s, "group_id") for s in group)

        # The synthetic slot was padded with a removed sibling copy.
        pad = group[-1]
        assert pad.status == Sample.Status.FAILED
        assert pad.remove_sample is True
        assert pad.reward == 0.0
        assert sum(pad.loss_mask) == 0
        assert pad.rollout_log_probs == [0.0] * pad.response_length
        assert pad.metadata["mode"] == "failed"

    def test_second_group_gets_disjoint_indices(self):
        """The per-worker group counter makes index unique ACROSS groups."""
        worker = _make_worker(_make_args())
        for _gid, task in enumerate(["task-A.g0.", "task-B.g1."]):
            worker._process_group(
                task, [_result(task, [_fast_traj(float(i % 2)) for i in range(4)])],
                instance_id=task.split(".")[0], dedup_key=f"{task}#e0",
            )
        group_a, group_b = _drain(worker)
        assert [s.group_index for s in group_a] == [0] * 4
        assert [s.group_index for s in group_b] == [1] * 4
        assert [s.index for s in group_a] == [0, 1, 2, 3]
        assert [s.index for s in group_b] == [4, 5, 6, 7]


# ===========================================================================
# 2. GRPO conversion: per-GROUP reward normalization, per-trajectory mask sums
# ===========================================================================


class TestGrpoGroupNormalization:
    def test_per_group_normalization_and_per_trajectory_mask_sums(self):
        """Two groups with within-group reward variance flow through miles'
        real postprocess_rollout_data + convert_samples_to_train_data and come
        out normalized per GROUP (not globally, not per-rollout_id)."""
        args = _make_args()
        worker = _make_worker(args)
        rewards_a = [1.0, 0.0, 0.5]  # + synthetic slot -> pad reward 0.0
        rewards_b = [0.0, 1.0, 1.0, 0.0]
        trajs_a = [_fast_traj(r, tok_n=12 + i) for i, r in enumerate(rewards_a)]
        trajs_a.append(_fast_traj(0.0, synthetic=True))
        trajs_b = [_fast_traj(r, tok_n=16 + i) for i, r in enumerate(rewards_b)]
        worker._process_group(
            "task-A.g0.", [_result("task-A.g0.", trajs_a)],
            instance_id="task-A", dedup_key="task-A#e0",
        )
        worker._process_group(
            "task-B.g1.", [_result("task-B.g1.", trajs_b)],
            instance_id="task-B", dedup_key="task-B#e0",
        )
        groups = _drain(worker)
        assert len(groups) == 2

        data, metadata = postprocess_rollout_data(args, groups, _TRAIN_PARALLEL_CONFIG)
        assert len(data) == args.global_batch_size  # 2 groups x 4, no trim
        train_data = convert_samples_to_train_data(args, data, metadata, None, None)

        # Per-GROUP normalization, hand-computed with torch. Group A's raw
        # rewards include the pad's 0.0 (the pad fills the failed slot).
        expected = _expected_group_normalized(rewards_a + [0.0]) + _expected_group_normalized(rewards_b)
        assert train_data["rewards"] == pytest.approx(expected)
        # ... and NOT global normalization over the flattened batch.
        global_norm = _expected_group_normalized(rewards_a + [0.0] + rewards_b)
        assert train_data["rewards"] != pytest.approx(global_norm)

        assert train_data["raw_reward"] == rewards_a + [0.0] + rewards_b
        assert train_data["sample_indices"] == list(range(8))
        # rollout_id is None on every sample, so rollout_ids fall back to index...
        assert train_data["rollout_ids"] == list(range(8))
        # ...making the loss denominators per-TRAJECTORY (the pad contributes 0).
        assert train_data["rollout_mask_sums"] == [sum(m) for m in train_data["loss_masks"]]
        assert train_data["rollout_mask_sums"][3] == 0  # removed pad
        assert all(lm_sum > 0 for i, lm_sum in enumerate(train_data["rollout_mask_sums"]) if i != 3)

        # use_rollout_logprobs: every row present and aligned 1:1 with response.
        assert all(
            len(lp) == rl
            for lp, rl in zip(train_data["rollout_log_probs"], train_data["response_lengths"], strict=True)
        )


# ===========================================================================
# 3. Slow-path rollout_log_probs zero-fill (mixed fast/slow group)
# ===========================================================================


class TestSlowPathLogprobFill:
    def test_mixed_group_yields_zero_filled_removed_sample(self, monkeypatch):
        """A messages-only trajectory in a fast-path group under
        use_rollout_logprobs is zero-filled + removed, never a None row."""
        monkeypatch.setattr(mask_utils, "MultiTurnLossMaskGenerator", _StubMaskGenerator)
        args = _make_args(rollout_batch_size=1, global_batch_size=4)
        worker = _make_worker(args)
        trajs = [_fast_traj(1.0), _fast_traj(0.0), _fast_traj(0.5), _slow_traj(1.0)]
        worker._process_group(
            "task-M.g0.", [_result("task-M.g0.", trajs)],
            instance_id="task-M", dedup_key="task-M#e0",
        )
        (group,) = _drain(worker)

        slow = group[3]
        assert slow.status == Sample.Status.ABORTED
        assert slow.remove_sample is True
        assert slow.rollout_log_probs == [0.0] * slow.response_length
        assert slow.response_length == 6  # stub mask: 10 tokens, 6-token response

        data, metadata = postprocess_rollout_data(args, [group], _TRAIN_PARALLEL_CONFIG)
        train_data = convert_samples_to_train_data(args, data, metadata, None, None)
        assert all(row is not None for row in train_data["rollout_log_probs"])
        assert all(
            len(lp) == rl
            for lp, rl in zip(train_data["rollout_log_probs"], train_data["response_lengths"], strict=True)
        )
        # Pre-fix failure mode: torch.tensor(None row) -> TypeError at
        # tensorization / object-store packing.
        for row in train_data["rollout_log_probs"]:
            torch.tensor(row, dtype=torch.float32)
        # The removed slow sample trains on nothing.
        assert sum(train_data["loss_masks"][3]) == 0

    def test_slow_path_without_rollout_logprobs_stays_none(self, monkeypatch):
        """With use_rollout_logprobs off, the slow path keeps the pre-existing
        behavior: no logprobs, sample stays trainable (COMPLETED)."""
        monkeypatch.setattr(mask_utils, "MultiTurnLossMaskGenerator", _StubMaskGenerator)
        worker = _make_worker(_make_args(use_rollout_logprobs=False))
        worker._process_group(
            "task-S.g0.", [_result("task-S.g0.", [_slow_traj(1.0)] * 4)],
            instance_id="task-S", dedup_key="task-S#e0",
        )
        (group,) = _drain(worker)
        assert all(s.rollout_log_probs is None for s in group)
        assert all(s.status == Sample.Status.COMPLETED for s in group)
        assert all(not s.remove_sample for s in group)
