"""R3 rollout routing replay on the arena NATS path (miles arena ADR-0012).

Covers the trainer side: the ``capture_routed_experts`` task-message flag,
the drain-time decode of inline and file-referenced payloads (stop-edge trim,
hard-overflow trim, sha/byte checks, file deletion), the fatal path for a
trainable sample without a payload, the all-zero payload guard, the zero
arrays that failed pads and DP-alignment pads carry, the ``ARENA_ROUTING_DIR``
containment check on ref paths, the slow-path removal, and the worker
fatal-error path across ``generate_rollout`` calls.

Run: python -m pytest tests/fast/plugins/arena/test_routing_replay.py -v
"""

from __future__ import annotations

import hashlib
import queue
import types as pytypes
from pathlib import Path

import numpy as np
import pybase64
import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.utils.types import Sample
from miles_plugins.arena.nats_arena import nats_rollout
from miles_plugins.arena.nats_arena.message_format import build_task_message, sample_to_task
from miles_plugins.arena.nats_arena.nats_rollout import (
    NATSRolloutWorker,
    _pad_rows_to_dp_alignment,
    _result_to_episodes_full_trajectory,
    generate_rollout,
)
from miles_plugins.arena.nats_arena.routing_replay import (
    INLINE_KEY,
    REF_KEY,
    ROUTING_DIR_ENV,
    RoutingReplayError,
    decode_routing,
    materialize_group_routing,
    reap_result_refs,
    reap_sample_refs,
)

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

L = 3  # num_layers
K = 2  # moe_router_topk
N = 2  # n_samples_per_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _args(**overrides: object) -> pytypes.SimpleNamespace:
    fields = dict(
        use_rollout_routing_replay=True,
        num_layers=L,
        moe_router_topk=K,
        n_samples_per_prompt=N,
        arena_train_segments="final",
        arena_mask_clipped_final_turn=False,
        arena_keep_timeout_trajectories=False,
        arena_keep_context_error_trajectories=False,
        use_rollout_logprobs=True,
        rollout_max_context_len=None,
        sglang_context_length=None,
        max_tokens_per_gpu=None,
    )
    fields.update(overrides)
    return pytypes.SimpleNamespace(**fields)


def _raw(rows: int) -> bytes:
    # arange from 1 so the buffer is never all zeros (the guard rejects that).
    return np.arange(1, rows * L * K + 1, dtype=np.int32).tobytes()


def _b64(raw: bytes) -> str:
    return pybase64.b64encode(raw).decode("ascii")


def _ref(tmp_path: Path, raw: bytes, name: str = "blob.routing", **overrides: object) -> dict:
    path = tmp_path / name
    path.write_bytes(raw)
    ref: dict = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    ref.update(overrides)
    return ref


def _traj(tok_n: int = 8, prompt_n: int = 3, stop_reason: str = "stop", **step_extra: object) -> dict:
    """Fast-path trajectory whose final step carries the routing payload."""
    step = {
        "has_generate_tokens": True,
        "token_ids": list(range(tok_n)),
        "loss_mask": [0] * prompt_n + [1] * (tok_n - prompt_n),
        "log_probs": [-0.1] * tok_n,
        "stop_reason": stop_reason,
    }
    step.update(step_extra)
    return {"reward": 1.0, "messages": [{"role": "user", "content": "x"}], "steps": [step]}


def _result(tid: str, trajs: list[dict]) -> dict:
    return {"task_id": tid, "status": "success", "gym_name": "g", "trajectories": trajs}


def _make_worker(args: pytypes.SimpleNamespace) -> NATSRolloutWorker:
    worker = object.__new__(NATSRolloutWorker)
    worker.args = args
    worker.n_per_prompt = args.n_samples_per_prompt
    worker.output_queue = queue.Queue()
    worker._output_group_counter = 0
    worker._tokenizer = None
    worker._train_segments = args.arena_train_segments
    worker.fatal_error = None
    return worker


def _one_sample(args: pytypes.SimpleNamespace, traj: dict) -> Sample:
    episodes = _result_to_episodes_full_trajectory(_result("t.g0.", [traj]), None, args)
    assert len(episodes) == 1 and len(episodes[0]) == 1
    return episodes[0][0]


# ---------------------------------------------------------------------------
# T1: task message flag
# ---------------------------------------------------------------------------


def test_task_message_flag_off_is_byte_identical() -> None:
    base = build_task_message("t", lakefs_uri="x", lakefs_commit_id="y", session="s")
    off = build_task_message("t", lakefs_uri="x", lakefs_commit_id="y", session="s", capture_routed_experts=False)
    assert off == base
    assert "capture_routed_experts" not in off


def test_task_message_flag_on_is_top_level_true() -> None:
    msg = build_task_message("t", lakefs_uri="x", lakefs_commit_id="y", capture_routed_experts=True)
    assert msg["capture_routed_experts"] is True
    # The gym contract reads raw.get("capture_routed_experts"), not metadata.
    assert "capture_routed_experts" not in msg["metadata"]


def test_sample_to_task_passes_flag_through() -> None:
    s = Sample()
    s.metadata = {"instance_id": "i", "lakefs_uri": "lakefs://r/b/i/", "lakefs_commit_id": "c"}
    assert "capture_routed_experts" not in sample_to_task(s)
    assert sample_to_task(s, capture_routed_experts=True)["capture_routed_experts"] is True


# ---------------------------------------------------------------------------
# T2: decode
# ---------------------------------------------------------------------------


def test_decode_exact_rows() -> None:
    arr = decode_routing(_raw(7), num_tokens=8, num_layers=L, topk=K)
    assert arr.shape == (7, L, K)
    assert arr.dtype == np.int32
    assert arr.flags.writeable
    np.testing.assert_array_equal(arr.ravel(), np.arange(1, 7 * L * K + 1))


def test_decode_trims_one_stop_edge_row() -> None:
    arr = decode_routing(_raw(8), num_tokens=8, num_layers=L, topk=K)
    assert arr.shape == (7, L, K)
    np.testing.assert_array_equal(arr, decode_routing(_raw(7), num_tokens=8, num_layers=L, topk=K))


@pytest.mark.parametrize("rows", [5, 9])
def test_decode_other_size_mismatch_raises(rows: int) -> None:
    with pytest.raises(RoutingReplayError, match="rows"):
        decode_routing(_raw(rows), num_tokens=8, num_layers=L, topk=K)


def test_decode_allow_extra_rows_keeps_prefix() -> None:
    arr = decode_routing(_raw(12), num_tokens=8, num_layers=L, topk=K, allow_extra_rows=True)
    np.testing.assert_array_equal(arr, decode_routing(_raw(7), num_tokens=8, num_layers=L, topk=K))


def test_decode_misaligned_buffer_raises() -> None:
    with pytest.raises(RoutingReplayError, match="int32"):
        decode_routing(_raw(7)[:-1], num_tokens=8, num_layers=L, topk=K)
    with pytest.raises(RoutingReplayError, match="multiple"):
        decode_routing(_raw(7)[:-4], num_tokens=8, num_layers=L, topk=K)


def test_decode_all_zeros_raises() -> None:
    with pytest.raises(RoutingReplayError, match="all zeros"):
        decode_routing(np.zeros(7 * L * K, dtype=np.int32).tobytes(), num_tokens=8, num_layers=L, topk=K)


# ---------------------------------------------------------------------------
# T2: sample build keeps the pointer, drain materializes
# ---------------------------------------------------------------------------


def test_inline_payload_is_pending_then_materialized() -> None:
    args = _args()
    s = _one_sample(args, _traj(tok_n=8, routed_experts=_b64(_raw(7))))
    assert s.rollout_routed_experts is None
    assert s.metadata[INLINE_KEY] == _b64(_raw(7))

    materialize_group_routing([s], args)
    assert s.rollout_routed_experts.shape == (7, L, K)
    assert INLINE_KEY not in s.metadata and REF_KEY not in s.metadata


def test_inline_stop_edge_row_is_trimmed() -> None:
    args = _args()
    s = _one_sample(args, _traj(tok_n=8, routed_experts=_b64(_raw(8))))
    materialize_group_routing([s], args)
    assert s.rollout_routed_experts.shape == (7, L, K)


def test_ref_materializes_from_file_and_deletes_it(tmp_path: Path) -> None:
    args = _args()
    ref = _ref(tmp_path, _raw(7))
    s = _one_sample(args, _traj(tok_n=8, routed_experts_ref=ref))
    # Lazy: only the pointer sits on the sample while it waits in the queue.
    assert s.rollout_routed_experts is None
    assert s.metadata[REF_KEY] == ref
    assert Path(ref["path"]).exists()

    materialize_group_routing([s], args)
    assert s.rollout_routed_experts.shape == (7, L, K)
    np.testing.assert_array_equal(s.rollout_routed_experts.ravel(), np.arange(1, 7 * L * K + 1))
    assert not Path(ref["path"]).exists()
    assert REF_KEY not in s.metadata


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"sha256": "0" * 64}, "sha256"),
        ({"bytes": 4}, "byte count"),
    ],
)
def test_ref_mismatch_raises_and_deletes_file(tmp_path: Path, override: dict, match: str) -> None:
    args = _args()
    ref = _ref(tmp_path, _raw(7), **override)
    s = _one_sample(args, _traj(tok_n=8, routed_experts_ref=ref))
    with pytest.raises(RoutingReplayError, match=match):
        materialize_group_routing([s], args)
    assert not Path(ref["path"]).exists(), "a consumed ref must not leak its blob"


def test_ref_missing_file_raises(tmp_path: Path) -> None:
    args = _args()
    ref = {"path": str(tmp_path / "gone.routing"), "bytes": 4, "sha256": "x"}
    s = _one_sample(args, _traj(tok_n=8, routed_experts_ref=ref))
    with pytest.raises(RoutingReplayError, match="unreadable"):
        materialize_group_routing([s], args)


def test_all_zero_real_payload_raises_at_drain() -> None:
    args = _args()
    s = _one_sample(args, _traj(tok_n=8, routed_experts=_b64(np.zeros(7 * L * K, dtype=np.int32).tobytes())))
    with pytest.raises(RoutingReplayError, match="all zeros"):
        materialize_group_routing([s], args)


def test_hard_overflow_clip_trims_payload_rows() -> None:
    # 12 tokens generated, max_ctx 8: tokens are clipped to 8 and the sample
    # is removed; the gym's blob still covers all 12 -> keep the first 7 rows.
    args = _args(rollout_max_context_len=8)
    s = _one_sample(args, _traj(tok_n=12, prompt_n=3, routed_experts=_b64(_raw(11))))
    assert len(s.tokens) == 8 and s.remove_sample
    materialize_group_routing([s], args)
    assert s.rollout_routed_experts.shape == (7, L, K)
    np.testing.assert_array_equal(s.rollout_routed_experts.ravel(), np.arange(1, 7 * L * K + 1))


def test_flag_off_ignores_payload() -> None:
    args = _args(use_rollout_routing_replay=False)
    s = _one_sample(args, _traj(tok_n=8, routed_experts=_b64(_raw(7))))
    assert INLINE_KEY not in s.metadata and REF_KEY not in s.metadata
    assert s.rollout_routed_experts is None


# ---------------------------------------------------------------------------
# Missing payload: fatal for trainable, zeros for removed
# ---------------------------------------------------------------------------


def test_missing_payload_on_trainable_sample_raises_at_build() -> None:
    with pytest.raises(RoutingReplayError, match="no routed_experts payload"):
        _one_sample(_args(), _traj(tok_n=8))


def test_process_group_does_not_swallow_routing_error() -> None:
    worker = _make_worker(_args())
    with pytest.raises(RoutingReplayError):
        worker._process_group("t.g0.", [_result("t.g0.", [_traj(tok_n=8)])])
    assert worker.output_queue.empty(), "a group with missing routing must never reach the queue"


def test_missing_payload_on_removed_sample_gets_zeros() -> None:
    args = _args()
    # stop_reason "length" -> TRUNCATED -> remove_sample; no payload shipped.
    s = _one_sample(args, _traj(tok_n=8, stop_reason="length"))
    assert s.remove_sample
    materialize_group_routing([s], args)
    assert s.rollout_routed_experts.shape == (7, L, K)
    assert (s.rollout_routed_experts == -1).all()


def test_materialize_missing_payload_trainable_raises() -> None:
    # Slow-path (messages-only) samples never get a pointer; materialize
    # must still refuse to train them without routing.
    s = Sample()
    s.tokens = list(range(6))
    s.metadata = {"task_id": "slow"}
    with pytest.raises(RoutingReplayError, match="no routed_experts payload"):
        materialize_group_routing([s], _args())


# ---------------------------------------------------------------------------
# Slow path (messages-only): no payload -> removed, never a run-killer
# ---------------------------------------------------------------------------


class _StubMaskGenerator:
    """Stands in for MultiTurnLossMaskGenerator: 10 tokens, 4-token prompt."""

    def __init__(self, tokenizer: object, tokenizer_type: object = None) -> None:
        pass

    def get_loss_mask(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        return list(range(10)), [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    def get_response_lengths(self, loss_masks: list[list[int]]) -> list[int]:
        return [len(mask[mask.index(1):]) if 1 in mask else 0 for mask in loss_masks]


def _slow_traj() -> dict:
    return {"reward": 1.0, "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], "steps": []}


def test_slow_path_under_replay_is_removed_and_materializes_zeros(monkeypatch: pytest.MonkeyPatch) -> None:
    from miles.utils import mask_utils

    monkeypatch.setattr(mask_utils, "MultiTurnLossMaskGenerator", _StubMaskGenerator)
    # use_rollout_logprobs off: nothing else removes a COMPLETED slow-path sample.
    args = _args(use_rollout_logprobs=False)
    s = _one_sample(args, _slow_traj())
    assert s.remove_sample
    assert s.status == Sample.Status.ABORTED
    assert s.metadata["removal_reason"] == "no_routing"
    materialize_group_routing([s], args)
    assert s.rollout_routed_experts.shape == (9, L, K)
    assert (s.rollout_routed_experts == -1).all()


def test_slow_path_without_replay_stays_trainable(monkeypatch: pytest.MonkeyPatch) -> None:
    from miles.utils import mask_utils

    monkeypatch.setattr(mask_utils, "MultiTurnLossMaskGenerator", _StubMaskGenerator)
    s = _one_sample(_args(use_rollout_routing_replay=False, use_rollout_logprobs=False), _slow_traj())
    assert not s.remove_sample
    assert s.status == Sample.Status.COMPLETED


# ---------------------------------------------------------------------------
# ARENA_ROUTING_DIR containment: never read or delete outside the root
# ---------------------------------------------------------------------------


@pytest.fixture
def routing_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "routing"
    root.mkdir()
    monkeypatch.setenv(ROUTING_DIR_ENV, str(root))
    return root


def test_ref_inside_root_is_consumed(routing_root: Path) -> None:
    args = _args()
    (routing_root / "sub").mkdir()
    ref = _ref(routing_root / "sub", _raw(7))
    s = _one_sample(args, _traj(tok_n=8, routed_experts_ref=ref))
    materialize_group_routing([s], args)
    assert s.rollout_routed_experts.shape == (7, L, K)
    assert not Path(ref["path"]).exists()


def test_ref_outside_root_on_trainable_sample_raises_at_build(routing_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "checkpoints"
    outside.mkdir()
    ref = _ref(outside, _raw(7))
    with pytest.raises(RoutingReplayError, match="outside"):
        _one_sample(_args(), _traj(tok_n=8, routed_experts_ref=ref))
    assert Path(ref["path"]).exists(), "a rejected ref must never be deleted"


def test_ref_outside_root_on_removed_sample_gets_zeros_and_keeps_file(routing_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "checkpoints"
    outside.mkdir()
    ref = _ref(outside, _raw(7))
    args = _args()
    s = _one_sample(args, _traj(tok_n=8, stop_reason="length", routed_experts_ref=ref))
    assert s.remove_sample and REF_KEY not in s.metadata
    materialize_group_routing([s], args)
    assert (s.rollout_routed_experts == -1).all()
    assert Path(ref["path"]).exists()


def test_ref_outside_root_at_drain_raises_and_keeps_file(routing_root: Path, tmp_path: Path) -> None:
    # A stale pointer that reached the queue before the root changed.
    outside = tmp_path / "data"
    outside.mkdir()
    s = _queued_sample(outside, "stale", reward=1.0, index=0)
    path = Path(s.metadata[REF_KEY]["path"])
    with pytest.raises(RoutingReplayError, match="outside"):
        materialize_group_routing([s], _args())
    assert path.exists()


def test_ref_with_wrong_suffix_raises(tmp_path: Path) -> None:
    ref = _ref(tmp_path, _raw(7), name="model.safetensors")
    with pytest.raises(RoutingReplayError, match="does not end with"):
        _one_sample(_args(), _traj(tok_n=8, routed_experts_ref=ref))
    assert Path(ref["path"]).exists()


def test_reap_refuses_paths_outside_root(routing_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "ref_load"
    outside.mkdir()
    s = _queued_sample(outside, "victim", reward=1.0, index=0)
    victim = Path(s.metadata[REF_KEY]["path"])
    inside = _ref(routing_root, _raw(7), name="ok.routing")
    reap_sample_refs([s])
    reap_result_refs([_result("t", [_traj(routed_experts_ref={"path": str(victim), "bytes": 1, "sha256": "x"}), _traj(routed_experts_ref=inside)])])
    assert victim.exists(), "reap must not unlink outside ARENA_ROUTING_DIR"
    assert REF_KEY not in s.metadata
    assert not Path(inside["path"]).exists()


# ---------------------------------------------------------------------------
# T3: pads
# ---------------------------------------------------------------------------


def test_failed_pads_share_one_zero_array_at_drain(tmp_path: Path) -> None:
    args = _args(n_samples_per_prompt=3)
    worker = _make_worker(args)
    ref = _ref(tmp_path, _raw(7))
    # One real trajectory + two synthetic failures -> two pads copy its tokens.
    synthetic = {"synthetic": True, "stop_reason": "errored", "messages": [], "steps": []}
    worker._process_group("t.g0.", [_result("t.g0.", [_traj(tok_n=8, routed_experts_ref=ref), synthetic, synthetic])])
    group = worker.output_queue.get_nowait()
    assert len(group) == 3
    real, pad_a, pad_b = group
    assert pad_a.status == Sample.Status.FAILED and pad_a.rollout_routed_experts is None

    materialize_group_routing(group, args)
    assert real.rollout_routed_experts.shape == (7, L, K) and real.rollout_routed_experts.any()
    for pad in (pad_a, pad_b):
        assert pad.rollout_routed_experts.shape == (len(pad.tokens) - 1, L, K)
        assert pad.rollout_routed_experts.dtype == np.int32
        assert (pad.rollout_routed_experts == -1).all()
    assert pad_a.rollout_routed_experts is pad_b.rollout_routed_experts, "one donor array per group"
    assert not Path(ref["path"]).exists()


def test_dp_alignment_pads_carry_shared_zeros() -> None:
    src = Sample()
    src.tokens = list(range(6))
    src.response_length = 3
    src.loss_mask = [1, 1, 1]
    src.rollout_log_probs = [-0.1] * 3
    src.reward = 1.0
    src.group_index = 0
    src.rollout_id = 0
    src.index = 0
    src.rollout_routed_experts = np.arange(1, 5 * L * K + 1, dtype=np.int32).reshape(5, L, K)
    src.metadata = {"task_id": "t", "segment": 0}
    data = [[src]]
    # dp_size 4, one row -> three pads.
    args = pytypes.SimpleNamespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        use_dynamic_batch_size=True,
        micro_batch_size=None,
    )
    assert _pad_rows_to_dp_alignment(data, args) == 3
    pads = data[0][1:]
    assert len(pads) == 3
    for pad in pads:
        assert pad.rollout_routed_experts.shape == (5, L, K)
        assert (pad.rollout_routed_experts == -1).all()
    assert pads[0].rollout_routed_experts is pads[2].rollout_routed_experts
    # The source keeps its real routing.
    assert src.rollout_routed_experts.any()


def test_dp_alignment_pads_without_replay_stay_none() -> None:
    src = Sample()
    src.tokens = list(range(6))
    src.response_length = 3
    src.loss_mask = [1, 1, 1]
    src.reward = 1.0
    src.group_index = 0
    src.rollout_id = 0
    src.index = 0
    src.metadata = {"task_id": "t", "segment": 0}
    data = [[src]]
    args = pytypes.SimpleNamespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        use_dynamic_batch_size=True,
        micro_batch_size=None,
    )
    assert _pad_rows_to_dp_alignment(data, args) == 1
    assert data[0][1].rollout_routed_experts is None


# ---------------------------------------------------------------------------
# Drain loop: materialize kept groups, reap dropped ones, re-raise fatal
# ---------------------------------------------------------------------------


class _FakeWorker:
    def __init__(self, groups: list[list[Sample]], fatal: BaseException | None = None) -> None:
        self.output_queue: queue.Queue = queue.Queue()
        for g in groups:
            self.output_queue.put(g)
        self._train_segments = "final"
        # The drain loop pulls ``remaining`` groups per pass and stops when the
        # worker thread is dead; stay alive while groups remain so a dropped
        # group is followed by a second pass.
        self.worker_thread = pytypes.SimpleNamespace(is_alive=lambda: not self.output_queue.empty())
        self.fatal_error = fatal

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()


def _rollout_args(**overrides: object) -> pytypes.SimpleNamespace:
    fields = dict(
        rollout_global_dataset=True,
        global_batch_size=1,
        n_samples_per_prompt=1,
        dynamic_sampling_filter_path=None,
        use_wandb=False,
        arena_train_segments="final",
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        use_dynamic_batch_size=True,
        micro_batch_size=None,
        use_rollout_routing_replay=True,
        num_layers=L,
        moe_router_topk=K,
    )
    fields.update(overrides)
    return pytypes.SimpleNamespace(**fields)


def _queued_sample(tmp_path: Path, name: str, reward: float, index: int) -> Sample:
    s = Sample()
    s.tokens = list(range(8))
    s.response_length = 5
    s.loss_mask = [1] * 5
    s.rollout_log_probs = [-0.1] * 5
    s.reward = reward
    s.group_index = index
    s.index = index
    s.status = Sample.Status.COMPLETED
    s.metadata = {"task_id": name, "mode": "full_trajectory", "segment": 0, "n_segments": 1}
    s.metadata[REF_KEY] = _ref(tmp_path, _raw(7), name=f"{name}.routing")
    return s


def test_generate_rollout_materializes_kept_and_reaps_dropped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dropped = _queued_sample(tmp_path, "dropped", reward=0.0, index=0)
    kept = _queued_sample(tmp_path, "kept", reward=1.0, index=1)
    dropped_path = Path(dropped.metadata[REF_KEY]["path"])
    kept_path = Path(kept.metadata[REF_KEY]["path"])
    monkeypatch.setattr(nats_rollout, "get_global_worker", lambda args, ds: _FakeWorker([[dropped], [kept]]))
    # A filter that drops zero-reward groups stands in for check_reward_nonzero_std.
    args = _rollout_args(dynamic_sampling_filter_path="tests.fast.plugins.arena.test_routing_replay._keep_positive")

    data = generate_rollout(args, 0, data_source=pytypes.SimpleNamespace())
    assert [g[0].metadata["task_id"] for g in data] == ["kept"]
    assert kept.rollout_routed_experts.shape == (7, L, K)
    assert dropped.rollout_routed_experts is None
    assert not kept_path.exists() and not dropped_path.exists()


def _keep_positive(args: object, group: list[Sample]) -> bool:
    """Dynamic-sampling filter for the drain test: keep groups with reward > 0."""
    return group[0].reward > 0


def test_generate_rollout_reraises_worker_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    fatal = RoutingReplayError("boom")
    monkeypatch.setattr(nats_rollout, "get_global_worker", lambda args, ds: _FakeWorker([], fatal=fatal))
    with pytest.raises(RoutingReplayError, match="boom"):
        generate_rollout(_rollout_args(), 0, data_source=pytypes.SimpleNamespace())


def _forbid_worker_construction(*a: object, **kw: object) -> None:
    raise AssertionError("a dead worker with fatal_error must not be replaced by a new one")


def test_dead_worker_with_fatal_is_not_recreated(monkeypatch: pytest.MonkeyPatch) -> None:
    # The thread died between two generate_rollout calls (during the train
    # step): worker_thread.is_alive() is False and fatal_error is set. The
    # real get_global_worker must raise, not build a fresh worker.
    dead = _FakeWorker([], fatal=RoutingReplayError("died between steps"))
    assert not dead.worker_thread.is_alive()
    monkeypatch.setattr(nats_rollout, "_global_worker", dead)
    monkeypatch.setattr(nats_rollout, "NATSRolloutWorker", _forbid_worker_construction)
    with pytest.raises(RoutingReplayError, match="died between steps"):
        generate_rollout(_rollout_args(), 0, data_source=pytypes.SimpleNamespace())
    assert nats_rollout._global_worker is dead


def test_dead_worker_with_full_queue_still_raises_fatal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A full batch sits in the queue, so the wait loop (the only pre-fix
    # check) never runs. The fatal error must still surface.
    dead = _FakeWorker([[_queued_sample(tmp_path, "q", reward=1.0, index=0)]], fatal=RoutingReplayError("late"))
    monkeypatch.setattr(nats_rollout, "get_global_worker", lambda args, ds: dead)
    with pytest.raises(RoutingReplayError, match="late"):
        generate_rollout(_rollout_args(), 0, data_source=pytypes.SimpleNamespace())


def test_reap_sample_refs_deletes_and_clears(tmp_path: Path) -> None:
    s = _queued_sample(tmp_path, "r", reward=1.0, index=0)
    path = Path(s.metadata[REF_KEY]["path"])
    reap_sample_refs([s])
    assert not path.exists()
    assert REF_KEY not in s.metadata


def test_untrained_segment_refs_are_reaped_under_final_mode(tmp_path, monkeypatch) -> None:
    """Under train_segments=final the archived segments' ref files must not orphan."""
    from miles_plugins.arena.nats_arena import routing_replay as rr

    monkeypatch.setenv(rr.ROUTING_DIR_ENV, str(tmp_path))
    archived = tmp_path / "a-0-s0-x.routing"
    archived.write_bytes(b"\x00" * 8)
    final = tmp_path / "a-0-s1-x.routing"
    final.write_bytes(b"\x00" * 8)
    steps = [
        {"routed_experts_ref": {"path": str(archived), "bytes": 8, "sha256": ""}},
        {"routed_experts_ref": {"path": str(final), "bytes": 8, "sha256": ""}, "has_generate_tokens": True},
    ]
    training = steps[-1:]
    untrained = [st for st in steps if not any(st is t for t in training)]
    rr.reap_result_refs([{"trajectories": [{"steps": untrained}]}])
    assert not archived.exists()
    assert final.exists()


def test_pad_routing_rows_are_minus_one_not_zero() -> None:
    """Zeros mean expert 0 x topk per token; Megatron's dropless dispatcher then
    permutes fewer rows than tokens*topk and the EP all-to-all fails (r16)."""
    from miles_plugins.arena.nats_arena.routing_replay import pad_routing

    a = pad_routing((3, 45, 8))
    assert a.dtype.name == "int32" and a.shape == (3, 45, 8) and (a == -1).all()
