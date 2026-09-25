"""R3 rollout routing replay on the arena NATS path (miles arena ADR-0012).

Covers the trainer side: the ``capture_routed_experts`` task-message flag,
the drain-time decode of inline and file-referenced payloads (stop-edge trim,
hard-overflow trim, sha/byte checks, file deletion), the fatal path for a
trainable sample without a payload, the all-zero payload guard, the zero
arrays that failed pads and DP-alignment pads carry, the ``ARENA_ROUTING_DIR``
containment check on ref paths, the slow-path removal, and the worker
fatal-error path across ``generate_rollout`` calls. It also covers the
lost-ref path (r27 and r28, 2026-09-20): a redelivered result keeps the refs
of its queued group, and a group whose blob is gone leaves the batch without
an exception.

It also covers per-step token arrays by file reference (ADR-0014): the
``token_arrays_by_ref`` task flag, inline and ref results that give equal
Samples, bit-identical float64 decode, the sha256, byte-count and
``n_tokens`` checks, a missing file, file deletion after the read, the
containment check, the reap on every drop path, and the per-trajectory drop
that never kills the run.

Run: python -m pytest tests/fast/plugins/arena/test_routing_replay.py -v
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import queue
import struct
import threading
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
    TOKEN_BYTES_PER_TOKEN,
    TOKEN_REF_KEY,
    TOKEN_SUFFIX,
    RoutingRefLostError,
    RoutingReplayError,
    TokenArraysRefError,
    decode_routing,
    materialize_group_routing,
    reap_result_refs,
    reap_sample_refs,
    resolve_token_arrays,
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
    # A whole-group drop counts here (pop_dropped_group_metrics).
    worker._dropped_groups = {}
    worker._dropped_groups_lock = threading.Lock()
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


def test_token_flag_off_is_byte_identical() -> None:
    base = build_task_message("t", lakefs_uri="x", lakefs_commit_id="y", session="s", capture_routed_experts=True)
    off = build_task_message(
        "t", lakefs_uri="x", lakefs_commit_id="y", session="s", capture_routed_experts=True, token_arrays_by_ref=False
    )
    assert off == base
    assert json.dumps(off) == json.dumps(base)
    assert "token_arrays_by_ref" not in off


def test_token_flag_on_is_top_level_true() -> None:
    msg = build_task_message("t", lakefs_uri="x", lakefs_commit_id="y", token_arrays_by_ref=True)
    assert msg["token_arrays_by_ref"] is True
    # The gym contract reads raw.get("token_arrays_by_ref"), not metadata.
    assert "token_arrays_by_ref" not in msg["metadata"]


def test_sample_to_task_passes_token_flag_through() -> None:
    s = Sample()
    s.metadata = {"instance_id": "i", "lakefs_uri": "lakefs://r/b/i/", "lakefs_commit_id": "c"}
    assert "token_arrays_by_ref" not in sample_to_task(s)
    assert sample_to_task(s, token_arrays_by_ref=True)["token_arrays_by_ref"] is True


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
    # A hard context overflow (8 tokens > max_ctx 7) removes the sample; no payload shipped.
    args = _args(rollout_max_context_len=7)
    s = _one_sample(args, _traj(tok_n=8))
    assert s.remove_sample
    assert s.metadata["removal_reason"] == "context_overflow"
    materialize_group_routing([s], args)
    assert s.rollout_routed_experts.shape == (6, L, K)
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
    # A hard context overflow (8 tokens > max_ctx 7) removes the sample.
    args = _args(rollout_max_context_len=7)
    s = _one_sample(args, _traj(tok_n=8, routed_experts_ref=ref))
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


# ---------------------------------------------------------------------------
# Lost ref: reap on a redelivered result, then read at drain time
# ---------------------------------------------------------------------------


def test_missing_file_is_lost_error_but_corrupt_file_is_not(tmp_path: Path) -> None:
    args = _args()
    gone = {"path": str(tmp_path / "gone.routing"), "bytes": 4, "sha256": "x"}
    with pytest.raises(RoutingRefLostError, match="unreadable"):
        materialize_group_routing([_one_sample(args, _traj(tok_n=8, routed_experts_ref=gone))], args)
    corrupt = _ref(tmp_path, _raw(7), sha256="0" * 64)
    with pytest.raises(RoutingReplayError, match="sha256") as info:
        materialize_group_routing([_one_sample(args, _traj(tok_n=8, routed_experts_ref=corrupt))], args)
    assert not isinstance(info.value, RoutingRefLostError)


def test_redelivered_accepted_result_keeps_its_refs(tmp_path: Path) -> None:
    # The r27/r28 trigger: the result of an accepted group comes back a second
    # time after its tid left pending_expected. The queued group still owns
    # the file, so the stale-result path leaves it alone. An unknown tid
    # (nobody accepted it) still reaps.
    ref = _ref(tmp_path, _raw(7), name="accepted.routing")
    result = _result("t.g1.", [_traj(tok_n=8, routed_experts_ref=ref)])
    nats_rollout._discard_stale_result(result, "t.g1.", {"t.g1."})
    assert Path(ref["path"]).exists()
    nats_rollout._discard_stale_result(result, "t.g1.", set())
    assert not Path(ref["path"]).exists()


def test_redelivered_accepted_result_reaps_only_its_token_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A second run of an accepted task (a task redelivery after the gym
    # published) writes new token files that nobody reads. The accepted-tid
    # path deletes them and keeps the routing file (ADR-0014).
    routing = _ref(tmp_path, _raw(7), name="rerun.routing")
    wire = _by_ref(tmp_path, _chain(tok_n=8), "rerun")
    wire["steps"][-1][REF_KEY] = routing
    result = _result("t.g1.", [wire])
    nats_rollout._discard_stale_result(result, "t.g1.", {"t.g1."})
    assert list(tmp_path.iterdir()) == [Path(routing["path"])]
    # A redelivery of the same message: the token files are already gone.
    with caplog.at_level(logging.WARNING):
        nats_rollout._discard_stale_result(result, "t.g1.", {"t.g1."})
    assert list(tmp_path.iterdir()) == [Path(routing["path"])]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_generate_rollout_drops_group_with_lost_ref_and_continues(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Reap then read: the trainer accepted a result, queued its group, then a
    # redelivered copy of the same result went down the reap path. The whole
    # group leaves the batch, its remaining blobs are deleted, the next group
    # trains, and no exception reaches the caller.
    lost_a = _queued_sample(tmp_path, "lost_a", reward=1.0, index=0)
    lost_b = _queued_sample(tmp_path, "lost_b", reward=0.0, index=0)
    kept = _queued_sample(tmp_path, "kept", reward=1.0, index=1)
    sibling_path = Path(lost_b.metadata[REF_KEY]["path"])
    redelivered = {"trajectories": [{"steps": [{"routed_experts_ref": lost_a.metadata[REF_KEY]}]}]}
    reap_result_refs([redelivered])
    assert not Path(lost_a.metadata[REF_KEY]["path"]).exists()
    monkeypatch.setattr(nats_rollout, "get_global_worker", lambda args, ds: _FakeWorker([[lost_a, lost_b], [kept]]))

    data = generate_rollout(_rollout_args(), 0, data_source=pytypes.SimpleNamespace())
    assert [g[0].metadata["task_id"] for g in data] == ["kept"]
    assert kept.rollout_routed_experts.shape == (7, L, K)
    assert lost_b.rollout_routed_experts is None
    assert not sibling_path.exists(), "the drain reaps the remaining blobs of the dropped group"


def test_generate_rollout_still_fails_on_corrupt_ref(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = _queued_sample(tmp_path, "bad", reward=1.0, index=0)
    bad.metadata[REF_KEY]["sha256"] = "0" * 64
    monkeypatch.setattr(nats_rollout, "get_global_worker", lambda args, ds: _FakeWorker([[bad]]))
    with pytest.raises(RoutingReplayError, match="sha256"):
        generate_rollout(_rollout_args(), 0, data_source=pytypes.SimpleNamespace())


# ---------------------------------------------------------------------------
# Token arrays by file reference (ADR-0014)
# ---------------------------------------------------------------------------

_ARRAY_KEYS = ("token_ids", "loss_mask", "log_probs")
_ROLLOUT_LOGGER = "miles_plugins.arena.nats_arena.nats_rollout"


def _token_ref(dir_: Path, step: dict, name: str = "seg.tokens", **overrides: object) -> dict:
    """Write ``step``'s arrays in the gym layout; return the ``token_arrays_ref`` value."""
    raw = (
        np.asarray(step["log_probs"], dtype="<f8").tobytes()
        + np.asarray(step["token_ids"], dtype="<i4").tobytes()
        + np.asarray(step["loss_mask"], dtype="u1").tobytes()
    )
    path = dir_ / name
    path.write_bytes(raw)
    ref: dict = {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_tokens": len(step["token_ids"]),
    }
    ref.update(overrides)
    return ref


def _by_ref(dir_: Path, traj: dict, name: str, **overrides: object) -> dict:
    """The wire copy that a new gym sends: arrays in ``.tokens`` files, no ``messages``."""
    wire = {k: v for k, v in copy.deepcopy(traj).items() if k != "messages"}
    steps = []
    for j, step in enumerate(wire["steps"]):
        moved = {k: v for k, v in step.items() if k not in _ARRAY_KEYS}
        moved[TOKEN_REF_KEY] = _token_ref(dir_, step, name=f"{name}-s{j}.tokens", **overrides)
        steps.append(moved)
    wire["steps"] = steps
    return wire


def _chain(tok_n: int = 8) -> dict:
    """A two-segment episode: one archived compaction segment, then the final step."""
    traj = _traj(tok_n=tok_n)
    final = traj["steps"][0]
    archived = {k: v for k, v in final.items() if k != "stop_reason"}
    archived["segment_end"] = "compaction"
    traj["steps"] = [archived, final]
    return traj


def _token_paths(traj: dict) -> list[Path]:
    return [Path(st[TOKEN_REF_KEY]["path"]) for st in traj["steps"]]


def test_token_ref_error_is_not_fatal() -> None:
    # The worker loop stores any RoutingReplayError as fatal_error.
    assert not issubclass(TokenArraysRefError, RoutingReplayError)


def test_token_ref_resolves_bit_identical_and_deletes_file(tmp_path: Path) -> None:
    step = _traj(tok_n=8)["steps"][0]
    step["log_probs"] = [-0.0, float("nan"), 5e-324, 1e-300, -12.345678901234567, -0.1, 0.0, -3.5]
    step["token_ids"] = [0, 1, 2**31 - 1, -(2**31), 151_000, 5, 6, 7]
    inline = copy.deepcopy(step)
    wire = {k: v for k, v in step.items() if k not in _ARRAY_KEYS}
    wire[TOKEN_REF_KEY] = _token_ref(tmp_path, step)
    path = Path(wire[TOKEN_REF_KEY]["path"])

    resolve_token_arrays(wire)
    assert wire["token_ids"] == inline["token_ids"]
    assert wire["loss_mask"] == inline["loss_mask"]
    # float64 keeps every value, NaN and -0.0 included, bit-identical to the
    # double that json.loads gives for the inline list today.
    today = json.loads(json.dumps(inline["log_probs"]))
    assert np.array(wire["log_probs"], "<f8").tobytes() == np.array(today, "<f8").tobytes()
    assert all(type(v) is int for v in wire["token_ids"] + wire["loss_mask"])
    assert all(type(v) is float for v in wire["log_probs"])
    assert TOKEN_REF_KEY not in wire
    assert not path.exists(), "a consumed token file must be deleted"
    assert {k: v for k, v in wire.items() if k != "log_probs"} == {
        k: v for k, v in inline.items() if k != "log_probs"
    }


def test_token_wire_literals_match_the_gym(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Mirror of AREnATasks test_token_arrays_ref_shape_matches_trainer_decoder.
    # The literals are the wire: a rename on this side breaks every new gym.
    assert TOKEN_REF_KEY == "token_arrays_ref"
    assert TOKEN_SUFFIX == ".tokens"
    assert TOKEN_BYTES_PER_TOKEN == 13
    assert ROUTING_DIR_ENV == "ARENA_ROUTING_DIR"
    monkeypatch.setenv("ARENA_ROUTING_DIR", str(tmp_path))
    raw = struct.pack("<4d", 0.0, 0.0, -0.25, -1.5) + struct.pack("<4i", 1, 2, 3, 901) + struct.pack("4B", 0, 0, 1, 1)
    path = tmp_path / "t-0-0.tokens"
    path.write_bytes(raw)
    ref = {"path": str(path), "bytes": 52, "sha256": hashlib.sha256(raw).hexdigest(), "n_tokens": 4}
    step = {"has_generate_tokens": True, "token_arrays_ref": ref}
    resolve_token_arrays(step)
    assert step == {
        "has_generate_tokens": True,
        "log_probs": [0.0, 0.0, -0.25, -1.5],
        "token_ids": [1, 2, 3, 901],
        "loss_mask": [0, 0, 1, 1],
    }
    assert not path.exists()


def test_inline_step_is_left_unchanged() -> None:
    step = _traj(tok_n=8)["steps"][0]
    before = copy.deepcopy(step)
    resolve_token_arrays(step)
    assert step == before


@pytest.mark.parametrize("mode", ["final", "all"])
def test_process_group_ref_equals_inline(tmp_path: Path, mode: str) -> None:
    args = _args(use_rollout_routing_replay=False, arena_train_segments=mode)
    trajs = [_chain(tok_n=8), _traj(tok_n=6, prompt_n=2)]
    trajs[0]["steps"][-1]["log_probs"] = [-0.5 - k / 7 for k in range(8)]
    trajs[1]["reward"] = 0.0
    wire = [_by_ref(tmp_path, t, f"t{i}") for i, t in enumerate(trajs)]
    inline_worker, ref_worker = _make_worker(args), _make_worker(args)
    inline_worker._process_group("t.g0.", [_result("t.g0.", copy.deepcopy(trajs))])
    ref_worker._process_group("t.g0.", [_result("t.g0.", wire)])

    inline_group = inline_worker.output_queue.get_nowait()
    ref_group = ref_worker.output_queue.get_nowait()
    assert len(ref_group) == len(inline_group) == (3 if mode == "all" else 2)
    for a, b in zip(inline_group, ref_group, strict=True):
        assert b.tokens == a.tokens
        assert b.loss_mask == a.loss_mask
        assert b.response_length == a.response_length
        assert b.rollout_log_probs == a.rollout_log_probs
        assert b.reward == a.reward
        assert b.status == a.status and b.remove_sample == a.remove_sample
        assert b.metadata == a.metadata
    assert list(tmp_path.iterdir()) == [], "every token file is read and deleted, or reaped"


def test_fast_path_without_messages_trains() -> None:
    # The gym leaves out messages only when steps[-1] has has_generate_tokens.
    traj = _traj(tok_n=8)
    del traj["messages"]
    s = _one_sample(_args(use_rollout_routing_replay=False), traj)
    assert not s.remove_sample and s.response_length == 5


@pytest.mark.parametrize("mode", ["final", "all"])
def test_chain_without_messages_takes_the_token_path_in_both_modes(mode: str) -> None:
    traj = _chain(tok_n=8)
    del traj["messages"]
    args = _args(use_rollout_routing_replay=False, arena_train_segments=mode)
    episodes = _result_to_episodes_full_trajectory(_result("t.g0.", [traj]), None, args)
    assert [len(e) for e in episodes] == [2 if mode == "all" else 1]
    assert not any(s.remove_sample for s in episodes[0])


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"sha256": "0" * 64}, "sha256 mismatch"),
        ({"bytes": 4}, "byte count mismatch"),
        ({"n_tokens": 5}, "n_tokens=5"),
        ({"n_tokens": True}, "n_tokens=True"),
        (None, "unreadable"),
    ],
    ids=["sha256", "bytes", "n_tokens", "n_tokens_bool", "missing_file"],
)
def test_bad_token_ref_drops_only_that_trajectory(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, override: dict | None, match: str
) -> None:
    args = _args()  # R3 on: the bad trajectory's routing file must be reaped too.
    routing = _ref(tmp_path, _raw(7), name="bad.routing")
    bad = _by_ref(tmp_path, _traj(tok_n=8, routed_experts_ref=routing), "bad", **(override or {}))
    if override is None:
        _token_paths(bad)[0].unlink()
    good = _by_ref(tmp_path, _traj(tok_n=8, routed_experts=_b64(_raw(7))), "good")
    worker = _make_worker(args)
    with caplog.at_level(logging.WARNING, logger=_ROLLOUT_LOGGER):
        worker._process_group("t.g0.", [_result("t.g0.", [bad, good])])

    assert worker.fatal_error is None
    real, pad = worker.output_queue.get_nowait()
    assert not real.remove_sample and real.tokens == list(range(8))
    assert pad.status == Sample.Status.FAILED
    assert pad.metadata["failed_reason"] == "token_ref"
    assert list(tmp_path.iterdir()) == [], "the bad file, its routing file and the good file are all gone"
    skipped = [r.getMessage() for r in caplog.records if "skipping synthetic trajectory" in r.getMessage()]
    assert len(skipped) == 1 and "reason=token_ref" in skipped[0] and match in skipped[0]
    assert worker.pop_dropped_group_metrics()["rollout/dropped_groups/token_ref"] == 0


def test_all_bad_token_refs_drop_the_group_as_token_ref(tmp_path: Path) -> None:
    args = _args(use_rollout_routing_replay=False)
    trajs = [_by_ref(tmp_path, _traj(tok_n=8), f"t{i}", sha256="0" * 64) for i in range(N)]
    worker = _make_worker(args)
    worker._process_group("t.g0.", [_result("t.g0.", trajs)])
    assert worker.output_queue.empty()
    assert worker.fatal_error is None
    assert worker.pop_dropped_group_metrics()["rollout/dropped_groups/token_ref"] == 1
    assert list(tmp_path.iterdir()) == []


def test_token_ref_outside_root_or_wrong_suffix_is_not_read_or_deleted(routing_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "checkpoints"
    outside.mkdir()
    step = _traj(tok_n=8)["steps"][0]
    cases = ((outside, "x.tokens", "outside"), (routing_root, "model.safetensors", "does not end with"))
    for dir_, name, match in cases:
        ref = _token_ref(dir_, step, name=name)
        with pytest.raises(TokenArraysRefError, match=match):
            resolve_token_arrays({TOKEN_REF_KEY: ref})
        assert Path(ref["path"]).exists(), "a rejected ref must never be read or deleted"
        reap_result_refs([_result("t", [{"steps": [{TOKEN_REF_KEY: ref}]}])])
        assert Path(ref["path"]).exists(), "reap must not unlink a rejected ref"

    # Through the worker: the slot becomes a pad, the sibling trains, the file stays.
    victim = _by_ref(outside, _traj(tok_n=8), "victim")
    worker = _make_worker(_args(use_rollout_routing_replay=False))
    worker._process_group("t.g0.", [_result("t.g0.", [victim, _by_ref(routing_root, _traj(tok_n=8), "ok")])])
    real, pad = worker.output_queue.get_nowait()
    assert not real.remove_sample and pad.metadata["failed_reason"] == "token_ref"
    assert all(p.exists() for p in _token_paths(victim))
    # The sibling's file is read and deleted; the rejected file from above stays.
    assert [p.name for p in routing_root.iterdir()] == ["model.safetensors"]


# Every drop path deletes the token files it owns.


def test_reap_result_refs_deletes_token_and_routing_files(routing_root: Path) -> None:
    routing = _ref(routing_root, _raw(7), name="a.routing")
    wire = _by_ref(routing_root, _chain(tok_n=8), "a")
    wire["steps"][-1]["routed_experts_ref"] = routing
    reap_result_refs([_result("t", [wire])])
    assert list(routing_root.iterdir()) == []


def test_stale_result_reaps_token_files(tmp_path: Path) -> None:
    wire = _by_ref(tmp_path, _traj(tok_n=8), "stale")
    nats_rollout._discard_stale_result(_result("t.g5.", [wire]), "t.g5.", set())
    assert list(tmp_path.iterdir()) == []


def test_failed_status_reaps_token_files(tmp_path: Path) -> None:
    result = {**_result("t.g0.", [_by_ref(tmp_path, _chain(tok_n=8), f"f{i}") for i in range(N)]), "status": "failed"}
    worker = _make_worker(_args(use_rollout_routing_replay=False))
    worker._process_group("t.g0.", [result])
    assert worker.output_queue.empty()
    assert list(tmp_path.iterdir()) == []


def test_unexpected_load_error_drops_the_result_and_reaps_every_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(step: dict) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(nats_rollout, "resolve_token_arrays", boom)
    worker = _make_worker(_args(use_rollout_routing_replay=False))
    worker._process_group("t.g0.", [_result("t.g0.", [_by_ref(tmp_path, _chain(tok_n=8), f"u{i}") for i in range(N)])])
    assert worker.fatal_error is None
    assert worker.output_queue.empty()
    assert list(tmp_path.iterdir()) == []


def test_conversion_error_reaps_unread_token_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Under final only the last step is read; the archived file waits unread
    # until the conversion. A conversion error must still delete it.
    def boom(result: dict, tokenizer: object, args: object) -> list:
        raise ValueError("boom")

    monkeypatch.setattr(nats_rollout, "_result_to_episodes_full_trajectory", boom)
    worker = _make_worker(_args(use_rollout_routing_replay=False))
    worker._process_group("t.g0.", [_result("t.g0.", [_by_ref(tmp_path, _chain(tok_n=8), f"c{i}") for i in range(N)])])
    assert worker.output_queue.empty()
    assert list(tmp_path.iterdir()) == []


def test_final_mode_reaps_untrained_token_refs(tmp_path: Path) -> None:
    wire = _by_ref(tmp_path, _chain(tok_n=8), "final")
    # A corrupt archived file proves that final mode never reads it.
    wire["steps"][0][TOKEN_REF_KEY]["sha256"] = "0" * 64
    worker = _make_worker(_args(use_rollout_routing_replay=False, n_samples_per_prompt=1))
    worker._process_group("t.g0.", [_result("t.g0.", [wire])])
    (sample,) = worker.output_queue.get_nowait()
    assert not sample.remove_sample and sample.metadata["mode"] == "full_trajectory"
    assert list(tmp_path.iterdir()) == [], "the archived step's file is reaped unread"
