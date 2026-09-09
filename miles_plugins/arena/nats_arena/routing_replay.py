"""R3 rollout routing replay payloads on the arena NATS path (ADR-0012).

SGLang returns ``meta_info.routed_experts`` for every generate call: base64 of
a flat little-endian int32 buffer with logical shape
``(len(tokens) - 1, num_layers, moe_router_topk)``. Row ``i`` is the routing
that produced token ``i + 1``. The gym ships that blob to the trainer in one
of two forms on the training step:

- ``step["routed_experts"]``: the base64 string inline. Rare; only a gym
  without ``ARENA_ROUTING_DIR`` does this.
- ``step["routed_experts_ref"]``: ``{"path", "bytes", "sha256"}`` for a file
  on a mount the gym and trainer share. One GLM-5.3-Flash episode is about
  58 MB, far above the 8 MiB NATS payload cap.

The trainer holds thousands of in-flight samples for hours, so a decoded
array per queued sample would cost over 100 GB of RAM. ``stash_step_payload``
keeps only the pointer on ``Sample.metadata`` while the group waits in the
output queue. ``materialize_group_routing`` decodes at drain time, right
before the group joins the train batch, and deletes the file.

The trainer deletes every ref file it consumes or drops. ``ARENA_ROUTING_DIR``
(the same value the gym uses) bounds those deletes: a ref whose path resolves
outside that directory, or that lacks the producer's ``.routing`` suffix, is
rejected before any read or unlink. Without the variable the trainer trusts
the path (local development, inline payloads).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pybase64

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# Sample.metadata keys that hold a pending (not yet decoded) payload.
REF_KEY = "routed_experts_ref"
INLINE_KEY = "routed_experts_b64"

# Shared with the producer (amzn_arena_contract.rl.routing): the gym stages
# blobs as ``<digest>-<traj>[-s<seg>]-<uuid>.routing`` under this directory.
ROUTING_DIR_ENV = "ARENA_ROUTING_DIR"
BLOB_SUFFIX = ".routing"


class RoutingReplayError(RuntimeError):
    """A trainable sample cannot get its rollout routing.

    Under ``--use-rollout-routing-replay`` the actor replays these expert
    choices in the forward and backward pass. A sample without them, or with
    a corrupt payload, would silently train the MoE on the wrong experts. The
    NATS worker therefore treats this error as fatal: it is never folded into
    the per-group ``n_failed`` pad path.
    """


def replay_enabled(args: Any) -> bool:
    """Return True when the trainer runs R3 (``--use-rollout-routing-replay``)."""
    return bool(getattr(args, "use_rollout_routing_replay", False))


def _dims(args: Any) -> tuple[int, int]:
    num_layers = int(getattr(args, "num_layers", 0) or 0)
    topk = int(getattr(args, "moe_router_topk", 0) or 0)
    if num_layers < 1 or topk < 1:
        raise RoutingReplayError(
            f"routing replay needs positive num_layers and moe_router_topk, got {num_layers} and {topk}"
        )
    return num_layers, topk


def _has_ref_path(ref: Any) -> bool:
    return isinstance(ref, dict) and isinstance(ref.get("path"), str) and bool(ref["path"])


def _ref_path(ref: dict[str, Any]) -> Path:
    """Return the blob path of ``ref`` after the containment checks.

    Raises:
        RoutingReplayError: the path is empty, lacks ``BLOB_SUFFIX``, or
            resolves outside ``ARENA_ROUTING_DIR`` when that variable is set.
            The trainer unlinks every ref it touches, so an unchecked path
            from a mis-set gym would delete unrelated files on the shared
            mount (checkpoints, datasets, ``ref_load``).
    """
    raw = ref.get("path")
    if not isinstance(raw, str) or not raw:
        raise RoutingReplayError("routed_experts_ref has no path")
    path = Path(raw)
    if path.suffix != BLOB_SUFFIX:
        raise RoutingReplayError(f"routed_experts_ref path {path} does not end with {BLOB_SUFFIX}")
    root = os.environ.get(ROUTING_DIR_ENV)
    if root and not path.resolve().is_relative_to(Path(root).resolve()):
        raise RoutingReplayError(f"routed_experts_ref path {path} is outside {ROUTING_DIR_ENV}={root}")
    return path


# ---------------------------------------------------------------------------
# Sample build time: keep only the pointer
# ---------------------------------------------------------------------------


def stash_step_payload(sample: Sample, step: dict[str, Any], *, task_id: str) -> None:
    """Move the step's routing payload onto ``sample.metadata`` for drain-time decode.

    Raises:
        RoutingReplayError: the sample is trainable and the step carries no
            payload, or its ref points outside ``ARENA_ROUTING_DIR``. A removed
            sample gets a zero array at drain time instead.
    """
    ref = step.get("routed_experts_ref")
    inline = step.get("routed_experts")
    if _has_ref_path(ref):
        try:
            _ref_path(ref)
        except RoutingReplayError:
            if not sample.remove_sample:
                raise
            # Never stash it: the reap and drain paths would unlink it.
            logger.warning("Task %s: removed sample has a rejected routed_experts_ref; ignoring it", task_id, exc_info=True)
        else:
            sample.metadata[REF_KEY] = ref
    elif isinstance(inline, str) and inline:
        sample.metadata[INLINE_KEY] = inline
    elif not sample.remove_sample:
        raise RoutingReplayError(
            f"Task {task_id}: trainable sample has no routed_experts payload. "
            "Is the gym built with capture support and did the task message set capture_routed_experts?"
        )


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def decode_routing(
    raw: bytes,
    *,
    num_tokens: int,
    num_layers: int,
    topk: int,
    allow_extra_rows: bool = False,
) -> np.ndarray:
    """Turn the raw int32 buffer into ``(num_tokens - 1, num_layers, topk)``.

    SGLang can append one stop-edge row (the routing of the final token, which
    feeds no training position); that row is dropped. ``allow_extra_rows``
    accepts a longer buffer and keeps the leading rows: the hard context
    overflow path clips ``tokens`` after the gym recorded the full sequence,
    and the leading rows are still the true routing for the surviving prefix.

    Raises:
        RoutingReplayError: on alignment or shape mismatch, or an all-zero
            payload (a top-k bypassing MoE runner backend records nothing).
    """
    per_row = num_layers * topk
    if len(raw) % 4 != 0:
        raise RoutingReplayError(f"routed_experts payload is {len(raw)} bytes, not int32-aligned")
    n_values = len(raw) // 4
    if n_values % per_row != 0:
        raise RoutingReplayError(
            f"routed_experts payload has {n_values} int32 values, not a multiple of "
            f"num_layers x topk = {num_layers} x {topk}"
        )
    rows = n_values // per_row
    expected = max(0, num_tokens - 1)
    if rows == expected + 1 or (allow_extra_rows and rows > expected):
        rows = expected
    elif rows != expected:
        raise RoutingReplayError(
            f"routed_experts rows {rows} != len(tokens) - 1 = {expected} "
            f"(num_layers={num_layers}, topk={topk})"
        )
    # .copy(): frombuffer over bytes is read-only (torch.from_numpy warns in the
    # actor's replay fill) and the copy drops the reference to the raw blob.
    arr = np.frombuffer(raw, dtype=np.int32)[: rows * per_row].reshape(rows, num_layers, topk).copy()
    if arr.size and not arr.any():
        raise RoutingReplayError(
            "routed_experts payload is all zeros: the SGLang engine did not capture routed experts "
            "(top-k bypassing --moe-runner-backend such as flashinfer_trtllm?)"
        )
    return arr


def _read_ref(path: Path, ref: dict[str, Any]) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RoutingReplayError(
            f"routed_experts_ref file unreadable: {path} "
            "(is ARENA_ROUTING_DIR on a filesystem the trainer sees at the same mount path?)"
        ) from exc
    expected_bytes = ref.get("bytes")
    if not isinstance(expected_bytes, int) or len(raw) != expected_bytes:
        raise RoutingReplayError(
            f"routed_experts_ref byte count mismatch for {path}: got {len(raw)}, expected {expected_bytes!r}"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ref.get("sha256"):
        raise RoutingReplayError(f"routed_experts_ref sha256 mismatch for {path}")
    return raw


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # Tier 3: a leaked blob on the shared mount is a disk-space problem,
        # not a training-correctness problem. Log and go on.
        logger.warning("Failed to delete routed_experts_ref file %s", path, exc_info=True)


def _reap_ref(ref: Any) -> None:
    """Delete one unconsumed ref file; refuse a path that fails containment."""
    if not _has_ref_path(ref):
        return
    try:
        path = _ref_path(ref)
    except RoutingReplayError:
        logger.warning("Refusing to delete rejected routed_experts_ref %s", ref.get("path"), exc_info=True)
        return
    _unlink(path)


def _load_payload(ref: dict[str, Any] | None, inline: str | None) -> bytes:
    """Return the raw buffer; a consumed ref file is deleted even on failure.

    The containment check runs before any read, so a rejected ref is neither
    read nor deleted.
    """
    if ref is not None:
        path = _ref_path(ref)
        try:
            return _read_ref(path, ref)
        finally:
            _unlink(path)
    assert inline is not None
    try:
        return pybase64.b64decode(inline.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise RoutingReplayError("inline routed_experts is not valid base64") from exc


# ---------------------------------------------------------------------------
# Drain time: materialize one group
# ---------------------------------------------------------------------------


def materialize_group_routing(group: list[Sample], args: Any) -> None:
    """Decode every pending payload in ``group`` and fill the rest with zeros.

    Trainable samples decode strictly. A removed sample (failed pad, truncated
    or overflow sibling) without a usable payload gets a zero array of its own
    row count, because ``convert_samples_to_train_data`` gates on
    ``samples[0]`` and ``fill_replay_data`` asserts routing on every packed
    row. Zero arrays are shared per row count within the group, so the failed
    pads of one group (all copies of one sibling) share one donor array.

    Raises:
        RoutingReplayError: a trainable sample has no payload, or its payload
            is corrupt or has the wrong shape.
    """
    num_layers, topk = _dims(args)
    zeros_by_rows: dict[int, np.ndarray] = {}
    for s in group:
        if s.rollout_routed_experts is not None:
            continue
        meta = s.metadata if isinstance(s.metadata, dict) else {}
        ref = meta.pop(REF_KEY, None)
        inline = meta.pop(INLINE_KEY, None)
        task_id = meta.get("task_id", "?")
        try:
            if ref is None and inline is None:
                raise RoutingReplayError(f"Task {task_id}: trainable sample has no routed_experts payload")
            raw = _load_payload(ref, inline)
            s.rollout_routed_experts = decode_routing(
                raw,
                num_tokens=len(s.tokens),
                num_layers=num_layers,
                topk=topk,
                allow_extra_rows=s.remove_sample,
            )
            continue
        except RoutingReplayError:
            if not s.remove_sample:
                raise
            if ref is not None or inline is not None:
                logger.warning(
                    "Task %s: removed sample has unusable routed_experts; using zeros", task_id, exc_info=True
                )
        rows = max(0, len(s.tokens) - 1)
        if rows not in zeros_by_rows:
            zeros_by_rows[rows] = np.zeros((rows, num_layers, topk), dtype=np.int32)
        s.rollout_routed_experts = zeros_by_rows[rows]


# ---------------------------------------------------------------------------
# Cleanup for payloads the trainer never consumes
# ---------------------------------------------------------------------------


def reap_sample_refs(samples: Iterable[Sample]) -> None:
    """Delete the ref files of samples that leave the pipeline undecoded."""
    for s in samples:
        meta = s.metadata if isinstance(s.metadata, dict) else {}
        _reap_ref(meta.pop(REF_KEY, None))
        meta.pop(INLINE_KEY, None)


def reap_result_refs(results: Iterable[dict[str, Any]]) -> None:
    """Delete the ref files of raw result envelopes dropped before conversion."""
    for result in results:
        for traj in result.get("trajectories") or []:
            for step in traj.get("steps") or []:
                _reap_ref(step.get("routed_experts_ref"))
