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

The same reader serves per-step token arrays (ADR-0014). Under the
``token_arrays_by_ref`` task flag the gym stages ``log_probs`` (float64),
``token_ids`` (int32) and ``loss_mask`` (uint8) of one step as one raw
little-endian ``.tokens`` file, 13 bytes per token, under the same
``ARENA_ROUTING_DIR``. The step then carries ``token_arrays_ref``:
``{"path", "bytes", "sha256", "n_tokens"}``. ``resolve_token_arrays`` loads
the file EAGERLY, when the result is converted, because the Sample build
reads the mask. The RAM cost equals the JSON decode of the inline arrays that
it replaces. The routing payload stays lazy. A bad token file raises
``TokenArraysRefError``, which drops one trajectory and is never fatal.
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

# Per-step token arrays by file reference (ADR-0014). Shared with the
# producer (amzn_arena_contract.rl.routing.stage_token_arrays): float64
# log_probs at offset 0, int32 token_ids at 8n, uint8 loss_mask at 12n.
TOKEN_REF_KEY = "token_arrays_ref"
TOKEN_SUFFIX = ".tokens"
TOKEN_BYTES_PER_TOKEN = 13

# Every step key that names a staged file, with the suffix its producer uses.
_STEP_REF_SUFFIXES = {REF_KEY: BLOB_SUFFIX, TOKEN_REF_KEY: TOKEN_SUFFIX}
_STEP_REF_KEYS = tuple(_STEP_REF_SUFFIXES)


class RoutingReplayError(RuntimeError):
    """A trainable sample cannot get its rollout routing.

    Under ``--use-rollout-routing-replay`` the actor replays these expert
    choices in the forward and backward pass. A sample without them, or with
    a corrupt payload, would silently train the MoE on the wrong experts. The
    NATS worker therefore treats this error as fatal: it is never folded into
    the per-group ``n_failed`` pad path. ``RoutingRefLostError`` is the one
    subclass the drain survives.
    """


class RoutingRefLostError(RoutingReplayError):
    """The ref file is gone before the trainer read it.

    The trainer owns every ref it accepts, so a missing file means that
    another path deleted it first. On 2026-09-20 a NATS redelivery of an
    accepted result went down the stale-result reap while the group waited
    in the output queue (r27 ``opq-admm-calibration.g549``, r28
    ``python-ec2-reaper_s0.g1075``). Every other sample still has its
    payload, so ``generate_rollout`` drops the affected group and continues.
    A corrupt or all-zero payload stays a plain ``RoutingReplayError`` and
    stays fatal.
    """


class TokenArraysRefError(RuntimeError):
    """A step's ``token_arrays_ref`` file is missing, corrupt, or rejected.

    This is NOT a ``RoutingReplayError`` subclass. The NATS worker loop stores
    every ``RoutingReplayError`` as ``fatal_error`` and stops the run, and
    ``_process_group`` re-raises it. A bad token file loses the tokens of one
    trajectory only, so the trainer drops that trajectory and its siblings
    still train (ADR-0014).
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


def _ref_path(ref: dict[str, Any], key: str = REF_KEY) -> Path:
    """Return the file path of ``ref`` (the value of step ``key``) after the containment checks.

    Raises:
        RoutingReplayError: the path is empty, lacks the suffix of ``key``, or
            resolves outside ``ARENA_ROUTING_DIR`` when that variable is set.
            The trainer unlinks every ref it touches, so an unchecked path
            from a mis-set gym would delete unrelated files on the shared
            mount (checkpoints, datasets, ``ref_load``).
    """
    suffix = _STEP_REF_SUFFIXES[key]
    raw = ref.get("path")
    if not isinstance(raw, str) or not raw:
        raise RoutingReplayError(f"{key} has no path")
    path = Path(raw)
    if path.suffix != suffix:
        raise RoutingReplayError(f"{key} path {path} does not end with {suffix}")
    root = os.environ.get(ROUTING_DIR_ENV)
    if root and not path.resolve().is_relative_to(Path(root).resolve()):
        raise RoutingReplayError(f"{key} path {path} is outside {ROUTING_DIR_ENV}={root}")
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


def _read_ref(path: Path, ref: dict[str, Any], key: str = REF_KEY) -> bytes:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise RoutingRefLostError(
            f"{key} file unreadable: {path} (deleted before the trainer read it)"
        ) from exc
    except OSError as exc:
        raise RoutingReplayError(
            f"{key} file unreadable: {path} "
            "(is ARENA_ROUTING_DIR on a filesystem the trainer sees at the same mount path?)"
        ) from exc
    expected_bytes = ref.get("bytes")
    if not isinstance(expected_bytes, int) or len(raw) != expected_bytes:
        raise RoutingReplayError(
            f"{key} byte count mismatch for {path}: got {len(raw)}, expected {expected_bytes!r}"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ref.get("sha256"):
        raise RoutingReplayError(f"{key} sha256 mismatch for {path}")
    return raw


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # Tier 3: a leaked blob on the shared mount is a disk-space problem,
        # not a training-correctness problem. Log and go on.
        logger.warning("Failed to delete staged ref file %s", path, exc_info=True)


def _reap_ref(ref: Any, key: str = REF_KEY) -> None:
    """Delete one unconsumed ref file; refuse a path that fails containment."""
    if not _has_ref_path(ref):
        return
    try:
        path = _ref_path(ref, key)
    except RoutingReplayError:
        logger.warning("Refusing to delete rejected %s %s", key, ref.get("path"), exc_info=True)
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


PAD_EXPERT = -1


def pad_routing(shape: tuple[int, ...]) -> np.ndarray:
    """Routing for a sample that carries no real routing (pads, removed rows).

    Rows of -1 are the upstream convention: ``replay_base._get_replay_result``
    rewrites an all -1 row to ``arange(topk) % num_experts``, so each token
    still selects ``topk`` distinct experts. An all-zero row selects expert 0
    ``topk`` times; Megatron's dropless dispatcher then sizes the all-to-all
    for ``tokens * topk`` rows but permutes fewer, and fails with
    "Split sizes doesn't match total dim 0 size" (r16 step 0, 2026-09-09).
    """
    return np.full(shape, PAD_EXPERT, dtype=np.int32)


def materialize_group_routing(group: list[Sample], args: Any) -> None:
    """Decode every pending payload in ``group`` and fill the rest with -1 rows.

    Trainable samples decode strictly. A removed sample (failed pad, truncated
    or overflow sibling) without a usable payload gets an all -1 array of its
    own row count, because ``convert_samples_to_train_data`` gates on
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
                    "Task %s: removed sample has unusable routed_experts; using -1 rows", task_id, exc_info=True
                )
        rows = max(0, len(s.tokens) - 1)
        if rows not in zeros_by_rows:
            zeros_by_rows[rows] = pad_routing((rows, num_layers, topk))
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


def reap_result_refs(results: Iterable[dict[str, Any]], keys: tuple[str, ...] = _STEP_REF_KEYS) -> None:
    """Delete the ref files of raw result envelopes dropped before conversion.

    ``keys`` selects the step ref keys to reap: routing and token arrays by
    default. The accepted-tid stale path reaps ``TOKEN_REF_KEY`` only.
    """
    for result in results:
        for traj in result.get("trajectories") or []:
            for step in traj.get("steps") or []:
                for key in keys:
                    _reap_ref(step.get(key), key)


# ---------------------------------------------------------------------------
# Token arrays by file reference (ADR-0014): eager, at conversion
# ---------------------------------------------------------------------------


def resolve_token_arrays(step: dict[str, Any]) -> None:
    """Replace ``step["token_arrays_ref"]`` with the ``log_probs``, ``token_ids`` and ``loss_mask`` lists.

    A step without the key (an inline step from any gym image) is left as it
    is. The file is deleted once the containment check passes, whether the
    read succeeds or fails. The decoded values are Python floats and ints,
    bit-identical to what ``json.loads`` gives for the inline arrays.

    Raises:
        TokenArraysRefError: the ref has no path, fails containment (the file
            is then neither read nor deleted), is missing or unreadable, has
            the wrong byte count or sha256, or its ``n_tokens`` does not match
            the file at 13 bytes per token.
    """
    if TOKEN_REF_KEY not in step:
        return
    ref = step[TOKEN_REF_KEY]
    if not _has_ref_path(ref):
        raise TokenArraysRefError(f"{TOKEN_REF_KEY} has no path")
    try:
        path = _ref_path(ref, TOKEN_REF_KEY)
    except RoutingReplayError as exc:
        raise TokenArraysRefError(str(exc)) from exc
    try:
        raw = _read_ref(path, ref, TOKEN_REF_KEY)
    except RoutingReplayError as exc:
        raise TokenArraysRefError(str(exc)) from exc
    finally:
        _unlink(path)
    n = ref.get("n_tokens")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1 or len(raw) != TOKEN_BYTES_PER_TOKEN * n:
        raise TokenArraysRefError(
            f"{TOKEN_REF_KEY} {path}: {len(raw)} bytes do not hold n_tokens={n!r} "
            f"at {TOKEN_BYTES_PER_TOKEN} bytes per token"
        )
    step["log_probs"] = np.frombuffer(raw, dtype="<f8", count=n).tolist()
    step["token_ids"] = np.frombuffer(raw, dtype="<i4", count=n, offset=8 * n).tolist()
    step["loss_mask"] = np.frombuffer(raw, dtype="u1", count=n, offset=12 * n).tolist()
    del step[TOKEN_REF_KEY]
