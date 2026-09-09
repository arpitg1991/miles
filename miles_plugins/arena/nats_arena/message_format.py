"""NATS message format helpers for miles <-> AREnABase integration.

Defines the wire format for task messages (trainer -> Gym) and result
messages (Gym -> trainer) exchanged via NATS JetStream.

Task messages carry a lakefs_uri + commit_id pointing to the task directory.
The gym worker pulls and materializes the task from lakeFS.
"""

from __future__ import annotations

import gzip
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task message: trainer -> NATS -> Gym
# ---------------------------------------------------------------------------

def build_task_message(
    instance_id: str,
    *,
    lakefs_uri: str,
    lakefs_commit_id: str,
    n_samples: int = 1,
    metadata: dict[str, Any] | None = None,
    gym_name: str = "",
    session: str | None = None,
    capture_routed_experts: bool = False,
) -> dict[str, Any]:
    """Build a task message for publishing to NATS.

    The gym worker receives this, pulls the task directory from lakeFS
    at the pinned commit, and builds the instance from the filesystem.
    It then runs ``n_samples`` independent rollouts in parallel (fork pool)
    and returns all trajectories in a single result message.

    Args:
        instance_id: Task ID extracted from the lakefs_uri.
        lakefs_uri: Full lakefs:// URI to the task directory.
        lakefs_commit_id: Pinned commit SHA for reproducibility.
        n_samples: Number of parallel rollouts the gym worker should run.
        metadata: Arbitrary metadata forwarded to the gym.
        gym_name: Target gym name for routing.
        session: Trainer session token. Echoed back by the gym in the
            result so the trainer can drop results from a prior session
            (e.g., from before a checkpoint rollback). Optional — older
            gyms that don't echo this back will simply have a None session
            on their results, which the trainer treats as legacy/accepted.
        capture_routed_experts: Ask the gym to record SGLang's MoE routing
            (``return_routed_experts``) on every generate call and ship it
            on the training step (ADR-0012, R3). The key is emitted only
            when True so the message stays byte-identical for runs
            without ``--use-rollout-routing-replay``.

    Returns:
        JSON-serialisable dict ready for NATS publish.

    Note: in earlier shapes the publisher injected a W3C trace_context
    carrier so gym-side rollout spans could parent under a trainer-side
    task.<id> span. With the gym-pod-fork-pool model, the parent span
    is opened by the gym dispatcher itself (it owns the n_samples
    fan-out), so no carrier propagation across NATS is needed.
    """
    meta = metadata or {}
    if gym_name:
        meta["gym_name"] = gym_name

    msg: dict[str, Any] = {
        "id": instance_id,
        "lakefs_uri": lakefs_uri,
        "lakefs_commit_id": lakefs_commit_id,
        "n_samples": n_samples,
        "metadata": meta,
    }
    if session is not None:
        # Top-level field so old gyms can still parse the message even
        # if they ignore it. Echoed back by gyms that know to forward it.
        msg["session"] = session
    if capture_routed_experts:
        # Top-level, like ``session``: the gym contract reads
        # ``raw.get("capture_routed_experts", False)`` (amzn_arena_contract).
        msg["capture_routed_experts"] = True
    return msg


def serialize_task(task: dict[str, Any]) -> bytes:
    """JSON-encode and gzip-compress a task message for NATS."""
    raw = json.dumps(task, ensure_ascii=False).encode("utf-8")
    return gzip.compress(raw)


# ---------------------------------------------------------------------------
# Result message: Gym -> NATS -> trainer
# ---------------------------------------------------------------------------

_GZIP_MAGIC = b"\x1f\x8b"

# Result envelope statuses whose trajectories the trainer salvages.
# "truncated" = the worker ran the group but at least one rollout stopped
# on length; per-step stop_reason marks those samples TRUNCATED
# (remove_sample), so the clean siblings stay usable. Anything else
# ("failed", unknown) is dropped and only its error is logged.
SALVAGEABLE_RESULT_STATUSES = ("success", "truncated")


def parse_result(data: bytes) -> dict[str, Any]:
    """Decode a result message received from NATS.

    Transparently handles both gzip-compressed and plain JSON payloads.

    Returns:
        Dict with at least ``task_id``, ``status``, and ``trajectories``.
    """
    if data[:2] == _GZIP_MAGIC:
        data = gzip.decompress(data)
    return json.loads(data.decode("utf-8"))


def extract_trajectories(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the trajectory list from a result message.

    Each trajectory dict contains:
      - ``reward``: float (0.0-1.0)
      - ``messages``: list[dict]  (re-tokenized locally when no token-level data)
      - ``steps``: list[dict]     (GenerateClient token_ids/loss_mask/log_probs);
        may hold several self-contained segments (ADR-0011); the final
        segment is last and is the only step that carries ``stop_reason``

    Returns:
        List of trajectory dicts, or empty list on failure.
    """
    trajectories = result.get("trajectories", [])
    if not trajectories:
        logger.warning(
            "Result %s has no trajectories (status=%s, error=%s)",
            result.get("task_id", "?"),
            result.get("status", "?"),
            result.get("error", "none"),
        )
    return trajectories


# ---------------------------------------------------------------------------
# Sample -> task conversion (miles dataset -> NATS task)
# ---------------------------------------------------------------------------

def sample_to_task(
    sample,  # miles.utils.types.Sample
    *,
    rollout_id: int = 0,
    group_index: int = 0,
    n_samples: int = 1,
    gym_name: str = "",
    session: str | None = None,
    capture_routed_experts: bool = False,
) -> dict[str, Any]:
    """Convert a miles Sample (from manifest dataset) into a NATS task.

    The Sample's metadata contains lakefs_uri and lakefs_commit_id
    populated by the data source from the manifest JSONL.
    """
    meta = sample.metadata or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}

    instance_id = str(meta.get("instance_id", sample.index or sample.group_index or 0))
    lakefs_uri = meta.get("lakefs_uri", "")
    lakefs_commit_id = meta.get("lakefs_commit_id", "")

    if not lakefs_uri:
        raise ValueError(
            f"Sample {instance_id} missing lakefs_uri in metadata. "
            "Dataset must be in manifest format."
        )

    return build_task_message(
        instance_id=instance_id,
        lakefs_uri=lakefs_uri,
        lakefs_commit_id=lakefs_commit_id,
        n_samples=n_samples,
        metadata={
            "rollout_id": rollout_id,
            "group_index": group_index,
        },
        gym_name=gym_name,
        session=session,
        capture_routed_experts=capture_routed_experts,
    )
