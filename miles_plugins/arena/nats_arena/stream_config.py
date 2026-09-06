"""JetStream stream configuration for the arena NATS rollout path.

The results stream holds one message per completed rollout task. The trainer
consumes and acknowledges each message. Before this module the stream used
``RetentionPolicy.LIMITS`` with no size or age bound, so acknowledged messages
lived forever. On a JetStream file store backed by a fixed ``emptyDir`` the
store grew until the volume filled, NATS was evicted, and every in-flight
rollout task was lost (glm53 r6, 2026-09-05). The bounds here cap the store so a
slow or stalled trainer consumer never fills the disk.

The tasks stream uses ``RetentionPolicy.WORK_QUEUE``: the server deletes each
task message once a gym worker acknowledges it, so it needs no explicit size
bound and stays in ``nats_rollout``.
"""

from __future__ import annotations

import os

from nats.js.api import DiscardPolicy, RetentionPolicy, StreamConfig

# 128 MiB per-message ceiling (matches the prior hardcoded value).
RESULTS_MAX_MSG_SIZE = 134_217_728

# Bound the results store so a stalled trainer consumer never fills the
# JetStream volume. Default age is 4 h — longer than the DLQ task deadline
# (NATS_TASK_DEADLINE_SECS, default 12600 s => 3.5 h) so a live task's result is
# never discarded before the trainer reads it. Default size is 6 GiB — below the
# nats.yaml emptyDir so the volume never fills. discard=OLD drops the oldest
# acknowledged message when a bound is reached and never blocks a publish.
DEFAULT_RESULTS_MAX_AGE_SECS = 4 * 60 * 60
DEFAULT_RESULTS_MAX_BYTES = 6 * 1024 * 1024 * 1024


def _env_int(name: str, default: int) -> int:
    """Return the int environment variable ``name`` or ``default``.

    Args:
        name: Environment variable name.
        default: Value to return when the variable is unset or unparseable.

    Returns:
        The parsed integer, or ``default`` when unset, empty, or not an int.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def results_max_age_secs() -> int:
    """Return the results-stream max age in seconds (ARENA_RESULTS_MAX_AGE_SECS)."""
    return _env_int("ARENA_RESULTS_MAX_AGE_SECS", DEFAULT_RESULTS_MAX_AGE_SECS)


def results_max_bytes() -> int:
    """Return the results-stream max size in bytes (ARENA_RESULTS_MAX_BYTES)."""
    return _env_int("ARENA_RESULTS_MAX_BYTES", DEFAULT_RESULTS_MAX_BYTES)


def results_stream_config(name: str, subject: str) -> StreamConfig:
    """Build the bounded results-stream config.

    Args:
        name: JetStream stream name.
        subject: Results subject the stream binds.

    Returns:
        A LIMITS-retention ``StreamConfig`` with per-message, age, and size
        bounds and ``discard=OLD``.
    """
    return StreamConfig(
        name=name,
        subjects=[subject],
        retention=RetentionPolicy.LIMITS,
        max_msg_size=RESULTS_MAX_MSG_SIZE,
        max_age=float(results_max_age_secs()),
        max_bytes=results_max_bytes(),
        discard=DiscardPolicy.OLD,
    )
