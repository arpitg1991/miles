"""Bounds on the NATS results stream config (stream_config.py).

The results stream must carry an age and size bound with discard=OLD so a
stalled trainer consumer cannot fill the JetStream volume — the eviction that
lost glm53 r6. These checks pin the defaults, the env overrides, and the
StreamConfig the trainer builds.
"""

from __future__ import annotations

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def test_defaults_are_bounded_and_below_the_volume():
    from miles_plugins.arena.nats_arena.stream_config import (
        DEFAULT_RESULTS_MAX_AGE_SECS,
        DEFAULT_RESULTS_MAX_BYTES,
        results_max_age_secs,
        results_max_bytes,
    )

    # 4 h age > the 3.5 h default DLQ deadline so a live result is never dropped.
    assert DEFAULT_RESULTS_MAX_AGE_SECS == 4 * 60 * 60
    assert DEFAULT_RESULTS_MAX_AGE_SECS > 12600
    # 6 GiB < the 20 GiB emptyDir so the volume never fills.
    assert DEFAULT_RESULTS_MAX_BYTES == 6 * 1024 * 1024 * 1024
    assert DEFAULT_RESULTS_MAX_BYTES < 20 * 1024 * 1024 * 1024
    assert results_max_age_secs() == DEFAULT_RESULTS_MAX_AGE_SECS
    assert results_max_bytes() == DEFAULT_RESULTS_MAX_BYTES


def test_stream_config_carries_bounds_and_discard_old():
    from nats.js.api import DiscardPolicy, RetentionPolicy

    from miles_plugins.arena.nats_arena.stream_config import (
        RESULTS_MAX_MSG_SIZE,
        results_max_age_secs,
        results_max_bytes,
        results_stream_config,
    )

    cfg = results_stream_config("ARENA_RESULTS", "arena.results")

    assert cfg.name == "ARENA_RESULTS"
    assert cfg.subjects == ["arena.results"]
    assert cfg.retention == RetentionPolicy.LIMITS
    assert cfg.discard == DiscardPolicy.OLD
    assert cfg.max_msg_size == RESULTS_MAX_MSG_SIZE
    # max_age is seconds here; nats-py converts to nanoseconds on the wire.
    assert cfg.max_age == float(results_max_age_secs())
    assert cfg.max_bytes == results_max_bytes()

    # The on-wire form must carry the age as nanoseconds and stay positive:
    # an unset (0) or negative age would leave the stream unbounded.
    wire = cfg.as_dict()
    assert wire["max_age"] == results_max_age_secs() * 1_000_000_000
    assert wire["max_bytes"] == results_max_bytes()


@pytest.mark.parametrize(
    ("age_env", "bytes_env", "want_age", "want_bytes"),
    [
        ("7200", "1073741824", 7200, 1073741824),
        ("", "", 4 * 60 * 60, 6 * 1024 * 1024 * 1024),
        ("not-an-int", "also-bad", 4 * 60 * 60, 6 * 1024 * 1024 * 1024),
    ],
)
def test_env_overrides_and_fallback(monkeypatch, age_env, bytes_env, want_age, want_bytes):
    from miles_plugins.arena.nats_arena.stream_config import (
        results_max_age_secs,
        results_max_bytes,
    )

    monkeypatch.setenv("ARENA_RESULTS_MAX_AGE_SECS", age_env)
    monkeypatch.setenv("ARENA_RESULTS_MAX_BYTES", bytes_env)
    assert results_max_age_secs() == want_age
    assert results_max_bytes() == want_bytes
