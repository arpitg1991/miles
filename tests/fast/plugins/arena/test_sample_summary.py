"""--arena-sample-summary-dir: one JSONL row per training sample, no tensors.

``_write_sample_summary`` decodes the ADR-0011 packed ``index`` with
``_SEGMENT_INDEX_STRIDE`` and writes through a ``.tmp`` rename. It is a
diagnostics path, so a failure logs a warning and never raises.

Run: python -m pytest tests/fast/plugins/arena/test_sample_summary.py -v
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from miles.utils.types import Sample
from miles_plugins.arena.nats_arena.nats_rollout import (
    _SEGMENT_INDEX_STRIDE,
    _add_arena_arguments,
    _write_sample_summary,
)


def _samples() -> list[Sample]:
    return [
        Sample(
            index=5 + 2 * _SEGMENT_INDEX_STRIDE,
            group_index=1,
            rollout_id=5,
            tokens=[1, 2, 3, 4, 5],
            response_length=3,
            reward=0.5,
            status=Sample.Status.COMPLETED,
            metadata={"raw_reward": 0.5, "agent_stop_reason": "submit"},
        ),
        Sample(
            index=6,
            group_index=1,
            tokens=[1, 2, 3],
            response_length=1,
            reward=None,
            status=Sample.Status.TRUNCATED,
        ),
        Sample(
            index=7,
            group_index=1,
            tokens=[1, 2],
            response_length=1,
            reward=1,
            remove_sample=True,
            status=Sample.Status.COMPLETED,
        ),
    ]


def test_rows_decode_segment_index(tmp_path: Path) -> None:
    _write_sample_summary(SimpleNamespace(arena_sample_summary_dir=tmp_path), 42, _samples())

    rows = [json.loads(line) for line in (tmp_path / "rollout_42.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert not list(tmp_path.glob("*.tmp"))

    first, second, third = rows
    assert first["rollout_id"] == 42
    assert first["index"] == 5 + 2 * _SEGMENT_INDEX_STRIDE
    assert first["segment_k"] == 2
    assert first["episode_index"] == 5
    assert first["sample_rollout_id"] == 5
    assert "span_tokens" not in first
    assert "has_advantage_scale" not in first
    assert first["reward"] == 0.5
    assert first["raw_reward"] == 0.5
    assert first["total_length"] == 5
    assert first["response_length"] == 3
    assert first["status"] == "completed"
    assert first["stop_reason"] == "submit"

    assert second["reward"] is None
    assert second["raw_reward"] is None
    assert second["segment_k"] == 0
    assert second["stop_reason"] is None
    assert second["status"] == "truncated"

    assert third["reward"] == 1.0
    assert third["remove_sample"] is True


def test_failure_logs_and_does_not_raise(tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    # mkdir on a path that is a regular file fails; the helper must swallow it.
    _write_sample_summary(SimpleNamespace(arena_sample_summary_dir=blocker), 1, _samples())
    assert not (blocker / "rollout_1.jsonl").exists()


def test_argparse_default_is_off() -> None:
    parser = argparse.ArgumentParser()
    _add_arena_arguments(parser)
    assert parser.parse_args([]).arena_sample_summary_dir is None
    assert parser.parse_args(["--arena-sample-summary-dir", "/x"]).arena_sample_summary_dir == "/x"
