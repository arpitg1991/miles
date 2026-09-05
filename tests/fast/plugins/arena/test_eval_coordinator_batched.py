"""Unit tests for the batched n_samples model in eval_coordinator.

The eval coordinator publishes ONE NATS task per prompt carrying
``n_samples = n_samples_per_eval_prompt``; the eval gym pod fans out the whole
group of N rollouts internally (mirroring the trainer). These tests pin the
publish-side contract: the task message carries the right ``n_samples`` and a
plain instance id with no ``-s<idx>`` replication suffix.
"""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from miles_plugins.arena.nats_arena.eval_coordinator import _row_to_task
from miles_plugins.arena.nats_arena.message_format import parse_result, serialize_task


def _row():
    return {
        "id": "task-123",
        "lakefs_uri": "lakefs://arena-mcp/main/tasks/task-123",
        "lakefs_commit_id": "abc123",
        "label": "the-answer",
    }


def test_row_to_task_carries_n_samples():
    task = _row_to_task(_row(), {"name": "ds", "gym_name": "mygym"}, 0, n_samples=4)
    assert task["n_samples"] == 4
    # Base instance id — no -s<idx> replication suffix.
    assert task["id"] == "task-123"
    assert "-s" not in task["id"]


def test_row_to_task_defaults_to_single_sample():
    task = _row_to_task(_row(), {"name": "ds", "gym_name": "mygym"}, 0)
    assert task["n_samples"] == 1


def test_n_samples_survives_serialize_roundtrip():
    task = _row_to_task(_row(), {"name": "ds", "gym_name": "mygym"}, 0, n_samples=8)
    # serialize_task gzips + json-encodes; parse_result is the inverse used on
    # the gym side. n_samples must survive the wire format the gym reads.
    decoded = parse_result(serialize_task(task))
    assert decoded["n_samples"] == 8
    assert decoded["id"] == "task-123"
    assert decoded["metadata"]["eval"] is True
    assert decoded["metadata"]["eval_dataset"] == "ds"
