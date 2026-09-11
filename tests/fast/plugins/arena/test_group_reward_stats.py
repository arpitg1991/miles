"""Unit test for _group_reward_stats (per-group reward variance metrics).

Run: python -m pytest tests/fast/plugins/arena/test_group_reward_stats.py -v
"""

from __future__ import annotations

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.utils.types import Sample
from miles_plugins.arena.nats_arena.nats_rollout import _group_reward_stats

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def _group(group_index: int, rewards: list[float]) -> list[Sample]:
    out = []
    for i, r in enumerate(rewards):
        s = Sample()
        s.group_index = group_index
        s.index = i
        s.reward = r
        s.status = Sample.Status.COMPLETED
        s.metadata = {"mode": "full_trajectory", "task_id": f"t{group_index}", "gym_name": "g"}
        out.append(s)
    return out


def test_two_binary_groups():
    # group 0: [1, 0, 0, 0] -> mean 0.25, var 0.1875, std 0.4330
    # group 1: [1, 1, 0, 0] -> mean 0.5,  var 0.25,   std 0.5
    m = _group_reward_stats([_group(0, [1, 0, 0, 0]), _group(1, [1, 1, 0, 0])])
    assert m["rollout/group_reward_var/mean"] == pytest.approx((0.1875 + 0.25) / 2)
    assert m["rollout/group_reward_std/mean"] == pytest.approx((0.1875**0.5 + 0.5) / 2)
    assert m["rollout/group_reward_mean/std"] == pytest.approx(0.125)  # std of [0.25, 0.5]


def test_empty_returns_no_metrics():
    assert _group_reward_stats([]) == {}
