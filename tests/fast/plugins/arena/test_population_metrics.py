"""Population learning metrics (rollout/population/*) and the verifier regression flag.

The kept-batch mean reward excludes every all-correct group the dynamic-sampling
filter drops, so it stays flat while the policy solves more chains. These tests
pin the pre-filter metrics that follow the whole population, and the
``grader_metadata`` -> ``metadata["regression"]`` path that feeds them.
"""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci
from tests.fast.plugins.arena.test_multi_segment_episodes import (
    _args,
    _result,
    _three_segments,
    _traj,
)

from miles.utils.types import Sample
from miles_plugins.arena.nats_arena.nats_rollout import (
    _grader_regression,
    _population_metrics,
    _result_to_episodes_full_trajectory,
)

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def test_grader_regression_reads_the_chain_level_verifier_keys():
    # agentic-debt live-baseline: no_regression is the mean of the per-step flags.
    assert _grader_regression({"no_regression": 1.0, "broken": 0.0}) is False
    assert _grader_regression({"no_regression": 0.5, "broken": 0.02}) is True
    # A step-1 regression carries penalty 0 but broken > 0.
    assert _grader_regression({"broken": 0.0036}) is True
    assert _grader_regression({"broken": 0.0}) is False
    # A gym without these keys reports nothing, not a false clean chain.
    assert _grader_regression({"reward": 1.0}) is None
    assert _grader_regression(None) is None


def test_regression_flag_lands_on_every_segment():
    clean = _traj(_three_segments(), reward=1.0, grader_metadata={"no_regression": 1.0, "broken": 0.0})
    broke = _traj(_three_segments(), reward=0.0, grader_metadata={"no_regression": 0.5, "broken": 0.1})
    plain = _traj(_three_segments(), reward=1.0)
    episodes = _result_to_episodes_full_trajectory(_result([clean, broke, plain]), tokenizer=None, args=_args("all"))
    assert [s.metadata["regression"] for s in episodes[0]] == [False, False, False]
    assert [s.metadata["regression"] for s in episodes[1]] == [True, True, True]
    assert all("regression" not in s.metadata for s in episodes[2])


_NEXT_EPISODE = iter(range(1, 10_000))


def _episode(reward: float, regression: bool | None = None, n_segments: int = 2) -> list[Sample]:
    """One multi-segment episode; ``rollout_id`` is the episode identity under ``all`` mode."""
    rid = next(_NEXT_EPISODE)
    segs = []
    for k in range(n_segments):
        s = Sample(tokens=[1, 2], response_length=1, reward=reward, rollout_id=rid, group_index=0)
        s.metadata = {"segment": k, "n_segments": n_segments}
        if regression is not None:
            s.metadata["regression"] = regression
        segs.append(s)
    return segs


def test_population_metrics_count_dropped_groups_once_per_episode():
    solved = [s for r in (1.0, 1.0) for s in _episode(r, False)]  # all-1.0: the filter drops it
    mixed = [s for r, g in ((1.0, False), (0.0, True)) for s in _episode(r, g)]
    m = _population_metrics([solved, mixed])
    assert m["rollout/population/episodes"] == 4
    assert m["rollout/population/mean_reward"] == 0.75
    assert m["rollout/population/chains_at_one_frac"] == 0.75
    assert m["rollout/population/regression_frac"] == 0.25
    assert m["rollout/population/all_one_groups_frac"] == 0.5
    assert m["rollout/population/zero_variance_groups_frac"] == 0.5


def test_population_metrics_omit_regression_without_verifier_keys():
    m = _population_metrics([[s for r in (1.0, 0.5) for s in _episode(r)]])
    assert "rollout/population/regression_frac" not in m
    assert m["rollout/population/chains_at_one_frac"] == 0.5
    assert _population_metrics([]) == {}
