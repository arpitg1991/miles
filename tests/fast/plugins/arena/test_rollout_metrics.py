"""Unit tests for rollout metrics: truncated_ratio and group_metrics aggregation.

Covers:
  - truncated_ratio computation from nats_rollout.py (lines ~1498-1505)
  - compute_group_metrics_from_samples from rollout_metrics.py

Run: python -m pytest tests/fast/plugins/arena/test_rollout_metrics.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum

import pytest


# ---------------------------------------------------------------------------
# Lightweight Sample mock — mirrors miles.utils.types.Sample just enough for
# the metrics code paths under test.
# ---------------------------------------------------------------------------


@dataclass
class MockSample:
    """Minimal Sample stand-in with Status enum matching miles.utils.types.Sample."""

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    status: "MockSample.Status" = Status.COMPLETED
    reward: float | None = 0.0
    metadata: dict = field(default_factory=dict)
    remove_sample: bool = False
    weight_versions: list = field(default_factory=list)

    def get_reward_value(self, args) -> float:
        """Mirror miles.utils.types.Sample.get_reward_value."""
        reward_key = getattr(args, "reward_key", None)
        return self.reward if not reward_key else self.reward[reward_key]


@dataclass
class MockArgs:
    """Minimal args stand-in for the metrics code paths under test."""

    update_weights_interval: int = 1
    reward_key: str | None = None
    custom_reward_post_process_path: str | None = None


# ===========================================================================
# 1. truncated_ratio computation
# ===========================================================================


def _compute_truncated_ratio(all_data: list[list[MockSample]], Sample=MockSample) -> float:
    """Replicate the truncated_ratio logic from nats_rollout.py (~lines 1498-1505).

    The plugin logs this pre-filter value as rollout/truncated_ratio_prefilter;
    miles-native log_rollout_data owns the post-filter rollout/truncated_ratio.
    This is a faithful extraction so we can unit-test it without importing the
    full nats_rollout module (which pulls torch, miles, nats, etc.). Keep in
    sync with miles_plugins/arena/nats_arena/nats_rollout.py by hand.
    """
    truncated_count = sum(
        1 for g in all_data for s in g
        if getattr(s, "status", None) == Sample.Status.TRUNCATED
    )
    all_data_total = sum(len(g) for g in all_data)
    return truncated_count / max(all_data_total, 1)


class TestTruncatedRatio:
    def test_truncated_ratio_counts_truncated_status(self):
        """Groups with TRUNCATED samples are counted; ratio = truncated / total."""
        all_data = [
            [MockSample(status=MockSample.Status.TRUNCATED),
             MockSample(status=MockSample.Status.COMPLETED)],
            [MockSample(status=MockSample.Status.TRUNCATED),
             MockSample(status=MockSample.Status.TRUNCATED)],
            [MockSample(status=MockSample.Status.COMPLETED),
             MockSample(status=MockSample.Status.COMPLETED)],
        ]
        # 3 truncated out of 6 total
        ratio = _compute_truncated_ratio(all_data)
        assert ratio == pytest.approx(3.0 / 6.0)

    def test_truncated_ratio_zero_when_none_truncated(self):
        """All COMPLETED samples yield truncated_ratio = 0."""
        all_data = [
            [MockSample(status=MockSample.Status.COMPLETED) for _ in range(4)],
            [MockSample(status=MockSample.Status.COMPLETED) for _ in range(3)],
        ]
        ratio = _compute_truncated_ratio(all_data)
        assert ratio == 0.0

    def test_truncated_ratio_all_truncated(self):
        """When every sample is TRUNCATED, ratio should be 1.0."""
        all_data = [
            [MockSample(status=MockSample.Status.TRUNCATED) for _ in range(5)],
        ]
        ratio = _compute_truncated_ratio(all_data)
        assert ratio == 1.0

    def test_truncated_ratio_empty_data(self):
        """Empty all_data (no samples) should return 0 (no division by zero)."""
        ratio = _compute_truncated_ratio([])
        assert ratio == 0.0

    def test_truncated_ratio_mixed_statuses(self):
        """ABORTED and FAILED samples are not counted as truncated."""
        all_data = [
            [MockSample(status=MockSample.Status.TRUNCATED),
             MockSample(status=MockSample.Status.ABORTED),
             MockSample(status=MockSample.Status.FAILED),
             MockSample(status=MockSample.Status.COMPLETED)],
        ]
        # Only 1 truncated out of 4
        ratio = _compute_truncated_ratio(all_data)
        assert ratio == pytest.approx(0.25)


# ===========================================================================
# 2. compute_group_metrics_from_samples
# ===========================================================================


class TestComputeGroupMetrics:
    def test_compute_group_metrics_aggregates_from_metadata(self):
        """Samples with group_metrics in metadata get aggregated correctly."""
        from miles_plugins.arena.rollout_metrics import compute_group_metrics_from_samples

        # Two groups: first sample of each carries group_metrics
        samples = [
            MockSample(metadata={"group_metrics": {
                "group.reward.mean": 0.8,
                "group.reward.std": 0.1,
                "group.reward.max": 1.0,
                "n_samples_total": 4,
            }}),
            MockSample(metadata={}),  # second sample of group 0, no group_metrics
            MockSample(metadata={"group_metrics": {
                "group.reward.mean": 0.6,
                "group.reward.std": 0.2,
                "group.reward.max": 0.9,
                "n_samples_total": 4,
            }}),
            MockSample(metadata={}),  # second sample of group 1
        ]

        result = compute_group_metrics_from_samples(samples)

        # n_groups should equal 2 (the number of samples with group_metrics)
        assert result["n_groups"] == 2.0

        # n_samples_total starts with "n_samples" so it's summed
        assert result["n_samples_total"] == 8.0

        # Spread stats for within_group.reward_std (min/max/mean of [0.1, 0.2])
        assert result["within_group.reward_std.min"] == pytest.approx(0.1)
        assert result["within_group.reward_std.max"] == pytest.approx(0.2)
        assert result["within_group.reward_std.mean"] == pytest.approx(0.15)

        # Spread stats for within_group.reward_mean
        assert result["within_group.reward_mean.min"] == pytest.approx(0.6)
        assert result["within_group.reward_mean.max"] == pytest.approx(0.8)
        assert result["within_group.reward_mean.mean"] == pytest.approx(0.7)

        # reward.max uses max aggregation (ends with .max)
        assert result["reward.max"] == pytest.approx(1.0)

    def test_compute_group_metrics_empty_when_no_metadata(self):
        """Samples without group_metrics in metadata yield an empty dict."""
        from miles_plugins.arena.rollout_metrics import compute_group_metrics_from_samples

        samples = [
            MockSample(metadata={"other_key": 123}),
            MockSample(metadata={}),
            MockSample(metadata=None),
        ]
        result = compute_group_metrics_from_samples(samples)
        assert result == {}

    def test_compute_group_metrics_empty_samples_list(self):
        """Empty input returns empty dict."""
        from miles_plugins.arena.rollout_metrics import compute_group_metrics_from_samples

        assert compute_group_metrics_from_samples([]) == {}

    def test_compute_group_metrics_single_group(self):
        """Single group produces n_groups=1 and direct values."""
        from miles_plugins.arena.rollout_metrics import compute_group_metrics_from_samples

        samples = [
            MockSample(metadata={"group_metrics": {
                "group.reward.mean": 0.5,
                "group.reward.std": 0.3,
                "completion_length": 120.0,
            }}),
        ]
        result = compute_group_metrics_from_samples(samples)
        assert result["n_groups"] == 1.0
        # completion_length is averaged (only 1 value)
        assert result["completion_length"] == pytest.approx(120.0)

    def test_compute_group_metrics_count_keys_are_summed(self):
        """Keys ending with .count or starting with n_received/n_failed are summed."""
        from miles_plugins.arena.rollout_metrics import compute_group_metrics_from_samples

        samples = [
            MockSample(metadata={"group_metrics": {
                "tool_calls.count": 3,
                "n_received_total": 10,
                "n_failed_tasks": 2,
            }}),
            MockSample(metadata={"group_metrics": {
                "tool_calls.count": 5,
                "n_received_total": 8,
                "n_failed_tasks": 1,
            }}),
        ]
        result = compute_group_metrics_from_samples(samples)
        assert result["tool_calls.count"] == 8.0
        assert result["n_received_total"] == 18.0
        assert result["n_failed_tasks"] == 3.0

    def test_compute_group_metrics_min_keys_use_min(self):
        """Keys ending with .min are aggregated with min()."""
        from miles_plugins.arena.rollout_metrics import compute_group_metrics_from_samples

        samples = [
            MockSample(metadata={"group_metrics": {"group.reward.min": 0.1}}),
            MockSample(metadata={"group_metrics": {"group.reward.min": 0.3}}),
        ]
        result = compute_group_metrics_from_samples(samples)
        assert result["reward.min"] == pytest.approx(0.1)


# ===========================================================================
# 3. off-policy staleness (compute_off_policy_metrics wrapper)
# ===========================================================================


class TestOffPolicyMetrics:
    """compute_off_policy_metrics reads Sample.weight_versions (SGLang-tagged)."""

    def test_empty_samples_returns_empty(self):
        from miles_plugins.arena.rollout_metrics import compute_off_policy_metrics

        assert compute_off_policy_metrics(MockArgs(), [], rollout_id=5) == {}

    def test_on_policy_samples_report_zero_lag(self):
        """Samples generated by the current weights: off-policy round == 0."""
        from miles_plugins.arena.rollout_metrics import compute_off_policy_metrics

        # reference_step=0, interval=1 → rollout_weight_step = (1-1)*1 = 0.
        samples = [MockSample(weight_versions=["1"]), MockSample(weight_versions=["1"])]
        result = compute_off_policy_metrics(MockArgs(), samples, rollout_id=0)
        assert result["off_policy_round/mean"] == pytest.approx(0.0)
        assert result["off_policy_round/on_policy_frac"] == pytest.approx(1.0)
        assert result["off_policy_round/untagged_frac"] == pytest.approx(0.0)
        assert result["off_policy_round/current_step"] == pytest.approx(0.0)

    def test_stale_samples_report_positive_lag(self):
        """A sample from weight v1 consumed at step 3 is 3 rounds stale."""
        from miles_plugins.arena.rollout_metrics import compute_off_policy_metrics

        samples = [MockSample(weight_versions=["1"])]
        result = compute_off_policy_metrics(MockArgs(), samples, rollout_id=3)
        assert result["off_policy_round/mean"] == pytest.approx(3.0)
        assert result["off_policy_round/on_policy_frac"] == pytest.approx(0.0)

    def test_untagged_samples_tracked(self):
        """Samples with no weight_versions count toward untagged_frac."""
        from miles_plugins.arena.rollout_metrics import compute_off_policy_metrics

        samples = [MockSample(weight_versions=["2"]), MockSample(weight_versions=[])]
        result = compute_off_policy_metrics(MockArgs(), samples, rollout_id=3)
        assert result["off_policy_round/untagged_frac"] == pytest.approx(0.5)


# ===========================================================================
# 4. binary_reward (rollout/binary_reward gating + fraction)
# ===========================================================================


def _compute_binary_reward(args, all_samples: list[MockSample]) -> dict[str, float]:
    """Replicate the rollout/binary_reward block from nats_rollout.py (~lines 1694-1699).

    Faithful extraction so we can test the gating + fraction without importing
    the full nats_rollout module (torch/miles/nats). Keep in sync with
    miles_plugins/arena/nats_arena/nats_rollout.py by hand.
    """
    metrics: dict[str, float] = {}
    if getattr(args, "custom_reward_post_process_path", None) and "binary" in args.custom_reward_post_process_path:
        raw_rewards = [s.get_reward_value(args) for s in all_samples]
        if raw_rewards:
            metrics["rollout/binary_reward"] = sum(
                1.0 if r >= 1.0 else 0.0 for r in raw_rewards
            ) / len(raw_rewards)
    return metrics


class TestBinaryReward:
    """rollout/binary_reward is gated on a binarizing post-processor."""

    _BINARY_PATH = "miles_plugins.arena.nats_arena.reward_binary.binarize_reward"

    def test_not_emitted_without_binary_post_processor(self):
        args = MockArgs(custom_reward_post_process_path=None)
        samples = [MockSample(reward=1.0)]
        assert "rollout/binary_reward" not in _compute_binary_reward(args, samples)

    def test_not_emitted_for_nonbinary_post_processor(self):
        args = MockArgs(custom_reward_post_process_path="some.other.reward_fn")
        samples = [MockSample(reward=1.0)]
        assert "rollout/binary_reward" not in _compute_binary_reward(args, samples)

    def test_fraction_of_samples_at_or_above_one(self):
        args = MockArgs(custom_reward_post_process_path=self._BINARY_PATH)
        samples = [
            MockSample(reward=1.0),
            MockSample(reward=0.0),
            MockSample(reward=1.0),
            MockSample(reward=0.5),
        ]
        result = _compute_binary_reward(args, samples)
        assert result["rollout/binary_reward"] == pytest.approx(0.5)

    def test_all_zero_reward(self):
        args = MockArgs(custom_reward_post_process_path=self._BINARY_PATH)
        samples = [MockSample(reward=0.0), MockSample(reward=0.2)]
        result = _compute_binary_reward(args, samples)
        assert result["rollout/binary_reward"] == pytest.approx(0.0)


# ===========================================================================
# Run standalone
# ===========================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
