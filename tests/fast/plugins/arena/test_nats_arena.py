"""Comprehensive unit tests for the miles_plugins.arena.nats_arena modules.

Covers:
  - message_format.py: build_task_message, serialize_task/parse_result round-trip,
    extract_trajectories, sample_to_task
  - mixture_controller.py: MixtureController initialization, record_completion,
    maybe_adjust weight corrections, state_dict/load_state_dict, thread safety
  - rollout_timing_tracker.py: record_publish/record_completion, get_stats p50/p95,
    window eviction, multi-gym tracking
  - reward_binary.py: binarize_reward (requires torch)
  - data_source.py: _read_manifest, ArenaDataSourceWithBuffer.get_samples
  - eval_coordinator.py: _row_to_task edge cases, env var parsing, _handle_result_msg
    dedup/DLQ state machine
  - gym_autoscaler.py: desired_replicas calculation without K8s

Run: python -m pytest tests/fast/plugins/arena/test_nats_arena.py -v
"""

from __future__ import annotations

import gzip
import json
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])


# ===========================================================================
# 1. message_format.py
# ===========================================================================


class TestBuildTaskMessage:
    def test_basic_fields(self):
        from miles_plugins.arena.nats_arena.message_format import build_task_message

        msg = build_task_message(
            "task-1",
            lakefs_uri="lakefs://repo/branch/tasks/task-1/",
            lakefs_commit_id="deadbeef",
        )
        assert msg["id"] == "task-1"
        assert msg["lakefs_uri"] == "lakefs://repo/branch/tasks/task-1/"
        assert msg["lakefs_commit_id"] == "deadbeef"
        assert msg["n_samples"] == 1
        assert msg["metadata"] == {}
        assert "session" not in msg

    def test_n_samples(self):
        from miles_plugins.arena.nats_arena.message_format import build_task_message

        msg = build_task_message(
            "t", lakefs_uri="x", lakefs_commit_id="y", n_samples=8
        )
        assert msg["n_samples"] == 8

    def test_session_field(self):
        from miles_plugins.arena.nats_arena.message_format import build_task_message

        msg = build_task_message(
            "t", lakefs_uri="x", lakefs_commit_id="y", session="sess-42"
        )
        assert msg["session"] == "sess-42"

    def test_session_none_not_included(self):
        from miles_plugins.arena.nats_arena.message_format import build_task_message

        msg = build_task_message(
            "t", lakefs_uri="x", lakefs_commit_id="y", session=None
        )
        assert "session" not in msg

    def test_gym_name_in_metadata(self):
        from miles_plugins.arena.nats_arena.message_format import build_task_message

        msg = build_task_message(
            "t", lakefs_uri="x", lakefs_commit_id="y", gym_name="my_gym"
        )
        assert msg["metadata"]["gym_name"] == "my_gym"

    def test_metadata_not_mutated_if_none(self):
        from miles_plugins.arena.nats_arena.message_format import build_task_message

        msg = build_task_message(
            "t", lakefs_uri="x", lakefs_commit_id="y", metadata=None
        )
        assert msg["metadata"] == {}

    def test_custom_metadata_merged_with_gym_name(self):
        from miles_plugins.arena.nats_arena.message_format import build_task_message

        msg = build_task_message(
            "t",
            lakefs_uri="x",
            lakefs_commit_id="y",
            metadata={"foo": "bar"},
            gym_name="g",
        )
        assert msg["metadata"]["foo"] == "bar"
        assert msg["metadata"]["gym_name"] == "g"


class TestSerializeParseRoundTrip:
    def test_roundtrip_plain_message(self):
        from miles_plugins.arena.nats_arena.message_format import (
            build_task_message,
            parse_result,
            serialize_task,
        )

        original = build_task_message(
            "task-99",
            lakefs_uri="lakefs://r/b/tasks/task-99/",
            lakefs_commit_id="abc",
            n_samples=4,
            metadata={"k": "v"},
            session="s1",
        )
        data = serialize_task(original)
        # Should be gzip compressed
        assert data[:2] == b"\x1f\x8b"
        decoded = parse_result(data)
        assert decoded == original

    def test_parse_result_uncompressed(self):
        from miles_plugins.arena.nats_arena.message_format import parse_result

        payload = {"task_id": "t1", "status": "ok", "trajectories": []}
        raw = json.dumps(payload).encode("utf-8")
        assert parse_result(raw) == payload

    def test_parse_result_compressed(self):
        from miles_plugins.arena.nats_arena.message_format import parse_result

        payload = {"task_id": "t2", "trajectories": [{"reward": 1.0}]}
        raw = gzip.compress(json.dumps(payload).encode("utf-8"))
        assert parse_result(raw) == payload

    def test_unicode_survives_roundtrip(self):
        from miles_plugins.arena.nats_arena.message_format import parse_result, serialize_task

        msg = {"id": "t", "content": "Hello 世界"}
        decoded = parse_result(serialize_task(msg))
        assert decoded["content"] == "Hello 世界"


class TestExtractTrajectories:
    def test_returns_trajectories_list(self):
        from miles_plugins.arena.nats_arena.message_format import extract_trajectories

        result = {
            "task_id": "t1",
            "status": "ok",
            "trajectories": [{"reward": 0.5}, {"reward": 1.0}],
        }
        trajs = extract_trajectories(result)
        assert len(trajs) == 2
        assert trajs[0]["reward"] == 0.5

    def test_empty_trajectories(self):
        from miles_plugins.arena.nats_arena.message_format import extract_trajectories

        result = {"task_id": "t1", "status": "error", "trajectories": []}
        assert extract_trajectories(result) == []

    def test_missing_trajectories_key(self):
        from miles_plugins.arena.nats_arena.message_format import extract_trajectories

        result = {"task_id": "t1", "status": "error"}
        assert extract_trajectories(result) == []


class TestSampleToTask:
    def test_basic_conversion(self):
        from miles_plugins.arena.nats_arena.message_format import sample_to_task

        sample = SimpleNamespace(
            metadata={
                "instance_id": "inst-1",
                "lakefs_uri": "lakefs://r/b/tasks/inst-1/",
                "lakefs_commit_id": "sha1",
            },
            index=0,
            group_index=0,
        )
        task = sample_to_task(sample, rollout_id=5, group_index=3, n_samples=2, gym_name="g1")
        assert task["id"] == "inst-1"
        assert task["lakefs_uri"] == "lakefs://r/b/tasks/inst-1/"
        assert task["n_samples"] == 2
        assert task["metadata"]["rollout_id"] == 5
        assert task["metadata"]["group_index"] == 3
        assert task["metadata"]["gym_name"] == "g1"

    def test_missing_lakefs_uri_raises(self):
        from miles_plugins.arena.nats_arena.message_format import sample_to_task

        sample = SimpleNamespace(
            metadata={"instance_id": "x"},
            index=0,
            group_index=0,
        )
        with pytest.raises(ValueError, match="missing lakefs_uri"):
            sample_to_task(sample)

    def test_string_metadata_parsed(self):
        from miles_plugins.arena.nats_arena.message_format import sample_to_task

        sample = SimpleNamespace(
            metadata=json.dumps({
                "instance_id": "t",
                "lakefs_uri": "lakefs://r/b/tasks/t/",
                "lakefs_commit_id": "c",
            }),
            index=0,
            group_index=0,
        )
        task = sample_to_task(sample)
        assert task["id"] == "t"

    def test_invalid_string_metadata_treated_as_empty(self):
        from miles_plugins.arena.nats_arena.message_format import sample_to_task

        sample = SimpleNamespace(
            metadata="not-valid-json",
            index=0,
            group_index=0,
        )
        with pytest.raises(ValueError, match="missing lakefs_uri"):
            sample_to_task(sample)

    def test_session_propagated(self):
        from miles_plugins.arena.nats_arena.message_format import sample_to_task

        sample = SimpleNamespace(
            metadata={
                "instance_id": "t",
                "lakefs_uri": "lakefs://r/b/tasks/t/",
                "lakefs_commit_id": "c",
            },
            index=0,
            group_index=0,
        )
        task = sample_to_task(sample, session="my-session")
        assert task["session"] == "my-session"


# ===========================================================================
# 2. mixture_controller.py
# ===========================================================================


class TestMixtureController:
    def test_initialization_normalizes_weights(self):
        from miles_plugins.arena.nats_arena.mixture_controller import MixtureController

        mc = MixtureController(target_ratios={"a": 3.0, "b": 1.0})
        assert abs(mc.target_ratios["a"] - 0.75) < 1e-9
        assert abs(mc.target_ratios["b"] - 0.25) < 1e-9
        # current_weights initialized to target_ratios
        assert mc.current_weights == mc.target_ratios

    def test_record_completion_increments(self):
        from miles_plugins.arena.nats_arena.mixture_controller import MixtureController

        mc = MixtureController(target_ratios={"a": 1.0, "b": 1.0})
        mc.record_completion("a")
        mc.record_completion("a")
        mc.record_completion("b")
        assert mc.gym_completions["a"] == 2
        assert mc.gym_completions["b"] == 1
        assert mc.total_completions == 3

    def test_maybe_adjust_returns_none_before_interval(self):
        from miles_plugins.arena.nats_arena.mixture_controller import MixtureController

        mc = MixtureController(
            target_ratios={"a": 1.0}, adjustment_interval=9999
        )
        for _ in range(20):
            mc.record_completion("a")
        assert mc.maybe_adjust() is None

    def test_maybe_adjust_returns_none_with_few_completions(self):
        from miles_plugins.arena.nats_arena.mixture_controller import MixtureController

        mc = MixtureController(
            target_ratios={"a": 1.0}, adjustment_interval=0.0
        )
        mc.record_completion("a")  # only 1 completion, need >= 10
        assert mc.maybe_adjust() is None

    def test_maybe_adjust_produces_weights_after_interval(self):
        from miles_plugins.arena.nats_arena.mixture_controller import MixtureController

        mc = MixtureController(
            target_ratios={"a": 1.0, "b": 1.0},
            adjustment_interval=0.0,
            smoothing=1.0,  # full correction, no EMA
        )
        # Gym a is over-represented
        for _ in range(8):
            mc.record_completion("a")
        for _ in range(2):
            mc.record_completion("b")
        new_weights = mc.maybe_adjust()
        assert new_weights is not None
        # b should get boosted since it was under-represented
        assert new_weights["b"] > new_weights["a"]

    def test_weight_corrections_clamped(self):
        from miles_plugins.arena.nats_arena.mixture_controller import MixtureController

        mc = MixtureController(
            target_ratios={"a": 1.0, "b": 1.0},
            adjustment_interval=0.0,
            smoothing=1.0,
            correction_bounds=(0.5, 2.0),
        )
        # Extreme imbalance: all completions from a
        for _ in range(100):
            mc.record_completion("a")
        new_weights = mc.maybe_adjust()
        assert new_weights is not None
        # Even with extreme imbalance, correction is bounded
        ratio = new_weights["b"] / new_weights["a"]
        # max correction of 2.0: b gets target*2.0 vs a gets target*0.5
        # so max ratio is 2.0/0.5 = 4.0, but after normalization it's bounded
        assert ratio <= 4.1  # small float tolerance

    def test_state_dict_load_state_dict_roundtrip(self):
        from miles_plugins.arena.nats_arena.mixture_controller import MixtureController

        mc = MixtureController(target_ratios={"a": 0.7, "b": 0.3})
        mc.current_weights = {"a": 0.6, "b": 0.4}
        mc.adjustment_count = 5

        state = mc.state_dict()
        assert state["current_weights"] == {"a": 0.6, "b": 0.4}
        assert state["adjustment_count"] == 5

        mc2 = MixtureController(target_ratios={"a": 0.7, "b": 0.3})
        mc2.load_state_dict(state)
        assert mc2.current_weights == {"a": 0.6, "b": 0.4}
        assert mc2.adjustment_count == 5

    def test_thread_safety_record_completion(self):
        from miles_plugins.arena.nats_arena.mixture_controller import MixtureController

        mc = MixtureController(target_ratios={"a": 1.0, "b": 1.0})
        n = 1000

        def record_a():
            for _ in range(n):
                mc.record_completion("a")

        def record_b():
            for _ in range(n):
                mc.record_completion("b")

        t1 = threading.Thread(target=record_a)
        t2 = threading.Thread(target=record_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert mc.total_completions == 2 * n
        assert mc.gym_completions["a"] == n
        assert mc.gym_completions["b"] == n

    def test_adjustment_resets_counters(self):
        from miles_plugins.arena.nats_arena.mixture_controller import MixtureController

        mc = MixtureController(
            target_ratios={"a": 1.0, "b": 1.0},
            adjustment_interval=0.0,
        )
        for _ in range(10):
            mc.record_completion("a")
        mc.maybe_adjust()
        # Counters should be reset
        assert mc.total_completions == 0
        assert mc.gym_completions["a"] == 0

    def test_get_observed_snapshot_before_adjustment(self):
        from miles_plugins.arena.nats_arena.mixture_controller import MixtureController

        mc = MixtureController(target_ratios={"a": 0.6, "b": 0.4})
        targets, observed = mc.get_observed_snapshot()
        assert abs(targets["a"] - 0.6) < 1e-9
        assert observed == {}  # no adjustment yet

    def test_get_weight_snapshot(self):
        from miles_plugins.arena.nats_arena.mixture_controller import MixtureController

        mc = MixtureController(target_ratios={"a": 0.6, "b": 0.4})
        targets, weights = mc.get_weight_snapshot()
        # Before any adjustment, current_weights == target_ratios
        assert targets == weights

    def test_all_zero_target_ratios_handled(self):
        from miles_plugins.arena.nats_arena.mixture_controller import _normalize

        result = _normalize({"a": 0.0, "b": 0.0})
        assert abs(result["a"] - 0.5) < 1e-9
        assert abs(result["b"] - 0.5) < 1e-9


# ===========================================================================
# 3. rollout_timing_tracker.py
# ===========================================================================


class TestRolloutTimingTracker:
    def test_record_publish_and_completion(self):
        from miles_plugins.arena.nats_arena.rollout_timing_tracker import RolloutTimingTracker

        tracker = RolloutTimingTracker(short_window_size=10, long_window_size=100)
        tracker.record_publish("t1", "gym_a", ts=100.0)
        tracker.record_completion("t1", ts=110.0)
        stats = tracker.get_stats("gym_a")
        assert stats is not None
        assert stats.p50_short == 10.0
        assert stats.p95_short == 10.0
        assert stats.sample_count_short == 1
        assert stats.sample_count_long == 1

    def test_get_stats_returns_none_for_unknown_gym(self):
        from miles_plugins.arena.nats_arena.rollout_timing_tracker import RolloutTimingTracker

        tracker = RolloutTimingTracker(short_window_size=5, long_window_size=50)
        assert tracker.get_stats("nonexistent") is None

    def test_p50_p95_multiple_samples(self):
        from miles_plugins.arena.nats_arena.rollout_timing_tracker import RolloutTimingTracker

        tracker = RolloutTimingTracker(short_window_size=100, long_window_size=1000)
        # Record 100 tasks with durations 1..100
        for i in range(1, 101):
            tracker.record_publish(f"t{i}", "g", ts=0.0)
            tracker.record_completion(f"t{i}", ts=float(i))

        stats = tracker.get_stats("g")
        assert stats is not None
        # p50 should be around 50
        assert 49.0 <= stats.p50_short <= 51.0
        # p95 should be around 95
        assert 94.0 <= stats.p95_short <= 96.0
        assert stats.sample_count_short == 100

    def test_window_eviction_short(self):
        from miles_plugins.arena.nats_arena.rollout_timing_tracker import RolloutTimingTracker

        tracker = RolloutTimingTracker(short_window_size=5, long_window_size=50)
        # Record more than the short window size
        for i in range(10):
            tracker.record_publish(f"t{i}", "g", ts=0.0)
            tracker.record_completion(f"t{i}", ts=float(i + 1))

        stats = tracker.get_stats("g")
        assert stats is not None
        # Short window holds only the last 5 entries (durations 6,7,8,9,10)
        assert stats.sample_count_short == 5
        # Long window holds all 10
        assert stats.sample_count_long == 10

    def test_multi_gym_tracking(self):
        from miles_plugins.arena.nats_arena.rollout_timing_tracker import RolloutTimingTracker

        tracker = RolloutTimingTracker(short_window_size=10, long_window_size=100)
        tracker.record_publish("t1", "gym_x", ts=0.0)
        tracker.record_completion("t1", ts=5.0)
        tracker.record_publish("t2", "gym_y", ts=0.0)
        tracker.record_completion("t2", ts=20.0)

        stats_x = tracker.get_stats("gym_x")
        stats_y = tracker.get_stats("gym_y")
        assert stats_x.p50_short == 5.0
        assert stats_y.p50_short == 20.0
        assert sorted(tracker.all_gyms()) == ["gym_x", "gym_y"]

    def test_duplicate_completion_ignored(self):
        from miles_plugins.arena.nats_arena.rollout_timing_tracker import RolloutTimingTracker

        tracker = RolloutTimingTracker(short_window_size=10, long_window_size=100)
        tracker.record_publish("t1", "g", ts=0.0)
        tracker.record_completion("t1", ts=10.0)
        # Second completion for same task: in_flight was already popped
        tracker.record_completion("t1", ts=20.0)
        stats = tracker.get_stats("g")
        # Should only have 1 sample
        assert stats.sample_count_short == 1

    def test_in_flight_count(self):
        from miles_plugins.arena.nats_arena.rollout_timing_tracker import RolloutTimingTracker

        tracker = RolloutTimingTracker(short_window_size=10, long_window_size=100)
        tracker.record_publish("t1", "g", ts=0.0)
        tracker.record_publish("t2", "g", ts=0.0)
        assert tracker.in_flight_count() == 2
        tracker.record_completion("t1", ts=5.0)
        assert tracker.in_flight_count() == 1

    def test_state_dict_load_state_dict(self):
        from miles_plugins.arena.nats_arena.rollout_timing_tracker import RolloutTimingTracker

        tracker = RolloutTimingTracker(short_window_size=10, long_window_size=100)
        for i in range(5):
            tracker.record_publish(f"t{i}", "g", ts=0.0)
            tracker.record_completion(f"t{i}", ts=float(i + 1))

        state = tracker.state_dict()
        tracker2 = RolloutTimingTracker(short_window_size=10, long_window_size=100)
        tracker2.load_state_dict(state)

        stats = tracker2.get_stats("g")
        assert stats is not None
        assert stats.sample_count_short == 5

    def test_growth_ratio_increasing(self):
        from miles_plugins.arena.nats_arena.rollout_timing_tracker import RolloutTimingTracker

        # Use a tracker where short window fills with longer durations than
        # the long window baseline. Long window must be large enough that the
        # few recent slow values don't dominate its p95.
        tracker = RolloutTimingTracker(short_window_size=5, long_window_size=100)
        # Long window: many quick rollouts (fills most of the 100-slot window)
        for i in range(95):
            tracker.record_publish(f"old{i}", "g", ts=0.0)
            tracker.record_completion(f"old{i}", ts=10.0)  # all 10s
        # Short window: overwrite with long rollouts
        for i in range(5):
            tracker.record_publish(f"new{i}", "g", ts=0.0)
            tracker.record_completion(f"new{i}", ts=100.0)  # all 100s

        stats = tracker.get_stats("g")
        assert stats is not None
        # Short p95 (100s) should be higher than long p95 (~10s with some 100s mixed in)
        assert stats.growth_ratio > 1.0

    def test_invalid_window_sizes(self):
        from miles_plugins.arena.nats_arena.rollout_timing_tracker import RolloutTimingTracker

        with pytest.raises(ValueError):
            RolloutTimingTracker(short_window_size=0, long_window_size=100)
        with pytest.raises(ValueError):
            RolloutTimingTracker(short_window_size=100, long_window_size=50)

    def test_empty_task_id_ignored(self):
        from miles_plugins.arena.nats_arena.rollout_timing_tracker import RolloutTimingTracker

        tracker = RolloutTimingTracker(short_window_size=10, long_window_size=100)
        tracker.record_publish("", "g", ts=0.0)
        tracker.record_completion("", ts=5.0)
        assert tracker.in_flight_count() == 0


# ===========================================================================
# 4. reward_binary.py (requires torch)
# ===========================================================================


class TestBinarizeReward:
    @pytest.fixture(autouse=True)
    def _skip_without_torch(self):
        pytest.importorskip("torch")

    def _make_sample(self, reward: float, metadata: dict | None = None):
        """Create a mock Sample with get_reward_value returning the given reward."""
        sample = MagicMock()
        sample.get_reward_value = MagicMock(return_value=reward)
        sample.metadata = metadata if metadata is not None else {}
        return sample

    def _make_args(self, n_samples: int = 2, grpo_std: bool = False):
        return SimpleNamespace(
            n_samples_per_prompt=n_samples,
            grpo_std_normalization=grpo_std,
        )

    def test_all_zero_rewards(self):
        from miles_plugins.arena.nats_arena.reward_binary import binarize_reward

        args = self._make_args(n_samples=2)
        samples = [self._make_sample(0.0) for _ in range(4)]
        raw, processed = binarize_reward(args, samples)
        # All zero -> binary all 0 -> mean=0 per group -> normalized all 0
        assert all(r == 0.0 for r in raw)
        assert all(r == 0.0 for r in processed)

    def test_all_perfect_rewards(self):
        from miles_plugins.arena.nats_arena.reward_binary import binarize_reward

        args = self._make_args(n_samples=2)
        samples = [self._make_sample(1.0) for _ in range(4)]
        raw, processed = binarize_reward(args, samples)
        assert all(r == 1.0 for r in raw)
        # All 1.0 -> binary [1,1,...] -> mean=1 -> normalized all 0
        assert all(r == 0.0 for r in processed)

    def test_mixed_rewards(self):
        from miles_plugins.arena.nats_arena.reward_binary import binarize_reward

        args = self._make_args(n_samples=2)
        # Group 0: [0.5, 1.0] -> binary [0, 1] -> mean=0.5 -> [-0.5, 0.5]
        samples = [
            self._make_sample(0.5),
            self._make_sample(1.0),
        ]
        raw, processed = binarize_reward(args, samples)
        assert raw == [0.5, 1.0]
        assert abs(processed[0] - (-0.5)) < 1e-6
        assert abs(processed[1] - 0.5) < 1e-6

    def test_grpo_std_normalization(self):
        from math import sqrt

        from miles_plugins.arena.nats_arena.reward_binary import binarize_reward

        args = self._make_args(n_samples=4, grpo_std=True)
        # 2 perfect, 2 not -> binary [1,1,0,0] -> mean=0.5 -> [0.5,0.5,-0.5,-0.5].
        # torch's Tensor.std defaults to the UNBIASED (Bessel, n-1) estimator,
        # so std = sqrt(4*0.25/3) = sqrt(1/3) and each value normalizes to
        # +/- 0.5/(sqrt(1/3)+1e-6) ~= 0.866, NOT +/-1 (the population-std
        # value the upstream AGISlime test wrongly asserted — it failed
        # against the original code too; the ported code matches the original
        # behavior bit-for-bit).
        samples = [
            self._make_sample(1.0),
            self._make_sample(1.0),
            self._make_sample(0.3),
            self._make_sample(0.0),
        ]
        raw, processed = binarize_reward(args, samples)
        expected = 0.5 / (sqrt(1.0 / 3.0) + 1e-6)
        assert abs(abs(processed[0]) - expected) < 1e-4
        assert abs(abs(processed[2]) - expected) < 1e-4

    def test_binary_reward_metadata_stamped(self):
        from miles_plugins.arena.nats_arena.reward_binary import binarize_reward

        args = self._make_args(n_samples=2)
        s1 = self._make_sample(0.5, metadata={})
        s2 = self._make_sample(1.0, metadata={})
        binarize_reward(args, [s1, s2])
        assert s1.metadata["binary_reward"] == 0.0
        assert s2.metadata["binary_reward"] == 1.0

    def test_non_dict_metadata_converted(self):
        from miles_plugins.arena.nats_arena.reward_binary import binarize_reward

        args = self._make_args(n_samples=1)
        s = self._make_sample(1.0)
        s.metadata = None  # not a dict
        binarize_reward(args, [s])
        assert isinstance(s.metadata, dict)
        assert s.metadata["binary_reward"] == 1.0


class TestRenormalizeAfterMask:
    """renormalize_after_mask + binarize_renormalize_after_mask (ADR-0030 gap)."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self):
        # reward_binary imports torch at module level (via binarize_reward); skip
        # rather than hard-error in a torch-less env, matching sibling test classes.
        pytest.importorskip("torch")

    def _make_sample(self, reward: float, remove: bool = False, resp_len: int = 1):
        sample = MagicMock()
        sample.get_reward_value = MagicMock(return_value=reward)
        sample.metadata = {}
        sample.remove_sample = remove
        sample.response_length = resp_len
        return sample

    def _make_args(self, n_samples: int, grpo_std: bool = True):
        return SimpleNamespace(
            n_samples_per_prompt=n_samples,
            grpo_std_normalization=grpo_std,
        )

    def test_rejects_mismatched_lengths(self):
        from miles_plugins.arena.nats_arena.reward_binary import renormalize_after_mask

        with pytest.raises(ValueError):
            renormalize_after_mask([0.0, 0.0], [self._make_sample(0.0)], 2, True)

    def test_rejects_indivisible_group(self):
        from miles_plugins.arena.nats_arena.reward_binary import renormalize_after_mask

        samples = [self._make_sample(0.0) for _ in range(3)]
        with pytest.raises(ValueError):
            renormalize_after_mask([0.0] * 3, samples, 2, True)

    def test_single_survivor_zeroed(self):
        from miles_plugins.arena.nats_arena.reward_binary import renormalize_after_mask

        # Group of 2, one removed -> 1 survivor -> zeroed; removed keeps value.
        samples = [self._make_sample(1.5), self._make_sample(-1.5, remove=True)]
        out = renormalize_after_mask([1.5, -1.5], samples, 2, True)
        assert out[0] == 0.0
        assert out[1] == -1.5

    def test_zero_length_disqualifies_survivor(self):
        from miles_plugins.arena.nats_arena.reward_binary import renormalize_after_mask

        # response_length == 0 -> not a survivor; no survivors -> values unchanged.
        samples = [self._make_sample(1.0, resp_len=0), self._make_sample(-1.0, resp_len=0)]
        out = renormalize_after_mask([1.0, -1.0], samples, 2, True)
        assert out == [1.0, -1.0]

    def test_removes_positive_bias(self):
        from miles_plugins.arena.nats_arena.reward_binary import renormalize_after_mask

        # Group of 4: three real (advantages carry a positive-mean bias from the
        # removed sample being in the full-group baseline) + one removed.
        normed = [1.5, -0.5, -0.5, -0.5]  # e.g. raw [1,0,0,0] after full-group norm
        samples = [
            self._make_sample(1.0),
            self._make_sample(0.0),
            self._make_sample(0.0),
            self._make_sample(0.0, remove=True),
        ]
        pre = normed[:3]
        assert sum(pre) / 3 > 1e-9  # biased positive before renorm
        out = renormalize_after_mask(normed, samples, 4, True)
        assert abs(sum(out[:3]) / 3) < 1e-9  # survivors recentred to zero mean
        assert all(v == v for v in out)  # finite (no NaN)

    def test_binarize_renormalize_end_to_end(self):
        from miles_plugins.arena.nats_arena.reward_binary import (
            binarize_renormalize_after_mask,
        )

        args = self._make_args(n_samples=4, grpo_std=True)
        samples = [
            self._make_sample(1.0),
            self._make_sample(1.0),
            self._make_sample(0.2),
            self._make_sample(0.0, remove=True),
        ]
        raw, processed = binarize_renormalize_after_mask(args, samples)
        assert raw == [1.0, 1.0, 0.2, 0.0]
        # binarize metadata side effect preserved
        assert all(s.metadata["binary_reward"] in (0.0, 1.0) for s in samples)
        # survivors (0,1,2) recentred to zero mean
        assert abs(sum(processed[:3]) / 3) < 1e-6

    def test_binarize_renormalize_empty(self):
        from miles_plugins.arena.nats_arena.reward_binary import (
            binarize_renormalize_after_mask,
        )

        args = self._make_args(n_samples=1)
        assert binarize_renormalize_after_mask(args, []) == ([], [])


# ===========================================================================
# 5. data_source.py
# ===========================================================================


class TestReadManifest:
    @pytest.fixture(autouse=True)
    def _skip_without_deps(self):
        # data_source imports torch (and miles internals) at module level.
        pytest.importorskip("torch")

    def test_basic_manifest_parsing(self, tmp_path):
        from miles_plugins.arena.nats_arena.data_source import _read_manifest

        manifest = tmp_path / "manifest.jsonl"
        rows = [
            {"lakefs_uri": "lakefs://repo/branch/tasks/task-1/", "lakefs_commit_id": "abc"},
            {"lakefs_uri": "lakefs://repo/branch/tasks/task-2/", "lakefs_commit_id": "def"},
        ]
        manifest.write_text("\n".join(json.dumps(r) for r in rows))

        result = _read_manifest(str(manifest))
        assert len(result) == 2
        assert result[0]["metadata"]["instance_id"] == "task-1"
        assert result[0]["metadata"]["lakefs_uri"] == "lakefs://repo/branch/tasks/task-1/"
        assert result[1]["metadata"]["instance_id"] == "task-2"
        # Messages should have placeholder user message
        assert result[0]["messages"][0]["role"] == "user"
        assert "task-1" in result[0]["messages"][0]["content"]

    def test_blank_lines_skipped(self, tmp_path):
        from miles_plugins.arena.nats_arena.data_source import _read_manifest

        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            '{"lakefs_uri": "lakefs://r/b/tasks/t1/", "lakefs_commit_id": "x"}\n'
            "\n"
            '{"lakefs_uri": "lakefs://r/b/tasks/t2/", "lakefs_commit_id": "y"}\n'
            "\n"
        )
        result = _read_manifest(str(manifest))
        assert len(result) == 2

    def test_uri_without_tasks_segment(self, tmp_path):
        from miles_plugins.arena.nats_arena.data_source import _read_manifest

        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            '{"lakefs_uri": "lakefs://repo/branch/some/path/myid", "lakefs_commit_id": "x"}\n'
        )
        result = _read_manifest(str(manifest))
        # Falls back to last segment
        assert result[0]["metadata"]["instance_id"] == "myid"


class TestTaskIdFromUri:
    @pytest.fixture(autouse=True)
    def _skip_without_deps(self):
        pytest.importorskip("torch")

    def test_standard_uri(self):
        from miles_plugins.arena.nats_arena.data_source import _task_id_from_uri

        assert _task_id_from_uri("lakefs://repo/branch/prefix/tasks/task-42/") == "task-42"

    def test_no_tasks_segment(self):
        from miles_plugins.arena.nats_arena.data_source import _task_id_from_uri

        assert _task_id_from_uri("lakefs://repo/branch/some/path") == "path"

    def test_trailing_slash(self):
        from miles_plugins.arena.nats_arena.data_source import _task_id_from_uri

        assert _task_id_from_uri("lakefs://r/b/tasks/id123/") == "id123"


class TestArenaDataSourceWeightedSampling:
    """Test get_samples weighted allocation without needing torch/tokenizer."""

    @pytest.fixture(autouse=True)
    def _skip_without_deps(self):
        pytest.importorskip("torch")

    def test_deficit_based_allocation(self):
        """Test the _allocate method produces correct weighted distribution."""
        from miles_plugins.arena.nats_arena.data_source import ArenaDataSourceWithBuffer

        # We can't easily instantiate ArenaDataSourceWithBuffer without mocking
        # a ton of miles internals, so test the _allocate logic directly by
        # creating a minimal instance with just the fields _allocate needs.
        class _MinimalDS:
            def __init__(self):
                self.gym_names = ["a", "b"]
                self.weights = {"a": 0.75, "b": 0.25}
                self._deficit = {"a": 0.0, "b": 0.0}
                self._weights_lock = threading.Lock()

        ds = _MinimalDS()
        # Borrow the _allocate method
        ds._allocate = ArenaDataSourceWithBuffer._allocate.__get__(ds)

        alloc = ds._allocate(100)
        # Should be approximately 75/25 split
        assert alloc["a"] == 75
        assert alloc["b"] == 25


# ===========================================================================
# 6. eval_coordinator.py
# ===========================================================================


class TestRowToTask:
    def test_basic_row(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _row_to_task

        row = {
            "id": "eval-task-1",
            "lakefs_uri": "lakefs://repo/main/tasks/eval-task-1/",
            "lakefs_commit_id": "sha",
            "label": "expected",
        }
        task = _row_to_task(row, {"name": "ds1", "gym_name": "g"}, 0)
        assert task["id"] == "eval-task-1"
        assert task["lakefs_uri"] == "lakefs://repo/main/tasks/eval-task-1/"
        assert task["metadata"]["eval"] is True
        assert task["metadata"]["eval_dataset"] == "ds1"
        assert task["metadata"]["label"] == "expected"
        assert task["metadata"]["gym_name"] == "g"

    def test_missing_lakefs_uri_raises(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _row_to_task

        row = {"id": "x"}
        with pytest.raises(ValueError, match="missing 'lakefs_uri'"):
            _row_to_task(row, {"name": "ds"}, 0)

    def test_metadata_as_string(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _row_to_task

        row = {
            "metadata": json.dumps({
                "instance_id": "from-meta",
                "lakefs_uri": "lakefs://r/b/tasks/from-meta/",
                "lakefs_commit_id": "c",
            }),
        }
        task = _row_to_task(row, {"name": "ds", "gym_name": "g"}, 0)
        assert task["id"] == "from-meta"

    def test_invalid_metadata_string(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _row_to_task

        row = {
            "lakefs_uri": "lakefs://r/b/tasks/t/",
            "lakefs_commit_id": "c",
            "metadata": "not-json",
        }
        task = _row_to_task(row, {"name": "ds", "gym_name": "g"}, 0)
        # Should not crash; metadata falls back to {}
        assert task["id"] == "t"

    def test_n_samples_forwarded(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _row_to_task

        row = {
            "lakefs_uri": "lakefs://r/b/tasks/t/",
            "lakefs_commit_id": "c",
        }
        task = _row_to_task(row, {"name": "ds", "gym_name": "g"}, 0, n_samples=16)
        assert task["n_samples"] == 16

    def test_id_fallback_chain(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _row_to_task

        # Falls to task_id field
        row = {
            "task_id": "from-task-id",
            "lakefs_uri": "lakefs://r/b/tasks/xxx/",
            "lakefs_commit_id": "c",
        }
        task = _row_to_task(row, {"name": "ds", "gym_name": "g"}, 0)
        assert task["id"] == "from-task-id"

    def test_id_fallback_to_uri(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _row_to_task

        # No id, no task_id, no metadata.instance_id -> extract from URI
        row = {
            "lakefs_uri": "lakefs://r/b/tasks/from-uri/",
            "lakefs_commit_id": "c",
        }
        task = _row_to_task(row, {"name": "ds", "gym_name": "g"}, 0)
        assert task["id"] == "from-uri"


class TestEnvVarParsingGuards:
    """Test that json.loads("") edge case is handled by the `or "[]"` fix."""

    def test_empty_eval_gyms_config_does_not_crash(self):
        """The `or "[]"` pattern should prevent json.loads("") from crashing."""
        # Simulate the pattern from eval_coordinator (line ~900):
        # json.loads(os.environ.get("EVAL_GYMS_CONFIG", "[]") or "[]")
        raw = ""  # simulates PRESENT-but-EMPTY env var
        result = json.loads(raw or "[]")
        assert result == []

    def test_none_eval_gyms_config_handled(self):
        # os.environ.get returns "[]" as default
        raw = None
        default = "[]"
        value = raw if raw is not None else default
        result = json.loads(value or "[]")
        assert result == []

    def test_valid_json_passes_through(self):
        raw = '[{"replicas": 4, "concurrency": 2}]'
        result = json.loads(raw or "[]")
        assert result == [{"replicas": 4, "concurrency": 2}]


class TestHandleResultMsgDedup:
    """Test the _handle_result_msg dedup/DLQ state machine.

    This is the highest-priority test target per reviewer concern.
    We simulate the state machine by directly invoking the logic
    with the same closure variables the real function uses.

    NOTE: the real ``_handle_result_msg`` is a closure inside
    ``eval_coordinator`` (miles_plugins/arena/nats_arena/eval_coordinator.py,
    ~lines 1187-1269) and cannot be imported without standing up the whole
    coordinator, so this inline copy MUST be kept in sync with the ported
    source by hand. Trace-span emission (a no-op side channel) is elided.
    """

    def _make_state_machine(self):
        """Recreate the _handle_result_msg closure's state variables."""
        from miles_plugins.arena.nats_arena.message_format import extract_trajectories

        collected = [0]
        seen_task_ids: set[str] = set()
        dlq_swept: set[str] = set()
        task_registry: dict[str, str] = {}
        results_by_dataset: dict[str, dict[str, list[float]]] = {}
        publish_times: dict[str, float] = {}
        pass_threshold_per_dataset: dict[str, float] = {"ds": 1.0}

        def handle_result_msg(result: dict) -> bool:
            """Inline reimplementation of _handle_result_msg logic."""
            tid = result.get("task_id", "")

            is_override = tid in dlq_swept
            if is_override:
                dlq_swept.discard(tid)
            elif tid in seen_task_ids:
                return False
            else:
                seen_task_ids.add(tid)

            dataset_name = task_registry.get(tid)
            if dataset_name is None:
                base_id = tid.rsplit("-s", 1)[0]
                dataset_name = task_registry.get(base_id, "unknown")

            trajectories = extract_trajectories(result)
            base_tid = tid
            for traj in trajectories:
                reward = traj.get("reward", 0.0)
                results_by_dataset.setdefault(dataset_name, {}).setdefault(base_tid, []).append(reward)

            publish_times.pop(tid, None)
            if is_override:
                return False
            collected[0] += 1
            return True

        return (
            handle_result_msg,
            collected,
            seen_task_ids,
            dlq_swept,
            task_registry,
            results_by_dataset,
            publish_times,
        )

    def test_first_arrival_increments_collected(self):
        handle, collected, seen, dlq, registry, results, ptimes = self._make_state_machine()
        registry["t1"] = "ds"
        ptimes["t1"] = 100.0

        result = {"task_id": "t1", "trajectories": [{"reward": 0.8}]}
        assert handle(result) is True
        assert collected[0] == 1
        assert "t1" in seen
        assert results["ds"]["t1"] == [0.8]

    def test_duplicate_arrival_does_not_double_count(self):
        handle, collected, seen, dlq, registry, results, ptimes = self._make_state_machine()
        registry["t1"] = "ds"
        ptimes["t1"] = 100.0

        result = {"task_id": "t1", "trajectories": [{"reward": 0.8}]}
        handle(result)  # first
        assert handle(result) is False  # duplicate
        assert collected[0] == 1
        # Rewards should NOT be duplicated
        assert results["ds"]["t1"] == [0.8]

    def test_dlq_swept_task_then_real_result_overrides(self):
        """A DLQ-swept task getting a late real result should override the miss."""
        handle, collected, seen, dlq, registry, results, ptimes = self._make_state_machine()
        registry["t1"] = "ds"

        # Simulate DLQ sweep: mark task as seen + swept, increment collected
        seen.add("t1")
        dlq.add("t1")
        collected[0] = 1  # DLQ sweep incremented this

        # Now a real result arrives
        result = {"task_id": "t1", "trajectories": [{"reward": 1.0}]}
        is_fresh = handle(result)

        # Should NOT double-count (is_override=True returns False)
        assert is_fresh is False
        assert collected[0] == 1  # not incremented again
        # But the reward IS recorded (overriding the miss)
        assert results["ds"]["t1"] == [1.0]
        # dlq_swept entry should be removed
        assert "t1" not in dlq

    def test_dlq_swept_override_then_duplicate_ignored(self):
        """After DLQ override, a further duplicate is still deduped."""
        handle, collected, seen, dlq, registry, results, ptimes = self._make_state_machine()
        registry["t1"] = "ds"

        # DLQ sweep
        seen.add("t1")
        dlq.add("t1")
        collected[0] = 1

        # Real result overrides
        result = {"task_id": "t1", "trajectories": [{"reward": 1.0}]}
        handle(result)

        # Third arrival (pure duplicate)
        assert handle(result) is False
        assert collected[0] == 1
        # Reward recorded only once during override
        assert results["ds"]["t1"] == [1.0]

    def test_multiple_tasks_independent(self):
        handle, collected, seen, dlq, registry, results, ptimes = self._make_state_machine()
        registry["t1"] = "ds"
        registry["t2"] = "ds"
        ptimes["t1"] = 100.0
        ptimes["t2"] = 100.0

        handle({"task_id": "t1", "trajectories": [{"reward": 0.5}]})
        handle({"task_id": "t2", "trajectories": [{"reward": 1.0}]})
        assert collected[0] == 2
        assert results["ds"]["t1"] == [0.5]
        assert results["ds"]["t2"] == [1.0]

    def test_multi_trajectory_result(self):
        """A single result with multiple trajectories (n_samples > 1)."""
        handle, collected, seen, dlq, registry, results, ptimes = self._make_state_machine()
        registry["t1"] = "ds"
        ptimes["t1"] = 100.0

        result = {
            "task_id": "t1",
            "trajectories": [
                {"reward": 0.0},
                {"reward": 0.5},
                {"reward": 1.0},
            ],
        }
        assert handle(result) is True
        assert collected[0] == 1
        # All three rewards recorded under the same base_tid
        assert results["ds"]["t1"] == [0.0, 0.5, 1.0]


class TestTaskIdFromLakefsUri:
    def test_standard_uri(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _task_id_from_lakefs_uri

        assert _task_id_from_lakefs_uri("lakefs://repo/branch/prefix/tasks/my-id/") == "my-id"

    def test_no_tasks_segment(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _task_id_from_lakefs_uri

        assert _task_id_from_lakefs_uri("lakefs://repo/branch/some/thing") == "thing"

    def test_empty_string(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _task_id_from_lakefs_uri

        assert _task_id_from_lakefs_uri("") == ""


# ===========================================================================
# 7. gym_autoscaler.py
# ===========================================================================


class TestGymAutoscalerScalingDecision:
    """Test scaling decision logic without K8s."""

    def _make_scaler(self, **kwargs) -> "GymAutoscaler":
        from miles_plugins.arena.nats_arena.gym_autoscaler import GymAutoscaler

        defaults = {
            "gym_specs": {
                "gym_a": {
                    "deploy_name": "deploy-gym-a",
                    "concurrency_per_pod": 4,
                    "min_replicas": 2,
                    "max_replicas": 20,
                    "initial_replicas": 4,
                }
            },
            "namespace": "test-ns",
            "tasks_stream": "ARENA_TASKS",
            "dry_run": True,
            "warmup_secs": 0.0,  # disable warmup for tests
        }
        defaults.update(kwargs)
        scaler = GymAutoscaler(**defaults)
        # Mark as seeded so _ensure_seeded doesn't try K8s
        scaler._seeded_from_api = True
        return scaler

    def test_decide_action_scale_up(self):
        scaler = self._make_scaler()
        action = scaler._decide_action(
            gym_name="gym_a",
            current=4,
            desired=8,
            num_ack_pending=16,
            concurrency=4,
            now=time.time(),
            in_warmup=False,
            cooldown_live=180.0,
            in_profile=False,
        )
        assert action == "up"

    def test_decide_action_noop(self):
        scaler = self._make_scaler()
        action = scaler._decide_action(
            gym_name="gym_a",
            current=4,
            desired=4,
            num_ack_pending=16,
            concurrency=4,
            now=time.time(),
            in_warmup=False,
            cooldown_live=180.0,
            in_profile=False,
        )
        assert action == "noop"

    def test_decide_action_warmup_blocks_scale_down(self):
        scaler = self._make_scaler(warmup_secs=3600.0)
        action = scaler._decide_action(
            gym_name="gym_a",
            current=8,
            desired=4,
            num_ack_pending=4,
            concurrency=4,
            now=time.time(),
            in_warmup=True,
            cooldown_live=180.0,
            in_profile=False,
        )
        assert action == "warmup_blocked"

    def test_decide_action_cooldown_before_scale_down(self):
        scaler = self._make_scaler(cooldown_secs=180.0)
        now = time.time()
        # First call: starts cooldown
        action = scaler._decide_action(
            gym_name="gym_a",
            current=8,
            desired=4,
            num_ack_pending=4,
            concurrency=4,
            now=now,
            in_warmup=False,
            cooldown_live=180.0,
            in_profile=False,
        )
        assert action == "cooldown"

        # Second call shortly after: still in cooldown
        action = scaler._decide_action(
            gym_name="gym_a",
            current=8,
            desired=4,
            num_ack_pending=4,
            concurrency=4,
            now=now + 60,
            in_warmup=False,
            cooldown_live=180.0,
            in_profile=False,
        )
        assert action == "cooldown"

        # Third call after cooldown elapsed
        action = scaler._decide_action(
            gym_name="gym_a",
            current=8,
            desired=4,
            num_ack_pending=4,
            concurrency=4,
            now=now + 200,
            in_warmup=False,
            cooldown_live=180.0,
            in_profile=False,
        )
        assert action == "down"

    def test_decide_action_busy_pods_prevent_scale_down(self):
        scaler = self._make_scaler(cooldown_secs=0.0)
        # ack_pending (20) > desired (2) * concurrency (4) = 8
        action = scaler._decide_action(
            gym_name="gym_a",
            current=8,
            desired=2,
            num_ack_pending=20,
            concurrency=4,
            now=time.time(),
            in_warmup=False,
            cooldown_live=0.0,
            in_profile=False,
        )
        assert action == "noop"

    def test_decide_action_profile_blocks_all(self):
        scaler = self._make_scaler(profile_mode=True)
        action = scaler._decide_action(
            gym_name="gym_a",
            current=4,
            desired=8,
            num_ack_pending=16,
            concurrency=4,
            now=time.time(),
            in_warmup=False,
            cooldown_live=180.0,
            in_profile=True,
        )
        assert action == "profile_blocked"

    def test_desired_replicas_from_demand(self):
        """Test the core scaling formula: pending > 0 triggers demand-based scaling."""
        import math

        from miles_plugins.arena.nats_arena.gym_autoscaler import _clamp

        # Simulate: num_pending=40, num_ack_pending=16, concurrency=4, headroom=1.25
        num_pending = 40
        num_ack_pending = 16
        concurrency = 4
        headroom = 1.25
        target_util = _clamp(1.0 / max(1.0, headroom), 0.3, 0.95)
        total_demand = num_ack_pending + num_pending
        desired = math.ceil(total_demand / (concurrency * target_util))
        # 56 / (4 * 0.8) = 56/3.2 = 17.5 -> 18
        assert desired == 18

    def test_min_max_clamping(self):
        scaler = self._make_scaler()
        # desired > max_replicas should be clamped
        spec = scaler.gym_specs["gym_a"]
        min_r = spec["min_replicas"]
        max_r = spec["max_replicas"]
        assert max(min_r, min(max_r, 100)) == 20
        assert max(min_r, min(max_r, 1)) == 2

    def test_cooldown_candidate_reset_on_noop(self):
        scaler = self._make_scaler(cooldown_secs=180.0)
        now = time.time()
        # Start cooldown
        scaler._decide_action("gym_a", 8, 4, 4, 4, now, False, 180.0, False)
        assert "gym_a" in scaler._scale_down_candidate_since
        # Noop (desired == current) resets
        scaler._decide_action("gym_a", 8, 8, 16, 4, now + 10, False, 180.0, False)
        assert "gym_a" not in scaler._scale_down_candidate_since

    def test_derive_live_params_without_tracker(self):
        scaler = self._make_scaler(headroom=1.5, cooldown_secs=120.0)
        headroom_live, cooldown_live = scaler._derive_live_params("gym_a", None)
        # Falls back to configured values
        assert headroom_live == 1.5
        assert cooldown_live == 120.0

    def test_derive_live_params_with_warm_tracker(self):
        from miles_plugins.arena.nats_arena.rollout_timing_tracker import TimingStats

        scaler = self._make_scaler(
            headroom=1.25,
            cooldown_secs=60.0,
            auto_tune=True,
            min_tracker_samples=5,
        )
        stats = TimingStats(
            p50_short=30.0,
            p95_short=60.0,
            mean_short=35.0,
            p50_long=25.0,
            p95_long=50.0,
            mean_long=30.0,
            cov_long=0.8,  # fairly bursty
            growth_ratio=1.2,
            sample_count_short=50,
            sample_count_long=100,
        )
        headroom_live, cooldown_live = scaler._derive_live_params("gym_a", stats)
        # headroom should be increased above configured 1.25
        assert headroom_live > 1.25
        # cooldown should be at least 1.5 * p95_long = 75s
        assert cooldown_live >= 75.0

    def test_state_dict_load_state_dict(self):
        scaler = self._make_scaler()
        scaler._current_replicas = {"gym_a": 6}
        scaler._tick_count = 42
        scaler._scale_down_candidate_since = {"gym_a": 12345.0}

        state = scaler.state_dict()
        assert state["current_replicas"] == {"gym_a": 6}
        assert state["tick_count"] == 42

        scaler2 = self._make_scaler()
        scaler2.load_state_dict(state)
        assert scaler2._tick_count == 42
        assert scaler2._scale_down_candidate_since == {"gym_a": 12345.0}


# ===========================================================================
# Additional edge case: _estimate_pass_at_k
# ===========================================================================


class TestEstimatePassAtK:
    def test_all_correct(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _estimate_pass_at_k

        assert _estimate_pass_at_k(4, 4, 1) == 1.0

    def test_none_correct(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _estimate_pass_at_k

        result = _estimate_pass_at_k(4, 0, 1)
        assert result == 0.0

    def test_some_correct(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _estimate_pass_at_k

        # 4 samples, 2 correct, pass@1
        result = _estimate_pass_at_k(4, 2, 1)
        # 1 - C(2,1)/C(4,1) = 1 - 2/4 = 0.5
        assert abs(result - 0.5) < 1e-9

    def test_k_equals_num_samples(self):
        from miles_plugins.arena.nats_arena.eval_coordinator import _estimate_pass_at_k

        # If k >= num_samples - num_correct, should return 1.0
        result = _estimate_pass_at_k(4, 3, 2)
        # num_samples - num_correct = 1 < k=2, so 1.0
        assert result == 1.0


# ===========================================================================
# Run standalone
# ===========================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


class TestSalvageableStatuses:
    def test_truncated_is_salvageable_failed_is_not(self):
        from miles_plugins.arena.nats_arena.message_format import (
            SALVAGEABLE_RESULT_STATUSES,
        )

        assert "success" in SALVAGEABLE_RESULT_STATUSES
        assert "truncated" in SALVAGEABLE_RESULT_STATUSES
        assert "failed" not in SALVAGEABLE_RESULT_STATUSES


class TestGradedRenormalizeAfterMask:
    """graded_reward + graded_renormalize_after_mask (graded, no binarization)."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self):
        pytest.importorskip("torch")

    def _make_sample(self, reward: float, *, remove: bool = False, resp_len: int = 1):
        sample = MagicMock()
        sample.get_reward_value = MagicMock(return_value=reward)
        sample.metadata = {}
        sample.remove_sample = remove
        sample.response_length = resp_len
        return sample

    def _make_args(self, n_samples: int = 2, grpo_std: bool = False):
        return SimpleNamespace(
            n_samples_per_prompt=n_samples,
            grpo_std_normalization=grpo_std,
        )

    def test_empty_batch_returns_empty(self):
        from miles_plugins.arena.nats_arena.reward_binary import (
            graded_renormalize_after_mask,
        )

        assert graded_renormalize_after_mask(self._make_args(), []) == ([], [])

    def test_preserves_partial_credit_no_binarization(self):
        """A near-solve keeps proportional advantage instead of being zeroed.

        This is the whole point of the graded path: with binarize_reward a
        composite of 0.97 (< 1.0) becomes 0, so a group of [0.97, 0.0] binarizes
        to [0, 0] -> all-zero -> no gradient. Graded keeps [0.97, 0.0] and gives
        the near-solve a positive advantage.
        """
        from miles_plugins.arena.nats_arena.reward_binary import (
            graded_renormalize_after_mask,
        )

        args = self._make_args(n_samples=2)
        samples = [self._make_sample(0.97), self._make_sample(0.0)]
        raw, processed = graded_renormalize_after_mask(args, samples)
        assert raw == [0.97, 0.0]  # not binarized
        assert processed[0] > 0.0 > processed[1]  # signal survives

    def test_renormalizes_survivors_after_mask(self):
        """Masked (removed) samples are excluded from the survivor mean/std."""
        from miles_plugins.arena.nats_arena.reward_binary import (
            graded_renormalize_after_mask,
        )

        args = self._make_args(n_samples=4)
        # 3 survivors + 1 removed; raw graded is preserved for all four.
        samples = [
            self._make_sample(1.0),
            self._make_sample(0.5),
            self._make_sample(0.0),
            self._make_sample(0.0, remove=True),
        ]
        raw, processed = graded_renormalize_after_mask(args, samples)
        assert raw == [1.0, 0.5, 0.0, 0.0]
        # Survivors (first three) recentred to ~zero mean over survivors only.
        assert abs(sum(processed[:3]) / 3) < 1e-6

    def test_single_survivor_group_zeroed(self):
        from miles_plugins.arena.nats_arena.reward_binary import (
            graded_renormalize_after_mask,
        )

        args = self._make_args(n_samples=2)
        # One removed -> a single survivor -> no usable variance -> zeroed.
        samples = [self._make_sample(1.0), self._make_sample(0.0, remove=True)]
        _raw, processed = graded_renormalize_after_mask(args, samples)
        assert processed[0] == 0.0
