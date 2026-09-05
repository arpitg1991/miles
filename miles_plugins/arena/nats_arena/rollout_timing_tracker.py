"""Per-gym rollout walltime tracker for adaptive autoscaling.

Rollout durations for RL training are non-stationary: as the model
improves, successful agents take more turns before finishing, p95 tends
to grow, and the autoscaler's static ``cooldown`` / ``headroom``
parameters drift out of calibration. This tracker maintains two sliding
windows per gym so consumers (``GymAutoscaler``) can derive live
cooldown and headroom values from recent behavior and detect when
rollouts are lengthening via a short-vs-long p95 ratio.

Thread-safe: ``record_publish`` and ``record_completion`` are called
from the NATS worker loop; ``get_stats`` may be read from the same
thread or via the autoscaler tick.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# 99th-percentile Winsor trim on the long window keeps single stuck
# rollouts (e.g. a 3-hour outlier from a hung container) from skewing
# p95 forever. Applied only to the long window since the short window
# turns over quickly and self-corrects.
_WINSOR_PCT = 99.0


@dataclass
class TimingStats:
    """Snapshot of per-gym rollout timing stats.

    All durations in seconds. ``sample_count_*`` report the current
    fill level of each window; consumers should fall back to config
    defaults when below some minimum (e.g. 50).
    """

    p50_short: float
    p95_short: float
    mean_short: float
    p50_long: float
    p95_long: float
    mean_long: float
    cov_long: float
    growth_ratio: float
    sample_count_short: int
    sample_count_long: int


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile on a pre-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(sorted_values[lo])
    frac = rank - lo
    return float(sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo]))


def _winsorize(sorted_values: list[float], pct: float) -> list[float]:
    """Clip values above the ``pct``th percentile to the percentile itself."""
    if len(sorted_values) < 20:
        return sorted_values  # too few samples to trim meaningfully
    cap = _percentile(sorted_values, pct)
    return [min(v, cap) for v in sorted_values]


class RolloutTimingTracker:
    """Per-gym short/long-window rollout-duration tracker."""

    def __init__(
        self,
        short_window_size: int = 100,
        long_window_size: int = 1000,
    ):
        if short_window_size <= 0 or long_window_size <= 0:
            raise ValueError("Window sizes must be positive")
        if short_window_size >= long_window_size:
            raise ValueError(
                "short_window_size must be strictly less than long_window_size"
            )
        self.short_size = short_window_size
        self.long_size = long_window_size

        self._lock = threading.Lock()
        # task_id -> (gym_name, publish_ts) — not per-gym because task_ids are global.
        self._in_flight: dict[str, tuple[str, float]] = {}
        # gym_name -> deque[duration_secs]
        self._short: dict[str, deque[float]] = {}
        self._long: dict[str, deque[float]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_publish(self, task_id: str, gym_name: str, ts: float) -> None:
        if not task_id or not gym_name:
            return
        with self._lock:
            self._in_flight[task_id] = (gym_name, ts)

    def record_completion(self, task_id: str, ts: float) -> None:
        if not task_id:
            return
        with self._lock:
            entry = self._in_flight.pop(task_id, None)
            if entry is None:
                # Unknown task — possibly a result from before this tracker
                # was instantiated, or a duplicate delivery. Ignore.
                return
            gym_name, publish_ts = entry
            duration = max(0.0, ts - publish_ts)
            self._ensure_windows(gym_name)
            self._short[gym_name].append(duration)
            self._long[gym_name].append(duration)

    def _ensure_windows(self, gym_name: str) -> None:
        if gym_name not in self._short:
            self._short[gym_name] = deque(maxlen=self.short_size)
            self._long[gym_name] = deque(maxlen=self.long_size)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, gym_name: str) -> TimingStats | None:
        """Return a fresh ``TimingStats`` or ``None`` if no samples yet."""
        with self._lock:
            short = self._short.get(gym_name)
            long_ = self._long.get(gym_name)
            if not short and not long_:
                return None
            short_list = list(short) if short else []
            long_list = list(long_) if long_ else []

        short_sorted = sorted(short_list)
        long_sorted = sorted(long_list)
        long_trimmed = _winsorize(long_sorted, _WINSOR_PCT)
        long_trimmed_sorted = sorted(long_trimmed)

        p50_short = _percentile(short_sorted, 50.0)
        p95_short = _percentile(short_sorted, 95.0)
        mean_short = (sum(short_list) / len(short_list)) if short_list else 0.0
        p50_long = _percentile(long_trimmed_sorted, 50.0)
        p95_long = _percentile(long_trimmed_sorted, 95.0)
        mean_long = (sum(long_trimmed) / len(long_trimmed)) if long_trimmed else 0.0

        # CoV on the trimmed long window: std/mean. Return 0 if mean is 0
        # (gym never completed) to avoid division-by-zero.
        if mean_long > 0 and len(long_trimmed) >= 2:
            variance = sum((v - mean_long) ** 2 for v in long_trimmed) / len(long_trimmed)
            cov_long = math.sqrt(variance) / mean_long
        else:
            cov_long = 0.0

        # Growth ratio: how much longer recent p95 is vs. the long-run baseline.
        # >1.0 means rollouts are stretching; <1.0 means shortening.
        if p95_long > 1e-6:
            growth_ratio = p95_short / p95_long
        else:
            growth_ratio = 1.0

        return TimingStats(
            p50_short=p50_short,
            p95_short=p95_short,
            mean_short=mean_short,
            p50_long=p50_long,
            p95_long=p95_long,
            mean_long=mean_long,
            cov_long=cov_long,
            growth_ratio=growth_ratio,
            sample_count_short=len(short_list),
            sample_count_long=len(long_list),
        )

    def all_gyms(self) -> list[str]:
        with self._lock:
            return sorted(self._short.keys())

    def in_flight_count(self) -> int:
        with self._lock:
            return len(self._in_flight)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "short_window_size": self.short_size,
                "long_window_size": self.long_size,
                "short": {g: list(d) for g, d in self._short.items()},
                "long": {g: list(d) for g, d in self._long.items()},
                # Note: _in_flight is intentionally not persisted — after a
                # restart the NATS stream is purged and prior publishes are
                # gone, so their completion records would never arrive.
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        with self._lock:
            for gym_name, values in (state.get("short") or {}).items():
                self._short[gym_name] = deque(values, maxlen=self.short_size)
            for gym_name, values in (state.get("long") or {}).items():
                self._long[gym_name] = deque(values, maxlen=self.long_size)
            restored_gyms = sorted(set(self._short) | set(self._long))
            if restored_gyms:
                logger.info(
                    "RolloutTimingTracker restored: gyms=%s short=%s long=%s",
                    restored_gyms,
                    {g: len(self._short.get(g, [])) for g in restored_gyms},
                    {g: len(self._long.get(g, [])) for g in restored_gyms},
                )
