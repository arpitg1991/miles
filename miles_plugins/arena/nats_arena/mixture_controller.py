"""Dynamic mixture controller for multi-gym training.

Monitors per-gym rollout completion rates and computes corrected
sampling weights so the actual training-batch composition converges
to a user-specified target ratio.

Algorithm:
  1. Count completions per gym over a rolling window.
  2. Every ``adjustment_interval`` seconds, compare observed ratios
     to target ratios.
  3. Compute a proportional correction: weight_new = target * (target / observed).
  4. Clamp corrections to [min_correction, max_correction].
  5. Normalize, then EMA-smooth with previous weights.

The EMA smoothing ensures restart stability: after a checkpoint
restore the controller starts from restored weights and makes
small corrections, never snapping back to initial config values.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights} if n else {}
    return {k: v / total for k, v in weights.items()}


class MixtureController:
    """Feedback controller that adjusts per-gym prompt sampling weights
    based on observed rollout completion rates.

    Thread-safe: ``record_completion`` is called from the NATS collect
    path and ``maybe_adjust`` from the same loop, but weight reads
    happen from the data-source thread.
    """

    def __init__(
        self,
        target_ratios: dict[str, float],
        adjustment_interval: float = 60.0,
        smoothing: float = 0.3,
        correction_bounds: tuple[float, float] = (0.2, 5.0),
    ):
        self.target_ratios: dict[str, float] = _normalize(target_ratios)
        self.adjustment_interval = adjustment_interval
        self.smoothing = smoothing
        self.min_correction = correction_bounds[0]
        self.max_correction = correction_bounds[1]

        self._lock = threading.Lock()
        self.gym_completions: dict[str, int] = defaultdict(int)
        self.total_completions: int = 0
        self.last_adjustment_time: float = time.time()

        self.current_weights: dict[str, float] = dict(self.target_ratios)

        self.observed_ratios: dict[str, float] = {}
        self.adjustment_count: int = 0

    def record_completion(self, gym_name: str) -> None:
        with self._lock:
            self.gym_completions[gym_name] += 1
            self.total_completions += 1

    def maybe_adjust(self) -> dict[str, float] | None:
        """Return new weights if the adjustment interval has elapsed
        and enough data exists, otherwise ``None``."""
        with self._lock:
            now = time.time()
            if now - self.last_adjustment_time < self.adjustment_interval:
                return None
            if self.total_completions < 10:
                return None
            return self._compute_adjusted_weights_locked(now)

    def _compute_adjusted_weights_locked(self, now: float) -> dict[str, float]:
        previous_weights = dict(self.current_weights)
        elapsed = now - self.last_adjustment_time

        observed: dict[str, float] = {}
        known_total = sum(self.gym_completions.get(g, 0) for g in self.target_ratios)
        denom = known_total if known_total > 0 else self.total_completions
        for gym_name in self.target_ratios:
            observed[gym_name] = (
                self.gym_completions.get(gym_name, 0) / denom
            )
        self.observed_ratios = observed

        corrections: dict[str, float] = {}
        raw_weights: dict[str, float] = {}
        for gym_name, target in self.target_ratios.items():
            obs = max(observed.get(gym_name, 0.0), 0.001)
            correction = target / obs
            correction = max(self.min_correction, min(self.max_correction, correction))
            corrections[gym_name] = correction
            raw_weights[gym_name] = target * correction

        raw_weights = _normalize(raw_weights)

        smoothed: dict[str, float] = {}
        alpha = self.smoothing
        for gym_name in raw_weights:
            old = self.current_weights.get(gym_name, raw_weights[gym_name])
            smoothed[gym_name] = alpha * raw_weights[gym_name] + (1 - alpha) * old
        smoothed = _normalize(smoothed)

        self.current_weights = smoothed
        self.adjustment_count += 1

        logger.info(
            "Mixture adjustment #%d (%.0fs window, %d completions):",
            self.adjustment_count, elapsed, self.total_completions,
        )
        logger.info(
            "  completions: %s",
            {g: self.gym_completions.get(g, 0) for g in self.target_ratios},
        )
        logger.info(
            "  target_ratios:    %s",
            {g: f"{r:.3f}" for g, r in self.target_ratios.items()},
        )
        logger.info(
            "  observed_ratios:  %s",
            {g: f"{r:.3f}" for g, r in observed.items()},
        )
        logger.info(
            "  corrections:      %s",
            {g: f"{c:.3f}" for g, c in corrections.items()},
        )
        logger.info(
            "  weights_before:   %s",
            {g: f"{w:.3f}" for g, w in previous_weights.items()},
        )
        logger.info(
            "  weights_after:    %s",
            {g: f"{w:.3f}" for g, w in smoothed.items()},
        )

        self.gym_completions = defaultdict(int)
        self.total_completions = 0
        self.last_adjustment_time = now

        return smoothed

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "current_weights": dict(self.current_weights),
                "adjustment_count": self.adjustment_count,
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with self._lock:
            restored_weights = state.get("current_weights")
            if restored_weights:
                self.current_weights = dict(restored_weights)
            self.adjustment_count = state.get("adjustment_count", 0)
            logger.info(
                "MixtureController restored: weights=%s, adjustments=%d",
                {g: f"{w:.3f}" for g, w in self.current_weights.items()},
                self.adjustment_count,
            )

    # ------------------------------------------------------------------
    # Read-only snapshot for external consumers (e.g. GymAutoscaler)
    # ------------------------------------------------------------------

    def get_observed_snapshot(self) -> tuple[dict[str, float], dict[str, float]]:
        """Thread-safe copy of ``(target_ratios, observed_ratios)``.

        ``observed_ratios`` is updated at every adjustment tick. Between
        ticks the returned value is the most recent computed ratio (empty
        dict before the first adjustment).
        """
        with self._lock:
            return dict(self.target_ratios), dict(self.observed_ratios)

    def get_weight_snapshot(self) -> tuple[dict[str, float], dict[str, float]]:
        """Thread-safe copy of ``(target_ratios, current_weights)``.

        Used by the autoscaler to pre-warm gyms whose sampling weight has
        been raised above their target ratio — i.e. the controller has
        decided to skew the next prompt batch toward this gym. Acting on
        ``current_weights`` (the controller's *decision*) instead of
        ``observed_ratios`` (a lagging measurement) ensures the
        autoscaler scales in lockstep with the controller, not ahead of
        a phantom deficit before the first adjustment, and not chasing
        stale completion ratios between adjustment ticks.

        Before the first adjustment ``current_weights == target_ratios``,
        so weight-skew is zero and no pre-warm pressure is generated.
        """
        with self._lock:
            return dict(self.target_ratios), dict(self.current_weights)
