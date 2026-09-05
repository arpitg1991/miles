"""Per-gym CPU worker autoscaler for multi-gym training.

Polls NATS JetStream consumer backlog (``num_pending`` + ``num_ack_pending``)
for each gym's durable consumer ``gym-worker-<gym>`` and patches the per-gym
K8s Deployment ``<JOB_NAME>-gym-<gym_dashed>`` when the desired replica
count differs from the current count.

The autoscaler runs inside the trainer's ``NATSRolloutWorker`` thread,
alongside the ``MixtureController``. Their signals are complementary: the
mixture controller adjusts the *input* rate (prompts/s routed to a gym),
this scaler adjusts the *capacity* (pods available to run them). See
``docs/multi-gym-autoscaling.md`` (if present) for the full design.

Key guards:
    - 30-min warmup after start/restart: scale-up only, no scale-down.
    - Scale-down cooldown: ``desired < current`` must hold continuously for
      ``cooldown_secs`` AND active rollouts must fit in the new pod count.
    - RBAC 403: autoscaler disables itself (one warning, no crash).
    - Dry-run mode: log decisions without patching.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

from miles_plugins.arena.nats_arena.rollout_timing_tracker import (
    RolloutTimingTracker,
    TimingStats,
)

logger = logging.getLogger(__name__)


# Below this many samples in the long window the tracker's stats are too
# noisy to drive parameter derivation. Fall back to configured values.
# Tunable via --gym-autoscale-min-samples for long-rollout workloads where
# reaching 50 completions in a profile window is impractical.
_DEFAULT_MIN_TRACKER_SAMPLES = 50


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _format_tracker_stats(stats: TimingStats | None) -> str:
    """Render the tracker slice of the per-tick log line, or empty string."""
    if stats is None or stats.sample_count_long == 0:
        return ""
    return (
        f"p95_short={stats.p95_short:.1f}s p95_long={stats.p95_long:.1f}s "
        f"growth={stats.growth_ratio:.2f} cov={stats.cov_long:.2f} "
        f"n_long={stats.sample_count_long} "
    )


def _get_k8s_apps_client():
    """Lazy-import kubernetes SDK with in-cluster-config fallback.

    The runtime image must pin kubernetes==35.0.0: kubernetes-py 36.x breaks
    Authorization handling on EKS — every request from the pod returns 401
    even with a valid SA token — so autoscaling silently freezes (see the
    trainer entrypoint's pip install pin).
    """
    import kubernetes  # type: ignore[import-untyped]

    try:
        kubernetes.config.load_incluster_config()
    except kubernetes.config.ConfigException:
        kubernetes.config.load_kube_config()
    return kubernetes.client.AppsV1Api(), kubernetes.client.ApiException


class GymAutoscaler:
    """Feedback autoscaler for per-gym CPU worker Deployments.

    Args:
        gym_specs: ``{gym_name: {"deploy_name": str, "concurrency_per_pod": int,
            "min_replicas": int, "max_replicas": int, "initial_replicas": int}}``.
        namespace: Kubernetes namespace holding the gym Deployments.
        tasks_stream: JetStream tasks stream name (e.g. ``ARENA_TASKS``).
        consumer_prefix: Durable consumer prefix (e.g. ``gym-worker``).
        mixture_controller: Optional ``MixtureController`` — its
            ``get_observed_snapshot()`` is used to derive a ratio-deficit
            scale signal alongside the raw backlog signal.
        interval_secs: Min seconds between scaling ticks (default 30).
        cooldown_secs: Min seconds ``desired < current`` must hold before
            scaling down (default 180).
        warmup_secs: At startup, only scale-up decisions are honored for
            this many seconds (default 1800 = 30 min). Protects against
            falsely scaling down right after a restart, before the
            post-purge NATS queue has refilled.
        headroom: Target in-flight multiplier per pod — each pod should
            have ~``concurrency * headroom`` work outstanding before the
            fleet is considered fully fed (default 2.0).
        deficit_alpha: Damping factor for the observed-ratio nudge
            (default 0.5). 0 disables the mixture-controller signal.
        dry_run: Log decisions without issuing K8s patches.
    """

    def __init__(
        self,
        gym_specs: dict[str, dict[str, Any]],
        namespace: str,
        tasks_stream: str,
        consumer_prefix: str = "gym-worker",
        mixture_controller=None,
        timing_tracker: RolloutTimingTracker | None = None,
        interval_secs: float = 30.0,
        cooldown_secs: float = 180.0,
        warmup_secs: float = 1800.0,
        headroom: float = 1.25,
        deficit_alpha: float = 0.5,
        dry_run: bool = False,
        auto_tune: bool = True,
        growth_threshold: float = 1.3,
        growth_preemption: float = 1.25,
        profile_mode: bool = False,
        profile_duration_secs: float = 1800.0,
        profile_only: bool = False,
        min_tracker_samples: int = _DEFAULT_MIN_TRACKER_SAMPLES,
    ):
        self.gym_specs = dict(gym_specs)
        self.namespace = namespace
        self.tasks_stream = tasks_stream
        self.consumer_prefix = consumer_prefix
        self.mixture_controller = mixture_controller
        self.timing_tracker = timing_tracker
        self.interval_secs = interval_secs
        self.cooldown_secs = cooldown_secs
        self.warmup_secs = warmup_secs
        self.headroom = headroom
        self.deficit_alpha = deficit_alpha
        self.dry_run = dry_run
        self.auto_tune = auto_tune
        self.growth_threshold = growth_threshold
        self.growth_preemption = growth_preemption
        self.profile_mode = profile_mode
        self.profile_duration_secs = profile_duration_secs
        self.profile_only = profile_only
        self.min_tracker_samples = max(1, int(min_tracker_samples))

        self._lock = threading.Lock()
        self._started_at = time.time()
        self._last_tick_at = 0.0
        self._disabled = False

        # Per-gym state
        self._current_replicas: dict[str, int] = {
            g: spec["initial_replicas"] for g, spec in gym_specs.items()
        }
        self._scale_down_candidate_since: dict[str, float] = {}
        self._seeded_from_api: bool = False
        self._tick_count = 0
        # Remember which gyms we've already logged "tracker is now warm" for
        # so we only log once per gym when it becomes usable.
        self._tracker_warm_logged: set[str] = set()
        self._profile_report_emitted = False

        self._apps: Any = None
        self._api_exc: type[BaseException] = Exception

        logger.info(
            "GymAutoscaler configured: gyms=%s, interval=%.0fs, "
            "cooldown=%.0fs (floor), warmup=%.0fs, headroom=%.1f (floor), "
            "auto_tune=%s, growth_threshold=%.2f, profile_mode=%s, dry_run=%s",
            list(gym_specs.keys()),
            interval_secs,
            cooldown_secs,
            warmup_secs,
            headroom,
            auto_tune,
            growth_threshold,
            profile_mode,
            dry_run,
        )

    # ------------------------------------------------------------------
    # Public entry point: called from the worker loop.
    # ------------------------------------------------------------------

    async def maybe_scale(self, js) -> None:
        """If the poll interval has elapsed, read backlog + patch deployments.

        Safe to call every loop tick; internally rate-limited.
        """
        if self._disabled:
            return
        now = time.time()
        if now - self._last_tick_at < self.interval_secs:
            return
        self._last_tick_at = now
        self._tick_count += 1

        await self._ensure_seeded()
        if self._disabled:
            return

        target_ratios, observed_ratios = self._get_mixture_snapshot()
        # ``current_weights`` is the controller's *decision* about how to
        # split the next prompt batch (vs. ``observed_ratios``, which is
        # a lagging completion measurement). Pre-warming on weight-skew
        # ensures we add capacity in lockstep with — not ahead of — the
        # controller actually steering work toward a deficit gym.
        # Empty before any controller setup; falls back to no skew.
        current_weights = self._get_current_weights()
        in_warmup = (now - self._started_at) < self.warmup_secs
        warmup_remaining = max(0.0, self.warmup_secs - (now - self._started_at))
        in_profile = self._in_profile_window(now)

        # Fire the profile report before this tick's scaling decisions so
        # that, on the transition tick, --profile-only can disable the
        # autoscaler *before* it would otherwise start patching.
        self._maybe_emit_profile_report(now, in_profile)
        if self._disabled:
            return

        for gym_name, spec in self.gym_specs.items():
            try:
                await self._scale_one(
                    js, gym_name, spec, target_ratios, observed_ratios,
                    current_weights,
                    now, in_warmup, warmup_remaining, in_profile,
                )
            except Exception:
                logger.exception("Autoscaler error for gym=%s; skipping this tick", gym_name)

    def _in_profile_window(self, now: float) -> bool:
        if not self.profile_mode:
            return False
        return (now - self._started_at) < self.profile_duration_secs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _ensure_seeded(self) -> None:
        """On first tick, read live ``spec.replicas`` from K8s and use it as
        ground truth for ``self._current_replicas``. This matters after an
        auto-restart: a checkpoint-restored value may not match reality."""
        if self._seeded_from_api:
            return
        try:
            self._apps, self._api_exc = _get_k8s_apps_client()
        except Exception as exc:
            logger.warning(
                "GymAutoscaler: failed to init kubernetes client (%s); disabling.",
                exc,
            )
            self._disabled = True
            return

        for gym_name, spec in self.gym_specs.items():
            try:
                scale = self._apps.read_namespaced_deployment_scale(
                    spec["deploy_name"], self.namespace,
                )
                live = int(scale.spec.replicas or 0)
                self._current_replicas[gym_name] = live
                logger.info(
                    "GymAutoscaler: seeded gym=%s from live replicas=%d "
                    "(deploy=%s)",
                    gym_name, live, spec["deploy_name"],
                )
            except self._api_exc as exc:
                if getattr(exc, "status", None) == 403:
                    logger.warning(
                        "GymAutoscaler: RBAC denied reading Deployment scale "
                        "(403). Disabling autoscaler. Grant "
                        "get/patch on deployments/scale to the trainer "
                        "ServiceAccount."
                    )
                    self._disabled = True
                    return
                logger.warning(
                    "GymAutoscaler: failed to read scale for gym=%s (%s); "
                    "falling back to configured initial_replicas=%d.",
                    gym_name, exc, self._current_replicas[gym_name],
                )
        self._seeded_from_api = True

    def _get_mixture_snapshot(self) -> tuple[dict[str, float], dict[str, float]]:
        if self.mixture_controller is None:
            return {}, {}
        getter = getattr(self.mixture_controller, "get_observed_snapshot", None)
        if getter is None:
            return {}, {}
        try:
            return getter()
        except Exception:
            logger.debug("MixtureController snapshot failed", exc_info=True)
            return {}, {}

    def _get_current_weights(self) -> dict[str, float]:
        """Read the controller's current sampling weights (empty if no
        controller). Returns the dict that drives the data source's next
        gym pick — i.e. the *decision*, not the observation.
        """
        if self.mixture_controller is None:
            return {}
        getter = getattr(self.mixture_controller, "get_weight_snapshot", None)
        if getter is None:
            return {}
        try:
            _, current_weights = getter()
            return current_weights
        except Exception:
            logger.debug("MixtureController weight snapshot failed", exc_info=True)
            return {}

    async def _scale_one(
        self,
        js,
        gym_name: str,
        spec: dict[str, Any],
        target_ratios: dict[str, float],
        observed_ratios: dict[str, float],
        current_weights: dict[str, float],
        now: float,
        in_warmup: bool,
        warmup_remaining: float,
        in_profile: bool,
    ) -> None:
        consumer_name = f"{self.consumer_prefix}-{gym_name}"
        try:
            info = await js.consumer_info(self.tasks_stream, consumer_name)
        except Exception as exc:
            logger.warning(
                "autoscale[%s]: consumer_info(%s/%s) failed: %s",
                gym_name, self.tasks_stream, consumer_name, exc,
            )
            return

        num_pending = int(getattr(info, "num_pending", 0) or 0)
        num_ack_pending = int(getattr(info, "num_ack_pending", 0) or 0)

        concurrency = int(spec["concurrency_per_pod"])
        min_r = int(spec["min_replicas"])
        max_r = int(spec["max_replicas"])
        current = self._current_replicas.get(gym_name, int(spec["initial_replicas"]))

        stats = self._get_tracker_stats(gym_name)
        headroom_live, cooldown_live = self._derive_live_params(gym_name, stats)

        # `headroom` is a slack multiplier: headroom=1.0 means target 100%
        # utilization (no slack, sure to oscillate), headroom=2.0 means
        # target 50% utilization (lots of buffer). We convert it to a
        # target utilization bounded away from 0 and 1 so the math is
        # always meaningful.
        target_util = _clamp(1.0 / max(1.0, headroom_live), 0.3, 0.95)
        capacity = max(1, current * concurrency)
        utilization = num_ack_pending / capacity

        # --- Core scaling formula (see design notes):
        # `num_ack_pending` is a CAPACITY-UTILIZATION measurement, not an
        # independent demand signal. At steady saturation it equals
        # current × concurrency by definition. Mixing it into "backlog" and
        # dividing by capacity double-counts and causes runaway downscale.
        # Instead:
        #   - If num_pending > 0, the queue is the demand signal. Desired
        #     pods = total work / per-pod sustainable throughput.
        #   - If num_pending = 0, the fleet is keeping up. Only shrink when
        #     utilization drops below a scale-down threshold.
        scale_down_util_threshold = target_util * 0.75  # e.g. 0.6 if target=0.8
        if num_pending > 0:
            total_demand = num_ack_pending + num_pending
            desired_from_demand = math.ceil(total_demand / (concurrency * target_util))
        else:
            if utilization < scale_down_util_threshold:
                # Fleet is under-loaded; shrink toward target utilization.
                desired_from_demand = max(
                    1, math.ceil(num_ack_pending / (concurrency * target_util))
                )
            else:
                # Busy enough — don't touch the fleet. Preserve current size
                # as the baseline for the other scaling signals below.
                desired_from_demand = current

        # Pre-warm signal — *scale-up-only*: when the MixtureController
        # has decided to skew the next prompt batch toward this gym (i.e.
        # ``current_weights[gym] > target_ratios[gym]``), pre-emptively
        # add capacity so pods are ready when those prompts arrive in
        # NATS. Acting on ``current_weights`` (the *decision*) rather
        # than ``observed_ratios`` (a lagging measurement) means:
        #
        #   * No phantom deficit at startup. Before the first adjustment
        #     ``current_weights == target_ratios``, so skew==0 and no
        #     pressure is generated even though observed_ratios is empty.
        #     This stops the runaway 6 → 9 → 14 → 21 ... scale-up loop
        #     that fired every tick until completions started landing.
        #
        #   * No racing the controller. The autoscaler only pre-warms in
        #     lockstep with the controller's actual sampling decision,
        #     not based on stale completion ratios between adjustment
        #     ticks. The controller leads (decides skew), the autoscaler
        #     follows (adds capacity to absorb the skew).
        #
        # The floor is computed as an ABSOLUTE margin above ``min_r``,
        # not a multiplicative growth on ``current``. Compounding (e.g.
        # ``current * 1.2`` every tick) was the original design but
        # produced 6 → 8 → 10 → 12 over one controller window even
        # when the queue was empty between pushes — those pods then sat
        # idle until the cooldown timer evicted them, wasting CPU. The
        # absolute floor pre-warms a fixed buffer above the configured
        # minimum; the queue-demand path (above) is what actually grows
        # the fleet under real load. Self-clears when the controller
        # resets weights (skew → 0 → floor → 0).
        target = target_ratios.get(gym_name, 0.0)
        observed = observed_ratios.get(gym_name, 0.0)  # logging only
        decided_weight = current_weights.get(gym_name, target)
        weight_skew = max(0.0, decided_weight - target)
        deficit_floor = 0
        if target > 0 and self.deficit_alpha > 0 and weight_skew > 0:
            deficit_floor = int(math.ceil(
                min_r * (1.0 + self.deficit_alpha * weight_skew / target)
            ))

        desired = max(desired_from_demand, deficit_floor)

        # Growth preemption: if short-window p95 has stretched noticeably
        # above the long-run baseline, overshoot desired so backlog doesn't
        # chase the trend while pods spin up.
        growth_applied = False
        if (
            self.auto_tune
            and stats is not None
            and stats.sample_count_long >= self.min_tracker_samples
            and stats.growth_ratio > self.growth_threshold
            and desired > current  # only preempt when scaling up anyway
        ):
            desired = int(math.ceil(desired * self.growth_preemption))
            growth_applied = True

        desired = max(min_r, min(max_r, desired))

        action = self._decide_action(
            gym_name, current, desired, num_ack_pending, concurrency,
            now, in_warmup, cooldown_live, in_profile,
        )

        # Log every tick at INFO, including noops. Noop ticks are needed to
        # study the system: they reveal invisible candidate-since resets that
        # happen when a transient burst makes desired==current, which defeats
        # the cooldown dwell. Without noop visibility, operators see the
        # cooldown ticks but not the resets between them, which is confusing.
        logger.info(
            "autoscale[%s] current=%d desired=%d "
            "pending=%d ack_pending=%d util=%.2f target_util=%.2f "
            "observed=%.3f target=%.3f "
            "%sheadroom_live=%.2f cooldown_live=%.0fs action=%s%s%s",
            gym_name, current, desired,
            num_pending, num_ack_pending, utilization, target_util,
            observed, target,
            _format_tracker_stats(stats),
            headroom_live, cooldown_live, action,
            " [growth-preempt]" if growth_applied else "",
            (
                f" (warmup {warmup_remaining:.0f}s remaining)"
                if in_warmup and not in_profile
                else " (profile-mode: no patches)" if in_profile
                else ""
            ),
        )

        if action in ("up", "down"):
            if self._apply_scale(gym_name, spec["deploy_name"], desired):
                self._current_replicas[gym_name] = desired
            self._scale_down_candidate_since.pop(gym_name, None)

    def _decide_action(
        self,
        gym_name: str,
        current: int,
        desired: int,
        num_ack_pending: int,
        concurrency: int,
        now: float,
        in_warmup: bool,
        cooldown_live: float,
        in_profile: bool,
    ) -> str:
        """Return one of 'up', 'down', 'noop', 'cooldown', 'warmup_blocked',
        'profile_blocked'."""
        if in_profile:
            # Profile mode: observe only, never patch. Cooldown candidate
            # tracking is paused so profile does not prime scale-down.
            self._scale_down_candidate_since.pop(gym_name, None)
            return "profile_blocked"
        if desired > current:
            return "up"
        if desired == current:
            self._scale_down_candidate_since.pop(gym_name, None)
            return "noop"
        # desired < current — scale-down path
        if in_warmup:
            self._scale_down_candidate_since.pop(gym_name, None)
            return "warmup_blocked"
        # Don't evict pods that are actively running tasks.
        if num_ack_pending > desired * concurrency:
            self._scale_down_candidate_since.pop(gym_name, None)
            return "noop"
        first_seen = self._scale_down_candidate_since.get(gym_name)
        if first_seen is None:
            self._scale_down_candidate_since[gym_name] = now
            return "cooldown"
        if (now - first_seen) < cooldown_live:
            return "cooldown"
        return "down"

    def _get_tracker_stats(self, gym_name: str) -> TimingStats | None:
        if self.timing_tracker is None:
            return None
        try:
            stats = self.timing_tracker.get_stats(gym_name)
        except Exception:
            logger.debug("timing_tracker.get_stats(%s) failed", gym_name, exc_info=True)
            return None
        if (
            stats is not None
            and stats.sample_count_long >= self.min_tracker_samples
            and gym_name not in self._tracker_warm_logged
        ):
            logger.info(
                "autoscale[%s]: timing tracker is now warm "
                "(long=%d, p50=%.1fs, p95=%.1fs, cov=%.2f)",
                gym_name, stats.sample_count_long,
                stats.p50_long, stats.p95_long, stats.cov_long,
            )
            self._tracker_warm_logged.add(gym_name)
        return stats

    def _derive_live_params(
        self, gym_name: str, stats: TimingStats | None
    ) -> tuple[float, float]:
        """Return ``(headroom_live, cooldown_live)``, both clamped below
        by the configured values — autotune only increases conservatism."""
        if (
            not self.auto_tune
            or stats is None
            or stats.sample_count_long < self.min_tracker_samples
        ):
            return self.headroom, self.cooldown_secs

        # Headroom (slack multiplier). Bursty rollouts (high CoV) need more
        # slack so saturated pods don't cause backlog blow-up when the next
        # slow rollout lands. Clamp the CoV contribution so a single weird
        # window can't drive target_util below ~0.3 (headroom=3.0).
        headroom_live = max(
            self.headroom,
            1.0 + _clamp(stats.cov_long * 0.5, 0.1, 2.0),
        )
        # Cooldown: cover at least 3× poll interval and 1.5× p95 of the
        # long-window rollout duration.
        cooldown_live = max(
            self.cooldown_secs,
            3.0 * self.interval_secs,
            1.5 * stats.p95_long,
        )
        return headroom_live, cooldown_live

    def _maybe_emit_profile_report(self, now: float, in_profile: bool) -> None:
        """At the moment profile mode ends, emit a one-shot recommendation
        block and (if profile_only) disable further scaling."""
        if not self.profile_mode or self._profile_report_emitted:
            return
        if in_profile:
            return
        # Transition tick — profile just expired.
        self._profile_report_emitted = True

        lines = [
            "=" * 70,
            "AUTOSCALE PROFILE COMPLETE — recommended per-gym parameters",
            "=" * 70,
        ]
        for gym_name in self.gym_specs:
            stats = None
            if self.timing_tracker is not None:
                stats = self.timing_tracker.get_stats(gym_name)
            if stats is None or stats.sample_count_long < self.min_tracker_samples:
                lines.append(
                    f"  {gym_name}: insufficient samples "
                    f"(long={getattr(stats, 'sample_count_long', 0)}); "
                    f"keeping configured defaults."
                )
                continue
            headroom_rec = max(1.5, 1.0 + _clamp(stats.cov_long, 0.5, 3.0))
            cooldown_rec = max(3.0 * self.interval_secs, 1.5 * stats.p95_long)
            warmup_rec = max(600.0, 3.0 * stats.p95_long + 300.0)
            lines.append(
                f"  {gym_name}: p50={stats.p50_long:.1f}s p95={stats.p95_long:.1f}s "
                f"cov={stats.cov_long:.2f} growth={stats.growth_ratio:.2f} "
                f"(n={stats.sample_count_long})"
            )
            lines.append(
                f"    → recommended: "
                f"headroom={headroom_rec:.1f}, cooldown={cooldown_rec:.0f}s, "
                f"warmup={warmup_rec:.0f}s"
            )
        lines.append("=" * 70)
        for line in lines:
            logger.info(line)

        if self.profile_only:
            logger.info(
                "GymAutoscaler: --gym-autoscale-profile-only set; disabling "
                "further scaling. Copy the recommendations above into your "
                "training config and restart with --gym-autoscale."
            )
            self._disabled = True

    def _apply_scale(self, gym_name: str, deploy_name: str, replicas: int) -> bool:
        if self.dry_run:
            logger.info(
                "autoscale[%s] DRY-RUN: would patch %s to replicas=%d",
                gym_name, deploy_name, replicas,
            )
            return True
        try:
            self._apps.patch_namespaced_deployment_scale(
                deploy_name, self.namespace,
                {"spec": {"replicas": int(replicas)}},
            )
            logger.info(
                "autoscale[%s] patched %s -> replicas=%d",
                gym_name, deploy_name, replicas,
            )
            return True
        except self._api_exc as exc:
            if getattr(exc, "status", None) == 403:
                logger.warning(
                    "GymAutoscaler: RBAC denied patching %s (403). "
                    "Disabling autoscaler for the remainder of this run.",
                    deploy_name,
                )
                self._disabled = True
            else:
                logger.warning(
                    "autoscale[%s] patch failed (%s); will retry next tick.",
                    gym_name, exc,
                )
            return False

    # ------------------------------------------------------------------
    # Checkpoint helpers (optional; pod state is the real source of truth)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "current_replicas": dict(self._current_replicas),
                "scale_down_candidate_since": dict(self._scale_down_candidate_since),
                "tick_count": self._tick_count,
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with self._lock:
            # Note: ``current_replicas`` from the checkpoint is informational
            # only. The first tick seeds from live K8s via
            # ``read_namespaced_deployment_scale``, which is the true state.
            self._scale_down_candidate_since = dict(
                state.get("scale_down_candidate_since", {})
            )
            self._tick_count = int(state.get("tick_count", 0))
