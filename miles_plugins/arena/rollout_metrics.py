"""Rollout-level training metrics: group aggregation and off-policy staleness.

Standalone implementations extracted from gym-evals slime/ray/rollout.py and
slime/utils/metric_utils.py. Importable without the full training stack.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_group_metrics_from_samples(samples: list[Any]) -> dict[str, float]:
    """Aggregate gym-side per-task group_metrics across a rollout batch.

    The gym dispatcher stamps a ``group_metrics`` dict on the first sample's
    metadata for each prompt group. This function collects and aggregates them
    into batch-level values for W&B logging under ``rollout/group_metrics/``.
    """
    if not samples:
        return {}
    per_group_metrics: list[dict] = []
    for s in samples:
        meta = s.metadata or {}
        gm = meta.get("group_metrics") if isinstance(meta, dict) else None
        if gm:
            per_group_metrics.append(gm)
    if not per_group_metrics:
        return {}

    all_keys: set[str] = set()
    for gm in per_group_metrics:
        all_keys.update(gm.keys())

    aggregated: dict[str, float] = {}
    aggregated["n_groups"] = float(len(per_group_metrics))
    for key in all_keys:
        out_key = key[len("group."):] if key.startswith("group.") else key
        values = [gm[key] for gm in per_group_metrics if key in gm and gm[key] is not None]
        if not values:
            continue
        if out_key.endswith(".max"):
            aggregated[out_key] = float(max(values))
        elif out_key.endswith(".min"):
            aggregated[out_key] = float(min(values))
        elif (
            out_key.endswith(".count")
            or out_key.startswith(("n_samples", "n_received", "n_failed", "n_rewarded"))
        ):
            aggregated[out_key] = float(sum(values))
        else:
            aggregated[out_key] = float(sum(values) / len(values))

    def _spread(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        n = len(values)
        return {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(sum(values) / n),
        }

    def _collect(suffix: str) -> list[float]:
        out: list[float] = []
        for gm in per_group_metrics:
            v = gm.get(f"group.{suffix}")
            if v is not None:
                out.append(float(v))
        return out

    for suffix, out_prefix in (
        ("reward.std", "within_group.reward_std"),
        ("reward.mean", "within_group.reward_mean"),
        ("reward.max", "within_group.reward_max"),
    ):
        for stat_key, stat_val in _spread(_collect(suffix)).items():
            aggregated[f"{out_prefix}.{stat_key}"] = stat_val

    return aggregated


def compute_off_policy_round_metrics(
    per_sample_weight_versions: list[list] | None,
    interval: int,
    reference_step: int | None,
    prefix: str = "off_policy_round",
) -> dict[str, float]:
    """Rollout staleness in train-step units.

    Measures how far the trainer's weights have advanced past the weights that
    generated a rollout. Returns ``{prefix}/*`` metrics or ``{}`` when nothing
    meaningful can be reported.
    """
    if not per_sample_weight_versions or reference_step is None:
        return {}

    try:
        import numpy as np
    except ImportError:
        return {}

    interval = max(1, int(interval or 1))

    off_policy_rounds = []
    num_untagged = 0
    for weight_versions in per_sample_weight_versions:
        if not weight_versions:
            num_untagged += 1
            continue
        try:
            versions = [int(v) for v in weight_versions]
        except (ValueError, TypeError):
            num_untagged += 1
            continue
        if not versions:
            num_untagged += 1
            continue
        mean_version = sum(versions) / len(versions)
        rollout_weight_step = (mean_version - 1) * interval
        off_policy_rounds.append(max(0.0, reference_step - rollout_weight_step))

    untagged_frac = float(num_untagged / len(per_sample_weight_versions))

    if not off_policy_rounds:
        return {f"{prefix}/untagged_frac": untagged_frac}

    off_policy_array = np.array(off_policy_rounds)

    return {
        f"{prefix}/mean": float(np.mean(off_policy_array)),
        f"{prefix}/max": float(np.max(off_policy_array)),
        f"{prefix}/min": float(np.min(off_policy_array)),
        f"{prefix}/std": float(np.std(off_policy_array)),
        f"{prefix}/on_policy_frac": float(np.mean(off_policy_array == 0)),
        f"{prefix}/untagged_frac": untagged_frac,
        f"{prefix}/current_step": float(reference_step),
    }


def compute_off_policy_metrics(args: Any, all_samples: list, rollout_id: int | None = None) -> dict[str, float]:
    """Wrapper that extracts weight_versions from samples and calls the core function."""
    if not all_samples:
        return {}
    return compute_off_policy_round_metrics(
        per_sample_weight_versions=[s.weight_versions for s in all_samples],
        interval=getattr(args, "update_weights_interval", 1),
        reference_step=rollout_id,
    )
