"""Reward post-processors for miles.

binarize_reward: collapse graded rewards to {0, 1} then GRPO-normalize.
Adapted from AGISlime slime_plugins/nats_arena/reward_binary.py — imports only
miles' public Sample type.
"""

from __future__ import annotations

import logging
from math import sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def normalize_grouped_rewards(
    rewards: list[float], n_samples: int, std_normalization: bool
) -> list[float]:
    """Center rewards per group and optionally divide by sample stddev."""
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError("n_samples_per_prompt must be a positive integer")
    if len(rewards) % n_samples:
        raise ValueError(
            f"sample count {len(rewards)} is not divisible by "
            f"n_samples_per_prompt={n_samples}"
        )

    normalized = []
    for offset in range(0, len(rewards), n_samples):
        group = rewards[offset : offset + n_samples]
        mean = sum(group) / n_samples
        centered = [reward - mean for reward in group]
        if std_normalization:
            variance = (
                sum(reward * reward for reward in centered) / (n_samples - 1)
                if n_samples > 1
                else 0.0
            )
            std = sqrt(variance)
            centered = [reward / (std + 1e-6) for reward in centered]
        normalized.extend(centered)
    return normalized


def binarize_reward(args, samples: list[Sample]):
    """Binarize graded rewards: perfect (>=1.0) -> 1, anything else -> 0.

    Returns (raw_rewards, processed_rewards) where processed_rewards has GRPO
    group normalization applied after binarization. Intended as a reward
    post-processor over a group of ``args.n_samples_per_prompt`` rollouts.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    binary_rewards = [1.0 if r >= 1.0 else 0.0 for r in raw_rewards]

    n_perfect = int(sum(binary_rewards))
    n_total = len(binary_rewards)
    n_samples = args.n_samples_per_prompt
    rewards = normalize_grouped_rewards(
        binary_rewards, n_samples, args.grpo_std_normalization
    )
    if not samples:
        return raw_rewards, rewards

    n_groups = n_total // n_samples
    per_group_perfect = [
        int(sum(binary_rewards[offset : offset + n_samples]))
        for offset in range(0, n_total, n_samples)
    ]
    groups_with_signal = sum(1 for p in per_group_perfect if p > 0)
    groups_all_zero = n_groups - groups_with_signal

    logger.info(
        f"binarize_reward: {n_perfect}/{n_total} perfect ({n_perfect / n_total * 100:.1f}%), "
        f"groups_with_signal={groups_with_signal}/{n_groups}, "
        f"groups_all_zero={groups_all_zero}/{n_groups} (no gradient), "
        f"avg_graded={sum(raw_rewards) / n_total:.4f}, avg_binary={n_perfect / n_total:.4f}"
    )

    for i, sample in enumerate(samples):
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["binary_reward"] = binary_rewards[i]

    return raw_rewards, rewards
