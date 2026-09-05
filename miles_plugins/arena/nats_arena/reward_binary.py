"""Binary reward post-processor: score=1.0 → reward=1, else → reward=0."""

import logging
from math import sqrt

import torch

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def binarize_reward(args, samples: list[Sample]):
    """Binarize graded rewards: perfect (1.0) → 1, anything else → 0.

    Returns (raw_rewards, processed_rewards) where processed_rewards
    has GRPO group normalization applied after binarization.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    binary_rewards = [1.0 if r >= 1.0 else 0.0 for r in raw_rewards]

    n_perfect = int(sum(binary_rewards))
    n_total = len(binary_rewards)
    n_samples = args.n_samples_per_prompt

    rewards = torch.tensor(binary_rewards, dtype=torch.float)
    rewards = rewards.reshape(-1, n_samples)

    n_groups = rewards.shape[0]
    per_group_perfect = rewards.sum(dim=-1).int().tolist()
    groups_with_signal = sum(1 for p in per_group_perfect if p > 0)
    groups_all_zero = n_groups - groups_with_signal

    mean = rewards.mean(dim=-1, keepdim=True)
    rewards = rewards - mean

    if args.grpo_std_normalization:
        std = rewards.std(dim=-1, keepdim=True)
        rewards = rewards / (std + 1e-6)

    logger.info(
        f"binarize_reward: {n_perfect}/{n_total} perfect ({n_perfect/n_total*100:.1f}%), "
        f"groups_with_signal={groups_with_signal}/{n_groups}, "
        f"groups_all_zero={groups_all_zero}/{n_groups} (no gradient), "
        f"avg_graded={sum(raw_rewards)/n_total:.4f}, avg_binary={n_perfect/n_total:.4f}"
    )

    for g in range(n_groups):
        start = g * n_samples
        end = start + n_samples
        graded_g = raw_rewards[start:end]
        tid = ""
        if samples[start].metadata and "instance_id" in samples[start].metadata:
            tid = samples[start].metadata["instance_id"]
        logger.info(
            f"  group {g}: task={tid} "
            f"graded=[{', '.join(f'{r:.2f}' for r in graded_g)}] "
            f"perfect={per_group_perfect[g]}/{n_samples}"
        )

    for i, sample in enumerate(samples):
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["binary_reward"] = binary_rewards[i]

    return raw_rewards, rewards.flatten().tolist()


def renormalize_after_mask(
    rewards: list[float],
    samples: list[Sample],
    n_samples: int,
    std_normalization: bool,
) -> list[float]:
    """Re-normalize GRPO advantages per prompt-group over surviving samples.

    When failed/truncated samples are removed from training (``remove_sample``
    zeros their loss mask) but remain counted in the group mean/std, the
    surviving samples get a biased — typically inflated positive — advantage.
    This recomputes each group's mean/std using only survivors (``remove_sample``
    is False and ``response_length > 0``), removing that bias.

    Faithful port of slime's gym-evals ``--renormalize-after-mask`` block in
    ``rollout.py::_convert_samples_to_train_data``: operates on a copy of
    ``rewards`` (already group-normalized advantages, contiguous groups of
    ``n_samples``) and returns the renormalized list. Removed samples keep their
    incoming value (masked out of the loss regardless). A group with fewer than
    two survivors has its survivors zeroed (no usable variance -> no gradient).
    The survivor std uses the population convention (divide by count), matching
    upstream's ``std(unbiased=False)``.
    """
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError("n_samples_per_prompt must be a positive integer")
    if len(rewards) != len(samples):
        raise ValueError(
            f"rewards/samples length mismatch: {len(rewards)} != {len(samples)}"
        )
    if len(rewards) % n_samples:
        raise ValueError(
            f"sample count {len(rewards)} is not divisible by "
            f"n_samples_per_prompt={n_samples}"
        )

    out = list(rewards)
    for offset in range(0, len(out), n_samples):
        survivors = [
            offset + i
            for i in range(n_samples)
            if not getattr(samples[offset + i], "remove_sample", False)
            and getattr(samples[offset + i], "response_length", 0) > 0
        ]
        if len(survivors) < 2:
            for i in survivors:
                out[i] = 0.0
            continue
        surv_rewards = [out[i] for i in survivors]
        mean = sum(surv_rewards) / len(surv_rewards)
        if std_normalization:
            variance = sum((r - mean) ** 2 for r in surv_rewards) / len(surv_rewards)
            std = sqrt(variance)
            for i in survivors:
                out[i] = (out[i] - mean) / (std + 1e-6)
        else:
            for i in survivors:
                out[i] = out[i] - mean
    return out


def binarize_renormalize_after_mask(args, samples: list[Sample]):
    """``binarize_reward`` then GRPO advantage renormalization over survivors.

    miles ``custom_reward_post_process_func`` for binary-reward runs. Runs
    :func:`binarize_reward` (preserving its ``binary_reward`` metadata and
    per-group logging), then re-normalizes each prompt group's advantages over
    surviving (non-removed, non-empty) samples only via
    :func:`renormalize_after_mask`, fixing the positive-advantage bias from
    masked failed/truncated samples. Returns ``(raw_rewards, processed_rewards)``.
    """
    # Guard BEFORE binarize_reward: it divides by len(samples) in its per-group
    # logging (e.g. n_perfect/n_total), so an empty batch would ZeroDivisionError
    # before we ever reach a check placed after the call.
    if not samples:
        return [], []
    raw_rewards, rewards = binarize_reward(args, samples)
    rewards = renormalize_after_mask(
        rewards, samples, args.n_samples_per_prompt, args.grpo_std_normalization
    )
    return raw_rewards, rewards


def graded_reward(args, samples: list[Sample]):
    """Group-normalize GRADED (fractional) rewards — no binarization.

    Mirrors :func:`binarize_reward`'s GRPO group normalization but keeps the
    graded composite score (e.g. ``0.7*verifier + 0.3*rubric``) instead of
    thresholding it to {0, 1}. Binarization turns a genuine near-solve into
    reward 0 and collapses every group whose samples all fall short of a perfect
    1.0 to all-zero (no gradient); the graded signal preserves partial progress.
    Returns ``(raw_rewards, processed_rewards)`` where ``processed_rewards`` has
    GRPO group normalization applied.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    n_total = len(raw_rewards)
    n_samples = args.n_samples_per_prompt

    rewards = torch.tensor(raw_rewards, dtype=torch.float).reshape(-1, n_samples)
    n_groups = rewards.shape[0]
    # Pre-normalization per-group std: a group with ~zero reward variance yields
    # all-zero advantages (no gradient). Surfaced like binarize_reward's count.
    group_std = rewards.std(dim=-1)
    groups_no_signal = int((group_std <= 1e-6).sum().item())

    mean = rewards.mean(dim=-1, keepdim=True)
    rewards = rewards - mean
    if args.grpo_std_normalization:
        std = rewards.std(dim=-1, keepdim=True)
        rewards = rewards / (std + 1e-6)

    logger.info(
        f"graded_reward: n={n_total}, avg_graded={sum(raw_rewards)/n_total:.4f}, "
        f"groups_no_variance={groups_no_signal}/{n_groups} (no gradient)"
    )
    for g in range(n_groups):
        start = g * n_samples
        graded_g = raw_rewards[start : start + n_samples]
        tid = ""
        if samples[start].metadata and "instance_id" in samples[start].metadata:
            tid = samples[start].metadata["instance_id"]
        logger.info(
            f"  group {g}: task={tid} "
            f"graded=[{', '.join(f'{r:.2f}' for r in graded_g)}]"
        )

    return raw_rewards, rewards.flatten().tolist()


def graded_renormalize_after_mask(args, samples: list[Sample]):
    """``graded_reward`` then GRPO advantage renormalization over survivors.

    miles ``custom_reward_post_process_func`` for GRADED (non-binary) runs — the
    graded counterpart of :func:`binarize_renormalize_after_mask`. Runs
    :func:`graded_reward` (fractional composite, GRPO group-normalized), then
    :func:`renormalize_after_mask` to recompute each group's mean/std over
    surviving (non-removed, non-empty) samples only, fixing the
    positive-advantage bias that masked failed/truncated samples introduce.
    Returns ``(raw_rewards, processed_rewards)``.

    Prefer this over :func:`binarize_renormalize_after_mask` when a genuine
    partial solve should carry a proportional reward rather than being
    thresholded to 0 — the composite ``>= 1.0`` binarization otherwise zeroes
    near-solves and collapses most groups to all-zero (no gradient).
    """
    if not samples:
        return [], []
    raw_rewards, rewards = graded_reward(args, samples)
    rewards = renormalize_after_mask(
        rewards, samples, args.n_samples_per_prompt, args.grpo_std_normalization
    )
    return raw_rewards, rewards
