"""Off-policy corrections and per-age mismatch metrics for the arena train loop.

Both functions plug into ``--custom-tis-function-path`` and keep the built-in
seam: ``pg_loss`` comes in as ``-A * ratio_ppo`` per token; the function
multiplies the behavior correction weight ``w(r)`` into it, where
``r = exp(train_log_probs - rollout_log_probs)`` is the trainer-at-step-start
versus engine ratio (one optimizer step per batch, so the trainer's start-of-step
policy is the current policy).

- ``arena_tis_function``: the built-in truncated importance sampling,
  ``w = clamp(r, tis_clip_low, tis_clip)``, plus the metrics below.
- ``arena_m2po_function``: M2PO (Zheng, Zhao, Chen 2025, arXiv:2510.01161).
  ``w = r * M`` with the mask ``M`` that drops, largest first, the trust-region
  tokens (``A > 0 and r > 1`` or ``A < 0 and r < 1``) until the mean of
  ``(log r)^2`` over the remaining trust-region tokens is at most
  ``--arena-m2po-tau`` (paper default 0.04). Masked tokens stay in the
  denominator, as in the paper.

Metrics (token level, aggregated by the loss hub like ``tis``):
``m2`` = ``(log r)^2``; ``m2_masked_frac`` = masked trust-region tokens;
``age_b<k>_tokens`` / ``age_b<k>_abs_log_ratio`` / ``age_b<k>_m2`` per weight-age
bucket, so ``age_b<k>_abs_log_ratio / age_b<k>_tokens`` is the mean gap of the
tokens whose sample is ``k`` weight versions stale (``train_metadata["weight_age"]``
set at drain time by ``nats_rollout``).

ponytail: the M2PO mask is computed per micro-batch, not per global batch; the
paper's batch-level mean is approximated by the local token set (tens of
thousands of tokens at per-token loss). Upgrade path: all-reduce a histogram of
``(log r)^2`` across the data-parallel group before the cutoff search.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import torch

AGE_BUCKETS: tuple[tuple[int, int], ...] = ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 10**9))
M2PO_TAU_DEFAULT = 0.04


def add_off_policy_arguments(parser: Any) -> Any:
    """Register the M2PO threshold (the TIS bounds come from miles core)."""
    parser.add_argument(
        "--arena-m2po-tau",
        type=float,
        default=M2PO_TAU_DEFAULT,
        help="M2PO: mask the largest trust-region tokens until the mean (log r)^2 of the rest is at most tau.",
    )
    return parser


def _age_of(batch: dict[str, Any] | None, index: int) -> int | None:
    """Weight age of sample ``index`` from the drain-time tag; None when untagged."""
    if batch is None:
        return None
    metadata = batch.get("metadata")
    if metadata is None or index >= len(metadata) or metadata[index] is None:
        return None
    age = metadata[index].get("weight_age")
    return None if age is None else int(age)


def _expand_per_sample(values: list[float], lengths: list[int], like: torch.Tensor) -> torch.Tensor:
    """Repeat one scalar per sample over that sample's local token chunk."""
    return torch.cat(
        [like.new_full((n,), v) for v, n in zip(values, lengths, strict=True)]
    )


def _local_masks(args: Namespace, kwargs: dict[str, Any], like: torch.Tensor) -> torch.Tensor:
    """Local (context-parallel chunk) response masks concatenated, same shape as the log probs."""
    from miles.backends.training_utils.cp_utils import get_local_response_loss_masks

    masks = get_local_response_loss_masks(
        kwargs["total_lengths"],
        kwargs["response_lengths"],
        kwargs["loss_masks"],
        args.qkv_format,
        kwargs.get("max_seq_lens"),
    )
    return torch.cat(masks, dim=0).to(device=like.device, dtype=like.dtype)


def mismatch_metrics(
    log_ratio: torch.Tensor,
    active: torch.Tensor,
    ages: list[int | None],
    lengths: list[int],
) -> dict[str, torch.Tensor]:
    """Token-level ``m2`` and per-age-bucket indicator / gap / m2 tensors."""
    abs_log_ratio = log_ratio.abs() * active
    m2 = log_ratio.square() * active
    metrics = {"m2": m2.clone().detach(), "abs_log_ratio": abs_log_ratio.clone().detach()}
    if all(a is None for a in ages):
        return metrics
    age_tokens = _expand_per_sample([float(-1 if a is None else a) for a in ages], lengths, log_ratio)
    for lo, hi in AGE_BUCKETS:
        tag = f"age_b{lo}" if lo == hi else f"age_b{lo}_{hi if hi < 10**9 else 'up'}"
        in_bucket = ((age_tokens >= lo) & (age_tokens <= hi)).to(log_ratio.dtype) * active
        metrics[f"{tag}_tokens"] = in_bucket.clone().detach()
        metrics[f"{tag}_abs_log_ratio"] = (abs_log_ratio * in_bucket).clone().detach()
        metrics[f"{tag}_m2"] = (m2 * in_bucket).clone().detach()
    return metrics


def m2po_mask(log_ratio: torch.Tensor, trust_region: torch.Tensor, tau: float) -> torch.Tensor:
    """M2PO Algorithm 1 on one token set: 1 keeps the token, 0 masks it.

    Drops the trust-region tokens with the largest ``(log r)^2`` until the mean
    over the remaining trust-region tokens is at most ``tau``. Non-trust-region
    tokens are never masked. Sorting replaces the paper's one-at-a-time loop:
    with ``s`` the descending squares and ``c`` their suffix means, the first
    ``k`` with ``c[k] <= tau`` is the number of tokens to drop.
    """
    keep = torch.ones_like(log_ratio)
    idx = trust_region.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return keep
    sq = log_ratio[idx].square()
    if sq.mean() <= tau:
        return keep
    s, order = torch.sort(sq, descending=True)
    n = s.numel()
    # suffix_mean[k] = mean(s[k:]) = mean of the tokens that remain after dropping the k largest
    suffix_sum = torch.flip(torch.cumsum(torch.flip(s, dims=[0]), dim=0), dims=[0])
    counts = torch.arange(n, 0, -1, device=s.device, dtype=s.dtype)
    suffix_mean = suffix_sum / counts
    ok = (suffix_mean <= tau).nonzero(as_tuple=True)[0]
    k = int(ok[0].item()) if ok.numel() > 0 else n
    keep[idx[order[:k]]] = 0.0
    return keep


def _ratio_inputs(
    args: Namespace, kwargs: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int | None]]:
    train = torch.cat(kwargs["train_log_probs"], dim=0)
    rollout = torch.cat(kwargs["rollout_log_probs"], dim=0)
    log_ratio = torch.nan_to_num(train - rollout, nan=0.0, posinf=0.0, neginf=0.0)
    active = _local_masks(args, kwargs, log_ratio) if kwargs.get("total_lengths") is not None else torch.ones_like(log_ratio)
    lengths = [lp.shape[0] for lp in kwargs["train_log_probs"]]
    ages = [_age_of(kwargs.get("batch"), i) for i in range(len(lengths))]
    return log_ratio, active, lengths, ages


def arena_tis_function(
    args: Namespace,
    *,
    pg_loss: torch.Tensor,
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """Built-in truncated IS (``clamp(r, tis_clip_low, tis_clip)``) plus mismatch metrics."""
    kwargs["loss_masks"] = loss_masks
    log_ratio, active, lengths, ages = _ratio_inputs(args, kwargs)
    ratio = log_ratio.exp()
    weights = torch.clamp(ratio, min=args.tis_clip_low, max=args.tis_clip)
    metrics = {
        "tis": ratio.clone().detach(),
        "tis_clipfrac": (weights != ratio).to(ratio.dtype).clone().detach(),
        "tis_abs": (ratio - 1).abs().clone().detach(),
        **mismatch_metrics(log_ratio, active, ages, lengths),
    }
    return pg_loss * weights, loss_masks, metrics


def arena_m2po_function(
    args: Namespace,
    *,
    pg_loss: torch.Tensor,
    loss_masks: list[torch.Tensor],
    advantages: list[torch.Tensor] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """M2PO: unclipped IS weight times the second-moment trust-region mask."""
    kwargs["loss_masks"] = loss_masks
    log_ratio, active, lengths, ages = _ratio_inputs(args, kwargs)
    ratio = log_ratio.exp()
    if advantages is not None:
        adv = torch.cat(advantages, dim=0)
    else:
        # pg_loss = -A * ratio_ppo with ratio_ppo > 0, so the advantage sign is -sign(pg_loss).
        adv = -pg_loss.detach()
    trust_region = (((adv > 0) & (log_ratio > 0)) | ((adv < 0) & (log_ratio < 0))) & active.bool()
    keep = m2po_mask(log_ratio, trust_region, args.arena_m2po_tau)
    masked = (1.0 - keep) * trust_region.to(ratio.dtype)
    weights = ratio * keep
    metrics = {
        "tis": ratio.clone().detach(),
        "tis_clipfrac": masked.clone().detach(),
        "tis_abs": (ratio - 1).abs().clone().detach(),
        "m2_masked_frac": masked.clone().detach(),
        "m2_trust_region_frac": trust_region.to(ratio.dtype).clone().detach(),
        **mismatch_metrics(log_ratio, active, ages, lengths),
    }
    return pg_loss * weights, loss_masks, metrics
