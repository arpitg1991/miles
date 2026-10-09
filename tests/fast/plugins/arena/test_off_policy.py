"""CPU checks for miles_plugins.arena.off_policy (M2PO mask, TIS seam, per-age metrics)."""

from argparse import Namespace

import pytest
import torch

from miles_plugins.arena import off_policy
from miles_plugins.arena.off_policy import (
    AGE_BUCKETS,
    arena_m2po_function,
    arena_tis_function,
    m2po_mask,
    mismatch_metrics,
)


def _kwargs(train, rollout, advantages, ages=None):
    lengths = [t.shape[0] for t in train]
    batch = {"metadata": [None if a is None else {"weight_age": a} for a in ages]} if ages is not None else None
    adv = [torch.full((n,), float(a)) for a, n in zip(advantages, lengths, strict=True)]
    ratio_ppo = torch.ones(sum(lengths))
    pg_loss = -torch.cat(adv) * ratio_ppo
    return dict(
        pg_loss=pg_loss,
        train_log_probs=list(train),
        rollout_log_probs=list(rollout),
        loss_masks=[torch.ones(n) for n in lengths],
        total_lengths=[n + 4 for n in lengths],
        response_lengths=lengths,
        max_seq_lens=None,
        advantages=adv,
        batch=batch,
    )


def _args(**over):
    base = dict(tis_clip=2.0, tis_clip_low=0.0, arena_m2po_tau=0.04, qkv_format="thd")
    base.update(over)
    return Namespace(**base)


@pytest.fixture(autouse=True)
def _no_cp(monkeypatch):
    """The local-mask helper needs Megatron parallel state; identity masks stand in on CPU."""

    def local(args, kwargs, like):
        return torch.cat(kwargs["loss_masks"], dim=0).to(like.dtype)

    monkeypatch.setattr(off_policy, "_local_masks", local)


def test_m2po_mask_keeps_everything_under_tau():
    log_ratio = torch.tensor([0.1, -0.1, 0.05, 0.0])
    keep = m2po_mask(log_ratio, torch.ones(4, dtype=torch.bool), tau=0.04)
    assert keep.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_m2po_mask_drops_largest_until_mean_under_tau():
    log_ratio = torch.tensor([1.0, 0.5, 0.1, 0.1, 0.1, 0.1])  # squares 1, .25, .01 x4
    keep = m2po_mask(log_ratio, torch.ones(6, dtype=torch.bool), tau=0.04)
    # dropping 1.0 leaves mean (.25+.04)/5 = .058 > tau; dropping .5 too leaves .01 <= tau
    assert keep.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    kept = log_ratio[keep.bool()].square().mean().item()
    assert kept <= 0.04


def test_m2po_mask_never_touches_non_trust_region_tokens():
    log_ratio = torch.tensor([3.0, 3.0, 0.1])
    trust = torch.tensor([False, True, True])
    keep = m2po_mask(log_ratio, trust, tau=0.04)
    assert keep.tolist() == [1.0, 0.0, 1.0]


def test_m2po_function_matches_paper_objective_and_reports_mask():
    # sample 0: A>0, ratio>1 on token 0 by a lot (trust region, masked); sample 1: A<0, ratio>1 (not trust region)
    train = [torch.tensor([2.0, 0.0, 0.0]), torch.tensor([0.5, 0.5])]
    rollout = [torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0])]
    kw = _kwargs(train, rollout, advantages=[1.0, -1.0], ages=[0, 7])
    pg_loss, masks, metrics = arena_m2po_function(_args(), **kw)
    ratio = torch.cat(train).exp()
    expected_keep = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
    expected = kw["pg_loss"] * ratio * expected_keep
    assert torch.allclose(pg_loss, expected)
    assert [m.tolist() for m in masks] == [m.tolist() for m in kw["loss_masks"]]  # denominator unchanged
    assert metrics["m2_masked_frac"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert metrics["m2_trust_region_frac"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert torch.allclose(metrics["m2"], torch.cat(train).square())
    # the age-7 sample lands in the 6-10 bucket with its two tokens
    assert metrics["age_b6_10_tokens"].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert metrics["age_b0_tokens"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert torch.allclose(metrics["age_b6_10_abs_log_ratio"], torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5]))


def test_m2po_function_falls_back_to_pg_loss_sign_without_advantages():
    train = [torch.tensor([2.0, 0.0])]
    rollout = [torch.tensor([0.0, 0.0])]
    kw = _kwargs(train, rollout, advantages=[1.0])
    kw["advantages"] = None
    pg_loss, _, metrics = arena_m2po_function(_args(), **kw)
    assert metrics["m2_masked_frac"].tolist() == [1.0, 0.0]
    assert pg_loss[0].item() == 0.0


def test_tis_function_equals_builtin_and_adds_metrics():
    from miles.backends.training_utils.loss_hub.corrections import vanilla_tis_function

    train = [torch.tensor([0.3, -0.2, 1.5]), torch.tensor([0.0, 0.1])]
    rollout = [torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0])]
    kw = _kwargs(train, rollout, advantages=[1.0, -1.0], ages=[3, None])
    ours, masks, metrics = arena_tis_function(_args(), **kw)
    theirs, _, their_metrics = vanilla_tis_function(
        _args(),
        pg_loss=kw["pg_loss"],
        train_log_probs=kw["train_log_probs"],
        rollout_log_probs=kw["rollout_log_probs"],
        loss_masks=kw["loss_masks"],
    )
    assert torch.allclose(ours, theirs)
    assert torch.allclose(metrics["tis_clipfrac"], their_metrics["tis_clipfrac"])
    assert metrics["age_b3_5_tokens"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    # the untagged sample is in no bucket
    assert sum(metrics[f"{tag}_tokens"][3:].sum().item() for tag in _tags()) == 0.0


def test_mismatch_metrics_without_ages_reports_only_m2():
    log_ratio = torch.tensor([0.1, -0.2])
    metrics = mismatch_metrics(log_ratio, torch.ones(2), [None], [2])
    assert set(metrics) == {"m2", "abs_log_ratio"}


def _tags():
    return [f"age_b{lo}" if lo == hi else f"age_b{lo}_{hi if hi < 10**9 else 'up'}" for lo, hi in AGE_BUCKETS]


def test_train_metadata_reaches_every_dp_shard():
    """The drain-time weight age travels as ``train_metadata`` -> ``metadata``; each DP shard keeps its rows."""
    from tests.fast.ray.rollout.conftest import make_args, make_sample

    from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp_raw

    samples = [make_sample(reward=float(i % 2)) for i in range(4)]
    for i, s in enumerate(samples):
        s.train_metadata = {"weight_age": i}
    data = convert_samples_to_train_data(
        make_args(rewards_normalization=False),
        samples,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )
    assert data["metadata"] == [{"weight_age": i} for i in range(4)]
    shards = split_train_data_by_dp_raw(make_args(rewards_normalization=False), data, dp_size=2)
    ages = sorted(m["weight_age"] for shard in shards for m in shard["metadata"])
    assert ages == [0, 1, 2, 3]
    for shard in shards:
        assert len(shard["metadata"]) == len(shard["tokens"])
