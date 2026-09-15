"""Per-token ``advantage_scale`` in the advantage path.

``apply_advantage_scale`` multiplies a rollout-supplied per-token scale into
the per-token advantages before OPD and whitening. ``mask`` (0) and ``flip``
(-1) are the plain product; ``flip_positive`` keeps a negative scale only
where the advantage is positive and zeroes the token otherwise.
``apply_truncated_turn_shift`` (rule ``shift``) reads the scale as a span
marker and moves a positive advantage to ``max(adv - lam, min_adv)``.
"""

from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils import loss as loss_utils
from miles.backends.training_utils.log_utils import log_train_step
from miles.backends.training_utils.loss_hub.advantages import (
    apply_advantage_scale,
    apply_truncated_turn_shift,
)

# This module intentionally has no explicit CI registration call: modules under
# tests/fast are implicitly assigned to the stage-a-cpu suite by the CI collector.

_ADV = [2.0, -1.0, 0.5, -3.0]


def test_mask_and_flip_are_a_plain_multiply():
    advantages = [torch.tensor(_ADV)]
    scale = [torch.tensor([1.0, 0.0, -1.0, -1.0])]

    apply_advantage_scale(advantages, scale)

    torch.testing.assert_close(advantages[0], torch.tensor([2.0, 0.0, -0.5, 3.0]))


def test_flip_positive_flips_positive_and_zeroes_negative_advantages():
    advantages = [torch.tensor(_ADV)]
    scale = [torch.tensor([1.0, 0.0, -1.0, -1.0])]

    apply_advantage_scale(advantages, scale, positive_only=True)

    # Token 2: adv > 0 under -1 -> flipped. Token 3: adv < 0 under -1 -> zero.
    torch.testing.assert_close(advantages[0], torch.tensor([2.0, 0.0, -0.5, 0.0]))


def test_scale_is_detached_and_moved_to_the_advantage_dtype():
    source = torch.tensor([1.0, 0.0], requires_grad=True)
    advantages = [torch.tensor([1.0, 1.0], dtype=torch.float64)]

    apply_advantage_scale(advantages, [source * 1.0])

    assert advantages[0].requires_grad is False
    assert advantages[0].dtype == torch.float64
    torch.testing.assert_close(advantages[0], torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_shape_and_length_mismatch_raise():
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_advantage_scale([torch.ones(3)], [torch.ones(2)])
    with pytest.raises(ValueError, match="length mismatch"):
        apply_advantage_scale([torch.ones(3)], [])
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_truncated_turn_shift([torch.ones(3)], [torch.ones(2)], lam=0.5, min_adv=-1.0)
    with pytest.raises(ValueError, match="length mismatch"):
        apply_truncated_turn_shift([torch.ones(3)], [], lam=0.5, min_adv=-1.0)


@pytest.mark.parametrize(
    ("adv", "lam", "min_adv", "expected"),
    [
        (0.8, 0.5, -1.0, 0.3),  # plain shift
        (0.2, 0.5, -1.0, -0.3),  # crosses zero, above the floor
        (2.0, 5.0, -1.0, -1.0),  # clamped at the floor
        (0.0, 0.5, -1.0, 0.0),  # zero is unchanged
        (-0.7, 0.5, -1.0, -0.7),  # negative is unchanged
    ],
)
def test_shift_on_a_span_token(adv, lam, min_adv, expected):
    advantages = [torch.tensor([adv])]

    apply_truncated_turn_shift(advantages, [torch.tensor([-1.0])], lam=lam, min_adv=min_adv)

    torch.testing.assert_close(advantages[0], torch.tensor([expected]))


def test_shift_leaves_non_span_tokens_unchanged():
    advantages = [torch.tensor(_ADV)]

    apply_truncated_turn_shift(advantages, [torch.ones(4)], lam=0.5, min_adv=-1.0)

    torch.testing.assert_close(advantages[0], torch.tensor(_ADV))


def test_shift_mixed_spans_and_signs():
    advantages = [torch.tensor([0.8, 0.8, -1.0, 0.2, 5.0])]
    scale = [torch.tensor([1.0, -1.0, -1.0, -1.0, -1.0])]

    apply_truncated_turn_shift(advantages, scale, lam=0.5, min_adv=-1.0)

    torch.testing.assert_close(advantages[0], torch.tensor([0.8, 0.3, -1.0, -0.3, 4.5]))


def _rollout_data(scale: list[float] | None) -> dict:
    data = {
        "log_probs": [torch.zeros(4)],
        "rewards": [0.0],
        "values": None,
        "response_lengths": [4],
        "loss_masks": [torch.ones(4)],
        "total_lengths": [4],
    }
    if scale is not None:
        data["advantage_scale"] = [torch.tensor(scale)]
    return data


def _args(**overrides) -> Namespace:
    fields = dict(
        skip_actor_forward_only=False,
        use_rollout_logprobs=False,
        kl_coef=0.0,
        use_opd=False,
        normalize_advantages=False,
    )
    fields.update(overrides)
    return Namespace(**fields)


@pytest.fixture
def fixed_advantages(monkeypatch):
    def fake_compute_advantages(**kwargs):
        adv = torch.tensor(_ADV)
        return [adv], [adv.clone()]

    monkeypatch.setattr(loss_utils, "compute_advantages", fake_compute_advantages)


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("mask", [2.0, 0.0, -0.5, 3.0]),
        ("flip", [2.0, 0.0, -0.5, 3.0]),
        ("flip_positive", [2.0, 0.0, -0.5, 0.0]),
        # Marker -1 on tokens 2 and 3: 0.5 -> 0.0, -3.0 unchanged. Token 1 keeps
        # its 0 marker from the shared fixture; the shift ignores a scale >= 0.
        ("shift", [2.0, -1.0, 0.0, -3.0]),
    ],
)
def test_compute_advantages_and_returns_applies_the_rule(fixed_advantages, rule, expected):
    rollout_data = _rollout_data([1.0, 0.0, -1.0, -1.0])

    loss_utils.compute_advantages_and_returns(_args(arena_truncated_turn_rule=rule), rollout_data)

    torch.testing.assert_close(rollout_data["advantages"][0], torch.tensor(expected))
    torch.testing.assert_close(rollout_data["returns"][0], torch.tensor(_ADV))  # returns untouched
    if rule == "shift":
        torch.testing.assert_close(rollout_data["truncated_turn_shifted"][0], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    else:
        assert "truncated_turn_shifted" not in rollout_data


def test_compute_advantages_and_returns_shift_reads_lambda_and_floor(fixed_advantages):
    rollout_data = _rollout_data([-1.0, -1.0, -1.0, -1.0])
    args = _args(
        arena_truncated_turn_rule="shift", arena_truncated_turn_lambda=5.0, arena_truncated_turn_min_adv=-0.25
    )

    loss_utils.compute_advantages_and_returns(args, rollout_data)

    torch.testing.assert_close(rollout_data["advantages"][0], torch.tensor([-0.25, -1.0, -0.25, -3.0]))
    torch.testing.assert_close(rollout_data["truncated_turn_shifted"][0], torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_compute_advantages_and_returns_shift_defaults(fixed_advantages):
    # No lambda / floor on args: 0.5 and -1.0 (the argparse defaults).
    rollout_data = _rollout_data([-1.0, -1.0, -1.0, -1.0])

    loss_utils.compute_advantages_and_returns(_args(arena_truncated_turn_rule="shift"), rollout_data)

    torch.testing.assert_close(rollout_data["advantages"][0], torch.tensor([1.5, -1.0, 0.0, -3.0]))


def test_compute_advantages_and_returns_without_scale_or_rule_is_a_noop(fixed_advantages):
    rollout_data = _rollout_data(None)

    loss_utils.compute_advantages_and_returns(_args(), rollout_data)

    torch.testing.assert_close(rollout_data["advantages"][0], torch.tensor(_ADV))


def test_log_train_step_derives_the_neg_pos_ratio_from_reduced_masses():
    # The masses reduce as sums under one divisor; the ratio is taken after
    # the reduction, so it is a ratio of totals and not a mean of ratios.
    out = log_train_step(
        args=Namespace(),
        loss_dict={"adv_pos_mass": 2.0, "adv_neg_mass": 1.0, "truncated_turn_shifted_tokens": 3.0},
        grad_norm=0.0,
        rollout_id=0,
        step_id=0,
        num_steps_per_rollout=1,
        should_log=False,
    )

    assert out["train/adv_neg_pos_ratio"] == pytest.approx(0.5, rel=1e-5)
    assert out["train/truncated_turn_shifted_tokens"] == 3.0
    assert "train/adv_neg_pos_ratio" not in log_train_step(
        args=Namespace(),
        loss_dict={"loss": 1.0},
        grad_norm=0.0,
        rollout_id=0,
        step_id=0,
        num_steps_per_rollout=1,
        should_log=False,
    )
