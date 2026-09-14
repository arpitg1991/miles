"""Per-token ``advantage_scale`` in the advantage path.

``apply_advantage_scale`` multiplies a rollout-supplied per-token scale into
the per-token advantages before OPD and whitening. ``mask`` (0) and ``flip``
(-1) are the plain product; ``flip_positive`` keeps a negative scale only
where the advantage is positive and zeroes the token otherwise.
"""

from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils import loss as loss_utils
from miles.backends.training_utils.loss_hub.advantages import apply_advantage_scale

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
    ],
)
def test_compute_advantages_and_returns_applies_the_rule(fixed_advantages, rule, expected):
    rollout_data = _rollout_data([1.0, 0.0, -1.0, -1.0])

    loss_utils.compute_advantages_and_returns(_args(arena_truncated_turn_rule=rule), rollout_data)

    torch.testing.assert_close(rollout_data["advantages"][0], torch.tensor(expected))
    torch.testing.assert_close(rollout_data["returns"][0], torch.tensor(_ADV))  # returns untouched


def test_compute_advantages_and_returns_without_scale_or_rule_is_a_noop(fixed_advantages):
    rollout_data = _rollout_data(None)

    loss_utils.compute_advantages_and_returns(_args(), rollout_data)

    torch.testing.assert_close(rollout_data["advantages"][0], torch.tensor(_ADV))
