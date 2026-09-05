"""Runnable self-checks for miles_plugins.arena plugins.

Run: python tests/fast/plugins/arena/test_plugins.py
(or via pytest). No fixtures/frameworks beyond pytest's runner.
"""

from math import isclose, isfinite, sqrt
from types import SimpleNamespace

from miles_plugins.arena.parsers import (
    register_nova_reasoning_parser,
    split_reasoning,
)
from miles_plugins.arena.rewards import (
    binarize_reward,
    normalize_grouped_rewards,
)


def test_split_reasoning_complete():
    r, c = split_reasoning("<|begin_internal_thought|>think<|end_internal_thought|>answer")
    assert r == "think"
    assert c == "answer"


def test_split_reasoning_none():
    r, c = split_reasoning("just an answer")
    assert r == ""
    assert c == "just an answer"


def test_split_reasoning_unterminated():
    r, c = split_reasoning("<|begin_internal_thought|>still thinking")
    assert r == "still thinking"
    assert c == ""


def test_register_nova_reasoning_parser():
    if not register_nova_reasoning_parser():
        return
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    assert ReasoningParser.DetectorMap["nova"] is register_nova_reasoning_parser.detector_cls
    parser = ReasoningParser("nova")
    assert parser.parse_non_stream(
        "<|begin_internal_thought|>think<|end_internal_thought|>answer"
    ) == ("think", "answer")


class _Sample:
    def __init__(self, reward):
        self.reward = reward
        self.metadata = {}

    def get_reward_value(self, _args):
        return self.reward


def _args(n_samples_per_prompt):
    return SimpleNamespace(
        n_samples_per_prompt=n_samples_per_prompt,
        grpo_std_normalization=True,
        reward_key=None,
    )


def test_binarize_reward_importable():
    assert callable(binarize_reward)


def test_binarize_reward_empty_group():
    assert binarize_reward(_args(1), []) == ([], [])


def test_binarize_reward_single_sample_is_finite():
    raw, processed = binarize_reward(_args(1), [_Sample(1.0)])
    assert raw == [1.0]
    assert processed == [0.0]
    assert all(isfinite(value) for value in processed)


def test_binarize_reward_rejects_incomplete_group():
    try:
        binarize_reward(_args(2), [_Sample(1.0)])
    except ValueError:
        return
    raise AssertionError("incomplete reward group accepted")


def test_binarize_reward_rejects_invalid_group_size():
    try:
        binarize_reward(_args(0), [])
    except ValueError:
        return
    raise AssertionError("invalid reward group size accepted")


def test_binarize_reward_preserves_sample_std_normalization():
    processed = normalize_grouped_rewards([0.0, 1.0], 2, True)
    expected = 0.5 / (sqrt(0.5) + 1e-6)
    assert isclose(processed[0], -expected)
    assert isclose(processed[1], expected)


if __name__ == "__main__":
    # Run every test_* in this module with plain python (no pytest required).
    failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {_name}: {_e}")
    raise SystemExit(1 if failures else 0)
