"""--arena-truncated-turn-rule: per-token ``Sample.advantage_scale`` from the
gym's ``truncated_spans``.

The gym ships every mid-episode generate that hit the per-turn cap as a
``[start, end)`` span in the index space of the step's cumulative
``token_ids`` / ``loss_mask`` and keeps the loss mask at 1. The trainer maps
the span into the response window (from the first 1 on) and writes the rule
value: ``mask`` -> 0, ``flip`` -> -1, ``flip_positive`` -> -1 (the loss turns
it into ``where(adv > 0, -1, 0)``). No span, or an older gym image without
the field, leaves ``advantage_scale`` None.

Run: python -m pytest tests/fast/plugins/arena/test_truncated_spans_advantage_scale.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles_plugins.arena.nats_arena.nats_rollout import (
    _batch_telemetry,
    _result_to_episodes_full_trajectory,
    _result_to_samples_full_trajectory,
    _truncated_spans_scale,
)

# Two prompt tokens, then ten response tokens: response index i is token index i + 2.
_PROMPT_LEN = 2
_RESPONSE_LEN = 10
_LOSS_MASK = [0] * _PROMPT_LEN + [1] * _RESPONSE_LEN
# Token span [5, 8) -> response indices 3..5.
_SPAN = [[5, 8]]


def _step(loss_mask: list[int], **extra) -> dict:
    return {
        "token_ids": [7] * len(loss_mask),
        "loss_mask": list(loss_mask),
        "log_probs": [-0.1] * len(loss_mask),
        "stop_reason": "stop",
        "has_generate_tokens": True,
        **extra,
    }


def _result(steps: list[dict]) -> dict:
    return {
        "task_id": "t1",
        "gym_name": "g",
        "status": "success",
        "trajectories": [{"reward": 1.0, "steps": steps}],
    }


def _args(rule: str | None = "mask", **overrides) -> SimpleNamespace:
    fields = {
        "arena_mask_clipped_final_turn": False,
        "arena_keep_timeout_trajectories": False,
        "arena_keep_context_error_trajectories": False,
    }
    if rule is not None:
        fields["arena_truncated_turn_rule"] = rule
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _expected(value: float) -> list[float]:
    scale = [1.0] * _RESPONSE_LEN
    scale[3:6] = [value] * 3
    return scale


@pytest.mark.parametrize(
    ("rule", "value"),
    [("mask", 0.0), ("flip", -1.0), ("flip_positive", -1.0)],
)
def test_rule_value_written_on_span(rule: str, value: float) -> None:
    (s,) = _result_to_samples_full_trajectory(
        _result([_step(_LOSS_MASK, truncated_spans=_SPAN)]), tokenizer=None, args=_args(rule)
    )
    assert s.response_length == _RESPONSE_LEN
    assert s.advantage_scale == _expected(value)
    assert s.loss_mask == [1] * _RESPONSE_LEN  # the mask is not touched
    assert s.remove_sample is False
    assert "truncated_spans_out_of_range" not in s.metadata
    s.validate()


def test_default_rule_is_mask() -> None:
    (s,) = _result_to_samples_full_trajectory(
        _result([_step(_LOSS_MASK, truncated_spans=_SPAN)]), tokenizer=None, args=_args(rule=None)
    )
    assert s.advantage_scale == _expected(0.0)


@pytest.mark.parametrize("extra", [{}, {"truncated_spans": []}], ids=["field_missing", "empty_list"])
def test_no_spans_leaves_none(extra: dict) -> None:
    (s,) = _result_to_samples_full_trajectory(
        _result([_step(_LOSS_MASK, **extra)]), tokenizer=None, args=_args("flip")
    )
    assert s.advantage_scale is None


def test_span_past_token_stream_is_clipped_and_counted() -> None:
    (s,) = _result_to_samples_full_trajectory(
        _result([_step(_LOSS_MASK, truncated_spans=[[10, 20]])]), tokenizer=None, args=_args("mask")
    )
    assert s.advantage_scale == [1.0] * 8 + [0.0, 0.0]
    assert s.metadata["truncated_spans_out_of_range"] == 1


def test_span_before_response_window_is_a_silent_noop() -> None:
    # A first turn the gym did not train on (loss_mask 0) can carry a span;
    # nothing in the response window is affected and nothing is reported.
    mask = [0, 0, 0, 0, 1, 1, 1, 1]
    (s,) = _result_to_samples_full_trajectory(
        _result([_step(mask, truncated_spans=[[2, 4]])]), tokenizer=None, args=_args("mask")
    )
    assert s.advantage_scale is None
    assert "truncated_spans_out_of_range" not in s.metadata


def test_helper_clips_and_counts_malformed_spans() -> None:
    scale, bad = _truncated_spans_scale([[3, 3], [-1, 4], [4, 6]], prompt_len=2, response_length=6, args=_args("flip"))
    assert scale == [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
    assert bad == 2
    assert _truncated_spans_scale([[1, 2]], prompt_len=0, response_length=0, args=_args()) == (None, 0)


def test_multi_segment_spans_stay_per_segment() -> None:
    archived = _step([0, 1, 1, 1, 1], truncated_spans=[[1, 3]], truncated_generates=1, segment_end="compaction")
    archived.pop("stop_reason")
    final = _step(_LOSS_MASK, truncated_spans=[])
    (episode,) = _result_to_episodes_full_trajectory(
        _result([archived, final]), tokenizer=None, args=_args("mask", arena_train_segments="all")
    )
    a, f = episode
    assert a.advantage_scale == [0.0, 0.0, 1.0, 1.0]
    assert f.advantage_scale is None


def test_truncated_turn_tokens_metric() -> None:
    (s,) = _result_to_samples_full_trajectory(
        _result([_step(_LOSS_MASK, truncated_spans=_SPAN)]), tokenizer=None, args=_args("flip")
    )
    tm = _batch_telemetry([[s]], [[s]])
    assert tm["truncated_turn_tokens"] == pytest.approx(3 / _RESPONSE_LEN)
    assert tm["truncated_ratio"] == 0.0  # the final generate stopped cleanly
    assert tm["truncated_spans_out_of_range"] == 0
