"""Gym ``weight_version_spans`` become upstream ``Sample.weight_versions`` (ADR-0018).

Each step that the training gym ships carries one ``{"version", "start",
"end"}`` entry per model call that has an SGLang weight version.
``[start, end)`` is the output of that call in the step ``token_ids``. The
trainer builds one ``WeightVersionsPerCall`` per entry, so the upstream
checks and metrics read arena samples. The file also covers the
consume-time staleness filter of ``generate_rollout`` and ``NatsRolloutFn``.

Run:
    python -m pytest tests/fast/plugins/arena/test_weight_versions.py -v
"""

from __future__ import annotations

import logging
import sys
import types as pytypes

import pytest
from tests.fast.plugins.arena.test_group_identity import _make_args, _make_worker, _StubMaskGenerator
from tests.fast.plugins.arena.test_multi_segment_episodes import _FakeWorker, _rollout_args, _sample

import miles.utils.mask_utils as mask_utils
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.inference_rollout.compatibility import load_rollout_function
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall
from miles.utils.weight_version import assert_samples_weight_version_sane
from miles_plugins.arena.nats_arena import nats_rollout
from miles_plugins.arena.nats_arena.nats_rollout import (
    _MIN_GYM_IMAGE_FOR_WEIGHT_VERSION_SPANS,
    NatsRolloutFn,
    WeightVersionSpansError,
    _add_arena_arguments,
    _result_to_samples_full_trajectory,
    generate_rollout,
)

_LOGGER = "miles_plugins.arena.nats_arena.nats_rollout"

# One step of a two-call episode, in the gym token index space:
# prompt 101-103 | call 1 output 201-202 | tool output 301-302 | call 2 output 401-403.
_TOKEN_IDS = [101, 102, 103, 201, 202, 301, 302, 401, 402, 403]
_LOSS_MASK = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]
_SPANS = [{"version": 4, "start": 3, "end": 5}, {"version": 5, "start": 7, "end": 10}]


def _step(spans: list | None = _SPANS, **extra) -> dict:
    step = {
        "token_ids": list(_TOKEN_IDS),
        "loss_mask": list(_LOSS_MASK),
        "log_probs": [-0.1] * len(_TOKEN_IDS),
        "stop_reason": "stop",
        "has_generate_tokens": True,
        **extra,
    }
    if spans is not None:
        step["weight_version_spans"] = spans
    return step


def _result(steps: list[dict], weight_versions: list | None = (4, 5), **traj_extra) -> dict:
    traj: dict = {"reward": 1.0, "steps": steps, **traj_extra}
    if weight_versions is not None:
        traj["weight_versions"] = list(weight_versions)
    return {"task_id": "t1", "gym_name": "g", "status": "success", "trajectories": [traj]}


def _convert(result: dict, args=None) -> list[Sample]:
    # The fast path never touches the tokenizer.
    return _result_to_samples_full_trajectory(result, tokenizer=None, args=args)


def _calls(*spans: tuple[str, int, int]) -> list[WeightVersionsPerCall]:
    return [WeightVersionsPerCall(spans=[WeightVersionSpan(v, a, b)]) for v, a, b in spans]


# ===========================================================================
# 1. Conversion: one upstream call per span, offset 0
# ===========================================================================


def test_spans_become_one_upstream_call_each():
    (s,) = _convert(_result([_step()]))
    assert s.weight_versions == _calls(("4", 3, 5), ("5", 7, 10))
    assert s.oldest_weight_version == 4
    assert "arena_weight_versions" not in s.metadata


def test_span_offset_is_zero_against_tokens_and_loss_mask():
    """The gym index space is the Sample.tokens index space: the same tokens, and loss mask 1 on each output."""
    (s,) = _convert(_result([_step()]))
    assert s.tokens == _TOKEN_IDS
    prompt_length = len(s.tokens) - s.response_length
    assert prompt_length == 3
    outputs = [s.tokens[span.abs_start : span.abs_end] for span in s.all_weight_version_spans]
    assert outputs == [[201, 202], [401, 402, 403]]
    for span in s.all_weight_version_spans:
        assert s.loss_mask[span.abs_start - prompt_length : span.abs_end - prompt_length] == [1] * (
            span.abs_end - span.abs_start
        )
        assert len(s.rollout_log_probs[span.abs_start - prompt_length : span.abs_end - prompt_length]) == (
            span.abs_end - span.abs_start
        )


def test_upstream_checks_accept_arena_samples():
    (s,) = _convert(_result([_step()]))
    s.validate()
    args = pytypes.SimpleNamespace(debug_rollout_only=False, debug_skip_weight_update=False)
    assert_samples_weight_version_sane(args, [s])
    # The debug rollout dump keeps the spans.
    assert Sample.from_dict(s.to_dict()).weight_versions == s.weight_versions


def test_no_weight_versions_gives_no_spans():
    """Eval and no-SGLang paths: no version on any call, so no spans, as upstream allows."""
    (s,) = _convert(_result([_step(spans=None)], weight_versions=None))
    assert s.weight_versions == []
    (s,) = _convert(_result([_step(spans=[])], weight_versions=None))
    assert s.weight_versions == []


def test_hard_overflow_cuts_spans_as_upstream_strip_does():
    args = pytypes.SimpleNamespace(rollout_max_context_len=8)
    (cut,) = _convert(_result([_step()]), args=args)
    assert cut.remove_sample and len(cut.tokens) == 8
    assert cut.weight_versions == _calls(("4", 3, 5), ("5", 7, 8))
    (full,) = _convert(_result([_step()]))
    full.strip_last_output_tokens(2, tokenizer=pytypes.SimpleNamespace(decode=lambda ids: ""))
    assert cut.weight_versions == full.weight_versions
    # A call whose output starts past the cut keeps its entry with no span, as upstream.
    (s,) = _convert(_result([_step()]), args=pytypes.SimpleNamespace(rollout_max_context_len=6))
    assert s.weight_versions == [WeightVersionsPerCall(spans=[WeightVersionSpan("4", 3, 5)]), WeightVersionsPerCall()]


def _inspect_step(prompt: list[int], output: list[int], spans: list | None) -> dict:
    """One step in the shape of the Inspect streaming ``TrajectoryState`` (no Harbor keys)."""
    step = {
        "token_ids": prompt + output,
        "loss_mask": [0] * len(prompt) + [1] * len(output),
        "log_probs": [0.0] * len(prompt) + [-0.1] * len(output),
        "has_generate_tokens": True,
        "finish_reasons": ["stop"],
    }
    if spans is not None:
        step["weight_version_spans"] = spans
    return step


@pytest.mark.parametrize("mode", ["final", "all"])
@pytest.mark.parametrize("provider", ["sglang_full", "sglang_perstep"])
def test_inspect_streaming_spans_have_offset_zero(provider, mode):
    """Both Inspect providers: the trained step token_ids are Sample.tokens, so the span offset is 0."""
    if provider == "sglang_full":
        # One cumulative step: call 1 output 201-202, tool output 301, call 2 output 401-402.
        steps = [
            _inspect_step(
                [101, 102],
                [201, 202, 301, 401, 402],
                [{"version": 4, "start": 2, "end": 4}, {"version": 5, "start": 5, "end": 7}],
            )
        ]
        steps[0]["loss_mask"][4] = 0
        outputs = [[201, 202], [401, 402]]
    else:
        # One self-contained step per call. The trainer trains the last step.
        steps = [
            _inspect_step([101, 102], [201, 202], [{"version": 4, "start": 2, "end": 4}]),
            _inspect_step([101, 102, 201, 202, 301], [401, 402], [{"version": 5, "start": 5, "end": 7}]),
        ]
        outputs = [[401, 402]]
    (s,) = _convert(_result(steps), args=_make_args(arena_train_segments=mode))
    assert s.tokens == steps[-1]["token_ids"]
    assert [s.tokens[span.abs_start : span.abs_end] for span in s.all_weight_version_spans] == outputs
    prompt_length = len(s.tokens) - s.response_length
    for span in s.all_weight_version_spans:
        width = span.abs_end - span.abs_start
        assert s.loss_mask[span.abs_start - prompt_length : span.abs_end - prompt_length] == [1] * width


def test_slow_path_has_no_spans(monkeypatch):
    """The messages-only path makes its own tokens, so the gym spans do not apply."""
    monkeypatch.setattr(mask_utils, "MultiTurnLossMaskGenerator", _StubMaskGenerator)
    traj_messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    result = _result([{"stop_reason": "stop"}], messages=traj_messages)
    (s,) = _result_to_samples_full_trajectory(result, tokenizer=None, args=_make_args(use_rollout_logprobs=False))
    assert s.weight_versions == []


# ===========================================================================
# 2. Old gym images and contract violations stop the run
# ===========================================================================


@pytest.mark.parametrize(
    "result",
    [
        _result([_step(spans=None)]),
        _result([_step(spans=None, weight_version=7)], weight_versions=None),
    ],
    ids=["trajectory_list", "per_step_key"],
)
def test_old_gym_without_spans_names_the_field_and_the_image(result):
    with pytest.raises(WeightVersionSpansError, match="weight_version_spans") as info:
        _convert(result)
    assert _MIN_GYM_IMAGE_FOR_WEIGHT_VERSION_SPANS in str(info.value)


def test_inspect_streaming_without_spans_names_both_gym_runtimes():
    """An Inspect streaming gym sends weight_versions and no spans: the error names its gym class too."""
    steps = [_inspect_step([101, 102], [201, 202, 401], spans=None)]
    with pytest.raises(WeightVersionSpansError, match="weight_version_spans") as info:
        _convert(_result(steps, weight_versions=[4, 5]), args=_make_args(arena_train_segments="all"))
    message = str(info.value)
    assert "gym g" in message
    assert "amzn_arena_streaming.sglang_provider.TrajectoryState" in message
    assert "amzn_arena_harbor.sglang_rollout.RolloutState" in message
    assert _MIN_GYM_IMAGE_FOR_WEIGHT_VERSION_SPANS in message


@pytest.mark.parametrize(
    "spans",
    [
        [{"version": 4, "start": 3, "end": 5}, {"version": 5, "start": 4, "end": 10}],
        [{"version": 4, "start": 3, "end": 11}],
        [{"version": 4, "start": 3, "end": 3}],
        [{"version": 4, "start": 5, "end": 3}],
        [{"version": True, "start": 3, "end": 5}],
        [{"version": "4", "start": 3, "end": 5}],
        [{"version": 4, "start": 3}],
        [[4, 3, 5]],
    ],
    ids=["overlap", "past_tokens", "empty", "reversed", "bool_version", "str_version", "missing_end", "not_a_dict"],
)
def test_contract_violation_stops(spans):
    with pytest.raises(WeightVersionSpansError, match="breaks the contract"):
        _convert(_result([_step(spans=spans)]))


def test_process_group_raises_instead_of_padding():
    worker = _make_worker(_make_args(n_samples_per_prompt=2))
    good = _result([_step()])["trajectories"][0]
    old = _result([_step(spans=None)])["trajectories"][0]
    result = {"task_id": "t.g0.", "gym_name": "g", "status": "success", "trajectories": [good, old]}
    with pytest.raises(WeightVersionSpansError):
        worker._process_group("t.g0.", [result])
    assert worker.output_queue.empty()


def test_generate_rollout_reraises_the_worker_fatal_error(monkeypatch):
    dead = _FakeWorker([], "final")
    dead.fatal_error = WeightVersionSpansError("old gym")
    monkeypatch.setattr(nats_rollout, "get_global_worker", lambda args, ds: dead)
    with pytest.raises(WeightVersionSpansError, match="old gym"):
        generate_rollout(_rollout_args("final"), 0, data_source=pytypes.SimpleNamespace())


# ===========================================================================
# 3. Consume-time staleness filter (upstream DefaultDataBuffer.get)
# ===========================================================================


def _group(version: int, group_index: int) -> list[Sample]:
    s = _sample(group_index=group_index, index=group_index, rollout_id=None, reward=1.0, response_length=2)
    s.weight_versions = _calls((str(version), 1, 3))
    return [s]


def _run(monkeypatch, *, max_weight_staleness: int | None, weight_version: int | None, gbs: int):
    """Queue a version-1 group, then a version-5 group, and drain ``gbs`` groups."""
    worker = _FakeWorker([_group(version=1, group_index=0), _group(version=5, group_index=1)], "final")
    # A live thread lets the drain poll the queue again after a drop.
    worker.worker_thread = pytypes.SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(nats_rollout, "get_global_worker", lambda args, ds: worker)
    args = _rollout_args("final", gbs=gbs, max_weight_staleness=max_weight_staleness)
    return generate_rollout(args, 0, data_source=pytypes.SimpleNamespace(), weight_version=weight_version)


def test_stale_group_is_dropped_and_counted(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        output = _run(monkeypatch, max_weight_staleness=2, weight_version=5, gbs=1)
    # Staleness 5 - 1 = 4 > 2: dropped. Staleness 0: kept.
    assert [g[0].group_index for g in output.samples] == [1]
    assert output.metrics == {
        "rollout/fully_async/stale_groups_filtered": 1,
        "rollout/fully_async/avg_staleness": 0.0,
        "rollout/fully_async/max_staleness": 0,
    }
    assert any(r.getMessage().startswith("Weight staleness: dropped=1 stale groups") for r in caplog.records)


def test_default_keeps_every_group(monkeypatch):
    output = _run(monkeypatch, max_weight_staleness=None, weight_version=5, gbs=2)
    assert [g[0].group_index for g in output.samples] == [0, 1]
    assert output.metrics == {
        "rollout/fully_async/stale_groups_filtered": 0,
        "rollout/fully_async/avg_staleness": 2.0,
        "rollout/fully_async/max_staleness": 4,
    }


def test_without_a_weight_version_no_staleness_is_known(monkeypatch):
    output = _run(monkeypatch, max_weight_staleness=None, weight_version=None, gbs=2)
    assert len(output.samples) == 2
    assert output.metrics == {"rollout/fully_async/stale_groups_filtered": 0}


def test_filter_without_a_weight_version_names_the_class(monkeypatch):
    with pytest.raises(ValueError, match="NatsRolloutFn"):
        _run(monkeypatch, max_weight_staleness=2, weight_version=None, gbs=1)


def test_nats_rollout_fn_passes_the_engine_weight_version(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(nats_rollout, "generate_rollout", lambda *a, **kw: calls.append({"args": a, **kw}))
    args, data_source = pytypes.SimpleNamespace(), pytypes.SimpleNamespace()
    fn = load_rollout_function(
        RolloutFnConstructorInput(args=args, data_source=data_source),
        "miles_plugins.arena.nats_arena.nats_rollout.NatsRolloutFn",
    )
    assert isinstance(fn, NatsRolloutFn)
    assert NatsRolloutFn.add_arguments is _add_arena_arguments
    fn(RolloutFnTrainInput(rollout_id=7, weight_version=5))
    assert calls == [{"args": (args, 7, data_source), "evaluation": False, "weight_version": 5}]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
