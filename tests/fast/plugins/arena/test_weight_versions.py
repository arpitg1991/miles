"""Gym ``weight_version_spans`` become upstream ``Sample.weight_versions`` (ADR-0018).

Each step that the training gym ships carries ``{"call", "version", "start",
"end"}`` entries. ``call`` is the 0-based index of the model call in the step.
A call that continues across a weight update gives one entry per weight
version, from the SGLang ``meta_info["weight_versions"]`` list.
``[start, end)`` is that part of the call output in the step ``token_ids``.
The trainer builds one ``WeightVersionsPerCall`` per call, as upstream
``WeightVersionsPerCall.from_meta_info`` does, so the upstream checks and
metrics read arena samples. The file also covers the consume-time staleness
filter of ``generate_rollout`` and ``NatsRolloutFn``.

Run:
    python -m pytest tests/fast/plugins/arena/test_weight_versions.py -v
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types as pytypes

import pytest
from tests.fast.plugins.arena.test_group_identity import _make_args, _make_worker, _StubMaskGenerator
from tests.fast.plugins.arena.test_multi_segment_episodes import _FakeWorker, _rollout_args, _sample

import miles.utils.mask_utils as mask_utils
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.filter_hub.common_filters import group_staleness
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
from miles_plugins.arena.rollout_metrics import compute_off_policy_metrics

_LOGGER = "miles_plugins.arena.nats_arena.nats_rollout"

# One step of a two-call episode, in the gym token index space:
# prompt 101-103 | call 1 output 201-202 | tool output 301-302 | call 2 output 401-403.
_TOKEN_IDS = [101, 102, 103, 201, 202, 301, 302, 401, 402, 403]
_LOSS_MASK = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]
_SPANS = [{"call": 0, "version": 4, "start": 3, "end": 5}, {"call": 1, "version": 5, "start": 7, "end": 10}]


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


# Call 0 continues across the weight update from 4 to 5, so SGLang gives its output two spans.
# Call 1 runs under 5. Only the first span of call 0 has the oldest version.
_SPLIT_SPANS = [
    {"call": 0, "version": 4, "start": 3, "end": 4},
    {"call": 0, "version": 5, "start": 4, "end": 5},
    {"call": 1, "version": 5, "start": 7, "end": 10},
]


def _upstream_split_sample() -> Sample:
    """The same two calls through upstream ``from_meta_info``, with the sglang-miles ``meta_info`` of each call."""
    meta_infos = [
        {
            "output_token_logprobs": [(-0.1, 201, None), (-0.1, 202, None)],
            "weight_versions": [{"version": "4", "start": 0, "end": 1}, {"version": "5", "start": 1, "end": 2}],
            "weight_version": "5",
        },
        {
            "output_token_logprobs": [(-0.1, 401, None), (-0.1, 402, None), (-0.1, 403, None)],
            "weight_versions": [{"version": "5", "start": 0, "end": 3}],
            "weight_version": "5",
        },
    ]
    return Sample(
        tokens=list(_TOKEN_IDS),
        response_length=7,
        reward=1.0,
        weight_versions=[
            WeightVersionsPerCall.from_meta_info(meta_infos[0], output_end=5),
            WeightVersionsPerCall.from_meta_info(meta_infos[1], output_end=10),
        ],
    )


def test_split_call_becomes_one_upstream_call_with_two_spans():
    (s,) = _convert(_result([_step(spans=_SPLIT_SPANS)]))
    assert s.weight_versions == _upstream_split_sample().weight_versions
    assert s.weight_versions[0].spans == [WeightVersionSpan("4", 3, 4), WeightVersionSpan("5", 4, 5)]
    # One entry per model call, as the dashboard turn count reads it.
    assert len(s.weight_versions) == 2
    s.validate()
    args = pytypes.SimpleNamespace(debug_rollout_only=False, debug_skip_weight_update=False)
    assert_samples_weight_version_sane(args, [s])


def test_split_call_staleness_and_metrics_read_its_oldest_version():
    """The newest-version label of the first span contract gave staleness 1 and min 5 here."""
    from miles.ray.rollout.metrics import _compute_metrics_from_samples

    (arena,) = _convert(_result([_step(spans=_SPLIT_SPANS)]))
    upstream = _upstream_split_sample()
    assert arena.oldest_weight_version == upstream.oldest_weight_version == 4
    assert group_staleness([arena], 6) == group_staleness([upstream], 6) == 2

    args = pytypes.SimpleNamespace(
        reward_key=None, advantage_estimator="grpo", sglang_speculative_algorithm=None, log_reward_category=None
    )

    def weight_version_keys(sample: Sample) -> dict:
        metrics = _compute_metrics_from_samples(args, [sample])
        return {k: v for k, v in metrics.items() if k.startswith("weight_version/")}

    assert weight_version_keys(arena) == weight_version_keys(upstream)
    assert weight_version_keys(arena)["weight_version/min"] == 4
    assert weight_version_keys(arena)["weight_version/mixed_version_ratio"] == 1.0


def test_hard_overflow_cuts_a_split_call_as_upstream_strip_does():
    (cut,) = _convert(_result([_step(spans=_SPLIT_SPANS)]), args=pytypes.SimpleNamespace(rollout_max_context_len=4))
    (full,) = _convert(_result([_step(spans=_SPLIT_SPANS)]))
    full.strip_last_output_tokens(6, tokenizer=pytypes.SimpleNamespace(decode=lambda ids: ""))
    expected = [WeightVersionsPerCall(spans=[WeightVersionSpan("4", 3, 4)]), WeightVersionsPerCall()]
    assert cut.weight_versions == full.weight_versions == expected


def test_a_call_without_an_entry_gives_no_upstream_call():
    """Call 1 has an empty output, so the gym sends no entry for it. The trainer adds no empty call."""
    spans = [{"call": 0, "version": 4, "start": 3, "end": 5}, {"call": 2, "version": 5, "start": 7, "end": 10}]
    (s,) = _convert(_result([_step(spans=spans)]))
    assert s.weight_versions == _calls(("4", 3, 5), ("5", 7, 10))


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
                [{"call": 0, "version": 4, "start": 2, "end": 4}, {"call": 1, "version": 5, "start": 5, "end": 7}],
            )
        ]
        steps[0]["loss_mask"][4] = 0
        outputs = [[201, 202], [401, 402]]
    else:
        # One self-contained step per call. The trainer trains the last step.
        steps = [
            _inspect_step([101, 102], [201, 202], [{"call": 0, "version": 4, "start": 2, "end": 4}]),
            _inspect_step([101, 102, 201, 202, 301], [401, 402], [{"call": 0, "version": 5, "start": 5, "end": 7}]),
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


def test_entry_without_a_call_index_stops_with_a_clear_error():
    """An entry of the first span contract has no ``call``: the trainer cannot group the spans of a call."""
    with pytest.raises(WeightVersionSpansError, match='has no "call" index') as info:
        _convert(_result([_step(spans=[{"version": 4, "start": 3, "end": 5}])]))
    assert "gym g" in str(info.value)
    assert "one entry per weight version" in str(info.value)


@pytest.mark.parametrize(
    "spans",
    [
        [{"call": 0, "version": 4, "start": 3, "end": 5}, {"call": 1, "version": 5, "start": 4, "end": 10}],
        [{"call": 0, "version": 4, "start": 3, "end": 11}],
        [{"call": 0, "version": 4, "start": 3, "end": 3}],
        [{"call": 0, "version": 4, "start": 5, "end": 3}],
        [{"call": 0, "version": True, "start": 3, "end": 5}],
        [{"call": 0, "version": "4", "start": 3, "end": 5}],
        [{"call": 0, "version": 4, "start": 3}],
        [[0, 4, 3, 5]],
        [{"call": 1, "version": 4, "start": 3, "end": 5}, {"call": 0, "version": 5, "start": 7, "end": 10}],
        [{"call": -1, "version": 4, "start": 3, "end": 5}],
        [{"call": True, "version": 4, "start": 3, "end": 5}],
        [{"call": "0", "version": 4, "start": 3, "end": 5}],
    ],
    ids=[
        "overlap",
        "past_tokens",
        "empty",
        "reversed",
        "bool_version",
        "str_version",
        "missing_end",
        "not_a_dict",
        "call_goes_back",
        "negative_call",
        "bool_call",
        "str_call",
    ],
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


@pytest.mark.parametrize(
    ("rollout_id", "interval", "engine_version", "train_version"),
    [
        (3, 1, 5, 5),  # the first rollout drains after the initial update
        (7, 1, 5, 6),  # an update runs between the drain and the train step
        (7, 2, 5, 5),  # no update between them: 7 is not a multiple of 2
        (8, 2, 5, 6),
        (7, 1, None, None),  # no published version yet
    ],
)
def test_nats_rollout_fn_passes_the_version_that_trains_the_batch(
    monkeypatch, rollout_id, interval, engine_version, train_version
):
    calls: list[dict] = []
    monkeypatch.setattr(nats_rollout, "generate_rollout", lambda *a, **kw: calls.append({"args": a, **kw}))
    args = pytypes.SimpleNamespace(start_rollout_id=3, update_weights_interval=interval)
    data_source = pytypes.SimpleNamespace()
    fn = load_rollout_function(
        RolloutFnConstructorInput(args=args, data_source=data_source),
        "miles_plugins.arena.nats_arena.nats_rollout.NatsRolloutFn",
    )
    assert isinstance(fn, NatsRolloutFn)
    assert NatsRolloutFn.add_arguments is _add_arena_arguments
    fn(RolloutFnTrainInput(rollout_id=rollout_id, weight_version=engine_version))
    assert calls == [{"args": (args, rollout_id, data_source), "evaluation": False, "weight_version": train_version}]


@pytest.mark.parametrize(("start_rollout_id", "interval"), [(0, 1), (0, 2), (3, 1), (3, 2)])
def test_train_async_arena_order_gives_the_version_that_trains_each_batch(monkeypatch, start_rollout_id, interval):
    """Run the real train_async_arena loop with fakes; record the version each batch gets and trains under.

    The driver starts the drain of rollout r before it trains rollout r - 1, and it updates the weights
    after that drain. The executor reads its version when the drain starts, as
    RolloutExecutor._get_rollout_data does.
    """
    # The driver imports the inference controller, which needs sglang.
    pytest.importorskip("sglang.srt.constants")
    import miles_plugins.arena.train_async_arena as driver

    passed: dict[int, int | None] = {}
    trained: dict[int, int] = {}

    def fake_generate_rollout(args, rollout_id, data_source, evaluation=False, weight_version=None):
        passed[rollout_id] = weight_version
        return rollout_id

    monkeypatch.setattr(nats_rollout, "generate_rollout", fake_generate_rollout)
    args = pytypes.SimpleNamespace(
        colocate=False,
        fully_async=False,
        start_rollout_id=start_rollout_id,
        num_rollout=start_rollout_id + 6,
        update_weights_interval=interval,
        check_weight_update_equal=False,
        eval_interval=None,
        skip_eval_before_train=True,
        prefetch_rollout_data=False,
        use_critic=False,
        save_interval=None,
        save_trigger_sentinel=None,
        debug_exit_after_rollout=None,
        use_wandb=False,
        load=None,
    )

    class Actor:
        version = 0  # the trainer weight version; the first update makes it 1

        async def train(self, rollout_id, rollout_data_ref):
            assert rollout_data_ref == rollout_id
            trained[rollout_id] = self.version

    class Executor:
        version = None  # the version that update_weights publishes
        fn = NatsRolloutFn(RolloutFnConstructorInput(args=args, data_source=None))

        async def get(self, rollout_id):
            batch_input = RolloutFnTrainInput(rollout_id=rollout_id, weight_version=self.version)
            await asyncio.sleep(0)  # the drain takes time; the driver runs on
            return self.fn(batch_input)

    class Controller:
        async def prepare_rollout(self, rollout_id):
            return None

    actor, executor = Actor(), Executor()

    async def update_weights(args, actor_model, rollout_executor, inference_controller, *, rollout_id=None):
        actor_model.version += 1
        rollout_executor.version = actor_model.version

    async def create_rollout_components(args):
        return Controller(), executor, None

    async def create_training_models(args, rollout_executor):
        return actor, None

    noop = lambda *a, **kw: None  # noqa: E731
    for name, value in {
        "validate_async_off_policy_correction": noop,
        "init_orchestration_script": noop,
        "maybe_start_api_server": noop,
        "maybe_start_mini_ft_controller": noop,
        "remove_rollout_data_refs": noop,
        "create_rollout_components": create_rollout_components,
        "create_training_models": create_training_models,
        "update_weights": update_weights,
        "EvalDispatcher": lambda *a: pytypes.SimpleNamespace(drain=noop),
    }.items():
        monkeypatch.setattr(driver, name, value)

    asyncio.run(driver.train(args, disposer=pytypes.SimpleNamespace(add=noop)))

    assert sorted(trained) == list(range(start_rollout_id, start_rollout_id + 6))
    assert passed == trained
    if start_rollout_id == 0 and interval == 1:
        # Then the upstream staleness and the arena off-policy round give the same number.
        for rollout_id, version in passed.items():
            group = _group(version=1, group_index=0)
            off_policy = compute_off_policy_metrics(
                pytypes.SimpleNamespace(update_weights_interval=1), [group], rollout_id
            )
            assert group_staleness(group, version) == off_policy["off_policy_round/mean"] == rollout_id


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
