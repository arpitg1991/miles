"""Consume-time staleness cap of the NATS rollout path (``--max-weight-staleness``, ``NatsRolloutFn``).

Port of the arpit-ns-20261005 cap (9775f958, c830cff7, 19dee1af) to the
acuadron/dsa-qp line, where ``Sample.weight_versions`` is the per-call version
list of the gym (ADR-0019).
"""

from __future__ import annotations

import asyncio
import logging
import types as pytypes

import pytest
from tests.fast.plugins.arena.test_multi_segment_episodes import _FakeWorker, _rollout_args, _sample

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.compatibility import load_rollout_function
from miles_plugins.arena.nats_arena import nats_rollout
from miles_plugins.arena.nats_arena.nats_rollout import (
    NatsRolloutFn,
    _add_arena_arguments,
    _group_staleness,
    _staleness_metrics,
    _train_weight_version,
    generate_rollout,
)

_LOGGER = "miles_plugins.arena.nats_arena.nats_rollout"
_FN_PATH = "miles_plugins.arena.nats_arena.nats_rollout.NatsRolloutFn"


def _group(versions: list[int], group_index: int, reward: float = 1.0) -> list:
    s = _sample(group_index=group_index, index=group_index, rollout_id=None, reward=reward, response_length=2)
    s.weight_versions = [str(v) for v in versions]
    return [s]


class _LiveWorker(_FakeWorker):
    """A worker whose thread is alive, so the drain polls the queue again after a drop."""

    def __init__(self, groups, mode="final"):
        super().__init__(groups, mode)
        self.worker_thread = pytypes.SimpleNamespace(is_alive=lambda: True)
        self.fatal_error = None

    def pop_dropped_group_metrics(self) -> dict:
        return {}


def _run(monkeypatch, groups, *, max_weight_staleness, weight_version, gbs, **overrides):
    worker = _LiveWorker(groups)
    monkeypatch.setattr(nats_rollout, "get_global_worker", lambda args, ds: worker)
    args = _rollout_args("final", gbs=gbs, max_weight_staleness=max_weight_staleness, **overrides)
    return generate_rollout(args, 0, data_source=pytypes.SimpleNamespace(), weight_version=weight_version), worker


# ---------------------------------------------------------------------------
# the drop
# ---------------------------------------------------------------------------


def test_stale_group_is_dropped_and_the_next_fresh_group_fills_the_batch(monkeypatch, caplog):
    groups = [_group([1, 3], 0), _group([5], 1)]
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        data, worker = _run(monkeypatch, groups, max_weight_staleness=2, weight_version=5, gbs=1)
    # Group 0: oldest version 1, staleness 4 > 2, dropped. Group 1: staleness 0, kept.
    assert [g[0].group_index for g in data] == [1]
    assert worker.output_queue.qsize() == 0
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Weight staleness:"))
    assert line == (
        "Weight staleness: dropped=1 stale groups (max_weight_staleness=2), kept=1 groups, "
        "avg=0.0 max=0 at weight_version=5"
    )


def test_the_cap_is_inclusive(monkeypatch):
    # Staleness exactly at the cap stays, as upstream (staleness > max drops).
    data, _ = _run(monkeypatch, [_group([3], 0)], max_weight_staleness=2, weight_version=5, gbs=1)
    assert [g[0].group_index for g in data] == [0]


def test_the_oldest_call_of_any_sample_decides(monkeypatch):
    # One sample of the group has a version-1 call: the whole group is stale.
    stale = _group([5, 5], 0) + _group([4, 1, 5], 0)
    data, _ = _run(monkeypatch, [stale, _group([5], 1)], max_weight_staleness=3, weight_version=5, gbs=1)
    assert [g[0].group_index for g in data] == [1]


def test_untagged_samples_have_no_staleness_and_stay(monkeypatch):
    untagged = _group([], 0)
    data, _ = _run(monkeypatch, [untagged], max_weight_staleness=0, weight_version=9, gbs=1)
    assert len(data) == 1


def test_default_keeps_every_group(monkeypatch):
    data, _ = _run(monkeypatch, [_group([1], 0), _group([5], 1)], max_weight_staleness=None, weight_version=5, gbs=2)
    assert [g[0].group_index for g in data] == [0, 1]


def test_the_cap_without_a_weight_version_names_the_class(monkeypatch):
    with pytest.raises(ValueError, match="NatsRolloutFn"):
        _run(monkeypatch, [_group([1], 0)], max_weight_staleness=2, weight_version=None, gbs=1)


def test_a_stale_group_under_r3_reaps_its_routing_and_never_decodes(monkeypatch):
    reaped: list[list] = []
    decoded: list[list] = []
    monkeypatch.setattr(nats_rollout, "reap_sample_refs", lambda group: reaped.append(list(group)))
    monkeypatch.setattr(nats_rollout, "materialize_group_routing", lambda group, args: decoded.append(group))
    stale, fresh = _group([1], 0), _group([5], 1)
    data, _ = _run(
        monkeypatch, [stale, fresh], max_weight_staleness=2, weight_version=5, gbs=1, use_rollout_routing_replay=True
    )
    assert [g[0].group_index for g in data] == [1]
    assert reaped == [stale]
    assert decoded == [fresh]


def test_metrics_reach_the_wandb_dict(monkeypatch):
    logged: list[dict] = []
    monkeypatch.setattr(
        "miles.utils.tracking_utils.tracking.log", lambda args, metrics, step_key: logged.append(metrics)
    )
    stale_a, stale_b = _group([1], 0, reward=0.25), _group([2], 1, reward=0.75)
    _run(
        monkeypatch,
        [stale_a, stale_b, _group([4], 2), _group([5], 3)],
        max_weight_staleness=2,
        weight_version=5,
        gbs=2,
        use_wandb=True,
        wandb_always_use_train_step=False,
    )
    metrics = logged[0]
    assert metrics["rollout/fully_async/stale_groups_filtered"] == 2
    assert metrics["rollout/num_old_age_dropped"] == 2
    assert metrics["rollout/reward_old_age_dropped"] == 0.5
    assert metrics["rollout/fully_async/avg_staleness"] == 0.5
    assert metrics["rollout/fully_async/max_staleness"] == 1


def test_staleness_metrics_have_no_reward_key_without_a_drop():
    assert _staleness_metrics(0, [], [2, 4]) == {
        "rollout/fully_async/stale_groups_filtered": 0,
        "rollout/num_old_age_dropped": 0,
        "rollout/fully_async/avg_staleness": 3.0,
        "rollout/fully_async/max_staleness": 4,
    }
    assert _staleness_metrics(0, [], []) == {
        "rollout/fully_async/stale_groups_filtered": 0,
        "rollout/num_old_age_dropped": 0,
    }


def test_group_staleness_matches_the_upstream_buffer_rule():
    from miles.rollout.fully_async_data_buffer import DefaultDataBuffer

    group = _group([7, 4], 0) + _group([6], 0)
    assert _group_staleness(group, 9) == DefaultDataBuffer._staleness(group, 9) == 5
    assert _group_staleness(group, None) is None


# ---------------------------------------------------------------------------
# the version that trains the batch
# ---------------------------------------------------------------------------


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
def test_train_weight_version(rollout_id, interval, engine_version, train_version):
    args = pytypes.SimpleNamespace(update_weights_interval=interval)
    assert _train_weight_version(args, rollout_id, engine_version, start_rollout_id=3) == train_version


def test_nats_rollout_fn_loads_through_the_class_seam_and_passes_the_train_version(monkeypatch):
    calls: list[tuple] = []

    def fake(args, rollout_id, data_source, evaluation=False, weight_version=None):
        calls.append((rollout_id, weight_version))
        return [["batch", rollout_id]]

    monkeypatch.setattr(nats_rollout, "generate_rollout", fake)
    args = pytypes.SimpleNamespace(start_rollout_id=None, update_weights_interval=1)
    fn = load_rollout_function(RolloutFnConstructorInput(args=args, data_source=None), _FN_PATH)
    assert isinstance(fn, NatsRolloutFn)
    assert NatsRolloutFn.add_arguments is _add_arena_arguments
    out = fn(RolloutFnTrainInput(rollout_id=20, weight_version=1))  # a resume: the first call is the start
    assert isinstance(out, RolloutFnTrainOutput) and out.samples == [["batch", 20]] and out.metrics is None
    fn(RolloutFnTrainInput(rollout_id=21, weight_version=1))
    assert calls == [(20, 1), (21, 2)]


def test_nats_rollout_fn_prefers_a_start_rollout_id_from_the_command_line(monkeypatch):
    calls: list[int | None] = []
    monkeypatch.setattr(nats_rollout, "generate_rollout", lambda *a, **kw: calls.append(kw["weight_version"]) or [])
    args = pytypes.SimpleNamespace(start_rollout_id=4, update_weights_interval=1)
    fn = NatsRolloutFn(RolloutFnConstructorInput(args=args, data_source=None))
    fn(RolloutFnTrainInput(rollout_id=5, weight_version=2))  # 5 > start 4: an update runs before it trains
    assert calls == [3]


@pytest.mark.parametrize(("start_rollout_id", "interval"), [(0, 1), (0, 2), (3, 1), (3, 2)])
def test_train_async_arena_order_gives_the_version_that_trains_each_batch(monkeypatch, start_rollout_id, interval):
    """Run the real train_async_arena loop with fakes; each batch must get the version that it trains under.

    The fake RolloutManager reads its weight version when a drain starts, as RolloutManager._get_rollout_data
    does, and the fake actor publishes each update to it, as MegatronTrainRayActor.update_weights does.
    """
    # The driver imports the Ray placement, sglang and megatron; on a host without libcuda that import fails.
    pytest.importorskip("ray")
    pytest.importorskip("sglang")
    try:
        import miles_plugins.arena.train_async_arena as driver
    except OSError as exc:  # libcuda.so.1 missing: a CPU-only host; the node job runs this test
        pytest.skip(f"train_async_arena needs libcuda: {exc}")

    passed: dict[int, int | None] = {}
    trained: dict[int, int] = {}

    def fake_generate_rollout(args, rollout_id, data_source, evaluation=False, weight_version=None):
        passed[rollout_id] = weight_version
        return rollout_id

    monkeypatch.setattr(nats_rollout, "generate_rollout", fake_generate_rollout)
    args = pytypes.SimpleNamespace(
        colocate=False,
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
        control_server_port=None,
    )
    # The RolloutManager's own copy of args: start_rollout_id is None there.
    manager_args = pytypes.SimpleNamespace(**{**vars(args), "start_rollout_id": None})

    class _Remote:
        def __init__(self, fn):
            self.remote = fn

    class Manager:
        weight_version = None
        fn = NatsRolloutFn(RolloutFnConstructorInput(args=manager_args, data_source=None))

        def __init__(self):
            self.generate = _Remote(self._generate)
            self.dispose = _Remote(self._noop)

        def _generate(self, rollout_id):
            async def drain():
                version = self.weight_version  # read when the drain starts
                await asyncio.sleep(0)  # the drain takes time; the driver runs on
                return self.fn(RolloutFnTrainInput(rollout_id=rollout_id, weight_version=version)).samples

            return asyncio.ensure_future(drain())

        def _noop(self, *a, **kw):
            async def done():
                return None

            return asyncio.ensure_future(done())

    manager = Manager()

    class Actor:
        version = 0  # the first update makes it 1

        async def train(self, rollout_id, rollout_data_ref):
            assert rollout_data_ref == rollout_id
            trained[rollout_id] = self.version

        async def update_weights(self, rollout_id=None):
            self.version += 1
            manager.weight_version = self.version

    class Dispatcher:
        def __init__(self, *a, **kw):
            pass

        async def dispatch(self, *a, **kw):
            pass

        async def drain(self):
            pass

    async def create_training_models(args, pgs, rollout_manager):
        return Actor(), None

    for name, value in {
        "validate_async_off_policy_correction": lambda args: None,
        "configure_logger": lambda *a, **kw: None,
        "maybe_start_periodic_pyspy_dump": lambda: None,
        "_load_extra_state": lambda args: None,
        "create_placement_groups": lambda args: {"rollout": None},
        "object_store": pytypes.SimpleNamespace(init_instance=lambda *a, **kw: None),
        "init_tracking": lambda args: None,
        "create_rollout_manager": lambda args, pg: (manager, 6),
        "create_training_models": create_training_models,
        "maybe_start_mini_ft_controller": lambda args: None,
        "EvalDispatcher": Dispatcher,
        "remove_rollout_data_refs": lambda args, ref: None,
        "_maybe_drain_eval_metrics": lambda *a, **kw: None,
    }.items():
        monkeypatch.setattr(driver, name, value)

    asyncio.run(driver.train(args))

    assert sorted(trained) == list(range(start_rollout_id, start_rollout_id + 6))
    assert passed == trained
