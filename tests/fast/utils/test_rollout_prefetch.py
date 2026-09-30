"""--prefetch-rollout-data: the prefetch of the next shard and its hand-off to the fetch of train()."""

import logging
import threading
import time
from argparse import Namespace

import numpy as np
import pytest
import ray

from miles.ray.specs.train import TRAINER_CONCURRENCY_GROUPS
from miles.ray.train_actor import TrainRayActor
from miles.utils import data as data_utils
from miles.utils import object_store
from miles.utils.object_store import ObjectStoreGetResult, StoreObjectInfo, _RayStoreObjectRef
from miles.utils.workers.ray_worker_manager import _route_method_to_concurrency_group
from miles.utils.workers.rpc.common.metadata import declared_concurrency_groups, rpc


def _args(**overrides) -> Namespace:
    defaults = dict(
        delay_split_train_data_by_dp=False,
        prefetch_rollout_data=True,
        object_store_backend="ray",
        worker_comm_backend="ray",
        use_fault_tolerance=True,
    )
    return Namespace(**{**defaults, **overrides})


class _FakeStore:
    """Records the pulls and gets; a pull waits for ``release`` when ``block`` is set."""

    def __init__(self, *, block: bool = False, fail: bool = False) -> None:
        self.pulled: list[str] = []
        self.got: list[str] = []
        self.release = threading.Event()
        self.block, self.fail = block, fail

    def locate(self, ref) -> StoreObjectInfo:
        return StoreObjectInfo(key=ref, size=100, local=0)

    def pull(self, ref) -> None:
        self.pulled.append(ref)
        if self.fail:
            raise RuntimeError("object lost")
        if self.block:
            self.release.wait(timeout=30)

    def get(self, ref) -> ObjectStoreGetResult:
        self.got.append(ref)
        return ObjectStoreGetResult(value={"shard": ref}, release_fn=lambda _: None)


@pytest.fixture
def fake_store(monkeypatch):
    def install(**kwargs) -> _FakeStore:
        store = _FakeStore(**kwargs)
        monkeypatch.setattr(object_store, "_INSTANCE", store)
        return store

    return install


def test_take_returns_the_prefetched_shard(fake_store):
    store = fake_store()
    prefetcher = data_utils.RolloutDataPrefetcher()
    prefetcher.prefetch("ref-41", rollout_id=41, dp_rank=1)
    result, mode = prefetcher.take("ref-41")
    assert mode == "hit" and result.value == {"shard": "ref-41"}
    assert store.pulled == store.got == ["ref-41"]
    # The shard goes to one train() only.
    assert prefetcher.take("ref-41") == (None, "miss")


def test_take_of_another_object_misses_and_keeps_the_prefetch(fake_store):
    fake_store()
    prefetcher = data_utils.RolloutDataPrefetcher()
    prefetcher.prefetch("ref-42", rollout_id=42, dp_rank=0)
    assert prefetcher.take("ref-41") == (None, "miss")
    assert prefetcher.take("ref-42")[1] == "hit"


def test_take_does_not_wait_for_a_running_prefetch(fake_store):
    store = fake_store(block=True)
    prefetcher = data_utils.RolloutDataPrefetcher()
    thread = threading.Thread(target=prefetcher.prefetch, args=("ref-41",), kwargs=dict(rollout_id=41, dp_rank=0))
    thread.start()
    while not store.pulled:
        time.sleep(0.01)
    t0 = time.monotonic()
    assert prefetcher.take("ref-41") == (None, "miss")
    assert time.monotonic() - t0 < 1.0
    store.release.set()
    thread.join(timeout=30)


def test_prefetch_after_train_took_the_object_does_nothing(fake_store):
    store = fake_store()
    prefetcher = data_utils.RolloutDataPrefetcher()
    assert prefetcher.take("ref-41") == (None, "miss")
    prefetcher.prefetch("ref-41", rollout_id=41, dp_rank=0)
    assert store.pulled == []


def test_failed_prefetch_makes_train_fetch_itself(fake_store, caplog):
    fake_store(fail=True)
    prefetcher = data_utils.RolloutDataPrefetcher()
    with caplog.at_level(logging.WARNING):
        prefetcher.prefetch("ref-41", rollout_id=41, dp_rank=0)
    assert "Prefetch of rollout 41 failed" in caplog.text
    assert prefetcher.take("ref-41") == (None, "miss")


def test_the_train_actor_routes_the_prefetch_to_its_own_group():
    """A prefetch in the default group would wait for train(); Ray rejects a group the spec does not declare."""
    assert declared_concurrency_groups(TrainRayActor)["prefetch_rollout_data"] == "rollout_prefetch"
    assert TRAINER_CONCURRENCY_GROUPS["rollout_prefetch"] == 1


class _Trainer:
    """A train actor with the real prefetch and fetch path; ``train`` only holds the default group."""

    def __init__(self, dp_rank: int) -> None:
        object_store.init_instance(Namespace(object_store_backend="ray", worker_comm_backend="ray"))
        self.args = _args()
        self.dp_rank = dp_rank
        self.lines: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda record: self.lines.append(record.getMessage())
        logging.getLogger(data_utils.__name__).addHandler(handler)
        logging.getLogger(data_utils.__name__).setLevel(logging.INFO)

    @rpc(concurrency_group="rollout_prefetch")
    def prefetch_rollout_data(self, rollout_id: int, rollout_data_ref) -> None:
        data_utils.prefetch_rollout_data(self.args, rollout_data_ref, rollout_id=rollout_id, dp_rank=self.dp_rank)

    def train(self, seconds: float) -> float:
        time.sleep(seconds)
        return time.time()

    def fetch(self, rollout_id: int, rollout_data_ref) -> tuple[list, list[str]]:
        result = data_utils._fetch_rollout_data(
            self.args, rollout_data_ref[self.dp_rank], rollout_id=rollout_id, dp_rank=self.dp_rank
        )
        return result.value["tokens"], self.lines


@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(address="local", num_cpus=4, include_dashboard=False, log_to_driver=False)
    yield


def test_prefetch_runs_next_to_train_and_hands_over_the_right_shard(ray_cluster):
    # The same routing as _ServeActorRayCommManager._compute_actor_class with the fault-tolerance groups.
    routed = {
        name: _route_method_to_concurrency_group(getattr(_Trainer, name), group=group)
        for name, group in declared_concurrency_groups(_Trainer).items()
    }
    actor_cls = type("_Trainer", (_Trainer,), routed)
    trainer = ray.remote(concurrency_groups=TRAINER_CONCURRENCY_GROUPS)(actor_cls).remote(dp_rank=1)
    refs = [
        _RayStoreObjectRef(
            payload=ray.put({"tokens": [dp], "rollout_routed_experts": [np.full((4, 45, 8), dp, np.int16)]})
        )
        for dp in range(2)
    ]

    train = trainer.train.remote(5.0)
    time.sleep(0.5)
    prefetch = trainer.prefetch_rollout_data.remote(41, refs)
    ray.get(prefetch, timeout=30)
    prefetch_done = time.time()
    assert prefetch_done < ray.get(train), "the prefetch waited for train()"

    tokens, lines = ray.get(trainer.fetch.remote(41, refs))
    assert tokens == [1]
    key = refs[1].payload.hex()
    (done,) = [line for line in lines if "phase=prefetch_done" in line]
    (fetch,) = [line for line in lines if "phase=fetch" in line]
    assert f"ref={key}" in done and "dp=1" in done
    assert f"ref={key}" in fetch and "prefetch=hit" in fetch and "local_before=1" in fetch
    assert f"routing_bytes={4 * 45 * 8 * 2}" in fetch
