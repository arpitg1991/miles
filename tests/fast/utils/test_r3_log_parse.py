"""The r3 log lines of the real call sites parse with the kdatp harness parser parse_r3_timing.py."""

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp
from miles.utils import data as data_utils
from miles.utils import object_store
from miles.utils.r3_log import log_r3_mem, log_replay_digest, r3_timing

PARSER = Path(__file__).parents[3] / "examples/arena/harbor-rl-glm53-flash/kdatp/prof/parse_r3_timing.py"
logger = logging.getLogger(__name__)


def _parser():
    spec = importlib.util.spec_from_file_location("parse_r3_timing", PARSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=True)
def _ray_cluster():
    if not ray.is_initialized():
        ray.init(address="local", num_cpus=2, include_dashboard=False, log_to_driver=False)
    yield


def test_r3_lines_parse(tmp_path, caplog):
    object_store.init_instance(make_args())
    args = make_args(rewards_normalization=False, balance_data=False, prefetch_rollout_data=True)
    samples = [make_sample(index=i) for i in range(2)]
    for sample in samples:
        sample.rollout_routed_experts = np.zeros((len(sample.tokens) - 1, 45, 8), dtype=np.int32)
    rollout_id = 41
    with caplog.at_level(logging.INFO):
        # RolloutManager.generate: load, convert, put.
        with r3_timing(logger, rank="rm", rollout=rollout_id, phase="load"):
            pass
        with r3_timing(logger, rank="rm", rollout=rollout_id, phase="convert") as line:
            data = convert_samples_to_train_data(
                args,
                samples,
                metadata={},
                custom_convert_samples_to_train_data_func=None,
                custom_reward_post_process_func=None,
            )
            line.update(bytes=sum(r.nbytes for r in data["rollout_routed_experts"]), dtype="int16")
        refs = split_train_data_by_dp(args, data, {"dp_size": 2}, rollout_id=rollout_id)
        # A train rank of DP 1: the prefetch during the previous step, then the fetch of train().
        data_utils.prefetch_rollout_data(args, refs, rollout_id=rollout_id, dp_rank=1)
        data_utils._fetch_rollout_data(args, refs[1], rollout_id=rollout_id, dp_rank=1)
        log_r3_mem(logger, rollout=rollout_id, at="start")
        buffers = [torch.zeros((4, 8), dtype=torch.int16)]
        # The keys of the fill line in MegatronTrainRayActor.train_actor.
        with r3_timing(logger, rank=0, rollout=rollout_id, phase="fill") as line:
            line.update(bytes=64, layers=1, buffers=1)
        log_replay_digest(logger, rollout=rollout_id, buffers=buffers)
        with r3_timing(logger, rank=0, rollout=rollout_id, phase="optimizer"):
            pass
        log_r3_mem(logger, rollout=rollout_id, at="end")
    log = tmp_path / "trainer-0.log"
    log.write_text("\n".join(record.getMessage() for record in caplog.records) + "\n")

    parser = _parser()
    view = parser._r3_view(parser.Arm("r3-prefetch", log), rollout_id)

    routing_bytes = sum((len(s.tokens) - 1) * 45 * 8 * 2 for s in samples)
    assert view["put_routing_bytes"] == routing_bytes
    assert view["put_bytes"] > routing_bytes
    assert view["fetch_put_ratio"] == [1.0]
    assert view["prefetch_modes"] == {"hit": 1}
    assert view["local_before"] == 1
    assert view["hit_ref_mismatch"] == 0
    assert view["prefetch_pull_s"]["n"] == view["prefetch_deser_s"]["n"] == 1
    assert view["fill_bytes_max"] == 64
    assert view["pre_exact"] > 0
    assert view["optimizer_s"]["n"] == 1
    assert len(view["digests"]) == 1
    assert set(next(iter(view["mem"].values()))) == {"start", "end"}
    for key in ("load_s", "convert_s", "prefetch_nodes", "fetch_nodes"):
        assert key in view, key
