"""The r3 log lines of the real call sites parse with the kdatp harness parser parse_r3_timing.py."""

import importlib.util
import logging
import re
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp
from miles.utils import data as data_utils
from miles.utils import object_store
from miles.utils.r3_log import log_r3_mem, log_replay_digest, r3_line, r3_timing

PROF = Path(__file__).parents[3] / "examples/arena/harbor-rl-glm53-flash/kdatp/prof"
PARSER = PROF / "parse_r3_timing.py"
logger = logging.getLogger(__name__)


def _parser():
    spec = importlib.util.spec_from_file_location("parse_r3_timing", PARSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _ray_cluster(ray_local_mode):
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
            # The lines carry 3 decimals, so a body shorter than 1 ms can log t0 == t1 (fill t1 == fetch t0).
            time.sleep(0.02)
            line.update(bytes=64, layers=1, buffers=1)
        log_replay_digest(logger, rollout=rollout_id, buffers=buffers)
        with r3_timing(logger, rank=0, rollout=rollout_id, phase="optimizer"):
            pass
        log_r3_mem(logger, rollout=rollout_id, at="end")
    log = tmp_path / "trainer-0.log"
    log.write_text("\n".join(record.getMessage() for record in caplog.records) + "\n")

    parser = _parser()
    arm = parser.Arm("r3-prefetch", log)
    view = parser._r3_view(arm, rollout_id)

    routing_bytes = sum((len(s.tokens) - 1) * 45 * 8 * 2 for s in samples)
    assert view["put_routing_bytes"] == routing_bytes
    assert view["put_bytes"] > routing_bytes
    # G1c windows: the prefetch pull and the fetch of train() on this node, with the size of the DP 1 object.
    (windows,) = view["wire_windows"].values()
    (put_dp1,) = [f["bytes"] for f in arm.r3_lines(rollout_id, "put") if f["dp"] == 1]
    assert windows["bytes"] == put_dp1
    assert len(windows["intervals"]) == 2
    assert view["prefetch_modes"] == {"hit": 1}
    assert view["local_before"] == 1
    assert view["hit_ref_mismatch"] == 0
    assert view["prefetch_pull_s"]["n"] == view["prefetch_deser_s"]["n"] == 1
    assert view["fill_bytes_max"] == 64
    assert view["pre_exact"] >= 0.01
    assert view["optimizer_s"]["n"] == 1
    assert len(view["digests"]) == 1
    assert set(next(iter(view["mem"].values()))) == {"start", "end"}
    for key in ("load_s", "convert_s", "prefetch_nodes", "fetch_nodes"):
        assert key in view, key


T = 1_790_000_000.0
OBJECT = 10**9
BACKGROUND = 1000  # eth0 bytes per second outside a transfer


def _sampler_columns() -> list[str]:
    """The header that kdatp-prof-run.sh writes, so that the test fails if the parser and the sampler differ."""
    (header,) = re.findall(r"printf '(epoch[^']*)\\n' > \"\$out\"", (PROF / "kdatp-prof-run.sh").read_text())
    return header.split("\\t")


def _moved(transfers: list, t: float) -> int:
    """Bytes of the transfers (start, end, bytes) that arrived by time ``t``, at an even rate."""
    return int(sum(b * min(max((t - s) / (e - s), 0.0), 1.0) for s, e, b in transfers))


def _write_sampler(path: Path, rx: list, tx: list) -> None:
    """One row per second from T, with background traffic plus the transfers."""
    columns = _sampler_columns()
    rows = ["\t".join(columns)]
    for i in range(21):
        t = T + i
        row = dict.fromkeys(columns, 0)
        row.update(epoch=t, net_rx_bytes=BACKGROUND * i + _moved(rx, t), net_tx_bytes=BACKGROUND * i + _moved(tx, t))
        rows.append("\t".join(str(row[c]) for c in columns))
    path.write_text("\n".join(rows) + "\n")


def _line(node: str, rank: int, phase: str, t0: float, t1: float, **extra) -> str:
    """A line of another pod: the keys of log_r3_timing, with ``node`` set."""
    return r3_line(
        "timing",
        rank=rank,
        rollout=41,
        node=node,
        phase=phase,
        bytes=OBJECT,
        seconds=t1 - t0,
        t0=T + t0,
        t1=T + t1,
        **extra,
    )


HEAD, REMOTE = "kdatp-prof-x-worker-0", "kdatp-prof-x-worker-1"
PULL = [(T + 5.3, T + 6.7, OBJECT)]
# r3: the head fetches its local object; the remote node pulls its object from the head.
R3_LINES = [_line(HEAD, 8, "fetch", 5.1, 5.2, dp=1), _line(REMOTE, 0, "fetch", 5.2, 6.8, dp=0)]
# r3-prefetch on the remote node: rank 16 prefetches during train(N-1) and hits; rank 17 misses, so its
# fetch shares the running pull, and the overlap counts once.
PREFETCH_LINES = [
    _line(REMOTE, 16, "prefetch_done", 2.2, 3.8, dp=0),
    _line(REMOTE, 17, "prefetch_done", 2.3, 3.9, dp=0),
    _line(REMOTE, 17, "fetch", 3.5, 3.95, dp=0, prefetch="miss"),
    _line(REMOTE, 16, "fetch", 10.1, 10.2, dp=0, prefetch="hit"),
]


@pytest.mark.parametrize(
    ("arm_name", "lines", "remote_rx", "verdict", "objects"),
    [
        ("r3", R3_LINES, PULL, "PASS", 1.0),
        # A second copy on the wire.
        ("r3", R3_LINES, [(T + 5.3, T + 6.7, 2 * OBJECT)], "FAIL", 2.0),
        ("r3-prefetch", PREFETCH_LINES, [(T + 2.3, T + 3.7, OBJECT)], "PASS", 1.0),
        # A sampler file without the eth0 columns (a job before this change): no value.
        ("r3", R3_LINES, None, "NA", None),
    ],
)
def test_g1c_reads_eth0_bytes_per_node(tmp_path, arm_name, lines, remote_rx, verdict, objects):
    parser = _parser()
    log = tmp_path / "trainer-0.log"
    log.write_text("\n".join(lines) + "\n")
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    _write_sampler(nodes / "node-0.tsv", rx=[], tx=PULL)
    _write_sampler(nodes / "node-1.tsv", rx=remote_rx or [], tx=[])
    if remote_rx is None:
        for path in nodes.iterdir():
            path.write_text("\n".join(line.rsplit("\t", 2)[0] for line in path.read_text().splitlines()) + "\n")
    row = parser._r3_view(parser.Arm(arm_name, log), 41)
    results = {arm_name: {"rc": 0, "out_of_memory": False, "window": [T, T + 20], "rollouts": {41: row}}}

    parser.add_node_windows(results, parser.read_nodes(nodes), "r3", "r3-prefetch")
    (gate,) = [g for g in parser.gates(results, [], "r3", "r3-prefetch", None, None) if g["id"].startswith("G1c")]

    assert gate["id"] == f"G1c[{arm_name}]"
    assert gate["verdict"] == verdict
    if objects is None:
        assert gate["value"]["remote"] is None
        return
    # The bracket samples add a few seconds of background traffic only.
    assert gate["value"]["remote"]["max"] == pytest.approx(objects, abs=1e-5)
    if arm_name == "r3":
        assert gate["value"]["head_max"] == pytest.approx(0, abs=1e-5)
        assert gate["value"]["head_tx_share"] == [pytest.approx(1.0, abs=1e-5)]
