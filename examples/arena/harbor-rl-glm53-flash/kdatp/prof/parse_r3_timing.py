"""Pre-train data-path table of kdatp arms: fetch, fill, prefetch, bytes, host memory and numerics.

The measurement plan of job ``kdatp-prof-20260930r`` defines every column and
gate (r3-run/reports/plan.md). For each arm, the parser reads
``<arm>/trainer-0.log`` (and ``<arm>/rc``) and prints one row per rollout.
It reads two kinds of lines:

- Lines that the control image also logs (job ``20260929d``, miles
  ``4716a367a``): the per-rank ``fn=train`` and ``fn=train_actor`` start
  lines, the rank-0 ``Timer`` lines, the ``perf N`` and ``step N`` dicts,
  ``[peak-memory]`` and the RolloutManager ``Rollout-side DP schedule`` line.
  Every arm gets the same columns from them, so the control compares like
  for like.
- Lines that only the r3 image logs, one ``key=value`` list each:
  ``[r3-timing]`` (``phase`` = ``load``, ``convert``, ``put``, ``fetch``,
  ``fill``, ``prefetch_start``, ``prefetch_done`` or ``optimizer``),
  ``[r3-mem]`` and ``[r3-digest]``.

With ``--nodes``, it also reads the harness sampler files
``<run>/nodes/node-<K>.tsv`` (one row per second per pod; the header row
names the columns; node-0 is the Ray head). The pod with hostname
``<job>-worker-<K>`` writes ``node-<K>.tsv``. The ``eth0`` byte counters of
these files are the wire measure of gate G1c.

Columns from the first kind (seconds from ``t0``, the first ``fn=train``
start of the rollout over all ranks):

- ``first``/``last``: the first and the last node whose 8 ranks all reached
  ``fn=train_actor``, that is, finished ``get_rollout_data``.
- ``head``: the same for the Ray head node (its lines carry no ``ip=``).
- ``serial``: 1 - first/last over the 7 remote nodes. 0 means that all
  remote nodes finish together; 0.86 means one after another at an even rate.
- ``s/node``: the slope of the sorted remote-node times.
- ``fill0``: rank 0 ``Timer train start`` minus ``Timer data_preprocess end``
  (``get_data_iterator`` plus ``fill_replay_data``).
- ``pre``: ``last`` + ``fill0``, the critical path before training.
- ``gen``: the ``Rollout-side DP schedule`` line of ``generate(N)`` minus
  ``t0`` of rollout N-1 (``generate(N)`` starts right before ``train(N-1)``).

Rollout ``start`` (the first one) compiles kernels, so "warm" means the other
rollouts.

    python3 parse_r3_timing.py base=<run>/base r3=<run>/r3 r3-prefetch=<run>/r3-prefetch \\
        --control base --control base-a --manifest <data>/t2/manifest.json [--nodes <run>/nodes] [--json out.json]
"""

import argparse
import ast
import bisect
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PREFIX = re.compile(
    r"\((?P<actor>\w+) pid=\d+(?:, ip=(?P<ip>[\d.]+))?\)\x1b\[0m "
    r"\[(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) (?P<who>[^\]]+)\] (?P<msg>.*)$"
)
RANK = re.compile(r"actor_cell\d+_rank(\d+)$")
WITH_LOGS = re.compile(r"ft cls=MegatronTrainRayActor fn=(train|train_actor) phase=start")
TRAIN_STEP = re.compile(r"ft op=train_step rollout=(\d+) ")
TIMER = re.compile(r"Timer (\w+) (start|end)")
PEAK = re.compile(r"\[peak-memory\] rollout=(\d+) rank=(\d+) pp=(\d+) tp=(\d+) max_allocated_gib=([\d.]+)")
METRICS = re.compile(r"\b(perf|step) (\d+): (\{.*\})\s*$")
OOM = re.compile(r"OutOfMemoryError|CUDA out of memory")
DP_SCHEDULE = "Rollout-side DP schedule"
R3 = re.compile(r"\[r3-(timing|mem|digest)\] (.*)$")
HEAD = "head"
HEAD_SAMPLER = "node-0"
NUM_RANKS = 64
# Step-0 forward metrics: bit-identical in the two base runs 20260929a and 20260929d (same seed and rows).
FORWARD_KEYS = (
    "train/loss",
    "train/pg_loss",
    "train/tis",
    "train/tis_abs",
    "train/tis_clipfrac",
    "train/train_rollout_kl",
    "train/train_rollout_logprob_abs_diff",
)
ROUTING_BYTES_PER_TOKEN_INT32 = 45 * 8 * 4
# G1c bounds, in objects per remote node: TCP/IP headers and background traffic add a few percent;
# a second copy of the shard reads 2.0.
WIRE_OBJECTS = (0.95, 1.25)
CLK_TCK = 100


def _epoch(ts: str) -> float:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()


def _num(value: str):
    for kind in (int, float):
        try:
            return kind(value)
        except ValueError:
            pass
    return value


def _kv(text: str) -> dict:
    return {key: _num(value) for key, sep, value in (t.partition("=") for t in text.split()) if sep}


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values), "max": max(values), "min": min(values)}


def _median(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    xs = range(len(values))
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(values)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values, strict=True)) / sum(
        (x - mean_x) ** 2 for x in xs
    )


class Arm:
    """All lines of one ``trainer-0.log``, grouped by kind."""

    def __init__(self, name: str, path: Path):
        self.name = name
        self.log = path / "trainer-0.log" if path.is_dir() else path
        rc_file = self.log.parent / "rc"
        self.rc = int(rc_file.read_text()) if rc_file.is_file() else None
        self.out_of_memory = False
        self.rank_node: dict[int, str] = {}
        self.train_start: dict[int, list[float]] = defaultdict(list)
        self.actor_start: dict[int, list[float]] = defaultdict(list)
        self.train_step_ids: set[int] = set()
        self.timer0: list[tuple[str, str, float]] = []
        self.metrics: dict[str, dict[int, dict]] = {"perf": {}, "step": {}}
        self.peak: dict[int, dict[int, float]] = defaultdict(dict)
        self.dp_schedule: list[float] = []
        self.r3: list[dict] = []
        self.first_ts = self.last_ts = None
        for line in self.log.read_text(errors="replace").splitlines():
            self._line(line)

    def _line(self, line: str) -> None:
        if OOM.search(line):
            self.out_of_memory = True
        # r3 lines carry their own times, so they parse with or without the Ray prefix (driver, tests).
        if r3 := R3.search(line):
            fields = _kv(r3[2])
            fields["kind"] = r3[1]
            self.r3.append(fields)
            return
        match = PREFIX.search(line)
        if match is None:
            return
        ts = _epoch(match["ts"])
        self.first_ts = ts if self.first_ts is None else self.first_ts
        self.last_ts = ts
        msg = match["msg"]
        rank_match = RANK.search(match["who"])
        if rank_match is None:
            if DP_SCHEDULE in msg:
                self.dp_schedule.append(ts)
            return
        rank = int(rank_match[1])
        self.rank_node[rank] = match["ip"] or HEAD
        if m := WITH_LOGS.search(msg):
            (self.train_start if m[1] == "train" else self.actor_start)[rank].append(ts)
        elif m := TRAIN_STEP.search(msg):
            self.train_step_ids.add(int(m[1]))
        elif rank == 0 and (m := TIMER.search(msg)):
            self.timer0.append((m[1], m[2], ts))
        elif m := PEAK.search(msg):
            slot = self.peak[int(m[1])]
            slot[int(m[3])] = max(slot.get(int(m[3]), 0.0), float(m[5]))
        elif m := METRICS.search(msg):
            by_id = self.metrics[m[1]]
            if int(m[2]) not in by_id:  # the step dict is logged twice (log_utils.py and model.py)
                try:
                    by_id[int(m[2])] = ast.literal_eval(m[3])
                except (ValueError, SyntaxError):
                    pass

    def rollout_ids(self) -> list[int]:
        start = min(self.train_step_ids | set(self.metrics["perf"]) or {0})
        count = max((len(v) for v in self.train_start.values()), default=0)
        return list(range(start, start + count))

    def r3_lines(self, rollout_id: int, phase: str | None = None, kind: str = "timing") -> list[dict]:
        return [
            f
            for f in self.r3
            if f["kind"] == kind and f.get("rollout") == rollout_id and (phase is None or f.get("phase") == phase)
        ]


def _fill0(arm: Arm) -> list[float]:
    """Rank 0: seconds from each ``data_preprocess end`` to the next ``train start``."""
    gaps, pending = [], None
    for name, event, ts in arm.timer0:
        if name == "data_preprocess" and event == "end":
            pending = ts
        elif name == "train" and event == "start" and pending is not None:
            gaps.append(ts - pending)
            pending = None
    return gaps


def _node_view(rank_node: dict, starts: dict, ends: dict, head: str = HEAD) -> dict | None:
    """Per node, the time when the last of its ranks reached ``ends``, from the first of ``starts`` over all ranks."""
    ranks = [r for r in starts if r in ends and r in rank_node]
    if not ranks:
        return None
    t0 = min(starts[r] for r in ranks)
    per_node: dict[str, float] = defaultdict(float)
    for r in ranks:
        per_node[rank_node[r]] = max(per_node[rank_node[r]], ends[r] - t0)
    remote = sorted(v for k, v in per_node.items() if k != head)
    return {
        "t0": t0,
        "ranks": len(ranks),
        "first": min(per_node.values()),
        "last": max(per_node.values()),
        "head": per_node.get(head),
        "serial": (1 - remote[0] / remote[-1]) if len(remote) > 1 and remote[-1] > 0 else None,
        "per_node_slope": _slope(remote),
        "staircase": sorted(per_node.items(), key=lambda kv: kv[1]),
    }


def _nth(by_rank: dict[int, list[float]], index: int) -> dict[int, float]:
    return {r: v[index] for r, v in by_rank.items() if len(v) > index}


def _r3_view(arm: Arm, rollout_id: int) -> dict:
    """Columns from the r3-only lines of one rollout (empty for the control)."""
    out: dict = {}
    puts = arm.r3_lines(rollout_id, "put")
    if puts:
        out["put_bytes"] = sum(f["bytes"] for f in puts)
        out["put_routing_bytes"] = sum(f.get("routing_bytes", 0) for f in puts)
        out["put_s"] = _stats([f["seconds"] for f in puts])
        out["put_t1"] = max(f["t1"] for f in puts)
    for phase in ("load", "convert"):
        if lines := arm.r3_lines(rollout_id, phase):
            out[f"{phase}_s"] = lines[0]["seconds"]
    fetch = arm.r3_lines(rollout_id, "fetch")
    if fetch:
        out["fetch_s"] = _stats([f["seconds"] for f in fetch])
        out["fetch_node_bytes"] = {f["node"]: f["bytes"] for f in fetch}
        # G1c: per node, the object size and the windows in which its ranks can move the shard, that is,
        # the prefetch pulls and the fetches of train(). The fetch bytes are the owner's object_size,
        # the same number as the put bytes, so only the eth0 counters of the node sampler measure the wire.
        windows: dict = {}
        for f in [*fetch, *arm.r3_lines(rollout_id, "prefetch_done")]:
            slot = windows.setdefault(f["node"], {"bytes": -1, "intervals": []})
            slot["intervals"].append((f["t0"], f["t1"]))
            slot["bytes"] = max(slot["bytes"], f["bytes"])
        out["wire_windows"] = windows
        out["prefetch_modes"] = dict(sorted(_count(str(f.get("prefetch")) for f in fetch).items()))
        out["local_before"] = sum(1 for f in fetch if f.get("local_before") == 1)
        done = {f["rank"]: f.get("ref") for f in arm.r3_lines(rollout_id, "prefetch_done")}
        out["hit_ref_mismatch"] = sum(
            1 for f in fetch if f.get("prefetch") == "hit" and done.get(f["rank"]) != f.get("ref")
        )
        # The exact staircase: the head node is the one whose hostname ends in -0.
        head = next((f["node"] for f in fetch if str(f["node"]).endswith("-0")), HEAD)
        rank_node = {f["rank"]: f["node"] for f in fetch}
        out["fetch_nodes"] = _node_view(
            rank_node, {f["rank"]: f["t0"] for f in fetch}, {f["rank"]: f["t1"] for f in fetch}, head
        )
    if fill := arm.r3_lines(rollout_id, "fill"):
        out["fill_s"] = _stats([f["seconds"] for f in fill])
        out["fill_bytes_max"] = max(f["bytes"] for f in fill)
        if fetch:
            out["pre_exact"] = max(f["t1"] for f in fill) - min(f["t0"] for f in fetch)
    if lines := arm.r3_lines(rollout_id, "optimizer"):
        out["optimizer_s"] = _stats([f["seconds"] for f in lines])
        out["optimizer_window"] = (min(f["t0"] for f in lines), max(f["t1"] for f in lines))
    if done_lines := arm.r3_lines(rollout_id, "prefetch_done"):
        out["prefetch_done_s"] = _stats([f["seconds"] for f in done_lines])
        # deser_s holds the actor GIL while the training thread runs, so it gets its own column.
        for key in ("pull_s", "deser_s"):
            out[f"prefetch_{key}"] = _stats([f[key] for f in done_lines if f.get(key, -1) >= 0])
        starts = {f["rank"]: f["t0"] for f in arm.r3_lines(rollout_id, "prefetch_start")}
        head = next((f["node"] for f in done_lines if str(f["node"]).endswith("-0")), HEAD)
        rank_node = {f["rank"]: f["node"] for f in done_lines}
        out["prefetch_nodes"] = _node_view(rank_node, starts, {f["rank"]: f["t1"] for f in done_lines}, head)
    if digests := arm.r3_lines(rollout_id, kind="digest"):
        out["digest_s"] = _stats([f["seconds"] for f in digests])
        out["digests"] = {f["rank"]: f["sha256"] for f in digests}
    if mem := arm.r3_lines(rollout_id, kind="mem"):
        per_node: dict = defaultdict(dict)
        for f in mem:
            slot = per_node[f["node"]].setdefault(f["at"], {"rss_gib_max": 0.0, "avail_gib_min": float("inf")})
            slot["rss_gib_max"] = max(slot["rss_gib_max"], f["rss_gib"])
            slot["avail_gib_min"] = min(slot["avail_gib_min"], f["avail_gib"])
            slot["shmem_gib"] = f["shmem_gib"]
            slot["cgroup_gib"] = max(slot.get("cgroup_gib", 0.0), f["cgroup_gib"])
        out["mem"] = dict(per_node)
    return out


def _count(items) -> dict:
    counts: dict = defaultdict(int)
    for item in items:
        counts[item] += 1
    return counts


def parse_arm(arm: Arm) -> dict:
    fill0 = _fill0(arm)
    rows = {}
    for index, rollout_id in enumerate(arm.rollout_ids()):
        view = _node_view(arm.rank_node, _nth(arm.train_start, index), _nth(arm.actor_start, index)) or {}
        perf = arm.metrics["perf"].get(rollout_id, {})
        step = arm.metrics["step"].get(rollout_id, {})
        row = {
            k: view.get(k) for k in ("t0", "ranks", "first", "last", "head", "serial", "per_node_slope", "staircase")
        }
        row["fill0"] = fill0[index] if index < len(fill0) else None
        row["pre"] = row["last"] + row["fill0"] if row["last"] is not None and row["fill0"] is not None else None
        # generate(N) starts right before train(N-1); its DP schedule line comes after convert.
        previous = rows.get(rollout_id - 1, {}).get("t0")
        row["gen"] = arm.dp_schedule[index] - previous if previous and index < len(arm.dp_schedule) else None
        row["dp48"] = perf.get("perf/data_preprocess_time")
        row["actor_train"] = perf.get("perf/actor_train_time")
        row["step_time"] = perf.get("perf/step_time")
        row["grad_norm"] = step.get("train/grad_norm")
        row["forward"] = {k: step[k] for k in FORWARD_KEYS if k in step}
        row["peak_gib"] = arm.peak.get(rollout_id)
        row.update(_r3_view(arm, rollout_id))
        rows[rollout_id] = row
    return {
        "log": str(arm.log),
        "rc": arm.rc,
        "out_of_memory": arm.out_of_memory,
        "window": [arm.first_ts, arm.last_ts],
        "rollouts": rows,
    }


def read_nodes(nodes_dir: Path) -> dict[str, list[dict]]:
    """``node-<K>.tsv`` files of the harness sampler: a header row, then one row per sample."""
    out = {}
    for path in sorted(nodes_dir.glob("node-*.tsv")):
        lines = path.read_text(errors="replace").splitlines()
        if not lines:
            continue
        header = lines[0].split("\t")
        rows = [
            {k: _num(v) for k, v in zip(header, values, strict=True)}
            for values in (line.split("\t") for line in lines[1:])
            if len(values) == len(header)
        ]
        out[path.stem] = rows
    return out


def window_stats(samples: list[dict], t0: float, t1: float) -> dict | None:
    """CPU busy share, raylet CPU seconds, memory extremes and Ray counters of one node between t0 and t1."""
    inside = [s for s in samples if t0 <= s["epoch"] <= t1]
    if len(inside) < 2:
        return None
    first, last = inside[0], inside[-1]
    total = last["cpu_total_ticks"] - first["cpu_total_ticks"]
    return {
        "seconds": last["epoch"] - first["epoch"],
        "cpu_busy": (last["cpu_busy_ticks"] - first["cpu_busy_ticks"]) / total if total else None,
        "raylet_cpu_s": (last["raylet_ticks"] - first["raylet_ticks"]) / CLK_TCK,
        "mem_avail_gib_min": min(s["mem_avail_kb"] for s in inside) / 2**20,
        "shmem_gib_max": max(s["shmem_kb"] for s in inside) / 2**20,
        "cgroup_gib_max": max(s["cgroup_bytes"] for s in inside) / 2**30,
        "pushes_remaining_max": max(s["pushes_remaining"] for s in inside),
        "pulls_active_max": max(s["objects_actively_pulled"] for s in inside),
        "pull_bytes_available_min": min(s["pull_bytes_available"] for s in inside),
        "spill_requests": last["spill_requests"] - first["spill_requests"],
        "restore_requests": last["restore_requests"] - first["restore_requests"],
    }


def sampler_name(host: str) -> str:
    """The sampler file stem of a pod: hostname ``<job>-worker-<K>`` writes ``node-<K>.tsv``."""
    return f"node-{str(host).rsplit('-', 1)[-1]}"


def net_bytes(samples: list[dict], intervals: list[tuple[float, float]]) -> tuple[int, int] | None:
    """eth0 bytes received and sent over the union of ``intervals``.

    Each interval runs from the last sample at or before its start to the
    first sample at or after its end. Intervals that share sampled seconds
    merge, so no second counts twice. None if a bracket sample or the counter
    is missing.
    """
    if not samples or samples[0].get("net_rx_bytes", -1) < 0:
        return None
    epochs = [s["epoch"] for s in samples]
    spans: list[list[int]] = []
    for t0, t1 in sorted(intervals):
        first, last = bisect.bisect_right(epochs, t0) - 1, bisect.bisect_left(epochs, t1)
        if first < 0 or last >= len(samples):
            return None
        if spans and first < spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], last)
        else:
            spans.append([first, last])
    return tuple(sum(samples[b][key] - samples[a][key] for a, b in spans) for key in ("net_rx_bytes", "net_tx_bytes"))


def add_wire(results: dict, nodes: dict) -> None:
    """G1c inputs per rollout: eth0 rx of each node over its shard windows, in objects, and the head tx share."""
    for arm in results.values():
        for row in arm["rollouts"].values():
            windows = {sampler_name(host): w for host, w in (row.get("wire_windows") or {}).items()}
            objects = {}
            for name, w in windows.items():
                moved = net_bytes(nodes.get(name, []), w["intervals"])
                if moved is not None and w["bytes"] > 0:
                    objects[name] = moved[0] / w["bytes"]
            remote = [w for name, w in windows.items() if name != HEAD_SAMPLER]
            head = net_bytes(nodes.get(HEAD_SAMPLER, []), [i for w in remote for i in w["intervals"]])
            remote_bytes = sum(w["bytes"] for w in remote)
            row["wire"] = {
                "objects": objects,
                # 1.0: the head sent every remote copy; less: remote nodes served some copies to each other.
                "head_tx_share": head[1] / remote_bytes if head is not None and remote_bytes > 0 else None,
            }


def add_node_windows(results: dict, nodes: dict, r3: str, prefetch: str) -> None:
    """Sampler stats per arm, per r3 fetch window, and per pull window.

    The pull window of rollout N is the prefetch of N during train(N-1). The
    r3 arm gets the same window, placed at the same offset from its own
    ``put`` of rollout N, so the two arms compare the same part of the step.
    """
    add_wire(results, nodes)
    for arm in results.values():
        arm["nodes"] = {node: window_stats(s, *arm["window"]) for node, s in nodes.items()}
        for row in arm["rollouts"].values():
            if view := row.get("fetch_nodes"):
                row["fetch_window_nodes"] = {
                    n: window_stats(s, view["t0"], view["t0"] + view["last"]) for n, s in nodes.items()
                }
    for rollout_id, row in results.get(prefetch, {}).get("rollouts", {}).items():
        view = row.get("prefetch_nodes")
        if not view or "put_t1" not in row:
            continue
        offset, length = view["t0"] - row["put_t1"], view["last"]
        for name in (r3, prefetch):
            other = results.get(name, {}).get("rollouts", {}).get(rollout_id)
            if other and "put_t1" in other:
                t0 = other["put_t1"] + offset
                other["pull_window_nodes"] = {n: window_stats(s, t0, t0 + length) for n, s in nodes.items()}


def gates(
    results: dict, controls: list[str], r3: str, prefetch: str, manifest: dict | None, reference_digests: dict | None
) -> list[dict]:
    """The gates of the plan. A gate whose inputs are missing reads NA; INFO gates only report."""
    out = []

    def gate(gate_id: str, text: str, value, verdict: str | None) -> None:
        out.append({"id": gate_id, "gate": text, "value": value, "verdict": verdict or "NA"})

    def rows(name: str) -> dict:
        return results.get(name, {}).get("rollouts", {})

    def warm_ids(name: str) -> list[int]:
        return sorted(rows(name))[1:]

    def warm(name: str, key: str, sub: str | None = None) -> list:
        values = [rows(name)[i].get(key) for i in warm_ids(name)]
        return [v[sub] if sub and v else v for v in values if v is not None]

    base = controls[0] if controls else None
    for name in [*controls, r3, prefetch]:
        if name in results:
            arm = results[name]
            head = (arm.get("nodes") or {}).get(HEAD_SAMPLER) or {}
            value = {
                "rc": arm["rc"],
                "oom": arm["out_of_memory"],
                "spill": head.get("spill_requests"),
                "restore": head.get("restore_requests"),
            }
            ok = (
                arm["rc"] == 0
                and not arm["out_of_memory"]
                and not head.get("spill_requests")
                and not head.get("restore_requests")
            )
            gate(
                f"G0[{name}]",
                "rc 0, no out-of-memory, no head spill or restore in the arm window (null: no sampler)",
                value,
                "PASS" if ok else "FAIL",
            )
    # Q1: bytes on the wire.
    if manifest and rows(r3):
        routing_int32 = (manifest["tokens"] - manifest["rows"]) * ROUTING_BYTES_PER_TOKEN_INT32
        ratio = _median([row.get("put_routing_bytes") for row in rows(r3).values()])
        ratio = ratio / routing_int32 if ratio else None
        gate(
            "G1a",
            "r3 routing bytes put / control int32 routing bytes (manifest), expect 0.500 +- 0.001",
            ratio,
            None if ratio is None else ("PASS" if abs(ratio - 0.5) <= 0.001 else "FAIL"),
        )
    totals = [(row["put_bytes"], row["put_routing_bytes"]) for row in rows(r3).values() if row.get("put_bytes")]
    if totals:
        total, routing = totals[0]
        value = {"routing_share": routing / total, "object_vs_int32_object": total / (total + routing)}
        gate(
            "G1b",
            "routing share of the r3 objects (>= 0.97, so the object bytes halve with the routing bytes)",
            value,
            "PASS" if routing / total >= 0.97 else "FAIL",
        )
    # One object per remote node and no second copy: 2.0 fails the upper bound, a missed pull the lower one.
    for name in (r3, prefetch):
        if not rows(name):
            continue
        objects = [
            (node, x) for row in rows(name).values() for node, x in (row.get("wire") or {}).get("objects", {}).items()
        ]
        remote = [x for node, x in objects if node != HEAD_SAMPLER]
        head = [x for node, x in objects if node == HEAD_SAMPLER]
        value = {
            "remote": _stats(remote),
            "head_max": max(head, default=None),
            "head_tx_share": [(row.get("wire") or {}).get("head_tx_share") for row in rows(name).values()],
        }
        gate(
            f"G1c[{name}]",
            f"eth0 rx of each remote node over its pull and fetch windows / its object bytes "
            f"({WIRE_OBJECTS[0]}-{WIRE_OBJECTS[1]}); INFO: head (about 0), head tx share",
            value,
            (
                None
                if not remote
                else ("PASS" if all(WIRE_OBJECTS[0] <= x <= WIRE_OBJECTS[1] for x in remote) else "FAIL")
            ),
        )
    if base and rows(r3):
        new, old = _median(warm(r3, "last")), _median(warm(base, "last"))
        gate(
            "G1d",
            "INFO: last node ready, r3 / control, warm median (bytes model: about 0.5)",
            new / old if new and old else None,
            "INFO",
        )
    if base and rows(r3):
        value = {
            "convert_s": _median(warm(r3, "convert_s")),
            "gen_r3_minus_control_s": (
                _median(warm(r3, "gen")) - _median(warm(base, "gen"))
                if _median(warm(r3, "gen")) is not None and _median(warm(base, "gen")) is not None
                else None
            ),
        }
        gate("G1e", "INFO: head cost of the int16 cast: r3 convert s, and gen(r3) - gen(control) s", value, "INFO")
    # Q2: fill.
    if base and rows(r3):
        old = _median(warm(base, "fill0"))
        new = _median(warm(r3, "fill0"))
        if old and new is not None:
            ratio = new / old
            gate(
                "G2a",
                "r3 fill0 / control fill0, warm median (PASS <= 0.25; FAIL >= 0.50, the int16-only bytes model)",
                ratio,
                "PASS" if ratio <= 0.25 else ("FAIL" if ratio >= 0.5 else "UNCLEAR"),
            )
        fill_max = max(warm(r3, "fill_s", "max"), default=None)
        if old and fill_max is not None:
            gate(
                "G2b",
                "r3 per-rank fill max / control fill0 warm median (<= 0.25)",
                fill_max / old,
                "PASS" if fill_max / old <= 0.25 else "FAIL",
            )
    # Q3: prefetch hides the fetch.
    if rows(prefetch) and rows(r3):
        ids = warm_ids(prefetch)
        hits = [rows(prefetch)[i].get("prefetch_modes", {}).get("hit", 0) for i in ids]
        fetch_max = warm(prefetch, "fetch_s", "max")
        mismatch = sum(rows(prefetch)[i].get("hit_ref_mismatch", 0) for i in ids)
        ok = hits and all(h == NUM_RANKS for h in hits) and fetch_max and max(fetch_max) <= 1.0 and mismatch == 0
        gate(
            "G3a",
            "r3-prefetch warm: prefetch=hit on 64 ranks, ref match, max fetch <= 1.0 s",
            {"hits": hits, "fetch_max": fetch_max, "ref_mismatch": mismatch},
            "PASS" if ok else "FAIL",
        )
        last_pf, head_r3 = _median(warm(prefetch, "last")), _median(warm(r3, "head"))
        if last_pf is not None and head_r3 is not None:
            gate(
                "G3b",
                "r3-prefetch last node ready - r3 head node ready, warm median (<= 1.0 s)",
                last_pf - head_r3,
                "PASS" if last_pf - head_r3 <= 1.0 else "FAIL",
            )
        pre = warm(prefetch, "pre_exact")
        gate(
            "G3c",
            "r3-prefetch exact pre-train window, warm max (<= 5.0 s); r3 for reference",
            {"r3-prefetch": pre, "r3": warm(r3, "pre_exact")},
            None if not pre else ("PASS" if max(pre) <= 5.0 else "FAIL"),
        )
    # Q4: replay tensors, log-probs and grad norms.
    for name in (r3, prefetch):
        if not (base and rows(name) and rows(base)):
            continue
        first = min(rows(name))
        new, ref = rows(name)[first].get("forward", {}), rows(base).get(first, {}).get("forward", {})
        diff = {k: (ref.get(k), new.get(k)) for k in FORWARD_KEYS if ref.get(k) != new.get(k)}
        gate(
            f"G4a[{name}]",
            f"step {first} forward metrics bit-identical to {base}",
            diff or "all equal",
            "PASS" if new and not diff else "FAIL",
        )
        norms = [rows(c).get(first, {}).get("grad_norm") for c in controls]
        norms = [n for n in norms if n]
        value = rows(name)[first].get("grad_norm")
        if norms and value:
            rel = abs(value - statistics.fmean(norms)) / statistics.fmean(norms)
            gate(
                f"G4b[{name}]",
                f"step {first} grad_norm vs control mean, relative (PASS <= 2e-4; FAIL > 1e-3)",
                rel,
                "PASS" if rel <= 2e-4 else ("FAIL" if rel > 1e-3 else "UNCLEAR"),
            )
        later = {}
        for rollout_id in warm_ids(name):
            refs = [rows(c).get(rollout_id, {}).get("grad_norm") for c in controls]
            refs = [n for n in refs if n]
            value = rows(name)[rollout_id].get("grad_norm")
            if len(refs) >= 1 and value:
                spread = (max(refs) - min(refs)) / statistics.fmean(refs) if len(refs) > 1 else None
                later[rollout_id] = {
                    "rel": abs(value - statistics.fmean(refs)) / statistics.fmean(refs),
                    "base_spread": spread,
                }
        gate(f"G4e[{name}]", "INFO: warm grad_norm vs control mean (base-vs-base spread for scale)", later, "INFO")
        worst = None
        for rollout_id, row in rows(name).items():
            ref_peak, new_peak = rows(base).get(rollout_id, {}).get("peak_gib"), row.get("peak_gib")
            if ref_peak and new_peak:
                d = max(abs(new_peak[pp] - ref_peak[pp]) for pp in ref_peak if pp in new_peak)
                worst = d if worst is None else max(worst, d)
        gate(
            f"G4d[{name}]",
            "max over rollouts and stages of |GPU peak allocated - control| GiB (<= 0.5)",
            worst,
            None if worst is None else ("PASS" if worst <= 0.5 else "FAIL"),
        )
    digests: dict = defaultdict(set)
    for name in (r3, prefetch):
        for row in rows(name).values():
            for rank, sha in row.get("digests", {}).items():
                digests[rank].add(sha)
    if reference_digests and digests:
        checked = {r: (sha, digests.get(int(r), set())) for r, sha in reference_digests.items()}
        bad = sorted(int(r) for r, (sha, seen) in checked.items() if seen != {sha})
        gate(
            "G4g",
            "in-job replay digest == offline reference of the 4716a367a fill (check_r3_fill.py) on each checked rank",
            {"checked": len(checked), "mismatch": bad},
            "PASS" if not bad else "FAIL",
        )
    digest_s = [x for name in (r3, prefetch) for x in warm(name, "digest_s", "max")]
    if digest_s:
        gate(
            "G4f",
            "digest cost, max over ranks and warm rollouts, s (<= 1.0)",
            max(digest_s),
            "PASS" if max(digest_s) <= 1.0 else "FAIL",
        )
    if digests:
        bad = sorted(rank for rank, shas in digests.items() if len(shas) != 1)
        gate(
            "G4c",
            "one replay digest per rank over all rollouts of r3 and r3-prefetch",
            {"ranks": len(digests), "ranks_with_2+": bad},
            "PASS" if not bad and len(digests) == NUM_RANKS else "FAIL",
        )
    # Q5: the cost of prefetch.
    if rows(prefetch) and rows(r3):
        a, b = warm(r3, "actor_train"), warm(prefetch, "actor_train")
        if a and b:
            rel = statistics.fmean(b) / statistics.fmean(a) - 1
            gate(
                "G5a", "actor_train warm mean, r3-prefetch / r3 - 1 (<= +1.0%)", rel, "PASS" if rel <= 0.01 else "FAIL"
            )
        a, b = _median(warm(r3, "optimizer_s", "median")), _median(warm(prefetch, "optimizer_s", "median"))
        if a is not None and b is not None:
            gate(
                "G5b",
                "optimizer step median, r3-prefetch - r3, s (<= max(1 s, 5%))",
                b - a,
                "PASS" if b - a <= max(1.0, 0.05 * a) else "FAIL",
            )
        shard_gib = _median([row["put_bytes"] / 2 / 2**30 for row in rows(r3).values() if row.get("put_bytes")])
        deltas = {}
        for rollout_id in warm_ids(prefetch):
            mem_a, mem_b = rows(r3).get(rollout_id, {}).get("mem", {}), rows(prefetch)[rollout_id].get("mem", {})
            for node in set(mem_a) & set(mem_b):
                end_a, end_b = mem_a[node].get("end"), mem_b[node].get("end")
                if end_a and end_b:
                    slot = deltas.setdefault(node, {"rss": 0.0, "shmem": 0.0})
                    slot["rss"] = max(slot["rss"], end_b["rss_gib_max"] - end_a["rss_gib_max"])
                    slot["shmem"] = max(slot["shmem"], end_b["shmem_gib"] - end_a["shmem_gib"])
        if deltas and shard_gib:
            rss, shmem = max(d["rss"] for d in deltas.values()), max(d["shmem"] for d in deltas.values())
            ok = rss <= 2.0 and shmem <= shard_gib + 2.0
            gate(
                "G5c",
                f"train end, r3-prefetch - r3: max rank RssAnon delta <= 2 GiB, max node Shmem delta <= one object ({shard_gib:.1f} GiB) + 2",
                {"rss_gib": rss, "shmem_gib": shmem},
                "PASS" if ok else "FAIL",
            )
        cpu = {}
        for rollout_id in warm_ids(prefetch):
            wa, wb = rows(r3).get(rollout_id, {}).get("pull_window_nodes"), rows(prefetch)[rollout_id].get(
                "pull_window_nodes"
            )
            for node in set(wa or {}) & set(wb or {}):
                if wa[node] and wb[node]:
                    cpu.setdefault(node, []).append(
                        {
                            "cpu_busy_delta": wb[node]["cpu_busy"] - wa[node]["cpu_busy"],
                            "raylet_cpu_s_delta": wb[node]["raylet_cpu_s"] - wa[node]["raylet_cpu_s"],
                        }
                    )
        gate(
            "G5d",
            "INFO: pull window, r3-prefetch - r3: node CPU busy share and raylet CPU seconds",
            cpu or None,
            "INFO" if cpu else None,
        )
        overlap = {}
        for rollout_id in warm_ids(prefetch):
            pull, opt = rows(prefetch)[rollout_id].get("prefetch_nodes"), rows(prefetch).get(rollout_id - 1, {}).get(
                "optimizer_window"
            )
            if pull and opt:
                overlap[rollout_id] = max(0.0, min(pull["t0"] + pull["last"], opt[1]) - max(pull["t0"], opt[0]))
        if overlap:
            gate(
                "G5e",
                "pull of N overlaps the optimizer step of train(N-1), s (0 everywhere = not exercised)",
                overlap,
                "NOT-EXERCISED" if not any(overlap.values()) else "INFO",
            )
    for name in (r3, prefetch):
        ids = warm_ids(name)
        growth = {}
        if len(ids) >= 2:
            first, last = rows(name)[ids[0]].get("mem", {}), rows(name)[ids[-1]].get("mem", {})
            for node in set(first) & set(last):
                if first[node].get("end") and last[node].get("end"):
                    growth[node] = last[node]["end"]["shmem_gib"] - first[node]["end"]["shmem_gib"]
        if growth:
            worst = max(growth.values())
            gate(
                f"G5f[{name}]",
                f"no plasma leak: node Shmem at train end, rollout {ids[-1]} - {ids[0]}, GiB (<= 1.0)",
                worst,
                "PASS" if worst <= 1.0 else "FAIL",
            )
    # Q6: node by node.
    serial = {}
    for name in [*controls, r3]:
        serial[f"{name}:train"] = _median(warm(name, "serial"))
    serial[f"{r3}:fetch-lines"] = _median(warm(r3, "fetch_nodes", "serial"))
    serial[f"{prefetch}:pulls"] = _median(warm(prefetch, "prefetch_nodes", "serial"))
    pushes = [
        ((row.get("fetch_window_nodes") or {}).get(HEAD_SAMPLER) or {}).get("pushes_remaining_max")
        for row in rows(r3).values()
    ]
    verdicts = {
        k: None if v is None else ("SERIAL" if v >= 0.6 else ("CONCURRENT" if v <= 0.3 else "PARTIAL"))
        for k, v in serial.items()
    }
    gate(
        "G6",
        "serial index, warm median (>= 0.6 SERIAL, <= 0.3 CONCURRENT); head pushes remaining max per r3 fetch",
        {"serial": serial, "verdict": verdicts, "head_pushes_remaining_max": pushes},
        "INFO",
    )
    return out


def _fmt(value, width: int = 7, digits: int = 1) -> str:
    if value is None:
        return "-".rjust(width)
    if isinstance(value, float):
        return f"{value:{width}.{digits}f}"
    return str(value).rjust(width)


R3_KEYS = (
    "put_bytes",
    "put_routing_bytes",
    "convert_s",
    "fetch_s",
    "prefetch_modes",
    "local_before",
    "fill_s",
    "fill_bytes_max",
    "digest_s",
    "pre_exact",
    "prefetch_done_s",
    "prefetch_pull_s",
    "prefetch_deser_s",
    "optimizer_s",
)


def print_tables(results: dict, gate_rows: list[dict]) -> None:
    for name, arm in results.items():
        print(f"\n## {name}  rc={arm['rc']}  ({arm['log']})")
        print(
            " rollout  ranks  first   last   head serial  s/node  fill0    pre   gen   dp48  actor_train  grad_norm   logprob_abs_diff"
        )
        for rollout_id, row in arm["rollouts"].items():
            lp = row["forward"].get("train/train_rollout_logprob_abs_diff")
            print(
                f" {rollout_id:7d} {_fmt(row['ranks'], 6)} {_fmt(row['first'])} {_fmt(row['last'])} {_fmt(row['head'], 6)}"
                f" {_fmt(row['serial'], 6, 2)} {_fmt(row['per_node_slope'])} {_fmt(row['fill0'], 6)} {_fmt(row['pre'], 6)}"
                f" {_fmt(row.get('gen'), 5, 0)} {_fmt(row['dp48'], 6)} {_fmt(row['actor_train'], 12)}"
                f"  {_fmt(row['grad_norm'], 10, 7)}  {lp!r}"
            )
        for rollout_id, row in arm["rollouts"].items():
            stairs = " ".join(f"{node}:{t:.1f}" for node, t in row["staircase"] or [])
            print(f"   staircase {rollout_id}: {stairs}")
        for rollout_id, row in arm["rollouts"].items():
            r3 = {k: row[k] for k in R3_KEYS if k in row}
            if r3:
                print(f"   r3 {rollout_id}: {json.dumps(r3, default=str)}")
    if gate_rows:
        print("\n## gates")
        for g in gate_rows:
            print(f" {g['verdict']:13s} {g['id']:16s} {g['gate']}: {json.dumps(g['value'], default=str)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("arms", nargs="+", help="NAME=PATH, PATH = the arm dir or its trainer-0.log")
    parser.add_argument(
        "--control", action="append", default=[], help="control arm; the first one is the timing control"
    )
    parser.add_argument("--r3", default="r3")
    parser.add_argument("--prefetch", default="r3-prefetch")
    parser.add_argument("--manifest", type=Path, help="data/t2/manifest.json of the rows (control bytes)")
    parser.add_argument("--nodes", type=Path, help="harness sampler dir <run>/nodes")
    parser.add_argument("--reference-digests", type=Path, help="check_r3_fill.py output: JSON {rank: sha256}")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = {}
    for spec in args.arms:
        name, _, path = spec.partition("=")
        results[name] = parse_arm(Arm(name, Path(path)))
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    if manifest:
        routing = (manifest["tokens"] - manifest["rows"]) * ROUTING_BYTES_PER_TOKEN_INT32
        print(
            f"manifest: {manifest['rows']} rows, {manifest['tokens']} tokens, int32 routing {routing} B ({routing / 1e9:.2f} GB)"
        )
    if args.nodes:
        add_node_windows(results, read_nodes(args.nodes), args.r3, args.prefetch)
    reference = json.loads(args.reference_digests.read_text()) if args.reference_digests else None
    gate_rows = gates(results, args.control, args.r3, args.prefetch, manifest, reference)
    print_tables(results, gate_rows)
    if args.json:
        args.json.write_text(json.dumps({"arms": results, "gates": gate_rows}, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
