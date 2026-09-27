"""Collect the per-arm results of a kdatp run dir into one JSON (stdout).

For each ``<run>/<arm>/trainer-0.log`` it reads:

- the ``perf N: {...}``, ``step N: {...}`` and ``rollout N: {...}`` dicts
  that miles logs (all ranks write to the ray driver log),
- the ``[peak-memory]`` lines (``MILES_LOG_PEAK_MEMORY=1``): the max over the
  ranks of each pipeline stage, per rollout,
- ``<arm>/rc`` (the launcher exit code) and an out-of-memory marker.

It also gives, per arm, the mean of each perf and step value over the steps
after the first one (the first step pays the kernel compile).

    python3 parse_logs.py /mnt/scratch-s3files-rw/guparpit/kdatp/t2/<stamp> > results.json
"""

import ast
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRICS = re.compile(r"\b(perf|step|rollout) (\d+): (\{.*\})\s*$")
PEAK = re.compile(
    r"\[peak-memory\] rollout=(\d+) rank=(\d+) pp=(\d+) tp=(\d+) max_allocated_gib=([\d.]+) max_reserved_gib=([\d.]+)"
)
OOM = re.compile(r"OutOfMemoryError|CUDA out of memory")


def parse_arm(arm_dir: Path) -> dict:
    log = arm_dir / "trainer-0.log"
    metrics: dict[str, dict[int, dict]] = defaultdict(lambda: defaultdict(dict))
    peaks: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    oom = False
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            if match := METRICS.search(line):
                kind, rollout_id, payload = match.groups()
                try:
                    values = ast.literal_eval(payload)
                except (ValueError, SyntaxError):
                    continue
                metrics[kind][int(rollout_id)].update(v for v in values.items() if isinstance(v[1], (int, float)))
            elif match := PEAK.search(line):
                rollout_id, _rank, pp, _tp, allocated, reserved = match.groups()
                slot = peaks[int(rollout_id)][int(pp)]
                slot[0] = max(slot[0], float(allocated))
                slot[1] = max(slot[1], float(reserved))
            elif OOM.search(line):
                oom = True
    rc_file = arm_dir / "rc"
    result = {
        "rc": int(rc_file.read_text()) if rc_file.is_file() else None,
        "out_of_memory": oom,
        "metrics": {kind: dict(sorted(by_id.items())) for kind, by_id in metrics.items()},
        "peak_memory_gib": {
            rollout_id: {pp: {"max_allocated": a, "max_reserved": r} for pp, (a, r) in sorted(by_pp.items())}
            for rollout_id, by_pp in sorted(peaks.items())
        },
    }
    steady = {}
    for kind in ("perf", "step"):
        ids = sorted(metrics.get(kind, {}))[1:]
        keys = {key for rollout_id in ids for key in metrics[kind][rollout_id]}
        for key in sorted(keys):
            values = [metrics[kind][i][key] for i in ids if key in metrics[kind][i]]
            if values:
                steady[key] = statistics.fmean(values)
    result["steady_mean"] = steady
    return result


def main() -> None:
    run_dir = Path(sys.argv[1])
    arms = sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / "trainer-0.log").exists())
    json.dump({arm.name: parse_arm(arm) for arm in arms}, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
