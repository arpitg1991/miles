"""All-to-all bandwidth of two expert-parallel layouts on the kdatp-prof nodes.

    python3 -m torch.distributed.run --nnodes 8 --nproc-per-node 8 --node-rank <n> \\
        --rdzv-backend static --rdzv-endpoint <job>-worker-0:23456 a2a_bench.py --out <dir>

The layouts run one after the other; all groups of a layout run at the same time:

- ``ep16``: groups of 16 ranks, global ranks 16k..16k+15 on nodes 2k and 2k+1,
  the expert groups of the r47 pipeline stages. Half of the bytes of each
  rank cross EFA.
- ``ep8``: groups of the 8 GPUs of one node (NVLink only).

A size is the bytes that one rank sends in one ``all_to_all_single`` call
(bf16, equal splits). 536,870,912 B is the r47 dispatch estimate: 8192 tokens
x top-8 x hidden 4096 x 2 B. Each size runs 3 warmup and 10 timed calls, and
a world barrier starts each call. The time of a call is the max over the
ranks of the slowest group; the table gives the median over the 10 calls and
the range of the per-group medians.
``algbw`` is the bytes of one rank / time; ``cross-node`` is the bytes of one
rank that go to the other node / time. Rank 0 writes ``a2a.json`` and
``a2a.md`` to ``--out``.
"""

import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

SIZES = (32 << 20, 64 << 20, 128 << 20, 256 << 20, 8192 * 8 * 4096 * 2, 1 << 30)
WARMUP, ITERS = 3, 10


def time_calls(group: dist.ProcessGroup, numel: int, device: torch.device) -> torch.Tensor:
    """Seconds of each timed call on this rank."""
    send = torch.ones(numel, dtype=torch.bfloat16, device=device)
    recv = torch.empty_like(send)
    times = torch.zeros(ITERS, dtype=torch.float64, device=device)
    for i in range(WARMUP + ITERS):
        dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_to_all_single(recv, send, group=group)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= WARMUP:
            times[i - WARMUP] = time.perf_counter() - start
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--group-sizes", default="16,8", help="ranks per group, one value per layout")
    parser.add_argument("--sizes", default=",".join(map(str, SIZES)), help="bytes that one rank sends")
    parser.add_argument("--backend", default="nccl", help="gloo runs on the CPU (a local check)")
    cli = parser.parse_args()
    per_node = int(os.environ["LOCAL_WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if cli.backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # A hung call fails after 180 s, inside the 420 s bound of the driver.
    bound = device if device.type == "cuda" else None
    dist.init_process_group(cli.backend, timeout=timedelta(seconds=180), device_id=bound)
    rank, world = dist.get_rank(), dist.get_world_size()
    layouts = []
    for group_size in map(int, cli.group_sizes.split(",")):
        members = [list(range(first, first + group_size)) for first in range(0, world, group_size)]
        group, _ = dist.new_subgroups_by_enumeration(members, backend=cli.backend)
        mine = members[rank // group_size]
        cross = sum(peer // per_node != rank // per_node for peer in mine) / group_size
        rows = []
        for size in map(int, cli.sizes.split(",")):
            numel = size // 2 // group_size * group_size
            gathered = [torch.zeros(ITERS, dtype=torch.float64, device=device) for _ in range(world)]
            dist.all_gather(gathered, time_calls(group, numel, device))
            by_rank = torch.stack(gathered).cpu()
            # Per group and call: the slowest rank. Per call: the slowest group.
            group_calls = [by_rank[ranks].max(dim=0).values.tolist() for ranks in members]
            calls = [max(values) for values in zip(*group_calls, strict=True)]
            median = statistics.median(calls)
            nbytes = numel * 2
            rows.append(
                {
                    "bytes": nbytes,
                    "median_s": median,
                    "algbw_gbps": nbytes / median / 1e9,
                    "cross_node_gbps": nbytes * cross / median / 1e9,
                    "group_median_s": [statistics.median(values) for values in group_calls],
                    "calls_s": calls,
                }
            )
        layouts.append({"layout": f"ep{group_size}", "cross_node_fraction": cross, "rows": rows})
    if rank == 0:
        out = Path(cli.out)
        out.mkdir(parents=True, exist_ok=True)
        meta = {"world": world, "per_node": per_node, "warmup": WARMUP, "iters": ITERS, "backend": cli.backend}
        (out / "a2a.json").write_text(json.dumps({**meta, "layouts": layouts}, indent=2))
        lines = [
            "| Layout | Bytes per rank | Median ms | algbw GB/s | Cross-node GB/s | Group medians ms |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for layout in layouts:
            for row in layout["rows"]:
                lines.append(
                    f"| {layout['layout']} | {row['bytes']:,} | {row['median_s'] * 1e3:.2f} | "
                    f"{row['algbw_gbps']:.1f} | {row['cross_node_gbps']:.1f} | "
                    f"{min(row['group_median_s']) * 1e3:.2f} to {max(row['group_median_s']) * 1e3:.2f} |"
                )
        (out / "a2a.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
