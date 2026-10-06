"""Offline byte check of the r3 replay fill on the real T2 rows, before the image build.

It fills the rows with the ``fill_replay_data`` of miles ``4716a367a`` (on the
int32 routing of the rows, as in the control job) and with the fill of this
checkout (on the int16 shards of ``convert_samples_to_train_data``), for 8
train ranks of the r47 layout, and it compares every replay buffer. The
micro-batches come from the rollout-side DP schedule of the control job's
arguments, read from the argument dump in its ``trainer-0.log``.

Outputs in ``--out``:

- ``check-r3-fill.txt``: the mismatch count per rank, the layers and the
  micro-batches, and the CPU seconds of three fills on one rank (old fill on
  int32, old fill on int16, new fill on int16).
- ``check-r3-fill-digests.json``: ``{rank: sha256}`` of the old buffers cast to
  int16 (``replay_digest``, the function of the in-job ``[r3-digest]`` line).
  ``parse_r3_timing.py --reference-digests`` reads it (gate G4g).

Run it from the repo root, with the miles dependencies (the r17 image):

    git show 4716a367a:miles/backends/training_utils/replay_data.py > <dir>/replay_data_4716a367a.py
    python3 examples/arena/harbor-rl-glm53-flash/kdatp/prof/check_r3_fill.py --rows <t2>/rollout_40.pt \\
        --args-log <control>/base/trainer-0.log --old-fill <dir>/replay_data_4716a367a.py --out <dir>
"""

import argparse
import ast
import hashlib
import json
import re
import time
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

from miles.backends.training_utils import parallel
from miles.backends.training_utils.data import get_data_iterator
from miles.backends.training_utils.replay_data import fill_replay_data, register_replay_list_sequential
from miles.ray.rollout.debug_data import _load_rollout_data_file
from miles.ray.rollout.train_data_conversion import (
    convert_samples_to_train_data,
    process_rollout_data_shard,
    split_train_data_by_dp_scheduled_raw,
)
from miles.utils.r3_log import replay_digest

KEY = "rollout_routed_experts"
# The plan's ranks: DP {0, 1} x PP {0, 3} x TP {0, 7}; rank = pp * 16 + dp * 8 + tp (the control's [peak-memory] lines).
RANKS = (0, 7, 8, 15, 48, 55, 56, 63)
ARG_LINE = re.compile(r"^\s+(\w+) \.+ (.*)$")


def read_args(log: Path) -> Namespace:
    """The parsed arguments of a train job, from the Megatron argument dump in its log."""
    values, inside = {}, False
    for line in log.read_text(errors="replace").splitlines():
        if "------------------------ arguments ------------------------" in line:
            inside = True
        elif "end of arguments" in line:
            break
        elif inside and (m := ARG_LINE.match(line)):
            try:
                values[m[1]] = ast.literal_eval(m[2])
            except (ValueError, SyntaxError):
                values[m[1]] = m[2]
    assert values, f"no argument dump in {log}"
    return Namespace(**values)


def load_module(path: Path, package: str) -> ModuleType:
    """A module from a source file, with its relative imports resolved in ``package``."""
    module = ModuleType(f"{package}.{path.stem}")
    module.__package__ = package
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def stage_moe_layers(args: Namespace) -> list[list[int]]:
    """The global MoE layer ids of each pipeline stage (register_replay_list_moe on the job layout)."""
    pp, first, last = (
        args.pipeline_model_parallel_size,
        args.decoder_first_pipeline_num_layers,
        args.decoder_last_pipeline_num_layers,
    )
    middle = (args.num_layers - first - last) // (pp - 2)
    sizes = [first] + [middle] * (pp - 2) + [last]
    assert sum(sizes) == args.num_layers, sizes
    starts = np.cumsum([0] + sizes[:-1])
    return [[i for i in range(s, s + n) if args.moe_layer_freq[i]] for s, n in zip(starts, sizes, strict=True)]


class _Replay:
    """Replay.record without pinned memory: the same layout and bytes."""

    def __init__(self, stream_idx: int) -> None:
        self.stream_idx = stream_idx
        self.top_indices_list: list[torch.Tensor] = []

    def record(self, top_indices: torch.Tensor) -> None:
        self.top_indices_list.append(torch.empty_like(top_indices, device="cpu").copy_(top_indices))


def run_fill(fill, args: Namespace, shard: dict, rows: list[torch.Tensor], layers: list[int]):
    """Buffers in replay order, then micro-batch order (the order of the in-job digest), and the seconds."""
    rollout_data = process_rollout_data_shard(args, dict(shard))
    rollout_data["tokens"] = [torch.as_tensor(t) for t in rollout_data["tokens"]]
    rollout_data[KEY] = rows
    data_iterator, num_microbatches = get_data_iterator(args, None, rollout_data)
    replays = [_Replay(layer) for layer in layers]
    t0 = time.perf_counter()
    fill(
        args=args,
        models=None,
        data_iterator=data_iterator,
        num_microbatches=num_microbatches,
        rollout_data=rollout_data,
        data_key=KEY,
        replay_list=replays,
        register_replay_list_func=register_replay_list_sequential,
        if_sp_region=True,
        indices_are_token_positions=False,
    )
    seconds = time.perf_counter() - t0
    return [buf for replay in replays for buf in replay.top_indices_list], num_microbatches, seconds


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("--rows", type=Path, required=True, help="data/t2/rollout_40.pt")
    cli.add_argument("--args-log", type=Path, required=True, help="trainer-0.log of the control arm")
    cli.add_argument("--old-fill", type=Path, required=True, help="replay_data.py of 4716a367a (git show)")
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--threads", type=int, default=1, help="torch CPU threads for the timed fills")
    opts = cli.parse_args()
    torch.set_num_threads(opts.threads)
    opts.out.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    def say(text: str) -> None:
        print(text, flush=True)
        report.append(text)

    args = read_args(opts.args_log)
    old_fill = load_module(opts.old_fill, "miles.backends.training_utils").fill_replay_data
    say(f"old fill: {opts.old_fill} sha256 {hashlib.sha256(opts.old_fill.read_bytes()).hexdigest()}")
    say(
        f"layout: tp {args.tensor_model_parallel_size} sp {args.sequence_parallel} cp {args.context_parallel_size} "
        f"pp {args.pipeline_model_parallel_size} qkv {args.qkv_format} pad multiplier {args.data_pad_size_multiplier} "
        f"allgather_cp {args.allgather_cp}"
    )
    t0 = time.perf_counter()
    samples, metadata = _load_rollout_data_file(opts.rows)
    routing32 = [s.rollout_routed_experts for s in samples]
    say(
        f"rows: {len(samples)} samples, {sum(len(s.tokens) for s in samples)} tokens, load {time.perf_counter() - t0:.0f} s"
    )
    assert {r.dtype for r in routing32} == {np.dtype(np.int32)}, {r.dtype for r in routing32}

    data = convert_samples_to_train_data(args, samples, metadata, None, None)
    config = {
        "dp_size": 2,
        "cp_size": args.context_parallel_size,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": None,
    }
    shards = split_train_data_by_dp_scheduled_raw(args, data, train_parallel_config=config)
    say(f"schedule: num_microbatches {shards[0]['num_microbatches']} per DP rank (the control job logged [156])")
    layers = stage_moe_layers(args)
    say(f"MoE layers per stage: {[f'{s[0]}-{s[-1]} ({len(s)})' for s in layers]}")

    digests, failures = {}, 0
    tp_size = args.tensor_model_parallel_size
    for rank in RANKS:
        pp, dp, tp = rank // 16, (rank // 8) % 2, rank % tp_size
        shard = shards[dp]
        rows16 = [torch.from_numpy(r) for r in shard[KEY]]
        rows32 = [torch.from_numpy(routing32[i]) for i in shard["partition"]]
        assert all(
            a.dtype == torch.int16 and torch.equal(a.to(torch.int32), b) for a, b in zip(rows16, rows32, strict=True)
        )
        parallel._parallel_state = SimpleNamespace(
            tp=SimpleNamespace(rank=tp, size=tp_size),
            cp=SimpleNamespace(rank=0, size=args.context_parallel_size),
            vpp_size=1,
        )
        new, num_mb, new_s = run_fill(fill_replay_data, args, shard, rows16, layers[pp])
        old, _, old_s = run_fill(old_fill, args, shard, rows32, layers[pp])
        mismatch = sum(1 for a, b in zip(new, old, strict=True) if not torch.equal(a.to(torch.int32), b))
        mismatch += abs(len(new) - len(old))
        failures += mismatch
        digests[str(rank)] = replay_digest([b.to(torch.int16) for b in old])
        same = replay_digest(new) == digests[str(rank)]
        say(
            f"rank {rank:2d} (pp {pp} dp {dp} tp {tp}): {len(new)} buffers, {sum(b.nbytes for b in new)} B int16, "
            f"{num_mb} micro-batches, mismatch {mismatch}, new digest == old digest: {same}, "
            f"fill s new {new_s:.2f} old int32 {old_s:.2f}"
        )
        failures += not same
        if rank == RANKS[0]:
            _, _, old16_s = run_fill(old_fill, args, shard, rows16, layers[pp])
            say(
                f"timing rank {rank}, {opts.threads} thread(s): old fill int32 {old_s:.2f} s, old fill int16 "
                f"{old16_s:.2f} s, new fill int16 {new_s:.2f} s"
            )
    (opts.out / "check-r3-fill-digests.json").write_text(json.dumps(digests, indent=2) + "\n")
    say(f"RESULT: {'PASS' if failures == 0 else 'FAIL'} ({failures} mismatches over {len(RANKS)} ranks)")
    (opts.out / "check-r3-fill.txt").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
