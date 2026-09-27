"""T1 gate isolation: which of the two kernel-call changes moves the GLM-5.3 KDA gradients.

The shared layer changes two things in the ``chunk_kda`` call of the r45 trainer: the gate moves into
the kernel, and ``safe_gate`` turns on. For each checkpoint and layer, the script runs the four
combinations on the inputs of ``t1_parity.py`` and compares each with its fp32 reference (the r45 call
with fp32 weights and inputs). ``outside`` is A of ``t1_parity.py`` and ``kernel`` is S, so these two
rows repeat the parity numbers. The script reports numbers only. It has no pass gate.

Run with ``torchrun --nproc-per-node 1``. Rank 0 writes ``<out>/GATE_ISOLATION.json``.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from t1_parity import (
    KDA_PARAMS,
    build_a,
    copy_a,
    forward_backward_a,
    grads_a,
    init_distributed,
    load_a,
    packed_inputs,
    rel_l2,
    resolve_dcp_dir,
)

MODES = ("outside", "outside_safe", "kernel_unsafe", "kernel")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", action="append", required=True, help="name=DCP dir")
    parser.add_argument("--layers", default="0,44")
    parser.add_argument("--seqlens", default="512,2048,5000,24576", help="the t1_parity.py default")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1234, help="the t1_parity.py default")
    cli = parser.parse_args()
    init_distributed(cli.seed)
    seqlens = [int(x) for x in cli.seqlens.split(",")]
    rows = {}
    for spec in cli.ckpt:
        name, dcp = spec.split("=", 1)
        for layer in (int(x) for x in cli.layers.split(",")):
            module_a = build_a()
            load_a(module_a, resolve_dcp_dir(dcp), layer)
            hidden, grad, psp = packed_inputs(seqlens, cli.seed + layer)
            ref = copy_a(module_a, gate="outside", dtype=torch.float32)
            out_ref, dx_ref = forward_backward_a(ref, hidden.float(), grad.float(), psp)
            g_ref = grads_a(ref)
            del ref
            errors = {}
            for mode in MODES:
                module = copy_a(module_a, gate=mode)
                out, dx = forward_backward_a(module, hidden, grad, psp)
                g = grads_a(module)
                errors[mode] = {
                    "output": rel_l2(out, out_ref),
                    "dx": rel_l2(dx, dx_ref),
                    **{n: rel_l2(g[n], g_ref[n]) for n in KDA_PARAMS},
                }
                del module
            rows[f"{name}/layer{layer}"] = errors
            print(json.dumps({f"{name}/layer{layer}": errors}), flush=True)
            del module_a
            torch.cuda.empty_cache()
    Path(cli.out).mkdir(parents=True, exist_ok=True)
    (Path(cli.out) / "GATE_ISOLATION.json").write_text(json.dumps(rows, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
