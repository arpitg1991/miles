"""T1: GLM-5.3 KDA tensor-parallel parity on one node (8 GPUs, TP 8, sequence parallel on).

Run with ``torchrun --nproc-per-node 8``. For each checkpoint and layer, the
script builds the same ``Glm5NextKDAAttention`` twice:

- A: the default module, all 64 heads on each rank.
- B: ``--glm5-next-kda-tp``, 8 heads per rank.

Both load the layer from the same DCP checkpoint. Then the script checks:

- L1 load: each B shard is bitwise equal to its slice of the A tensor.
- L2 keys: B asks for no checkpoint key that A does not ask for, except the
  TE ``_extra_state`` objects. The production strictness
  (``assume_ok_unexpected``) loads B without an error.
- E1 export: the weight-sync gather (``all_gather_params_async``) plus
  ``convert_glm5_next_to_hf`` gives the same HF tensors for A and B. When an
  HF directory is given, they are also bitwise equal to its safetensors.
- F1 forward and B1 backward: the output, the input gradient and each weight
  gradient of B match A. A second A pass gives the noise floor.
- N1 negative control: B with a contiguous conv slice moves the output by
  more than 10%, so F1 can see a wrong conv layout.
- R1 round trip: B saves a DCP. A fresh A and a fresh B load it bitwise.
- T7 timing (optional): forward plus backward time of A and B per length.

Rank 0 writes ``<out>/SUMMARY.json``. The script exits 0 only when all checks pass.
"""

import argparse
import json
import os
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core import parallel_state as mpu
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked

from miles.backends.megatron_utils.megatron_to_hf.glm5_next import convert_glm5_next_to_hf
from miles.backends.megatron_utils.update_weight.common import all_gather_params_async
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.types import ParamInfo
from miles_plugins.models.glm5_next.kda import Glm5NextKDAAttention

KDA_PARAMS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "conv1d.weight",
    "b_proj.weight",
    "f_a_proj.weight",
    "f_b_proj.weight",
    "g_a_proj.weight",
    "g_b_proj.weight",
    "A_log",
    "dt_bias",
    "o_norm.weight",
    "o_proj.weight",
)
# How a full tensor splits into the TP shards: (dim, parts). parts > 1 is the packed conv.
SHARDING = {
    "q_proj.weight": (0, 1),
    "k_proj.weight": (0, 1),
    "v_proj.weight": (0, 1),
    "conv1d.weight": (0, 3),
    "b_proj.weight": (0, 1),
    "f_b_proj.weight": (0, 1),
    "g_b_proj.weight": (0, 1),
    "A_log": (0, 1),
    "dt_bias": (0, 1),
    "o_proj.weight": (1, 1),
}
# Gates (proposed in the kdatp design; not calibrated yet, so the JSON keeps the values).
MAX_REL_L2_OUTPUT = 2e-3
MAX_REL_L2_GRAD = 5e-3
MIN_REL_L2_NEGATIVE = 1e-1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-checkpoint", required=True, help="HF dir: config.json for the model shape")
    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help="name=DCP dir[=HF dir]. The DCP dir is an iter_* or release dir, or a dir with a tracker file. "
        "The optional HF dir holds the same weights as safetensors (E1 compares them bitwise).",
    )
    parser.add_argument("--layers", default="0,44", help="KDA layer numbers (0-based) to check")
    parser.add_argument("--seqlens", default="512,2048,5000,24576", help="packed sequence lengths")
    parser.add_argument("--timing-lengths", default="", help="optional T7 lengths, for example 8192,32768,131072")
    parser.add_argument("--out", required=True, help="output dir; the round-trip DCP goes under it")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def resolve_dcp_dir(path: str) -> Path:
    path = Path(path)
    tracker = path / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        value = tracker.read_text().strip()
        return path / ("release" if value == "release" else f"iter_{int(value):07d}")
    return path


def init_distributed(seed: int) -> None:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mpu.initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    model_parallel_cuda_manual_seed(seed)

    def info(rank, size, group):
        return GroupInfo(rank=rank, size=size, group=group)

    tp = info(
        mpu.get_tensor_model_parallel_rank(),
        mpu.get_tensor_model_parallel_world_size(),
        mpu.get_tensor_model_parallel_group(),
    )
    one = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=one,
            intra_dp_cp=one,
            cp=one,
            tp=tp,
            pp=one,
            ep=one,
            etp=one,
            indep_dp=one,
        )
    )


def make_config(tp_size: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=64,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        layernorm_epsilon=1e-5,
    )


def build(hf_checkpoint: str, config: TransformerConfig, layer: int, kda_tp: bool) -> Glm5NextKDAAttention:
    args = Namespace(hf_checkpoint=hf_checkpoint, sequence_parallel=True, allgather_cp=False, glm5_next_kda_tp=kda_tp)
    module = Glm5NextKDAAttention(args, config, layer_number=layer + 1).cuda()
    # Float16Module does the same cast: bf16 except the tensors marked keep_in_fp32.
    convert_module_to_dtype_except_fp32_marked(module, torch.bfloat16)
    return module


def layer_prefix(layer: int) -> str:
    return f"decoder.layers.{layer}.self_attention."


def load_layer(module: Glm5NextKDAAttention, ckpt_dir: Path, layer: int, strict: StrictHandling):
    """Load one layer from a DCP the way Megatron loads the model (sharded state dict, then load_state_dict)."""
    prefix = layer_prefix(layer)
    sharded = module.sharded_state_dict(prefix=prefix, metadata={"non_homogeneous_layers": True})
    keys = set(sharded)
    result = dist_checkpointing.load(sharded, str(ckpt_dir), strict=strict)
    unexpected = set()
    if strict == StrictHandling.RETURN_UNEXPECTED:
        result, _missing, unexpected = result
    state_dict = {key[len(prefix) :]: value for key, value in result.items() if key in keys}
    incompatible = module.load_state_dict(state_dict, strict=False)
    return keys, sorted(unexpected), sorted(incompatible.missing_keys), sorted(incompatible.unexpected_keys)


def param(module: Glm5NextKDAAttention, name: str) -> torch.Tensor:
    return module.kda.get_parameter(name)


def expected_shard(full: torch.Tensor, name: str, rank: int, tp_size: int) -> torch.Tensor:
    if name not in SHARDING:
        return full
    dim, parts = SHARDING[name]
    return torch.cat([part.chunk(tp_size, dim=dim)[rank] for part in full.chunk(parts, dim=dim)], dim=dim)


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)).item()


def gather_sp(x: torch.Tensor) -> torch.Tensor:
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x.contiguous())
    return torch.cat(parts, dim=0)


def gather_full(module: Glm5NextKDAAttention, layer: int, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Gather tensors laid out like the module params with the production weight-sync gather."""
    infos_and_params = []
    for name in KDA_PARAMS:
        p = param(module, name)
        attrs = {
            "tensor_model_parallel": getattr(p, "tensor_model_parallel", False),
            "partition_dim": getattr(p, "partition_dim", -1),
            "partition_stride": getattr(p, "partition_stride", 1),
            "parallel_mode": getattr(p, "parallel_mode", None),
        }
        full_name = f"module.module.decoder.layers.{layer}.self_attention.kda.{name}"
        info = ParamInfo(
            name=full_name,
            dtype=p.dtype,
            shape=p.shape,
            attrs=attrs,
            size=p.numel() * p.element_size(),
            src_rank=dist.get_rank(),
        )
        # _get_megatron_full_params builds the same fresh Parameter with the ParamInfo attributes.
        copy = torch.nn.Parameter(tensors[name].detach().clone(), requires_grad=False)
        for key, value in attrs.items():
            setattr(copy, key, value)
        infos_and_params.append((info, copy))
    gathered = all_gather_params_async(Namespace(swiglu=True), infos_and_params)
    return {name: tensor for name, tensor in zip(KDA_PARAMS, gathered, strict=True)}


def hf_export(layer: int, full: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for name in KDA_PARAMS:
        megatron_name = f"module.module.decoder.layers.{layer}.self_attention.kda.{name}"
        for hf_name, tensor in convert_glm5_next_to_hf(Namespace(), megatron_name, full[name]):
            out[hf_name] = tensor
    return out


def read_hf(hf_dir: Path, names: list[str]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    weight_map = json.loads((hf_dir / "model.safetensors.index.json").read_text())["weight_map"]
    # The upstream checkpoint may nest the language tower under model.language_model.
    alias = {key.replace("model.language_model.", "model.", 1): key for key in weight_map}
    out = {}
    for name in names:
        key = alias[name]
        with safe_open(str(hf_dir / weight_map[key]), framework="pt") as f:
            out[name] = f.get_tensor(key)
    return out


def packed_inputs(seqlens: list[int], seed: int) -> tuple[torch.Tensor, torch.Tensor, PackedSeqParams]:
    device = torch.device("cuda", torch.cuda.current_device())
    total = sum(seqlens)
    tp = dist.get_world_size()
    assert total % tp == 0, f"packed length {total} is not divisible by tp={tp}"
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(total, 1, 4096, generator=generator, device=device, dtype=torch.bfloat16)
    grad = torch.randn(total, 1, 4096, generator=generator, device=device, dtype=torch.bfloat16) * 1e-2
    cu = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    psp = PackedSeqParams(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max(seqlens),
        max_seqlen_kv=max(seqlens),
        qkv_format="thd",
    )
    return hidden, grad, psp


def forward_backward(module, hidden_full, grad_full, psp):
    """One SP-sharded forward and backward. Returns the gathered output and input gradient."""
    shard = hidden_full.chunk(dist.get_world_size(), dim=0)[dist.get_rank()].clone().requires_grad_(True)
    grad = grad_full.chunk(dist.get_world_size(), dim=0)[dist.get_rank()]
    module.zero_grad(set_to_none=True)
    output, bias = module(shard, attention_mask=None, packed_seq_params=psp)
    assert bias is None
    output.backward(grad)
    return gather_sp(output.detach()), gather_sp(shard.grad)


def tp_reduce_grads(module) -> dict[str, torch.Tensor]:
    """Apply the TP gradient sum of finalize_model_grads (sequence_parallel params) and return the grads."""
    grads = {}
    for name in KDA_PARAMS:
        p = param(module, name)
        grad = p.grad.detach().clone()
        if getattr(p, "sequence_parallel", False):
            dist.all_reduce(grad, group=mpu.get_tensor_model_parallel_group())
        grads[name] = grad
    return grads


def time_fwd_bwd(module, length: int, seed: int) -> float:
    hidden, grad, psp = packed_inputs([length], seed)
    forward_backward(module, hidden, grad, psp)  # warm-up (autotune, JIT)
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    iters = 3
    for _ in range(iters):
        forward_backward(module, hidden, grad, psp)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, ok: bool, **detail) -> None:
        # Every rank must agree before rank 0 records the result.
        flag = torch.tensor([0 if ok else 1], device="cuda")
        dist.all_reduce(flag)
        row = {"check": name, "pass": bool(flag.item() == 0), **detail}
        self.rows.append(row)
        if dist.get_rank() == 0:
            print(json.dumps(row), flush=True)


def check_layer(cli, checks: Checks, ckpt_name: str, ckpt_dir: Path, hf_dir: Path | None, layer: int) -> None:
    config = make_config(dist.get_world_size())
    tp_rank, tp_size = dist.get_rank(), dist.get_world_size()
    tag = f"{ckpt_name}/layer{layer}"
    module_a = build(cli.hf_checkpoint, config, layer, kda_tp=False)
    module_b = build(cli.hf_checkpoint, config, layer, kda_tp=True)

    keys_a, unexpected_a, missing_a, _ = load_layer(module_a, ckpt_dir, layer, StrictHandling.RETURN_UNEXPECTED)
    keys_b, unexpected_b, missing_b, extra_b = load_layer(module_b, ckpt_dir, layer, StrictHandling.RETURN_UNEXPECTED)
    only_extra_state = all(key.endswith("._extra_state") and ".kda." in key for key in unexpected_b)
    checks.add(
        f"{tag}/L2_keys",
        not unexpected_a and only_extra_state and not extra_b and all(k.endswith("_extra_state") for k in missing_b),
        unexpected_a=unexpected_a,
        unexpected_b=unexpected_b,
        missing_after_load_b=missing_b,
        keys_only_in_b=sorted(keys_b - keys_a),
    )
    # The production strictness keeps the unexpected _extra_state objects in the request.
    module_b_prod = build(cli.hf_checkpoint, config, layer, kda_tp=True)
    try:
        load_layer(module_b_prod, ckpt_dir, layer, StrictHandling.ASSUME_OK_UNEXPECTED)
        prod_error = None
    except Exception as exc:  # noqa: BLE001 - the check records the error
        prod_error = repr(exc)
    prod_equal = prod_error is None and all(
        torch.equal(param(module_b_prod, n), param(module_b, n)) for n in KDA_PARAMS
    )
    checks.add(f"{tag}/L2_production_strictness", prod_equal, error=prod_error)
    del module_b_prod

    mismatched = [
        name
        for name in KDA_PARAMS
        if not torch.equal(param(module_b, name), expected_shard(param(module_a, name), name, tp_rank, tp_size))
    ]
    checks.add(f"{tag}/L1_shards", not mismatched, mismatched=mismatched)

    weights_a = {name: param(module_a, name) for name in KDA_PARAMS}
    weights_b = {name: param(module_b, name) for name in KDA_PARAMS}
    export_a = hf_export(layer, gather_full(module_a, layer, weights_a))
    export_b = hf_export(layer, gather_full(module_b, layer, weights_b))
    diff = sorted(n for n in export_a if n not in export_b or not torch.equal(export_a[n], export_b[n]))
    checks.add(f"{tag}/E1_export_a_vs_b", not diff and export_a.keys() == export_b.keys(), mismatched=diff)
    if hf_dir is not None:
        reference = read_hf(hf_dir, sorted(export_b))
        diff = sorted(n for n, t in export_b.items() if not torch.equal(t.to(reference[n].dtype).cpu(), reference[n]))
        checks.add(f"{tag}/E1_export_vs_hf", not diff, hf_dir=str(hf_dir), mismatched=diff)

    seqlens = [int(x) for x in cli.seqlens.split(",")]
    hidden, grad, psp = packed_inputs(seqlens, cli.seed + layer)
    out_a, dx_a = forward_backward(module_a, hidden, grad, psp)
    grads_a = tp_reduce_grads(module_a)
    out_a2, dx_a2 = forward_backward(module_a, hidden, grad, psp)
    out_b, dx_b = forward_backward(module_b, hidden, grad, psp)
    grads_b = gather_full(module_b, layer, tp_reduce_grads(module_b))

    marked = sorted(n for n in KDA_PARAMS if getattr(param(module_b, n), "sequence_parallel", False))
    checks.add(f"{tag}/B1_tp_sum_marks", marked == ["o_norm.weight"], sequence_parallel_params=marked)

    floor = {"output": rel_l2(out_a2, out_a), "dx": rel_l2(dx_a2, dx_a)}
    out_err, dx_err = rel_l2(out_b, out_a), rel_l2(dx_b, dx_a)
    checks.add(
        f"{tag}/F1_forward",
        out_err < MAX_REL_L2_OUTPUT,
        rel_l2=out_err,
        mean_abs=(out_b.float() - out_a.float()).abs().mean().item(),
        max_abs=(out_b.float() - out_a.float()).abs().max().item(),
        noise_floor=floor["output"],
        seqlens=seqlens,
    )
    checks.add(f"{tag}/B1_input_grad", dx_err < MAX_REL_L2_GRAD, rel_l2=dx_err, noise_floor=floor["dx"])
    grad_err = {name: rel_l2(grads_b[name], grads_a[name]) for name in KDA_PARAMS}
    # A norm ratio near 8, 1/8 or sqrt(8) is a missing or doubled TP reduction.
    norm_ratio = {name: (grads_b[name].float().norm() / grads_a[name].float().norm()).item() for name in KDA_PARAMS}
    checks.add(
        f"{tag}/B1_weight_grads",
        all(err < MAX_REL_L2_GRAD for err in grad_err.values()),
        rel_l2=grad_err,
        norm_ratio=norm_ratio,
    )

    # N1: a contiguous conv slice (the wrong layout) must move the output.
    conv = param(module_b, "conv1d.weight")
    saved = conv.detach().clone()
    with torch.no_grad():
        conv.copy_(param(module_a, "conv1d.weight").chunk(tp_size, dim=0)[tp_rank])
    out_wrong, _ = forward_backward(module_b, hidden, grad, psp)
    with torch.no_grad():
        conv.copy_(saved)
    wrong_err = rel_l2(out_wrong, out_a)
    checks.add(f"{tag}/N1_negative_control", wrong_err > MIN_REL_L2_NEGATIVE, rel_l2=wrong_err)

    # R1: B saves; a fresh A (old code) and a fresh B load that save bitwise.
    round_dir = Path(cli.out) / "roundtrip" / f"{ckpt_name}-layer{layer}-{int(time.time())}"
    if dist.get_rank() == 0:
        round_dir.mkdir(parents=True)
    dist.barrier()
    dist_checkpointing.save(
        module_b.sharded_state_dict(prefix=layer_prefix(layer), metadata={"non_homogeneous_layers": True}),
        str(round_dir),
    )
    dist.barrier()
    fresh_a = build(cli.hf_checkpoint, config, layer, kda_tp=False)
    fresh_b = build(cli.hf_checkpoint, config, layer, kda_tp=True)
    load_layer(fresh_a, round_dir, layer, StrictHandling.ASSUME_OK_UNEXPECTED)
    load_layer(fresh_b, round_dir, layer, StrictHandling.ASSUME_OK_UNEXPECTED)
    bad = [n for n in KDA_PARAMS if not torch.equal(param(fresh_a, n), param(module_a, n))]
    bad += [f"tp:{n}" for n in KDA_PARAMS if not torch.equal(param(fresh_b, n), param(module_b, n))]
    checks.add(f"{tag}/R1_round_trip", not bad, mismatched=bad, dir=str(round_dir))

    if cli.timing_lengths:
        timing = {}
        for length in (int(x) for x in cli.timing_lengths.split(",")):
            timing[length] = {
                "replicated_s": time_fwd_bwd(module_a, length, cli.seed),
                "tp_s": time_fwd_bwd(module_b, length, cli.seed),
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        checks.add(f"{tag}/T7_timing", True, seconds=timing)


def main() -> None:
    cli = parse_args()
    init_distributed(cli.seed)
    out = Path(cli.out)
    if dist.get_rank() == 0:
        out.mkdir(parents=True, exist_ok=True)
    checks = Checks()
    for spec in cli.ckpt:
        name, dcp, *hf = spec.split("=")
        for layer in (int(x) for x in cli.layers.split(",")):
            check_layer(cli, checks, name, resolve_dcp_dir(dcp), Path(hf[0]) if hf else None, layer)
            torch.cuda.empty_cache()
    passed = all(row["pass"] for row in checks.rows)
    if dist.get_rank() == 0:
        summary = {"pass": passed, "checks": checks.rows}
        (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"T1 parity: {'PASS' if passed else 'FAIL'} ({sum(r['pass'] for r in checks.rows)}/{len(checks.rows)})")
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
