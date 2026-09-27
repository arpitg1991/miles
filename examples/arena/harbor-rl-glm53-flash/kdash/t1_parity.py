"""T1: GLM-5.3 KDA on the shared head-sharded layer, parity on one node (8 GPUs, TP 8, sequence parallel on).

Run with ``torchrun --nproc-per-node 8``. For each checkpoint and layer, the script loads the layer
from the same DCP checkpoint three times:

- A: the replicated layer of the r45 trainer (all 64 heads on each rank; the gate computed outside
  ``chunk_kda``). ``tests/fast-gpu/delta_rule_reference.py`` keeps its code, names and shapes.
- S: A with the kernel call of the shared layer (the gate inside ``chunk_kda``, the safe-gate path,
  as SGLang runs GLM-5.3). S against A isolates the kernel call; B against S isolates the TP split.
- B: the production ``Glm5NextKDAAttention`` (8 heads per rank, runtime names ``linear_attn.*``).

Then the script checks:

- L1 load: each B shard is bitwise equal to its slice of the A tensor.
- L2 keys: B asks for exactly the checkpoint keys of A (the 13 KDA tensors, no new key), the load
  returns no unexpected key, and ``load_state_dict(strict=True)`` passes. The production strictness
  (``assume_ok_unexpected``) gives the same weights.
- E1 export: the weight-sync gather (``all_gather_params_async``) plus ``convert_glm5_next_to_hf``
  gives the HF tensors of A. When an HF directory is given, they are bitwise equal to its safetensors.
- F1 forward and B1 backward: the output, the input gradient and each weight gradient of A, S and B
  are compared with an fp32 reference (A with fp32 weights and inputs). B passes when its error is at
  most 2 x the error of A plus one bf16 rounding (2**-8). The row-parallel ``out_proj`` of B sums 8 bf16
  partial outputs, so B is not bitwise equal to A. A weight gradient norm ratio B/A outside 0.99 to 1.01
  is a TP reduction bug (a ratio near 8, 1/8 or 2.83).
- B1 marks: no KDA parameter of B has the ``sequence_parallel`` mark (the TP sums of the replicated
  weights are in the TP copy op, so a mark sums them twice).
- N1 negative control: B with a contiguous slice of the packed conv moves the output by more than 10%.
- R1 round trip: B saves a DCP. A fresh A (old code) and a fresh B load it bitwise.
- T7 timing (optional): forward plus backward time and peak memory of A and B per length.

Rank 0 writes ``<out>/SUMMARY.json``. The script exits 0 only when all checks pass.
"""

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core import parallel_state as mpu
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.utils import sharded_state_dict_default

from miles.backends.megatron_utils.megatron_to_hf.glm5_next import convert_glm5_next_to_hf
from miles.backends.megatron_utils.update_weight.common import all_gather_params_async
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.kernels.attention.delta_rule import DeltaRuleHeads
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.types import ParamInfo
from miles_plugins.models.glm5_next.kda import Glm5NextKDAAttention

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tests" / "fast-gpu"))
from delta_rule_reference import ReplicatedGlm5NextKDA  # noqa: E402

HIDDEN, HEADS, HEAD_DIM, RMS_EPS = 4096, 64, 128, 1e-5
# A (checkpoint) name -> the B names that hold it and the TP split dim of each (None: replicated).
# The packed conv [q; k; v] is held by three convs, one per part.
LAYOUT = {
    "q_proj.weight": [("linear_attn.q_proj.weight", 0)],
    "k_proj.weight": [("linear_attn.k_proj.weight", 0)],
    "v_proj.weight": [("linear_attn.v_proj.weight", 0)],
    "conv1d.weight": [
        ("linear_attn.q_conv1d.weight", 0),
        ("linear_attn.k_conv1d.weight", 0),
        ("linear_attn.v_conv1d.weight", 0),
    ],
    "b_proj.weight": [("linear_attn.b_proj.weight", 0)],
    "f_a_proj.weight": [("linear_attn.f_a_proj.weight", None)],
    "f_b_proj.weight": [("linear_attn.f_b_proj.weight", 0)],
    "g_a_proj.weight": [("linear_attn.g_a_proj.weight", None)],
    "g_b_proj.weight": [("linear_attn.g_b_proj.weight", 0)],
    "A_log": [("linear_attn.A_log", 0)],
    "dt_bias": [("linear_attn.dt_bias", 0)],
    "o_norm.weight": [("linear_attn.norm.weight", None)],
    "o_proj.weight": [("linear_attn.out_proj.weight", 1)],
}
KDA_PARAMS = tuple(LAYOUT)
B_PARAMS = tuple(name for entries in LAYOUT.values() for name, _ in entries)
# F1 and B1 gate: err(B, ref) <= REF_ERR_FACTOR * err(A, ref) + BF16_ROUNDING (relative L2).
REF_ERR_FACTOR = 2.0
BF16_ROUNDING = 2.0**-8
# A missing or doubled TP sum moves a gradient norm by 8, 1/8 or sqrt(8).
NORM_RATIO_RANGE = (0.99, 1.01)
# Used only when the fp32 reference fails (the JSON records the error): B against A.
FALLBACK_MAX_REL_L2 = 1e-2
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
        ParallelState(intra_dp=one, intra_dp_cp=one, cp=one, tp=tp, pp=one, ep=one, etp=one, indep_dp=one)
    )


def make_config(tp_size: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        layernorm_epsilon=RMS_EPS,
    )


def build_a(gate: str = "outside", dtype: torch.dtype = torch.bfloat16) -> ReplicatedGlm5NextKDA:
    heads = DeltaRuleHeads(num_k_heads=HEADS, num_v_heads=HEADS, head_k_dim=HEAD_DIM, head_v_dim=HEAD_DIM)
    return ReplicatedGlm5NextKDA(HIDDEN, heads, dtype, gate=gate, eps=RMS_EPS).cuda()


def copy_a(source: ReplicatedGlm5NextKDA, gate: str, dtype: torch.dtype = torch.bfloat16) -> ReplicatedGlm5NextKDA:
    """A replicated layer with the weights of ``source`` (cast to ``dtype``; A_log and dt_bias stay fp32)."""
    module = build_a(gate, dtype)
    with torch.no_grad():
        for name in KDA_PARAMS:
            target = module.get_parameter(name)
            target.copy_(source.get_parameter(name).to(target.dtype))
    return module


def build_b(hf_checkpoint: str, config: TransformerConfig, layer: int) -> Glm5NextKDAAttention:
    args = Namespace(hf_checkpoint=hf_checkpoint, allgather_cp=False)
    module = Glm5NextKDAAttention(args, config, layer_number=layer + 1).cuda()
    # Float16Module does the same cast: bf16 except the tensors marked keep_in_fp32.
    convert_module_to_dtype_except_fp32_marked(module, torch.bfloat16)
    for p in module.parameters():  # Megatron's get_model sets the TP defaults on every parameter
        set_defaults_if_not_set_tensor_model_parallel_attributes(p)
    return module


def layer_prefix(layer: int) -> str:
    return f"decoder.layers.{layer}.self_attention."


def sharded_a(module: ReplicatedGlm5NextKDA, layer: int) -> dict:
    """The old replicated layer saved each tensor as one chunk under ``self_attention.kda.``."""
    return sharded_state_dict_default(module, prefix=f"{layer_prefix(layer)}kda.")


def sharded_b(module: Glm5NextKDAAttention, layer: int) -> dict:
    return module.sharded_state_dict(prefix=layer_prefix(layer), metadata={"non_homogeneous_layers": True})


def load(module, sharded: dict, prefix: str, ckpt_dir: Path, strict: StrictHandling):
    """Load one layer from a DCP the way Megatron loads the model (sharded state dict, then load_state_dict)."""
    keys = sorted({sh.key for sh in sharded.values()})
    result = dist_checkpointing.load(sharded, str(ckpt_dir), strict=strict)
    unexpected = set()
    if strict == StrictHandling.RETURN_UNEXPECTED:
        result, _missing, unexpected = result
    module.load_state_dict({key[len(prefix) :]: value for key, value in result.items()}, strict=True)
    return keys, sorted(unexpected)


def load_a(module, ckpt_dir: Path, layer: int, strict: StrictHandling = StrictHandling.ASSUME_OK_UNEXPECTED):
    return load(module, sharded_a(module, layer), f"{layer_prefix(layer)}kda.", ckpt_dir, strict)


def load_b(module, ckpt_dir: Path, layer: int, strict: StrictHandling = StrictHandling.ASSUME_OK_UNEXPECTED):
    return load(module, sharded_b(module, layer), layer_prefix(layer), ckpt_dir, strict)


def expected_shards(full: torch.Tensor, name: str, rank: int, tp_size: int) -> dict[str, torch.Tensor]:
    """The B tensors of rank ``rank`` that hold the A tensor ``name``."""
    entries = LAYOUT[name]
    parts = full.chunk(len(entries), dim=0) if len(entries) > 1 else [full]
    return {
        b_name: part if dim is None else part.chunk(tp_size, dim=dim)[rank]
        for (b_name, dim), part in zip(entries, parts, strict=True)
    }


def to_a_layout(full_b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Full-layout B tensors keyed by B name -> A tensors keyed by A name."""
    return {name: torch.cat([full_b[b_name] for b_name, _ in entries]) for name, entries in LAYOUT.items()}


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)).item()


def max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def compare(a: torch.Tensor, s: torch.Tensor, b: torch.Tensor, ref: torch.Tensor | None) -> dict:
    """Errors of A, S and B against the fp32 reference, and of B and S against A. ``ok`` is the F1/B1 gate."""
    row = {
        "rel_l2_b_vs_a": rel_l2(b, a),
        "rel_l2_s_vs_a": rel_l2(s, a),
        "rel_l2_b_vs_s": rel_l2(b, s),
        "max_abs_b_vs_a": max_abs(b, a),
        "max_abs_a": a.float().abs().max().item(),
    }
    if ref is None:
        row["ok"] = row["rel_l2_b_vs_a"] < FALLBACK_MAX_REL_L2
        return row
    row.update(
        rel_l2_a_vs_ref=rel_l2(a, ref),
        rel_l2_s_vs_ref=rel_l2(s, ref),
        rel_l2_b_vs_ref=rel_l2(b, ref),
        max_abs_a_vs_ref=max_abs(a, ref),
        max_abs_b_vs_ref=max_abs(b, ref),
    )
    row["ok"] = row["rel_l2_b_vs_ref"] <= REF_ERR_FACTOR * row["rel_l2_a_vs_ref"] + BF16_ROUNDING
    return row


def gather_sp(x: torch.Tensor) -> torch.Tensor:
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x.contiguous())
    return torch.cat(parts, dim=0)


def gather_b(module: Glm5NextKDAAttention, layer: int, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Gather tensors laid out like the B params with the production weight-sync gather."""
    infos_and_params = []
    for name in B_PARAMS:
        p = module.get_parameter(name)
        attrs = {
            "tensor_model_parallel": p.tensor_model_parallel,
            "partition_dim": p.partition_dim,
            "partition_stride": p.partition_stride,
            "parallel_mode": getattr(p, "parallel_mode", None),
        }
        info = ParamInfo(
            name=f"module.module.decoder.layers.{layer}.self_attention.{name}",
            dtype=p.dtype,
            shape=p.shape,
            attrs=attrs,
            size=p.numel() * p.element_size(),
            src_rank=dist.get_rank(),
        )
        # HfWeightIteratorDirect builds the same fresh Parameter with the ParamInfo attributes.
        copy = torch.nn.Parameter(tensors[name].detach().clone(), requires_grad=False)
        for key, value in attrs.items():
            setattr(copy, key, value)
        infos_and_params.append((info, copy))
    gathered = all_gather_params_async(Namespace(swiglu=True), infos_and_params)
    return dict(zip(B_PARAMS, gathered, strict=True))


def hf_export_b(layer: int, full_b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for name in B_PARAMS:
        megatron_name = f"module.module.decoder.layers.{layer}.self_attention.{name}"
        for hf_name, tensor in convert_glm5_next_to_hf(Namespace(), megatron_name, full_b[name]):
            assert hf_name not in out, hf_name
            out[hf_name] = tensor
    return out


def hf_export_a(layer: int, module: ReplicatedGlm5NextKDA) -> dict[str, torch.Tensor]:
    """What the old converter gave for the replicated layer: the names as is, the packed conv split in 3."""
    out = {}
    for name in KDA_PARAMS:
        tensor = module.get_parameter(name).detach()
        if name == "conv1d.weight":
            for part, hf_name in zip(tensor.chunk(3), ("q_conv1d", "k_conv1d", "v_conv1d"), strict=True):
                out[f"model.layers.{layer}.self_attn.{hf_name}.weight"] = part
        else:
            out[f"model.layers.{layer}.self_attn.{name}"] = tensor
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
    hidden = torch.randn(total, 1, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16)
    grad = torch.randn(total, 1, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16) * 1e-2
    cu = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    psp = PackedSeqParams(
        cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(seqlens), max_seqlen_kv=max(seqlens), qkv_format="thd"
    )
    return hidden, grad, psp


def forward_backward_a(module, hidden_full, grad_full, psp):
    """The replicated layer on the full sequence ([s, b, h] in and out, like the old HuggingfaceAttention)."""
    x = hidden_full.transpose(0, 1).clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    output = module(x, psp.cu_seqlens_q)
    output.backward(grad_full.transpose(0, 1))
    return output.detach().transpose(0, 1), x.grad.transpose(0, 1)


def forward_backward_b(module, hidden_full, grad_full, psp):
    """One SP-sharded forward and backward. Returns the gathered output and input gradient."""
    shard = hidden_full.chunk(dist.get_world_size(), dim=0)[dist.get_rank()].clone().requires_grad_(True)
    grad = grad_full.chunk(dist.get_world_size(), dim=0)[dist.get_rank()]
    module.zero_grad(set_to_none=True)
    output, bias = module(shard, attention_mask=None, packed_seq_params=psp)
    assert bias is None
    output.backward(grad)
    return gather_sp(output.detach()), gather_sp(shard.grad)


def grads_a(module) -> dict[str, torch.Tensor]:
    return {name: module.get_parameter(name).grad.detach().clone() for name in KDA_PARAMS}


def grads_b(module, layer: int) -> dict[str, torch.Tensor]:
    """B's gradients in the A layout. The TP copy op already summed the replicated ones across TP."""
    local = {name: module.get_parameter(name).grad.detach().clone() for name in B_PARAMS}
    return to_a_layout(gather_b(module, layer, local))


def timed(fn, *args) -> tuple[float, float]:
    fn(*args)  # warm-up (autotune, JIT)
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    iters = 3
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters, torch.cuda.max_memory_allocated() / 2**30


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
    module_a = build_a()
    module_b = build_b(cli.hf_checkpoint, config, layer)

    keys_a, unexpected_a = load_a(module_a, ckpt_dir, layer, StrictHandling.RETURN_UNEXPECTED)
    keys_b, unexpected_b = load_b(module_b, ckpt_dir, layer, StrictHandling.RETURN_UNEXPECTED)
    checks.add(
        f"{tag}/L2_keys",
        keys_a == keys_b and len(keys_b) == len(KDA_PARAMS) and not unexpected_a and not unexpected_b,
        keys_b=keys_b,
        keys_only_in_a=sorted(set(keys_a) - set(keys_b)),
        keys_only_in_b=sorted(set(keys_b) - set(keys_a)),
        unexpected_a=unexpected_a,
        unexpected_b=unexpected_b,
    )
    # The production strictness.
    module_b_prod = build_b(cli.hf_checkpoint, config, layer)
    try:
        load_b(module_b_prod, ckpt_dir, layer)
        prod_error = None
    except Exception as exc:  # noqa: BLE001 - the check records the error
        prod_error = repr(exc)
    prod_equal = prod_error is None and all(
        torch.equal(module_b_prod.get_parameter(n), module_b.get_parameter(n)) for n in B_PARAMS
    )
    checks.add(f"{tag}/L2_production_strictness", prod_equal, error=prod_error)
    del module_b_prod

    mismatched = [
        b_name
        for name in KDA_PARAMS
        for b_name, want in expected_shards(module_a.get_parameter(name), name, tp_rank, tp_size).items()
        if not torch.equal(module_b.get_parameter(b_name), want)
    ]
    checks.add(f"{tag}/L1_shards", not mismatched, mismatched=mismatched)

    export_a = hf_export_a(layer, module_a)
    export_b = hf_export_b(layer, gather_b(module_b, layer, {n: module_b.get_parameter(n) for n in B_PARAMS}))
    diff = sorted(n for n in export_a if n not in export_b or not torch.equal(export_a[n], export_b[n]))
    checks.add(f"{tag}/E1_export_a_vs_b", not diff and export_a.keys() == export_b.keys(), mismatched=diff)
    if hf_dir is not None:
        reference = read_hf(hf_dir, sorted(export_b))
        diff = sorted(n for n, t in export_b.items() if not torch.equal(t.to(reference[n].dtype).cpu(), reference[n]))
        checks.add(f"{tag}/E1_export_vs_hf", not diff, hf_dir=str(hf_dir), mismatched=diff)

    marked = sorted(n for n in B_PARAMS if getattr(module_b.get_parameter(n), "sequence_parallel", False))
    split = {n: bool(module_b.get_parameter(n).tensor_model_parallel) for n in B_PARAMS}
    want_split = {b_name: dim is not None for entries in LAYOUT.values() for b_name, dim in entries}
    checks.add(f"{tag}/B1_tp_marks", not marked and split == want_split, sequence_parallel=marked, split=split)

    seqlens = [int(x) for x in cli.seqlens.split(",")]
    hidden, grad, psp = packed_inputs(seqlens, cli.seed + layer)
    module_s = copy_a(module_a, gate="kernel")
    out_a, dx_a = forward_backward_a(module_a, hidden, grad, psp)
    g_a = grads_a(module_a)
    out_a2, dx_a2 = forward_backward_a(module_a, hidden, grad, psp)
    out_s, dx_s = forward_backward_a(module_s, hidden, grad, psp)
    g_s = grads_a(module_s)
    del module_s
    out_b, dx_b = forward_backward_b(module_b, hidden, grad, psp)
    g_b = grads_b(module_b, layer)

    # The fp32 reference: A with fp32 weights, inputs and output gradient.
    try:
        module_ref = copy_a(module_a, gate="outside", dtype=torch.float32)
        out_ref, dx_ref = forward_backward_a(module_ref, hidden.float(), grad.float(), psp)
        g_ref = grads_a(module_ref)
        ref_error = None
        del module_ref
    except Exception as exc:  # noqa: BLE001 - the check falls back to B against A and records the error
        out_ref = dx_ref = None
        g_ref = dict.fromkeys(KDA_PARAMS)
        ref_error = repr(exc)
    ref_failed = torch.tensor([0 if ref_error is None else 1], device="cuda")
    dist.all_reduce(ref_failed)
    if ref_failed.item():  # every rank MUST use the same gate
        out_ref = dx_ref = None
        g_ref = dict.fromkeys(KDA_PARAMS)

    floor = {"output": rel_l2(out_a2, out_a), "dx": rel_l2(dx_a2, dx_a)}
    out_cmp = compare(out_a, out_s, out_b, out_ref)
    checks.add(
        f"{tag}/F1_forward",
        out_cmp.pop("ok"),
        **out_cmp,
        mean_abs_b_vs_a=(out_b.float() - out_a.float()).abs().mean().item(),
        a_vs_a=floor["output"],
        reference="fp32" if out_ref is not None else f"none: {ref_error}",
        seqlens=seqlens,
    )
    dx_cmp = compare(dx_a, dx_s, dx_b, dx_ref)
    checks.add(f"{tag}/B1_input_grad", dx_cmp.pop("ok"), **dx_cmp, a_vs_a=floor["dx"])
    grad_cmp = {name: compare(g_a[name], g_s[name], g_b[name], g_ref[name]) for name in KDA_PARAMS}
    norm_ratio = {name: (g_b[name].float().norm() / g_a[name].float().norm()).item() for name in KDA_PARAMS}
    low, high = NORM_RATIO_RANGE
    grads_ok = [row.pop("ok") for row in grad_cmp.values()]  # a list: every row loses its "ok" key
    checks.add(
        f"{tag}/B1_weight_grads",
        all(grads_ok) and all(low <= r <= high for r in norm_ratio.values()),
        errors=grad_cmp,
        norm_ratio=norm_ratio,
    )

    # N1: a contiguous slice of the packed conv (the wrong layout) must move the output.
    convs = [module_b.get_parameter(b_name) for b_name, _ in LAYOUT["conv1d.weight"]]
    saved = [conv.detach().clone() for conv in convs]
    with torch.no_grad():
        wrong = module_a.get_parameter("conv1d.weight").chunk(tp_size)[tp_rank].chunk(3)
        for conv, part in zip(convs, wrong, strict=True):
            conv.copy_(part)
    out_wrong, _ = forward_backward_b(module_b, hidden, grad, psp)
    with torch.no_grad():
        for conv, part in zip(convs, saved, strict=True):
            conv.copy_(part)
    wrong_err = rel_l2(out_wrong, out_a)
    checks.add(f"{tag}/N1_negative_control", wrong_err > MIN_REL_L2_NEGATIVE, rel_l2=wrong_err)

    # R1: B saves; a fresh A (old code) and a fresh B load that save bitwise.
    round_dir = Path(cli.out) / "roundtrip" / f"{ckpt_name}-layer{layer}-{int(time.time())}"
    if dist.get_rank() == 0:
        round_dir.mkdir(parents=True)
    dist.barrier()
    dist_checkpointing.save(sharded_b(module_b, layer), str(round_dir))
    dist.barrier()
    fresh_a = build_a()
    fresh_b = build_b(cli.hf_checkpoint, config, layer)
    load_a(fresh_a, round_dir, layer)
    load_b(fresh_b, round_dir, layer)
    bad = [n for n in KDA_PARAMS if not torch.equal(fresh_a.get_parameter(n), module_a.get_parameter(n))]
    bad += [f"b:{n}" for n in B_PARAMS if not torch.equal(fresh_b.get_parameter(n), module_b.get_parameter(n))]
    checks.add(f"{tag}/R1_round_trip", not bad, mismatched=bad, dir=str(round_dir))
    del fresh_a, fresh_b

    if cli.timing_lengths:
        timing = {}
        for length in (int(x) for x in cli.timing_lengths.split(",")):
            hidden_t, grad_t, psp_t = packed_inputs([length], cli.seed)
            a_s, a_gib = timed(forward_backward_a, module_a, hidden_t, grad_t, psp_t)
            b_s, b_gib = timed(forward_backward_b, module_b, hidden_t, grad_t, psp_t)
            timing[length] = {
                "replicated_s": a_s,
                "shared_s": b_s,
                "replicated_peak_gib": a_gib,
                "shared_peak_gib": b_gib,
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
