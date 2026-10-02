"""T1-QP: GLM-5.3 DSA query-parallel parity and timing on one node (8 GPUs, TP 8, sequence parallel on).

Run with ``torchrun --nproc-per-node 8``. The script builds the same ``Glm5NextDSAAttention`` layer twice
with the same weights:

- A: the default head-parallel core (each rank: all queries on its 8 heads, 64-wide zero tail).
- B: ``--glm5-next-dsa-qp`` (each rank: its sequence-parallel chunk of queries on all 64 heads).

Checks, for each packed-sequence case:

- I1 indices: the top-k token indices of B (gathered over the tensor-parallel ranks) equal those of A
  bit for bit. The indexer runs on the local query chunk in B, so a wrong token id shows here first.
- F1 forward: the layer output of B against A, relative L2 <= ``--max-rel-l2`` (default 1e-2: the
  two cores differ by the FlashMLA / TileLang forward (rel_diff about 2.5e-6, that is a relative L2
  of about 2e-3), the fp32 atomic order of dKV, and bf16 roundings of the all-to-all layout).
- B1 backward: the input gradient and every weight gradient of B against A, the same bound, and a
  weight-gradient norm ratio B/A inside 0.99 to 1.01. The TP-summed gradients (``sequence_parallel``
  params: the duplicated linears and the kv layer-norm weight) are summed before the comparison,
  as ``finalize_model_grads`` does.
- N1 negative control: B with a reversed head-group order after the first all-to-all moves the
  output by a relative L2 above 0.1, so F1 can see a layout bug.
- T7 timing: forward plus backward of A and B per length (one sequence, CUDA events, max over ranks),
  and the peak CUDA memory of each above the inputs.

Rank 0 writes ``<out>/SUMMARY.json`` and ``<out>/SUMMARY.md``. The exit code is 0 only when every check passes.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig

from miles_plugins.models.glm5.glm5 import DSASelfAttentionSubmodules
from miles_plugins.models.glm5_next import dsa as dsa_module
from miles_plugins.models.glm5_next.dsa import Glm5NextDSAAttention

# GLM-5.3-Flash (scripts/models/glm5.3-flash.py and config.json)
HIDDEN, HEADS, Q_LORA, KV_LORA, QK_HEAD_DIM, V_HEAD_DIM = 4096, 64, 1536, 512, 256, 256
INDEX_HEADS, INDEX_DIM, INDEX_TOPK, INDEX_KPOOL = 32, 128, 2048, 4
LAYER_NUMBER = 4  # the first DSA layer (0-based layer 3)
NORM_RATIO_RANGE = (0.99, 1.01)
MIN_REL_L2_NEGATIVE = 1e-1
WARMUP, TIMED = 2, 3
GIB = 2**30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--cases",
        default="4096;1024,3072;3000,5192,16384;8192,24576",
        help="';'-separated packed cases, each a ','-separated list of sequence lengths (total divisible by 32)",
    )
    parser.add_argument("--timing-lengths", default="8192,32768,65536,131072")
    parser.add_argument("--max-rel-l2", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-timing", action="store_true")
    return parser.parse_args()


def init_distributed(seed: int) -> None:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mpu.initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    model_parallel_cuda_manual_seed(seed)


def make_config(tp_size: int) -> MLATransformerConfig:
    config = MLATransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        kv_channels=QK_HEAD_DIM,
        multi_latent_attention=True,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_head_dim=QK_HEAD_DIM,
        qk_pos_emb_head_dim=0,
        v_head_dim=V_HEAD_DIM,
        qk_layernorm=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        rope_type="rope",
        rotary_base=10000,
        add_bias_linear=False,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    # _apply_glm5_next_config (glm5_next.py) sets these from the HF config.
    config.index_num_attention_heads = INDEX_HEADS
    config.index_head_dim = INDEX_DIM
    config.index_topk = INDEX_TOPK
    config.index_kpool = INDEX_KPOOL
    config.glm5_next_full_attn_layers = [i for i in range(45) if i % 4 == 3]
    config.freeze_indexer = False
    return config


def build(config: MLATransformerConfig, query_parallel: bool) -> Glm5NextDSAAttention:
    backend = TESpecProvider()
    spec = ModuleSpec(
        module=Glm5NextDSAAttention,
        params={"attn_mask_type": AttnMaskType.causal, "topk_backend": "torch", "query_parallel": query_parallel},
        submodules=DSASelfAttentionSubmodules(
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_layer_norm_linear(),
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=backend.column_parallel_layer_norm_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
            linear_v_up_proj=IdentityOp,
            wq_b=backend.linear(),
            wk=backend.linear(),
            k_norm=backend.layer_norm(),
            weights_proj=backend.linear(),
        ),
    )
    module = build_module(spec, config=config, layer_number=LAYER_NUMBER).cuda()
    convert_module_to_dtype_except_fp32_marked(module, torch.bfloat16)
    return module


def copy_params(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    src_params = dict(src.named_parameters())
    with torch.no_grad():
        for name, p in dst.named_parameters():
            assert name in src_params and src_params[name].shape == p.shape, name
            p.copy_(src_params[name])
        # The layer uses the kv up-projection norm weight and the index-pool gate; make them non-trivial.
        for module in (src, dst):
            torch.manual_seed(7)
            module.index_kpool_compress_gate.normal_(std=0.02)
            module.index_kpool_compress_ape.normal_(std=0.5)
            module.linear_kv_up_proj.layer_norm_weight.copy_(
                1.0 + 0.1 * torch.randn_like(module.linear_kv_up_proj.layer_norm_weight)
            )


def packed_inputs(seqlens: list[int], seed: int):
    device = torch.device("cuda", torch.cuda.current_device())
    total = sum(seqlens)
    tp = dist.get_world_size()
    assert total % (4 * tp) == 0, f"packed length {total} must be divisible by 4 x tp={tp}"
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(total, 1, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16)
    grad = torch.randn(total, 1, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16) * 1e-2
    cu = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    psp = PackedSeqParams(
        cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(seqlens), max_seqlen_kv=max(seqlens), qkv_format="thd"
    )
    return hidden, grad, psp


class IndexRecorder:
    """Records the top-k indices that ``kpool_select_topk`` returns inside the dsa module."""

    def __init__(self):
        self.calls = []
        self._orig = dsa_module.kpool_select_topk

    def __enter__(self):
        recorder = self

        def wrapped(*args, **kwargs):
            out = recorder._orig(*args, **kwargs)
            recorder.calls.append(out.detach())
            return out

        dsa_module.kpool_select_topk = wrapped
        return self

    def __exit__(self, *exc):
        dsa_module.kpool_select_topk = self._orig


def gather_sp(x: torch.Tensor) -> torch.Tensor:
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x.contiguous())
    return torch.cat(parts, dim=0)


def forward_backward(module, hidden_full, grad_full, psp):
    """One SP-sharded forward and backward. Returns the gathered output, the gathered input gradient and the indices."""
    tp, rank = dist.get_world_size(), dist.get_rank()
    shard = hidden_full.chunk(tp, dim=0)[rank].clone().requires_grad_(True)
    grad = grad_full.chunk(tp, dim=0)[rank]
    module.zero_grad(set_to_none=True)
    with IndexRecorder() as rec:
        output, bias = module(shard, attention_mask=None, packed_seq_params=psp)
    assert bias is None
    output.backward(grad)
    assert len(rec.calls) == 1, len(rec.calls)
    return gather_sp(output.detach()), gather_sp(shard.grad), rec.calls[0]


def reduced_grads(module) -> dict[str, torch.Tensor]:
    """The gradients after the TP sum of finalize_model_grads for the sequence_parallel params."""
    grads = {}
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        grad = p.grad.detach().clone().float()
        if getattr(p, "sequence_parallel", False):
            dist.all_reduce(grad, group=mpu.get_tensor_model_parallel_group())
        grads[name] = grad
    return grads


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)).item()


def max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def all_ranks_true(ok: bool) -> bool:
    t = torch.tensor([1 if ok else 0], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def max_over_ranks(value: float) -> float:
    t = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def run_case(a, b, seqlens, seed, max_rel):
    hidden, grad, psp = packed_inputs(seqlens, seed)
    out_a, din_a, idx_a = forward_backward(a, hidden, grad, psp)
    grads_a = reduced_grads(a)
    out_b, din_b, idx_b = forward_backward(b, hidden, grad, psp)
    grads_b = reduced_grads(b)
    tp = dist.get_world_size()
    total = sum(seqlens)
    # I1: A holds [S, 1, W] on every rank; B holds the rank's chunk [S / tp, 1, W].
    assert idx_a.shape[0] == total and idx_b.shape[0] == total // tp, (idx_a.shape, idx_b.shape)
    idx_b_full = gather_sp(idx_b)
    i1 = all_ranks_true(torch.equal(idx_a, idx_b_full))
    f1 = {"rel_l2": max_over_ranks(rel_l2(out_b, out_a)), "max_abs": max_over_ranks(max_abs(out_b, out_a))}
    f1["ok"] = f1["rel_l2"] <= max_rel
    b1 = {
        "input_grad": {
            "rel_l2": max_over_ranks(rel_l2(din_b, din_a)),
            "max_abs": max_over_ranks(max_abs(din_b, din_a)),
        }
    }
    b1["input_grad"]["ok"] = b1["input_grad"]["rel_l2"] <= max_rel
    weights = {}
    ok_w = True
    for name in sorted(grads_a):
        ga, gb = grads_a[name], grads_b.get(name)
        row = {"present_in_b": gb is not None, "norm_a": ga.norm().item()}
        if gb is None:
            row["ok"] = False
        else:
            ratio = (gb.norm() / ga.norm().clamp_min(1e-30)).item()
            row.update(
                rel_l2=max_over_ranks(rel_l2(gb, ga)),
                max_abs=max_over_ranks(max_abs(gb, ga)),
                norm_ratio_b_over_a=ratio,
                sequence_parallel=bool(getattr(dict(a.named_parameters())[name], "sequence_parallel", False)),
            )
            row["ok"] = all_ranks_true(
                row["rel_l2"] <= max_rel and NORM_RATIO_RANGE[0] <= ratio <= NORM_RATIO_RANGE[1]
            )
        ok_w = ok_w and row["ok"]
        weights[name] = row
    extra_b = sorted(set(grads_b) - set(grads_a))
    b1["weights"] = weights
    b1["grads_only_in_b"] = extra_b
    b1["ok"] = ok_w and not extra_b and b1["input_grad"]["ok"]
    return {"seqlens": seqlens, "I1": {"ok": i1}, "F1": f1, "B1": b1, "ok": i1 and f1["ok"] and b1["ok"]}, (
        hidden,
        grad,
        psp,
        out_a,
    )


def negative_control(b, hidden, grad, psp, out_a):
    """B with the head groups reversed after the first all-to-all: the attention output reaches the wrong w_vc."""
    orig = dsa_module.heads_to_sequence_a2a_output

    def reversed_groups(y):
        return orig(y.flip(0))

    dsa_module.heads_to_sequence_a2a_output = reversed_groups
    try:
        out_wrong, _, _ = forward_backward(b, hidden, grad, psp)
    finally:
        dsa_module.heads_to_sequence_a2a_output = orig
    err = max_over_ranks(rel_l2(out_wrong, out_a))
    return {"rel_l2_vs_a": err, "ok": err > MIN_REL_L2_NEGATIVE}


def timing(a, b, lengths, seed):
    rows = []
    for length in lengths:
        hidden, grad, psp = packed_inputs([length], seed)
        row = {"tokens": length}
        for name, module in (("A_head_parallel", a), ("B_query_parallel", b)):
            times = []
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            for i in range(WARMUP + TIMED):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                dist.barrier()
                start.record()
                forward_backward(module, hidden, grad, psp)
                end.record()
                torch.cuda.synchronize()
                if i >= WARMUP:
                    times.append(start.elapsed_time(end))
            row[name] = {
                "fwd_bwd_ms_mean_max_rank": max_over_ranks(statistics.fmean(times)),
                "fwd_bwd_ms_all_rank0": [round(t, 2) for t in times],
                "peak_gib_above_inputs_max_rank": max_over_ranks((torch.cuda.max_memory_allocated() - base) / GIB),
            }
        row["speedup_b_over_a"] = (
            row["A_head_parallel"]["fwd_bwd_ms_mean_max_rank"] / row["B_query_parallel"]["fwd_bwd_ms_mean_max_rank"]
        )
        rows.append(row)
        del hidden, grad
        torch.cuda.empty_cache()
    return rows


def kernel_bench(lengths, seed):
    """Rank 0: the sparse-attention kernels alone, today's layout (8 heads, all S queries, 576 wide, TileLang
    forward) against the query-parallel layout (64 heads, S / 8 queries of the last chunk, 512 wide, FlashMLA
    or TileLang forward). Indices come from the kpool indexer on random index tensors (one sequence)."""
    from miles.kernels.attention.dsa import sparse_attention
    from miles.kernels.attention.dsa.sparse_attention import flash_mla_sparse_fwd
    from miles_plugins.models.glm5_next.ops.kpool_indexer import build_pooled_keys, kpool_select_topk, pool_boundaries

    tp = dist.get_world_size()
    scale = QK_HEAD_DIM**-0.5
    rows = []
    for tokens in lengths:
        torch.manual_seed(seed + tokens)
        index_q = torch.randn(tokens, INDEX_HEADS, INDEX_DIM, device="cuda", dtype=torch.bfloat16)
        index_k = torch.randn(tokens, INDEX_DIM, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(tokens, INDEX_DIM, device="cuda", dtype=torch.bfloat16)
        ape = torch.zeros(INDEX_KPOOL, INDEX_DIM, device="cuda", dtype=torch.float32)
        weights = (
            torch.randn(tokens, INDEX_HEADS, device="cuda", dtype=torch.float32) * (INDEX_HEADS * INDEX_DIM) ** -0.5
        )
        cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        pooled = build_pooled_keys(index_k, gate, ape, cu, INDEX_KPOOL)
        pool_cu = pool_boundaries(cu, INDEX_KPOOL)
        idx = kpool_select_topk(index_q, pooled, weights, cu, pool_cu, INDEX_TOPK, INDEX_KPOOL)  # [S, 1, W]
        kv = torch.randn(tokens, 1, KV_LORA, device="cuda", dtype=torch.bfloat16)
        chunk = tokens // tp
        variants = {
            "today_h8_allq_d576_tilelang": (8, slice(0, tokens), 576, "tilelang"),
            "qp_h64_chunk_d512_tilelang": (64, slice(tokens - chunk, tokens), 512, "tilelang"),
        }
        if flash_mla_sparse_fwd is not None:
            variants["qp_h64_chunk_d512_flash_mla"] = (64, slice(tokens - chunk, tokens), 512, "flash_mla")
            variants["today_h8_allq_d576_flash_mla"] = (8, slice(0, tokens), 576, "flash_mla")
        row = {"tokens": tokens}
        for name, (heads, qs, width, backend) in variants.items():
            q = torch.randn(qs.stop - qs.start, heads, KV_LORA, device="cuda", dtype=torch.bfloat16)
            kv_w, q_w = kv, q
            if width > KV_LORA:
                q_w = torch.nn.functional.pad(q, (0, width - KV_LORA))
                kv_w = torch.nn.functional.pad(kv, (0, width - KV_LORA))
            q_w = q_w.unsqueeze(0).contiguous().requires_grad_()
            kv_w = kv_w.unsqueeze(0).contiguous().requires_grad_()
            ind = idx[qs].unsqueeze(0).contiguous()
            grad_out = torch.randn(1, qs.stop - qs.start, heads, KV_LORA, device="cuda", dtype=torch.bfloat16)
            fwd_ms, bwd_ms = [], []
            try:
                for i in range(WARMUP + TIMED):
                    e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                    e[0].record()
                    out = sparse_attention(q_w, kv_w, ind, scale, d_v=KV_LORA, forward_backend=backend)
                    e[1].record()
                    torch.autograd.grad(out, (q_w, kv_w), grad_out)
                    e[2].record()
                    torch.cuda.synchronize()
                    if i >= WARMUP:
                        fwd_ms.append(e[0].elapsed_time(e[1]))
                        bwd_ms.append(e[1].elapsed_time(e[2]))
                row[name] = {
                    "queries": qs.stop - qs.start,
                    "heads": heads,
                    "fwd_ms": round(statistics.fmean(fwd_ms), 2),
                    "bwd_ms": round(statistics.fmean(bwd_ms), 2),
                    "finite": bool(torch.isfinite(out).all()),
                }
            except Exception as exc:  # an out-of-memory error or a kernel failure is a result
                row[name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            del q, q_w, kv_w, grad_out
            torch.cuda.empty_cache()
        rows.append(row)
        print(f"[t1-qp] kernels {row}", flush=True)
        del idx, kv, pooled, index_q, index_k, gate
        torch.cuda.empty_cache()
    return rows


def write_summary(out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# T1-QP DSA query-parallel parity: {'PASS' if summary['ok'] else 'FAIL'}", ""]
    lines.append("| case | I1 indices | F1 out rel_l2 | B1 input grad rel_l2 | B1 worst weight rel_l2 | ok |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for case in summary["cases"]:
        worst = max((w["rel_l2"] for w in case["B1"]["weights"].values() if "rel_l2" in w), default=float("nan"))
        lines.append(
            f"| {case['seqlens']} | {case['I1']['ok']} | {case['F1']['rel_l2']:.2e} | "
            f"{case['B1']['input_grad']['rel_l2']:.2e} | {worst:.2e} | {case['ok']} |"
        )
    lines += [
        "",
        f"N1 negative control: rel_l2 {summary['N1']['rel_l2_vs_a']:.3f} (want > {MIN_REL_L2_NEGATIVE}): {summary['N1']['ok']}",
        "",
    ]
    if summary.get("timing"):
        lines.append("| tokens | A fwd+bwd ms | B fwd+bwd ms | speedup | A peak GiB | B peak GiB |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in summary["timing"]:
            lines.append(
                f"| {row['tokens']} | {row['A_head_parallel']['fwd_bwd_ms_mean_max_rank']:.1f} | "
                f"{row['B_query_parallel']['fwd_bwd_ms_mean_max_rank']:.1f} | {row['speedup_b_over_a']:.2f}x | "
                f"{row['A_head_parallel']['peak_gib_above_inputs_max_rank']:.2f} | "
                f"{row['B_query_parallel']['peak_gib_above_inputs_max_rank']:.2f} |"
            )
    if summary.get("kernels"):
        lines += ["", "Kernels alone (rank 0): fwd / bwd ms", ""]
        for row in summary["kernels"]:
            parts = [
                (
                    f"{k}: {v.get('fwd_ms')} / {v.get('bwd_ms')} (q={v.get('queries')})"
                    if "error" not in v
                    else f"{k}: {v['error']}"
                )
                for k, v in row.items()
                if k != "tokens"
            ]
            lines.append(f"- {row['tokens']} tokens: " + "; ".join(parts))
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    init_distributed(args.seed)
    rank, tp = dist.get_rank(), dist.get_world_size()
    config = make_config(tp)
    a = build(config, query_parallel=False)
    b = build(config, query_parallel=True)
    copy_params(a, b)
    summary = {"tp": tp, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cases": []}
    last = None
    for case in args.cases.split(";"):
        seqlens = [int(x) for x in case.split(",")]
        result, last = run_case(a, b, seqlens, args.seed, args.max_rel_l2)
        summary["cases"].append(result)
        if rank == 0:
            print(f"[t1-qp] case {seqlens}: I1 {result['I1']['ok']} F1 {result['F1']} ok={result['ok']}", flush=True)
    summary["N1"] = negative_control(b, *last)
    if rank == 0:
        print(f"[t1-qp] N1 {summary['N1']}", flush=True)
    if not args.skip_timing:
        summary["timing"] = timing(a, b, [int(x) for x in args.timing_lengths.split(",")], args.seed)
        if rank == 0:
            for row in summary["timing"]:
                print(f"[t1-qp] timing {row}", flush=True)
    if not args.skip_timing:
        if rank == 0:
            summary["kernels"] = kernel_bench(
                [int(x) for x in args.timing_lengths.split(",") if int(x) >= 8192], args.seed
            )
        dist.barrier()
    summary["ok"] = all(c["ok"] for c in summary["cases"]) and summary["N1"]["ok"]
    if rank == 0:
        write_summary(Path(args.out), summary)
        print(f"[t1-qp] {'PASS' if summary['ok'] else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
