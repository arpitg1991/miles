"""Torch against flashinfer top-k in the GLM-5.3 kpool indexer, one GPU.

``--miles-dsa-topk-backend flashinfer`` selects the pools with ``flashinfer.top_k`` instead of
``torch.topk``. Both pick the same pools unless logits tie, so the token index sets of a query differ
only on ties. This script runs ``kpool_select_topk`` with both backends on random packed cases and
reports, per case: the share of queries with the same index set, the mean and worst Jaccard distance,
and the time of each backend on the indexer logits. Rank 0 writes ``<out>/SUMMARY.json``.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from miles.kernels.attention.dsa.kpool import build_pooled_keys, pool_boundaries
from miles.kernels.attention.dsa.topk import get_dsa_topk_fn
from miles_plugins.models.glm5_next.ops.kpool_indexer import kpool_select_topk

INDEX_HEADS, INDEX_DIM, INDEX_TOPK, INDEX_KPOOL = 32, 128, 2048, 4
WARMUP, TIMED = 2, 5


def case(seqlens: list[int], seed: int) -> dict:
    device = torch.device("cuda")
    tokens = sum(seqlens)
    g = torch.Generator(device=device).manual_seed(seed)
    index_q = torch.randn(tokens, INDEX_HEADS, INDEX_DIM, generator=g, device=device, dtype=torch.bfloat16)
    index_k = torch.randn(tokens, INDEX_DIM, generator=g, device=device, dtype=torch.bfloat16)
    gate = torch.randn(tokens, INDEX_DIM, generator=g, device=device, dtype=torch.bfloat16)
    ape = torch.zeros(INDEX_KPOOL, INDEX_DIM, device=device, dtype=torch.float32)
    weights = torch.randn(tokens, INDEX_HEADS, generator=g, device=device, dtype=torch.float32) * (INDEX_HEADS * INDEX_DIM) ** -0.5
    cu = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], device=device, dtype=torch.int32)
    pooled = build_pooled_keys(index_k, gate, ape, cu, INDEX_KPOOL)
    pool_cu = pool_boundaries(cu, INDEX_KPOOL)
    out = {"seqlens": seqlens}
    idx = {}
    for backend in ("torch", "flashinfer"):
        times = []
        for i in range(WARMUP + TIMED):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = kpool_select_topk(index_q, pooled, weights, cu, pool_cu, INDEX_TOPK, INDEX_KPOOL, topk_backend=backend)
            torch.cuda.synchronize()
            if i >= WARMUP:
                times.append((time.perf_counter() - t0) * 1000)
        idx[backend] = res.squeeze(1)
        out[backend] = {"select_ms_mean": statistics.fmean(times), "width": res.shape[-1]}
    a, b = idx["torch"], idx["flashinfer"]
    same_rows = 0
    jaccard = []
    for q in range(tokens):
        sa = set(a[q][a[q] >= 0].tolist())
        sb = set(b[q][b[q] >= 0].tolist())
        same_rows += sa == sb
        union = len(sa | sb)
        jaccard.append(1.0 - len(sa & sb) / union if union else 0.0)
    out["queries"] = tokens
    out["same_index_set_share"] = same_rows / tokens
    out["jaccard_distance_mean"] = statistics.fmean(jaccard)
    out["jaccard_distance_max"] = max(jaccard)
    out["bitwise_equal_tensors"] = bool(torch.equal(a, b))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default="4096;1024,3072;3000,5192,16384;8192,24576;65536")
    parser.add_argument("--seed", type=int, default=1234)
    cli = parser.parse_args()
    torch.cuda.set_device(0)
    rows = [case([int(x) for x in c.split(",")], cli.seed) for c in cli.cases.split(";")]
    summary = {"cases": rows, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    for r in rows:
        print(
            f"[topk] {r['seqlens']}: same set {r['same_index_set_share']:.4f}, jaccard mean {r['jaccard_distance_mean']:.2e} "
            f"max {r['jaccard_distance_max']:.2e}, bitwise {r['bitwise_equal_tensors']}, "
            f"torch {r['torch']['select_ms_mean']:.1f} ms, flashinfer {r['flashinfer']['select_ms_mean']:.1f} ms",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
