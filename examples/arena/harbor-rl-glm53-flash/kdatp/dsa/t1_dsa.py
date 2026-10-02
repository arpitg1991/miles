"""kdatp-dsa T1: the DSA sparse attention kernels of miles PR #3608 against the current kernel, one B200.

One process per GPU (``CUDA_VISIBLE_DEVICES=<i>``); ``dsa-run.sh`` starts 8. The process of
``--group g`` runs ``GROUPS[g]``: one head count per rank, one q/kv width and one index set, at
8,192, 65,536 and 131,072 tokens. Batch 1, one KV group, index width 2,112, d_v 512, and the model
softmax scale 256**-0.5 (``glm5_next/dsa.py``). Width 576 is 512 plus the 64 zero columns of
``glm5_next/dsa.py`` (the zero tail, as today). Width 512 is the optional "no zero tail" row. The
seed depends on the index set and the length only, so the 512 and 576 cases get the same values.

Kernels (variants):

- ``old``: ``old_dsa.SparseMLA`` (miles 4716a367a, the r17 image). Width 576 only.
- ``tl``: ``miles.kernels.attention.dsa.sparse_attention(..., forward_backend="tilelang")``.
- ``fmla``: the same with ``forward_backend="flash_mla"``, when ``flash_mla`` imports.

Index sets:

- ``causal``: ``causal_indices`` of the PR bench ``tests/manual/bench_dsa.py``.
- ``kpool``: ``build_pooled_keys`` of ``miles/kernels/attention/dsa/kpool.py`` and
  ``kpool_select_topk`` of ``glm5_next/ops/kpool_indexer.py`` (index_topk 2048, kpool 4,
  32 index heads of 128) on random index tensors, one sequence per row.

Per case and variant: 2 warm-up and 5 timed calls. Each call records CUDA events before the
forward, between the forward and the backward, and after the backward (``torch.autograd.grad``
with a fixed bf16 upstream gradient). Then one more call gives the parity tensors and the peak
memory above the inputs (``max_memory_allocated`` minus ``memory_allocated`` before the call). A
handler on the ``tilelang`` logger counts the TileLang compiles and their time.

Gates. ``rel_diff`` is the PR metric (``dsa_reference.rel_diff``, 1 - 2xy / (x^2 + y^2)); the PR
tables print it. JSON also holds the relative L2 error and the max abs error.

- ``tl_vs_old`` (width 576): out, lse and dq bitwise equal; dkv rel_diff <= 1e-6.
- ``fmla_vs_old`` (width 576): out, dq and dkv rel_diff <= 1e-5. A wrong LSE base shows here as a
  large dq and dkv error.
- ``fmla_vs_tl`` (width 512): out, dq and dkv rel_diff <= 1e-5.
- ``ref`` (8,192 tokens): tl and fmla out rel_diff to the fp32 dense reference <= 2.5e-6.
- ``finite``: out, lse, dq and dkv of each variant hold no NaN and no inf.
- ``no_error``: every variant of the case ran. An out-of-memory error is a result, not a harness bug.
- ``tl_bwd_one_compile`` (per group): the new backward compiles at most once for all lengths.

After round 1, a process with ``--busy`` repeats the timed calls (rounds 2, 3, ...) until the stop
file exists, so that the GPUs stay busy while pytest runs on the pytest GPU.

    python3 t1_dsa.py --group 3 --out <run>/gpu3 --marker <sync>/gpu3.round1 --stop-file <sync>/stop --busy
    python3 t1_dsa.py --merge <run>   # results.json and results.md from the case files
    python3 t1_dsa.py --plan          # the case plan and the kernel imports, no GPU call
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("AGISLIME_DIR", "/root/miles"))

TOKENS = (8192, 65536, 131072)
REF_TOKENS = 8192  # the fp32 dense reference needs [S, H, S] scores; the PR checks it up to 8K
D_V = 512
SM_SCALE = 256**-0.5  # glm5_next/dsa.py: softmax_scale = q_head_dim**-0.5, q_head_dim 256
INDEX_TOPK, KPOOL, INDEX_HEADS, INDEX_DIM = 2048, 4, 32, 128  # GLM-5.3-Flash config.json
INDEX_WIDTH = (INDEX_TOPK + KPOOL - 1 + 63) // 64 * 64  # 2112, kpool_indexer.py out_width
WARMUP, TIMED = 2, 5
# (heads per rank, q/kv width, index set); GPU g runs GROUPS[g]. 8 heads is TP8, 16 heads is TP4.
GROUPS = (
    (8, 576, "causal"),
    (8, 576, "kpool"),
    (16, 576, "causal"),
    (16, 576, "kpool"),
    (8, 512, "causal"),
    (8, 512, "kpool"),
    (16, 512, "causal"),
    (16, 512, "kpool"),
)
TL_DKV_MAX = 1e-6  # PR: dkv differs at the 4e-7 level (fp32 atomic order)
FMLA_MAX = 1e-5  # PR test_forward_backends_agree tolerance
REF_MAX = 2.5e-6  # PR: "within 2e-6" printed with the .0e format, so values below 2.5e-6
COMPILE_RE = re.compile(r"TileLang (begins|completes) to compile kernel `([^`]+)`")
GIB = 2**30


class CompileLog(logging.Handler):
    """Records the TileLang compiles. TileLang logs a begin line and an end line for each kernel
    that misses its disk cache (tilelang/jit/kernel.py, TILELANG_PRINT_ON_COMPILATION, default 1).
    ``label`` names the case and the variant that runs at that time."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.label = "setup"
        self.events = []
        self._start = {}

    def emit(self, record):
        match = COMPILE_RE.search(record.getMessage())
        if match is None:
            return
        verb, kernel = match.groups()
        now = time.perf_counter()
        if verb == "begins":
            self._start[kernel] = now
        else:
            seconds = now - self._start.pop(kernel, now)
            self.events.append({"label": self.label, "kernel": kernel, "s": round(seconds, 3)})

    def of(self, label):
        return [e for e in self.events if e["label"] == label]


def load_kernels():
    """Imports the new kernels as the PR exposes them, the old kernel from the harness dir, the
    glm5_next kpool index path, and the PR bench helpers (``causal_indices``, ``dsa_reference``)."""
    for path in (str(HERE), str(REPO)):  # old_dsa from the harness dir; miles_plugins from the repo
        if path not in sys.path:
            sys.path.append(path)
    import tilelang  # noqa: F401  the handler needs the tilelang logger
    from old_dsa import SparseMLA

    from miles.kernels.attention.dsa import sparse_attention
    from miles.kernels.attention.dsa.sparse_attention import flash_mla_sparse_fwd
    from miles.kernels.attention.dsa.kpool import build_pooled_keys, pool_boundaries
    from miles_plugins.models.glm5_next.ops.kpool_indexer import kpool_select_topk

    spec = importlib.util.spec_from_file_location("bench_dsa", REPO / "tests/manual/bench_dsa.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)  # loads dsa_reference as bench.reference; main() does not run
    return SimpleNamespace(
        sparse_attention=sparse_attention,
        has_fmla=flash_mla_sparse_fwd is not None,
        SparseMLA=SparseMLA,
        build_pooled_keys=build_pooled_keys,
        kpool_select_topk=kpool_select_topk,
        pool_boundaries=pool_boundaries,
        causal_indices=bench.causal_indices,
        reference=bench.reference,
    )


def versions():
    out = {}
    for dist in ("torch", "tilelang", "triton", "flash_mla", "sglang-kernel"):
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = None
    return out


def variants_of(width, has_fmla):
    return (["old"] if width == 576 else []) + ["tl"] + (["fmla"] if has_fmla else [])


def case_name(heads, width, kind, tokens):
    return f"h{heads}-d{width}-{kind}-t{tokens // 1024}k"


def kpool_indices(k, tokens):
    """The index set of glm5_next/dsa.py _kpool_select on random index tensors, one sequence."""
    index_q = torch.randn(tokens, INDEX_HEADS, INDEX_DIM, device="cuda", dtype=torch.bfloat16)
    index_k = torch.randn(tokens, INDEX_DIM, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(tokens, INDEX_DIM, device="cuda", dtype=torch.bfloat16)
    ape = torch.zeros(KPOOL, INDEX_DIM, device="cuda", dtype=torch.float32)
    head_weights = (
        torch.randn(tokens, INDEX_HEADS, device="cuda", dtype=torch.float32) * (INDEX_HEADS * INDEX_DIM) ** -0.5
    )
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    pooled_k = k.build_pooled_keys(index_k, gate, ape, cu_seqlens, KPOOL)
    pool_cu_seqlens = k.pool_boundaries(cu_seqlens, KPOOL)
    idx = k.kpool_select_topk(index_q, pooled_k, head_weights, cu_seqlens, pool_cu_seqlens, INDEX_TOPK, KPOOL)
    return idx.view(1, tokens, 1, -1)


def make_inputs(k, heads, width, kind, tokens):
    """q [1, S, H, width], kv [1, S, 1, width], indices [1, S, 1, 2112] int32, upstream grad [1, S, H, 512]."""
    torch.manual_seed(1000 * (kind == "kpool") + tokens // 1024)
    t0 = time.perf_counter()
    if kind == "causal":
        idx = k.causal_indices(tokens, tokens, INDEX_WIDTH)
    else:
        idx = kpool_indices(k, tokens)
    idx = idx.contiguous()
    assert idx.shape == (1, tokens, 1, INDEX_WIDTH) and idx.dtype == torch.int32, (idx.shape, idx.dtype)
    q = torch.randn(1, tokens, heads, D_V, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, tokens, 1, D_V, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(1, tokens, heads, D_V, device="cuda", dtype=torch.bfloat16)
    if width > D_V:  # the zero tail: glm5_next/dsa.py F.pad(query, (0, 64)) and F.pad(key, (0, 64))
        q = F.pad(q, (0, width - D_V)).contiguous()
        kv = F.pad(kv, (0, width - D_V)).contiguous()
    torch.cuda.synchronize()
    stats = {
        "build_s": round(time.perf_counter() - t0, 3),
        "pad_fraction": (idx == -1).float().mean().item(),
        "max_index": idx.max().item(),
    }
    return (q, kv, idx, grad_out), stats


def forward_fn(k, name, q, kv, idx):
    """Returns the leaves, fwd(read_lse) -> (out, lse), and the map of the [1, S, H, 512] grad to the out layout."""
    # The 4716a367a call: SparseMLA.apply(query [S, H, 576], key [S, 1, 576], indices [S, 1, 2112], scale).
    if name == "old":
        leaves = (q[0].detach().requires_grad_(), kv[0].detach().requires_grad_())
        idx3 = idx[0]

        def fwd(read_lse=False):
            return k.SparseMLA.apply(leaves[0], leaves[1], idx3, SM_SCALE)

        return leaves, fwd, lambda t: t[0]

    backend = {"tl": "tilelang", "fmla": "flash_mla"}[name]
    leaves = (q.detach().requires_grad_(), kv.detach().requires_grad_())

    def fwd(read_lse=False):  # the glm5_next/dsa.py call on the PR branch
        out = k.sparse_attention(leaves[0], leaves[1], idx, SM_SCALE, d_v=D_V, forward_backend=backend)
        lse = None
        if read_lse:  # _SparseAttention saves (q, kv, indices, attn_sink, out, lse); the backward reads this lse
            try:
                lse = out.grad_fn.saved_tensors[5]
            except (AttributeError, IndexError, RuntimeError):
                pass
        return out, lse

    return leaves, fwd, lambda t: t


def call(fwd, leaves, grad):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    events[0].record()
    out, lse = fwd()
    events[1].record()
    dq, dkv = torch.autograd.grad(out, leaves, grad)
    events[2].record()
    torch.cuda.synchronize()
    return (out, lse, dq, dkv), events[0].elapsed_time(events[1]), events[1].elapsed_time(events[2])


def time_calls(fwd, leaves, grad):
    """2 warm-up and 5 timed calls: the timed fwd, bwd and host wall ms, and the wall s of the first call.
    The host wall time of a call includes the final synchronize, so it is at least the event total."""
    fwd_ms, bwd_ms, wall_ms, first_s = [], [], [], None
    for i in range(WARMUP + TIMED):
        t0 = time.perf_counter()
        res, f, b = call(fwd, leaves, grad)
        wall = time.perf_counter() - t0
        del res
        if i == 0:
            first_s = round(wall, 3)
        if i >= WARMUP:
            fwd_ms.append(f)
            bwd_ms.append(b)
            wall_ms.append(wall * 1e3)
    return fwd_ms, bwd_ms, wall_ms, first_s


def ms_stats(values):
    return {
        "mean": round(statistics.fmean(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "all": [round(v, 3) for v in values],
    }


def sha256(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def run_variant(k, log, name, label, inputs):
    """Timing, first-call wall time, compiles, peak memory and the parity tensors of one variant."""
    q, kv, idx, grad_out = inputs
    leaves, fwd, view = forward_fn(k, name, q, kv, idx)
    grad = view(grad_out)
    log.label = label
    fwd_ms, bwd_ms, wall_ms, first_s = time_calls(fwd, leaves, grad)
    rec = {
        "fwd_ms": ms_stats(fwd_ms),
        "bwd_ms": ms_stats(bwd_ms),
        "total_ms": ms_stats([f + b for f, b in zip(fwd_ms, bwd_ms, strict=True)]),
        "wall_ms": ms_stats(wall_ms),
        "first_call_s": first_s,
    }
    # The parity call. The peak window starts after the inputs and the kept tensors of earlier variants.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out, lse = fwd(read_lse=True)
    torch.cuda.synchronize()
    rec["peak_fwd_gib"] = round((torch.cuda.max_memory_allocated() - base) / GIB, 3)
    dq, dkv = torch.autograd.grad(out, leaves, grad)
    torch.cuda.synchronize()
    rec["peak_fwd_bwd_gib"] = round((torch.cuda.max_memory_allocated() - base) / GIB, 3)
    compiles = log.of(label)
    rec["compiles"] = compiles
    rec["compile_count"] = len(compiles)
    rec["compile_s"] = round(sum(e["s"] for e in compiles), 3)
    if name == "old":  # to the new layout [1, S, ...]
        out, lse, dq, dkv = out.unsqueeze(0), lse.unsqueeze(0), dq.unsqueeze(0), dkv.unsqueeze(0)
    tensors = {"out": out.detach(), "lse": None if lse is None else lse.detach(), "dq": dq, "dkv": dkv}
    rec["lse"] = "missing" if lse is None else "saved"
    rec["finite"] = all(bool(torch.isfinite(t).all()) for t in tensors.values() if t is not None)
    rec["out_sha256"] = sha256(tensors["out"])
    rec["dq512_sha256"] = sha256(dq[..., :D_V])
    return rec, tensors


def diff(k, a, b):
    """a against b: bitwise equality, the PR rel_diff, the relative L2 error and the max abs error."""
    a32, b32 = a.float(), b.float()
    err = a32 - b32
    norm = b32.norm().item()
    return {
        "equal": bool(torch.equal(a, b)),
        "rel_diff": k.reference.rel_diff(a, b),
        "rel_l2": err.norm().item() / norm if norm > 0 else 0.0,
        "max_abs": err.abs().max().item(),
    }


def gate(ok, detail):
    return {"pass": ok, "detail": detail}


def compare_case(k, width, tokens, tensors, ref):
    """The comparisons and the gates of one case. A gate is None when a variant did not run."""
    comp, gates = {}, {}
    pairs = [("tl_vs_old", "tl", "old"), ("fmla_vs_old", "fmla", "old"), ("fmla_vs_tl", "fmla", "tl")]
    for key, a, b in pairs:
        if a in tensors and b in tensors:
            comp[key] = {
                n: diff(k, tensors[a][n], tensors[b][n])
                for n in ("out", "lse", "dq", "dkv")
                if tensors[a][n] is not None and tensors[b][n] is not None
            }
    if width == 576:
        c = comp.get("tl_vs_old")
        if c is None:
            gates["tl_vs_old"] = gate(None, "tl or old did not run")
        else:
            bitwise = all(c[n]["equal"] for n in ("out", "lse", "dq") if n in c)
            ok = bitwise and c["dkv"]["rel_diff"] <= TL_DKV_MAX
            detail = f"out/lse/dq bitwise {bitwise}, dkv rel_diff {c['dkv']['rel_diff']:.2e} (max {TL_DKV_MAX:g})"
            if "lse" not in c:  # the new lse was not read from the autograd node
                ok, detail = (None if ok else False), detail + "; lse not compared"
            gates["tl_vs_old"] = gate(ok, detail)
    fmla_key, fmla_base = ("fmla_vs_old", "old") if width == 576 else ("fmla_vs_tl", "tl")
    c = comp.get(fmla_key)
    if c is None:
        gates[fmla_key] = gate(None, f"fmla or {fmla_base} did not run")
    else:
        worst = max(c[n]["rel_diff"] for n in ("out", "dq", "dkv"))
        gates[fmla_key] = gate(
            worst <= FMLA_MAX,
            "rel_diff out {:.2e} dq {:.2e} dkv {:.2e} (max {:g})".format(
                *(c[n]["rel_diff"] for n in ("out", "dq", "dkv")), FMLA_MAX
            ),
        )
    if ref is not None:
        comp["ref"] = {name: k.reference.rel_diff(ref, t["out"]) for name, t in tensors.items()}
        new = {n: v for n, v in comp["ref"].items() if n != "old"}
        gates["ref"] = gate(
            all(v <= REF_MAX for v in new.values()) if new else None,
            " ".join(f"{n} {v:.2e}" for n, v in comp["ref"].items()) + f" (new max {REF_MAX:g})",
        )
    return comp, gates


def free_cuda():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def cuda_ok():
    try:
        torch.cuda.synchronize()
        return True
    except RuntimeError:
        return False


def run_group(k, log, group, out_dir):
    """Round 1: every case of the group. Writes case-<name>.json after each case."""
    heads, width, kind = GROUPS[group]
    names = variants_of(width, k.has_fmla)
    failed = set()
    for tokens in TOKENS:
        name = case_name(heads, width, kind, tokens)
        rec = {
            "case": name,
            "group": group,
            "heads": heads,
            "width": width,
            "indices": kind,
            "tokens": tokens,
            "d_v": D_V,
            "index_width": INDEX_WIDTH,
            "sm_scale": SM_SCALE,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "variants": {},
        }
        t0 = time.perf_counter()
        log.label = f"{name}/inputs"
        inputs, rec["index_stats"] = make_inputs(k, heads, width, kind, tokens)
        ref = None
        if tokens == REF_TOKENS:
            q, kv, idx, _ = inputs
            ref = k.reference.sparse_attention_ref(q, kv, idx, SM_SCALE, D_V)
            free_cuda()
        tensors = {}
        for v in names:
            try:
                rec["variants"][v], tensors[v] = run_variant(k, log, v, f"{name}/{v}", inputs)
                print(
                    f"[t1_dsa] {name} {v}: total {rec['variants'][v]['total_ms']['mean']:.2f} ms, {rec['variants'][v]['compile_count']} compiles",
                    flush=True,
                )
            except Exception as exc:  # a kernel failure is a test result; record it and go on while CUDA is usable
                failed.add(v)
                rec["variants"][v] = {
                    "error": repr(exc),
                    "oom": isinstance(exc, torch.cuda.OutOfMemoryError),
                    "traceback": traceback.format_exc(),
                }
                print(f"[t1_dsa] {name} {v}: ERROR {exc!r}", flush=True)
                if not cuda_ok():
                    rec["fatal"] = "CUDA context broken"
                    (out_dir / f"case-{name}.json").write_text(json.dumps(rec, indent=1))
                    return failed, False
                free_cuda()
        rec["compare"], rec["gates"] = compare_case(k, width, tokens, tensors, ref)
        ran = {v: r for v, r in rec["variants"].items() if "finite" in r}
        errors = [v for v, r in rec["variants"].items() if "error" in r]
        rec["gates"]["finite"] = gate(
            all(r["finite"] for r in ran.values()) if ran else None,
            " ".join(f"{v} {r['finite']}" for v, r in ran.items()),
        )
        rec["gates"]["no_error"] = gate(not errors, f"errors in {errors}" if errors else "all kernels ran")
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        (out_dir / f"case-{name}.json").write_text(json.dumps(rec, indent=1))
        del inputs, tensors, ref
        free_cuda()
    return failed, True


def busy_rounds(k, log, group, out_dir, stop_file, until, skip):
    """Rounds 2, 3, ...: the timed calls again until the stop file exists or the time is up."""

    def stop():
        return stop_file.exists() or time.time() > until

    heads, width, kind = GROUPS[group]
    names = [v for v in variants_of(width, k.has_fmla) if v not in skip]
    rnd = 1
    with open(out_dir / "rounds.jsonl", "a") as fh:
        while not stop():
            rnd += 1
            for tokens in TOKENS:
                if stop():
                    break
                name = case_name(heads, width, kind, tokens)
                inputs, _ = make_inputs(k, heads, width, kind, tokens)
                for v in names:
                    if stop():
                        break
                    log.label = f"{name}/{v}/round{rnd}"
                    q, kv, idx, grad_out = inputs
                    leaves, fwd, view = forward_fn(k, v, q, kv, idx)
                    fwd_ms, bwd_ms, wall_ms, _ = time_calls(fwd, leaves, view(grad_out))
                    del leaves, fwd
                    row = {
                        "round": rnd,
                        "case": name,
                        "variant": v,
                        "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "fwd_ms": round(statistics.fmean(fwd_ms), 3),
                        "bwd_ms": round(statistics.fmean(bwd_ms), 3),
                        "total_ms": round(statistics.fmean([f + b for f, b in zip(fwd_ms, bwd_ms, strict=True)]), 3),
                        "wall_ms": round(statistics.fmean(wall_ms), 3),
                        "compiles": len(log.of(log.label)),
                    }
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                del inputs
                free_cuda()
    return rnd


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    heads, width, kind = GROUPS[args.group]
    group = {
        "group": args.group,
        "heads": heads,
        "width": width,
        "indices": kind,
        "versions": versions(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log = CompileLog()
    t0 = time.perf_counter()
    ok = False
    try:
        k = load_kernels()
        tl_logger = logging.getLogger("tilelang")
        tl_logger.addHandler(log)
        if tl_logger.getEffectiveLevel() > logging.INFO:
            tl_logger.setLevel(logging.INFO)
        group.update(
            gpu=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()),
            has_fmla=k.has_fmla,
            variants=variants_of(width, k.has_fmla),
        )
        failed, ok = run_group(k, log, args.group, args.out)
    except Exception as exc:  # an import or setup failure ends this group; the driver merges the rest
        group["error"] = repr(exc)
        group["traceback"] = traceback.format_exc()
        print(f"[t1_dsa] group {args.group}: ERROR {exc!r}", flush=True)
    group["round1_s"] = round(time.perf_counter() - t0, 3)
    # One compile of the new backward serves every length (PR: T.dynamic batch and lengths).
    # The pod cache starts empty, so zero logged compiles means that the log hook failed.
    bwd = [e for e in log.events if e["kernel"].startswith("sparse_mla_bwd")]
    new_bwd = [e for e in bwd if e["label"].split("/")[1:2] in (["tl"], ["fmla"])]
    old_bwd = [e for e in bwd if e["label"].split("/")[1:2] == ["old"]]
    measured = ok and len(log.events) > 0
    group["compiles"] = log.events
    group["gates"] = {
        "tl_bwd_one_compile": gate(
            len(new_bwd) <= 1 if measured else None,
            f"new backward compiles {len(new_bwd)} for {len(TOKENS)} lengths; old backward compiles {len(old_bwd)}; all TileLang compiles {len(log.events)}",
        )
    }
    (args.out / "group.json").write_text(json.dumps(group, indent=1))
    if args.marker is not None:
        args.marker.parent.mkdir(parents=True, exist_ok=True)
        args.marker.touch()
    print(f"[t1_dsa] group {args.group}: round 1 done in {group['round1_s']:.0f} s, ok {ok}", flush=True)
    if ok and args.busy:
        try:
            rounds = busy_rounds(k, log, args.group, args.out, args.stop_file, args.busy_until or float("inf"), failed)
            print(f"[t1_dsa] group {args.group}: busy rounds end at round {rounds}", flush=True)
        except Exception as exc:  # the round 1 files are complete; a busy-round failure only ends the extra load
            print(f"[t1_dsa] group {args.group}: busy rounds ERROR {exc!r}\n{traceback.format_exc()}", flush=True)
    return 0 if ok else 1


def verdict_of(values):
    if any(v is False for v in values):
        return "FAIL"
    if all(v is True for v in values):
        return "PASS"
    return "NOT RUN" if all(v is None for v in values) else "INCOMPLETE"


def fmt(value, spec=".2f"):
    return "-" if value is None else format(value, spec)


def merge(run_dir):
    """results.json and results.md from gpu*/case-*.json, gpu*/group.json, gpu*/rounds.jsonl and pytest.*."""
    cases = [json.loads(p.read_text()) for p in sorted(run_dir.glob("gpu*/case-*.json"))]
    groups = [json.loads(p.read_text()) for p in sorted(run_dir.glob("gpu*/group.json"))]
    rounds = [
        json.loads(line)
        for p in sorted(run_dir.glob("gpu*/rounds.jsonl"))
        for line in p.read_text().splitlines()
        if line.strip()
    ]
    pytest_rc = (run_dir / "pytest.rc").read_text().strip() if (run_dir / "pytest.rc").exists() else None
    pytest_log = (
        (run_dir / "pytest.log").read_text(errors="replace").splitlines() if (run_dir / "pytest.log").exists() else []
    )
    pytest_summary = next(
        (line for line in reversed(pytest_log) if re.search(r"\d+ (passed|failed|error)", line)), None
    )
    by_key = {(c["heads"], c["width"], c["indices"], c["tokens"]): c for c in cases}

    gate_rows, verdict = [], {}
    for c in cases:
        for g, r in c.get("gates", {}).items():
            gate_rows.append((c["case"], g, r["pass"], r["detail"]))
            verdict.setdefault(g, []).append(r["pass"])
    for grp in groups:
        for g, r in grp.get("gates", {}).items():
            gate_rows.append(
                (f"group {grp['group']} h{grp['heads']}-d{grp['width']}-{grp['indices']}", g, r["pass"], r["detail"])
            )
            verdict.setdefault(g, []).append(r["pass"])
    verdict = {g: verdict_of(vals) for g, vals in verdict.items()}
    missing = [g for g in range(len(GROUPS)) if not any(grp["group"] == g for grp in groups)]

    zero_tail = []
    for (heads, width, kind, tokens), c in sorted(by_key.items()):
        other = by_key.get((heads, 576, kind, tokens))
        if width != 512 or other is None:
            continue
        row = {"heads": heads, "indices": kind, "tokens": tokens}
        for v in ("tl", "fmla"):
            a, b = c["variants"].get(v, {}), other["variants"].get(v, {})
            if "out_sha256" in a and "out_sha256" in b:
                row[v] = {
                    "out_equal": a["out_sha256"] == b["out_sha256"],
                    "dq512_equal": a["dq512_sha256"] == b["dq512_sha256"],
                    "total_ms_576": b["total_ms"]["mean"],
                    "total_ms_512": a["total_ms"]["mean"],
                }
        zero_tail.append(row)

    busy = {}
    for r in rounds:
        busy.setdefault((r["case"], r["variant"]), []).append(r["total_ms"])

    results = {
        "data": "Synthetic kernel inputs only. The agentic-debt dataset of record lakefs://arena-inspect/main/internal/agentic-debt-r3/agentic-debt-766/ is not used.",
        "verdict": verdict,
        "missing_groups": missing,
        "pytest": {"rc": pytest_rc, "summary": pytest_summary},
        "gates": {"tl_dkv_max": TL_DKV_MAX, "fmla_max": FMLA_MAX, "ref_max": REF_MAX},
        "groups": [{k: v for k, v in grp.items() if k != "compiles"} for grp in groups],
        "cases": cases,
        "zero_tail": zero_tail,
        "busy_rounds": {
            f"{c}/{v}": {"n": len(t), "median_total_ms": round(statistics.median(t), 3), "min": min(t), "max": max(t)}
            for (c, v), t in busy.items()
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=1))

    md = ["# kdatp-dsa T1 results", "", results["data"], ""]
    if groups:
        g0 = groups[0]
        md += [f"GPU {g0.get('gpu')}, capability {g0.get('capability')}, versions {g0.get('versions')}.", ""]
    md += ["| Gate | Verdict |", "| --- | --- |"] + [f"| {g} | {v} |" for g, v in verdict.items()]
    md += [f"| pytest `tests/fast-gpu/kernels/attention/dsa` | rc {pytest_rc}: {pytest_summary} |", ""]
    if missing:
        md += [f"Groups without group.json: {missing}.", ""]
    md += ["## Time and memory (round 1, mean of 5 timed calls)", ""]
    md += [
        "| Case | Kernel | fwd ms | bwd ms | fwd+bwd ms | Host wall ms | Speedup vs old 576 | Peak fwd GiB | Peak fwd+bwd GiB | TileLang compiles (n, s) | First call s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (heads, _width, kind, tokens), c in sorted(by_key.items(), key=lambda item: (-item[0][1], *item[0])):
        old = by_key.get((heads, 576, kind, tokens), {}).get("variants", {}).get("old", {})
        for v, r in c["variants"].items():
            if "error" in r:
                error = r["error"][:120].replace("|", "/")
                md.append(f"| {c['case']} | {v} | ERROR {'(OOM) ' if r.get('oom') else ''}{error} | | | | | | | | |")
                continue
            speed = old["total_ms"]["mean"] / r["total_ms"]["mean"] if "total_ms" in old else None
            md.append(
                f"| {c['case']} | {v} | {fmt(r['fwd_ms']['mean'])} | {fmt(r['bwd_ms']['mean'])} | {fmt(r['total_ms']['mean'])} | {fmt(r.get('wall_ms', {}).get('mean'))} | {fmt(speed)}x | {fmt(r['peak_fwd_gib'])} | {fmt(r['peak_fwd_bwd_gib'])} | {r['compile_count']}, {fmt(r['compile_s'], '.1f')} | {fmt(r['first_call_s'], '.1f')} |"
            )
    md += ["", "## Gates per case", "", "| Case | Gate | Pass | Detail |", "| --- | --- | --- | --- |"]
    md += [f"| {c} | {g} | {p} | {d} |" for c, g, p, d in gate_rows]
    if zero_tail:
        md += [
            "",
            "## Zero tail: width 512 against width 576 (same values, same indices)",
            "",
            "| Heads | Indices | Tokens | Kernel | out bitwise | dq[:512] bitwise | fwd+bwd ms 576 | fwd+bwd ms 512 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in zero_tail:
            for v in ("tl", "fmla"):
                if v in row:
                    r = row[v]
                    md.append(
                        f"| {row['heads']} | {row['indices']} | {row['tokens']} | {v} | {r['out_equal']} | {r['dq512_equal']} | {fmt(r['total_ms_576'])} | {fmt(r['total_ms_512'])} |"
                    )
    if busy:
        md += [
            "",
            "## Busy rounds (rounds 2 and later, fwd+bwd ms per round mean)",
            "",
            "| Case / kernel | Rounds | Median | Min | Max |",
            "| --- | --- | --- | --- | --- |",
        ]
        md += [
            f"| {key} | {r['n']} | {fmt(r['median_total_ms'])} | {fmt(r['min'])} | {fmt(r['max'])} |"
            for key, r in results["busy_rounds"].items()
        ]
    (run_dir / "results.md").write_text("\n".join(md) + "\n")
    print(
        json.dumps({"verdict": verdict, "pytest": results["pytest"], "cases": len(cases), "groups": len(groups)}),
        flush=True,
    )
    return 0


def plan():
    """Prints the case plan and checks the kernel imports. No CUDA call."""
    k = load_kernels()
    print(
        f"imports ok: sparse_attention, SparseMLA (old_dsa), kpool_select_topk, causal_indices, sparse_attention_ref; flash_mla {k.has_fmla}"
    )
    print(f"versions {versions()}")
    for g, (heads, width, kind) in enumerate(GROUPS):
        cases = ", ".join(case_name(heads, width, kind, t) for t in TOKENS)
        print(f"gpu {g}: {cases}; kernels {variants_of(width, True)}; fp32 reference at {REF_TOKENS}")
    print(
        f"index width {INDEX_WIDTH}, sm_scale {SM_SCALE}, {WARMUP} warm-up + {TIMED} timed calls; gates tl dkv {TL_DKV_MAX:g}, fmla {FMLA_MAX:g}, ref {REF_MAX:g}"
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", type=int, choices=range(len(GROUPS)), help="the GPU group to run")
    parser.add_argument("--out", type=Path, help="the output dir of the group")
    parser.add_argument("--marker", type=Path, help="touched at the end of round 1")
    parser.add_argument(
        "--stop-file", type=Path, default=Path("/tmp/kdatp-dsa/sync/stop"), help="ends the busy rounds"
    )
    parser.add_argument("--busy", action="store_true", help="repeat the timed calls after round 1")
    parser.add_argument("--busy-until", type=float, default=0.0, help="epoch s; the busy rounds end at this time")
    parser.add_argument("--merge", type=Path, help="merge the case files of this run dir")
    parser.add_argument("--plan", action="store_true", help="print the plan and check the imports")
    args = parser.parse_args(argv)
    if args.plan:
        return plan()
    if args.merge is not None:
        return merge(args.merge)
    if args.group is None or args.out is None:
        parser.error("--group and --out are required")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
