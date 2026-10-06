"""One-node SGLang benchmark of the DSA attention backends for GLM-5.3-Flash.

Per arm, the script launches ``sglang.launch_server`` with the engine arguments
of the live recipe (training-runs/.../acuadron-agentic-debt-final-v5/miles-config.yaml:
TP 8, EP 8, context 131072, chunked prefill 8192, mem fraction 0.7, bf16 KV,
page 64, routed-experts capture on) and the arm's DSA backends, waits for
``/health``, and runs:

1. greedy numerics: fixed prompts (text from the image tree, cut to 512, 4k,
   12k and 24k tokens), temperature 0, 160 new tokens, ``return_logprob`` and
   ``return_routed_experts``. Arms are compared with the first arm: tokens equal
   before the first divergence, max |delta log-prob| on the common prefix, and
   the share of (token, layer) expert sets that match.
2. ``sglang.bench_serving``: random prompts, 27,000 in / 2,048 out (fixed), 45
   requests at concurrency 45 (the live median running-req and context), then
   90 at 90. Median TPOT is the decode step time per request; output
   throughput is the engine's generation rate.
3. the server log's ``Decode batch`` lines: median gen throughput at 40-50
   running requests and the implied step time.

Writes ``<out>/<arm>/{server.log,greedy.json,bench-c45.jsonl,bench-c90.jsonl}``
and ``<out>/SUMMARY.md``. Lines that start with ``[sgl-bench]`` are the
progress log.
"""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "/mnt/scratch-fast-1a-rw/durable/model-artifacts/external/zai-org/GLM-5.3-Flash-BF16"
PORT = 30001
URL = f"http://127.0.0.1:{PORT}"
PROMPT_SOURCE = "/root/miles/miles/utils/arguments.py"
GREEDY_LENS = (512, 4096, 12000, 24000)
GREEDY_NEW_TOKENS = 160

# name -> (dsa prefill backend, dsa decode backend, extra server args)
ARMS = {
    "tl-tl": ("tilelang", "tilelang", []),  # the live recipe
    "tl-trt": ("tilelang", "trtllm", []),
    "trt-trt": ("trtllm", "trtllm", []),
    "tl-trt-cutedsl": ("tilelang", "trtllm", ["--dsa-paged-mqa-logits-backend", "cutedsl"]),
}

# The live engine arguments that matter for the kernels and the memory layout (sglang_engine.py
# passes the sglang_* keys of the recipe through; the rest are defaults).
SERVER_ARGS = [
    "--model-path", MODEL, "--trust-remote-code",
    "--tp-size", "8", "--ep-size", "8", "--dp-size", "1",
    "--context-length", "131072", "--chunked-prefill-size", "8192", "--max-prefill-tokens", "16384",
    "--mem-fraction-static", "0.7", "--kv-cache-dtype", "bfloat16", "--page-size", "64",
    "--dsa-topk-backend", "sgl-kernel",
    "--enable-return-routed-experts", "--enable-draft-weights-cpu-backup", "--skip-server-warmup",
    "--host", "127.0.0.1", "--port", str(PORT), "--decode-log-interval", "40", "--log-level", "info",
]


def log(msg: str) -> None:
    print(f"[sgl-bench] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def http_json(path: str, payload: dict | None = None, timeout: float = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_health(proc: subprocess.Popen, timeout_s: float) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with rc={proc.returncode}")
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as resp:
                if resp.status == 200:
                    return time.time() - t0
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(5)
    raise TimeoutError(f"server not healthy after {timeout_s} s")


def gpu_mem_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False
    ).stdout
    return sum(int(x) for x in out.split() if x.strip().isdigit())


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=60)
            break
        except subprocess.TimeoutExpired:
            continue
    # The schedulers hold the GPUs a little longer than the launcher.
    t0 = time.time()
    while time.time() - t0 < 180 and gpu_mem_used_mib() > 8 * 2048:
        time.sleep(5)
    log(f"server stopped; gpu mem used {gpu_mem_used_mib()} MiB")


def build_prompts(tokenizer) -> list[dict]:
    text = Path(PROMPT_SOURCE).read_text()
    text = (text + "\n") * 8  # about 400k tokens; each prompt is a prefix
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    prompts = []
    for n in GREEDY_LENS:
        if n > len(ids):
            break
        prompts.append({"tokens": n, "text": tokenizer.decode(ids[:n])})
    return prompts


def routed_experts_of(resp: dict, n_tokens: int):
    """The captured routing as [token][layer][topk] ints, or None if the server did not send it."""
    if not resp.get("meta_info", {}).get("routed_experts"):
        return None
    try:
        from sglang.srt.state_capturer.routed_experts import extract_routed_experts_from_meta_info

        flat = extract_routed_experts_from_meta_info(resp)
        if n_tokens <= 0 or flat.size % n_tokens:
            log(f"routed experts: {flat.size} values for {n_tokens} tokens, not reshaped")
            return None
        per_token = flat.size // n_tokens
        layers = 45 if per_token % 45 == 0 else 1
        return [[list(map(int, layer)) for layer in tok] for tok in flat.reshape(n_tokens, layers, per_token // layers).tolist()]
    except Exception as e:  # noqa: BLE001 - informational only
        log(f"routed experts not decoded: {e!r}")
        return None


def run_greedy(prompts: list[dict]) -> list[dict]:
    results = []
    for p in prompts:
        t0 = time.time()
        resp = http_json(
            "/generate",
            {
                "text": p["text"],
                "sampling_params": {"temperature": 0, "max_new_tokens": GREEDY_NEW_TOKENS, "ignore_eos": True},
                "return_logprob": True,
                "top_logprobs_num": 0,
                "return_routed_experts": True,
            },
            timeout=1800,
        )
        meta = resp["meta_info"]
        out_lp = meta.get("output_token_logprobs") or []
        results.append(
            {
                "prompt_tokens": p["tokens"],
                "seconds": round(time.time() - t0, 2),
                "output_ids": [int(x[1]) for x in out_lp],
                "logprobs": [float(x[0]) for x in out_lp],
                "routed_experts": routed_experts_of(resp, len(out_lp)),
                "completion_tokens": meta.get("completion_tokens"),
            }
        )
        log(f"greedy {p['tokens']} tokens: {len(out_lp)} out tokens in {results[-1]['seconds']} s")
    return results


def compare_greedy(base: list[dict], other: list[dict]) -> list[dict]:
    rows = []
    for b, o in zip(base, other, strict=False):
        n = 0
        for x, y in zip(b["output_ids"], o["output_ids"], strict=False):
            if x != y:
                break
            n += 1
        dlp = [abs(x - y) for x, y in zip(b["logprobs"][:n], o["logprobs"][:n], strict=False)]
        match = None
        if b.get("routed_experts") and o.get("routed_experts"):
            tot = ok = 0
            for tb, to in zip(b["routed_experts"][:n], o["routed_experts"][:n], strict=False):
                for lb, lo in zip(tb, to, strict=False):
                    tot += 1
                    ok += set(lb) == set(lo)
            match = ok / tot if tot else None
        rows.append(
            {
                "prompt_tokens": b["prompt_tokens"],
                "equal_tokens": n,
                "of": min(len(b["output_ids"]), len(o["output_ids"])),
                "max_abs_dlogprob": max(dlp) if dlp else None,
                "mean_abs_dlogprob": statistics.fmean(dlp) if dlp else None,
                "expert_set_match": match,
            }
        )
    return rows


def run_bench(out: Path, concurrency: int, num_prompts: int) -> dict | None:
    f = out / f"bench-c{concurrency}.jsonl"
    cmd = [
        sys.executable, "-m", "sglang.bench_serving", "--backend", "sglang", "--host", "127.0.0.1", "--port", str(PORT),
        "--model", MODEL, "--dataset-name", "random", "--random-input-len", "27000", "--random-output-len", "2048",
        "--random-range-ratio", "1.0", "--num-prompts", str(num_prompts), "--max-concurrency", str(concurrency),
        "--request-rate", "inf", "--seed", "1", "--disable-tqdm", "--output-file", str(f),
    ]
    t0 = time.time()
    with open(out / f"bench-c{concurrency}.log", "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600, check=False).returncode
    log(f"bench c{concurrency}: rc={rc} in {time.time() - t0:.0f} s")
    if not f.exists():
        return None
    lines = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return lines[-1] if lines else None


def parse_decode_log(server_log: Path) -> dict:
    rows = []
    for line in server_log.read_text(errors="replace").splitlines():
        m = re.search(r"Decode batch, #running-req: (\d+),.*gen throughput \(token/s\): ([0-9.]+)", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2))))
    mid = [tps for n, tps in rows if 40 <= n <= 50 and tps > 0]
    hi = [tps for n, tps in rows if 80 <= n <= 92 and tps > 0]
    step = lambda xs, n: (n / statistics.median(xs) * 1000) if xs else None  # noqa: E731
    return {
        "decode_lines": len(rows),
        "tps_bs40_50_median": statistics.median(mid) if mid else None,
        "step_ms_bs45": step(mid, 45),
        "tps_bs80_92_median": statistics.median(hi) if hi else None,
        "step_ms_bs90": step(hi, 90),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--concurrency", nargs="*", type=int, default=[45, 90])
    ap.add_argument("--health-timeout", type=float, default=1800)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompts = build_prompts(tokenizer)
    log(f"greedy prompts: {[p['tokens'] for p in prompts]} tokens")

    summary = {}
    base_greedy = None
    for arm in args.arms:
        prefill, decode, extra = ARMS[arm]
        d = out / arm
        d.mkdir(exist_ok=True)
        res: dict = {"prefill": prefill, "decode": decode, "extra": extra}
        summary[arm] = res
        cmd = [sys.executable, "-m", "sglang.launch_server", *SERVER_ARGS, "--dsa-prefill-backend", prefill,
               "--dsa-decode-backend", decode, *extra]
        log(f"arm {arm}: launch {' '.join(cmd[2:])}")
        server_log = d / "server.log"
        proc = None
        try:
            with open(server_log, "w") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
                res["startup_s"] = round(wait_health(proc, args.health_timeout), 1)
                log(f"arm {arm}: healthy after {res['startup_s']} s")
                greedy = run_greedy(prompts)
                (d / "greedy.json").write_text(json.dumps(greedy))
                if base_greedy is None:
                    base_greedy = greedy
                    res["greedy_vs_first"] = None
                else:
                    res["greedy_vs_first"] = compare_greedy(base_greedy, greedy)
                res["greedy_seconds"] = [g["seconds"] for g in greedy]
                for c in args.concurrency:
                    r = run_bench(d, c, c)
                    if r:
                        res[f"bench_c{c}"] = {
                            k: r.get(k)
                            for k in (
                                "completed", "duration", "output_throughput", "total_throughput", "mean_ttft_ms",
                                "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms", "mean_e2e_latency_ms",
                            )
                        }
        except Exception as e:  # noqa: BLE001 - record the failure and go on to the next arm
            res["error"] = repr(e)
            log(f"arm {arm}: FAILED {e!r}")
        finally:
            stop_server(proc)
            if server_log.exists():
                res["decode_log"] = parse_decode_log(server_log)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        write_summary(out, summary, args.arms)
    return 0


def write_summary(out: Path, summary: dict, arms: list[str]) -> None:
    fmt = lambda v, p=1: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["# SGLang DSA backend benchmark (GLM-5.3-Flash, TP8 EP8, one B200 node)", ""]
    lines.append("| arm | prefill / decode | startup s | c45: median TPOT ms | c45: out tok/s | c45: mean TTFT s | c90: median TPOT ms | c90: out tok/s | log: step ms @bs45 | error |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        r = summary.get(arm)
        if not r:
            continue
        b45, b90, dl = r.get("bench_c45") or {}, r.get("bench_c90") or {}, r.get("decode_log") or {}
        extra = " " + " ".join(r["extra"]) if r.get("extra") else ""
        lines.append(
            f"| {arm} | {r['prefill']} / {r['decode']}{extra} | {fmt(r.get('startup_s'))} | {fmt(b45.get('median_tpot_ms'))} | "
            f"{fmt(b45.get('output_throughput'), 0)} | {fmt((b45.get('mean_ttft_ms') or 0) / 1000 if b45 else None)} | "
            f"{fmt(b90.get('median_tpot_ms'))} | {fmt(b90.get('output_throughput'), 0)} | {fmt(dl.get('step_ms_bs45'))} | {r.get('error', '')} |"
        )
    lines += ["", "## Greedy numerics vs the first arm (temperature 0, 160 new tokens)", ""]
    lines.append("| arm | prompt tokens | equal leading tokens | max abs dlogprob on prefix | mean abs dlogprob | expert-set match |")
    lines.append("|---|---|---|---|---|---|")
    for arm in arms:
        r = summary.get(arm) or {}
        for row in r.get("greedy_vs_first") or []:
            lines.append(
                f"| {arm} | {row['prompt_tokens']} | {row['equal_tokens']}/{row['of']} | {fmt(row['max_abs_dlogprob'], 4)} | "
                f"{fmt(row['mean_abs_dlogprob'], 4)} | {fmt(row['expert_set_match'], 4)} |"
            )
    lines += ["", "Greedy seconds per prompt (512, 4k, 12k, 24k tokens; 160 new tokens, batch 1):", ""]
    for arm in arms:
        r = summary.get(arm) or {}
        if r.get("greedy_seconds"):
            lines.append(f"- {arm}: {r['greedy_seconds']}")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
