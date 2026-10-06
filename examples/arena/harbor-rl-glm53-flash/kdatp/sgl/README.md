# kdatp/sgl — one-node SGLang engine benchmark of the DSA attention backends

The rollout engines of the GLM-5.3-Flash recipe run SGLang with
`sglang_dsa_prefill_backend: tilelang` and `sglang_dsa_decode_backend: tilelang`.
GLM-5.3-Flash has `index_kpool = 4`, and SGLang 0.5.19 (the image) accepts only
the `fa3`, `tilelang` and `trtllm` DSA backends for a pooled indexer; `fa3` is
Hopper-only, so on B200 the choice is `tilelang` or `trtllm`. SGLang's own
default for bf16 KV on SM100 is `trtllm` decode.

Live engines of `acuadron-agentic-debt-final-v5` (2026-10-06, 201k decode-batch
log samples): median 45 running requests, median context 27.6k tokens, 1,548
generated tok/s per engine, 33.8 tok/s per request (a 30 ms decode step), KV
usage 0.58, 92% radix-cache hits on prefill. The engines are latency-bound,
not request- or memory-bound; the decode step is the lever.

| file | role |
|---|---|
| `sgl_bench.py` | Per arm: launch `sglang.launch_server` with the live engine arguments and the arm's backends, greedy numerics check (fixed prompts, temperature 0, log-probs and routed experts), `sglang.bench_serving` at 45 and 90 concurrent 27k-token requests, decode-log parse. Writes `SUMMARY.md`. |
| `sgl-run.sh` | Driver on the pod; copies the harness from the mount, runs `sgl_bench.py`, records versions and GPU power. |
| `sgl-job.yaml` | PyTorchJob `kdatp-sgl-<stamp>`, one p6-b200 node, 10,800 s deadline. Placeholders `__IMAGE__`, `__STAMP__`, `__ARMS__`. |

Arms (`sgl_bench.py` `ARMS`): `tl-tl` (the live recipe), `tl-trt`, `trt-trt`,
`tl-trt-cutedsl` (`--dsa-paged-mqa-logits-backend cutedsl`, the SM100 indexer
logits kernel).

## Run

```bash
K="kubectl --context arena-prod-bom-v2 -n arena-tasks"
STAMP=<fresh suffix>; IMAGE=<trainer image>; ARMS=""   # or "tl-tl tl-trt"
H=s3://arena-scratch-prod-bom-ap-south-1/acuadron/kdatp/sgl/$STAMP/harness/
aws --profile arena-prod-bom-user s3 cp sgl_bench.py ${H}sgl_bench.py
aws --profile arena-prod-bom-user s3 cp sgl-run.sh ${H}sgl-run.sh
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" -e "s#__ARMS__#$ARMS#" sgl-job.yaml | $K create -f -
# results: s3://arena-scratch-prod-bom-ap-south-1/acuadron/kdatp/sgl/$STAMP/SUMMARY.md
```

Delete the job by hand when it ends. The measures to read: `median TPOT ms` at
c45 (the decode step per request at the live load), `out tok/s` (the engine
rate), the greedy table (equal leading tokens and the log-prob delta on the
common prefix tell whether a backend changes the sampled distribution beyond
bf16 noise), and `startup s`.

## Results

### 2026-10-06a — DSA attention backends (`kdatp-sgl-20261006a`, image r20, one B200 node)

`bench_serving`, random prompts 27,000 in / 2,048 out, `c45` = 45 requests at concurrency 45, `c90` = 90 at 90.
`step ms` is the pure decode step from the server log at 40-50 running requests (no prefill interleaved).

| arm | prefill / decode | startup s | c45 median TPOT ms | c45 out tok/s | c45 mean TTFT s | c90 median TPOT ms | c90 out tok/s | step ms @bs45 |
|---|---|---|---|---|---|---|---|---|
| tl-tl (live recipe) | tilelang / tilelang | 616 | 32.9 | 1,033 | 22.1 | 31.7 | 1,515 | 17.5 |
| tl-trt | tilelang / trtllm | 466 | 27.0 | 1,204 | 21.4 | 30.8 | 1,566 | 16.8 |
| **trt-trt** | trtllm / trtllm | 476 | **25.9** | **1,282** | **19.0** | **25.7** | **1,787** | 16.9 |
| tl-trt-cutedsl | tilelang / trtllm, cutedsl indexer logits | 471 | 26.9 | 1,205 | 21.3 | 26.4 | 1,707 | 16.8 |

The live c45 TPOT of the recipe (33.8 tok/s per request, 30 ms) reproduces (32.9 ms). `trtllm/trtllm` cuts the
TPOT 20% and raises the engine output 18-24%; the pure decode step moves 4%, so the gain is in the prefill
(TTFT -14%) and its interleaving with decode. Live prefill is 92% radix-cached, so expect the live gain
between the two numbers. The job reproduces within 1%: `tl-tl` in job 20261006b gave 32.8 ms / 1,037 tok/s.

Greedy numerics (temperature 0, 160 new tokens; prompts of 512, 4k, 12k, 24k tokens): the trtllm arms share
13-121 leading tokens with the tilelang run and differ by a mean |delta log-prob| of 0.017-0.060 on the common
prefix. **The same tilelang/tilelang server on another node differs from itself by the same amount** (8-47
equal tokens, mean 0.016-0.039; the first-token log-prob of the 512 prompt moves 0.07 between two tilelang
runs), so the backend change stays inside the engine's own run-to-run noise. The train-rollout log-prob gap
of the live run (0.026-0.037) is the metric that decides; measure it in the live A/B.

### 2026-10-06b — the other engine knobs, on the live attention backends (`kdatp-sgl-20261006b`)

Same bench. `tl-tl` reproduces job a (32.8 ms / 1,037 tok/s).

| arm | flags added to the live recipe | startup s | c45 median TPOT ms | c45 out tok/s | c45 mean TTFT s | c90 median TPOT ms | c90 out tok/s | step ms @bs45 | accept len |
|---|---|---|---|---|---|---|---|---|---|
| tl-tl | - | 621 | 32.8 | 1,037 | 21.9 | 32.8 | 1,522 | 17.4 | - |
| **moe-cutlass** | `--moe-runner-backend flashinfer_cutlass` | 501 | **22.5** | **1,461** | **17.0** | **24.2** | **1,977** | **14.5** | - |
| arf | `--enable-flashinfer-allreduce-fusion` | 466 | 27.2 | 1,207 | 20.7 | 28.2 | 1,714 | 17.3 | - |
| ncds2 | `--num-continuous-decode-steps 2` | 471 | 27.5 | 1,188 | 21.3 | 31.0 | 1,674 | 17.4 | - |
| nextn2 | `--speculative-algorithm NEXTN`, 2 steps, 3 draft tokens | 783 | 19.1 | 1,484 | 22.1 | 21.1 | 1,645 | 8.2 | 2.97 of 3 |
| nextn3 | `--speculative-algorithm NEXTN`, 3 steps, 4 draft tokens | 611 | **17.7** | **1,544** | 22.1 | **12.2** | **2,349** | **7.3** | 3.95 of 4 |

- The MoE runner is the largest kernel lever: the live engines run the Triton fused MoE (SGLang 9a26e749
  falls back from `flashinfer_trtllm` to it when routed-experts capture is on). `flashinfer_cutlass`
  materializes the top-k ids like Triton, so the R3 capture stays on: the greedy check returns 160 routing
  rows for 160 output tokens, all ids in range, the three dense layers as zero rows, as with Triton.
- Allreduce fusion and 2 continuous decode steps help only when prefill interleaves with decode (the pure
  decode step does not move); the live prefill is 92% cached, so expect less than the bench shows.
- NEXTN (the model's own MTP layer as the draft) halves the per-token decode time at batch 45 and 90. The
  accept length here is inflated: bench_serving decodes greedily (temperature 0) from random-token prompts,
  and the greedy check uses repeated text; job 20261006c measures it on real code at temperature 1. The
  routing capture under NEXTN is well-formed and aligned: 160 rows for 160 accepted tokens, and the expert
  sets of the common-prefix tokens overlap the non-speculative run as much as any two runs do (7.0-7.4 of 8
  shared experts per layer; a misaligned capture would share ~0.2).
- Greedy numerics of every arm stay inside the tilelang-vs-tilelang run-to-run band (mean |delta log-prob|
  0.015-0.058; the two tilelang runs differ by 0.016-0.039; near-tie first tokens flip in both cases).

### 2026-10-06c — combinations, plus a real-code bench at temperature 1 (`kdatp-sgl-20261006c`)

`real c45` = 45 concurrent requests of 27,000 tokens of real, non-repeated Python source (windows of the
image's own tree), 2,048 output tokens, temperature 1.0; the other columns as before.

| arm | flags | startup s | c45 TPOT ms | c45 out tok/s | c45 TTFT s | c90 TPOT ms | c90 out tok/s | real c45 TPOT ms | real c45 out tok/s | real accept len |
|---|---|---|---|---|---|---|---|---|---|---|
| tl-tl | live recipe | 611 | 33.1 | 1,033 | 21.9 | 28.6 | 1,522 | 29.0 | 1,159 | - |
| combo | trtllm/trtllm + cutlass MoE + allreduce fusion | 501 | 20.6 | 1,638 | 14.2 | 25.1 | 1,950 | 22.9 | 1,518 | - |
| combo-ncds2 | combo + 2 continuous decode steps | 471 | 20.6 | 1,638 | 14.2 | 24.4 | 2,198 | 22.8 | 1,527 | - |
| combo-nextn2 | combo + NEXTN 2 steps / 3 draft tokens | 678 | 14.1 | 2,065 | 14.7 | 19.2 | 1,994 | 14.9 | 1,998 | 2.96 of 3 |
| **combo-nextn3** | combo + NEXTN 3 steps / 4 draft tokens | 611 | **12.8** | **2,206** | 14.6 | **10.4** | **2,742** | **13.5** | **2,106** | 3.93 of 4 |

Chosen for the live A/B (`training-runs/.../acuadron-agentic-debt-final-v6`): `combo-nextn3`. Real-code TPOT
29.0 -> 13.5 ms (-53%), engine output +82% at 45 concurrent; +80% at 90. The 2 continuous decode steps add
nothing at 45 and are left out. The accept length on real code at temperature 1 stays near the maximum; the
live agentic text (model reasoning and commands, not file copies) is the number that matters and the
engines log it (`accept len` in the `Decode batch` lines).

`log: step ms` is not comparable across jobs a-c: it is a median over every decode line of the server log,
and job c adds the temperature-1 phase to the mix. Use the bench columns.
