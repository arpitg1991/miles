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

