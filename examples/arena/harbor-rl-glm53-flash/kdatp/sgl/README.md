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

(see the dated sections below)
