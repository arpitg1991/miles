# Run record: acuadron-agentic-debt-engine-ab-v3 — live A/B of the speculation depth 5/6 plus the fp8 KV cache (r24)

**Status:** Stopped 2026-10-09 00:06 UTC after train step 1 (gate passed).
**Date:** 2026-10-08
**Argo workflow:** `acuadron-agentic-debt-engine-ab-v3-g5m5j` (submitted 22:22 UTC)
**Base:** `acuadron-agentic-debt-r3par-ab` (16 nodes: 8 actor = DP 2, 8 engines; batch 256; final-v9's dataset; r24).
**Config deltas vs r3par-ab:** `sglang_speculative_num_steps` 3 -> 5, `sglang_speculative_num_draft_tokens` 4 -> 6, `sglang_kv_cache_dtype` bfloat16 -> fp8_e4m3; run names.
**Outcome:** PASS, and far above the bench. Engines at the same load as r3par-ab (40-55 running requests, about 55 queued, 8 engines x 128 trials): decode 2,325 tok/s median against 1,579 (+47%; mean 2,397 vs 1,772), accept length 3.09 of 6 against 2.68 of 4, KV usage 0.29 against 0.52 over the matched 55-min windows of rollout 0. Rollout 0 54.7 min against 91.8 (first group 27.7 vs 39 min); step 1 1,733 s against 4,170 s. Numerics: log-prob gap 0.0220 / 0.0240 at steps 0 / 1 (r3par-ab 0.0330 / 0.0356), train-rollout KL 0.0021 / 0.0023, TIS clip fraction 0.0002 / 0.0003, grad_norm 0.087 / 0.081; 0 groups lost to a routing ref, no `RoutingReplayError`, 32-group materialization 32.5 s. The live gain exceeds the bench's +3-5% because the live engines are KV-bound (long sessions, 48 running + a queue): fp8 KV halves the footprint and the deeper draft adds 15% accepted tokens per step.

## Goal

Step 2 of `kdatp/sgl/ARMS-20261008.md`. The one-node bench (jobs 20261008a/c, the other session's d) left two arms with a gain and numerics
inside the run-to-run band: 5 draft steps / 6 draft tokens (real-code c45: TPOT 13.8 -> 12.1 ms, +3% tok/s, accept 3.73 -> 4.86 of 6)
and the fp8 KV cache (TPOT 12.7 ms, +5%). DP attention is 30-45% slower with every available MoE a2a backend (none, flashinfer) and
its greedy log-probs are off by 0.16-0.26; deepep has no bf16 MoE runner here (cutlass refuses, humming dies at init); fp8 weights
need a modelopt_fp8 checkpoint; top-k 2 drafts are refused by the DSA backend at page size 64. Gate: log-prob gap at steps 0-1 at most
0.045 (r3par-ab: 0.033 / 0.036), accept length above the live 2.6, 0 lost routing refs, decode tok/s at 48 running against r3par-ab's
1,150-1,590. Stop after step 1.

## Timeline

| time (UTC) | event |
|---|---|
| 2026-10-08 22:22 | Submitted. Engines ready 22:34 with `kv_cache_dtype='fp8_e4m3'`, `speculative_num_steps=5`, `speculative_num_draft_tokens=6`. |
| 22:34 -> 23:29 | Rollout 0: 32 groups in 54.7 min (r3par-ab 91.8); first group 23:02. |
| 23:38 | Step 0: `actor_train` 393 s; gap 0.0220, KL 0.0021, grad_norm 0.087. |
| 23:38 -> 00:03 | Rollout 1: 24 min; materialized 32 groups in 32.5 s, 0 lost refs. |
| 00:05 | Step 1: `actor_train` 350 s at 70.4k tok/s; gap 0.0240. Stopped. |

## For final-v10 (step 3 of ARMS-20261008.md)

Edit the three existing keys in `miles-config` in place: `sglang_speculative_num_steps: 5`, `sglang_speculative_num_draft_tokens: 6`,
`sglang_kv_cache_dtype: fp8_e4m3`. Optional, untested live: `sglang_max_running_requests: 96` (the fp8 KV leaves the pool at 0.3, and
final-v10's engines run 47 of 48 with 11 queued; the bench's c90 is 10-20% above c45).
