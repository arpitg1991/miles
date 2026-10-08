# Run record: acuadron-agentic-debt-engine-ab-v3 — live A/B of the speculation depth 5/6 plus the fp8 KV cache (r24)

**Status:** Running
**Date:** 2026-10-08
**Argo workflow:** `acuadron-agentic-debt-engine-ab-v3-g5m5j` (submitted 22:22 UTC)
**Base:** `acuadron-agentic-debt-r3par-ab` (16 nodes: 8 actor = DP 2, 8 engines; batch 256; final-v9's dataset; r24).
**Config deltas vs r3par-ab:** `sglang_speculative_num_steps` 3 -> 5, `sglang_speculative_num_draft_tokens` 4 -> 6, `sglang_kv_cache_dtype` bfloat16 -> fp8_e4m3; run names.
**Outcome:** (pending)

## Goal

Step 2 of `kdatp/sgl/ARMS-20261008.md`. The one-node bench (jobs 20261008a/c, the other session's d) left two arms with a gain and numerics
inside the run-to-run band: 5 draft steps / 6 draft tokens (real-code c45: TPOT 13.8 -> 12.1 ms, +3% tok/s, accept 3.73 -> 4.86 of 6)
and the fp8 KV cache (TPOT 12.7 ms, +5%). DP attention is 30-45% slower with every available MoE a2a backend (none, flashinfer) and
its greedy log-probs are off by 0.16-0.26; deepep has no bf16 MoE runner here (cutlass refuses, humming dies at init); fp8 weights
need a modelopt_fp8 checkpoint; top-k 2 drafts are refused by the DSA backend at page size 64. Gate: log-prob gap at steps 0-1 at most
0.045 (r3par-ab: 0.033 / 0.036), accept length above the live 2.6, 0 lost routing refs, decode tok/s at 48 running against r3par-ab's
1,150-1,590. Stop after step 1.
