# Run record: acuadron-agentic-debt-engine-ab-v1 — the SGLang engine flags of kdatp/sgl (trtllm DSA, cutlass MoE, allreduce fusion, NEXTN) in the full loop, 24 nodes

**Status:** Stopped 2026-10-06 ~20:25 UTC after step 20 (validation done; nodes reused by engine-ab-v2).
<!-- gen-workflow:begin -->
**Date:** 2026-10-06
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `acuadron-agentic-debt-engine-ab-v1-`
**Experiment name:** `acuadron-agentic-debt-engine-ab-v1`
**W&B project:** `agentic-debt` (group `acuadron-agentic-debt-engine-ab-v1`)
**Dataset:** the final-v3 dataset (`agentic-debt-final`, 871 checkpoints; gym `agentic-debt`)
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-r20-20261006a` (final-v5's)
**Template:** `guparpit-miles-deployer-v10`
**Base:** `acuadron-agentic-debt-final-v5`
<!-- gen-workflow:end -->
**Argo workflow:** `acuadron-agentic-debt-engine-ab-v1-krksx` (submitted 2026-10-06 12:15 UTC)
**Config deltas vs final-v5:** engine flags `sglang_dsa_prefill_backend`/`sglang_dsa_decode_backend` tilelang -> trtllm, `sglang_moe_runner_backend: flashinfer_cutlass`, `sglang_enable_flashinfer_allreduce_fusion: true`, `sglang_speculative_algorithm: NEXTN` (3 steps, eagle top-k 1, 4 draft tokens); shape 48 -> 24 nodes (8 actor = TP8 PP4 DP2, 16 engines), `gym-replicas` 288 -> 144, `rollout_batch_size` 64 -> 32, `global_batch_size` 512 -> 256 (same tokens per train rank as final-v5); run names.
**Checkpoints:** `/mnt/scratch-s3files-rw/acuadron/checkpoints/slime_experiments/acuadron-agentic-debt-engine-ab-v1` (fresh)
**Outcome:** Running; validation passed at steps 0-1 (2026-10-06 14:20 UTC). Engines serve with the four flags, two weight syncs done, R3 payloads under speculation accepted (32 groups materialized, 0 dropped, 720 B/token), log-prob gap 0.0198 / 0.0214 (final-v5 at the same steps: 0.0257 / 0.0270), engine decode 2,204 tok/s at 40-50 running requests (+42% vs final-v5's 1,548), accept length 2.69 of 4 on agentic text.

## Why this shape

A 48-node copy of final-v5 with the engine flags (`acuadron-agentic-debt-final-v6-m62n8`, submitted 11:51 UTC) was
stopped at 12:13 before it got nodes: another session had launched the production run
`acuadron-agentic-debt-final-v6-t66h9` at 09:44 UTC under the same experiment name (same checkpoint and W&B
paths), after stopping final-v5 at 09:43 for its nodes. The cluster then had ~30 free p6 nodes, so this run uses 24.

## Goal

Validate the flags that the one-node benchmarks selected (`examples/arena/harbor-rl-glm53-flash/kdatp/sgl`,
jobs 20261006a-c: real-code TPOT 29.0 -> 13.5 ms, engine output +82% at 45 concurrent) in the full RL loop:

1. engines start and serve with the four flags; weight sync works with the speculative draft (the MTP layer stays
   at the base weights; `enable_draft_weights_cpu_backup` is always on);
2. the R3 payloads under speculation pass the trainer's strict check and the replay fill (no
   "routed_experts payload" errors; `[r3-timing] phase=fill` normal);
3. numerics: `train_rollout_logprob_abs_diff` in the final-v5 band (0.026-0.029 at steps 0-2), `train_rollout_kl`,
   `tis_clipfrac`, `grad_norm`;
4. engine metrics on agentic text: `accept len` and gen throughput at 40-50 running requests in the `Decode batch`
   lines (final-v5: 1,548 tok/s per engine, 33.8 tok/s per request, no speculation).

Rollout supply (groups per hour) is not comparable to final-v5: half the engines, half the groups per step.

## Timeline

| time (UTC) | event |
|---|---|
| 2026-10-06 12:15 | Submitted. |
| 2026-10-06 12:21 | Engines up: `dsa_prefill_backend='trtllm' dsa_decode_backend='trtllm' moe_runner_backend='flashinfer_cutlass' speculative_algorithm='EAGLE' (NEXTN normalizes to EAGLE with the nextn draft) speculative_num_steps=3 speculative_num_draft_tokens=4 flashinfer_allreduce_fusion_backend='auto'` (the `enable_*` flag is deprecated and converts to the backend setting; the bench used the same flag). |
| 2026-10-06 13:13 | Rollout 0: 32 groups, 970 rows, 16.2M tokens, 2,309 s (final-v5: 64 groups on 32 engines in 3,090 s; per engine -25%, both cold). Routing materialized for 32 groups in 92 s, 0 dropped, 11.67 GB int16 = 720 B/token. |
| 2026-10-06 14:07 | Step 0 (cold): `actor_train` 2,828 s, `update_weights` 39 s (the draft keeps the base MTP weights); grad_norm 0.0826 (GBS 256), **log-prob gap 0.0198**, kl 0.00206, tis_clipfrac 0.00029. |
| 2026-10-06 14:20 | Step 1: 1,552 rows, 34.4M padded tokens (17.2M per DP rank, 736 micro-batches), `actor_train` 851 s (49.5 us/token/rank; final-v5 step 1: 62.4 us), `data_preprocess` 2.3 s, `train_wait` 27 s, `update_weights` 23 s; **log-prob gap 0.0214**, kl 0.00222, grad_norm 0.0648. Rollout 1: 1,093 s; rollout 2: 506 s with 42 groups queued: at DP 2 the trainer is the bottleneck again. |

## Engine metrics on agentic text (the `Decode batch` lines of 16 engines, 1,192 samples, 12:21-14:20 UTC)

| | final-v5 engines (tilelang, Triton MoE, no speculation) | engine-ab-v1 engines |
|---|---|---|
| running requests, median | 45 | 40 (p90 48) |
| gen throughput at 40-50 running requests | 1,548 tok/s | **2,204 tok/s (+42%)** |
| per request | 33.8 tok/s | ~50 tok/s |
| accept length (4 draft tokens max) | - | 2.69 (p10 2.54, p90 2.94); accept rate ~0.52 per draft token |
| KV usage | 0.58 | 0.59 (mamba state 0.24 vs 0.12: the draft's states) |

The one-node bench (real code, temperature 1) predicted +82% with an accept length of 3.93; agentic text accepts
2.7 of 4, which lands the live gain at +42% decode throughput and -25% rollout wall time per engine (cold rollout 0).
| 2026-10-06 14:20-20:25 | Steps 2-20: `actor_train` 694-1,006 s, `train_wait` 27-902 s (rollout-bound at multiplier 4 except steps 2-6 and 18), log-prob gap 0.025-0.029, kl 0.0028-0.0037, grad_norm 0.053-0.063. `rollout/weight_version/min` stayed 1 through rollout 18 (chains that run from the first weight version), then 7-10. |
| 2026-10-06 ~20:25 | Stopped with `argo stop`. |
