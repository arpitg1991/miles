# Run record: acuadron-agentic-debt-final-v6 — final-v5 + the SGLang engine flags of the kdatp/sgl benchmarks (trtllm DSA, cutlass MoE, allreduce fusion, NEXTN)

**Status:** Running
<!-- gen-workflow:begin -->
**Date:** 2026-10-06
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `acuadron-agentic-debt-final-v6-`
**Experiment name:** `acuadron-agentic-debt-final-v6`
**W&B project:** `agentic-debt` (group `acuadron-agentic-debt-final-v6`)
**Dataset:** the final-v3 dataset (`agentic-debt-final`, 871 checkpoints; gym `agentic-debt`)
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-r20-20261006a` (same as final-v5)
**Template:** `guparpit-miles-deployer-v10`
**Base:** `acuadron-agentic-debt-final-v5`
<!-- gen-workflow:end -->
**Argo workflow:** `acuadron-agentic-debt-final-v6-m62n8` (submitted 2026-10-06 11:51 UTC, next to the running final-v5 for a same-time A/B)
**Trainer config deltas vs base (engine only):** `sglang_dsa_prefill_backend`/`sglang_dsa_decode_backend` tilelang -> trtllm; `sglang_moe_runner_backend: flashinfer_cutlass`; `sglang_enable_flashinfer_allreduce_fusion: true`; `sglang_speculative_algorithm: NEXTN`, `sglang_speculative_num_steps: 3`, `sglang_speculative_eagle_topk: 1`, `sglang_speculative_num_draft_tokens: 4`; run names -> v6. Same shape (16 actor + 32 engine nodes, 288 gyms, multiplier 4, GBS 512).
**Checkpoints:** `/mnt/scratch-s3files-rw/acuadron/checkpoints/slime_experiments/acuadron-agentic-debt-final-v6` (fresh; starts from `ref_load`)
**Outcome:** (pending)

## Goal

final-v5 is rollout-bound (`train_wait` 590-1,385 s per step at steps 3-12 against a 650-800 s train step). Its
engines run at a 30 ms decode step (33.8 tok/s per request, median 45 running requests, 27.6k context). The
one-node benchmarks `examples/arena/harbor-rl-glm53-flash/kdatp/sgl` (jobs 20261006a-c, same image, same engine
arguments) selected the flags above: on real code prompts at temperature 1, 45 concurrent 27k-token requests,
median TPOT 29.0 -> 13.5 ms and output 1,159 -> 2,106 tok/s per engine (+82%); accept length 3.93 of 4.

Measure live: (1) engine `Decode batch` gen throughput at 40-50 running requests and `accept len` on agentic
text; (2) rollout supply: time between `Rollout N complete` lines and `perf/step_time` while rollout-bound
(final-v5: 1,350-2,600 s per step at steps 3-12); (3) numerics: `train_rollout_logprob_abs_diff` (final-v5:
0.026 at step 0 rising to 0.037 at step 12), `train_rollout_kl`, `tis_clipfrac`, `grad_norm`; (4) the R3 payloads
under speculative decoding (the trainer's strict check rejects a malformed payload: "routed_experts payload").

## Timeline

| time (UTC) | event |
|---|---|
| 2026-10-06 11:51 | Submitted. |
