# Run record: guparpit-agentic-debt-v3 (r50) — r49 with EP8

**Status:** Prepared
<!-- gen-workflow:begin -->
**Date:** 2026-10-02
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v3-`
**Experiment name:** `guparpit-agentic-debt-v3`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v3`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-recon-20261001b`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v2`
<!-- gen-workflow:end -->
**Argo workflow:** not submitted
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Trainer config deltas vs base:** `expert_model_parallel_size` 16 -> 8; `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v3`, seeded from r47 `iter_0000059` (2026-10-01 02:51Z)
**Outcome:** Prepared

Starts from r47 checkpoint `iter_0000059` (a fresh copy; sidecar `{"rollout_id": 59}`), the same weights as r49. Image `miles-glm53-recon-20261001b` and gym image `gym-glm53-adr72-20260927a`, as r49. Sibling arms: r51 (`guparpit-agentic-debt-v4`, r50 plus an output queue capped at 64 groups) and r52 (`guparpit-agentic-debt-v5`, r49 plus EP8 plus the FlashMLA DSA kernel).

## Change

`expert_model_parallel_size` 16 -> 8. The kdatp `ep8` arm (study `trainer-core-profile-glm53-flash`, section (i), job `kdatp-prof-20260929d`) measured a step of 672.6 s against 776.9 s for `base` (-13.4%) on the T2 rows. Peak allocated memory was 109.73 GiB per GPU. The EP16 all-to-all crosses two nodes on EFA (125.1 s per T2 step, fully exposed); EP8 keeps it inside one NVLink node (6.8x faster in the a2a bench). The recon branch adds about 8 GiB per stage (RECONCILE, T2), so the expected peak is about 118 GiB per GPU.

## Upstream check

- Upstream GLM-5.3-Flash uses EP16 at 64 GPUs and EP8 at 32 GPUs (`scripts/run_glm5_3_flash.py`, `cc76e23915`). No upstream layout uses EP8 at 64 GPUs.
- Published B200 NVL8 rows keep EP inside the node (Megatron-Bridge EP8 rows; veRL Qwen3-30B-A3B EP8 in one node). STUDY section "EP placement".
- The checkpoint is EP-agnostic: T1 (2026-10-01) loaded the EP16 save of `iter_0000059` at EP8.

## Measurement plan

| Question | Signal | Gate |
| --- | --- | --- |
| Faster per token | `perf/actor_train_tok_per_s`, `perf/actor_train_time` | at least 10% above r49 (25,300 to 27,900 tok/s) from step 61 on |
| Memory | `memsample-*.log` peak, no OOM | peak below 170 GiB per GPU |
| Same learning | `train/grad_norm`, `train/train_rollout_logprob_abs_diff`, `rollout/group_metrics/reward.mean` | within r49's range at the same rollout ids |
| Engine path | `perf/update_weights_time` | near r49 (31 to 37 s) |

## Launch

<!-- filled at launch -->
