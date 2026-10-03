# Run record: acuadron-agentic-debt-final-v2 — final-v1 recipe, length reward off, 2x batch (64 groups/step), 16 actor nodes

**Status:** Retired
<!-- gen-workflow:begin -->
**Date:** 2026-10-02
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `acuadron-agentic-debt-final-v2-`
**Experiment name:** `acuadron-agentic-debt-final-v2`
**W&B project:** `agentic-debt` (group `acuadron-agentic-debt-final-v2`)
**Dataset:** `lakefs://arena-inspect/0c328c7a4bef7311c2713a246e5623ebc19f4891e7ff5949931fb7e8a5c16b96/internal/agentic-debt-r3/agentic-debt-final/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `0c328c7a4bef7311c2713a246e5623ebc19f4891e7ff5949931fb7e8a5c16b96`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r17-20260928a`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `acuadron-agentic-debt-final-v1`
<!-- gen-workflow:end -->
**Argo workflow:** `acuadron-agentic-debt-final-v2-2slpg`
**W&B run:** (set at launch)
**Task pin:** `7af3bbbece69` (every manifest row)
**Trainer config deltas vs base:** `arena_length_reward_coef` 0.10 -> 0; `global_batch_size` 256 -> 512; `rollout_batch_size` 32 -> 64; `num_trainers` 8 -> 16 (`replicas` 40 -> 48). Workflow: `replica-trainer` 16, `replicas` 48.
**Checkpoints:** `/mnt/scratch-s3files-rw/acuadron/checkpoints/slime_experiments/acuadron-agentic-debt-final-v2` (fresh; starts from `ref_load`)
**Outcome:** Retired 2026-10-03 01:30 after 2 train steps (step 0 actor_train 4,229 s cold at DP 4) in favour of acuadron-agentic-debt-final-v3 (same recipe on the r19 query-parallel DSA trainer, EP8, PP 12/11/11/11; ADR-0018).

## Goal

Same question as final-v1 with the pure chain reward (no length term), twice the groups per optimizer step and DP 4, so that each step sees 64 chains (27% of the 237) and the ~20% all-perfect groups leave the batch instead of training length.

## Setup

Identical to guparpit-agentic-debt-v1 except the dataset and the run identity. Dataset facts that differ from
agentic-debt-766 and matter at runtime: `multi_step_reward_strategy = "mean"`, `min_reward = { segment_pass = 0.0 }`
(no cut), no `[agent] timeout_sec` (agent-timeout-multiplier inert; ack-wait 72,000 s bounds a group), images on
ap-south-1 ECR, `[environment.env] HOME=/home/agent` (template v10 pins `DOCKER_CONFIG`).
