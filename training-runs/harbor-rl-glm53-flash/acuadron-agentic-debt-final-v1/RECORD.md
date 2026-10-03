# Run record: acuadron-agentic-debt-final-v1 — r48 recipe on agentic-debt-final (237 chains, "mean" reward, no stop)

**Status:** Retired
<!-- gen-workflow:begin -->
**Date:** 2026-10-01
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `acuadron-agentic-debt-final-v1-`
**Experiment name:** `acuadron-agentic-debt-final-v1`
**W&B project:** `agentic-debt` (group `acuadron-agentic-debt-final-v1`)
**Dataset:** `lakefs://arena-inspect/0c328c7a4bef7311c2713a246e5623ebc19f4891e7ff5949931fb7e8a5c16b96/internal/agentic-debt-r3/agentic-debt-final/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `0c328c7a4bef7311c2713a246e5623ebc19f4891e7ff5949931fb7e8a5c16b96`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r17-20260928a`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v1`
<!-- gen-workflow:end -->
**Argo workflow:** `acuadron-agentic-debt-final-v1-tjdbb`
**W&B run:** (set at launch)
**Task pin:** `7af3bbbece69` (every manifest row)
**Trainer config deltas vs base:** `user` guparpit -> acuadron; `experiment_name`/`project_name` -> `acuadron-agentic-debt-final-v1`; `prompt-data-list` agentic-debt-766 @327e057c -> agentic-debt-final @0c328c7a4bef; `arena_sample_summary_dir` -> acuadron path. Workflow: `username`, `experiment-name`, `publish-jobs-dir` -> `/mnt/scratch-s3files-rw/acuadron/harbor-training`.
**Checkpoints:** `/mnt/scratch-s3files-rw/acuadron/checkpoints/slime_experiments/acuadron-agentic-debt-final-v1` (fresh; starts from `ref_load`)
**Outcome:** Retired 2026-10-02 22:33Z after 5 steps (no checkpoint) in favour of final-v2 (length reward off, 2x batch, DP 4). Rollouts 0-4: median R 0.95, 53% of attempts flawless, 11% negative, 20% of groups all-perfect.

## Goal

Train GLM-5.3-Flash with the current agentic-debt recipe on the agentic-debt-final dataset and watch whether
`episode_raw_reward` (now in (-inf, 1]) trends up over 100+ steps, as r44/r47 did on their sets.

## Setup

Identical to guparpit-agentic-debt-v1 except the dataset and the run identity. Dataset facts that differ from
agentic-debt-766 and matter at runtime: `multi_step_reward_strategy = "mean"`, `min_reward = { segment_pass = 0.0 }`
(no cut), no `[agent] timeout_sec` (agent-timeout-multiplier inert; ack-wait 72,000 s bounds a group), images on
ap-south-1 ECR, `[environment.env] HOME=/home/agent` (template v10 pins `DOCKER_CONFIG`).
