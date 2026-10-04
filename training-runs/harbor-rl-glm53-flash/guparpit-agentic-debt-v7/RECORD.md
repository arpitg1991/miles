# Run record: guparpit-agentic-debt-v7 (r54) — r52's FlashMLA trainer with the r53 loss (no bonus, token-level, no spread division)

**Status:** Prepared
<!-- gen-workflow:begin -->
**Date:** 2026-10-04
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v7-`
**Experiment name:** `guparpit-agentic-debt-v7`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v7`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-recon-20261002-flashmla-evfix`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v5`
<!-- gen-workflow:end -->
**Argo workflow:** not submitted
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Trainer config deltas vs base:** `arena_length_reward_coef` 0.10 -> 0.0; `calculate_per_token_loss` true; `disable_grpo_std_normalization` true; `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v7`, seeded from r52 `iter_0000069`
**Outcome:** Prepared

## Goal

The production candidate: the fastest trainer we have (EP8 and the FlashMLA DSA kernel of upstream PR #3608, 42,593 tok/s on r52 at step 61) with the loss that holds reward (r53: true reward 0.747 against r51's 0.663 at rollouts 85 to 88, effort flat). r52 collapsed under the length bonus to 0.06 to 0.13 by rollout 255. This run resumes r52's `iter_0000069` with the r53 loss, so r52 is its control at the same rollout ids. No output queue cap: that flag lives on `arpit-r51-queue-cap`, not in this image.

## Upstream check

As r53 (`guparpit-agentic-debt-v6` record): `--calculate-per-token-loss` (DAPO) and `--disable-grpo-std-normalization` (Dr. GRPO, arXiv 2503.20783) are upstream flags; no upstream framework puts a length term in the reward. FlashMLA: upstream PR #3608 head `557fb097`, ported line for line; T1 parity against image b passed on 2026-10-02 (37,534 of 37,534 weight-sync tensors bitwise equal, log-probs at the noise floor).

## Measurement plan

| Question | Signal | Gate |
| --- | --- | --- |
| Speed holds with the new loss | `perf/actor_train_tok_per_s` from step 71 on | at least 38,000 (r52: 42,593 at step 61) |
| Effort holds | calls, tool calls, and output tokens per trial from the published tree, as in the r53 comparison | within 20% of rollouts 70 to 76 at rollouts 85 to 99 (r52 fell to a third) |
| Reward holds | `rollout/group_metrics/reward.mean` | at least 0.70 at rollouts 85 to 99 (r52: 0.70 then 0.59 at 110 to 119) |
| Same-task reward | paired same-task change (`/workplace/guparpit/kdfast/scratch/drop/sametask/`) | within ±0.02 |
| Flags live | argument dump: `calculate_per_token_loss True`, `grpo_std_normalization False`, `arena_length_reward_coef 0.0`, `miles_dsa_sparse_attention_forward_backend flash_mla` | present at start |
| Reaper risk | `queue=` in `Rollout N complete`, rollout seconds | with no queue cap the queue may fill to 320; a full queue idles the engines (r47, r50 were reaped). Watch the 60-min power. |

## Launch

<!-- filled at launch -->
