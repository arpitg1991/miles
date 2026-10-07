# Run record: acuadron-agentic-debt-engine-ab-v2 — live check of the weight-staleness cap (r21, ADR-0019) on the final-v7 recipe, 32 nodes

**Status:** Stopped 2026-10-07 07:37 UTC at step 23 (diverged; 32 nodes freed for final-v8-9q9ww).
<!-- gen-workflow:begin -->
**Date:** 2026-10-06
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `acuadron-agentic-debt-engine-ab-v2-`
**Experiment name:** `acuadron-agentic-debt-engine-ab-v2`
**W&B project:** `agentic-debt` (group `acuadron-agentic-debt-engine-ab-v2`)
**Dataset:** `lakefs://arena-inspect/f75dbe74c8e33cf6ce5ad9a46b15f66b3a7339f8c3f01e5997cc972859161684/internal/agentic-debt-r3/acuadron-agentic-debt-final-110-sd015-20261006/manifest.jsonl` (final-v7's 110 chains; gym `agentic-debt`)
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-r21-20261006a` (digest `sha256:93a2e545…`; miles `acuadron/dsa-qp` `14128ef1`, clean clone, FROM r18dsa)
**Template:** `guparpit-miles-deployer-v10`
**Base:** `acuadron-agentic-debt-final-v7` (`final-v7-x44fz`, submitted by another session at 19:56 UTC)
<!-- gen-workflow:end -->
**Argo workflow:** `acuadron-agentic-debt-engine-ab-v2-ffptr` (submitted 2026-10-06 20:30 UTC; a first 24-node submission, `qqx4r`, was stopped at 20:30 before it got nodes)
**Config deltas vs final-v7:** `rollout_function_path` -> `...nats_rollout.NatsRolloutFn`; `max_weight_staleness: 8` (new); `arena_inflight_multiplier` 4 -> 6; shape 64 -> 32 nodes (8 actor = DP 2, 24 engines: the production 1:3 ratio, so each engine carries 64 trajectories as at 16 + 48 nodes), `gym-replicas` 288 -> 216 (1.125 x 192), `rollout_batch_size` 64 -> 32, `global_batch_size` 512 -> 256; trainer image r20 -> r21; run names.
**Checkpoints:** `/mnt/scratch-s3files-rw/acuadron/checkpoints/slime_experiments/acuadron-agentic-debt-engine-ab-v2` (fresh)
**Outcome:** The cap works live: from step 9 it dropped 4-21 stale groups per rollout and kept staleness at most 8 (average 3.3-6.0). The run diverged. With lr 1e-5, multiplier 6 and GBS 256, the train-rollout KL reached 0.18 at step 5 (final-v5 at lr 1.5e-6: 0.005; final-v6 and -v7 at lr 1e-5 with multiplier 4 and GBS 512: 0.067 and 0.081). Response length per sample went 9.4k -> 28.6k tokens and the reward 0.71 -> 0.28 by step 9. At step 16 grad_norm jumped 0.12 -> 1.48 (56 at step 20) and the log-prob gap 0.34 -> 3.0 -> 5.6; the engines then ran the broken weights and the reward reached about 0 at step 22. CloudWatch shows no engine restart, no mid-run weight load and no NaN, so the cause is the optimization, not the infrastructure.

## Goal

1. Version arithmetic: every rollout logs `Weight staleness: dropped=… kept=… avg=… max=… at weight_version=N`. On a fresh run
   with `update_weights_interval 1`, N = rollout id + 1 (rollout 0 trains at version 1, after the initial weight push).
   The kept groups' `max` must equal N - (oldest version of a kept group), at most 8.
2. Drop path: with long chains the oldest call of a group passes 8 versions by rollout ~9 (engine-ab-v1 at multiplier 4:
   `rollout/weight_version/min` stayed 1 through rollout 18). Dropped groups are reaped under R3; the batch still fills.
3. Drop rate and bias on this dataset: `rollout/num_old_age_dropped`, `rollout/reward_old_age_dropped` vs the batch reward.
4. Numerics unchanged: `train_rollout_logprob_abs_diff` in the engine-ab-v1 band (0.020-0.029 over steps 0-20).

## Before launch

- Fast tests in the r21 image on a B200 node (`kdatp/dsaqp/20261006r21`): 669 passed, 0 skipped, including the four
  `test_train_async_arena_order_gives_the_version_that_trains_each_batch` cases (the real driver loop with fakes, start
  rollout 0 and 3, interval 1 and 2). DSA-QP parity PASS (same numbers as r19/r20).

## Timeline

| time (UTC) | event |
|---|---|
| 2026-10-06 20:28 | Submitted (engine-ab-v1 stopped at 20:2x after step 20 to free its 24 nodes). |
| 2026-10-06 20:30 | Resubmitted at 32 nodes as `ffptr`: at 8 + 16 nodes, 192 groups in flight put 96 trajectories on each engine, 1.5x the production load. |

## Divergence (2026-10-07)

Per-step metrics from CloudWatch (`/aws/eks/arena-eks-prod-bom-v2/application`, trainer-worker-0):

| step | reward | response/sample | staleness avg/max/dropped | log-prob gap | train-rollout KL | grad_norm |
|---|---|---|---|---|---|---|
| 0 | 0.706 | 9,359 | 0/0/0 | 0.028 | 0.004 | 0.084 |
| 5 | 0.702 | 13,235 | 5.0/5/0 | 0.318 | 0.184 | 0.065 |
| 8 | 0.447 | 24,031 | 7.5/8/0 | 0.293 | 0.173 | 0.050 |
| 9 | 0.281 | 28,641 | 4.1/8/19 | 0.154 | 0.093 | 0.058 |
| 14 | 0.403 | 17,990 | 5.1/8/7 | 0.209 | 0.131 | 0.239 |
| 16 | 0.348 | 17,401 | 5.4/8/6 | 0.339 | 0.203 | 1.475 |
| 17 | 0.234 | 16,794 | 5.3/8/7 | 3.031 | 2.394 | 2.695 |
| 18 | 0.122 | 15,862 | 5.3/8/7 | 5.605 | 4.312 | 2.407 |
| 20 | 0.130 | 13,169 | 5.7/8/9 | 4.351 | 3.240 | 56.129 |
| 22 | -0.005 | 7,073 | 5.9/8/5 | 1.198 | 0.846 | 16.387 |

The KL between the trainer and the behavior policy grows as (lr x staleness)^2: final-v5's constant predicts 0.16 at lr 1e-5
and staleness 5 (measured 0.18). One update per batch makes the PPO ratio identically 1 (`pg_clipfrac` 0), so no clip
bounds the step against the behavior policy; TIS only reweights (clip 2.0).
