# Run record: acuadron-agentic-debt-engine-ab-v2 — live check of the weight-staleness cap (r21, ADR-0019) on the final-v7 recipe, 24 nodes

**Status:** Running
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
**Argo workflow:** `acuadron-agentic-debt-engine-ab-v2-qqx4r` (submitted 2026-10-06 20:28 UTC)
**Config deltas vs final-v7:** `rollout_function_path` -> `...nats_rollout.NatsRolloutFn`; `max_weight_staleness: 8` (new); `arena_inflight_multiplier` 4 -> 6; shape 64 -> 24 nodes (8 actor = DP 2, 16 engines), `gym-replicas` 288 -> 216 (1.125 x 192), `rollout_batch_size` 64 -> 32, `global_batch_size` 512 -> 256; trainer image r20 -> r21; run names.
**Checkpoints:** `/mnt/scratch-s3files-rw/acuadron/checkpoints/slime_experiments/acuadron-agentic-debt-engine-ab-v2` (fresh)
**Outcome:** (pending)

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
