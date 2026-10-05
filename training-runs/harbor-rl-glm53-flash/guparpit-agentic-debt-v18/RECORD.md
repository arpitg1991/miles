# Run record: guparpit-agentic-debt-v18 (r56-ns, resubmit of v14 with the class rollout seam, resubmit of v13) — r56 at inflight multiplier 8 with a staleness cap of 8 on the ns trainer stack

**Status:** Prepared
<!-- gen-workflow:begin -->
**Date:** 2026-10-05
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v18-`
**Experiment name:** `guparpit-agentic-debt-v18`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v18`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-wvspans-20261001a@sha256:e29ba91a3392fd250e78f551fca6f33061520a2b456134693d419e1bf5c9eab9`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-ns-20261005b` (digest recorded at submit)
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v9`
<!-- gen-workflow:end -->
**Argo workflow:** not submitted
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Trainer config deltas vs base:** `rollout_function_path` `generate_rollout` -> `NatsRolloutFn`; `arena_inflight_multiplier` 4 -> 8; `max_weight_staleness` 8 (new key); `arena_output_queue_groups` 64 (new key); `prefetch_rollout_data` true (from r56); `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v18`, seeded from r54 (`guparpit-agentic-debt-v7`), `iter_0000089` (its latest save at copy time, 2026-10-05 14:02Z; sidecar `{"rollout_id": 89}` without `wandb_run_id`, data state 89, tracker 89, 135 objects match)
**Outcome:** Prepared. Resubmit of v16 (r56-ns) on the ns-b trainer image, which fixes `NatsRolloutFn`: it read `args.start_rollout_id`, None inside the RolloutExecutor actor, and v15 and v16 stopped at their first rollout with `TypeError: '>' not supported between instances of 'int' and 'NoneType'`.

The owner's order of 2026-10-05: set the staleness cap to 8, log the count and
the reward of the dropped groups, and run it with and without the prefetch, as
`r54-ns` (v12) and `r56-ns` (v13). Both start from the same r54 checkpoint and
replace r53 to r56.

## Goal

Feed the FlashMLA trainer. At inflight multiplier 4 the trainer of r54 waited
444 s per step on average over 26 steps (23% of wall time; 15 of 26 steps over
300 s, max 1,295 s) and r56 waited 484 s (24%); the TileLang runs r53 and r55
waited about 115 s (5%). Multiplier 8 puts 256 groups in flight (8 batches of
32) instead of 128. The staleness cap of 8 drops a group whose oldest weight
version lags the version that trains the batch by more than 8 updates, so the
larger in-flight pool cannot feed the trainer arbitrarily old data. Two new
W&B metrics show what the cap removes.
r56-ns also keeps r56's `prefetch_rollout_data: true`, so the pair v12 against v13 repeats the prefetch comparison of r54 against r56 on the new stack.

## Trainer stack

Branch `arpit-ns-20261005` (the start-rollout fix on top of `19dee1af1a`), image `miles-glm53-ns-20261005b`
(fork `examples/arena/Dockerfile` on `radixark/miles:miles-base-d7f1a42-20261001a`;
pushed to `arena-github/miles`, the one repository the `ecr-dev` role can still
push to since 2026-10-05):

- `arpit-recon-20261001` at `43b7cd490e`: the upstream-main reconcile with the
  weight-version spans fill, the R3 triton check, the staleness drop
  (`--max-weight-staleness` on the NATS path), and the event-logger xattr fix;
- the FlashMLA kernel stack of `arpit-r52-flashmla` (PR #3605 to #3608 at
  `557fb097` plus the flag `--miles-dsa-sparse-attention-forward-backend`),
  cherry-picked; `miles/kernels` is byte-equal to that branch;
- `--arena-output-queue-groups` of `arpit-r51-queue-cap` (`2566c00328`);
- new: `rollout/num_old_age_dropped` and `rollout/reward_old_age_dropped`.

## Upstream check

- `--max-weight-staleness` is the upstream fully-async flag
  (`miles/utils/arguments.py`; `DefaultDataBuffer.get` in
  `miles/rollout/fully_async_data_buffer.py` drops a group whose oldest weight
  version lags the current version by more than N). The NATS path ported the
  same rule on 2026-10-01 (`9775f9587e`): consume-time staleness against the
  version that trains the batch, a dropped group is always dropped (no prompt
  recycle), metrics under the upstream names `rollout/fully_async/*`.
- The two reward metrics are fork additions (`19dee1af1a`); upstream logs only
  the count (`stale_groups_filtered`).
- `--arena-inflight-multiplier` is a fork plugin flag; upstream has no in-flight
  cap of this kind. The rule since r48: size multiplier x engines to the trainer
  step. 4x fit the 2,400 s TileLang step; the 1,450 s FlashMLA step needs more.
- `--prefetch-rollout-data` is fork-only (`examples/arena/RECONCILE.md`,
  "Prefetch port"); r56 is its first GPU run on this base and it worked (fetch
  3 to 4 s on clean steps against r54's 27 to 83 s).
- The gym image `gym-glm53-wvspans-20261001a` sends `weight_version_spans`
  (AREnATasks ADR-0074); this trainer refuses a gym without them
  (`WeightVersionSpansError`). It is a superset of `adr72-20260927a` (mainline
  `538bc63` is an ancestor of its commit `5d85d626`).

## Measurement plan

Control: r54 (`guparpit-agentic-debt-v7-h6668`) at matched rollout ids, and
v12 against v13 for the prefetch.

| Question | Signal | Gate |
| --- | --- | --- |
| The trainer no longer waits | `perf/train_wait_time` per step | under 300 s on more than 80% of steps (r54: 11 of 26) |
| The cap is live and visible | `rollout/num_old_age_dropped` every step; `rollout/reward_old_age_dropped` when a group was dropped; the log line "Weight staleness: dropped=... (max_weight_staleness=8)" | metric present every step from the first rollout |
| Data age | `rollout/weight_version/mean` and `/max` against the trainer version; `rollout/fully_async/avg_staleness`, `/max_staleness` | max staleness at most 8 by construction; record the mean (r54 at 4x: lag 4.8 to 6.3) |
| What the cap throws away | `rollout/num_old_age_dropped` as a share of 32 groups; `rollout/reward_old_age_dropped` against the batch reward | record; a share above 10% means the cap, not the multiplier, sets the batch |
| Reward | `rollout/group_metrics/reward.mean`, batch `avg_reward` | within noise of r54 at matched rollouts (0.63 to 0.67 per 5-rollout bin) |
| Spans | no `WeightVersionSpansError`; `rollout/weight_version/mixed_version_ratio` > 0 | from the first rollout |
| Throughput | `perf/actor_train_tok_per_s`, `perf/step_time` | tok/s equal to r54 (about 43K); step time lower by the removed wait |
| Prefetch (v13 only) | `perf/data_preprocess_time` and `[r3-timing] phase=fetch` on trainer-bound steps, v13 against v12 at matched rollouts | v13 lower (r56: 19 s against r54's 53 s mean) |

## Launch

| Time (UTC) | Event |
| --- | --- |
| 2026-10-05 15:4x | v16 (`guparpit-agentic-debt-v16-szq4x`) stopped at its first rollout: `_train_weight_version` compared the rollout id with `args.start_rollout_id`, which the driver sets on its own copy of `args` after the trainer loads (`miles/ray/placement_group.py`) and which stays None in the RolloutExecutor actor. The fix takes the start from the executor's `load(start - 1)` call, with the first call's rollout id as the fallback; regression test added. `miles/` is unchanged since the T1 image `20261005a`, so the T1 result stands for this image. |
