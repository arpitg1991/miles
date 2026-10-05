# Run record: guparpit-agentic-debt-v12 (r54-ns) — r54 at inflight multiplier 8 with a staleness cap of 8 on the ns trainer stack

**Status:** Crashed at the first rollout (config defect); replaced by v15
<!-- gen-workflow:begin -->
**Date:** 2026-10-05
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v12-`
**Experiment name:** `guparpit-agentic-debt-v12`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v12`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-wvspans-20261001a@sha256:e29ba91a3392fd250e78f551fca6f33061520a2b456134693d419e1bf5c9eab9`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-ns-20261005a@sha256:fc64f05a1b485a63e6e03858c54d6fa5ddf93a09b78ed716556d9c69ee053a4f`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v7`
<!-- gen-workflow:end -->
**Argo workflow:** `guparpit-agentic-debt-v12-zvmdk`
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Trainer config deltas vs base:** `arena_inflight_multiplier` 4 -> 8; `max_weight_staleness` 8 (new key); `arena_output_queue_groups` 64 (new key); `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v12`, seeded from r54 (`guparpit-agentic-debt-v7`), `iter_0000089` (its latest save at copy time, 2026-10-05 14:02Z; sidecar `{"rollout_id": 89}` without `wandb_run_id`, data state 89, tracker 89, 135 objects match)
**Outcome:** Crashed 2026-10-05 15:01:32Z: worker-0 exited with code 1 at the first rollout, the job went Failed and the training-operator deleted it (ttl 0), so the pod logs are gone. Cause: `rollout_function_path` was the legacy `generate_rollout`, which drops the engine weight version; with `max_weight_staleness` set, `generate_rollout` raises `ValueError("--max-weight-staleness needs the weight version that trains the batch... Only --rollout-function-path ...NatsRolloutFn passes it")` (ADR-0018). The fix is the class path `NatsRolloutFn`; v15 (r54-ns) carries it.

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

## Trainer stack

Branch `arpit-ns-20261005` (`19dee1af1a`), image `miles-glm53-ns-20261005a`
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

## Launch

| Time (UTC) | Event |
| --- | --- |
| 2026-10-05 13:40 | Image `miles-glm53-ns-20261005a` built from `19dee1af1a` (clean tree) and pushed to `arena-github/miles` us-east-1, digest `sha256:fc64f05a1b485a63e6e03858c54d6fa5ddf93a09b78ed716556d9c69ee053a4f`. The `ecr-dev` role is denied `ecr:InitiateLayerUpload` on `arena-slime-dev` in both regions since this date. |
| 13:49 | In-image parse of the rendered argv (`recon2/parse/fullparse10.py`, `CFG=/work/v12.yaml`, `WORLD_SIZE` 320, `RANK` 0, CUDA stub; `parse-v12.out`): exit 0. `max_weight_staleness 8`, `arena_inflight_multiplier 8`, `arena_output_queue_groups 64`, `prefetch_rollout_data False`, `object_store_backend ray`, `use_fault_tolerance True`, `flash_mla`, `triton`, EP 8, `glm5_next_kda_tp True`, `skip_actor_forward_only True`, `calculate_per_token_loss True`, `grpo_std_normalization False`, `arena_length_reward_coef 0.0`, 32 x 8 = 256, save dir v12. |
| 13:59 to 14:19 | T1 parity on the image (`recon-t1-new-20261005c`, 1 node, forward `flash_mla`, r47 `iter_0000059` through `guparpit-agentic-debt-v2`; harness = the fork T1 harness of 2026-10-02, `recon2/t1-ns/`): load 903 s, `dsa_forward_backend_modules {'flash_mla': 11}`, rc 0. Compare against the r17 reference run `20261001a-ref`: 37,534 of 37,534 gathered tensors bitwise equal; log-prob mean 0.0499 (limit 0.0990), p99 0.560 (1.140), logits relative L2 0.230 (0.459): PASS (`recon2/t1-ns/out/compare-20261005c.json`). Job and ConfigMap deleted after the run. |
| 14:02 | Seed copied from r54 `iter_0000089` (`metrics-1001/ckcopy-v12.sh`): 135 objects match by name and size; sidecar `{"rollout_id": 89}`; data state `arena_data_source_state_89.pt`; tracker 89 written last. Note: the names v10 and v11 were taken by `guparpit-agentic-debt-v10-9j466` (another session of the owner, created 06:22Z) and its directory; a first seed copy into the v10 directory was reverted at 14:01Z (iter_0000089 and state 89 removed, tracker restored to 79, before that run saved 89), so these runs are v12 and v13. |
| 14:25:48 | Submit `guparpit-agentic-debt-v12-zvmdk`. Node exclusions refreshed at submit: 304 B200 nodes, 49 bad (4 NotReady, 49 with a taint other than the GPU and EFA taints); 311 (the r54 list of 262 plus 49 bad) excluded. Trainer image by digest, gym image `gym-glm53-wvspans-20261001a` by digest, `prompt-data-list` unchanged from the base. |
| 14:28:56 | Trainer PyTorchJob `guparpit-agentic-debt-v12-zvmdk-trainer` created. Queued: the B200 queue fit 1 of 40 pods while r53 to r56 held 160 nodes. |
| 14:30:38 | r53, r54, r55, r56 stopped (`spec.shutdown: Stop`, cleanup ran by 14:34:39). |
| 14:33 | Kueue admitted the trainer; 40 worker pods Running. Argument dump: `max_weight_staleness 8`, `arena_inflight_multiplier 8`, `arena_output_queue_groups 64`, `prefetch_rollout_data False`, `miles_dsa_sparse_attention_forward_backend flash_mla`, `sglang_moe_runner_backend triton`, `expert_model_parallel_size 8`, `calculate_per_token_loss True`, `grpo_std_normalization False`, `arena_length_reward_coef 0.0`, `use_rollout_routing_replay True`. The one `Traceback` in the log is Ray's `_get_docker_cpus` cpuset parse at `ray start` (benign, as in every run on this base). |
| 15:01:32 | worker-0 `exitCode: 1` (training-operator events `ExitedWithCode`, `PyTorchJobFailed: 1 Worker replica(s) failed`), job deleted 15:01:33, pods garbage-collected. 28 min after start = the first `generate_rollout` call. The in-image parse cannot catch this: the refusal is a runtime check at the first rollout. Lesson for the recipe: `max_weight_staleness` requires `rollout_function_path: ...NatsRolloutFn`. |
