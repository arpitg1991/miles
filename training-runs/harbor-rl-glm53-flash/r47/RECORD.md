# Run record: r47 — agentic debt on agentic-debt-766, fresh from the base model, first-port KDA

**Status:** Running
<!-- gen-workflow:begin -->
**Date:** 2026-09-28
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `rl-glm53f47-`
**Experiment name:** `rl-glm53f-adebt-766-r47`
**W&B project:** `rl-glm53f-adebt-766` (group `rl-glm53f-adebt-766-r47`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r17-20260928a`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `r45`
<!-- gen-workflow:end -->
**Argo workflow:** `rl-glm53f47-wxj87` (second launch; the first launch `rl-glm53f47-qtzlx` was a zombie, see Issues)
**W&B run:** `0ix3m75e` (`https://mega.wandb.agi.amazon.dev/arena/rl-glm53f-adebt-766/runs/0ix3m75e`)
**Task pin:** `3cadc6b0a9a6c71bd8e7485e2d9da1cb04e2e608bfa1b9b531cb1df37cd82140` (`lakefs_commit_id` of each of the 766 manifest rows; manifest md5 `cc78c1ca74df94480a2239402e7fcce0`)
**Image digests:** gym `sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d`, trainer `sha256:e6f04a9ca1abc9df17a7bbf9e3d2aaf6a643f40ed5a457c98a4114719972bcd0` (worker-0 `imageID` read 2026-09-28 16:03Z)
**Trainer config deltas vs base:** `prompt-data-list` path `/mnt/scratch-s3files-rw/guparpit/data/agentic-debt/20260923-v3-locked-oracle/manifest-le5.jsonl` -> the lakeFS URI above; `sglang_disable_overlap_schedule` `true` -> `false`; `rollout_batch_size` 64 -> 32; `glm5_next_kda_tp` unset -> `true`; `arena_inflight_multiplier` 4 -> 8; `skip_actor_forward_only` unset -> `true`; `wandb_project` `rl-glm53f-adebt-v3` -> `rl-glm53f-adebt-766`; `experiment_name`, `project_name`, `arena_sample_summary_dir` renamed for r47 (`diff` of the non-comment lines of `r45/miles-config.yaml` and `r47/miles-config.yaml`)
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-adebt-766-r47` (S3 `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/rl-glm53f-adebt-766-r47/`), last save `iter_0000009` (2026-09-28 15:39Z; tracker reads 9; 64 `.distcp` shards, 583.8 GiB; sidecar `{"rollout_id": 9, "wandb_run_id": "0ix3m75e"}`; no `hf/` dir)
**Outcome:** Running at 2026-09-28 16:03Z. 12 rollouts complete (0 to 11), 10 train steps done (0 to 9), one save. No wrong-dataset string in any task download. Flip A confirmed: `train/ppo_kl` and `train/pg_clipfrac` read exactly 0 at every step.

## Goal

Train agentic debt on the dataset the user gave,
`lakefs://arena-inspect/main/internal/agentic-debt-r3/agentic-debt-766/`,
from the base model, on the latest configuration: the first port of KDA
tensor parallelism, the 16384 output cap, the radix cache, the overlap
schedule on, flip A, and 48 h deadlines. The run ends at `num_rollout`
300, or when the user retires it.

Two questions decide the run:

1. Does the first-port KDA layer train a live run without a wrong weight
   sync? The gate is `train/train_rollout_logprob_abs_diff` near the value
   of the replicated layer (`BUILD.md` risk 9; ADR-0016).
2. Does the reward on the full 766-chain set move up from the base model?
   The frontier reference on the same 766 tasks is
   `acuadron-ad-766-pass8-gpt56-xhigh` (reward 0.4730) and
   `acuadron-ad-766-pass8-opus48-high` (`BUILD.md` "Dataset").

## Setup

`BUILD.md` holds the preparation notes, the full `gen-workflow.py` command,
the diff checks, the risks, and the launch checklist. The run was prepared
on 2026-09-27 (`41efa9d0a`), finalized on the candidate r16 image
(`b2f41c864`), moved to the r17 first-port image (`07b779190`), and launched
on 2026-09-28 (`8c9c69c48`).

### Dataset mistake that this run corrects

r41, r41b, r43, and r45 trained on a different set: the acuadron
oracle-validation branch, v3-locked commit `77c2239b`, filtered to the 551
chains with 5 or fewer steps (`manifest-le5.jsonl`; `r41/BUILD.md` line 10,
`r45/BUILD.md` line 32). 203 of those 551 chains are not in agentic-debt-766,
and the 418 agentic-debt-766 chains with more than 5 steps were never trained
(`BUILD.md` "Why"; audit workflow `wf_e6084655-6ed`). The easier subset made
r45 look far above the frontier models. On the 344 tasks shared with
agentic-debt-766, r45 read 0.62 against GPT-5.6 0.30 and Opus-4.8 0.50, and
those are trained-on tasks, not held out (`wf_e6084655-6ed`, result 4). The
audit found no reward hack. The user statement of 2026-09-27, "I gave you
this dataset to train on", is recorded in the memory note
`adebt-dataset-must-be-766-main.md` (note, not a primary source). Since then
the dataset URI is a required header field of every run record, and the gym
task downloads are checked for the wrong commit at launch (Timeline, 10:20Z).

### Delta vs r45

| Item | Base r45 | This run |
| --- | --- | --- |
| Dataset | v3-locked `77c2239b`, `manifest-le5.jsonl`: 551 chains, 1,809 segments, K 2..5 | agentic-debt-766 at `327e057c`: 766 chains, 6,070 segments, K 2..66, mean K 7.92 |
| Task pin per row | `77c2239b` | `3cadc6b0a9a6...` |
| Start point | r43 `iter_0000039`, seeded into the r45 dir | base DCP (`ref_load`), `finetune` True, rollout id 0 |
| Trainer image | `miles-glm53-r14-20260927a` | `miles-glm53-r17-20260928a` (miles `arpit-glm-53` `4716a367a`) |
| `glm5_next_kda_tp` | unset: each TP rank holds all 64 KDA heads | `true`: each TP rank holds 8 KDA heads (ADR-0016) |
| `skip_actor_forward_only` (flip A) | off | on; `rollout_batch_size` 64 -> 32, `arena_inflight_multiplier` 4 -> 8 (the cap stays 256 groups) |
| `sglang_disable_overlap_schedule` | `true` | `false` (as r42, r44, r46) |
| `ack-wait` / `trainer-task-deadline-secs` | 86000 / 90000 | 172000 / 176000 (48 h) |
| HF export per save | yes (`--save-hf` always on the r14 launcher) | no (r15 and later add it only on `arena_save_hf: true`) |
| W&B | project `rl-glm53f-adebt-v3`, run `ajsur4ej` | project `rl-glm53f-adebt-766`, run `0ix3m75e` |
| Verifier memory cap | none | 16 GiB; 32 GiB on `MP_PyTorch_s0`, 48 GiB on `flamedisx_s0` (dataset authors) |

Unchanged from r45: gym image `gym-glm53-adr72-20260927a`, template
`guparpit-miles-deployer-v10` (uid `16ddd530-501f-4734-82ae-88500671bed8`,
generation 1), `agent-kwargs` `{}`, `publish-jobs-dir` `''`,
`rollout_max_response_len` 16384, window 131072, radix cache on, `replicas`
40 (8 actor nodes, 32 SGLang engines), `gym-replicas` 288, GBS 256, `lr`
1.5e-6, TIS, R3, full recompute, `save_interval` 10, `num_rollout` 300, the
83-id `excluded-nodes` list (`BUILD.md` "Delta vs r45").

Decisions of 2026-09-28 (`BUILD.md` "Pending items"): first port ON; flip A
ON; flip B (selective recompute) OFF, it ran out of memory in the 8-node T2
test; token-level loss average (r46) OFF; the shared-layer KDA image
`miles-glm53-r16-20260928a` not released. A move to the shared layer MUST
remove the `glm5_next_kda_tp` key and needs a separate user yes (ADR-0016).

## Timeline

All times UTC, 2026-09-28. Sources: `RUNLOG.md` lines 2323-2419 (the launch
log), `kubectl get workflow` and `get pod` read at 16:03Z, the
trainer-worker-0 log, and the W&B `_timestamp` of each row.

| UTC | Event |
| --- | --- |
| 07:39Z | `07b779190`: r47 moved to the r17 first-port image. Server dry run accepted `rl-glm53f47-bgls8`, not created. |
| 09:21Z | Server dry run again: `rl-glm53f47-jw7fz`, not created. |
| 09:23:31Z | r45 `rl-glm53f45-vvqhg` Stop patch (user go "ok go ahead"). r45 was in `actor_train` of step 55; last complete save `iter_0000049`. |
| 09:27:57Z | No r45 PyTorchJob, Deployment, Service, or pod left. |
| 09:28:18Z | First launch `rl-glm53f47-qtzlx` created (`creationTimestamp`; `RUNLOG.md` records the create command at 09:28:16Z). |
| ~09:31:25Z | Kueue admitted the PyTorchJob, then logged `EvictedDueToNodeFailures` for TAS nodes `i-00cfa767b52449c6f` and `i-06bf42135a1b37f40`. ttl 0 removed the PyTorchJob. |
| 09:32:52Z | `i-06bf42135a1b37f40` `NotReady` with `karpenter.sh/disrupted` (`AMIDrift` since 2026-09-23). |
| 09:42:09Z | The zombie `rl-glm53f47-qtzlx` stopped (`Stopped with strategy 'Stop'`, `finishedAt` 09:45:12Z). |
| 09:45:36Z | Second launch `rl-glm53f47-wxj87` created from the same `workflow.yaml` (`creationTimestamp`; create command 09:45:34Z). |
| 09:48:47Z | Kueue admitted the workload with no wait. `PodsReady` 09:49:07Z. worker-0 `startTime` 09:49:02Z. |
| 09:50:35Z | 40 trainer workers `Running`. |
| 09:51:08Z | `Failed to load slime extra state: invalid literal for int() with base 10: 'release'` (expected: `ref_load` tracker). `start_rollout_id` 0. |
| 09:51:12Z | W&B run `0ix3m75e` created in project `rl-glm53f-adebt-766`. |
| 09:51:38Z | `Pulled lakefs://arena-inspect/327e057c.../agentic-debt-766/manifest.jsonl`; pod copy 766 rows, md5 `cc78c1ca...`. |
| 10:03:36Z to 10:03:55Z | 32 of 32 SGLang engines up. `disable_radix_cache=False`, `disable_overlap_schedule=False`, `mamba_radix_cache_strategy='extra_buffer'`, `context_length=131072`. |
| 10:05:32Z | First weight sync `ok=true`, 44.9 s. |
| 10:05:33Z | `Rollout 0: collecting 32 groups (GBS=256, n_samples=8, queue=0)`; `NATS worker started: ... max_in_flight=256`. |
| ~10:09Z | Gym Deployment 288/288. |
| 10:17:37Z | First result (`realdiff_alonfnt_notata_s0`, zero std at 0.0, dropped). |
| 10:18:54Z | First kept group (`realdiff_benavlabs_fastcrud_s0`, reward 0.5). |
| 10:20Z | All 440 gym `Downloading` lines read the `3cadc6b0...`/`agentic-debt-766/tasks/` path. 0 `77c2239b`, 0 `20260923-v3-locked`, 0 `manifest-le5`. |
| 10:42:40Z | `Rollout 0 complete: 32 groups (agentic-debt=32), 256 samples, avg_reward=0.626 in 2227.6s`. |
| 11:39:30Z | Train step 0 logged: `actor_train_time` 3380 s (cold KDA kernel compile, no cache seed), `train_wait_time` 2787 s, `step_time` 6168 s. |
| 12:02:24Z | Train step 1 logged: `actor_train_time` 1302 s, `step_time` 1374 s. |
| 14:11:10Z, 14:39:23Z, 15:08:08Z, 15:41:56Z | `Error in NATS worker loop: nats: timeout` after a weight update; the loop continued (rollouts 7 to 11 completed after them). |
| 14:33:14Z | Gym pod `rl-glm53f47-wxj87-gym-dcc8bd575-2p9r8` `dind` exit 137; pod phase `Failed`, replaced. |
| 15:39:41Z | `successfully saved checkpoint from iteration 9`; tracker written 15:40:43Z. No HF export. |
| 15:48:15Z | `Rollout 11 complete: ... avg_reward=0.578 in 489.7s (queue=320)`. |
| 16:03Z | Status read: workflow `Running`, 40 trainer pods `Running` with 0 restarts, gym 288/288 ready, 330 pods `Running`. |

## Results

Read from W&B run `0ix3m75e` (history, 67 rows) and the trainer-worker-0
log at 2026-09-28 16:03Z. The run continues; these are the first 12
rollouts, about half an epoch (23.9 rollouts per epoch at 32 kept groups is
the lower bound, `BUILD.md` "Delta vs r45").

### Train steps (flip A and first-port KDA gates)

| Step | `train/grad_norm` | `train/train_rollout_logprob_abs_diff` | `train/train_rollout_kl` | `train/ppo_kl` | `train/pg_clipfrac` | `perf/actor_train_time` s | `perf/step_time` s | `perf/actor_train_tok_per_s` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.1678 | 0.03249 | 0.004758 | 0 | 0 | 3380 (cold compile) | 6168 | 3,092 |
| 1 | 0.1091 | 0.03218 | 0.004616 | 0 | 0 | 1302 | 1374 | 13,815 |
| 2 | 0.0994 | 0.03331 | 0.004575 | 0 | 0 | 1332 | 1437 | 18,434 |
| 3 | 0.0877 | 0.03191 | 0.004386 | 0 | 0 | 1327 | 1412 | 19,941 |
| 4 | 0.0927 | 0.03330 | 0.004553 | 0 | 0 | 1319 | 1453 | 24,274 |
| 5 | 0.0958 | 0.03162 | 0.004226 | 0 | 0 | 1491 | 1589 | 20,854 |
| 6 | 0.0931 | 0.03405 | 0.004547 | 0 | 0 | 1645 | 1804 | 21,807 |
| 7 | 0.0890 | 0.03226 | 0.004257 | 0 | 0 | 1538 | 1666 | 21,986 |
| 8 | 0.1147 | 0.03650 | 0.004915 | 0 | 0 | 1492 | 1634 | 25,020 |
| 9 | 0.0843 | 0.03782 | 0.005188 | 0 | 0 | 1686 | 1857 | 21,572 |

- Weight-sync gate (`BUILD.md` risk 9, ADR-0016): `abs_diff` reads
  0.0316 to 0.0378. The reference is 0.0272 (kdatp T2, r43 `iter_0000039`,
  r45 rollout-40 data; `kdatp/RESULTS.md` "Step-1 parity"). r47 starts from
  the base model on a different dataset, so the two are not the same
  measurement. A wrong SGLang weight gives a step of a different order
  (`kdatp/RESULTS.md`, review minor 2), and no such step occurred. No
  `rollout/log_probs` metric exists (flip A). `update_weights_time` 23.9 to
  27.3 s from step 1 on (44.9 s at step 0).
- Flip A: `ppo_kl` and `pg_clipfrac` read exactly 0 at every step, and no
  log-prob pass runs (`perf/` has no `log_probs_time` key).
- `grad_norm` 0.084 to 0.168; kdatp T2 read 0.127 at step 1 on r43
  `iter_0000039`. The gradient clip at 1.0 never fires (ADR-0016).
- Train step time: `actor_train_time` 1,302 to 1,686 s for 824 to 1,620
  training rows per step. r45 read 3,676 to 4,022 s at rollouts 48 to 50
  (`BUILD.md` "Capacity"), on a different dataset with fewer rows per
  episode, so the -21.3% of kdatp T2 is not isolated here. The trainer
  sets the pace: `perf/rollout_time` is 352 to 491 s from rollout 2 on,
  and the queue holds 320 groups at rollout 8 and later.
- `perf/actor_train_mfu` 0.023 to 0.050 (logged MFU; the MFU study notes
  that this figure charges dense attention to all 45 layers, ADR-0016).
- Peak GPU memory: not measured in the live run. `MILES_LOG_PEAK_MEMORY`
  is unset, and the `memsample-*.log` files under
  `s3://arena-scratch-prod-bom-ap-south-1/guparpit/logs/rl-glm53f-adebt-766-r47/`
  hold host RSS only. kdatp T2 read 105 to 110 GiB (DCGM) for the first
  port against 160 to 165 GiB replicated (`kdatp/RESULTS.md`); not
  re-verified on r47.

### Rollouts

`rollout/episode_raw_reward` is the mean over the 256 kept episodes and
equals the `avg_reward` of the trainer log line. `rollout/raw_reward` is the
mean over the training rows (one row per segment, `arena_train_segments`
`all`), so long chains weigh more.

| Rollout | `episode_raw_reward` | `raw_reward` (rows) | Rows (`num_training_samples`) | `response_len/mean` (row) | `response_len/max` | `perf/rollout_time` s | Queue at end |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.626 | 0.776 | 824 | 5,923 | 40,845 | 2,234 | 0 |
| 1 | 0.579 | 0.758 | 862 | 10,833 | 65,886 | 957 | 0 |
| 2 | 0.619 | 0.768 | 1,294 | 9,857 | 65,701 | 352 | 125 |
| 3 | 0.592 | 0.758 | 1,224 | 10,968 | 67,677 | 391 | 164 |
| 4 | 0.567 | 0.719 | 1,298 | 12,735 | 68,332 | 435 | 186 |
| 5 | 0.578 | 0.745 | 1,166 | 13,910 | 81,122 | 405 | 204 |
| 6 | 0.637 | 0.782 | 1,518 | 11,862 | 74,869 | 447 | 251 |
| 7 | 0.589 | 0.794 | 1,620 | 10,731 | 80,146 | 440 | 300 |
| 8 | 0.615 | 0.796 | 1,332 | 14,732 | 82,805 | 440 | 320 |
| 9 | 0.578 | 0.728 | 1,388 | 14,375 | 87,149 | 439 | 320 |
| 10 | 0.627 | 0.792 | 1,472 | 13,519 | 72,833 | 465 | 320 |
| 11 | 0.578 | not logged yet | 1,422 | 14,797 | 82,104 | 491 | 320 |

Rollout 11 summary values (W&B summary at 16:03Z):

| Metric | Value | Note |
| --- | --- | --- |
| `rollout/dyn_sampling_dropped` / `dyn_sampling_drop_frac` | 20 / 0.385 | 52 groups examined, 32 kept (`group_metrics/n_groups` 52) |
| `rollout/zero_reward_groups_frac` | 0.115 | |
| `rollout/group_metrics/reward.mean` | 0.549 | over the 52 examined groups |
| `rollout/reward_p25` / `p50` / `p75` | 0.125 / 0.5 / 1 | |
| `rollout/group_metrics/num_turns.mean` / `max` | 34.4 / 499 | |
| `rollout/episode_response_length/mean` / `max` | 82,193 / 295,948 | tokens per episode (all segments) |
| `rollout/clipped_turns` | 0.145 | mean per-trajectory count of turns clipped at the 16384 cap |
| `rollout/masked_output_tokens` | 2,369 | mean per trajectory |
| `rollout/off_policy_round/mean` / `max` | 8.5 / 11 | `weight_version/mean` 1.9: groups arrive 7 to 11 rollouts behind the policy |
| `rollout/dropped_groups/too_large`, `deadline`, `chain_abort`, `token_ref` | 0, 0, 0, 0 | |
| `rollout/failed_frac` | 0.0039 | 1 failed slot of 256 (`failed/no_reward`) |
| `rollout/repetition_frac`, `truncated_ratio`, `context_overflow_frac` | 0, 0, 0 | |
| `rollout/prefix_cache_hit_rate` | 0 | metric not fed by this engine path; SGLang logs show `#cached-token` above 0 |

Trainer log counts (worker-0, 09:49Z to 16:05Z): 0 `Requested token count
exceeds`, 0 `maximum context length`, 0 `too_large`, 0 `rollout deadline
exceeded`, 0 `CUDA out of memory`, 0 `NaN`, 22 `update_weights ... ok=true`.

## Issues

- **First launch zombie (`rl-glm53f47-qtzlx`).** Kueue admitted the trainer
  and then evicted it for TAS node failures. Cause: Karpenter AMI drift.
  When r45 freed its nodes, Karpenter started to replace the drifted ones
  (15 drifted B200 nodes in disruption at 09:41Z; all 53 drifted live nodes
  held arena-tasks pods). ttl 0 removed the PyTorchJob, and the workflow
  stayed `Running` in `wait-trainer-nats` with only NATS and the wait pod:
  the zombie signature of 2026-09-18 and of the first r45 launch
  (`rl-glm53f45-54bb8`). The operator stopped it at 09:42:09Z and created
  the second launch from the same file. That Stop and re-create were not
  on the user's cleared list (`wf_44fd3690-bac`, result 3). No checkpoint
  and no W&B run belong to `qtzlx`.
- **Cold kernel compile at step 0.** The r17 image has no kernel-cache
  seed, so step 0 `actor_train_time` read 3,380 s against 1,302 s at step
  1. kdatp T2 saw the same on its cold log-prob pass (2,135 s against 312
  s warm). Persistent `TILELANG_CACHE_DIR` and `TRITON_CACHE_DIR` on shared
  storage are a follow-up (`BUILD.md` "Pending items").
- **`dind` sidecar restarts in the gym.** At 16:03Z the 289 gym pods had
  85 `dind` restarts: 48 `Error` and 36 `OOMKilled`. One pod
  (`rl-glm53f47-wxj87-gym-dcc8bd575-2p9r8`) went `Failed` at 14:33:14Z
  (`dind` exit 137, `gym-worker` `ContainerStatusUnknown`) and was replaced.
  The 64 GiB Docker-in-Docker sidecar hosts 8 trials with verifier caps of
  16 to 48 GiB (`BUILD.md` risks 2 and 3). r45 saw one `dind` OOMKill at
  07:05Z on 2026-09-27 (`RUNLOG.md` r45 section). The effect on rewards is
  not measured: `rollout/dropped_groups/*` read 0 and `failed_frac` 0.0039,
  so the restarts did not remove groups, but a restart loses the trials in
  flight on that pod. Not investigated further.
- **`nats: timeout` in the NATS worker loop.** After the weight updates at
  14:11Z, 14:39Z, 15:08Z, and 15:41Z, `nats_rollout.py:1793` `js.publish`
  raised `nats.errors.TimeoutError`, logged as `Error in NATS worker loop`
  with a traceback. The loop retried; rollouts 7 to 11 completed after
  these lines with 0 dropped groups. Log-only so far.
- **Expected startup noise.** 21 tracebacks in total in the worker-0 log:
  1 Ray CPU-list read at 09:49Z, 10 SGLang `freeze_gc` connection-refused
  lines at 10:03Z during engine warmup, and the NATS timeouts above. 128
  `destroy_weights_update_group` HTTP 400 lines at 10:04:47Z are startup
  only. `Failed to load slime extra state ... 'release'` is the `ref_load`
  tracker; a fresh start has no W&B id to restore.
- **lakeFS start burst.** About 256 task downloads started at once. 111
  failed attempt 1 of 3 and 73 failed attempt 2 of 3 (connection errors to
  `prod.artifact-vault.agi.amazon.dev`). 0 tasks failed all 3 attempts at
  10:22Z. A gym restart gives a new burst.

## Follow-ups

- Move r47 to the shared-layer KDA image at a save: needs a separate user
  yes. The move MUST remove the `glm5_next_kda_tp` key. A direct load of a
  first-port save into the shared layer is not tested; test it first
  (ADR-0016; `BUILD.md` "KDA layer choice").
- After each epoch (about 24 rollouts), count `rollout deadline exceeded`,
  verifier out-of-memory exits, `rollout/dropped_groups/too_large`, and
  the chains with K > 5 that reach step 6 or more (`RUNLOG.md` lines 2415-2417).
- Explain the `dind` `OOMKilled` and `Error` restarts (85 at 16:03Z) and
  their effect on the trials in flight. Owner: operator. Record the result
  here or in a study.
- Persistent kernel caches on shared storage (a template env change), so
  a resume does not pay the 2,000 s compile again.
- PP split 12/11/11/11 (`decoder_first_pipeline_num_layers` 12,
  `decoder_last_pipeline_num_layers` 11): a memory and time test, not
  prepared (`BUILD.md` "Pending items").
- Off-policy lag: `off_policy_round/mean` 8.5 at rollout 11 with the queue
  full at 320 groups. Decide whether the 8x in-flight multiplier stays.
- r45 steps 50 to 54 exist only in W&B run `ajsur4ej`; the r45 dir keeps
  `iter_0000049` and the r43 seed `iter_0000039` for a later resume.

## Sources

- Run files: `BUILD.md`, `miles-config.yaml`, `workflow.yaml` (this folder).
  Commits `41efa9d0a`, `b2f41c864`, `07b779190`, `8c9c69c48` on `arpit-glm-53`.
- `RUNLOG.md` lines 2323-2419 (r47 launch log) and 2189-2265 (r45).
- `examples/arena/harbor-rl-glm53-flash/kdatp/RESULTS.md` (first-port KDA
  gates T1, T2, and the 0.0272 reference).
- `miles_plugins/arena/adr/0016-*.md` (KDA tensor-parallel sharding).
- `examples/arena/README.md` image lineage (`miles-glm53-r17-20260928a`,
  `gym-glm53-adr72-20260927a`).
- W&B: `https://mega.wandb.agi.amazon.dev/arena/rl-glm53f-adebt-766/runs/0ix3m75e`
  (history read 2026-09-28 16:03Z).
- S3 (`--profile arena-prod-bom-user --region ap-south-1`,
  `s3://arena-scratch-prod-bom-ap-south-1/guparpit/`):
  `checkpoints/slime_experiments/rl-glm53f-adebt-766-r47/`,
  `debug/rl-glm53f-adebt-766-r47/sample_summary/rollout_0.jsonl` to
  `rollout_11.jsonl`, `logs/rl-glm53f-adebt-766-r47/memsample-*.log`,
  `kdatp/` (T1 and T2 raw results).
- Cluster reads at 2026-09-28 16:03Z: `kubectl --context arena-prod-bom-v2
  -n arena-tasks get workflow rl-glm53f47-wxj87 rl-glm53f47-qtzlx`,
  `get pods`, `logs rl-glm53f47-wxj87-trainer-worker-0 -c pytorch`.
- Workflow journals: `wf_44fd3690-bac` (r17 image build, shared-layer
  gradient dump verdict, r45 stop, r47 launch), `wf_e6084655-6ed`
  (all-pass audit; r45 dataset overlap), `wf_58a8a79c-51c` (first-port
  KDA design).
- AREnATasks ADR-0069 (training reward is the Harbor trial reward),
  ADR-0071 (token arrays by file reference; 30 MiB result cap), ADR-0072
  (gym takes the output cap and window from the task message).
- Memory notes (not primary): `adebt-dataset-must-be-766-main.md`,
  `r45-mfu-investigation-2026-09-27.md`.
