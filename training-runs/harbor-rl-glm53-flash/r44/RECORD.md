# Run record: r44 — auctioneer caponly-1034 on the r42 recipe with the 16384 output cap

**Status:** Running
<!-- gen-workflow:begin -->
**Date:** 2026-09-27
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `rl-glm53f44-`
**Experiment name:** `rl-glm53f-auct-cap-r44`
**W&B project:** `rl-glm53f-auct-cap` (group `rl-glm53f-auct-cap-r44`)
**Dataset:** `lakefs://arena-inspect/dev/internal/auctioneer/caponly/caponly-1034/manifest.jsonl` (gym `auctioneer-caponly`)
**Manifest commit:** `dev`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r15-20260927a`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `r42`
<!-- gen-workflow:end -->
**Argo workflow:** `rl-glm53f44-qvjnv` (resume 1, created 2026-09-27 14:16:02Z, phase `Running` at 2026-09-28 16:08Z). First attempt `rl-glm53f44-lt8vm` (created 05:07:02Z, deleted 10:02:37Z, no longer in the cluster).
**W&B run:** `82i3g7lv` (`rl-glm53f-auct-cap-r44_ni9gdjwj-RANK_0`, one run across both attempts; the W&B group carries the `_ni9gdjwj` suffix)
**Task pin:** `e5ef91b0…` (manifest rows pin this lakeFS commit; the URI ref `dev` is a branch). Source: `r39/BUILD.md` and the `r42/miles-config.yaml` comment. The full id is not re-verified: this host has no lakeFS client.
**Image digests:** gym `sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d`, trainer (resume 1, r15) `sha256:67b57cdf92695b4ec1e3c065cb803349745bc83858b28ef046d59da4cf960247`, trainer (launch, r14) `sha256:48d52a6a431257f9f7b40ce00debe9b2a10b61f94cee85d83291b21a2439e60c`. All three read from ECR `arena-slime-dev` (ap-south-1) on 2026-09-28.
**Trainer config deltas vs base:** `rollout_max_response_len` 32768 -> 16384; `replicas` 40 -> 16 (resume 1 only; a record key, the launcher drops it). Workflow parameters: `trainer-image` `miles-glm53-r12-20260925a` -> `miles-glm53-r14-20260927a` (launch) -> `miles-glm53-r15-20260927a` (resume 1); `gym-image` `gym-glm53-adr69-20260925a` -> `gym-glm53-adr72-20260927a`; template `guparpit-miles-deployer-v7` -> `v10`; `agent-kwargs` `{}` (r42 had `compaction-max` 2); `publish-jobs-dir` `''`. The non-comment diff of `r42/miles-config.yaml` against `r44/miles-config-r0.yaml` is `rollout_max_response_len` plus the run names only.
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-auct-cap-r44` (S3 `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/rl-glm53f-auct-cap-r44/`), last save `iter_0000049` (2026-09-28 14:59Z log line; newest object 15:01Z). Tracker reads 49. Saves at iter 9, 19, 29, 39, 49; 68 entries and 584 GiB each; each sidecar holds `wandb_run_id` `82i3g7lv`. `rollout/arena_data_source_state_{9,19,29,39,49}.pt` exist. `hf/rollout_9` is the only HF export (r14 attempt).
**Outcome:** Running. The reward per 5 rollouts rose from 0.35 (rollouts 0-9) to 0.54 (35-39), then fell to 0.52, 0.48 and 0.46 (40-54). Since the resume with 8 engines, 91-100% of the episodes end at the agent timeout, and the run makes 8 rollouts per 4 h.

## Goal

Test the 16384 per-call output cap that the trainer sends in the task
message (miles ADR-0015, AREnATasks ADR-0072) on the r42 auctioneer recipe.
Success: no SGLang 131072 window overflow (HTTP 400), no lost Vulcan
compaction notes, and a reward at or above r42 at the same rollout indices.
The run ends at `num_rollout` 300, on a user retire, or on a collapse like
r39 and r42 (r42 fell to 0.06 at rollouts 50-54).

## Setup

`BUILD.md` holds the generation commands, the reasons, and the two launch
checklists. The header block above shows the trainer image of
`workflow-resume1.yaml` (r15, generated with `--base r44`) next to the
`--base r42` of the launch `workflow.yaml`; the launch trainer image was
r14 (Image digests). r39 and r42 sent about 500 SGLang requests per 90 min that went
over the 131072 window (98K-131K input plus a fixed 32768 output budget).
About 99% were Vulcan summary calls, and 38-48% of the compactions lost
their handoff note (`BUILD.md`, "Why"). r44 moves the cap, the window and
the sampling values into the task message, and the gym clamps
`max_new_tokens` to the room left in the window (AREnATasks ADR-0063
amendment).

| Item | Base r42 | This run |
| --- | --- | --- |
| `rollout_max_response_len` | 32768 (gym copy `ARENA_MAX_TOKENS` 32768) | 16384, sent in the task message |
| Window | gym env `ARENA_ROLLOUT_CONTEXT_LIMIT` 131072 | task message `max_seq_len` 131072 |
| Output cap near the window end | fixed 32768 | clamped to the room left |
| Vulcan `max_compactions` | 2 (`compaction-max` 2) | 4 (`agent-kwargs` `{}`, Vulcan default) |
| Template | `guparpit-miles-deployer-v7` | `guparpit-miles-deployer-v10` (uid `16ddd530-501f-4734-82ae-88500671bed8`) |
| Trainer image | `miles-glm53-r12-20260925a` | `miles-glm53-r14-20260927a` (miles `e0987aed6`), then `miles-glm53-r15-20260927a` (miles `bc31f88ac`) |
| Gym image | `gym-glm53-adr69-20260925a` | `gym-glm53-adr72-20260927a` (AREnATasks mainline `538bc63`) |
| Trajectory publication | off | off (`publish-jobs-dir` `''`) |
| `replicas` / SGLang engines | 40 / 32 | 40 / 32 at launch; 16 / 8 from resume 1 |
| HF export per save | yes | yes on r14 (`hf/rollout_9`); no on r15 |

Same as r42: dataset caponly-1034 (1,034 tasks), radix cache and overlap
schedule on, `agent-timeout-multiplier` 2, `ack-wait` 36000,
`trainer-task-deadline-secs` 39600, `gym-replicas` 288, `excluded-nodes`
(83 instance ids), `lr` 1.5e-6, `rollout_batch_size` 64,
`n_samples_per_prompt` 8, `global_batch_size` 256,
`arena_inflight_multiplier` 4, `use_tis` true, W&B project
`rl-glm53f-auct-cap`.

Resume 1 (`workflow-resume1.yaml`, `miles-config.yaml`) differs from the
launch (`workflow.yaml`, `miles-config-r0.yaml`) in `trainer-image`,
`replicas`, and the config header only. The r14 to r15 image delta is one
functional commit, `bc31f88ac` (`--save-hf` opt-in); the other seven
commits in `e0987aed6..bc31f88ac` are docs. The resume starts from
`iter_0000009` with the data source at offset 937 of 1034, epoch 0, and
352 consumed instance ids (`BUILD.md`, "Checkpoint").

## Timeline

| UTC | Event |
| --- | --- |
| 2026-09-27 04:48:19Z | Template `guparpit-miles-deployer-v10` created (`BUILD.md`) |
| 2026-09-27 05:02:08Z / 05:05:52Z | r42 and r39 finished after `shutdown: Stop` (RUNLOG.md:2144-2149) |
| 2026-09-27 05:07:02Z | `kubectl create -f r44/workflow.yaml` -> `rl-glm53f44-lt8vm` (RUNLOG.md:2160) |
| 2026-09-27 05:10:03Z | `deploy-trainer-pinned`; Kueue admitted the PyTorchJob with no wait; worker-0 start 05:10:20Z (RUNLOG.md:2165) |
| 2026-09-27 05:12:42Z | Manifest pulled from `lakefs://arena-inspect/dev/.../caponly-1034/manifest.jsonl` (`trainer-0-attempt1.log`:2373) |
| 2026-09-27 05:24:47Z-05:25:12Z | 32 SGLang engines up (`trainer-0-attempt1.log`:4380-5002) |
| 2026-09-27 05:26:50Z | Rollout 0 collect start; complete 05:56:25Z, 1774.6 s, `avg_reward` 0.312 (log:5230, 6718) |
| 2026-09-27 06:58Z | Train step 0 end; `perf/step_time` 6041 s (W&B) |
| 2026-09-27 09:33:41Z | Checkpoint `iter_0000009` saved (log:22353); HF export and sidecar followed, sidecar 09:57:57Z (`BUILD.md`) |
| 2026-09-27 10:00:11Z | Rollout 11 complete, `avg_reward` 0.334 (log:24471), last rollout of attempt 1 |
| 2026-09-27 10:02:37Z | AREnAThanatos idle-GPU reaper deleted `rl-glm53f44-lt8vm` (`BUILD.md`; memory note; the audit-log query is not re-run) |
| 2026-09-27 10:04:10Z | Last write of the attempt-1 `trainer-0.log` in S3 |
| 2026-09-27 14:06:09Z | Trainer image `miles-glm53-r15-20260927a` pushed (ECR) |
| 2026-09-27 14:16:02Z | `rl-glm53f44-qvjnv` created (Argo `creationTimestamp`) |
| 2026-09-27 14:19:03Z | `deploy-trainer-pinned` (83 excluded nodes, `deploy-trainer` skipped); `wait-trainer-nats` 14:21:03Z-14:34:12Z; `deploy-gym-workers` 14:35:11Z (Argo nodes) |
| 2026-09-27 18:39:13Z | Rollout 10 sample summary written; `perf/rollout_time` 14,600 s (W&B) |
| 2026-09-27 23:20Z | `iter_0000019` saved (S3 newest object 23:20:19Z) |
| 2026-09-28 03:52Z | `iter_0000029` saved (S3) |
| 2026-09-28 08:44Z | `iter_0000039` saved (S3) |
| 2026-09-28 14:59:17Z | `iter_0000049` saved (live worker-0 log); S3 tracker 15:00:20Z |
| 2026-09-28 15:54:32Z | `Rollout 54: collecting 32 groups (... queue=127)` (live worker-0 log); summary written 15:58:25Z |
| 2026-09-28 16:08Z | Status read: workflow `Running`, PyTorchJob `Running`, 306 pods `Running` (16 trainer workers) |

## Results

### Reward per 5 rollouts

The r44 reward is the pooled mean of `reward` over all rows of
`debug/rl-glm53f-auct-cap-r44/sample_summary/rollout_<N>.jsonl` (S3,
1,280-1,334 rows per bin). It equals the W&B `rollout/raw_reward` per
rollout to 3 decimals. Length, turn, clip and stop columns are W&B means
over the bin. r42 (`ob9qvkyg`) and r46 (`4iu54zov`) are W&B
`rollout/raw_reward` bin means, read on 2026-09-28. The r44 bin means pool
the rows, so they differ in the third decimal from a mean of the per-rollout
W&B values; a W&B bin that includes the attempt-1 rollout-10 point (0.405)
reads 0.425 for 10-14 (`../../studies/idle-gpu-reaper-and-hf-export/STUDY.md`).

| Rollouts | r44 reward | `episode_response_length/mean` | `num_turns.mean` | `clipped_turns` | `stop/timeout` | r42 | r46 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-4 | 0.354 | 17.8K | 69.7 | 0 | 0 | 0.362 | 0.345 |
| 5-9 | 0.347 | 18.7K | 71.0 | 0 | 0 | 0.403 | 0.364 |
| 10-14 | 0.429 | 24.0K | 47.1 | 0 | 0.941 | 0.339 | 0.492 |
| 15-19 | 0.462 | 25.0K | 47.5 | 0 | 0.930 | 0.349 | 0.411 |
| 20-24 | 0.476 | 24.5K | 45.5 | 0 | 0.945 | 0.380 | 0.414 |
| 25-29 | 0.482 | 28.2K | 43.7 | 0 | 0.961 | 0.512 | 0.499 |
| 30-34 | 0.456 | 29.3K | 41.7 | 0.001 | 0.973 | 0.580 | 0.503 |
| 35-39 | 0.540 | 35.2K | 39.0 | 0.003 | 0.976 | 0.652 | 0.564 |
| 40-44 | 0.524 | 39.5K | 35.3 | 0.027 | 0.984 | 0.464 | 0.600 |
| 45-49 | 0.477 | 43.6K | 33.4 | 0.052 | 0.993 | 0.186 | 0.613 |
| 50-54 | 0.461 | 47.4K | 29.9 | 0.116 | 0.984 | 0.063 | 0.703 |

Per-rollout reward (sample summary means), five per row:

| From | +0 | +1 | +2 | +3 | +4 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.308 | 0.301 | 0.338 | 0.392 | 0.428 |
| 5 | 0.306 | 0.414 | 0.360 | 0.342 | 0.311 |
| 10 | 0.445 | 0.443 | 0.465 | 0.434 | 0.361 |
| 15 | 0.453 | 0.460 | 0.455 | 0.476 | 0.464 |
| 20 | 0.487 | 0.449 | 0.507 | 0.447 | 0.489 |
| 25 | 0.366 | 0.536 | 0.498 | 0.450 | 0.559 |
| 30 | 0.439 | 0.489 | 0.415 | 0.456 | 0.481 |
| 35 | 0.569 | 0.580 | 0.476 | 0.530 | 0.543 |
| 40 | 0.473 | 0.486 | 0.515 | 0.545 | 0.598 |
| 45 | 0.525 | 0.486 | 0.458 | 0.481 | 0.435 |
| 50 | 0.393 | 0.447 | 0.415 | 0.552 | 0.495 |

Trend: the peak bin is 35-39 (0.540). The next three bins fall to 0.524,
0.477 and 0.461. Rollouts 50-52 (0.393, 0.447, 0.415) pull the last bin
down; rollouts 53 and 54 read 0.552 and 0.495. The fall is not settled.
r42 at the same indices collapsed (0.186, 0.063); r44 has not. r46 rose to
0.703, but r46 replayed epoch 0 from offset 0 after its own resume, so the
r44/r46 pair is confounded (memory note `r44-r45-launch-plan-2026-09-27.md`,
line 20; not re-verified here).

Attempt 1 rollouts 10 and 11 read 0.407 and 0.334 (`avg_reward` in
`trainer-0-attempt1.log` lines 21701, 24471; W&B `rollout/raw_reward` reads
0.405 for rollout 10). The resume produced them again (0.445, 0.443) and
overwrote the two sample summary files. W&B holds both points at steps 10
and 11.

### Train metrics (W&B `82i3g7lv`)

| Step | `grad_norm` | `ppo_kl` | `pg_clipfrac` | `ess_ratio` | `tis_clipfrac` | `train_rollout_kl` | `train_rollout_logprob_abs_diff` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.140 | 2.3e-5 | 2.6e-4 | 1.015 | 4.5e-4 | 0.0034 | 0.032 |
| 9 | 0.143 | 2.1e-5 | 2.7e-4 | 0.999 | 1.8e-3 | 0.0102 | 0.060 |
| 10 (resume) | 0.114 | -3.8e-6 | 2.9e-4 | 0.999 | 1.0e-3 | 0.0070 | 0.053 |
| 25 | 0.104 | -2.7e-5 | 2.4e-4 | 0.998 | 2.9e-3 | 0.0132 | 0.074 |
| 40 | 0.085 | 8.8e-6 | 1.5e-4 | 0.999 | 3.0e-3 | 0.0121 | 0.072 |
| 52 | 0.065 | -1.5e-5 | 1.1e-4 | 1.022 | 9.1e-4 | 0.0062 | 0.054 |

`train/step` 52 is the last train step at 16:08Z (`rollout/step` 53 in the
W&B summary). `grad_norm` fell from 0.14 to 0.065 and `pg_clipfrac` from
2.7e-4 to 1.1e-4 over the run. `ppo_kl` stays within 3e-5 of zero on every
step. `train_rollout_kl` climbs through each 8-step wave (0.0070 at step 10
to 0.0132 at step 25; 0.0091 at step 26 to 0.0143 at step 33) and drops one
step after each wave start (steps 26, 34, 42, 50).

### Throughput and health

| Metric | Value | Source |
| --- | --- | --- |
| Rollout 0 (32 engines) | first group 05:48:00Z, complete 1774.6 s, 32/32 groups, 256 samples in the log line (260 `num_training_samples` in W&B), `episode_raw_reward` 0.312, `truncated_ratio` 0 | RUNLOG.md:2178; log:6718 |
| Rollout collect time, attempt 1 rollouts 1-11 | 111-134 s with `queue=320` finished groups | log:7022-24471 |
| `perf/step_time`, attempt 1 steps 1-9 | 812-1389 s; `train_wait_time` 53-70 s; `actor_train_time` 616-1026 s; `log_probs_time` 135-310 s | W&B |
| `perf/actor_train_mfu` | 0.029-0.051 (attempt 1), 0.030-0.066 (resume) | W&B |
| `perf/step_time`, resume, inside a wave | 750-1350 s; `train_wait_time` 45-75 s | W&B |
| `perf/train_wait_time` at a wave start | 15,095 s (step 10), 4,702 (17), 8,068 (25), 7,601 (33), 6,287 (41), 5,885 (49) | W&B |
| `perf/save_model_time` (DCP only, r15) | 101-120 s at steps 19, 29, 39, 49 | W&B |
| `perf/update_weights_time` | 19-25 s each step; 44 s at step 0, 41 s at step 10 | W&B |
| Rollout cadence, attempt 1 | rollouts 2-10: 14-23 min apart; rollout 11: 40 min, across the step-9 save and HF export | sample summary S3 times; log:21701, 24471 |
| Rollout cadence, resume | 8 rollouts per wave; wave ends at rollouts 17, 25, 33, 41, 49 at 22:38:39Z, 02:39:33Z, 06:40:16Z, 10:41:16Z, 14:42:10Z (4 h 00-01 min apart); 30 min per rollout on average | sample summary S3 times |
| `rollout/stop/timeout` | 0 in rollouts 0-9 (`stop/unknown` 1.0: the gym sent no stop reason); 0.905-1.000 in rollouts 10-54 | W&B |
| `rollout/queue_depth_at_start` in a wave | 223, 191, 159, 127, 95, 63, 31, 14-27, then 223 again | W&B |
| `rollout/off_policy_round/mean`, `weight_version/mixed_version_ratio` | 0-10 and 0 in attempt 1; 15-22 and 0.98-1.00 from rollout 18 | W&B |
| `rollout/clipped_turns`, `masked_output_tokens` | 0 through rollout 33; 0.004-0.008 (64-128 tokens) at 34-39; 0.03-0.08 (448-1280) at 42-48; 0.08-0.16 (1,350-2,624) at 50-54 | W&B |
| `rollout/dyn_sampling_dropped`, `dropped_groups/*`, `truncated_ratio`, `context_overflow_frac` | 0 on every rollout | W&B |
| `failed_frac` | 0 except 0.004 (5), 0.023 (9), 0.016 (16), 0.004 (34, 53) | W&B; sample summary `status` `failed` rows 1-6 |
| SGLang overflow, attempt 1 | 0 "Requested token count exceeds", 0 `POST /generate` 400, 0 "maximum context length" in `trainer-0-attempt1.log` | grep of the 4.99 MB S3 file; RUNLOG.md:2181 |
| Vulcan compactions, 07:00Z | 180 lines, all `1/4`; 177 `(count, summary)`, 3 `(tokens, summary)`, 0 `structural`: no lost handoff note (r39, r42: 38-48% lost) | RUNLOG.md:2182 |
| Gym limits | Harbor `config.json`: `max_tokens` 16384, `context_limit` 131072, `temperature` 1.0, `top_p` 1.0 | RUNLOG.md:2177 |
| Gym env | `DOCKER_CONFIG=/root/.docker`, `HARBOR_AGENT_KWARGS={}`, `ARENA_PUBLISH_JOBS_DIR` empty, `ARENA_TOOL_CALL_PARSER=glm47`; none of the seven removed ADR-0072 names | RUNLOG.md:2175 |

## Issues

- **Idle-GPU reaper deleted attempt 1.** AREnAThanatos deletes a workload
  when the 60-min mean normalized GPU power is under 10%. At the 16384 cap,
  32 engines refilled the queue in about 2 min (`queue=320` on every
  rollout), then idled through each 14-23 min train step. Each save also
  wrote an HF export for about 25 min with all GPUs idle (`iter_0000009`
  saved 09:33:41Z, sidecar 09:57:57Z). Fix (user, 2026-09-27 about 14:00Z):
  resume with 8 engines (`replicas` 16) and no per-save HF export (trainer
  r15, miles `bc31f88ac`, plugin ADR-0006 amendment). Source: `BUILD.md`
  "Resume 1"; workflow `wf_2776ce0d-bc7`.
- **Since the resume, nearly every episode ends at the agent timeout.**
  `rollout/stop/timeout` is 0 in rollouts 0-9 and 0.905-1.000 in rollouts
  10-54. The gym scales the task `agent.timeout_sec` by
  `ARENA_AGENT_TIMEOUT_MULTIPLIER` 2 (template v10 lines 2637-2638;
  `gym_worker.py:659-683`). The auctioneer task declares 7200 s (memory note
  `auctioneer-caponly-r30-and-limits.md`:34; not re-verified from
  `task.toml`), so the limit is 4 h. Rollout 10 completed 4 h 4 min after
  `deploy-gym-workers`, and each later wave of 256 groups (the
  `arena_inflight_multiplier` 4 x 64 cap) returns 4 h 00-01 min after the
  previous one. The trainer then trains 8 steps in about 2 h and waits
  about 2 h (`train_wait_time` 4,702-15,095 s). Effects: 30 min per rollout
  against 15-17 min in attempt 1; `num_turns.mean` 70 -> 47 at the resume;
  all samples span 15-22 weight versions (`mixed_version_ratio` 1.0; TIS
  absorbs it, `tis_clipfrac` 1e-3 to 3e-3); the reward jump 0.31-0.34 ->
  0.43-0.46 at rollout 10 coincides with the stop-reason change, so the two
  attempts are not one comparable series. The 4 h interpretation rests on
  the metric name and the wave times; the gym logs (`AgentTimeoutError`
  counts) were not read.
- **Length growth and clipped turns since rollout 34.**
  `episode_response_length/mean` 24K (rollouts 10-14) -> 35K (35-39) ->
  47K (50-54); `num_turns.mean` 47 -> 30; `clipped_turns` 0 -> 0.16 at
  rollout 54 (2,624 masked tokens). r39 and r42 showed the same length
  growth before their collapse (r42 reward 0.652 at 35-39 -> 0.063 at
  50-54). r44 reads 0.461 at 50-54, so no collapse yet, but the bins fall
  since 35-39.
- **The resume rank-0 log is not in S3.** `logs/rl-glm53f-auct-cap-r44/trainer-0.log`
  (S3 mtime 2026-09-27 10:04:10Z, 4,986,851 bytes) is byte-identical in
  size to `trainer-0-attempt1.log` (the copy-aside from `BUILD.md`,
  14:16:56Z). The live pod log is rotated: `kubectl logs` on
  `rl-glm53f44-qvjnv-trainer-worker-0` at 16:08Z returned only the recent
  tail. The resume evidence in this record comes from W&B (one run id,
  history continues at step 10) and from the S3 checkpoint sidecars
  (`wandb_run_id` `82i3g7lv` in `iter_0000019` to `iter_0000049`). The
  lines `Restored wandb_run_id=82i3g7lv` and `Resume: loaded 352 consumed
  instance_ids` are not re-verified.
- **W&B duplicates.** Steps 10 and 11 hold two points each (attempt 1:
  0.407 and 0.334; resume: 0.445 and 0.443). Read the later point.
- **`trainer-1.log` to `trainer-39.log` in S3** mix both attempts: ranks
  1-15 are from the resume (14:22Z), ranks 16-39 from attempt 1 (05:12Z).
  Each is 4.6-5.0 KB and holds startup lines only.

## Follow-ups

- Decide the engine count for auctioneer at the 16384 cap. Eight engines
  make every episode hit the 4 h agent limit and halve the rollout rate.
  Candidates: 16 engines (`replicas` 24), a lower `arena_inflight_multiplier`
  (fewer concurrent episodes per engine), or a lower
  `agent-timeout-multiplier`. Confirm the timeout cause in the gym logs
  first (count `AgentTimeoutError` on one gym pod of `rl-glm53f44-qvjnv`).
  Owner: user decision; record in a study under `training-runs/studies/`.
- Watch the reward at rollouts 55-64 against r42 (collapsed at 45-54) and
  the `clipped_turns` rise. A bin under 0.40 with `clipped_turns` over 0.2
  repeats the r39/r42 pattern.
- Record the full `lakefs_commit_id` of one manifest row as the task pin.
  Needs lakeFS access from a pod or a lakeFS client.
- Compare r44 and r46 only after the r46 epoch-0 replay is quantified
  (r46 resumed from offset 0; r44 from offset 937).
- Idle-GPU reaper and HF-export findings:
  `../../studies/idle-gpu-reaper-and-hf-export/STUDY.md` (Closed). Loss
  A/B against r46: `../../studies/token-level-loss-average-r44-vs-r46/STUDY.md`
  (Open).

## Sources

- `training-runs/harbor-rl-glm53-flash/r44/BUILD.md`, `workflow.yaml`,
  `workflow-resume1.yaml`, `miles-config-r0.yaml`, `miles-config.yaml`,
  `guparpit-miles-deployer-v10.yaml`
- `training-runs/harbor-rl-glm53-flash/RUNLOG.md` lines 2142-2190 (launch and
  health), 2271-2320 (r46 against r44), 2401 (not touched on 2026-09-28)
- W&B: https://mega.wandb.agi.amazon.dev/arena/rl-glm53f-auct-cap/runs/82i3g7lv
  (332 history rows, read 2026-09-28 16:05Z); r42 `ob9qvkyg`; r46 `4iu54zov`
- S3 `s3://arena-scratch-prod-bom-ap-south-1/guparpit/debug/rl-glm53f-auct-cap-r44/sample_summary/rollout_{0..54}.jsonl`
  (55 files, 256-270 rows each; `reward`, `response_length`, `stop_reason`)
- S3 `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/rl-glm53f-auct-cap-r44/`
  (tracker, `iter_*`, `rollout/`, `hf/`)
- S3 `s3://arena-scratch-prod-bom-ap-south-1/guparpit/logs/rl-glm53f-auct-cap-r44/trainer-0-attempt1.log`
- Argo `rl-glm53f44-qvjnv` (`kubectl get workflow -o json`, context
  `arena-prod-bom-v2`, namespace `arena-tasks`, 2026-09-28 16:0xZ)
- ECR `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev`
  (`describe-images` for the three tags)
- miles ADR-0015 (`miles_plugins/arena/adr/0015-task-message-carries-output-cap-window-and-sampling.md`),
  plugin ADR-0006 amendment 2026-09-27 (HF export opt-in),
  `examples/arena/README.md` image rows r14, r15, adr72
- AREnATasks ADR-0063 (amendment: clamp to the room left), ADR-0072
  (`adr/0072-training-gym-takes-limits-from-the-task-message.md`);
  `src/amzn-arena-harbor/src/amzn_arena_harbor/gym_worker.py:659-683`
- Code reviews: AREnATasks CR-308323817, CR-308323860; Apps CR-308323875
- Workflow journal `wf_2776ce0d-bc7` (8-engine resume and HF-export-off)
- Memory notes (facts with dates, secondary): `r44-r45-launch-plan-2026-09-27.md`,
  `thanatos-idle-gpu-reaper-prod-bom-v2.md`, `r42-radix-overlap-timeout-2026-09-25.md`,
  `auctioneer-caponly-r30-and-limits.md`, `r39-auctioneer-caponly-1034-2026-09-25.md`
