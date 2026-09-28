# Run record: r45 — agentic-debt v3 resumed from r43 step 39 on the ADR-0015 images

**Status:** Retired
<!-- gen-workflow:begin -->
**Date:** 2026-09-27
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `rl-glm53f45-`
**Experiment name:** `rl-glm53f-adebt-v3-r45`
**W&B project:** `rl-glm53f-adebt-v3` (group `rl-glm53f-adebt-v3-r45`)
**Dataset:** `/mnt/scratch-s3files-rw/guparpit/data/agentic-debt/20260923-v3-locked-oracle/manifest-le5.jsonl` (gym `agentic-debt`)
**Manifest commit:** none (not a lakeFS URI)
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r14-20260927a`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `r43`
<!-- gen-workflow:end -->
**Argo workflow:** `rl-glm53f45-vvqhg` (second try; the first try `rl-glm53f45-54bb8` lost its trainer to two node failures and was stopped as a zombie)
**W&B run:** `ajsur4ej` (r43 ran as `2z0599lb`; r45 opened its own run)
**Task pin:** `77c2239b6d97b08248d294ec8cb7be73fa63d1a63e7382e122e6e2267508e703` on all 551 manifest rows; each `lakefs_uri` starts with `lakefs://arena-inspect/acuadron-v3-oracle-validation-20260924/internal/agentic-debt-r3/20260923-v3-locked/tasks/` (manifest copy read from S3 2026-09-28, md5 `54b7bdc0db049307f05a630923407927`)
**Image digests:** gym `sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d`, trainer `sha256:48d52a6a431257f9f7b40ce00debe9b2a10b61f94cee85d83291b21a2439e60c` (ECR `427267593057` `arena-slime-dev`, ap-south-1 and us-east-1, read 2026-09-28)
**Trainer config deltas vs base:** `rollout_max_response_len` 32768 -> 16384; `experiment_name`, `project_name`, `arena_sample_summary_dir` name r45 instead of r43. No other non-comment line of `miles-config.yaml` differs from `r43/miles-config.yaml`.
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-adebt-v3-r45` (S3 `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/rl-glm53f-adebt-v3-r45/`), last save `iter_0000049` (2026-09-28 00:25Z DCP, 00:47Z sidecar and data source state, 00:47Z `hf/rollout_49/.complete`). Sidecar `{"rollout_id": 49, "wandb_run_id": "ajsur4ej"}`. The seed `iter_0000039` (links to the r43 files) is also in the dir.
**Outcome:** Retired 2026-09-28 09:23Z on the user's go, to free its 40 nodes for r47 (`rl-glm53f47-wxj87`, agentic-debt-766). r45 trained 15 optimizer steps (40 to 54) over 16 rollouts (40 to 55) in 27.5 h. The reward stayed flat at about 0.68, the same level as r43 rollouts 30 to 41. The 16384 output cap removed the SGLang window overflow: 0 "Requested token count exceeds" lines, against 3,392 on r43. Steps 50 to 54 came after the last save and exist only in W&B. The run also trained on the wrong dataset (see Issues).

## Goal

Continue the r43 policy, data position, and RNG state on the new trainer and gym images (miles ADR-0015, AREnATasks ADR-0072), with the output cap lowered from 32768 to 16384 and the Vulcan `max_compactions` raised from 2 to 4. The run ends when the agentic-debt run on the correct dataset (r47) is ready.

## Setup

r45 is a resume, not a fresh start. `r45/BUILD.md` holds the seed commands and the resume mechanics. The seed made a real `iter_0000039` dir with 66 relative links to the r43 shards and metadata, its own sidecar `{"rollout_id": 39}` (no W&B id, so r45 opened a new run), a copy of `rollout/arena_data_source_state_39.pt`, and a tracker file that reads 39 (RUNLOG 2026-09-27 r45 entry; S3 listing 2026-09-28: `iter_0000039/` 64 shards plus `.metadata` 51 B, `metadata.json` 55 B, sidecar 18 B, all written 05:33-05:34Z).

| Item | Base r43 | This run |
| --- | --- | --- |
| Start point | base DCP `ref_load` (`finetune` True, `start_rollout_id` 0) | r43 `iter_0000039` seeded into the r45 dir (`finetune` False; `successfully loaded checkpoint ... at iteration 39`) |
| Rollout ids, data position | 0..41, row 0 | 40..55, r43 offset at 39 (`Resume: loaded 1312 consumed instance_ids to skip`) |
| W&B run | `2z0599lb` | `ajsur4ej` (no `Restored wandb_run_id` line) |
| `rollout_max_response_len` | 32768 (gym env `ARENA_MAX_TOKENS` 32768) | 16384, sent in the task message |
| Window | gym env `ARENA_ROLLOUT_CONTEXT_LIMIT` 131072 | task message `max_seq_len` 131072; `max_new_tokens` clamped to the room left |
| Vulcan `max_compactions` | 2 (`compaction-max` 2) | 4 (`agent-kwargs` `{}`) |
| Template | `guparpit-miles-deployer-v9` | `guparpit-miles-deployer-v10` (uid `16ddd530-501f-4734-82ae-88500671bed8`) |
| Trainer image | `miles-glm53-r13-20260925a` (miles `dd5e67391`) | `miles-glm53-r14-20260927a` (miles `e0987aed6`) |
| Gym image | `gym-glm53-adr71-20260925a` (AREnATasks `ff880da`) | `gym-glm53-adr72-20260927a` (AREnATasks mainline `538bc63`) |
| Trajectory publication | off (v9 has no publish env) | off (`publish-jobs-dir` `''`) |
| `step-cut-on-fail` parameter | `false` | removed (v10 declares none) |
| Optimizer state | not saved (`no_save_optim` true) | not loaded (`no_load_optim` true); Adam moments restart at zero |

Same as r43 (`r45/miles-config.yaml`, non-comment keys): `lr` 1.5e-6 constant, `advantage_estimator` grpo, `n_samples_per_prompt` 8, `global_batch_size` 256, `rollout_batch_size` 64, `use_tis` true, `use_rollout_routing_replay` true, `eps_clip` 0.2 / `eps_clip_high` 0.28, `kl_coef` 0, `arena_inflight_multiplier` 4, `arena_train_segments` all, `arena_length_reward_coef` 0, `save_interval` 10, radix cache on (`sglang_disable_radix_cache` false), overlap schedule off, TP 8, PP 4, EP 16, `max_tokens_per_gpu` 8192, `recompute_granularity` full. Workflow parameters (`r45/workflow.yaml`): `replicas` 40, `replica-trainer` 8, `gym-replicas` 288, `ack-wait` 86000, `agent-timeout-multiplier` 4, `trainer-task-deadline-secs` 90000, `excluded-nodes` = the r43 list (no listed node existed on 2026-09-27).

The trainer log shows `Rollout N: collecting 32 groups (GBS=256, n_samples=8)` and one optimizer step per rollout. The config comment that promises two steps per rollout is wrong (MFU study, `wf_545a45a0-6ca`).

## Timeline

| UTC | Event |
| --- | --- |
| 2026-09-27 03:46:50 | Trainer image `miles-glm53-r14-20260927a` pushed (ECR `imagePushedAt`) |
| 2026-09-27 04:44:57 | Gym image `gym-glm53-adr72-20260927a` pushed (ECR) |
| 2026-09-27 04:48:19 | Template `guparpit-miles-deployer-v10` created (`r44/BUILD.md`) |
| 2026-09-27 05:32 | Pre-seed checks: r43 tracker 39, 67 entries in `iter_0000039`, sidecar `2z0599lb` (RUNLOG) |
| 2026-09-27 05:33:22 | Seed of the r45 dir from r43 trainer-worker-0 (RUNLOG; S3 objects 05:33:28-05:34:31) |
| 2026-09-27 05:33:41 | r43 `shutdown: Stop`; phase `Failed` "Stopped with strategy 'Stop'" at 05:36:44 (RUNLOG); r43 log ends 05:36:19 |
| 2026-09-27 05:38:31 | First launch `rl-glm53f45-54bb8` created; Kueue admitted 05:41:41; `EvictedDueToNodeFailures` on `i-055b073cb40c3ac18` and `i-0b1fc7110a2f1ddda` at 05:41:44; zombie stopped 05:50:30, teardown done 05:53:47 (RUNLOG) |
| 2026-09-27 05:53:57 | Second launch `rl-glm53f45-vvqhg` created; Kueue admitted 05:57:09 (RUNLOG) |
| 2026-09-27 05:58:35 | Trainer reads the seed sidecar: `Checkpoint sidecar rollout_id=39.` (trainer log line 1994) |
| 2026-09-27 05:58:41 | W&B run `ajsur4ej` created (W&B API `created_at`) |
| 2026-09-27 06:11:14 | First SGLang engine "fired up" (trainer log line 4227) |
| 2026-09-27 06:12:22 | `Loading arena data source state from .../arena_data_source_state_39.pt` (line 4916) |
| 2026-09-27 06:13:07 | First weight sync `update_weights ok=true` 44.7 s (line 4968) |
| 2026-09-27 06:13:08 | `Rollout 40: collecting 32 groups`, `NATS connected` (line 4980) |
| 2026-09-27 06:17 | Gym Deployment 288/288 (RUNLOG) |
| 2026-09-27 06:55:53 | First `status=success` result (line 7063) |
| 2026-09-27 07:05:53 | One gym `dind` container OOMKilled, pod `rl-glm53f45-vvqhg-gym-756ff88f8c-c76sd` (RUNLOG) |
| 2026-09-27 10:02:36 | Rollout 40 complete after 13,768 s, `avg_reward` 0.704 (line 16238) |
| 2026-09-27 12:30:30 | Step 40 end; `perf/step_time` 23,162 s of which `train_wait_time` 14,450 s (line 27280) |
| 2026-09-28 00:25:36 | `successfully saved checkpoint from iteration 49` (line 64128); HF export done 00:46:44 (line 66426); data source state 49 written 00:47:52 (S3) |
| 2026-09-28 07:49:35 | Rollout 55 complete after 4,602 s (line 87439) |
| 2026-09-28 08:08:46 | Step 54 end, the last trained step (metrics table) |
| 2026-09-28 08:09:28 | `Rollout 56: collecting 32 groups (... queue=320)` (line 88399) |
| 2026-09-28 09:22 | Trainer in `actor_train` of step 55; next save at 59; no save in progress (RUNLOG r47 entry) |
| 2026-09-28 09:23:31 | `shutdown: Stop`; phase `Failed` "Stopped with strategy 'Stop'" 09:27:32; W&B last heartbeat 09:27:01; no PyTorchJob, Deployment, Service, or pod left at 09:27:57 (RUNLOG; W&B API) |
| 2026-09-28 09:45:34 | r47 `rl-glm53f47-wxj87` created on the freed nodes (RUNLOG) |

## Results

All rows below come from `trainer-0.log` (S3 `guparpit/logs/rl-glm53f-adebt-v3-r45/trainer-0.log`, 21.6 MB): the `Rollout N complete` lines, the RolloutManager `perf N` dicts, and the actor `step N` and `perf N` dicts. The W&B API returns the same 16 reward points and 15 `train/grad_norm` points for `ajsur4ej` (checked 2026-09-28).

### Rollouts 40 to 55

| Rollout | Done (UTC) | Collect time (s) | `episode_raw_reward` | Episode length mean / max (tokens) | Train rows | Groups examined / dropped zero-variance |
| --- | --- | --- | --- | --- | --- | --- |
| 40 | 09-27 10:02 | 13,768 | 0.704 | 87,867 / 192,682 | 758 | 103 / 71 |
| 41 | 09-27 10:45 | 2,537 | 0.636 | 118,548 / 263,362 | 920 | 59 / 27 |
| 42 | 09-27 12:41 | 610 | 0.771 | 166,438 / 352,364 | 1,134 | 75 / 43 |
| 43 | 09-27 13:33 | 652 | 0.623 | 181,864 / 418,928 | 1,188 | 75 / 43 |
| 44 | 09-27 14:48 | 677 | 0.742 | 178,235 / 459,512 | 1,224 | 78 / 46 |
| 45 | 09-27 16:10 | 840 | 0.687 | 228,902 / 545,555 | 1,398 | 66 / 34 |
| 46 | 09-27 17:25 | 702 | 0.703 | 187,192 / 627,194 | 1,258 | 72 / 40 |
| 47 | 09-27 19:04 | 744 | 0.662 | 190,650 / 746,395 | 1,262 | 68 / 36 |
| 48 | 09-27 20:31 | 726 | 0.708 | 198,522 / 621,404 | 1,300 | 72 / 40 |
| 49 | 09-27 21:50 | 637 | 0.623 | 161,670 / 840,335 | 1,132 | 78 / 46 |
| 50 | 09-27 23:18 | 706 | 0.620 | 198,188 / 765,144 | 1,256 | 72 / 40 |
| 51 | 09-28 01:00 | 799 | 0.687 | 224,341 / 727,718 | 1,414 | 78 / 46 |
| 52 | 09-28 02:49 | 2,050 | 0.719 | 191,593 / 621,706 | 1,246 | 64 / 32 |
| 53 | 09-28 04:03 | 701 | 0.702 | 188,703 / 785,799 | 1,262 | 66 / 34 |
| 54 | 09-28 05:24 | 841 | 0.666 | 216,566 / 667,958 | 1,326 | 57 / 25 |
| 55 | 09-28 07:49 | 4,602 | 0.662 | 218,431 / 1,022,969 | 1,402 | 72 / 40 |

Every rollout kept 32 of 32 groups, 256 samples, `truncated_ratio` 0.0, and 0 groups dropped for a lost routing ref. A "zero-variance group" is a group of 8 samples with one identical reward; the dynamic-sampling filter drops it because it carries no gradient. r45 examined 1,155 groups and dropped 643 (56%). Mean reward over rollouts 40 to 55: 0.682. Mean over r43 rollouts 30 to 41: 0.681.

### Train steps 40 to 54

| Step | End (UTC) | `grad_norm` | `ppo_kl` | `pg_clipfrac` | `train_rollout_logprob_abs_diff` | `train_rollout_kl` | `log_probs` (s) | `actor_train` (s) | `step_time` (s) | Logged `actor_train_mfu` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40 | 09-27 12:30 | 0.054 | -5.3e-6 | 1.3e-4 | 0.042 | 0.0046 | 2,390 | 6,321 | 23,162 | 0.017 |
| 41 | 09-27 13:22 | 0.055 | -4.9e-6 | 1.3e-4 | 0.041 | 0.0044 | 557 | 2,388 | 3,116 | 0.063 |
| 42 | 09-27 14:36 | 0.042 | -3.0e-6 | 1.3e-4 | 0.041 | 0.0044 | 806 | 3,416 | 4,463 | 0.065 |
| 43 | 09-27 15:55 | 0.048 | -4.7e-6 | 1.4e-4 | 0.040 | 0.0042 | 914 | 3,606 | 4,728 | 0.071 |
| 44 | 09-27 17:13 | 0.049 | 1.9e-6 | 1.2e-4 | 0.039 | 0.0040 | 897 | 3,600 | 4,663 | 0.068 |
| 45 | 09-27 18:51 | 0.037 | -9.9e-6 | 1.3e-4 | 0.040 | 0.0042 | 1,154 | 4,506 | 5,902 | 0.074 |
| 46 | 09-27 20:18 | 0.080 | -3.6e-6 | 1.3e-4 | 0.040 | 0.0041 | 923 | 4,022 | 5,209 | 0.067 |
| 47 | 09-27 21:39 | 0.044 | -5.9e-7 | 1.3e-4 | 0.038 | 0.0039 | 916 | 3,676 | 4,835 | 0.074 |
| 48 | 09-27 23:05 | 0.057 | -7.1e-6 | 1.4e-4 | 0.040 | 0.0042 | 986 | 3,997 | 5,203 | 0.070 |
| 49 | 09-28 00:22 | 0.049 | -1.2e-5 | 1.4e-4 | 0.039 | 0.0041 | 866 | 3,521 | 4,618 | 0.066 |
| 50 | 09-28 02:14 | 0.046 | 2.6e-6 | 1.5e-4 | 0.042 | 0.0044 | 1,017 | 4,053 | 6,696 | 0.072 |
| 51 | 09-28 03:51 | 0.041 | -1.3e-5 | 1.4e-4 | 0.043 | 0.0046 | 1,049 | 4,340 | 5,811 | 0.075 |
| 52 | 09-28 05:09 | 0.046 | -2.1e-7 | 1.3e-4 | 0.044 | 0.0047 | 826 | 3,608 | 4,725 | 0.074 |
| 53 | 09-28 06:32 | 0.044 | 1.7e-6 | 1.3e-4 | 0.043 | 0.0046 | 913 | 3,863 | 4,947 | 0.070 |
| 54 | 09-28 08:08 | 0.045 | 1.6e-6 | 1.4e-4 | 0.044 | 0.0047 | 1,196 | 4,382 | 5,780 | 0.072 |

`lr` 1.5e-6 on every step. Steps 41 to 54: `step_time` 3,116 to 6,696 s, mean 5,050 s; `actor_train` mean 3,784 s; `log_probs` mean 930 s; `update_weights` 24.0 to 25.7 s. Step 50 waited 1,625 s for the save of iteration 49 (`train_wait_time`).

### Comparisons and analyses

| Measure | Value | Source |
| --- | --- | --- |
| SGLang "Requested token count exceeds" lines, whole run | r45 0; r43 3,392 (32768 cap) | both `trainer-0.log` files, `grep -c` |
| `POST /generate` HTTP 400 lines | r45 2; r43 3,394 (any 400) | same |
| Result statuses in the trainer log | 2,701 `status=success`, 8 `status=failed`, 24 `HealthcheckError` lines, 0 `deadline exceeded`, 0 `too_large` | r45 `trainer-0.log` |
| Finished-group queue at rollout start | 0, 1, 161, 155, 173, 204, 243, 280, 319, then 320 (the cap) from rollout 49 on | `Rollout N: collecting ... queue=` lines |
| Rows per episode (`segment_k`), sample summary | rollout 40: 758 rows / 256 episodes = 2.96 (max 9); rollout 55: 1,402 / 256 = 5.48 (max 19) | `debug/rl-glm53f-adebt-v3-r45/sample_summary/rollout_{40,55}.jsonl` |
| Episodes at reward 1.0 / at 0.0 | rollout 40: 157 (61%) / 54; rollout 55: 141 (55%) / 54 | same |
| Episodes with a `context_error` row | rollout 40: 0; rollout 55: 7 (3%) | same |
| Per-row `response_length` P50 / P90 | rollout 40: 46,078 / 93,198; rollout 55: 78,756 / 97,446 | same |
| Logged vs true `actor_train` MFU, steps 41 and 42 | logged 6.3% / 6.5%; true 1.70% / 1.65%; step 40 true 0.47% (not steady state) | MFU study `wf_545a45a0-6ca`, verified by two agents; ADR-0016 Context |
| Share of executed training FLOPs in the replicated KDA layers | 62-64% (all 64 heads on all 8 TP ranks) | `wf_545a45a0-6ca`; ADR-0016 |
| Step 42 time breakdown (4,463 s) | TIS old-logprob forward 806 s (measured); full recompute about 782 s (inferred); TileLang/JIT stalls 360 s; PP stage imbalance about 300 s; gap between steps 241 s | memory note `r45-mfu-investigation-2026-09-27` from `wf_545a45a0-6ca`; scratch `/tmp/mfu` not in git, on tmpfs, present at 16:35Z 2026-09-28, not re-read here, so the 360 s and 300 s items are not re-verified |
| Step-level advantage A/B on r45 | not worth it: no dropped group becomes trainable; GiGPO-style B flips the sign on about 0.6% of tokens; per-step C zeroes 45% of trained tokens; 80-90% of zero-variance groups are all-pass | `wf_11893a4c-2b9`, verified; scratch `/tmp/stepadv` not in git, on tmpfs, present at 16:35Z 2026-09-28 |
| r45 vs frontier on the 344 tasks shared with agentic-debt-766 (trained-on, not held out) | mean reward r45 0.72, Opus 0.633, GPT 0.51; all-pass 0.623 / 0.498 / 0.298. The early 91-task snapshot subset read 0.700 / 0.643 / 0.532 (all 8 at 1.0 on 28 / 23 / 6 tasks) | all-pass audit `wf_e6084655-6ed`, verify pass; `../../studies/agentic-debt-all-pass-audit/STUDY.md` |
| Reward hack in the all-pass groups | none found: grader files identical across 343 fingerprints on 7 tasks, 0 writes to reward, verifier, or seal paths in 465 step transcripts | `wf_e6084655-6ed`; memory note `adebt-dataset-must-be-766-main` |
| Trials that probed the grading setup | 42 of 176 r45 trials vs 0 of 160 frontier trials; none succeeded | memory note `adebt-dataset-must-be-766-main`; not re-verified |
| Manifest overlap with agentic-debt-766 | 348 of the 551 r45 manifest tasks are in the 766 set | `wf_e6084655-6ed` |
| kdatp T2 projection on r45 step 42 | `actor_train` 3,416 -> 2,690 s (-21.3%) with the first KDA port; -24.5% with block recompute of 10 layers | `examples/arena/harbor-rl-glm53-flash/kdatp/RESULTS.md` |

## Issues

- **Wrong dataset.** r41 to r45 trained on the acuadron oracle-validation branch, commit `77c2239b`, filtered to 551 chains with at most 5 steps (`manifest-le5.jsonl`). The user's dataset is `lakefs://arena-inspect/main/internal/agentic-debt-r3/agentic-debt-766/` (766 chains, K 2 to 66, mean 7.9). The easier subset is why r45 looked far better than the frontier models and why 55-61% of its episodes scored 1.0. User decision 2026-09-27: keep r45 until the run on the correct dataset is ready, then stop it. r47 is that run, fresh from the base model (memory note `adebt-dataset-must-be-766-main`; `r47/BUILD.md`; RUNLOG 2026-09-28).
- **First launch was a zombie.** `rl-glm53f45-54bb8` lost its PyTorchJob 3 s after admission (`EvictedDueToNodeFailures` on two TAS nodes; ttl 0 removed the job) and stayed `Running` in `wait-trainer-nats` with no trainer. Same signature as 2026-09-18 and as the first r47 launch. The agent stopped it and relaunched without a user go (RUNLOG; memory note `r44-r45-launch-plan-2026-09-27`).
- **Rollout 40 took 3.8 h and step 40 waited 4.0 h.** The resume restarted every gym pod with an empty queue at a mid-epoch data position. The first result came at 06:55Z, 43 min after collection started; at 07:20Z all 6 returned groups had zero variance (RUNLOG). Step 40 is not a steady-state step: leave it out of any time or MFU average (`wf_545a45a0-6ca`).
- **The trainer is the slow side.** From rollout 49 on, the finished-group queue sat at the 320 cap; a gym rollout took 610 to 841 s while a train step took about 5,050 s. Training data was about 4 steps old, and about 68 of 288 gym pods sat idle (`wf_11893a4c-2b9`; the pod count is not re-verified).
- **Steps 50 to 54 exist only in W&B.** The last save is iteration 49 (00:25Z). The Stop at 09:23Z came in step 55. A later resume starts at rollout 50 (RUNLOG 2026-09-28; S3 tracker 49).
- **W&B shows `crashed` for `ajsur4ej` and `2z0599lb`.** The Stop kills the process before W&B gets a finish call. Neither run crashed (W&B API 2026-09-28).
- **Gym `dind` containers ran out of memory.** One OOMKilled restart at 07:05:53Z (RUNLOG). A later note reports steady restarts from 07:00Z 2026-09-27 on (memory note `miles-reconcile-upstream-2026-09-28`; not re-verified).
- **Per-call clips vs `truncated_ratio`.** The gym logged 1,482 "turn truncated at max_tokens" warnings in the first hour (the 16384 cap; RUNLOG), while `rollout/truncated_ratio` read 0.0 on every rollout. The two count different things (one model call vs one episode).
- **Slow rollouts 52 and 55.** 2,050 s and 4,602 s against 610 to 841 s for the others. Not investigated.
- **Episode length grows across the resume.** Mean episode length fell from 175K (r43 rollout 41) to 88K at rollout 40, then rose to 218K by rollout 55; the max reached 1.02M tokens. Rows per episode rose from 2.96 to 5.48. NEVER fit one trend across rollout 39 and rollout 40; the resume changed the cap, the compaction bound, the images, and the W&B run (`r45/BUILD.md`).

## Parent: r43 (`rl-glm53f43-qztzt`)

r43 is r41b on template `guparpit-miles-deployer-v9`, which adds the cluster-only gym env `DOCKER_CONFIG=/root/.docker` (Harbor 0.22.0 passes the task `HOME=/home/agent` to the Docker CLI, so r41 and r41b failed every pull). Launched 2026-09-25 21:26:30Z; trainer log starts 21:31:51Z; W&B `2z0599lb` created 21:31:58Z. Images: gym `gym-glm53-adr71-20260925a` `sha256:42c5bb8e1bb59429ed4adb066f34f9be9a9ebb38e4f90c1aaedc97d0a2167c8d`, trainer `miles-glm53-r13-20260925a` `sha256:f5e24d9ed73b4fcea815d2c1691cc9e85588df44bd72e8515d2d1aafbacbd2a4` (ECR, read 2026-09-28). Fresh start from the base DCP. It was the live check for CR-307888266 (token arrays as `.tokens` files): 0 groups dropped for a lost ref in all 42 rollouts (`r43/BUILD.md`; memory note `r41-adebt-v3-token-refs-2026-09-25`; r43 `trainer-0.log`).

| Measure | Value | Source |
| --- | --- | --- |
| Rollouts / steps | 42 rollouts (0 to 41), 41 steps (0 to 40) | r43 `trainer-0.log`; W&B `2z0599lb` (42 reward points, 41 `grad_norm` points) |
| Saves (DCP done) | iteration 9 at 09-26 03:40:03; 19 at 09:55:44; 29 at 17:21:04; 39 at 09-27 04:07:31; data source state 39 at 04:31:37 | r43 `trainer-0.log`; S3 listing |
| Reward | 0.695 at rollout 0; 0.681 mean over rollouts 30 to 41; low 0.558 at rollout 37 | `Rollout N complete` lines |
| Episode length mean | 16.7K at rollout 0 -> 103K at 29 -> 175K at 41 | RolloutManager `perf N` |
| Window overflow | 3,392 "Requested token count exceeds the model's maximum context length" lines | `grep -c` |
| `grad_norm` spike | 12.6 at step 34 (`ppo_kl` 3.7e-4, `tis_clipfrac` 2.7e-3); back to 0.047 at step 35 | actor `step N` dicts |
| Zero-variance groups | 2,721 examined, 1,377 dropped (51%) | `Dynamic sampling:` lines |
| Step time, steps 30 to 40 | mean 4,009 s | actor `perf N` dicts |
| End | Stop at 09-27 05:33:41Z; log ends 05:36:19Z with NATS timeouts | RUNLOG; `trainer-0.log` |

Lost at the Stop: rollouts 40 and 41 (collected 03:00Z and 04:44Z) and step 40 (trained 05:33:07Z). Rollout 42 started collection at 05:33:33Z, 8 s before the Stop. Step 41 never trained. The RUNLOG line "r43 had trained rollouts 40 and 41" overstates this by one step. Those metrics exist only in W&B `2z0599lb`. Checkpoint dir: `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/rl-glm53f-adebt-v3-r43/` (tracker 39; `iter_0000009`, `_19`, `_29`, `_39`; sample summaries `debug/rl-glm53f-adebt-v3-r43/sample_summary/rollout_0..41.jsonl`).

## Follow-ups

- Dataset: r47 trains on agentic-debt-766 (manifest commit `327e057c`, task pin `3cadc6b0`). Record: `../r47/RECORD.md`.
- KDA tensor-parallel shard (ADR-0016) and `--skip-actor-forward-only`: both ship in the r17 trainer image of r47. The r47 first-step gate expects `train_rollout_logprob_abs_diff` near 0.0272, the kdatp T2 value on the r43 `iter_0000039` start point (`kdatp/RESULTS.md`; RUNLOG 2026-09-28). The live r45 value was 0.038 to 0.044 on steps 40 to 54 (table above), and r43 read 0.039 at step 39. Re-check which value the r47 gate uses before you call a deviation a fault.
- HF export per save is off from `miles-glm53-r15-20260927a` on (`--save-hf` opt-in). r45 still wrote `hf/rollout_49` (21 min of idle GPUs). r47 has no `--save-hf`.
- Study records under `../../studies/`: MFU and step-time breakdown `mfu-glm53-flash-rl/STUDY.md` (`wf_545a45a0-6ca`), step-level advantage `step-level-advantage-agentic-debt/STUDY.md` (`wf_11893a4c-2b9`), all-pass and reward-hack audit `agentic-debt-all-pass-audit/STUDY.md` (`wf_e6084655-6ed`), generation length in `context-overflow-131k-and-clamp/STUDY.md` (`wf_c2a543f0-f9c`, r43 data), KDA gates `kda-tensor-parallel-sharding/STUDY.md` (`wf_58a8a79c-51c`, `wf_cce1530b-ce9`, `wf_add94c9c-caf`, `wf_1e9f5ba2-351`, `wf_60cbc331-c00`, `wf_44fd3690-bac`).
- RUNLOG 2026-09-27 r45 entry: the r43 loss is rollouts 40 and 41 plus step 40, not "trained rollouts 40 and 41". RUNLOG is closed; this record holds the correction.
- Config comment in `r45/miles-config.yaml` about two optimizer steps per rollout is wrong (one step per rollout on the NATS path). Fix it in the next config that copies from r45.
- Gym `dind` memory: confirm the restart count from the pod events of a live run before the next agentic-debt launch.

## Sources

- Run files: `r45/BUILD.md`, `r45/miles-config.yaml`, `r45/workflow.yaml`; base `r43/BUILD.md`, `r43/miles-config.yaml`, `r43/workflow.yaml`; images and template `r44/BUILD.md`.
- RUNLOG (`training-runs/harbor-rl-glm53-flash/RUNLOG.md`): 2026-09-27 ADR-0015 entry; 2026-09-27 r44 launch (retire of r39 and r42); 2026-09-27 r45 entry (seed, Stop of r43, both launches, resume evidence, health at 07:20Z); 2026-09-28 r47 entry (r45 retire, last checkpoint).
- Trainer logs: `s3://arena-scratch-prod-bom-ap-south-1/guparpit/logs/rl-glm53f-adebt-v3-r45/trainer-0.log` (21,619,127 B, last write 09:28:11Z 2026-09-28) and `.../rl-glm53f-adebt-v3-r43/trainer-0.log` (29,276,407 B). Line numbers in this record refer to these files.
- Checkpoints: `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/rl-glm53f-adebt-v3-r45/` and `.../rl-glm53f-adebt-v3-r43/` (listed 2026-09-28).
- Sample summaries: `s3://arena-scratch-prod-bom-ap-south-1/guparpit/debug/rl-glm53f-adebt-v3-r45/sample_summary/rollout_40.jsonl` to `rollout_55.jsonl`.
- Manifest: `s3://arena-scratch-prod-bom-ap-south-1/guparpit/data/agentic-debt/20260923-v3-locked-oracle/manifest-le5.jsonl` (139,999 B, 2026-09-25 08:10Z).
- W&B: `https://mega.wandb.agi.amazon.dev/arena/rl-glm53f-adebt-v3/runs/ajsur4ej` and `.../runs/2z0599lb`.
- ECR: `aws ecr describe-images --registry-id 427267593057 --repository-name arena-slime-dev` in ap-south-1 and us-east-1 (2026-09-28).
- Decisions and code: miles ADR-0015 (task message limits), ADR-0016 (KDA shard); AREnATasks ADR-0063 amendment, ADR-0072; `examples/arena/harbor-rl-glm53-flash/kdatp/RESULTS.md`; `r47/BUILD.md`.
- Workflow journals: `wf_545a45a0-6ca` (MFU), `wf_11893a4c-2b9` (step-level advantage), `wf_e6084655-6ed` (all-pass audit), `wf_2776ce0d-bc7` (r44 resume; read the r44 checkpoint from an r45 trainer pod).
- Memory notes (dated, secondary): `r44-r45-launch-plan-2026-09-27`, `r45-mfu-investigation-2026-09-27`, `adebt-dataset-must-be-766-main`, `r41-adebt-v3-token-refs-2026-09-25`, `gen-length-p50-p90-2026-09-27`, `miles-reconcile-upstream-2026-09-28`.
