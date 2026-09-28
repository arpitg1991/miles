# Run record: r46 — auctioneer caponly-1034 with the token-level loss average

**Status:** Running
<!-- gen-workflow:begin -->
**Date:** 2026-09-27
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `rl-glm53f46-`
**Experiment name:** `rl-glm53f-auct-cap-r46`
**W&B project:** `rl-glm53f-auct-cap` (group `rl-glm53f-auct-cap-r46`)
**Dataset:** `lakefs://arena-inspect/dev/internal/auctioneer/caponly/caponly-1034/manifest.jsonl` (gym `auctioneer-caponly`)
**Manifest commit:** `dev`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r14-20260927a`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `r44`
<!-- gen-workflow:end -->
**Argo workflow:** `rl-glm53f46-clhv6` (launch, 2026-09-27 06:15:41Z; deleted by the idle-GPU reaper 11:02:24Z), then `rl-glm53f46-rxn6b` (resume 1, created 14:16:04Z, `Running` at 2026-09-28 16:00Z)
**W&B run:** `4iu54zov` (both attempts; the resume continued it through a hand-written sidecar)
**Task pin:** `8c710cf5576ef516bdcb8bce13e56928629e429972e9a3cba61c419d3b688676` on all 1034 rows of the manifest copy that the resume pulled (`/root/.arena_cache/.../caponly-1034/manifest.jsonl` on `rl-glm53f46-rxn6b-trainer-worker-0`, mtime 2026-09-27 14:22Z, md5 `3e65fa5e263d7734cf50565738ef2f5d`). The launch-time copy died with `rl-glm53f46-clhv6`; not re-verified for attempt 1.
**Image digests:** gym `sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d`; trainer r14 (launch) `sha256:48d52a6a431257f9f7b40ce00debe9b2a10b61f94cee85d83291b21a2439e60c`; trainer r15 (resume 1) `sha256:67b57cdf92695b4ec1e3c065cb803349745bc83858b28ef046d59da4cf960247` (ECR `describe-images` 2026-09-28; the live worker-0 `imageID` is the r15 digest)
**Trainer config deltas vs base:** `calculate_per_token_loss` unset (miles default `sum_of_sample_mean`) -> `true`; `experiment_name`, `project_name`, `arena_sample_summary_dir` carry the r46 names. Resume 1 vs launch: `replicas` 40 -> 16 (record key; workflow param `replicas` 16), trainer image r14 -> r15 (no `--save-hf`).
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-auct-cap-r46` (S3 `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/rl-glm53f-auct-cap-r46/`), tracker 49, last save `iter_0000049` (2026-09-28 13:51:50Z in the trainer log; `.metadata` on S3 13:52:56Z)
**Outcome:** Running at rollout 56 of 300 (2026-09-28 15:40Z). Reward 0.65-0.79 in rollouts 50-56 against r44 0.39-0.55 at the same indices. The A/B against r44 is confounded: the resume replayed epoch 0 from offset 0, and 75% of the episodes after the resume end on the 4 h agent timeout (r44: 95-99%). See Issues 2 and 3.

## Goal

Test the token-level loss average (DAPO, arXiv 2503.14476; Dr. GRPO length
bias, arXiv 2503.20783) against the r44 per-sample mean on auctioneer
caponly-1034. With `calculate_per_token_loss`, each optimizer step divides the
loss sum by the total trained token count, so every token has the same weight.
The r44 default divides each rollout by its own token count, then divides the
sum by the rollout count. Expected effect: a length drop on the failed episodes
(`r46/BUILD.md` "Why"). The run ends at `num_rollout` 300 or on a retire.

## Setup

r46 is r44 with one functional change. Everything else is the r44 launch:
dataset caponly-1034, `lr` 1.5e-6, `rollout_max_response_len` 16384,
`rollout_max_context_len` 131072, `global_batch_size` 256,
`rollout_batch_size` 64, `n_samples_per_prompt` 8, `arena_inflight_multiplier`
4, radix cache and overlap schedule on, `agent-timeout-multiplier` 2,
`ack-wait` 36000, `trainer-task-deadline-secs` 39600, `agent-kwargs` `{}`,
`publish-jobs-dir` `''`, `gym-replicas` 288, template v10 (uid
`16ddd530-501f-4734-82ae-88500671bed8`). Preparation notes, the code check of
the flag, and the two launch checklists are in [BUILD.md](BUILD.md).

| Item | Base r44 | This run |
| --- | --- | --- |
| `calculate_per_token_loss` | unset (`sum_of_sample_mean`) | `true` (trainer flag `--calculate-per-token-loss`, live in the worker-0 argv) |
| Loss normalizer per step | each rollout / its token count, then / rollout count | loss sum / total trained token count |
| `experiment-name`, `generateName` | `rl-glm53f-auct-cap-r44`, `rl-glm53f44-` | `rl-glm53f-auct-cap-r46`, `rl-glm53f46-` |
| Sample summary dir | `debug/rl-glm53f-auct-cap-r44/sample_summary` | `debug/rl-glm53f-auct-cap-r46/sample_summary` |
| Workflow parameters | - | equal to r44 except `experiment-name` and `miles-config` (`BUILD.md` diff check) |

Resume 1 (`workflow-resume1.yaml`, `miles-config.yaml`; the launch config is
`miles-config-r0.yaml`). r44 got the same two changes at the same time
(`r44/BUILD.md` "Resume 1"; `rl-glm53f44-qvjnv` created 14:16:02Z).

| Item | r46 launch (`rl-glm53f46-clhv6`) | r46 resume 1 (`rl-glm53f46-rxn6b`) |
| --- | --- | --- |
| Start point | base DCP (`ref_load`) | r46 `iter_0000009` (`--load` = `--save` = the r46 dir) |
| Rollout ids | 0..10 | 10.. (attempt-1 rollout 10 discarded) |
| `replicas` / trainer nodes / SGLang engines | 40 / 8 / 32 (`--rollout-num-gpus 256`) | 16 / 8 / 8 (`--rollout-num-gpus 64`, live argv) |
| Trainer image | `miles-glm53-r14-20260927a` (miles `e0987aed6`) | `miles-glm53-r15-20260927a` (miles `bc31f88ac`) |
| HF export per save | yes (`--save-hf`) | no (`arena_save_hf` unset) |
| Sidecar `iter_0000009/slime_extra_state.json` | not written (reaper cut the save) | hand-written 14:07:30Z: `{"rollout_id": 9, "wandb_run_id": "4iu54zov"}`, 45 bytes |
| Data-source state `rollout/arena_data_source_state_9.pt` | not written | none; the data source restarted at offset 0, epoch 0 (Issue 2) |
| Trainer pods / gym pods (2026-09-28 16:00Z) | 40 / 288 | 16 / 288, 0 restarts |

## Timeline

| UTC | Event |
| --- | --- |
| 2026-09-27 06:15:41Z | `kubectl create -f r46/workflow.yaml` -> `rl-glm53f46-clhv6` (RUNLOG.md:2282) |
| 2026-09-27 06:18:49Z | Kueue admitted the PyTorchJob with no wait; `PodsReady` 06:19:34Z (RUNLOG.md:2284) |
| 2026-09-27 06:21:47Z | Manifest pulled from the `dev` branch path; 1034 prompts loaded (`trainer-0-attempt1.log` 2348, 2351) |
| 2026-09-27 06:33:46Z-06:34:08Z | 32 of 32 SGLang engines up (`trainer-0-attempt1.log` 4415-4978) |
| 2026-09-27 07:03:50Z | Rollout 0 complete, 32 groups, 256 samples, `avg_reward` 0.363, 1685.5 s (`trainer-0-attempt1.log` 6710) |
| 2026-09-27 08:00:15Z | Train step 0 end (`trainer-0-attempt1.log` 11910) |
| 2026-09-27 10:28:10Z | Rollout 10 complete, `avg_reward` 0.330, queue 320 (`trainer-0-attempt1.log` 22696) |
| 2026-09-27 10:44:56Z | Step-9 DCP save complete; the HF export starts (`trainer-0-attempt1.log` 23462-23463) |
| 2026-09-27 11:02:24Z | AREnAThanatos reaper deletes `rl-glm53f46-clhv6` during the HF export (`BUILD.md` "Resume 1"; EKS audit log not re-read). Last log line 11:02:46Z |
| 2026-09-27 14:06:09Z | Trainer image r15 in ap-south-1 ECR (`describe-images` `imagePushedAt`) |
| 2026-09-27 14:07:30Z | Sidecar written by hand from r45 trainer-worker-39 (`BUILD.md`; S3 object 14:08:32Z, 45 bytes) |
| 2026-09-27 14:16:04Z | `kubectl create -f r46/workflow-resume1.yaml` -> `rl-glm53f46-rxn6b` (Workflow `creationTimestamp`) |
| 2026-09-27 14:19:41Z | `rl-glm53f46-rxn6b-trainer-worker-0` started (pod `startTime`); worker logs 1-15 on S3 14:21-14:22Z; manifest re-pulled 14:22Z (pod cache mtime) |
| 2026-09-27 18:38Z | Resume rollout 10 complete, reward 0.427 (W&B; `sample_summary/rollout_10.jsonl` rewritten 18:39:34Z) |
| 2026-09-27 23:05:56Z | `iter_0000019` save; sidecar 23:07:04Z; `arena_data_source_state_19.pt` 23:05:56Z (S3) |
| 2026-09-28 03:29:25Z | `iter_0000029` save (S3) |
| 2026-09-28 08:02:00Z | `iter_0000039` save (S3) |
| 2026-09-28 13:51:50Z | `iter_0000049` save (pod log); `arena_data_source_state_49.pt` 13:53:45Z (S3) |
| 2026-09-28 15:40:01Z | Rollout 56 complete, `avg_reward` 0.663, queue 30 (pod log); train step 56 at 16:00Z (W&B) |

## Results

Train step 0 against r44 step 0 (RUNLOG.md:2305-2312; W&B `train/*`).
`pg_loss` is not comparable: on r46 it is a token-weighted mean. On r46
`pg_clipfrac`, `ppo_kl`, and `ess_ratio` are token-weighted means; on r44 they
are per-rollout means.

| Metric | r44 | r46 |
| --- | --- | --- |
| `train/pg_clipfrac` | 0.000262 | 0.000268 |
| `train/ppo_kl` | 2.35e-5 | -3.75e-5 |
| `train/grad_norm` | 0.140 | 0.120 |
| `train/ess_ratio` | 1.015 | 0.999 |
| `train/pg_loss` | 1.42e-5 | -0.0183 |
| `train/train_rollout_kl` | 0.0034 | 0.0041 |

Reward series. Mean of W&B `rollout/episode_raw_reward` per window, with the
min and max, and the mean of `rollout/episode_response_length/mean` (model
tokens per episode). Read on 2026-09-28 16:00Z from runs `4iu54zov` (r46) and
`82i3g7lv` (r44). Attempt 1 is the 32-engine period before the reaper; the
resume is the 8-engine period. NEVER fit one trend across rollout 10: both
runs log rollout 10 twice (r46 0.330 at 10:28Z, 0.427 at 18:38Z).

| Rollouts | r46 reward | r46 length | r44 reward | r44 length |
| --- | --- | --- | --- | --- |
| Attempt 1: r46 0-10, r44 0-11 | 0.354 (0.291-0.432) | 18148 | 0.355 (0.300-0.429) | 18427 |
| 10-14 | 0.492 (0.427-0.526) | 14345 | 0.430 (0.361-0.465) | 23956 |
| 15-24 | 0.412 (0.335-0.484) | 14174 | 0.469 (0.447-0.507) | 24745 |
| 25-32 | 0.495 (0.383-0.560) | 15206 | 0.469 (0.366-0.559) | 27928 |
| 33-40 | 0.550 (0.425-0.626) | 18306 | 0.513 (0.456-0.580) | 33990 |
| 41-48 | 0.625 (0.588-0.676) | 19492 | 0.512 (0.460-0.598) | 42362 |
| 49-56 (r44 49-54) | 0.671 (0.535-0.789) | 18177 | 0.458 (0.385-0.554) | 46875 |

Train metrics, medians over the resume steps 10-56 (r44 10-52), W&B.

| Metric | r46 | r44 | Note |
| --- | --- | --- | --- |
| `train/grad_norm` | 0.123 | 0.097 | r46 range 0.108-0.141; r44 range 0.062-0.114, 0.065 at step 52 |
| `train/pg_clipfrac` | 2.85e-4 | 2.1e-4 | token-weighted on r46 |
| `train/train_rollout_kl` | 0.0123 | 0.0090 | |
| `train/ess_ratio` | 0.999 | 0.999 | |
| `perf/actor_train_time` (s) | 532 | 730 | attempt 1: 767 / 763 |
| `perf/step_time` (s) | 722 | 1010 | attempt 1: 1010 / 1020 |

Throughput and stop reasons (W&B, pod log, `sample_summary/`).

| Metric | Attempt 1 (32 engines) | Resume 1 (8 engines) | Source |
| --- | --- | --- | --- |
| Wall time per rollout, r46 | 20.5 min (rollouts 0-10) | 27.4 min (rollouts 10-56) | W&B `_timestamp` |
| Wall time per rollout, r44 | 22.2 min (0-11) | 29.1 min (10-54) | W&B `_timestamp` |
| `perf/rollout_time` median (s) | 119 | 112 | W&B; queue-fed rollouts only |
| Rollouts with an empty queue | none after rollout 1 (queue 320) | 49: 6267 s, 50: 3789 s (`queue=0`) | `trainer-0-attempt1.log`, pod log |
| Episodes with `agent_stop_reason` `timeout`, r46 | 0 of 2646 (rollouts 0-9) | 0.72-0.78 per window (10-48), 0.49 (49-56) | `sample_summary`, W&B `rollout/stop/timeout` |
| Episodes with `timeout`, r44 | no `rollout/stop/timeout` key logged | 0.94-0.99 per window | W&B |
| `rollout/dyn_sampling_drop_frac`, both runs | 0 | 0 | W&B; 32 of 32 groups kept on every rollout |

Data order after the resume (`rollout/arena_data_source_state_<N>.pt`, loaded
with `torch.load`; attempt-1 ids from the `rollout_tasks` lines of
`trainer-0-attempt1.log`).

| State file | `offset` | `epoch` | `sample_group_index` | Trained rollouts in `metadata` |
| --- | --- | --- | --- | --- |
| `_19.pt` | 767 | 0 | 767 | 10-20 |
| `_29.pt` | 1022 | 0 | 1022 | 10-30 |
| `_39.pt` | 244 | 1 | 1278 | 10-40 |
| `_49.pt` | 504 | 1 | 1538 | 10-49 |

- Resume rollouts 10-40 trained on epoch-0 ids only (`-e0` suffix): 992
  distinct seeds. All 320 seeds that attempt 1 trained in steps 0-9 are
  among them. Thus 320 prompts were trained twice from the same draw order.
- Per rollout, the count of trained ids that attempt 1 had trained: 10-17:
  32 of 32 each; 18: 18; 19: 24; 20: 11; 21: 10; 22: 2; 23-40: 0. The short
  episodes finish first on both attempts, so the first eight resume rollouts
  re-trained only prompts of the attempt-1 set.
- Epoch-1 ids appear at rollout 41 (1 of 32), 42 (21), and 43 on (32). By
  rollout 49 every one of the 1034 seeds was trained at least once in the
  resume. Against the attempt-1 rollout at the same index (k-10), the
  overlap is 3-9 of 32, as in the r44-vs-r46 check of `BUILD.md`.
- r44 resumed with its saved state at offset 937 of 1034, epoch 0
  (`r44/BUILD.md` "Checkpoint"), so r44 entered epoch 1 about 30 rollouts
  before r46.

## Issues

1. **Reaper delete during the HF export (2026-09-27 11:02:24Z).** With the
   16384 cap, 32 engines finished a rollout in about 2 min and idled through
   each 14-26 min train step; the step-9 HF export idled every GPU for about
   25 min. The AREnAThanatos rule (60-min mean normalized GPU power under 10%)
   fired on r44 at 10:02:37Z and on r46 at 11:02:24Z (`BUILD.md` "Resume 1";
   memory note `thanatos-idle-gpu-reaper-prod-bom-v2`, power data not
   pulled). Fix: 8 engines, and the HF export off by default in the launcher
   (miles `bc31f88ac`, image r15, plugin ADR-0006 amendment).
2. **The cut save lost the sidecar and the data-source state.**
   `train_async_arena.py` runs `save_model` (DCP, then HF export), then
   `_save_extra_state`, then `rollout_manager.save`. The delete came between
   the first and the second. `hf/rollout_9` holds 489 of 671 files and is not
   a model. The sidecar was hand-written (Setup). The data-source state was
   not restorable: `offset` and the consumed ids depend on the rewards and on
   the completion order, so the r44 state does not fit r46 (`BUILD.md`
   "Repair decision"). `ArenaDataSourceWithBuffer.load(9)` then logged
   "does not exist, starting fresh" (expected line; the resume log is not
   readable yet, Issue 5) and restarted at offset 0 of the `shuffle(0)` order.
   `state_19.pt` confirms the restart: offset 767, epoch 0. Effect: the
   resume replayed all of epoch 0 (rollouts 10-40) and re-trained the 320
   attempt-1 prompts first (Results). This is the data-order confound of the
   loss A/B. Open item; see Follow-ups.
3. **Timeouts after the resume.** With 8 engines, each engine serves up to
   256 groups x 8 samples / 8 = 256 episodes at once (64 before). 75% of the
   r46 episodes and 95-99% of the r44 episodes now end with
   `agent_stop_reason` `timeout`, the 4 h agent limit (task timeout x
   `agent-timeout-multiplier` 2; `r42/BUILD.md` line 12). Attempt 1 had 0
   such episodes. Groups now complete in about 4 h waves: 8 fast rollouts on
   the queue, then a 1-2 h wait with `queue=0` (W&B gaps before rollouts 18,
   25, 33, 41, 49; pod log rollouts 49-50). The auctioneer reward is scored
   on the frozen state at the cut, so the resumed runs train on a changed
   episode. Both the attempt-1 comparison and the r44 comparison carry this
   change; r44 carries more of it (longer episodes, more timeouts). The
   r46 timeout share fell to 0.49 in rollouts 49-56 with no config change.
4. **W&B x-axis.** Rollout 10 has two points in both runs (attempt 1 and the
   resume). `sample_summary/rollout_10.jsonl` holds the resume data only.
   Split every fit at the boundary (`wandb-resumed-run-restart-discontinuity`).
5. **The resume trainer log is not readable yet.** `trainer-0.log` on S3 is
   byte-equal to `trainer-0-attempt1.log` (5058594 bytes, 11:03:59Z);
   mountpoint-s3 uploads on close. `kubectl logs` of worker-0 was rotated
   and starts on 2026-09-28 about 13:30Z. Thus the resume startup lines
   (`Restored wandb_run_id=4iu54zov`, `starting fresh`, the engine count)
   are not re-verified. Indirect evidence: run `4iu54zov` continued at
   18:38Z; `state_19.pt` offset 767 epoch 0; live argv `--rollout-num-gpus
   64`, `--calculate-per-token-loss`, no `--save-hf`.
6. **Dataset ref is a branch.** `prompt-data-list` points at `dev`. The
   trainer logs the path only (`data_source.py:126`), so the branch head at
   pull time is unknown. The resume copy pins task commit `8c710cf5...`. The r39 note of 2026-09-25
   recorded rows at `e5ef91b0...` (memory note, id truncated, not
   re-verified). If both are right, the `dev` content changed between r39
   and r46. r47 pins a commit ref instead (`r47/BUILD.md`).

## Follow-ups

- Study: separate the loss change from the two confounds before any credit
  goes to `calculate_per_token_loss`. Compare epoch-1 rollouts (r46 43 on,
  r44 12 on) on the same prompts with the `prompt_reward` maps of both state
  files; split by `agent_stop_reason`; report the timed-out share next to
  each reward. Record:
  `../../studies/token-level-loss-average-r44-vs-r46/STUDY.md` (Open).
- miles: write the sidecar and the data-source state before any HF export,
  or make a resume with `--load` set and no data-source state a hard error
  behind an explicit flag. Today it logs one line and restarts at offset 0.
- miles: pin `prompt-data-list` to a lakeFS commit ref for every new run,
  as r47 does; add the pulled commit id to the `Pulled lakefs://` log line.
- Operator: when the run ends, copy the resume `trainer-0.log` from S3 and
  add the startup lines of Issue 5 to this record.
- Operator: decide whether the 8-engine layout stays. It survived the reaper
  for 26 h, but it made 75% of the episodes time out and cut the throughput
  from 20.5 to 27.4 min per rollout.

## Sources

- W&B: `https://mega.wandb.agi.amazon.dev/arena/rl-glm53f-auct-cap/runs/4iu54zov` (r46), `.../runs/82i3g7lv` (r44); history read 2026-09-28 16:00Z with the `wandb` API (688 and 658 rows).
- S3 (profile `arena-prod-bom-user`, `ap-south-1`): `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/rl-glm53f-auct-cap-r46/` (iter dirs, `rollout/`, `hf/rollout_9`), `.../guparpit/logs/rl-glm53f-auct-cap-r46/` (`trainer-0-attempt1.log`, `trainer-0.log`, `trainer-1..39.log`), `.../guparpit/debug/rl-glm53f-auct-cap-r46/sample_summary/` (57 files, rollouts 0-56).
- Cluster (read only, context `arena-prod-bom-v2`, namespace `arena-tasks`): Workflows `rl-glm53f46-rxn6b`, `rl-glm53f44-qvjnv`; pod `rl-glm53f46-rxn6b-trainer-worker-0` (`/proc` argv of `train_async_arena.py`, `imageID`, `startTime`, `kubectl logs`, cached manifest); ECR `describe-images` on `arena-slime-dev`.
- Repo: [BUILD.md](BUILD.md), [miles-config.yaml](miles-config.yaml), [miles-config-r0.yaml](miles-config-r0.yaml), [workflow.yaml](workflow.yaml), [workflow-resume1.yaml](workflow-resume1.yaml); `../r44/BUILD.md`; `../r42/BUILD.md`; `../RUNLOG.md` lines 2266-2320 (launch) and 2401 (r47 entry, "Not touched"); `miles_plugins/arena/nats_arena/nats_rollout.py` `_stop_metrics`; `examples/arena/README.md` HF export note.
- miles commits: `e0987aed6` (r14 image), `bc31f88ac` (HF export opt-in, r15 image), `aa8cd8e70` and `662e0cc95` (r46 prepare and launch), `3028bc358` (resume prep).
- ADRs: miles plugin ADR-0006 (launcher argv parity, HF export amendment), miles ADR-0015 (task-message limits), AREnATasks ADR-0072 and ADR-0063 amendment (gym limits and clamp).
- Workflow journals: `wf_128c0acc-6c4` (launch and first health check), `wf_2776ce0d-bc7` (HF export off, r15 image, resume prep; its resume-phase result was not journaled).
- Memory notes (secondary): `r44-r45-launch-plan-2026-09-27`, `thanatos-idle-gpu-reaper-prod-bom-v2`, `r39-auctioneer-caponly-1034-2026-09-25`, `wandb-resumed-run-restart-discontinuity`.
