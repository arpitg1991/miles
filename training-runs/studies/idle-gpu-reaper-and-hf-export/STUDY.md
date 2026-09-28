# Study: Idle-GPU reaper deletions and the per-save HF export

**Date:** 2026-09-27
**Status:** Closed
**Question:** Why did the AREnAThanatos idle-GPU reaper delete four GLM-5.3-Flash runs, and which changes stop it?
**Runs and data used:** r41 `rl-glm53f41-kfjhv` (W&B `pdi51hm0`), r41b `rl-glm53f41-c44lw` (W&B `12qql5pf`), r44 `rl-glm53f44-lt8vm` and its resume `rl-glm53f44-qvjnv` (W&B `82i3g7lv`), r46 `rl-glm53f46-clhv6` and its resume `rl-glm53f46-rxn6b` (W&B `4iu54zov`); W&B projects `rl-glm53f-adebt-v3` and `rl-glm53f-auct-cap`; `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/rl-glm53f-auct-cap-r4{4,6}/` and `.../guparpit/logs/rl-glm53f-auct-cap-r4{4,6}/trainer-0-attempt1.log`; EKS audit log `/aws/eks/arena-eks-prod-bom-v2/cluster`.
**Code SHAs:** miles `bc31f88ac`: the per-save HF export becomes opt-in (`arena_save_hf`); miles `c10201fc5`: README row for image `miles-glm53-r15-20260927a`; miles `3028bc358`: r44 and r46 `workflow-resume1.yaml` with `replicas` 16; AREnAInfraCDK `e5e6e06d` (local checkout, 2026-09-22): `resources/lambda/thanatos/index.py` and `lib/stacks/workloads/thanatos-stack.ts` as read here.
**Workflow ids:** `wf_2776ce0d-bc7` (code, image, resume preparation, resume launch).

## Method

- Deletes: one CloudWatch Logs Insights query on 2026-09-28 over the EKS audit
  log (`verb = "delete"`, `objectRef.resource = "workflows"`, names
  `rl-glm53f4[146]*`, 2026-09-25 to 2026-09-28), profile `arena-prod-bom-user`.
- Reaper policy: the Lambda source in the local AREnAInfraCDK checkout. The
  deployed Lambda version is not re-verified.
- Save timeline: the first-attempt trainer logs, which the resume procedure
  copied aside as `trainer-0-attempt1.log`. Line numbers below refer to those
  files. S3 object times give a second view; the S3 view lags the log by about
  one minute.
- Sizes: `aws s3 ls --recursive --summarize` on `iter_0000009/` and
  `hf/rollout_9/` of both runs, read 2026-09-28.
- Cadence and reward: W&B history through the API, with `rollout/step` as the
  rollout id and the `perf/*` keys. Read 2026-09-28 about 16:00Z, both runs at
  rollout 53 to 56.
- Save order after the fix: the live pod logs of the two resumes
  (`kubectl logs`, last 10k lines only) at the step-49 save.
- Not measured: the DCGM power series that the reaper reads. The idle
  attribution below is a reconstruction from log timing.

## Results

### The four deletes (EKS audit log, re-verified 2026-09-28)

All four deletes came from
`assumed-role/AREnAThanatos-Prod-bom-v2-ThanatosRoleADC18D21-K5OJV79S7Pa8/AREnAThanatos-Prod-bom-v2-ThanatosFnCD707E78-2vPpG7nviXdQ`
with response code 200.

| Run | Workflow | Created (UTC) | Deleted (UTC) | State at the delete | Source |
| --- | --- | --- | --- | --- | --- |
| r41 | `rl-glm53f41-kfjhv` | 2026-09-25 08:13:08 | 2026-09-25 10:59:26 | 0 train steps. Every trial failed in seconds with "pull access denied"; the trainer sat at `0/32 groups collected` for 8286 s. W&B `pdi51hm0` created 08:26:29Z, last heartbeat 10:59:50Z, 0 summary keys | audit log; memory note `r41-adebt-v3-token-refs-2026-09-25` (trainer numbers not re-verified); W&B |
| r41b | `rl-glm53f41-c44lw` | 2026-09-25 15:38:31 | 2026-09-25 16:47:41 | 0 train steps, same pull failure (`HOME=/home/agent` hid `/root/.docker`). W&B `12qql5pf` created 15:44:17Z, last heartbeat 16:48:08Z | audit log; `r41b/BUILD.md`; W&B |
| r44 | `rl-glm53f44-lt8vm` | 2026-09-27 05:07:02 | 2026-09-27 10:02:37 | 9 saved steps. The step-9 HF export ended 09:57:57; step 10 had started | audit log; RUNLOG r44 section; `trainer-0-attempt1.log` |
| r46 | `rl-glm53f46-clhv6` | 2026-09-27 06:15:41 | 2026-09-27 11:02:24 | 9 saved steps. The step-9 HF export was in progress | audit log; RUNLOG r46 section; `trainer-0-attempt1.log` |

### Reaper policy (AREnAInfraCDK local checkout `e5e6e06d`)

| Rule | Value | Source |
| --- | --- | --- |
| Metric | `DCGM_FI_DEV_POWER_USAGE`, raw mean over the window, normalized per GPU model; B200: `(mean_W - 250) / (1000 - 250) * 100`, clamped to 0..100 | `index.py:577-601` |
| Window and samples | 3600 s; 30 s scrape; at least 114 samples (95 %); no gap over 120 s | `index.py:79-82, 589-597` |
| Threshold | job mean over its GPUs below 10 % under `DPU_POLICY` (5 % otherwise) | `thanatos-stack.ts:64, 140-142`; `index.py:128` |
| Runtime guard | every GPU pod of the job `Running` for the full 3600 s | `index.py:560-574` |
| Scope | namespaces `default` and `arena-tasks`; opt-out only by namespace (`IGNORE_NAMESPACES`) | `index.py:78, 121-123` |
| Action | delete the verified top workload root (the Argo Workflow) with `propagationPolicy: Foreground`; owner notification "Stopping workload" | `index.py:925, 994` |

### The step-9 save on the first attempts

| Event | r44 (UTC) | r46 (UTC) | Source |
| --- | --- | --- | --- |
| Rollout 9 complete, `queue=320` | 09:04:09, 112.0 s | 10:12:02, 112.2 s | r44 log line 20612; r46 line 21621 |
| Rollout 10 complete, `queue=320` | 09:20:43, 113.7 s | 10:28:10, 109.1 s | r44 21701; r46 22696 |
| `saving checkpoint at iteration 9` | 09:31:57.256 | 10:42:31.074 | r44 22257; r46 23438 |
| `successfully saved checkpoint` (DCP done) | 09:33:41.309 (104 s) | 10:44:56.431 (145 s) | r44 22353; r46 23463 |
| `Saving model in HuggingFace format` | 09:33:41.316 | 10:44:56.438 | r44 22354; r46 23462 |
| `Successfully saved merged HuggingFace model` | 09:57:57.05 (1456 s) | never | r44 24183 |
| `Saved slime extra state` (sidecar) | 09:57:57.086 | never | r44 24182 |
| `Arena data source state saved` | 09:57:57.107 | never | r44 24196 |
| Rollout 11 complete, `queue=320` | 10:00:11, 110.8 s | - | r44 24471 |
| Last log timestamp | 10:03:04 | 11:02:46 | tail of each log |
| Delete | 10:02:37 | 11:02:24 | audit log |

### Save cost: DCP against HF export

| Artifact | Objects | Bytes | Duration | Throughput | Writers | Source |
| --- | --- | --- | --- | --- | --- | --- |
| r44 `iter_0000009` (DCP) | 68 | 626,842,102,696 | 104 s | 6.03 GB/s | 64 ranks in parallel | S3 listing; log lines 22257, 22353 |
| r46 `iter_0000009` (DCP) | 68 | 626,842,102,696 | 145 s | 4.32 GB/s | 64 ranks | S3 listing; log lines 23438, 23463 |
| r44 `hf/rollout_9` (HF export, complete) | 674 | 626,752,948,749 | 1456 s | 0.430 GB/s | rank 0 only | S3 listing; log lines 22354, 24183 |
| r46 `hf/rollout_9` (HF export, cut) | 490 | 463,764,098,896 | 1048 s to the delete | 0.443 GB/s | rank 0 only | S3 listing; log line 23462; audit log |

The HF export path is `export_hf_model_direct` (`--megatron-to-hf-mode raw`,
the default at `miles/utils/arguments.py:387-389`). Every rank takes part in
the collective gather, but `is_writer = torch.distributed.get_rank() == 0`, and
rank 0 alone moves each chunk to the CPU and writes each `safetensors` shard in
sequence (`miles/backends/megatron_utils/hf_export.py:55-70`). The DCP save
writes 64 `.distcp` shards from 64 ranks. Thus the same 627 GB takes 14 times
longer as an HF export, and every GPU of the job idles for that time.

Other counts of the same directories: `r44/BUILD.md` says 671 files and
`r46/BUILD.md` says 489; ADR-0006 says 491 of 675. The S3 counts above include
the directory marker and, for r44, `.complete` and the index.

### Why the job looked idle (reconstruction, power not pulled)

| Fact | r44 | r46 | Source |
| --- | --- | --- | --- |
| GPU split at 40 nodes | 64 trainer GPUs (20 %), 256 engine GPUs (80 %) | same | `r44/BUILD.md` "Engine count" |
| Rollout time, rollouts 1-9 | 113-135 s, median 127 s | median 119 s | W&B `perf/rollout_time` |
| Step time, rollouts 1-9 | 834-1389 s, median 1011 s; step 0 6041 s | median 998 s; step 0 5592 s | W&B `perf/step_time` |
| Trainer wait for data, rollouts 1-5 | 53-64 s (ratio 0.04-0.06) | 53-57 s | W&B `perf/train_wait_time`, `perf/wait_time_ratio` |
| Finished-group queue at each rollout end | 320 | 320 | log lines above |
| Engine busy time in the 60 min before the delete | about 4 min (rollouts 10 and 11) | about 4 min (rollouts 9 and 10) | log lines above |
| Trainer GPU busy time in the same 60 min | about 33 min (steps 8 and 9, then 3 min of step 10) | about 16 min (step 9) | `perf/step_time` 992 s and 812 s; save start times |
| GPU idle for the HF export | 24.3 min | 17.5 min (cut) | log lines above |

At the 16384 output cap the engines finished each rollout in about 2 min, and
the queue sat at its 320-group cap. So the engines, 80 % of the GPUs, had no
work for most of each 17-min step. The step-9 save then idled the trainer GPUs
too: 104-145 s of DCP and 17-24 min of rank-0 HF export. The weighted GPU duty
in the r44 window is about (64 x 33 + 256 x 4) / (320 x 60) = 16 % of GPU
minutes, and an idle B200 sits near the 250 W floor (normalized 0). A 60-min
normalized power mean under 10 % is consistent with these numbers. The DCGM
series itself was not read.

### What the cut save left on disk

| File | r44 `iter_0000009` | r46 `iter_0000009` | Source |
| --- | --- | --- | --- |
| 64 `.distcp` shards, `.metadata`, `metadata.json` | present, `.metadata` 09:34:46 | present, `.metadata` 10:46:00 | S3 listing |
| `latest_checkpointed_iteration.txt` = 9 | yes | yes | `r44/BUILD.md`, `r46/BUILD.md` "Checkpoint" |
| `slime_extra_state.json` (sidecar) | 09:59:00, `{"rollout_id": 9, "wandb_run_id": "82i3g7lv"}` | absent; written by hand 14:07:30Z (S3 14:08:32) with `wandb_run_id` `4iu54zov` | S3 listing; `r46/BUILD.md` "Repair decision" |
| `rollout/arena_data_source_state_9.pt` | 09:58:59, offset 937 of 1034, epoch 0, 352 consumed ids | absent on 2026-09-28; `rollout/` first appears at 23:05:44 with the step-19 state | S3 listing; `r44/BUILD.md` |
| `hf/rollout_9` | complete, `.complete` 09:59:01 | 489 files, no `.complete`, no index | S3 listing |

The order is fixed in code: `save_model` runs the DCP `save(...)`
(`miles/backends/megatron_utils/actor.py:741`) and then `save_hf_model`
(`actor.py:746-749`). The driver writes the sidecar and the data-source state
only after `save_model` returns (`miles_plugins/arena/train_async_arena.py:208,
241, 245`). A delete during the export leaves a loadable DCP without the two
files that carry the W&B id and the data position.

### After the fix

| Check | r44 resume `rl-glm53f44-qvjnv` | r46 resume `rl-glm53f46-rxn6b` | Source |
| --- | --- | --- | --- |
| Workflow created | 2026-09-27 14:16:02Z | 14:16:04Z | Argo `creationTimestamp` (`kubectl get workflow`, read 2026-09-28). The `wf_2776ce0d-bc7` resume-agent transcript read 14:16:05Z for r46; the journal holds no result entry for that agent |
| Trainer image, argv | `miles-glm53-r15-20260927a`, 257 tokens, no `--save-hf`, 8 engines | same image, 258 tokens (adds `--calculate-per-token-loss`), 8 engines | same transcript |
| Sidecar restore | `Restored wandb_run_id=82i3g7lv from checkpoint sidecar.` 14:21:43 | W&B run `4iu54zov` continued | same transcript; W&B |
| Data position | offset 937, 352 ids skipped | offset 0 (`starting fresh`); about 90 % of epoch 0 drawn again | `r44/BUILD.md`, `r46/BUILD.md` "Resume 1"; the resume log line is not re-verifiable (pod log rotated) |
| First resumed step (rollout 10) | `perf/rollout_time` 14646 s, `perf/train_wait_time` 15095 s, logged 19:33:13 | 14668 s, 15093 s, logged 19:28:56 | W&B |
| `perf/save_model_time`, saves 19/29/39/49 | 101, 108, 120, 115 s | 185, 179, 178, 164 s | W&B |
| Step-49 save order | save 14:57:22.329, DCP done 14:59:17.006, sidecar 14:59:17.043, data state 14:59:17.078 (2026-09-28) | 13:49:06.398, 13:51:50.807, 13:51:50.841, 13:51:50.875 | pod logs, lines 6366-6467 and 1487-1646 |
| HF export lines in the pod log | 0 | 0 | pod logs |
| `hf/` content on 2026-09-28 | `rollout_9` only | `rollout_9` only | S3 listing |

The step-9 save with the export took 1560 s on r44. The DCP-only saves take
101-185 s, 8 to 15 times less, and the sidecar and the data-source state follow
the DCP within 0.1 s.

The resume itself cost about 5 h of the 16-node job: the first attempts' queues
of 320 finished groups were lost, and rollout 10 had to fill from an empty
queue with 8 engines (`perf/rollout_time` 14646 s).

### Cadence at 8 engines

| Metric | r44 | r46 | Source |
| --- | --- | --- | --- |
| Rollout time, rollouts 12-56, median | 119 s | 104 s | W&B `perf/rollout_time` |
| Rollout time spikes (rollout: s) | 17: 5621, 25: 8744, 33: 8320, 41: 7052, 49: 6771 | 18: 6280, 25: 7111, 26: 3011, 33: 8028, 34: 1876, 41: 6550, 42: 2996, 49: 6268, 50: 3790 | W&B |
| Trainer wait at the spikes | rollout 49: 5885 s (ratio 0.87) | rollout 49: 5632 s; 50: 3144 s | W&B `perf/train_wait_time` |
| Wall time per rollout, 32 engines (rollouts 1-9) | 136.8 min / 8 = 17.1 min | 142.7 min / 8 = 17.8 min | W&B `rollout/raw_reward` timestamps 07:04:49-09:21:39 and 08:07:10-10:29:53 |
| Wall time per rollout, 8 engines (rollouts 11 to the last read) | 1202.8 min / 42 = 28.6 min (11 to 53) | 1219.3 min / 45 = 27.1 min (11 to 56) | timestamps 19:38:28 to 15:41:14 and 19:32:46 to 15:52:01 |
| Node-minutes per rollout | 40 x 17.1 = 684, then 16 x 28.6 = 458 | 40 x 17.8 = 712, then 16 x 27.1 = 434 | derived |

With 8 engines the finished-group queue drains about every 8 rollouts, and the
trainer then waits 1 to 2.4 h for the engines. 8 rollouts x 32 groups = 256
groups = the in-flight cap (`arena_inflight_multiplier` 4 x
`rollout_batch_size` 64), so the cap is the likely cause. The cause is not
established. The run is 1.5 to 1.7 times slower per rollout in wall time and
uses 33 to 39 % fewer node-minutes per rollout.

### Reward by rollout window (`rollout/raw_reward`, mean of the window)

| Rollouts | r44 (per-sample loss mean) | r46 (token-level loss mean) |
| --- | --- | --- |
| 0-4 | 0.354 | 0.345 |
| 5-9 | 0.347 | 0.364 |
| 10-14 | 0.425 | 0.492 |
| 15-19 | 0.462 | 0.411 |
| 20-24 | 0.476 | 0.414 |
| 25-29 | 0.482 | 0.499 |
| 30-34 | 0.456 | 0.503 |
| 35-39 | 0.539 | 0.564 |
| 40-44 | 0.523 | 0.600 |
| 45-49 | 0.477 | 0.613 |
| 50-54 | 0.452 (4 rollouts) | 0.703 |

Source: W&B runs `82i3g7lv` and `4iu54zov`, read 2026-09-28 about 16:00Z. W&B
holds rollout 10 of r44 twice (0.405 at 10:01:08 from the first attempt, 0.445
at 18:55:05 from the resume); rollout 10 of r46 exists only from the resume.
r46 drew the epoch-0 order again from offset 0, so its rollouts 10 to about 21
repeat prompts that rollouts 0-10 had seen (937 of 1034 groups; `r44/BUILD.md`
"Checkpoint", `r46/BUILD.md` "Repair decision"). The r46 dip at rollouts 15-24
sits in and after that replay window. The r44 against r46 loss comparison is
confounded by the replay.

## Verdict

The reaper worked as designed. Four runs each idled their whole GPU footprint
for a full hour. r41 and r41b idled because every trial failed at the image
pull, so no group ever arrived. r44 and r46 idled because the 32-engine layout
was training-bound at the 16384 cap (engines busy about 4 min per hour) and
because the step-9 save added 17-24 min in which rank 0 alone wrote a 627 GB HF
export at 0.43 GB/s while the DCP had taken 104-145 s from 64 ranks. The delete
during the export also cost r46 its sidecar and its data-source state, because
the driver writes both after the export. The two changes remove the trigger:
the export is now opt-in, so a save is DCP-only and the progress files follow
within 0.1 s; and the auctioneer runs use 8 engines, so the trainer GPUs are
50 % of the job. The numbers do not show that 8 engines is the right count: the
queue now drains every 8 rollouts, and the wall time per rollout rose from
17-18 min to 27-29 min. The numbers also do not separate the r46 loss change
from the epoch-0 replay.

## Caveats and open items

- The DCGM power series was not read. The idle attribution is a
  reconstruction from log timing; the reaper's own log group had no events
  since 2026-07-09 (memory note `thanatos-idle-gpu-reaper-prod-bom-v2`, not
  re-verified).
- The deployed Lambda version is not re-verified; the policy table comes from
  the local checkout `e5e6e06d`.
- The r41 trainer numbers (8286 s at `0/32 groups`, 16,517 dropped groups)
  come from a memory note; the r41 workflow and pods are gone.
- The resume log lines (`starting fresh`, 8 engines) are not re-verifiable
  now: the pod logs hold the last 10k lines only, and the S3 `trainer-0.log`
  of both runs still shows the first attempt (same size and last timestamp as
  `trainer-0-attempt1.log`).
- Save order in code is unchanged. An opt-in export (`arena_save_hf: true` or
  `ARENA_EVAL_TASKS`) still runs before the sidecar and the data-source state.
- The periodic 1-2.4 h trainer waits at 8 engines need a cause. Candidates:
  the in-flight cap of 256 groups, or the engine count. An in-flight
  multiplier of 8, or 12-16 engines, is untested on auctioneer.
- The r46 replay of epoch 0 confounds the r44 against r46 comparison. The
  overlap of prompts that r46 saw twice, and the reward on unseen prompts
  after the replay, are not quantified.
- `hf/rollout_9` of r44 (627 GB) and the partial `hf/rollout_9` of r46
  (464 GB) stay on scratch. Nothing reads them.
- Earlier trainer deaths right after an HF export (r20, r23, r25) were later
  attributed to routing-replay file reaping, not to the export (memory note
  `glm53-post-export-trainer-crash`, not re-verified here).

## Actions taken

- miles `bc31f88ac`: `scripts/run_arena_harbor.py` appends `--save-hf` only
  when the YAML sets `arena_save_hf: true` or the pod sets `ARENA_EVAL_TASKS`
  (`_save_hf_requested`, lines 209-221, 259-260). A non-boolean value stops the
  launcher. New test `tests/fast/launch_scripts/test_run_arena_harbor.py`
  (6 cases); snapshot `train.txt` without `--save-hf`; ADR-0006 amendment
  2026-09-27.
- Trainer image `miles-glm53-r15-20260927a`, digest
  `sha256:67b57cdf92695b4ec1e3c065cb803349745bc83858b28ef046d59da4cf960247`,
  in us-east-1 and ap-south-1 (README row, miles `c10201fc5`). r17
  (`miles-glm53-r17-20260928a`, r47) is a superset; the r47 argv has no
  `--save-hf` (RUNLOG r47 section).
- miles `3028bc358`: `r44/workflow-resume1.yaml` and `r46/workflow-resume1.yaml`
  with `replicas` 16 and `replica-trainer` 8 (64 rollout GPUs, 8 engines),
  `gym-replicas` 288 unchanged; `miles-config.yaml` became the resume config and
  `miles-config-r0.yaml` the launch config.
- r46 sidecar written by hand at 14:07:30Z (`{"rollout_id": 9, "wandb_run_id":
  "4iu54zov"}`, 45 bytes). The r44 data-source state was not copied to r46.
- Resumes launched 2026-09-27 14:16Z as `rl-glm53f44-qvjnv` and
  `rl-glm53f46-rxn6b` on image r15. Both reached `iter_0000049` on 2026-09-28
  (trackers 15:00:20Z and 13:52:56Z).
- Operating rule (memory note `thanatos-idle-gpu-reaper-prod-bom-v2`): a run
  with 0 rollouts or all groups failed must be fixed or stopped inside 60 min.
  No per-job opt-out exists.

## Sources

- EKS audit log query 2026-09-28: log group `/aws/eks/arena-eks-prod-bom-v2/cluster`, stream `kube-apiserver-audit`, 4 records matched.
- `s3://arena-scratch-prod-bom-ap-south-1/guparpit/logs/rl-glm53f-auct-cap-r44/trainer-0-attempt1.log` (24621 lines) and `.../rl-glm53f-auct-cap-r46/trainer-0-attempt1.log` (24563 lines).
- `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/rl-glm53f-auct-cap-r44/` and `.../rl-glm53f-auct-cap-r46/` (`iter_0000009/`, `iter_0000019/`, `hf/rollout_9/`, `rollout/`, `latest_checkpointed_iteration.txt`).
- W&B `arena/rl-glm53f-auct-cap/82i3g7lv`, `arena/rl-glm53f-auct-cap/4iu54zov`, `arena/rl-glm53f-adebt-v3/pdi51hm0`, `arena/rl-glm53f-adebt-v3/12qql5pf`.
- Pod logs `rl-glm53f44-qvjnv-trainer-worker-0` and `rl-glm53f46-rxn6b-trainer-worker-0`, container `pytorch`, read 2026-09-28.
- Journal `wf_2776ce0d-bc7`: results of the `code+image` and `resume-prep` agents; transcript of the `resume` agent.
- `training-runs/harbor-rl-glm53-flash/r41/BUILD.md`, `r41b/BUILD.md`, `r44/BUILD.md`, `r46/BUILD.md`; `RUNLOG.md` sections "r44 launch", "r46", "r47".
- `miles_plugins/arena/adr/0006-launcher-argv-parity.md`, amendment 2026-09-27.
- `miles/backends/megatron_utils/actor.py`, `miles/backends/megatron_utils/hf_export.py`, `miles_plugins/arena/train_async_arena.py`, `miles_plugins/arena/nats_arena/data_source.py:538`, `scripts/run_arena_harbor.py`, `miles/utils/arguments.py:387-389`.
- AREnAInfraCDK `resources/lambda/thanatos/index.py`, `lib/stacks/workloads/thanatos-stack.ts`, `lib/constants/gpu-job-tracking.ts:55` (local checkout `e5e6e06d`).
- Memory notes `thanatos-idle-gpu-reaper-prod-bom-v2`, `r41-adebt-v3-token-refs-2026-09-25`, `r44-r45-launch-plan-2026-09-27`, `glm53-post-export-trainer-crash`.
