# r44: r42 with a 16384 output cap from the task message (ADR-0015)

Prepared 2026-09-27. Not launched. Generated with:

```
.venv/bin/python gen-workflow.py 44 --base r42 --template guparpit-miles-deployer-v10 \
  --experiment-name rl-glm53f-auct-cap-r44 \
  --gym-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a \
  --trainer-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r14-20260927a \
  --param 'agent-kwargs={}' --param 'publish-jobs-dir='
```

| Delta vs r42 | r42 | r44 |
| --- | --- | --- |
| `rollout_max_response_len` | 32768 (gym copy `ARENA_MAX_TOKENS` 32768) | 16384, sent in the task message |
| Window | gym env `ARENA_ROLLOUT_CONTEXT_LIMIT` 131072 | task message `max_seq_len` 131072 |
| Output cap near the window end | fixed 32768 | clamped to the room left in the window |
| Vulcan `max_compactions` | 2 (`compaction-max` 2) | 4 (`agent-kwargs` `{}`, Vulcan default) |
| Template | `guparpit-miles-deployer-v7` | `guparpit-miles-deployer-v10` |
| Trainer image | `miles-glm53-r12-20260925a` | `miles-glm53-r14-20260927a` (miles `e0987aed6`) |
| Gym image | `gym-glm53-adr69-20260925a` | `gym-glm53-adr72-20260927a` (AREnATasks mainline `538bc63`, tree `ab266688`) |
| Trajectory publication | off (v7 has no publish env) | off (`publish-jobs-dir` `''`) |
| Names | `rl-glm53f-auct-cap-r42` | `rl-glm53f-auct-cap-r44` |

Same as r42: dataset caponly-1034, radix cache and overlap schedule on,
`agent-timeout-multiplier` 2, `ack-wait` 36000, trainer deadline 39600,
`excluded-nodes`, W&B project `rl-glm53f-auct-cap`. Fresh run: no
checkpoint dir under `slime_experiments/rl-glm53f-auct-cap-r44` (checked
2026-09-27 04:50Z).

Why:

- r39 and r42 sent about 500 SGLang requests per 90 minutes that SGLang
  rejected with HTTP 400. Each had 98K-131K input tokens plus the fixed
  32768 output budget, more than the 131072 window.
- About 99% of those requests were Vulcan summary calls. 38-48% of the
  compactions lost their handoff note.
- Healthy runs show a per-call P99 of 17K-20K output tokens. The collapsed
  calls sit at P90 = 32768, the cap. A 16384 cap keeps the healthy range
  and leaves 16K more room for the input.
- The trainer is now the only source of the cap, the window, and the
  sampling values. The gym copies drifted with no error.

Code reviews, all merged: AREnATasks CR-308323817 (clamp to the room left
in the window, `0b12642`) and CR-308323860 (task message limits, ADR-0072,
`538bc63`); Apps CR-308323875 (gym env copies removed, `agent-kwargs`,
mainline `e4bf764`). The trainer side is miles ADR-0015 (`4c9e97b0f`,
`e0987aed6`).

Images:

- Trainer `miles-glm53-r14-20260927a`: `sha256:48d52a6a431257f9f7b40ce00debe9b2a10b61f94cee85d83291b21a2439e60c`
  in us-east-1 and ap-south-1.
- Gym `gym-glm53-adr72-20260927a`: `sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d`
  in us-east-1 and ap-south-1. Label `org.opencontainers.image.revision`
  is `538bc6378870a31a7e29451c163dca1d6c161676`.

## Template v10

`guparpit-miles-deployer-v10.yaml` in this directory is the Apps mainline
`e4bf764` generated template (CR-308323875, same tree as `07e4970`) with
`__AWS_REGION__` set to `ap-south-1`. It adds the two cluster-only gym
additions of v8 and v9: the us-east-1 ECR login and
`DOCKER_CONFIG=/root/.docker`. The default `trainer-image` and
`gym-image` are the two new images. RBAC allows create but not patch, so
each revision has a new name.

Created 2026-09-27T04:48:19Z on `arena-prod-bom-v2`, namespace
`arena-tasks`, uid `16ddd530-501f-4734-82ae-88500671bed8`. The live spec
equals this file.

The template default of `publish-jobs-dir` is
`/mnt/scratch-s3files-rw/harbor-training`. r44 sets `''`, so the gym
publishes no trajectories.

## Launch checklist

1. Done: the gym image exists in ap-south-1 ECR (digest above).
2. Done: template `guparpit-miles-deployer-v10` exists (uid above).
3. Pair check: the r14 trainer with an ADR-0072 gym only. An older trainer
   fails every group. An older gym on v10 falls back to 8192/32768 with no
   error.
4. `excluded-nodes` is the r42 list. On 2026-09-27 no listed instance
   existed in the cluster, so the list excludes no node. A non-empty list
   selects the `deploy-trainer-pinned` step, as on r42.
5. Stop r42 (`rl-glm53f42-5cthp`) to free its 40 nodes.
6. `kubectl create -f r44/workflow.yaml`.
7. On one gym pod, read `HARBOR_AGENT_KWARGS` (`{}`), `DOCKER_CONFIG`
   (`/root/.docker`), and `ARENA_PUBLISH_JOBS_DIR` (empty). In the gym
   log, look for `max_new_tokens` 16384 and `max_seq_len` 131072.
8. Watch the SGLang 400 count and the Vulcan compaction note loss against
   r42 at the same rollout indices. Record `max_compactions` 4.

## Resume 1 (2026-09-27): 8 engines, no HF export

Prepared 2026-09-27. Not launched. The user asked for 8 SGLang engines on
auctioneer and for no per-save HF export on every run (2026-09-27 about
14:00Z). The user rule of 2026-09-09 (resume a crashed run from its last
checkpoint with no new question) covers the resume.

Why:

- The AREnAThanatos idle-GPU reaper deleted `rl-glm53f44-lt8vm` at
  10:02:37Z. The reaper deletes a workflow when the 60-min mean normalized
  GPU power is less than 10%.
- At the 16384 cap, 32 engines finished a rollout in about 2 min. Then the
  engines stayed idle for each 20-60 min train step. The 8 actor nodes are
  only 20% of the 40 nodes.
- Each save also wrote an HF export for about 25 min, with all GPUs idle.
- Rollouts 9 and 10 ended with 320 finished groups in the queue
  (`Rollout N complete ... queue=320`). The trainer sets the pace, not the
  engines.
- With 16 nodes, the 8 actor nodes are 50% of the GPUs.
- Resume needs only the DCP, the tracker, the sidecar, and the data-source
  state. No r44 part reads the HF export: `ARENA_EVAL_TASKS` and
  `eval_interval` are unset. SGLang starts from `hf_checkpoint` and gets the
  trained weights through `update_weights`.

Generated with:

```
python3 gen-workflow.py 44 --base r44 --template guparpit-miles-deployer-v10 \
  --experiment-name rl-glm53f-auct-cap-r44 \
  --gym-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a \
  --trainer-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r15-20260927a \
  --param replicas=16
mv r44/workflow.yaml r44/workflow-resume1.yaml && git checkout r44/workflow.yaml
```

`gen-workflow.py` always writes `r<N>/workflow.yaml` and reads
`r<N>/miles-config.yaml`. Thus `miles-config.yaml` is now the resume
config, and `miles-config-r0.yaml` is the launch config. The `mv` and the
`git checkout` keep `workflow.yaml` as the launch record.

| Delta vs the r44 launch | r44 launch | r44 resume 1 |
| --- | --- | --- |
| Start point | base DCP (`ref_load`) | r44 `iter_0000009`, same dir |
| Rollout ids | 0.. | 10.. |
| Data position | row 0 | offset 937, epoch 0 |
| W&B run | `82i3g7lv` | `82i3g7lv` (from the sidecar) |
| `replicas` (param and config record) | 40 | 16 |
| `replica-trainer` | 8 | 8 |
| Rollout GPUs, engines | 256, 32 | 64, 8 |
| Trainer image | `miles-glm53-r14-20260927a` | `miles-glm53-r15-20260927a` |
| HF export per save | yes (`--save-hf`) | no |
| Config file | `miles-config-r0.yaml` | `miles-config.yaml` |
| Workflow file | `workflow.yaml` | `workflow-resume1.yaml` |

Diff check: the parameters of `workflow-resume1.yaml` are those of
`workflow.yaml`, in the same order. Only `trainer-image`, `replicas`, and
`miles-config` change. In `miles-config`, the header comment and the
`replicas` line with its comment are the only changes. Same as the launch:
gym image, template v10, `gym-replicas` 288, `agent-kwargs` `{}`,
`publish-jobs-dir` `''`, `excluded-nodes`, and the 16384 cap.

The trainer image `miles-glm53-r15-20260927a` is
`sha256:67b57cdf92695b4ec1e3c065cb803349745bc83858b28ef046d59da4cf960247`
in us-east-1 and ap-south-1 (14:06:09Z). Its `/root/miles/scripts/run_arena_harbor.py`
is equal to miles `bc31f88ac` (`feat(arena): make the per-save HF export
opt-in`). A stub run of that launcher on `r44/miles-config.yaml` with
`REPLICA` 16 and `REPLICA_TRAINER` 8 gives 187 argv tokens, no
`--save-hf`, `--actor-num-nodes 8`, `--rollout-num-gpus 64`, and
`--load` = the r44 dir.

### Engine count

- The template sets the PyTorchJob `replicas`, `minReplicas`,
  `maxReplicas`, the pod env `REPLICA`, and the report `GPU_NODES` from the
  `replicas` param. `REPLICA_TRAINER` comes from `replica-trainer`.
- Kueue and TAS count the PyTorchJob pods, so the workload asks for 16
  `p6-b200` nodes. No other pod count exists.
- `scripts/run_arena_harbor.py` sets `--actor-num-nodes` =
  `REPLICA_TRAINER` = 8 and `--rollout-num-gpus` = (`REPLICA` -
  `REPLICA_TRAINER`) x `GPUS_PER_NODE` = (16 - 8) x 8 = 64.
- miles makes `rollout_num_gpus` / `rollout_num_gpus_per_engine` = 64 / 8
  = 8 engines at TP8/EP8.
- The trainer layout does not change: 8 nodes, 64 GPUs, TP8 x PP4 x CP1,
  EP16, DP 2. Thus the DCP loads as it is.
- The `replicas` key in `miles-config.yaml` is a record only. The launcher
  removes it before the trainer argv. It is 16, so that it agrees with the
  param.
- `gym-replicas` stays 288. The gym count covers the publisher in-flight
  cap: `arena_inflight_multiplier` 4 x `rollout_batch_size` 64 = 256
  groups. The engine count does not change that cap.
- Each engine now gets 4 times more requests: at most 256 groups x 8
  samples = 2048 episodes in flight, 256 for each engine (32 engines: 64).
  The router connection limit is `sglang_server_concurrency` 512 x 8 =
  4096, more than 2048.

### Checkpoint (read 2026-09-27 14:05Z from r45 trainer-worker-39)

`slime_experiments/rl-glm53f-auct-cap-r44`:

- `latest_checkpointed_iteration.txt` = 9. `trainer-0.log`: "successfully
  saved checkpoint from iteration 9" at 09:33:41Z.
- `iter_0000009`: 64 `__*_0.distcp` shards, `.metadata`, `metadata.json`,
  and `slime_extra_state.json` (67 entries).
- Sidecar (09:57:57Z): `{"rollout_id": 9, "wandb_run_id": "82i3g7lv"}`,
  45 bytes, no newline.
- `rollout/arena_data_source_state_9.pt` (21593 bytes, 09:57:57Z): offset
  937 of 1034, epoch 0, `sample_group_index` 937, `sample_index` 7496,
  consumed ids for rollouts 0-10 (352), 352 prompt rewards, no timing
  tracker state.
- `hf/rollout_9`: 671 files. The resume does not read it.

The checkpoint resumes as it is. No file changed.

### Launch checklist

1. Done: the r15 trainer image is in ap-south-1 (digest above).
2. `kubectl create --dry-run=server -f r44/workflow-resume1.yaml`. Done
   2026-09-27: accepted (`rl-glm53f44-l44xm`, not created).
3. Before the create, copy
   `/mnt/scratch-s3files-rw/guparpit/logs/rl-glm53f-auct-cap-r44/trainer-0.log`
   to `trainer-0-attempt1.log`. The template `tee` writes the same path
   again, so the resume replaces the first log.
4. `kubectl create -f r44/workflow-resume1.yaml`.
5. Read the trainer-worker-0 argv: `--rollout-num-gpus 64`,
   `--actor-num-nodes 8`, and no `--save-hf`.
6. In `trainer-0.log`, look for:
   - the load of `iter_0000009` from the r44 dir
   - `Restored wandb_run_id=82i3g7lv from checkpoint sidecar.`
   - `Loading arena data source state from .../arena_data_source_state_9.pt`
   - `Resume: loaded 352 consumed instance_ids to skip`
   - a first rollout id of 10
   - 8 SGLang engines
7. The first attempt logged the rollout metrics of rollouts 10 and 11 to
   W&B before the delete. Thus W&B shows two points at those steps.
8. Watch the rollout time, the SGLang queue depth, and the agent timeouts
   against r44 rollouts 0-9. The step-19 save writes only the DCP, the
   sidecar, and the data-source state.
