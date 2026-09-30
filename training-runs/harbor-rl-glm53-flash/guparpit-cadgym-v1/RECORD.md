# Run record: guparpit-cadgym-v1 — cadgym, first RL run, r44 recipe with the r47 trainer settings

**Status:** Prepared
<!-- gen-workflow:begin -->
**Date:** 2026-09-30
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-cadgym-v1-`
**Experiment name:** `guparpit-cadgym-v1`
**W&B project:** `cadgym` (group `guparpit-cadgym-v1`)
**Dataset:** `lakefs://arena-inspect/91b6618d09ea79c79b5b5167880ebdb4bf60fc6e7ba2361435322e1793a74a52/internal/cadgym-20260917/current/manifest.jsonl` (gym `cadgym`)
**Manifest commit:** `91b6618d09ea79c79b5b5167880ebdb4bf60fc6e7ba2361435322e1793a74a52`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r17-20260928a`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `r44`
<!-- gen-workflow:end -->
**Argo workflow:** not created. The server dry run of `workflow.yaml` passed on 2026-09-30 (Timeline).
**W&B run:** none yet. The project `cadgym` does not exist on 2026-09-30; `wandb.init` creates it. The run name and the group are `guparpit-cadgym-v1` (`disable_wandb_random_suffix: true`).
**Task pin:** `64ea81c53f019a11f787139a35e91f2051f3e450c07d3beeb1222c8c4ea78d5d` (`lakefs_commit_id` of each of the 199 manifest rows; manifest md5 `fdc9f818b221ea072ab45d4547983459`, 51,095 bytes)
**Image digests:** gym `sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d`, trainer `sha256:e6f04a9ca1abc9df17a7bbf9e3d2aaf6a643f40ed5a457c98a4114719972bcd0` (ECR `arena-slime-dev`, ap-south-1, `describe-images` read 2026-09-30)
**Trainer config deltas vs base:** `prompt-data-list` caponly-1034 on `dev` (gym `auctioneer-caponly`) -> cadgym at `91b6618d` (gym `cadgym`); `rollout_batch_size` 64 -> 32; `arena_inflight_multiplier` 4 -> 8; `skip_actor_forward_only` unset -> `true`; `glm5_next_kda_tp` unset -> `true`; `arena_length_reward_coef` 0 -> 0.10; `disable_wandb_random_suffix` unset -> `true`; `wandb_project` `rl-glm53f-auct-cap` -> `cadgym`; `experiment_name`, `project_name`, `arena_sample_summary_dir` renamed. Workflow parameters: `trainer-image` `miles-glm53-r15-20260927a` -> `miles-glm53-r17-20260928a`; `gym` `auctioneer-caponly` -> `cadgym`; `ack-wait` 36000 -> 39600; `trainer-task-deadline-secs` 39600 -> 43200; `excluded-nodes` 83 -> 86 (the r47 list). Base is `r44/workflow-resume1.yaml` and `r44/miles-config.yaml`.
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-cadgym-v1` (S3 `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/guparpit-cadgym-v1/`). No save yet; the prefix is empty on 2026-09-30.
**Outcome:** Prepared, not launched.

## Goal

Train cadgym from the base model on the r44 recipe with the trainer
settings that the user approved for agentic debt on 2026-09-30 ("give
cadgym the new settings you wanted for agentic-debt"). The dataset is
`lakefs://arena-inspect/main/internal/cadgym-20260917/current/`, pinned at
commit `91b6618d`. The run tests whether the raw reward
(`rollout/group_metrics/reward.mean`) moves up from its rollout 0-4
baseline. The run ends at `num_rollout` 300, or when the user retires it.

## Setup

### Dataset

| Item | Value |
| --- | --- |
| URI (user) | `lakefs://arena-inspect/main/internal/cadgym-20260917/current/` |
| `prompt-data-list` path | `lakefs://arena-inspect/91b6618d09ea79c79b5b5167880ebdb4bf60fc6e7ba2361435322e1793a74a52/internal/cadgym-20260917/current/manifest.jsonl` |
| Manifest commit | `91b6618d` ("gym-lineage: publish run 3", 2026-09-24), the last change under the path. The two-dot diff of `main` head `31da1243` against `91b6618d` under `internal/cadgym-20260917/` is empty. |
| Rows | 199 of 200 tasks. `offset-plate-00002262` failed its oracle check and is left out. |
| Task pin | every row pins `64ea81c5`. NEVER read the manifest at `64ea81c5`: that commit holds the stale run-2 manifest (md5 `6d74f3c4...`). |
| Tasks | 8 part shapes, each in CadQuery (99) and OpenSCAD (100). One agent container (2 CPU, 4 GB) and one separate verifier container (4 CPU, 8 GB). No network, no steps, no MCP servers, no GPUs. |
| Timeouts (all tasks) | agent 14400 s, verifier 900 s, build 1800 s |
| Reward | Harbor `reward` key, programmatic (gmsh mesh and CalculiX solve). 0 when a geometry gate fails; 0.25 when geometry passes and a structural gate fails; 0.5 + 0.5 x `mass_score` when all gates pass. |
| Gym name | `cadgym` (lakeFS `gym_id` `cadgym-20260917`). Not in `KNOWN_GYMS`, and it does not need to be: `--gym` names only the NATS subjects. Not the same gym as `cad-bench`. |

Sources: `/workplace/guparpit/kdfast/scratch/cadgym-v1/dataset.md` and
`history.md` (2026-09-30, read-only checks of lakeFS, ECR, and the r44 gym
pod).

### Delta vs r44

| Item | Base r44 (resume 1) | This run |
| --- | --- | --- |
| Dataset | caponly-1034 on `dev` (1,034 tasks), gym `auctioneer-caponly` | cadgym at `91b6618d` (199 tasks), gym `cadgym` |
| Start point | r44 `iter_0000009` | base DCP (`ref_load`), rollout id 0 |
| Trainer image | `miles-glm53-r15-20260927a` | `miles-glm53-r17-20260928a` (miles `4716a367a`), as r47 |
| `glm5_next_kda_tp` | unset | `true`, as r47 (ADR-0016 first port) |
| `skip_actor_forward_only` (flip A) | unset | `true`, as r47 |
| `rollout_batch_size` / `arena_inflight_multiplier` | 64 / 4 | 32 / 8. Flip A requires 256 = 32 x 8. The in-flight cap stays 256 groups. |
| `arena_length_reward_coef` | 0 | 0.10 (length-reward check PASS, `cadgym-v1/length-reward.md`) |
| `disable_wandb_random_suffix` | unset (run name `<group>_<id>-RANK_0`) | `true` (run name and group `guparpit-cadgym-v1`) |
| W&B project | `rl-glm53f-auct-cap` | `cadgym` |
| `ack-wait` / `trainer-task-deadline-secs` | 36000 / 39600 | 39600 / 43200 |
| `excluded-nodes` | 83 ids | 86 ids (the r47 list) |

Unchanged from r44 resume 1: gym image `gym-glm53-adr72-20260927a`,
template `guparpit-miles-deployer-v10` (uid
`16ddd530-501f-4734-82ae-88500671bed8`, generation 1; the live template
equals `r44/guparpit-miles-deployer-v10.yaml`), `replicas` 16 (8 actor
nodes, 8 SGLang engines), `gym-replicas` 288, `agent-timeout-multiplier`
2, `agent-kwargs` `{}` (4 Vulcan compactions), `publish-jobs-dir` `''`,
`rollout_max_response_len` 16384, window 131072, radix cache and overlap
schedule on, GBS 256, `n_samples_per_prompt` 8, `lr` 1.5e-6, TIS, R3,
full recompute, per-sample loss average, `save_interval` 10, `num_rollout`
300, no HF export.

### Deadline rule

Harbor 0.22 multiplies only the agent timeout
(`JobConfig.agent_timeout_multiplier`; `timeout_multiplier` stays 1.0, read
from the r44 gym pod). The 8 trials of a group run at the same time
(`n_concurrent_trials` 8). The longest trial is therefore:

| Phase | Seconds |
| --- | --- |
| agent environment build | 1800 |
| agent setup | 360 |
| agent, 14400 x 2 | 28800 |
| verifier environment build | 1800 |
| verifier | 900 |
| total | 33660 (9.35 h) |

The gym cuts a message at `ack-wait` - 300 = 39300 s. That is 5640 s (17%)
above the longest trial, for the task download, teardown, and DinD load.
r47 used the same idea: 20 h against a longest observed task of 15.05 h.
The trainer deadline is `ack-wait` + 3600 = 43200 s, the r44 and template
relation.

### Names

| Name | Value | How it gets there |
| --- | --- | --- |
| Argo generateName | `guparpit-cadgym-v1-` | `workflow.yaml` `metadata.generateName` |
| `experiment-name` | `guparpit-cadgym-v1` | template pod env `EXPERIMENT_NAME` and `PROJECT_NAME` |
| Checkpoint dir | `.../checkpoints/slime_experiments/guparpit-cadgym-v1` | template `ARENA_CHECKPOINTS_DIR` + launcher `ckpt_dir` (`--load` = `--save`) |
| W&B group | `guparpit-cadgym-v1` | launcher appends `--wandb-group $EXPERIMENT_NAME` |
| W&B run name | `guparpit-cadgym-v1` | `disable_wandb_random_suffix: true` |
| W&B project | `cadgym` | `wandb_project` |
| Routing, logs, sample summary | `.../routing/guparpit-cadgym-v1`, `.../logs/guparpit-cadgym-v1`, `.../debug/guparpit-cadgym-v1/sample_summary` | template env and `arena_sample_summary_dir` |

All five S3 prefixes are empty on 2026-09-30.

### Gates and metrics

Read the raw reward from `rollout/group_metrics/reward.mean` (gym side).
With the length term on, `rollout/raw_reward`, `rollout/episode_raw_reward`,
the reward percentiles, and the sample summary `reward` show the shaped
value (`cadgym-v1/length-reward.md`, condition 1).

1. Smoke: the first groups return a `reward` from the separate verifier.
   This is the first training run with a separate verifier container.
2. Length term on: the trainer log shows `length-reward rescued=`.
3. Baseline: `rollout/group_metrics/reward.mean` and
   `rollout/avg_response_length` for rollouts 0-4.
4. Signal: `rollout/zero_reward_groups_frac` and
   `rollout/dyn_sampling_drop_frac`. Near 1.0 means that most groups give
   no gradient and the length term is idle.
5. Timeouts: the share of trials that end by `AgentTimeoutError` or
   `VerifierTimeoutError`, `rollout deadline exceeded` log lines, and
   `rollout/dropped_groups/deadline`.
6. DinD: `dind` restarts and `OOMKilled` counts in the gym pods.

Not measured by this configuration: the cube exploit alarm (0.25 with
fewer than 8 gates) and the fake-mass alarm (1.0 far below the anchor).
Both need the extra `reward.json` channels. The gym sends them as
`grader_metadata`, but the trainer does not keep them, and
`publish-jobs-dir` is `''`.

### Risks

- Separate verifier container: no earlier training run used one.
- Sparse reward: the vendor rollouts (Opus 4.8, Terminus-2, 7200 s) have
  111 of the 199 tasks at all zero, and no group tied above zero. The filter
  drops each group with no reward spread.
- Agent timeout: since r44 moved to 8 engines, 91-100% of its episodes end
  at the agent timeout. Here the timeout is 8 h, so a group can take 9 h.
- Length term: a group tied at a partial reward (for example all 8 at
  0.25) is kept, and its shortest episode gets the top advantage. The 0.25
  level holds a known cheap exploit (a small cube scores 0.25). Watch for
  a rise in the 0.25 share.
- DinD: 8 trials share 8 CPU and 64 GiB. The container limits sum to more
  than that.
- 199 tasks and a 256-group in-flight cap: the first publish reaches into
  epoch 1, so one task can have two groups in flight.

### Launch checklist

- The dataset URI with its pinned commit is in the launch message.
- `kubectl --context arena-prod-bom-v2 -n arena-tasks create -f workflow.yaml`.
- Kueue can place the trainer on nodes that Karpenter removes. On
  `EvictedDueToNodeFailures`, add the failed nodes to `excluded-nodes` and
  create the trainer PyTorchJob again from the rendered manifest, at most
  twice.

## Timeline

| UTC | Event |
| --- | --- |
| 2026-09-30 06:20Z | Prepared. The five S3 prefixes are empty, and the W&B project `cadgym` does not exist. |
| 2026-09-30 06:24:41Z | Server dry run accepted `guparpit-cadgym-v1-bzrv8` (not created). The parameters it returned equal `workflow.yaml`. |

## Results

None yet.

## Issues

- None yet.

## Follow-ups

- Keep `grader_metadata` (or publish the Harbor job dirs) so the cube and
  fake-mass alarms can run. Owner: operator.
- `gen-workflow.py` takes only an integer run number. This folder was
  written with its functions from a one-off script
  (`/workplace/guparpit/kdfast/scratch/cadgym-v1/prep/make_workflow.py`).

## Sources

- Run files: `miles-config.yaml`, `workflow.yaml` (this folder).
- Base: `r44/workflow-resume1.yaml`, `r44/miles-config.yaml`,
  `r44/RECORD.md`. Trainer settings: `r47/miles-config.yaml`,
  `r47/BUILD.md` ("Flip A").
- Checks (2026-09-30, read-only): `/workplace/guparpit/kdfast/scratch/cadgym-v1/`
  `dataset.md`, `history.md`, `length-reward.md`, `prepare-cadgym.md`.
- Launcher and W&B code: miles `4716a367a` `scripts/run_arena_harbor.py`,
  `miles/utils/tracking_utils/wandb_utils.py`.
- Template: `kubectl get wftmpl guparpit-miles-deployer-v10 -o yaml`.
