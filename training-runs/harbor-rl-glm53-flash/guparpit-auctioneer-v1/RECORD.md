# Run record: guparpit-auctioneer-v1 — auctioneer caponly-1034 (the r44 gym side) on the r52/r53 trainer stack

**Status:** Stopped (never trained)
<!-- gen-workflow:begin -->
**Date:** 2026-10-04
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-auctioneer-v1-`
**Experiment name:** `guparpit-auctioneer-v1`
**W&B project:** `auctioneer` (group `guparpit-auctioneer-v1`)
**Dataset:** `lakefs://arena-inspect/dev/internal/auctioneer/caponly/caponly-1034/manifest.jsonl` (gym `auctioneer-caponly`)
**Manifest commit:** `dev`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-recon-20261002-flashmla-evfix@sha256:7ae37cd518c89834738aad94153b7df4cb9b7073354b0b4312319b3d39745ce3`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `r44` (resume 1) for the gym side and the dataset; `guparpit-agentic-debt-v7` for the trainer
<!-- gen-workflow:end -->
**Argo workflow:** `guparpit-auctioneer-v1-gzq6l`
**W&B run:** not created. The project `auctioneer` is new (user naming rule 2026-09-30); `wandb.init` creates it. Run name and group are `guparpit-auctioneer-v1` (`disable_wandb_random_suffix: true`).
**Task pin:** `e5ef91b0…` (the manifest rows pin this lakeFS commit; the URI ref `dev` is a branch). Source: `r44/RECORD.md` and `r39/BUILD.md`. Not re-verified on 2026-10-04: this host has no lakeFS client. Add the `Pulled lakefs://...` log line at launch.
**Image digests:** gym `sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d` (pushed 2026-09-27 04:44Z), trainer `sha256:7ae37cd518c89834738aad94153b7df4cb9b7073354b0b4312319b3d39745ce3` (pushed 2026-10-02 19:17Z; pinned in `workflow.yaml`). Both read from ECR `arena-slime-dev` (registry 427267593057, ap-south-1, `describe-images`) on 2026-10-04; the local us-east-1 copy of the trainer tag has the same digest
**Trainer config deltas vs base:** against `r44/miles-config.yaml` (resume 1): `expert_model_parallel_size` 16 -> 8; `glm5_next_kda_tp` unset -> `true`; `skip_actor_forward_only` unset -> `true`; `miles_dsa_sparse_attention_forward_backend` unset (`tilelang`) -> `flash_mla`; `sglang_moe_runner_backend` unset (`auto`) -> `triton`; `calculate_per_token_loss` unset -> `true`; `disable_grpo_std_normalization` unset -> `true`; `arena_length_reward_coef` `0` -> `0.0` (same value); `rollout_batch_size` 64 -> 32 (Setup, "Batch shape"); `disable_wandb_random_suffix` unset -> `true`; `wandb_project` `rl-glm53f-auct-cap` -> `auctioneer`; `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run. Unchanged: `prompt-data-list`, `replicas` 16, `num_trainers` 8, TP 8 with `sequence_parallel`, PP 4 with the 11/11/11/12 split, `use_rollout_routing_replay`, `use_tis`, `lr` 1.5e-6, `save_interval` 10, `no_load_optim`, `global_batch_size` 256, `n_samples_per_prompt` 8, `arena_inflight_multiplier` 4, `rollout_max_response_len` 16384, window 131072, radix cache and overlap schedule on, `num_rollout` 300. Workflow parameters against `r44/workflow-resume1.yaml`: `trainer-image` `miles-glm53-r15-20260927a` -> `miles-glm53-recon-20261002-flashmla-evfix@sha256:7ae37cd5…`; `excluded-nodes` 83 -> 262 ids; the other ten parameters are equal.
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-auctioneer-v1` (S3 `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/guparpit-auctioneer-v1/`). The prefix does not exist on 2026-10-04 (read-only `s3 ls`), so the trainer loads `ref_load` (the base DCP) and rollout ids start at 0. The `debug`, `routing`, and `logs` prefixes of this name are empty too.
**Outcome:** Stopped 2026-10-05 05:14Z on the owner's word. The trainer PyTorchJob never ran (see Launch). Resubmit on the default recipe is pending.

## Goal

Train the auctioneer gym from the base model with the trainer stack that r52
validated for speed (EP8 and the FlashMLA DSA forward: 42,593 tok/s at step
61) and the r53 loss that held reward and effort on agentic debt (no length
bonus, token-level loss, no spread division). The gym side and the dataset are
r44's, the last healthy auctioneer run (reward per 5 rollouts 0.35 at 0-9 to
0.54 at 35-39). r44 is the control at the same rollout ids. The run ends at
`num_rollout` 300, on a user retire, or on a collapse like r39 and r42.

## Setup

| Item | Base r44 (resume 1) | This run |
| --- | --- | --- |
| Trainer image | `miles-glm53-r15-20260927a` (miles `bc31f88ac`) | `miles-glm53-recon-20261002-flashmla-evfix` (miles `0ffc4044ec`, branch `arpit-r52-flashmla`), as v7 |
| `expert_model_parallel_size` | 16 | 8 (the r50/r52 arm: -13.4% step, peak 109.7 GiB per GPU) |
| `glm5_next_kda_tp` | unset | `true` (TP-sharded KDA, as r47 and later) |
| `skip_actor_forward_only` | unset | `true` (flip A: the training forward supplies the old log-probs) |
| DSA sparse-attention forward | TileLang | FlashMLA (`miles_dsa_sparse_attention_forward_backend: flash_mla`, upstream PR #3608) |
| `sglang_moe_runner_backend` | unset (`auto`) | `triton`: routing replay (R3) needs the materialized top-k ids on the new SGLang |
| Loss | per-sample average, GRPO std normalization, no length term (`arena_length_reward_coef` 0) | `calculate_per_token_loss` true, `disable_grpo_std_normalization` true, `arena_length_reward_coef` 0.0 |
| `rollout_batch_size` x `n_samples_per_prompt` / `global_batch_size` | 64 x 8 = 512 / 256 (two optimizer steps per rollout) | 32 x 8 = 256 / 256 (one step per rollout) |
| Start point | r44 `iter_0000009` (resume 1) | base DCP (`ref_load`), rollout id 0 |
| W&B | project `rl-glm53f-auct-cap`, run `<group>_<id>-RANK_0` | project `auctioneer`, run and group `guparpit-auctioneer-v1` |
| `excluded-nodes` | 83 ids | 262 ids: the r52 list (215) plus the 47 `p6-b200.48xlarge` nodes that were NotReady or carried a taint other than `nvidia.com/gpu` and `vpc.amazonaws.com/efa` on 2026-10-04 (`arena.agif.amazon.dev/burn-in`, `karpenter.sh/disrupted`, `node.kubernetes.io/unreachable`); equal to the v7 list |

Unchanged from r44 resume 1: dataset caponly-1034 on `dev` (1,034 tasks), gym
`auctioneer-caponly`, gym image `gym-glm53-adr72-20260927a`, template
`guparpit-miles-deployer-v10`, `replicas` 16 (8 actor nodes and 8 SGLang
engines: the launcher derives `--rollout-num-gpus` (16 - 8) x 8 = 64, and 64 /
`rollout_num_gpus_per_engine` 8 = 8), `gym-replicas` 288, `ack-wait` 36000,
`agent-timeout-multiplier` 2, `trainer-task-deadline-secs` 39600,
`agent-kwargs` `{}` (4 Vulcan compactions), `publish-jobs-dir` `''`,
`arena_inflight_multiplier` 4, `rollout_max_response_len` 16384, window
131072, radix cache and overlap schedule on, TP 8 with sequence parallel, PP 4
with the 11/11/11/12 split, R3, TIS, `lr` 1.5e-6, `save_interval` 10,
`no_load_optim`, `num_rollout` 300, no HF export.

### Batch shape

`skip_actor_forward_only` requires one optimizer step per rollout:
`validate_skip_actor_forward_only` in `miles/utils/arguments.py` asserts
`global_batch_size == rollout_batch_size x n_samples_per_prompt`. r44 ran
64 x 8 = 512 samples per rollout over two steps of 256. The in-image parse of
the r44 shape with this trainer fails:
`AssertionError: --skip-actor-forward-only requires exactly one optimizer step
for 512 rollout samples; got --global-batch-size 256`. This record keeps the
256-sample optimizer step and halves the samples per rollout
(`rollout_batch_size` 32), as `guparpit-cadgym-v1` and
`guparpit-agentic-debt-v7` did when they took this flag from r44. Effects: each
rollout trains on 256 samples, so 300 rollouts cover half the data of r44's
300; with `arena_inflight_multiplier` 4 the in-flight cap is 128 groups, not
r44's 256. The alternative that keeps 512 samples per rollout is
`global_batch_size` 512 (one step of 512, a different optimizer batch than
r44). The user did not approve either in advance. MUST decide before launch.

## Upstream check

- `--calculate-per-token-loss` (DAPO) and `--disable-grpo-std-normalization`
  (Dr. GRPO, arXiv 2503.20783) are upstream flags (r53 and v7 records). No
  upstream framework puts a length term in the reward.
- FlashMLA: upstream radixark/miles PR #3608 head `557fb097`, ported line for
  line (v5 record); T1 parity passed on 2026-10-02 (37,534 of 37,534
  weight-sync tensors bitwise equal, log-probs at the noise floor).
- `--sglang-moe-runner-backend triton`: the upstream GLM launcher
  `scripts/run_glm5_3_flash.py` (l.133-134) forces it for routing replay
  ("routing replay needs materialized topk ids; the SM100 default resolves to
  a fused runner"). The arena launcher takes the config key. The image code
  has no launch-time assertion for it; the guard is the runtime check in
  `miles/rollout/sglang_rollout.py` ("routed_experts payload is all zeros").
- `--skip-actor-forward-only`: upstream flag; `validate_skip_actor_forward_only`
  requires one optimizer step per rollout (Batch shape).
- EP8 and TP-sharded KDA: the r50 and r47 arms of the
  `trainer-core-profile-glm53-flash` study.

## Measurement plan

Compare with r44 at the same rollout ids (`82i3g7lv`, W&B project
`rl-glm53f-auct-cap`). The reward source is `rollout/group_metrics/reward.mean`
(the gym reward; no length term is on).

| Question | Signal | Gate |
| --- | --- | --- |
| Flags live | the Megatron argument dump in the trainer log | `calculate_per_token_loss True`, `grpo_std_normalization False`, `arena_length_reward_coef 0.0`, `miles_dsa_sparse_attention_forward_backend flash_mla`, `sglang_moe_runner_backend triton`, `expert_model_parallel_size 8`, `glm5_next_kda_tp True`, `skip_actor_forward_only True` at start |
| Fresh start | the first trainer log: load path, rollout id; W&B | load from `ref_load`, rollout 0, run `guparpit-auctioneer-v1` in project `auctioneer`, group the same |
| Speed | `perf/actor_train_tok_per_s` from step 10 on | at least 38,000 (r52 on agentic debt: 42,593 at step 61; ESTIMATE for this gym and the 256-sample step) |
| Memory | `memsample-*.log` peak, no OOM | below 170 GiB per GPU |
| Reward | `rollout/group_metrics/reward.mean` per 5 rollouts | at or above r44 at the same ids (r44: 0.354 at 0-4, 0.54 at 35-39) |
| No collapse | `rollout/avg_response_length`, `episode_response_length/mean`, `clipped_turns` | mean episode length stays below 100K tokens and reward stays above 0.30 (r39 and r42 collapsed past 100K); no "do less" fall to a third of the rollout 0-9 length (r52) |
| Window | SGLang HTTP 400 on the 131072 window in the engine logs | none (r44 fixed this with the 16384 cap) |
| Rollout time | `perf/rollout_time`, `stop/timeout` share | record; r44 with 8 engines made 8 rollouts per 4 h with 91-100% agent timeouts |
| Reaper risk | `queue=` in `Rollout N complete`, 60-min mean GPU power | the queue must not idle the engines (r47, r50 were reaped) |

## Checks done before launch

| Check | Result |
| --- | --- |
| `miles-config.yaml` vs `r44/miles-config.yaml` (resume 1) | 7 new keys (`sglang_moe_runner_backend`, `glm5_next_kda_tp`, `miles_dsa_sparse_attention_forward_backend`, `disable_wandb_random_suffix`, `calculate_per_token_loss`, `disable_grpo_std_normalization`, `skip_actor_forward_only`), no removed key; value changes `expert_model_parallel_size`, `rollout_batch_size`, `wandb_project`, and the 3 identity keys |
| `miles-config.yaml` vs `guparpit-agentic-debt-v7/miles-config.yaml` | same key set; values differ in `replicas` (40 -> 16), `prompt-data-list`, `wandb_project`, and the 3 identity keys only |
| `workflow.yaml` vs `r52-final2.yaml` (the v5 launch) | `generateName`, `experiment-name`, `replicas` 40 -> 16, `excluded-nodes` 215 -> 262 (47 added, 0 removed), `miles-config`, `gym`, `ack-wait`, `agent-timeout-multiplier`, `trainer-task-deadline-secs`, `publish-jobs-dir`; unchanged `username`, `trainer-image`, `replica-trainer`, `gym-replicas`, `gym-image`, `agent-kwargs`. The embedded `miles-config` is the JSON string of `miles-config.yaml` and round-trips byte for byte |
| `workflow.yaml` vs `r44/workflow-resume1.yaml` | `gym`, `gym-image`, `gym-replicas`, `ack-wait`, `agent-timeout-multiplier`, `trainer-task-deadline-secs`, `agent-kwargs`, `publish-jobs-dir`, `replicas`, `replica-trainer` all equal |
| In-image parse (`recon2/parse/fullparse-auct.py` in `miles-glm53-recon-20261002-flashmla-evfix`, `WORLD_SIZE` 64, `RANK` 0, `replicas` 16, `num_trainers` 8, `LD_LIBRARY_PATH` with the CUDA compat `libcuda.so.1`; `parse-auct.out`) | exit 0, 266 argv tokens. `calculate_per_token_loss True`, `grpo_std_normalization False`, `arena_length_reward_coef 0.0`, `miles_dsa_sparse_attention_forward_backend flash_mla`, `sglang_moe_runner_backend triton`, `expert_model_parallel_size 8`, `glm5_next_kda_tp True`, `skip_actor_forward_only True`, `use_rollout_routing_replay True`, `use_routing_replay True`, `rollout_batch_size 32`, `n_samples_per_prompt 8`, `global_batch_size 256`, `num_steps_per_rollout None`, `lr 1.5e-06`, `save_interval 10`, `no_load_optim True`, `prompt_data` = the r44 caponly-1034 row, `save` `.../slime_experiments/guparpit-auctioneer-v1`, `load` = `ref_load` (the save dir has no checkpoint), `wandb_project auctioneer`, `wandb_group guparpit-auctioneer-v1`, `wandb_random_suffix False`, `actor_num_nodes 8`, `rollout_num_gpus 64`, `rollout_num_gpus_per_engine 8`, `world_size 64`, `data_parallel_size 2`, `num_layers 45` |
| In-image parse of the r44 batch shape (`auct-r44shape.yaml`: `rollout_batch_size` 64; `parse-auct-r44shape.out`) | `AssertionError: --skip-actor-forward-only requires exactly one optimizer step for 512 rollout samples; got --global-batch-size 256` |
| S3 `slime_experiments/` (read-only `s3 ls`, 74 prefixes) | no `guparpit-auctioneer-v1` prefix; `debug`, `routing`, `logs` have none either |
| Nodes (read-only `kubectl get nodes`, `p6-b200.48xlarge`) | 302 nodes; 47 NotReady or with a foreign taint; the union with the r52 list is 262 ids |
| Record tests (`pt.sh`, `STUB=recon2/tmp/stub`: `test_training_runs_index.py`, `test_training_run_identity.py`, `test_run_arena_harbor.py`) | 19 passed, 14 warnings in 1.82 s |
| Image | the local `us-east-1` copy of the tag has digest `7ae37cd5…`; `/root/miles` is `0ffc4044ec`, a clean tree |

## Launch

| Time (UTC) | Event |
| --- | --- |
| 2026-10-04 18:19 | Workflow `guparpit-auctioneer-v1-gzq6l` created with 262 nodes excluded (every NotReady or tainted B200 node at launch). Fresh start from the base DCP; in-image parse confirmed the flags (see Parse results). |
| 2026-10-04 18:23:02 | Trainer PyTorchJob `guparpit-auctioneer-v1-gzq6l-trainer` created; Kueue admitted it at 18:23:05 and the operator created 16 worker pods. |
| 18:23:15 | The training-operator deleted the PyTorchJob 12 s after creation, while the pods were still scheduling (EKS audit log: `delete pytorchjobs` by `kubeflow-system:training-operator`, then the garbage collector deleted the pods). No pod reached `Failed` before the delete, and no `PyTorchJobFailed` event exists. `runPolicy` was `ttlSecondsAfterFinished: 0`, no `activeDeadlineSeconds`, `restartPolicy: Never`. The operator logs are closed to `guparpit` (RBAC), so the code path is unknown. r54 and r55, submitted 15 min earlier on the same template, were not affected. |
| 2026-10-05 05:14 | The workflow had waited 11 h on `wait-trainer-nats` with one NATS pod and no GPU. Stopped with `spec.shutdown: Stop` (onExit cleanup ran). Audit queries: `/workplace/guparpit/kdfast/scratch/recon2/audit.sh`. |

| UTC | Event |
| --- | --- |

## Sources

- Run files: `miles-config.yaml`, `workflow.yaml` (this folder).
- Base: `arpit-glm-53:training-runs/harbor-rl-glm53-flash/r44/` (`workflow-resume1.yaml`, `miles-config.yaml`, `RECORD.md`); trainer: `guparpit-agentic-debt-v7/miles-config.yaml`, v5 and v7 records; workflow shape: `/workplace/guparpit/kdfast/scratch/resume1002/r52-final2.yaml`.
- Checks: `/workplace/guparpit/kdfast/scratch/recon2/parse/` (`fullparse-auct.py`, `auct.yaml`, `auct-r44shape.yaml`, `parse-auct.out`, `parse-auct-r44shape.out`); `/workplace/guparpit/kdfast/scratch/tmpdir/auct/` (`p6nodes.json`, `bad_nodes.json`, `make_workflow.py`).
- Validator: `miles/utils/arguments.py` `validate_skip_actor_forward_only`; R3 backend: `scripts/run_glm5_3_flash.py` l.133-134, `miles/rollout/sglang_rollout.py`.
