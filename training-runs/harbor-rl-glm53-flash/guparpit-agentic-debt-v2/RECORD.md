# Run record: guparpit-agentic-debt-v2 (r49) — the r48 config on the refreshed upstream reconcile branch

**Status:** Prepared
<!-- gen-workflow:begin -->
**Date:** 2026-10-01
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v2-`
**Experiment name:** `guparpit-agentic-debt-v2`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v2`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-recon-20261001a`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v1`
<!-- gen-workflow:end -->
**Argo workflow:** not submitted
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Trainer config deltas vs base:** none for training; `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v2`, seeded from r47 `iter_0000059` (2026-10-01 02:51Z)
**Outcome:** Prepared

Starts from r47 checkpoint `iter_0000059`; miles resumes at rollout 60. Sibling of r48 `guparpit-agentic-debt-v1`: the same weights and the same training keys.

## What differs from r48

The trainer code and the image differ. The training keys do not.

- **Code:** miles `arpit-recon-20261001` at `b167425ce1`. That is upstream radixark/miles `main` `d7f1a421` (2026-09-30), plus every fork commit of `arpit-glm-53` and `arpit-r3-datapath`. See `examples/arena/RECONCILE.md`, section "Refresh 2026-10-01".
- **Image:** `arena-slime-dev:miles-glm53-recon-20261001a`, built on base `arena-slime-dev:miles-base-d7f1a42-20261001a`.
- **Base recipe:** upstream `docker/build.py --variant cu13-x86` at `d7f1a421`, with these pins:

  | Component | Pin |
  | --- | --- |
  | SGLang | `sglang-miles` `14a1fa7d` |
  | Megatron-LM | `miles-main` `a84b1054` |
  | fla | 0.5.2, with the KDA patch and `fla_conv_int64_offsets.py` |
  | TileLang | 0.1.14, with the #3647 Blackwell patch |

- **Config:** `miles-config.yaml` is the r48 file. Only the run identity changes: `experiment_name`, `project_name`, and `arena_sample_summary_dir`. r48 writes its sample summaries into the r47 folder. This run has its own folder.

Behavior changes that come with the upstream base, for this config:

| Area | r48 (fork) | r49 (this branch) |
| --- | --- | --- |
| GLM-5.3 DSA indexer | raw query | RMSNorm on the query, as SGLang. It changes the top-k selection and the log-probs |
| GLM-5.3 MLP | no clamp | `--activation-func-clamp-value 10`, the SGLang `swiglu_limit` |
| Failed engine | killed at the first failed health check, recovered before the next weight update | the mini fault-tolerance controller heals it on port 18080, at any time |
| Event logger | off | on: a checksum on each engine at each weight update, and an `events/` copy at each save |
| `train_rollout_logprob_abs_diff` | rollout log-probs against trainer log-probs | trainer-scored log-probs (#3655); with one optimizer step per rollout the value means the same |
| R3 routing | int32 | int16 in the trainer shards, widened to int32 at the router; the same indices |
| `rollout/weight_version/*` | upstream metrics | the same keys from `Sample.metadata["arena_weight_versions"]` (ADR-0018 amendment) |

## Checks done before launch

| Check | Result |
| --- | --- |
| Branch tests (venv, see RECONCILE) | arena 343 passed, 1 known failure (`test_training_runs_index`); launcher 65 passed, 1 upstream failure; snapshots 6 passed |
| Adversarial review (3 lenses) | 4 major findings: 3 fixed, 1 deferred to the T1 job below |
| Image | `/root/miles` at `b167425ce1`, clean tree; SGLang `14a1fa7`; Megatron `a84b10547`; fla conv int64 patch applied; fla KDA patch already upstream |
| Full `parse_args` of the r49 argv in the image, Megatron included, B200 device stubbed | passes. World 64, TP 8 with SP, PP 4, EP 16, DP 2. `glm5_next_kda_tp` true, multiplier 4, length coefficient 0.1, `skip_actor_forward_only` true, TIS and R3 on, GBS 256, lr 1.5e-6, `save_interval` 10, `wandb_random_suffix` false |
| Checkpoint copy | 66 objects, names and sizes equal to r47 `iter_0000059`; sidecar `{"rollout_id": 59}`; `latest_checkpointed_iteration.txt` 59; `rollout/arena_data_source_state_59.pt` |

## Upstream check

- The run uses the upstream base recipe and pins only the branch heads that the upstream Dockerfile tracks on 2026-10-01.
- Engine recovery: the upstream mini fault-tolerance controller replaces the fork recovery path. The config leaves `mini_ft_controller_enable` unset. A YAML `false` emits no flag, and turning it off would leave a dead engine dead.
- Not included: the upstream PR #3608 DSA kernel stack (`arpit-dsa-3608`). It is an open PR and it changes the DSA numerics.

## Measurement plan

| Question | Signal | Gate |
| --- | --- | --- |
| KDA TP checkpoint and weight sync on the new base | T1 job: HF gather SHA-256 of `iter_0000059`, r17 image against this image; log-probs on real rows | every tensor equal; log-prob difference within 2 x the same-image floor + 1e-3 |
| Checkpoint load | `successfully loaded checkpoint ... at iteration 59` | present |
| Engine path | `perf/update_weights_time` and the checksum time per update | update succeeds; the time is recorded against r48 (about 25 s) |
| First step | `train/train_rollout_logprob_abs_diff`, `train/grad_norm` | near r48: 0.048 to 0.050, and 0.033 to 0.042 |
| Staleness | `rollout/weight_version/*` in the perf line and W&B | present; max - min at most 4 by rollout 65 |
| Outcome | `rollout/group_metrics/reward.mean`, `rollout/episode_response_length/mean`, `perf/step_time` | compare with r48 at the same rollout ids |

If a gate fails, report it. Ask the user before you stop the run.

## Launch

<!-- filled at launch -->
