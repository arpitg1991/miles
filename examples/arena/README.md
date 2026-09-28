# Arena examples: Harbor RL over NATS on miles

Index of the arena Harbor RL jobs built on `miles_plugins/arena` (the AGISlime
-> miles port of the NATS-gym training path). Each subdirectory is the trainer
side of one job: the training YAML that `scripts/run_arena_harbor.py` converts
to the miles argv, the PyTorchJob and Service manifests, a README runbook, and
a `RUNLOG.md` with every launch (identity, image, shape, numbers, root causes,
decisions). Decisions with lasting effect are ADRs in `miles_plugins/arena/adr/`.
The GLM-5.3 run records (`r<N>/`, `RUNLOG.md`, `experiment-list.md`) live in
`training-runs/harbor-rl-glm53-flash/`; `training-runs/README.md` holds the
record convention and the index.
State as of 2026-09-05: GLM-5.3-Flash r7 (`rl-glm53f-gbash-r7`) is running.

## Layout

```
examples/arena/
  Dockerfile                          trainer image: <miles base> + arena deps + EFA layer + this tree + fla patch
  patches/fla_kda_next_power_of_2.py  fla 0.4.2 x triton 3.7.1 KDA kernel fix (image layer; no-op without fla)
  harbor-rl-27b/                      Qwen3.5-27B + financeagent smoke (port of AGISlime ADR-0039); CPU-validated only
  harbor-rl-27b-snorkel/              Qwen3.5-27B + snorkel-general-bash-harbor, plain stack (ADR-0007)
    smoke-3node/                      as-applied manifests of the only 27B GPU run, rl-milesgb1-smoke1
  harbor-rl-glm53-flash/              GLM-5.3-Flash (321B MoE, glm5_next) + snorkel-general-bash-harbor, r1-r7
    memprobe/                         single-node SGLang host-memory probe (run-2 OOM hypotheses R1-R5)
```

## Lineage

```
AGISlime tmp_staging/smoke/harbor-rl-27b (ADR-0039 2026-08-18; run rl-smoke27-financeagent)
  --> harbor-rl-27b/                 launcher + argv parity (old 106 / new 97); never GPU-launched on miles
AGISlime snorkel r1 .. run5 (2026-08-26 .. 08-29; rl-snork27b-gbash-r5)
  --> harbor-rl-27b-snorkel/         plain stack: r2 zero-variance filter kept, r3b/r4/r5 customs descoped
        --> smoke-3node/             rl-milesgb1-smoke1 (2026-09-01 16:39 -> 09-02 00:34 PT): 4/4 rollouts, ray SUCC
        --> harbor-rl-glm53-flash/   arena wiring / batch shape / GRPO block from the snorkel example;
              |                      model-bound blocks from radixark/miles PR #2786 run_glm5_3_flash.py
              +-- gym side: nats.yaml + gym-worker.yaml = the smoke-3node assets renamed (rl-glm53f*-)
```

## Examples

| directory | model / gym | shape | status (2026-09-05) | records |
|---|---|---|---|---|
| `harbor-rl-27b/` | Qwen3.5-27B / financeagent | 3x p6-b200 (workers 0-1 actor TP4/PP2/CP2, worker 2 = 2 engines x 4 GPUs) | CPU-validated (argv parity, strict parse `PARSE_OK`, e2e harness 49/50); no GPU launch on miles | [RUNLOG](harbor-rl-27b/RUNLOG.md), [ARGV_PARITY](harbor-rl-27b/ARGV_PARITY.md) |
| `harbor-rl-27b-snorkel/` | Qwen3.5-27B / snorkel-general-bash-harbor (2,922 lakeFS tasks) | 6x p6-b200 (2 actor TP4/PP2/CP2 + 4 engine nodes, 8 engines x 4 GPUs) | e2e-snorkel 145/145; the 3-node GPU smoke derived from it completed (4 rollouts, 4 steps) | [RUNLOG](harbor-rl-27b-snorkel/RUNLOG.md), [ARGV_PARITY](harbor-rl-27b-snorkel/ARGV_PARITY.md), [smoke-3node](harbor-rl-27b-snorkel/smoke-3node/) |
| `harbor-rl-glm53-flash/` | GLM-5.3-Flash-BF16 (rev 61f77a1e) / snorkel-general-bash-harbor | r1-r5: 12x p6-b200 (8 actor TP8+SP/PP4/EP16, DP2 + 4 engines TP8/EP8); r6-r7: 24x (8 actor + 16 engines) | r7 in progress; r1..r7 summary table and open questions at the end of its RUNLOG | [RUNLOG](../../training-runs/harbor-rl-glm53-flash/RUNLOG.md), [README](harbor-rl-glm53-flash/README.md), [memprobe](harbor-rl-glm53-flash/memprobe/README.md) |

## Trainer image lineage (`arena-slime-dev:*`, built with `examples/arena/Dockerfile`)

Pushed to us-east-1 (the only region `arena-ecr-dev` can push to); the
repository replicates to ap-south-1, where prod-bom pulls. Each GLM tag is a
superset of the previous one. miles argparse is strict: a config key whose flag
the image does not know kills the trainer at start, so bump the tag with the key.

| tag | base | adds | used by |
|---|---|---|---|
| `miles-arena-20260901b` | `radixark/miles:latest` (cu13; the 2026-09-01 `latest-cu12` nightly had a broken TE<->torch pairing) | Dockerfile v1: arena deps (`nats-py`, `kubernetes==35.0.0`, `lakefs`, `boto3`, `omegaconf`, `typer`) + fork tree + import smoke; no EFA layer | rl-milesgb1-smoke1 (TCP fabric); the GLM fetch Job |
| a `miles-glm53-20260902a` | ECR mirror of `radixark/miles:glm53next` (SGLang `sglang-miles-glm53next@9a26e749` + Megatron-LM PR#89 `e8f57451`) | fork + PR #2786 merge + session-import guards + arena port; EFA layer (aws-efa-installer 1.47.0 + aws-ofi-nccl, build gate on `libnccl-net.so`) | GLM run 1; convert job |
| b `miles-glm53-20260902b` | a | `patches/fla_kda_next_power_of_2.py` as an image layer | GLM run 2 |
| c `miles-glm53-20260902c` | b | `--arena-mask-clipped-final-turn`; `hf_export` ENOTSUPP fix | GLM r2, r3 |
| d `miles-glm53-20260903d` | c | `--arena-keep-timeout-trajectories`; per-rollout `Removal reasons:` summary | GLM r4, r5, r6, r7 |
| `miles-glm53-20260908a` | d + r8 `--arena-keep-context-error-trajectories` (`miles-glm53-20260905a`, r8-r10) + r11 `--arena-inflight-multiplier` (`miles-glm53-20260907a`, `arpit-glm-53` b209d9b4) | `--arena-train-segments {final,all}` (plugin ADR-0011) and the all-mode DP alignment pad (zero-loss sibling rows, so `build_dp_schedule` can align singleton micro-batches); drops `rollout/compaction_segments_mean` for the upstream `rollout/num_training_samples` and `rollout/episode_raw_reward`. Built 2026-09-08 from `arpit-glm-53` d28503cf | GLM r12 (`arena_train_segments: all`) |
| `miles-glm53-r14-20260927a` | `glm53next-upstream-20260902` (the ap-south-1 ECR mirror, the Dockerfile base of each GLM tag); superset of `miles-glm53-r13-20260925a` (image `HEAD` `82f4a288c`, code `dd5e67391`). The `training-runs/harbor-rl-glm53-flash/r<N>/BUILD.md` files and the GLM RUNLOG record the tags from r3 to r13 | Plugin ADR-0015: each Harbor task message carries `sampling_params` {`max_new_tokens`, `temperature`, `top_p`} and `max_seq_len`; a `ValueError` stops the trainer when `rollout_max_response_len` or both window values are not set, or when `rollout_top_k` is not -1; `gen-workflow.py` changes a base `compaction-max` N to `agent-kwargs` `{"max_compactions": N}`. Built 2026-09-27 from `arpit-glm-53` `e0987aed6` with `docker build -f examples/arena/Dockerfile --build-arg MILES_BASE_IMAGE=<glm53next-upstream-20260902>` at the repo root. Digest `sha256:48d52a6a431257f9f7b40ce00debe9b2a10b61f94cee85d83291b21a2439e60c` in us-east-1 and ap-south-1. Image checks: `git rev-parse HEAD` = `e0987aed6`, a clean tree, and `build_task_message` emits both keys. A gym image with AREnATasks ADR-0072 MUST use this trainer or a later one. An older trainer fails each group. NEVER pair the template without the gym env entries with an older gym image. That gym falls back to 8192 and 32768 without an error | none on 2026-09-27 |
| `miles-glm53-r15-20260927a` | `glm53next-upstream-20260902`; superset of `miles-glm53-r14-20260927a` | Plugin ADR-0006 amendment 2026-09-27: `scripts/run_arena_harbor.py` appends `--save-hf` only when the YAML sets `arena_save_hf: true` or the pod env sets `ARENA_EVAL_TASKS`. The default save is DCP only, so the sidecar and the data-source state come right after the DCP. `--load` and `--save` do not change. Built 2026-09-27 from `arpit-glm-53` `bc31f88ac` with `docker build -f examples/arena/Dockerfile --build-arg MILES_BASE_IMAGE=<glm53next-upstream-20260902>` at the repo root. Digest `sha256:67b57cdf92695b4ec1e3c065cb803349745bc83858b28ef046d59da4cf960247` in us-east-1 and ap-south-1. Image checks: `git rev-parse HEAD` = `bc31f88ac`, a clean tree, and `_build_train_args` gives no `--save-hf` by default and `<ckpt>/hf/rollout_{rollout_id}` with `arena_save_hf: true`. It keeps all r14 behavior, so it pairs with the same gym images | none on 2026-09-27 |
| `miles-glm53-recon-test-20260927a` | `recon-miles-base-20260927a` (upstream `docker/build.py --variant cu13-x86` at radixark/miles `23d41d711`: SGLang `sglang-miles` `571212b6`, Megatron-LM `f148a32b`, fla 0.5.2, TileLang 0.1.14; digest `sha256:f83c7828a756`) | TEST ONLY, NEVER for a live run. Branch `arpit-reconcile-upstream` `c5b138782` (the rebase onto upstream `main`, `examples/arena/RECONCILE.md`). Digest `sha256:8f5fe3dc5715` in us-east-1 and ap-south-1. A GLM-5.3 row longer than 87,381 tokens stops the train step (fla int32 offsets) | `recon-t1-*-20260927b` (forward parity and HF gather pass), `recon-warm-20260927a` (fails) |
| `miles-glm53-recon-test-20260927b` | `recon-miles-base-20260927a` | TEST ONLY. `4d6c39a7d`: `patches/fla_conv_int64_offsets.py` (fla PR #1082 lines for `causal_conv1d`). Digest `sha256:429d4ebffec4`. A launch with `pin_rollout_manager_to_head: true` can stop with `ServerUnavailable` | `recon-convchk-20260927b`, `recon-warm-20260927b` (pass), `recon-t2-20260927b` (fails at the launch) |
| `miles-glm53-recon-test-20260927c` | `recon-miles-base-20260927a` | TEST ONLY. `4316e4957`: the head node lookup reads the GCS node table. Digest `sha256:2cd2c030c567`. Image checks: `git rev-parse HEAD` = `4316e4957`, a clean tree, and the patched fla kernel file equal to fla `31d15f75` | `recon-t2-20260927c` (train-only r45 layout, 3 steps, pass) |
| `miles-glm53-r17-20260928a` | `glm53next-upstream-20260902`; superset of `miles-glm53-r15-20260927a` | The first port of KDA tensor parallelism: `--glm5-next-kda-tp` (YAML `glm5_next_kda_tp`, default off; `a9207dd7d`, the rebase of `3803da20d`). With the flag, each TP rank holds 8 of the 64 KDA heads, and the module requires sequence parallelism. The checkpoint keys and the global shapes do not change, so a save loads in each layout. The kdatp harness also ships: `kdatp/`, a peak-memory log in `actor.py` that runs only when `MILES_LOG_PEAK_MEMORY` is set, and a train-only launcher branch that runs only with `--load-debug-rollout-data`. The image has no kernel-cache seed, so the first train step compiles the KDA kernels (kdatp T2 cold log-prob pass: 2,135 s). Built 2026-09-28 from `arpit-glm-53` `4716a367a` with `docker build -f examples/arena/Dockerfile --build-arg MILES_BASE_IMAGE=<glm53next-upstream-20260902>` at the repo root. Digest `sha256:e6f04a9ca1abc9df17a7bbf9e3d2aaf6a643f40ed5a457c98a4114719972bcd0` in us-east-1 and ap-south-1. Image checks: `git rev-parse HEAD` = `4716a367a`, a clean tree, the flag in `miles/utils/arguments.py` and `glm5_next/kda.py`, and 15 passed for `test_glm5_next_kda_tp.py` and `test_run_arena_harbor.py`. The launcher gives the r47 argv with `--glm5-next-kda-tp` and no `--save-hf`, and the image parser reads it. It keeps all r15 behavior, so it pairs with the same gym images | r47 |

`miles-glm53-r16-20260928a` has no row. It is the shared-layer candidate,
not a release: miles `arpit-kda-shared-layer-rebased` `389d2630`, digest
`sha256:ca9505454ad7e2bcaac433e48f86443074534a9e936a336d5b0c8f6154af5e87`.
At step 1 its `grad_norm` is 1.2% higher than the old layer, and the
gradient check is open. NEVER use it for a live run before that check
passes.

## Gym image lineage (`arena-tasks-dev:*`, built from AREnATasks with `brazil-build docker-push-arena <tag>`)

The gym side is unchanged on the wire (ADR-0002: streams, subjects, durable and
envelopes bit-identical to `amzn_arena_contract`); only the worker image moved.
r11 moved the gym image to `arena-slime-dev`: the shared `arena-tasks-dev`
repository keeps only the 50 most recent images and pruned the r9/r10 gym tags
within hours. A row with a repository prefix lives in `arena-slime-dev`.

| tag | AREnATasks source | adds | used by |
|---|---|---|---|
| `rl-smoke-20260821b` | ADR-0039 era; Harbor 0.21.0 | -- | rl-milesgb1-smoke1; GLM run 1, run 2, r2, r3, r4 |
| `glm53-reasoning-20260903b` | HEAD 35f7ba7 + the (then uncommitted) reasoning-effort patch; Harbor 0.22.0, amzn-arena-harbor 1.0.1061.0 | `ARENA_REASONING_EFFORT` rendered on the first turn (GLM-5.x template honours `low`/`high` only); `<|user|><|user|>` role-token splice fix; CLI contract `--mode rollout --agent arena-terminus-2`; timed-out groups now reported `truncated` (0.21 said `success`; the trainer salvages both) | GLM r5 (effort `low`) |
| `glm53-reasoning-20260904a` | mainline c1a0439 + 0fb3e54 + bae6a6b | `ARENA_AGENT_TIMEOUT_MULTIPLIER` honoured on the rollout path (mainline wired eval only) | GLM r6 (multiplier 2, `ARENA_NATS_ACK_WAIT` 6000) |
| `glm53-sgltimeout-20260904b` | bae6a6b + `ARENA_SGLANG_REQUEST_TIMEOUT_SEC` | configurable `/generate` httpx read timeout (default 600 s, previously hardcoded) | GLM r7 (1800 s, effort `high`) |
| `arena-slime-dev:gym-glm53-vulcan-20260908a` | `arpit-glm-53` a337c2d; `brazil-build docker-arena`, then `docker tag amzn-arena-tasks:local` + `docker push` to us-east-1 `arena-slime-dev` (replicates to ap-south-1) | Vulcan agent with token-aware context compaction (AREnATasks ADR-0048): the policy writes a handoff note (pi prompts) before each compaction; the compacted history is text only (`[system, instruction, bridge]`, `compaction_tail_fraction` 0); each compaction starts a new rollout segment and the gym ships one step per segment; `ARENA_COMPACTION_*` knobs; CLI `--mode rollout --agent vulcan`. Vulcan does not read `ARENA_CONTEXT_NUDGE_TOKENS` | GLM r12 (`ARENA_COMPACTION_MAX` 2, effort `high`) |
| `arena-slime-dev:gym-glm53-r5-20260914a` | `arpit-glm-53` 1b07eda (on r4 `dd0df46`, `ARENA_PARTIAL_REWARD=ctrf`; AREnATasks ADR-0069 removes that knob and the `partial-reward` template parameter, so the gym trains on the Harbor trial reward); `brazil-build docker-arena`, `docker tag` + `docker push` to us-east-1 `arena-slime-dev`, `sha256:97bc7a13...` | Vulcan continues after a per-turn `length` cut-off with a Terminus-2 nudge, bounded by `ARENA_TRUNCATED_TURN_MAX` (default 5 in this image; see the ADR-0072 note below the table); each segment ships `truncated_spans` (`[[start, end), ...]` in the cumulative `token_ids` index space) with the loss mask left at 1, so the trainer decides the credit (`--arena-truncated-turn-rule`); `truncated_turns` in the trajectory | GLM r25 |
| `arena-slime-dev:gym-glm53-adr72-20260927a` | AREnATasks mainline `538bc63` (CR-A `0b12642` + CR-B `538bc63`, tree `ab266688`, package version 1.0.1536.0); `brazil-build release` and `brazil-build docker-arena` at the detached commit, then `docker tag amzn-arena-tasks:local` + `docker push` to us-east-1 `arena-slime-dev`. Built 2026-09-27. Digest `sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d` in us-east-1 and ap-south-1; label `org.opencontainers.image.revision` = `538bc63` | ADR-0072: the training gym reads the output cap, the window, and the sampling values from each task message, and stops at startup when a name in `REMOVED_ROLLOUT_ENV` is set; `sglang_rollout.py` asks for `min(max_tokens, context_limit - len(input_ids))` and raises `ContextLengthExceededError` before the `/generate` POST when no room is left; `TRAINING_MAX_TRUNCATED_TURNS = 5` in `agents/training_vulcan.py`. It keeps the Harbor trial reward (ADR-0069, ADR-0070) and the token arrays by file reference with the 30 MiB result cap (ADR-0071). Image checks: `training_gym_worker.py`, `sglang_rollout.py`, `gym_worker.py`, and `agents/{training_vulcan,vulcan,constants}.py` match `538bc63` byte for byte. This image MUST use `miles-glm53-r14-20260927a` or a later trainer | none on 2026-09-27 |

Gym images with AREnATasks ADR-0072 (plugin ADR-0015) start at
`gym-glm53-adr72-20260927a`. The adr69 and adr71 images still read the env
names below.
A trainer with ADR-0015 sends the per-call output cap, the window, and the
sampling values in every task message. A training gym with ADR-0072 reads
them from the message. It stops at startup when one of these env names is
set: `ARENA_MAX_TOKENS`, `ARENA_ROLLOUT_CONTEXT_LIMIT`, `ARENA_TEMPERATURE`,
`ARENA_TOP_P`, `ARENA_COMPACTION_MAX`, `ARENA_TRUNCATED_TURN_MAX`, or
`ARENA_COMPACTION_FRACTION`. Its Vulcan defaults are `max_compactions` 4 and
`max_truncated_turns` 5 in the training subclass (0 in eval). The template
parameter `agent-kwargs` (the gym `--agent-kwargs` JSON) overrides them.

## Shared conventions

- Run identity is the pod env `EXPERIMENT_NAME` (checkpoint dir
  `<ARENA_CHECKPOINTS_DIR>/slime_experiments/<name>` = `--load` = `--save`, and
  the W&B group); the YAML `experiment_name` is a record only. Never reuse a
  finished run's name: an existing dir resumes silently (plugin ADR-0004 note
  in each README).
- Driver logs survive only on EFS,
  `/mnt/scratch-s3files-rw/guparpit/logs/<EXPERIMENT_NAME>/trainer-<idx>.log`
  (the kubeflow operator deletes a failed PyTorchJob with its pods). Metrics:
  W&B `mega.wandb.agi.amazon.dev/arena/rl-snorkel27`, one group per `EXPERIMENT_NAME`.
- Every job runs `scripts/run_arena_harbor.py` (`train` on replica 0 = ray head
  + job submit, `worker` elsewhere); the launcher pops the keys it consumes
  (`user`, `cluster`, `experiment_name`, `project_name`, `agislime_dir`,
  `replicas`, `num_trainers`, `model_arch`, `arena_save_hf`) and appends
  `--wandb-group`, `--load/--save` and the node math from the pod env.
- HF export (2026-09-27, plugin ADR-0006 amendment): each save is DCP only, and
  resume needs only the DCP. The launcher appends
  `--save-hf <ckpt>/hf/rollout_{rollout_id}` only when the YAML sets
  `arena_save_hf: true` or the pod env sets `ARENA_EVAL_TASKS`. The per-save
  export idles every GPU for approximately 25 min. The idle-GPU reaper deleted
  r44 and r46 during such an export.
- Clipped turns and stop reasons (2026-09-23): the gym masks a clipped or
  empty generate in `loss_mask`; the trainer applies no advantage transform
  and keeps every episode whose verifier ran, whatever `agent_stop_reason`
  says. Only a broken token stream is removed (`context_overflow`,
  `bad_logprobs`). The flags in the lineage tables above
  (`--arena-mask-clipped-final-turn`, `--arena-keep-*-trajectories`,
  `--arena-truncated-turn-*`) are history and no longer parse.
- Cluster: kubectl context `arena-prod-bom-v2`, namespace `arena-tasks`, kueue
  queue `gpu.p6-b200-48xlarge`. Check for kueue TAS mis-pins within ~2 min of
  every apply (GLM README step 4); one NATS broker per run (JetStream stream
  names are per-server); restart the gym Deployment together with any NATS
  restart (the durable consumers live in the broker's emptyDir).

## Where things are recorded

- `examples/arena/<job>/RUNLOG.md`: launches, numbers, RCAs and decisions per job, chronological, times in PT.
- `training-runs/<family>/r<N>/RECORD.md`: one record per GLM-5.3 run (dataset, images, template, checkpoints, outcome);
  `training-runs/studies/<slug>/STUDY.md`: one record per investigation. `training-runs/README.md` holds the index.
- `miles_plugins/arena/RUNLOG.md`: package milestones (port, verification, post-port flags, test state).
- `miles_plugins/arena/adr/`: 0001 port scope and module mapping; 0002 NATS wire contract bit-identical;
  0003 group identity via `group_index`; 0004 per-trajectory loss weighting and rollout logprobs;
  0005 asyncio driver and sidecar resume; 0006 launcher argv parity; 0007 plain-stack descoping for snorkel;
  0008 glm5_next via the PR #2786 merge; 0009 `--arena-mask-clipped-final-turn`; 0010 `--arena-keep-timeout-trajectories`.
- `miles_plugins/arena/README.md`: module mapping, wiring knobs, NATS defaults, install extra, known limitations.
- Evidence too large for the repo (analysis/verify reports, e2e harnesses, raw trainer logs, probe scripts):
  `/workplace/guparpit/miles/arena-port-artifacts/` (outside the repo).
- Gym-side records for the same runs live in AREnATasks (`adr/`, `eval-runs/`, `.claude/skills/run-eval/SKILL.md`).
