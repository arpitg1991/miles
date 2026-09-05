# Arena examples: Harbor RL over NATS on miles

Index of the arena Harbor RL jobs built on `miles_plugins/arena` (the AGISlime
-> miles port of the NATS-gym training path). Each subdirectory is the trainer
side of one job: the training YAML that `scripts/run_arena_harbor.py` converts
to the miles argv, the PyTorchJob and Service manifests, a README runbook, and
a `RUNLOG.md` with every launch (identity, image, shape, numbers, root causes,
decisions). Decisions with lasting effect are ADRs in `miles_plugins/arena/adr/`.
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
| `harbor-rl-glm53-flash/` | GLM-5.3-Flash-BF16 (rev 61f77a1e) / snorkel-general-bash-harbor | r1-r5: 12x p6-b200 (8 actor TP8+SP/PP4/EP16, DP2 + 4 engines TP8/EP8); r6-r7: 24x (8 actor + 16 engines) | r7 in progress; r1..r7 summary table and open questions at the end of its RUNLOG | [RUNLOG](harbor-rl-glm53-flash/RUNLOG.md), [README](harbor-rl-glm53-flash/README.md), [memprobe](harbor-rl-glm53-flash/memprobe/README.md) |

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

## Gym image lineage (`arena-tasks-dev:*`, built from AREnATasks with `brazil-build docker-push-arena <tag>`)

The gym side is unchanged on the wire (ADR-0002: streams, subjects, durable and
envelopes bit-identical to `amzn_arena_contract`); only the worker image moved.

| tag | AREnATasks source | adds | used by |
|---|---|---|---|
| `rl-smoke-20260821b` | ADR-0039 era; Harbor 0.21.0 | -- | rl-milesgb1-smoke1; GLM run 1, run 2, r2, r3, r4 |
| `glm53-reasoning-20260903b` | HEAD 35f7ba7 + the (then uncommitted) reasoning-effort patch; Harbor 0.22.0, amzn-arena-harbor 1.0.1061.0 | `ARENA_REASONING_EFFORT` rendered on the first turn (GLM-5.x template honours `low`/`high` only); `<|user|><|user|>` role-token splice fix; CLI contract `--mode rollout --agent arena-terminus-2`; timed-out groups now reported `truncated` (0.21 said `success`; the trainer salvages both) | GLM r5 (effort `low`) |
| `glm53-reasoning-20260904a` | mainline c1a0439 + 0fb3e54 + bae6a6b | `ARENA_AGENT_TIMEOUT_MULTIPLIER` honoured on the rollout path (mainline wired eval only) | GLM r6 (multiplier 2, `ARENA_NATS_ACK_WAIT` 6000) |
| `glm53-sgltimeout-20260904b` | bae6a6b + `ARENA_SGLANG_REQUEST_TIMEOUT_SEC` | configurable `/generate` httpx read timeout (default 600 s, previously hardcoded) | GLM r7 (1800 s, effort `high`) |

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
  `replicas`, `num_trainers`, `model_arch`) and appends `--wandb-group`,
  `--load/--save/--save-hf` and the node math from the pod env.
- Cluster: kubectl context `arena-prod-bom-v2`, namespace `arena-tasks`, kueue
  queue `gpu.p6-b200-48xlarge`. Check for kueue TAS mis-pins within ~2 min of
  every apply (GLM README step 4); one NATS broker per run (JetStream stream
  names are per-server); restart the gym Deployment together with any NATS
  restart (the durable consumers live in the broker's emptyDir).

## Where things are recorded

- `examples/arena/<job>/RUNLOG.md`: launches, numbers, RCAs and decisions per job, chronological, times in PT.
- `miles_plugins/arena/RUNLOG.md`: package milestones (port, verification, post-port flags, test state).
- `miles_plugins/arena/adr/`: 0001 port scope and module mapping; 0002 NATS wire contract bit-identical;
  0003 group identity via `group_index`; 0004 per-trajectory loss weighting and rollout logprobs;
  0005 asyncio driver and sidecar resume; 0006 launcher argv parity; 0007 plain-stack descoping for snorkel;
  0008 glm5_next via the PR #2786 merge; 0009 `--arena-mask-clipped-final-turn`; 0010 `--arena-keep-timeout-trajectories`.
- `miles_plugins/arena/README.md`: module mapping, wiring knobs, NATS defaults, install extra, known limitations.
- Evidence too large for the repo (analysis/verify reports, e2e harnesses, raw trainer logs, probe scripts):
  `/workplace/guparpit/miles/arena-port-artifacts/` (outside the repo).
- Gym-side records for the same runs live in AREnATasks (`adr/`, `eval-runs/`, `.claude/skills/run-eval/SKILL.md`).
