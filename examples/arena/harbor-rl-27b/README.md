# Harbor RL smoke — miles trainer side (Qwen3.5-27B + financeagent)

The AGISlime ADR-0039 27B smoke ported to miles: Qwen3.5-27B GRPO against the
`financeagent` Harbor gym over NATS, 3x p6-b200.48xlarge on prod-bom (kueue
queue `gpu.p6-b200-48xlarge`). Workers 0-1 run the Megatron actor
(TP4/PP2/CP2, 16 GPUs); worker 2 hosts the SGLang engines (8 GPUs).

This directory is the trainer side only. The gym-worker side (NATS service,
gym workers, dataset staging) is **unchanged** and still lives in
`AREnATasks/tmp_staging/harbor-rl-smoke/` — the NATS wire contract
(`arena.tasks.<gym>` / `arena.results`, streams `ARENA_TASKS`/`ARENA_RESULTS`,
durable `slime-trainer`) is bit-identical in the miles port.

## Build the trainer image

From the miles repo root (the whole fork ships in the image — the arena plugin
under `miles_plugins/arena/`, `scripts/run_arena_harbor.py`, and
`scripts/models/qwen3.5-27B.py`):

```bash
docker build -f examples/arena/Dockerfile \
    --build-arg MILES_BASE_IMAGE=<your miles GPU image> \
    -t <registry>/miles-arena-dev:rl-smoke-<date> .
docker push <registry>/miles-arena-dev:rl-smoke-<date>
```

`MILES_BASE_IMAGE` is a miles GPU runtime image from `docker/build.py`
(`radixark/miles:<tag>` or your ECR mirror). Then set the image in
`trainer-pytorchjob.yaml`. If you push to a replicated ECR registry, verify the
tag exists in the cluster's pull region AND that the digest changed since the
last push (`aws ecr describe-images --region <pull-region> --image-ids
imageTag=<tag>`) — a stale build context ships old code under a new tag with no
error.

## Deploy

Gym-worker side first (unchanged): `kubectl apply -f nats.yaml` and
`kubectl apply -f gym-worker.yaml` from
`AREnATasks/tmp_staging/harbor-rl-smoke/`, plus the sglang router Service in
this directory (`sglang-svc.yaml`, verbatim from
`AGISlime/tmp_staging/smoke/harbor-rl-27b/sglang-svc.yaml`): Service
`rl-smoke27-sglang` targets job-name `rl-smoke27-trainer` replica-index 0 on
port 30000. Do not substitute the non-27B `harbor-rl` variant — its Service
name and job-name selector do not match this job, leaving the router
unreachable with zero endpoints.

Trainer:

```bash
kubectl apply -f sglang-svc.yaml
kubectl create configmap rl-smoke27-trainer-config -n arena-tasks \
  --from-file=slime-config.yaml=financeagent-27b-smoke.yaml
kubectl apply -f trainer-pytorchjob.yaml
kubectl logs -n arena-tasks -f rl-smoke27-trainer-worker-0
```

**Before any launch that must train from step 0** (ADR-0004): the pod env
`EXPERIMENT_NAME` (not the config yaml) drives the checkpoint dir
`${ARENA_CHECKPOINTS_DIR}/slime_experiments/${EXPERIMENT_NAME}`, passed as both
`--load` and `--save`. An existing checkpoint dir resumes silently — a finished
run trains nothing and exits 0. Bump `EXPERIMENT_NAME`/`PROJECT_NAME` in the
PyTorchJob for a fresh run.

Every replica runs `scripts/run_arena_harbor.py`; the pod command picks the
role from the kubeflow replica index. Worker-0 (`train`) starts the Ray head,
waits for all `REPLICA` nodes, converts the ConfigMap YAML to a flat argv and
submits `miles_plugins/arena/train_async_arena.py`; the other replicas
(`worker`) join the Ray cluster and block. `--ref-load` points at a
pre-converted TP4 DCP checkpoint, so there is no HF->torch_dist conversion
step on start.

**Eval-enabled jobs** (any config with `eval_datasets` or
`hallmark_benchmarks`) MUST set the `MILES_ARENA_DIR` pod env to an external
tree carrying `experiments/k8s/templates/` — the miles repo does not ship
those K8s templates, so the baked `AGISLIME_DIR=/root/miles` satisfies
DCP->HF converter resolution only. Without it every eval trigger logs and
skips instead of submitting the coordinator Job. The smoke job configures no
eval and needs nothing.

## Verify

- Worker-0 log: the rendered `--load` path names the new experiment dir, and
  there is no `Loaded slime extra state ... rollout_id` /
  `Restored wandb_run_id` line (those mean you resumed an old run).
- Reward per step: `grep rollout/` in the worker-0 log — `raw_reward` /
  `group_metrics/group.reward.mean` trend up on a passing smoke
  (the AGISlime run 1 baseline: quarter means 0.173 / 0.242 / 0.320 / 0.331).
- Gym results: gym-worker logs show `Completed <task>: n/m real`; the workers
  wait up to 30 min for the trainer to create the JetStream streams.
- Argv parity against the AGISlime original: `ARGV_PARITY.md`.

## Teardown

```bash
kubectl delete pytorchjob rl-smoke27-trainer -n arena-tasks
kubectl delete configmap rl-smoke27-trainer-config -n arena-tasks
kubectl delete -f sglang-svc.yaml
# gym-worker side: see AREnATasks/tmp_staging/harbor-rl-smoke/README.md
```

Teardown includes the checkpoint tree: each 27B experiment leaves ~328 GB on
prod scratch (DCP iters + HF exports) under
`${ARENA_CHECKPOINTS_DIR}/slime_experiments/${EXPERIMENT_NAME}`.
