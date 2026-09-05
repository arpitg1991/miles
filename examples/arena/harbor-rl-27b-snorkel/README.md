# Harbor RL: snorkel-general-bash-harbor — miles trainer side (Qwen3.5-27B, plain stack)

Qwen3.5-27B GRPO against the `snorkel-general-bash-harbor` Harbor gym over
NATS, 6x p6-b200.48xlarge on prod-bom (kueue queue `gpu.p6-b200-48xlarge`).
Derived from the AGISlime snorkel run 5 deployment (`rl-snork27b-gbash-r5`)
but deliberately descoped to the plain miles stack — miles-native reward
normalization, r1 batch shape, no custom post-processing hooks (see "Lineage
and descoping"). Workers 0-1 run the Megatron actor (TP4/PP2/CP2, 16 GPUs,
DP=1); workers 2-5 host the SGLang engines (32 GPUs, 8 engines at 4 GPUs
each) — ample headroom at rollout_batch_size 32.

**Reward: binary in-container pytest verifier only.** Each task's `tests/`
writes 0/1 to `/logs/verifier/reward.txt` — the harbor default reward shape;
no judge model, no `GYM_REWARD_KEYS`, no rubric leg is wired anywhere in this
loop.

This directory is the trainer side only. The gym-worker side (NATS service,
gym workers) is **unchanged** and still lives in
`AREnATasks/tmp_staging/harbor-rl-snorkel/run5/` — the NATS wire contract
(`arena.tasks.<gym>` / `arena.results`, streams `ARENA_TASKS`/`ARENA_RESULTS`,
durable `slime-trainer`) is bit-identical in the miles port.

## Lineage and descoping

The AGISlime snorkel lineage layered custom training-path machinery on top of
its r1 baseline (r2: zero-variance group removal; r3b: an "8x pushed to NATS"
batch budget requiring a group-keyed survivor reward normalization; r4:
removed-sample replacement; r5: r3b's algorithm on a lean topology). This
port **keeps only the r2-era piece and reverts the rest** (a deliberate user
decision — port the plain stack first):

- **Kept — zero-variance dynamic sampling** (`dynamic_sampling_filter_path` =
  `miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`,
  a stock miles function): all-pass / all-fail groups carry zero GRPO
  advantage and are dropped at collection time, fetching extra groups until
  GBS/n survivors, bounded at 4x examined (the
  `--dynamic-sampling-max-examine-mult` default; the flag stays unset).
- **Descoped — group-keyed survivor normalization**
  (`custom_reward_post_process_path`): not needed on the plain shape. miles'
  native `_normalize_rewards_by_rollout` already keys reward groups on
  `Sample.group_index`, so filter survivors normalize per-group natively —
  the r3b failure mode (a whole-batch degenerate reshape when rbs*n != GBS)
  does not exist on miles.
- **Descoped — removed-sample replacement** (r4's
  `removed_sample_replacement` module and its
  `ARENA_NATS_REPLACE_REMOVED_SAMPLES` toggle): not ported.
- **Descoped — the r3b batch budget**: `rollout_batch_size` reverted 128 -> 32
  (the r1 baseline, rbs*n == GBS) and the
  `ARENA_DYN_SAMPLING_MAX_EXAMINE_MULT` pod-env override dropped with it.
  `num_rollout` is 90 (~1 epoch: 2922 tasks / rbs 32) instead of run5's
  100000 run-until-stopped sentinel.
- `log_passrate` stays `false` to match the run5 record, but its original
  rationale (ragged replacement batches breaking pass@k's fixed-size group
  reshape) no longer applies at rbs*n == GBS — it could be re-enabled.

The survivor-norm / replacement lineage can be revisited later; the reference
implementation is the harbor AGISlime commits `637d714` (survivor norm +
replacement) and `b8aae83` (examine-mult env override).

## Dataset (provenance — a NAMED difference from run5)

`prompt-data-list` points at the `KNOWN_GYMS` registry pin for this gym
(AREnATasks `amzn_arena_harbor/gyms.py`):

```
lakefs://arena-inspect/dev/internal/snorkel-general-bash-harbor/ecr-20260823/
```

with `manifest.jsonl` appended, because `ArenaDataSourceWithBuffer` pulls
exactly the lakeFS object the URI names — a bare variant-dir URI (with or
without the trailing slash) 404s at trainer startup; only *local* directory
paths get `manifest.jsonl` appended by the resolver.

- `manifest.jsonl` (manifest commit `b4aeedec…`) pins the **2922 validated
  tasks** at payload commit `ca22976f…`; rows are per-task
  `{lakefs_uri, lakefs_commit_id}` pointers.
- The AGISlime run5 instead trained on a scratch-staged `train.jsonl` —
  a **2600-task training subset**, with the remaining 322 tasks held out as
  `dev.jsonl` for offline per-checkpoint eval. This port trains on all 2922,
  **including those 322 dev tasks**: an offline eval against run5's
  `dev.jsonl` would be contaminated. If you need the run5 train/dev split,
  point `prompt-data-list` back at the staged `train.jsonl`
  (`AREnATasks/tmp_staging/harbor-rl-snorkel/stage_dataset.sh`).
- No dataset staging step: the trainer pulls the one manifest object itself
  (LAKECTL_* pod env, IAM via the `arena-tasks-runner` SA), and the gym
  workers materialize each task from lakeFS at the pinned commit
  (`resolve_task_dir` fallback, cache on the `/trials` volume) — the 212k
  payload objects never land on scratch.

## Build the trainer image

From the miles repo root (the whole fork ships in the image — the arena plugin
under `miles_plugins/arena/`, `scripts/run_arena_harbor.py`, and
`scripts/models/qwen3.5-27B.py`):

```bash
docker build -f examples/arena/Dockerfile \
    --build-arg MILES_BASE_IMAGE=<your miles GPU image> \
    -t <registry>/miles-arena-dev:rl-snorkel-<date> .
docker push <registry>/miles-arena-dev:rl-snorkel-<date>
```

Then set the image in `trainer-pytorchjob.yaml`. If you push to a replicated
ECR registry, verify the tag exists in the cluster's pull region AND that the
digest changed since the last push (`aws ecr describe-images --region
<pull-region> --image-ids imageTag=<tag>`) — a stale build context ships old
code under a new tag with no error.

## Deploy

Gym-worker side first, **unchanged from the AGISlime run**. The run5 gym
assets lived in `AREnATasks/tmp_staging/harbor-rl-snorkel/run5/`, which is
not versioned; the copies versioned here are the 8-replica smoke variants of
those exact files (renamed `rl-milesgb1-*`, as applied for `rl-milesgb1-smoke1`,
see `RUNLOG.md`):

```bash
kubectl apply -f smoke-3node/nats.yaml        # NATS JetStream (rl-milesgb1-nats)
kubectl apply -f smoke-3node/gym-worker.yaml  # 8 workers + dind, per-task lakeFS pulls
```

For the 6-node job rename the objects back to `rl-snork27b5-*` (the trainer
manifest's `NATS_URL` is `nats://rl-snork27b5-nats:4222` and `sglang-svc.yaml`
publishes `rl-snork27b5-sglang`), set `replicas` back to 128, and keep the
worker's `--nats-url` / `--model-base-url` consistent with those names.

- Worker sampling env (`ARENA_MAX_TOKENS=2048`, `ARENA_TEMPERATURE=1.0`) is
  already in parity with this trainer config.
- **ADR-0046 caveat for worker redeploys:** the run5 `gym-worker.yaml` exec
  line and its pinned image (`rl-smoke-20260821b`) predate ADR-0046. If you
  rebuild the worker image from current AREnATasks head, add `--mode rollout`
  to the `python3.12 -m amzn_arena_harbor.gym_worker` invocation — the flag is
  REQUIRED post-ADR-0046 and the worker never infers the mode (an eval-mode
  worker would silently produce a run with no training data).

Trainer (this directory):

```bash
kubectl apply -f sglang-svc.yaml    # verbatim from the run5 assets: Service
                                    # rl-snork27b5-sglang -> job rl-snork27b5-trainer
                                    # replica-index 0, port 30000
kubectl create configmap rl-snork27b5-trainer-config -n arena-tasks \
  --from-file=slime-config.yaml=snorkel-27b.yaml
kubectl apply -f trainer-pytorchjob.yaml
kubectl logs -n arena-tasks -f rl-snork27b5-trainer-worker-0
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
DCP->HF converter resolution only. This job configures no eval and needs
nothing.

## Verify

- Worker-0 log: the rendered `--load` path names the new experiment dir, and
  there is no `Loaded slime extra state ... rollout_id` /
  `Restored wandb_run_id` line (those mean you resumed an old run).
- Data source startup: `Pulled lakefs://arena-inspect/dev/internal/…/
  manifest.jsonl` and a 2922-row dataset for gym
  `snorkel-general-bash-harbor`.
- Gym results: gym-worker logs show `Completed <task>: n/m real` with real > 0
  and successful cross-region ECR pulls (task images are digest-pinned
  us-west-2 refs); workers wait up to 30 min for the trainer to create the
  JetStream streams.
- **Binary-reward signal check** (from the run-1 launch checklist): watch
  `group_metrics/group.reward.mean` and the fraction of nonzero-variance
  groups on W&B (`arena/rl-snorkel27`). With binary rewards, GRPO only gets
  gradient where pass@8 is strictly between 0 and 1; if >~80% of groups are
  zero-variance in the first steps, stop and revisit.
- Dynamic sampling: worker-0 log shows `NATS dynamic sampling enabled:
  filter=miles.rollout.filter_hub…check_reward_nonzero_std` at startup, and
  each collection logs `examined` against the cap — `max_examined` is 4x the
  group target (the `--dynamic-sampling-max-examine-mult` default; no
  pod-env override on this stack).
- Argv parity against the AGISlime run5 original (minus the descoped
  customs): `ARGV_PARITY.md`.

## Teardown

```bash
kubectl delete pytorchjob rl-snork27b5-trainer -n arena-tasks
kubectl delete configmap rl-snork27b5-trainer-config -n arena-tasks
kubectl delete -f sglang-svc.yaml
# gym-worker side (the versioned smoke variants, or the renamed copies you
#   applied): kubectl delete -f smoke-3node/gym-worker.yaml -f smoke-3node/nats.yaml
```

Teardown includes the checkpoint tree under
`${ARENA_CHECKPOINTS_DIR}/slime_experiments/${EXPERIMENT_NAME}`: with
`num_rollout: 90` and `save_interval: 20` the run leaves 4 saves plus the
final one — DCP iters + HF exports at ~80 GB+ per save point for 27B; prune
when the run is done.
