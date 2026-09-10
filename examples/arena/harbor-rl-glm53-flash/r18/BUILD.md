# r18 (2026-09-10): r17 images, TIS back on, first Argo launch

No image change. Trainer `arena-slime-dev:miles-glm53-r3-20260909b`, gym
`arena-slime-dev:gym-glm53-r3-20260909a` (see `r16/BUILD.md`).

Deltas vs r17:

1. `use_tis: true`, `use_rollout_logprobs: false` (the r16 values). r17 step
   15 logged `train/ppo_kl` 0.0062 and `pg_clipfrac` 2% with the SGLang
   rollout logprobs as `old_log_probs`. r16 under R3 with the trainer
   `log_probs` pass logged `ppo_kl` 1.8e-5. The trainer pass is load-bearing;
   the 5 min per step it costs buys an on-policy update.
2. Fresh run. No checkpoint dir and no warm-start symlink exist under
   `slime_experiments/rl-glm53f-gbash-r18/`. Weights load from the converted
   base model; rollout ids start at 0; the data source starts at row 0.
3. Launch path: the Argo `arena-miles-deployer` WorkflowTemplate
   (AREnATasksApps ADR-0008) instead of five ordered `kubectl apply` calls.
   `r18/workflow.yaml` is the submitted manifest.

Kept from r17: curriculum manifest (`r17/curriculum/`), `rollout_shuffle:
false`, `lr` 1e-6, node split 16 actor + 24 engine (`REPLICA_TRAINER` 16),
and the 21-node `NotIn` exclusion list.

## Launch (Argo)

Context `arena-prod-bom-v2`, namespace `arena-tasks`. No `argo` CLI; use
`kubectl` only.

1. Install or refresh the template from AREnATasksApps (commit 3141706 or
   later):
   `kubectl apply --server-side --force-conflicts -f apps/arena-miles-deployer/generated/workflowtemplate.yaml`
2. Confirm the r18 checkpoint dir does not exist. From any live trainer pod:
   `kubectl exec <pod> -c pytorch -- ls /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-gbash-r18`
   must return `No such file or directory`. If it exists, stop.
3. Regenerate `r18/workflow.yaml` after any `miles-config.yaml` change. The
   `miles-config` parameter is the file verbatim as a YAML block scalar;
   `excluded-nodes` is the `NotIn` list of `trainer-pytorchjob.yaml` as
   compact JSON. Check with
   `kubectl create --dry-run=client -o json -f r18/workflow.yaml`.
4. Submit: `kubectl create -f r18/workflow.yaml`. The workflow name is
   `rl-glm53f18-<suffix>`; every resource the template creates carries that
   prefix (`-trainer`, `-nats`, `-sglang`, `-gym-sgb`, `-config`).
5. The DAG runs `render-config`, `deploy-nats`, `wait-nats-ready`, then one
   of `deploy-trainer` (empty list) or `deploy-trainer-pinned` (non-empty
   list; r18 uses this one), then `deploy-sglang-svc`, `wait-trainer-nats`
   (polls worker-0 for `NATS connected (initial)`), then
   `deploy-gym-workers`. Watch with
   `kubectl get wf <name> -o jsonpath='{range .status.nodes.*}{.displayName}{"\t"}{.phase}{"\t"}{.message}{"\n"}{end}'`.
6. Kueue admits the PyTorchJob when 40 `p6-b200` nodes free up. Pods Pending
   more than 6 min after admission mean a TAS mis-pin; add the node to the
   `NotIn` list, delete the workflow, and resubmit.

## Manual fallback

If the template cannot be created or the DAG misfires, delete the workflow
and its `rl-glm53f18-*` leftovers, then apply in this order:

1. `kubectl create configmap rl-glm53f18-trainer-config --from-file=miles-config.yaml=r18/miles-config.yaml`
2. `kubectl apply -f r18/nats.yaml`
3. `kubectl apply -f r18/sglang-svc.yaml`
4. `kubectl apply -f r18/trainer-pytorchjob.yaml`
5. After worker-0 logs `NATS connected (initial)`: `kubectl apply -f r18/gym-worker.yaml`.
