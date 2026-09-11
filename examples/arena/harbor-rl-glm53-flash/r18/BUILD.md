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
3. Launch path attempted: the Argo `arena-miles-deployer` WorkflowTemplate
   (AREnATasksApps ADR-0008). `r18/workflow.yaml` is the submitted manifest.
   Three template defects stopped it (see "Argo attempt"), so the run that
   is live came from the manual fallback below.

Kept from r17: curriculum manifest (`r17/curriculum/`), `rollout_shuffle:
false`, `lr` 1e-6, node split 16 actor + 24 engine (`REPLICA_TRAINER` 16),
and the 21-node `NotIn` exclusion list.

## Launch (Argo)

Context `arena-prod-bom-v2`, namespace `arena-tasks`. No `argo` CLI; use
`kubectl` only.

1. Install the template from AREnATasksApps (commit 3141706 or later). The
   `arena-validate-workflowtemplate-name` admission policy denies user
   templates named `arena*`, and the user role has no `patch` on
   WorkflowTemplates, so rename and `create`:
   `sed 's/^  name: arena-miles-deployer$/  name: rl-glm53f18-miles-deployer/' apps/arena-miles-deployer/generated/workflowtemplate.yaml | kubectl create -f -`
   `r18/workflow.yaml` references `rl-glm53f18-miles-deployer`. The
   platform-owned `arena-miles-deployer` name arrives through AREnAInfraCDK.
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
7. Do not submit while the previous run is still Terminating. On
   2026-09-10 05:43Z kueue admitted r18 the second r17 was deleted, TAS
   pinned pods onto nodes r17 still occupied, node `i-089c3bbbf596a26cb`
   went unreachable, kueue terminated worker-2, and the job failed and was
   removed within 60 s. Wait for `kubectl get pods | grep <old prefix>` to
   return nothing, then apply.

## Argo attempt (2026-09-10 05:30Z to 05:40Z)

Three workflows failed against AREnATasksApps 3141706. Fix these before the
next Argo launch:

1. `rl-glm53f18-zr6lk`: `invalid spec: templates.deploy-trainer: failed to
   resolve {{workflow.*}}`. The vendored manifests carry the literal
   `{{workflow.*}}` inside a YAML comment (`miles/manifests/pytorch_job.yaml`
   line 7, `gym_worker_deployment.yaml` line 5). Argo resolves comments too.
2. `rl-glm53f18-hpdwc`: `render-config` and `annotate` pods
   `InvalidImageName`: the `os-images` reference keeps the `__AWS_REGION__`
   placeholder. A `kubectl create` of the generated file does not fill it;
   only the InfraCDK chart install does. Set it to `ap-south-1` by hand.
3. `rl-glm53f18-cdlqp`: both `deploy-trainer` and `deploy-trainer-pinned`
   ended in `Error`: `Invalid 'when' expression ''["i-0161...","i-0e64..."]'
   != '[]'': Cannot transition token types from STRING [[] to VARIABLE [i]`.
   The JSON list holds double quotes inside a single-quoted govaluate string.
   Neither trainer task ran. The DAG then omitted the Service, the gate,
   and the gym.

The user role can `create` and `patch` Workflows but cannot `delete` or
`update` Workflows or WorkflowTemplates. Stop a bad workflow with
`kubectl patch wf <name> --type merge -p '{"spec":{"shutdown":"Terminate"}}'`
and delete its `-nats` Deployment and Service by hand. Its `-config`
ConfigMap belongs to the Argo service account and stays.

## Manual fallback (the live r18 path)

If the template cannot be created or the DAG misfires, delete the workflow
and its `rl-glm53f18-*` leftovers, then apply in this order:

1. `kubectl create configmap rl-glm53f18-trainer-config --from-file=miles-config.yaml=r18/miles-config.yaml`
2. `kubectl apply -f r18/nats.yaml`
3. `kubectl apply -f r18/sglang-svc.yaml`
4. `kubectl apply -f r18/trainer-pytorchjob.yaml`
5. After worker-0 logs `NATS connected (initial)`: `kubectl apply -f r18/gym-worker.yaml`.

## Resume (Argo, 2026-09-11)

The manual r18 run trained through step 32 and stopped. `save_interval: 5`
left `iter_0000029` as the newest checkpoint
(`latest_checkpointed_iteration.txt` = 29, `rollout/arena_data_source_state_29.pt`),
so steps 30-32 are lost. The resume runs as a second 40-node job next to r19.

1. Verify the checkpoint from any live trainer pod:
   `kubectl exec <pod> -c pytorch -- ls /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-gbash-r18 .../rollout`
   and `cat .../latest_checkpointed_iteration.txt`. Expect `iter_0000029` and
   `arena_data_source_state_29.pt`.
2. `r18/workflow-resume.yaml` targets `guparpit-miles-deployer` (the r19
   path). It keeps `experiment-name: rl-glm53f-gbash-r18`, so the trainer
   resumes the weights and the data source state from the same directory. It
   keeps the original r18 image `miles-glm53-r3-20260909b`, `replica-trainer`
   16, the r18 `NotIn` list as `excluded-nodes`, and `r18/miles-config.yaml`
   verbatim. `generateName: rl-glm53f18r-` keeps the resource names apart from
   the retired manual `rl-glm53f18-*` objects. Regenerate with PyYAML as in
   r19/BUILD.md step 2 (emit `miles-config` as a `|` block scalar, round-trip
   check, then `kubectl create --dry-run=client -f r18/workflow-resume.yaml`).
3. Check capacity first: `kubectl get nodes -l node.kubernetes.io/instance-type=p6-b200.48xlarge`
   and the ClusterQueue `gpu.p6-b200-48xlarge` usage. At submit time (12:00Z)
   306 nodes were Ready, 293 held GPU pods (13 of them backfill), and the
   queue used 2304 of 2440 GPUs: 17 free nodes, so the job queued behind no
   other pending workload (BestEffortFIFO, no preemption).
4. `kubectl create -f r18/workflow-resume.yaml` -> `rl-glm53f18r-gzc97`.
   Watch the DAG with the r19/BUILD.md step 4 command. The PyTorchJob stays
   Suspended in kueue until 40 nodes free.
