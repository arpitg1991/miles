# r19 (2026-09-11): skip1000 curriculum, reward-gated prompt skip, r16 node shape

Trainer image `arena-slime-dev:miles-glm53-r3-20260911b` (arpit-glm-53
09f477834, pushed 2026-09-11 01:33Z). It adds
`--arena-skip-prompt-above-reward`. Gym image unchanged:
`arena-slime-dev:gym-glm53-r3-20260909a` (see `r16/BUILD.md`).

## Deltas vs r18

1. Manifest: `ecr-20260823-curriculum-gpt56-skip1000/manifest.jsonl`, rows
   1001..2922 of the r17 curriculum manifest (1922 prompts). Built on EFS
   from `rl-glm53f18-trainer-worker-1` with `tail -n +1001`; row 1 of the new
   file equals row 1001 of the old file, and the last rows match.
   Source md5 `5899d99d7e4b7f396dca4a661bef101c` (2922 rows); skip1000 md5
   `36ce6d409a002b7efb4c25b6983ec08e` (1922 rows). r18 consumed 31 steps x
   64 = 1984 prompts of the easiest-first order; r19 skips the easiest 1000.
2. `arena_skip_prompt_above_reward: 0.5`. From epoch 1 on, the data source
   drops a prompt whose last recorded mean raw reward exceeds 0.5 and logs
   `Gym <name> epoch N: skipped K/1922 prompts with reward > 0.500`. If the
   filter would drop every prompt, it disables itself for that pass and
   warns. Epoch 0 (1922 / 64 = 30 steps) runs unfiltered.
3. Node split back to the r16 shape: `REPLICA_TRAINER` 8 (8 actor nodes,
   DP=2; 32 engine nodes). The Megatron parallelism block (TP8 x PP4 x CP1,
   EP16/ETP1, `max_tokens_per_gpu` 8192) is identical in r16, r18, and r19;
   only `REPLICA_TRAINER` differed between r16 and r18.
4. Trainer image `miles-glm53-r3-20260911b`.
5. Launch path: the user-owned WorkflowTemplate `guparpit-miles-deployer`
   (AREnATasksApps ADR-0008 with fix df26b80; region filled). `r19/workflow.yaml`
   is the submitted manifest.

Kept from r18: `use_tis: true`, `use_rollout_logprobs: false`,
`rollout_shuffle: false`, `lr` 1e-6, GBS 512, `rollout_batch_size` 64,
`n_samples_per_prompt` 8, `arena_inflight_multiplier` 4, `gym-replicas` 288,
the 22-node `NotIn` list (incl. `i-089c3bbbf596a26cb`). Fresh run: no
checkpoint dir and no warm-start symlink under
`slime_experiments/rl-glm53f-gbash-r19/`.

## Comparison

| | r16 | r18 | r19 |
| --- | --- | --- | --- |
| Trainer image | `miles-glm53-r3-20260909b` | `miles-glm53-r3-20260909b` | `miles-glm53-r3-20260911b` |
| Gym image | `gym-glm53-r3-20260909a` | same | same |
| REPLICA / REPLICA_TRAINER | 40 / 8 | 40 / 16 | 40 / 8 |
| Actor DP (TP8 x PP4) | 2 | 4 | 2 |
| SGLang engines (TP8) | 32 | 24 | 32 |
| Manifest | lakeFS `ecr-20260823` (2922, shuffled) | curriculum-gpt56 (2922) | curriculum-gpt56-skip1000 (1922) |
| `rollout_shuffle` | true | false | false |
| `use_tis` / `use_rollout_logprobs` | true / false | true / false | true / false |
| `arena_skip_prompt_above_reward` | - | - | 0.5 |
| `lr` | 1e-6 | 1e-6 | 1e-6 |
| Start | fresh | fresh | fresh |
| Launch | manual | manual (Argo failed x3) | Argo `guparpit-miles-deployer` |

## Launch (Argo)

Context `arena-prod-bom-v2`, namespace `arena-tasks`. No `argo` CLI; use
`kubectl` only.

1. Confirm the r19 checkpoint dir does not exist. From any live trainer pod:
   `kubectl exec <pod> -c pytorch -- ls /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-gbash-r19`
   must return `No such file or directory`. If it exists, stop.
2. Regenerate `r19/workflow.yaml` after any `miles-config.yaml` or `NotIn`
   change (the `miles-config` parameter is the file verbatim; `excluded-nodes`
   is the `NotIn` list as compact JSON). Use PyYAML: read both files, emit
   `miles-config` as a `|` block scalar, and confirm that a parse of the
   output returns the config byte-for-byte. Check with
   `kubectl create --dry-run=client -f r19/workflow.yaml`.
3. Submit: `kubectl create -f r19/workflow.yaml`. The workflow name is
   `rl-glm53f19-<suffix>`; the template creates `<name>-trainer`, `-nats`,
   `-sglang`, `-gym-sgb`, `-config`.
4. Watch the DAG: `render-config`, `deploy-nats`, `wait-nats-ready`,
   `deploy-trainer-pinned` (non-empty `excluded-nodes`), `deploy-sglang-svc`,
   `wait-trainer-nats` (polls worker-0 for `NATS connected (initial)`),
   `deploy-gym-workers`:
   `kubectl get wf <name> -o jsonpath='{range .status.nodes.*}{.displayName}{"\t"}{.phase}{"\t"}{.message}{"\n"}{end}'`
5. Kueue holds the PyTorchJob (Suspended) until 40 `p6-b200` nodes free up.
   Retire the previous run only after the r19 PyTorchJob exists, then wait
   for `kubectl get pods | grep rl-glm53f18` to return nothing before kueue
   admits r19 (see r18 step 7: a TAS mis-pin onto still-occupied nodes killed
   the first r18 attempt).
6. If an Argo node errors or the DAG misorders, stop the workflow with
   `kubectl patch wf <name> --type merge -p '{"spec":{"shutdown":"Terminate"}}'`,
   delete its `rl-glm53f19-*` leftovers, and use the manual fallback.

## Manual fallback

Apply in this order:

1. `kubectl create configmap rl-glm53f19-trainer-config --from-file=miles-config.yaml=r19/miles-config.yaml`
2. `kubectl apply -f r19/nats.yaml`
3. `kubectl apply -f r19/sglang-svc.yaml`
4. `kubectl apply -f r19/trainer-pytorchjob.yaml`
5. After worker-0 logs `NATS connected (initial)`: `kubectl apply -f r19/gym-worker.yaml`.
6. Start the auto-resume watcher: `sed 's/r18/r19/g' /tmp/r18-resume.sh > /tmp/r19-resume.sh && nohup bash /tmp/r19-resume.sh >> /tmp/r19-resume.log 2>&1 &`.

## Checks after admission

- Trainer log (worker-0) prints the parsed `arena_skip_prompt_above_reward`
  as `0.5` and `Loaded gym snorkel-general-bash-harbor: 1922 prompts`.
- 10 min of worker-0 log after rollout start: count `failed: None`,
  `No task directory`, `Traceback`, `RoutingReplay`.
