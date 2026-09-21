# r31: agentic-debt with cut-at-first-failure (DRAFT — NOT launched)

Derived from the r29 (agentic-debt) recipe. r31 trains the same `agentic-debt`
le5 dataset as r28/r29, but the harbor chain cuts at the first failed step
(`ARENA_STEP_CUT_ON_FAIL=true`, AREnATasks 7ea168d). Reward =
passed-prefix / declared_K. Logs to the NEW W&B project `rl-glm53f-adebt-cut`.

## Deltas vs r29 (already applied to r31/miles-config.yaml)

- experiment_name / project_name: rl-glm53f-gbash-r26 -> rl-glm53f-adebt-cut-r31
- wandb_project: rl-glm53f-adebt -> rl-glm53f-adebt-cut  (wandb_team stays arena)
- arena_sample_summary_dir -> .../rl-glm53f-adebt-cut-r31/sample_summary
- Dataset unchanged: same agentic-debt le5 manifest as r28/r29
  (manifest-adebt-le5.jsonl). partial-reward stays off.

## Deltas vs r29 (workflow parameters)

- template guparpit-miles-deployer-v6 -> guparpit-miles-deployer-v7
- NEW gym image gym-glm53-adebt-cut-r31-20260921a (AREnATasks 7ea168d)
- NEW param step-cut-on-fail=true (string "true")
- Trainer image reused as-is: miles-glm53-r9-20260920a (routing-ref fix).

## Regenerate

```
.venv/bin/python gen-workflow.py 31 --base r29 --template guparpit-miles-deployer-v7 \
  --partial-reward off --experiment-name rl-glm53f-adebt-cut-r31 \
  --gym-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adebt-cut-r31-20260921a \
  --trainer-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r9-20260920a \
  --param gym=agentic-debt --param ack-wait=36000 --param agent-timeout-multiplier=1 \
  --param trainer-task-deadline-secs=39600 --param compaction-max=2 \
  --param step-cut-on-fail=true
```

## Launch checklist (do NOT execute until confirmed)

1. Gym image `gym-glm53-adebt-cut-r31-20260921a` and trainer image
   `miles-glm53-r9-20260920a` visible in ap-south-1 ECR.
2. WorkflowTemplate `guparpit-miles-deployer-v7` created (RBAC is create-only;
   a new template revision needs a new name).
3. Confirm the v7 template carries the `step-cut-on-fail` parameter.
4. Smoke-test agentic-debt at --limit 1 on the r31 gym image.
5. Refresh excluded-nodes for current prod-bom capacity.
6. `kubectl create -f r31/workflow.yaml`; start the r31 resume watcher after
   the PyTorchJob exists.
