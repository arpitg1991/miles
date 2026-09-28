# r30: first auctioneer-caponly run (DRAFT — NOT launched)

Derived from the r29 (agentic-debt) recipe. r30 trains the
`auctioneer-caponly` gym (1,047 single-step tasks; MCP daemon over a frozen
simulator, root-only world; reward = profit over the seed do-nothing
baseline) and logs to the NEW W&B project `rl-glm53f-auct-cap`.

NO prior auctioneer run exists in this workspace. The gym-description
"pricing-base-nats training run" is NOT committed here; there is no local
precedent to copy auctioneer agent/tool/timeout settings from. Resolve the
OPEN DECISIONS below before regenerating and launching.

## Deltas vs r29 (already applied to r30/miles-config.yaml)

- experiment_name / project_name: rl-glm53f-gbash-r26 -> rl-glm53f-auct-cap-r30
- wandb_project: rl-glm53f-adebt -> rl-glm53f-auct-cap  (wandb_team stays arena)
- arena_sample_summary_dir -> .../rl-glm53f-auct-cap-r30/sample_summary
- prompt-data-list -> auctioneer-caponly (path is a PLACEHOLDER, see OPEN 3)
- Single-step is unchanged config: arena_train_segments: all trains the
  length-1 chain as one Sample (one-code-path). NO multi_step knob exists to
  turn off. partial-reward stays off (correct for continuous reward).
  arena_length_reward_coef stays 0. GRPO group-normalizes the float reward;
  no reward-clip config to change.

## Regenerate (run only AFTER resolving OPEN DECISIONS)

```
.venv/bin/python gen-workflow.py 30 --base r29 --template guparpit-miles-deployer-v6 \
  --partial-reward off --experiment-name rl-glm53f-auct-cap-r30 \
  --gym-image <GYM_IMAGE — see OPEN 1> \
  --trainer-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r9-20260920a \
  --param gym=auctioneer-caponly --param ack-wait=36000 --param agent-timeout-multiplier=1 \
  --param trainer-task-deadline-secs=39600 --param compaction-max=2
```

Trainer image reused as-is (miles-glm53-r9-20260920a, routing-ref fix); no new
trainer code is needed for a gym swap.

## OPEN DECISIONS / RISKS (need user confirmation before a 40-node launch)

1. GYM IMAGE. The harbor gym worker is gym-agnostic (selects by --gym,
   materializes task assets per-task from lakeFS). No auctioneer code lives in
   AREnATasks; the simulator + MCP daemon come from the dataset. So the r29
   image gym-glm53-adebt-r29-20260920a is FUNCTIONALLY reusable. Its
   agentic-debt hardening (ARENA_TASK_MEMORY_MB, docker prune, Vulcan dead-env
   abort) is docker-mode specific — inert or helpful, not harmful. DECISION:
   reuse the r29 gym image, OR build a clean auctioneer-tagged image. Verify
   with a --limit 1 smoke before the fleet run.
2. WORLD MODE / TEMPLATE. auctioneer-caponly is described as an MCP daemon
   over a frozen simulator (like sample-ha/direct2home), NOT "docker-mode
   workers only". The per-task environment type is declared inside the
   materialized dataset task dir (unreadable without lakeFS creds). If it is
   NOT docker-mode, the v6 DinD-sized template (64Gi sidecar, 250Gi ephemeral)
   is overkill but works. If it IS docker-mode, v6 is correct. DECISION:
   confirm the environment type; v6 is the safe default either way. A simpler
   non-DinD template would need a NEW template name (RBAC is create-only).
3. DATASET / MANIFEST. prompt-data-list path is a PLACEHOLDER
   (lakefs://.../caponly/20260916-v1/manifest.jsonl). data_source accepts a
   lakefs:// file (auto-pulled) or a local jsonl; each row must carry
   {lakefs_uri, lakefs_commit_id}. CONFIRM which file to use — manifest.jsonl
   (ADR-0029 gate) vs tasks.jsonl (training order) — and that its row schema
   matches. If tasks.jsonl has a different schema, stage/convert a manifest to
   /mnt/scratch-s3files-rw/guparpit/data/auctioneer-caponly/20260916-v1/.
4. TRAINING ORDER. rollout_shuffle is still true. tasks.jsonl "carries the
   training order"; if order matters (curriculum), set rollout_shuffle: false.
5. AGENT + LIMITS. No local precedent for the auctioneer agent (Vulcan vs
   Terminus), tool set (MCP), termination, max tokens/turns, timeouts. r29
   inherits Vulcan + ARENA_MAX_TOKENS + compaction-max=2 + the v5 deadline
   knobs. CONFIRM these fit a single-step MCP auction task (likely far shorter
   than an adebt chain; deadlines may be over-provisioned).
6. CAPACITY / QUEUE. prod-bom inference queue has no preemption; a 40-node run
   waits for capacity. Refresh excluded-nodes; launch on separate queued nodes.
7. STALE-FILE HYGIENE. r29/gym-worker.yaml still names snorkel; r29/workflow.yaml
   embeds a stale config (wandb_project rl-snorkel27) vs r29/miles-config.yaml.
   The deployer template creates the pods; gen-workflow embeds
   r30/miles-config.yaml verbatim. Make the r30 identity fields consistent
   (done) and regenerate rather than hand-editing workflow.yaml.

## Launch checklist (do NOT execute until 1-6 resolved)

1. Resolve OPEN 1-5; smoke-test auctioneer-caponly at --limit 1 on the chosen
   gym image.
2. Confirm gym + trainer images visible in ap-south-1 ECR.
3. Confirm WorkflowTemplate guparpit-miles-deployer-v6 exists (or the chosen
   template).
4. Refresh excluded-nodes for current capacity.
5. Run the gen-workflow command above; verify r30/workflow.yaml params
   (gym=auctioneer-caponly, experiment-name, images, prompt-data-list).
6. `kubectl create -f r30/workflow.yaml`; start the r30 resume watcher after
   the PyTorchJob exists.
