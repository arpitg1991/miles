# r28: multi-step agentic-debt chains

r27 batch shape and trainer image (`miles-glm53-r8-20260917a`) on the
`agentic-debt` gym. Template `guparpit-miles-deployer-v5`, gym image
`gym-glm53-adebt-r28-20260920a` (AREnATasks `01f7a13`, digest `sha256:fcc52063...`, ap-south-1 replica confirmed), 40 replicas
(8 actor + 32 engine), 288 gym replicas.

## Why

r1-r27 trained single-step snorkel tasks. The agentic-debt-r3 dataset holds
repository chains: one task is one repository, and its `[[steps]]` are
consecutive change requests on the same worktree. harbor runs the chain in one
container and grades each step with its own hidden tests. The gym reader
(AREnATasks ADR-0064) ships one segment per step with `segment_end="step"`
under one `rollout_id`. The trainer trains the chain on its existing
ADR-0011 segment path. No trainer code changes.

## Data

`manifest-adebt-le5.jsonl`: the 556 chains of the 834-chain manifest with at
most 5 segments, 1,816 segments in total. Order and bytes of the kept rows are
those of `manifest-adebt-834.jsonl`.

| Item | Value |
| --- | --- |
| Builder | `~/glm53-prep/mkman-le5.py` over `manifest-adebt-834.jsonl` + `/tmp/glm53/segcounts.json` |
| Object | `s3://arena-scratch-prod-bom-ap-south-1/guparpit/data/agentic-debt/20260916-v1/manifest-adebt-le5.jsonl` |
| Mount path | `/mnt/scratch-s3files-rw/guparpit/data/agentic-debt/20260916-v1/manifest-adebt-le5.jsonl` |
| Size, md5 | 151,890 bytes, `a1ac62c5ec56b280275ec62663b65967` |
| Row shape | `{"lakefs_uri": ".../agentic-debt-r3/20260916-v1/tasks/<task>/", "lakefs_commit_id": "<sha>"}` |

The gym name is not a manifest field. `prompt-data-list` carries
`gym_name: agentic-debt`, and the workflow `gym` parameter sets the same
name on the workers (`--gym`) and on the trainer (`ARENA_DEFAULT_GYM`). The
trainer publishes to `arena.tasks.<gym_name>`; the workers subscribe on
`arena.tasks.<--gym>`. The two names MUST match. Otherwise no worker consumes
a task.

## Batch shape

The trainer counts episodes, not rows. `global_batch_size` 256 = 32 groups x 8
attempts per optimizer step, and one optimizer step per rollout
(`num_steps_per_rollout` 1). Each episode keeps weight 1/256 and one
advantage regardless of its segment count. Rows per step grow with the
segments: about 837 mean for le5 against 310 measured in r27.
`arena_inflight_multiplier` 4 keeps 4 x 64 prompts in flight (r27: 2 x 64),
because a chain holds a worker for up to 5 segments.

## Deadlines

| Knob | r27 | r28 | Reason |
| --- | --- | --- | --- |
| per-segment agent cap | 1,800 s task timeout | 3,600 s task timeout | Dataset value (`task.toml`), not a knob. |
| `agent-timeout-multiplier` | 8 | 1 | One segment gets 3,600 s. At 8x one timed-out segment runs 28,800 s past the group deadline. |
| `ack-wait` | 18,000 | 36,000 | 5 segments x (3,600 s agent + 2,400 s verifier) = 30,000 s, plus env build and the 300 s deadline margin. |
| `trainer-task-deadline-secs` | 21,600 | 39,600 | `ack-wait` + 3,600 s slack. MUST exceed `ack-wait`; otherwise the trainer gives up on a group before the gym redelivers it. |
| `compaction-max` | 5 | 2 | Every step adds segments to one result message; the 60 MiB result guard (ADR-0064) reaps the largest trajectory when the envelope does not fit. |
| segments per attempt | 1 step, up to 6 segments | 5 steps max, up to 3 segments each | le5 manifest. |

## Reward rule

The chain reward is the sum of graded step rewards divided by the declared
step count K from `task.toml`. Unreached steps count 0, so a stalled chain
scores below a complete one. The gym broadcasts the value to every segment
of the attempt (ADR-0062 shared reward). `partial-reward` is `off`: each step
reward is the binary harbor verifier value. An aborted chain (a step with
`exception_info` and no `verifier_result`) is unscored, not 0.
`reward_harbor_mean` stays in metadata.

## Deltas vs r27

| Knob | r27 | r28 |
| --- | --- | --- |
| `workflowTemplateRef.name` | `guparpit-miles-deployer-v4` | `guparpit-miles-deployer-v5` |
| `experiment-name` | `rl-glm53f-gbash-r27` | `rl-glm53f-adebt-r28` |
| `gym-image` | `gym-glm53-r5-20260914a` | `gym-glm53-adebt-r28-20260920a` |
| `gym` | template default `snorkel-general-bash-harbor` | `agentic-debt` |
| `partial-reward` | `ctrf` | `off` |
| `ack-wait`, `agent-timeout-multiplier`, `trainer-task-deadline-secs`, `compaction-max` | template defaults 18000, 8, 21600, 5 | 36000, 1, 39600, 2 |
| `prompt-data-list` | `manifest-849.jsonl`, `snorkel-general-bash-harbor` | `manifest-adebt-le5.jsonl`, `agentic-debt` |
| `arena_inflight_multiplier` | 2 | 4 |
| `arena_length_reward_coef` | 0.10 | 0 |
| `arena_sample_summary_dir` | `.../rl-glm53f-gbash-r27/sample_summary` | `.../rl-glm53f-adebt-r28/sample_summary` |

Everything else is r27: trainer image `miles-glm53-r8-20260917a`, 40
replicas, `replica-trainer` 8, `gym-replicas` 288, the r27 `excluded-nodes`
list (226 IDs), `global_batch_size` 256, `save_interval` 10, `num_rollout`
300, lr 1.5e-6, eps_clip 0.2/0.28, TP8 PP4 EP16, TIS, R3 routing replay,
`arena_train_segments` `all`, `arena_mask_clipped_final_turn` true,
`arena_keep_timeout_trajectories` true, `arena_truncated_turn_rule` `shift`
with lambda 0.5 and `min_adv` -1.0, 64 MiB NATS `max_payload`.

`gym-worker.yaml`, `trainer-pytorchjob.yaml`, `nats.yaml` and
`sglang-svc.yaml` are the pre-Argo manual-launch manifests (r19/BUILD.md,
"Launch (manual)"). r25-r27 carried them unedited; r28 keeps them identical
to r27. The Argo template renders the live objects from the workflow
parameters, so these files do not reflect the r28 deadlines.

## Regenerate

```
.venv/bin/python gen-workflow.py 28 --base r27 --template guparpit-miles-deployer-v5 \
  --partial-reward off --experiment-name rl-glm53f-adebt-r28 \
  --gym-image <ECR image> --param gym=agentic-debt --param ack-wait=36000 \
  --param agent-timeout-multiplier=1 --param trainer-task-deadline-secs=39600 \
  --param compaction-max=2
```

The script re-parses the output and exits non-zero when the embedded
`miles-config` differs from `r28/miles-config.yaml`.

## Launch checklist

1. DONE 2026-09-20 02:33Z: AREnATasksApps `2c9d0f0` committed;
   `guparpit-miles-deployer-v5` created from
   `~/glm53-prep/guparpit-miles-deployer-v5.yaml`. The live v4 and v5 differ
   only in the four deadline parameters, their env wiring, and the DinD
   sidecar (`arena-dind-container:latest`, Apps `03030e3`, not yet run).
2. DONE 2026-09-20 02:09Z: gym image `gym-glm53-adebt-r28-20260920a`
   (AREnATasks `01f7a13`, digest `sha256:fcc52063...`) pushed to us-east-1,
   replica visible in ap-south-1; reference set in `r28/workflow.yaml`.
3. Manifest present: `aws s3 ls --profile arena-prod-bom-user s3://arena-scratch-prod-bom-ap-south-1/guparpit/data/agentic-debt/20260916-v1/` shows 151890 bytes.
4. No silent resume: no directory under
   `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-adebt-r28`.
5. `kubectl create --dry-run=client -f r28/workflow.yaml` passes.
6. Kill any live resume watcher for another run before a take-down; start
   the r28 watcher only after the PyTorchJob exists (r27/BUILD.md step 4).
7. Queued run via `kubectl create -f r28/workflow.yaml` after user go.

Trainer pods are `<wf>-trainer-worker-N`, not `<wf>-trainer-N`.
