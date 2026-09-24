# r38: auctioneer-caponly redo on the rebuilt stack

r30's recipe and config, retrained on the rebuilt gym image (AREnATasks
`2befd21`, ADR-0065..0068 subclasses, gym-owned loss mask) and a trainer
from miles `a639ea445` (train every verified episode, stop-reason and
clipped-turn telemetry). r30 ran the pre-rebuild stack and its forensics
lacked stop reasons, clipped-turn counts and transcripts.

## Deltas vs r30

| Item | r30 | r38 |
| --- | --- | --- |
| gym image | `gym-glm53-adebt-r29-20260920a` | `gym-glm53-adr68-20260924a` (AREnATasks `2befd21`) |
| trainer image | `miles-glm53-r9-20260920a` (miles `03791ce86`) | `miles-glm53-r11-20260924a` (miles `a639ea445`) |
| template | `guparpit-miles-deployer-v6` | `guparpit-miles-deployer-v7` (r30 values passed explicitly) |
| config keys | six clipped-turn keys | dropped: `arena_mask_clipped_final_turn`, `arena_keep_timeout_trajectories`, `arena_keep_context_error_trajectories`, `arena_truncated_turn_rule`, `arena_truncated_turn_lambda`, `arena_truncated_turn_min_adv` (removed by miles `b04f433e7`; strict argparse rejects them) |
| task memory cap | 6144 MiB code default | task's own `memory_mb = 4096` (no default override at `2befd21`; template sets no `ARENA_TASK_MEMORY_MB`) |
| transcripts | none | none (v7 sets no `ARENA_PUBLISH_*`) |

Every other knob identical: dataset `20260916-v1` auctioneer-caponly,
global_batch_size 256, n_samples_per_prompt 8, lr 1.5e-6, partial-reward
off, compaction-max 2, ack-wait 36000, agent-timeout-multiplier 1,
trainer-task-deadline-secs 39600, 40 replicas, replica-trainer 8, 288 gym
replicas, the r30 excluded-nodes list (83 nodes, identical to the
2026-09-24 f32 relaunch).

## Why the trainer changed too

The new gym masks clipped generates itself and stamps
`stop_reason="length"`. The r9 trainer's `--arena-mask-clipped-final-turn`
salvage then finds nothing to mask and removes the whole episode; on r30 a
third to a half of the episodes per rollout ended on a clipped turn. The
`a639ea445` trainer trains every verified episode and reports
`rollout/stop/<reason>`, `rollout/clipped_turns`,
`rollout/masked_output_tokens`. Confounds vs r30 to note in the report:
clipped non-final turns are now masked (r30 shifted their advantage), and
`ARENA_TRUNCATED_TURN_MAX` now counts only clipped turns without a tool
call.

## Launch

```
.venv/bin/python gen-workflow.py 38 --base r30 --template guparpit-miles-deployer-v7 \
  --partial-reward off --experiment-name rl-glm53f-auct-cap-r38 \
  --gym-image .../arena-slime-dev:gym-glm53-adr68-20260924a \
  --trainer-image .../arena-slime-dev:miles-glm53-r11-20260924a \
  --param gym=auctioneer-caponly --param ack-wait=36000 --param agent-timeout-multiplier=1 \
  --param trainer-task-deadline-secs=39600 --param compaction-max=2
kubectl create -f r38/workflow.yaml
```

Launched 2026-09-24T21:03Z as `rl-glm53f38-htf6t` (W&B `rl-glm53f-auct-cap-r38`).
Trainer image digest `sha256:3aefa1fe31deb3690aadbcbe2716dd6152002b63abe9372d0c1e49490f03f183`,
pushed to us-east-1 21:00:48Z, ap-south-1 replica 21:01:01Z. Fresh run, no
resume watcher, queued behind
the running jobs (no preemption).
