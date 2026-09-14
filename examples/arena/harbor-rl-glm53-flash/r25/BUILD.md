# r25: fresher samples, live PPO ratio, longer episodes

r23 with a shorter publisher pipeline, two optimizer steps per rollout, and a
gym that survives a per-turn token cut-off. Template
`guparpit-miles-deployer-v4` (AREnATasksApps, 64 MiB NATS `max_payload`).

## Deltas vs r23

| Knob | r23 | r25 | Reason |
| --- | --- | --- | --- |
| `arena_inflight_multiplier` | 4 | 2 | At 4x a sample waited 6-7 optimizer steps between generation and training; 2x cuts the age to 2-3 steps. |
| `global_batch_size` | 512 | 256 | One rollout is 64 x 8 = 512 episodes, so 256 gives two optimizer steps per rollout and the PPO ratio and clip become live. |
| `save_interval` | 5 | 10 | A step at gbs 256 is half as long, so 10 halves the checkpoint idle at the same wall-clock exposure. |
| `num_rollout` | 130 | 300 | The run reaches 100+ optimizer steps before a verdict. |
| `arena_truncated_turn_rule` | absent | `mask` | New trainer arg; how the trainer treats a turn the gym continued after a per-turn cut-off. Decision pending, see below. |
| gym `ARENA_COMPACTION_MAX` | 2 | 5 | The 64 MiB `max_payload` holds one result at 5 compactions; 2 was the 8 MiB bound. |
| gym `ARENA_TRUNCATED_TURN_MAX` | absent | 5 | Vulcan continues after a per-turn token cut-off up to 5 times per episode, then finalizes as truncated. |
| NATS `max_payload` | 8 MiB | 64 MiB | r19 lost 76% of the gym fleet to the 8 MiB reject + redelivery loop; the results stream caps a message at 128 MiB, so 64 MiB fits. |

Unchanged: lr 1.5e-6, eps_clip 0.2/0.28, TP8 PP4 EP16, `manifest-871.jsonl`,
rollout_shuffle, TIS, R3 routing replay, partial reward `ctrf`, 40 replicas
(8 actor + 32 engine), 288 gym replicas, the r23 `excluded-nodes` list.

Token budget check: `rollout_max_context_len` and `sglang_context_length`
stay 131072 and bound one segment; `rollout_max_response_len` 32768 bounds
one turn; `max_tokens_per_gpu` 8192 is a micro-batch packing budget that a
131k sample already exceeds by design. A continued turn adds no tokens past
the 131072 episode window, so no trainer budget changes.

## Open decision: `arena_truncated_turn_rule`

The trainer arg lands in parallel with this directory; the key name is
fixed, the value is not.

- `mask`: zero the loss on the cut-off turn, train the continuation.
- `flip`: keep the turn and flip its advantage sign.
- `flip_positive`: flip only when the episode advantage is positive.

`mask` is the conservative default. Decide before you fill the image tags.

## Launch

Both images do not exist yet. Fill the tags before you submit.

1. Build the gym image from AREnATasks (`arpit-glm-53`, the commit that adds
   `ARENA_TRUNCATED_TURN_MAX`): `brazil-build docker-arena`, then push to
   `arena-slime-dev` in us-east-1 and wait for the ap-south-1 replica.
2. Build the trainer image from miles (`arpit-glm-53`, the commit that adds
   `arena_truncated_turn_rule`) and push it the same way.
3. Replace the two `TODO r25 image` tags: the gym tag in `r25/gym-worker.yaml`,
   then rerun `.venv/bin/python gen-workflow.py 25` (AREnATasks `.venv`; it
   reads the gym tag and the config), then set the trainer tag in
   `r25/workflow.yaml`. Keep the `# TODO r25 image` lines out of the final file.
4. `kubectl create -f /tmp/guparpit-miles-deployer-v4.yaml` (RBAC allows
   `create`, not `patch`, on workflowtemplates; the file is the committed
   AREnATasksApps `apps/arena-miles-deployer/generated/workflowtemplate.yaml`
   with `__AWS_REGION__` -> `ap-south-1` and the name -> `guparpit-miles-deployer-v4`).
5. `kubectl create -f r25/workflow.yaml`.
6. Start `/tmp/r25-resume.sh` only after the PyTorchJob exists.
