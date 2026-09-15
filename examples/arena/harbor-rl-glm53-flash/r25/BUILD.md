# r25: fresher samples, live PPO ratio, longer episodes

r24 with a shorter publisher pipeline, two optimizer steps per rollout, and a
gym that survives a per-turn token cut-off. Template
`guparpit-miles-deployer-v4` (AREnATasksApps, 64 MiB NATS `max_payload`).

## Deltas vs r24

| Knob | r24 | r25 | Reason |
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
(8 actor + 32 engine), 288 gym replicas, the r23/r24 `excluded-nodes` list.

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

1. DONE 2026-09-14 22:40Z: gym image `gym-glm53-r5-20260914a` (AREnATasks
   `1b07eda`, `sha256:97bc7a13...`), built with `brazil-build docker-arena`
   and pushed to `arena-slime-dev` us-east-1; ap-south-1 replica in 15 s.
2. DONE 2026-09-14 22:39Z: trainer image `miles-glm53-r5-20260914a` (miles
   `e247e5e80`, `sha256:c258e933...`), `docker build -f
   examples/arena/Dockerfile` on base `glm53next-upstream-20260902`.
3. DONE: both tags set in `r25/gym-worker.yaml`, `r25/trainer-pytorchjob.yaml`
   and `r25/workflow.yaml` (`gen-workflow.py 25` rerun, round-trip OK).
4. DONE 2026-09-15 00:45Z: `kubectl create -f /tmp/guparpit-miles-deployer-v4.yaml`
   -> WorkflowTemplate `guparpit-miles-deployer-v4`.
5. DONE 2026-09-15 00:46Z: `kubectl create -f r25/workflow.yaml` ->
   `rl-glm53f25-6ggcw`. NATS ready 00:48Z; PyTorchJob created 00:49Z and held
   by kueue (`Created,Suspended`) while r23 (resumed as `rl-glm53f23-8skq6`)
   and r24 hold 80 GPU nodes.
6. DONE 2026-09-15 00:52Z: `/tmp/r25-resume.sh` started after the PyTorchJob
   existed. This watcher needs two empty reads 120 s apart and a zero kubectl
   exit code before it resumes; the r23 watcher fired on a single empty read
   at 00:26Z and restarted a healthy run mid-checkpoint.
