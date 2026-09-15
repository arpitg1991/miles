# r26: shift credit on continued clipped turns

r25 with one change: the trainer applies the `shift` rule to a turn the gym
continued after a per-turn token cut-off. Template
`guparpit-miles-deployer-v4`, gym image `gym-glm53-r5-20260914a`, r25 batch
shape, 40 replicas (8 actor + 32 engine), 288 gym replicas.

## What r26 tests

r25 masks the loss on a continued clipped turn (`mask`). r26 keeps the turn
in the loss and lowers its credit when the episode advantage is positive:

```
A_turn = A_episode                          if not span token or A_episode <= 0
A_turn = max(A_episode - lambda, min_adv)   otherwise
```

`A_episode` is the group-normalized advantage before OPD and before
advantage whitening (`miles/backends/training_utils/loss.py`,
`apply_truncated_turn_shift`). A span token in a failed episode keeps its
negative advantage. A span token in a successful episode loses `lambda` of
credit and can go negative, down to `min_adv`.

## Deltas vs r25

| Knob | r25 | r26 | Reason |
| --- | --- | --- | --- |
| `arena_truncated_turn_rule` | `mask` | `shift` | Train on the continued turn with reduced credit instead of no gradient. |
| `arena_truncated_turn_lambda` | absent | 0.5 | The amount subtracted from a positive span advantage. |
| `arena_truncated_turn_min_adv` | absent | -1.0 | The floor of the shifted advantage. |
| trainer image | `miles-glm53-r5-20260914a` | `miles-glm53-r6-20260915a` | miles `aca2e9520`: the `shift` rule, the two knobs, and the advantage mass metrics. |

Everything else is r25: `arena_inflight_multiplier` 2, `global_batch_size`
256, `save_interval` 10, `num_rollout` 300, lr 1.5e-6, eps_clip 0.2/0.28,
TP8 PP4 EP16, `manifest-871.jsonl`, rollout_shuffle, TIS, R3 routing replay,
partial reward `ctrf`, gym `ARENA_COMPACTION_MAX` 5 +
`ARENA_TRUNCATED_TURN_MAX` 5, 64 MiB NATS `max_payload`, the r25
`excluded-nodes` list.

## Monitoring

The r6 image logs four new `train/*` gauges over the loss-masked tokens after
the rule (monitoring only, no rebalancing):

| Metric | Meaning |
| --- | --- |
| `train/adv_pos_mass` | Sum of positive advantages, per sample. |
| `train/adv_neg_mass` | Sum of the absolute negative advantages, per sample. |
| `train/adv_neg_pos_ratio` | `adv_neg_mass / (adv_pos_mass + 1e-6)`, a ratio of the reduced sums. |
| `train/truncated_turn_shifted_tokens` | Tokens the shift changed, per sample. |

Watch `train/adv_neg_pos_ratio` against the r25 baseline (the r5 image does
not log it; compare the r26 curve to a value near 1.0, which is what a
group-normalized advantage gives before the rule). The shift only moves
mass from positive to negative, so the ratio rises with
`truncated_turn_shifted_tokens`. Consider a CalibAdv-style rebalancing only
if the ratio drifts. Also watch `arena/truncated_turn_tokens`,
`rollout/episode_raw_reward`, `train/ppo_kl` and `train/pg_clipfrac` as in
r25.

## Launch

1. DONE 2026-09-15 01:13Z: trainer image `miles-glm53-r6-20260915a` (miles
   `aca2e9520`, `sha256:444d710f12dcf69b72a166b7b5eb0b422ceeb88394221d02d804ba43cfd518e3`),
   `docker build -f examples/arena/Dockerfile` on base
   `glm53next-upstream-20260902`, pushed to `arena-slime-dev` us-east-1;
   ap-south-1 replica in 16 s with the same digest. The image carries
   `shift` in `nats_rollout.py` and `apply_truncated_turn_shift` in
   `loss_hub/advantages.py` (checked with `inspect.getsource` inside the
   image).
2. DONE: tags set in `r26/trainer-pytorchjob.yaml` and `r26/workflow.yaml`
   (`gen-workflow.py 26` rerun, round-trip OK; the generator copies the r19
   trainer image, so `trainer-image` was set to the r6 tag by hand).
3. TODO: `kubectl create -f r26/workflow.yaml`. The workflow creates NATS,
   the SGLang service and the PyTorchJob; kueue holds the job while other
   runs hold the GPU nodes.
4. TODO: start `/tmp/r26-resume.sh` only after the PyTorchJob exists. The
   watcher needs two empty reads 120 s apart and a zero kubectl exit code
   before it resumes (r25 note: a single-read watcher restarted a healthy
   run mid-checkpoint).
5. TODO: after the first optimizer steps, confirm `train/adv_neg_pos_ratio`
   and `train/truncated_turn_shifted_tokens` appear in W&B.
