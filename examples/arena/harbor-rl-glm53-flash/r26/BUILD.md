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

1. DONE 2026-09-15 01:29Z: `kubectl create -f r26/workflow.yaml` ->
   `rl-glm53f26-kk68n` (template `guparpit-miles-deployer-v4`). NATS ready
   01:31Z; PyTorchJob created 01:32Z and admitted by kueue at once (41 pods
   Running by 01:36Z) next to r24 and r25, so three 40-node runs are live.
2. DONE 2026-09-15 01:36Z: `/tmp/r26-resume.sh` started after the PyTorchJob
   existed (two-read guard, same as r25).

## r7 probe resume (2026-09-17)

r26 resumed from iter 69 as `rl-glm53f26-dcskp` on trainer image
`miles-glm53-r7-20260917a` (miles `f60ead4cc`). The probe writes one
`rollout_{rollout_id}.jsonl` per training rollout with per-sample reward,
raw_reward, response_length, total_length, status, group_index, and
segment_k. No tensors. The goal is the within-group advantage-vs-episode-
length correlation behind the CTRF length drift (length grows on r21, r24,
r25, r26; flat on pre-CTRF r11).

| Item | r26 as launched | r7 probe resume | Reason |
| --- | --- | --- | --- |
| trainer image | `miles-glm53-r6-20260915a` | `miles-glm53-r7-20260917a` | miles `8989cfb49`: `--arena-sample-summary-dir`, `save_dashboard_columns` sample_index Int32 -> Int64, non-fatal wrapper. |
| `arena_sample_summary_dir` | absent | `/mnt/scratch-s3files-rw/guparpit/debug/rl-glm53f-gbash-r26/sample_summary` | Per-sample (reward, length) capture, ~100 KB per rollout, not on a tensor path. |
| everything else | r26 | unchanged | Same checkpoint line (`iter_*` 69), same gym image, batch shape, and rule knobs. |

Remove `arena_sample_summary_dir` after 2-3 rollouts only if it shows any
cost.

Crash lesson from `rl-glm53f26-6ddss` and `rl-glm53f26-7g5p7`:
`save_debug_rollout_data` also runs `save_dashboard_columns`. That writer
typed `sample_index` as Int32 and crashed on the ADR-0011 segment index
(`base + k * (1 << 40)`) before `torch.save`. Each cycle lost one ~2.5 h
rollout on 40 nodes with no optimizer step. Debug sidecars are on the hot
path. Trace every call to its guard before you enable one on a production
run.
