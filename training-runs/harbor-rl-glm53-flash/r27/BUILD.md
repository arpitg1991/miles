# r27: group-relative token-efficiency reward

r26 with one change: each episode reward gains a term based on how its token
count ranks inside its own group. Template `guparpit-miles-deployer-v4`, gym
image `gym-glm53-r5-20260914a`, r26 batch shape, 40 replicas (8 actor + 32
engine), 288 gym replicas.

## Why

The r26 sample-summary probe (rollouts 70-76, 288 groups) shows the reward
ignores tokens completely:

| Measure | Value |
| --- | --- |
| Groups with 2+ episodes tied at the group max reward | 229 of 288 (80%) |
| Longest/shortest tokens among those tied-best episodes | median 2.09x, max 14.9x |
| Groups with 2+ episodes at reward 1.0 | 205 |
| Groups holding both a <150k and a >300k full success | 23 |

GRPO gives every tied episode the identical advantage. Nothing favours the
cheap solution, so length drifts upward: mean response length grew 31k -> 77k
tokens per sample with `corr(length, rollout_id)` = +0.903, while reward at
fixed truncated share stayed flat over ~150 optimizer steps.

The term also recovers wasted compute. `check_reward_nonzero_std` deletes 7-11
groups per rollout (median 9) out of ~41 examined to fill 32, so ~22% of rollout
compute is discarded. Of 766 observed drops, 677 are `zero_std_1.0`: every
attempt in the group passed. Those groups carry zero reward variance only
because the reward ignores tokens. The term gives them variance whose only
content is "solve it again, shorter".

## The rule

`shape_group_length_reward` runs on each group BEFORE the dynamic-sampling
filter (`miles_plugins/arena/nats_arena/nats_rollout.py`):

```
lam   = 0.5 - (episode_tokens - group_min) / (group_max - group_min)
shift = coef * lam                       if reward == group best reward
shift = coef * min(0, lam)               otherwise
shift = max(shift, -0.49 * gap_below)    when shift < 0
reward = reward + shift
```

`gap_below` is the distance to the next lower distinct reward in the group.
Episode tokens are the sum over the ADR-0011 segments of one episode. A removed
episode never sets the length scale.

Two deviations from the Kimi k1.5 length reward, both deliberate:

- The best-reward branch keeps the full `+-coef/2` range. Kimi clips the long
  half to 0, which would leave the whole long tail undifferentiated, and the
  long tail is what needs separating here.
- Each downward shift is clamped to `0.49x` the gap to the next lower reward
  level. A fixed coefficient is not safe on its own: 13.5% of observed
  within-group reward gaps are under 0.05 and the minimum is 0.0099 (a task with
  ~101 tests). An unclamped 0.05 reorders rewards in 20 pairs over the same 320
  groups. With the clamp a reordering is impossible at any coefficient, because
  only the best level ever moves upward and nothing sits above it.

An all-failed group stays untouched. Paying its shortest attempt would reward
giving up early, and all-zero groups are only 42 of 766 drops.

## Deltas vs r26

| Knob | r26 | r27 | Reason |
| --- | --- | --- | --- |
| `arena_length_reward_coef` | absent (0.0, off) | 0.10 | User-selected. In a fully tied group the term is the only variance, so GRPO normalization makes the value irrelevant there; the value only sets how hard length competes with reward in a MIXED group. |
| `arena_sample_summary_dir` | `.../rl-glm53f-gbash-r26/sample_summary` | `.../rl-glm53f-gbash-r27/sample_summary` | Same probe, own path. It measures whether the term bends the length trend. |
| trainer image | `miles-glm53-r7-20260917a` | `miles-glm53-r8-20260917a` | miles `33697784f`: `--arena-length-reward-coef`, `shape_group_length_reward`, the `length-reward rescued=` counter. |

Everything else is r26: `arena_truncated_turn_rule` `shift` with lambda 0.5 and
`min_adv` -1.0, `arena_inflight_multiplier` 2, `global_batch_size` 256,
`save_interval` 10, `num_rollout` 300, lr 1.5e-6, eps_clip 0.2/0.28, TP8 PP4
EP16, `manifest-871.jsonl`, TIS, R3 routing replay, partial reward `ctrf`, gym
`ARENA_COMPACTION_MAX` 5 + `ARENA_TRUNCATED_TURN_MAX` 5, 64 MiB NATS
`max_payload`, the r25 `excluded-nodes` list.

## Experiment design

r27 starts fresh from `ref_load`, not from an r26 checkpoint. Two reasons:

1. The template derives `--load` and `--save` from `experiment-name`, so a fork
   from another run's checkpoint needs either a full DCP copy or an unproven
   `load` override. No run in this series does either.
2. The length drift appears early. r26 grew 31k -> 51k by rollout 5 and 64k by
   rollout 10. r27 therefore answers the question inside ~10-15 rollouts.

The control is r26 rollouts 0-20, same config, same base checkpoint, same batch
shape, one flag different. r26 keeps running as the long arm; r27 does NOT
replace it.

## Monitoring

| Signal | Expectation if the term works |
| --- | --- |
| `rollout/response_length/mean` vs rollout_id | Flat or slow, against r26's 31k -> 64k by rollout 10 |
| `length-reward rescued=N` in the trainer log | Non-zero. It counts all-passed groups the filter would delete |
| `Dynamic sampling: dropped=` | Falls from the r26 median of 9 |
| `rollout/raw_reward` | No fall against r26 at the same rollout_id |
| `rollout/truncated_ratio` | Falls, because fewer episodes reach the wall |

Watch `raw_reward` first. The term MUST NOT cost task success. Read the
first sample-summary rollouts with `/tmp/glm53/tiebreak.py` to confirm the
tied-best length spread narrows.

## Launch

1. FAILED 2026-09-18 00:02Z: `rl-glm53f27-tvgrz` admitted in 29 s, then evicted
   37 s later: `EvictedDueToNodeFailures ... unhealthy node(s):
   i-08faf7d33aedffdc4`. The PyTorchJob was garbage-collected and the workflow
   hung in `wait-trainer-nats` with zero GPU pods, because `deploy-trainer` is
   `Skipped` and nothing recreates a deleted PyTorchJob. Stopped at 00:24Z.
   `i-08faf7d33aedffdc4` added to `excluded-nodes`.
2. FAILED 2026-09-18 00:26Z: `rl-glm53f27-44wsc` evicted under 60 s on
   `i-05c9a8a01c250e0d7`. Node exclusion of one named node does not help. The
   cause is Karpenter drift: 203 of 337 p6 nodeclaims read `Drifted: True`, and
   Karpenter recycles them ~30 at a time. TAS gang placement is all-or-nothing,
   so one tainted node out of 40 kills the workload.
3. DONE 2026-09-18 13:21Z: `excluded-nodes` grown from 23 to 226 IDs (the prior
   23 plus all 203 drifted). `kubectl create -f r27/workflow.yaml` ->
   `rl-glm53f27-vv8g7`. **Zero evictions.** Admitted 15:56:51Z after 2 h 35 min
   queued. The unlock was kueue preemption inside our own ClusterQueue
   (`withinClusterQueue: LowerPriority`, preemptor priority 1000 over a
   priority-0 tenant), NOT an `arena-backfill` drain. Backfill never yields,
   because `gpu.p6-b200-48xlarge` sets `reclaimWithinCohort: Never` by design
   (ADR-0035 in AREnAInfraCDK; automatic reclaim caused the 2026-08-17 Kueue
   webhook outage).
4. DONE 2026-09-18 20:37Z: resume watcher `/tmp/r27-resume.sh` started, after
   the PyTorchJob existed. Kill it by PID before any planned take-down.

Trainer pods are `<wf>-trainer-worker-N`, not `<wf>-trainer-N`.
