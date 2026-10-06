# ADR-0019: Consume-time weight-staleness cap on the NATS rollout path

**Status:** Accepted
**Date:** 2026-10-06

**Amends:** nothing. `--rollout-function-path ...nats_rollout.generate_rollout`
keeps its behavior; the cap is reachable only through the new class seam.
**Port of:** `arpit-ns-20261005` 9775f958 (the filter), c830cff7 (the version
that trains a batch), 19dee1af (the owner metrics), e343987e (the class seam).

## Context

With the DSA-QP trainer and the R3 data path (ADR-0018, final-v5) the loop is
rollout-bound. More engines only help with more groups in flight
(`arena_inflight_multiplier`), and more groups in flight make the batch older.
The upstream `--max-weight-staleness` flag (fully-async `DataBuffer.get`)
bounds that age, but the NATS path ignored it. The ns line ports it to the NATS
drain; this line has no such filter.

## Decision

1. `NatsRolloutFn` (`--rollout-function-path
   miles_plugins.arena.nats_arena.nats_rollout.NatsRolloutFn`) wraps
   `generate_rollout` behind the upstream class seam, which passes the engine
   weight version (`RolloutFnTrainInput.weight_version`; the function seam
   drops it).
2. `_train_weight_version` turns the engine version at drain start into the
   version that trains the batch: `train_async_arena` drains rollout `r`
   during the train step of `r - 1`, waits for the drain, and updates the
   weights before rollout `r` trains when `r` is a multiple of
   `--update-weights-interval` (+1). The first rollout of a process trains
   with no update after its drain (+0). The start rollout is the command-line
   `--start-rollout-id`, else the first call (the RolloutManager's copy of
   `args` keeps `start_rollout_id` None).
3. In the drain, after the dynamic-sampling filter, a group whose oldest
   weight version (upstream `group_oldest_weight_version`: the minimum over the
   per-call versions of every sample) is more than `--max-weight-staleness`
   versions behind that version is dropped, and its routing files are reaped
   under R3. The cap is inclusive (`staleness > max` drops), as upstream. A
   sample without a numeric version counts as unknown.
4. Metrics, every rollout: `rollout/fully_async/stale_groups_filtered` and
   `rollout/num_old_age_dropped` (count), `rollout/reward_old_age_dropped`
   (mean reward of the dropped groups, per episode, only when one was
   dropped), `rollout/fully_async/avg_staleness` and `max_staleness` (kept
   groups); one `Weight staleness:` log line.
5. `--max-weight-staleness` without the class seam stops the first rollout
   with an error that names `NatsRolloutFn`.

## Consequences

- The versions are the gym's per-call `weight_versions` list (the scalar
  `meta_info["weight_version"]` of each call). A call that continues across a
  weight update reports the newer version, so its oldest tokens read one
  version too new. A group whose oldest call straddled an update can pass the
  cap one version over it. The ns line closes this with per-call version
  spans (AREnATasks ADR-0074, gym image `gym-glm53-wvspans-20261001a` or
  later, and its `Sample.weight_versions` span types).
  ponytail: upgrade path is that gym image plus the span types.
- Under `--arena-train-segments all` every segment trains and every sample
  carries the whole trajectory's version list, so the oldest version of a
  group is the first call of its oldest episode. A long multi-step chain can
  outlive the cap by itself (final-v5: `rollout/weight_version/min` stayed 1
  through rollout 13 at multiplier 4), and its whole group is dropped. Watch
  `rollout/num_old_age_dropped` and `rollout/reward_old_age_dropped` for a
  length or reward bias.
- A dropped group is rollout compute spent for nothing, and its prompt is not
  retried (`--async-unused-samples-handler retry` has no effect here).

## Verification

`tests/fast/plugins/arena/test_weight_staleness.py`: the drop, the inclusive
cap, the oldest call of any sample, untagged samples, the default, the error
without the seam, the R3 reap, the W&B keys, the upstream rule, the
`_train_weight_version` table, the class seam, and the real
`train_async_arena` loop with fakes for start rollout 0 and 3 and interval 1
and 2 (each batch gets the version that it trains under; needs libcuda, so it
runs on a GPU node).
