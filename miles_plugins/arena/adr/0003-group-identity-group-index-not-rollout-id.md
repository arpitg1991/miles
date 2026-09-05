# ADR-0003: Group identity is a shared `group_index` plus a unique `index`, never `rollout_id`

**Status:** Accepted
**Date:** 2026-09-01

**Builds on:** ADR-0001 (port scope and module mapping), ADR-0002 (NATS wire contract stays bit-identical)

## Context

The AGISlime overlay ran on a vendored slime 0.3.0 whose `Sample` had a
`group_id` field. `NATSRolloutWorker._process_group` stamped one shared
`group_id` per GRPO prompt group (`s.group_id = gid`, AGISlime
`nats_rollout.py:1252`) so that the vendored
`_convert_samples_to_train_data` could key per-group loss denominators
(`group_mask_sums`, vendored `slime/ray/rollout.py:703-719`). The vendored
GRPO reward normalisation never read `group_id`: it reshaped the batch
contiguously to `(-1, n_samples_per_prompt)` (`rollout.py:634-652`).

miles has no `group_id`, and the analysis report `milesApi.md` flagged this
as the most load-bearing `Sample` drift of the whole port:

- `Sample.rollout_id` is a *different* concept in miles: the compact
  siblings of ONE rollout execution, which must share a single reward.
  `_normalize_rewards_by_rollout` (`miles/ray/rollout/train_data_conversion.py`)
  raises `ValueError("all samples in rollout N must share one reward")`
  otherwise. `rollout_id` also keys `rollout_mask_sums` and, when set on any
  sample, marks the batch compact and disables trimming to
  `global_batch_size` (`rollout_data_conversion.py`).
- `Sample.group_index` is what `_reward_group_segments` uses to segment GRPO
  reward normalisation.
- `Sample.index` is packed into int64 `sample_indices` / `rollout_ids`
  arrays; a `None` there crashes the packer (the AGISlime samples carried
  `None`).

The obvious mechanical mapping `group_id -> rollout_id` was tried first and
fails empirically: stamping `rollout_id = gid` on a group with rewards
`[1, 0, 1, 0]` raises
`ValueError: all samples in rollout 99 must share one reward; rows [0, 1, 2, 3] have rewards [1.0, 0.0, 1.0, 0.0]`
from `_normalize_rewards_by_rollout`, and the same stamp silently disables
batch trimming.

## Decision

`_process_group` stamps, per emitted group (`gid` = the worker's
`_output_group_counter`, incremented once per group):

```python
for i, s in enumerate(group_samples):
    s.group_index = gid                       # shared: keys GRPO reward segments
    s.index = gid * self.n_per_prompt + i     # unique: int64-packable, per-trajectory unit
# s.rollout_id deliberately stays None
```

A `PORT NOTE` comment at the stamping site records the concept mismatch so a
future "fix" back to `rollout_id` does not look like a cleanup. This is the
port's one deliberate semantic adaptation; every other non-comment change
in `nats_rollout.py` is an import swap or the new argument hook.

### Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| `rollout_id = gid` (mechanical rename) | Raises the shared-reward `ValueError` on any within-group reward variance (reproduced); also flips the batch to compact and disables trimming. |
| Leave `group_index` unset and rely on the contiguous `n_samples_per_prompt` layout, as vendored slime did | miles segments explicitly; on partial or filtered batches the vendored fallback collapsed everything into one group, the explicit key stays correct. |
| Re-introduce `group_id` on miles' `Sample` and port `group_mask_sums` into the converter | Core edit to `train_data_conversion.py` for one plugin; the per-group weighting it would restore is unproven (see ADR-0004). |
| Keep a dead `group_id` instance attribute on the miles `Sample` for "compatibility" | Silently loses the aggregation it pretends to carry; nothing in miles reads it. |

## Consequences

- GRPO mean-centering / std-normalisation runs per prompt group. On a full
  batch the math is identical to the vendored contiguous reshape (verified
  numerically at port time: two ported groups through
  `postprocess_rollout_data` + `convert_samples_to_train_data` gave
  `[0.866, -0.866, ...]` per group); on a partial batch it is correct where
  the vendored path was not.
- Loss denominators become per-trajectory (`rollout_mask_sums` keyed on the
  unique `index`) instead of per-group. That divergence is recorded and
  accepted separately in ADR-0004.
- Batch trimming to `global_batch_size` stays active, matching vendored
  behaviour; a short batch (worker thread death) now fails in
  `postprocess_rollout_data` (`Not enough samples`) rather than in the
  vendored dp-schedule assert.
- `sample_indices` are real unique ints, so miles' rollout-side DP
  scheduling (`can_schedule_on_rollout_side`, needs `len(set(rollout_ids)) >= GBS`)
  activates for this path: 64 unique ids >= GBS 64 on the 27B smoke shape,
  a code path the AGISlime deployment never took.
- The wire format is untouched: `group_index` still rides task metadata as
  before (ADR-0002); only trainer-side `Sample` fields changed.
- Regression coverage at the time of this decision: none. The ported
  wire-format suite pins publish-side metadata only and never drives
  `_process_group`; a regression to `rollout_id = gid` would pass the suite
  and crash the first training step.
