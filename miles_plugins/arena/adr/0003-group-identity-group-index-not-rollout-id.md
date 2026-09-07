# ADR-0003: Group identity is a shared `group_index` plus a unique `index`, never `rollout_id`

**Status:** Accepted
**Date:** 2026-09-01

**Builds on:** ADR-0001 (port scope and module mapping), ADR-0002 (NATS wire contract stays bit-identical)
**Amended by:** ADR-0011 (`--arena-train-segments all` stamps the EPISODE id, never the group id, on `rollout_id`; the default `final` mode keeps this decision)

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

## Verification (2026-09-01 ~01:50-02:58 PT): executed semantics and the regression guard

Appended with the `verification-hardening` commit. The Decision above was
taken from source reading during the port; this section records how the
adversarial wave executed it and what now guards it.

### What verify-natsGrpo executed (CPU venv, real miles conversion path)

- Normalized diff of `nats_rollout.py` against the AGISlime original: 267
  diff lines; every hunk is doc-only, an import reorder, the sanctioned
  stamping block (`s.group_id = gid` -> `s.group_index = gid` +
  `s.index = gid * n_per_prompt + i`, `rollout_id` untouched) or the new
  `_add_arena_arguments` hook. No wire drift: task build ->
  `amzn_arena_contract.rl.parse_task_message` round-trips, the degenerate
  stop set (6 values), salvageable statuses and subjects matched the live
  contract package; the trainer-side streams, `slime-trainer` durable,
  ack_wait 3600 and max_deliver 3 are byte-identical to the original.
- Repro: 2 groups x 8 built through the real `NATSRolloutWorker._process_group`
  (fast path, one synthetic slot -> sibling-copy pad, one truncated), then
  `postprocess_rollout_data` + `convert_samples_to_train_data` with
  smoke-shaped args (grpo, GBS 16, n 8, use_rollout_logprobs,
  rewards_normalization + grpo_std_normalization). Output: `group_index`
  `[0]*8 / [1]*8`, `index` 0..15, `rollout_id` all None; rewards matched the
  per-group torch mean / unbiased-std computation exactly and differed from
  global normalization; `rollout_mask_sums` equalled per-sample mask sums
  (pads 0); at the smoke shape (8 groups x 8 = 64, GBS 64)
  `can_schedule_on_rollout_side` is True because the 64 unique indices act as
  64 rollout ids.
- Counterfactual (the alternative rejected above): stamping `rollout_id = gid`
  on a group with rewards [1, 0, 1, 0] raises
  `ValueError("all samples in rollout 99 must share one reward; rows [0, 1, 2, 3]
  have rewards [1.0, 0.0, 1.0, 0.0]")` from `_normalize_rewards_by_rollout`,
  and additionally sets `is_compact=True`, which silently disables batch
  trimming at group boundaries.
- Vendored-baseline check: `vendor/slime/0.3.0/slime/utils/dp_schedule.py:120-124`
  computes `num_steps = len(group_ids) // global_batch_size` and asserts
  `>= 1`; the overlay stamped one `group_id` per prompt group, so on the 27B
  smoke config 8 groups // GBS 64 = 0 fires the assert (vendored
  `rollout.py:769` calls `build_dp_schedule` unconditionally). The AGISlime
  tree could never complete a training step on that config, so neither its
  stamping nor its per-group loss weighting is demonstrated behaviour the
  port could regress against.
- Other counterfactuals: all-failed group is dropped (0 groups emitted);
  all-truncated group converts with mask sums 0 (train side clamps the
  denominator); 1 group vs GBS 16 raises `Not enough samples 8 for
  global_batch_size 16` in postprocess (the failure moved from the vendored
  dp-schedule assert to postprocess, same severity); 3 groups vs GBS 16
  trims to 16 at an exact group boundary.

verify-e2e (real NATS 2.14.6 JetStream + fake financeagent worker, 02:15 PT)
saw the same on the wire: shared `group_index`, unique `index`,
`rollout_id` None on every group; GRPO rows exact vs hand math
(raw [1.0, 0.5, 0.0, 0.25] -> [1.31746, 0.14638, -1.02469, -0.43915]); no
"must share one reward" error despite within-group variance.

### Finding: the decision had zero regression coverage

verify-natsGrpo [minor]: the 102 ported `test_nats_arena.py` tests pin wire
format, dedup and salvage but never assert `group_index` / `index` /
`rollout_id` on output samples (its only `group_index` asserts, lines
198-199, are task metadata on the publish side). A refactor of
`_process_group`, or of miles' `_reward_group_segments` fallback order, that
regressed to `rollout_id = gid` or dropped the `index` stamp would pass the
whole arena suite and crash the 27B job at step 1: the ValueError above, or
int64 packing on None indices. Both were reproduced in the venv.
verify-e2e [minor] added the mixed fast/slow-path None-row case that ADR-0004
fixes by zero-filling.

### Guard: `tests/fast/plugins/arena/test_group_identity.py` (written 2026-09-01 02:58 PT)

5 tests, `register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])`,
real torch + miles `postprocess_rollout_data` / `convert_samples_to_train_data`.
It is a separate file, not a class in `test_nats_arena.py`, because that suite
is deliberately import-light; this one needs the real conversion path. It is
the only arena test with no AGISlime counterpart.

| Class | Test | Pins |
| --- | --- | --- |
| TestGroupIdentityStamping | test_stamping_with_synthetic_pad | shared `group_index`, `index == gid*n + i`, `rollout_id` None, no `group_id` attribute; the pad is FAILED / remove_sample / reward 0.0 / zero loss mask / zero logprobs / `metadata["mode"] == "failed"` |
| TestGroupIdentityStamping | test_second_group_gets_disjoint_indices | per-worker group counter: `[0,1,2,3]` then `[4,5,6,7]`, `group_index` 0 then 1 |
| TestGrpoGroupNormalization | test_per_group_normalization_and_per_trajectory_mask_sums | 2 groups x 4 == GBS 8 through the real conversion: rewards equal per-group torch normalization and differ from global; `rollout_ids` fall back to `index`; `rollout_mask_sums` per trajectory with the pad at 0; logprob rows 1:1 with response lengths |
| TestSlowPathLogprobFill | test_mixed_group_yields_zero_filled_removed_sample | messages-only trajectory in a fast-path group under use_rollout_logprobs is ABORTED + removed, logprobs zero-filled to response_length, every row tensorizable |
| TestSlowPathLogprobFill | test_slow_path_without_rollout_logprobs_stays_none | with the flag off the slow path keeps its pre-existing behaviour (None logprobs, COMPLETED, trainable) |

Maintainer consequence: `rollout_id` stays None on arena samples on
purpose; a "fix" that stamps the group id there fails this file before it
fails the job. The per-trajectory loss-denominator consequence remains
tracked in ADR-0004 and still needs job-owner sign-off.
