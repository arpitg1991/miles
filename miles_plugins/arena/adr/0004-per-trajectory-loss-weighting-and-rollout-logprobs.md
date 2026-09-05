# ADR-0004: Accept per-trajectory loss weighting; zero-fill slow-path rollout log-probs

**Status:** Accepted (loss weighting pending job-owner sign-off)
**Date:** 2026-09-01

**Builds on:** ADR-0003 (group identity)

## Context

Two training-math consequences of running the NATS rollout on miles' native
`convert_samples_to_train_data` instead of the vendored slime 0.3.0 one.

**Loss denominators.** The AGISlime overlay intended per-GROUP token-weighted
loss means: vendored `slime/ray/rollout.py:703-719` computed
`group_mask_sums[i]` = sum of loss-mask totals over every sample in sample
`i`'s group and broadcast it per sample, so the reducer used the whole-group
denominator even when first-fit packing split a group across micro-batches.
miles computes `rollout_mask_sums` the same way but keyed on `rollout_ids`
(`rollout_id`, falling back to `index`; `train_data_conversion.py:183-190`).
With ADR-0003 the key is the unique `index`, so every trajectory is its own
loss-aggregation unit. A plugin cannot restore group denominators: the only
stamp that groups siblings for `rollout_mask_sums` is `rollout_id`, which
also forces one shared reward per group (ADR-0003).

Mitigating fact, verified against the vendored source: the AGISlime baseline
could never have completed a training step on the harbor-rl-27b smoke
config with its own stamping. Vendored `rollout.py:769` calls
`build_dp_schedule` unconditionally with `group_ids`, and
`dp_schedule.py:120-124` asserts `num_groups >= global_batch_size`; with 8
stamped groups and GBS 64 that is `8 // 64 = 0` and the assert fires. The
per-group weighting was therefore never demonstrated behaviour.

**Rollout log-probs.** Every arena config (`financeagent-27b-smoke.yaml`,
`snorkel-27b.yaml`) sets `use_rollout_logprobs: true`, and miles gates the
whole batch on `samples[0].rollout_log_probs` (`train_data_conversion.py:114-115`).
The slow path in `_result_to_samples_full_trajectory` (messages-only
trajectories re-tokenised through `MultiTurnLossMaskGenerator`) never set
`rollout_log_probs`. The 2026-09-01 e2e verification executed the case: a
group `[fast, fast, fast, slow]` produced `rollout_log_probs` rows
`[20, 20, 20, None]` and `torch.tensor` failed with
`TypeError: must be real number, not NoneType`; with the slow sample first
the field was silently dropped for the entire batch. The region was
byte-identical to AGISlime, latent only because Harbor gym workers always
emit `has_generate_tokens` steps.

## Decision

1. **Per-trajectory loss means are accepted for the arena path** as miles'
   standard GRPO semantics. The divergence from the overlay's intent is
   recorded here and in the package README ("Known limitations") and needs
   an explicit sign-off from the job owner before the 27B job is treated as
   a like-for-like replacement of the AGISlime run.
2. **Slow path under `use_rollout_logprobs`:** zero-fill
   `s.rollout_log_probs = [0.0] * response_length`, set
   `status = ABORTED`, `remove_sample = True`, removal reason `no_logprobs`,
   and log a warning naming the task. This mirrors the fast path's existing
   bad-logprob handling (zero-fill + ABORTED + remove). Without
   `use_rollout_logprobs` the slow path is unchanged: `rollout_log_probs`
   stays `None` and the sample remains `COMPLETED` and trainable.

### Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Re-implement group weighting in the plugin through `--custom-convert-samples-to-train-data-path` | Duplicates the converter to restore a behaviour AGISlime never ran end to end; do it only if the owner asks for it. |
| Add `group_mask_sums` keyed on `group_index` to miles core | Core change for one plugin, same unproven-semantics objection. |
| Turn `use_rollout_logprobs` off | The configs rely on gym-side log-probs for the importance-ratio correction; and the crash only moves (a `None` row still breaks tensorisation for whichever sample is first). |
| Reject messages-only trajectories loudly when `use_rollout_logprobs` is set | One slow trajectory would kill its whole group and the step; zero-fill + remove keeps the siblings trainable and `removed_sample_frac` honest. |

## Consequences

- Within a group, a short trajectory's tokens carry proportionally more
  gradient weight than under group weighting. Removed and padded samples
  contribute zero tokens; miles' reducer clamps zero denominators to 1, so
  an all-removed group is safe.
- Mixed fast/slow groups no longer crash the step. Slow-path samples become
  zero-gradient rows counted in `rollout/removed_sample_frac` and visible as
  `no_logprobs` in the logs, so a gym that suddenly stops sending token-level
  data shows up as a removal spike instead of a tensorisation error.
- Any config that leans on `use_rollout_logprobs` must keep it on; the
  fallback now yields zero-gradient samples rather than failures, which is
  what makes it safe to run gyms whose trajectories may lack token data.
- Regression guard: `tests/fast/plugins/arena/test_group_identity.py`
  (`TestGrpoGroupNormalization` for the per-trajectory `rollout_mask_sums`,
  `TestSlowPathLogprobFill` for the zero-fill and the `None`-stays-`None`
  case without the flag), landing with the verification-hardening commit.
