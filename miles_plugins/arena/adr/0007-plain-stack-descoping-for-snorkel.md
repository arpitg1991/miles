# ADR-0007: Plain-stack descoping for the snorkel job

**Status:** Accepted
**Date:** 2026-09-01

**Builds on:** ADR-0001 (port scope), ADR-0003 (group identity =
`Sample.group_index`), ADR-0006 (launcher argv parity)

## Summary

Stage `snorkel-general-bash-harbor` (Qwen3.5-27B) on miles as the PLAIN
stack: keep the AGISlime r2 zero-variance filter through miles-native
`check_reward_nonzero_std`, the r1 batch shape and the run5 lean topology;
descope the r3b/r4/r5 customs (group-keyed survivor normalization,
removed-sample replacement, `ARENA_DYN_SAMPLING_MAX_EXAMINE_MULT`, the 8x
batch budget). User decision, 2026-09-01: port the plain stack first.

## Context

The AGISlime snorkel lineage (`AGISlime/tmp_staging/smoke/harbor-rl-27b-snorkel/`, PT dates):

| run | date | shape | delta |
| --- | --- | --- | --- |
| r1 | 08-26 | 14 nodes, rbs 32 / GBS 256 / n 8, 80 rollouts | baseline; scratch `train.jsonl` (2,600 tasks) |
| r2 | 08-27 | 16 nodes | zero-variance group removal (`slime.rollout.filter_hub...check_reward_nonzero_std`), 4x examine cap |
| r3 | 08-27 | 18 nodes, rbs 256 / GBS 2048 | big-batch variant |
| r3b | 08-27 | 16 nodes, rbs 128 / GBS 256, run until stopped | "8x pushed to NATS": `ARENA_DYN_SAMPLING_MAX_EXAMINE_MULT=8` (b8aae83); survivor norm REQUIRED because rbs*n != GBS collapses slime's reward-norm reshape to one whole-batch group |
| r4 | 08-27 | 16 nodes | + removed-sample replacement, `ARENA_NATS_REPLACE_REMOVED_SAMPLES=1` (637d714) |
| r5 | 08-29 | 6 nodes = 2 actor (TP4/PP2/CP2) + 4 rollout (8 engines) | r3b algorithm on a lean topology |

b8aae83's rationale for 8x: measured keep rates of ~20-30%, so a 4x cap
back-fills unfiltered zero-variance groups at `target_groups=32`.

On miles the premises differ: `_normalize_rewards_by_rollout` keys reward
groups on `Sample.group_index` (ADR-0003), so survivors normalize per group
natively and the r3b failure mode does not exist at rbs*n == GBS;
`check_reward_nonzero_std` ships in `miles.rollout.filter_hub` (proven by
loading and calling it: mixed group keep=True, zero-variance keep=False);
`--dynamic-sampling-max-examine-mult` is a registered flag (default 4.0), not
the getattr-only attribute that forced AGISlime into a pod-env override.

## Decision

1. Keep: the r2 filter via `dynamic_sampling_filter_path =
   miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`
   with the registered 4x cap (flag unset); the r1 batch shape
   (`rollout_batch_size` 128 -> 32, GBS 256, n 8, `num_rollout` 100000 -> 90,
   ~1 epoch of 2,922 tasks); lr 1.5e-6; run5's 6-replica topology; W&B
   `arena/rl-snorkel27`.
2. Descope: `custom_reward_post_process_path` (survivor norm), the
   `removed_sample_replacement` module and its test (deleted from the plugin),
   `ARENA_NATS_REPLACE_REMOVED_SAMPLES`, the examine-mult pod-env override, the
   8x budget. `log_passrate: false` stays to match the run5 record.
3. ARGV_PARITY.md applies the descoping to BOTH sides (old = run5-minus-customs
   through AGISlime's converter + entrypoint, new = the miles launcher), so the
   record names only porting deltas (old 110 vs new 101 flag occurrences, 9
   named) and a config decision cannot hide a porting error.
4. Dataset (also per user): the `KNOWN_GYMS` lakeFS pin
   `lakefs://arena-inspect/dev/internal/snorkel-general-bash-harbor/ecr-20260823/manifest.jsonl`
   (2,922 tasks) replaces run5's scratch-staged `train.jsonl`.

### Alternatives considered

- **Port run5 verbatim** (survivor norm + replacement + 8x budget): rejected;
  three custom pieces would ride along before the plain path was proven on
  GPU, and their justification (rbs*n != GBS) does not hold on miles.
- **Keep only the survivor norm as a post-process hook**: rejected; at rbs*n
  == GBS it differs from the native norm only in how removed rows are treated
  (forced 0.0, population std over survivors) - recorded as the consequence
  to watch instead of carried as code.
- **Keep the 8x budget via the registered flag**: rejected for the first port;
  the flag stays available without any env override.
- **Drop the r2 filter too**: rejected; binary verifier rewards make
  all-pass/all-fail groups carry zero GRPO advantage, and the filter is stock
  miles.

## Consequences

Easier: no custom hooks or env toggles; the parse probe asserts
`custom_reward_post_process_path=None`, `dynamic_sampling_max_examine_mult=4.0`,
`rollout_batch_size=32`, `global_batch_size=256`; the GLM-5.3-Flash configs
inherit the plain stack unchanged.

Harder / to watch:

- **Removed/truncated samples' rewards participate in the native group
  baseline.** Asserted by the e2e-snorkel harness (round r2-heavy): rewards
  `[1, 0, 0, 1]` with rows 1,3 removed give +/-0.866 on all four rows, where
  the survivor norm gives +/-1.0 to the survivors and 0.0 to removed rows.
  Removed rows train nothing (mask sum 0), so the effect is on the survivors'
  advantage scale. First suspect if training regresses; reference
  implementation AGISlime `637d714` / `b8aae83`.
- The 4x cap binds at the measured keep rate: the GPU smoke kept 8 of 31
  examined groups (rollout 0) and 8 of 32 = cap (rollout 1). Raising
  `--dynamic-sampling-max-examine-mult` is the miles-native lever.
- Training on all 2,922 tasks includes run5's 322 held-out `dev.jsonl` tasks;
  an offline eval against that split is contaminated.
