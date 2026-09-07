# ADR-0011: Multi-segment episodes share one `rollout_id` behind `--arena-train-segments all`

**Status:** Accepted
**Date:** 2026-09-07

**Amends:** ADR-0003 (group identity: `rollout_id` is no longer always `None`;
in `all` mode it carries the EPISODE id, never the group id)
**Builds on:** ADR-0004 (per-trajectory loss weighting keyed on `rollout_ids`),
ADR-0009 (`--arena-mask-clipped-final-turn`), ADR-0010 (`--arena-keep-timeout-trajectories`)
**Pairs with:** AREnATasks ADR-0048 (Vulcan context compaction and rollout
segments; the gym side of this contract)

## Context

- AREnATasks ADR-0048 gives the Vulcan agent token-aware context compaction
  under `arena-sglang`. A compaction is a segment boundary:
  `RolloutState.begin_segment("compaction")` archives the live token stream
  and the next `/generate` re-renders the compacted history from scratch. The
  gym therefore ships a trajectory whose `steps` holds several self-contained
  segments. The final segment (the one that produced the episode end) is
  last and keeps today's top-level shape. Each archived segment carries
  `token_ids`, `loss_mask`, `log_probs`, `has_generate_tokens: true`,
  `truncated_generates`, and `segment_end` (`"compaction"` or
  `"prefix_mismatch"`), and NO `stop_reason`. A single-segment episode is
  byte-identical to today's `rollout.json`.
- `_result_to_samples_full_trajectory` reads `steps[-1]`. On a compacted
  episode that trains the post-compaction tail only: the earlier segments,
  including the policy-written handoff note, get the reward but no loss. An
  old trainer on a new gym degrades this way, aligned but silent.
- The Inspect `sglang_perstep` provider (`amzn_arena_streaming/sglang_provider.py`)
  already ships cumulative per-turn steps with `has_generate_tokens: True` on
  EVERY step. Those steps are prefixes of one stream, not self-contained
  segments; a trainer that expands them trains the same tokens several
  times. The trainer therefore needs an explicit marker before it expands
  a step list.
- miles core keys the training math on `rollout_id` (fallback `index`):
  `_normalize_rewards_by_rollout` demands one reward per key and broadcasts
  it (`train_data_conversion.py:238-285`); `_compute_rollout_mask_sums` sums
  the loss mask per key (`:193-200`); `build_dp_schedule` keeps one key's
  samples in one training step or drops them together (`dp_schedule.py:55-70`);
  `postprocess_rollout_data` sets `is_compact` when any `rollout_id` is set
  and then skips the sample-count trim (`rollout_data_conversion.py:30-32`);
  `can_schedule_on_rollout_side` needs `len(set(rollout_ids)) >= global_batch_size`
  (`train_data_conversion.py:319-330`).
- ADR-0003 rejected stamping the GROUP id on `rollout_id`, because miles then
  demands one reward per group and raises `all samples in rollout N must
  share one reward` on any within-group variance. That rejection was about
  the group id. The EPISODE id is exactly `Sample.rollout_id`'s documented
  meaning: compact siblings of ONE rollout execution that share one reward.
  Every segment of one episode has one verifier reward.

## Decision

Add `--arena-train-segments {final,all}` (`type=str`, `default="final"`,
registered in `_add_arena_arguments`, read via `getattr` so `args=None`
callers keep the old behaviour). The default is byte-identical to today.

| Mode | Steps trained | Stamping per episode `e` of group `gid` (`base = gid*n + e`) |
| --- | --- | --- |
| `final` | `steps[-1:]` | `group_index=gid`, `index=base`, `rollout_id=None` (ADR-0003 pins hold) |
| `all` | every step with `has_generate_tokens`, only when some step carries `segment_end`; else `steps[-1:]` | on EVERY sample `k` of the episode: `group_index=gid`, `rollout_id=base`, `index=base + k * (1 << 40)` |

- **One `Sample` per segment, one episode.** `_result_to_episodes_full_trajectory`
  returns `list[list[Sample]]`; `_step_to_sample` lifts the fast-path body and
  runs once per training step. `_result_to_samples_full_trajectory` flattens
  it, so its signature and default-mode output are unchanged. `_process_group`
  pads and trims in episodes, then flattens before `output_queue.put`.
- **Gate on `segment_end`.** `all` expands a step list only when some step
  carries the explicit marker. Inspect `sglang_perstep` step lists never
  carry it and stay on the `steps[-1:]` path in both modes.
- **Episode id on `rollout_id`.** `base = gid*n + e` is today's `index`, so
  `k=0` reproduces today's value; `_SEGMENT_INDEX_STRIDE = 1 << 40` keeps
  `index` unique and int64-packable. With this stamp miles core does the
  right thing without a core edit: `_normalize_rewards_by_rollout` gives one
  advantage per episode (the shared-reward assert holds by construction);
  `_compute_rollout_mask_sums` gives one token-weighted loss unit per
  episode across micro-batches; `build_dp_schedule` keeps all segments of an
  episode in one step; `can_schedule_on_rollout_side` still counts episodes
  (r11: 64 groups x 8 = 512 = `global_batch_size`).
- **Compact mode.** Any non-`None` `rollout_id` flips `postprocess_rollout_data`
  to compact and skips the sample-count trim. `generate_rollout` already
  delivers exactly `GBS // n * n` episodes, so the trim was a no-op. A short
  batch (worker death) now fails at `build_dp_schedule`'s
  `num_full_steps >= 1` assert, or earlier at the new guard in
  `generate_rollout` (`RuntimeError` naming the episode count when
  `len({episode keys}) < global_batch_size`), instead of `postprocess`'s
  `Not enough samples`. Same severity, different line.
- **Truncation stays an episode property.** Only the final step can carry
  `stop_reason="length"` (archived segments have no `stop_reason`). A final
  `length` marks every segment TRUNCATED. `--arena-mask-clipped-final-turn`
  (ADR-0009) zeroes the trailing 1-run of the FINAL segment only; earlier
  segments contain no clipped tokens and stay trainable (they record
  `final_clip_masked` without a change to their mask). With the flag off,
  every segment is removed, today's whole-episode removal. `--arena-keep-timeout-trajectories`
  (ADR-0010) and `--arena-keep-context-error-trajectories` are trajectory-level
  and salvage every segment. `hard_overflow` and `bad_logprobs` are checked
  per segment. `weight_versions` is the trajectory list on every segment.
- **Telemetry counts episodes.** `generate_rollout` keys on
  `(group_index, rollout_id if not None else index)`: `total_samples`,
  `avg_reward`, `nonzero_count`, `failed_frac`, `removed_frac`,
  `truncated_ratio` count one representative per episode;
  `avg_response_length` sums `response_length` per episode; the new
  `rollout/compaction_segments_mean` is the mean number of samples per
  episode. Under `final` every episode has one sample, so every number is
  unchanged. `s.metadata["segment"] = k` and `s.metadata["n_segments"] = n`
  on every sample.

### Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Concatenate the segments into one sample | Breaks importance sampling. Each segment's `rollout_log_probs` were produced under that segment's exact prompt; the actor recomputes log-probs on the same `tokens`, so the ratios are exact only per segment. A concatenation changes the conditioning context of every post-compaction token. miles asserts prefix extension when it merges samples for this reason (`miles/rollout/generate_utils/sample_utils.py:138`). |
| Train the final segment only (`steps[-1]`) | Kept as the default `final`. Loses the handoff note and every pre-compaction turn from the gradient; an old trainer on a new gym degrades to this path. |
| Stamp the group id on `rollout_id` | Rejected in ADR-0003 and still wrong: miles demands one reward per group. |
| Per-segment truncation (mark only the clipped segment TRUNCATED) | Drifts from today's whole-episode semantics and from the `truncated_ratio` meaning "the agent loop did not finish". ADR-0009's final-turn salvage stays the only exception. |
| Expand every multi-step trajectory without a marker | Inspect `sglang_perstep` cumulative steps train the same tokens once per turn. |
| Per-segment `weight_versions` slices | Not needed for `compute_off_policy_metrics` (counted once per episode); named as the upgrade path. |

## Consequences

- **Lockstep order: gym first, trainer second.** An old trainer on a new gym
  reads `steps[-1]` = the final segment with the episode reward: aligned but
  degraded. A new trainer in `all` mode on an old gym, or on an Inspect gym,
  sees no `segment_end` marker and behaves exactly as today.
- **r12 config.** Set `arena_train_segments: all` in the r12
  `miles-config.yaml` (not part of this change; strict argparse kills an
  older trainer image on the key, so bump the image together with the key).
  Start the gym with `ARENA_COMPACTION_MAX=2`: the NATS result envelope
  grows with `8 trials x (compactions + 1) segments x up to 131k tokens`
  against the r11 `max_payload: 8388608`; the gym warns above 6 MiB.
- **Loss weighting.** A 3-segment episode is one entry in the GRPO
  statistics and one `rollout_mask_sums` denominator, so the episode's total
  weight does not depend on how many times it was compacted (ADR-0004's
  per-trajectory weighting, unchanged in spirit).
- **Summary-turn loss mask is a gym-side knob.** `ARENA_COMPACTION_TRAIN_SUMMARY`
  (default `1`) sets the `loss_mask` of a used handoff note; a discarded note
  is always masked by the gym. The trainer trains whatever mask it receives.
- **Failure mode moves.** In `all` mode a partial batch fails in
  `generate_rollout`'s guard or `build_dp_schedule`, not in
  `postprocess_rollout_data`. Loud in both cases.
- **Metrics.** Read `rollout/compaction_segments_mean` next to reward on the
  first r12 comparison; under `final` it is 1.0.
- **Tests.** `tests/fast/plugins/arena/test_multi_segment_episodes.py` pins
  both modes, the marker gate, the stamping, the shared reward through the
  real conversion path, the truncation semantics, and the guard.
  `test_group_identity.py` is unchanged and still pins `rollout_id=None`
  under the default mode; its module docstring records this amendment.
- **ADR-0003 status.** Its decision stands for the default mode and its
  rejection of the GROUP id stands in every mode. Read its sentence
  "`rollout_id` deliberately stays None" as scoped to `final` mode; its
  header carries an `Amended by` pointer to this ADR.
