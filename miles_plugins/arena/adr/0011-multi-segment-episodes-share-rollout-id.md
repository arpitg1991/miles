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

### Precedent

miles upstream already runs this design for its own rollout functions. PR
#1927 (`11cc2326`, variable global batch size) made `global_batch_size` count
rollouts. `rollout_id` marks compact siblings. `_compute_rollout_mask_sums`
sums the loss mask per rollout, and `build_dp_schedule` keeps one rollout's
samples in one training step. PR #2368 (`95ecc712`) stamps one `rollout_id`
per leaf. PR #2710 (`f2b7c792`) added `_compute_training_sample_metrics` to
`miles/ray/rollout/metrics.py`: `rollout/num_training_samples` and
`rollout/episode_raw_reward`, keyed by `(group_index, rollout_id)`. miles logs
them for every rollout function through `rollout_manager.generate` ->
`log_rollout_data`. PR #2741 (`bfe13117`,
`examples/experimental/terminus-compaction`) ships one `Sample` per kept
leaf. Siblings share a rollout ID and the terminal reward. A shared
completion is masked on every owner but the first. This ADR ports that design
onto the NATS path. It adds no core edit.

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
- **DP alignment pad.** `build_dp_schedule` needs the micro-batch count of a
  step to be a multiple of `align_to = dp_size * (mb_group if vpp_size > 1
  else 1)`. The dynamic path grows the count with `expand_bins_by_splitting`
  (`miles/utils/seqlen_balancing.py`), which splits multi-sample bins only. A
  row that alone exceeds `max_tokens_per_gpu` is a singleton bin it cannot
  split. The static path never splits. Under `final` the row count is
  `global_batch_size`, aligned by config. Under `all` the row count is the sum
  of segments. An odd count on `dp_size == 2` therefore asserts on the rollout
  side, in `RolloutManager.generate` -> `split_train_data_by_dp` ->
  `split_train_data_by_dp_scheduled_raw` -> `build_dp_schedule`. The message
  reads `dynamic path: could only produce N micro-batches after maximal
  splitting`. Under `delay_split_train_data_by_dp` or `indep_dp`
  (`train_parallel_config` is `{}`) the legacy `split_train_data_by_dp_raw`
  runs instead. There `--balance-data` asserts `rows % dp_size == 0`
  (`get_seqlen_balanced_partitions(equal_size=True)`); `unit` is a multiple of
  `dp_size`, so the pad covers that path too. Upstream tests
  (`tests/fast/utils/test_dp_schedule.py`) cover multi-sample bins and an even
  oversized count only. The upstream Terminus recipe (PR #2741: TP4 on 8 GPUs,
  `dp_size` 2, `--max-tokens-per-gpu 16384` against 32k-token rows) has the
  same exposure. `generate_rollout` therefore calls
  `_pad_rows_to_dp_alignment(data, args)` right after the episode-count guard.
  `dp_size = actor_num_nodes * actor_num_gpus_per_node // (tp * pp *
  context_parallel_size)`, with `tp` and `pp` read through `getattr(args, ...,
  1)`. FSDPArgs (`miles/backends/fsdp_utils/arguments.py`) defines
  `context_parallel_size` only, and FSDP `dp` is `world // cp`. `vpp_size`
  reads `args.virtual_pipeline_model_parallel_size` (set by Megatron
  `validate_args`); `mb_group` mirrors `_compute_vpp_fields`, which returns
  `pipeline_model_parallel_size` when vpp > 1. `unit = align_to` on the
  dynamic path and `align_to * micro_batch_size` on the static path;
  `pads = (-rows) % unit`. Each pad is a zero-loss sibling segment of the
  SHORTEST kept row (fewest tokens, `remove_sample` False). It copies `tokens`,
  `response_length`, `reward`, `group_index`, `rollout_id` and
  `weight_versions`. It sets `status = COMPLETED`, `loss_mask = [0] *
  response_length`, and `rollout_log_probs = [0.0] * response_length` when the
  source carries log-probs. It takes `index = base + len(episode) *
  _SEGMENT_INDEX_STRIDE`. The pad joins the source's group, so a second pad
  from the same episode takes the next stride and `index` stays unique.
  `metadata` is a copy with `mode = "dp_pad"` and `segment = len(episode)`.
  `n_segments` keeps the copied value; no code in miles or the plugin
  validates `segment < n_segments` (both keys are informational).
  `remove_sample` stays False: `removed_frac` flags an episode when ANY
  segment is removed, and a pad is not lost training signal. The pad adds one
  row with the episode reward to `_normalize_rewards_by_rollout` and zero to
  `_compute_rollout_mask_sums`. The episode's advantage and loss weight
  therefore do not change. The pad status is `COMPLETED` by design: a
  zero-loss row never trains, and a copied `TRUNCATED` inflates the per-sample
  `rollout/truncated_ratio` and `train_data["truncated"]`. Known drift: the
  pad adds the shortest row's tokens to the response-length sums
  (`avg_response_length`, upstream `rollout/episode_total_response_length/mean`).
  That drift is bounded by `unit - 1` rows per batch. Under `--log-passrate`,
  `_compute_passrate_from_samples` (`miles/ray/rollout/metrics.py`) keeps only
  groups with exactly `n_samples_per_prompt` rows. A padded group holds
  `n_samples_per_prompt + 1` rows or more, so the metric drops it as incomplete
  and logs a warning. The metric drops a group with a multi-segment episode the
  same way. One `DP alignment:` warning names rows, pads and unit. The upgrade
  path is an upstream fix in `build_dp_schedule` (pad there, or let a rank take
  zero micro-batches). The helper then pads nothing and can go.
- **Truncation stays an episode property.** Only the final step can carry
  `stop_reason="length"` (archived segments have no `stop_reason`). A final
  `length` marks every segment TRUNCATED. `--arena-mask-clipped-final-turn`
  (ADR-0009) zeroes the trailing 1-run of the FINAL segment only; earlier
  segments contain no clipped tokens and stay trainable (they record
  `final_clip_masked` without a change to their mask). A final segment
  whose only turn is the clipped one has nothing to salvage: that segment
  alone is removed with `removal_reason="length"` and the earlier segments
  still train. With the flag off, every segment is removed, today's
  whole-episode removal. `--arena-keep-timeout-trajectories` (ADR-0010) and
  `--arena-keep-context-error-trajectories` are trajectory-level and salvage
  every segment. `hard_overflow` and `bad_logprobs` are checked per segment,
  so `remove_sample`, `removal_reason` and `status` can differ between the
  segments of one episode. `weight_versions` is the trajectory list on every
  segment.
- **Telemetry counts episodes.** `generate_rollout` groups samples into
  episodes by `(group_index, rollout_id if not None else index)`
  (`_episodes`). `avg_reward`, `nonzero_count` and
  `failed_frac` read the first segment (`_episode_representatives`; reward,
  `weight_versions` and `group_metrics` live there).
  The per-segment flags reduce over the episode: `removed_frac` counts an
  episode when ANY segment has `remove_sample`, `truncated_ratio` when ANY
  segment is TRUNCATED, and the "Removal reasons" breakdown reads the
  removed segment (`_removal_representative`), so a final segment dropped
  under `--arena-mask-clipped-final-turn` or a per-segment `bad_logprobs`
  is visible. `avg_response_length` sums `response_length` per episode. The
  plugin logs no sample count of its own: miles `log_rollout_data` already
  logs `rollout/num_training_samples` and `rollout/episode_raw_reward` for
  this batch (PR #2710). Under `final` every episode has one sample, so every
  number is unchanged. `s.metadata["segment"] = k` and `s.metadata["n_segments"] = n`
  on every sample: `_finish_sample` stamps `0` and `1`, so a messages-only
  (slow-path) sample and a pad episode read as segment 0 of 1, and the fast
  path overwrites both for a multi-segment episode.

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
- **Metrics.** miles `log_rollout_data` (`miles/ray/rollout/metrics.py`;
  `_compute_metrics_from_samples` output gets the `rollout/` prefix through
  `dict_add_prefix`) logs for this path: `rollout/num_training_samples` (every
  row, pads included) and `rollout/episode_raw_reward` (mean over episodes
  keyed by `(group_index, rollout_id)`), plus
  `rollout/episode_response_length/{mean,...}` and
  `rollout/episode_total_response_length/mean` summed per episode. Read
  `rollout/num_training_samples / global_batch_size` as the mean segments per
  episode on the first r12 comparison; under `final` it is 1.0. The plugin
  emits no segment metric of its own. A pad shows up in the `DP alignment:`
  warning line and as `metadata["mode"] == "dp_pad"`.
- **Tests.** `tests/fast/plugins/arena/test_multi_segment_episodes.py` pins
  one behaviour per test:
  - both modes and the marker gate;
  - the stamping on the fast path, the slow path and the gym-side pads;
  - the shared reward through the real conversion path;
  - the truncation and salvage semantics per segment;
  - the partial-batch guard;
  - the per-episode `removed_frac` and "Removal reasons" line for an episode
    whose final segment alone is dropped;
  - the DP alignment pad: 9 rows on `dp_size` 2 pad to 10 with a zero-loss
    sibling of the shortest kept row (shared `rollout_id`, `group_index` and
    reward, unique `index`, `remove_sample` False, `mode == "dp_pad"`);
  - a `TRUNCATED` source yields a `COMPLETED` pad, so the per-sample truncated
    count and the per-episode `truncated_ratio` do not grow;
  - FSDP-shaped args (no `tensor_model_parallel_size` or
    `pipeline_model_parallel_size`) pad to `dp_size = world // cp`;
  - an even row count adds nothing and logs nothing;
  - the static path pads to a multiple of `dp_size * micro_batch_size`;
  - `build_dp_schedule` asserts on the 9 oversized rows and schedules the
    padded 10;
  - `final` mode never pads.

  `test_group_identity.py` is unchanged and still pins `rollout_id=None`
  under the default mode; its module docstring records this amendment.
- **ADR-0003 status.** Its decision stands for the default mode and its
  rejection of the GROUP id stands in every mode. Read its sentence
  "`rollout_id` deliberately stays None" as scoped to `final` mode; its
  header carries an `Amended by` pointer to this ADR.
