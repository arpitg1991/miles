# ADR-0013: The gym owns the loss mask; the trainer trains every verified episode

**Status:** Accepted
**Date:** 2026-09-23

**Supersedes:** ADR-0009 (`--arena-mask-clipped-final-turn`), ADR-0010
(`--arena-keep-timeout-trajectories` and its r8 sibling
`--arena-keep-context-error-trajectories`)
**Amends:** ADR-0002 (the "Degenerate stops" row: `_DEGENERATE_AGENT_STOP` is
gone and no agent stop removes a sample), ADR-0011 (the "Truncation stays an
episode property" and "Harbor chain step" bullets: no per-segment salvage, no
`truncated_spans`, no `--arena-truncated-turn-rule`)
**Builds on:** ADR-0004 (`remove_sample` and rollout-logprob semantics),
ADR-0007 (a removed sample's reward still enters the group baseline),
ADR-0011 (one `Sample` per segment)
**Pairs with:** AREnATasks ADR-0066 (the gym side of this contract, same date)

## Summary

Two rules replace six flags. (1) The trainer keeps every episode whose
verifier ran, whatever `agent_stop_reason` says. It removes a sample only
when the token stream is broken. (2) The gym masks every clipped or empty
generate in `loss_mask` itself, so the trainer applies no advantage transform
to clipped turns. The code paths of ADR-0009 and ADR-0010 and the
`--arena-truncated-turn-*` ladder were removed on 2026-09-23.

## Context

- **Three treatments for one event.** A generate that hit the per-turn cap
  (`finish_reason == "length"`) took a different path by position and by
  flag. A clipped final turn set `stop_reason = "length"` on the step,
  marked the sample TRUNCATED, and removed it unless
  `--arena-mask-clipped-final-turn` zeroed the trailing 1-run (ADR-0009). A
  clipped mid-episode turn rode `truncated_spans` with `loss_mask = 1` and
  took `--arena-truncated-turn-rule` (`mask`, `flip`, `flip_positive`,
  `shift`, with `--arena-truncated-turn-lambda` and
  `--arena-truncated-turn-min-adv`) through `Sample.advantage_scale`. A
  clipped turn inside an archived compaction segment carried no
  `stop_reason` and trained at full weight (ADR-0011).
- **Two salvage flags for two agent stops.** `_DEGENERATE_AGENT_STOP` marked
  `context_error`, `context_churn`, `empty_response`, `max_budget`,
  `timeout` and `no_choices` as TRUNCATED and removed them.
  `--arena-keep-timeout-trajectories` (ADR-0010) and
  `--arena-keep-context-error-trajectories` (r8, no ADR) each kept one stop
  back, with precedence rules against the final-turn mask. The verifier had
  run on every one of those episodes; the reward was real.
- **Six flags across two repositories set the fate of one token run.** The
  gym decided what to ship (`truncated_spans`, `stop_reason`); the trainer
  decided what to train. Per-run behaviour depended on the flag set in
  `miles-config.yaml` and on the image that parsed it.
- **Misread telemetry.** `Sample.status == TRUNCATED` flipped on any single
  clipped turn or any degenerate stop. `rollout/truncated_ratio` and
  `rollout/truncated_ratio_prefilter` were read as "episodes ended by
  truncation" when they meant "any clipped turn or degenerate stop". A
  comment in `miles/ray/rollout/metrics.py` had to warn against that
  reading.
- **No measurable effect.** Analysis of r30 (auctioneer, 2026-09-21 to
  2026-09-23) showed that the `shift` rule touched about 0.3 percent of
  response tokens. The run already kept timeouts and context errors through
  both salvage flags. The ladder changed nothing the loss could see.
- **Gym side.** AREnATasks ADR-0066 (2026-09-23) moves the mask into the
  gym: `mask_last_output()` zeroes every clipped generate and every empty
  un-clipped generate, `truncated_spans` leaves the payload, and
  `masked_output_tokens` enters it.

## Decision

### 1. Every verified episode trains

`_step_to_sample` and `_messages_to_sample` set `status = COMPLETED` for
every kept sample. `agent_stop_reason` is telemetry. `completed`,
`truncated`, `empty_response`, `timeout`, `context_error`, `max_budget`,
`max_iterations`, `error`, `cancelled` and any other value keep the sample,
keep `loss_mask` as shipped, and keep `remove_sample = False`. A caller with
`args=None` gets the same result.

A sample leaves the loss only when its token stream is broken:

| Removal | Condition | `status` | `removal_reason` |
| --- | --- | --- | --- |
| Gym-side pad | `metadata["mode"] == "failed"`; the verifier never ran | as stamped by `_process_group` | `failed` (counted by `_removal_reason_counts`) |
| Hard overflow | `len(token_ids) > max_ctx`, both paths | TRUNCATED | `context_overflow` |
| Log-prob mismatch | fast path, `len(rollout_log_probs) != response_length` | ABORTED | `bad_logprobs` |
| No log-probs | slow path under `use_rollout_logprobs` | ABORTED | `no_logprobs` (unchanged) |
| No routing | slow path under `use_rollout_routing_replay` (ADR-0012) | ABORTED | `no_routing` (unchanged) |

`hard_overflow` and `bad_logprobs` stay per segment (ADR-0011). The miles
`log_rollout_data` metric `rollout/truncated_ratio` therefore equals the
hard-context-overflow share of the batch and nothing else.

### 2. No advantage transform for clipped turns

The gym masks every clipped generate (`finish_reason == "length"`) and every
empty un-clipped generate with `mask_last_output()`. Those tokens arrive with
`loss_mask = 0` and get no gradient. The trainer does not touch the mask.
The gym no longer sends `truncated_spans`; when an old gym image sends the
field, `_step_to_sample` ignores it. `Sample.advantage_scale` stays `None`
on the arena path. `apply_advantage_scale` stays in
`miles/backends/training_utils/loss_hub/advantages.py` as a plain multiply
for other callers.

### 3. Removed flags

argparse rejects all six. A `miles-config.yaml` key for any of them kills the
trainer at startup on an image built from this commit or later.

| Flag | Was |
| --- | --- |
| `--arena-truncated-turn-rule` | `mask` / `flip` / `flip_positive` / `shift` on `truncated_spans` |
| `--arena-truncated-turn-lambda` | the `shift` amount |
| `--arena-truncated-turn-min-adv` | the `shift` floor |
| `--arena-mask-clipped-final-turn` | ADR-0009 |
| `--arena-keep-timeout-trajectories` | ADR-0010 |
| `--arena-keep-context-error-trajectories` | ADR-0010 sibling (r8) |

### 4. Removed code

- `miles_plugins/arena/nats_arena/nats_rollout.py`: `_DEGENERATE_AGENT_STOP`,
  `_TRUNCATED_TURN_RULE_SCALE`, `_truncated_spans_scale`,
  `final_clip_masked`, the salvage blocks (`stop_salvageable`,
  `kept_timeout`, `kept_context_error`, `timeout_salvageable`,
  `context_salvageable`), `any_step_truncated`, `_EpisodeContext.status`,
  the `is_final` argument of `_step_to_sample`, the
  `truncated_spans_out_of_range` warning, the `kept_timeout=` /
  `kept_context_error=` suffix of the `Removal reasons:` log line, and the
  `span_tokens` / `has_advantage_scale` columns of the sample summary.
- `miles/backends/training_utils/loss_hub/advantages.py`:
  `apply_truncated_turn_shift` and the `positive_only` argument of
  `apply_advantage_scale`.
- `miles/backends/training_utils/loss.py`: the rule branch in
  `compute_advantages_and_returns`.
- `miles/backends/training_utils/loss_hub/losses.py`: the
  `arena_truncated_turn_rule` metric block.
- `miles/backends/training_utils/log_utils.py`: the `adv_neg_pos_ratio`
  derivation.
- `miles/backends/megatron_utils/model.py`: `truncated_turn_shifted` in the
  padded-key list.
- Metrics: `train/adv_pos_mass`, `train/adv_neg_mass`,
  `train/adv_neg_pos_ratio`, `train/truncated_turn_shifted_tokens`,
  `rollout/truncated_turn_shifted`, `rollout/truncated_ratio_prefilter`,
  `arena/truncated_turn_tokens`.
- Tests: `test_mask_clipped_final_turn.py`,
  `test_keep_timeout_trajectories.py`,
  `test_keep_context_error_trajectories.py`,
  `test_truncated_spans_advantage_scale.py`, and the rule cases of
  `test_advantage_scale.py`.

### 5. New telemetry

`_stop_metrics(all_data)` runs over the pre-filter episodes that came from a
real trajectory. Pads with `mode == "failed"` are excluded.

| Metric | Meaning |
| --- | --- |
| `rollout/stop/<agent_stop_reason>` | share of episodes per lowercase terminal reason; `unknown` when the gym sent none |
| `rollout/clipped_turns` | mean clipped generates per episode, from the gym's `truncated_turns` |
| `rollout/masked_output_tokens` | mean masked output tokens per episode, from the gym's `masked_output_tokens`; 0 on an older gym image |

`_finish_sample` stamps `truncated_turns` and `masked_output_tokens` into
`Sample.metadata` on every sample. Read `rollout/stop/*` to explain why
episodes end. NEVER read `rollout/truncated_ratio` for that purpose; it
counts hard overflows only.

### 6. Contract with the gym

AREnATasks ADR-0066 (2026-09-23) is the gym side. Per trajectory the gym
ships `agent_stop_reason`, `truncated_turns`, `masked_output_tokens`,
`compactions` and `steps[*].stop_reason`. `truncated_turns` and
`masked_output_tokens` are present only when non-zero; a missing field reads
as 0. The gym's default compaction cap (`ARENA_COMPACTION_MAX`) is 5, and the
clipped-turn nudge cap (`ARENA_TRUNCATED_TURN_MAX`) defaults to 5. The
trainer trusts the gym mask and the gym verdict.

### Alternatives considered

| Alternative | Why not |
| --- | --- |
| Keep the flags with new defaults (`mask`, both salvages on) | Six flags still set one token run's fate; an old config still parses and drifts in silence. |
| Keep `truncated_spans` and the `mask` rule as a trainer-side backstop | Two owners of one mask; every change ships twice; the r30 span share was about 0.3 percent. |
| Keep `_DEGENERATE_AGENT_STOP` for the stops nobody salvaged (`context_churn`, `no_choices`) | The verifier ran on those episodes too. A mangled conversation is a gym defect to fix in the gym, not a trainer filter. |
| Keep `Sample.status = TRUNCATED` on a clipped final turn as telemetry | The status drives `remove_sample` in `_messages_to_sample` and drove the misreads of `truncated_ratio`; `rollout/stop/*` and `rollout/clipped_turns` carry the signal. |
| Disable the code paths behind a default and keep them in the tree | Dead code. Version control is the safety net. |

## Consequences

- The code paths of ADR-0009 and ADR-0010 no longer exist in the repo. They
  were removed, not disabled. Both ADRs are marked superseded and stay as
  history.
- Historical run directories `examples/arena/harbor-rl-glm53-flash/r9` to
  `r35` and the RUNLOGs keep the old keys (`arena_mask_clipped_final_turn`,
  `arena_keep_timeout_trajectories`,
  `arena_keep_context_error_trajectories`, `arena_truncated_turn_*`) as
  history. NEVER edit them; they record what each run parsed. A new run
  copies the live `miles-config.yaml`, which no longer carries them.
- Lockstep. A trainer image built from this commit rejects the old keys at
  argparse. Bump the image tag together with the config. An old gym image on
  the new trainer trains its clipped turns at full weight (it sent
  `truncated_spans`, which the trainer ignores) and reports
  `rollout/masked_output_tokens = 0`. Pair the trainer with a gym image that
  carries AREnATasks ADR-0066.
- Trained-on samples now include every timed-out, context-ended,
  empty-response and budget-capped episode. Their reward is the verifier's
  verdict on the partial work.
- `rollout/truncated_ratio` drops to the hard-overflow share, near 0 on a
  healthy run. A dashboard that read it as a truncation rate reads
  `rollout/stop/truncated` and `rollout/clipped_turns` instead.
- `rollout/removed_sample_frac` and the `Removal reasons:` line now name
  `context_overflow`, `bad_logprobs`, `no_logprobs`, `no_routing`, `failed`
  and `other` only.
- Tests: `tests/fast/plugins/arena/test_stop_reason_policy.py` pins every
  stop reason as telemetry, the two broken-stream removals, `_stop_metrics`,
  `_removal_reason_counts`, `advantage_scale is None` under a legacy
  `truncated_spans`, and the argparse rejection of all six flags.
  `test_advantage_scale.py` keeps the plain-multiply cases.
- ADR-0002's "Degenerate stops" row and ADR-0011's truncation bullets
  describe removed behaviour. Both headers carry an `Amended by` pointer to
  this ADR.
