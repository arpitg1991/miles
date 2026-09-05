# ADR-0010: Keep Harbor agent-timeout trajectories as training samples, opt-in, only when the timeout is the sole defect

**Status:** Accepted
**Date:** 2026-09-03

**Builds on:** ADR-0002 (`success` and `truncated` envelopes are both salvageable), ADR-0007 (plain stack: removed samples' rewards already enter the group baseline), ADR-0009 (`--arena-mask-clipped-final-turn`)

## Context

GLM-5.3-Flash runs r1/r2 (2026-09-02/03) trained on ~10% of every 256-sample
batch: run 1 kept 23/256; run 2 (image b) removed 230/256 and 227/256
(`truncated_ratio` 0.898 / 0.883, ess_ratio 0.099); r2 (image c) removed
229/256 (0.8945). The wire looked clean throughout (run 1 rollout 0: 272
`success` / 54 `failed` / 0 `truncated` — the Harbor 0.21 gym envelope reports
timed-out groups as `success`), and ADR-0009's mask-clipped salvage, written
on the length-clip reading of run 1, salvaged nothing in r2.

The truncation RCA (2026-09-03 ~00:40-01:30 PT) therefore went to the gym pods
of r2: 96 trials (12 task groups x 8, run 23:07-00:29 PT) from 12 of the 128
`rl-glm53f-gym-sgb-*` pods, the gym source as deployed (`amzn_arena_harbor
1.0.763.0`) and the task-events logs:

- 91/96 trials ended in `AgentTimeoutError: Agent execution timed out after N
  seconds`, N = the task's `task.toml` `[agent] timeout_sec` (1800 s x59,
  1500 x16, 1200 x8, 900 x8; no gym override), reached 31-85 s after trial
  start. 5/96 finished.
- No clipping: 0/1033 per-turn outputs at the 32768 cap (max 15033, p50 630,
  p90 4948). All 96 trajectories end at the model's turn-close token with no
  tokens after the last assistant turn; every step `stop_reason=stop`;
  `weight_versions` count == turn count.
- Real episodes with real rewards: timeouts had a median of 8 turns / 34k
  tokens (max 74k < 131k cap); the verifier ran and 8/91 scored 1 (~9%) —
  >=19 of the ~46 reward-1 samples per step were being discarded.
- Cause: GLM at template-default `max` reasoning effort thinks 8-15k
  tokens/turn at p50 14.5 tok/s per sample (~400 concurrent trials on 4
  saturated engines); a >8k-token turn takes ~500 s (61 such turns, median
  latency 494 s).
- Trainer mechanism: the gym's `_EXCEPTION_AGENT_STOPS` maps
  `AgentTimeoutError` to `agent_stop_reason="timeout"`; `timeout` is in
  `nats_rollout._DEGENERATE_AGENT_STOP`, so the sample is TRUNCATED +
  `remove_sample` (the r5-lineage policy — right for `context_error`,
  `context_churn`, `empty_response`, `max_budget`, `no_choices`, whose
  conversations are mangled; wrong for a clean episode that ran out of clock).

Removing the cause (more wall-clock or less thinking) is gym- or dataset-side
and could not ship in a trainer image that night; r3 had just launched on
image c (01:32 PT) with the same loss.

## Decision

Add `--arena-keep-timeout-trajectories` (store_true, default off, registered
in `_add_arena_arguments`, read via `getattr` so `args=None` callers keep the
old behaviour). With the flag on, a fast-path (token-level) trajectory with
`agent_stop_reason == "timeout"` is kept — `loss_mask` untouched,
`remove_sample` False — **only when the timeout is its sole defect**: no step
with `stop_reason=length`, no hard context overflow, no zero-filled log-probs,
non-empty response. Other degenerate stops are unaffected. `Sample.status`
stays TRUNCATED in both modes, so `rollout/truncated_ratio_prefilter` and
miles' `rollout/truncated_ratio` keep meaning "the agent loop did not finish".

Precedence with ADR-0009: timeout + clipped final turn is salvaged by the
mask-clipped path when both flags are on (the timeout no longer blocks it);
with only this flag on it stays removed (this branch never keeps a clipped
turn, reason `timeout`); with only mask-clipped on the degenerate stop still
blocks the salvage (unchanged). Slow-path (messages-only) trajectories are out
of scope, as in ADR-0009.

Bookkeeping: every removal stamps `metadata["removal_reason"]`
(`bad_logprobs` / `no_logprobs`, `context_overflow`, the agent stop such as
`timeout`, `length`, `other`; gym-side pads count as `failed`), every salvage
stamps `metadata["kept_timeout"]`, every real sample records
`agent_stop_reason`, and `generate_rollout` logs once per rollout, after the
failed-sample distribution:
`Removal reasons: timeout=N, context_error=N, length=N (kept_timeout=N)`
(`_removal_reason_counts`). Under the flag `timeout=N` means "timeout AND
another defect".

### Alternatives considered

| Alternative | Why not (then) |
| --- | --- |
| Raise the Harbor per-task `timeout_sec` | Gym/dataset-side: `gym_worker.build_rollout_job_config` hard-wires `task.toml`; needs a gym image and `ARENA_NATS_ACK_WAIT` in lockstep. Pursued in AREnATasks as `ARENA_AGENT_TIMEOUT_MULTIPLIER` (r6). |
| Cut thinking via `reasoning_effort` / `max_thinking_tokens` | The gym's `ArenaSGLangLLM` dropped the kwarg; needs a gym image. Pursued in AREnATasks as `ARENA_REASONING_EFFORT` (r5 gym image). |
| More engine throughput | Capacity; taken in r6 (16 engines). Does not change what the trainer does with the timeouts it still receives. |
| Reclassify `timeout` as COMPLETED / drop it from `_DEGENERATE_AGENT_STOP` | Hides the signal (`truncated_ratio` would read ~0.1 while ~89% of episodes never finished) and silently flips the r5-lineage default for every config. |
| Loosen the degenerate-stop policy wholesale | The other five stops are mangled conversations and must not be trained on. |
| Accept the loss | ~10% of GBS carried loss and the reward-1 signal sat disproportionately in the discarded set. |

## Consequences

- Default off: r3 (image c) and r4 (image d, flag unset) train as before; r4
  is the first run whose log attributes the loss (`Removal reasons:
  timeout=230 (kept_timeout=0)` in rollout 0; 222-245/256 per step after).
  First enabled in r5 (`arena_keep_timeout_trajectories: true`; image d
  required — strict argparse kills the trainer on image c).
- With the flag on, `truncated_ratio` no longer approximates the removed
  fraction: r5 logged `kept_timeout` 183-221/256 with truncated 0.71-0.86;
  read `kept_timeout` and the removal-reason line. r6 (2x timeout multiplier,
  16 engines) brought it to 228/512 (45%).
- Trained-on samples now include episodes cut mid-task by the wall clock;
  their reward is the verifier's verdict on the partial work (mostly 0, ~9%
  1). Previously only their reward entered the group baseline (ADR-0007);
  now their tokens carry loss too.
- The trainer trusts the gym's `agent_stop_reason` mapping. Harbor 0.22 gyms
  (r5+) also mark timed-out groups `truncated` on the wire — telemetry only,
  both statuses are salvageable (ADR-0002).
- Pinned by `tests/fast/plugins/arena/test_keep_timeout_trajectories.py`
  (19 cases covering both flag states, every other defect, the precedence
  with ADR-0009 and `_removal_reason_counts`); `test_mask_clipped_final_turn.py`
  passes both knobs explicitly.
- Not a fix for the wall clock: the gym-side levers (timeout multiplier,
  reasoning effort, SGLang request timeout) are recorded in AREnATasks
  (ADR-0047) and in `examples/arena/harbor-rl-glm53-flash/RUNLOG.md`.
