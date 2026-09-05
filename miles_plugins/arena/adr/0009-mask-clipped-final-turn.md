# ADR-0009: Salvage a final-turn-clipped trajectory by masking only the clipped turn

**Status:** Accepted
**Date:** 2026-09-02

**Builds on:** ADR-0002 (wire contract: per-step `stop_reason`, degenerate
agent-stop set), ADR-0004 (`remove_sample` / rollout-logprob semantics),
ADR-0007 (the "r5-lineage" removal policy inherited from AGISlime snorkel run 5)
**Related:** ADR-0010 (`--arena-keep-timeout-trajectories`, the flag that
addressed what the removals actually were)

## Summary

`--arena-mask-clipped-final-turn` (default off) keeps a multi-turn trajectory
whose FINAL generate hit the per-turn token cap: only that turn's tokens are
zeroed in the response `loss_mask`, the earlier, cleanly stopped turns stay
trainable, and the sample keeps status TRUNCATED. Written 2026-09-02 16:44 PT
(integration commit 5f8925db0) while GLM run 2 was alive; shipped in trainer
image `arena-slime-dev:miles-glm53-20260902c`; `true` in
`examples/arena/harbor-rl-glm53-flash/miles-config.yaml` from r2 on.

## Context

- Inherited policy: `_result_to_samples_full_trajectory` marks a sample
  TRUNCATED and `remove_sample=True` when ANY step's `stop_reason` contains
  `length` or the trajectory's `agent_stop_reason` is in
  `_DEGENERATE_AGENT_STOP`. The Harbor gym ships one step per trajectory
  (`agent/rollout.json`, cumulative `token_ids`/`loss_mask`/`log_probs`) and
  stamps `stop_reason="length"` only when the final generate ended on
  `length` (earlier per-turn caps only count in `truncated_generates`;
  any-generate stamping had removed 67% of samples in the 27B smoke and was
  rejected gym-side). So on this gym "any step clipped" means "final
  generate clipped", and one clipped turn dropped the whole trajectory.
- What the GLM-5.3-Flash runs showed (`EXPERIMENT_NAME rl-glm53f-gbash-r1`,
  W&B `arena/rl-snorkel27`, 2026-09-02 PT):
  - run 1 (image a): after the 32k-per-turn / 131k-episode cap raise (10:41
    PT), rollout 0 came back 272 success / 54 failed / 0 truncated on the
    wire, yet only 23/256 training samples carried loss (91% removed).
  - run 2 (image b, same identity, ~14:15 PT): rollout 0 summary 15:56 PT
    `removed_total=230/256 (0.898)`, `rollout/truncated_ratio` 0.898,
    episode reward 0.152, response_len mean ~19k / max 36k tokens, 12
    zero-std groups; rollout 1 16:53 PT `227/256 (0.887)`, reward 0.258;
    train step 0 `ess_ratio` 0.099, i.e. GBS 256 training on ~26 samples.
- Reading at the time: GLM thinks 8-15k tokens per turn, so long episodes
  clip somewhere even when the gym reports success, and the intact turns
  before the clip are thrown away with it. (Wrong; see Consequences.)

## Decision

Add `--arena-mask-clipped-final-turn` (`store_true`, default `False`; YAML
`arena_mask_clipped_final_turn` via the launcher) in `_add_arena_arguments`.
With the flag on, a TRUNCATED fast-path sample whose truncation comes from a
`length` step is salvaged: the trailing run of 1s in the response `loss_mask`
(the final turn's generated tokens; nothing follows the last generate) is
zeroed, `remove_sample` stays `False`, and the rollout logs `Task <id>: masked
clipped final turn (<n> of <m> response tokens); earlier turns stay trainable.`

| Case | Behaviour |
| --- | --- |
| flag off | unchanged: whole sample removed (r5-lineage default) |
| final turn clipped, >= 1 trainable token before it | trailing 1-run zeroed, sample kept |
| single-turn trajectory, whole response clipped | still removed (nothing left to train on) |
| degenerate agent stop (`context_error`, `timeout`, ...) | still removed: a mangled conversation, not a per-turn clip |
| hard context overflow (`len(token_ids) > max_ctx`, new `hard_overflow` flag) | still removed |
| `rollout_log_probs` length mismatch (zero-filled, ABORTED) | still removed |
| slow path (messages-only trajectory) | out of scope |
| `Sample.status` | stays TRUNCATED so `rollout/truncated_ratio_prefilter` remains an honest rate of clipped episodes |

Alternatives considered:

| Alternative | Why not |
| --- | --- |
| Loosen the any-step policy wholesale (train on clipped turns too) | trains on a turn with no proper terminal and rewards being cut off; token-level removal was a deliberate AGISlime r5 choice |
| Larger caps | already 32k/turn + 131k/episode since run 1; more KV and rollout time, and no help if clips were mid-trajectory as assumed |
| Cut thinking (`reasoning_effort`) | gym-side change (the gym's `ArenaSGLangLLM` dropped the kwarg); pursued later in AREnATasks for r5 |
| Accept the loss | ~26 trainable samples per step, `ess_ratio` 0.099 |
| Default on | the 27B snorkel configs are r5-parity and smoke-proven; opt-in leaves them untouched |

## Consequences

- Easier: a trajectory with one clipped final turn contributes its clean
  turns; group size and the ADR-0003 identity stamping are unchanged.
- Harder: `truncated_ratio_prefilter` no longer equals the removed fraction;
  read removals from the `Failed-sample distribution ... removed_total=` line
  (and, from ADR-0010, `Removal reasons:`).
- Strict argparse: the YAML key kills the trainer at startup on an image
  older than `miles-glm53-20260902c` (image b, 13:55 PT, predates the flag);
  bump the image tag together with the key.
- With ADR-0010 also on, a `timeout` stop no longer blocks this salvage (the
  guard moved from `not agent_degenerate` to `not agent_blocks_training` in
  32da04357); a timeout trajectory that ALSO has a clipped final turn is kept
  only through this flag.
- Naming trap: "r5-lineage default" in the help text and test docstring means
  the AGISlime snorkel run-5 policy (ADR-0007), not GLM r5.
- Tests: `tests/fast/plugins/arena/test_mask_clipped_final_turn.py` (5).
- **Outcome: correct but inert on this workload.** r2 (image c, identity
  `rl-glm53f-gbash-r2`, 2026-09-02 22:49 PT -> 09-03 00:36 PT) rollout 0
  removed 229/256 (`truncated_ratio` 0.8945) with zero `masked clipped final
  turn` lines. The truncation RCA (2026-09-03 ~00:40-01:30 PT, gym-pod trial
  artifacts) found 0/1033 generates at the 32k cap (max seen 15k): the
  removals were Harbor `AgentTimeoutError` -> `agent_stop_reason="timeout"`,
  which this flag rightly refuses. That case is ADR-0010. The key stays
  `true` in the GLM config (r2-r7) as a no-cost guard for real clips.
