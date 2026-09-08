# GLM-5.3-Flash RL — experiment backlog

Context-overflow experiments for the snorkel-general-bash-harbor run family.
Each row lists status, effort, expected benefit, and the stability risk.
Read `RUNLOG.md` for per-run history and `README.md` for the task.

## The problem r8 exposed

r8 reward is flat near 0.40. Per-episode analysis of 8,744 episodes shows the
episodes die at about 98 k total tokens, not the 131 k window. The fixed
`ARENA_MAX_TOKENS` reservation (32768) is subtracted from every generate, so
the real input ceiling is `131072 - 32768 = 98304`. Key findings:

- PASS episodes are SHORTER than FAIL episodes (61 k vs 77 k policy tokens).
- 42% of passes end in `context_error` AFTER the work finished.
- 58% of trained tokens are `context_error` failures, mean advantage -0.67.
- Only 0.2% of turns hit the per-turn 32768 cap. This is not the #2588
  degenerate-loop signature.

The conclusion: this is a termination problem, not a reward problem. The model
keeps starting new work past the point where it can finish inside the window.

## In flight

### 1. Budget nudge (r9) — SHIPPED

- **Change.** `ArenaTerminus2` injects a "wrap up now" user turn once the
  running prompt gets within `ARENA_CONTEXT_NUDGE_TOKENS` (default 16384) of the
  effective ceiling (98304). Gym-side only. No loss-mask or advantage change.
- **Effort.** Low. One agent method, one env var, one gym image.
- **Expected benefit.** Medium. Converts a fraction of the 42% "finished then
  overflowed" passes into clean completions, and gives failing episodes a chance
  to submit partial work before the wall.
- **Stability risk.** Low. The nudge is a normal user turn. It does not
  re-render history, so it does not corrupt the cumulative token state the way
  summarization does. r9 keeps the r8 loss and advantage path unchanged, so r9
  isolates the nudge as a single variable.
- **Note.** r9 reuses the r8 trainer image `miles-glm53-20260905a`, which
  predates the committed NATS results-stream bound (miles `arpit-glm-53`
  @ 777c3a97). Apply the live `js.update_stream` bound to r9's fresh NATS after
  launch, the same guard that protects r8. Bake the trainer fix into the next
  run image.

### 7. Vulcan compaction + every-segment training (r12) — PREPARED

- **Change.** Gym: `--agent vulcan` with token-aware context compaction
  (AREnATasks ADR-0048, image `gym-glm53-vulcan-20260908a`). The policy writes
  a handoff note before each compaction; the compacted history is text only
  (`[system, instruction, bridge]`); each compaction starts a new rollout
  segment and the gym ships one step per segment. `ARENA_COMPACTION_MAX=2`
  keeps one NATS result message under the 8 MiB `max_payload`. Trainer:
  `arena_train_segments: all` (miles ADR-0011, image `miles-glm53-20260908a`)
  trains every segment as its own Sample with a shared `rollout_id` and the
  episode reward; the trainer pads the row count to the DP alignment with
  zero-loss sibling rows. Every other knob is r11. Assets: `r12/`.
- **Effort.** Done. Both images are built and pushed; `r12/BUILD.md` lists the
  launch commands in order.
- **Expected benefit.** High. An episode continues past the 98 k wall instead
  of dying at it, and every segment of the work carries gradient.
- **Stability risk.** Medium. New agent loop and new sample shape on the arena
  NATS path. The row count per step is data dependent. Watch
  `rollout/num_training_samples`, `rollout/episode_raw_reward`,
  `train_rollout_logprob_abs_diff` and the gym's 6 MiB payload warning. The
  loss and advantage path is unchanged (no KL, no entropy term), so a shift
  from the new sample shape has no gauge; candidate 5 stays open.

## Remaining candidates

### 2. Group-relative length penalty on passes (#1573 pattern)

- **Change.** Inside a GRPO group, a shorter pass beats a longer pass. Penalty
  is group-relative and applies only among the group's own samples.
- **Effort.** Medium. Reward post-process hook plus a length signal per sample.
- **Expected benefit.** Medium. Teaches the policy to finish sooner, which
  directly attacks the 98 k wall.
- **Stability risk.** Medium. r8 runs `kl_coef 0`, `entropy_coef 0`. A length
  term with no KL anchor can drift toward terse, low-quality passes. Pair it
  with a small KL or entropy gauge to keep the shift visible.

### 3. Length shaping on failures only

- **Change.** Apply the group-relative length term to failing samples only, so
  a failure that ran out of context is pushed below a failure that stopped
  cleanly. Passes are untouched.
- **Effort.** Medium.
- **Expected benefit.** Low. Simulation shows GRPO per-group std normalization
  absorbs the term. For `context_error` samples the advantage delta was only
  about -0.02.
- **Stability risk.** Low. The change is small because the effect is small.

### 4. Clean-fail reward floor

- **Change.** Give a clean failure (stopped, marked done, wrong) a small floor
  reward (about 0.1) above a truncated failure (0.0).
- **Effort.** Low. One reward post-process branch.
- **Expected benefit.** Low to medium. Separates "tried and stopped" from "ran
  out of room".
- **Stability risk.** HIGH. All-fail groups trap: simulation shows a +2.65
  advantage blowup when every sample in a group takes the floor and the std
  collapses. Do not ship without an all-fail-group guard.

### 5. Stability gauges (add back a small KL or entropy term)

- **Change.** Add a small `kl_loss_coef` and lower `eps_clip` from 0.4.
- **Effort.** Low. Config only.
- **Expected benefit.** Indirect. It does not fix the wall. It makes a policy
  shift visible before it becomes a collapse.
- **Stability risk.** Low. This is the safety instrument, not the experiment.
- **Note.** Kept OUT of r9 on purpose so r9 isolates the nudge. Add it to the
  first run that changes the loss or advantage path (candidates 2, 3, 4).

### 6. Harness compaction (upstream session server v2 + Terminus 2)

- **Change.** Adopt the upstream Terminus 2 compaction path (#2741, #2124-2128)
  driven by the session server v2 trajectory tree, instead of the arena
  fixed-cap rollout that disables summarization.
- **Effort.** HIGH. It reworks the rollout token accounting and the arena
  splice logic. It touches the trainer, the gym, and the wire format.
- **Expected benefit.** High. Compaction is the real long-horizon fix. It lets
  an episode continue past the window instead of dying at it.
- **Stability risk.** Medium to high, and unknown for the arena splice path.
  Summarization re-renders history, which is exactly what the arena backend
  disables today because it corrupts the cumulative token state. Needs a design
  and an ADR.
- **Note.** r12 ships the Vulcan variant (AREnATasks ADR-0048, miles ADR-0011),
  not the session-server-v2 path; see item 7.

## Rejected

### R1. Mask truncated failures out of the loss

- **Idea.** `remove_sample=True` only when `context_error` and reward 0. Keep
  passes that overflowed. Keep the samples in advantage normalization.
- **Why rejected.** The remaining loss is mostly positive advantage. With r8's
  zero KL and zero entropy coefficients this risks entropy collapse. The user
  rejected it as unstable.

### R2. Fix the 98 k wall directly (raise the effective ceiling)

- **Idea.** Lower `ARENA_MAX_TOKENS` reservation or clamp `max_new_tokens` to
  remaining context, so the input ceiling rises toward 131 k.
- **Why rejected.** The 98 k wall is an artifact the run created. The overflowing
  generations fail anyway. The user assessed the benefit as marginal: the
  episodes just get stuck near 120 k instead of 98 k.

### R3. Reduce the per-turn cap (32768 → 16384 or 8192)

- **Idea.** Smaller per-turn cap leaves more turns inside the window.
- **Why rejected.** Only 0.2% of turns hit the cap, so the cap is not the
  binding constraint. The user assessed the benefit as marginal.
