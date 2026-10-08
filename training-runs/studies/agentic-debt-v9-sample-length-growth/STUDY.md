# Study: why `rollout/total_lengths` grows on final-v9 and not on v23

**Date:** 2026-10-08
**Status:** Closed (cause located to the turn count; the recipe attribution needs an A/B)
**Runs:** `acuadron-agentic-debt-final-v9-dx8b7` (W&B `i8ufpnjc`, steps 0-40) and `guparpit-agentic-debt-v23-sw9v6` (W&B `3tirhnqh`, steps 0-64).
**Data:** W&B history; per-sample ledgers `sample_summary/rollout_N.jsonl` (both runs); Harbor `result.json` per trial (v9 22,495 files, v23 23,994, synced 2026-10-08 03:40 UTC); 400 `segment-01` `trajectory.json` files (80 at version 1 and 120 at versions 18-30 for v9; 80 and 120 at versions 24-34 for v23). Scripts: `/tmp/glm53f/hj/{ctx,trajstats,pick}.py` (not in git).

## What the metric is

`rollout/total_lengths` is the mean token count of one training sample: one compaction segment of one chain step (prompt about 1.5-2k tokens, then everything the agent wrote and read in that segment). A segment ends at the compaction ceiling (about 100k tokens; p90 of the ledger sits at 95k) or at the end of the chain step.

## Results

Per-sample length (ledger, k tokens), all samples and the first chain step of each episode (`segment_k = 0`, 512 or 256 samples per step, same prompt type every step):

| step | v9 all | v9 first step | v23 all | v23 first step |
|---|---|---|---|---|
| 0 | 20.6 | 26.4 | 16.5 | 18.3 |
| 8 | 31.4 | 34.3 | 25.6 | 31.8 |
| 20 | 30.0 | 34.7 | 32.7 | 39.3 |
| 28 | 39.8 | 47.7 | 27.1 | 36.9 |
| 36 | 47.3 | 55.6 | 31.7 | 39.0 |
| 38-40 | 52.4-56.0 | 58.2 | 30.2 | — |

The growth is the same inside every segment-position bucket, so it is not the batch mix (v9 has no staleness cap and trains more late segments; that adds nothing here).

First chain step by policy version (Harbor `result.json`; v9 110 tasks at version 1, 26-29 at versions 13-40 as of the sync):

| versions | v9 calls | v9 output tokens | v9 compactions | v9 tool errors | v9 pass | v23 calls | v23 output tokens | v23 compactions | v23 pass |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 29.9 | 27.1k | 0.13 | 4.0 | 0.79 | 29.9 | 27.6k | 0.13 | 0.80 |
| 9-12 | 31.0 | 26.8k | 0.15 | 4.2 | 0.78 | 32.2 | 30.6k | 0.17 | 0.79 |
| 18-22 | 49.5 | 51.2k | 0.43 | 6.1 | 0.79 | 33.9 | 30.5k | 0.17 | 0.81 |
| 28-33 | 54.6 | 56.2k | 0.49 | 10.1 | 0.80 | 32.9 | 28.6k | 0.16 | 0.81 |
| 34-40 | 57.8 | 64.1k | 0.55 | 8.6 | 0.81 | 40.0 | 39.5k | 0.30 | 0.79 |

Output tokens per call: v9 906 -> 1,030-1,108 (+15%), v23 923 -> 870-1,134. Tool output per call (trajectory chars): v9 1,455 -> 1,345, v23 1,600 -> 1,478. So the growth is the number of tool calls per chain step, not the size of a call or of a tool result. Both passing and failing first steps grew (v9 pass 27.6 -> 49.7 calls, fail 38.6 -> 76.9).

Trajectory content (400 `segment-01` files, calls per trajectory): v9 heredoc file writes 13.1 -> 20.1, `echo` 0 -> 5.0, `sed` 2.9 -> 4.4, `grep` 2.1 -> 3.5, `pytest` 4.0 -> 5.1, `sleep` 0.7 -> 2.0, `python` 1.0 -> 1.8; v23 heredoc 10.7 -> 11.2, `echo` 0 -> 1.7, `pytest` 3.2 -> 4.6. The `echo attempt-N` and `sleep 280; tail log` loops sit in two tasks (`realdiff_Quantco_slim-trees_s0`, `realdiff_JudeWells_chainsaw_s0`) in both runs, 2-4x more often in v9; excluding them v9 still grew x1.4 in calls and x1.5 in completion tokens against v23's x1.2.

## Why the policy takes more turns

- The within-group signal rewards it on v9's tasks. On the first chain step at version 1, Spearman(calls, step reward) inside a group is +0.085 on v9 (174 mixed groups, about 3 standard errors) and +0.000 on v23 (157 groups). v9 trains on the 110 low-mean, high-variance chains, where the attempt that iterates longer passes more tests; GRPO trains on the within-group sign only (the auctioneer study records the same mechanism). Pass rate did not move (0.78-0.81), so the learned extra turns earn nothing at the population level.
- The recipe differences all point the same way, and the logs cannot separate them: v9 uses the per-sample loss mean and standard-deviation-normalised advantages, v23 `calculate_per_token_loss: true` and `disable_grpo_std_normalization: true` (the r44/r46 study saw the token-level average keep turns short); v9 trains on data 7-8 versions old (uncapped) against v23's about 4; v9 has half the updates per sample (batch 512 against 256). The same recipe at lr 1e-5 (final-v6, final-v7, engine-ab-v2) grew tokens per call 7x in 6 steps; at lr 1.5e-6 the drift is the turn count, 2x in 40 steps.
- Ruled out: speculation (`speculative_accept_threshold_single/acc = 1.0`, exact sampling; the log-prob gap is equal in both runs), composition, tool-output size.

## Cost

Tokens per step 34M (step 0) -> 110M (step 9) -> 233M (step 40); `actor_train` 375 -> 1,300 s; the serial routing load scales with bytes (1,440 B per token raw): 1,200 -> 2,500 s; step time 20 -> 46 min at steps 39-40. With r24's parallel loads the load falls to about 6 min at this size, and the trainer alone is then 22 min per step.

## Options (recipe changes, not taken here)

`calculate_per_token_loss: true` and `disable_grpo_std_normalization: true` (v23's settings); `max_weight_staleness: 8`; `arena_length_reward_coef` above 0 (the group-relative token-efficiency term, coefficient below the reward granularity); a per-chain-step turn budget in the gym.
