# Study: Episode length runaway and reward collapse on GLM-5.3-Flash (CTRF runs r21-r26, auctioneer r30, r39, r42)

**Date:** 2026-09-22 (evidence window 2026-09-13 to 2026-09-28)
**Status:** Closed
**Question:** Why does the reward fall after a climb on every long GLM-5.3-Flash run, what co-moves with the fall, which runs recovered, and which change stopped it?
**Runs and data used:** `rl-glm53f-gbash-r11` (W&B `arena/rl-snorkel27/sbsp3drn`, binary-reward control), `r21` (`5tyk9fx2`), `r24` (`d8oak55n`), `r25` (`xlhqt1lv`), `r26` (`c46c9swg`, ledger `s3://arena-scratch-prod-bom-ap-south-1/guparpit/debug/rl-glm53f-gbash-r26/sample_summary/` rollouts 70-82); auctioneer `r30` (`rl-glm53f30-swrx2`, W&B `arena/rl-glm53f-auct-cap/lwkruqyz`, ledger `debug/rl-glm53f-auct-cap-r30/` rollouts 0-160), `r38` (`6h5t1jzm`, ledger rollouts 0-33), `r39` (`rl-glm53f39-25pdn`, `j6todjkb`, ledger 0-70), `r42` (`rl-glm53f42-5cthp`, `ob9qvkyg`, ledger 0-58); follow-ups `r44` (`82i3g7lv`, ledger 0-54) and `r46` (`4iu54zov`, ledger 0-56). Durable checkpoint `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/rl-glm53f-auct-cap-r30/hf/rollout_29/`. Reports under `s3://arena-scratch-prod-bom-ap-south-1/guparpit/2026/09/{22,25,26}/`.
**Code SHAs:** miles `8989cfb49` (`--arena-sample-summary-dir`, the per-sample ledger); miles `33697784f` (`shape_group_length_reward`, `--arena-length-reward-coef`, r27 only); miles `db3955b70` (trainer image `miles-glm53-r12-20260925a`, r39/r42); AREnATasks `4d3ecdf` (gym image `gym-glm53-adr69-20260925a`, r39/r42); miles plugin ADR-0015 commit "feat(arena): send the output cap, window, and sampling in the task message" (trainer image `miles-glm53-r14-20260927a`, r44+); AREnATasks ADR-0072 gym image `gym-glm53-adr72-20260927a` (r44+).
**Workflow ids:** `wf_c2a543f0-f9c` (generation length per call, segment, episode), `wf_795e8c03-107` (131K overflow trace on r39/r42).

## Method

Three sources, in order of trust.

1. **Per-sample ledgers** (`--arena-sample-summary-dir`, miles `8989cfb49`): one `rollout_<N>.jsonl` per trained rollout, keyed by the true rollout id, so a resume does not shift the index. The study reads them from the S3 prefixes above with a short script (scratch, deleted). Row filters: drop `mode == "dp_pad"`, drop `remove_sample == true`, drop `reward == null`. Episode = all rows with one `episode_index`; episode length = sum of `response_length` over its segments (ADR-0011: a Vulcan compaction opens a new segment, ADR-0063). Stop reason = `stop_reason` of the last segment; `null` means the agent stopped on its own. "Truncated share" = share of episodes with `status == "truncated"`. Reward = mean of `reward` over episodes; this equals W&B `rollout/episode_raw_reward`, not the row-weighted `rollout/raw_reward`.
2. **W&B** (host `https://mega.wandb.agi.amazon.dev`, key read from the r47 trainer pod, each key fetched alone and indexed by row order). Used for r11, r21, r24, r25, r26 rollouts 0-69, where no ledger exists, and for r30 optimizer metrics. Caveat: a resumed run replays rollouts into the same W&B run, so the row index exceeds the rollout id on r25 (five restarts) and on r26 after row 69.
3. **Journals** `wf_c2a543f0-f9c` and `wf_795e8c03-107` for per-call token counts and the overflow path in the gym code. Their `/tmp` scratch (`/tmp/ctx131k/`, `/tmp/genlen/`) is not in git, on tmpfs, present at 16:35Z 2026-09-28; the result text in the journal is the record.

Configuration comes from `training-runs/harbor-rl-glm53-flash/r<N>/miles-config.yaml`, `r<N>/BUILD.md`, and `RUNLOG.md`. Hardware: 40 nodes (32 SGLang engines x 8 GPU + 8 trainer nodes), 288 gym pods, prod-bom cluster.

### Shared configuration of every run in this study

| Knob | Value | Source |
| --- | --- | --- |
| `kl_coef`, `kl_loss_coef`, `entropy_coef` | 0.0, 0.0, 0.0 | `r26/miles-config.yaml:359-361`, `r30:404-406`, `r39:424-426`, `r42:432-434` |
| `lr` | 1.5e-6, constant | `r30/miles-config.yaml:413` |
| `global_batch_size` / `n_samples_per_prompt` | 256 / 8 (32 groups) | `r30:246`, `r30:232` |
| `rollout_max_response_len` / `rollout_max_context_len` | 32768 / 131072 | `r30:241`, `r30:258` |
| Effective input budget per call | 131072 - 32768 = 98304 tokens | `wf_795e8c03-107` (gym `sglang_rollout.py:703` sends `max_new_tokens = 32768` on every call) |
| Vulcan compaction trigger / cap | 0.8 x 98304 = 78643 tokens; `ARENA_COMPACTION_MAX` 2 (r30-r42), 5 (r25/r26) | `wf_795e8c03-107`; `experiment-list.md` item 8; RUNLOG 2026-09-15 |
| `arena_keep_timeout_trajectories`, `arena_keep_context_error_trajectories` | true (walled episodes still train) | `r30:460,466` |
| `arena_length_reward_coef` | 0 (only r27 ran 0.10) | `r30:542` |
| `sglang_disable_radix_cache` | true, except r42 and r44+ (false) | `r39:281`, `r42:289` |
| `ARENA_PARTIAL_REWARD` | `ctrf` on r21/r24/r25/r26; off on r30/r39/r42 | RUNLOG 2026-09-13; `r39/BUILD.md` |

## Results

### 1. CTRF family (snorkel-general-bash-harbor): the first observation, 2026-09-17

Datasets: r21 the 871-prompt manifest; r24, r25, r26 `/mnt/scratch-s3files-rw/guparpit/data/snorkel-general-bash-harbor/robust-20260913-r19signal/manifest-849.jsonl` (variant commit `f6d04286ce2a...`; `r26/miles-config.yaml:152`, RUNLOG 2026-09-13 r22 entry).

W&B rows (`rollout/raw_reward`, `rollout/response_lengths` = per-segment mean, `rollout/truncated`; r24 uses `rollout/episode_response_length/mean` and `rollout/truncated_ratio`).

| Run | Reward | Rows | Reward first -> last | Length first -> last (tokens) | Truncated first -> last | Source |
| --- | --- | --- | --- | --- | --- | --- |
| r11 (control) | binary | 53 | 0.463 -> 0.400; range 0.395-0.475 | 80.7k -> 85.8k (flat, 82-88k) | 0.33 -> 0.44 (flat) | W&B `sbsp3drn` |
| r21 | CTRF | 23 | 0.821 -> 0.672 | 28.6k -> 53.6k | 0.00 -> 0.24 | W&B `5tyk9fx2` |
| r24 | CTRF | 21 | 0.883 -> 0.647 (row 19) | episode 20.9k -> 47.9k | 0.00 -> 0.73 | W&B `d8oak55n` |
| r25 | CTRF, `truncated_turn_rule: mask` | 182 | 0.865 -> 0.634 (row 175); 0.59-0.67 over rows 155-175 | episode 22.2k -> 242.9k | 0.00 -> 0.54 | W&B `xlhqt1lv` (row index > rollout id, five restarts) |
| r26 | CTRF, `truncated_turn_rule: shift` | 83 | 0.894 -> 0.709 (row 82); low 0.582 at row 75 | segment 31.4k -> 71.7k; episode 108k (L70) -> 266k (L82) | 0.00 -> 0.18 (W&B); 0.03 -> 0.19 status share (ledger) | W&B `c46c9swg`; ledger r26 |

Ledger r26 rollouts 70-82 (fresh pipeline after the `dcskp` resume at iter 69, RUNLOG 2026-09-17): reward 0.830, 0.720, 0.582, 0.495, 0.597, 0.712, then 0.60-0.75 through rollout 82; episode length 108k -> 272k (rollout 73) -> 212k-266k; segments per episode 1.74 -> 3.55; `timeout` share 3% -> 16-20%, `context_error` share 0% -> 2-10%. The first three post-resume rollouts are a short-biased sample (a rollout closes after about 2 h and long groups land later), so the 0.830 is not a policy gain.

Findings recorded on 2026-09-17 (memory note `glm53-length-explosion-ctrf`; the analysis scripts under `/tmp/glm53/` are not in git, on tmpfs, present at 16:35Z 2026-09-28, and were not re-run for this record, so every number in this paragraph is **not re-verified** except where RUNLOG repeats it):

- Regression of per-rollout `raw_reward` on rollout id with and without the truncated share: r26 rows 0-70 raw slope -0.00250 per rollout, controlled slope +0.00019, truncation coefficient -0.548; r25 rows 20-71 raw -0.00217, controlled +0.00009, coefficient -0.476. The whole decline is the truncated share times a 0.4 reward gap (completed 0.65-0.85 vs truncated 0.28-0.38). Task competence at a fixed truncated share is flat over about 140 optimizer steps.
- Simpson's paradox: pooled across tasks, successes are shorter than failures (gap -35k to -68k tokens). Within one group, the longer attempt scores higher (completed-only within-group corr(advantage, length) +0.18 to +0.34; longer half of a group 0.676 vs 0.567). GRPO normalizes per group, so it trains on the within-group sign only.
- The reward ignores tokens: 229 of 288 groups (rollouts 70-76) hold 2 or more episodes tied at the group best reward, median length spread 2.09x, max 14.9x; mean response length 31k -> 77k with corr(length, rollout id) +0.903 (RUNLOG 2026-09-18, r27 entry, repeats these three numbers). 677 of 766 dynamic-sampling drops are `zero_std_1.0` groups.
- A blanket length penalty is wrong: among completed episodes the reward plateaus at 0.70-0.75 from 100k tokens up (bins 100-150k 0.748, >500k 0.727). The cost is the wall, not the tokens.

### 2. Auctioneer r30: two full cycles, no intervention

Ledger `debug/rl-glm53f-auct-cap-r30/sample_summary/`, 161 rollouts, 256 episodes each, `0/32` zero-std groups in every rollout. Dataset `lakefs://arena-inspect/main/internal/auctioneer/caponly/20260916-v1/manifest.jsonl` (1,047 tasks; memory `auctioneer-caponly-r30-and-limits`). Agent timeout multiplier 1 = 2 h (`r30/BUILD.md`). Workflow `Failed 2026-09-25T15:19:59Z Stopped with strategy 'Stop'` (kubectl, read-only); the retire reason is not in RUNLOG.

| Rollout | Reward (episode mean) | Episode length mean / p90 (tokens) | Segments per episode | Truncated share | Stop mix (`null` = agent stopped) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.335 | 41.5k / 48.1k | 1.00 | 0.00 | null 1.00 |
| 29 | 0.583 | 54.0k / 67.6k | 1.04 | 0.12 | null 0.88, timeout 0.12 |
| 38 | 0.655 (W&B row avg 0.637, first peak) | 81.7k / 106.5k | 1.63 | 0.43 | null 0.57, timeout 0.43 |
| 47 | 0.339 | 141.7k / 173.8k | 2.27 | 0.85 | timeout 0.84, null 0.15 |
| 54 | 0.084 | 172.3k / 203.2k | 2.64 | 0.91 | timeout 0.90, null 0.09, context_error 0.02 |
| 57 | 0.060 (W&B 0.050, trough) | not tabulated | | | |
| 64 | 0.079 | 229.2k / 276.1k | 2.86 | 0.82 | context_error 0.48, timeout 0.33, null 0.19 |
| 66 | 0.435 | 230.8k / 278.3k | 2.86 | 0.66 | context_error 0.56, null 0.34, timeout 0.10 |
| 79 | 0.585 | 141.2k / 221.6k | 2.22 | 0.09 | null 0.91, timeout 0.05, context_error 0.04 |
| 90 | 0.653 | 85.6k / 124.5k | 1.53 | 0.00 | null 1.00 |
| 102 / 105 | 0.739 (ledger best) / W&B row avg 0.708 (W&B max) | 102: not tabulated | | | |
| 120 | 0.700 | 180.1k / 237.6k | 2.67 | 0.55 | timeout 0.52, null 0.45, context_error 0.03 |
| 130 | 0.390 | 256.9k / 292.5k | 2.97 | 0.89 | context_error 0.55, timeout 0.34, null 0.11 |
| 140 | 0.368 | 278.4k / 301.4k | 2.98 | 0.95 | context_error 0.89, null 0.07, timeout 0.04 |
| 150 | 0.558 | 210.3k / 283.6k | 2.70 | 0.48 | null 0.52, timeout 0.27, context_error 0.22 |
| 160 | 0.715 | 91.1k / 128.6k | 1.58 | 0.00 | null 1.00 |

Read: reward and episode length move in opposite directions with a lag. Length climbs 42k -> 229k over rollouts 0-64 while reward peaks at 38 and bottoms at 57-64; length falls back to 86k by rollout 90 and reward returns to 0.65. The same cycle repeats: length 86k (90) -> 278k (140), reward 0.65 -> 0.37, then length 91k (160) and reward 0.72. Period about 65-70 rollouts, not the "about 30" written on 2026-09-23. No config, image, or resume changed during the run (W&B is one run id; no `episode_response_length` collapse marks a restart).

Optimizer side (W&B `lwkruqyz`, row = rollout): `train/grad_norm` 0.136 (0) -> 0.093 (30) -> 0.032 (60) -> 0.069 (90) -> 0.029 (130); `train/tis_clipfrac` peaks 0.0093 at row 30 and sits at 0.0003-0.0008 at the floor. No update spike precedes either fall. Rows 145-158 show a `grad_norm` burst (0.204, 0.72, 2.896, 2.163, 1.0, 0.501, ..., 1.292) with `tis_clipfrac` 0.045 at row 150, during the second recovery. It was not analyzed while the run lived.

### 3. Auctioneer r39 and r42: collapse without recovery

Dataset `lakefs://arena-inspect/dev/internal/auctioneer/caponly/caponly-1034` (1,034 tasks, rows pin commit `e5ef91b0`; same reward function as 20260916-v1, `r39/BUILD.md`). r42 = r39 + radix cache on + overlap schedule on + agent timeout multiplier 2 (`r42/BUILD.md`). Both retired 2026-09-27 (`shutdown: Stop`, r42 05:02:08Z at tracker 49, r39 05:05:52Z at tracker 59; RUNLOG 2026-09-27 r44 entry).

| Rollout | r39 reward | r39 length mean (k) | r39 stop mix | r42 reward | r42 length mean (k) | r42 stop mix |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.385 | 45.1 | null 1.00 | 0.385 | 44.8 | null 1.00 |
| 20 | 0.392 | 54.8 | null 0.91, timeout 0.09 | 0.334 | 49.2 | null 1.00 |
| 30 | 0.580 | 60.4 | null 0.73, timeout 0.27 | 0.545 | 61.8 | null 1.00 |
| 35 / 36 | 0.488 | 83.5 | timeout 0.70 | 0.740 at 36 (best) | 87.0 at 35 | null 1.00 |
| 39 / 40 | 0.444 (39); 0.616 at 40 (best) | 104.8 | timeout 0.82 | 0.670 (40) | 151.1 | null 0.80, context_error 0.20 |
| 45 | 0.322 | 146.2 | timeout 0.89 | 0.399 | 252.8 | context_error 0.84 |
| 49 / 50 | 0.100 (50) | 211.3 | timeout 0.94, context_error 0.04 | 0.076 (49) | 273.8 | context_error 0.80 |
| 55 | 0.238 | 228.8 | context_error 0.52, timeout 0.39 | -0.036 (worst) | 259.5 | context_error 0.70 |
| 58 | | | | 0.023 (last) | 249.3 | context_error 0.73 |
| 60 | 0.081 | 234.4 | context_error 0.875 | | | |
| 64 | 0.011 (worst) | 232.0 | context_error 0.85 | | | |
| 66 | 0.298 | 225.8 | context_error 0.83 | | | |
| 70 | 0.136 (last) | 206.5 | context_error 0.59, null 0.41 | | | |

Segments per episode reach 2.93-2.99 at the floor on both runs: the two-compaction cap is hit on almost every episode (`wf_c2a543f0-f9c`: r39 rollouts 50-66 mean 2.95; r42 rollouts 45-53 mean 2.96). r39 `status == truncated` share is 0.00 in every ledger row because the adr69 gym records the wall in `stop_reason` only.

Per-call view (`wf_c2a543f0-f9c`, verified pass): healthy auctioneer rollouts 0-29 pooled over r30/r39/r42 have per-segment response P50 49,910 / P90 63,280 tokens (n 23,885) and about 130 generated tokens per model call at P50; 0.06% (r42) to 0.50% (r39) of calls hit the 32,768 cap. Collapsed windows: r39 rollouts 50-66 per-segment P50 81,293 / P90 101,184 and episode P50 231,024, mean reward 0.184; r42 rollouts 45-53 per-segment P50 91,896 / P90 107,136, episode P50 274,948, mean reward 0.135; 15-19% of calls hit the cap. The task text "responses 52K -> 112K" is not in any source as written; the nearest primary numbers are r39 healthy per-segment P50 52,622 (rollouts 0-39) and collapsed per-segment total_length P90 109,372 (r39) / 114,202 (r42).

Overflow path (`wf_795e8c03-107`, verified pass): no request carries more than 131,072 input tokens. The gym sends `max_new_tokens = 32768` on every call (`amzn_arena_harbor/sglang_rollout.py:703`), so SGLang rejects any input above 98,304 with HTTP 400. One clipped 32,768-token reply moves a history from below the 78,643 compaction trigger past 98,304 in one turn, so the compaction summary call itself gets the 400 and the agent falls back to a structural compaction with no handoff note (38-48% of compactions, RUNLOG 2026-09-27 ADR-0015 entry). After two compactions the next 400 ends the episode as `model_length`, recorded as `agent_stop_reason = context_error`. No retry loop exists; engine and gym 400 counts agree. The journal trace counted r42 512 vs 513 and r39 545 vs 567 over its 90 min window; the re-verified engine count from the saved engine logs is r39 429 and r42 453 (`/tmp/ctx131k/verify/eng90_*.txt`, a different window and method; see `../context-overflow-131k-and-clamp/STUDY.md`). In the last 90 min before the trace, 81% of r42 and 28% of r39 episodes ended at the context limit. Every such episode still trains with the verifier reward.

r38 (20260916-v1, adr68 gym, 34 rollouts, ledger): reward 0.316 -> 0.647 (30) -> 0.533 (33); length 44k -> 78k; `timeout` share 0.76 at rollout 33. It shows the same approach phase and stopped before a floor; its retire is not in RUNLOG.

### 4. What recovered and what did not

| Run | Family | First wall | Trough (rollout, reward) | Recovery | Observation window past trough |
| --- | --- | --- | --- | --- | --- |
| r26 | CTRF | timeout, then context_error | plateau, no floor: 0.58-0.75 at 210-270k tokens | none needed; stayed on a truncation-limited plateau | rollouts 70-82 |
| r25 | CTRF | timeout | drift to 0.59-0.67, truncation 0.5 (W&B rows 155-175) | none | run continued; not re-tabulated past row 181 |
| r30 | auctioneer | timeout (2 h), then context_error | 57, 0.050-0.060; second trough 134-140, 0.07-0.39 | yes, twice, unaided: 0.435 at 66, 0.585 at 79, 0.739 at 102; 0.715 at 160 | 103 rollouts |
| r39 | auctioneer | timeout, then context_error | 64, 0.011 | none observed (0.298 at 66, 0.136 at 70) | 6 rollouts past 64, 20 past the first floor at 50 |
| r42 | auctioneer | context_error only (4 h timeout never fires) | 55, -0.036 | none observed (0.023 at 58) | 3 rollouts past 55, 9 past 49 |

The r30 recovery is not a config effect: reward and length turned around inside one W&B run with no resume. r39 and r42 were retired inside a window shorter than the r30 recovery lag (9 rollouts from trough to 0.435). The data cannot show whether a later turn was possible for them.

### 5. Follow-up runs on the fixed cap (r44, r46)

r44 = r42 base + trainer `miles-glm53-r14-20260927a` (plugin ADR-0015) + gym `gym-glm53-adr72-20260927a` (AREnATasks ADR-0072) + `rollout_max_response_len: 16384` + `agent-kwargs {}` (Vulcan default 4 compactions) + `agent-timeout-multiplier 2` (`r44/workflow.yaml:35-40`, RUNLOG 2026-09-27). r46 = r44 + `calculate_per_token_loss: true` (`r46/miles-config.yaml:496`). Ledger through 2026-09-28 15:58Z.

| Run | Rollouts | Reward first -> best | Episode length mean range | Segments per episode | Truncated share | context_error share |
| --- | --- | --- | --- | --- | --- | --- |
| r44 | 0-54 | 0.312 -> 0.598 (44); 0.49 at 54 | 44k-66k | 1.00-1.05 | 0.00 | 0.00 |
| r46 | 0-56 | 0.363 -> 0.789 (50); 0.66 at 56 | 36k-50k | 1.00-1.05 | 0.00 | 0.00 |

Fifty-five rollouts is past the point where r39 (rollout 35), r42 (40), and r30 (35) had already crossed 80k tokens. Neither follow-up shows the climb. r46 lengths trend down (46k -> 36-43k). RUNLOG 2026-09-27 r44 entry: 0 "Requested token count exceeds" lines in the trainer log, all compactions `1/4`, 0 lost handoff notes.

## Verdict

Every long GLM-5.3-Flash run in this series climbed, then lost reward when episode length ran into a wall. The wall was the 2 h agent timeout first (r30, r39; `timeout` share 0.84-0.94 at rollouts 45-54) and the context window second (`context_error` 0.5-0.9 once episodes reached about 230k response tokens over 3 segments). The reward loss tracks the walled share: on r25/r26 the truncation term absorbs the whole decline, and on r30/r39/r42 reward troughs coincide with 82-95% of episodes at a wall. The optimizer did not misbehave (`grad_norm` fell, `tis_clipfrac` fell, no zero-variance groups). CTRF partial credit was the first suspect, but r30 (continuous `oracle_fraction`, partial reward off) reproduced the pattern, so the driver is the within-group gradient that favors the longer attempt plus a reward that is blind to tokens, with no KL, entropy, or length term to oppose it. r30 recovered twice on its own with a period near 65 rollouts; r39 and r42 were retired before any turn. The change that removed the wall in practice is the 16,384 output cap with the clamp to the room left in the window (ADR-0072, plugin ADR-0015): r44 and r46 hold 36k-66k tokens with zero walled episodes through 55 rollouts. The data do not say whether the policy stopped its length growth or whether the larger effective window (114,688 tokens, four compactions) only moved the wall out of reach; a longer r44/r46 window settles that.

## Caveats and open items

- **The r30 extrinsic report (2026-09-22) conflicts with the ledger.** It states "total episode length grows from ~45k to ~84k, never near the 131k context limit" and a floor mix of 59-89% completed. It read W&B `rollout/total_lengths` (per-segment mean, not per-episode) and a time-bucketed 30-pod log sample. The ledger for the same rollouts shows episode means of 172k-234k and `timeout + context_error` shares of 0.82-0.91. Treat the ledger as primary; the report's "completed-but-wrong floor" reading is not supported.
- **Not re-verified** (source is a memory note only; the scripts under `/tmp/glm53/` are not in git, on tmpfs, present at 16:35Z 2026-09-28, and were not re-run): the 2026-09-17 regression coefficients, the Simpson's-paradox numbers, the length-bin table, the "2-4.9M cumulative tokens and up to 96 tool-call errors per episode" on r30 (gym pods are gone), and the r24 reward "0.38 by rollout 20" (W&B shows 0.647 at row 19 and no row 20 value).
- **Period "about 30 rollouts"** in the 2026-09-23 memory note is wrong; the ledger shows about 65-70.
- **r30 `grad_norm` burst at rows 145-158** (up to 2.896, `tis_clipfrac` 0.045) during the second recovery is unexplained.
- **r44/r46 ledgers record `stop_reason = timeout` with `status = completed`** on 50-97% of episodes at 36k-66k tokens (r44 rollouts 30 and 50: 503 of 518 rows). Under the ADR-0013 gym the status no longer flips, and the meaning of that `timeout` for a short auctioneer episode is not verified here.
- **No transcripts and no task id** in the ledger, so what the policy did inside the long episodes is unknown for every run before r47.
- **r25 and r26 W&B row index is not the rollout id** after a resume; the tables above cite W&B rows for r25 and rollout ids only where the ledger exists.
- **r39/r42 observation windows** after the trough are 3-20 rollouts, shorter than the r30 recovery lag; "did not recover" means "not within the window".

## Actions taken

- 2026-09-17: per-sample ledger `--arena-sample-summary-dir` (miles `8989cfb49`, image `miles-glm53-r7-20260917a`), armed on r26 and on every run since (RUNLOG 2026-09-17).
- 2026-09-18: r27 group-relative length reward, `arena_length_reward_coef 0.10` (miles `33697784f`, image `miles-glm53-r8-20260917a`; RUNLOG 2026-09-18). r28 and later set the coefficient back to 0 (`r30/miles-config.yaml:542`).
- 2026-09-22: r30 extrinsic report published to `s3://arena-scratch-prod-bom-ap-south-1/guparpit/2026/09/22/auctioneer-cap-r30-extrinsic-report/report.html` (28,108 B, verified on S3); browser URL `https://browser.bom.prod.arena.agif.amazon.dev/scratch/guparpit/2026/09/22/auctioneer-cap-r30-extrinsic-report/report.html`.
- 2026-09-22/23: r30 peak checkpoint `hf/rollout_29` (training reward 0.579; saved checkpoints rollout_9/19/29/39/49/59 scored 0.372/0.421/0.579/0.504/0.093/0.202 per the report) copied to `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/rl-glm53f-auct-cap-r30/hf/rollout_29/` (prefix verified).
- 2026-09-25/26: step-40 report `guparpit/2026/09/25/auctioneer-cap-r30-steps0-40-report/report.html` (96,696 B, verified) and the concise brief `guparpit/2026/09/26/auctioneer-r30-vending-brief-final/report.html` (19,413 B, verified; rollout_29 vs base on Vending-Bench: 360-day mean profit +$71,587 vs -$6,705). `TRAINING_REPORT.html` committed to `lakefs://arena-inspect/dev/internal/auctioneer/caponly/caponly-1034/` at commit `e7f00c95215e` (memory note; not re-verified).
- 2026-09-27: `rollout_max_response_len` 32768 -> 16384, gym takes the cap from the task message and clamps to the room left in the window (AREnATasks ADR-0072, plugin ADR-0015; reverses `experiment-list.md` R2 and R3). r39 and r42 retired; r44 launched (RUNLOG 2026-09-27). r46 adds the token-level loss average (RUNLOG 2026-09-27 r46 entry).
- Rejected on the evidence: a blanket length penalty (completed-episode reward plateaus with length), DAPO dynamic sampling (already on), a higher learning rate (1.5e-6 is already above the 1e-6 upstream GLM-5.2 precedent), a return to `use_kl_loss` (the ref forward cost 31% of a step and it runs without R3 replay). Memory notes `glm53-length-explosion-ctrf`, `glm53-ppo-kl-is-numerical-noise`.

## Sources

- `training-runs/harbor-rl-glm53-flash/RUNLOG.md`: 2026-09-13 (r20/r21 CTRF launch, first reward table), 2026-09-14 (r23/r24, completion-order bias), 2026-09-15 (r25/r26 launch), 2026-09-17 (r26 probe, ledger), 2026-09-18 (r27 cause and length reward), 2026-09-27 (ADR-0015 entry, r39/r42 400 counts and retire, r44 launch, r46 launch).
- `training-runs/harbor-rl-glm53-flash/experiment-list.md`: items 8, 9, 10; rejected R2, R3 and their 2026-09-27 reversal.
- Run records: `r21/`, `r24/`, `r25/`, `r26/`, `r30/`, `r39/`, `r42/`, `r44/`, `r46/` (`miles-config.yaml`, `BUILD.md`, `workflow.yaml`).
- Ledgers: `s3://arena-scratch-prod-bom-ap-south-1/guparpit/debug/{rl-glm53f-gbash-r26,rl-glm53f-auct-cap-r30,rl-glm53f-auct-cap-r38,rl-glm53f-auct-cap-r39,rl-glm53f-auct-cap-r42,rl-glm53f-auct-cap-r44,rl-glm53f-auct-cap-r46}/sample_summary/`.
- W&B (`https://mega.wandb.agi.amazon.dev`): `arena/rl-snorkel27/{sbsp3drn,5tyk9fx2,d8oak55n,xlhqt1lv,c46c9swg}`, `arena/rl-glm53f-auct-cap/{lwkruqyz,6h5t1jzm,j6todjkb,ob9qvkyg,82i3g7lv,4iu54zov}`.
- Journals: `wf_c2a543f0-f9c`, `wf_795e8c03-107`.
- Reports: `s3://arena-scratch-prod-bom-ap-south-1/guparpit/2026/09/22/auctioneer-cap-r30-extrinsic-report/report.html`, `.../2026/09/25/auctioneer-cap-r30-steps0-40-report/report.html`, `.../2026/09/26/auctioneer-r30-vending-brief-final/report.html`.
- ADRs: AREnATasks ADR-0063 (Vulcan compaction and rollout segments), ADR-0065 (`rollout/stop/<reason>` metrics), ADR-0069 (training reward is the Harbor trial reward), ADR-0072 (gym takes the cap, window, sampling from the task message); miles plugin ADR-0011 (segments share one `rollout_id`), ADR-0013 (gym-owned mask, every verified episode trains), ADR-0015 (task message carries the cap and window).
- Memory notes (facts with dates; treated as notes): `glm53-length-explosion-ctrf`, `glm53-r30-collapse-r35-reports-2026-09-22`, `auctioneer-caponly-r30-and-limits`, `r39-auctioneer-caponly-1034-2026-09-25`, `r42-radix-overlap-timeout-2026-09-25`, `sglang-speed-vs-inference-hosting-2026-09-25`, `report-truncated-turn-shifted-only`, `wandb-resumed-run-restart-discontinuity`, `glm53-ppo-kl-is-numerical-noise`.
