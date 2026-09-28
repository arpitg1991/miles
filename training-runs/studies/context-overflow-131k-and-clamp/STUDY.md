# Study: The 131072 context overflow on r39 and r42, the room-left clamp, and the 16384 cap

**Date:** 2026-09-27
**Status:** Closed
**Question:** Why did the r39 and r42 gyms send SGLang requests over the 131072-token window, was a retry loop behind the repeated 400s, and which clamp and cap stop it without a loss of healthy output?
**Runs and data used:** r39 `rl-glm53f39-25pdn` (experiment `rl-glm53f-auct-cap-r39`, W&B `j6todjkb`); r42 `rl-glm53f42-5cthp` (`rl-glm53f-auct-cap-r42`, W&B `ob9qvkyg`); r43 `rl-glm53f43-qztzt` (`rl-glm53f-adebt-v3-r43`, W&B `2z0599lb`); r30 (`rl-glm53f-auct-cap-r30`, W&B `lwkruqyz`, pods gone); r44 `rl-glm53f44-lt8vm` first 90 minutes as the check after the fix. Data: gym pod logs of all 288 pods per run (90 min, 3 h, and 4 h windows); trainer-worker-0 logs; `rollout.json`, `trajectory.json`, and `result.json` of the last finished group on every pod (`/trials/arena-rollout-trash`; 848 r42 and 946 r39 episodes); `sample_summary/rollout_<N>.jsonl` under `/mnt/scratch-s3files-rw/guparpit/debug/<experiment>/`; 5,648 r43 `.tokens` files at `s3://arena-scratch-prod-bom-ap-south-1/guparpit/routing/rl-glm53f-adebt-v3-r43/`. Saved copies (not in git, on tmpfs, present at 16:35Z 2026-09-28): `/tmp/ctx131k/`, `/tmp/genlen/`, `/tmp/budget-design/`.
**Code SHAs:** AREnATasks `0b12642`: the facade clamps `max_new_tokens` to the room left (ADR-0063 amendment 2026-09-27, CR-308323817). AREnATasks `538bc63`: the gym takes the cap, window, and sampling from the task message (ADR-0072, CR-308323860). miles `4c9e97b0f`: the trainer sends `sampling_params` and `max_seq_len` (ADR-0015). miles `e0987aed6`: `gen-workflow.py` maps `compaction-max` to `agent-kwargs` and the trainer rejects `rollout_top_k`. AREnATasksApps mainline `e4bf764`: the template drops the gym env copies (CR-308323875). Runtime code read from the images: gym `gym-glm53-adr69-20260925a`, trainer `miles-glm53-r12-20260925a` (miles `db3955b70` inside), SGLang `0.5.19.dev52+g9a26e74`.
**Workflow ids:** `wf_795e8c03-107` (overflow trace, 2026-09-26 23:56Z to 2026-09-27 00:27Z), `wf_c2a543f0-f9c` (generation length, 00:09Z to 00:51Z), `wf_6ef80880-514` (env-knob and clamp design, 01:27Z to 01:54Z).

## Method

Terms. A "400" is the HTTP Bad Request that SGLang returns when a request
asks for more tokens than the window holds. "Compaction" is the Vulcan step
that shortens the chat history; a "summary" compaction first asks the model
for a handoff note, a "structural" compaction keeps no note. A "clipped
turn" is a model call that stops at the per-call cap. The "clamp" is the
rule `max_new_tokens = min(cap, window - input)`.

1. **Overflow trace (`wf_795e8c03-107`).** One agent copied the runtime
   code from a live gym pod (`amzn_arena_harbor/agents/vulcan.py`,
   `training_vulcan.py`, `sglang_rollout.py`, `constants.py`) and the
   SGLang check from `/sgl-workspace`. One agent pulled `kubectl logs
   --since=4h` from all 288 gym pods of each run, parsed them into per-pod
   timelines, and ran a read-only probe over `/trials/arena-rollout-trash`
   on every pod. It tied each 400 to one trial with two keys: the failed
   summary input equals the archived segment length plus 278 tokens (nudge
   plus `SUMMARY_PROMPT`) or plus 356 (update prompt), and the "removed M
   messages" count equals the rebuilt message count. Engine-side 400 lines
   came from trainer-worker-0 with Ray "[repeated Nx]" markers expanded. A
   verify agent recounted a 90-minute window.
2. **Generation length (`wf_c2a543f0-f9c`).** Episode and segment lengths
   come from `sample_summary` rows with `mode == "dp_pad"` and
   `remove_sample == true` dropped; the episode key is (`rollout_id`,
   `index mod 2^40`) (miles ADR-0011). Per-call lengths come from the r43
   `.tokens` files (ADR-0071): a call spans from the token after
   `<|assistant|><think>` to the next `<|assistant|>`; a token counts as
   generated when `loss_mask == 1` or `log_prob != 0`. No token arrays
   survive for the healthy auctioneer window, so its per-call figures are a
   log-normal fit on the W&B per-call mean and variance.
3. **Knob and clamp design (`wf_6ef80880-514`).** A survey of every reader
   and setter of `ARENA_MAX_TOKENS`, `ARENA_ROLLOUT_CONTEXT_LIMIT`,
   `ARENA_COMPACTION_MAX`, and the sampling env names across AREnATasks,
   AREnATasksApps, miles, AGISlime, AREnABase, and the 77 live
   WorkflowTemplates on `arena-prod-bom-v2`. The clamp patch ran on a
   scratch copy of AREnATasks: `tests/harbor_runtime` 1736 passed, 9
   skipped.
4. **Recount for this record (2026-09-28).** The saved logs and records in
   `/tmp/ctx131k/` and `/tmp/genlen/` were re-read with fresh scripts. A
   value marked "re-verified" matches that recount. A value marked "journal"
   comes from the workflow report only.

## Results

### A. The mechanism

| Arm | Metric | Value | Source |
| --- | --- | --- | --- |
| r39, r42 | Window and per-call cap | 131072 and 32768 (`ARENA_ROLLOUT_CONTEXT_LIMIT`, `ARENA_MAX_TOKENS`; trainer `--rollout-max-context-len 131072`, `--rollout-max-response-len 32768`) | pod env; r42 `miles-config.yaml:269,286-287`; `wf_795e8c03-107` |
| r39, r42 | Send site | `sampling_params = {"max_new_tokens": self._max_tokens}` on every `/generate`, no room check | gym image `amzn_arena_harbor/sglang_rollout.py:703`; re-derived in `wf_795e8c03-107` verify |
| r39, r42 | Reject site | SGLang rejects `input + max_new_tokens > 131072`; `validate_total_tokens` hardcoded True; `allow_auto_truncate` False | engine image `sglang/srt/managers/tokenizer_manager.py:1233-1256`, `server_args.py:1460` |
| r39, r42 | Largest input that fits | 131072 - 32768 = 98304 | arithmetic |
| r39, r42 | Compaction trigger | `int(0.8 * 98304)` = 78643 projected tokens, checked once per turn; headroom 19661 | gym image `agents/vulcan.py:1501-1516`; `ARENA_COMPACTION_FRACTION` unset, default 0.8 |
| r39, r42 | One clipped turn | +32768 tokens, so the history jumps from under 78643 to over 98304 in one hop; the summary request (history + 278) then overflows | `wf_795e8c03-107` trace:episodes |
| r39, r42 | Engine 400s, 90 min (23:5xZ-00:1xZ) | r39 429, r42 453; every line says "32768 tokens for the completion"; input 98,332-130,839 (r39) and 98,311-130,919 (r42); no input alone over 131072 | `/tmp/ctx131k/verify/eng90_*.txt`, re-verified |
| r39 | Repeated request size | 102,815 (83 lines), 102,817 (50), 102,813 (49), 102,816 (26) in 90 min = a 4,233-token head + 3 x 32768 + nudges + 278 | `/tmp/ctx131k/verify/g39`, re-verified; shape from `wf_795e8c03-107` |

### B. Who sends the 400s

| Arm | Metric | Value | Source |
| --- | --- | --- | --- |
| r39, r42 | Gym-side "Context length exceeded" lines, 3 h | r39 772, r42 567; every one is `vulcan: summary call failed ... no new note` | `/tmp/ctx131k/r39-gym3h.txt`, `r42-gym3h.txt`, re-verified; memory note counted 770 of 772 and 567 of 574 |
| r39, r42 | Compactions, 3 h (summary / structural) | r39 871 / 799 (48% lost the note); r42 1071 / 650 (38%) | same files, re-verified |
| r39, r42 | Compactions, 90 min (summary / structural) | r39 410 / 488 (54%); r42 504 / 271 (35%) | `/tmp/ctx131k/verify/g39`, `g42`, re-verified |
| r39, r42 | Stop reasons, 3 h | r39 truncated 659, model_length 357, completed 24; r42 model_length 799, truncated 212, completed 17 | same files, re-verified |
| r39, r42 | Stop reasons, 90 min | r39 250 / 97 / 5; r42 61 / 272 / 4 (truncated / model_length / completed) | `/tmp/ctx131k/verify`, re-verified |
| r39, r42 | Engine-side split, 3 h | summary 400s vs terminal 400s: r39 772 vs 357 (68% summary); r42 567 vs 799 (41% summary) | derived from the two rows above; the terminal 400 logs only `stop_reason=model_length` on the gym side |
| r39, r42 | Engine vs gym totals, 4 h (21:13Z-00:12Z) | r42 1351 engine vs 523 summary + 831 model_length = 1354 gym; r39 1167 vs 824 + 345 = 1169 | `wf_795e8c03-107` trace:episodes, journal |
| r39, r42 | "overflow" compactions | 0 in both runs | `/tmp/ctx131k/verify`, re-verified |

### C. No retry loop

| Arm | Metric | Value | Source |
| --- | --- | --- | --- |
| code | Overflow retry | none: `_generate` re-raises an overflow at once; only other errors retry up to `_LLM_MAX_ATTEMPTS = 3` | gym image `agents/vulcan.py:1581-1584` |
| code | Summary overflow | caught in `_summarize`, logged, returns None, structural compaction follows | `agents/vulcan.py:1618-1623` |
| code | Main-turn overflow | one compact-and-retry when a compaction is left; else `_on_context_exceeded` and `model_length` | `agents/vulcan.py:1730-1751`; `training_vulcan.py:166-168` |
| code | Trial retries | `ARENA_TRIAL_MAX_RETRIES` unset, so 0 | `gym_worker.py:697-714` |
| code | Router | pass-through proxy, no length check, no retry | `sglang::router` 0.3.2, `miles/router/router.py` |
| r42 | 400s per finished episode (848) | 0: 85, 1: 326, 2: 347, 3: 90; none above 3 | `/tmp/ctx131k/rec_r42.json`, re-verified |
| r39 | 400s per finished episode (946) | 0: 109, 1: 583, 2: 229, 3: 25 | `/tmp/ctx131k/rec_r39.json`, re-verified |
| r42 | Distinct input sizes, 4 h | 712 summary 400s with 697 distinct sizes | `wf_795e8c03-107`, journal |
| r39, r42 | Clipped turns per episode, mean | r39 5.34, r42 4.44 | `rec_*.json`, re-verified |
| r39, r42 | Generated tokens that sit in clipped, masked turns | journal: r42 58% (124.7M of 214M), r39 73% (166.3M of 229M); recount by `rollout.json` mask fields: 59% and 71% | `wf_795e8c03-107`; `rec_*.json` |
| r39, r42 | Reward of `model_length` episodes | r42 mean 0.015, 479 of 659 at 0; r39 mean 0.150 | `rec_*.json`, re-verified |
| r39, r42 | Samples removed by the trainer | 0 ("Removal reasons: none"); `context_error` is telemetry only | trainer image `nats_rollout.py:417-421`; `wf_795e8c03-107` |
| r39, r42 | Time from first 400 to episode end | median 32.6 min (r42, n=371), 33.6 min (r39, n=245) | `wf_795e8c03-107`, journal, not re-verified |
| r39, r42 | Cost of one 400 | under 1 ms; rejected before prefill | `wf_795e8c03-107`, journal |

Traced episodes (`wf_795e8c03-107`, group `seed-00000428.g2447` on r42 pod
`...-9jww7` and `seed-00001028.g2946` on r39 pod `...-shc9d`):

| Run, trial | 400s (input tokens) | Path | End |
| --- | --- | --- | --- |
| r42 `__AQmVf8S` | 101,541; 109,118; 119,831 | two failed summaries, two structural compactions, clipped turn 5/5, terminal overflow | `model_length`, 55 min, reward 0 |
| r42 `__tNbZZEf` | 103,904; 108,834; terminal | same path, 86 tool calls | `model_length`, reward -0.131 (the only non-zero reward in its group) |
| r42 `__XrRno7L` | 114,694 | both summaries fit; two more clipped turns; one terminal 400, no retry | `model_length` 2 s after the 400 |
| r39 `__gd6BLvf` | 102,815 | 3 clipped turns with no tool call, failed summary, 3 more clipped turns | `truncated`, 0 tools, 0 trainable tokens, 66 min |

### D. Generation length: why 16384 keeps the healthy range

| Arm | Metric | Value | Source |
| --- | --- | --- | --- |
| r43 (~rollouts 38-40) | Generated tokens per model call, all calls (167,592 calls, 5,648 `.tokens` files, 1,176 episodes) | P50 387, P90 3,063, P99 17,143, max 32768, mean 1,358; clipped at 32768: 517 (0.31%) | `/tmp/genlen/r43_calls.jsonl`, re-verified |
| r43 (newer files) | Same, 101,740 calls | P50 423, P90 3,268, P99 20,017 | `wf_c2a543f0-f9c` verify, journal |
| r43 | Model calls per episode | P50 123, P90 271; per Vulcan step P50 33, P90 89 | `wf_c2a543f0-f9c`, journal |
| auctioneer healthy (r30, r39, r42 rollouts 0-29) | Generated tokens per call | P50 129-137, P90 606-661, ESTIMATE from a log-normal fit on the W&B per-call mean (244-403) and variance | `wf_c2a543f0-f9c` verify; `/tmp/genlen/wandb_*.json` |
| auctioneer healthy | Calls that hit 32768 | r42 0-29: 0.06% (480 of 773,717); r39 0-19: 0%; r39 0-39: 0.50% | gym logs, `wf_c2a543f0-f9c`, journal |
| auctioneer collapsed | Generated tokens per call | r39 57-66: P50 804, P90 32768, 16-19% clipped; r42 45+: P50 1,712, P90 32768, 15% clipped | trash `rollout.json`, `wf_c2a543f0-f9c`, journal |
| auctioneer healthy pooled 0-29 | Segment `response_length` (model plus tool output) | P50 49,909, P90 63,275, P99 74,654 (n 23,845) | `/tmp/genlen/rl-glm53f-auct-cap-*/sample_summary`, re-verified; journal 49,910 / 63,280 |
| auctioneer healthy 0-29 | Episode `response_length`, per run | r39 P50 51,800 / P90 65,547; r42 51,034 / 64,135; r30 48,291 / 61,823; segments per episode mean 1.03-1.05 | same, re-verified |
| auctioneer healthy | Model-only tokens per episode | P50 about 18-20K (W&B `rollout/episode_response_length` median per rollout), P90 about 25K (estimate) | `wf_c2a543f0-f9c` verify, journal |
| r39 50-66 (collapsed) | Segment / episode `response_length` | 81,293 / 101,184 (P50 / P90); episode 231,024 / 274,738; 2.95 segments per episode (cap 3) | re-verified |
| r42 45-53 (collapsed) | Segment / episode `response_length` | 91,896 / 107,136; episode 274,948 / 302,044; 2.96 segments | re-verified |
| r43 0-37 | Segment / episode `response_length` | 33,048 / 80,599; episode 105,285 / 288,369; segments P50 3, P90 6 | re-verified |
| r43 rollout 38 | Episode `response_length` | P50 264K, P90 534K; 5.03 segments; 25% `context_error`; per-row reward 0.783 | `/tmp/genlen/check/r43_rollout_38.raw`, journal |
| design | Trigger at a 16384 cap | `int(0.8 * (131072 - 16384))` = 91,750 (from 78643); largest input that fits 114,688 | arithmetic; ADR-0072 Consequences |
| design | r43 calls a 16384 cap clips | about 1.1% (P99 is 17-20K) | `wf_c2a543f0-f9c` verify, journal |

### E. After the fix: r44 first 90 minutes, and r45 to r47

| Arm | Metric | Value | Source |
| --- | --- | --- | --- |
| r44 | Cap, window, sampling in the gym | Harbor job `config.json`: `max_tokens` 16384, `context_limit` 131072, `temperature` 1.0, `top_p` 1.0; trainer args `rollout_max_response_len` 16384, `rollout_max_context_len` 131072, `sglang_context_length` 131072 | RUNLOG.md:2177 |
| r44 | Engine 400s, 05:30Z-07:00Z | 0 "Requested token count exceeds", 0 `POST /generate` HTTP 400 | RUNLOG.md:2181 |
| r44 | Compactions by 07:00Z | 180, all `1/4`; 177 `(count, summary)`, 3 `(tokens, summary)`, 0 structural | RUNLOG.md:2182 |
| r44 | Clipped-turn warnings by 07:00Z | 0 | RUNLOG.md:2183 |
| r44 | Rollout 0 | 32 of 32 groups kept, `episode_raw_reward` 0.312, episode response length mean 16,791, max 28,620 | RUNLOG.md:2178 |
| r45 (agentic debt, resumed from r43 step 39) | 400s and clipped turns by 07:20Z | 0 "Requested token count exceeds"; 1,482 "turn truncated at max_tokens" warnings at the 16384 cap | RUNLOG.md:2249-2257 |
| r46 | 400s since launch | 0 | RUNLOG.md:2300 |
| r47 (agentic-debt-766) | 400s and clipped turns at 10:20Z | 0 engine 400s; 51 "turn truncated at max_tokens" | RUNLOG.md:2398 |
| r44-r47 | `rollout_max_response_len` in the run config | 16384 in all four; window values 131072 | `r44/miles-config.yaml:302,319-320`, `r45:278,295-296`, `r46:310,327-328`, `r47:330,348-349` |

## Verdict

No request carried more than 131072 input tokens. Every 400 was an input of
98K to 131K tokens plus the fixed 32768 output budget that the gym facade
attached to every call. The compaction trigger at 78643 tokens left 19661
tokens of headroom, and one clipped turn added 32768, so the history passed
the 98304 line before the check ran. The summary request then failed, and
the compaction kept no note; after two compactions the next overflow ended
the episode as `model_length`. The repeated sizes near 102,815 on r39 were
many sibling episodes with the same shape (a fixed head plus three clipped
turns), not one request resent in a loop. The code has no retry on an
overflow, and the data shows at most three 400s per episode. The numbers
support two fixes: clamp the output budget to the room left, and lower the
per-call cap to 16384. Healthy calls sit at P99 17-20K tokens (agentic debt)
and a few hundred tokens (auctioneer); only collapsed runs reach P90 = 32768.
A 16384 cap keeps the healthy range and moves the trigger to 91750. r44
then ran 90 minutes with 0 engine 400s and 0 structural compactions. The
numbers do not show that the clamp or the cap stops the length explosion
itself: the clipped turns that filled 58-73% of generated tokens on r39 and
r42 come from the policy, and the cap only changes where they stop.

## Caveats and open items

- **"About 99% summary calls" is a gym-log share.** 772 of the 772 r39 and
  567 of the 567 r42 gym-side "Context length exceeded" lines in 3 h are
  summary failures, because the terminal overflow logs only
  `stop_reason=model_length`. Of all engine-side 400s in the same 3 h, the
  summary share is 68% on r39 and 41% on r42. ADR-0063 (amendment
  2026-09-27), ADR-0072, miles ADR-0015, and `r44/BUILD.md` carry the 99%
  figure without this note.
- The trash directory holds only the last finished group of 8 trials per
  pod, so per-episode counts cover finished episodes after 20:00Z on
  2026-09-26. Live episodes appear only as lower bounds.
- Healthy auctioneer per-call percentiles are estimates. No token arrays
  survive for that window; the 5 trash jobs from it (4 r42 tasks, 1 r39
  task) have group means 2.5 to 4 times the W&B batch mean and are not
  representative.
- `response_length` includes tool output and counts from the first
  `loss_mask == 1` token. A clipped first turn lands in the prompt region, so
  the collapsed r39 window undercounts about 33K tokens in 34% of its
  episodes. `total_length` is in the journal for contrast.
- The 4-h engine-versus-gym totals, the 712-of-697 distinct-size count, the
  time-to-end medians, and the 1.1% clip estimate at 16384 are journal
  values, not re-verified.
- r45 logged 1,482 "turn truncated at max_tokens" warnings in about 70
  minutes at the 16384 cap, against 0.31% of r43 calls at 32768. The cap
  bites agentic debt more than the estimate. Nobody has counted the clipped
  share per call on r45 to r47 yet.
- The r44 watch item in RUNLOG.md:2185-2187 (compare the 400 count and the
  structural share with r42 at the same rollout indices) is not done. The
  first r44 compactions came from the message-count trigger, not the token
  trigger.
- Side result, unverified: the r42 trainer at 00:14Z trained rollout 54 on
  groups g1780-g1879 while the gym ran g2440+, so 400-hit episodes reach the
  trainer hours later.
- The clamp does not cover the streaming path:
  `amzn_arena_streaming/sglang_provider.py:800-801` still sends an unclamped
  `max_new_tokens`.
- `raw_response_length/response_length_clip_ratio` compares the whole
  multi-turn response with the per-call cap and reads near 1 at 16384. The
  `correct_length/pNN` buckets lose responses above the cap. Neither changes
  the loss.
- After a clip at the room left, the Vulcan nudge still names the per-call
  cap, which is then too high (`ponytail:` comment in `agents/vulcan.py`).
- Rejected options, for the record: `sglang_allow_auto_truncate` (it deletes
  input tokens once the input alone fills the window,
  `tokenizer_manager.py:1224`, and breaks the TITO token alignment);
  `ARENA_COMPACTION_FRACTION` about 0.6 as an env-only mitigation (it spends
  the compaction budget sooner and ADR-0072 removed the env name).
- The saved data under `/tmp/ctx131k/`, `/tmp/genlen/`, and
  `/tmp/budget-design/` is not in git and can vanish with the host.

## Actions taken

- AREnATasks `0b12642` (CR-308323817, merged): `ArenaSGLangLLM` sends
  `min(max_tokens, context_limit - len(input_ids))` and raises
  `ContextLengthExceededError` before the POST when no room is left.
  ADR-0063 amendment 2026-09-27; ADR-0066 text updated.
- AREnATasks `538bc63` (CR-308323860, merged): ADR-0072. The task message
  carries `sampling_params` and `max_seq_len`; the gym reads no
  `ARENA_MAX_TOKENS`, `ARENA_ROLLOUT_CONTEXT_LIMIT`, `ARENA_TEMPERATURE`,
  `ARENA_TOP_P`, `ARENA_COMPACTION_MAX`, `ARENA_TRUNCATED_TURN_MAX`, or
  `ARENA_COMPACTION_FRACTION`; a set name stops `--mode rollout` at startup.
  `TrainingVulcanAgent` defaults `max_truncated_turns` to 5.
- miles `4c9e97b0f` and `e0987aed6`: ADR-0015. `NATSRolloutWorker` sends
  `sampling_params` {`max_new_tokens`, `temperature`, `top_p`} and
  `max_seq_len` = min(`rollout_max_context_len`, `sglang_context_length`);
  it raises at startup without a cap or window, or with `rollout_top_k`
  set. `gen-workflow.py` maps a base `compaction-max` N to `agent-kwargs`
  `{"max_compactions": N}`. RUNLOG.md:2065-2141 records the decision;
  `experiment-list.md` marks R2 and R3 (the r8 rejections of the clamp and
  the 16384 cap) as reversed on 2026-09-27.
- AREnATasksApps mainline `e4bf764` (CR-308323875): the miles-deployer
  template drops the gym env copies and the `compaction-max` parameter and
  adds `agent-kwargs`. Template `guparpit-miles-deployer-v10` created
  2026-09-27T04:48:19Z on `arena-prod-bom-v2` (`r44/BUILD.md`).
- Images: trainer `miles-glm53-r14-20260927a`
  (`sha256:48d52a6a431257f9f7b40ce00debe9b2a10b61f94cee85d83291b21a2439e60c`),
  gym `gym-glm53-adr72-20260927a`
  (`sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d`,
  revision label `538bc63`). Deploy order: trainer first, then gym.
- r39 and r42 retired 2026-09-27 (`shutdown: Stop`, finished 05:02:08Z and
  05:05:52Z; last checkpoints `iter` 49 and 59). r44 `rl-glm53f44-lt8vm`
  launched 05:07:02Z with `rollout_max_response_len: 16384` and Vulcan
  `max_compactions` 4. r45, r46, and r47 carry the same cap and window.

## Sources

- Journals: `wf_795e8c03-107` (agents `code:guards`, `trace:episodes`,
  `verify`), `wf_c2a543f0-f9c` (`episode-lengths`, `per-turn-lengths`,
  `verify`), `wf_6ef80880-514` (`consumers`, `single-source`,
  `clamp-design`, `plan`).
- Memory notes: `auct-131k-summary-overflow-2026-09-27.md`,
  `gen-length-p50-p90-2026-09-27.md` (notes; numbers checked against the
  saved data where the table says "re-verified").
- Saved data: `/tmp/ctx131k/verify/eng90_*.txt`, `g39/`, `g42/`;
  `/tmp/ctx131k/r39-gym3h.txt`, `r42-gym3h.txt`, `rec_r39.json`,
  `rec_r42.json`; `/tmp/genlen/r43_calls.jsonl`,
  `/tmp/genlen/rl-glm53f-auct-cap-{r30,r39,r42}/sample_summary/`,
  `/tmp/genlen/rl-glm53f-adebt-v3-r43/sample_summary/`,
  `/tmp/genlen/wandb_*.json`; `/tmp/budget-design/facade.diff`.
- S3: `s3://arena-scratch-prod-bom-ap-south-1/guparpit/routing/rl-glm53f-adebt-v3-r43/`
  (`.tokens` files; the trainer deletes each file after it reads it).
- AREnATasks (`/workplace/guparpit/arena-glm53/src/AREnATasks`):
  `adr/0063-vulcan-context-compaction-and-rollout-segments.md` lines
  583-640 (amendment 2026-09-27),
  `adr/0072-training-gym-takes-limits-from-the-task-message.md`,
  `adr/0066-mask-clipped-and-empty-generates-train-every-verified-episode.md`;
  commits `0b12642`, `538bc63`.
- miles: `miles_plugins/arena/adr/0015-task-message-carries-output-cap-window-and-sampling.md`;
  `training-runs/harbor-rl-glm53-flash/RUNLOG.md` lines 1542-1600 (r6 and
  r7: 2945 and 711 overflow errors at 98-102K prompt tokens, the first
  clamp proposal), 2065-2141 (ADR-0015 entry), 2142-2188 (r44 launch),
  2240-2262 (r45), 2266-2322 (r46), 2323-2400 (r47);
  `training-runs/harbor-rl-glm53-flash/experiment-list.md` lines 160-176
  (R2, R3); `training-runs/harbor-rl-glm53-flash/r44/BUILD.md`;
  `r44..r47/miles-config.yaml`, `workflow.yaml`.
- Upstream precedent: miles
  `miles/rollout/generate_utils/generate_endpoint_utils.py:59-63`
  (`min(max_new, rollout_max_context_len - len(input_ids))`, else
  TRUNCATED); pi `packages/ai/src/api/simple-options.ts:15-18`
  (`clampMaxTokensToContext`) and `compaction.ts:734-736` (summary capped at
  0.8 x reserve); Inspect AI `_providers/sglang.py:260-273`
  (`handle_bad_request` maps the 400 to `stop_reason="model_length"`).
