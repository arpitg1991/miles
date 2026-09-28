# Run record: r42 — r39 with the SGLang radix cache, the overlap schedule, and a 4 h agent limit

**Status:** Retired
<!-- gen-workflow:begin -->
**Date:** 2026-09-25
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `rl-glm53f42-`
**Experiment name:** `rl-glm53f-auct-cap-r42`
**W&B project:** `rl-glm53f-auct-cap` (group `rl-glm53f-auct-cap-r42`)
**Dataset:** `lakefs://arena-inspect/dev/internal/auctioneer/caponly/caponly-1034/manifest.jsonl` (gym `auctioneer-caponly`)
**Manifest commit:** `dev`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr69-20260925a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r12-20260925a`
**Template:** `guparpit-miles-deployer-v7`
**Base:** `r39`
<!-- gen-workflow:end -->
**Argo workflow:** `rl-glm53f42-5cthp`
**W&B run:** `ob9qvkyg` (`RANK_0`; W&B group `rl-glm53f-auct-cap-r42_c66qd2uv`; state `crashed` after the Stop)
**Task pin:** `e5ef91b0` (8-character prefix, the same rows as r39; the full `lakefs_commit_id` is not re-verified). `dev` is a branch ref: the trainer logged `Pulled lakefs://...` at 18:12:13Z with no commit id.
**Image digests:** same images as r39: gym `sha256:6b18f411d482521a316e2f7211fe5046199e4e549dd0f9c781879bc58c097302`, trainer `sha256:fa9a1252f8f292564fddb0e7092ee78604b9ba211571a0672da339bec2498020`; ECR `describe-images`, ap-south-1
**Trainer config deltas vs base:** `sglang_disable_overlap_schedule` `true` -> `false`; `sglang_disable_radix_cache` `true` -> `false`; `experiment_name`, `project_name`, and `arena_sample_summary_dir` `-r39` -> `-r42`. Workflow parameter `agent-timeout-multiplier` `1` -> `2`.
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-auct-cap-r42`, last save `iter_0000049` (2026-09-26 20:04Z); holds `hf/rollout_<N>` exports and `rollout/arena_data_source_state_<N>.pt`.
**Outcome:** Collapsed like r39, about 5 rollouts earlier. Reward rose to 0.740 at rollout 36, then the mean episode length grew past 100K tokens and reward fell to -0.04-0.15 in the last 10 rollouts. The radix cache and overlap schedule cut the cold-start rollout time 3.3x (1738 s against 5746 s) with no change in policy drift, and had no effect in the drain-bound steady state. 11,137 SGLang HTTP 400 lines; 41 percent of the engine-side 400s were Vulcan summary calls. Retired 2026-09-27 05:02Z on the user go, to free 80 nodes for r44.

## Goal

Measure the rollout-time gain of the SGLang radix cache and the overlap
schedule on the same task, dataset, and images as r39, and remove the 2 h
agent timeouts of r39. r39 continued, so that the two runs compare at
equal rollout indices.

## Setup

All r39 settings (`../r39/RECORD.md` Setup) except the deltas below.
Generated with `gen-workflow.py 42 --base r39 --param
agent-timeout-multiplier=2` (`BUILD.md`).

| Item | Base r39 | This run |
| --- | --- | --- |
| `sglang_disable_radix_cache` | `true` | `false` |
| `sglang_disable_overlap_schedule` | `true` | `false` |
| `mamba_radix_cache_strategy` (engine log) | `auto` | `extra_buffer` (resolved from `auto`) |
| `agent-timeout-multiplier` | 1 (2 h) | 2 (4 h) |

## Timeline

| UTC | Event |
| --- | --- |
| 2026-09-25 16:45 | SGLang speed probe on r39 (16 tok/s, TTFT 29 s; memory note) motivates the run. |
| 2026-09-25 18:06 | `kubectl create` -> `rl-glm53f42-5cthp`; run record commit `ed3dbdc9c` (rebased copy `c22a23681`). |
| 2026-09-25 18:11 | `trainer-0.log` starts; manifest pulled 18:12:13Z. |
| 2026-09-25 18:24 | SGLang engines up: `disable_radix_cache=False`, `disable_overlap_schedule=False`, `linear_attn_backend='triton'`, `mamba_radix_cache_strategy='extra_buffer'`. Rollout 0 collect starts 18:26:14Z. |
| 2026-09-25 18:55 | Rollout 0 complete: reward 0.385, 1732 s (r39: 5740 s). |
| 2026-09-25 20:05 | First step: `train_time` 3527 s against r39 3802 s; independent of SGLang (memory note). |
| 2026-09-25 22:27 | Save `iter_0000009`; then 09-26 01:13 iter 19, 04:06 iter 29, 09:05 iter 39, 20:03 iter 49. |
| 2026-09-26 01:07 | First SGLang "Requested token count exceeds" 400 (between rollouts 20 and 21). |
| 2026-09-26 06:13 | Peak reward 0.740 at rollout 36. |
| 2026-09-26 23:40 | Read as collapsed: rollouts 49-52 reward 0.05-0.15 (memory note). |
| 2026-09-27 00:05-01:00 | 131K overflow trace (`wf_795e8c03-107`) and generation-length study (`wf_c2a543f0-f9c`) on the live pods. |
| 2026-09-27 03:00 | User go: launch r44 from the r42 settings; retire r39 and r42 when r44 is ready (memory note). |
| 2026-09-27 04:58 | `shutdown: Stop` patch. |
| 2026-09-27 05:01 | Rollout 58 complete (last) 05:01:33Z: reward 0.023. |
| 2026-09-27 05:02 | Workflow finished 05:02:08Z (`../RUNLOG.md:2144-2146`). r44 `rl-glm53f44-lt8vm` created 05:07Z (`../RUNLOG.md:2160`). |

## Results

Per-rollout values come from the `Rollout N complete` lines
(`nats_rollout.py:2495`) and the metric dictionaries in each
`trainer-0.log` on S3 (Sources). `avg_reward` equals
`rollout/episode_raw_reward`. Buckets are means over 10 rollouts. The r39
columns repeat the r39 record for the side-by-side read.

### Reward and episode length

| Rollouts | r39 reward | r39 length mean | r42 reward | r42 length mean |
| --- | --- | --- | --- | --- |
| 0-9 | 0.382 | 18.7K | 0.383 | 18.9K |
| 10-19 | 0.362 | 19.8K | 0.346 | 19.8K |
| 20-29 | 0.465 | 26.0K | 0.451 | 23.7K |
| 30-39 | 0.514 | 51.7K | 0.625 | 56.2K |
| 40-49 | 0.356 | 106K | 0.338 | 126K |
| 50-59 | 0.206 | 122K | 0.041 (50-58) | 95.4K |
| 60-69 | 0.142 | 111K | - | - |
| 70 | 0.136 | 92.0K | - | - |

Length is `rollout/episode_response_length/mean` (model-written tokens per
episode, loss-mask sum). Peak reward: r39 0.616 at rollout 40; r42 0.740 at
rollout 36. First rollout with mean length over 50K: r39 34, r42 35; over
100K: r39 42, r42 37. Maximum mean length: r39 129K (rollout 57), r42 147K
(rollout 45). Last 10 rollouts: r39 0.011-0.298, r42 -0.036-0.151.

### Rollout time and engine settings

| Metric | r39 | r42 | Source |
| --- | --- | --- | --- |
| Rollout 0 `perf/rollout_time` (empty pipeline) | 5746 s | 1738 s (3.3x faster) | trainer logs; W&B |
| Rollout 1 time | 363 s (queue 0) | 121 s (queue 1) | trainer logs; W&B |
| Rollout time, rollouts 10-29 (queue 320, drain-bound) | 141-152 s | 134 s | trainer logs |
| Rollout time, collapse phase | 654-687 s | 656-717 s | trainer logs |
| `perf/step_time`, rollouts 10-19 -> last bucket | 998 s -> 4310 s | 994 s -> 4530 s | trainer logs |
| `perf/train_wait_time`, first -> last bucket | 234 s -> 353 s | 223 s -> 451 s | trainer logs |
| `perf/update_weights_time` | 24-27 s, flat | 24-26 s, flat (no overlap stall) | trainer logs |
| `perf/actor_train_mfu` (logged; the MFU study shows the formula over-counts) | 0.010 -> 0.082 | 0.011 -> 0.083 | trainer logs |
| `rollout/prefix_cache_hit_rate` | 0 (radix off) | 0 (metric not wired for the gym path) | trainer logs |
| `rollout/stop/timeout`, rollouts 0-29 (W&B, partial) | 0.07-0.16 | 0 | W&B history |

After rollout 1 the queue holds 320 finished groups on both runs, so the
rollout time measures the drain of the queue, not generation. The radix
gain is visible only on the empty pipeline (rollouts 0 and 1). Both runs
were trainer-bound: `perf/step_time` grew with episode length. Study:
`../../studies/radix-cache-and-overlap-schedule/STUDY.md`.

### Policy drift and optimizer (R3 routing replay on)

| Metric | r39 first step | r39 last | r42 first step | r42 last |
| --- | --- | --- | --- | --- |
| `train/train_rollout_kl` | 0.0067 | 0.0069 | 0.0037 | 0.0030 |
| `train/train_rollout_logprob_abs_diff` | 0.051 | 0.059 | 0.034 | 0.039 |
| `train/ppo_kl` | -1.9e-6 | -2.9e-6 | 6.8e-7 | 1.7e-6 |
| `train/grad_norm` (bucket means) | 0.194 | 0.071 | 0.153 | 0.043 |
| `train/pg_clipfrac` (bucket means) | 2.6e-4 | 1.3e-4 | 2.6e-4 | 7.7e-5 |
| `train/ess_ratio` (bucket means) | 1.02 | 2.94 | 1.03 | 2.96 |

The radix cache and overlap schedule did not raise the rollout-trainer
gap: r42 drift is lower than r39 at step 0. `ess_ratio` above 1 is an
aggregation artifact of long samples, not an importance-sampling problem
(memory note `glm53-length-explosion-ctrf`). Dynamic sampling dropped 0
groups on r39 and 10 groups in total on r42; `Removal reasons` were `none`
on all but 3 rollouts (r39 `failed=2`, `failed=1`; r42 `failed=1`).

### 131K overflow (SGLang HTTP 400)

| Metric | r39 | r42 | Source |
| --- | --- | --- | --- |
| "Requested token count exceeds" lines, whole run | 11,419 | 11,137 | trainer logs |
| First 400 | 2026-09-25 17:03Z | 2026-09-26 01:07Z | trainer logs |
| 400 lines per day (09-25 / 09-26 / 09-27) | 1,507 / 8,346 / 1,566 | - / 9,045 / 2,092 | trainer logs |
| Engine 400s in 90 min (23:5xZ-00:1xZ), re-verified from the saved engine logs | 429 | 453 | `/tmp/ctx131k/verify/eng90_*.txt`; `../../studies/context-overflow-131k-and-clamp/STUDY.md` |
| Journal trace count, engine vs gym side, 90 min (a different window and method) | 545 vs 567 | 512 vs 513 | `wf_795e8c03-107` result text |
| Input tokens of a rejected request | 98,332-130,839 (+32,768 fixed budget) | 98,311-130,919 | `/tmp/ctx131k/verify`, re-verified |
| Gym-side context 400s that were Vulcan summary calls (3 h) | 772 of 772 | 567 of 567 | `/tmp/ctx131k/r39-gym3h.txt`, `r42-gym3h.txt`, re-verified |
| Summary share of all engine-side 400s (3 h) | 68 percent | 41 percent | context-overflow study, re-verified |
| Compactions summary / structural (3 h) | 871 / 799 (48 percent lost the note) | 1071 / 650 (38 percent) | same files, re-verified |
| Stop reasons, last 90 min (`model_length` / `truncated` / `completed`) | 97 / 250 / 5 | 272 / 61 / 4 | `wf_795e8c03-107` verify |
| Episodes that ended at the context limit, last 90 min | 28 percent | 81 percent | `wf_795e8c03-107` verify |
| `rollout/clipped_turns`, rollouts 40-49 | 0.32 | 4.66 at rollout 58 (summary) | W&B |
| `rollout/masked_output_tokens`, rollouts 40-49 | 10.6K | 153K at rollout 58 (summary) | W&B |

Mechanism (`wf_795e8c03-107`, confirmed by its verify agent): the gym
sent `max_new_tokens` 32768 on every call with no room check
(`sglang_rollout.py:703` in the adr69 image). SGLang rejects input +
32768 > 131072. The Vulcan compaction trigger was `int(0.8 * (131072 -
32768))` = 78,643 tokens, but one clipped 32,768-token turn pushed the
history past 98,304 before the check. The compaction summary call then
failed with a 400 and fell back to a structural compaction with no handoff
note. There was no retry loop: at most 2 summary 400s plus 1 terminal 400
per episode. The r39 repeated size 102,815 was 3 clipped turns after the
same task header on sibling trials, not one request resent. The trainer
kept every such sample; `context_error` is telemetry only.

### Generation length (`wf_c2a543f0-f9c`, verified)

| Unit | Healthy (rollouts 0-29) | Collapsed |
| --- | --- | --- |
| Tokens per model call, P50 / P90 | about 130 / 600-660 (estimate from W&B mean and variance) | r39 rollouts 57-66: 804 / 32,768; r42 45+: 1,712 / 32,768 |
| Calls clipped at 32,768 | r42 0.06 percent, r39 0.50 percent (0-39) | 15-19 percent |
| Model tokens per episode, P50 | r39 20,133; r42 19,122 | 92K-147K mean (trainer logs) |
| Response tokens per episode incl. tool output, P50 / P90 | 50,418 / 64,027 (r30+r39+r42 pooled, n=23,020) | 231-275K / 275-302K (memory note, not re-verified) |

The 32,768 cap almost never bit in healthy rollouts. It bit only after the
length growth. A 16,384 cap keeps the healthy range and moves the
compaction trigger to 91,750 tokens; r44 adopted it.

### Other observations

- About 54 percent of r42 turns had empty thinking; the `glm47` parser
  turned `true` into `True` once at step 55 (memory note
  `vulcan-thinking-history-2026-09-25`, not re-verified).
- The r12 trainer reports `rollout/stop/unknown` for every gym stop reason
  other than `timeout` and `context_error`: rollouts 0-29 show
  `stop/unknown` 1.0; the summary at rollout 58 is `context_error` 0.72,
  `unknown` 0.28, `timeout` 0.008.

## Issues

- **Length growth, then collapse.** Repeats the r30 and r39 pattern. The
  r42 changes did not alter the shape; r42 collapsed about 5 rollouts
  earlier than r39 (mean length over 100K at rollout 37 against 42). Root
  cause analysis:
  `../../studies/auctioneer-length-explosion-and-collapse/STUDY.md`.
- **Fixed 32,768 output budget on every call.** Caused the 131K 400s, the
  lost compaction notes (38 percent, `../RUNLOG.md:2112`), and the
  `model_length` endings. Fix: clamp `max_new_tokens` to the room left
  (AREnATasks CR-308323817, `0b12642`), send the cap, the window, and the
  sampling values in the task message (AREnATasks ADR-0072 `538bc63`; miles
  ADR-0015 `4c9e97b0f`, `e0987aed6`), cap 16384. Shipped in r44. Study:
  `../../studies/context-overflow-131k-and-clamp/STUDY.md`.
- **Vulcan `max_compactions` 2 too low for the trigger.** r44 uses the
  Vulcan default 4 via `agent-kwargs` `{}`. `gen-workflow.py` turns a base
  `compaction-max` 2 into `agent-kwargs` `{"max_compactions": 2}`, so a run
  generated from r39 or r42 keeps 2 unless it passes `--param
  'agent-kwargs={}'` (`../RUNLOG.md:2093-2101`).
- **4 h agent limit.** `rollout/stop/timeout` 0 (r39 at 2 h: 0.07-0.16 in
  rollouts 0-29). The longer limit let more episodes reach the context
  wall: 81 percent against 28 percent in the last 90 min.
- **W&B history is incomplete.** `scan_history` returns rows to step 180
  (rollout 30) plus a few later points. The trainer log on S3 is the
  complete series.
- **Manifest commit not logged.** The trainer pulled the `dev` branch path
  and logged no commit id. Only the 8-character prefix `e5ef91b0` of the
  row pin is on record.
- **Sample summaries.** `debug/rl-glm53f-auct-cap-r42/sample_summary/`
  holds `rollout_0..58.jsonl` (59 files). Rows carry no task id.

## Follow-ups

- r44 (`rl-glm53f44-lt8vm`): r42 settings + 16384 cap from the task
  message, `max_compactions` 4; `../r44/RECORD.md`.
- r46 (`rl-glm53f46-clhv6`): r44 + `calculate_per_token_loss` (token-level
  loss average) as the length-bias test; `../r46/RECORD.md`.
- Studies: `../../studies/radix-cache-and-overlap-schedule/STUDY.md` (r33
  vs r34, r39 vs r42), `../../studies/context-overflow-131k-and-clamp/STUDY.md`
  (`wf_795e8c03-107`; holds the generation-length results of
  `wf_c2a543f0-f9c`),
  `../../studies/auctioneer-length-explosion-and-collapse/STUDY.md`.
- Wire the gym stop reasons `completed`, `truncated`, and `model_length`
  into `rollout/stop/<reason>` instead of `unknown` (trainer, open).
- Log the resolved lakeFS commit id on `Pulled lakefs://...` so a branch
  ref can be pinned after the fact (trainer, open).

## Sources

- Run files: `BUILD.md`, `miles-config.yaml`, `workflow.yaml` (this
  folder); commits `ed3dbdc9c` / `c22a23681`. r39 control:
  `../r39/RECORD.md`.
- Legacy log: `../RUNLOG.md:2065-2141` (ADR-0015 entry: overflow evidence,
  `compaction-max` carry-over), `:2142-2187` (retire, last checkpoints,
  r44 launch, 38-48 percent note loss).
- Trainer logs (complete per-rollout series):
  `s3://arena-scratch-prod-bom-ap-south-1/guparpit/logs/rl-glm53f-auct-cap-r42/trainer-0.log`
  (38.0 MB) and `.../rl-glm53f-auct-cap-r39/trainer-0.log` (46.9 MB).
- Checkpoints:
  `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/rl-glm53f-auct-cap-r42/`
  (`iter_0000009..49`, `latest_checkpointed_iteration.txt` = 49,
  `iter_0000049/slime_extra_state.json` = `{"rollout_id": 49, "wandb_run_id": "ob9qvkyg"}`).
- Sample summaries:
  `s3://arena-scratch-prod-bom-ap-south-1/guparpit/debug/rl-glm53f-auct-cap-r42/sample_summary/`.
- W&B: `https://mega.wandb.agi.amazon.dev/arena/rl-glm53f-auct-cap/runs/ob9qvkyg`
  (created 2026-09-25 18:11Z, runtime 125,383 s).
- Images: ECR `arena-slime-dev` in ap-south-1, tags
  `gym-glm53-adr69-20260925a` and `miles-glm53-r12-20260925a`.
- Workflow journals: `wf_795e8c03-107` (131K overflow trace, verified),
  `wf_c2a543f0-f9c` (generation length, verified).
- ADRs: AREnATasks ADR-0063 (Vulcan compaction and segments), ADR-0069
  (training reward is the Harbor trial reward), ADR-0072 (limits from the
  task message); miles ADR-0015 (task message carries cap, window,
  sampling).
- Memory notes (facts with dates; numbers not re-verified unless stated):
  `r42-radix-overlap-timeout-2026-09-25`,
  `sglang-speed-vs-inference-hosting-2026-09-25`,
  `auct-131k-summary-overflow-2026-09-27`,
  `gen-length-p50-p90-2026-09-27`, `r44-r45-launch-plan-2026-09-27`,
  `glm53-length-explosion-ctrf`, `vulcan-thinking-history-2026-09-25`.
