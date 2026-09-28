# Run record: r39 — auctioneer caponly-1034 on the ADR-0069 gym and the r12 trainer

**Status:** Retired
<!-- gen-workflow:begin -->
**Date:** 2026-09-25
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `rl-glm53f39-`
**Experiment name:** `rl-glm53f-auct-cap-r39`
**W&B project:** `rl-glm53f-auct-cap` (group `rl-glm53f-auct-cap-r39`)
**Dataset:** `lakefs://arena-inspect/dev/internal/auctioneer/caponly/caponly-1034/manifest.jsonl` (gym `auctioneer-caponly`)
**Manifest commit:** `dev`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr69-20260925a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r12-20260925a`
**Template:** `guparpit-miles-deployer-v7`
**Base:** `r38`
<!-- gen-workflow:end -->
**Argo workflow:** `rl-glm53f39-25pdn`
**W&B run:** `j6todjkb` (`RANK_0`; W&B group `rl-glm53f-auct-cap-r39_29ix2wj4`; state `crashed` after the Stop)
**Task pin:** `e5ef91b0` (8-character prefix from `miles-config.yaml:4`; the full `lakefs_commit_id` is not re-verified). `dev` is a branch ref: the trainer logged `Pulled lakefs://...` at 07:12:45Z with no commit id; the `dev` head at launch was `4476af5e` (memory note, not re-verified).
**Image digests:** gym `sha256:6b18f411d482521a316e2f7211fe5046199e4e549dd0f9c781879bc58c097302` (pushed 2026-09-25 06:20Z), trainer `sha256:fa9a1252f8f292564fddb0e7092ee78604b9ba211571a0672da339bec2498020` (pushed 06:15Z); ECR `describe-images`, ap-south-1
**Trainer config deltas vs base:** `prompt-data-list` path `.../main/internal/auctioneer/caponly/20260916-v1` -> `.../dev/internal/auctioneer/caponly/caponly-1034`; `experiment_name`, `project_name`, and `arena_sample_summary_dir` `-r38` -> `-r39`. Workflow parameter `partial-reward` `off` (explicit) -> template default `off`.
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-auct-cap-r39`, last save `iter_0000059` (2026-09-26 17:18Z); holds `hf/rollout_<N>` exports and `rollout/arena_data_source_state_<N>.pt`.
**Outcome:** Collapsed. Reward rose to 0.616 at rollout 40, then the mean episode length grew past 100K tokens and reward fell to 0.01-0.30 in the last 10 rollouts. 11,419 SGLang HTTP 400 "Requested token count exceeds" lines; every gym-logged context 400 was a Vulcan summary call, and 68 percent of the engine-side 400s were. Retired 2026-09-27 05:05Z on the user go, to free 80 nodes for r44.

## Goal

Run auctioneer-caponly on the canonical `caponly-1034` dataset with the
ADR-0069 gym (the training reward is the Harbor trial reward) and the r12
trainer (new `rollout/failed/<reason>` and `rollout/dropped_groups/<reason>`
counters). r38 continued on the old dataset. The run ends at `num_rollout`
300, on a user retire, or on a collapse like r30.

## Setup

40 replicas (8 trainer nodes, 32 SGLang engines at TP8), 288 gym pods,
`ack-wait` 36000, `trainer-task-deadline-secs` 39600, `compaction-max` 2,
`agent-timeout-multiplier` 1 (2 h), 83 `excluded-nodes`,
`rollout_batch_size` 64 x `n_samples_per_prompt` 8, `global_batch_size`
256, `lr` 1.5e-6, `use_tis`, `kl_coef` 0, `rollout_max_response_len`
32768, `rollout_max_context_len` 131072, `arena_inflight_multiplier` 4,
`save_interval` 10, `--save-hf`, `sglang_disable_radix_cache` `true`,
`sglang_disable_overlap_schedule` `true`. No `eval_interval`, so no
in-training eval. Source: `miles-config.yaml`, `workflow.yaml:13-40`.

Dataset `caponly-1034` versus `20260916-v1`: 13 seeds removed; only
`instruction.md` changed; `task.toml`, task image digest, verifier, and
`tests/oracle.json` byte-identical, so the reward function is the r38
reward (`BUILD.md`). Generated with `gen-workflow.py 39 --base r38`
(`BUILD.md`).

| Item | Base r38 | This run |
| --- | --- | --- |
| Dataset | `.../main/.../caponly/20260916-v1` (1,047 tasks) | `.../dev/.../caponly/caponly-1034` (1,034 tasks) |
| Gym image | `gym-glm53-adr68-20260924a` | `gym-glm53-adr69-20260925a` (AREnATasks `4d3ecdf`) |
| Trainer image | `miles-glm53-r11-20260924a` | `miles-glm53-r12-20260925a` (miles `db3955b70`) |
| `partial-reward` parameter | `off` (explicit) | template default `off` |

## Timeline

| UTC | Event |
| --- | --- |
| 2026-09-25 05:49 | miles `db3955b70` (r12 trainer source). AREnATasks `4d3ecdf` (adr69 gym source) at 02:14Z. |
| 2026-09-25 06:15 | Trainer image `miles-glm53-r12-20260925a` pushed; gym image `gym-glm53-adr69-20260925a` at 06:20Z (ECR). |
| 2026-09-25 06:21 | `kubectl create` -> `rl-glm53f39-25pdn` (`BUILD.md`). |
| 2026-09-25 06:23-07:07 | Trainer PyTorchJob suspended: Kueue quota short by 184 GPUs. r34 retired on the user go at 07:07:55Z; r39 admitted (memory note, not re-verified). |
| 2026-09-25 07:12 | `trainer-0.log` starts; manifest pulled 07:12:45Z; 1034 prompts loaded 07:12:47Z. |
| 2026-09-25 07:25 | SGLang engines up (`disable_radix_cache=True`, `disable_overlap_schedule=True`). Rollout 0 collect starts 07:27:28Z. |
| 2026-09-25 07:38 | Run record commit `82f4a288c` (rebased copy `895983104`). |
| 2026-09-25 09:03 | Rollout 0 complete: reward 0.385, 5740 s. |
| 2026-09-25 12:37 | Save `iter_0000009`; then 15:23 iter 19, 18:25 iter 29, 22:44 iter 39, 09-26 06:17 iter 49, 09-26 17:18 iter 59. |
| 2026-09-25 17:03 | First SGLang "Requested token count exceeds" 400 (between rollouts 25 and 26). |
| 2026-09-25 18:06 | r42 launched from the r39 settings with the radix cache and overlap schedule on (`../r42/RECORD.md`). r39 continued as its control. |
| 2026-09-25 22:19 | Peak reward 0.616 at rollout 40. |
| 2026-09-26 23:40 | Read as collapsed: rollouts 60-66 reward 0.11 (memory note). |
| 2026-09-27 00:05-01:00 | 131K overflow trace (`wf_795e8c03-107`) and generation-length study (`wf_c2a543f0-f9c`) on the live pods. |
| 2026-09-27 03:00 | User go: post the clamp and task-message CRs, launch r44 from the r42 settings, retire r39 and r42 when r44 is ready (memory note). |
| 2026-09-27 04:13 | Rollout 70 complete (last): reward 0.136. |
| 2026-09-27 05:02 | `shutdown: Stop` patch 05:02:52Z; workflow finished 05:05:52Z (`../RUNLOG.md:2144-2146`). r44 `rl-glm53f44-lt8vm` created 05:07Z (`../RUNLOG.md:2160`). |

## Results

Per-rollout values come from the `Rollout N complete` lines
(`nats_rollout.py:2495`) and the metric dictionaries in `trainer-0.log`
on S3 (Sources). `avg_reward` equals `rollout/episode_raw_reward`. Buckets
are means over 10 rollouts. `../r42/RECORD.md` holds the r39 versus r42
side-by-side tables (rollout time, policy drift, 131K overflow, generation
length).

### Reward and episode length

| Rollouts | Reward | Length mean |
| --- | --- | --- |
| 0-9 | 0.382 | 18.7K |
| 10-19 | 0.362 | 19.8K |
| 20-29 | 0.465 | 26.0K |
| 30-39 | 0.514 | 51.7K |
| 40-49 | 0.356 | 106K |
| 50-59 | 0.206 | 122K |
| 60-69 | 0.142 | 111K |
| 70 | 0.136 | 92.0K |

Length is `rollout/episode_response_length/mean` (model-written tokens per
episode, loss-mask sum). Peak reward 0.616 at rollout 40. First rollout
with mean length over 50K: 34; over 100K: 42. Maximum mean length 129K
(rollout 57). Last 10 rollouts: 0.011-0.298.

### Other metrics

| Metric | Value | Source |
| --- | --- | --- |
| Rollout 0 `perf/rollout_time` (empty pipeline) | 5746 s | trainer log; W&B |
| Rollout 1 time | 363 s (queue 0) | trainer log; W&B |
| Rollout time, rollouts 10-29 (queue 320, drain-bound) | 141-152 s | trainer log |
| Rollout time, collapse phase | 654-687 s | trainer log |
| `perf/step_time`, rollouts 10-19 -> last bucket | 998 s -> 4310 s | trainer log |
| `perf/actor_train_mfu` (logged; the MFU study shows the formula over-counts) | 0.010 -> 0.082 | trainer log |
| `train/train_rollout_kl`, first -> last step | 0.0067 -> 0.0069 | trainer log |
| `train/ppo_kl`, first -> last step | -1.9e-6 -> -2.9e-6 | trainer log |
| `train/grad_norm` (bucket means), first -> last | 0.194 -> 0.071 | trainer log |
| `rollout/stop/timeout`, rollouts 0-29 / 30-49 | 0.07-0.16 / 0.63-0.85 (partial history) | W&B |
| "Requested token count exceeds" lines, whole run | 11,419 (09-25 1,507; 09-26 8,346; 09-27 1,566) | trainer log |
| Engine 400s in 90 min (23:5xZ-00:1xZ), re-verified | 429 | `/tmp/ctx131k/verify/eng90_rl-glm53f39-25pdn.txt`; `../../studies/context-overflow-131k-and-clamp/STUDY.md` |
| Gym-side context 400s that were Vulcan summary calls, 3 h | 772 of 772 | `/tmp/ctx131k/r39-gym3h.txt`, re-verified |
| Compactions summary / structural, 3 h | 871 / 799 (48 percent lost the note) | same file, re-verified |
| Stop reasons, last 90 min (`model_length` / `truncated` / `completed`) | 97 / 250 / 5 | `wf_795e8c03-107` verify |
| Episodes that ended at the context limit, last 90 min | 28 percent | `wf_795e8c03-107` verify |
| Dynamic sampling | 0 groups dropped; `Removal reasons` `none` on all but 2 rollouts (`failed=2`, `failed=1`) | trainer log |

SGLang speed probe (2026-09-25 16:45Z, memory note, not re-verified): a
630-token prompt decoded at 16 tok/s with TTFT 29 s on the r39 router,
against 238 tok/s and 0.04 s on the idle `inference-hosting/glm-5-3-flash`
vLLM. The r39 engines sat at KV 0.97 median, about 51 requests per engine,
about 890 gen tok/s per engine. This probe motivated r42.

## Issues

- **Length growth, then collapse.** Repeats the r30 pattern: a reward rise
  to rollout 40, mean length past 100K tokens, then reward near 0. Root
  cause analysis:
  `../../studies/auctioneer-length-explosion-and-collapse/STUDY.md`.
- **Fixed 32,768 output budget on every call.** Caused the 131K 400s, the
  lost compaction notes (48 percent, `../RUNLOG.md:2112`), and the
  `model_length` endings. Fix: clamp `max_new_tokens` to the room left
  (AREnATasks CR-308323817, `0b12642`), send the cap, the window, and the
  sampling values in the task message (AREnATasks ADR-0072 `538bc63`; miles
  ADR-0015 `4c9e97b0f`, `e0987aed6`), cap 16384. Shipped in r44. Study:
  `../../studies/context-overflow-131k-and-clamp/STUDY.md`.
- **2 h agent limit.** W&B `rollout/stop/timeout` 0.07-0.16 in rollouts
  0-29 and 0.63-0.85 in rollouts 30-49 (partial history). r42 at 4 h shows
  0.
- **Kueue admission wait.** 184 GPUs short at launch; admitted only after
  r34 was retired (memory note). Not a code issue.
- **W&B history is incomplete.** `scan_history` returns rows to step 270
  (rollout 45). The trainer log on S3 is the complete series.
- **Manifest commit not logged.** The trainer pulled the `dev` branch path
  and logged no commit id. Only the 8-character prefix `e5ef91b0` of the
  row pin is on record.
- **Sample summaries.** `debug/rl-glm53f-auct-cap-r39/sample_summary/`
  holds `rollout_0..70.jsonl` (71 files). Rows carry no task id.
- **`rollout/stop/unknown`.** The r12 trainer reports `unknown` for every
  gym stop reason other than `timeout` and `context_error`.

## Follow-ups

- r42 (`rl-glm53f42-5cthp`): radix cache and overlap schedule on, 4 h
  agent limit; `../r42/RECORD.md`.
- r44 (`rl-glm53f44-lt8vm`): r42 settings + 16384 cap from the task
  message, `max_compactions` 4; `../r44/RECORD.md`.
- Studies: `../../studies/radix-cache-and-overlap-schedule/STUDY.md`,
  `../../studies/context-overflow-131k-and-clamp/STUDY.md` (holds the
  generation-length results of `wf_c2a543f0-f9c`),
  `../../studies/auctioneer-length-explosion-and-collapse/STUDY.md`.
- Wire the gym stop reasons `completed`, `truncated`, and `model_length`
  into `rollout/stop/<reason>` instead of `unknown` (trainer, open).
- Log the resolved lakeFS commit id on `Pulled lakefs://...` so a branch
  ref can be pinned after the fact (trainer, open).

## Sources

- Run files: `BUILD.md`, `miles-config.yaml`, `workflow.yaml` (this
  folder); commits `82f4a288c` / `895983104`.
- Legacy log: `../RUNLOG.md:2065-2141` (ADR-0015 entry: overflow evidence,
  `compaction-max` carry-over), `:2142-2187` (retire, last checkpoints,
  r44 launch, 38-48 percent note loss).
- Trainer log (complete per-rollout series):
  `s3://arena-scratch-prod-bom-ap-south-1/guparpit/logs/rl-glm53f-auct-cap-r39/trainer-0.log`
  (46.9 MB).
- Checkpoints:
  `s3://arena-scratch-prod-bom-ap-south-1/guparpit/checkpoints/slime_experiments/rl-glm53f-auct-cap-r39/`
  (`iter_0000009..59`, `latest_checkpointed_iteration.txt` = 59,
  `iter_0000059/slime_extra_state.json` = `{"rollout_id": 59, "wandb_run_id": "j6todjkb"}`).
- Sample summaries:
  `s3://arena-scratch-prod-bom-ap-south-1/guparpit/debug/rl-glm53f-auct-cap-r39/sample_summary/`.
- W&B: `https://mega.wandb.agi.amazon.dev/arena/rl-glm53f-auct-cap/runs/j6todjkb`
  (created 2026-09-25 07:12Z, runtime 162,311 s).
- Images: ECR `arena-slime-dev` in ap-south-1, tags
  `gym-glm53-adr69-20260925a` and `miles-glm53-r12-20260925a`.
- Workflow journals: `wf_795e8c03-107` (131K overflow trace, verified),
  `wf_c2a543f0-f9c` (generation length, verified).
- ADRs: AREnATasks ADR-0063 (Vulcan compaction and segments), ADR-0069
  (training reward is the Harbor trial reward), ADR-0072 (limits from the
  task message); miles ADR-0015 (task message carries cap, window,
  sampling).
- Memory notes (facts with dates; numbers not re-verified unless stated):
  `r39-auctioneer-caponly-1034-2026-09-25`,
  `sglang-speed-vs-inference-hosting-2026-09-25`,
  `auct-131k-summary-overflow-2026-09-27`, `r44-r45-launch-plan-2026-09-27`,
  `glm53-length-explosion-ctrf`.
