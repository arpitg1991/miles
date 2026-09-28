# Study: Token-level loss average against the per-sample mean (r44 vs r46)

**Date:** 2026-09-27 (A/B launch); numbers read 2026-09-28 16:00Z
**Status:** Open (both runs still run; last complete save `iter_0000049` on each)
**Question:** Does `calculate_per_token_loss: true` (one weight per trained token) change the auctioneer reward or the episode length against the miles default (one weight per rollout), at otherwise equal settings?
**Runs and data used:** r44 `rl-glm53f-auct-cap-r44` (Argo `rl-glm53f44-lt8vm`, then resume `rl-glm53f44-qvjnv`; W&B `arena/rl-glm53f-auct-cap/82i3g7lv`) and r46 `rl-glm53f-auct-cap-r46` (`rl-glm53f46-clhv6`, then `rl-glm53f46-rxn6b`; W&B `4iu54zov`). Dataset `lakefs://arena-inspect/dev/internal/auctioneer/caponly/caponly-1034/manifest.jsonl`, gym `auctioneer-caponly`. S3 under `s3://arena-scratch-prod-bom-ap-south-1/guparpit/`: `debug/rl-glm53f-auct-cap-r4{4,6}/sample_summary/rollout_N.jsonl` (r44 N=0..54, r46 N=0..56), `logs/rl-glm53f-auct-cap-r4{4,6}/trainer-0-attempt1.log`, `checkpoints/slime_experiments/rl-glm53f-auct-cap-r4{4,6}/rollout/arena_data_source_state_{9,19,29,39,49}.pt` (r46 has no `_9`).
**Code SHAs:** miles `e0987aed6` (trainer image `miles-glm53-r14-20260927a`, launch); `bc31f88ac` (image `miles-glm53-r15-20260927a`, resume; HF export opt-in); `aa8cd8e70` (prepare r46); `662e0cc95` (launch r46); `3028bc358` (prepare the two resumes). AREnATasks `538bc63` (gym image `gym-glm53-adr72-20260927a`, both runs).
**Workflow ids:** `wf_128c0acc-6c4` (r46 launch and first-step check), `wf_2776ce0d-bc7` (HF-export-off, 8-engine resumes, r46 sidecar repair). The reward, length, and replay analysis below has no journal; the scratch under `/workplace/guparpit/kdfast/scratch/tokloss/` was deleted after this record.

## Method

**The two normalizers.** The miles default (`sum_of_sample_mean`) divides
the loss of each rollout by its own token count (`rollout_mask_sums`), then
divides the sum by the rollout count. `--calculate-per-token-loss` makes
`get_sum_of_sample_mean` return the token sum
(`miles/backends/training_utils/cp_utils.py:98-110`), and `loss.py` returns
the token count as the Megatron normalizer instead of `1`
(`miles/backends/training_utils/loss.py:165-166,213,229`). Megatron then
divides the gradients by the token count reduced over DP. The launcher emits
the bare flag from the config key (`r46/BUILD.md`, "Code check"). The
argument is DAPO (arXiv 2503.14476, token-level policy gradient loss) and
Dr. GRPO (arXiv 2503.20783, length bias of the per-sample mean): with the
per-sample mean, each token of a long wrong episode gets a small share of the
gradient. `lr` stays 1.5e-6 on both runs. Upstream miles precedent:
`examples/experimental/DrGRPO/custom_reducer.py` (constant divisor).

**The A/B.** r46 is r44 with one functional change. The workflow parameters
differ only in `experiment-name` and `miles-config`; the config differs in
the three name lines, the sample-summary dir, and
`calculate_per_token_loss: true` (`diff r44/miles-config-r0.yaml
r46/miles-config-r0.yaml`; `r46/BUILD.md` "Diff check": 174 argv tokens
against 173). Same trainer image, gym image, template
`guparpit-miles-deployer-v10`, dataset, `rollout_seed` 42, `seed` 1234,
`rollout_batch_size` 64, `n_samples_per_prompt` 8, `global_batch_size` 256,
`rollout_max_response_len` 16384, `agent-timeout-multiplier` 2 (auctioneer
agent limit 7,200 s x 2 = 14,400 s; `r46/miles-config-r0.yaml:30-31`),
`arena_inflight_multiplier` 4, dynamic sampling `check_reward_nonzero_std`,
`kl_coef` 0, `kl_loss_coef` 0. W&B config confirms
`calculate_per_token_loss` False on `82i3g7lv` and True on `4iu54zov`.

**Both runs changed shape at rollout 10.** The idle-GPU reaper deleted both
launch workflows (r44 10:02:37Z, r46 11:02:24Z) after the step-9 save. The
resumes (14:16Z, image r15, `replicas` 16) run 8 SGLang engines instead of
32 and no per-save HF export (`r44/BUILD.md`, `r46/BUILD.md` "Resume 1").
The r46 delete cut the step-9 save before `slime_extra_state.json` and
`rollout/arena_data_source_state_9.pt` were written. The operator wrote
only the sidecar; the r46 data source therefore started at offset 0 of the
`shuffle(0)` order again, while r44 continued at offset 937 of 1034 with
352 consumed ids in its skip guard. Thus the comparison has three parts:
rollouts 0-9 (clean, 32 engines), rollouts 10-49 (8 engines, r46 replay),
and rollouts 50+ (8 engines, both in a later epoch).

**Measurements.**

- Reward and length per rollout: `sample_summary/rollout_N.jsonl`, one row
  per trained sample (`reward`, `response_length`, `stop_reason`,
  `status`). Per-rollout means, then means over 5-rollout bins. The W&B
  `rollout/episode_raw_reward` agrees with the sample_summary mean within
  0.016 (r44) and 0.009 (r46) at every step. `stop_reason` is the gym `agent_stop_reason`
  (`miles_plugins/arena/nats_arena/nats_rollout.py:2364`); `timeout` maps
  from Harbor `AgentTimeoutError` (AREnATasks
  `amzn_arena_harbor/gym_worker.py:340-344`); `None` means the agent
  finished on its own.
- Turns and per-turn length: W&B `rollout/group_metrics/num_turns.mean` and
  `rollout/group_metrics/completion_length.turn.mean`; episode length also
  as W&B `rollout/episode_response_length/mean`.
- Prompt identity per trained group: `metadata[rollout]` of the data-source
  state files (32 prompt ids per rollout, completion order) and the
  `rollout_tasks:` lines of the attempt-1 logs (rollouts 0-10 r46, 0-11
  r44). The r44 rollout-0 `metadata` list equals the log list (order and
  set). Groups were joined to prompt ids through `prompt_reward` equality
  (exact float match; 32/32 groups on most resume rollouts, 22-30/32 on
  r44 rollouts 40-50). The attempt-1 files were joined by position; that
  join checked 251/320 against r44 `state_9.prompt_reward`, so the
  attempt-1 per-prompt rewards are approximate.
- Exposure count: how many times a prompt had been trained (attempt 1 plus
  resume) at the time of a rollout. Both runs have 1,632 training events
  through rollout 49 (r44 through 50).

## Results

### Settings that differ between the arms (all from the run records)

| Setting | r44 | r46 |
| --- | --- | --- |
| `calculate_per_token_loss` | off (default `sum_of_sample_mean`) | `true` (`--calculate-per-token-loss`) |
| Loss normalizer per step | per-rollout mean, then mean over rollouts | one token sum over all trained tokens |
| Data position at the resume (rollout 10) | offset 937/1034, epoch 0, 352 ids skipped (`state_9.pt`) | offset 0, no skip guard (`r46/BUILD.md`; `state_19.pt` offset 767, epoch 0) |
| First rollout with epoch-1 ids | 10 (`state_19.pt`: offset 671, epoch 1) | 41 (`state_39.pt`: offset 244, epoch 1) |
| First rollout with epoch-2 ids | 44 (`state_39.pt`: offset 149, epoch 2) | none through 49 |
| SGLang engines | 32 to rollout 9, then 8 | 32 to rollout 9, then 8 |

### Reward, length, and stop reason per 5-rollout bin

Reward and `response_length` (sample_summary; all trained samples). Turns
and per-turn completion tokens (W&B group metrics). Timeout share = share
of trained samples with `stop_reason` `timeout` (W&B `rollout/stop/timeout`
gives the same values).

| Bin | r44 reward | r46 reward | r44 resp. tokens | r46 resp. tokens | r44 timeout | r46 timeout | r44 turns | r46 turns | r44 tok/turn | r46 tok/turn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0-4 | 0.354 | 0.345 | 46,715 | 46,069 | 0.00 | 0.00 | 69.7 | 69.3 | 256 | 253 |
| 5-9 | 0.347 | 0.364 | 47,412 | 47,592 | 0.00 | 0.00 | 71.0 | 70.9 | 265 | 267 |
| 10-14 | 0.429 | 0.492 | 45,967 | 39,970 | 0.94 | 0.75 | 47.1 | 57.7 | 513 | 250 |
| 15-19 | 0.462 | 0.411 | 47,224 | 39,881 | 0.93 | 0.79 | 47.5 | 58.3 | 533 | 248 |
| 20-24 | 0.476 | 0.414 | 46,189 | 39,199 | 0.95 | 0.76 | 45.5 | 57.2 | 546 | 246 |
| 25-29 | 0.482 | 0.499 | 49,574 | 39,656 | 0.96 | 0.74 | 43.7 | 55.9 | 655 | 273 |
| 30-34 | 0.456 | 0.503 | 50,102 | 40,228 | 0.97 | 0.74 | 41.7 | 55.4 | 714 | 292 |
| 35-39 | 0.539 | 0.564 | 54,641 | 42,114 | 0.98 | 0.78 | 39.0 | 52.8 | 918 | 363 |
| 40-44 | 0.523 | 0.600 | 57,824 | 41,586 | 0.98 | 0.70 | 35.3 | 52.2 | 1,148 | 363 |
| 45-49 | 0.477 | 0.613 | 61,206 | 42,228 | 0.99 | 0.76 | 33.4 | 51.1 | 1,326 | 396 |
| 50-54 | 0.461 | 0.703 | 63,751 | 39,911 | 0.98 | 0.41 | 30.0 | 53.8 | 1,607 | 328 |
| 55-56 | - | 0.658 | - | 40,932 | - | 0.50 | - | 53.2 | - | 350 |

Per-rollout extremes after the resume (sample_summary): r44 low 0.361 at
rollout 14, peak 0.598 at rollout 44, then 0.393 at rollout 50; r46 low
0.335 at rollout 23, peak 0.789 at rollout 50.
The memory note of 2026-09-28 read "r44 fell to 0.42 by 50-52" (0.393,
0.447, 0.415); rollouts 53-54 (0.552, 0.495) arrived later and lift the bin
to 0.461.

Reward by stop reason (sample_summary): in r46 the samples that finished on
their own score higher than the timed-out ones in every bin from 10 on
(50-54: 0.785 against 0.589; 25-29: 0.619 against 0.456). In r44 the
non-timeout samples are too few for a comparison: 35, 31, 20, 9, and 20
per 5-rollout bin from 30-34 on, out of about 1,280.
Across all samples the reward-length correlation is 0.041 (r44) and -0.027
(r46).

### Reward by prompt exposure (first, second, third visit)

Exposure 1 = the prompt had not been trained before in this run. r46
rollouts 10-17 trained only prompts that attempt 1 had already trained
(replay share 1.00); rollouts 18-19 0.78-0.81; 20-22 0.56, 0.34, 0.34;
23-24 0.06; 25-41 0.00 (28: 0.03); 42-49 (epoch 1) 0.16-0.50. The r44
resume rollouts mix epoch-0 tail and epoch-1 prompts from rollout 10 on.

| Bin | r44 visit 1 (n) | r44 visit 2 (n) | r44 visit 3+ (n) | r46 visit 1 (n) | r46 visit 2 (n) | r46 visit 3+ (n) |
| --- | --- | --- | --- | --- | --- | --- |
| 0-4 | 0.354 (160) | - | - | 0.345 (160) | - | - |
| 5-9 | 0.350 (160) | - | - | 0.366 (160) | - | - |
| 10-14 | 0.415 (101) | 0.450 (38) | - | - | 0.492 (160) | - |
| 15-19 | 0.427 (96) | 0.519 (63) | - | 0.297 (13) | 0.421 (147) | - |
| 20-24 | 0.496 (87) | 0.452 (73) | - | 0.418 (116) | 0.402 (44) | - |
| 25-29 | 0.504 (85) | 0.457 (75) | - | 0.500 (159) | 0.370 (1) | - |
| 30-34 | 0.425 (88) | 0.491 (71) | - | 0.503 (160) | - | - |
| 35-39 | 0.513 (101) | 0.572 (56) | - | 0.564 (160) | - | - |
| 40-44 | 0.523 (73) | 0.592 (45) | 0.724 (2) | 0.572 (54) | 0.594 (42) | 0.648 (44) |
| 45-49 | 0.536 (5) | 0.481 (77) | 0.455 (64) | - | 0.597 (111) | 0.650 (49) |

Paired gain on the same prompt, second visit minus first visit (attempt-1
side approximate, see Method): r44 +0.03 to +0.21 per bin (n 33-70), r46
+0.08 to +0.12 (n 23-159). A repeat visit is worth about +0.1 reward in
both arms. Rollouts 50-56 cannot be split by exposure yet: the next state
file (`_59.pt`) does not exist.

### Train metrics

Step 0 (RUNLOG r46 entry): `pg_clipfrac` 0.000262 / 0.000268,
`ppo_kl` 2.35e-5 / -3.75e-5, `grad_norm` 0.140 / 0.120, `ess_ratio` 1.015 /
0.999, `train_rollout_kl` 0.0034 / 0.0041 (r44 / r46). `pg_loss` is not
comparable: r44 1.42e-5, r46 -0.0183 (token-weighted; it moves away from
zero when advantage and length correlate). On r46 `pg_clipfrac`, `ppo_kl`,
`ess_ratio`, and `entropy_loss` are token-weighted means; on r44 they are
per-rollout means (`r46/BUILD.md` "Metrics").

| Bin (W&B, 5-step means) | r44 `grad_norm` | r46 `grad_norm` | r44 `pg_clipfrac` | r46 `pg_clipfrac` | r44 `train_rollout_kl` | r46 `train_rollout_kl` | r44 `actor_train_time` s | r46 `actor_train_time` s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0-4 | 0.188 | 0.117 | 2.76e-4 | 2.56e-4 | 0.0048 | 0.0049 | 1,177 | 1,192 |
| 5-9 | 0.128 | 0.115 | 2.73e-4 | 2.42e-4 | 0.0079 | 0.0074 | 719 | 740 |
| 20-24 | 0.104 | 0.134 | 2.68e-4 | 2.60e-4 | 0.0111 | 0.0133 | 609 | 549 |
| 35-39 | 0.086 | 0.112 | 1.61e-4 | 2.97e-4 | 0.0100 | 0.0131 | 773 | 599 |
| 50-54 | 0.071 | 0.127 | 1.18e-4 | 3.27e-4 | 0.0059 | 0.0120 | 839 | 522 |

r44 `grad_norm` and `pg_clipfrac` fall over the run; r46 holds both flat.
`train/entropy_loss` reads 0 on every step of both runs (the metric is
inert here, not a result). The r44 `grad_norm` 0.428 at step 2 is a single
step and lifts the 0-4 bin.

### Timeline (UTC)

| Time | Event | Source |
| --- | --- | --- |
| 2026-09-27 05:07:02 | r44 created (`rl-glm53f44-lt8vm`), 32 engines | RUNLOG r44 entry |
| 06:15:41 | r46 created (`rl-glm53f46-clhv6`), 32 engines | RUNLOG r46 entry |
| 09:33:41 / 10:44:56 | step-9 DCP save complete, r44 / r46 | `r4{4,6}/BUILD.md` |
| 10:02:37 / 11:02:24 | reaper deletes r44 / r46 (r46 save cut before sidecar and data state) | `r4{4,6}/BUILD.md` |
| 14:07:30 | r46 sidecar written by the operator; no data-source state | `r46/BUILD.md` |
| 14:16:02 / 14:16:04 | resumes created `rl-glm53f44-qvjnv` / `rl-glm53f46-rxn6b`, 8 engines, image r15 | `kubectl get workflows` (read 2026-09-28) |
| 23:19 / 23:05 | `arena_data_source_state_19.pt` r44 / r46 | S3 listing |
| 2026-09-28 15:01 / 13:53 | `arena_data_source_state_49.pt` r44 / r46 (last save) | S3 listing |
| 2026-09-28 15:58 / 15:41 | r44 rollout 54 / r46 rollout 56 sample summaries; both `Running` | S3 listing, `kubectl` |

## Verdict

Promising, not clean. In the only clean window (rollouts 0-9, 32 engines,
no timeouts, same data positions) the two arms are equal: reward 0.35-0.36,
46-48k response tokens, 70 turns of about 260 tokens. Every difference
appears after the resume at rollout 10, and three things changed there at
once: the engine count (32 to 8), the r46 data order (a replay of the
`shuffle(0)` order from offset 0), and the stop regime (the 14,400 s agent
timeout cut 93-99% of r44 samples and 70-79% of r46 samples). Inside that
regime the token-level arm kept 52-58 short turns (250-400 tokens) per
episode while the per-sample arm drifted to 30-47 long turns (513 to 1,607
tokens) and its episodes grew from 46k to 64k tokens. That is the direction
the DAPO and Dr. GRPO argument predicts. The r46 reward dip at rollouts
15-24 sits inside the replay window and on second-visit prompts (0.42,
0.40); the climb at rollouts 25-39 (0.50 to 0.56) is on first-visit
prompts, and r46 first-visit reward exceeds r44 first-visit reward from
bin 30-34 on (0.503 against 0.425; 0.564 against 0.513; 0.572 against
0.523). On like-for-like repeat visits at rollouts 40-49, r46 also leads
(second visit 0.594-0.597 against 0.592-0.481; third 0.648-0.650 against
0.455). The headline 0.70 against 0.46 at rollouts 50-54 mixes exposures
and stop regimes and overstates the effect. The numbers support "the
token-level average did no harm and kept turns short under a wall-clock
cut"; they do not yet support "the token-level average raises auctioneer
reward" as a loss-function result.

## Caveats and open items

- The engine change is a confound: after rollout 10 the trainer waits 1.5
  to 4 h for the gym every 8 rollouts (W&B `perf/rollout_time` 5.6e3 to
  1.5e4 s at r44 steps 10, 17, 25, 33, 41, 49). Episodes live about 4 h
  in flight and the Harbor agent timeout cuts most of them. A cut episode
  keeps its auctioneer score; its token length depends on the generation
  speed, not only on the policy. Shorter turns (r46) finish more auctions
  before the cut. Cause and effect cannot be separated in this data.
- The data-order confound is quantified above (replay share per rollout;
  reward by exposure) but the attempt-1 per-prompt rewards are from a
  positional join that matched 251/320 on r44; treat the paired gains as
  approximate.
- Rollouts 50-56 have no state file yet, so their exposure split is
  unknown. r46 bins 50-54 and 55-56 are epoch-1 repeat visits by
  construction (epoch 1 started at rollout 41).
- The r44 publisher drew 512 groups between the step-39 and step-49 saves
  (`sample_group_index` 2217 to 2729) against 256 in each earlier window;
  W&B shows no dropped or failed groups there. Not explained.
- Two length metrics disagree in scale: sample_summary `response_length`
  (46k at rollout 0) and W&B `rollout/episode_response_length/mean`
  (16.8k). Both move in the same direction after the resume for r46
  (shorter) and r44 (longer). The definition gap is not resolved here.
- Single seed, one dataset, one `lr`. No held-out eval ran on either run
  (`eval_interval` unset). `entropy_loss` is not logged (reads 0).
- The r46 `pg_loss`, `pg_clipfrac`, `ppo_kl`, and `ess_ratio` are
  token-weighted, so their trends against r44 partly reflect the weighting.
- Numbers from the memory notes (0.54 peak at 35-39, 0.41 dip at 15-24,
  0.70 at 50-54) re-verified against the sample summaries. "0.42 at 50-52"
  re-verified as of rollout 52; the bin now reads 0.461.

## Actions taken

- r46 launched as the A/B arm on 2026-09-27 (miles `aa8cd8e70`,
  `662e0cc95`; `wf_128c0acc-6c4`). Both runs resumed with 8 engines and
  no HF export on 2026-09-27 14:16Z (`3028bc358`; `wf_2776ce0d-bc7`).
- The recipe default stays `calculate_per_token_loss` off. r47
  (agentic-debt-766, 2026-09-28) runs without the flag (RUNLOG r47 entry,
  trainer argv check).
- Proposed, not done: rerun the A/B at one engine count from step 0, with
  the data-source state written before any export (the HF-export-off
  default now gives that order), with an in-training eval on held-out
  prompts, and with an agent timeout that the 8-engine generation speed
  does not reach; or read the first-visit reward only, as in the exposure
  table, when the runs finish.

## Sources

- `training-runs/harbor-rl-glm53-flash/r44/BUILD.md`, `r46/BUILD.md`,
  `r44/miles-config-r0.yaml`, `r46/miles-config-r0.yaml`,
  `r46/workflow.yaml` (settings, resume checkpoints, sidecar repair).
- `training-runs/harbor-rl-glm53-flash/RUNLOG.md` entries "2026-09-27 r44
  launch", "2026-09-27 r46", "2026-09-28 r47" (launch times, step-0
  metrics, argv checks).
- `miles/backends/training_utils/cp_utils.py:98-110`,
  `miles/backends/training_utils/loss.py:165-166,213,229`,
  `miles/utils/arguments.py:1385`,
  `miles_plugins/arena/nats_arena/nats_rollout.py:2364`,
  `miles_plugins/arena/nats_arena/data_source.py:435-595`; AREnATasks
  `src/amzn-arena-harbor/src/amzn_arena_harbor/gym_worker.py:340-344`.
- S3 `guparpit/debug/rl-glm53f-auct-cap-r44/sample_summary/` and
  `.../r46/sample_summary/` (55 and 57 files); `guparpit/logs/...`
  `trainer-0-attempt1.log` (both); `guparpit/checkpoints/slime_experiments/
  rl-glm53f-auct-cap-r4{4,6}/rollout/arena_data_source_state_*.pt`.
- W&B `arena/rl-glm53f-auct-cap` runs `82i3g7lv` (658 rows, steps 0-54)
  and `4iu54zov` (688 rows, steps 0-56), read with the API on 2026-09-28.
  Steps 10-11 hold two points each (first attempt and resume); the last
  point was used.
- Journals `wf_128c0acc-6c4`, `wf_2776ce0d-bc7`; memory notes
  `r44-r45-launch-plan-2026-09-27.md` (user request of 2026-09-28 to
  quantify the replay), `thanatos-idle-gpu-reaper-prod-bom-v2.md`,
  `glm53-length-explosion-ctrf.md` (earlier length-bias hypothesis on
  r21-r27).
- DAPO arXiv 2503.14476; Dr. GRPO arXiv 2503.20783.
