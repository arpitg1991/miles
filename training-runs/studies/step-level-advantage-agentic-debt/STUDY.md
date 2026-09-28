# Study: Step-level advantage (GiGPO style) for agentic debt on r45

**Date:** 2026-09-27
**Status:** Closed
**Question:** Does a per-step advantage (B, GiGPO style, or C, per-step only) add training signal over the miles episode-level GRPO (A) on the r45 agentic-debt v3 chains?
**Runs and data used:** r45 `rl-glm53f45-vvqhg` (experiment `rl-glm53f-adebt-v3-r45`, W&B run `ajsur4ej` in project `rl-glm53f-adebt-v3`, resumed from r43 `iter_0000039`); 137 finished groups (1,096 trials) read from the r45 gym pod trash dirs on 2026-09-27 21:36Z to 21:44Z; trainer sample summaries `s3://arena-scratch-prod-bom-ap-south-1/guparpit/debug/rl-glm53f-adebt-v3-r45/sample_summary/rollout_40.jsonl` to `rollout_49.jsonl` (320 trained groups); the `rl-glm53f45-vvqhg-trainer-worker-0` log for steps 44 to 49; dataset `/mnt/scratch-s3files-rw/guparpit/data/agentic-debt/20260923-v3-locked-oracle/manifest-le5.jsonl` (S3 `s3://arena-scratch-prod-bom-ap-south-1/guparpit/data/agentic-debt/20260923-v3-locked-oracle/manifest-le5.jsonl`, 139,999 bytes, 551 rows)
**Code SHAs:** miles `e0987aed6` (trainer image `miles-glm53-r14-20260927a`): `miles/ray/rollout/train_data_conversion.py` `_normalize_rewards_by_rollout` (line 248) and `miles/rollout/filter_hub/dynamic_sampling_filters.py` `check_reward_nonzero_std` (line 18); AREnATasks `538bc63` (gym image `gym-glm53-adr72-20260927a`): `src/amzn-arena-harbor/src/amzn_arena_harbor/gym_worker.py` `_DEADLINE_MARGIN_SECS = 300` (line 640) and `_sweep_trash` (line 731). This study produced no code change.
**Workflow ids:** `wf_11893a4c-2b9` (phases Extract, Analyze, Check; journal closed 2026-09-27 22:17Z)

## Method

### Terms

- **Advantage.** The number the trainer multiplies into the loss of each
  token. A positive value pushes the response up. A negative value pushes it
  down.
- **Group.** The 8 trials of one task (`n_samples_per_prompt` 8).
- **Zero-variance group.** All 8 trial rewards are equal. The filter
  `check_reward_nonzero_std` (float64 std > 1e-8) drops such a group. The
  trainer learns nothing from it.
- **Chain, step, K.** An agentic-debt task is a chain of K steps
  (`metadata.n_segments`, K = 2 to 5 in this dataset). The Harbor verifier
  grades each step and writes `segment_pass` (0 or 1) and `reward`.
- **Gate.** Each step has `min_reward = {segment_pass = 1.0}` and the task
  uses `multi_step_reward_strategy = "final"`. A trial stops at its first
  failed step. Its reward is the `reward` of the last step it reached, which
  is (steps passed) / K.
- **Step group `S_k`.** The trials of one group that reached step k.
- **Segment.** One trainer sample. A step yields 1 to 5 segments, one per
  context compaction.
- **GiGPO.** Group-in-Group Policy Optimization (Feng et al., 2025). It adds
  a step-level group advantage to the episode-level one.

### Data collection (Extract phase)

- The gym worker moves a finished job dir into `arena-rollout-trash`. On each
  new task message, `_sweep_trash` deletes entries 300 s or older
  (`gym_worker.py` lines 640 and 731). Thus the trash holds a short window
  of finished groups, not the run history.
- At 21:36Z on 2026-09-27, 118 of the 288 gym pods held trash (99 pods with
  1 group, 19 with 2). All 137 groups had 8 trials. Access was read-only:
  `kubectl get`, `logs`, and `exec` of `find`, `stat`, `grep`, `tar` to
  stdout, and a read-only Python script.
- 26 groups (208 trials, 23 pods) lost their `rollout.json` between the file
  snapshot (21:37Z) and the token scan (21:44Z). Their rewards and step
  results are complete; they have no token data (`rollout_scanned=false`).
- The groups finished between 10:54Z and 21:42Z; 96 of 137 finished at 20Z
  or 21Z. Group ids run from g119 to g1210. 129 distinct tasks; 8 tasks
  appear twice.

### Variants

- **Step reward.** `d_ik = segment_pass_ik / K`, and 0 for a step the trial
  did not reach. The Harbor `reward` key is a running total (k/K), so a sum
  over it counts earlier steps again. The analysis does not use it.
- **A (today).** `(r_i - mean) / (sample std + 1e-6)` in float32, the same
  value on every segment of the episode. A group with std <= 1e-8 is
  dropped. This is `_normalize_rewards_by_rollout` plus
  `check_reward_nonzero_std`. The live r45 trainer argv holds
  `--advantage-estimator grpo`, `--dynamic-sampling-filter-path
  miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`,
  `--n-samples-per-prompt 8`, `--rollout-batch-size 64`,
  `--global-batch-size 256`, `--lr 1.5e-06`, and none of
  `--disable-grpo-std-normalization`, `--disable-rewards-normalization`,
  `--normalize-advantages`, or a custom reward post-process path.
- **B (GiGPO style).** `B_ik = A_i + A_step(i, k)`. `A_step` applies the A
  normalization to the reward-to-go `G_ik = sum_{j >= k} d_ij` inside
  `S_k`. Step weight 1. A step group with fewer than 2 members or with
  std <= 1e-8 gets a step term of 0.
- **C (per-step only).** The A normalization applied to `d_ik` inside `S_k`,
  with no episode term.
- **Identity.** Under the gate, every member of `S_k` passed steps 1 to k-1.
  So `G_ik - mean_Sk(G) = r_i - mean_Sk(r)`. The script asserted this on
  every (trial, step) cell. The B step term is plain GRPO on the episode
  reward over `S_k`. At k = 1, B = 2A exactly.

### Verification (Check phase)

- A second agent re-ran `analysis.py`. Its output is byte-identical to
  `analysis_out.md` (`cmp` confirmed again on 2026-09-28).
- It read 4 groups (32 trials) and 19 `rollout.json` steps in the pods.
  Every step name, step reward, exception, trial reward, token length,
  `loss_mask` sum, `segment_end`, `stop_reason`, and `compactions` value
  matched the extracted data.
- It recomputed the headline counts from `trials.jsonl` with its own script
  (`indep.py`) and compared the sample against the trainer's own
  `sample_summary` records for rollouts 40 to 49.
- It confirmed the trainer image copies of `train_data_conversion.py` and
  `dynamic_sampling_filters.py` are byte-identical to the workspace files.

### Re-verification for this record (2026-09-28)

A fresh recount of `trials.jsonl` matched: 1,096 trials, 137 groups, 118
pods, K counts 47/40/30/20, stop causes 722/267/99/8, 70 zero-variance
groups of 136 scored (55 at 1.0, 12 at 0.0, 2 at 0.5, 1 at 1/3), window
10:54Z to 21:42Z, 129 tasks, strategy `final` on all 1,096 trials, gate
`{segment_pass: 1.0}` on every step. The manifest copy has 551 rows. The
trainer log lines for the queue sizes and the four `Dynamic sampling`
summaries matched. The per-segment and token numbers below rest on the
byte-identical re-run, not on a third recount.

## Results

### 1. Sample

| Item | Value | Source |
| --- | --- | --- |
| Trials / groups / pods with trash | 1,096 / 137 / 118 | `trials.jsonl` recount 2026-09-28 |
| Groups by K (2 / 3 / 4 / 5) | 47 / 40 / 30 / 20 | `trials.jsonl` recount |
| Manifest K mix (2 / 3 / 4 / 5), 551 rows | 182 / 139 / 122 / 108; chi-square p = 0.44 vs the sample | journal Check (p-value not re-verified) |
| Trials that passed all steps | 722 | `trials.jsonl` recount |
| Trials stopped by the gate before the last step | 267 | `trials.jsonl` recount |
| Trials that reached the last step and failed it | 99 | `trials.jsonl` recount |
| Trials with `EnvironmentStartTimeoutError` before step 1 (one group, K = 3) | 8 | `trials.jsonl` recount |
| Groups with token data / without | 111 / 26 | journal Extract |
| Trainable tokens (`loss_mask` sum) in scanned groups | 138.33 M | `RESULTS.md` section 3 |
| Trainable tokens per step: median / p90 / max | 40,961 / 124,631 / 382,079 | journal Extract |
| Segments per step (1 / 2 / 3 / 4 / 5) | 1,565 / 539 / 160 / 62 / 79 | journal Extract |

### 2. Episode level (136 scored groups)

| K | Groups | Zero variance | Mixed (trained) | Zero-variance reward values | Source |
| --- | --- | --- | --- | --- | --- |
| 2 | 47 | 26 (55.3%) | 21 | 0.0: 5, 0.5: 2, 1.0: 19 | `RESULTS.md` section 1 |
| 3 | 39 | 21 (53.8%) | 18 | 0.0: 3, 1/3: 1, 1.0: 17 | `RESULTS.md` section 1 |
| 4 | 30 | 14 (46.7%) | 16 | 0.0: 1, 1.0: 13 | `RESULTS.md` section 1 |
| 5 | 20 | 9 (45.0%) | 11 | 0.0: 3, 1.0: 6 | `RESULTS.md` section 1 |
| all | 136 | 70 (51.5%) | 66 (48.5%) | 55 of 70 at 1.0 | recount 2026-09-28 |

The 70 zero-variance groups hold 59.21 M of 138.33 M trainable tokens
(42.8%) (`RESULTS.md` section 3). With the all-failed group, 71 of 137
groups (51.8%) carry no gradient.

### 3. Step level (a step group is the set of trials that reached step k)

| k | Step groups | Size >= 2 | Variance under B | Variance under C | From zero-variance episode groups | Of those with variance (B / C) | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 136 | 136 | 66 (48.5%) | 32 (23.5%) | 70 | 0 / 0 | `RESULTS.md` section 2 |
| 2 | 148 | 120 | 40 (27.0%) | 21 (14.2%) | 58 | 0 / 0 | `RESULTS.md` section 2 |
| 3 | 97 | 77 | 21 (21.6%) | 10 (10.3%) | 36 | 0 / 0 | `RESULTS.md` section 2 |
| 4 | 56 | 42 | 13 (23.2%) | 11 (19.6%) | 19 | 0 / 0 | `RESULTS.md` section 2 |
| 5 | 25 | 15 | 4 (16.0%) | 4 (16.0%) | 6 | 0 / 0 | `RESULTS.md` section 2 |
| all | 462 | 390 | 144 (31.2%) | 78 (16.9%) | 189 | 0 / 0 | `RESULTS.md` section 2 |

The 462 step groups include 62 empty ones. Over the 400 non-empty groups,
the variance shares are 36.0% (B) and 19.5% (C) (journal Check). None of
the 189 step groups inside zero-variance episode groups has variance under
B or C. Inside the 66 mixed groups at k >= 2: 149 step groups, 14 of size 1
(step term 0), and a 2-member group always normalizes to +/-0.707.

### 4. Per segment (2,054 segments in the 55 mixed groups with token data)

| Pair | Segments | Sign differs | Strict flip (+ vs -) | A != 0 set to 0 | Tokens in strict flips | Source |
| --- | --- | --- | --- | --- | --- | --- |
| A vs B, all k | 2,054 | 37 (1.8%) | 37 (1.8%) | 0 | 0.6% | `RESULTS.md` section 3 |
| A vs B, k = 1 | 834 | 0 | 0 | 0 | 0.0% | `RESULTS.md` section 3 |
| A vs B, k >= 2 | 1,220 | 37 (3.0%) | 37 (3.0%) | 0 | 1.1% | `RESULTS.md` section 3 |
| A vs C, all k | 2,054 | 1,054 (51.3%) | 50 (2.4%) | 1,004 | 1.2% | `RESULTS.md` section 3 |
| A vs C, k >= 2 | 1,220 | 701 (57.5%) | 44 (3.6%) | 657 | 1.7% | `RESULTS.md` section 3 |

| Segments | corr(A, B) | corr(A, C) | mean abs A | mean abs B | mean abs C | Source |
| --- | --- | --- | --- | --- | --- | --- |
| all k | 0.978 | 0.705 | 0.729 | 1.312 | 0.384 | `RESULTS.md` section 3 |
| k >= 2 | 0.965 | 0.680 | 0.716 | 1.188 | 0.345 | `RESULTS.md` section 3 |

Only 5 of the 66 mixed groups have any A-vs-B sign flip. On (trial, step)
cells over all 66 mixed groups (n = 1,417): B flips 13 (0.9%), C flips 24
(1.7%), C zeroes 828 (58.4%) (`RESULTS.md` section 3).

### 5. Tokens (138.33 M trainable, all scanned groups)

| Variant | Tokens with nonzero advantage | Share | abs(adv) x tokens, relative to A | Source |
| --- | --- | --- | --- | --- |
| A | 79.12 M | 57.2% | 1.00 | `RESULTS.md` section 3 |
| B | 79.12 M | 57.2% | 1.83 (1.87 with the miles per-episode loss weight) | `RESULTS.md` section 3; journal Check |
| C | 43.67 M | 31.6% | 0.57 (0.63 with the per-episode weight) | `RESULTS.md` section 3; journal Check |

Tokens that B trains and A does not: 0. Tokens that C trains and A does
not: 0. C sets 45% of the tokens that A trains to zero (79.12 M to 43.67 M).

### 6. Sign against the step's own outcome (mixed groups)

| Variant | Failed-step segments with advantage > 0 | Passed-step segments with advantage < 0 | Source |
| --- | --- | --- | --- |
| A | 110 of 510 (21.6%) | 219 of 1,544 (14.2%) | `RESULTS.md` section 3 |
| B | 75 of 510 (14.7%) | 221 of 1,544 (14.3%) | `RESULTS.md` section 3 |
| C | 0 of 510 (0.0%) | 0 of 1,544 (0.0%) | `RESULTS.md` section 3 |

Worked examples (`RESULTS.md` section 4): in g1044 (`realdiff_mcurlej_joft_s0`,
K = 5) trial `xVEJGPg` passed steps 1 to 3 and failed step 4. A gives
+0.59 on all steps. B gives +1.18 on step 1 and -0.56 on steps 2 to 4, so
B blames steps the trial passed. C gives 0 on steps 2 and 3 and -1.15 on
step 4. In g1047 (`realdiff_tornede_py_experimenter_s0`) both trials that
reached step 5 failed it; B keeps A's +1.35 on that failed step, C gives 0.

### 7. Start-state check (k >= 2)

| Item | Value | Source |
| --- | --- | --- |
| Peer slots in step groups of size >= 2 | 1,831; 100% passed every earlier step | `RESULTS.md` section 5 |
| Cells where an earlier-outcome match changes B | 0 | `RESULTS.md` section 5 |
| Step groups whose members share the step-k prompt length | 319 of 320 (99.7%) | `RESULTS.md` section 5 |
| Median CV of earlier-step output tokens, k = 2 / 3 / 4 / 5 | 0.30 / 0.27 / 0.23 / 0.20 | `RESULTS.md` section 5 |
| AUC "the passer used fewer earlier tokens", k = 2 / k = 3 | 0.62 (13 vs 8 groups) / 0.61 (7 vs 2) | `RESULTS.md` section 5 |

The context resets at each step, so the prompt matches. The repository
state that earlier steps leave behind differs. A group by step index is not
a true same-state match. The AUC counts are small; treat them as weak
evidence.

### 8. Is the trash sample fair? (against the trainer's own records)

| Check | Trash sample | Trainer | Source |
| --- | --- | --- | --- |
| Zero-variance group rate | 51.5% (70 of 136; SE 4.3 points) | 55.9% (162 of 290 examined, rollouts 46 to 49) | recount; trainer log `Dynamic sampling` lines |
| Logged drop reasons | 12 of 70 at 0.0 | 24 of 27 logged at 1.0, 0 at 0.0 (sampled log lines) | trainer log `dyn-sampling drop` lines |
| Group mean reward, KS test | p = 0.87 | 320 trained groups, rollouts 40 to 49 | journal Check (not re-verified) |
| Samples per episode, KS test | p = 0.84 | same | journal Check (not re-verified) |
| Groups with more than 2 reward values | 15.2% | 18.4% | journal Check (not re-verified) |
| Groups where B flips any sign | 7.6% | 4.1% | journal Check (not re-verified) |

The 0-at-0.0 gap is likely timing: 9 of the 12 all-0.0 groups finished
after 20:38Z, and the trainer had not consumed them (journal Check,
unverified). If anything, the full run has fewer B flips than the sample.

### 9. r45 trainer state at the time of the sample (side findings)

| Item | Value | Source |
| --- | --- | --- |
| Finished-group queue at the start of rollouts 46 / 47 / 48 / 49 | 243 / 280 / 319 / 320 (cap 320) | trainer log `Rollout N: collecting` lines |
| Groups examined per step (kept 32 of 32) | 72 / 68 / 72 / 78 (cap 128) | trainer log `Dynamic sampling` lines |
| Age of the training data | about 4 steps | journal Check (derived: 320 queued / about 72 consumed per step) |
| Gym pods with no live job at 21:36Z | 68 of 288 | journal Check (not re-verified) |
| `train/grad_norm`, steps 44 to 47 | 0.0485 / 0.0372 / 0.0799 / 0.0443 (clip 1.0 never fires) | trainer log step lines |
| Trained groups with reward-0 crash-padding copies, rollouts 40 to 49 | 5 of 320; 1 group trainable only because of the copies | journal Check (not re-verified) |

## Verdict

On r45 agentic debt v3, B and C add no new training signal over A. The gate
makes the episode reward fix the whole pass pattern, so a step group holds
no fact that the episode reward does not. B and C recover none of the 70
zero-variance groups and train no token that A skips. B is close to a
rescaled A: corr 0.978, 1.83x the advantage mass (2A at step 1), and a
sign flip on 37 of 2,054 segments (0.6% of tokens) in 5 of 66 groups. Adam
cancels most of a uniform scale, and the gradient clip never fires, so the
net effect of B is a small shift of loss weight away from steps 2 and later
(42.7% instead of 46.3%) plus those few flips. C alone matches the sign to
the step's own result (it removes the 21.6% and 14.2% wrong-sign rates),
but it zeroes 45% of the tokens A trains, drops all cross-step credit, and
still pays the forward and backward pass for the zeroed segments. The real
limit is the 51 to 56% of groups with zero variance, and 55 of the 70 in
the sample (79%) are solved by all 8 trials. A step-level group cannot
change that. Decision: no A/B run of B or C on r45.

## Caveats and open items

- The sample is 136 scored groups from a 300 s trash window, 25 of them
  without token data. The per-segment numbers rest on 2,054 segments in 55
  mixed groups. The K mix and the zero-variance rate match the trainer's
  own records (section 8), so the sample is fair for this question.
- The analysis divides C by the step-group std, which the task text did not
  specify. Signs and zero counts do not depend on that choice; C's mass
  (0.57x) does.
- The 1.8x scale of B is not a learning-rate change under Adam. The Analyze
  phase called it one; the Check phase corrected that (journal Check).
- Late-step groups are small: at k >= 2, 14 of 149 mixed step groups have
  one member, and a 2-member group always gives +/-0.707.
- The start-state AUC (0.62 and 0.61) rests on 13 vs 8 and 7 vs 2 groups.
  It is weak evidence that a smaller earlier-step footprint helps the next
  step.
- To build B or C in miles, `_normalize_rewards_by_rollout` raises
  `ValueError` when segments of one rollout carry different rewards
  (`train_data_conversion.py` line 276). A variant needs
  `custom_reward_post_process_func` plus a step index on each `Sample`;
  today the metadata holds only `segment` and `n_segments`.
- The scratch files (`/tmp/stepadv/`: `trials.jsonl`, `rollouts_scan.jsonl`,
  `analysis.py`, `RESULTS.md`, `analysis_out.md`, `verify/`) are not in git, on
  tmpfs, present at 16:35Z 2026-09-28, and not on S3. The workflow journal holds
  the same tables.
- Not tested: a step weight below 1 for B, a mixed "A plus C" variant, and
  any variant on a dataset without the stop-on-fail gate. Open only if a
  future dataset drops the gate or uses `multi_step_reward_strategy = "mean"`.

## Actions taken

- No config change, image, ADR, or run came from this study. r45 kept
  episode-level GRPO with `check_reward_nonzero_std` until its retirement on
  2026-09-28 09:23Z (RUNLOG line 2323).
- The follow-up direction stated in the verdict, harder or filtered tasks,
  matches the later dataset move: r47 runs on agentic-debt-766
  (`r47/BUILD.md` section Dataset). That record does not cite this study,
  so no causal link is claimed. The 766-task format keeps `final` and the
  `{segment_pass = 1.0}` gate, so the gate identity above holds for r47
  too.
- Side findings on the full trainer queue (data about 4 steps old, about a
  quarter of gym pods idle) went to the r45 throughput work
  (`wf_545a45a0-6ca`, memory note `r45-mfu-investigation-2026-09-27.md`).

## Sources

- Workflow journal `wf_11893a4c-2b9`: Extract, Analyze, and Check results,
  2026-09-27 21:35Z to 22:17Z.
- Scratch (not in git): `/tmp/stepadv/RESULTS.md`, `analysis_out.md`,
  `analysis.py`, `trials.jsonl`, `rollouts_scan.jsonl`,
  `verify/verify_rerun.md`, `verify/indep.py`, `verify/cmd1912.txt` (trainer
  argv), `verify/trainer_full.log` (worker-0 log excerpt),
  `verify/manifest-le5.jsonl`, `verify/sample_summary/rollout_40..49.jsonl`.
- S3: `s3://arena-scratch-prod-bom-ap-south-1/guparpit/debug/rl-glm53f-adebt-v3-r45/sample_summary/rollout_40.jsonl`
  to `rollout_55.jsonl` (listed 2026-09-28);
  `s3://arena-scratch-prod-bom-ap-south-1/guparpit/data/agentic-debt/20260923-v3-locked-oracle/manifest-le5.jsonl`.
- W&B: run `ajsur4ej`, project `rl-glm53f-adebt-v3` (r45).
- Run records: `training-runs/harbor-rl-glm53-flash/r45/BUILD.md`,
  `r45/miles-config.yaml` (lines 231, 265, 266, 283, 402, 450, 541, 547),
  `RUNLOG.md` lines 2189 (r45 launch) and 2323 (r45 retired).
- Code: miles `e0987aed6` `miles/ray/rollout/train_data_conversion.py`
  lines 248 to 296; `miles/rollout/filter_hub/dynamic_sampling_filters.py`
  lines 18 to 24; `miles/utils/arguments.py` lines 1506 and 1512
  (`grpo_std_normalization`, `rewards_normalization`). AREnATasks `538bc63`
  `gym_worker.py` lines 640 and 731.
- ADRs: AREnATasks ADR-0065 (chain steps ride the rollout segment list),
  ADR-0069 (the training reward is Harbor's trial reward; the `final` plus
  `{segment_pass = 1.0}` gate), ADR-0070 (the Harbor reward key is `reward`).
- Memory note `r45-mfu-investigation-2026-09-27.md`, step-level entry
  (a summary of the journal; every number in this record comes from the
  journal or a primary source above).
