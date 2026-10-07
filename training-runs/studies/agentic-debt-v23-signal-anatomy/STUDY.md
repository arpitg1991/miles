# Study: what the GRPO signal in guparpit-agentic-debt-v23 teaches

**Date:** 2026-10-07
**Run:** `guparpit-agentic-debt-v23-sw9v6` (dataset `agentic-final-v2`, lr 1.5e-6, 32 groups x 8, `max_weight_staleness 8`)
**Data:** 11,312 Harbor trials (28 weight versions) from
`s3://arena-scratch-prod-bom-ap-south-1/harbor-training/guparpit-agentic-debt-v23/`;
`result.json` per trial, plus `reward-details.json`, `test-stdout.txt`, and `trajectory.json` for 160
contested groups (2,560 files). Scripts: `/tmp/glm53f/v23t/analyze{1..9}.py`.

## Reward mechanics (measured, not assumed)

- Step reward `z_k = (passed_after - passed_before) / (total - passed_before)`: the fraction of tests that
  failed before the step and pass after it. Partial credit per step.
- A regression at step k gives `z_k = -(k-1)` and stops the chain. At step 1 the penalty is 0.
- Episode reward `R = mean(z_k)` over the steps that ran. Verified on 11,253 of 11,253 trials.
- Tests that an earlier step left failing stay in the denominator of every later step. One early miss is charged
  again at each later step unless the agent fixes those hidden tests without feedback.

## Where the tokens go

| Episode R | episodes | tokens | steps/episode | note |
| --- | ---: | ---: | ---: | --- |
| 1.0 | 46% | 41% | 5.1 | |
| [0.9, 1) | 11% | 17% | 9.8 | long chains with one partial step |
| (0, 0.9) | 22% | 26% | 5.9 | ran the full chain; 37/2,488 stopped early |
| 0 | 17% | 11% | 3.9 | 1,561/1,929 are regressions (chain stopped) |
| < 0 | 4% | 5% | 8.8 | regression after unfixed steps |

Step verdicts across all 64,392 steps: 82% pass, 5.6% own tests partially fixed, 4.6% own tests pass but carried
debt lowers the reward, 3.2% own tests 0 fixed, 2.8% penalised regression, 0.8% regression at step 1.

Inside the mid-range episodes: 56% of steps (47% of tokens) passed; 16% have their own tests passing with
carried debt; 13% partial; 11% zero; 3% carried debt only. Carried debt accounts for 24-44% of the mid-range
reward deficit (bounds from unknown fix order).

## What separates pass from non-pass at the same step in the same group

160 groups, 712 pass and 568 non-pass attempts at the contested step.

Process features are indistinguishable (means pass vs non-pass; share of groups where pass > non-pass):
turns 29.1 vs 28.3 (59%); bash calls 30.7 vs 29.7 (60%); test runs 5.6 vs 5.3 (61%); probes 11.4 vs 10.8 (52%);
verified after the last edit 87% vs 83% (31%); reasoning characters 66.8k vs 63.1k (57%). 99.7% of non-pass
steps end with `stop_reason=completed` and a confident submit summary.

The difference is in solution content against hidden tests:

| Family | share of non-pass attempts | examples |
| --- | ---: | --- |
| Own tests pass; carried debt only | 34% | `chinese-calendar` step 16: own 1/1, eff 2 -> 0.50 |
| Own tests partially pass (exact semantics) | 38% | `MetricSnapshotDataFrame` shape (9,6) vs (6839,6); `KeyError: 'timestamp'`; exact markdown string in `redlines`; `{}` vs populated dict in `BMS_BLE` |
| Own tests 0, hidden test file not collected (layout, names, deps) | 11% | `smooth/optimization.py` vs `smooth/optimization/run_optimization.py`; `job` package placed under `spotify/` not `apple/`; `jk_bms.py` vs `jikong_bms.py`; `retrying` shim written into site-packages; `pandas` import in an env without pandas |
| Own tests 0, tests ran and failed | 18% | wrong behaviour on the spec's edge cases |

The contrast lives in 366 (task, step) cells across 101 tasks. The 25 largest cells cover 22% of non-pass
attempts. Within one task the discriminating detail repeats across weight versions (same layout mistake at wv13
and wv17), so the lesson is task-specific, not a shared behaviour.

## Sign-wrong gradient mass

In groups with variance, by advantage sign and step verdict: 55.5% of tokens positive on a passed step;
17.2% negative on a failed step; **16.3% negative on a passed step; 10.9% positive on a failed step**.
27% of the gradient mass carries the wrong sign at step level because the episode advantage is broadcast to
every segment.

## Is anything learned?

Per-cell pass rate by weight-version quartile (1-7 | 8-14 | 15-21 | 22-28):
`smooth` step 1 0.10 | 0.29 | 0.50 | 0.46; `BMS_BLE` step 3 0.30 | 0.42 | 0.54 | -;
`requests-hawk` step 1 0.04 | 0.12 | 0.25 | 0.19; `prometheus` step 2 0.82 | 0.78 | 0.75 | 0.79;
`conda-package-handling` step 1 0.19 | 0.19 | 0.16 | 0.25; `ttfautohint` step 3 0.12 | 0.06 | 0.12 | 0.00.
Across 293 contested cells with 8+ attempts in both halves: pass rate 0.638 -> 0.659; 56 cells improved by
more than 0.1, 37 worsened. The run memorises task-specific details on some cells; the aggregate stays flat because
solved groups leave the batch through the zero-variance filter and the remaining contrast is per-task detail.

## Implications

1. Step-level credit: compute advantages per step from `z_k` against siblings at the same step; the episode mean
   broadcast puts 27% of the gradient on the wrong sign.
2. Remove carried debt from the step denominator or charge it once; 24-44% of the mid-range deficit is re-charged
   debt that the agent cannot see.
3. The learnable general lessons are few: follow the spec's file paths literally, do not add dependencies or edit
   the environment, do not declare "already implemented". Everything else is per-task hidden-test detail.

## Why good SD does not produce visible learning

**90% of the within-group variance comes from the 521 groups that contain a regression attempt.** Partial-credit
groups (451) supply 10%. A regression yields `R <= 0` against siblings near 0.9, so it dominates the normalised
advantage; a partial step moves `R` by 0.05-0.2.

The regression lesson is real and learnable. In 150 sampled regressed steps: the broken P2P test file name appears in
the agent transcript in 78% (visible test); the agent ran a broad `pytest` after its last edit in only 49%; in 47% its
own pytest output showed the test `FAILED` and it submitted anyway. Median 2 broken tests per regression.

The model does learn it, slowly, and the dashboard hides it:

| weight versions | regression episodes | all-1.0 groups | zero-variance groups | population mean R |
| --- | ---: | ---: | ---: | ---: |
| 1-7 | 21.8% | 8.1% | 11.3% | 0.695 |
| 8-14 | 21.3% | 10.4% | 14.2% | 0.707 |
| 15-21 | 20.8% | 11.7% | 15.1% | 0.716 |
| 22-28 | 16.6% | 20.9% | 26.4% | 0.755 |

Paired per-task mean R, late minus early, all trials: +0.029 mean, +0.015 median, 15 tasks improved by more than
0.1, 3 worsened (n=115). The trainer's `avg_reward` is the kept-group mean after the zero-variance filter removes
every all-1.0 group, so solved tasks leave the batch and the metric stays flat (0.696 -> 0.694) while the population
rises.

Why the learning is slow:

1. The regression episode stops early and carries fewer tokens than its 1.0 siblings. With per-token loss, 55% of the
   gradient mass is positive advantage on already-passing steps of long chains; the negative lesson sits on a short
   episode and lands on its earlier passing steps as well as the offending step.
2. 27% of the gradient mass has the wrong sign at step level (episode advantage broadcast to every segment).
3. The partial-credit 10% is task-specific hidden-test detail with identical agent behaviour on both sides: noise plus
   memorisation.
4. Step size: lr 1.5e-6, grad norm 0.013-0.018, `pg_clipfrac` 0 at every step, train-rollout log-prob gap 0.035.

## Is the trend real? (16,218 episodes, 1,715 groups, 121 tasks, weight versions 1-38; `ci.py`)

Weight versions above 38 are excluded: they are still filling and skew to short chains.

Block rates with 95% CIs (cluster bootstrap by group; episodes in one group share a task and a policy):

| weight versions | regression | success (R = 1) |
| --- | --- | --- |
| 1-7 | 21.8% [18.0, 25.7] | 43.8% [39.3, 48.5] |
| 8-14 | 21.5% [18.2, 24.9] | 43.5% [39.9, 47.2] |
| 15-21 | 21.3% [18.1, 24.8] | 45.3% [41.5, 49.2] |
| 22-28 | 19.8% [16.3, 23.2] | 46.4% [42.6, 50.4] |
| 29-35 | 19.3% [16.0, 22.9] | 48.7% [44.4, 52.8] |
| 36-38 | 18.0% [11.9, 24.8] | 55.2% [48.4, 62.3] |

Adjacent blocks overlap; no single block-to-block change is significant. The trend test is the right instrument:

| outcome | slope per 10 weight versions (task fixed effects) | 95% CI (cluster bootstrap) | permutation p |
| --- | --- | --- | --- |
| regression rate | -1.1 points | [-1.5, -0.6] | < 0.003 |
| success (R = 1) | +1.6 points | [+1.0, +2.2] | < 0.003 |
| mean reward | +0.015 | [+0.010, +0.020] | < 0.003 |

Paired per task, late (29-38) minus early (1-14), n = 114 tasks: regression -3.0 points [-5.0, -0.9], sign test 52 down
/ 31 up, p = 0.028; success +4.3 points [+1.9, +6.9], 65 up / 38 down, p = 0.010; mean reward +0.041 [+0.020, +0.063].

Conclusion: the learning is real and slow. About 1 point fewer regressions and 1.6 points more successes per 10 updates.
At 22 minutes per update, a drop from 22% to 10% regressions needs about 110 more updates (~40 hours).
