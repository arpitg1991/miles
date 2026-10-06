# Run record: acuadron-agentic-debt-final-v6 — final-v5 trainer (r20) on the republished live-baseline agentic-debt-final

**Status:** Running
<!-- gen-workflow:begin -->
**Date:** 2026-10-06
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `acuadron-agentic-debt-final-v6-`
**Experiment name:** `acuadron-agentic-debt-final-v6`
**W&B project:** `agentic-debt` (group `acuadron-agentic-debt-final-v6`)
**Dataset:** `lakefs://arena-inspect/249fc194d84a…/internal/agentic-debt-r3/agentic-debt-final/manifest.jsonl` (gym `agentic-debt`): 233 chains / 1,687 checkpoints, live-baseline stop-on-regression reward (ADR-0076 vault)
**Manifest commit:** `249fc194d84aef7ac8047d87b0b07bd558ec68369bbc77854e40e7fbdfa17f2b` (main, 2026-10-06 02:40 UTC)
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-r20-20261006a` (digest `sha256:437bf865682262d2708c3204e00614b69f24b9202ce30554d25e271b76f9a59c`; miles `acuadron/dsa-qp` `bf9f3370`)
**Template:** `guparpit-miles-deployer-v10`
**Base:** `acuadron-agentic-debt-final-v5`
<!-- gen-workflow:end -->
**Argo workflow:** `acuadron-agentic-debt-final-v6-t66h9` (created 2026-10-06 09:44 UTC; admitted 09:48, 48/48 trainer pods Running 09:49)
**W&B run:** `jzfxq3v2` — https://mega.wandb.agi.amazon.dev/arena/agentic-debt/runs/jzfxq3v2
**Task pin:** `fec8e468c659` (every manifest row; rows name branch `acuadron-ad-final-onegrade-20261005t232628`, the materializer pins the commit)
**Trainer config deltas vs base:** `prompt-data-list` agentic-debt-final@0c328c7a (cumulative-shortfall reward, 237 chains) -> agentic-debt-final@249fc194 (live-baseline stop-on-regression, 233 chains); `lr` 1.5e-6 -> 1e-5 (the final-v4 decision); run names -> v6. Workflow: `experiment-name` only (48 replicas = 16 actor + 32 engine, 288 gyms, the 101-node exclusion list unchanged).
**Checkpoints:** `/mnt/scratch-s3files-rw/acuadron/checkpoints/slime_experiments/acuadron-agentic-debt-final-v6` (fresh; starts from `ref_load`)
**Outcome:** (running)

## Goal

Train on the two-grading live-baseline reward (`r_k = fixed_k`, or `-(K-1)` and chain stop on a regression) with the
r20 trainer (DSA-QP + R3 data path). Watch: reward integrity (no `no_reward` drops), the regression-stop rate, step time
with two gradings per checkpoint, and whether the policy moves at lr 1e-5.

## Setup

final-v5 unchanged except the dataset and lr. The dataset is the 2026-10-06 republish of `agentic-debt-final`: the pre-step
tree waits in a root-only vault inside the agent container (`/var/lib/ad-vault`, 0700), the root collect hook kills the
agent, wipes `/logs/artifacts`, publishes `prestep/` and `handoff/` with hashes; the verifier grades both trees. On gym
image adr72 (no host-side record relay) every baseline is graded in full (`baseline-plan.json` `mode=full`,
`reason=no_record`): two gradings per checkpoint.

Why not final-v4's `agentic-final-v2`: that payload (and `agentic-debt-final-live`) wipes the baseline in `setup.sh`
line 91 and stores it in the 0777 `/logs/artifacts`; every attempt dropped `no_reward` (final-v4 RECORD).

Gate on the gym image before launch (2026-10-06 09:28-09:40 UTC, `harbor run` on a v5 gym pod, chain
`realdiff_4n4nd_prometheus-api-client-python_s0`, 4 checkpoints): nop agent 0.0 at every step, `no_regression` 1.0,
F_k 6/10/11/14 (debt carries), P_k 12, chain 0.0; oracle 1.0 at every step, chain 1.0. Both ran to the end, no exception.

## Timeline

| time (UTC) | event |
|---|---|
| 2026-10-06 09:40 | final-v5 stopped at step 14 (kueue 2,136 + 256 of 2,424 GPUs in use; 48 more nodes did not fit). |
| 2026-10-06 09:44 | `kubectl create -f workflow.yaml` -> `acuadron-agentic-debt-final-v6-t66h9`. |
| 2026-10-06 09:50 | 233 prompts loaded from the pinned manifest; engines load 120 shards; NATS connected 10:04:41 (concurrency 64, n 8, multiplier 4); 288/288 gyms ready 10:23. |
| 2026-10-06 10:09-10:16 | Startup thundering herd on lakeFS: 34 of the first 256 groups dropped `reason=other` (`lakeFS download failed`, 5 retries exhausted) while 288 gyms fetched tasks at once. No repeat seen after 10:16. |
| 2026-10-06 10:35-10:52 | dind OOMKilled (64Gi limit) on 2 gym pods and 3 restarts on a third: every trial on those pods lost its `main` container -> collect hook `service "main" is not running` -> no handoff -> `no_reward` (15 trajectories in rollout 0: 7 `no verifier reward`, 5 at segment-02, 3 at segment-03). Not a payload fault; verifier containers declare 16 GiB each, 8 per pod. |
| 2026-10-06 10:57 | Rollout 0 complete: 64 groups, 512 attempts, avg_reward 0.779 (kept groups), raw_reward 0.825, response len 16,197, 3,116 s (final-v5 rollout 0: 3,090 s). Zero-variance drops logged: 4 all-1.0, 6 all-0.0. `train_wait` 3,514 s, `data_preprocess` 4.4 s; step 0 (cold JIT) started. Published trials carry `prestep/` and `handoff/`, `baseline-plan` mode=full. |
| 2026-10-06 09:49 | 48/48 trainer pods Running on 48 distinct nodes, none on the exclusion list; digest `437bf865…`; argv `--glm5-next-dsa-qp --glm5-next-kda-tp --prefetch-rollout-data --expert-model-parallel-size 8 --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11 --lr 1e-05 --global-batch-size 512 --rollout-batch-size 64 --n-samples-per-prompt 8 --arena-inflight-multiplier 4`. W&B `jzfxq3v2`. |

## Results

Steps 0-7 (2026-10-06 17:45 UTC), W&B `jzfxq3v2`:

| rollout | avg_reward (kept) | raw_reward | graded steps/episode | tokens/segment | tokens/episode | %chains==1.0 | rollout wall s |
|---|---|---|---|---|---|---|---|
| 0 | 0.779 | 0.825 | 3.76 | 16.2k | 61k | 46% | 3,116 |
| 3 | 0.700 | 0.773 | 5.12 | 24.8k | 127k | 39% | 1,122 |
| 5 | 0.791 | 0.816 | 6.07 | 31.2k | 189k | 42% | 2,731 |
| 7 | 0.660 | 0.720 | 7.00 | 44.9k | 314k | 27% | 6,690 |

- The headline reward is flat (~0.70-0.79) but the policy moves fast underneath: graded steps per episode
  3.76 -> 7.00 (dataset mean chain length 7.24 - chains now run to the end; early regressions/stops vanishing),
  tokens per segment 16.2k -> 44.9k, `train_rollout_logprob_abs_diff` 0.030 -> 0.161 (step 0 matches v5's
  0.026-0.030; the growth is the off-policyness of inflight-4 rollouts under lr 1e-5). TIS healthy (tis ~1.0,
  clipfrac <= 1.7%), grad_norm 0.028-0.066, truncated 0.
- avg_reward is the kept-group mean under dynamic sampling (logged drop reasons: 28 all-1.0, 12 all-0.0,
  3 all-0.8). Range-restricted by construction; population improvement shows first as more all-1.0 drops,
  not as a higher kept mean. Per-step `fixed` sample mean 0.906, 82% of steps fully fixed; episode mix at
  rollouts 0-3: 44% chains at 1.0, 13% at 0, 3% negative.
- Cost of the behavior change: ~5x tokens per episode -> rollouts 6-7 took 5,500-6,700 s, step_time ~7,000 s
  (train_wait ~5,700 s), actor_train 1,050-1,100 s. Loop strongly rollout-bound.
- Drops tapered after startup: 324 in hour 10 (lakeFS herd + first dind OOMs), then 2/1/2 in hours 15/16/17.
  19 gym container restarts total; `no_reward` 112 cumulative (~2.7% of attempts).

## Issues

- dind OOM (template v10: dind memory 64Gi, 8 trials per pod, verifier `memory_mb` 16384 each) kills in-flight trials; ~3 % of rollout-0 attempts. Mitigation needs an owned template variant (dind 128Gi, as `acuadron-miles-deployer-v3` had) and a relaunch or the NATS/gym restart dance; not applied.
- lakeFS startup herd: one-time, 34 groups.

## Follow-ups

- `acuadron-ad-final-vaultrecord-20261006t075544` (`f82814b4655a`, 08:05 UTC, not on main) keeps the grading record in the vault: one grading per tree without a launcher change. Adopt once validated; halves verifier time.
- Rollout-bound loop at multiplier 4 (final-v5: 600-1,400 s wait per step); two gradings per checkpoint make it worse. Levers: engine share, multiplier 6-8 with gyms >= 1.125x the cap.

## Sources

- final-v4 RECORD (the payload bug), final-v5 RECORD (the trainer), lakeFS main `249fc194d84a` README (reward), ADR-0076.
