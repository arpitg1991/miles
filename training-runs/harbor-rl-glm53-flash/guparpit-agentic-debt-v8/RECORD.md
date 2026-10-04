# Run record: guparpit-agentic-debt-v8 (r55) — the ablation of r53: r51 with only the length bonus off

**Status:** Prepared
<!-- gen-workflow:begin -->
**Date:** 2026-10-04
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v8-`
**Experiment name:** `guparpit-agentic-debt-v8`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v8`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-recon-20261002-qcap-evfix`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v4`
<!-- gen-workflow:end -->
**Argo workflow:** not submitted
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Trainer config deltas vs base:** `arena_length_reward_coef` 0.10 -> 0.0 only; `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v8`, seeded from r51 `iter_0000069` (a fresh copy)
**Outcome:** Prepared

## Goal

Attribution. Three arms share r51's image, layout, queue cap, and `iter_0000069`:

| Arm | Length bonus | Loss averaging | Spread division |
| --- | --- | --- | --- |
| r51 `guparpit-agentic-debt-v4` | on (0.10) | per-sample (GRPO default) | on |
| r55 this run | off | per-sample (GRPO default) | on |
| r53 `guparpit-agentic-debt-v6` | off | token-level (DAPO) | off (Dr. GRPO) |

If r55 holds reward and effort like r53, the bonus alone explains the r48, r50, r51, r52 collapse, and the two loss changes are a separate choice for the fork cleanup. If r55 drifts long (the Dr. GRPO per-sample bias: r47 grew from 82K to 314K tokens without the bonus), the loss changes earn their place.

## Upstream check

The GRPO defaults of this arm are upstream's defaults. The two flags r53 adds are upstream flags (`--calculate-per-token-loss`, `--disable-grpo-std-normalization`). See the r53 record.

## Measurement plan

Compare at the same rollout ids with r51 and r53.

| Question | Signal | Read |
| --- | --- | --- |
| Does removing the bonus alone stop the "do less" drift? | calls, tool calls, output tokens per trial (published tree); `rollout/episode_response_length/mean` | r55 near r53 means the bonus is the cause; r55 near r51 means the loss changes matter |
| Reward | `rollout/group_metrics/reward.mean`; paired same-task change | r53 held 0.74 at 85 to 88, r51 0.66 |
| Length drift without the bonus under per-sample averaging | episode tokens over 30 rollouts | drift above +30% points at the Dr. GRPO bias |
| Flags live | argument dump: `arena_length_reward_coef 0.0`, `calculate_per_token_loss False`, `grpo_std_normalization True`, `arena_output_queue_groups 64` | present at start |

## Launch

<!-- filled at launch -->
