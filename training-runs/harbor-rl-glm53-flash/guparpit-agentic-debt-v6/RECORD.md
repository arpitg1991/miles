# Run record: guparpit-agentic-debt-v6 (r53) — r51 without the length bonus, with token-level loss and no spread division

**Status:** Running
<!-- gen-workflow:begin -->
**Date:** 2026-10-04
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v6-`
**Experiment name:** `guparpit-agentic-debt-v6`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v6`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-recon-20261002-qcap-evfix`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v4`
<!-- gen-workflow:end -->
**Argo workflow:** `guparpit-agentic-debt-v6-w4889`
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Trainer config deltas vs base:** `arena_length_reward_coef` 0.10 -> 0.0; `calculate_per_token_loss` true; `disable_grpo_std_normalization` true; `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v6`, seeded from r51 `iter_0000069` (2026-10-02)
**Outcome:** Running

Starts from r51's `iter_0000069`, so rollouts 60 to 69 are shared with r51. The sidecar holds `{"rollout_id": 69}` and no `wandb_run_id`, so this run opens its own W&B run. Image, layout (EP8), gym, and the output queue cap (64 groups) are r51's. Control: r51 `guparpit-agentic-debt-v4-8l6jh`.

## Goal

Test whether the reward drop of r48, r50, and r51 comes from the length bonus, and remove the two length biases of the loss at the same time.

On the same tasks, r51 lost 0.054 by rollouts 85 to 89 and 0.070 by 95 to 99 (p = 0.0003). Tokens per chain step fell from 41K to 11K to 16K. The analysis is in the memory note `length-bonus-tied-group-do-less` and under `/workplace/guparpit/kdfast/scratch/drop/`.

## Changes, and why each one

1. `arena_length_reward_coef` 0.10 -> 0.0. The bonus runs before the zero-variance filter and rescues tied groups. With the spread division of GRPO, the ±0.05 term became an advantage of about ±1.7 in those groups. Tied groups grew to 33 to 49% of trained groups, and length drove 43 to 49% of all advantage. On cad-gym the same bonus opened a second hole: it measures only unmasked tokens, and clipped calls are masked, so the policy moved its thinking into clipped calls.
2. `calculate_per_token_loss: true` (DAPO, upstream flag). Without it, each sample's loss is divided by its own length (`get_sum_of_sample_mean` in `miles/backends/training_utils/cp_utils.py`). Dr. GRPO (arXiv 2503.20783) shows that this favors long failures and short successes. r47 without the bonus grew from 82K to 314K tokens per episode. r46 ran the flag on auctioneer.
3. `disable_grpo_std_normalization: true` (Dr. GRPO, upstream flag). No division by the group reward spread. Any future shaping term then acts as a plain coefficient. The launcher drops a YAML `false`, so the negative flag is set as `true`; the in-image parse resolves `grpo_std_normalization` to `False`.

## Upstream check

- Upstream miles has both flags: `--calculate-per-token-loss` and `--disable-grpo-std-normalization` (`miles/utils/arguments.py`). No upstream framework puts a length term into the reward; DAPO uses a soft overlong penalty near the context cap, which miles does not ship.
- The bonus was a port of the Kimi k1.5 length penalty. Kimi uses mean-baseline advantages with no spread division, which is why the same term stays small there.
- Tied groups are filtered by `check_reward_nonzero_std`, the standard dynamic-sampling rule.

## Measurement plan

The control is r51 at the same rollout ids. r53 and r51 share rollouts 60 to 69.

| Question | Signal | Gate at rollouts 85 to 99 |
| --- | --- | --- |
| Does effort hold? | tokens per chain step and turns, from the published trials (`trials.csv` method) | at least 30K tokens per step (r51: 11K to 16K); at least 45 turns (r51: 25 to 33) |
| Does reward hold on the same tasks? | paired same-task change, `/workplace/guparpit/kdfast/scratch/drop/sametask/paired_sametask.py` | within ±0.02 (r51: -0.054 to -0.070) |
| True reward | `rollout/group_metrics/reward.mean` (W&B) | at least 0.70 (r51: 0.667 to 0.673) |
| Length drift | `rollout/episode_response_length/mean` | within ±30% over 30 rollouts, no run-away in either direction |
| Flags live | `calculate_per_token_loss True`, `grpo_std_normalization False`, `arena_length_reward_coef 0.0` in the argument dump; `Dynamic sampling: ... rescued=0` | present at start |
| Trainer behavior | `train/grad_norm`, `train/train_rollout_logprob_abs_diff`, `perf/actor_train_tok_per_s` | grad norm may change scale with the new averaging; record it, no gate |

If r53 drops the same way, the bonus is cleared and the next step is the trajectories.

## Launch

| Time (UTC) | Event |
| --- | --- |
| 2026-10-04 00:27 | Seed copied: 64 shards, `.metadata`, `metadata.json`, `debug_events/` (69 files), sidecar `{"rollout_id": 69}`, `rollout/arena_data_source_state_69.pt`, `latest_checkpointed_iteration.txt` 69. 135 objects match r51 by name and size. |
| 00:28:23 | Workflow `guparpit-agentic-debt-v6-w4889` created. 64 B200 nodes excluded as NotReady or tainted, 208 excluded in all. In-image parse of the argv: `calculate_per_token_loss True`, `grpo_std_normalization False`, `arena_length_reward_coef 0.0`, `arena_output_queue_groups 64`, EP 8, triton MoE runner. |
