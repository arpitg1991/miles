# r46: r44 with the token-level loss average

Prepared 2026-09-27. Generated with:

```
.venv/bin/python gen-workflow.py 46 --base r44 --template guparpit-miles-deployer-v10 \
  --experiment-name rl-glm53f-auct-cap-r46 \
  --gym-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a
```

The base r44 workflow supplies the trainer image, `agent-kwargs` `{}`,
`publish-jobs-dir` `''`, and all other parameters.

| Delta vs r44 | r44 | r46 |
| --- | --- | --- |
| `calculate_per_token_loss` | off (miles default `sum_of_sample_mean`) | `true` (trainer flag `--calculate-per-token-loss`) |
| Loss normalizer per step | each rollout divides by its own token count, then the sum divides by the rollout count | the sum over all trained tokens divides by the total token count |
| Names | `rl-glm53f-auct-cap-r44`, `rl-glm53f44-` | `rl-glm53f-auct-cap-r46`, `rl-glm53f46-` |
| Sample summary dir | `debug/rl-glm53f-auct-cap-r44/sample_summary` | `debug/rl-glm53f-auct-cap-r46/sample_summary` |

Same as r44: trainer image `miles-glm53-r14-20260927a` (miles
`e0987aed6`), gym image `gym-glm53-adr72-20260927a`, template
`guparpit-miles-deployer-v10` (uid `16ddd530-501f-4734-82ae-88500671bed8`,
generation 1), dataset caponly-1034, `lr` 1.5e-6, `agent-kwargs` `{}`,
`publish-jobs-dir` `''`, `agent-timeout-multiplier` 2, `ack-wait` 36000,
trainer deadline 39600, `excluded-nodes`, W&B project `rl-glm53f-auct-cap`.
Fresh run: no checkpoint dir under
`slime_experiments/rl-glm53f-auct-cap-r46` (checked 2026-09-27).

Diff check: every workflow parameter except `experiment-name` and
`miles-config` is equal to r44, in the same order. `generateName` is the
only other change. In `miles-config`, the three name lines and the new key
with its comments are the only changes. A local copy of the launcher
`_flatten` gives 174 argv tokens for r46 and 173 for r44. The extra token is
`--calculate-per-token-loss`. The sample summary path is the only other
token change.

## Why

- DAPO (arXiv 2503.14476) uses a token-level policy gradient loss. Dr. GRPO
  (arXiv 2503.20783) shows the length bias of the per-sample mean.
- With the per-sample mean, a long episode spreads its weight over many
  tokens. Each token of a long wrong episode gets a small share of the
  gradient, so the penalty on long failures is weak.
- With the token-level average, every trained token has the same weight.
- `lr` is unchanged. When the episode lengths are similar, the mean step
  size per token is about the same.

## Code check (miles `e0987aed6`, Megatron `e8f57451` in the r14 image)

- `scripts/run_arena_harbor.py` `_flatten`: a true boolean becomes a bare
  flag. `calculate_per_token_loss: true` becomes
  `--calculate-per-token-loss`.
- `miles/utils/arguments.py`: `reset_arg(parser, "--calculate-per-token-loss",
  action="store_true")`.
- `miles/backends/training_utils/cp_utils.py` `get_sum_of_sample_mean`: the
  flag returns `sum_of_token` and ignores `rollout_mask_sums`.
- `miles/backends/training_utils/loss.py`: the flag skips the division by
  the rollout count and returns the token count as the Megatron normalizer.
- `loss_hub/losses.py`: the TIS path builds the reducer with the same flag.
  The TIS masks change only the numerator.
- Megatron `schedules.py` does not divide by the micro-batch count.
  `finalize_model_grads` reduces the token count over DP and divides the
  gradients by it.
- Asserts that block the flag: `ft/indep_dp.py` (only with `--indep-dp`),
  `utils/multi_lora.py` (only with multi-LoRA), Megatron DDP (only with
  `ddp_average_in_collective`, default false). r46 uses none of them. CP is
  1, and MTP and the custom loss paths are off.
- No MoE auxiliary loss and no DSA indexer loss: the GLM-5.3 plugin
  attention has no loss term, and the MoE auxiliary loss coefficient is the
  default 0.

## Metrics

- W&B `train/pg_loss` has a different normalizer. Its scale is not
  comparable with r44.
- `train/pg_clipfrac`, `train/ppo_kl`, and `train/entropy_loss` become
  token-weighted means. On r44 they are per-rollout means.
- The rollout-data metrics (`log_probs`, `advantages`) keep the per-rollout
  mean. They are comparable with r44.

## Launch checklist

1. `kubectl create --dry-run=server -f r46/workflow.yaml`.
2. `kubectl create -f r46/workflow.yaml`. No stop of r44 or r45.
3. Read the trainer-worker-0 argv for `--calculate-per-token-loss`.
4. On one gym pod, read `DOCKER_CONFIG` (`/root/.docker`) and
   `HARBOR_AGENT_KWARGS` (`{}`).
5. Compare the first train step `pg_clipfrac`, `ppo_kl`, and `grad_norm`
   with the r44 first step.
