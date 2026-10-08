# Run record: acuadron-agentic-debt-final-v10

**Status:** Prepared, not launched (2026-10-08). The user asked for these settings on the next run and no restart of final-v9.
**Family:** `harbor-rl-glm53-flash`
**Experiment name:** `acuadron-agentic-debt-final-v10`
**Base:** final-v9 recipe from the live workflow `acuadron-agentic-debt-final-v9-dx8b7` through `relaunch.py`.
**Template:** `acuadron-miles-deployer-v13` (passthrough from v9)
**Trainer image:** `arena-github/miles:miles-glm53-r24-20261008a` (290940d7: `rollout/population/*` W&B keys, parallel routing loads)
**Dataset:** 110-chain subset, manifest commit `f75dbe74…` (unchanged from v9)

## Config delta vs final-v9

- `calculate_per_token_loss: true` (DAPO token-level loss)
- `disable_grpo_std_normalization: true` (Dr.GRPO: advantage = reward - group mean, no spread division)
- `trainer-image` r22 -> r24

Everything else is v9: lr 1.5e-6, 64 groups x 8 attempts, GBS 512, `arena_inflight_multiplier` 4, 64 nodes, 288 gyms, base weights.

## Why

On final-v9 (per-sample loss, std normalization) the kept groups whose 8 rewards lie within 0.2 of each other
are 22% of the groups and carry 22% of the gradient mass at full strength; the hidden-test-detail and
carried-debt noise gets the same push as a regression. With these two flags on the same batches they carry 4%
and regressions 29% (`studies/agentic-debt-v23-signal-anatomy/STUDY.md`). guparpit-agentic-debt-v23 already
ran this way and showed no output-length drift over 60 updates, while v9 went 101k -> 191k output tokens per
episode in 36 updates.

## Checks done

- `kubectl create --dry-run=server`: accepted (`acuadron-agentic-debt-final-v10-n995d`).
- The r23/r24 image's argparse knows `--calculate-per-token-loss` and `--disable-grpo-std-normalization`;
  the launcher's `_flatten` emits both as bare flags.
- Parameter delta vs the live v9 workflow: `trainer-image`, `experiment-name`, `miles-config` (two keys).

## Launch

`kubectl --context arena-prod-bom-v2 -n arena-tasks create -f workflow.yaml`, then the `eval-monitor` skill.
To resume v9's weights instead of the base, set `experiment-name` and `experiment_name` to
`acuadron-agentic-debt-final-v9` (latest checkpoint iter 39 at 2026-10-08 08:40 UTC); note that the Adam
state resets (`no_save_optim`).

## Engine speed-up hand-off (2026-10-08, acuadron session 2)

`examples/arena/harbor-rl-glm53-flash/kdatp/sgl/ARMS-20261008.md`: the one-node bench of the remaining engine levers
(DP attention + DeepEP + two-batch overlap, speculation depth, fp8 KV, fp8 weights, KV pool 0.85) and the 16-node live
A/B gate run in session 2; the winner's `sglang_*` keys go into this run's `miles-config` on its next restart.
`workflow-512.yaml` (batch 512, 16 + 48, the arms commented at the end of `miles-config`) and `workflow-256.yaml`
(batch 256, 16 + 64) are the two prepared shapes; the launched workflow is `workflow.yaml`.
Session 2's commit `05065fac` overwrote this file's earlier text by mistake; restored here from `e23fcdde`.
