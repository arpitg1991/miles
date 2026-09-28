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

## Resume 1 (2026-09-27): 8 engines, no HF export, repaired sidecar

Prepared 2026-09-27. Not launched. The user asked for 8 SGLang engines on
auctioneer and for no per-save HF export on every run (2026-09-27 about
14:00Z). The user rule of 2026-09-09 (resume a crashed run from its last
checkpoint with no new question) covers the resume. The reason for
the two changes is in `r44/BUILD.md` "Resume 1".

The AREnAThanatos idle-GPU reaper deleted `rl-glm53f46-clhv6` at
11:02:24Z. The DCP save of step 9 was complete at 10:44:56Z. The HF export
of step 9 then ran, and the delete cut it.

Generated with:

```
python3 gen-workflow.py 46 --base r46 --template guparpit-miles-deployer-v10 \
  --experiment-name rl-glm53f-auct-cap-r46 \
  --gym-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a \
  --trainer-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r15-20260927a \
  --param replicas=16
mv r46/workflow.yaml r46/workflow-resume1.yaml && git checkout r46/workflow.yaml
```

`miles-config.yaml` is now the resume config, and `miles-config-r0.yaml`
is the launch config (`r44/BUILD.md` "Resume 1" gives the reason).

| Delta vs the r46 launch | r46 launch | r46 resume 1 |
| --- | --- | --- |
| Start point | base DCP (`ref_load`) | r46 `iter_0000009`, same dir |
| Rollout ids | 0.. | 10.. |
| Data position | row 0 | row 0 again (no data-source state) |
| W&B run | `4iu54zov` | `4iu54zov` (from the repaired sidecar) |
| `replicas` (param and config record) | 40 | 16 |
| `replica-trainer` | 8 | 8 |
| Rollout GPUs, engines | 256, 32 | 64, 8 |
| Trainer image | `miles-glm53-r14-20260927a` | `miles-glm53-r15-20260927a` |
| HF export per save | yes (`--save-hf`) | no |
| Config file | `miles-config-r0.yaml` | `miles-config.yaml` |
| Workflow file | `workflow.yaml` | `workflow-resume1.yaml` |

Diff check: the parameters of `workflow-resume1.yaml` are those of
`workflow.yaml`, in the same order. Only `trainer-image`, `replicas`, and
`miles-config` change. In `miles-config`, the header comment and the
`replicas` line with its comment are the only changes.
`calculate_per_token_loss` stays true. A stub run of the r15 launcher on
`r46/miles-config.yaml` with `REPLICA` 16 and `REPLICA_TRAINER` 8 gives
188 argv tokens: the 187 of r44 plus `--calculate-per-token-loss`. It has
no `--save-hf`, and `--rollout-num-gpus` is 64. The engine count math is in
`r44/BUILD.md` "Resume 1".

### Checkpoint (read 2026-09-27 14:05Z from r45 trainer-worker-39)

`slime_experiments/rl-glm53f-auct-cap-r46`:

- `latest_checkpointed_iteration.txt` = 9. `trainer-0.log`: "successfully
  saved checkpoint from iteration 9" at 10:44:56Z.
- `iter_0000009`: 64 `__*_0.distcp` shards, `.metadata`, and
  `metadata.json`. Each shard and `.metadata` has the same size as in r44
  `iter_0000009`. No file has size 0.
- No `slime_extra_state.json` and no `rollout/` dir. `train_async_arena.py`
  runs `save_model` (DCP, then HF export), then `_save_extra_state`, then
  `rollout_manager.save`. The delete came before the last two.
- `hf/rollout_9`: 489 files (r44: 671). It is not a complete HF model. The
  resume does not read it. It stays on disk.

### Resume without the two files (miles code)

- Rollout id: Megatron loads iteration 9 from the tracker, and the actor
  returns 9 + 1 = 10. The sidecar `rollout_id` goes to the log only.
- W&B: `_load_extra_state` finds no sidecar, and v10 sets no
  `WANDB_RUN_ID`. Thus the resume opens a new W&B run.
- Data source: `ArenaDataSourceWithBuffer.load(9)` logs `Checkpoint
  .../arena_data_source_state_9.pt does not exist, starting fresh.` and
  returns. The run does not fail. The data position restarts at offset 0,
  epoch 0.

### Repair decision

Done 2026-09-27 14:07:30Z from r45 trainer-worker-39, after `test ! -e`:
`iter_0000009/slime_extra_state.json` =
`{"rollout_id": 9, "wandb_run_id": "4iu54zov"}`. It has 45 bytes, no
newline, owner `root:ubuntu`, and mode 644, as the r44 sidecar. The key
order and the separators are those of `json.dump`. The S3 view shows the
file at 14:08:32Z. The resume continues W&B run `4iu54zov`.

No data-source state was written. A copy of the r44 state is not correct
for r46, because the state is not a function of the dataset, the seed, and
the rollout id:

- `offset` and `sample_group_index` count the groups that the publisher
  drew. Dynamic sampling (`check_reward_nonzero_std`) and the in-flight cap
  make that count depend on the rewards and on the completion times.
- `metadata` holds the ids of the 32 groups that each rollout trained on,
  in the completion order. The `rollout_tasks` lines of the two `trainer-0.log`
  files share only 3 to 9 of 32 ids for each rollout 0-10. Both runs drew
  the same epoch-0 order: the same `seed-N.gK.` pairs occur in both.
- `prompt_reward` holds the mean rewards of the policy of each run. r46
  trains with a different loss from step 0.
- With a copy, the resume guard skips the 352 r44 ids for the rest of
  epoch 0. Only 320 of them are in the r46 rollouts 0-10.

Effect of the restart:

- The data source starts again at offset 0 of the `shuffle(0)` order
  (`rollout_seed` 42). This is the order of the r46 launch. Thus r46 draws
  again the prompts that it drew in rollouts 0-10. At the same point, r44
  had drawn 937 of the 1034 groups, so the replay is about 90% of epoch 0.
- The resume guard is empty, so no prompt is skipped. `prompt_reward`
  starts empty. No r46 key reads it (`arena_skip_prompt_above_reward` is
  unset).
- `sample_group_index` and `sample_index` restart at 0, so the `.gK.`
  numbers in the task ids repeat. The new workflow has a new NATS stream,
  so the ids do not collide.
- After the replay, epoch 1 starts with the `shuffle(1)` order, as with no
  restart. At the r44 rate of about 85 drawn groups for each rollout, 300
  rollouts are about 24 epochs. The replay adds about one pass of the
  epoch-0 order.

### Launch checklist

1. Done: the r15 trainer image is in ap-south-1 (`r44/BUILD.md` "Resume 1").
2. `kubectl create --dry-run=server -f r46/workflow-resume1.yaml`. Done
   2026-09-27: accepted (`rl-glm53f46-8zxjz`, not created).
3. Before the create, copy
   `/mnt/scratch-s3files-rw/guparpit/logs/rl-glm53f-auct-cap-r46/trainer-0.log`
   to `trainer-0-attempt1.log`. The resume writes the same path again.
4. `kubectl create -f r46/workflow-resume1.yaml`.
5. Read the trainer-worker-0 argv: `--rollout-num-gpus 64`,
   `--calculate-per-token-loss`, and no `--save-hf`.
6. In `trainer-0.log`, look for:
   - the load of `iter_0000009` from the r46 dir
   - `Restored wandb_run_id=4iu54zov from checkpoint sidecar.`
   - `Checkpoint sidecar rollout_id=9.`
   - `.../arena_data_source_state_9.pt does not exist, starting fresh.`
   - a first rollout id of 10
   - 8 SGLang engines
7. The first attempt logged the rollout metrics of rollout 10 to W&B
   before the delete. Thus W&B shows two points at that step.
8. The step-19 save writes the DCP, the sidecar, and the data-source
   state, with no HF export.
