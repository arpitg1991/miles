# guparpit-agentic-debt-v1 (r48)

**Status:** submitted 2026-10-01.
**Gym:** agentic-debt. **W&B:** project `arena/agentic-debt`, run `guparpit-agentic-debt-v1`.
**Dataset:** `lakefs://arena-inspect/main/internal/agentic-debt-r3/agentic-debt-766/` (manifest commit 327e057c, as r47).
**Start:** r47 checkpoint `iter_0000059` (saved 2026-10-01 02:51Z). miles resumes at rollout 60.
**Parent:** r47 `rl-glm53f47-hqdb5`. r47 keeps running unchanged.

## Change vs r47

The only training change is `arena_inflight_multiplier` 10 -> 4. The run
name and the W&B project follow the naming rule of 2026-09-30.
`miles-config.yaml` holds the full list and the reasons.

Evidence from r47 rollouts 55 to 64 (trainer log and W&B `0ix3m75e`):

| Signal | r47 value |
| --- | --- |
| Finished groups in the queue | 320 (the cap, 10 rollouts) from rollout 62 on |
| `rollout/weight_version` at rollout 63, trainer on version 10 | min 1, mean 3.1, max 9 |
| `train/train_rollout_logprob_abs_diff` | 0.033 (rollouts 0 to 7) to 0.049 (rollouts 56 to 63) |
| `perf/step_time` | 3,585 to 4,876 s |

A full queue means that the gym makes groups faster than the trainer uses
them. A higher cap then only adds stale data.

## Checkpoint copy

`slime_experiments/guparpit-agentic-debt-v1/` is a server-side S3 copy:

- `iter_0000059/`: 64 shards, `.metadata`, `metadata.json` (583.8 GiB). Names
  and sizes match the source.
- `iter_0000059/slime_extra_state.json`: `{"rollout_id": 59}`. The source also
  holds `wandb_run_id` 0ix3m75e. `_load_extra_state` restores that id, so the
  copy omits it and the run opens a new W&B run.
- `rollout/arena_data_source_state_59.pt`: the data position at rollout 59.
- `latest_checkpointed_iteration.txt`: `59`, written last.

The S3 Files mount showed all 66 files (584G) and the new sidecar 40 s after the copy.

## Upstream check

- miles `fully_async_data_buffer.py` bounds staleness when it gets a group:
  it drops groups older than `--max-weight-staleness` and logs
  `avg_staleness` and `max_staleness`. The arena NATS rollout does not use
  that buffer. It bounds staleness only through the dispatch cap,
  `arena_inflight_multiplier` x `rollout_batch_size` (`nats_rollout.py`).
- AReaL bounds the same lag with `max_head_offpolicyness`.
- A lower dispatch cap is the existing arena control. r44 runs at 4 with an empty queue and a lag of 1 to 3 versions.
- Limit: a long chain that a worker holds for many hours can still arrive
  stale. The cap does not stop that. A consume-time filter like
  `--max-weight-staleness` would. This run does not add one.

## Measurement plan

All signals exist in the r47 image and were read from r47 on 2026-10-01:

| Question | Signal | Gate |
| --- | --- | --- |
| Is the data fresher? | `rollout/weight_version/{min,mean,max}` (perf line of `metrics.py`); the counter restarts at 1 | by rollout 65: max - min <= 4 |
| Does the queue stay below the cap? | `queue=` in `Rollout N complete` | below 128 |
| Does the trainer wait for data? | `perf/train_wait_time`, `perf/wait_time_ratio` | wait ratio < 0.3 after the first rollout |
| Off-policy gap | `train/train_rollout_logprob_abs_diff`, `train/pg_clipfrac`, `train/tis_clipfrac` | gap falls toward 0.033 |
| Outcome | `rollout/group_metrics/reward.mean`, `rollout/episode_response_length/mean`, `rollout/clipped_turns` | compare with r47 at the same rollout ids |

r47 and this run start from the same weights at rollout 60. The comparison
is not a clean A/B test, because r47 starts with a full queue and this run
starts with an empty one.

## Launch

<!-- filled at launch -->
