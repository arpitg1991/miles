# Run record: guparpit-agentic-debt-v4 (r51) — r50 with the output queue capped at 64 groups

**Status:** Prepared
<!-- gen-workflow:begin -->
**Date:** 2026-10-02
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v4-`
**Experiment name:** `guparpit-agentic-debt-v4`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v4`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-recon-20261002-qcap@sha256:8246fcae103253c75cfad780ad39ee0ea6b1e5c0119a6d7ec916c58c18ae086c`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v3`
<!-- gen-workflow:end -->
**Argo workflow:** not submitted
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Image digests:** gym `sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d`, trainer `sha256:8246fcae103253c75cfad780ad39ee0ea6b1e5c0119a6d7ec916c58c18ae086c`
**Trainer config deltas vs base:** `arena_output_queue_groups` unset (320 groups) -> 64; `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v4`, seeded from r47 `iter_0000059` (2026-10-01 02:51Z)
**Outcome:** Prepared

Starts from r47 checkpoint `iter_0000059`: a copy that was made before this record, with a sidecar without `wandb_run_id`. miles resumes at rollout 60. Sibling arms: r50 (`guparpit-agentic-debt-v3`, r49 plus EP8) and r52 (`guparpit-agentic-debt-v5`, r49 plus EP8 plus the FlashMLA DSA kernel).

## Goal

Find out if a smaller output queue makes the trained groups fresher at the same throughput. The run ends when the gates below pass or fail at rollout 75.

## Setup

r51 is r50 with one more key. EP8 stays.

| Item | Base r50 (`guparpit-agentic-debt-v3`) | This run |
| --- | --- | --- |
| `arena_output_queue_groups` | unset: `10 * 256 // 8` = 320 groups | 64 groups (2 rollouts of 32) |
| trainer image | `miles-glm53-recon-20261001b` (miles `da93977ba8`) | `miles-glm53-recon-20261002-qcap` (miles `2566c00328`, branch `arpit-r51-queue-cap`) |
| `expert_model_parallel_size` | 8 | 8 |
| `arena_inflight_multiplier` | 4 (128 groups in flight) | 4 |

The worker output queue holds finished groups until the trainer takes them. Its capacity was `10 * global_batch_size // n_samples_per_prompt` groups, 320 here. The worker `put` blocks on a full queue, so the cap is back-pressure: it slows the gyms and drops no group. `arena_inflight_multiplier` limits only the dispatched tasks.

The queue is not the only wait. `_process_group` (`nats_rollout.py`) calls the blocking `queue.Queue.put` inside the asyncio worker loop. A full queue thus stops the whole loop: no fetch, no ack, no publish. The gyms still finish the in-flight tasks (`arena_inflight_multiplier` 4 x 32 = 128 groups), and their results wait unfetched in the JetStream results stream. `in_flight` drops only at a fetch, so no new task starts. When the trainer is the bottleneck, a group waits about (cap + in-flight) / D steps from dispatch to drain. D is the number of groups that one step takes from the queue (`rollout/group_metrics/n_groups`). D is more than 32 when dynamic sampling drops groups. The 128 / D part does not depend on the episode length: the in-flight stage holds 128 groups and passes D groups per step (Little's law). The drain of rollout r runs while rollout r - 1 trains, and that adds 1 version. The expected lag of the mean is thus:

| D | r51: (64 + 128) / D + 1 | r50: (320 + 128) / D + 1 |
| --- | --- | --- |
| 32 (no group dropped) | 7.0 | 15.0 |
| 52 (r47 rollout 11, 38.5% dropped) | 4.7 | 9.6 |
| 65 (r45, 51% dropped) | 4.0 | 7.9 |

The cap removes (320 - 64) / D versions of lag: 8 at D 32, 3.9 at D 65. The in-flight term stays, and `arena_inflight_multiplier` is its lever. r47 is not at this steady state in the baseline below: it had 320 groups in flight (multiplier 10), and its queue was full only from rollout 62 on (the `guparpit-agentic-debt-v1` record).

The image commit `2566c00328` is the r50 image commit `da93977ba8` plus `--arena-output-queue-groups` (YAML `arena_output_queue_groups`). The default (None) keeps the 320-group formula. A value less than `rollout_batch_size` stops the worker at startup. At startup, the worker logs `NATSRolloutWorker output queue: maxsize=64 groups (--arena-output-queue-groups=64)`.

## Upstream check

Upstream radixark/miles `main` `d7f1a421` (2026-09-30, the base of the recon branch):

| Upstream knob | Role | Use here |
| --- | --- | --- |
| `--async-data-buffer-capacity-factor` (default 2.0) | Capacity of the fully-async finished-group buffer, `floor(factor * rollout_batch_size)` groups; the producer blocks when the buffer is full (`miles/rollout/fully_async_data_buffer.py`) | The same role. Not reused: only the fully-async driver reads it, and its default would change every arena run from 320 to 64 groups. r51 sets 64, which is the upstream default for `rollout_batch_size` 32 |
| `--max-weight-staleness` (default None) | Recycles a group older than N versions to the data buffer; fully-async only | Not used. A filter, not a cap. A follow-up candidate if the cap alone does not bound the lag |
| `--async-max-concurrent-samples` | Generation concurrency in fully-async mode | The counterpart of `arena_inflight_multiplier`; unchanged |
| `--over-sampling-batch-size` | Sampling granularity of the synchronous rollout | Not the same queue |

## Measurement plan

Compare r51 with r50 at the same rollout ids. Start the gates at rollout 66, after the queue first fills.

The lag of a sample is the trainer version minus the version that generated the sample. The weight-version counter starts at 0 in each trainer process (`updater.py`). The startup sync makes version 1, and version 1 trains rollout 60. Thus the trainer version at rollout id r is r - 59 in r50 and in r51. `rollout/weight_version/*` holds the oldest version of each training sample, so:

- lag of the mean = (r - 59) - `rollout/weight_version/mean`
- lag of the oldest sample = (r - 59) - `rollout/weight_version/min`
- cross-check: `rollout/off_policy_round/mean` - 60. This metric uses the mean version of the turns of each sample, over all groups taken from the queue, and adds the start rollout 60.

`rollout/weight_version/max` - `rollout/weight_version/mean` is not a lag. It is the spread inside one batch. The queue is FIFO (first in, first out), so a longer wait in the queue moves `min`, `mean` and `max` by the same amount, and the spread stays the same.

| Question | Signal | Gate |
| --- | --- | --- |
| The cap holds | startup log line; `rollout/queue_depth_at_start` | `maxsize=64` in the trainer log; queue depth 64 or less at every rollout |
| Fresher groups | lag of the mean and lag of the oldest sample (definitions above) | over rollouts 66 to 75: the mean lag of the mean at most (64 + 128) / D + 1, with D the mean `rollout/group_metrics/n_groups` over the same rollouts (4.0 at D 65, 7.0 at D 32); the mean lag of the mean and of the oldest sample below r50. r47 at rollout 63, trainer on version 10: mean 10 - 3.1 = 6.9, oldest 10 - 1 = 9 |
| Same throughput | `perf/rollout_time`, `rollout/queue_depth_at_start` | the trainer does not wait: queue depth 32 or more at most rollouts, `perf/rollout_time` within 10% of r50 |
| Same learning | `train/train_rollout_logprob_abs_diff`, `train/grad_norm`, `rollout/group_metrics/reward.mean` | log-prob diff at or below r50 (r48 rose from 0.033 to 0.049 on a stale queue); grad norm and reward within r50's range |

## Checks done before launch

| Check | Result |
| --- | --- |
| Tests, plain stub (`tests/fast/plugins/arena`, `tests/fast/launch_scripts`) | 419 passed, 1 known upstream failure (`test_workplace_backend`). The 10 new tests are in `test_output_queue_groups.py` |
| Tests, stubfull, conftests, local Ray (`tests/fast/plugins/arena`, `test_run_arena_harbor.py`) | 365 passed, 1 known stubfull failure (`test_register_nova_reasoning_parser`) |
| Image `miles-glm53-recon-20261002-qcap` | `git rev-parse HEAD` = `2566c00328bb5fa287e4678c13351b96a63b079d`, clean tree, `opentelemetry-api` 1.44.0 = `opentelemetry-sdk` 1.44.0. Pushed to us-east-1; the ap-south-1 replica has the same digest |
| r51 argv | the launcher builds `--arena-output-queue-groups 64`; the miles parser resolves 64 in the venv and in the image. Against the r50 config, the parsed args differ only in `arena_output_queue_groups` and `arena_sample_summary_dir` |
| `workflow.yaml` | against r50, only `generateName`, `experiment-name`, `trainer-image`, and `miles-config` change; `miles-config` is this `miles-config.yaml` verbatim |

## Timeline

| UTC | Event |
| --- | --- |
| 2026-10-02 04:25Z | trainer image pushed |

## Results

| Metric | Value | Source |
| --- | --- | --- |
| none yet | | |

## Issues

- None yet.

## Follow-ups

- While `put` blocks, the whole worker loop stops, and the finished in-flight results wait unfetched in JetStream. That wait is the 128 / D term of the lag, and the cap does not remove it. If the lag stays too high at the cap, lower `arena_inflight_multiplier`, or evaluate a staleness filter like upstream `--max-weight-staleness`.

## Sources

- miles `arpit-r51-queue-cap` `2566c00328` (`miles_plugins/arena/nats_arena/nats_rollout.py`: `_output_queue_groups`; the blocking `put` in `_process_group`; the `in_flight` count in `_worker_loop`).
- Base run files: `arpit-recon-20261001:training-runs/harbor-rl-glm53-flash/guparpit-agentic-debt-v3/`.
- r47 queue and version numbers: the `guparpit-agentic-debt-v1` header of `miles-config.yaml`.
- Lag: `miles/backends/training_utils/weight_update/updater.py` (the version counter), `miles_plugins/arena/train_async_arena.py` (the startup sync), `miles_plugins/arena/rollout_metrics.py` (`compute_weight_version_metrics`, `compute_off_policy_round_metrics`).
