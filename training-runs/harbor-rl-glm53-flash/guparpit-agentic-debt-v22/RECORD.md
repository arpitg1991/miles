# Run record: guparpit-agentic-debt-v22 (r56-ns, relaunch of guparpit-agentic-debt-v18 at inflight multiplier 12) — r54 stack, staleness cap 8, queue cap 64, prefetch on, 400 gym workers

**Status:** Running
<!-- gen-workflow:begin -->
**Date:** 2026-10-06
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v22-`
**Experiment name:** `guparpit-agentic-debt-v22`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v22`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-wvspans-20261001a@sha256:e29ba91a3392fd250e78f551fca6f33061520a2b456134693d419e1bf5c9eab9`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-ns-20261005b@sha256:dbf0a3c78744ab28d96ac420e785cb5b1aa5cc2c96fb1b0b446a95de7fcc1551`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v18`
<!-- gen-workflow:end -->
**Argo workflow:** `guparpit-agentic-debt-v22-kdh5d`
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Trainer config deltas vs base:** `arena_inflight_multiplier` 8 -> 12; `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run. Workflow: `gym-replicas` 288 -> 400. Everything else as guparpit-agentic-debt-v18.
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v22`, seeded from guparpit-agentic-debt-v18 `iter_0000109` (its latest save at copy time, 2026-10-06 07:34Z; 135 objects match; sidecar `{"rollout_id": 109}` without `wandb_run_id`; data state 109; tracker 109 written last)
**Outcome:** Submitted 2026-10-06 07:43Z; admitted 07:50Z after v17 and v18 retired; first batch collecting

The owner's order of 2026-10-06: "make r54-ns and r56-ns run with 12x multiple". guparpit-agentic-debt-v18 (r56-ns at multiplier 8) is relaunched from its own `iter_0000109` save with `arena_inflight_multiplier` 12 and 400 gym workers; the staleness cap (8) and the queue cap (64) stay. guparpit-agentic-debt-v18 retires once this run's trainer job exists.

## Goal

Feed the FlashMLA trainer. At multiplier 8 the engines were not saturated and the trainer waited for data on 13 of 15 steps once the staleness cap engaged (mean wait 721 s on v17, 772 s on v18, rollouts 100 to 115). The engine-side evidence from v17's worker-0 log (1,243 SGLang decode-batch lines): `#running-req` per engine p50 36 / p90 42 / max 90; `full token usage` p50 0.58 / p90 0.71 / max 0.99; `#queue-req` 0 on every line; decode throughput per engine rises with concurrency (1,460 tok/s at 24 to 39 requests, 1,694 at 48, 1,864 at 56, 2,100 to 2,250 at 64 to 88). Multiplier 4 -> 8 raised engine output from 125 to 173 to 176 tokens per GPU per second (r54 vs v17/v18). Multiplier 12 puts 384 groups in flight; 288 gym pods served 256, so the workflow runs 400.

## Trainer stack

As guparpit-agentic-debt-v18: image `miles-glm53-ns-20261005b` (fork `arpit-ns-20261005`, `45cd7e47ff`), FlashMLA sparse forward, EP8, TP8 + SP, PP4 11/11/11/12, `NatsRolloutFn`, `max_weight_staleness` 8, `arena_output_queue_groups` 64, prefetch on, `glm5_next_kda_tp`, R3 with the triton MoE runner, token-level loss, no spread division, no length reward.

## Upstream check

- `arena_inflight_multiplier` is the fork's knob (`miles_plugins/arena`): the number of groups the NATS publisher keeps in flight, as a multiple of `rollout_batch_size`. Upstream miles has no equivalent; its async rollout is bounded by `--async-data-buffer-capacity-factor` on the consumer side only.
- The engine side is upstream SGLang: more concurrent requests raise decode throughput until the KV cache fills; `full token usage` near 1.0 makes SGLang retract and re-prefill requests. The gate below watches for that.
- Gym worker count is a workflow parameter of the deployer (`gym-replicas`); no trainer code changes.

## Measurement plan

Control: guparpit-agentic-debt-v18 at multiplier 8 (rollouts 91 to 115 in its record and raw log). Compare at matched rollout ids after the startup buffer is gone (about 10 rollouts after the seed).

| Question | Signal | Gate |
| --- | --- | --- |
| Do the engines produce more? | `perf/tokens_per_gpu_per_sec` | above 176 (v17/v18 at multiplier 8); expected about 220 |
| Are the engines saturated now? | SGLang decode lines in worker-0: `#running-req` p50/p90/max, `full token usage` p50/p90/max, share of `#queue-req` > 0; retract or eviction messages | running-req p50 near 54; token usage p50 near 0.85 with p90 below 1.0; no retract storms |
| Does the trainer stop waiting? | `perf/train_wait_time` | under 300 s on more than 80% of steps once the buffer is gone (v17: 13 of 15 over 300 s) |
| What does the cap drop at 12x? | `Weight staleness:` lines; `rollout/num_old_age_dropped`, `rollout/reward_old_age_dropped` | recorded per step; expected to rise with the longer episodes |
| Data age | `rollout/weight_version` mean and max; `rollout/fully_async/avg_staleness` | recorded |
| Reward | batch reward per 5-rollout bin; W&B true reward | 0.63 to 0.67, no collapse |
| Prefetch (on) | `perf/data_preprocess_time` | under 10 s |

## Launch

| Time (UTC) | Event |
| --- | --- |
| 2026-10-06 07:34 | Seed copied from guparpit-agentic-debt-v18 `iter_0000109` (`metrics-1001/ckcopy-v22.sh`): 135 objects match by name and size; sidecar `{"rollout_id": 109}` without `wandb_run_id`; data state `arena_data_source_state_109.pt`; tracker 109 written last. |
| 07:41 | In-image parse of the rendered argv (`recon2/parse/fullparse10.py`, `CFG=/work/v22.yaml` with `hf_checkpoint` remapped to the local HF config copy, `WORLD_SIZE` 320, `RANK` 0, CUDA stub on `LD_LIBRARY_PATH`; `parse-v22.out`): exit 0, 271 argv tokens. `arena_inflight_multiplier 12`, `max_weight_staleness 8`, `arena_output_queue_groups 64`, `prefetch_rollout_data True`, `rollout_function_path ...NatsRolloutFn`, `flash_mla`, `triton`, EP 8, 11/11/11/12, `glm5_next_kda_tp True`, `calculate_per_token_loss True`, `grpo_std_normalization False`, `arena_length_reward_coef 0.0`, save dir guparpit-agentic-debt-v22. |
| 07:43:06 | Submit `guparpit-agentic-debt-v22-kdh5d` (`make_runs_v21.py`): gym-replicas 400, node exclusions refreshed at submit (304 B200 nodes, 2 bad, both tainted); 347 (the v18 list of 345 plus 2 bad nodes) excluded. Trainer and gym images by digest, as v17/v18. Queue at submit: pending 0, admitted 17. |
| 07:46 | Trainer PyTorchJob `guparpit-agentic-debt-v22-kdh5d-trainer` created; Kueue: "insufficient unused quota for nvidia.com/gpu, 88 more needed" (v17 to v20 held 160 nodes). |
| 07:48 | v17 and v18 stopped (owner-named retirement). |
| 07:50 | Kueue admitted the trainer: 37 Running + 3 Pending at 07:50 after one Kueue TAS eviction and requeue (`Evicted=False, Requeued=True`), 40 Running by 07:55. Argument dump: `arena_inflight_multiplier 12`, `max_weight_staleness 8`, `arena_output_queue_groups 64`, `prefetch_rollout_data True`; `Checkpoint sidecar rollout_id=109`; `successfully loaded checkpoint ... at iteration 109`. |
| 08:05 to 08:20 | Engines loaded (120 shards, about 14 min), weights synced, `inflight_multiplier=12, max_in_flight=384`, `Rollout 110: collecting`. Gym Deployment 400 of 400 ready by 08:30 (193 at 08:22). |
| 08:32 | Engine side, 12 min in (83 lines): running-req p50 41, token usage p50 0.14, queue-req 0, gen tok/s p50 1,625. |
| 08:52 | Engine side after 33 min of generation, 327 lines, 14 engines: running-req p50 71 / p90 85 / max 142; full token usage p50 0.82 / p90 0.98 / max 0.99; queue-req above 0 on 41% of lines; gen tok/s per engine p50 1,685 / p90 2,979. Last 200 lines (08:4x): running-req p50 79, token usage p50 0.97 / p90 0.98, queue-req above 0 on 67%, gen tok/s p50 1,556 / p90 3,072. The KV cache is full on most engines and requests queue, which is the saturation point the plan watched for; the first-batch burst (all 384 groups in their first calls) inflates concurrency above the steady state. Watch whether usage eases once episodes stagger into tool phases. First batch: 2/32 groups at 1,982 s (v17 at multiplier 8 took 4,351 s for its first batch). |
