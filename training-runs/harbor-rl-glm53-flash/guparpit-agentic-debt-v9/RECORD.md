# Run record: guparpit-agentic-debt-v9 (r56) — r54 with the next rollout shard prefetched during the train step

**Status:** Retired
<!-- gen-workflow:begin -->
**Date:** 2026-10-04
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v9-`
**Experiment name:** `guparpit-agentic-debt-v9`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v9`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-recon-20261002-flashmla-evfix`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v7`
<!-- gen-workflow:end -->
**Argo workflow:** `guparpit-agentic-debt-v9-d6tx9`
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Trainer config deltas vs base:** `prefetch_rollout_data` true; `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v9`, seeded from r52 `iter_0000069` (the seed r54 uses)
**Outcome:** Retired 2026-10-05 14:30Z on the owner's word, at rollout 89, last save `iter_0000079`. Replaced by v13 (`r56-ns`, the same config at inflight multiplier 8 with `max_weight_staleness` 8), seeded from r54 `iter_0000089`. Stopped with `spec.shutdown: Stop`; the onExit cleanup ran.

A paired speed arm of r54 (`guparpit-agentic-debt-v7-h6668`). Same image, same
config, same seed; the one change is `--prefetch-rollout-data`. The owner chose
this arm over an 8-node replay profile because it measures the live path: real
shards, the live NATS cadence, and the alternation between trainer-bound and
rollout-bound steps.

## Goal

Measure what `--prefetch-rollout-data` saves per train step. The flag pulls
each train rank's shard of the next rollout to its node while the current step
trains, so the next step does not wait for the transfer. It hides the
`[r3-timing] phase=fetch` window and nothing else.

Expected effect, from r54 at rollouts 70 to 72: the fetch of about 21 GB per
node takes 33 to 37 s mean and 65 to 77 s max per rank per step; `fill` takes
1 s; `perf/data_preprocess_time` reads 30 to 69 s. A trainer-bound step is
1,421 s (rollout 71, `perf/train_wait_time` 101 s), so the flag can save 2 to 5%
of such a step. When the trainer waits for rollouts (rollout 70:
`train_wait_time` 3,966 s of a 7,472 s step) it saves nothing.

## Upstream check

- `--prefetch-rollout-data` is a fork flag (`miles/utils/arguments.py`;
  `examples/arena/RECONCILE.md`, "Prefetch port"). Upstream `main` has no
  equivalent: its train loop fetches the shard at the start of the step.
- The flag needs `--use-fault-tolerance` (only the fault-tolerant trainer actor
  is threaded, so the prefetch can run in its own concurrency group next to
  `train()`) and `--object-store-backend ray` (the prefetch is a Ray
  `fetch_local` pull; Mooncake has no such pull). Both hold in this config:
  `use_fault_tolerance: true`, and `ray` is the argument default.
- The port to the upstream worker structure passes the CPU tests with a local
  Ray cluster. It has no GPU run on this base (RECONCILE open item 5): the only
  GPU run was on the older branch `arpit-r3-datapath`. This run is also that
  check.
- The routed-method fix `1b5d2978c` is in this image: without it the prefetch
  ran in the default concurrency group, behind `train()`, and hid nothing.

## Measurement plan

Control: r54 at the same rollout ids. Both runs start from the same
`iter_0000069`; the data order is not pinned, so the rollouts differ in content
and only aggregate step timing is compared.

| Question | Signal | Gate |
| --- | --- | --- |
| The prefetch runs | `[r3-timing] phase=prefetch_start` and `phase=prefetch_done` lines on every train rank, every step from rollout 71 on | present each step; a step without them is a defect |
| The fetch window is hidden | `[r3-timing] phase=fetch seconds=` per rank, and `perf/data_preprocess_time`, on trainer-bound steps (`perf/train_wait_time` < 300 s) | fetch under 5 s and `data_preprocess_time` under 10 s (r54: 33 to 37 s mean, 65 to 77 s max; 30 to 69 s) |
| Step time | `perf/step_time` and `perf/actor_train_time` on trainer-bound steps against r54 at matched rollout ids | `step_time` lower by the hidden fetch (about 30 to 70 s); `actor_train_time` equal within noise |
| Throughput | `perf/actor_train_tok_per_s` | equal to r54 within noise (36,025 at rollout 71); the flag changes no compute |
| Memory | `[r3-mem]` RssAnon and node MemAvailable; no OOM | each node holds two shards at once; no `OutOfMemory` in the trainer log |
| Reward | `rollout/group_metrics/reward.mean` | within noise of r54 (0.717 at rollout 70); the flag changes no data |
| Rollout-bound steps | `perf/train_wait_time` | excluded from the comparison; the flag cannot help there |

## Launch

| Time (UTC) | Event |
| --- | --- |
| 2026-10-04 23:13 | Seed copied from r52 `iter_0000069` (`guparpit-agentic-debt-v5`, a fresh copy, separate from r54's): 135 objects match by name and size; sidecar `{"rollout_id": 69}` without `wandb_run_id`; data state `arena_data_source_state_69.pt`; tracker 69 written last (`metrics-1001/ckcopy-v9.sh`). |
| 23:15 | In-image parse of the rendered argv (`recon2/parse/fullparse9.py` in `miles-glm53-recon-20261002-flashmla-evfix`, tree `0ffc4044ec`, `WORLD_SIZE` 320, `RANK` 0, the CUDA stub `libcuda.so.1` on `LD_LIBRARY_PATH`; `parse-v9.out`): exit 0, 267 argv tokens. `prefetch_rollout_data True`, `object_store_backend ray`, `use_fault_tolerance True`, `mini_ft_controller_enable True`, `calculate_per_token_loss True`, `grpo_std_normalization False`, `arena_length_reward_coef 0.0`, `miles_dsa_sparse_attention_forward_backend flash_mla`, `sglang_moe_runner_backend triton`, `expert_model_parallel_size 8`, `glm5_next_kda_tp True`, `skip_actor_forward_only True`, `rollout_batch_size 32`, `global_batch_size 256`, `save` the v9 dir. Against the r54 argv the only new token is `--prefetch-rollout-data`. |
| 23:17:36 | Submit `guparpit-agentic-debt-v9-d6tx9`. Node exclusions refreshed at launch: 304 B200 nodes, 63 bad (20 NotReady, 48 with the `burn-in` taint, 5 both); 311 excluded = the r54 list (262, of which 227 name instances that no longer exist) plus the 63 bad nodes (49 new). `prompt-data-list` byte-equal to r54's. |
| 23:20:45 | Trainer PyTorchJob `guparpit-agentic-debt-v9-d6tx9-trainer` created by the deployer and still present at 23:38 (the auctioneer trainer of 18:23 was deleted by the training-operator 12 s after creation; this one was not). Kueue workload `pytorchjob-guparpit-agentic-debt-v9-d6tx9-trainer-416f3`, queue `gpu.p6-b200-48xlarge`, priority 1000. |
| 23:29 to 23:38 | Not admitted: `topology "p6-network" allows to fit only 37 out of 40 pod(s). Total nodes: 300; excluded: affinity: 19, resource "memory": 135, resource "nvidia.com/gpu": 47, taint burn-in: 61`. The run is the only queued workload in the queue and waits for 3 more free B200 nodes (r53, r54 and r55 hold 120). No preemption on prod-bom-v2. The gym Deployment starts after the trainer connects to NATS, so nothing of this run uses a GPU while it waits. |
| 2026-10-05 14:30 | Retired on the owner's word together with r53, r54, r55 and r56 (r56 = this run), at rollout 89, last save `iter_0000079`. Replaced by v13 (`r56-ns`, the same config at inflight multiplier 8 with `max_weight_staleness` 8), seeded from r54 `iter_0000089`. Stopped with `spec.shutdown: Stop`. |
