# Run record: guparpit-agentic-debt-v26 (qp-fi-final-v2-m12) — v23 at in-flight multiplier 12

**Status:** Queued
<!-- gen-workflow:begin -->
**Date:** 2026-10-07
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v26-`
**Experiment name:** `guparpit-agentic-debt-v26`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v26`)
**Dataset:** `lakefs://arena-inspect/6e4b517ce5355126d1ec65e7da35a2b85dee083cd6bc5adc02943d1ab5b2c2ac/internal/agentic-debt-r3/agentic-final-v2/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `6e4b517ce5355126d1ec65e7da35a2b85dee083cd6bc5adc02943d1ab5b2c2ac` (lakeFS `main` on 2026-10-07 01:40 UTC)
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-wvspans-20261001a@sha256:e29ba91a3392fd250e78f551fca6f33061520a2b456134693d419e1bf5c9eab9`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-qp-20261006a@sha256:5e540e4b3f04e93480d992154da0686783439114780b9a41ca4affb194a3c667`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v23`
<!-- gen-workflow:end -->
**Argo workflow:** `guparpit-agentic-debt-v26-5znln`
**W&B run:** (set at launch)
**Task pin:** `a7b647a2df19` (every manifest row)
**Trainer config deltas vs base:** `arena_inflight_multiplier` 8 -> 12 (384 groups in flight); `experiment_name`, `project_name`, `arena_sample_summary_dir` -> v26. Workflow: `gym-replicas` 288 -> 400. Everything else is the v23 file: QP DSA core and the flashinfer top-k, staleness cap 8, output queue cap 64, prefetch on, `NatsRolloutFn`, FlashMLA forward, triton MoE runner, EP8, layers 11/11/11/12, router defaults (cache_aware, abs 64, rel 1.5), 8 trainer + 32 engine nodes.
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v26`, seeded from v23 `iter_0000029` (2026-10-07 21:28Z)
**Outcome:** (set at the end)

## Goal

The owner asked for a new run with a higher in-flight multiplier, started from v23's most recent save (2026-10-07). v23 keeps running at multiplier 8 and is the control at the same rollout ids (30 onward).

v23 at multiplier 8 waits for rollouts on every step: since rollout 20, trainer compute is about 670 s per step and the wait about 745 s, with all 12 steps over 300 s. This run tests whether more in-flight work on the same 32 engines closes that wait.

## Why 12, and the risks

Measured on 2026-10-07 at 19:38 UTC from the SGLang decode lines of v23 (multiplier 8, 30 of 32 engines, last 15 minutes):

| Measure | v23 at 8x |
| --- | --- |
| Running requests per engine, min / median / max | 31 / 36 / 83 |
| KV cache use, median / max | 0.54 / 0.96 |
| Engines with queued requests | 1 |

v23 episodes are shorter than on agentic-debt-766 (about 91K tokens against about 140K). On agentic-debt-766, multiplier 12 filled the KV cache (median 0.97) and halved decode throughput per engine (v21 and v22, retired 2026-10-06). On v23 the expected load at 12 is about 54 running requests per engine and a KV median near 0.8.

Risks:

- One engine per run already runs hot (83 requests, KV 0.96) because the cache_aware router keeps sessions on their cached engine. At 12 it can saturate first. The router balance threshold is tested separately in v24 and v25; this run keeps the router defaults so that the multiplier is the only change.
- The dataset has 121 chains. With 384 groups in flight, about 3.2 epochs are dispatched ahead of training, so each chain has about 3 groups out at once, and data age rises. The staleness cap (8) then drops more groups (`rollout/num_old_age_dropped`).
- The 11 spec-conflict chains of agentic-final-v2 (report `v23-regression-validity-180858`) are still in the dataset; they affect v23 and this run equally.

## Upstream check

- The in-flight multiplier is a fork setting (`arena_inflight_multiplier`, `miles_plugins/arena/nats_arena`); upstream miles has no NATS publisher and no equivalent.
- The other settings are v23's; its record holds their upstream checks (QP is a fork port of `acuadron/dsa-qp`; flashinfer top-k is an upstream choice of `--miles-dsa-topk-backend`).

## Measurement plan

Compare with v23 at matched rollout ids (30 onward).

| Question | Signal | Gate |
| --- | --- | --- |
| Does the wait drop? (primary) | `perf/train_wait_time`, rollout collect time (`Rollout N complete ... in S s`) | wait under 300 s on more than 80% of steps after the first-batch burst; collect time below v23's |
| Do the engines cope? | SGLang decode lines: running requests per engine (min/median/max), `full token usage`, share of lines with `#queue-req` > 0, decode tok/s per engine | KV median below 0.9; queued share below 20% |
| Data age | `rollout/weight_version` mean and max, `rollout/fully_async/*`, `rollout/num_old_age_dropped`, `rollout/reward_old_age_dropped` | record; compare with v23 |
| Reward | batch reward, W&B `rollout/group_metrics/reward.mean` | within the v23 band at the same rollouts; no collapse |
| Errors | `WeightVersionSpansError`, CUDA faults, Kueue-taint death | none |

Abort criterion (report to the owner; do not stop the run without the owner's word): after the first-batch burst, KV median at or above 0.95 with queued requests on more than 50% of decode lines for 30 minutes.

## Launch

| Time (UTC) | Event |
| --- | --- |
| 2026-10-07 21:28 | Seed copied from v23 `iter_0000029` (`metrics-1001/ckcopy-v26.sh`): 135 objects match by name and size; sidecar `{"rollout_id": 29}` without `wandb_run_id`; data state `arena_data_source_state_29.pt`; tracker 29 written last. `iter_0000039` had not landed. |
| 21:33 | Run files from v23 (`/workplace/guparpit/kdfast/scratch/v26/make_v26.py`): config differs from v23 in 4 non-comment lines (`experiment_name`, `project_name`, `arena_inflight_multiplier`, `arena_sample_summary_dir`); workflow params differ in `experiment-name`, `gym-replicas`, `miles-config`, `excluded-nodes`. `prompt-data-list` is byte-equal to v23's. |
| 21:36 | In-image parse (`recon2/parse/fullparse12.py`, `CFG=/work/v26.yaml` with `hf_checkpoint` remapped to `/work/hf`, image `miles-glm53-qp-20261006a`; `parse-v26.out`): exit 0, 274 argv tokens. `arena_inflight_multiplier 12`, `glm5_next_dsa_qp True`, `miles_dsa_topk_backend flashinfer`, `max_weight_staleness 8`, `arena_output_queue_groups 64`, `prefetch_rollout_data True`, `NatsRolloutFn`, `flash_mla`, `triton`, EP 8, 11/11/11/12, router cache_aware 0.3 / 64 / 1.5, the pinned agentic-final-v2 manifest. On the desktop `load` shows the base DCP because the checkpoint mount is absent; the pod log confirms the loaded iteration. |
| 21:34:59 | Submit `guparpit-agentic-debt-v26-5znln`. Node exclusions refreshed at submit: 303 B200 nodes, 4 bad (tainted); 349 excluded (the v23 list plus the 4 bad and `i-06818d629c2521eb7`). |
| 21:38:07 | Trainer PyTorchJob `guparpit-agentic-debt-v26-5znln-trainer` created. Queued: Kueue "insufficient unused quota for nvidia.com/gpu in flavor gpu.p6-b200-48xlarge, 152 more needed" (19 nodes); queue `gpu.p6-b200-48xlarge` pending 2, admitted 18. No run was stopped to make room. Log capture: `qp/v26-follow.log` (`router-ab/follow.sh`, reconnects until worker-0 exists). |
