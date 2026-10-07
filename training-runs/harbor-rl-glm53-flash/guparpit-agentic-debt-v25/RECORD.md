# Run record: guparpit-agentic-debt-v25 (router-abs16) — v20 (qp-fi) with the router imbalance threshold lowered from 64 to 16

**Status:** Prepared
<!-- gen-workflow:begin -->
**Date:** 2026-10-07
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v25-`
**Experiment name:** `guparpit-agentic-debt-v25`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v25`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-wvspans-20261001a@sha256:e29ba91a3392fd250e78f551fca6f33061520a2b456134693d419e1bf5c9eab9`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-qp-20261006a@sha256:5e540e4b3f04e93480d992154da0686783439114780b9a41ca4affb194a3c667`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v20`
<!-- gen-workflow:end -->
**Argo workflow:** not submitted
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Trainer config deltas vs base:** `router_balance_abs_threshold` 64 (router default) -> 16 (new key; miles passes every `RouterArgs` flag with the `router_` prefix, `miles/backends/sglang_utils/router_args_utils.py`); `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run. Workflow parameters differ from v20 only in `experiment-name`, `excluded-nodes` and `miles-config`.
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v25`, seeded from v20 `iter_0000169` (its latest save at copy time, 2026-10-07 20:00Z; 135 objects match by name and size; sidecar `{"rollout_id": 169}` without `wandb_run_id`; data state `arena_data_source_state_169.pt`; tracker 169 written last; `metrics-1001/ckcopy-v25.sh`).
**Outcome:** Prepared

## Goal

Test whether the router, not the engine count, makes the trainer wait. On 2026-10-07 19:38Z the three live runs (v19, v20, v23, all at the default threshold 64) showed per-engine running requests of 31 to 37 at the median and 76 to 94 on one engine per run. That engine had KV usage 0.96 to 0.98 and was the only one with queued requests; the others ran at KV 0.54 to 0.63. The trainer waited 745 to 1,400 s per step for rollouts. A lower threshold makes the router send new turns to the least-loaded engine sooner. Sibling arm: guparpit-agentic-debt-v24 (32). Control: v20 itself at the same rollout ids (it continues from the same save).

## Upstream check

- Router in the image: `sglang-router 0.3.2` (Python package; `sglang_router_rs.abi3.so`), built with SGLang `0.5.21.dev67+g14a1fa7`. Source: `sgl-project/sglang` commit `14a1fa7d93e87355375b2c3b6fa17255a4b8aa07`, `sgl-model-gateway/Cargo.toml` version `0.3.2`, policy `sgl-model-gateway/src/policies/cache_aware.rs`.
- Decision code, `select_worker`:
  ```rust
  let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
      && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);
  if is_imbalanced { return self.select_worker_min_load(...); }   // shortest queue, ties random
  // balanced: prefix match on the approximate radix tree
  let selected_idx = if match_rate > self.config.cache_threshold { /* the worker with the longest prefix match */ }
                     else { /* a worker with the minimum load, ties random */ };
  ```
  `load` is the router's count of in-flight requests per worker. Both conditions must hold. At 64 / 1.5, the observed spread (max 76 to 94, min 31 to 32, difference 45 to 63) never counted as imbalanced, so every turn whose prefix matched above 0.3 went back to its cached engine.
- At 16 / 1.5 the router treats the system as imbalanced when `max - min > 16` and `max > 1.5 x min`. With min about 32, the switch happens once the busiest engine passes 49 requests.
- Cost: a turn sent off its cached engine re-prefills its whole context there. agentic-debt episodes run 100K to 140K tokens, so a moved turn can cost a prefill of that size. The arm wins only if shorter batch collect times outweigh that cost.
- Upstream router defaults are 64 / 1.5 with `cache_threshold` 0.3 (the values the live runs use, unset in the config). No upstream recipe sets a lower threshold for long multi-turn agent sessions; this test is the measurement.

## Measurement plan

All at matched rollout ids (170 onward) against v20, which trained the same rollout ids from the same save.

| Signal | Source | Expectation |
| --- | --- | --- |
| Engine load spread | SGLang `Decode batch` lines in worker-0: per engine `#running-req`, max / median ratio, share of lines with `#queue-req` > 0, max `full token usage` | lower spread and less queueing than v20 |
| Prefix-cache cost | SGLang `Prefill batch` lines: `#cached-token` / (`#cached-token` + `#new-token`), per engine and total; prefill tokens per rollout | some loss of cache hits; record it |
| Rollout supply | `Rollout N complete ... in X s` collect time; `perf/train_wait_time` | the arm wins only if both drop |
| Decode speed | `gen throughput (token/s)` per engine | no drop beyond noise |
| Reward | batch reward and `rollout/group_metrics/reward.mean` | within noise of v20 |
| Errors | trainer log, Kueue | none new |

Router argument check (in-image parse, `recon2/parse/fullparse12.py`, image `miles-glm53-qp-20261006a`, `parse-v25.out`): the rendered router args read `policy cache_aware, cache_threshold 0.3, balance_abs_threshold 16, balance_rel_threshold 1.5`; v20 reads 64. Other parsed keys differ from v20 only in the run names and paths. On the desktop the parse shows `load` = the base DCP because the checkpoint mount is absent there (v20's parse shows the same); the pod log confirms the loaded iteration.

Compare script: `/workplace/guparpit/kdfast/scratch/router-ab/compare.py`.

## Launch

| Time (UTC) | Event |
| --- | --- |
| 2026-10-07 20:02 | Seed copied from v20 `iter_0000169`: 135 objects match; sidecar `{"rollout_id": 169}`; data state 169; tracker 169. |
| 20:04 | Run files written (`router-ab/make_router_runs.py`); exclusions refreshed: 304 B200 nodes, 5 bad, plus `i-06818d629c2521eb7`; 350 excluded. |
