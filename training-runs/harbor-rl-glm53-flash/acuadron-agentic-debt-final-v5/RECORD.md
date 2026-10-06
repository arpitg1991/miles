# Run record: acuadron-agentic-debt-final-v5 — final-v3 recipe on the r20 trainer (dsa-qp + the R3 data path of arpit-r3-datapath)

**Status:** Retired
<!-- gen-workflow:begin -->
**Date:** 2026-10-06
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `acuadron-agentic-debt-final-v5-`
**Experiment name:** `acuadron-agentic-debt-final-v5`
**W&B project:** `agentic-debt` (group `acuadron-agentic-debt-final-v5`)
**Dataset:** the final-v3 dataset (`agentic-debt-final`, 871 checkpoints; gym `agentic-debt`)
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-r20-20261006a` (digest `sha256:437bf865…`; miles `acuadron/dsa-qp` at `bf9f3370`, built on `miles-glm53-r18dsa-20260930a`; pushed to `arena-github/miles` because `ecr-dev` lost push rights on `arena-slime-dev` on 2026-10-05)
**Template:** `guparpit-miles-deployer-v10`
**Base:** `acuadron-agentic-debt-final-v3`
<!-- gen-workflow:end -->
**Argo workflow:** `acuadron-agentic-debt-final-v5-nbz8w` (submitted 2026-10-06 00:35 UTC)
**Trainer config deltas vs base:** `prefetch_rollout_data: true` (new key); run names -> v5. Workflow: `trainer-image` r19 -> r20, `excluded-nodes` = the final-v4 list (final-v3's 98 + the 3 NVLink-fault nodes), `replicas` 48 (16 actor + 32 engines) as in v3.
**Checkpoints:** `/mnt/scratch-s3files-rw/acuadron/checkpoints/slime_experiments/acuadron-agentic-debt-final-v5` (fresh; starts from `ref_load`)
**Outcome:** Stopped 2026-10-06 09:40 UTC at step 14 (last `iter_0000009`, saved 06:45 UTC in 90 s) to free the 48 nodes for final-v6 (the same trainer on the republished live-baseline `agentic-debt-final`). Goal met at steps 0-2. Resumable under the same experiment name. Steps 0-2 measured (2026-10-06 03:20 UTC): the data path is gone from the step (`data_preprocess` 177 s -> 2.8 s, `train_wait` 249 -> 29 s at step 2 vs final-v3), the per-token train rate is unchanged (34.4 vs 33.6 us/token/rank at step 2), numerics in the final-v3 range. Step 2: 679 s vs 941 s (-28%).

## Goal

Measure the R3 data path of the merged trainer against final-v3 on the same recipe and shape, and check the numerics.

final-v3 (r19: int32 routing, pad-concat-slice fill, no prefetch) at step 2 spent 177 s in
`perf/data_preprocess_time` and 249 s in `perf/train_wait_time` of a 941 s step. The merge of
`arpit-r3-datapath` (int16 routing, local-rows fill, `--prefetch-rollout-data`) measured 3-9 s of
`data_preprocess` on Arpit's ns stack (guparpit-agentic-debt-v18, GBS 256, DP 2). Expected here at GBS 512,
DP 4: `data_preprocess` under 15 s, `[r3-timing] phase=fetch` under 10 s with `prefetch=hit`, `phase=fill` 1-3 s.

Numerics to match final-v3: `train_rollout_logprob_abs_diff` 0.0217 / 0.0226 (steps 0 / 1), `grad_norm` 0.057 / 0.044,
`train_rollout_kl` 0.0023-0.0025; `[r3-digest]` lines equal across the TP ranks of a (dp, pp) cell.

## Trainer stack

`acuadron/dsa-qp` at `bf9f3370`: `arpit-glm-53` 6ae5099f + `arpit-dsa-3608` (FlashMLA sparse forward, dynamic-shape
TileLang backward) + `--glm5-next-dsa-qp` (ADR-0018) + `arpit-r3-datapath` e193eae7 merged at `8a14af78`.
Verification before launch: 403 fast tests in the image on the build host (2 Mooncake tests need the missing
package); 312 fast tests on a B200 node (`kdatp/dsaqp/20261006a`: prefetch concurrency group, dsa-qp layouts,
`test_thd_fill_matches_the_old_fill`, rollout prefetch); DSA-QP parity PASS on the same job (indices bitwise,
output bitwise, grads <= 5.7e-3 rel-L2, negative control 0.76; layer 64K 404.7 -> 77.3 ms).

## Timeline

| time (UTC) | event |
|---|---|
| 2026-10-06 00:35 | Submitted as `acuadron-agentic-debt-final-v5-nbz8w`. |
| 2026-10-06 00:42 | 237 prompts loaded; args carry `prefetch_rollout_data True`, `glm5_next_dsa_qp True`, EP 8, `object_store_backend ray`. |
| 2026-10-06 01:47 | Rollout 0: 64 groups, 1,944 rows, 35.8M tokens (39.8M padded), 3,090 s; routing 25.75 GB int16 (720 B/token), 6.5 GB per DP shard, `convert` 12.3 s, `put` 2.0 s per shard. |
| 2026-10-06 02:35 | Step 0 (cold): `actor_train` 2,795 s (final-v3: 2,779), `data_preprocess` 23.7 s (first rollout, no prefetch possible: fetch 3.6-29.9 s by rank, fill 0.3-0.5 s), `update_weights` 43.5 s; grad_norm 0.0603, log-prob gap 0.0257, kl 0.00316. |
| 2026-10-06 02:54 | Step 1: 2,980 rows, 71.0M padded tokens (17.75M per DP rank, 721 micro-batches), `actor_train` 1,107 s (warm-up variance: final-v3 step 1 725 s for 17.3M), `data_preprocess` 2.5 s, `train_wait` 28 s, step 1,136 s; fetch `prefetch=hit` 0.0 s on all 128 ranks, fill 0.4-0.5 s, prefetch pull 6-74 s hidden inside step 0; grad_norm 0.0420, gap 0.0270. |
| 2026-10-06 03:05 | Step 2: 2,816 rows, 75.4M padded tokens (18.86M per DP rank, 662 micro-batches), `actor_train` 649.6 s = 34.4 us/token/rank (final-v3 step 2: 691 s for 20.6M = 33.6 us), `data_preprocess` 2.8 s, `train_wait` 28.9 s, `update_weights` 24.5 s, **step 678.6 s** (final-v3: 940.5 s), 107k tok/s; grad_norm 0.0448, gap 0.0290, kl 0.00375. |
| 2026-10-06 03:17 | Rollout 3 complete (1,381 s, queue 3): the loop is rollout-bound now; the trainer started step 3 at once, so 94 ranks fetched during the still-running prefetch (`prefetch=miss`, 9-73 s, the same pull shared). |

## Results: the R3 data path, final-v3 (r19) vs final-v5 (r20), same recipe and shape (step 2)

| | final-v3 (int32, pad-concat-slice fill, no prefetch) | final-v5 (int16, local-rows fill, prefetch) |
|---|---|---|
| routing bytes per DP shard | ~13 GB (int32, 1,440 B/token) | 6.4-12.7 GB (int16, 720 B/token) |
| `perf/data_preprocess_time` | 177 s | **2.8 s** |
| `[r3-timing] phase=fetch` | - | 0.0 s, `prefetch=hit`, 128/128 ranks |
| `[r3-timing] phase=fill` | - (the old fill: 45-109 s in the r47 study) | 0.4-0.5 s |
| `perf/train_wait_time` | 249 s | 28.9 s |
| `perf/actor_train_time` per padded token per DP rank | 33.6 us | 34.4 us |
| `perf/step_time` | 940.5 s | **678.6 s** |
| `train_rollout_logprob_abs_diff` steps 0 / 1 / 2 | 0.0217 / 0.0226 / 0.0226 | 0.0257 / 0.0270 / 0.0290 (final-v1 r17: 0.0240 / 0.0252 / 0.0257) |
| `grad_norm` steps 0 / 1 / 2 | 0.0571 / 0.0442 / - | 0.0603 / 0.0420 / 0.0448 |

The int16 cast is exact (288 experts, -1 pad), the fill is byte-equal to the old fill
(`test_thd_fill_matches_the_old_fill`, run on the B200 node before launch) and the prefetch returns the
same object, so the trainer numerics do not change; the log-prob gap moves with the batch (reward 0.50 ->
0.66 -> 0.68 -> 0.24 over rollouts 0-3).

With the data path gone the loop is rollout-bound at `arena_inflight_multiplier` 4: step 3 waited for
rollout 3 (1,381 s between rollouts against a 650 s train step). Supply (multiplier, engine share) is the
next lever, not the trainer.

| 2026-10-06 09:36 | Step 14 done; rollout 14: 64 groups, avg_reward 0.650, 2,051 s (queue 2): rollout-bound at multiplier 4. |
| 2026-10-06 09:40 | `argo stop` for final-v6 capacity (kueue: 2,136 + 256 of 2,424 B200 GPUs in use; 48 more nodes did not fit). Last checkpoint `iter_0000009`. |
