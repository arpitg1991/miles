# Run record: acuadron-agentic-debt-final-v5 — final-v3 recipe on the r20 trainer (dsa-qp + the R3 data path of arpit-r3-datapath)

**Status:** Running
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
**Outcome:** (pending)

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
