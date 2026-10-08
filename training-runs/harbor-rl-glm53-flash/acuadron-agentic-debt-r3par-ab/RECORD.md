# Run record: acuadron-agentic-debt-r3par-ab — live check of the parallel routing-replay loads (r23)

**Status:** Stopped 2026-10-08 02:47 UTC after train step 1 (check done).
<!-- gen-workflow:begin -->
**Date:** 2026-10-07
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `acuadron-agentic-debt-r3par-ab-`
**Experiment name:** `acuadron-agentic-debt-r3par-ab`
**W&B project:** `agentic-debt` (group `acuadron-agentic-debt-r3par-ab`)
**Dataset:** final-v9's (`lakefs://arena-inspect/f75dbe74…/acuadron-agentic-debt-final-110-sd015-20261006/manifest.jsonl`)
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-r23-20261007a` (digest `sha256:72dca7c1…`; miles `290940d7`, clean clone; kernel-cache seed `v7-20261007`)
**Template:** `acuadron-miles-deployer-v13`
**Base:** `acuadron-agentic-debt-final-v9` (workflow `dx8b7`)
<!-- gen-workflow:end -->
**Argo workflow:** `acuadron-agentic-debt-r3par-ab-2f2pf` (submitted 2026-10-07 23:46 UTC)
**Config deltas vs final-v9:** image r22 -> r23 (`--arena-routing-workers`, default 16, set explicitly); shape 64 -> 16 nodes (8 actor = DP 2, 8 engines), `gym-replicas` 288 -> 144, `rollout_batch_size` 64 -> 32, `global_batch_size` 512 -> 256; run names.
**Outcome:** The parallel loads are correct and fast. Rollout 0: 32 groups materialized in 39.1 s (1.2 s per group of about 670 MB raw, 550 MB/s); rollout 1: 32 groups in 32.4 s (1.0 s per group of about 1.1 GB raw, 1.1 GB/s); final-v9's serial loop: 19 s per 2.6 GB group (137 MB/s). 0 groups dropped for a lost ref, no `RoutingReplayError`, dynamic sampling reaped its 10 zero-variance groups as before. Numerics with the replayed routing in final-v9's band: log-prob gap 0.0330 / 0.0356 at steps 0 / 1 (final-v9: 0.032 / 0.034), train-rollout KL 0.0046 / 0.0050, grad_norm 0.086 / 0.077, TIS clip fraction 0.0008 / 0.0009. Train step 0 385 s at 38.6k tok/s (seeded kernels, as restart-ab-b), step 1 374 s at 67.0k tok/s. The image is re-tagged `miles-glm53-r24-20261008a` (same digest) for production.

## Goal

final-v9 loads each group's 50-60 routing payload files one at a time (about 2.5 GB per group at the mount's
single-stream 133-188 MB/s, about 19 s per group). That loop took 1,140-1,250 s per 64-group batch and set the step
time while 225-251 finished groups waited in the queue. r23 loads the payloads of one group in 16 threads; on the real
mount from a CPU pod (2026-10-07 23:40 UTC, 60 real blobs per set) one thread read 133-188 MB/s and 16 threads about
900 MB/s (2.7-3.7 s per group). This run checks the code in the loop: `Routing replay: materialized expert blobs for
32 groups in Xs` (expect X/32 of about 3 s, final-v9: 18-19 s), `rollout/routing_ref_lost_groups` 0, no
`RoutingReplayError`, and the log-prob gap at steps 0-1 in final-v9's band (0.032-0.035). Stop after step 2.

## Timeline

| time (UTC) | event |
|---|---|
| 2026-10-07 23:46 | Submitted. |
| 2026-10-07 23:50 | Ray head up; engines ready 23:55 (dummy weights, draft export); trainer ready 23:57, initial push 27.7 s. |
| 2026-10-07 23:58 | Rollout 0 collecting (128 groups in flight on 8 engines: 48 running requests per engine, about 30 tok/s per request). First group 00:37. |
| 2026-10-08 01:29 | Rollout 0 complete: 32 groups in 5,511 s; `materialized expert blobs for 32 groups in 39.1s`; 0 lost refs; 10 zero-variance groups dropped. |
| 2026-10-08 01:43 | Train step 0: `actor_train` 385 s at 38.6k tok/s; log-prob gap 0.0330, KL 0.0046, grad_norm 0.086. |
| 2026-10-08 02:39 | Rollout 1 complete: 32 groups in 4,154 s; `materialized expert blobs for 32 groups in 32.4s` (18.0 GB int16 batch). |
| 2026-10-08 02:46 | Train step 1: `actor_train` 374 s at 67.0k tok/s; log-prob gap 0.0356, KL 0.0050, grad_norm 0.077. Stopped. |
