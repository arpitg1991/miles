# Run record: acuadron-agentic-debt-r3par-ab — live check of the parallel routing-replay loads (r23)

**Status:** Running
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
**Outcome:** (pending)

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
