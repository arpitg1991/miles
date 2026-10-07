# Run record: acuadron-agentic-debt-restart-ab-b — restart-time check B: A plus dummy engine weights and the NEXTN draft export

**Status:** Running
<!-- gen-workflow:begin -->
**Date:** 2026-10-07
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `acuadron-agentic-debt-restart-ab-b-`
**Experiment name:** `acuadron-agentic-debt-restart-ab-b`
**W&B project:** `agentic-debt` (group `acuadron-agentic-debt-restart-ab-b`)
**Dataset:** final-v7's 110 chains (`lakefs://arena-inspect/f75dbe74…/acuadron-agentic-debt-final-110-sd015-20261006/manifest.jsonl`)
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-r22-20261007a` (digest `sha256:d3552c75…`; miles `c4a16409`, clean clone; kernel-cache seed `v7-20261007`, 48,948 files)
**Template:** `acuadron-miles-deployer-v4` (gyms start with the trainer pod)
**Base:** `acuadron-agentic-debt-final-v7`
<!-- gen-workflow:end -->
**Argo workflow:** `acuadron-agentic-debt-restart-ab-b-pfkgk` (submitted 2026-10-07 00:33 UTC)
**Config deltas vs final-v7:** shape 64 -> 24 nodes (8 actor = DP 2, 16 engines), `gym-replicas` 288 -> 144, `rollout_batch_size` 32, `global_batch_size` 256; image r20 -> r22; template v10 -> acuadron v4; `sglang_load_format: dummy`, `sglang_speculative_draft_model_path: /mnt/scratch-fast-1a-rw/acuadron/models/GLM-5.3-Flash-BF16-nextn`, `sglang_speculative_draft_load_format: auto`; run names.
**Outcome:** (pending)

## Goal

The final-v7 restart took 6,440 s from submit to the first optimizer update: Argo + scheduling 4.4 min,
engine start 19.8 min (220 GB disk read 11.3 min, NEXTN draft re-read of all 120 shards 3 min), initial push
2.7 min, rollout 0 33.6 min (gyms deployed only after the trainer's NATS handshake), train step 0 46.8 min
(about 41 min of kernel compile and autotune). Check B measures the engine start with dummy weights (no 220 GB read) and the draft export (15 GB), and the whole restart time. Numerics: `train_rollout_logprob_abs_diff` at step 0 in the final-v7 band (~0.026), `accept len` about 2.6 (a broken draft gives about 1).

## Timeline

| time (UTC) | event |
|---|---|
| 2026-10-07 00:33 | Submitted. |
