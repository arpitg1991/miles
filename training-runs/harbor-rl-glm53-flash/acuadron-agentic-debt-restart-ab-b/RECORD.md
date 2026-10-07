# Run record: acuadron-agentic-debt-restart-ab-b — restart-time check B: A plus dummy engine weights and the NEXTN draft export

**Status:** Stopped 2026-10-07 ~01:40 UTC after train step 1 (check done).
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
**Outcome:** Restart (submit to the end of train step 0) 51.3 min at 24 nodes, against 112 min for engine-ab-v1 at the same shape on r20 and template v10, and 107 min for final-v7 at 64 nodes. Engines ready 4.7 min after the Ray head (dummy target 3.4-3.7 s, draft export 11-16 s), so the trainer init (7.2 min) is now the critical path; train step 0 380 s (637 tok/s per GPU; cold r20/r21 at this shape: 2,707-2,828 s at 90-93 tok/s per GPU); log-prob gap 0.0212 at step 0 (engine-ab-v1: 0.0198), accept length 2.67 (the draft export works). Rollout 0 (32.2 min) is now 63% of the restart.

## Goal

The final-v7 restart took 6,440 s from submit to the first optimizer update: Argo + scheduling 4.4 min,
engine start 19.8 min (220 GB disk read 11.3 min, NEXTN draft re-read of all 120 shards 3 min), initial push
2.7 min, rollout 0 33.6 min (gyms deployed only after the trainer's NATS handshake), train step 0 46.8 min
(about 41 min of kernel compile and autotune). Check B measures the engine start with dummy weights (no 220 GB read) and the draft export (15 GB), and the whole restart time. Numerics: `train_rollout_logprob_abs_diff` at step 0 in the final-v7 band (~0.026), `accept len` about 2.6 (a broken draft gives about 1).

## Timeline

| time (UTC) | event |
|---|---|
| 2026-10-07 00:33 | Submitted. |
| 2026-10-07 00:38:12 | Ray head up (4.7 min after submit: Argo steps 3 min, image pull and start 1.7 min). Gyms deployed 00:39:31 (template v4). |
| 2026-10-07 00:40:43-00:41:08 | Engines: dummy target weights 3.4-3.7 s; NEXTN draft from the export 11.4-16.1 s (final-v7: 650-689 s and 166-196 s). Ready 00:42:48-00:42:54. |
| 2026-10-07 00:45:23 | Trainer ready (7.2 min after the Ray head; now the critical path). Initial push 30.2 s fills the engines. |
| 2026-10-07 00:45:55 | Rollout 0 collecting (12.4 min after submit; final-v7: 26.9 min). First group 00:56:24, 32nd 01:18:06 (32.2 min). |
| 2026-10-07 01:24:46 | Train step 0 done: `actor_train` 379.6 s for 15.5M tokens (40.8k tok/s), `update_weights` 30.1 s; log-prob gap 0.0212, grad_norm 0.093, accept length 2.67. |
| 2026-10-07 01:39 | Train step 1: `actor_train` 374.5 s at 69.8k tok/s (1,091 tok/s per GPU, warm; cold r20/r21 step 1 at this shape: 515-571 per GPU). Stopped. |
