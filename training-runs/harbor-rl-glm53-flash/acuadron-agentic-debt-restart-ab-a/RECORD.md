# Run record: acuadron-agentic-debt-restart-ab-a — restart-time check A: kernel-cache seed, early gyms, weight-push coverage

**Status:** Stopped 2026-10-07 01:06 UTC after the weight check (its purpose).
<!-- gen-workflow:begin -->
**Date:** 2026-10-07
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `acuadron-agentic-debt-restart-ab-a-`
**Experiment name:** `acuadron-agentic-debt-restart-ab-a`
**W&B project:** `agentic-debt` (group `acuadron-agentic-debt-restart-ab-a`)
**Dataset:** final-v7's 110 chains (`lakefs://arena-inspect/f75dbe74…/acuadron-agentic-debt-final-110-sd015-20261006/manifest.jsonl`)
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-r22-20261007a` (digest `sha256:d3552c75…`; miles `c4a16409`, clean clone; kernel-cache seed `v7-20261007`, 48,948 files)
**Template:** `acuadron-miles-deployer-v4` (gyms start with the trainer pod)
**Base:** `acuadron-agentic-debt-final-v7`
<!-- gen-workflow:end -->
**Argo workflow:** `acuadron-agentic-debt-restart-ab-a-2qbrz` (submitted 2026-10-07 00:33 UTC)
**Config deltas vs final-v7:** shape 64 -> 24 nodes (8 actor = DP 2, 16 engines), `gym-replicas` 288 -> 144, `rollout_batch_size` 32, `global_batch_size` 256; image r20 -> r22; template v10 -> acuadron v4; `check_weight_update_equal: true`, `check_weight_update_selector: target`, `check_weight_update_skip_list: [visual]`; run names.
**Outcome:** The trainer's initial push covers every target weight of the engines: the weight checker compared about 1,300 tensors per engine rank on all 128 ranks after the push, with no mismatch (no `check tensor equality failed`; only the skipped vision tower, which the reset does not poison, stays at its disk value). Dummy engine weights (check B) are therefore safe. Side effect to know: `reset_tensors` runs with the default selector and poisons the NEXTN draft too, so a run with `check_weight_update_equal` and speculation decodes with accept length 1.00; use the checker only in check runs.

## Goal

The final-v7 restart took 6,440 s from submit to the first optimizer update: Argo + scheduling 4.4 min,
engine start 19.8 min (220 GB disk read 11.3 min, NEXTN draft re-read of all 120 shards 3 min), initial push
2.7 min, rollout 0 33.6 min (gyms deployed only after the trainer's NATS handshake), train step 0 46.8 min
(about 41 min of kernel compile and autotune). Check A measures the step-0 time with the kernel-cache seed and the gym start with template v4, and it verifies that the trainer's initial push covers every target weight of the engines (the weight checker snapshots the disk weights, poisons them, and compares them after the push), the precondition of check B. Stop after train step 1.

## Timeline

| time (UTC) | event |
|---|---|
| 2026-10-07 00:33 | Submitted. |
| 2026-10-07 00:37:56 | Ray head up; gyms deployed at 00:39:31 by template v4 (wait-trainer-running), 20 min before the first task. |
| 2026-10-07 00:46:45-00:50:14 | Engines load from disk: target 382-395 s (16 engines; final-v7's 48 took 650-689 s), NEXTN draft re-reads all shards 158-193 s; ready 00:51:56. |
| 2026-10-07 00:52-00:59 | Weight checker: snapshot and reset (skip list `visual`) on all engine ranks, then the trainer's push (30.6 s), then compare: passed on every rank. |
| 2026-10-07 01:00 | Rollout 0 collecting; accept len 1.00 (the reset poisoned the draft). Stopped at 01:06. |
