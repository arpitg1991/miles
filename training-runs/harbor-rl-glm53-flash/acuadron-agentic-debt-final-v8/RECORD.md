# Run record: acuadron-agentic-debt-final-v8

**Status:** Running
**Date:** 2026-10-07
**Family:** `harbor-rl-glm53-flash`
**Experiment name:** `acuadron-agentic-debt-final-v8`
**Argo generateName:** `acuadron-agentic-debt-final-v8-`
**Argo workflow:** `acuadron-agentic-debt-final-v8-kn8t8`
**W&B project:** `agentic-debt`
**Base:** v7 configuration; original GLM base weights, not a v7 checkpoint.
**Template:** `acuadron-miles-deployer-v12`
**Dataset:** 110 selected chains; manifest commit `f75dbe74c8e33cf6ce5ad9a46b15f66b3a7339f8c3f01e5997cc972859161684`.
**Task pin:** `fec8e468c659469489431100e54d7088c24fb72d1cb26bf313e3b8c0455d329d`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-r22-20261007a`
**Trainer digest:** `sha256:d3552c757fba60d160a1926402f780b3e9e820802280a901997aff019862e890`
**Gym image:** `arena-slime-dev:gym-glm53-adr72-20260927a`
**Checkpoints:** `/mnt/scratch-s3files-rw/acuadron/checkpoints/slime_experiments/acuadron-agentic-debt-final-v8`
**W&B run:** `7rri6l81` — https://mega.wandb.agi.amazon.dev/arena/agentic-debt/runs/7rri6l81
**Outcome:** All 64 GPU pods run; 110 prompts loaded; base initialization confirmed. Gym startup remains in progress.

## Setup

The user corrected the resume plan: use the base model. The checkpoint-wait process was terminated before any resume submission.
The v8 checkpoint prefix was empty before launch. No v7 weights or optimizer state are loaded.
Keep 16 trainer nodes, 48 engine nodes, GBS 512, 64 groups, eight attempts, LR 1e-5, and save interval 10.
Use r22 kernel-cache seeds, dummy engine load, and the dedicated NEXTN draft export with load format auto.
Retain DSA-QP, R3 prefetch, EP8, the 12/11/11/11 pipeline split, and validated engine flags.
Keep multiplier 4 and 288 gyms. The function path is `generate_rollout`; the staleness cap is off.
The proposed 121-chain expansion remains document-only.

## Gym resources and cleanup

Template v12 preserves the v4 early-gym DAG and adds the approved Docker resource patch.
Docker memory: 128Gi request / 192Gi limit. Docker storage: 300Gi request / 350Gi limit; Docker emptyDir: 300Gi.
The cleanup sidecar checks usage every 60 seconds and removes unused task images without force at 250Gi.
It preserves containers, volumes, networks, infrastructure images, and all images referenced by containers.
An isolated canary exercised cleanup at a low threshold and confirmed those boundaries.
The full fleet resource shape and actual 250Gi pressure remain unverified.

## Timeline

- 02:25 UTC: cancelled the checkpoint-gated v7 resume process; no submission receipt existed.
- 02:25 UTC: `argo stop acuadron-agentic-debt-final-v7-x44fz` accepted.
- Created `acuadron-agentic-debt-final-v8-kn8t8` with `kubectl create -f workflow.yaml` after server dry-run.
- Read-only monitor: `/tmp/glm53f/watch-v8.sh`, PID 3762655; output `/tmp/glm53f/status-v8.log`.
- Monitor captures status every five minutes, at most 24 checks. It does not send automatic alerts.
- By 02:31 UTC, all 64 GPU pods ran on the expected r22 digest. Live args confirm `start_rollout_id=0` and load from the original base checkpoint.
- At 02:31:36 UTC, the data source loaded 110 prompts. Engine config confirms dummy main weights and auto draft load.
- Gym pod specs confirm 128/192Gi Docker memory, 300Gi storage request, and the cleanup sidecar. Its log reports threshold 262144000 KiB and interval 60 seconds.
- The old v7 deployments and PyTorchJob are gone. No optimizer update or full-fleet gym readiness is verified yet.

## Validation targets

Confirm base checkpoint load, start rollout 0, 110 prompts, all 64 GPU pods, and 288 three-container gym pods.
Confirm r22 digest, dummy load, separate draft load, and allreduce fusion backend auto.
Check R3 replay, log-prob gap, speculation acceptance length, and warm throughput after step 2.
Confirm cleanup logs and absence of OOM or disk evictions under load.
