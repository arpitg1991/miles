# Run record: acuadron-agentic-debt-final-v7

**Status:** Running
**Date:** 2026-10-06
**Family:** `harbor-rl-glm53-flash`
**Experiment name:** `acuadron-agentic-debt-final-v7`
**W&B project:** `agentic-debt`
**Argo generateName:** `acuadron-agentic-debt-final-v7-`
**Base:** `acuadron-agentic-debt-final-v6`
**Template:** `guparpit-miles-deployer-v10`
**Dataset:** 110 chains; `lakefs://arena-inspect/f75dbe74c8e33cf6ce5ad9a46b15f66b3a7339f8c3f01e5997cc972859161684/internal/agentic-debt-r3/acuadron-agentic-debt-final-110-sd015-20261006/manifest.jsonl`
**Manifest commit:** `f75dbe74c8e33cf6ce5ad9a46b15f66b3a7339f8c3f01e5997cc972859161684`
**Task pin:** `fec8e468c659469489431100e54d7088c24fb72d1cb26bf313e3b8c0455d329d`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-r20-20261006a`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Checkpoints:** `/mnt/scratch-s3files-rw/acuadron/checkpoints/slime_experiments/acuadron-agentic-debt-final-v7`
**Argo workflow:** `acuadron-agentic-debt-final-v7-x44fz`
**W&B run:** `7pp226vc` — https://mega.wandb.agi.amazon.dev/arena/agentic-debt/runs/7pp226vc
**Outcome:** All 64 trainer pods run. Dataset load confirms 110 prompts; model initialization remains in progress.

## Setup

The user authorized an immediate stop of `t66h9` and a fresh base start at iteration 0. No v6 checkpoint is used.
The subset preserves existing task payloads and pins. Its selection estimate is 0.515; this is not a forecast.
See `training-runs/studies/agentic-debt-final-110-sd015/STUDY.md` for the exact chain list and selection method.

Use 16 trainer nodes, 48 engine nodes, 288 gyms, multiplier 4, GBS 512, 64 groups, eight attempts, and LR 1e-5.
Keep DSA-QP, R3 prefetch, EP8, and pipeline layout 12/11/11/11.
Add trtllm DSA, flashinfer_cutlass MoE, allreduce fusion, and NEXTN with 3 steps, top-k 1, and 4 draft tokens.
Keep the r20 function entry point. Do not configure a staleness cap.

## Validation and risks

Manifest readback equals the uploaded bytes; all 110 IDs match the approved selection document.
No task payload changed. Engine flags have prior live evidence in `acuadron-agentic-debt-engine-ab-v1`.
The 64-node shape has no prior live timing evidence. Judge speed after step 2.
Template v10 retains the known 64Gi dind limit; monitor restarts and verifier errors.
A log-prob gap above 0.05 requires review of rollout staleness. Do not infer performance from the workflow phase alone.

## Timeline

- Prepared the immutable subset manifest and the r20 configuration. No source-code overlay or image build.
- `argo stop acuadron-agentic-debt-final-v6-t66h9` succeeded. The exit handler removed its deployments and PyTorchJob.
- `kubectl --context arena-prod-bom-v2 -n arena-tasks create -f workflow.yaml` created `acuadron-agentic-debt-final-v7-x44fz`.
- Read-only shell capture: `/tmp/glm53f/watch-v7.sh`; output `/tmp/glm53f/status-v7.log`; PID file `/tmp/glm53f/watch-v7.pid`.
- The shell captures status every five minutes for at most 24 checks. It does not diagnose or notify autonomously.
- At 20:02 UTC, all 64 trainer pods ran on distinct nodes, with no excluded-node placements. Image digest: `sha256:437bf865682262d2708c3204e00614b69f24b9202ce30554d25e271b76f9a59c`.
- Live arguments confirm `start_rollout_id=0`, load from the original reference checkpoint, LR 1e-5, 16 actors, and 384 rollout GPUs.
- At 20:03 UTC, the data source loaded exactly 110 prompts from manifest `f75dbe74c8e33cf6…`.
- Engine arguments confirm trtllm prefill/decode, flashinfer_cutlass MoE, speculative algorithm EAGLE, four draft tokens, and allreduce fusion backend auto.
- First optimizer update, R3 replay checks, and warm step timing were pending at launch.

## Resume 1 preparation — 2026-10-07

The user approved r22, the early-gym startup flow, and the Docker memory/storage repair.
The selected path is a resume without the staleness cap. Keep the 110-chain manifest, LR, shape, exclusions, and save interval.
Adam state is not saved. The optimizer state resets on resume.

- Trainer image: `arena-github/miles:miles-glm53-r22-20261007a`, BOM digest `sha256:d3552c757fba60d160a1926402f780b3e9e820802280a901997aff019862e890`.
- Owned template: `acuadron-miles-deployer-v12`, derived from live `acuadron-miles-deployer-v4`. Only the gym resource manifest differs.
- Docker resources: 128Gi memory request, 192Gi memory limit, 300Gi storage request, 350Gi storage limit, 300Gi Docker emptyDir.
- Cleanup sidecar: 250Gi threshold; scan every 60 seconds; non-forced removal of unused task images only.
- Engine initialization: `sglang_load_format: dummy`; dedicated NEXTN draft path; draft load format `auto`.
- Keep `generate_rollout`, multiplier 4, 288 gyms, 16 trainer nodes, and 48 engine nodes. No staleness cap.
- Prepared files: `workflow-resume1.yaml` and `miles-config-resume1.yaml`.
- Template and fully resolved gym Deployment passed Kubernetes server dry-run. The owned template is installed.
- End-to-end cleanup canary at a forced low threshold deleted an unused task image and preserved active and infrastructure images.
- The canary was deleted after validation. Full-fleet resource admission and a real 250Gi pressure test remain unverified.

At 02:12 UTC, no checkpoint marker existed. Step 7 had finished; rollout 8 was only partially collected.
A checkpoint-gated operation waits for a marker of at least 9 and the corresponding `slime_extra_state.json` before it stops the old run.
It waits for old workload cleanup, then creates exactly one resume and starts a read-only monitor.
The operation checks the prepared workflow hash and rejects another active workflow under the same experiment identity.
It has a six-hour wait limit and halts on uncertainty. It does not send automatic user notifications.

- Operation: `/tmp/glm53f/relaunch/checkpoint_resume_v7.py`, PID 3748268.
- Status: `/tmp/glm53f/relaunch/checkpoint-resume-status.json`.
- Log: `/tmp/glm53f/relaunch/checkpoint-resume-v7.log`.
- Submission receipt after launch: `/tmp/glm53f/relaunch/resume-v7-submitted.json`.
- State at preparation: `awaiting_checkpoint`; no resumed workflow submitted.
- Superseded at 02:25 UTC: the user requested base weights instead. The operation was terminated before it submitted a resume.
- Stop requested for `x44fz`. Fresh v8 launched as `acuadron-agentic-debt-final-v8-kn8t8`; see its record.
