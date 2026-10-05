# Run record: acuadron-agentic-debt-final-v4 — final-v3 recipe, fresh, on agentic-final-v2 (113 chains, fixed−N·broken reward), lr 1e-5

**Status:** Running
<!-- gen-workflow:begin -->
**Date:** 2026-10-05
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `acuadron-agentic-debt-final-v4-`
**Experiment name:** `acuadron-agentic-debt-final-v4`
**W&B project:** `agentic-debt` (group `acuadron-agentic-debt-final-v4`)
**Dataset:** `lakefs://arena-inspect/c519f32c424d17823cf3f375fd873590942dc2a5ae78f1033a8cfe9ac412e21f/internal/agentic-debt-r3/agentic-final-v2/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `c519f32c424d17823cf3f375fd873590942dc2a5ae78f1033a8cfe9ac412e21f`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r19-20261003a`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `acuadron-agentic-debt-final-v3`
<!-- gen-workflow:end -->
**Argo workflow:** `acuadron-agentic-debt-final-v4-x29f2`
**W&B run:** (set at launch)
**Task pin:** `c2ce399f5749` (every manifest row)
**Trainer config deltas vs base:** `prompt-data-list` agentic-debt-final@0c328c7a -> agentic-final-v2@c519f32c424d; `lr` 1.5e-6 -> 1e-5; `experiment_name`/`project_name`/`arena_sample_summary_dir` -> v4. `replicas` 48 -> 40 (16 actor + 24 engines; kueue headroom next to the running final-v3), `arena_inflight_multiplier` 4 -> 3. Workflow: `experiment-name`, `replicas` 40 (excluded-nodes stays at the 101 of v3 resume 2).
**Checkpoints:** `/mnt/scratch-s3files-rw/acuadron/checkpoints/slime_experiments/acuadron-agentic-debt-final-v4` (fresh; starts from `ref_load`)
**Outcome:** stopped 2026-10-05 23:38 UTC after ~15.5 h with 0 optimizer steps and 0 checkpoints; every trajectory dropped (`no_reward`). Dataset bug, not trainer: see Outcome below.

## Goal

Does the policy move at lr 1e-5 (train/ppo_kl 1e-4..1e-3, bf16 weights changing within ~10 steps) and does the
fixed−N·broken reward trend up on agentic-final-v2 over 100+ steps without resumes?

## Setup

final-v3 (Arpit r19 image, EP8, DSA-QP, GBS 512 / 64 groups x 8, 16 actor + 32 engine nodes, length reward off) with
the dataset and lr changed. agentic-final-v2 differs from agentic-debt-final: baseline is the agent's own tree, untouched
scores 0 (not 1−N), regressions cost N x fraction broken, debt carries forward, tasks declare `verifier.timeout_sec` 4800 (section-aware read; `[agent]` carries no `timeout_sec`).

## Outcome

Stopped with `argo stop` at 23:38 UTC on 2026-10-05 (`x29f2`, W&B `cnv27v3t`). 0/64 groups collected for the whole run;
1,092 trajectories, all dropped `reason=no_reward error='no verifier reward'`. No `iter_*` written.

Root cause (payload, reproduced with `harbor run --agent nop` on the gym image, Harbor 0.22.0):

- `steps/segment-NN/workdir/setup.sh` line 91 runs `rm -rf ... /logs/artifacts/handoff ...` as the agent user before
  every agent phase. It deletes the live-baseline snapshot that the `[environment.healthcheck]` (segment 1) or the previous
  `[[verifier.collect]]` hook wrote. The collect hook then finds no `handoff/` to move to `prestep/`, `tests/test.sh`
  reports `grading_setup_failed:prestep_missing`, `normalize_reward.py` raises, no `reward.txt` is written, Harbor raises
  `RewardFileNotFoundError`.
- The delete succeeds because Harbor's `Trial._chmod_artifact_mount_chain` makes `/logs/artifacts` 0777 in the mounted
  docker environment; the payload comment "stays root 755" is false here. `/logs/verifier` is 0777 and shared with the
  separate verifier container too (`test.sh` deletes `reward.*` first, so no accepted hack, only dropped samples).
- Both `agentic-final-v2` and `agentic-debt-final-live` carry the same line. Post-agent grading itself works
  (example: `reward 0.25`, F2P/P2P 1.0).

Fix is payload-side (root-only baseline vault in `main`, hook wipes `/logs/artifacts` before publication, drop the
`setup.sh` cleanup entry). No Miles, gym, or Harbor change. Relaunch from base against the republished pin.
