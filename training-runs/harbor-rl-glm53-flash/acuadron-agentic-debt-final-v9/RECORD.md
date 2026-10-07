# Run record: acuadron-agentic-debt-final-v9

**Status:** Running
**Date:** 2026-10-07
**Family:** `harbor-rl-glm53-flash`
**Experiment name:** `acuadron-agentic-debt-final-v9`
**Argo workflow:** `acuadron-agentic-debt-final-v9-dx8b7` (created 15:28 UTC; admitted and all pods ready by 15:35)
**W&B run:** `i8ufpnjc` — https://mega.wandb.agi.amazon.dev/arena/agentic-debt/runs/i8ufpnjc
**Base:** final-v8 configuration from the original GLM base weights; no checkpoint load.
**Template:** `acuadron-miles-deployer-v13`
**Dataset:** 110 selected chains; manifest commit `f75dbe74c8e33cf6ce5ad9a46b15f66b3a7339f8c3f01e5997cc972859161684`.
**Trainer image:** `arena-github/miles:miles-glm53-r22-20261007a`
**Checkpoints:** `/mnt/scratch-s3files-rw/acuadron/checkpoints/slime_experiments/acuadron-agentic-debt-final-v9`
**Config delta vs final-v8:** `lr` 1e-5 -> 1.5e-6. Nothing else.

## Why final-v8 was stopped

final-v8 (`9q9ww`) was not frozen. It was starved by a policy length explosion under lr 1e-5.
Evidence from the trainer log and 72 sampled gym workers:

| Rollout | Mean tokens per episode | Retained mean reward |
| ---: | ---: | ---: |
| 0 | 66k | 0.761 |
| 3 | 167k | 0.648 |
| 5 | 509k | 0.656 |
| 6 | 938k | 0.571 |

Per hour across the sampled gyms, completed checkpoints fell 3,733 (08h) -> 335 (12h) while
`turn truncated at max_tokens` warnings rose 160 -> 2,154 (14h). The agent generated 16k-token turns
without a tool call, was nudged, compacted its context, and repeated. Engines sat at 86-97% KV usage with
about 30 requests each, so each turn took minutes. Step 5 took 12,347 s (10,206 s waiting for rollouts).
No checkpoint existed (first save after step 9), so a rollback was impossible; the fallback the user set for the
failure case was a base restart at the reduced learning rate.

Secondary findings on v8, not fixed in place because the run was replaced:
- Two gym workers had a cached, timed-out Harbor Docker kernel probe and rejected every no-network task
  (27 groups, 216 attempts, zero trajectories). Fresh pods re-probe; v9 shows 0 poisoned workers at 15:55 UTC.
- Docker sidecar OOM kills continued at the 192Gi limit (18 in 7 h; v7 had 30 in 6 h at 64Gi).
- No disk evictions; the 300Gi volume and cleanup sidecar held.

## v9 startup checks (15:55 UTC)

- 64/64 trainer pods and 288/288 gyms Running; zero container restarts.
- Live args: `lr 1.5e-06`, load from the base DCP, 110 prompts loaded, `HARBOR_AGENT_KWARGS={}`, ARENA_JOB correct.
- Engines loaded the NEXTN draft in 11-21 s. NATS connected; 218 rollout jobs dispatched; 0/64 groups at 270 s.
- Read-only monitor: `/tmp/glm53f/watch-v9.sh` (PID in `/tmp/glm53f/watch-v9.pid`), log `/tmp/glm53f/status-v9.log`.

## Watch next

Tokens per episode and `turn truncated` rate over rollouts 0-6; if they climb as in v8, the learning rate is not the
only cause and the 16,384 per-turn cap or the nudge loop needs attention. First checkpoint after step 9.

## Restart path (2026-10-08)

`python3 examples/arena/harbor-rl-glm53-flash/relaunch.py acuadron-agentic-debt-final-v9-dx8b7 acuadron-agentic-debt-final-v9 > workflow.yaml`
writes the resume manifest. The only parameter delta against the live v9 is `trainer-image` ->
`arena-github/miles:miles-glm53-r24-20261008a` (290940d7: `rollout/population/*` W&B keys plus parallel
routing loads; digest `sha256:72dca7c1…`). Server dry run accepted on 2026-10-08. The same experiment name
resumes from the latest checkpoint (first save after step 9); a new name starts from the base weights.
After the restart, W&B shows `rollout/population/mean_reward`, `rollout/population/chains_at_one_frac` and
`rollout/population/regression_frac` over every collected group, next to the kept-group `rollout/raw_reward`.
