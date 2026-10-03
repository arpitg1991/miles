# Run record: acuadron-agentic-debt-final-v3 — final-v2 on the query-parallel DSA trainer (r19), EP8, PP 12/11/11/11

**Status:** Running
<!-- gen-workflow:begin -->
**Date:** 2026-10-03
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `acuadron-agentic-debt-final-v3-`
**Experiment name:** `acuadron-agentic-debt-final-v3`
**W&B project:** `agentic-debt` (group `acuadron-agentic-debt-final-v3`)
**Dataset:** `lakefs://arena-inspect/0c328c7a4bef7311c2713a246e5623ebc19f4891e7ff5949931fb7e8a5c16b96/internal/agentic-debt-r3/agentic-debt-final/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `0c328c7a4bef7311c2713a246e5623ebc19f4891e7ff5949931fb7e8a5c16b96`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r19-20261003a`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `acuadron-agentic-debt-final-v2`
<!-- gen-workflow:end -->
**Argo workflow:** `acuadron-agentic-debt-final-v3-gblzd`
**W&B run:** (set at launch)
**Task pin:** `7af3bbbece69` (every manifest row)
**Image digests:** gym `sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d`, trainer `sha256:b814139f278672d40a6e645b2a98b1d241f48020c2a8a186379504ac5f8f1762`
**Trainer config deltas vs base:** trainer image r17 (`4716a367a`) -> r19 (`a56d9a9c`: arpit-glm-53 + the PR #3608 DSA kernel backport + ADR-0018); `glm5_next_dsa_qp` unset -> `true`; `expert_model_parallel_size` 16 -> 8; `decoder_first_pipeline_num_layers` 11 -> 12; `decoder_last_pipeline_num_layers` 12 -> 11; `experiment_name`, `project_name`, `arena_sample_summary_dir` renamed. Workflow: `experiment-name`, `trainer-image`, `miles-config` only.
**Checkpoints:** `/mnt/scratch-s3files-rw/acuadron/checkpoints/slime_experiments/acuadron-agentic-debt-final-v3` (fresh; starts from `ref_load`)
**Outcome:** (pending)

## Goal

The same RL question as final-v2 (pure chain reward, 64 chains per step, DP 4) on a trainer that spends its time
on the model instead of on the DSA gather: the trace of the r47 layout put the DSA sparse attention at 57% of the
step. This run measures the live step time and the loop balance (`train_wait`) of the r19 trainer.

## Setup

RL recipe identical to final-v2. Trainer changes only (ADR-0018 and the arms of `kdatp/qp-t2`):

| Item | Base final-v2 | This run |
| --- | --- | --- |
| Trainer image | `miles-glm53-r17-20260928a` (miles `4716a367a`) | `miles-glm53-r19-20261003a` (miles `a56d9a9c`) |
| DSA kernels | first-port TileLang fwd and bwd, static shapes, 64-wide zero tail | PR #3608 backport: FlashMLA sparse fwd, dynamic-shape TileLang bwd |
| DSA layout | head split: all queries on 8 heads per TP rank | `glm5_next_dsa_qp: true`: each rank its sequence chunk on 64 heads (ADR-0018) |
| `expert_model_parallel_size` | 16 (two nodes per expert group, EFA all-to-all) | 8 (one node per expert group; expert DP 4) |
| Pipeline split | 11/11/11/12 | 12/11/11/11 |
| Everything else | - | unchanged (GBS 512, rbs 64, n 8, 16 actor + 32 engine nodes, lr 1.5e-6, TIS, R3, length reward 0, ack-wait 72,000) |

Evidence before the launch: 1-node parity and timing `kdatp/dsaqp/20261003a` (PASS; one DSA layer 5.1x faster at
64K tokens, bitwise outputs); 8-node train-only arms `kdatp/prof/20261003q` (`qp`, `qp-ep8`, `qp-ep8-pp12`, `base`
on the T2 rows; see the table in `examples/arena/harbor-rl-glm53-flash/kdatp/qp-t2/README.md`).

## Timeline

| UTC | Event |
| --- | --- |
| 2026-10-03 01:38 | Launched as `acuadron-agentic-debt-final-v3-gblzd` after final-v2 was stopped at 01:30 (its step 0: actor_train 4,229 s cold at DP 4, 2,864 rows). |
