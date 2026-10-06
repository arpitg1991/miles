# Run record: guparpit-agentic-debt-v20 (qp-fi) — r56-ns with the query-parallel DSA core and the flashinfer top-k (ADR-0019)

**Status:** Prepared
<!-- gen-workflow:begin -->
**Date:** 2026-10-06
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v20-`
**Experiment name:** `guparpit-agentic-debt-v20`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v20`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-wvspans-20261001a@sha256:e29ba91a3392fd250e78f551fca6f33061520a2b456134693d419e1bf5c9eab9`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-qp-20261006a@sha256:5e540e4b3f04e93480d992154da0686783439114780b9a41ca4affb194a3c667`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v18`
<!-- gen-workflow:end -->
**Argo workflow:** not submitted
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Trainer config deltas vs base:** `glm5_next_dsa_qp: true` (new key, `--glm5-next-dsa-qp`); `miles_dsa_topk_backend` torch -> flashinfer (the kpool indexer selects its pools with `flashinfer.top_k`); `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run. Everything else is the r56-ns config: prefetch on, inflight multiplier 8, `max_weight_staleness` 8, `arena_output_queue_groups` 64, FlashMLA forward, `NatsRolloutFn`, TP8 SP, PP4 11/11/11/12, EP8.
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v20`, seeded from r54-ns (`guparpit-agentic-debt-v17`) `iter_0000109` (its latest save at copy time, 2026-10-06 04:44Z; 135 objects match by name and size; sidecar `{"rollout_id": 109}` without `wandb_run_id`; data state `arena_data_source_state_109.pt`; tracker 109 written last)
**Outcome:** Prepared

The owner's order of 2026-10-06: port `--glm5-next-dsa-qp` onto the ns stack, gate it with the
1-node parity harness and T1, and run two arms at once from the same seed: `--glm5-next-dsa-qp`
(v19) and `--glm5-next-dsa-qp` plus `--miles-dsa-topk-backend flashinfer` (v20). r54-ns (v17) and
r56-ns (v18) keep running as the controls at matched rollout ids.

## Goal

Measure the query-parallel DSA core (ADR-0019, ported from `acuadron/dsa-qp` a56d9a9c36) on the
live ns stack. The head split runs the sparse-attention kernel for all queries on each rank's 8
heads; the per-query KV gather and the backward dKV atomics repeat 8 times. The query split runs
each rank's sequence chunk on all 64 heads. Expected from the source branch: 1.5x live trainer
tokens per second (their final-v5: 837 to 1,057 per GPU at steps 2 to 4 against our 672 to 680),
1.94x trainer compute on the shared T2 rows against the FlashMLA EP8 arm. v20 adds the flashinfer top-k in the kpool indexer, which the GLM-5.3 path ignored before ADR-0019; its indices are not bitwise against torch on ties, so the parity run records the index difference, and the comparison of v20 against v19 measures the indexer time and any drift.

## Trainer stack

Branch `arpit-qp-20261006` (worktree `/workplace/guparpit/kdfast/scratch/qp/miles`) = `arpit-ns-20261005`
tip `59a3016356` + `6edfa76be4` (top-k backend through the kpool indexer) + `d67055a3ad` (query-parallel
core, flag, spec hook, CPU tests) + `61f60b0aed` (parity harness, qp-t2 arms, ADR-0019). Image
`miles-glm53-qp-20261006a`, digest `sha256:5e540e4b3f04e93480d992154da0686783439114780b9a41ca4affb194a3c667`,
built from `61f60b0aed` (clean tree) on `radixark/miles:miles-base-d7f1a42-20261001a`, pushed to
`arena-github/miles` us-east-1 at 04:43Z, replica in ap-south-1 at 04:43:36Z.

## Upstream check

- Upstream `main` has no query-parallel DSA (checked 2026-10-06). Its latest change to the GLM-5.3 DSA
  module is #3866 (2026-10-01, native Megatron DSA in raw model mode); #2786 added GLM 5.3 Flash. The
  source branch sits on the fork lineage from 2026-08-31, so the change is a port against our plugin.
- `--miles-dsa-topk-backend` is an upstream miles flag (`torch`, `flashinfer`); upstream applies it to the
  GLM-5 lightning indexer only. The kpool indexer of GLM-5.3 took it in `6edfa76be4`.
- The rest of the recipe is the r56-ns stack (v18 record).

## Measurement plan

Controls: r54-ns (`guparpit-agentic-debt-v17`) and r56-ns (`guparpit-agentic-debt-v18`) at the same rollout ids
(both run the same seed lineage; v20 starts at 109). Only trainer-bound steps count (`perf/train_wait_time` under
300 s), as before.

| Question | Signal | Gate |
| --- | --- | --- |
| Trainer speed | `perf/actor_train_time` and `perf/actor_train_tok_per_s` at matched rollouts vs r56-ns | at least 1.4x tokens per second (r56-ns: 43.7K on 64 GPUs); the T2 arms give the matched number |
| Numerics | `train_rollout_logprob_abs_diff` after each weight sync | about 0.03, as r54 to r56; a jump above 0.05 stops the run |
| Reward | batch and W&B true reward per 5-rollout bin vs the controls | within noise (0.60 to 0.67) |
| Loop balance | `perf/train_wait_time`, queue size | recorded; expected to grow, the trainer gets faster and multiplier 8 with 32 engines is the next limit |
| Staleness | `rollout/num_old_age_dropped`, `rollout/reward_old_age_dropped`, mean age | recorded as in v17 and v18 |
| Flags live | the argument dump | `glm5_next_dsa_qp True`, `miles_dsa_topk_backend flashinfer`, and the one-time log line "glm5_next DSA query-parallel core: layer ..., 64 heads x N local queries, forward backend flash_mla" |
| Indexer | v20 against v19: step time, `train_rollout_logprob_abs_diff`, reward | the flashinfer top-k changes indices on ties only; a reward or drift gap larger than the noise is a stop |
Gates before the submit: the 1-node parity harness (`kdatp/dsaqp`, jobs `kdatp-dsaqp-20261006qt` torch and
`kdatp-dsaqp-20261006qf` flashinfer) and T1 on the image (`recon-t1-new-20261006q`, flag on). The T2 arms
(`kdatp-prof-20261006q`: control, qp, qp-fi, qp-split, split) run in parallel as the matched measurement.

## Launch

| Time (UTC) | Event |
| --- | --- |
| 2026-10-06 04:43 | Image `miles-glm53-qp-20261006a` built from `61f60b0aed` and pushed (digest above). |
| 04:44 | Seed copied from r54-ns `iter_0000109` (`metrics-1001/ckcopy-v20.sh`): 135 objects match; sidecar `{"rollout_id": 109}`; data state 109; tracker 109 last. |
| 04:46 | Gate jobs created: `kdatp-dsaqp-20261006qt`, `kdatp-dsaqp-20261006qf` (1 node each), `kdatp-prof-20261006q` (8 nodes), `recon-t1-new-20261006q` (1 node); all admitted within a minute. |
