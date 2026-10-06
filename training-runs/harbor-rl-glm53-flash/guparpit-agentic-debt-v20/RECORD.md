# Run record: guparpit-agentic-debt-v20 (qp-fi) — r56-ns with the query-parallel DSA core and the flashinfer top-k (ADR-0019)

**Status:** Running
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
**Argo workflow:** `guparpit-agentic-debt-v20-xxmm9`
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Trainer config deltas vs base:** `glm5_next_dsa_qp: true` (new key, `--glm5-next-dsa-qp`); `miles_dsa_topk_backend` torch -> flashinfer (the kpool indexer selects its pools with `flashinfer.top_k`); `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run. Everything else is the r56-ns config: prefetch on, inflight multiplier 8, `max_weight_staleness` 8, `arena_output_queue_groups` 64, FlashMLA forward, `NatsRolloutFn`, TP8 SP, PP4 11/11/11/12, EP8.
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v20`, seeded from r54-ns (`guparpit-agentic-debt-v17`) `iter_0000109` (its latest save at copy time, 2026-10-06 04:44Z; 135 objects match by name and size; sidecar `{"rollout_id": 109}` without `wandb_run_id`; data state `arena_data_source_state_109.pt`; tracker 109 written last)
**Outcome:** Running

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
| 04:50 | 1-node parity, flashinfer top-k (`kdatp-dsaqp-20261006qf`, forward `flash_mla`): PASS. All 4 packed cases: I1 top-k indices bitwise equal between the head split and the query split; F1 layer output relative L2 0 (bitwise); B1 input gradient 4.6e-3 to 5.5e-3, worst weight gradient 5.7e-3, norm ratios within 2.3e-4 (bounds 1e-2 and 1%); N1 negative control 0.760. Per-layer fwd+bwd 2.93x at 8K tokens to 5.39x at 128K; peak memory 24.6 GiB -> 7.1 GiB at 128K. `s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/dsaqp/20261006qf/parity/SUMMARY.md`. |
| 04:54 | 1-node parity, torch top-k (`kdatp-dsaqp-20261006qt`): PASS with the same numbers (F1 bitwise, B1 4.6e-3 to 5.7e-3, N1 0.760, 2.93x to 5.2x). The PR GPU kernel tests (`tests/fast-gpu/kernels/attention/dsa`) passed 60 of 60 on the image first. |
| 04:55 | Torch against flashinfer top-k in the kpool indexer (`kdatp-dsaqp-20261006qx`, 1 GPU, random packed cases up to 65,536 tokens): the index set of every query is the same in all cases but one 65,536-token query, where the two backends pick different tokens on a tie (Jaccard distance 3.9e-3 for that query, 5.9e-8 mean); the column order differs, so the tensors are not bitwise equal. Selection time 53.1 ms (torch) against 48.6 ms (flashinfer) at 65,536 tokens, about 9% less. |
| 05:07 | T1 on the image with the flag on (`recon-t1-new-20261006q`, 1 node, forward `flash_mla`, r47 `iter_0000059` through `guparpit-agentic-debt-v2`; harness `recon2/t1-qp/`): load 912 s, `dsa_forward_backend_modules {'flash_mla': 11}`, `dsa_query_parallel_modules {True: 11}`, rc 0. Against the r17 reference run `20261001a-ref`: 37,534 of 37,534 gathered tensors bitwise equal; log-prob mean 0.0527 (limit 0.1013), p99 0.586 (1.140), logits relative L2 0.229 (0.453): PASS (`recon2/t1-qp/out/compare-20261006q.json`). Job and ConfigMap deleted after the run. All three gates passed; the submit follows. |
| 05:11:47 | Submit `guparpit-agentic-debt-v20-xxmm9` (the two arms 2 s apart). Node exclusions refreshed at submit: 304 B200 nodes, 1 bad (tainted), 346 excluded (the v18 list of 345 plus 1). Queue `gpu.p6-b200-48xlarge` at submit: pending 0, admitted 16. Trainer image by digest, gym image `gym-glm53-wvspans-20261001a` by digest. |
| 05:14:55 | Trainer PyTorchJob `guparpit-agentic-debt-v20-xxmm9-trainer` created. Queued: Kueue fits 28 of 40 pods (304 nodes; 221 hold memory, 16 hold GPUs, 33 excluded by affinity, 6 with the `karpenter.sh/disrupted` taint). It waits for 12 free B200 nodes; r54-ns, r56-ns and v19 keep running. |
| 05:4x | Kueue reserved the quota (QuotaReserved=True) after 12 more B200 nodes came free; the 40 worker pods start. |
| 05:58 to 06:08 | T2 job `kdatp-prof-20261006q`: the `control` arm (flag off, EP8, FlashMLA, R3 on) hung in its first train step for 70 min and hit the arm timeout (rc 124) with no error and no step; the R3 fill of the replayed rows read 252 bytes per rank (the live runs read about 20 GB), so the 2026-09-29 T2 rows carry the routing payload of the old data path and the ns image's R3 fill cannot use it. Job stopped; resubmitted as `kdatp-prof-20261006r` with `use_rollout_routing_replay: false` in every arm (control, qp, qp-fi, qp-split, split). The T2 numbers of 30 Sep and 03 Oct ran with R3 on; the in-job control carries the comparison. |
| 05:43 to 06:14 | 40 worker pods Running at 05:43 (no operator deletion). Argument dump: `glm5_next_dsa_qp True`, `miles_dsa_topk_backend flashinfer`, `miles_dsa_sparse_attention_forward_backend flash_mla`, `max_weight_staleness 8`, `arena_inflight_multiplier 8`, `prefetch_rollout_data True`. `Checkpoint sidecar rollout_id=109`. `Rollout 110` collecting since 06:04 (0 of 32 at 630 s). Live log capture: `/workplace/guparpit/kdfast/scratch/qp/v20-follow.log`. |
