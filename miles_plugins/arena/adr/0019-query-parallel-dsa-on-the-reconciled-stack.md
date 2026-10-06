# ADR-0019: Query-parallel DSA sparse attention on the reconciled stack

**Status:** Accepted
**Date:** 2026-10-06

**Amends:** nothing. The checkpoint contract (ADR-0004), the KDA sharding
(ADR-0016) and the R3 routing replay (ADR-0012) stay as they are.
**Source:** Alex Cuadron's `acuadron/dsa-qp` (commit `a56d9a9c36`, 2026-10-02,
and its ADR-0018 in that branch's numbering), re-implemented against the
upstream GLM-5.3 plugin that ADR-0018 (ours) took from upstream `main`.

## Summary

With `glm5_next_dsa_qp: true` (`--glm5-next-dsa-qp`, default off) each
tensor-parallel rank runs the DSA indexer and the sparse-attention kernel for
its sequence-parallel chunk of `S / 8` queries on all 64 heads, instead of all
`S` queries on its 8 heads. Two all-to-alls over the tensor-parallel group move
the absorbed queries into that layout and the attention output back.
Parameters, sharding, checkpoint keys and the weight sync do not change. The
core drops the 64-wide zero tail. The flag requires sequence parallelism and
rejects the indexer replay.

The same change lets `--miles-dsa-topk-backend` reach the kpool indexer: the
pool top-k (`_pool_topk`) takes the backend's top-k function, so `flashinfer`
selects the pools with `flashinfer.top_k` in both layouts. Before this change
the flag applied only to the GLM-5 lightning indexer, and the GLM-5.3 path
ignored it.

## Context

- The DSA core was 57% of the train step on the head split (trainer-core
  profile, 2026-09-29): the per-query KV gather and the backward `dKV` atomics
  repeat on every rank and do not shrink with the head count.
- The source branch measured, on the shared T2 rows (8 nodes, DP 2): head
  split with FlashMLA and EP8 470.7 s `actor_train` (our job `20260930f`);
  query-parallel with EP8 259.5 s; with the 12/11/11/11 split 242.0 s; the
  same-job control 581.5 s (their job `20261003q`). Live: their `final-v3`
  reached 866 and 1,021 trainer tokens per second per GPU at steps 2 and 3,
  and `final-v5` (query-parallel plus our R3 data path) 837 to 1,057 at steps
  2 to 4, against 672 to 680 for `r54-ns` and `r56-ns`.
- Upstream check (2026-10-06): upstream `main` has no query-parallel DSA. Its
  latest change to the GLM-5.3 DSA module is #3866 (2026-10-01, native
  Megatron DSA in raw model mode); #2786 added GLM 5.3 Flash. The source
  branch sits on the fork lineage from 2026-08-31, so this is a port, not a
  cherry-pick.

## Decision

1. `miles_plugins/models/glm5_next/dsa.py`: `query_parallel` constructor
   argument; `_core_attention_head_parallel` (unchanged math) and
   `_core_attention_query_parallel`; the indexer runs on the local query chunk
   with global token ids. The layout reshapes live in
   `miles_plugins/models/glm5_next/ops/qp_layout.py` (torch only) so the CPU
   tests pin them.
2. `kpool_select_topk` takes `token_ids` and `topk_backend`;
   `miles/kernels/attention/dsa/kpool.py` threads a `topk_fn` through
   `_pool_topk`, `select_expand_tail` and `pool_topk_to_token_fn`;
   `topk_with_scores` in `topk.py` adapts a backend's indices to the
   `(scores, indices)` the expand kernel reads. The torch path is unchanged.
3. The spec hook passes `query_parallel`; the flag is `--glm5-next-dsa-qp`.
4. Gates before a live run: the 1-node parity harness
   (`examples/arena/harbor-rl-glm53-flash/kdatp/dsaqp`, with torch and with
   flashinfer top-k), T1 on the image, and the T2 arms of `kdatp/qp-t2`.

## Evidence

Filled in by the run records of `guparpit-agentic-debt-v19` (query-parallel)
and `-v20` (query-parallel plus flashinfer top-k) and the qp-t2 job.

## Consequences

- The T2 rows of 2026-09-29 (`kdatp/data/t2`) carry the routing payload of the
  old data path; the reconciled image's R3 fill reads 252 bytes of it and the
  train step never completes. T2 timing on this stack runs with
  `use_rollout_routing_replay: false` until the rows are rebuilt with the
  image's `build_rollout_data.py`.

- The step cost moves from the DSA kernels to the MoE path and the dense work.
  The trainer gets faster, so at multiplier 8 the engines become the limit
  again; the next recipe raises the engine count or the multiplier.
- The flashinfer top-k is not bitwise against torch on ties. The parity run
  records the index difference; a run that needs bitwise indices keeps `torch`.
