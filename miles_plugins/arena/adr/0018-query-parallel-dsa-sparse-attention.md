# ADR-0018: Query-parallel DSA sparse attention for GLM-5.3-Flash training

**Status:** Accepted
**Date:** 2026-10-03

**Amends:** nothing. The checkpoint contract of ADR-0004 (the `--load`/`--save`
DCP layout), the KDA sharding of ADR-0016 and the R3 routing replay of
ADR-0012 stay as they are.
**Pairs with:** the upstream radixark/miles PR stack #3605-#3608 (the DSA
kernel backport on `arpit-dsa-3608`: FlashMLA sparse forward, dynamic-shape
TileLang backward, no zero tail), which this decision ships in the same image.

## Summary

The GLM-5.3-Flash DSA layers run their sparse-attention core and their
indexer in the sequence split instead of the head split. With
`glm5_next_dsa_qp: true`, each tensor-parallel rank handles its
sequence-parallel chunk of `S / 8` queries on all 64 heads. Two all-to-alls
over the tensor-parallel group move the absorbed queries into that layout
and the attention output back. Parameters, their sharding, the checkpoint
keys and the weight sync do not change. Release image:
`miles-glm53-r19-20261003a` (miles `a56d9a9c`); first user:
`acuadron-agentic-debt-final-v3`.

## Context

- **The DSA core was 57% of the train step.** The Kineto trace of the r47
  layout (`training-runs/studies/trainer-core-profile-glm53-flash/STUDY.md`,
  round 20260929c) put the sparse-attention kernels at 376 s and the indexer
  plus top-k at 67 s of a 779 s T2 step. The GPU idled 1.2 to 1.7% inside
  the pipeline, so the step was not launch-bound; the kernels were the cost.
- **The cost did not shrink with the head count.** In the head split each
  rank runs the kernel for all `S` queries on its 8 of 64 heads. The kernel
  gathers the 2,112 selected KV rows (2.4 MB) per query and, in the
  backward, adds 4.9 MB of fp32 atomics into `dKV` per query. That work is
  per query, not per head: the kernel test of 2026-09-30 (`kdatp/dsa`,
  run 20260930a) measured 8 and 16 heads at the same time per call. So the
  8 ranks of a TP group repeated the same gather 8 times, and each rank also
  ran the indexer for all `S` tokens with an `[S, S / 4]` fp32 logits
  buffer (4.3 GiB at 64K tokens, 17 GiB at 128K).
- **The DeepSeek reference layout amortises the gather over 64 heads.** The
  TileLang backward kernel takes 64 heads per block at 256 threads; the
  FlashMLA forward runs 64 heads per query in any case (it pads 8 heads to
  64, which is why its forward used 9.6 GiB at 64K in the head split).
- **TP4 was the known lever and did not fit.** The study measured TP4 + EP8
  at -42%, with the GPUs full (36 MiB free). TP4 halves the DSA cost; the
  sequence split divides it by 8 at the same memory as TP8.

## Decision

1. `miles_plugins/models/glm5_next/dsa.py` gains the query-parallel core
   behind `--glm5-next-dsa-qp` (YAML `glm5_next_dsa_qp`, default off). The
   absorbed queries `[S, 8, 512]` go through `all_to_all` over the TP group
   into `[S / 8, 64, 512]` (rank `r` holds its sequence-parallel chunk `r`,
   heads in the column-parallel order), the kernel runs there, and the
   output goes back the same way for the head-split `w_vc` and the
   row-parallel output projection. The indexer runs on the local chunk:
   `kpool_select_topk` takes the global token ids of the queries. The core
   drops the 64-wide zero tail. The flag requires sequence parallelism and
   rejects the indexer replay (which holds the top-k of all tokens).
2. The image ships the upstream kernel backport (`arpit-dsa-3608`): the
   FlashMLA sparse forward and the TileLang backward with dynamic shapes
   (one compile per head count, not one per padded length).
3. The head-parallel path stays available with the flag off (the r18dsa
   path) until a live run has used the new layout for a full schedule.

## Evidence

| Measure | Head split (flag off) | Query-parallel | Source |
| --- | --- | --- | --- |
| Top-k indices, 4 packed cases | - | bitwise equal | `kdatp/dsaqp/20261003a` |
| Layer output | - | bitwise equal (same FlashMLA forward) | same |
| Input and weight gradients | - | within 5.7e-3 relative L2 (dKV atomic order); norm ratios within 2e-4 | same |
| Negative control (reversed head groups) | - | output moves by 0.76 relative L2 | same |
| One DSA layer fwd+bwd at 65,536 tokens, TP8 | 406 ms | 79 ms (5.1x) | same |
| One DSA layer fwd+bwd at 131,072 tokens | 936 ms | 169 ms (5.6x) | same |
| DSA layer peak memory above inputs at 131,072 tokens | 24.6 GiB | 7.0 GiB | same |
| Kernels alone at 65,536 tokens, one rank: fwd / bwd | 19.9 (FlashMLA) / 322.6 ms; r17 kernels 102.7 / 496.3 ms | 2.3 / 42.3 ms | same; `kdatp/dsa/20260930a` |
| T2 step (22.3M tokens, 8 nodes, TP8 PP4 EP16 DP2), `actor_train` warm mean | r17 776.9 s; r18dsa 588.9 s | 376.5 s (steps 41-43: 377.3 / 380.9 / 371.2; -51.5% vs r17, -36% vs r18dsa); grad_norm and the log-prob gap track the r18dsa base step by step (0.027294 vs 0.027268 at step 40) | `kdatp/prof/20261003q` |
| T2 step with EP8 | r17 672.6 s; r18dsa 470.7 s | 259.5 s (steps 41-42: 259.7 / 259.2; -66.6% vs r17, -45% vs r18dsa ep8); torch max allocated 85.6 / 97.6 / 92.1 / 96.8 GiB by stage (r18dsa ep8: 95.7 / 106.9 / 100.9 / 102.4) | same |
| T2 step with EP8 and the 12/11/11/11 split | - | 242.0 s (steps 41-42: 242.6 / 241.3; -68.8% vs r17, 3.2x); same-job control (flag off, the r18dsa path) 581.5 s; torch max allocated 93.6 / 97.4 / 91.9 / 90.9 GiB by stage | same |

## Consequences

- The step cost moves from the DSA kernels to the MoE path and the dense
  work. The next levers are the ones the trace ranks after DSA: the EP
  all-to-all (EP8 in node removes the EFA leg), the pipeline split, and
  the recompute pass (20% of the step); the GEMMs themselves are about 9%.
- The `[S, S / 4]` fp32 indexer logits shrink by 8 on every rank, and the
  FlashMLA forward no longer pads heads, so the DSA layers free 17 GiB of
  transient memory at 128K tokens. That margin pays for EP8 (+30 GiB of
  expert weights and gradients per rank) on the same layout.
- The all-to-alls add about 6 ms per DSA layer per 64K micro-batch over
  NVLink; the DSA layers no longer dominate any pipeline stage, so the
  12/11/11/11 split moves work from the LM-head stage to the first stage.
- A live run keeps `train_rollout_logprob_abs_diff` (about 0.029) as the
  drift check after each weight sync, as before.

## Sources

- `examples/arena/harbor-rl-glm53-flash/kdatp/dsaqp/README.md` (1-node parity and timing)
- `examples/arena/harbor-rl-glm53-flash/kdatp/qp-t2/README.md` (8-node train-only arms)
- `training-runs/studies/trainer-core-profile-glm53-flash/STUDY.md` (the trace that ranked the cost)
- `examples/arena/harbor-rl-glm53-flash/kdatp/dsa/README.md` (the PR #3608 kernel test)
