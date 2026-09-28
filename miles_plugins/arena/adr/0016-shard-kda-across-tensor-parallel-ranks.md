# ADR-0016: Shard the GLM-5.3 KDA layers across the tensor-parallel ranks

**Status:** Accepted
**Date:** 2026-09-28

**Amends:** nothing; the checkpoint contract of ADR-0004 (the `--load`/
`--save` DCP layout) stays byte-compatible.
**Pairs with:** upstream radixark/miles PR stack #3605-#3610, #3632, #3634
(the shared head-sharded delta-rule layer, all OPEN on 2026-09-28).
**Numbering note:** the `arpit-reconcile-upstream` branch holds a different,
Proposed ADR-0016 (weight versions). Renumber one of the two at the merge.

## Summary

The GLM-5.3-Flash KDA linear-attention layers run sharded: at TP8, each
tensor-parallel rank holds 8 of the 64 KDA heads. Two implementations
exist. The first port (`glm5_next_kda_tp: true`, Kimi-K3 #1825 style) is
the released path and runs r47. The upstream shared head-sharded layer
(branch `arpit-kda-shared-layer-rebased`, image candidate
`miles-glm53-r16-20260928a`) is the successor; its measured gradient
difference is rounding, not a defect. A switch of a live run to it needs
its own approval.

## Context

- **Every TP rank ran the full KDA layer.** `hf_attention.py` replicated
  all 64 heads on all 8 ranks. The MFU study of r45 (2026-09-27) measured
  this replication at 62-64% of the executed training FLOPs, inside a true
  MFU of about 1.7% against the logged 6.5% (the logged figure charged
  dense `s^2` attention to all 45 layers; GLM-5.3-Flash has 34 KDA and 11
  DSA layers).
- **Upstream sharded KDA for other models.** Kimi-K3 PR #1825 (merged)
  shards KDA with TE column/row linears. The open #3609 stack adds one
  shared head-sharded delta-rule layer; its PR text benchmarks the
  GLM-5.3-Flash KDA shape at TP8 as 2.2-5.5x faster per layer with
  3.6-5.6x less activation memory. No upstream PR moves `glm5_next` onto
  it.

## Decision

1. The first port is the released path. `miles_plugins/models/glm5_next`
   gains `glm5_next_kda_tp` (default off). With the flag on, each rank
   holds `num_heads / tp_size` heads. Checkpoint key names and full tensor
   shapes DO NOT change: old saves load into the sharded layout, and
   sharded saves load back into the replicated layout. Release image:
   `miles-glm53-r17-20260928a`; first user: r47
   (`rl-glm53f47-wxj87`, dataset agentic-debt-766).
2. The shared layer is the successor. Branch
   `arpit-kda-shared-layer-rebased` imports the #3609/#3634 files at the
   stack tip `ff6193f26` (see
   `miles/kernels/attention/delta_rule/PROVENANCE.md`), moves `glm5_next`
   onto them as one code path, keeps the stored checkpoint keys, and fixes
   the offline DCP-to-HF converters for those keys. It merges after one
   more live-run validation, and the fork drops the first port then.
3. `safe_gate=True` stays in the shared layer's kernel call. It matches
   the SGLang inference kernel. The measured cost is a less exact
   `dt_bias` gradient (2.65e-2 against 9.4e-3 relative, one small
   parameter).

## Evidence

All arms ran the same synthetic step (rollout 40 data, r43
`iter_0000039`, 8 nodes, TP8 PP4 EP16) unless noted. A = the replicated
layer.

| Measure | First port | Shared layer |
| --- | --- | --- |
| T1 single-layer parity checks | 48 of 48 pass | 46 of 48 (the 2 fails are the `safe_gate` `dt_bias` gradient) |
| One KDA layer at 131K tokens | ~4x faster | 5.2-5.5x faster, 4.3x less memory |
| Train time per step vs A | -21.3% | -24% |
| Peak memory per GPU (A: 160-165 GiB) | 105-110 GiB | 70-77 GiB |
| Step-1 logprob gap vs A (`abs_diff`) | +0.22% | +0.24% |
| Step-1 `grad_norm` vs A | -0.69% | +1.20% (safe_gate off: +0.84%) |
| Checkpoint round trip old<->new | bit-exact | bit-exact |
| HF export / weight-sync gather | bit-exact | bit-exact |

The shared layer's `grad_norm` gap is rounding, not a defect. The
full-model per-parameter dump (2026-09-28,
`s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdfinal/dump/20260928a/analysis/`)
shows: no parameter-tag or TP all-reduce fault (replicated gradients
identical across ranks; counted norm equals the true norm to 8 digits);
the gap concentrates in 11 `alpha_post` hyper-connection weights of the
DSA layers, which sum rounding from the whole model; without them the
shared layer matches A within 0.04% while the first port is 1.3% off; the
KDA gradient cosine against A is 0.938 (shared) against 0.937 (first
port). A recompute-only change moves the same total by about 0.82%, so
the gap sits at the size of known-harmless changes. Gradient clipping
(1.0) never fires at these norms (~0.13).

## Consequences

- r47 trains with the first port. Watch
  `train_rollout_logprob_abs_diff` (~0.027) after each weight sync.
- A live run moves to the shared layer only at a DCP save, with an
  explicit approval per run. The rollback path (old image, same save) is
  tested.
- The offline DCP-to-HF converters accept the stored KDA keys
  (`self_attention.kda.*`, packed conv) on the shared-layer branch. The
  first-port path never changed the keys.
- The `--glm5-next-kda-tp` flag disappears when the shared layer merges;
  one code path remains (the repo rule).
- Send the `glm5_next` switch upstream after the #3609 stack merges, so
  the fork stops carrying the layer copy.

## Sources

- `examples/arena/harbor-rl-glm53-flash/kdatp/RESULTS.md` (first-port T1/T2)
- `miles/kernels/attention/delta_rule/PROVENANCE.md` (upstream SHAs)
- kdsg/kdrel arm logs (`s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdrel/`, `.../kdsg/`)
- Full-model gradient dump analysis (`.../kdfinal/dump/20260928a/analysis/`)
- MFU study 2026-09-27 (workflow records; true MFU ~1.7%, KDA replication 62-64% of executed FLOPs)
