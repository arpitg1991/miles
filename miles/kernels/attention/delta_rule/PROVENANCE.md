# Provenance: the shared head-sharded delta-rule layer

Source: https://github.com/radixark/miles, the open PR stack #3605-#3610,
#3632 and #3634, based on upstream `main` `3cdaad1027`. The files below are
taken at the stack tip, PR #3634 head `ff6193f26b3d983910d6af2c8e6956a139c06e68`.
This fork branched from upstream at `2799fe386` (2026-08-31).

| PR | Head SHA | Title |
| --- | --- | --- |
| #3609 | `71fa168d5a73cf2ec418e8c4e0256a06c66fe2fa` | feat(models): shared head-sharded delta-rule attention layer |
| #3610 | `62a6d89ee812d0cf7c3fef89016587cda027af1c` | feat(models): Qwen3.5 / Qwen3-Next GDN on the head-sharded layer |
| #3632 | `5228139cb224240fa594125cd40009d45000727e` | feat(models): Kimi-K3 KDA on the shared head-sharded layer |
| #3634 | `ff6193f26b3d983910d6af2c8e6956a139c06e68` | perf(kernels): use fla's Triton KDA dqkg backward on Blackwell (and `c5483c94e`, the causal-conv1d short-conv backward) |

## Files taken at `ff6193f26`

| File | Last upstream change | Fork adaptation |
| --- | --- | --- |
| `miles/kernels/__init__.py`, `miles/kernels/attention/__init__.py` | #3605 | none (empty) |
| `miles/kernels/attention/delta_rule/{__init__,backend,conv,heads,rule}.py` | #3606, #3609, #3634 | none |
| `miles_plugins/models/linear_attn.py` | #3609, #3610, #3632, #3634 | `mark_param_dtype` also sets `mark_keep_in_fp32`. This fork does not call `enforce_marked_param_dtypes`, so `Float16Module` keeps a tensor in fp32 only with that mark. |
| `miles/backends/megatron_utils/megatron_to_hf/gdn_layout.py` | #3609, #3610 | `load_hf_config` from `miles.utils.hf_config` |
| `tests/fast-gpu/delta_rule_reference.py` | #3609, #3610, #3632 | GDN references removed (their layers need #3610); GLM-5.3 reference added |
| `tests/fast-gpu/test_delta_rule_head_sharded.py` | #3609, #3610, #3632 | GDN cases removed; two GLM-5.3 cases added; no `hardware=` argument (this fork's `register_cuda_ci` has none) |
| `tests/fast-gpu/test_linear_attn_layer.py`, `tests/fast-gpu/test_delta_rule_conv_chunking.py` | #3609, #3634 | no `hardware=` argument |
| `tests/fast/models/test_delta_rule_layout.py` | #3609 | skips without megatron (this fork's `megatron_to_hf` package imports it) |
| `tests/fast/models/test_delta_rule_backend.py` | #3606, #3634 (`tests/fast/test_qwen_gdn_backend.py`) | the kernel selection tests only |

The linear-attention layer at the stack tip also holds the parts that #3610
and #3632 moved into it: the `weight_layout_version` buffer, the
`_build_convolutions` and `convolve` hooks, the reduce-scatter output under
sequence parallelism, and the CP relayout of `cp_utils.py`.

## Not taken

- The model switches of #3610 (Qwen3.5 / Qwen3-Next) and #3632 (Kimi-K3).
  This fork switches only GLM-5.3 (`miles_plugins/models/glm5_next/kda.py`).
- The checkpoint check of #3610 that rejects a DCP without
  `weight_layout_version`. GLM-5.3 keeps its checkpoint layout, so its layer
  makes the buffer non-persistent.
- The kernel refactors of #3605-#3608 (quant, MoE, DSA, dense attention).
- `miles/kernels/README.md` (it describes those refactors) and
  `tests/manual/bench_delta_rule.py` (it needs the #3610 GDN layers).
