# Study: KDA sharding across the tensor-parallel ranks

**Date:** 2026-09-28
**Status:** Closed
**Question:** Which KDA head split does GLM-5.3-Flash training release, the first port (`glm5_next_kda_tp`) or the upstream shared head-sharded layer, and is the shared layer's `grad_norm` gap a defect?
**Runs and data used:** r45 layout (TP8 SP, PP4 11/11/11/12, CP1, EP16, 8 nodes, 64 B200); start point r43 `iter_0000039` (the r45 seed, symlinks byte for byte); T2 data = 8 groups of r45 rollout 43 with r43 tokens and routing (313 rows + 1 pad, 22.3M tokens, max 131,070); base DCP `glm5.3-flash_torch_dist/release`; HF `GLM-5.3-Flash-BF16`; r45 `hf/rollout_39`. First live user: r47 `rl-glm53f47-wxj87`, W&B `rl-glm53f-adebt-766/0ix3m75e`. S3 prefixes: `guparpit/kdatp/`, `kdash/`, `kdrel/`, `kdsg/`, `kdfast/dump/l2a/`, `kdfinal/dump/20260928a/` under `s3://arena-scratch-prod-bom-ap-south-1/`.
**Code SHAs:** miles `3803da20d`: first port, flag `--glm5-next-kda-tp` (rebased as `a9207dd7d` on `arpit-glm-53`); `4716a367a`: `arpit-glm-53` head that built `miles-glm53-r17-20260928a`; `51629e714`: `glm5_next` on the shared head-sharded layer (`arpit-kda-shared-layer-rebased`; original `8770fcbab`); `c2a21839f`: kdash parity and round-trip harness (original `3c354c773`); `b4670791e`: gate-location and `safe_gate` isolation test (original `f88262f79`); `91939db95`: offline DCP-to-HF converters accept the stored KDA keys (original `db5158873`); `389d26303`: Dockerfile kernel-cache seed (original `ab6069ac9`; branch tip, image `miles-glm53-r16-20260928a`); `2fe28faba`: `safe_gate=False` test branch `arpit-kda-safegate-test`; `3ad0037e8`, `07b779190`, `8c9c69c48`: r17 lineage, r47 config, r47 launch docs; `12fae498b`: ADR-0016. Upstream: radixark/miles stack tip `ff6193f26` (#3634), heads #3609 `71fa168d5`, #3610 `62a6d89ee`, #3632 `5228139cb`; Kimi-K3 KDA TP PR #1825.
**Workflow ids:** `wf_545a45a0-6ca` (MFU study, the trigger), `wf_58a8a79c-51c` (first port, kdatp T1/T2), `wf_cce1530b-ce9` (shared layer, kdash T1/T2), `wf_add94c9c-caf` (converter fix, arm C, `baseline2`, gate 1), `wf_1e9f5ba2-351` (`safe_gate` off arm D, gate 2), `wf_60cbc331-c00` (tag audit, 1-node per-parameter dump, gate 3), `wf_44fd3690-bac` (r17 release, full-model dump, r47 launch).

## Method

Every KDA layer ran replicated: `hf_attention.py` gathered the full sequence
on each of the 8 TP ranks and `Glm5NextKDA` ran all 64 heads on each rank.
The MFU study (`wf_545a45a0-6ca`) estimated the 7 extra copies at 62-64% of
the executed training FLOPs, at a true MFU of about 1.7% against the logged
6.5%. That estimate is a FLOPs model, not a profile.

Two implementations split the heads, 8 per rank at TP8:

- **First port** (`kdatp`): `Glm5NextKDATensorParallel`, a port of the
  Kimi-K3 `_init_kda` layout (PR #1825). TE column linears for
  `q/k/v/b/f_b/g_b`, duplicated TE linears for `f_a/g_a`, TE row linear for
  `o_proj`, `A_log`/`dt_bias`/packed conv split by head (conv stride 3),
  `o_norm` replicated with the `sequence_parallel` mark. Gate computed by
  `fused_kda_gate` before `chunk_kda`, `safe_gate` off. Default off; the
  checkpoint keys and global shapes do not change.
- **Shared layer** (`kdash`): the upstream `linear_attn.py` and
  `miles/kernels/attention/delta_rule/` files at `ff6193f26`, byte-identical
  except a `mark_keep_in_fp32` wrapper and one import
  (`miles/kernels/attention/delta_rule/PROVENANCE.md`). One code path, no
  flag. Gate computed inside `chunk_kda` with `safe_gate=True` (the SGLang
  prefill path). Conv backward on causal-conv1d (`mix` backend). A prefix map
  keeps the stored `self_attention.kda.*` keys.

Harness files: `examples/arena/harbor-rl-glm53-flash/kdatp/` (`t1_parity.py`,
`build_rollout_data.py`, `gen_arm_configs.py`, `parse_logs.py`, `t1-job.yaml`,
`t2-job.yaml`, `RESULTS.md`) on `arpit-glm-53`; `kdash/` (same set plus
`t1_gate_isolation.py`, `check_dcp_to_hf.py`) on
`arpit-kda-shared-layer-rebased`.

Tests, in order:

1. **T1 parity**, 1 node. Module A (replicated) against B (split), layers 0 and
   44, base DCP and r43 `iter_0000039`, 4 packed sequences (512 to 24,576
   tokens). Gate: err(B) against an fp32 reference at most 2 x err(A) +
   2^-8; weight-grad norm ratio B/A in [0.99, 1.01]. Also L1 shard equality,
   L2 keys, E1 export against HF safetensors, N1 negative control (wrong conv
   layout), R1 save and reload both ways, T7 one-layer time and memory.
2. **T2 train-only smoke**, 8 nodes, r45 layout, `--debug-train-only` with
   `--load-debug-rollout-data`, R3 on, TIS on, no SGLang, no save, no W&B.
   Steps 40 to 43; step 40 is the parity step, steps 41 to 43 are timed.
3. **Gates** on the step-40 gap to A: run-to-run noise (`baseline2`), conv
   backend (arm C, `FLA_CONV_BACKEND=triton`), `safe_gate` off (arm D).
4. **Per-parameter gradient dumps** after `finalize_model_grads`: 1 node
   (5-layer slice, group 243) and 8 nodes (full model, step 40). Checks: tag
   audit, counted norm against true norm, cross-TP spread of replicated
   gradients, per-group norm ratio, cosine, relative L2.
5. **Converter** on the base DCP against `GLM-5.3-Flash-BF16`.
6. **Release** of the first port as `miles-glm53-r17-20260928a`; first live
   use on r47.

## Results

### T1: single-layer parity (base and r43 checkpoints, layers 0 and 44)

| Arm | Metric | Value | Source |
| --- | --- | --- | --- |
| First port | Checks passed | 48 of 48 | `kdatp/t1/20260927b/parity/SUMMARY.json` |
| First port | Output rel. L2 B-A; to fp32 A / B | 0.0033; 0.0056-0.0058 / 0.0060-0.0062 | `kdatp/RESULTS.md` T1 table |
| First port | Worst weight-grad rel. L2 B-A | 0.0040 (`g_a_proj`, `f_a_proj`) | same |
| First port | Weight-grad norm ratio B/A | 0.99893 to 1.00065 | same |
| First port | Conv, `b_proj`, `A_log`, `dt_bias` grads B-A | rel. L2 0 | same |
| Shared | Checks passed | 46 of 48; fails `base/layer0/B1_weight_grads`, `r45/layer0/B1_weight_grads` | `kdash/t1/20260927c/parity/SUMMARY.json` |
| Shared | Layer-0 `dt_bias` grad rel. error: A / B / limit | 9.4e-3 / 2.65e-2 / 2.26e-2 (base); 1.05e-2 / 2.70e-2 / 2.49e-2 (r45) | `wf_cce1530b-ce9` result 3 |
| Shared | Layer-44 `dt_bias` grad: B / limit | 1.89e-2 / 2.10e-2 | same |
| Shared | Output rel. error A / B | 5.6-5.8e-3 / 6.6-6.8e-3 | same |
| Shared | All 52 grad-norm ratios B/A | 0.9984 to 1.0005 | same |
| Shared | B against S (A with the new kernel call), `dt_bias` grad | at most 3e-8: the TP split is exact | same |
| Both | L1, L2, E1 (against HF and r43 `hf/rollout_39`), N1, R1 | pass, bit-exact where applicable | both `SUMMARY.json` |

Gate-location and `safe_gate` isolation, `dt_bias` grad rel. error, base layer 0
(job `kdash-t1-20260927d`, 4 GPUs; `wf_cce1530b-ce9` result 3; the S3 prefix
`kdash/t1/20260927d/` exists, the table itself is not re-verified):

| Gate computed | `safe_gate` off | `safe_gate` on |
| --- | --- | --- |
| Before `chunk_kda` (old way) | 9.37e-3 | 2.64e-2 |
| Inside `chunk_kda` (new way) | 9.16e-3 | 2.65e-2 |

`safe_gate` alone causes the 2 T1 fails. It does not change the forward output
or the input gradient (5.72e-3 on, 5.78e-3 off).

### T7: one KDA layer, forward plus backward, per GPU

| Arm | Tokens | A | B | Speedup | Source |
| --- | --- | --- | --- | --- | --- |
| First port | 131,072 | 270-273 ms | 66-70 ms | 3.9-4.1x | `kdatp/RESULTS.md` T7 |
| First port | 32,768 | 63-64 ms | 17.5-18.5 ms | 3.4-3.7x | same |
| Shared | 131,072 | 265-276 ms, 77.2 GiB | 50-51 ms, 17.9 GiB | 5.2-5.5x, 4.3x less memory | `wf_cce1530b-ce9` result 3 |
| Shared | 32,768 | 62 ms, 23.7 GiB | 13 ms, 9.0 GiB | 4.7-4.9x | same |

The layer is 4x to 5.5x faster, not 8x: the fla recurrent kernels fill few SMs
at 8 heads. The 62-64% FLOPs share therefore overstates the time share
(`kdatp/RESULTS.md`, an inference).

### T2: full model, 8 nodes, same data and checkpoint

Mean of steps 41 to 43 (seconds). A = `baseline`.

| Arm | `log_probs` | `actor_train` | `train_time` | `step_time` | tok/s | vs A (`actor_train`) | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A `baseline` | 223.2 | 992.0 | 1215.2 | 1279.4 | 22,509 | 0 | `kdatp/t2/20260927e/results.json` |
| `baseline2` (step 41 only) | 219.1 | 988.9 | 1208.1 | 1269.6 | 22,579 | -0.3% | same |
| First port `kdatp` | 185.4 | 780.9 | 966.3 | 1022.7 | 28,595 | -21.3% | same |
| First port + `block10` | 183.1 | 748.7 | 931.8 | 993.3 | 29,824 | -24.5% | same |
| First port + selective | 250.3 (step 40) | out of memory, stage 0 | - | - | - | - | same |
| Shared `shared` | 181.2 | 754.7 | 935.9 | 997.2 | 29,587 | -23.9% | `kdash/t2/20260927b/results.json` |
| Shared + selective | 267.5 (step 40) | out of memory, stage 0 | - | - | - | - | same |
| Shared, Triton conv (C) | - | 756.1, 755, 766 | - | 987, 993, 1010 | - | -23.5% | `kdrel/t2/20260928a/results.json` |
| Shared, `safe_gate` off (D), step 41 | 200.2 | 759.3 | 959.6 | 1013.2 | 29,406 | -23.5% | `kdsg/t2/20260928a/results.json` |

Peak torch `max_allocated` per pipeline stage 0/1/2/3 (GiB, max over ranks and
steps; B200 178.35 GiB):

| Arm | Stage 0 / 1 / 2 / 3 | DCGM used (AMP, 1-min) | Source |
| --- | --- | --- | --- |
| A | 128.79 / 129.72 / 124.72 / 124.90 | 163.5 / 164.9 / 160.4 / 160.6 | `kdatp/t2/20260927e/results.json`; `kdatp/RESULTS.md` |
| First port | 72.94 / 76.55 / 70.55 / 69.56 | 106.8 / 110.0 / 104.5 / 108.6 | same |
| First port + `block10` | 107.15 / 100.68 / 82.57 / 89.80 | 143.0 / 138.7 / 122.1 / 122.8 | same |
| Shared | 72.93 / 76.53 / 70.53 / 69.54 | not measured | `kdash/t2/20260927b/results.json` |
| Shared, C and D | 76.53 max (stage 1) | not measured | `kdrel/...`, `kdsg/...results.json` |

DCGM minus torch reserved is about 29 GiB per GPU (NCCL, CUDA context,
workspaces; `kdatp/RESULTS.md`). On the same measure, the first port and the
shared layer use the same memory within 0.02 GiB.

### Step-40 parity and the gates

Step 40, same checkpoint and data. N = run-to-run noise of A (`baseline2`).

| Arm | `rollout/log_probs` | `abs_diff` | vs A | `grad_norm` | vs A | Source |
| --- | --- | --- | --- | --- | --- | --- |
| A | -0.87510949 | 0.027205374 | 0 | 0.12739533 | 0 | `kdatp/t2/20260927e/results.json` |
| `baseline2` (N) | -0.87510949 (same bits) | 0.027205374 (same bits) | 0 | 0.12736258 | -0.026% | same |
| First port | -0.87509397 | 0.027264241 | +0.22% | 0.12652251 | -0.69% | same |
| First port + `block10` | -0.87509397 | 0.027264241 | +0.22% | 0.12756713 | +0.13% | same |
| Shared B (`safe_gate` on, `mix` conv) | -0.87511712 | 0.027270453 | +0.24% | 0.12891941 | +1.20% | `kdash/t2/20260927b/results.json` |
| Shared C (Triton conv) | -0.87511712 | 0.027270453 (same bits as B) | +0.24% | 0.12894483 | +1.22% | `kdrel/t2/20260928a/results.json` |
| Shared D (`safe_gate` off) | -0.87512133 | 0.027280699 | +0.28% | 0.12846518 | +0.84% | `kdsg/t2/20260928a/results.json` |

Step 41 `grad_norm` vs A: first port -0.35%, `block10` +1.07%, B +1.79%,
C +1.21%, D +1.45%; `abs_diff` vs A: first port +0.17%, B +0.16%, C +0.30%,
D +0.23% (same files). N at step 41: `grad_norm` -0.29%, `abs_diff` +0.09%.

| Gate | Rule | Result | Source |
| --- | --- | --- | --- |
| 1 (`wf_add94c9c-caf`) | B-A within 2N, or arm C moves toward A | NO-GO. A is bit-deterministic at step 40, so 2N = 0 for the forward; any correct rewrite fails. C equals B bit for bit on the forward and is +0.020% on `grad_norm`; the conv backend is not the cause | journal gate report |
| 2 (`wf_1e9f5ba2-351`) | D-A `grad_norm` at most 0.7% and `abs_diff` at most 0.3% | NO-GO. +0.84% and +0.28%. `safe_gate` removes 30% of the step-40 gap and 19% of the step-41 gap | journal gate report |
| 3 (`wf_60cbc331-c00`) | (a) a pure counting artifact, or (b) a real bug fixed and re-tested | NO-GO by default: the gap is numerics, neither branch fits | journal gate report |
| Decision (`wf_44fd3690-bac`) | user 2026-09-28: launch r47 on the first port; finish the shared-layer check in parallel | first port released as r17; shared layer verdict ROUNDING (below) | RUNLOG r47 entry; ADR-0016 |

Lesson: a "2 x run-to-run noise" gate fails every correct rewrite of a
bit-deterministic forward. The size of another correct rewrite (the first
port: `abs_diff` +0.22%, `grad_norm` -0.69%) is the usable yardstick.

### `grad_norm` gap analysis

**Tag audit** (`wf_60cbc331-c00` lane 1, read of `linear_attn.py`,
`glm5_next/kda.py`, Megatron `e8f574511` `finalize_model_grads`): every
head-sharded parameter carries `tensor_model_parallel=True` and is counted on
every rank; `f_a_proj`, `g_a_proj`, `norm.weight` are replicated, summed once
through the TP copy op, counted on rank 0 only. No tag can raise the norm by
+1.2%: the only over-count path is 8x on a replicated parameter, and none
carries the TP mark.

**1-node dump**, 5-layer slice, one step, group 243
(`kdfast/dump/l2a/analysis/contrib.txt`):

| Arm | Logged `grad_norm` | vs A | Counted / true norm | Max cross-TP spread | Largest single contributor (share of A's grad^2) | Source |
| --- | --- | --- | --- | --- | --- | --- |
| A old | 0.01062744 | 0 | 1.000000 | 0 | - | `contrib.txt` |
| First port | 0.01062790 | +0.004% | 1.000000 | 0 | L3 `self_attention_hyper_connection.alpha_post` (25.24%): ratio 0.99597 | same |
| B shared | 0.01057915 | -0.454% | 1.000000 | 0 | same parameter: ratio 0.98316, -0.8426% of the -0.9067% total | same |
| D `safe_gate` off | 0.01062505 | -0.022% | 1.000000 | 0 | same parameter: ratio 0.99379 | same |

The slice did not reproduce the full-model size or sign (+1.2% at 8 nodes).
The KDA parameter gradients of B match A as closely as the first port does
(per-parameter cosine at least 0.9996, `wf_60cbc331-c00` lane 2).

**Full-model dump**, 8 nodes, step 40, 4 arms
(`kdfinal/dump/20260928a/analysis/report-4arm.txt`, `extra.txt`):

| Arm | Logged `grad_norm` | vs A | Counted / true | Norm ratio without the 11 DSA-layer attention `alpha_post` | KDA params (49.7% of grad^2): net dsq / cos / rel. L2 | Source |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.12737089 | 0 | 1.000000 | - | - | `report-4arm.txt` |
| A2 (rerun of A) | 0.12739350 | +0.018% | 1.000000 | 0.99991 | -0.07% / 0.99992 / 0.013 | same; `extra.txt` |
| FP first port | 0.12652215 | -0.666% | 1.000000 | 0.98690 | -0.40% / 0.93686 / 0.355 | same |
| B shared | 0.12893170 | +1.225% | 1.000000 | 1.00043 | +0.21% / 0.93797 / 0.353 | same |

| Group | Share of A's grad^2 | FP: ratio / dsq | B: ratio / dsq | Source |
| --- | --- | --- | --- | --- |
| `alpha_post`, attention, 11 DSA layers | 2.11% | 1.2569 / +1.22% | 1.4597 / +2.38% | `report-4arm.txt` |
| `kda.conv1d` | 39.70% | 1.0001 / +0.01% | 1.0050 / +0.40% | same |
| DSA attention | 17.99% | 0.9864 / -0.49% | 0.9942 / -0.21% | same |
| Experts (sample) | 10.09% | 0.9786 / -0.43% | 0.9889 / -0.22% | same |
| `output_layer` | 4.25% | 0.9980 / -0.02% | 1.0022 / +0.02% | same |
| Largest scalar: layer 15 attention `alpha_post` | 0.89% | 1.11 | 1.74 (+1.80% of total) | same |

Read: the norm rebuilt from the dumps equals the logged norm to 8 digits in
every arm; replicated gradients are identical across TP ranks. Eleven scalar
mHC `alpha_post` weights of the DSA layers, not KDA parameters, hold nearly
the whole gap. Without them the shared layer matches A within 0.04%; the
first port is 1.3% off under the same test. The KDA gradient cosine against A
is 0.938 (B) against 0.937 (FP); the median per-layer KDA cosine is 0.985 for
both, the same-code rerun gives 0.9999. B's `A_log` gradient points the other
way at layers 4 (cos -0.24) and 10 (cos -0.01); FP has cos 0.44 at layer 9
and 0.53 at layer 1; both are at 0.99 or above from layer 17 on, and `A_log`
is 0.013% of the squared norm. A recompute-only change (`kdatp` against
`kdatp-block10`, same forward) moves the full-model `grad_norm` by 0.82%.
Gradient clipping (1.0) never fires at norms of about 0.13.

### Converter and kernel cache

| Item | Result | Source |
| --- | --- | --- |
| Offline DCP-to-HF on the shared-layer branch before the fix | `ValueError: Unknown parameter name: ...self_attention.kda.q_proj.weight` on every GLM-5.3 checkpoint; the eval fallback and ADR-0006 export path run this tool | `wf_cce1530b-ce9` review major 1 |
| After `db5158873` / `91939db95`, job `kdrel-conv-20260928e`, base DCP against `GLM-5.3-Flash-BF16` | both tools 780 of 780 tensors, 0 missing, 0 extra, 0 mismatch; 690 bit-exact, 90 `hc_*_fn` fp32 against bf16 with equal values (the same before the fix) | `kdrel/conv/20260928e/results.json` |
| Cold first log-prob pass, 8 nodes | 2,135 s (`kdatp-t2-20260927b`); AREnAThanatos deleted that job at 60 min of GPU power below 10% | `kdatp/RESULTS.md` jobs table |
| Warm cache from a 1-node warm-up (`kdatp-w1-*`, `kdash-w1-20260927bw`) | 312 s (kdatp) / 239 s (kdash) first log-prob pass | `kdatp/RESULTS.md`; `kdash/t2/20260927b/results.json` |
| Dockerfile build input `kernel-cache` (`ab6069ac9` / `389d26303`) | 15,794 files, 540 MB, unpacked to `/tmp/kernel_cache`; seed `kdash/kcache/20260927bw.tar` sha256 `72d6f3ea...` | `wf_add94c9c-caf` result 1 |
| Kernel cache in `miles-glm53-r16-20260928a` | present (15,794 files) | `wf_60cbc331-c00` result 2 |
| Kernel cache in `miles-glm53-r17-20260928a` | absent; the first r47 train step compiles | `examples/arena/README.md` lineage row r17 |

### Release and first live use (r47)

| Item | Value | Source |
| --- | --- | --- |
| Release image | `arena-slime-dev:miles-glm53-r17-20260928a`, `sha256:e6f04a9ca1abc9df17a7bbf9e3d2aaf6a643f40ed5a457c98a4114719972bcd0`, `arpit-glm-53` `4716a367a`, flag `glm5_next_kda_tp: true` | `examples/arena/README.md` lineage; `r47/BUILD.md` |
| Shared-layer candidate, not released | `miles-glm53-r16-20260928a`, `sha256:ca9505454ad7e2bcaac433e48f86443074534a9e936a336d5b0c8f6154af5e87`, `arpit-kda-shared-layer-rebased` `389d26303` | `wf_60cbc331-c00` result 2 |
| r47 launch | `rl-glm53f47-wxj87`, 2026-09-28 09:45:34Z; first try `rl-glm53f47-qtzlx` died on a Karpenter AMI-drift node replacement | RUNLOG r47 entry |
| r47 argv | `--glm5-next-kda-tp`, `--sequence-parallel`, `--skip-actor-forward-only`, `--recompute-granularity full` | RUNLOG r47 checks table |
| r47 step 0 (base DCP, agentic-debt-766, cold compile) | `train_rollout_logprob_abs_diff` 0.03249, `grad_norm` 0.1678, `ppo_kl` 0, `actor_train_time` 3,380 s, `step_time` 6,168 s | trainer-worker-0 log 11:39:30Z, `perf 0` |
| r47 step 1 | `abs_diff` 0.03218, `grad_norm` 0.1091, `actor_train_time` 1,302 s, 13,815 tok/s, `train_wait` 67 s | trainer-worker-0 log 12:02:24Z, `perf 1` |
| r47 steps 2 to 9 | `abs_diff` 0.0319 to 0.0378, `grad_norm` 0.084 to 0.115, `ppo_kl` 0 on every step | trainer-worker-0 log, 12:26Z to 15:36Z |

The r47 `abs_diff` of 0.032 to 0.038 sits at the size of the T2 value
(0.027 on r43 `iter_0000039` with r45 data). No large jump means the SGLang
weight sync received correct sharded weights. The r47 data differs (base
model, 766-task chains, 16,384 output cap, `--skip-actor-forward-only`), so
the r47 time and `abs_diff` are not a direct comparison with T2.

## Verdict

The numbers support both implementations as correct. The first port passes
every T1 check, loads and exports bit for bit, trains 21.3% faster on the
r45 layout, and cuts peak memory by 53 to 56 GiB per GPU; it is released as
r17 and runs r47. The shared layer is 3.4% faster still (754.7 s against 780.9
s `actor_train`), uses the same memory, and is one code path with the
upstream kernels. Its `grad_norm` gap (+1.2% at step 40) is rounding, not a
defect: the counted norm equals the true norm, replicated gradients are
identical across ranks, no tag or all-reduce fault exists, and 11 scalar mHC
`alpha_post` weights of the DSA layers hold the gap; without them the shared
layer is within 0.04% of the old layer while the first port is 1.3% off.
`safe_gate=True` is the one measured cost: 2.8x the rounding error on the
`dt_bias` gradient, one small parameter. The numbers do not support a claim
that either split changes training outcomes; no run has compared reward
curves across layouts. The step-40 gates as first written (2 x noise; 0.7%)
were not usable yardsticks for a bit-deterministic forward.

## Caveats and open items

- ADR-0016's memory row compares DCGM for the first port (105-110 GiB)
  with torch `max_allocated` for the shared layer (70-77 GiB). On the same
  measure the two are equal within 0.02 GiB. DCGM adds about 29 GiB.
- The `safe_gate` isolation table (job `kdash-t1-20260927d`) is from the
  journal only; not re-verified against S3.
- The 62-64% FLOPs share and the 1.7% true MFU are a FLOPs model from
  `wf_545a45a0-6ca`, not a profile. T7 shows 4x to 5.5x per layer, not 8x.
- No test loaded a first-port save into the shared layer. A move of r47 to
  the shared layer MUST remove the `glm5_next_kda_tp` key (the r16 and r15
  images reject the flag) and MUST happen at a DCP save with an explicit
  approval per run (ADR-0016). The rollback path old<->new is tested (R1).
- T2 had no SGLang. The weight sync was tested at module level (E1) and
  then live on r47 (`abs_diff` near the T2 value). The shared layer has no
  live SGLang check yet.
- r44 and r46 checkpoints were not loaded in any test; they hold the same
  442 KDA tensors as r43 (`wf_58a8a79c-51c` design).
- `baseline2` ran 2 steps (`debug_exit_after_rollout: 2`), so no noise
  number exists for steps 42 and 43.
- The 5-layer dump did not reproduce the full-model gap size or sign
  (-0.45% against +1.2%). The full-model dump is the evidence.
- The shared layer runs the conv backward on causal-conv1d (`mix`); arm C
  shows the Triton backend gives the same forward and +0.02% `grad_norm`.
- `--recompute-method block --recompute-num-layers 10` on the first port is
  GO after the first port runs clean (-24.5%, 35 GiB free DCGM on stage 0);
  `block9` and selective recompute are NO-GO (out of memory). Not used on
  r47.
- The `arpit-reconcile-upstream` branch holds a different Proposed
  ADR-0016 (weight versions); renumber one at the merge. The shared-layer
  branch sits on the old base and needs a rebase onto the reconciled branch
  before the two combine (`wf_3ff0709b-51c`).
- Upstream #3609 PR text (2.2-5.5x per layer, 3.6-5.6x less activation
  memory) is not re-verified. The four PRs were OPEN on 2026-09-28.
- The r17 image has no kernel-cache seed; r47 step 0 paid 3,380 s of
  `actor_train` against 1,302 s at step 1.
- Peak memory on r47 is not logged (`MILES_LOG_PEAK_MEMORY` unset).

## Actions taken

- `3803da20d` / `a9207dd7d`: `--glm5-next-kda-tp` on `arpit-glm-53`;
  `tests/fast/backends/megatron_utils/test_glm5_next_kda_tp.py`.
- Image `miles-glm53-r17-20260928a` built and released; lineage row
  `3ad0037e8`; r47 config `07b779190` (`glm5_next_kda_tp: true`); r47 launch
  `8c9c69c48` (`rl-glm53f47-wxj87`, 2026-09-28 09:45Z); r45 retired at
  `iter_0000049`.
- Branch `arpit-kda-shared-layer-rebased` at `389d26303`: `glm5_next` on the
  shared layer, `test_glm5_next_kda_shared_layer.py`, converter fix
  (`91939db95`), Dockerfile `kernel-cache` build input. Candidate image
  `miles-glm53-r16-20260928a` built, not released.
- Test branch `arpit-kda-safegate-test` (`2fe28faba`), not merged.
- ADR-0016 (`12fae498b`): first port released, shared layer the successor,
  `safe_gate=True` kept.
- Test images (`arena-slime-dev`): `miles-glm53-kdatp-test-20260927{b,c,d}`,
  `miles-glm53-kdash-test-20260927{a,b,c,d}`, `miles-glm53-kdrel-conv-20260928{a..d}`,
  `miles-glm53-kdsg-test-20260928a`. NEVER use them for a live run.

## Sources

- `miles_plugins/arena/adr/0016-shard-kda-across-tensor-parallel-ranks.md`
- `examples/arena/harbor-rl-glm53-flash/kdatp/RESULTS.md`, `kdatp/README.md`
- `miles/kernels/attention/delta_rule/PROVENANCE.md` (branch `arpit-kda-shared-layer-rebased`)
- `training-runs/harbor-rl-glm53-flash/r47/BUILD.md` ("KDA layer choice"), `RUNLOG.md` r47 entry (2026-09-28)
- `examples/arena/README.md`, trainer image lineage rows r14 to r17
- S3 `s3://arena-scratch-prod-bom-ap-south-1/guparpit/`: `kdatp/t1/20260927b/parity/SUMMARY.json`, `kdatp/t2/20260927e/results.json`, `kdash/t1/20260927c/parity/SUMMARY.json`, `kdash/t2/20260927b/results.json`, `kdrel/t2/20260928a/results.json`, `kdrel/conv/20260928e/results.json`, `kdsg/t2/20260928a/results.json`, `kdfast/dump/l2a/analysis/{contrib,report2,report3}.txt`, `kdfinal/dump/20260928a/analysis/{report-4arm,report-AB,report-A-FP-B,extra}.txt`, `kdash/kcache/20260927bw.tar`
- Journals `wf_545a45a0-6ca`, `wf_58a8a79c-51c`, `wf_cce1530b-ce9`, `wf_add94c9c-caf`, `wf_1e9f5ba2-351`, `wf_60cbc331-c00`, `wf_44fd3690-bac`, `wf_3ff0709b-51c`
- r47 trainer log `rl-glm53f47-wxj87-trainer-worker-0` (`step 0` to `step 9`, `perf 0`, `perf 1`), read 2026-09-28 at about 16:02Z
- Memory note `r45-mfu-investigation-2026-09-27.md` (safe_gate facts and gate history; notes only)
- Upstream radixark/miles PRs #1825, #3605-#3610, #3632, #3634
