# kdatp results: KDA tensor parallelism (item 3) and partial recompute (item 4)

The tests ran on prod-bom-v2 from 2026-09-27 18:41Z to 2026-09-28 02:58Z,
away from the live runs. No test wrote into a live-run directory or a W&B
run. At 2026-09-28 03:05Z the namespace held no `kdatp` PyTorchJob, Argo
workflow, pod, service, ConfigMap or Kueue workload.

## Decision

| Item | Decision | Evidence |
| --- | --- | --- |
| 3: `glm5_next_kda_tp: true` | GO | Module parity in bf16 limits, bit-exact checkpoint load and HF export, full-model step-1 log-probs equal to baseline within 1.6e-5 nats per token. `actor_train` -21.3%, log-prob pass -16.9%, 53 GiB less memory per GPU. |
| 4: block recompute, first 10 layers of each stage, with item 3 | GO, after item 3 runs clean | `actor_train` -4.1% against item 3 alone (-24.5% against baseline). Stage 0 peak: 143 GiB of 178 GiB (DCGM). |
| 4: selective recompute (`mhc`, `moe_act`, `layernorm`) | NO-GO | Out of memory on stage 0 in T2, also with item 3. |
| 4: no recompute | NO-GO | Out of memory on the 5-layer slice with replicated KDA. It keeps more activations than selective, which fails in T2. |
| 4: any recompute change without item 3 | NO-GO | The baseline has 13 GiB free (DCGM). One layer without recompute on stage 0 adds 34 GiB even with item 3. |

Gates for the first live use of item 3:

- The T2 harness has no SGLang. The weight sync to SGLang was tested only
  at module level (E1: the same gather and converter, bit-exact). On the
  first rollout after the resume, `train/train_rollout_logprob_abs_diff`
  MUST stay near the value before the resume (about 0.027 on r45). A wrong
  SGLang weight gives a large step.
- The test images hold the harness. NEVER use them for a live run. Build a
  live image from `arpit-glm-53` plus `3803da20d`.
- The live step time falls only when the trainer is the slower side of the
  asynchronous loop.

## Setup

- T2 layout: the r45 layout. TP 8 with sequence parallel, PP 4 (11, 11,
  11, 12 layers), CP 1, EP 16, micro batch 1, dynamic batch,
  `max_tokens_per_gpu` 8192, full uniform recompute of each layer,
  `use_tis`, R3 routing replay. 8 nodes, 64 B200 GPUs.
- T2 start point: r43 `iter_0000039`, linked into `kdatp/checkpoints/`.
  r45 `iter_0000039` holds only symlinks to these files (S3 Files mode
  `0120777`, target `../../rl-glm53f-adebt-v3-r43/iter_0000039/...`), so
  this is the r45 start point, byte for byte.
- T2 data: 8 groups of r45 rollout 43 with r43 tokens and routing. 64
  episodes, 313 rows plus 1 DP pad, 22.3M tokens. Mean row 71K tokens,
  62 rows at 100K tokens or more, maximum 131,070. Each arm trains the same
  file as rollouts 40 to 43. `global_batch_size` 64 gives one optimizer
  step per train call. Step 40 is the warm-up step. Steps 41 to 43 are
  timed. `baseline2` trains rollouts 40 and 41 only.
- The T2 token rate equals the r45 rate: baseline `actor_train` 22.5K
  tokens/s, r45 step 42 21.4K tokens/s (73M tokens in 3,416 s).

## T2: full model, eight nodes (`kdatp-t2-20260927e`)

### Time per phase, mean of steps 41 to 43 (seconds)

| Arm | `data_preprocess` | `log_probs` | `actor_train` | `train_time` | `step_time` | TFLOPS/GPU | MFU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | 41.7 | 223.2 | 992.0 | 1215.2 | 1279.4 | 165.1 | 7.34% |
| `kdatp` | 34.7 | 185.4 | 780.9 | 966.3 | 1022.7 | 209.8 | 9.32% |
| `kdatp-block10` | 39.2 | 183.1 | 748.7 | 931.8 | 993.3 | 218.8 | 9.72% |
| `baseline2` (step 41) | 39.3 | 219.1 | 988.9 | 1208.1 | 1269.6 | 165.6 | 7.36% |
| `kdatp-selective` | - | 250.3 (step 40) | out of memory | - | - | - | - |

| Arm | `actor_train` vs baseline | `log_probs` vs baseline | `train_time` vs baseline | Projected r45 step 42: `actor_train` / `log_probs` |
| --- | --- | --- | --- | --- |
| `baseline` | 0 | 0 | 0 | 3,416 s / 806 s (measured) |
| `kdatp` | -21.3% | -16.9% | -20.5% | 2,690 s / 669 s (-864 s) |
| `kdatp-block10` | -24.5% | -18.0% | -23.3% | 2,578 s / 661 s (-983 s) |
| `baseline2` | -0.3% | -1.8% | -0.6% | noise floor |

The projection scales the r45 step 42 times by the T2 ratios. The step time
includes `train_wait` (about 60 s). That wait has no meaning in a train-only
run.

### Peak memory per GPU (GiB, B200 capacity 178.35 GiB)

Torch values are the maximum over the ranks of each stage and over steps 40
to 43. DCGM values are the maximum `DCGM_FI_DEV_FB_USED` of the stage pods
in the arm window (AMP, 1-minute samples).

| Arm | Torch allocated, stage 0 / 1 / 2 / 3 | Torch reserved, max | DCGM used, stage 0 / 1 / 2 / 3 | Free at the DCGM peak |
| --- | --- | --- | --- | --- |
| `baseline` | 128.8 / 129.7 / 124.7 / 124.9 | 135.9 | 163.5 / 164.9 / 160.4 / 160.6 | 13.4 |
| `kdatp` | 72.9 / 76.6 / 70.6 / 69.6 | 80.6 | 106.8 / 110.0 / 104.5 / 108.6 | 68.3 |
| `kdatp-block10` | 107.2 / 100.7 / 82.6 / 89.8 | 114.5 | 143.0 / 138.7 / 122.1 / 122.8 | 35.3 |
| `baseline2` | 128.8 / 129.7 / 124.7 / 124.9 | 134.8 | 163.1 / 164.1 / 160.4 / 160.5 | 14.2 |
| `kdatp-selective` | out of memory on stage 0 | - | 66.2 / 73.5 / 74.2 / 90.0 | - |

- DCGM minus torch reserved is about 29 GiB per GPU in each arm (NCCL,
  CUDA context and library workspaces).
- `kdatp-selective` failed in the first `actor_train` on stage 0 (pods
  `worker-2` and `worker-3`). At the failed 512 MiB request, PyTorch held
  159.9 GiB and the process used 178.1 GiB. The 1-minute DCGM samples did
  not see this spike. The log-prob pass before it completed in 250 s.
- `kdatp-block10` estimate for the worst case: one layer without recompute
  costs about 10 GiB per 131K-token micro batch (stage 3: +20.2 GiB for 2
  layers, 1 micro batch in flight). Stage 0 holds up to 4 micro batches.
  Four 131K rows give about +40 GiB against the measured +34 GiB, so about
  150 GiB DCGM and 28 GiB free. `block9` adds one more layer on stage 0
  (+34 GiB to +40 GiB) and leaves 1 GiB or less, so `block9` is a NO-GO.
  This is an estimate, not a measurement.

### Step-1 parity (rollout 40, same checkpoint, same data)

| Arm | `rollout/log_probs` | `train_rollout_logprob_abs_diff` | `ppo_kl` | `pg_clipfrac` | `grad_norm` | `loss` |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline` | -0.87510949 | 0.02720537 | -8.5e-06 | 1.11e-04 | 0.1273953 | -3.6e-06 |
| `baseline2` | -0.87510949 | 0.02720537 | -8.5e-06 | 1.11e-04 | 0.1273626 | -3.6e-06 |
| `kdatp` | -0.87509397 | 0.02726424 | 1.2e-05 | 1.18e-04 | 0.1265225 | -1.3e-05 |
| `kdatp-block10` | -0.87509397 | 0.02726424 | 1.6e-05 | 1.18e-04 | 0.1275671 | -2.3e-06 |

- The SGLang log-probs of the data are -0.87294340 in every arm.
- The baseline forward is deterministic: `baseline2` gives the same
  log-probs, bit for bit. Its `grad_norm` differs by 0.026%.
- Item 3 moves the mean log-prob by 1.55e-5 nats per token. The distance
  to the SGLang log-probs grows by 0.22%. This is the hard gate of the
  review (minor 2): a KDA layer with TE-initialized weights gives a large
  distance.
- `kdatp` and `kdatp-block10` have the same forward but a 0.82% gap in
  `grad_norm`. The gap of item 3 to the baseline (-0.69% and +0.13%) is
  inside that spread. The loss is near 0 by design, and each gap is below
  1e-5.

Steps 41 to 43 after the same updates:

| Step | `grad_norm`: baseline / kdatp / kdatp-block10 | `loss`: baseline / kdatp / kdatp-block10 | `train_rollout_logprob_abs_diff`: baseline / kdatp / kdatp-block10 |
| --- | --- | --- | --- |
| 41 | 0.12331 / 0.12288 / 0.12462 (`baseline2` 0.12295) | -5.30e-4 / -5.27e-4 / -5.17e-4 | 0.027410 / 0.027457 / 0.027463 |
| 42 | 0.12154 / 0.12389 / 0.12429 | -1.861e-3 / -1.893e-3 / -1.891e-3 | 0.028123 / 0.028153 / 0.028146 |
| 43 | 0.15396 / 0.14964 / 0.14925 | -5.034e-3 / -5.076e-3 / -5.073e-3 | 0.030916 / 0.030995 / 0.030995 |

## T1: one node (`kdatp-t1-20260927b`)

All 48 checks pass (`t1/20260927b/parity/SUMMARY.json`). Module A is the
replicated KDA (64 heads on each rank). Module B is item 3 (8 heads per
rank). Each loads layers 0 and 44 from the base DCP and from r43
`iter_0000039` (the r45 start point). Input: 4 packed sequences of 512,
2,048, 5,000 and 24,576 tokens.

| Checkpoint / layer | Output rel. L2 B-A | Output max abs B-A / max abs A | Output rel. L2 to fp32: A / B | dX rel. L2 B-A | dX max abs B-A / max abs A | dX rel. L2 to fp32: A / B | Worst weight-grad rel. L2 B-A | Weight-grad norm ratio B/A | N1 rel. L2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base / 0 | 0.0033 | 0.00098 / 0.217 | 0.0058 / 0.0062 | 0.0046 | 6.1e-05 / 0.0059 | 0.0071 / 0.0074 | 0.0040 (`g_a_proj`) | 0.99956 to 1.00065 | 1.172 |
| base / 44 | 0.0033 | 0.125 / 27.8 | 0.0056 / 0.0060 | 0.0048 | 0.00098 / 0.166 | 0.0071 / 0.0075 | 0.0040 (`g_a_proj`) | 0.99957 to 1.00002 | 1.173 |
| r43 / 0 | 0.0033 | 0.00098 / 0.216 | 0.0058 / 0.0062 | 0.0046 | 4.6e-05 / 0.0059 | 0.0071 / 0.0074 | 0.0040 (`g_a_proj`) | 0.99957 to 1.00012 | 1.172 |
| r43 / 44 | 0.0033 | 0.125 / 27.8 | 0.0056 / 0.0060 | 0.0048 | 0.00146 / 0.166 | 0.0072 / 0.0075 | 0.0040 (`f_a_proj`) | 0.99893 to 1.00002 | 1.173 |

- F1 and B1 gate: the error of B to the fp32 reference is at most 2 x the
  error of A plus 2**-8. The largest excess of B over A is 0.0014
  (`o_norm`, base layer 44: A 0.0049, B 0.0063). The gate limit there is
  0.0138.
- The conv, `b_proj`, `A_log` and `dt_bias` gradients of B equal A (rel.
  L2 0). The largest gaps are on `f_a_proj`, `g_a_proj` and `o_norm`
  (0.0038 to 0.0040). These weights see the TP sum of 8 bf16 partial
  gradients.
- N1: a contiguous conv slice moves the output by 117%. F1 finds a wrong
  conv layout.
- L1: each B shard equals its slice of the A tensor, bit for bit.
- L2: B adds only the 9 TE `_extra_state` keys. The old checkpoints do not
  hold them. The production strictness (`assume_ok_unexpected`) loads B
  with no error.
- E1: the weight-sync gather plus `convert_glm5_next_to_hf` gives the same
  HF tensors for A and B. They equal the base HF safetensors and r43
  `hf/rollout_39`, bit for bit.
- R1: B saves a DCP. A fresh A and a fresh B load it bit for bit, so a
  rollback to the default layout works.

T7, forward plus backward time of one KDA layer (ms, A / B, speedup):

| Checkpoint / layer | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | --- | --- | --- |
| base / 0 | 15.9 / 19.7 (first, cold) | 63.3 / 18.5 (3.4x) | 271.3 / 69.9 (3.9x) |
| base / 44 | 15.8 / 7.4 (2.1x) | 63.8 / 17.7 (3.6x) | 272.6 / 66.1 (4.1x) |
| r43 / 0 | 15.8 / 7.4 (2.1x) | 64.3 / 17.5 (3.7x) | 271.8 / 66.1 (4.1x) |
| r43 / 44 | 15.7 / 7.5 (2.1x) | 63.9 / 17.6 (3.6x) | 270.2 / 66.2 (4.1x) |

The KDA layer gets 4x faster, not 8x (review minor 5: the fla recurrent
kernels fill few SMs at 8 heads). A 4x speedup and a 21% model gain put the
KDA share of the baseline `actor_train` at 28% or more. The earlier
estimate of 62% to 64% of the executed FLOPs overstates the time share.
This is an inference.

### T1c: 5-layer slice, one node

TP 8, PP 1, EP 8. One step of group 243 (43 rows, 4.46M tokens, maximum
131,070) from the base DCP. The kernels compile in these steps, so the times
are not comparable.

| Arm | Result | Torch allocated / reserved (GiB) | `grad_norm` |
| --- | --- | --- | --- |
| `baseline` | pass | 91.9 / 97.0 | 0.0106285 |
| `kdatp` | pass | 41.4 / 47.6 | 0.0106320 (+0.03%) |
| `selective` | out of memory | - | - |
| `kdatp-selective` | pass | 82.5 / 93.6 | 0.0106323 |
| `none` | out of memory | - | - |
| `kdatp-none` | pass | 89.7 / 94.1 | 0.0106325 |

## Item 2: `--skip-actor-forward-only`

No arm tested it. The T2 data gives its upper limit: the log-prob pass is
18.4% of `train_time` on the baseline, 19.2% with item 3 and 19.6% with
item 3 plus `block10`. With item 3 on, the pass for 512 samples at r45
scale falls from 806 s to about 669 s. Item 2 then saves about 669 s per
512 samples, less one extra weight sync (25 s to 45 s). The step-1 `ppo_kl`
of each arm is 2e-5 or less, so the separate pass gives the same log-probs
as the first training forward, within noise. Item 2 needs one optimizer
step per train call (`rollout_batch_size` 32 at `global_batch_size` 256).

## Jobs and images

| Job | Image | Nodes | UTC | Result |
| --- | --- | --- | --- | --- |
| `kdatp-t1-20260927b` | b | 1 | 09-27 18:41 to 19:40 | T1 48 of 48 pass. T1c 6 arms. |
| `kdatp-t2-20260927b` | b | 8 | 09-27 19:25 to 20:32 | AREnAThanatos deleted it at 20:32:12Z: cold kernel compile, 60-minute mean GPU power below 10%. The first log-prob pass took 2,135 s. No result. |
| `kdatp-w1-20260927dwbaseline`, `kdatp-w1-20260927dwkdatp` | d | 1 each | 09-27 20:49 to 21:37 | Kernel caches `kdatp/kcache/*.tar`. |
| `kdatp-t2-20260927e` | d | 8 | 09-27 21:40 to 09-28 02:58 | 5 arms. First log-prob pass 312 s with the warm cache. |

Each job started with no queue wait. No other workload was pending at each
submit. After each job ended, the platform deleted it.

| Tag (`arena-slime-dev:`) | Commit | Digest (us-east-1 and ap-south-1) |
| --- | --- | --- |
| `miles-glm53-kdatp-test-20260927b` | `fa31dad68` | `sha256:31060ca91f8215dff980701ed8813804f1a75b05fc99cc87c290f8472a9f44a4` |
| `miles-glm53-kdatp-test-20260927c` | `a8e2887b7` | `sha256:432a1eca7e8db95e315d2d4bcfbc47e45013e0073c85731ffd07a499e2191c83` (no job) |
| `miles-glm53-kdatp-test-20260927d` | `4231902ce` | `sha256:9e970578a62021cf0d4ca3360d767c4e3420bca404730d236edd0babda23c8d4` |

Raw results: `s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/`
(`t1/20260927b/`, `t2/20260927e/`, `warm/`, `kcache/`, `data/`).
`t2/20260927e/results.json` is the `parse_logs.py` output.
