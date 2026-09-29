# Study: MFU of GLM-5.3-Flash RL training (r45)

**Date:** 2026-09-27
**Status:** Closed
**Question:** Is the 6.5% MFU that miles logs for r45 correct, where does a training step spend its time, how does r45 compare with published MoE MFU, and which fixes have a precedent?
**Runs and data used:** r45 (`rl-glm53f45-vvqhg`, experiment `rl-glm53f-adebt-v3-r45`, W&B `arena/rl-glm53f-adebt-v3/ajsur4ej`, steps 40 to 43); r43 (W&B `2z0599lb`, steps 3 to 35, time fit only); sample summaries `s3://arena-scratch-prod-bom-ap-south-1/guparpit/debug/rl-glm53f-adebt-v3-r45/sample_summary/rollout_40.jsonl` to `rollout_44.jsonl`; the r45 trainer-worker-0 log (scratch copy `r45-10h.txt`, deleted; not under `/tmp/mfu/` on 2026-09-28); AMP DCGM counters (workspace `ws-4b40ad7c`) for the 8 trainer pods
**Code SHAs:** miles `e0987aed6` (trainer image `miles-glm53-r14-20260927a`): `miles/utils/flops_utils.py`, `miles/utils/train_metric_utils.py`, `miles_plugins/models/hf_attention.py`, `miles_plugins/models/glm5_next/kda.py`, `dsa.py`; veRL `220e9039` `verl/utils/flops_counter.py` (counter precedent); miles `a9207dd7d` (KDA tensor-parallel shard, the action from this study)
**Workflow ids:** `wf_545a45a0-6ca` (five investigate reports, one synthesis, two adversarial verify reports)

Terms. **MFU** (model FLOPs utilization) is the useful model math per second
divided by the peak math rate of the GPUs. **HFU** (hardware FLOPs
utilization) is all the math the GPUs run, repeated work included, divided
by the same peak. The peak is the B200 dense BF16 rate, 2250 TFLOPS
(`perf/mfu_peak_tflops`). **KDA** is the linear-attention layer type of
GLM-5.3-Flash; its cost grows in proportion to the sequence length `s`.
**DSA** is its sparse-attention layer type; each query reads at most 2048
keys. Neither has the dense `s^2` cost of standard attention.

## Method

- **Replay of the miles formula.** The study rebuilt the logged
  `perf/actor_train_tflops` from the per-sample `total_length` values in the
  `sample_summary` files with the `flops_utils.py` formula. The rebuilt
  values equal the logged values to two decimals (141.44 and 145.56
  TFLOPS), so the arithmetic of miles is correct and the FLOPs model is
  the only thing under test.
- **First-principles count.** The true forward cost per token comes from
  `config.json` (34 `linear_attention` + 11 `deepseek_sparse_attention`
  layers, `index_topk` 2048, 288 routed experts top-8 plus 1 shared,
  hidden 4096, MoE FFN 2048) and the layer code (`kda.py:68-85`,
  `dsa.py:85-93,189-193`). Training FLOPs = 3 x forward + 1 x indexer (the
  indexer runs under `no_grad`, `dsa.py:105-106`). The count uses the
  absorbed MLA convention (512 + 512 per key), which is what the kernel
  runs. The denominator is 64 GPUs x `actor_train_time` x 2.25e15.
- **Step-time breakdown.** Timer values come from the `perf N` lines of
  rank 48 (last pipeline stage, the rank that logs perf) and from
  `timer.py` lines of rank 0. GPU activity comes from AMP DCGM 30 s
  samples (`PIPE_TENSOR_ACTIVE`, `DRAM_ACTIVE`, `GR_ENGINE_ACTIVE`,
  power) over the 64 trainer GPUs. Recompute cost comes from a linear fit
  of `actor_train_time` and `log_probs_time` against tokens over 31 rows
  (r43 steps 3 to 35 and r45 steps 41 to 42). TileLang stalls come from
  compile log lines plus DCGM bins with tensor activity below 0.05.
- **External comparison.** Published MoE MFU figures, each divided by the
  dense BF16 peak of the GPU that ran the job.
- **Hardware and layout.** 8 `p6-b200.48xlarge` nodes, 64 B200 trainer
  GPUs, TP8 with sequence parallel, PP4 (11/11/11/12 layers), EP16, CP1,
  full uniform recompute, micro batch 1, `max_tokens_per_gpu` 8192, TIS,
  R3 routing replay (`r45/miles-config.yaml`).
- **Harness files.** Scratch scripts under `/tmp/mfu/` (`mfu_calc.py`,
  `analyze.py`, `fit.py`, `verify/indep.py`, `verify/amp_*.py`,
  `gpuc/`). They are not in git, on tmpfs, present at 16:35Z 2026-09-28
  (132 files). The journal
  `wf_545a45a0-6ca` holds their outputs.
- **Re-verification for this record (2026-09-28).** Token sums and row
  counts from the S3 sample summaries: exact match. W&B perf rows of
  `ajsur4ej`: exact match for every timer and MFU value below.
  `flops_utils.py:38-49,101-126` and `hf_attention.py:195-203` at
  `e0987aed6`: read again, as described. DCGM values, TileLang log lines,
  and the first-principles FLOPs count: taken from the journal (the
  verify agent rebuilt the count on its own); marked "not re-verified"
  where they stand alone.

## Results

### The logged MFU against the true MFU

| Metric | Step 41 | Step 42 | Source |
| --- | --- | --- | --- |
| Rows (segments plus DP pad row) | 920 | 1134 | S3 `rollout_41.jsonl`, `rollout_42.jsonl` (re-verified) |
| Tokens (sum of `total_length`) | 52,694,856 | 73,060,888 | same |
| Mean / max sample length | 57,277 / 110,450 | 64,428 / 113,007 | same |
| `perf/actor_train_time` (s) | 2388.34 | 3416.23 | W&B `ajsur4ej` (re-verified) |
| `perf/log_probs_time` (s) | 557.36 | 805.68 | same |
| `perf/step_time` (s) | 3115.63 | 4463.04 | same |
| Logged `perf/actor_train_mfu` | 6.29% | 6.47% | same |
| Logged `perf/actor_train_tflops` | 141.44 | 145.56 | same |
| miles forward FLOPs (GF per token) | 7.21e18 (136.8) | 1.06e19 (145.2) | `mfu_calc.py`; replay equals the log |
| Share of miles FLOPs from the dense `s^2` term | 80.1% | 81.2% | `mfu_calc.py`, `verify/indep.py` |
| True forward cost (GF per token) | 37.64 | 37.71 | first-principles count, verified twice |
| Training FLOPs, (3 x 36.80 + 0.84) GF x tokens | 5.84e18 | 8.11e18 | verify:compute correction of 5.86e18 / 8.14e18 |
| GPU-seconds x peak | 3.439e20 | 4.919e20 | 64 x `actor_train_time` x 2.25e15 |
| **True `actor_train` MFU** | **1.70%** | **1.65%** | absorbed MLA convention; 1.63% / 1.58% non-absorbed |
| Check with the 6N rule (N = 16.7B active) | 1.54% | 1.49% | `verify/indep.py` |
| MFU over-count, logged / true | x3.70 | x3.93 | verify:compute correction (x3.63 / x3.85 are forward-FLOP ratios) |
| True MFU over the whole step (`step_time`) | 1.30% | 1.27% | verify:compute correction of 1.4% at step 41 |
| Useful TFLOPS per GPU | 38.2 | 37.1 | 1.70% and 1.65% of 2250 |
| Estimated HFU (8 KDA copies + full recompute) | 6.7% | 6.5% | FLOPs model, ~433 GF per token executed |
| DCGM tensor-pipe busy, `actor_train` | 9.8% | 9.4% | AMP `ws-4b40ad7c` (not re-verified) |
| DCGM DRAM busy / GR engine busy, `actor_train` | 10.9% / 96.6% | 11.4% / 99.5% | same |
| Trainer tokens per second per GPU | 344.7 | 334.2 | `perf/actor_train_tok_per_s` / 64 |

- Step 40 is the first step after the resume: logged 1.69%, true 0.47%,
  `actor_train_time` 6320.88 s, `train_wait_time` 14,450 s (W&B). Leave
  it out of every average.
- The error grows with sequence length. Forward-FLOP over-count: x3.51
  (step 40), x3.63 (41), x3.85 (42), x4.05 (43, lengths only). From step
  41 to 42 the logged MFU rose and the true MFU fell.
- Later r45 steps logged 6.62% to 7.45% (rollouts 43 to 54, W&B). Their
  true MFU was not computed.
- The logged number lands near the HFU by chance: the fake `s^2` term is
  about the size of the real repeated work.
- `perf/tokens_per_gpu_per_sec` (445 to 454) is a rollout metric over the
  256 SGLang GPUs, not a trainer metric.

True forward cost per token by component, step 41 (GF per token):

| Component | GF | Component | GF |
| --- | --- | --- | --- |
| MoE MLP (8 routed + 1 shared + router) | 19.12 | Dense MLP (3 layers) | 0.91 |
| KDA projections (34 layers) | 9.36 | Indexer logits (forward only) | 0.84 |
| DSA sparse attention (11 layers) | 2.90 | KDA chunk core (estimated) | 0.40 |
| DSA projections | 2.58 | Indexer linears | 0.16 |
| LM head | 1.27 | mHC | 0.09 |

miles charges 109.5 GF per token of attention that does not run, and
27.2 GF per token for the linear part (true 33.5 GF: `o_proj` counted as
`hidden^2`, KDA projections as MLA). Source: investigate:flops-accounting.

### Where step 42 spends its time (4463 s = `train_wait` 240.8 + `log_probs` 805.7 + `actor_train` 3416.2)

| Rank | Cost | Seconds | Share of step | Kind | How it was found |
| --- | --- | --- | --- | --- | --- |
| 1 | Separate TIS old-log-prob forward pass | 806 | 18.1% | Measured | `perf 42` line; rank 0 timer reads 872 s, ranks range 795 to 874 s |
| 2 | Full activation recompute | ~782 (700 to 850) | 17.5% | Inferred | fit `log_probs` = 10.7 us/token x 73.06M; `actor_train` / `log_probs` = 4.24, or 3.79 without the JIT stall |
| 3 | TileLang `sparse_mla_bwd` re-JIT stalls | 360 (300 to 360 from JIT) | 8.1% | Measured | 12 DCGM 30 s bins below 0.05 in two rounds (13:40 to 13:43:30Z, 14:35 to 14:36:30Z); 0 bins in step 41 |
| 4 | Pipeline stage imbalance (stage 3: 12 MoE layers + LM head) | ~300 | 6.7% | Inferred | per-stage tensor busy 0.096 / 0.103 / 0.103 / 0.113, mean/max 0.914 |
| 5 | Idle gap between steps | 241 | 5.4% | Measured | `update_weights` 24.6 + rank-48 `data_preprocess` 143.7 + R3 fill ~71; tensor 0.0007, power 257 W |
| 6 | Padding to multiples of 4096 tokens | ~100 | 2.2% | Inferred | 2.43M pad tokens (3.3%) |
| 7 | Optimizer step (CPU offload) and gradient sync | 30 or less | 0.7% or less | Bound | 30 s DCGM resolution |
| 8 | Pipeline bubble (1F1B, PP4, 566 micro-batches) | ~25 | 0.6% | Simulated | `analyze.py` |
| 9 | Data-parallel imbalance | 0 | 0 | Measured | paired micro-batch efficiency 0.998 to 0.999 |
| - | **Core forward + backward that remains** | **~1,820** | 41% | Derived | runs at ~3.1% useful MFU: 8.11e18 / (64 x 1820 x 2.25e15) |

- Inside the core, a kernel runs 98 to 99% of the time, but the tensor
  cores are busy about 10% and HBM about 11%. Power is about 600 W of
  1000 W; clocks stay at 1845 to 1965 MHz (no throttle).
- The largest known waste in the core: each KDA layer runs in full on all
  8 TP ranks (`hf_attention.py:195-203`, comment "duplicated on all
  ranks"; `kda.py:68-85` full-size `nn.Linear`). The 7 extra copies are
  62.4% of executed training FLOPs; all 8 copies are about 71%. The DSA
  indexer logits also run on every TP rank.
- Rows 1 to 6 together: about 2,300 s. With them removed, the step is
  about 1,900 to 2,100 s (2.1 to 2.3x faster), or about 2.7 to 3.0%
  whole-step useful MFU. Rows 3 and 4 can overlap; treat sums as +-10%.
- The JIT stall is only partly compile time. The log shows about 13 s of
  compile per stage, but each stage stalls about 46 s. EP16 spans both DP
  replicas of a stage (TP8 x DP2), so one replica's JIT freezes all 64
  GPUs. The bwd kernel takes the padded length as a constant
  (`tilelang_sparse_mla_bwd.py:87-119,301`), so each new length compiles
  again. The cache is pod-local (`/root/.tilelang`, `TILELANG_CACHE_DIR`
  unset), so every resume starts cold: step 40 lost about 6,600 s
  (about 118 GPU-hours), not all of it TileLang.
- The rollout side is not the bottleneck: `train_wait` is 5.4% of the
  step, and the next rollout is ready about 40 min before the trainer
  asks for it.
- One optimizer step per rollout. The r45 config comment at line 281
  ("256 gives two optimizer steps per rollout") is wrong on the NATS path:
  `nats_rollout.py:2396` sets `target_groups = gbs // n` = 32 groups, and
  the log shows `num_rollouts=[256]`. `r47/miles-config.yaml:313-317,
  332-336` carries the corrected comment.
- The dispatcher is `alltoall` (miles rewrites `allgather` at
  `arguments.py:2757-2762`), `moe_enable_deepep` is off, and every
  overlap flag (`tp_comm_overlap`, `overlap_grad_reduce`,
  `overlap_param_gather`, `overlap_moe_expert_parallel_comm`,
  `moe_shared_expert_overlap`, `moe_router_fusion`) is off. The training
  block is a near copy of upstream `scripts/run_glm5_3_flash.py`; no knob
  is mis-set against the recipe (investigate:config-vs-upstream).

### r45 against published MoE MFU (each against the dense BF16 peak of its GPU)

| Band | Examples | MFU | Source |
| --- | --- | --- | --- |
| MoE pretraining, tuned, BF16, 4K tokens | DeepSeek-V3 on 256 GB200: 857 TF, 34.3%; Qwen3-235B on GB200 30.0%, on H100 32.4% (force-balanced routing) | 30 to 34% | arXiv 2603.07685 Table 11 |
| MoE pretraining, other | Mixtral-8x22B 49.3%, Qwen2-57B-A14B 39.0% (H100, BF16, 4K); MegaScale-MoE 27.9 to 32.5% (H800, 352B) | 28 to 49% | arXiv 2504.14960 Table 1; arXiv 2505.11432 section 6 text |
| Fine-grained MoE on B200 | GPT-OSS-120B on 64 B200, BF16: 16.4%; Qwen3-30B-A3B on 8 B200, MXFP8: 27.5% (about 13.8% against the FP8 peak) | 14 to 28% | Megatron-Bridge perf archive 26.04.01 |
| Long context, 128K to 131K tokens | Qwen3-235B MXFP8 on GB300: 46% (about 23% against the FP8 peak); Mixtral 128K BF16 42.9%; Llama 3 405B dense 38% | 38 to 46% | 2603.07685 Table 11; 2504.14960 Table 5; arXiv 2407.21783 Table 4 |
| MoE RL, training phase | veRL DeepSeek-V3 on 96 H20: 19% (about 28 TF per GPU); veRL Qwen3-30B-A3B: 31 to 40%; miles Qwen3.5-35B-A3B on H200: 5.0 to 5.8% (same inflated dense formula on its GDN layers); miles DeepSeek-V4-Flash on MI355X: 1.6% (FP8, about 0.8% against the FP8 peak) | 1.6 to 40% | veRL `docs/perf/dpsk.md`; miles PR #2353, PR #2038 |
| **r45** | logged 6.3 to 6.5%; true 1.65 to 1.70%; whole step 1.27 to 1.30%; HFU about 6.5%; about 38 TF per GPU useful | | this study |

- r45 is about 18 to 20x below tuned MoE pretraining and about 11x below
  the veRL DeepSeek-V3 RL figure. Its 38 TF per GPU of useful work is
  1.4x the per-GPU output of that H20 run, on GPUs with 15x the peak.
- The synthesis first placed veRL DeepSeek-V3 at 1.25% (B200 peak). The
  verify pass corrected it to 19% (H20 peak, the GPU that ran the job).
  The MoE RL band is 1.6 to 40%, not 1.2 to 5.8%.
- The expert shapes do not explain the gap. GLM-5.3-Flash has MoE FFN
  2048 and 288 experts; Qwen3-235B has 1536 and 128. GLM expert GEMMs are
  larger, not smaller. Long sequences alone cost at most about 10%
  relative (Mixtral 4K to 128K: 47.6% to 42.9%).
- Three causes are this run's own: the 8 KDA copies, full recompute, and
  the extra log-prob pass. The in-kernel inefficiency is the fourth.

### Ranked fixes, with precedent and status on 2026-09-28

| Rank | Change | Precedent | Expected gain (step 42) | Risk | Status |
| --- | --- | --- | --- | --- | --- |
| 0 | Count FLOPs per attention type in `flops_utils.py` (KDA linear, DSA top-k plus indexer, true `o_proj` width) | veRL `flops_counter.py:343-350` at `220e9039`; Megatron-Core 0.19 in the image (`training.py:390-720`, `effective_topk`, `gdn_layer_flops`, `hybrid_flops`) | No speed gain; the logged MFU becomes honest (6.47% -> 1.65%) | None | Not done: `flops_utils.py` last changed at `1200fca8f` |
| 1 | `--skip-actor-forward-only`; `rollout_batch_size` 64 -> 32 and `arena_inflight_multiplier` 4 -> 8 (the validator needs gbs = rbs x n) | miles `arguments.py:1558` (PR #2219); upstream tests combine it with rollout routing replay and with TIS | -806 s (-18%), about 1.22x throughput; `ppo_kl` reads exactly 0, so watch `train/tis` | Medium | ON in r47 ("flip A", `r47/miles-config.yaml:31-33,660`) |
| 2 | Partial or selective recompute (`--recompute-method block`, or `selective` with `mhc moe_act layernorm`) | Megatron-Core; the image warns when hyper-connections run without the `mhc` module | Up to -782 s | High (OOM at 131K tokens) | `kdatp` T2: selective OOM on stage 0 = NO-GO; block recompute of 10 layers with item 3 = -4.1% more, GO after item 3 runs clean; r47 keeps full recompute (`kdatp/RESULTS.md`) |
| 3 | Shard the KDA heads across TP (8 heads per rank) | miles Kimi-K3 PR #1825 `_init_kda` (merged 2026-09-23); upstream shared layer PR #3609 (open) | -62% of executed FLOPs; upper bound about 4.5% MFU if time tracks FLOPs | High (weight layout, SGLang sync) | First port `a9207dd7d` (`glm5_next_kda_tp`): T2 `actor_train` -21.3%, `log_probs` -16.9%, peak memory 105 to 110 GiB against 160 to 165; ON in r47 (image `miles-glm53-r17-20260928a`); ADR-0016 |
| 4 | Pipeline split 12/11/11/11 (`decoder_first` 12, `decoder_last` 11) | Megatron-Core flags already in use; `--account-for-loss-in-pipeline-split` fails (46 % 4 != 0, and it rejects the first/last flags) | About -200 to -300 s | Low to medium | Not done: r47 keeps 11/12 (`r47/miles-config.yaml:399-400`) |
| 5 | DeepEP flex dispatcher | On by default in upstream GLM-5 and GLM-5.2 744B scripts; `deep_ep` is in the image | Unknown, 5 to 15% of MoE time (low confidence) | Medium; likely blocked: the nodes use EFA, and `deep_ep/buffer.py:97-101` forces NVSHMEM IBGDA across nodes | Not done |
| 6 | Faster between-step fetch: `--object-store-backend mooncake`, or R3 payload int32 -> uint16 (`train_data_conversion.py:25`; 288 experts fit) | miles `arguments.py:619`; LMSYS 2026-08-20 post (GET 10 to 14x faster than Ray, synthetic payload) | -100 to -140 s; uint16 halves the ~49 GB per DP shard | Medium (Mooncake on EFA untested); low (uint16) | Not done |
| 7 | Fewer re-JIT stalls: persistent `TILELANG_CACHE_DIR`, `data_pad_size_multiplier` 1024 (8192-token pads, about +3% pad tokens) | GLM-5.2 recipe uses 1024; upstream `run_glm5_3_flash.py` forwards Triton and Inductor cache dirs | 0 to 360 s per steady step; about 118 GPU-hours per resume | Low; an EFS cache raced across the 8 ranks of a node (Triton `ESTALE`, RUNLOG r2) | Not done: r47 keeps 512 and no cache env |

Not worth the effort now: VPP on bubble grounds alone (bubble about 1%; but
VPP is the prerequisite for EP all-to-all overlap, so keep it on the list
for the core), optimizer and gradient overlap (30 s or less), a larger
`max_tokens_per_gpu` (97% of samples are alone in their micro-batch), FP8
(changes numerics), fused mHC (blocked by the assert at `glm5_next.py:95`),
`--tp-comm-overlap` (untested with variable-length THD batches).

## Verdict

The 6.5% MFU that miles logs for r45 is not a measure of useful work. The
formula charges dense causal attention on all 45 layers, and GLM-5.3-Flash
has none: 34 layers are linear attention and 11 are sparse top-2048. That
one term is 80 to 81% of the counted FLOPs at 57K to 64K mean tokens. The
true `actor_train` MFU is about 1.7% (range 1.5 to 1.8% across counting
conventions), the whole-step MFU about 1.3%, and the useful throughput
about 38 TF per GPU. Two verify passes confirmed the arithmetic, the layer
types, and the DCGM picture, and corrected four numbers (the MFU
over-count factors, the step-41 whole-step MFU, the veRL comparison, and
the Qwen3-235B expert shape). The numbers support four run-specific
causes: the KDA layer replicated on all 8 TP ranks (62% of executed
FLOPs), full recompute (about 23% of `actor_train`), the separate TIS
log-prob pass (18% of the step), and TileLang re-JIT stalls (8% of step
42). They also support a fifth cause without a size: the core forward and
backward runs at about 3% useful MFU with the tensor cores 10% busy. The
numbers do not support a comparison of the logged 6.5% with any published
MoE MFU, and they do not split the core into MoE all-to-all, KDA kernels,
TileLang sparse MLA, and unfused mHC; only a profile can.

## Caveats and open items

- The recompute cost (782 s) is inferred from a fit that pools r43 and
  r45 (31 rows). The 4.24 ratio check does not hold once the 360 s stall
  is removed (3.79). Range 700 to 850 s. Only a profile measures it.
- The stage imbalance (300 s) and padding (100 s) rows come from the
  FLOPs model, not from timers.
- About 33 s of each 46 s JIT stall per stage has no logged cause.
  Candidates: Triton autotune in the fla KDA backward, `torch.compile`
  recompiles of the mHC `_reference_proj_rms`. A larger pad multiplier
  alone does not remove that part for certain; pre-warm of all padded
  lengths or a persistent TileLang cache tests it directly.
- The per-resume 118 GPU-hours is not all TileLang. The step-40 log-prob
  pass ran 5x longer than the fit predicts with zero backward compiles.
- The HFU estimate (6.5%) is a FLOPs model. DCGM tensor busy (9.4%)
  counts cycles, not FLOPs, and skips FP32 CUDA-core work. The gap
  between them is not explained.
- The KDA chunk core (0.40 GF per token, 1.1% of the total) is estimated.
- The MLA counting convention moves the true MFU between 1.65% (absorbed)
  and 1.58% (non-absorbed) at step 42. The study uses absorbed, which
  matches the kernel.
- The external bands mix MXFP8 and FP8 runs with BF16 runs. The GB300 and
  MI355X peaks are not checked against datasheets. The Kimi-K2.5
  "17 to 18%" figure has no source. Whether GLM releases publish an MFU
  was not checked.
- The wall-time share of the KDA copies was not measured in r45. The
  `kdatp` T2 test later measured the split at -21.3% `actor_train`
  (`kdatp/RESULTS.md`), which is well below the 62% FLOPs share: time
  does not track FLOPs here.
- `SM_ACTIVE`, `SM_OCCUPANCY`, and NVLink TX/RX are not exported to AMP,
  so communication waits and slow kernels cannot be told apart. On
  Megatron, miles supports only `--profile-target train_overall`, which
  traces every rank with stacks and memory (`actor.py:134-138`,
  `profile_utils.py:60-78`); a 50 min r45 step is too big. Profile one
  step of a small same-shape smoke run after JIT warm-up.
- DCGM values, TileLang log lines, and the r45 argv dump are not
  re-verified for this record; the scratch copies under `/tmp/mfu/` are
  not in git, on tmpfs, present at 16:35Z 2026-09-28, and were not
  re-read. The journal holds them.
- r47 logs its MFU with the same formula. Its logged MFU rises further as
  the KDA split and the skipped pass cut the time, and it stays
  incomparable to any published figure. Compare `perf/actor_train_time`
  per token, or a true MFU, instead.

## Actions taken

- r47 (`rl-glm53f47-wxj87`, launched 2026-09-28 09:45:34Z) runs with
  `skip_actor_forward_only: true`, `rollout_batch_size` 32,
  `arena_inflight_multiplier` 8 (fix 1) and `glm5_next_kda_tp: true`
  (fix 3) on trainer image `miles-glm53-r17-20260928a` (miles
  `4716a367a`). The worker-0 argv shows `--skip-actor-forward-only`,
  `--glm5-next-kda-tp`, `--rollout-batch-size 32`,
  `--arena-inflight-multiplier 8` (RUNLOG 2026-09-28 r47 entry).
- ADR-0016 (`miles_plugins/arena/adr/0016-shard-kda-across-tensor-parallel-ranks.md`,
  commit `12fae498b`) records the KDA sharding decision and cites this
  study for the 62 to 64% replication share and the true 1.7% MFU.
- The `kdatp` harness (`examples/arena/harbor-rl-glm53-flash/kdatp/`,
  commits `a26047618` to `4716a367a`) measured fix 3 and fix 2 on 8 nodes:
  first port -21.3% `actor_train`; block recompute of 10 layers -24.5%
  against baseline; selective recompute OOM (`kdatp/RESULTS.md`).
- The r47 config comment corrects the "two optimizer steps per rollout"
  error (`r47/miles-config.yaml:332-336`).
- Not done from this study: fix 0 (the FLOPs counter), fix 4 (12/11/11/11),
  fix 5 (DeepEP), fix 6 (Mooncake or uint16 R3 payload), fix 7 (kernel
  cache persistence, pad multiplier 1024), the profiled smoke step, and
  the AMP export request for `SM_ACTIVE`, `SM_OCCUPANCY`, and NVLink
  fields.

## Sources

- Workflow journal `wf_545a45a0-6ca` (2026-09-27 14:50Z to 15:37Z):
  investigate reports `flops-accounting`, `config-vs-upstream`,
  `gpu-counters`, `step-breakdown`, `external-benchmarks`; `synthesize`;
  verify reports `compute` and `external`.
- Memory note `r45-mfu-investigation-2026-09-27.md` (session notes; the
  numbers above come from the journal and the primary sources, not from
  the note).
- W&B `arena/rl-glm53f-adebt-v3/ajsur4ej` (`https://mega.wandb.agi.amazon.dev`),
  `perf/*` rows for rollouts 40 to 54 (read 2026-09-28).
- `s3://arena-scratch-prod-bom-ap-south-1/guparpit/debug/rl-glm53f-adebt-v3-r45/sample_summary/rollout_40.jsonl`
  to `rollout_44.jsonl` (`total_length` per row).
- AMP workspace `ws-4b40ad7c` (ap-south-1), DCGM fields
  `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `DCGM_FI_PROF_DRAM_ACTIVE`,
  `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, power, for the r45 trainer pods
  (worker-1, 7, 8, 10, 14, 19, 30, 38).
- miles `e0987aed6`: `miles/utils/flops_utils.py:38-49,52-53,89,101-128`;
  `miles/utils/train_metric_utils.py:33-48`; `miles/utils/device_flops.py:12`;
  `miles_plugins/models/hf_attention.py:195-203`;
  `miles_plugins/models/glm5_next/kda.py:68-85`, `dsa.py:85-133,187-193`,
  `glm5_next.py:95,146-155`, `ops/kpool_indexer.py:230-280`;
  `miles/backends/megatron_utils/train_data_conversion.py:25,358,452`;
  `miles_plugins/arena/nats_arena/nats_rollout.py:867-874,2396`;
  `miles/utils/arguments.py:618-623,1558,2757-2762,3592-3654`.
- Run records: `training-runs/harbor-rl-glm53-flash/r45/` (`BUILD.md`,
  `miles-config.yaml`), `r47/` (`BUILD.md`, `miles-config.yaml`);
  `training-runs/harbor-rl-glm53-flash/RUNLOG.md` entries 2026-09-27 r45
  and 2026-09-28 r47 (kernel cache on EFS and its `ESTALE` revert: the
  r2 and r3 rows of the run table).
- `examples/arena/harbor-rl-glm53-flash/kdatp/RESULTS.md`; ADR-0016.
- External: arXiv 2603.07685 (Table 11), arXiv 2504.14960 (Tables 1 and
  5), arXiv 2505.11432 (section 6), arXiv 2407.21783 (Table 4),
  Megatron-Bridge `docs/performance-summary-archive.md` (release
  26.04.01), veRL `docs/perf/dpsk.md` and `verl/utils/flops_counter.py`
  at `220e9039`, radixark/miles PR #2353, #2038, #2219, #1825, #3597,
  #3609, LMSYS post 2026-08-20 on Mooncake rollout data transfer,
  DeepEP README.

## Follow-up

The profile that this study asked for ran on 2026-09-29 as the `kdatp-prof`
harness (two jobs, both deleted). Both traces came back empty: Kineto ran
out of its 33 default CUPTI buffers in the warmup step, and torch 2.13 then
stopped device collection. The record
[trainer-core-profile-glm53-flash](../trainer-core-profile-glm53-flash/STUDY.md)
holds the cause chain, the config-only fix, the clean r47-layout step times
(778.6 s `actor_train` for the T2 rows with `skip_actor_forward_only`), a
first-principles time budget that explains 47% of the step, and the ranked
candidate causes for the rest.
