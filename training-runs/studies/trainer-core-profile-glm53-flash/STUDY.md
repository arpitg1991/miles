# Study: Trainer core profile of GLM-5.3-Flash RL training (r47 layout)

**Date:** 2026-09-29
**Status:** Open
**Question:** Where does the forward and backward time of one GLM-5.3-Flash training step go on the r47 layout, why does the trainer run 10 to 20x below published MoE training efficiency, and what does a profile need in order to answer this?
**Runs and data used:** profile jobs `kdatp-prof-20260929a` (T2 rows: 314 rows, 22.3M tokens) and `kdatp-prof-20260929ab` (groups 239 and 258 of r45 rollout 43: 66 rows, 4.72M tokens), both 3 steps from the r43 `iter_0000039` seed, both `rc 0`, both deleted; outputs under `s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/prof/20260929a/` and `.../20260929ab/` (`driver.log`, `kdatp-prof/trainer-0.log`, `results.json`, `tb/`); slice data `.../kdatp/data/prof-239-258/`; kdatp T2 and T7 results (`kdatp/RESULTS.md`); the r47 pod `rl-glm53f47-wxj87-trainer-worker-0` (read-only, runtime code and `config.json`); no live run was touched
**Code SHAs:** miles `4716a367a` (trainer image `miles-glm53-r17-20260928a`, digest `sha256:e6f04a9ca1abc9df17a7bbf9e3d2aaf6a643f40ed5a457c98a4114719972bcd0`): `miles/utils/profile_utils.py`, `miles_plugins/models/glm5_next/{kda.py,dsa.py,glm5_next.py}`, `miles_plugins/models/hf_attention.py`, `glm5/ops/tilelang_sparse_mla_{fwd,bwd}.py`, `glm5_next/ops/kpool_indexer.py`; harness commits `e7a0d35ba`, `82e080ec3`, `a441a1031` (`examples/arena/harbor-rl-glm53-flash/kdatp/prof/`); in the image: Megatron-Core `0.19.0+e8f574511`, TE 2.17.0, torch `2.13.0+cu130` (Kineto at `094d3c1d`), fla 0.4.2, tilelang 0.1.9
**Workflow ids:** `wf_269fcf06-9bc` (profile attempt, three research reports, three adversarial verify reports) and the synthesis run of 2026-09-29 that wrote this record

Terms. **MFU** (model FLOPs utilization) is the useful model math per second
divided by the peak math rate of the GPUs (B200 dense BF16, 2250 TFLOPS).
A **micro-batch** is the set of tokens one pipeline stage processes in one
forward and one backward pass; here it is almost always one sample of 57K
to 131K tokens. **TP** (tensor parallel) splits a layer over the 8 GPUs of a
node. **EP** (expert parallel) puts different experts on different GPUs; the
**all-to-all** is the network step that moves each token to its expert and
back. **EFA** is the AWS network card between nodes; **NVLink** is the fast
link inside a node. **Recompute** runs the forward pass a second time in the
backward pass to save memory. **Kineto** is the C++ library behind
`torch.profiler`; **CUPTI** is the NVIDIA library that records GPU events;
a **GPU record** is one CUPTI entry (a kernel, a copy, or an API call).
**KDA** is the linear-attention layer type (34 of 45 layers); **DSA** is the
sparse-attention layer type (11 of 45).

## Method

- **Isolation profile attempt.** The `kdatp-prof` harness
  (`examples/arena/harbor-rl-glm53-flash/kdatp/prof/`: `README.md`,
  `kdatp-prof-job.yaml`, `kdatp-prof-run.sh`, `patch_prof_arm.py`) runs the
  kdatp T2 arm on 8 `p6-b200.48xlarge` nodes (64 B200) with the r47 trainer
  layout: `glm5_next_kda_tp`, full uniform recompute, TP8 with sequence
  parallel, PP4 (11/11/11/12 layers), EP16 (two nodes per stage), DP2,
  `micro_batch_size` 1, `max_tokens_per_gpu` 8192, pad multiple 4,096
  tokens, `skip_actor_forward_only` (r47 flip A). The image is the released
  r47 trainer image `miles-glm53-r17-20260928a` (digest above); the pods read
  the driver from the scratch mount, so no image was built. The miles torch
  profiler (`--use-pytorch-profiler --profile-target train_overall
  --profile-step-start 1 --profile-step-end 2`) was set to warm up on step 0
  (rollout 40), record step 1 (rollout 41) on all 64 ranks, and run step 2
  (rollout 42) clean. Job `kdatp-prof-20260929a` ran the full T2 rows (156
  micro-batches per DP rank). Job `kdatp-prof-20260929ab` ran a 2-group
  slice (33 micro-batches per DP rank) with `KINETO_LOG_LEVEL=1`. Both ran 3
  steps to `rc 0` and were deleted when `driver.log` said done. No live run
  was touched.
- **Three research tracks (read-only).** (1) A first-principles time model
  of one micro-batch per cost family (`research/first-principles.md`,
  `model.py`): bytes over bandwidth for communication, a serial latency
  chain for KDA, a gather for DSA, FLOPs over peak times efficiency for
  GEMMs, plus launch and sync counts. (2) A KDA and DSA kernel note anchored
  on the measured T7 timings (`research/kda-dsa-kernels.md`). (3) A survey of
  how published 20 to 40% MoE runs are laid out, against arXiv 2603.07685
  (Tables 11 and 20, Appendix B.2), the Megatron-Bridge performance archive
  26.04.01, veRL `docs/perf/dpsk.md`, and the upstream miles recipes
  (`research/how-others-do-it.md`).
- **Adversarial verify pass.** Three verifiers tried to refute each claim
  against the trainer logs, the runtime code in the r47 pod, the Kineto
  source at the torch v2.13.0 pin, and the paper text on disk
  (`verify/time-model.md`, `verify/peers.md`, `verify/profile-next.md`).
  Every CORRECTED and REFUTED verdict is applied below; claims that no
  source settles are marked UNVERIFIED.
- **Scratch.** `/workplace/guparpit/kdfast/scratch/prof/20260929a/`
  (`run-20260929a/`, `run-20260929ab/`, `research/`, `verify/`, `traces/`
  with the stub traces, `extract_log.py`). Not in git.

## Results

### (a) What the profile attempt measured

The traces are empty (see (b)), but both jobs gave clean step times and
peak memory on the r47 layout with warm kernel caches. Source:
`results.json` of each run (`perf N` lines of rank 48, `[peak-memory]`
lines of every rank).

| Arm | Micro-batches per DP rank | Step | `actor_train` (s) | `train_wait` (s) | Torch max allocated by stage 0 / 1 / 2 / 3 (GiB) | Torch max reserved (GiB) |
| --- | --- | --- | --- | --- | --- | --- |
| T2 rows (314 rows, 22.3M tokens) | 156 | 40 (cold) | 1,049.5 | 363.9 | 72.9 / 76.6 / 70.6 / 69.5 | 79.6 |
| T2 rows | 156 | 41 | 800.2 | 48.0 | same | 79.9 |
| T2 rows | 156 | 42 | **778.6** | 67.8 | same | 80.1 |
| 2-group slice (66 rows, 4.72M tokens) | 33 | 40 (cold) | 401.4 | 41.4 | 62.6 / 66.3 / 61.4 / 61.2 | 71.8 |
| 2-group slice | 33 | 41 | 186.2 | 10.8 | same | 71.8 |
| 2-group slice | 33 | 42 | **182.3** | 10.0 | same | 71.8 |

- **Skip-forward effect.** With `skip_actor_forward_only` the step has no
  separate log-prob pass. The kdatp T2 arm without it measured `actor_train`
  780.9 s plus `log_probs` 185.4 s, `train_time` 966.3 s (`kdatp/RESULTS.md`,
  mean of steps 41 to 43). With it, `train_time` is 778.6 s (step 42). The
  saving is the whole log-prob pass, about 188 s or 19% of `train_time`;
  `actor_train` itself did not change (780.9 against 778.6 s).
- **Cold-compile cost.** Step 0 ran 271 s longer than step 42 on the T2 rows
  (1,049 against 779 s) and 219 s longer on the slice (401 against 182 s),
  with the kernel cache tarball loaded (2 TileLang entries, 2,549 Triton
  entries, `driver.log`). The compile share inside that difference was not
  measured. Step 1 carried the profiler warmup and ran 22 s (T2) and 4 s
  (slice) longer than step 2.
- **Two measured points fit one line.** Time = 65.9 us per token per DP
  replica plus 7.6 s fixed (verify: time-model, section 7). The fixed part
  plus the pipeline bubble is about 22 s, 3% of the step. So the step time
  is per-token work inside the micro-batches, not a fixed or a bubble cost.
- **Kineto record count as a datum.** In the slice run every rank processed
  2.19M to 2.80M GPU records (157 to 198 MB, 70.6 to 71.8 B per record) in
  the cold step 0 before the cut. Per micro-batch: 66K to 69K records on
  stage 0, 73K to 75K on stage 1, 72K to 77K on stage 2, 82K to 85K on stage
  3. The stage-3 to stage-0 ratio (1.22) matches the layer mix of the time
  model (kernel ratio 1.26). At 4 to 5 records per launch this reads as
  about 17K to 21K launches per stage-3 micro-batch, 3 to 4x the 6,000 the
  time model assumed. Plausible, not confirmed: only a trace with kernel
  records settles the mix, and step 0 holds some compile-time launches.

### (b) Why the trace is empty: the confirmed cause chain

1. `miles/utils/profile_utils.py` schedules `wait 0, warmup 1, active 1`
   for `profile_step_start 1`; `TrainProfiler.on_init_end` starts the
   profiler (`megatron_utils/actor.py:327`), so step 0 runs as WARMUP with
   GPU collection on.
2. Kineto keeps at most `1 + 128 MiB / 4 MiB = 33` open CUPTI buffers per
   rank by default (`CuptiActivityApi::setMaxBufferSize`). The log prints
   `Max GPU buffer size: 128MB` once per rank.
3. All 64 ranks logged `Exceeded max GPU buffer count (33 >= 33) -
   terminating tracing` at 09:23:37, in the same second, 4 s after the
   gradient sync of step 0 began (rank 0 `check_grads` at 09:23:33.7),
   while the record volumes differed by 28%. So the cap fired on a burst of
   open buffers in the gradient-sync and optimizer phase, not on total
   volume: every rank had already processed more than 128 MiB. Which CUPTI
   queue rule drives the burst is UNVERIFIED and does not change the fix.
4. torch 2.13 `profiler.py:1207-1222` reads the Kineto stop flag at the next
   `step()`; with the previous action WARMUP it warns `Device profiling
   activity collection was stopped early at step 1` and enters
   `DEVICE_STOPPED`, which runs `start_trace, stop_trace, _trace_ready` at
   once. Each rank wrote a stub trace (13 to 33 KB, under 10 ms) at 09:24
   and recorded nothing in step 1. Both runs show this; the first run had
   the Kineto log off and shows only the 64 torch warnings.
5. Two proposed ways out are refuted. A smaller slice does not avoid the
   burst, and on the torch sync path nothing clears the warmup buffers, so
   steps 0 and 1 share the budget: even by volume only about 8 micro-batches
   per DP rank per step fit, half a group. The one-line
   `custom_profiler_config="ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB=2048"` fix in
   the prof README (`a441a1031`) does not work on torch 2.13:
   `kineto_shim.cpp` wraps it as `CUSTOM_CONFIG=...`, the line then has two
   `=` signs, and `AbstractConfig::parse` rejects it. Do not build an image
   for it.
6. The config-only fix is feasible: Kineto at the torch v2.13.0 pin reads
   `KINETO_CONFIG` (or `/etc/libkineto.conf`) in
   `ActivityProfilerProxy::prepareTrace` and honors
   `ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB`; the pod `libtorch_cpu.so` holds the
   strings `KINETO_CONFIG`, `/etc/libkineto.conf`,
   `ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB`, and `Max GPU buffer size: `. The
   raylet environment delivered `KINETO_LOG_LEVEL=1` to all 64 ranks in the
   second run, so it delivers `KINETO_CONFIG` the same way. Live effect not
   tested (no launch allowed). Fallback: `profile_step_start 0`,
   `profile_step_end 1` saves the partial cold step 0 at the cut with the
   stock cap (the natural `(RECORD_AND_SAVE, NONE)` transition is excluded
   from the `DEVICE_STOPPED` override, `profiler.py:1217`).

### (c) First-principles time budget, T2 shape (seconds per step)

Stage 3 is the critical path (9 KDA, 3 DSA, 12 MoE layers, LM head). The
published model gave 347 s. The verify pass corrected five inputs: 156
micro-batches (not 157), `E[s^2]` 6.55e9 (not 5.5e9), expert tokens per
rank per MoE layer call `(s_A + s_B)/2`, about `s`, because the EP16 group
serves both DP replicas (the model had `s/2`; expert GEMM doubles to
27.4 s), sparse index width 2112 (not 2048), and 15 TP units per KDA layer
(not 14). Corrected central: 367 s.

| Family | s per step | Basis | Confidence |
| --- | --- | --- | --- |
| EP all-to-all over EFA | 96.7 | 12 MoE layers x 6 large calls (9 NCCL kernels with the probs) per micro-batch; `s x 8192 B` per rank per call; 8 of 16 destinations on the other node (`parallel_state.py:792-798`, rank map from the log); EFA at 35 GB/s per GPU = 70% of 400 Gbps | Counts CONFIRMED; realized EFA bandwidth UNVERIFIED (the widest lever: at 15 to 20 GB/s the family is 170 to 230 s) |
| DSA sparse MLA (fwd, recompute, bwd) | 87.3 | one block per query gathers 2112 x 576 x 2 B = 2.4 MB of KV; repeated on all 8 TP ranks; 8 heads padded to 16; bwd gathers again and adds dKV with fp32 atomics; gather at 60% of HBM, bwd 3x fwd | Structure CONFIRMED (`dsa.py:90-98`, `tilelang_sparse_mla_bwd.py`); efficiency UNVERIFIED (the KV of one 64K row, 75 MB, fits the 126 MB L2, so the HBM framing does not hold there); 5x bwd and 35% gather give 203 s |
| KDA chunk kernels | 53.6 | 4 serial passes per layer per micro-batch (`chunk_fwd.py:93`, `chunk_bwd.py:462,514`) over s/64 chunks at 8 us per chunk; grid 16 to 32 blocks on 148 SMs (`chunk_delta_h.py:691`) | Passes and grid CONFIRMED; 8 us per chunk UNVERIFIED (a fit to T7: one 8-head layer at 131K fwd+bwd = 66 ms; within 8% of the T7-anchored note); 15 us gives 99 s |
| TP/SP collectives | 29.0 | 221 units of [s, 4096] bf16 over TP8 on stage 3 (15 per KDA layer, 4 of them separate backward all-reduces of q, k, v, b) at 650 GB/s | Medium; bus bandwidth assumed |
| Expert grouped GEMM | 27.4 | about `s` routed tokens per rank per layer (3,640 per expert at 64K), 18 groups, 4 passes at 45% | Medium (CORRECTED 2x) |
| Dense TP GEMMs | 16.8 | 496 MFLOP per token per rank x 4 passes at 60% (LM head counted below) | Medium |
| Launch and host syncs | 16.8 | 6,000 kernels x 10 us + 2 dispatcher syncs per MoE layer x 2 ms; the Kineto datum implies 17K to 21K launches, so 31 s at 10 us and 125 s at 40 us; `CUDA_DEVICE_MAX_CONNECTIONS=1` makes any stream's host stall a GPU stall | Low; under-sized 3 to 4x (the one family the datum contradicts) |
| DSA indexer and top-k | 11.2 | `1024 x s^2` FLOP at 50%, 2 passes, on every TP rank; fp32 logits 4.3 to 17 GB per layer | Medium |
| 1F1B bubble, fixed per step, LM head, MoE glue, mHC, PP send/recv, DSA glue | 27.7 | bubble `3/156 = 1.9%`; fixed 5 s; LM head 4.4 s | Medium; the measured fit gives bubble plus fixed about 22 s |
| **Total, corrected central** | **367** | **47% of the measured 778.6 s** | |
| Measured T2 `actor_train`, step 42 | 778.6 | `results.json` | Measured |
| **Gap** | **about 410 s** | 434 s against the published 347 s | **UNEXPLAINED pending a trace** |

- The gap is the same 2.1x at both sizes (slice: 87 s modeled against
  182.3 s measured), so it is per-token work inside the micro-batches.
- The "all pessimistic" variant of the published model (699 s) is not a
  bound: with the expert fix it is 729 s, and with the measured kernel count
  (`KERNEL_MULT 4`) it is 843 s, above 781 s. The family that is under-sized
  is the host launch count, and it was already in the model.
- The whole T2 step at peak is 23 s of useful math (25 s with the replicated
  indexer). GEMMs are 6 to 13% of the modeled step and 3 to 6% of the
  measured step. The recompute share of the modeled step is about 24%.
- Stage 0 idles about 24% of the step in the model. UNVERIFIED: no
  per-stage timer exists.
- The ranking (all-to-all, sparse MLA, KDA chunk, then TP collectives) holds
  through every correction.

### (d) How the published 20 to 40% runs differ

| Property | Published rows | r47 | Source and status |
| --- | --- | --- | --- |
| EP placement | Every Blackwell row above 30% with a readable config keeps EP x TP inside the NVLink domain (GB200 and GB300 NVL72 at EP32 or EP64; B200 NVL8 rows at EP8: GPT-OSS-120B, Qwen3-30B-A3B, Qwen3-235B). H100 cross-node rows use DeepEP plus 1F1B overlap on InfiniBand. One B200 row crosses nodes (DeepSeek-V3, EP32 over 4 nodes, 38.4%) and its dispatcher, overlap, and fabric are unread. | EP16 over two nodes on EFA, NCCL `alltoall`, no overlap of any kind | arXiv 2603.07685 Table 20 and B.2 (`mcore-moe.txt` 2664-2710, guideline at 1645); Megatron-Bridge archive 26.04.01. CORRECTED: the cross-node B200 row is UNVERIFIED, not "DeepEP on InfiniBand" |
| TP | TP1 on all Blackwell MoE rows, TP2 on H100; TP4 only with CP4 at 128K | TP8 with SP on hidden 4096 | Table 20 CONFIRMED; arXiv 2504.14960 rows UNVERIFIED (not on disk) |
| Recompute | None or selective (`mlp`) in the NVIDIA rows; full uniform in veRL DeepSeek-V3 and upstream miles | Full uniform, every layer (about 24% of the modeled step) | Table 11; `kdatp/RESULTS.md` (selective OOM on stage 0 in T2; block recompute of 10 layers -4.1%) |
| Sequences and routing | Table 11 and Megatron-Bridge rows: packed 4K sequences, MBS 1 to 8, force-balanced routing. The band also holds 128K pretraining rows (Qwen3-235B 46%, Mixtral 42.9%, Llama 3 405B 38%) and one RL row (veRL Qwen3-30B-A3B, 31 to 40%, real routing, EP8 in one node) | One 57K to 131K sample per micro-batch, real replayed routing (R3), pad to 4,096 tokens | REFUTED as written: "the 20 to 40% band is pretraining with 4K sequences" |
| Attention accounting and kernels | Counters halve causal attention, count top-k plus shared experts, never count recompute (same convention as our true 1.7%); the models run TE fused dense attention or MLA with CUDA graphs | 34 KDA layers on Triton `fla` and 11 DSA layers on TileLang, plus a kpool indexer and mHC; no CUDA graphs | Megatron-Bridge `flop_utils.py`; MFU study |
| Precision | MXFP8 or FP8 on the top rows (1.2 to 1.3x over BF16); BF16 rows still reach 30 to 34% | BF16 | Table 11 pairs |
| RL peers | veRL DeepSeek-V3 on 96 H20: 19% = 28 TF per GPU; veRL Qwen3-30B-A3B: 31 to 40%; miles Qwen3.5-35B-A3B: 5.0 to 5.8% logged with the inflated dense formula (true value lower); miles DeepSeek-V4-Flash on MI355X: 1.64% | true 1.65 to 1.70% `actor_train` (r45); about 2.2% useful on the T2 kdatp arm | veRL `dpsk.md` CONFIRMED; miles PR #2353 and #2038 not fetched (UNVERIFIED beyond the MFU study) |

- **DeepEP over EFA blocker.** `deep_ep/buffer.py:96-101` in the image sets
  `NVSHMEM_IB_ENABLE_IBGDA=1` for any buffer with more than one node, and
  lines 359 and 420 gate the internode paths on the same test.
  `deep_ep_cpp*.so` holds 29 `ibgda` strings and zero `libfabric` or `efa`
  strings. NVSHMEM itself lists a libfabric transport, but the DeepEP
  kernels do not use it. The hosts have 8 EFA devices at 400 Gb/s and two
  ConnectX-7 ports at 100 Gb/s; `nvidia_peermem` and `gdrdrv` are not
  loaded. CONFIRMED as a code inference; nothing was run. Whether IBGDA
  starts on the two 100 Gb/s ports was not tested; their rejection is a
  bandwidth argument.
- **EP8 memory estimate.** One expert is 25,165,824 parameters. EP8 doubles
  the experts per rank: stage 3 (12 MoE layers) +30.4 GiB (10.1 GiB bf16
  weights + 20.2 GiB fp32 grads), stage 0 +20.2 GiB; host optimizer state
  unchanged at 5.44B parameters per rank (the config comment "~66 GB fp32
  state per rank", `r47/miles-config.yaml:538`). The headroom reference is
  the kdatp T2 arm on r45 data, not an r47 read: DCGM 106.8 / 110.0 / 104.5
  / 108.6 GiB by stage of 178 GiB, 68 GiB free, on 1-minute samples that
  missed a 178 GiB spike in the selective arm. This attempt saw torch
  reserved peaks of 80 GiB (T2 rows); DCGM adds about 29 GiB per GPU for
  NCCL and CUDA workspaces. `RESULTS.md:93` ("one layer without recompute
  costs about 10 GiB per 131K micro-batch") is a per-layer cost, not a
  per-micro-batch transient, and the research note reused it wrongly. EP8
  is likely to fit; it needs the smoke test. Upstream agrees that EP16 is a
  memory choice: `run_glm5_2_744b_a40b.py` says "EP=16 is what fits the
  fp32 optimizer states on 276GB GPUs", with the optimizer on the GPU.
- **Upstream GLM-5.3-Flash recipe.** `scripts/run_glm5_3_flash.py` at
  `4716a367a` hard-codes the shape: 64 GPUs and 32 GPUs TP8 PP4 EP16, 24
  GPUs TP8 PP3 EP8, 8 GPUs TP2 PP2 EP2; full uniform recompute,
  `max-tokens-per-gpu 8192`, no DeepEP, overlap, or VPP flag. It defaults
  to 4 GPUs per node and was validated on 16 nodes x 4 GB300 (PR #2786);
  "NVL72 domain" is an inference no source states. Our config is a near
  copy of that recipe.
- **Megatron 1F1B EP overlap.** `transformer_config.py:3408-3491` requires
  VPP when PP > 1, EP > 1, dispatcher `alltoall` or `flex`, bf16 or fp16,
  `moe` not in `recompute_modules`, shared-expert overlap off. Full
  recompute is allowed. The "2x activation memory" figure belongs to the
  paper's FWD-FWD pattern; Megatron's flag is the FWD-BWD 1F1B pattern the
  paper calls "No additional memory overhead" (extra dispatch and combine
  buffers still apply). `CUDA_DEVICE_MAX_CONNECTIONS > 1` is a Hopper-only
  warning (`arguments.py:1624-1647`, gated on arch < 10). `tp-comm-overlap`
  with variable-length THD is a plausible incompatibility, untested.

### (e) Ranked candidate causes of the unexplained time

Every estimate is modeled unless marked measured. The trace decides.

| Rank | Candidate | Estimate (s per T2 step) | Evidence quality | Precedent |
| --- | --- | --- | --- | --- |
| 1 | Host launch and sync gaps: 17K to 21K launches per stage-3 micro-batch at 10 to 40 us of Python per op; one hardware queue (`CUDA_DEVICE_MAX_CONNECTIONS=1`) | 31 to 125 (model had 17) | Low to medium: the Kineto record count is measured, the records-per-launch ratio is inferred; DCGM "kernel resident 97 to 99%" does not rule it out because the other 15 ranks of an EP group sit in NCCL kernels while one rank is host-bound | Megatron plus TE Python per op of 30 to 80 us is common knowledge, not sourced here |
| 2 | EP all-to-all realized bandwidth over EFA below 35 GB/s per GPU | 97 central; 170 to 230 at 15 to 20 GB/s | Low: no measurement of a 16-rank NCCL all-to-all on this fabric | Every published Blackwell row above 30% with a readable config keeps EP inside NVLink; Megatron-Bridge B200 rows use EP8; veRL DeepSeek-V3 uses EP8 per stage |
| 3 | DSA sparse-MLA backward: fp32 atomics on a 155 MB dKV target, gather repeated on 8 TP ranks | 87 central; up to 203 pessimistic | Low: code confirmed, no timing exists; L2 fit at 64K makes the range two-sided | DeepSeek reference layout runs 64 heads per block (8x fewer bytes per FLOP) |
| 4 | KDA serial chunk recurrence above 8 us per chunk in situ | 54 central; 99 at 15 us | Medium: T7 measured 66 ms per layer at 131K; in-situ contention unknown | Kimi Linear paper publishes relative speedups only |
| 5 | Full recompute | about 83 of 347 modeled (24%); MFU study fit 23% | Medium: fit and model agree | NVIDIA rows: none or selective; T2 selective OOM, block recompute of 10 layers GO after the KDA split |
| 6 | Routing skew: the largest expert group and the busiest all-to-all destination set the critical path | Unknown | Low: the model assumes uniform routing; R3 replays real SGLang routing | Table 11 rows force-balance routing |
| 7 | Allocator events on the 4 to 17 GB fp32 indexer logits per DSA layer, a new length every micro-batch | Unknown | Low: `expandable_segments` is on; not ruled out by the two-point fit (events scale with micro-batch count) | none |
| 8 | TileLang sparse-MLA backward recompile per step | 0 in T2 steady steps (same file, warm); 0 to 360 in r45 step 42 | Measured absent in T2 steps 41 to 42; open for live runs with new lengths each step | MFU study row 3; pod cache holds no backward variant after 33 steps |

## Verdict

The isolation profile did not deliver a trace. Both jobs wrote empty stub
traces because Kineto ran out of its 33 default CUPTI buffers in the
gradient sync of the warmup step, and torch 2.13 then stopped device
collection before the recorded step began. The cause chain is confirmed in
the logs, the torch source, and the Kineto source; the fix is one
environment variable in the harness driver, and it is untested live.

What is measured: on the r47 layout with warm caches, one T2 step of 22.3M
tokens takes 778.6 s of `actor_train` (65.9 us per token per DP replica
plus 7.6 s fixed), the skipped log-prob pass saves about 188 s of
`train_time`, the cold step costs 220 to 270 s more, peak torch memory is
70 to 77 GiB allocated per stage, and one stage-3 micro-batch produces
about 84K GPU records. What is modeled: the time budget in (c) explains 367
s (47%) of the step from confirmed counts and assumed efficiencies; the
other 410 s is UNEXPLAINED and scales with tokens.

Why it is so bad for us compared with others, in plain words: the useful
math of one step fits in 23 s of GPU time, and the layout spends the rest
of the 779 s on work around the math. The expert exchange rides a 50 GB/s
inter-node link six times per MoE layer with nothing scheduled behind it,
while every published Blackwell run above 30% keeps that exchange on
NVLink or overlaps it. TP8 leaves each GPU 8 attention heads, so the sparse
key gather runs 8 times over and the linear-attention state chain runs on
16 to 32 of 148 SMs; published runs use TP1 or TP2 with dense attention on
fused kernels. Full recompute repeats the forward pass; published Blackwell
runs use none or selective. One sample per micro-batch and about 20K
launches per micro-batch make the CPU a suspect the model did not size.
The published runs also pack fixed 4K sequences with balanced routing; we
train one 64K to 131K sample at a time with real routing. The ranking of
these causes is modeled; only the trace measures it.

## Caveats and open items

- **The unexplained 410 s.** The corrected central model reaches 47% of the
  measured step at both sizes. Which of candidates 1 to 4 holds the rest is
  not known. The trace check reads: NCCL all-to-all bin near 200 s means
  communication; `sparse_mla_bwd_kernel` near 150 s or more means the DSA
  backward; kernel-free gaps above 50 s on the slowest rank of each TP
  group mean the host.
- **The record-count datum.** 66K to 85K records per micro-batch per rank
  come from the cold step 0 and include compile-time launches; the split
  into kernels, copies, and API calls is inferred (4 to 5 records per
  launch). Size the next profile with 85K per micro-batch.
- **Realized EFA all-to-all bandwidth** for a 16-rank NCCL all-to-all with
  33 MB per peer is assumed at 70% of line rate. No measurement exists.
- **DSA kernel time was never isolated.** No T7-style microbenchmark exists
  for the sparse-MLA forward or backward or the indexer. The 87 s family is
  the widest range in the model.
- **Whether TileLang recompiles the backward per step** in a live run
  (new padded lengths every step) is open. The T2 steady steps reuse one
  file and cannot show it; the pod cache holds no backward variant.
- **Stage-0 idle share (24%)** is model output; no per-stage timer exists.
- **The 8 us per KDA chunk** is a fit to T7, not a measurement.
- **The CUPTI queue rule** behind the 33-buffer burst (per stream or per
  thread) is UNVERIFIED; it does not change the fix.
- **Trace export size and time** on 64 ranks with `record_shapes`,
  `with_stack`, `profile_memory`, and `with_flops` on: several GB of JSON
  per rank and minutes per rank are estimates. miles has no per-rank filter
  for `train_overall`.
- **EP8 headroom** rests on 1-minute DCGM samples of a different arm and
  needs a smoke test. EP8 plus block recompute of 10 layers does not fit by
  the same arithmetic.
- The Kineto commit inside the wheel is not recorded in the wheel; the torch
  v2.13.0 submodule pointer is the evidence. The live proof is the
  `Max GPU buffer size: 4096MB` line on the next run.
- arXiv 2504.14960 and miles PR #2353 and #2038 were not read for this
  study; their rows stand on the MFU study's citation.

## Next step

One action, then one more, each on a new explicit go from the user.

1. **One profiling job with the config-only Kineto fix.** In
   `kdatp-prof-run.sh`, next to the `KINETO_LOG_LEVEL` export and above the
   `REPLICA_IDX` branch (head and workers both), write `/tmp/kineto.conf`
   with `ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB=4096` and export
   `KINETO_CONFIG=/tmp/kineto.conf` (the pod `/tmp` is the overlay disk, and
   the file must exist on every pod). Keep `profile_step_start 1`,
   `profile_step_end 2`, `PROF_GROUPS 239,258`. 4096 MB gives 1,025
   buffers: about 10x the two-step volume and 31x the observed burst.
   Success signals in `trainer-0.log`: `Max GPU buffer size: 4096MB` on 64
   ranks, zero `Exceeded max GPU buffer count`, zero `stopped early`, trace
   files far above 100 KB. If the first line still reads `128MB`, stop the
   Ray job and fall back to `profile_step_start 0`, `profile_step_end 1`
   (partial cold step 0). Needs from the user: a new explicit go for this
   ONE job on 8 nodes (about 25 min plus the trace export), the same rules
   as before (delete the job when done, no live run touched), and a
   decision on the export cost if the 64-rank JSON takes long.
2. **Separately, an EP8 config-only T2 arm on the kdatp harness** as the
   first layout test: `expert_model_parallel_size 8`, everything else as the
   kdatp arm. Read `actor_train` per token against 778.6 s and the DCGM peak
   against 178 GiB. Precedent: the Megatron-Bridge 26.04.01 B200 rows
   (GPT-OSS-120B, Qwen3-30B-A3B, Qwen3-235B) all run EP8 inside the NVL8
   node, and the veRL DeepSeek-V3 recipe keeps EP8 per stage. Needs from
   the user: a separate go, after the profile, because the profile decides
   whether the all-to-all is the family to attack first.

## Actions taken

- Harness `examples/arena/harbor-rl-glm53-flash/kdatp/prof/` on miles
  `arpit-glm-53`: `e7a0d35ba` (the one-step profiled train job),
  `82e080ec3` (the 2-group slice and `KINETO_LOG_LEVEL`), `a441a1031` (the
  status note; section (b) refutes both of its ways out, the one-line
  `custom_profiler_config` fix and the "slice under 15 micro-batches fits"
  claim).
- Jobs `kdatp-prof-20260929a` (08:01 to 09:00Z) and `kdatp-prof-20260929ab`
  (09:08 to 09:31Z) on `arena-prod-bom-v2`, namespace `arena-tasks`, 8
  nodes each, both `rc 0`, both deleted. No image built. No live run
  touched. Read-only reads from the r47 pod only.
- Scratch: `/workplace/guparpit/kdfast/scratch/prof/20260929a/`
  (`run-20260929a/`, `run-20260929ab/`, `research/`, `verify/`, `traces/`,
  `extract_log.py`, `t2-manifest.json`, the two rendered job YAMLs).
- S3: `s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/prof/20260929a/`
  and `.../20260929ab/` (`harness/`, `arms/`, `driver.log`,
  `kdatp-prof/trainer-0.log`, `tb/`, `results.json`);
  `.../guparpit/kdatp/data/prof-239-258/` (`rollout_40.pt`, sha256
  `414454df...`, links 41 to 43, `manifest.json`).
- This record and the `## Follow-up` note in
  `studies/mfu-glm53-flash-rl/STUDY.md`. No change to the harness driver
  yet: the `KINETO_CONFIG` edit waits for the go in Next step 1.

## Sources

- Workflow journal `wf_269fcf06-9bc` (2026-09-29): research reports
  `first-principles.md`, `kda-dsa-kernels.md`, `how-others-do-it.md`
  (with `mcore-moe.txt`, the text of arXiv 2603.07685); verify reports
  `time-model.md`, `peers.md`, `profile-next.md`; `model.py` and
  `verify/model_verify.py`; `verify/records-by-stage.txt`,
  `verify/rankmap-ab.txt`, `verify/podsrc/`.
- Profile logs: `run-20260929a/kdatp-prof/trainer-0.log` (64 `stopped early
  at step 1`, no Kineto lines), `run-20260929ab/kdatp-prof/trainer-0.log`
  (`Max GPU buffer size: 128MB` x 64; `Exceeded max GPU buffer count (33 >=
  33)` first at 09:23:37 on all ranks; `Processed N GPU records` 2,189,091
  to 2,796,315; `num_microbatches=[33]`; `Timer actor_train` 395.4 / 184.8 /
  184.6 s on rank 0); `results.json` of both runs; `driver.log`
  (`image_head=4716a367a...`, `kcache ... tilelang_entries=2
  triton_entries=2549`, data build `rows 66, tokens 4721494, max_length
  109166`); `tb/listing.txt` (stubs 13 to 33 KB).
- Runtime code, read-only from pod `rl-glm53f47-wxj87-trainer-worker-0`
  (miles `4716a367a`, image `miles-glm53-r17-20260928a`): Megatron-Core
  `token_dispatcher.py:517,612,709-760,864-905,990`, `experts.py:791,969`,
  `parallel_state.py:792-798`, `tensor_parallel/mappings.py:426-460`,
  `transformer_config.py:2087-2093,3176-3182,3408-3491`,
  `schedules.py:2244-2246`, `training/arguments.py:1077-1085,1542-1545,1624-1647`,
  `training/initialize.py:216-219`; torch `profiler/profiler.py:1067-1071,1207-1222`,
  `lib/libtorch_cpu.so` strings; fla 0.4.2 `ops/common/chunk_delta_h.py:23-32,131,680,691`,
  `ops/kda/chunk.py:41,49`, `chunk_fwd.py:93`, `chunk_bwd.py:462,514`,
  `ops/utils/index.py:121`; `deep_ep/buffer.py:96-101,359,420`,
  `deep_ep_cpp*.so` and `libnvshmem_host.so.3` strings;
  `GLM-5.3-Flash-BF16/config.json`; the actor environment
  (`CUDA_DEVICE_MAX_CONNECTIONS=1`, `NCCL_PROTO=simple`, `FI_PROVIDER=efa`);
  `/sys/class/infiniband`, `lsmod`, `/proc/<pid>/cmdline`.
- miles `4716a367a` (`/workplace/guparpit/kdfast/miles-kdatp-rel`):
  `miles/utils/profile_utils.py`, `miles/backends/megatron_utils/actor.py:327,637`,
  `miles_plugins/arena/run_arena_harbor.py:315-325`,
  `miles/ray/train/actor_factory.py:85`, `miles/utils/data.py:277`,
  `miles/utils/arguments.py:2766-2772`, `miles_plugins/models/hf_attention.py:280`,
  `glm5_next/kda.py`, `dsa.py:60-150`, `glm5_next.py:42,95`,
  `glm5/ops/tilelang_sparse_mla_fwd.py:40-56`, `tilelang_sparse_mla_bwd.py:87-119,301`,
  `tilelang_indexer_fwd.py:27-28`, `glm5_next/ops/kpool_indexer.py`,
  `scripts/run_glm5_3_flash.py:44-52,82-125`, `scripts/run_glm5_2_744b_a40b.py:204`,
  `docs/models/glm/glm5-3-flash.md:99-100`.
- Kineto at `094d3c1d072362d0a919a77299459eee94f97931` (torch v2.13.0
  `third_party/kineto`): `ActivityProfilerProxy::prepareTrace`,
  `ConfigLoader::getConfString`, `Config::handleOption`,
  `GenericActivityProfiler::configure`, `CuptiActivityApi::setMaxBufferSize`,
  `bufferRequested`, `bufferCompleted`, `AbstractConfig::parse`; torch
  `torch/csrc/profiler/kineto_shim.cpp` `appendCustomConfig`.
- Records: `training-runs/studies/mfu-glm53-flash-rl/STUDY.md`;
  `training-runs/harbor-rl-glm53-flash/r47/RECORD.md` (image digests) and
  `r47/miles-config.yaml:444,448,527,538,540,542`;
  `examples/arena/harbor-rl-glm53-flash/kdatp/RESULTS.md` (T2 arms lines
  49-98, T7 KDA timings lines 164-170); `kdatp/prof/README.md`; ADR-0016.
- External: arXiv 2603.07685v2 (Table 11, Table 20, Appendix B.2; text
  lines 797, 802-819, 881-895, 1355-1361, 1645, 1846, 1869, 2664-2710),
  arXiv 2504.14960 (not read here), arXiv 2505.11432, arXiv 2510.26692 (Kimi
  Linear), Megatron-Bridge `docs/performance-summary-archive.md` release
  26.04.01 and `flop_utils.py`, veRL `docs/perf/dpsk.md` and
  `examples/grpo_trainer/run_deepseek_v3_671b_megatron.sh`, radixark/miles
  PR #2353, #2038, #2786, #1825, #3609 (the first two not fetched), DeepEP
  README, AWS p6-b200 instance page.
