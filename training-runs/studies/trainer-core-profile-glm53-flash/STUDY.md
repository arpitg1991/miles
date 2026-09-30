# Study: Trainer core profile of GLM-5.3-Flash RL training (r47 layout)

**Date:** 2026-09-29; updated 2026-09-30 with the results of round 20260929c (jobs `kdatp-prof-20260929c`, `-d` and `-e`)
**Status:** Open. The trace of round 20260929c answers where the time goes and what a profile needs. The choice of levers stays open, because every proposed job and live change waits for a user go
**Question:** Where does the forward and backward time of one GLM-5.3-Flash training step go on the r47 layout, why does the trainer run 10 to 20x below published MoE training efficiency, and what does a profile need in order to answer this?
**Runs and data used:** profile jobs `kdatp-prof-20260929a` (T2 rows: 314 rows, 22.3M tokens) and `kdatp-prof-20260929ab` (groups 239 and 258 of r45 rollout 43: 66 rows, 4.72M tokens), both 3 steps from the r43 `iter_0000039` seed, both `rc 0`, both deleted; outputs under `s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/prof/20260929a/` and `.../20260929ab/` (`driver.log`, `kdatp-prof/trainer-0.log`, `results.json`, `tb/`); slice data `.../kdatp/data/prof-239-258/`; kdatp T2 and T7 results (`kdatp/RESULTS.md`); the r47 pod `rl-glm53f47-wxj87-trainer-worker-0` (read-only, runtime code and `config.json`); no live run was touched. Round 20260929c: the profile job `kdatp-prof-20260929c` (2-group slice, profiled step 1, all-to-all bench) and the arms jobs `kdatp-prof-20260929d` (`ep8`, `base`, `pack`) and `kdatp-prof-20260929e` (`ep8pack`, `tp4ep8pack`) on the T2 rows with `replay_rollout: 40`, all `rc 0`, all removed by the cluster TTL at success; outputs under `s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/prof/20260929{c,d,e}/`; read-only evidence from the live run r47 `rl-glm53f47-wxj87` (W&B `arena/rl-glm53f-adebt-766/0ix3m75e`, Ray actor logs, NATS, DCGM). All these jobs are train-only timing tests on replayed T2 rows, not training runs. The agentic-debt dataset of record is `lakefs://arena-inspect/main/internal/agentic-debt-r3/agentic-debt-766/`
**Code SHAs:** miles `4716a367a` (trainer image `miles-glm53-r17-20260928a`, digest `sha256:e6f04a9ca1abc9df17a7bbf9e3d2aaf6a643f40ed5a457c98a4114719972bcd0`): `miles/utils/profile_utils.py`, `miles_plugins/models/glm5_next/{kda.py,dsa.py,glm5_next.py}`, `miles_plugins/models/hf_attention.py`, `glm5/ops/tilelang_sparse_mla_{fwd,bwd}.py`, `glm5_next/ops/kpool_indexer.py`; harness commits `e7a0d35ba`, `82e080ec3`, `a441a1031` (`examples/arena/harbor-rl-glm53-flash/kdatp/prof/`); in the image: Megatron-Core `0.19.0+e8f574511`, TE 2.17.0, torch `2.13.0+cu130` (Kineto at `094d3c1d`), fla 0.4.2, tilelang 0.1.9, NCCL 2.29.7, aws-ofi-nccl 1.18.0; round 20260929c harness `dabe8a8972` (multi-arm driver, `KINETO_CONFIG` fix, `a2a_bench.py`), driver fix `27ab1bccdd`, README update `c2e17e5870`
**Workflow ids:** `wf_269fcf06-9bc` (profile attempt, three research reports, three adversarial verify reports) and the synthesis run of 2026-09-29 that wrote this record; round 20260929c: 8 round reports (`reports/`), 4 verifier reports (`verify2/`) and `reports/synthesis.md` under `/workplace/guparpit/kdfast/scratch/prof/20260929c/` (the reports hold no workflow id)

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

Terms of round 20260929c. The **T2 step** is one `actor_train` step on the
replayed T2 rows; the control is 778.6 s. **pp0** to **pp3** are the four
pipeline stages; pp3 is the critical stage. **F**, **R** and **B** are the
forward, the recompute and the backward pass of a micro-batch. The
**coupling wait** is the time that one DP replica waits for the other inside
an EP16 all-to-all. **Hard slack** is the GPU memory that is left after the
torch allocated peak plus the memory outside torch. The **loop step** is one
full RL step of r47 (`perf/step_time`). **MEASURED** is a number that a
verifier CONFIRMED or CORRECTED. **MODEL** and **ESTIMATE** are arithmetic
from measured parts. The **pair-min** time of an all-to-all call is the
shorter of its two durations on the DP0 and DP1 ranks; it estimates the
transfer time.

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
- **Round 20260929c (2026-09-29 20:25Z to 2026-09-30 01:45Z).** The harness
  `dabe8a8972` runs several arms in one Ray cluster, writes the
  `KINETO_CONFIG` file of (b) item 6, and has an optional all-to-all bench
  (`a2a_bench.py`). Job `kdatp-prof-20260929c` recorded step 1 of the
  2-group slice on all 64 ranks and ran the bench. Jobs
  `kdatp-prof-20260929d` and `kdatp-prof-20260929e` ran five layout arms on
  the T2 rows with the profiler off. A trace analyst read 8 of the 64
  traces (`analysis/trace/`). Read-only tracks read the live run r47
  (`reports/live.md`), the upstream precedent (`reports/precedent.md`), and
  the memory and time fit of each arm (`reports/feasibility.md`). A second
  adversarial pass checked four claim clusters (`verify2/ep8.md`,
  `verify2/tp4.md`, `verify2/dsa.md`, `verify2/loop.md`), and
  `reports/synthesis.md` ranks the path. Sections (f) to (m) apply every
  CORRECTED and REFUTED verdict of that pass. Scratch:
  `/workplace/guparpit/kdfast/scratch/prof/20260929c/` (`reports/`, `run/`,
  `analysis/`, `arms-20260929d/`, `arms-20260929e/`, `live/`, `verify2/`,
  `precedent/`). Not in git. Paths in sections (f) to (m) are relative to
  that directory unless they name a repo.

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
  **Trace result (section (g)): REFUTED.** The warm trace counts 12,544
  kernels per pp3 micro-batch (`reports/trace-analysis.md` section 6.2),
  and the GPU idles only 1.2 to 1.7% of the step inside the pipeline.

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
   from the `DEVICE_STOPPED` override, `profiler.py:1217`). **Tested live on
   2026-09-29 in job `kdatp-prof-20260929c`: the fix works (section (f)).**

### (c) First-principles time budget, T2 shape (seconds per step)

Stage 3 is the critical path (9 KDA, 3 DSA, 12 MoE layers, LM head). The
published model gave 347 s. The verify pass corrected five inputs: 156
micro-batches (not 157), `E[s^2]` 6.55e9 (not 5.5e9), expert tokens per
rank per MoE layer call `(s_A + s_B)/2`, about `s`, because the EP16 group
serves both DP replicas (the model had `s/2`; expert GEMM doubles to
27.4 s), sparse index width 2112 (not 2048), and 15 TP units per KDA layer
(not 14). Corrected central: 367 s.

**Trace verdict (2026-09-30).** The trace of round 20260929c (section (g))
closes the gap and REFUTES the size of five rows: DSA sparse MLA (87 s
central, 203 s high), DSA indexer, KDA, host launch (the launch-bound case),
and the EFA 15 to 20 GB/s case of the all-to-all. The last column gives the
measured value, scaled to the T2 step.

| Family | s per step | Basis | Confidence | Trace, T2-scaled (section (g)) |
| --- | --- | --- | --- | --- |
| EP all-to-all over EFA | 96.7 | 12 MoE layers x 6 large calls (9 NCCL kernels with the probs) per micro-batch; `s x 8192 B` per rank per call; 8 of 16 destinations on the other node (`parallel_state.py:792-798`, rank map from the log); EFA at 35 GB/s per GPU = 70% of 400 Gbps | Counts CONFIRMED; realized EFA bandwidth UNVERIFIED (the widest lever: at 15 to 20 GB/s the family is 170 to 230 s) | 125.1 s (1.3x). The transfer runs at about the EFA line rate (about 50 GB/s per GPU across nodes); 28% of the time is DP coupling wait. The 15 to 20 GB/s case (170 to 230 s) is REFUTED |
| DSA sparse MLA (fwd, recompute, bwd) | 87.3 | one block per query gathers 2112 x 576 x 2 B = 2.4 MB of KV; repeated on all 8 TP ranks; 8 heads padded to 16; bwd gathers again and adds dKV with fp32 atomics; gather at 60% of HBM, bwd 3x fwd | Structure CONFIRMED (`dsa.py:90-98`, `tilelang_sparse_mla_bwd.py`); efficiency UNVERIFIED (the KV of one 64K row, 75 MB, fits the 126 MB L2, so the HBM framing does not hold there); 5x bwd and 35% gather give 203 s | 375.8 s (4.3x): bwd 269.2, fwd F+R 106.5. REFUTED: the 87 s central and the 203 s high case are both too low |
| KDA chunk kernels | 53.6 | 4 serial passes per layer per micro-batch (`chunk_fwd.py:93`, `chunk_bwd.py:462,514`) over s/64 chunks at 8 us per chunk; grid 16 to 32 blocks on 148 SMs (`chunk_delta_h.py:691`) | Passes and grid CONFIRMED; 8 us per chunk UNVERIFIED (a fit to T7: one 8-head layer at 131K fwd+bwd = 66 ms; within 8% of the T7-anchored note); 15 us gives 99 s | 25.8 s (0.48x). The state chain runs 1.3 us per 64-token chunk, not 8 us. REFUTED (smaller) |
| TP/SP collectives | 29.0 | 221 units of [s, 4096] bf16 over TP8 on stage 3 (15 per KDA layer, 4 of them separate backward all-reduces of q, k, v, b) at 650 GB/s | Medium; bus bandwidth assumed | 37.2 s (1.3x) |
| Expert grouped GEMM | 27.4 | about `s` routed tokens per rank per layer (3,640 per expert at 64K), 18 groups, 4 passes at 45% | Medium (CORRECTED 2x) | 21.2 s (0.77x) |
| Dense TP GEMMs | 16.8 | 496 MFLOP per token per rank x 4 passes at 60% (LM head counted below) | Medium | 49.4 s with the mHC projections, bmm, the LM head and the sm80 and fp32 fallback kernels (2.3x against 16.8 + 4.4) |
| Launch and host syncs | 16.8 | 6,000 kernels x 10 us + 2 dispatcher syncs per MoE layer x 2 ms; the Kineto datum implies 17K to 21K launches, so 31 s at 10 us and 125 s at 40 us; `CUDA_DEVICE_MAX_CONNECTIONS=1` makes any stream's host stall a GPU stall | Low; under-sized 3 to 4x (the one family the datum contradicts) | 13.8 s of host idle inside the pipeline (1.2 to 1.7% of the step on every stage); 12,544 kernels per pp3 micro-batch. REFUTED: the step is not launch-bound (31 to 125 s case) |
| DSA indexer and top-k | 11.2 | `1024 x s^2` FLOP at 50%, 2 passes, on every TP rank; fp32 logits 4.3 to 17 GB per layer | Medium | 55.7 s by the trace scale; CORRECTED to 66.7 s with the T2 E[s^2] of 6.55e9 (`verify2/dsa.md` 2.4). REFUTED (6x) |
| 1F1B bubble, fixed per step, LM head, MoE glue, mHC, PP send/recv, DSA glue | 27.7 | bubble `3/156 = 1.9%`; fixed 5 s; LM head 4.4 s | Medium; the measured fit gives bubble plus fixed about 22 s | PP fill, steady and drain 21.2 s; CPU Adam and offload copies 18.6 s; mHC, permute, elementwise and copies 38.8 s; EP-group all-gather 13.0 s |
| **Total, corrected central** | **367** | **47% of the measured 778.6 s** | | **795.6 s** (about 807 s with the corrected DSA scale of `verify2/dsa.md` 2.4, record-keeper arithmetic) |
| Measured T2 `actor_train`, step 42 | 778.6 | `results.json` | Measured | 778.6 s |
| **Gap** | **about 410 s** | 434 s against the published 347 s | **UNEXPLAINED pending a trace** | **Closed**: the trace budget sums to the wall time on all 8 traced ranks |

- The gap is the same 2.1x at both sizes (slice: 87 s modeled against
  182.3 s measured), so it is per-token work inside the micro-batches.
- The "all pessimistic" variant of the published model (699 s) is not a
  bound: with the expert fix it is 729 s, and with the measured kernel count
  (`KERNEL_MULT 4`) it is 843 s, above 781 s. The family that is under-sized
  is the host launch count, and it was already in the model.
- The whole T2 step at peak is 23 s of useful math (25 s with the replicated
  indexer). GEMMs are 6 to 13% of the modeled step and 3 to 6% of the
  measured step. The recompute share of the modeled step is about 24%.
  **Trace:** the recompute pass R is 19.9% of the pp3 step.
- Stage 0 idles about 24% of the step in the model. UNVERIFIED: no
  per-stage timer exists. **Trace:** pp0 waits 58.2 s of 186.9 s (31%) in
  PP receives. This wait is not on the critical path.
- The ranking (all-to-all, sparse MLA, KDA chunk, then TP collectives) holds
  through every correction. **REFUTED by the trace:** DSA (about 443 s) >
  EP all-to-all (125 s) > GEMM (71 s) > TP (37 s) > KDA (26 s).

### (d) How the published 20 to 40% runs differ

| Property | Published rows | r47 | Source and status |
| --- | --- | --- | --- |
| EP placement | Every Blackwell row above 30% with a readable config keeps EP x TP inside the NVLink domain (GB200 and GB300 NVL72 at EP32 or EP64; B200 NVL8 rows at EP8: GPT-OSS-120B, Qwen3-30B-A3B, Qwen3-235B). H100 cross-node rows use DeepEP plus 1F1B overlap on InfiniBand. One B200 row crosses nodes (DeepSeek-V3, EP32 over 4 nodes, 38.4%) and its dispatcher, overlap, and fabric are unread. | EP16 over two nodes on EFA, NCCL `alltoall`, no overlap of any kind | arXiv 2603.07685 Table 20 and B.2 (`mcore-moe.txt` 2664-2710, guideline at 1645); Megatron-Bridge archive 26.04.01. CORRECTED: the cross-node B200 row is UNVERIFIED, not "DeepEP on InfiniBand" |
| TP | TP1 on all Blackwell MoE rows, TP2 on H100; TP4 only with CP4 at 128K | TP8 with SP on hidden 4096 | Table 20 CONFIRMED; arXiv 2504.14960 rows UNVERIFIED (not on disk) |
| Recompute | None or selective (`mlp`) in the NVIDIA rows; full uniform in veRL DeepSeek-V3 and upstream miles | Full uniform, every layer (about 24% of the modeled step) | Table 11; `kdatp/RESULTS.md` (selective OOM on stage 0 in T2; block recompute of 10 layers -4.1%) |
| Sequences and routing | Table 11 and Megatron-Bridge rows: packed 4K sequences, MBS 1 to 8, force-balanced routing. The band also holds 128K pretraining rows (Qwen3-235B 46%, Mixtral 42.9%, Llama 3 405B 38%) and one RL row (veRL Qwen3-30B-A3B, 31 to 40%, real routing, EP8 in one node) | One 57K to 131K sample per micro-batch, real replayed routing (R3), pad to 4,096 tokens | REFUTED as written: "the 20 to 40% band is pretraining with 4K sequences". Packing measured in section (i): +0.2% at EP16 and -0.3% at EP8, so no gain |
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
  **Measured in job `kdatp-prof-20260929d` (section (i)):** allocated
  +20.25 / +27.84 / +27.84 / +30.38 GiB by stage, torch reserved max
  109.73 GiB. EP8 fits in the harness.
- **Upstream GLM-5.3-Flash recipe.** `scripts/run_glm5_3_flash.py` at
  `4716a367a` hard-codes the shape: 64 GPUs and 32 GPUs TP8 PP4 EP16, 24
  GPUs TP8 PP3 EP8, 8 GPUs TP2 PP2 EP2; full uniform recompute,
  `max-tokens-per-gpu 8192`, no DeepEP, overlap, or VPP flag. It defaults
  to 4 GPUs per node and was validated on 16 nodes x 4 GB300 (PR #2786);
  "NVL72 domain" is an inference no source states. Our config is a near
  copy of that recipe. **CORRECTED (`verify2/ep8.md` section 4):** upstream
  main `scripts/run_glm5_3_flash.py` l.75-86 now sets EP = GPUs / 4 (added
  in `cc76e23915`). So 64 GPUs use EP16 and 32 GPUs use EP8, both with
  expert DP 1. The `4716a367a` copy lists EP16 for 32 GPUs, which cannot
  run. No upstream GLM-5.3-Flash layout uses EP8 at 64 GPUs.
- **Megatron 1F1B EP overlap.** `transformer_config.py:3408-3491` requires
  VPP when PP > 1, EP > 1, dispatcher `alltoall` or `flex`, bf16 or fp16,
  `moe` not in `recompute_modules`, shared-expert overlap off. Full
  recompute is allowed. The "2x activation memory" figure belongs to the
  paper's FWD-FWD pattern; Megatron's flag is the FWD-BWD 1F1B pattern the
  paper calls "No additional memory overhead" (extra dispatch and combine
  buffers still apply). `CUDA_DEVICE_MAX_CONNECTIONS > 1` is a Hopper-only
  warning (`arguments.py:1624-1647`, gated on arch < 10). `tp-comm-overlap`
  with variable-length THD is a plausible incompatibility, untested.
  **Round 20260929c:** the image Megatron rejects VPP with the 11/11/11/12
  split (`transformer_config.py:2566-2598`). No RL framework sets the
  overlap flag (`reports/precedent.md` B.1). After EP8, at most about 23 s
  per T2 step is left to hide (`reports/trace-analysis.md` section 10).
  Parked.

### (e) Ranked candidate causes of the unexplained time

Every estimate is modeled unless marked measured. The trace decides. The
last column holds the trace result of round 20260929c (section (g)).

| Rank | Candidate | Estimate (s per T2 step) | Evidence quality | Precedent | Trace result (2026-09-30) |
| --- | --- | --- | --- | --- | --- |
| 1 | Host launch and sync gaps: 17K to 21K launches per stage-3 micro-batch at 10 to 40 us of Python per op; one hardware queue (`CUDA_DEVICE_MAX_CONNECTIONS=1`) | 31 to 125 (model had 17) | Low to medium: the Kineto record count is measured, the records-per-launch ratio is inferred; DCGM "kernel resident 97 to 99%" does not rule it out because the other 15 ranks of an EP group sit in NCCL kernels while one rank is host-bound | Megatron plus TE Python per op of 30 to 80 us is common knowledge, not sourced here | REFUTED. Pipeline idle 1.2 to 1.7% per stage (13.8 s at T2); 12,544 kernels per pp3 micro-batch; the host waits for the GPU (rank 48 spends 123 s in `cudaStreamSynchronize`) |
| 2 | EP all-to-all realized bandwidth over EFA below 35 GB/s per GPU | 97 central; 170 to 230 at 15 to 20 GB/s | Low: no measurement of a 16-rank NCCL all-to-all on this fabric | Every published Blackwell row above 30% with a readable config keeps EP inside NVLink; Megatron-Bridge B200 rows use EP8; veRL DeepSeek-V3 uses EP8 per stage | REFUTED as a bandwidth problem. The transfer runs at about the EFA line rate. The family is 125.1 s at T2, 100% exposed, 28% coupling wait; EP8 removes it (section (i)) |
| 3 | DSA sparse-MLA backward: fp32 atomics on a 155 MB dKV target, gather repeated on 8 TP ranks | 87 central; up to 203 pessimistic | Low: code confirmed, no timing exists; L2 fit at 64K makes the range two-sided | DeepSeek reference layout runs 64 heads per block (8x fewer bytes per FLOP) | CONFIRMED and larger. bwd 269 s at T2; all DSA about 443 s (57%). Each block computes 16 head rows for 8 real heads |
| 4 | KDA serial chunk recurrence above 8 us per chunk in situ | 54 central; 99 at 15 us | Medium: T7 measured 66 ms per layer at 131K; in-situ contention unknown | Kimi Linear paper publishes relative speedups only | REFUTED. 1.3 us per chunk; KDA 25.8 s at T2 |
| 5 | Full recompute | about 83 of 347 modeled (24%); MFU study fit 23% | Medium: fit and model agree | NVIDIA rows: none or selective; T2 selective OOM, block recompute of 10 layers GO after the KDA split | MEASURED. R is 37.6 s of the 189.4 s pp3 step (19.9%); the DSA core is 48% of R |
| 6 | Routing skew: the largest expert group and the busiest all-to-all destination set the critical path | Unknown | Low: the model assumes uniform routing; R3 replays real SGLang routing | Table 11 rows force-balance routing | MEASURED in part. The largest receive from one peer is 3.1x the mean, but the aggregate runs at line rate; the DP replicas wait for each other (coupling wait 28%) |
| 7 | Allocator events on the 4 to 17 GB fp32 indexer logits per DSA layer, a new length every micro-batch | Unknown | Low: `expandable_segments` is on; not ruled out by the two-point fit (events scale with micro-batch count) | none | Not a visible cost. All host gaps inside the pipeline sum to 2.3 to 3.3 s per stage |
| 8 | TileLang sparse-MLA backward recompile per step | 0 in T2 steady steps (same file, warm); 0 to 360 in r45 step 42 | Measured absent in T2 steps 41 to 42; open for live runs with new lengths each step | MFU study row 3; pod cache holds no backward variant after 33 steps | MEASURED. Live r47: a new padded length in 12 of steps 3-38 (11-15 s per rank), none after step 38 (`reports/live.md` section 3). Harness: 576 compiles in the `ep8pack` cold step, 768 in `tp4ep8pack` |

### (f) Round 20260929c: the profile job with the Kineto fix

Job `kdatp-prof-20260929c` ran the 2-group slice (groups 239,258: 66 rows,
4,721,494 tokens, 33 micro-batches per DP rank) on the r47 layout with the
fix of (b) item 6, after the all-to-all bench of (h). It ran on 8 nodes from
20:29Z to 21:00Z on 2026-09-29 and ended `rc 0`. Step 0 (rollout 40) warmed
up, step 1 (rollout 41) was recorded on all 64 ranks, and step 2 (rollout
42) ran clean. Source: `reports/prof-job.md`; `run/trainer-0.log`,
`run/results.json`, `run/driver.log`, `run/tb-sizes.txt`.

| Signal | Value | Want |
| --- | --- | --- |
| `Max GPU buffer size: 4096MB` | 64 ranks (0 at `128MB`) | 64 |
| `Exceeded max GPU buffer count` | 0 | 0 |
| torch warning `Device profiling activity collection was stopped early` | 0 (64 in each of runs `20260929a` and `20260929ab`) | 0 |
| `Processed N GPU records` | 7.36M to 8.93M per rank | 2M or more |
| Traces | 64 files, 393.5 to 460.0 MB each (gz), 27.21 GB in total; the 8 local copies hold 5.7 to 6.7 GB each after `gunzip` | 64, each above 1 MB |
| `actor_train`, steps 0 / 1 / 2 | 404.8 (cold) / 189.4 (recorded) / 186.3 s | - |
| Profiler cost in the step | at most +3.0 s (1.6%). Run `20260929ab` shows +3.9 s between the same steps with no trace, so the true cost is near 0 (UNVERIFIED) | - |
| Trace export after the step | 338 s on rank 0; 405 s on the slowest rank (`train_time` 594.4 s minus `actor_train` 189.4 s) | - |
| Host memory during the recorded step | about +30 GiB RSS per trainer rank (node 0: 113.7 to 145.0 GiB per rank); pod limit 1,800 Gi | - |
| GPU memory | the same as run `20260929ab` within 0.04 GiB | - |

- The driver of `dabe8a8972` logged `64 stopped early`. That count is false.
  It matched the Kineto summary line `GPU stopped early? = 0`. That line
  also read 0 on all ranks of the cut run `20260929ab`, so it is not a stop
  signal. Fix `27ab1bccdd` counts only the torch warning (Actions taken).
- The cluster adds `ttlSecondsAfterFinished: 0` to the job. The operator
  removed the job and its pods within seconds of success, so `kubectl logs`
  was lost. All outputs were on the scratch mount.
- The fallback job `kdatp-prof-20260929c2` did not run.
- A traced full T2 step (156 micro-batches) holds about 4.7x the records.
  Its host memory can pass the pod limit (UNVERIFIED). Keep a profiled arm
  on the slice.

### (g) Measured time budget from the trace

Method (`reports/trace-analysis.md` sections 1 to 3; scripts and outputs in
`analysis/trace/`): the analyst read 8 of the 64 traces, TP rank 0 of each
stage and DP replica (ranks 0, 8, 16, 24, 32, 40, 48, 56). Each instant of
the step goes to one owner: a compute kernel wins over a copy, a copy wins
over NCCL, and among NCCL kernels the EP all-to-all wins over TP, DP and PP.
An instant with no kernel is idle. So the budget of a rank sums to its wall
time. The `train_one_step` window on ranks 48 and 56 is 189.366 s, equal to
`perf/actor_train_time` of step 1 (189.37 s). The T2 scale is x 4.72 for
token-linear families (11.5M T2 tokens per DP rank against 2,437,120),
x 4.06 for the indexer and top-k (s^2), x 4.73 for per-micro-batch
families (156 / 33), and x 1 for the fill, the drain and the optimizer
(`analysis/trace/levers.json`).

Budget of the critical stage pp3 (mean of the DP pair; the slice step is
189.37 s), with the T2 scale and the old model of (c):

| Family | pp3 slice (s) | Share of 189.4 s | T2-scaled (s) | Share of 778.6 s | Old model (c) (s) | Verdict on the old row |
| --- | --- | --- | --- | --- | --- | --- |
| DSA sparse-MLA backward | 57.06 | 30.1% | 269.2 | 34.6% | 87.3 for the sparse MLA fwd, R and bwd | REFUTED: the sparse MLA is 375.8 s, 4.3x |
| DSA sparse-MLA forward, F and R | 22.57 | 11.9% | 106.5 | 13.7% | (in the row above) | |
| DSA indexer and `clean_logits`, F and R | 9.62 | 5.1% | 39.1 (CORRECTED 46.8) | 5.0% (6.0%) | 11.2 with the top-k | REFUTED: indexer plus top-k 66.7 s (CORRECTED), 6x |
| DSA top-k | 4.09 | 2.2% | 16.6 (CORRECTED 19.9) | 2.1% (2.6%) | (in the row above) | |
| EP all-to-all (tokens bf16 and probs fp32) | 26.47 | 14.0% | 125.1 | 16.1% | 96.7 | 1.3x; the 15 to 20 GB/s case is REFUTED, see (h) |
| EP-group all-gather (dispatcher counts) | 2.74 | 1.4% | 13.0 | 1.7% | not modeled | |
| TP/SP collectives | 7.88 | 4.2% | 37.2 | 4.8% | 29.0 | 1.3x |
| GEMM, dense and other | 10.47 | 5.5% | 49.4 | 6.3% | 16.8 + 4.4 (LM head) | 2.3x. The sm80 WMMA and fp32 SIMT fallback kernels take 5.8 s of the pp3 slice step |
| GEMM, expert (grouped) | 4.50 | 2.4% | 21.2 | 2.7% | 27.4 | 0.77x |
| KDA (state chain and other `fla` kernels) | 5.46 | 2.9% | 25.8 | 3.3% | 53.6 | REFUTED (smaller): 1.3 us per 64-token chunk, not 8 us |
| mHC inductor, MoE permute, elementwise, copies | 8.22 | 4.3% | 38.8 | 5.0% | part of 27.7 | |
| Idle inside the pipeline (host Python, syncs, gaps under 20 us) | 2.92 | 1.5% | 13.8 | 1.8% | 16.8 launch and syncs (31 to 125 s case) | REFUTED: the step is not launch-bound |
| PP fill, steady and drain | 8.84 | 4.7% | 21.2 | 2.7% | part of 27.7 | |
| Post phase: CPU Adam and offload copies | 18.59 | 9.8% | 18.6 | 2.4% | 5 (fixed per step) | 3.7x |
| **Total** | **189.44** | 100% | **795.6** | 102.2% | 367 | +2.2% against the measured 778.6 s |

- CORRECTED (`verify2/dsa.md` 2.4): the T2 rows padded to 4,096 give
  E[s^2] = 6.55e9, not the 5.5e9 of `levers.py:11`. Per-row interpolation
  of the measured per-call costs gives bwd 269.8 + fwd 106.5 + indexer
  46.8 + top-k 19.9 = **443.0 s of DSA per T2 step (56.9%)**. The scaled
  total then is about 807 s (+3.7%, record-keeper arithmetic). The DP split
  of the rows is UNVERIFIED (the estimate takes half the rows per DP rank).
- DSA is 93.3 s of the pp3 step (49.3%). On pp0, which holds 2 DSA layers,
  DSA is 62.1 s (33%), and pp0 waits 58.2 s (31%) in PP receives. One DSA
  layer costs as much as 6.7 KDA layers (36.0 against 5.4 s per step).
- Per DSA call at 65,536 tokens (rank 48): sparse-MLA forward 101 ms,
  backward 522 ms, indexer 31 ms (CORRECTED from 34 ms; `verify2/dsa.md`
  2.2). The grid is one block per query token. Each block computes 16
  head rows (`padded_H = max(next_pow2(H), 16)`; the backward uses
  `block_H = min(64, padded_H)`), but TP8 leaves 8 real heads per rank (`tilelang_sparse_mla_bwd.py:127-130`,
  `tilelang_sparse_mla_fwd.py:56,72,83` at `4716a367a`). Per query, the
  backward gathers 2,112 x 576 bf16 key values and does 4.87 MB of fp32
  atomic adds into dKV (index width 2,112, not 2,048;
  `kpool_indexer.py:233`). So the per-rank DSA work does not change from 8
  to 16 heads.
- Recompute: R is 37.6 s of the pp3 step (19.9%). The DSA core is 18.2 s
  (48%) of R.
- EP all-to-all: it runs on the compute stream (stream 7), so it is 100%
  exposed. 28% of it on pp3 is coupling wait, because EP16 = TP8 x DP2
  joins the two DP replicas in every MoE layer.
- Idle: 2.3 to 3.3 s per stage inside the pipeline (1.2 to 1.7%). pp3 runs
  12,544 kernels per micro-batch. Two thirds are under 10 us, but they sum
  to 0.6 s. The host waits for the GPU: rank 48 spends 123 s in
  `cudaStreamSynchronize`. So CUDA graphs can save at most about 2%.
- Stage balance: micro-batch work 112.8 / 149.1 / 150.7 / 159.1 s by stage.
  The best contiguous split is 12/11/11/11 (149.0 / 118.5 / 150.6 /
  153.6 s, `analysis/trace/pp.json`), about -19 to -26 s at T2.
- Limits: only TP rank 0 of each stage is traced, the other 56 traces are
  checked by size only, and the slice holds 33 micro-batches, not 156.

### (h) All-to-all bench (job `kdatp-prof-20260929c`)

`a2a_bench.py` ran bf16 `all_to_all_single` on the 64 ranks before Ray
started, with all groups at the same time. Each value is the median of 10
calls of the slowest group. `ep16` is the r47 EP group (16 ranks on 2
nodes; half of the bytes cross EFA). `ep8` is the 8 GPUs of one node.
Source: `run/a2a/a2a.md`, `reports/prof-job.md`.

| Bytes per rank | `ep16` ms | `ep16` cross-node GB/s per rank | `ep8` ms | `ep8` algbw GB/s |
| --- | --- | --- | --- | --- |
| 33,554,432 | 0.89 | 18.8 | 0.17 | 197.6 |
| 67,108,864 | 1.30 | 25.9 | 0.24 | 277.8 |
| 134,217,728 | 1.95 | 34.3 | 0.34 | 391.3 |
| 268,435,456 | 3.38 | 39.7 | 0.52 | 518.7 |
| 536,870,912 | 6.23 | 43.1 | 0.92 | 584.6 |
| 1,073,741,824 | 11.99 | 44.8 | 1.65 | 649.2 |

- At 536,870,912 B (8,192 tokens x top-8 x 4,096 x 2 B), EP16 takes
  6.23 ms and EP8 takes 0.92 ms (6.8x faster). The old model assumed
  35 GB/s and 7.7 ms.
- In the trace, a mean token call on pp3 sends 739 MB per rank. Its
  pair-min time is 0.86x the bench time at the same size, and send bytes
  over pair-min time is about 100 GB/s per rank. So about 50 GB/s crosses
  EFA, the line rate of one GPU (400 Gbps). The time above the bench is the
  coupling wait (about 1.7 ms per call) (`reports/trace-analysis.md`
  section 5).
- The largest receive from one peer is 127 MB against a mean of 41 MB
  (3.1x). The aggregate per rank still runs at line rate.
- So NCCL tuning on EFA cannot gain much. Only a change of the path (EP8)
  removes this time.

### (i) Layout arms on the T2 rows (jobs `kdatp-prof-20260929d` and `-e`)

These arms are train-only timing tests on replayed T2 rows, not training
runs. Every step loads `data/t2/rollout_40.pt` (314 rows, 64 episodes,
22,329,281 tokens, max row 131,070, sha256 `2ca13686...8e66`), the rows of
the 778.6 s control. Each arm starts from the r43 `iter_0000039` seed
without optimizer state (`no_load_optim`), with the profiler off and
`skip_actor_forward_only` on, and runs 4 steps (rollouts 40 to 43; 40 is
cold). No save, no W&B. Job `d` ran `ep8`, `base` and `pack` on one set of
8 pods (20:57Z to 00:19Z). Job `e` ran `ep8pack` and `tp4ep8pack` on another
set (21:31Z to 23:47Z). All 5 arms ended `rc 0` with no OOM. The warm mean
is the mean `actor_train` of rollouts 41 to 43 (rank-48 `perf N` lines,
`results.json` `steady_mean`). The control is one step: 778.6 s at rollout
42 of job `20260929a` (`20260929a/run-20260929a/kdatp-prof/trainer-0.log`
l.18553; rollout 41 was 800.2 s). MFU = 2.0% x 778.6 / warm mean. The
round takes 2.0% as the true MFU at 778.6 s (`reports/synthesis.md` section
1); the scale of the MFU study gives 2.2% (row "RL peers" of (d)), so read
the MFU column as a ratio. Source:
`reports/arms-20260929d.md`, `reports/arms-20260929e.md`,
`arms-20260929d/results.json`, `arms-20260929e/results.json`,
`verify2/ep8.md`, `verify2/tp4.md`.

Step time:

| Arm | Job | Override on the r47 layout | Micro-batches per DP rank | Cold step 40 (s) | Warm steps 41 / 42 / 43 (s) | Warm mean (s) | Change vs 778.6 s | Change vs the same-job reference | MFU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `base` | d | none | 156 | 865.2 | 780.1 / 775.5 / 775.1 | 776.9 | -0.2% | - | 2.0% |
| `pack` | d | `max_tokens_per_gpu: 131072` | 97 | 1,266.4 | 778.1 / 779.0 / 777.9 | 778.3 | -0.0% | +0.2% vs `base` | 2.0% |
| `ep8` | d | `expert_model_parallel_size: 8` | 156 | 896.4 | 668.8 / 669.8 / 679.2 | **672.6** | **-13.6%** | -13.4% vs `base` | 2.3% |
| `ep8pack` | e | `expert_model_parallel_size: 8`, `max_tokens_per_gpu: 131072` | 97 | 1,083.8 | 682.1 / 665.9 / 663.5 | 670.5 | -13.9% | -13.7% vs `base`; -0.3% vs `ep8` | 2.3% |
| `tp4ep8pack` | e | `tensor_model_parallel_size: 4`, `expert_model_parallel_size: 8`, `max_tokens_per_gpu: 131072`, `data_pad_size_multiplier: 1024`; DP4 (data `t2-dp4`: 316 rows, 22,336,299 tokens) | 49 | 2,050.7 | 442.1 / 457.2 / 455.2 | **451.5** | **-42.0%** | -32.7% vs `ep8pack` (the TP4 part alone; range -31.4% to -33.1%) | 3.45% |

Memory (torch peaks: max over all ranks of a stage and rollouts 40 to 43;
device memory: `nvidia-smi` snapshots, not peaks, except where marked):

| Arm | Torch max allocated, pp0 / pp1 / pp2 / pp3 (GiB) | Torch max reserved, pp0 / pp1 / pp2 / pp3 (GiB) | Device memory |
| --- | --- | --- | --- |
| `base` | 72.94 / 76.55 / 70.55 / 69.52 | 78.95 / 80.02 / 74.65 / 79.42 | 107.1 to 108.5 GiB (head node) |
| `pack` | the same as `base` | 81.33 / 83.81 / 77.30 / 80.61 | 105.7 to 113.6 GiB |
| `ep8` | 93.19 / 104.39 / 98.39 / 99.90 | 97.25 / 108.38 / 102.62 / **109.73** | 123.1 to 138.4 GiB per node |
| `ep8pack` | the same as `ep8` | 101.04 / 112.03 / 105.16 / 112.60 | 142,323 to 144,671 MiB (pp1 node) |
| `tp4ep8pack` | 131.08 / 138.65 / 129.21 / 133.22 | 148.66 / 149.24 / 140.66 / 154.13 | **Full.** DCGM 30 s series (all 64 GPUs): in steps 1 to 3, 26 of 64 GPUs had less than 1 GiB free; the maximum, 182,598 MiB, is 32 MiB below the usable 182,630 MiB. The 23:23Z `nvidia-smi` sample of 182,595 MiB leaves 36 MiB usable (CORRECTED from "764 MiB free", which counted the 728 MiB driver reserve) |

- EP8 memory: the growth over `base` in allocated memory is +20.25 /
  +27.84 / +27.84 / +30.38 GiB. That is exactly 2.53 GiB per MoE layer (18
  more experts x 25.17M parameters x 6 B) times 8 / 11 / 11 / 12 MoE layers
  (`verify2/ep8.md` section 2). The `[peak-memory]` IP map puts each block
  of 8 ranks on one node, so the EP8 group is inside one node. Expert DP is
  2 (ranks r and r+8).
- Live EP8 prediction (MODEL, 3 methods within 0.9 GiB,
  `verify2/ep8/mem_pred.py`): device peak 133.0 / 147.8 / 142.9 / 148.1
  GiB by stage. The margin is about 26 GiB after 2 GiB each for
  fragmentation and a change in NCCL memory.
- TP4 memory: over `ep8pack`, allocated +30.8 to +37.9 GiB and reserved
  +35.5 to +47.6 GiB by stage (CORRECTED). The hard floor on pp1 is 138.65
  GiB allocated plus at least 30.2 GiB outside torch = 168.9 GiB. That
  leaves 9.5 GiB of hard slack in the harness and about 5.5 to 8 GiB live
  (MODEL; `verify2/tp4.md` 4.2). `expandable_segments:True` is already on,
  and miles flushes the pool when free memory falls below 10%
  (`model.py:669-674`). So TP4 is fragile: NO-GO for a live run until a
  harness run shows at least 15 GiB of hard slack on every stage. The op
  at the TP4 peak is UNVERIFIED; the TP-invariant fp32 indexer logits are
  probably not it (`verify2/tp4.md` 4.3).
- Packing: no gain at EP16 (+0.2%) or at EP8 (-0.3%). The time per
  micro-batch grows with its tokens (4.31 s at 71.6K, 6.91 s at 115.1K
  tokens). So the step cost is per token, not per micro-batch. The packing
  gain of the earlier model (-2 to -8%) is REFUTED.
- EP8 gain: -104.3 s against `base`. It matches the EP16 all-to-all of
  125.1 s per T2 step minus the in-node remainder (about 18 s by the bench
  ratio 0.92 / 6.23 ms). The gain shrank over the 3 warm steps: -14.3 /
  -13.6 / -12.4% on rank 48 and -15.2 / -13.9 / -11.3% on rank 0. `ep8pack`
  does not shrink. The steady-state gain is UNVERIFIED.
- TP4 gain: it matches half of the DSA time. The DSA kernel does the same
  work per query at 8 or 16 heads, and DP4 halves the micro-batches per
  rank: 670.5 - 443 / 2 = 449 s against 451.5 s measured
  (`verify2/dsa.md` 3.3). No TP4 trace exists. The harness bubble at 49
  micro-batches (5.8%, MODEL) is larger than live, so the harness
  understates the TP4 gain against r47.
- Cold steps: `ep8pack` compiled 576 TileLang kernels (+413 s over its
  warm mean) and `tp4ep8pack` compiled 768 (+1,599 s, 26.7 min). The
  harness saves no kernel cache, and the TTL removed the pods, so the TP4
  kernels are lost. The TP4 arm survived the idle reaper: its lowest
  60-minute rolling mean was 16.7% against the 10% line.

Numerics (same seed, same rows; reference `base` of job d):

| Arm | Step-40 `grad_norm` (change vs `base` 0.126965) | `grad_norm` change at steps 41 / 42 / 43 | Step-40 `train_rollout_logprob_abs_diff` (change vs `base` 0.027279) |
| --- | --- | --- | --- |
| control (job `20260929a`) | 0.1269729 (+0.006%) | - | 0.027279 (bit-identical) |
| `pack` | 0.127514 (+0.43%) | +2.88 / -1.39 / -0.04% | 0.027248 (-0.11%) |
| `ep8` | **0.136139 (+7.23%)** | -1.22 / +1.10 / -0.54% | 0.027274 (-0.02%); steps 41 to 43: +0.09 / +0.07 / -0.22% |
| `ep8pack` | 0.127521 (+0.44%; +0.005% vs `pack`) | +1.71 / +2.68 / -2.62% | 0.027253 (-0.10%) |
| `tp4ep8pack` | 0.129292 (+1.83%) | -1.47 / -0.58 / +1.06% | 0.027191 (-0.32%); train/rollout KL -0.36% |

- A same-layout repeat across jobs moves the step-40 `grad_norm` by only
  0.006%. So the layout effects are deterministic, not run noise.
- The `ep8` step-40 +7.2%: the explanation "a constant scale on the expert
  gradients or a double count" is REFUTED (`verify2/ep8.md` section 3).
  `grad_norm` is computed before the Adam step, so a constant scale or a
  double count stays at steps 41 to 43, and these steps are in the noise.
  `ep8pack` has the same EP8 and expert-DP-2 layout and matches `pack` to
  0.005%. Dense and expert gradients both scale by 1/dp, and the norm is one
  all-reduce with an ETP-aware duplicate filter. The cause is UNVERIFIED:
  a one-time first-step event of the `ep8` arm only (not reproducible, or a
  first-step effect of EP8 with 8,192-token micro-batches). r47 runs
  `--no-load-optim`, so every live resume is a first step after load.
- `ep8` log-prob distance: within 0.1% at steps 40 to 42 and -0.22% at
  step 43 (CORRECTED from "within 0.1%").
- TP4: the step-40 log-prob gap of -0.32% is 3x the other layouts (-0.02%
  to -0.11%) and larger than the +0.22% of the accepted KDA port. The loss
  is within 0.3% only at steps 42 and 43 (CORRECTED). TP4 parity is
  PLAUSIBLE, not proven.
- The miles argument code switches the dispatcher from `allgather` to
  `alltoall` in every arm (`miles/utils/arguments.py:2766-2772`), so all
  arms use the all-to-all dispatcher. DeepEP is off.

### (j) The live run r47 (read-only)

Run `rl-glm53f47-wxj87` (r47, agentic-debt-766, trainer image
`miles-glm53-r17-20260928a`, W&B `arena/rl-glm53f-adebt-766/0ix3m75e`).
Source: `reports/live.md` (read 2026-09-29 19:50Z to 20:20Z) and
`verify2/loop.md` (2026-09-30 00:31Z to 01:30Z), raw files under `live/`
and `verify2/loop/`, and one read-only `kubectl logs` at 01:43:14Z
(`reports/synthesis.md` section 4.3). Nothing on the cluster was changed.

| Quantity | Value | Source |
| --- | --- | --- |
| `actor_train` share of the loop step, steps 38 to 52 | 90.77%. Mean loop step 3,224.4 s, `actor_train` 2,926.9 s | W&B history (`verify2/loop.md` section 1) |
| Trainer wait for rollouts | 0 s through step 52; the least slack was 30 s at step 49, when `generate(50)` took 4,306 s. After train step 53 ended (01:04:00Z), the trainer waits: `generate(54)` began at 23:52:26Z and had not ended at 01:43:14Z, so the wait was at least 39 min. The RolloutManager read rate fell to 6.3 to 12.8 MB/s (normal 140 to 160 MB/s), with one thread at 75 to 100% CPU. Cause UNVERIFIED: GIL starvation or slow NFS reads | `verify2/loop.md` section 2; `reports/synthesis.md` section 4.3 |
| Gym supply | 38.7 groups/h (153 results in 3.95 h, NATS `ARENA_RESULTS` publish times) | `verify2/loop/nats-msg-times.json` |
| Trainer demand | 59.3 groups/h (53.1 groups examined per step / 3,224 s) | W&B `rollout/group_metrics/n_groups` |
| Backlog | about 408 groups (queue 320 + NATS 48 to 56 pending + 32 unacked) | NATS `jsz` at 00:45Z |
| Supply ceiling | about 66 groups/h (56 to 74), ESTIMATE. About 30 (30 to 65) in-flight slots look orphaned; task `ack_wait` 172,000 s, `NATS_TASK_DEADLINE_SECS` 176,000 s, 0 redeliveries, 7 gym pods in `Error` | `verify2/loop.md` 3a, 3b |
| Steady-state loop limit | about 1.11x (0.94x to 1.25x), ESTIMATE. EP8 cuts the loop step about 12.2% (ESTIMATE) and sits at that limit | `verify2/loop.md` 3b |
| Sample staleness | 11.0 policy versions on average; up to 29 at step 53 | W&B `rollout/off_policy_round` |
| Pre-train data path | slowest node, fetch plus fill: mean 251 s (7.8% of the loop step), max 455 s (steps 38 to 52); 456 s at step 53. The rank-48 view is 225.9 s (steps 39 to 48) | `verify2/loop.md` 4a |
| Data path cause | int32 R3 routing is 99.1 to 99.2% of the Ray object bytes (88.6 to 158.1 GB per rollout, 2 objects). The head Ray plasma store serves each of the 8 trainer nodes its full 45-layer DP shard (44 to 79 GB) in partly serial pulls at 1.7 to 4.0 GB/s in total. A CPU fill of 45 to 109 s adds on top. The 200 GB head store spilled 3.0 TiB | `verify2/loop.md` 4b, 4c |
| Per-token cost | 70.0 us per padded token per DP replica (median, 66.3 to 74.1, steps 38 to 53) against 65.9 us in the harness (+6.3%). The per-micro-batch term is -0.14 ± 0.29 s, so no packing signal | `verify2/loop.md` section 5 |
| Device memory | torch reserved + 30.4 to 33.0 GiB outside torch. Reserved max 81.2 / 86.6 / 82.1 / 84.8 GiB; device used max 112.8 / 119.6 / 115.0 / 117.7 GiB by stage (DCGM `FB_USED` max agrees to 0.1 GiB) | 6,912 `memory_utils.py:41` lines; `live/amp/r47-fbused-max-by-pod.json` |
| Live against harness | about +6 to +10.5 GiB of device memory: pool fragmentation over many steps plus 1.5 to 4 GiB more outside torch. The exact offsets are UNVERIFIED | `reports/live.md` 1b; `verify2/ep8.md` section 2 |
| Live `max_allocated` | not logged, because `MILES_LOG_PEAK_MEMORY` is not set in r47 | `reports/live.md` |
| NCCL | 2.29.7 is loaded (`torch.cuda.nccl.version()`, `/proc/<pid>/maps`, pip `nvidia-nccl-cu13 2.29.7`). The env `NCCL_VERSION=2.28.3-1` is a stale base-image label. aws-ofi-nccl 1.18.0. `NCCL_PROTO=simple` is set; its cost is UNVERIFIED | `reports/live.md` l.146-154 |
| TileLang compile in live steps | a new padded length in 12 of steps 3 to 38 (11 to 15 s per rank); none after step 38. About 3.5 s per step on average (ESTIMATE) | `reports/live.md` section 3 |

- The earlier live estimate "a trainer speedup reaches the loop up to about
  1.5x" is REFUTED. It took the supply to be equal to the demand.
- The harness arm numbers do not hold the data path, because the harness
  loads `.pt` files.
- The harness figure of 29 GiB outside torch is UNVERIFIED for this round.
  The raw `nvidia-smi` snapshots were not saved.

### (k) Precedent checks

| Item | Finding | Source and status |
| --- | --- | --- |
| Upstream DSA kernel, miles PR #3608 | Open, not draft, stacked on #3605 to #3607 (all open), no human review, GPU CI skipped. It moves TileLang fragments to shared memory in the forward and the backward, gives the backward dynamic shapes, and adds a FlashMLA sparse forward. It measures 1.64x per layer on GB300 and 2.11x on H200 at 64K, TP8. No B200 number; numerics are unit-level only | `verify2/dsa.md` 4.1 to 4.3; CONFIRMED |
| Our DSA kernel files against the PR | Equal to upstream main `5ebc06f8` (blobs `e226bb08`, `022a6006`, `260b38ed`), not to the PR base. The algorithm is the same as the PR base | `verify2/dsa.md` 4.4; "byte-identical to the PR 'before' state" is REFUTED |
| PR #3608 on our code | Its base predates GLM-5.3-Flash support (#2786). It deletes `glm5/ops/sparse_mla.py`, which `glm5_next/dsa.py:11` imports. It needs FlashMLA; the image has no FlashMLA and has TileLang 0.1.9 (the PR ran 0.1.12 and 0.1.14) | `verify2/dsa.md` new findings |
| PR #3608 saving at T2 | TP8: 376 s to 173 to 232 s (save 144 to 203 s, 18.5 to 26% of 778.6 s). TP4: save 75 to 103 s. The FlashMLA forward adds about 18 GiB of transient memory at 131K tokens | `verify2/dsa.md` 4.5; MODEL, UNVERIFIED on B200. CORRECTED from "283 s to 176 s" |
| EP8 for GLM-5.3-Flash | Upstream sets EP = GPUs / 4 (`scripts/run_glm5_3_flash.py` l.75-86, added in `cc76e23915`), so 64 GPUs use EP16 with expert DP 1. EP8 with expert DP 2 at 64 GPUs is the generic Megatron expert-DP path (upstream `run_qwen3_30b_a3b.py` l.318-329, B200 on 2 to 4 nodes). No upstream GLM-5.3-Flash parity result exists for it | `verify2/ep8.md` section 4; CORRECTED (`reports/precedent.md` cited `5fb2865678`) |
| EP8 in node, other models | NVIDIA B200 recipes (DeepSeek-V3 on 256 B200, Qwen3-235B on 64 B200) and the AWS HyperPod RL recipes use EP8 | `reports/precedent.md` section 0 item 2 |
| CUDA graphs and EP all-to-all overlap | No RL framework (miles, slime, veRL, NeMo-RL) uses them in training. The image Megatron rejects per-layer CUDA graphs with full recompute (`transformer_config.py:3270-3274`) and VPP with the 11/11/11/12 split (`transformer_config.py:2566-2598`); the 1F1B overlap needs VPP when PP > 1 | `reports/precedent.md` B.1, C.1 |
| DSA layers kept out of recompute | No config option (the image Megatron has uniform or block only, `transformer_block.py:733-772`) and no upstream precedent; the NVIDIA GLM-5 DSA recipes use full uniform 1 | `verify2/dsa.md` 5.1 |
| CP2 | `dsa.py:136-137` raises `NotImplementedError` for CP above 1 | `verify2/tp4.md` new findings |
| Chunked DSA indexer | In-image Megatron `dsa_cudnn_kernels.py:135,503-674` (1 GiB query chunks); radixark/miles#2793 (open) | `verify2/tp4.md` new findings |
| int16 routing | verl `agent_loop.py:131,796`; NeMo-RL `interfaces.py:81-89`. Upstream miles main `3439ec7513` still ships int32 and all 45 layers | `verify2/loop.md` 4d |
| NCCL large-message regression (NCCL 2.28.3 and older) | Does not apply: the image loads NCCL 2.29.7 | `reports/live.md` l.146; the 2.28.3 claim of `reports/precedent.md` is CORRECTED |

### (l) Verifier corrections of round 20260929c

The second adversarial pass (`verify2/`) checked the round claims. This
record applies each verdict above. Source: `reports/synthesis.md` section 8
and the four verifier reports.

| Round claim | Verdict | Corrected value | Source |
| --- | --- | --- | --- |
| DSA about 431 s of 779 s at T2 | CORRECTED | about 443 s (57%). The indexer plus top-k is 66.7 s, not 55.7 s, because the T2 E[s^2] is 6.55e9, not 5.5e9 | `verify2/dsa.md` 2.4 |
| Indexer 34 ms per call at 64K | CORRECTED | 31 ms at 65,536 tokens. The index width is 2,112, not 2,048 | `verify2/dsa.md` 2.2, 3.1 |
| Our DSA files are byte-identical to the PR #3608 "before" state | REFUTED | Identical to upstream main. Same algorithm as the PR base, different files | `verify2/dsa.md` 4.4 |
| PR #3608 T2 estimate 283 s to 176 s | CORRECTED | The B200 base is 376 s with recompute. Saving 144 to 203 s at TP8 and 75 to 103 s at TP4 (MODEL) | `verify2/dsa.md` 4.5 |
| EP8 in node has strong upstream GLM-5.3-Flash precedent (commit `5fb2865678`) | CORRECTED | Upstream GLM-5.3-Flash at 64 GPUs is EP16 (rule added in `cc76e23915`). EP8 with expert DP 2 is the generic Megatron path | `verify2/ep8.md` section 4 |
| `ep8` log-prob distance within 0.1% | CORRECTED | Within 0.1% at steps 40 to 42; -0.22% at step 43 | `verify2/ep8.md` section 3 |
| The `ep8` +7.2% `grad_norm` comes from a constant expert scale or a double count | REFUTED | Cause UNVERIFIED. `ep8pack` (same layout) matches `pack` to 0.005% | `verify2/ep8.md` section 3 |
| The TP4 lever is -42% | CORRECTED | -42.0% is the whole config against 778.6 s. The TP4 part is -32.7% (range -31.4% to -33.1%) against `ep8pack` | `verify2/tp4.md` claim 3 |
| TP4 "764 MiB free" | CORRECTED | 36 MiB free of 182,630 MiB usable | `verify2/tp4.md` claim 10 |
| The true TP4 peak can be higher than the sample | REFUTED | The device was full: 26 of 64 GPUs under 1 GiB free; max 32 MiB below usable | `verify2/tp4.md` claim 11 |
| TP4 adds 35 to 39 GiB allocated and 36 to 44 GiB reserved | CORRECTED | +30.8 to +37.9 GiB allocated; +35.5 to +47.6 GiB reserved | `verify2/tp4.md` claim 9 |
| Loss and log-prob match within 0.3% at TP4 | CORRECTED | Step-40 log-prob gap -0.32%, KL -0.36%. The loss is within 0.3% only at steps 42 and 43 | `verify2/tp4.md` claim 6 |
| This TP4 config "would likely fail in production" | CORRECTED | Fragile: about 5.5 to 8 GiB of hard slack live (MODEL). NO-GO without at least 15 GiB of relief | `verify2/tp4.md` 4.2 |
| The fp32 indexer logits are the TP4 peak | UNVERIFIED | MODEL points to the MoE or sparse-MLA backward, or the pp3 loss path | `verify2/tp4.md` 4.3 |
| Keep DSA layers out of recompute: -86 to -134 s at T2 | CORRECTED | No config option and no precedent. It does not fit TP4 or live EP8 pp1 (MODEL). With pp3 only, the gain is about 40 s | `verify2/dsa.md` section 5 |
| r47 trainer waits 0 s | CORRECTED | True through step 52. After step 53 the trainer waited at least 39 min (01:43:14Z log check) | `verify2/loop.md` section 2; `reports/synthesis.md` 4.3 |
| A trainer speedup reaches the loop up to about 1.5x | REFUTED | About 1.11x in steady state (0.94x to 1.25x, ESTIMATE). The gyms make 38.7 groups/h, and the trainer needs 59.3 groups/h | `verify2/loop.md` 3a, 3b |
| Pre-train path 226 s (7.5%) | CORRECTED | 225.9 s is the rank-48 view. The slowest-node path is 251 s (7.8%), up to 455 s | `verify2/loop.md` 4a |
| The data path is served to one node at a time | CORRECTED | Head Ray plasma store; partly serial pulls at 1.7 to 4.0 GB/s in total; a CPU fill of 45 to 109 s on top | `verify2/loop.md` 4b, 4c |
| A per-micro-batch term points to packing | REFUTED | -0.14 ± 0.29 s per micro-batch | `verify2/loop.md` section 5 |
| NCCL is 2.28.3 (`reports/precedent.md`) | CORRECTED | 2.29.7 is loaded. The 2.28.3 env value is a stale label, so the NCCL 2.28.3 large-message issue does not apply | `reports/live.md` l.146 |
| Harness memory outside torch is 29 GiB | UNVERIFIED for this round | The raw `nvidia-smi` snapshots were not saved. The live 30.4 to 33.0 GiB is CONFIRMED | `verify2/ep8.md` section 2 |
| Driver fix: count `GPU stopped early? = [1-9]` (`reports/synthesis.md` 6.3) | CORRECTED by this record | That Kineto flag read 0 on all 64 ranks of the cut run `20260929ab`, so it misses a real cut. Fix `27ab1bccdd` counts the torch warning `activity collection was stopped early` (0 in job `20260929c`, 64 in runs `20260929a` and `20260929ab`) | `20260929a/run-20260929ab/kdatp-prof/trainer-0.log` |

No verifier of round 2 re-checked these values; this record uses them as
reported: the Kineto signals of (f), the driver count bug, the 3.0 s
profiler cost, and the all-to-all bench of (h) (`reports/prof-job.md`). The
Kineto counts, the trace sizes, and the step times of (f) were re-read from
the local logs for this record.

### (m) Ranked path (synthesis of round 20260929c)

Source: `reports/synthesis.md` sections 2 to 6. The levers overlap, so the
rows do not add. MFU = 2.0% x 778.6 / the new T2 step. The 778.6 s control
is one step; the same-job base is 776.9 s. Harness gains carry over to
live rows only as an ESTIMATE: live costs 6.3% more per token and has a
smaller pipeline bubble.

| Step | Lever | T2 step and change vs 778.6 s | MFU after | Prerequisites | Kind | Risk | Precedent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | EP8 in node (TP8 SP, PP4, EP8, DP2, expert DP 2) | **672.6 s MEASURED, -13.6%** (-13.4% vs the same-job base); `ep8pack` 670.5 s | 2.3% | Job X passes (step-40 `grad_norm` repeat, 7 warm steps); the live gates of Next step | Config only: `expert_model_parallel_size: 8` | Low to medium. The step-40 `grad_norm` +7.2% has no known cause. The gain shrank -14.3 / -13.6 / -12.4% over 3 warm steps. The expert-DP-2 weight sync never ran. Live peak 148.1 GiB (MODEL), about 26 to 30 GiB margin | Upstream GLM-5.3-Flash at 64 GPUs uses EP16. EP8 uses the generic Megatron expert-DP path (upstream miles `run_qwen3_30b_a3b.py` l.318-329) |
| 2 | PP 12/11/11/11 on top of EP8 | ESTIMATE 647 to 654 s, -16% to -17% (trace model -19 to -26 s) | about 2.4% | Job X arm `ep8rb` | Config only: `decoder_first_pipeline_num_layers: 12`, `decoder_last_pipeline_num_layers: 11` | Low. pp0 gets 1 more MoE layer and has the most margin (45.3 GiB, MODEL). The PP reshard at load is UNVERIFIED | Megatron-Bridge DeepSeek-V3 16/16/16/13; miles DeepSeek-V3 last stage 13 |
| 3 | TP4 + EP16 + DP4, packed | ESTIMATE 540 to 560 s, -28% to -31% | 2.8 to 2.9% | Job Y (time, hard slack at least 15 GiB, parity); rollout supply at least 81 groups/h | Config only (a kernel-cache restore in the template is optional) | Medium. TP4 parity is PLAUSIBLE only (step-40 log-prob gap -0.32%). About 27 min of compile at each pod start. The cross-node all-to-all returns | slime GLM-5 TP4; miles Kimi-K3 and DeepSeek-V4-Flash TP4; none for GLM-5.3-Flash |
| 4 | DSA kernel: PR #3608 backport (TileLang fix, dynamic backward shapes, FlashMLA forward) at TP8 on EP8 | ESTIMATE 470 to 529 s, -32% to -40% (saves 144 to 203 s) | 2.9 to 3.3% | Backport (the PR predates `glm5_next`); image with FlashMLA for sm_100; 1-GPU B200 bench; 8-node parity arms | Code + image | Medium. The PR is open and stacked, with no human review and GPU CI skipped. Numerics are unit-level only. FlashMLA adds about 18 GiB at 131K tokens (MODEL) | Upstream miles PR #3608 (open): 1.64x per layer on GB300, 2.11x on H200 |
| 5 | TP4 + EP8 + DP4, packed | **451.5 s MEASURED, -42.0%**; the TP4 part alone is -32.7% vs `ep8pack` | 3.45% | Hard slack at least 15 GiB per stage (now 9.5 GiB in the harness, 5.5 to 8 GiB live, MODEL); Job Y memory snapshot; memory relief | Config keys; the relief needs code + image | High. The device was full (36 MiB free; 26 of 64 GPUs under 1 GiB free) | As step 3 |
| 6 | TP4 + EP8 + PR #3608 | ESTIMATE 349 to 377 s, -52% to -55% | 4.1 to 4.5% | Steps 4 and 5; the TileLang path at TP4 (FlashMLA does not fit) | Code + image | High | As steps 4 and 5 |
| 7 | No zero tail (#3607 `D_tail = 0` and removal of the pad in `glm5_next/dsa.py:189-190`) on top of step 4 | ESTIMATE about 431 s, -45% (GB300 numbers only) | about 3.6% | Step 4, the #3607 kernel, a parity arm | Code + image | Medium to high | PR #3608 GB300 table |
| Fallback | Block recompute of 10 layers at EP16, only if EP8 fails | MEASURED -4.1% (780.9 to 748.7 s, log-prob pass on) | about 2.1% | EP16 only | Config only | Does not fit with EP8 (pp0 169 to 173 GiB live, MODEL) | Megatron-Bridge, pretraining only |
| Loop | Supply: `--arena-inflight-multiplier` 8 to 10; task `ack_wait` and `NATS_TASK_DEADLINE_SECS` at 16 to 20 h | No change to the T2 step; lifts the supply ceiling of about 66 groups/h (ESTIMATE) | n/a | Next launch | Config | Latency grows with engine load (UNVERIFIED) | Standard knobs |
| Loop | int16 routing (the data path is 251 s = 7.8% of the loop step, MEASURED) | ESTIMATE -125 s per loop step (-3.9%); no change to the T2 step | whole-step MFU up to +7.8% | Code, image, job `kdatp-r3-int16` | Code + image | Low | verl `agent_loop.py:131,796`; NeMo-RL `interfaces.py:81-89` |

Loop view: r47 needs 59.3 groups/h, but the gyms make only 38.7 groups/h
(MEASURED). The steady-state loop limit is about 1.11x (0.94x to 1.25x,
ESTIMATE). EP8 cuts the loop step by about 12.2% (ESTIMATE) and sits at that
limit. The levers past EP8 raise the trainer MFU, but they reach the loop
only after a rise in rollout supply. A faster trainer still drains the
320-group queue and so lowers sample staleness; the effect on learning is
UNVERIFIED.

Refuted or parked (`reports/synthesis.md` 6.4): packing (REFUTED as a
lever), CUDA graphs (REFUTED as a large lever: at most about 2%), EP
all-to-all overlap (parked: needs VPP, no RL precedent, at most about 23 s
left after EP8), DSA layers kept out of recompute (parked: no option, no
precedent), the GPU optimizer (parked: 16 to 18 s, about 58 GiB of Adam
state per GPU does not fit with EP8), bf16 main grads (parked: numerics
risk), CP2 (NO-GO), block recompute stacked on EP8 (NO-GO), the per-PP-stage
routing slice (parked: no precedent), and another EP library such as DeepEP
or UCCL-EP (parked: image and host changes, no gain at hidden size 4096).

## Verdict

Round 20260929c (2026-09-29 to 2026-09-30). The Kineto fix works, and the
trace explains the whole step. The measured budget sums to the wall time on
all 8 traced ranks. Scaled to the T2 step, it gives 795.6 s against the
measured 778.6 s (+2.2%). So the 410 s gap of (c) is closed.

Where the time goes, measured: the DSA sparse attention takes about 443 s
of the 778.6 s T2 step (57%, CORRECTED). The sparse-MLA backward alone
takes about 270 s. The EP16 all-to-all takes 125 s (16%). It runs at the
EFA line rate on the compute stream with no overlap, and 28% of it is wait
between the two DP replicas. GEMMs take 71 s, TP collectives 37 s, KDA
26 s, and the CPU Adam phase 19 s. The GPU idles only 1.2 to 1.7% inside
the pipeline, so the step is not launch-bound.

Why it is so bad for us compared with others, in plain words: TP8 leaves 8
attention heads on each GPU, but the DSA kernel computes 16 head rows for
each query and gathers the full key set for each query on every TP rank.
So the sparse attention costs each GPU the same as 16 heads, and all 8 GPUs
of a TP group repeat it. The expert exchange crosses nodes on EFA and
blocks the compute stream. Full recompute repeats the forward pass (20% of
the step). Launch overhead, KDA, and packing are not the cause.

What the layout arms measure: EP8 in node gives 672.6 s (-13.6%) with
memory to spare (torch reserved max 109.7 GiB). TP4 + EP8 + DP4, packed,
gives 451.5 s (-42.0%), but the GPUs were full (36 MiB free). Packing gives
nothing. The EP8 step-40 `grad_norm` is 7.2% high, and its cause is
UNVERIFIED.

What reaches the RL loop: r47 is trainer-bound only because of a backlog.
The gyms make 38.7 groups per hour, and the trainer uses 59.3. The
steady-state loop limit is about 1.11x (ESTIMATE). EP8 sits at that limit.
A lever past EP8 needs more rollout supply first. Separately, the data path
before each train step (int32 routing through the head Ray store) costs
251 s per loop step (7.8%).

The path of (m): EP8 first (config only, gated on Job X), then PP
12/11/11/11, then the DSA kernel of PR #3608. TP4 stays out of live runs
until a harness run shows at least 15 GiB of hard slack on every stage.
Status stays Open: each job and live change in Next step waits for a user
go.

### Earlier verdict (2026-09-29, before the trace)

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

Trace result for this earlier verdict: the Kineto fix works (f). The
410 s is explained (g), almost all by DSA. REFUTED: the host launch count
as a suspect (12,544 kernels per pp3 micro-batch, idle 1.2 to 1.7%), the
KDA state chain as a large cost (26 s at T2), and a gain from packing
(+0.2% and -0.3%, section (i)). CONFIRMED: the 50 GB/s EFA exchange with
nothing behind it (125 s at T2) and the 8-fold DSA gather at TP8.

## Caveats and open items

Closed by round 20260929c:

- **The unexplained 410 s.** Closed: the trace budget (g) sums to the step.
  The trace check of 2026-09-29 read `sparse_mla_bwd_kernel` at 269 s per
  T2 step, the NCCL all-to-all at 125 s, and no kernel-free gap above 3.3 s
  per stage.
- **The record-count datum.** The warm trace counts 12,544 kernels per pp3
  micro-batch. The estimate of 17K to 21K launches is REFUTED.
- **Realized EFA all-to-all bandwidth.** Measured (h): 43.1 GB/s across
  nodes per rank at 536 MB in the bench, and about 50 GB/s (line rate) in
  training.
- **DSA kernel time.** Measured per call (g): forward 101 ms, backward
  522 ms, indexer 31 ms at 65,536 tokens.
- **TileLang backward recompiles in a live run.** Measured (j): a new
  length in 12 of r47 steps 3 to 38, none after step 38.
- **Stage-0 idle share.** Measured: pp0 waits 58.2 s (31%) in PP receives;
  this wait is not on the critical path.
- **The 8 us per KDA chunk.** REFUTED: 1.3 us.
- **Trace export size and time.** Measured (f): 393.5 to 460.0 MB per rank
  (gz), 338 to 405 s, about +30 GiB of host memory per rank. miles still
  has no per-rank filter for `train_overall`, so all 64 ranks write a
  trace.
- **EP8 headroom.** Measured in the harness (i): torch reserved max
  109.7 GiB. The live peak is a MODEL (148.1 GiB).
- **Kineto commit in the wheel.** The live proof is in (f): `Max GPU buffer
  size: 4096MB` on 64 ranks.
- **The CUPTI queue rule** behind the 33-buffer burst stays UNVERIFIED. It
  does not matter with the 4096 MB cap.

Still open from 2026-09-29: arXiv 2504.14960 and miles PR #2353 and #2038
were not read for this study.

Open after round 20260929c (UNVERIFIED; `reports/synthesis.md` section 9):

1. Is the `ep8` step-40 `grad_norm` +7.2% a one-off, or a first-step effect
   of EP8 with 8,192-token micro-batches? It can recur at each live resume,
   because r47 runs `--no-load-optim`. Job X answers it.
2. Does the EP8 gain hold past 3 warm steps? It shrank from -14.3% to
   -12.4%.
3. Which op sets the TP4 peak? Does a chunked indexer help (0 to 15 GiB)?
4. What are the real time and memory of `tp4ep16pack`? Only MODEL numbers
   exist (540 to 560 s, 101 to 111 GiB allocated).
5. What does PR #3608 gain on B200 and on TileLang 0.1.9? How much memory
   does FlashMLA add at the real peak? When does upstream merge #3605 to
   #3608 and port `glm5_next`?
6. What causes the `generate(54)` stall of r47: GIL starvation or slow NFS
   reads? Is it still stalled after 01:43:14Z?
7. What is the real rollout supply ceiling (66 groups/h, ESTIMATE)? How
   does multiplier 10 change gym latency and engine load? Where is the task
   `ack_wait` set?
8. How do the harness gains carry over to live rows? Live costs 6.3% more
   per token and has a smaller bubble.
9. What are the live `max_allocated` (not logged) and the parts of the 29
   to 33 GiB outside torch? How do EP8 and TP4 change NCCL memory?
10. Does the EP8 weight sync at expert DP 2 work in a live run? The harness
    never runs `update_weights`. Does the PP reshard at 12/11/11/11 load
    cleanly?
11. Does the harness accept `debug_exit_after_rollout` 8 with replay, and a
    per-arm env?
12. What does `NCCL_PROTO=simple` cost the TP and in-node collectives? No
    arm measured it.
13. Only TP rank 0 of each stage was traced. No TP4 trace exists, so the
    16-head DSA kernel time on B200 is inferred from the source and from
    the arm time, not measured.

## Next step

The two steps of 2026-09-29 ran: the profile job (`kdatp-prof-20260929c`)
and the EP8 arm (`kdatp-prof-20260929d`, with `kdatp-prof-20260929e`).
None of the items below ran. No item below changes production. Each job,
image build, and live change needs a new explicit go from the user.

Decisions without a job (`reports/synthesis.md`, adopt now):

1. Pick EP8 (`expert_model_parallel_size: 8`, all other r47 values kept) as
   the next live layout. It stays gated on Job X.
2. Drop packing, CUDA graphs, EP all-to-all overlap, DSA layers kept out of
   recompute, the GPU optimizer, bf16 main grads, CP2, and the per-PP-stage
   routing slice from the plan. Keep block recompute of 10 layers only as
   an EP16 fallback.
3. Keep every TP4 layout out of live runs until a harness run shows at
   least 15 GiB of hard slack on every stage.
4. Gate every trainer lever past EP8 on a measured rise in rollout supply.
   Measure supply by the NATS publish-time method
   (`verify2/loop/nats_times.py`, `nats_ids.py`).
5. Put these on the next-launch checklist: `MILES_LOG_PEAK_MEMORY=1`,
   `--check-weight-update-equal` on the first EP8 launch
   (`placement_group.py:213-217`), an alarm when DCGM `FB_USED` goes above
   160 GiB, a flag on any single-step `grad_norm` jump above 5% (above all
   on the first step after a resume), and the per-token check (about 61 us
   target against 70.0 us now, ESTIMATE).
6. For the DSA kernel, wait for upstream #3605 to #3608 to merge and for
   the `glm5_next` port. A backport is the fallback.

Proposals that need a go (NOT run; `reports/synthesis.md` section 7):

1. **Diagnose the r47 `generate(54)` stall.** One `py-spy dump --pid 15059`
   in `rl-glm53f47-wxj87-trainer-worker-0 -c pytorch`. It attaches to the
   production RolloutManager for about 1 s. Only the user decides any
   action after it.
2. **Job X, `kdatp-prof-20260930a` (EP8 validation and rebalance).** 8 x
   `p6-b200.48xlarge`, queue `gpu.p6-b200-48xlarge`, context
   `arena-prod-bom-v2`, namespace `arena-tasks`, image
   `miles-glm53-r17-20260928a`, harness `dabe8a8972` (or later), seed r43
   `iter_0000039`, data `t2`, `replay_rollout: 40`, `KDATP_KCACHE`
   `20260927dwkdatp.tar`. Arms in this order: `ep8a` (EP8,
   `debug_exit_after_rollout` 2, timeout 3,600 s); `ep8long` (EP8,
   `debug_exit_after_rollout` 8, timeout 7,800 s); `ep8rb` (EP8 +
   `decoder_first_pipeline_num_layers` 12 +
   `decoder_last_pipeline_num_layers` 11, `debug_exit_after_rollout` 4,
   timeout 5,400 s). About 3.4 h in total. Decision rules: if both step-40
   norms are within 0.5% of 0.12697, the job-d value was a one-off; if one
   repeats 0.13614 within 0.1%, run the per-tensor dumper arm of
   `verify2/ep8.md` P2 before any live EP8. `ep8long` warm steps hold at
   least -12% against 776.9 s. `ep8rb` is GO at least 1% below `ep8long`
   with pp0 reserved at or below 115 GiB.
3. **Job Y, `kdatp-prof-20260930b` (TP4 memory and the EP16 fallback).** The
   same node shape and image, plus a harness change that tars
   `/tmp/kernel_cache` at each arm end. Arms in this order: `ep8pack2`
   (warm-up and TP8 control, 2 steps, data `t2`); `tp4ep16pack` (TP4, EP16,
   `max_tokens_per_gpu` 131072, `data_pad_size_multiplier` 1024, data
   `t2-dp4`, 4 steps, timeout 7,200 s); `tp4mem` (the `tp4ep8pack` settings
   plus `record_memory_history`, `memory_snapshot_num_steps` 41, a snapshot
   dir under `guparpit/kdatp/prof/20260930b/snap`,
   `MILES_LOG_PEAK_MEMORY=1`, and `NCCL_DEBUG=INFO` with subsystems
   `INIT,ALLOC,NVLS`; 2 steps). About 2.7 to 3.0 h. Jobs X and Y can run at
   the same time (at most 2 kdatp jobs).
4. **Live EP8 at the r47 step-60 save** (save interval 10). Retire r47 at
   the step-60 checkpoint and relaunch through the Argo
   `arena-miles-deployer` template. The only layout change is
   `expert_model_parallel_size` 16 to 8. Dataset: agentic-debt-766 on main.
   The user owns the retire, any resume watcher, and the relaunch. The
   checklist of decision 5 applies. Only after Job X passes.
5. **Supply config at the same relaunch.** `--arena-inflight-multiplier` 8
   to 10, and task `ack_wait` and `NATS_TASK_DEADLINE_SECS` from about 48 h
   to 16 to 20 h. Optional: PP 12/11/11/11, only if `ep8rb` passes.
6. **Image build 1.** The PR #3608 kernel backport plus FlashMLA for sm_100.
   Then Job Z, `kdatp-dsa-bench` (1 B200 pod, about 30 to 45 min). Then
   8-node parity arms `base`, `dsa3608` and `dsa3608tl` at the r47 layout
   (about 70 min per arm).
7. **Code and image build 2.** int16 routing in
   `routing_replay.decode_routing` and `ROLLOUT_DATA_TENSOR_DTYPES`, with
   one unit test. Then job `kdatp-r3-int16`: 8 nodes, int32 and int16 arms
   on replayed rollout-53 data through the real Ray path, 2 steps each,
   about 1.5 h.
8. **Template changes owned by AREnATasksApps, not by us:** a TP4 kernel
   tarball restore at pod start, head `--object-store-memory` and
   `/dev/shm` size, and a Mooncake master. Each needs its owner and a user
   go.

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
  (Superseded: `dabe8a8972` added the edit, and job
  `kdatp-prof-20260929c` tested it.)
- Round 20260929c (2026-09-29 to 2026-09-30). The job operators of the
  round created three PyTorchJobs on `arena-prod-bom-v2`, namespace
  `arena-tasks`, 8 nodes each: `kdatp-prof-20260929c` (20:29Z to 21:00Z),
  `kdatp-prof-20260929d` (20:55Z to 00:19Z) and `kdatp-prof-20260929e`
  (21:29Z to 23:47Z). All ended `rc 0`, and the cluster TTL removed each
  job at success. At most 2 kdatp jobs ran at the same time. No image was
  built, and no live run was changed. The harness was `dabe8a8972`
  (`examples/arena/harbor-rl-glm53-flash/kdatp/prof/`). S3 writes stayed
  under `s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/`.
- Round 20260929c reads of r47 were read-only (`get`, `logs`, `exec` of
  read commands, W&B history, NATS `jsz` and JetStream direct get, AMP
  DCGM). `verify2/loop.md` records one deviation: a per-thread CPU sample
  wrote two small temporary files in the r47 head pod `/tmp` for 20 s and
  deleted them.
- This record update, on miles `arpit-glm-53`: `27ab1bccdd` (fix(kdatp):
  the driver counts only the torch stop warning), `c2e17e5870`
  (docs(kdatp): the live Kineto result and the zero TTL in the prof
  README), and the docs(training-runs) commit that holds this text and the
  `INDEX.md` row.
- Not done in this update: the kernel-cache tar at the arm end (a harness
  change that belongs to Job Y and its go), and the `levers.py:11` fix to
  E[s^2] = 6.55e9 (a scratch file outside this record; section (g) applies
  the corrected values).

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
- Round 20260929c, under `/workplace/guparpit/kdfast/scratch/prof/20260929c/`:
  round reports `reports/harness.md`, `feasibility.md`, `live.md`,
  `precedent.md`, `prof-job.md`, `trace-analysis.md`, `arms-20260929d.md`,
  `arms-20260929e.md`, `synthesis.md`; verifier reports `verify2/ep8.md`,
  `verify2/tp4.md`, `verify2/dsa.md`, `verify2/loop.md` with their raw
  files (`verify2/ep8/`, `verify2/tp4/jobB-DCGM_*.json`, `verify2/dsa/`,
  `verify2/loop/`).
- Profile job: `run/driver.log`, `run/trainer-0.log` (`Max GPU buffer size:
  4096MB` x 64, 0 `Exceeded`, 0 torch stop warnings, `GPU stopped early? =
  0` x 64, `Processed N GPU records` 7,363,992 to 8,927,969),
  `run/results.json` (`perf 40` to `42`), `run/tb-sizes.txt` (64 traces,
  27,210,561,289 B), `run/traces/` (8 traces), `run/a2a/a2a.md`; S3
  `s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/prof/20260929c/`.
- Trace analysis: `analysis/trace/` (`extract.py`, `analyze.py`,
  `combine.py`, `pp.py`, `levers.py`, `tables.py`, `permb.py`;
  `combine.json`, `pp.json`, `levers.json`, `tables.md`,
  `rank<R>_res.json`, `rank48_ops.pkl`, `a2a_pairs.pkl`).
- Arms: `arms-20260929d/{ep8,base,pack}/trainer-0.log` (`perf 40` to `43`
  at l.17807/18210/18613 for `ep8` and l.15545/15934/16340 for `base`;
  step-40 `grad_norm` at `ep8` l.17383 and `base` l.15138),
  `arms-20260929d/results.json`,
  `arms-20260929e/{ep8pack,tp4ep8pack}/trainer-0.log` (`tp4ep8pack` `perf
  41` to `43` at l.15124/15662/16248; 576 and 768 `completes to compile`
  lines), `arms-20260929e/results.json`; S3
  `.../guparpit/kdatp/prof/20260929d/` and `.../20260929e/`. Control:
  `/workplace/guparpit/kdfast/scratch/prof/20260929a/run-20260929a/kdatp-prof/trainer-0.log`
  l.18553.
- Live r47: `live/r47-miles-config.yaml` (l.403 `expert_model_parallel_size`),
  `live/r47-argv.txt`, `live/r47-actor-logs/`, `live/r47-job-driver.log`,
  `live/amp/r47-fbused-max-by-pod.json`, `live/wandb/r47.history.json`;
  `verify2/loop/wandb_latest15.json`, `nats-msg-times.json`,
  `nats-msg-ids.json`, `fetch_per_node.json`,
  `pretrain_critical_verify.json`, `head-plasma-series.txt`,
  `bytes_per_rollout.json`, `per_token_verify.json`, `poll_wait.log`.
- Runtime code at miles `4716a367a` (git objects in
  `/workplace/guparpit/arena/src/miles` and the r47 pod copy):
  `glm5/ops/tilelang_sparse_mla_fwd.py:56,72,83`,
  `tilelang_sparse_mla_bwd.py:127-130,147`, `glm5_next/ops/kpool_indexer.py:233`,
  `glm5_next/dsa.py:11,108-117,136-137,189-190`,
  `miles/backends/megatron_utils/model.py:669-674`,
  `miles/utils/arguments.py:2766-2772`,
  `miles_plugins/arena/nats_rollout.py:2450-2504`,
  `train_async_arena.py:249-253`, `routing_replay.py:233`,
  `train_data_conversion.py:25,315-326`, `object_store.py:116-121`;
  image Megatron `transformer_config.py:2566-2598,3270-3274`,
  `transformer_block.py:733-772`, `optimizer.py:1928-1945,2023-2035`,
  `distributed_data_parallel.py:205-212`.
- External, round 20260929c: radixark/miles PR #3605 to #3608 (open;
  `verify2/dsa.md` 4.1), #2793 (open), #2786, #591, #2535; upstream miles
  main `scripts/run_glm5_3_flash.py` l.75-86 (`cc76e23915`) and
  `run_qwen3_30b_a3b.py` l.318-329; verl `agent_loop.py:131,796`; NeMo-RL
  `interfaces.py:81-89`; NVIDIA/nccl#2416.
