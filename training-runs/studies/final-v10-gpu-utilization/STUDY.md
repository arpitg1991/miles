# Study: where the GPU time goes on final-v10 (62jdk), 2026-10-08 22:30-22:40 UTC

Run: `acuadron-agentic-debt-final-v10-62jdk`, 16 actor + 48 engine nodes, batch 512, v9 engines
(trtllm DSA, flashinfer_cutlass, allreduce fusion, NEXTN 3/4, bf16 KV). Steps 1-2: 534 s and 568 s.
Sources: `nvidia-smi` on all 64 nodes (one 4 s sample each), the trainer perf keys, and the SGLang scheduler
logs read on two engine pods (`/tmp/ray/session_latest/logs/worker-*.err`, 3,600 prefill and 390 decode
log lines per engine over 18.6 min). Scripts: inline in the session; numbers below.

## Trainer (16 nodes): busy 51% of the wall

| step | step_time | actor_train | train_wait | update_weights |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2,462 s | 354 s | 2,098 s | 41 s |
| 1 | 534 s | 270 s | 264 s | 31 s |

The trainer trains 270 s and then waits 264 s for the next 64 groups: the step is rollout-bound. Inside the
270 s the PP4 pipeline leaves bubbles: one 4 s sample across the 16 nodes read 0-99% (mean 80%; w11 0%, w9
44%, w3 53%, the rest 83-99%).

## Engines (48 nodes): 81% kernel-busy, about half of it prefill for 5% of the tokens

Instantaneous `utilization.gpu`: mean 81% (30 nodes at 80-99%, 13 at 60-79%, 1 at 40-59%, 1 at 0-19%).

Scheduler state (per engine): `max_running_requests` derived to **48** (`sglang_max_running_requests` unset);
running pinned at 46-48 on every engine; **~10 requests queued**; KV pool 2.37M tokens at **52%** used; 512
groups x 8 / 48 = 85 trials in flight per engine, so ~27 are in tool execution or the verifier at any time.

Prefill interleave (engine worker-25; worker-40 identical within 10%):

| new tokens per prefill batch | batches | median cached tokens | interval since previous prefill |
| --- | ---: | ---: | ---: |
| 1-256 | 1,055 | 9,984 | 281 ms |
| 257-512 | 664 | 13,568 | 266 ms |
| 513-1,024 | 550 | 9,216 | 294 ms |
| 1,025-4,096 | 528 | 4,000-6,600 | 257-284 ms |
| 4,097-8,192 (cache-miss chunks) | 813 | 0 | 190 ms (back to back, pure compute) |

- **3.2 prefill batches per second per engine**, 87% with one request (`#new-seq: 1`), median 576 new tokens
  with 92% of the context served from the radix cache. Each is one agent turn: cached conversation + a few
  hundred tokens of tool output.
- Between prefills there are ~4.3 decode steps. Wall time per decode step is **71 ms** (390 lines x 40 steps
  over 1,116 s) against the ~30-40 ms pure decode step of the bench, so roughly half of the engine wall goes to
  prefill batches that carry ~5% of the tokens; the 8k cache-miss chunks (new trials and post-compaction
  restarts) take another ~14%.
- Decode itself is fine: accept length 2.67, 125 tokens per step, 2,000-2,100 tok/s logged per engine.

Why the small prefills are expensive and frequent:
1. SGLang admits prefills FCFS between decode steps, one arrival at a time, so a 45-layer MoE runs a full
   non-graphed forward for a few hundred tokens ~3 times a second.
2. `Glm5NextForConditionalGeneration` is on SGLang's `piecewise_cuda_graph_disabled_model_archs` list: no
   CUDA graph for prefill, so the small batch is launch-bound.
3. `--enable-mixed-chunk` (decode inside prefill batches) is refused together with speculative decoding.
4. `num_continuous_decode_steps` is 1 (default): the scheduler looks for prefills after every decode step.

## Levers (corrected 2026-10-08 23:10)

`num_continuous_decode_steps` is parsed by this SGLang's `server_args.py` and read by nothing else in
`sglang/srt/` (checked in the r23/r24 image): the knob is inert, which is also why the 2026-10-06 bench saw no
change from 2. Together with mixed chunk (refused with speculative decoding) and prefill CUDA graphs
(`Glm5NextForConditionalGeneration` is on `piecewise_cuda_graph_disabled_model_archs`), every scheduler-side
way to batch or overlap the small prefills is closed for this model in this build.

1. `sglang_max_running_requests: 64` absorbs the ~10 queued requests (KV at 52%); more concurrent turns also
   means more prefill interruptions, so expect a partial gain.
2. DP attention arms (`kdatp-sgl-20261008c/d`): eight schedulers per node spread the interruptions, but the
   MoE layers run in lockstep, so the gain is uncertain; the flashinfer-a2a variant collapsed speculation on
   `b98fw`.
3. More engines raise groups per hour, not per-GPU efficiency.
4. Agent side: fewer, larger turns per checkpoint attack the cause (3.2 prefill batches per second per engine).

Engine flags need a relaunch; the first checkpoint lands at step 9, the natural boundary.
