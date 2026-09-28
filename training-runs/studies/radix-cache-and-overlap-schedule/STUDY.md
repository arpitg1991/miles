# Study: SGLang radix cache and overlap schedule for the GLM-5.3-Flash gyms

**Date:** 2026-09-25 (record written 2026-09-28)
**Status:** Closed
**Question:** Do the SGLang radix cache and the overlap schedule make the GLM-5.3-Flash rollout engines faster without a change in policy drift, and why were both off?
**Runs and data used:** r33 `rl-glm53f33-2mjdm` and r34 `rl-glm53f34-2jbnl` (agentic-debt-cut; W&B `arena/rl-glm53f-adebt-cut` runs `4c2ttq0u`, `k5mb61z8`); r39 `rl-glm53f39-25pdn` and r42 `rl-glm53f42-5cthp` (auctioneer caponly-1034; W&B `arena/rl-glm53f-auct-cap` runs `j6todjkb`, `ob9qvkyg`); `s3://arena-scratch-prod-bom-ap-south-1/guparpit/debug/<experiment>/sample_summary/`; SGLang engine logs quoted in the `wf_f3b6585c-cb3` transcripts.
**Code SHAs:** miles `ed3dbdc9c`: r42 run record (memory note, not re-verified). miles `d28f0acf0` and `f01b8201f`: first `sglang_disable_overlap_schedule: true` in the 27B snorkel example, 2026-09-05 (memory note, not re-verified). AGISlime `aafbf37`: glm52-chakra-mini turned overlap on, 2026-08-17 (memory note, not re-verified). No git command ran for this study.
**Workflow ids:** `wf_f3b6585c-cb3` (2026-09-26, r39 vs r42 speed comparison; the user stopped it before any report; its agent transcripts hold the engine-log snapshots below). The 2026-09-25 vLLM comparison has no journal.

## Method

Terms. The *radix cache* is the SGLang prefix cache: a request whose prompt
starts with tokens the engine has seen reuses their KV state instead of a new
prefill. The *overlap schedule* runs the CPU scheduler work in parallel with
the GPU forward pass. *R3* is routing replay (miles ADR-0012; AREnATasks ADR-0064): the
trainer replays the expert routes the engine used. `extra_buffer` and
`no_buffer` are the two SGLang modes that let a hybrid Mamba model (GLM-5.3-Flash
is one) keep a radix cache; only `no_buffer` requires the overlap schedule off.

Two A/B pairs, each on prod-bom-v2 with 40 nodes per run, one TP8 SGLang
engine per engine node (`server_args` `tp_size=8, dp_size=1`), 288 gym pods,
R3 on, `arena_inflight_multiplier` 4:

- **r33 vs r34.** One delta: `sglang_disable_radix_cache` true (r33) vs false
  (r34). Diff of `training-runs/harbor-rl-glm53-flash/r33/miles-config.yaml`
  against `r34/miles-config.yaml`: lines 142-143 (names), 241 (radix), 506
  (sample summary dir). Both keep `sglang_disable_overlap_schedule: true`
  (line 178). Both launched 2026-09-21 (first rollout logged 21:08Z and
  20:58Z).
- **r39 vs r42.** Three deltas: radix on, overlap on, and workflow parameter
  `agent-timeout-multiplier` 1 -> 2 (2 h -> 4 h per episode). Diff of
  `r39/miles-config.yaml` against `r42/miles-config.yaml`: lines 218/226
  (overlap), 281/289 (radix); `workflow.yaml` line 36 (multiplier). Same
  dataset, gym image `gym-glm53-adr69-20260925a`, trainer image
  `miles-glm53-r12-20260925a`, template v7 (`r42/BUILD.md`).

Measurements:

- W&B RANK_0 history through `wandb.Api().run().history()` against
  `https://mega.wandb.agi.amazon.dev`, read on 2026-09-28. Rollout rows are
  ordered by `_step` and numbered from 0; the row counts (95, 66, 71, 59)
  equal the `rollout_N.jsonl` counts in each run's S3 `sample_summary/`
  prefix, so the ordinal is the rollout id. Train rows anchor on
  `train/ppo_kl`, stop-reason rows on `rollout/stop/unknown`.
- Wall-clock cadence from the `LastModified` time of the S3
  `sample_summary/rollout_N.jsonl` files.
- SGLang `server_args`, `Prefill batch` and `Decode batch` lines, and
  "exceeds the model's maximum context length" counts from the trainer pods,
  as captured on 2026-09-26 23:40-23:50Z in the `wf_f3b6585c-cb3`
  transcripts (`agent-af3abb156fd7ed11f.jsonl`, `agent-a4f859b457072ca96.jsonl`).
- `perf/rollout_time` in this asynchronous loop is the wait to assemble one
  batch of groups. It is not a per-request decode speed.

## Results

### Where the two flags were set

| Config | radix cache | overlap schedule | Source |
| --- | --- | --- | --- |
| Recipe `examples/arena/harbor-rl-glm53-flash/miles-config.yaml` | off (`:122`) | off (`:68`) | README `:278` "forced — recipe" for radix; `:328` "chosen — arena precedent (AGISlime run5 + smoke) ... candidate to lift later" for overlap |
| r9 (first GLM-5.3-Flash run) | off | off | `r9/miles-config.yaml:122`, `:68` |
| r33 | off | off | `r33/miles-config.yaml:241`, `:178` |
| r34 | on | off | `r34/miles-config.yaml:241` |
| r39 | off | off | `r39/miles-config.yaml:281`, `:218` |
| r42 | on | on | `r42/miles-config.yaml:289`, `:226` |
| r43, r45 (r34 lineage) | on | off | `r43/miles-config.yaml:272`, `:209`; `r45/miles-config.yaml:298`, `:232`, comment line 23 |
| r44, r46, r47 | on | on | `r44:322`, `:256`; `r46:330`, `:264`; `r47:351`, `:275`, header comment lines 16-21 |

The overlap flag had no recorded reason. Every arena SGLang example sets it
off (`examples/arena/harbor-rl-27b-snorkel/snorkel-27b.yaml:65`,
`smoke-3node/miles-config.yaml:32`, `harbor-rl-27b/financeagent-27b-smoke.yaml:35`),
and the GLM recipe copied it. `RUNLOG.md:1086` lists "overlap schedule" among
the AGISlime glm52 engine settings. The memory note adds that AGISlime turned
it on in glm52-chakra-mini with the comment "+15% single-stream decode ...
does not change numerics"; the local AGISlime checkouts hold no
`experiments/k8s/glm52` directory, so that part is not re-verified.

SGLang accepted both flags together. r42 engine `server_args`:
`disable_radix_cache=False`, `disable_overlap_schedule=False`,
`mamba_backend='triton'`, `mamba_radix_cache_strategy='extra_buffer'`
(resolved from `auto`). r39: `disable_radix_cache=True`,
`disable_overlap_schedule=True`, strategy left at `'auto'`. r47 shows the same
`extra_buffer` resolution (`RUNLOG.md:2391`). The source-level claims in the
memory note (only MPS forces overlap off; only `no_buffer` requires it off;
GLM-5.3-Flash is in `_MAMBA_EXTRA_BUFFER_ARCHS`; SGLang 0.5.19.dev52) are not
re-verified: no SGLang checkout exists on this host.

### r33 vs r34: radix cache alone

Whole-run medians reproduce the recorded numbers exactly: `perf/rollout_time`
1130 s (r33, n=95) vs 534.7 s (r34, n=66). The split by day shows where the
gap sits.

| UTC day | r33 rollout_time | r33 tok/GPU/s | r33 episode len | n | r34 rollout_time | r34 tok/GPU/s | r34 episode len | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 09-21 | 254 | 313 | 40.8k | 6 | 263 | 300 | 41.9k | 6 |
| 09-22 | 417 | 329 | 82.5k | 36 | 461 | 331 | 92.3k | 35 |
| 09-23 | 3506 | 68 | 150k | 22 | 880 | 337 | 191k | 19 |
| 09-24 | 2800 | 88 | 168k | 22 | 16000 | 24 | 258k | 5 |
| 09-25 | 685 | 342 | 155k | 9 | 25710 | 15 | 272k | 1 |

Source: W&B `4c2ttq0u`, `k5mb61z8` (`perf/rollout_time`,
`perf/tokens_per_gpu_per_sec`, `rollout/episode_response_length/mean`).

| Matched rollout idx | r33 rollout_time | r34 rollout_time | r33 tok/GPU/s | r34 tok/GPU/s | r33 len | r34 len |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (cold start) | 3343 | 2063 | 9.6 | 16.8 | 18.4k | 18.8k |
| 1 | 1563 | 675 | 35.8 | 77.7 | 31.0k | 30.9k |
| 2-20 (median) | 283 | 301 | 338 | 317 | 63.5k | 63.1k |
| 21-40 | 459 | 500 | 306 | 333 | 91.8k | 103k |
| 41-65 | 3287 | 913 | 69 | 330 | 150k | 219k |
| 66-94 (r33 only) | 2531 | - | - | - | 167k | - |

Cadence and training cost:

| Metric | r33 | r34 | Source |
| --- | --- | --- | --- |
| Rollouts in the first 48 h | 61 | 59 | W&B `_timestamp` |
| Median gap between `rollout_N.jsonl` files, 09-21 / 09-22 / 09-23 | 1610 / 2092 / 3878 s | 1743 / 2332 / 4416 s | S3 `sample_summary/` |
| `perf/train_time`, steps 1-20 / 21-40 / 41-64 | 1667 / 2205 / 3257 s | 1749 / 2485 / 4523 s | W&B |
| `perf/update_weights_time`, steps 1-64 | 24.4-24.5 s | 24.3-24.7 s | W&B |
| Groups lost to the NATS 32 MiB record cap, 24 h window on 09-24/25 | - | 58 of 92 (63%) | memory note `nats-max-payload-redelivery-loop`, not re-verified here |

Policy drift with R3 (`train/*`, medians):

| Steps | r33 `train_rollout_kl` | r34 | r33 `logprob_abs_diff` | r34 | r33 `tis_clipfrac` | r34 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.00528 | 0.00223 | 0.0381 | 0.0206 | 9.0e-4 | 3.4e-4 |
| 1-20 | 0.00666 | 0.00401 | 0.0461 | 0.0335 | 1.16e-3 | 7.0e-4 |
| 21-40 | 0.00753 | 0.00323 | 0.0502 | 0.0317 | 1.59e-3 | 3.9e-4 |
| 41-64 | 0.00561 | 0.00492 | 0.0485 | 0.0422 | 8.5e-4 | 7.5e-4 |

`train/ppo_kl` stays at 1e-6 scale in both. Source: W&B `4c2ttq0u`, `k5mb61z8`.

`rollout/prefix_cache_hit_rate` and `rollout/avg_cached_tokens_per_sample`
read 0 in every rollout of all four runs, radix on or off. The Harbor NATS
path does not feed these two metrics. Cache hits are visible only in the
engine `Prefill batch ... #cached-token` lines.

### r39 vs r42: radix cache, overlap schedule, and the 4 h limit together

| Rollout idx | r39 rollout_time | r42 rollout_time | r39 reward | r42 reward | r39 len | r42 len | r39 tok/GPU/s | r42 tok/GPU/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (cold start) | 5746 | 1738 | 0.385 | 0.385 | 15.9k | 16.0k | - | - |
| 1 | 363 | 121 | 0.386 | 0.370 | 17.5k | 18.5k | - | - |
| 1-29 (median) | 145 | 135 | 0.392 | 0.371 | 20.7k | 19.9k | 365 | 380 |
| 30-49 | 290 | 429 | 0.466 | 0.557 | 77.9k | 102k | 378 | 357 |
| 50-58 | 667 | 712 | 0.205 | 0.037 | 121k | 94.6k | 340 | 363 |
| 59-70 (r39 only) | 689 | - | 0.138 | - | 113k | - | - | - |

Source: W&B `j6todjkb`, `ob9qvkyg`. Rollouts in the first 24 h: r39 54, r42
50; first 36 h: 64 vs 59.

| Train metric | r39 | r42 | Source |
| --- | --- | --- | --- |
| Step 0 `perf/step_time` / `train_wait_time` / `train_time` | 10120 / 6321 / 3802 s | 5820 / 2293 / 3527 s | W&B |
| Steps 1-29 `step_time` / `train_time` / `train_wait_time` | 874 / 800 / 65 s | 866 / 796 / 61 s | W&B |
| Steps 1-29 `update_weights_time` | 24.9 s | 24.3 s | W&B |
| Step 0 `train_rollout_kl` / `logprob_abs_diff` / `tis_clipfrac` / `ppo_kl` | 0.00674 / 0.0509 / 1.0e-3 / -1.9e-6 | 0.00372 / 0.0338 / 4.9e-4 / 6.8e-7 | W&B |
| Steps 1-29 `train_rollout_kl` / `logprob_abs_diff` | 0.0181 / 0.0808 | 0.0149 / 0.0753 | W&B |
| Steps 30-49 `train_rollout_kl` / `logprob_abs_diff` | 0.0131 / 0.0756 | 0.0217 / 0.0987 | W&B |

Stop reasons (`rollout/stop/<reason>`, mean share per range; a missing key
counts as 0; `unknown` is the normal end of an episode):

| Rollout idx | r39 `timeout` | r42 `timeout` | r39 `context_error` | r42 `context_error` | r39 `unknown` | r42 `unknown` |
| --- | --- | --- | --- | --- | --- | --- |
| 0-29 | 0.079 (nonzero in 22 of 30; max 0.277) | 0.000 | 0.000 | 0.000 | 0.921 | 1.000 |
| 30-49 | 0.718 (20 of 20) | 0.001 (2 of 20, 1 episode each) | 0.001 | 0.375 | 0.281 | 0.624 |
| 50-58 | 0.555 | 0.000 | 0.373 | 0.714 | 0.072 | 0.286 |
| 59-70 (r39) | 0.008 | - | 0.801 | - | 0.190 | - |

Context overflow did not change. SGLang "Requested token count exceeds the
model's maximum context length of 131072" lines: r39 engine stderr on
worker-1, 417 in the run to 2026-09-26 23:50Z; r42, 385. Driver log on
worker-0, last 2 h at 23:50Z: r39 536, r42 700 (Ray-deduplicated lines).
`RUNLOG.md:2109-2111` records about 500 such requests per 90 min in both
runs, 99% of them Vulcan summary calls. Both runs collapsed and were retired
2026-09-27 05:02Z and 05:05Z (`RUNLOG.md:2142-2156`).

### Engine snapshots

| Time (UTC) | Run | Engine log line | Source |
| --- | --- | --- | --- |
| 2026-09-25 16:08:07 | r39 | `Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.97, #running-req: 46` | `/tmp/r39-full.log` quoted in `wf_f3b6585c-cb3` `agent-a4f859b457072ca96.jsonl` |
| 2026-09-26 23:40 | r39 | `Decode batch, #running-req: 11-12, full token usage: 0.27-0.30, gen throughput (token/s): 251-732`; prefill `#cached-token: 0` | `agent-af3abb156fd7ed11f.jsonl` |
| 2026-09-26 23:40 | r42 | `Prefill batch, #new-token: 320, #cached-token: 55552` (99.4% of a 55872-token prompt); `#new-token: 1856, #cached-token: 8448`; `Decode batch, #running-req: 5-23, full token usage: 0.16-0.47, gen throughput (token/s): 466.79` | same |

The 2026-09-25 about 16:45Z comparison against the inference-hosting vLLM
GLM-5.3-Flash deployment exists only in the memory note
`sglang-speed-vs-inference-hosting-2026-09-25`; its scratch files are
deleted, so every number here is **not re-verified**:

| Request shape | r39 SGLang router | inference-hosting vLLM (idle replica) |
| --- | --- | --- |
| 630-token prompt, 600 output tokens, sequential | TTFT 29 s, 16 tok/s | TTFT 0.04 s, 238 tok/s |
| Same, 8 concurrent | TTFT 6 s, 21 tok/s | TTFT 0.43 s, 190 tok/s |
| 55k-token prompt, run 1 / run 2 (same prompt) | TTFT 2.3 s / 11.1 s | TTFT 7.0 s / 0.21 s (prefix hit) |

At that time the r39 engines read KV usage median 0.97, about 51 running
requests and about 890 generated tok/s per engine (about 17 tok/s per
request). The vLLM deployment ran fp8 KV, `gpu-memory-utilization 0.95`,
MTP speculative decoding, `max-num-seqs 64`, and was idle.

## Verdict

Radix cache on plus overlap schedule on is safe for GLM-5.3-Flash on this
SGLang branch: the engines resolve `extra_buffer`, weight sync stays at about
24 s, `train/ppo_kl` stays at 1e-6 scale, and the R3 mismatch metrics are
equal or lower in the "on" arm of both pairs. The speed gain is large only
where the engines are saturated or the prompts are long: the cold-start
rollout (3.3x for r42 vs r39, 1.6x and 2.3x for r34 vs r33 at rollouts 0 and
1) and the long-episode regime of 2026-09-23 (r33 fell to 68 tok/GPU/s at
150k-token episodes while r34 held 337 at 191k). In the steady state of the
first 40 rollouts the two arms are equal within noise on rollout time,
tokens per GPU per second, step time, and rollouts per day, because the loop
waits on training (800-3300 s per step) and not on rollout assembly (130-500
s). The headline "1130 s -> 535 s (2.1x)" is a whole-run median that mixes
these regimes; it is correct arithmetic, but almost all of the gap sits in
the 09-23/09-24 window, and r34 stopped at the NATS wall before its own long
episodes could count. The near-zero timeout share of r42 cannot be split
between the cache and the 4 h limit; the episodes that no longer time out
end on the 131072-token window instead.

## Caveats and open items

- One run per arm. r33 and r34 ran on different nodes of the same cluster.
- The r33 slowdown on 09-23/09-24 (tok/GPU/s 329 -> 68 -> 88, back to 342
  on 09-25) is attributed to the missing cache at long contexts because r34
  held 337 with longer episodes. No other cause (node contention, gym mount
  loss, the r33 gym pods) was checked.
- r34 lost 63% of its groups to the NATS 32 MiB record cap on 09-24/25
  (memory note); its rows after 09-24 are 6 of 66. The r33/r34 comparison is
  clean only through 2026-09-23.
- r39 vs r42 changed three settings at once. Per-request tok/s under load was
  not measured for r42 beyond the single 23:40Z snapshot; the comparison
  workflow `wf_f3b6585c-cb3` produced no report.
- W&B carries no cache-hit series on the Harbor path
  (`rollout/prefix_cache_hit_rate` is always 0); a per-rollout hit rate needs
  the SGLang `#cached-token` lines or engine metrics.
- The 2026-09-25 vLLM comparison table and the SGLang source-level statements
  are from the memory note only: not re-verified.
- Commit SHAs in the header come from memory notes: not re-verified.
- r34 has two RANK_0 W&B runs: `8nbpn7iu` (created 19:52:56Z, no rows) and
  `k5mb61z8` (20:09:08Z, all data). The first attempt logged nothing.
- The recipe `examples/arena/harbor-rl-glm53-flash/miles-config.yaml` still
  sets both flags off (`:68`, `:122`), and README `:328` still calls overlap
  "candidate to lift later". A run generated from the recipe instead of from
  r42 or later inherits the old values.

## Actions taken

- r42 launched 2026-09-25 18:06:16Z with radix on, overlap on, and
  `agent-timeout-multiplier` 2; r39 kept for the comparison (`r42/BUILD.md`).
- r44 (2026-09-27), r46, and r47 keep both flags on; r47 lifted the overlap
  flag from the r45 lineage with a header comment that names this reason
  (`r47/miles-config.yaml:16-21`). r43 and r45 keep radix on, overlap off.
- r39 and r42 retired 2026-09-27 with `shutdown: Stop`; last checkpoints
  `iter` 59 and 49 (`RUNLOG.md:2142-2156`).
- No ADR and no recipe change. The recipe and README rows above are the open
  follow-up.

## Sources

- `training-runs/harbor-rl-glm53-flash/r33/miles-config.yaml`, `r34/miles-config.yaml`, `r39/miles-config.yaml`, `r42/miles-config.yaml`, `r42/workflow.yaml`, `r42/BUILD.md`, `r39/BUILD.md`, `r43/`, `r44/`, `r45/`, `r46/`, `r47/miles-config.yaml`
- `training-runs/harbor-rl-glm53-flash/RUNLOG.md` lines 1086, 2099-2111, 2142-2156, 2387-2391
- `examples/arena/harbor-rl-glm53-flash/README.md` lines 278, 328, 443, 550; `examples/arena/harbor-rl-glm53-flash/miles-config.yaml` lines 68, 122
- W&B `arena/rl-glm53f-adebt-cut` runs `4c2ttq0u` (r33), `8nbpn7iu` and `k5mb61z8` (r34); `arena/rl-glm53f-auct-cap` runs `j6todjkb` (r39), `ob9qvkyg` (r42); host `https://mega.wandb.agi.amazon.dev`
- `s3://arena-scratch-prod-bom-ap-south-1/guparpit/debug/rl-glm53f-adebt-cut-r33/sample_summary/` (95 files), `.../rl-glm53f-adebt-cut-r34/sample_summary/` (66), `.../rl-glm53f-auct-cap-r39/sample_summary/` (71), `.../rl-glm53f-auct-cap-r42/sample_summary/` (59); `.../checkpoints/slime_experiments/rl-glm53f-adebt-cut-r33/iter_0000089`, `.../rl-glm53f-adebt-cut-r34/iter_0000059`
- Workflow `wf_f3b6585c-cb3` transcripts `agent-a4f859b457072ca96.jsonl` (timing) and `agent-af3abb156fd7ed11f.jsonl` (server args, engine lines, overflow counts)
- Memory notes `sglang-speed-vs-inference-hosting-2026-09-25`, `r42-radix-overlap-timeout-2026-09-25`, `r39-auctioneer-caponly-1034-2026-09-25`, `nats-max-payload-redelivery-loop` (notes; numbers marked not re-verified where no primary source was read)
