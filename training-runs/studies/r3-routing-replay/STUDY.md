# Study: R3 rollout routing replay for GLM-5.3-Flash on the NATS gym path

**Date:** 2026-09-25
**Status:** Closed
**Question:** Does a replay of the SGLang expert routing in the Megatron forward and backward pass (R3, arXiv 2510.11370, miles arena ADR-0012) remove the step-0 train/inference mismatch of the GLM-5.3-Flash runs, and what does its payload cost on the NATS gym path?
**Runs and data used:** r13 `rl-glm53f-gbash-r13` (W&B `jnbq9gub`) and r14 `rl-glm53f-gbash-r14` (W&B `tor7hf5u`) as the pre-R3 baseline; r16 `rl-glm53f16` (W&B `qp51gz7c`, points 0-15; the crashed first attempt is W&B `ocpsk0ip` and holds no train metrics); r17 `rl-glm53f17` (W&B `qp51gz7c`, points 16-18, the run resumed from r16); r18 `rl-glm53f18` and its resume `rl-glm53f18r-gzc97` (W&B `fzl7n8vw`, 50 steps); r19 (8 MiB payload loop); r27 and r28 (lost routing ref); r34 `rl-glm53f34-2jbnl` (32 MiB store cap); r45 (payload size estimate). W&B project `arena/rl-snorkel27` on `https://mega.wandb.agi.amazon.dev`. Routing files lived under `/mnt/scratch-s3files-rw/guparpit/routing/<experiment>`; the trainer deletes each file after use, so nothing remains to read back.
**Code SHAs:** miles `3903e8f92` (ADR-0012 text), `a9951bb59` (R3 over NATS: `capture_routed_experts`, drain-time materialization, `routing_replay.py`), `70337a951` (pad rows -1, not 0), `969931b9e` and `805d7b382` (r16 image and manifests), `fc9847ab7` (r17: TIS off, rollout log-probs on), `ece3378e7` (r18: TIS back on), `03791ce86` (accepted-tid ownership rule and `RoutingRefLostError` after r27 and r28), `7cec2a4b8` (r29 on the trainer image with that fix), `dd5e67391` (token arrays by file reference, ADR-0014), `db3955b70` (drop-reason categories); fork pre-history `cee16d9d0`, `0dd04cc2a`, `4ac3f4b7f` (2026-08-27: indexer-replay guard, all-zero payload check, conversion-time check). AREnATasks `e266b87` (gym image `gym-glm53-r3-20260909a`, per `r16/BUILD.md`), `0bd5e57` (streaming `routing.py`, the ref protocol), `f0472e9` (token arrays staged as files, ADR-0071; from the memory note `nats-token-arrays-by-ref-2026-09-25`).
**Workflow ids:** `wf_545a45a0-6ca` (MFU study: 49 GB per DP shard estimate, R3 fill time), `wf_3ff0709b-51c` (fork reconcile: R3 math unchanged upstream; `triton` versus `auto`), `wf_58a8a79c-51c` (first-port KDA: replay cursors and recompute). From memory notes only, journals not opened: `wf_67beac2c-2fb` (r34 evidence), `wf_4b7e7dec-d14` (claim-check precedent), `wf_52efcae6-e11` (NATS file-store limit sources), `wf_a714d36a-283` and `wf_1f57d4a1-353` (token-ref build and integration rerun).

## Method

Period: 2026-09-09 (r16 launch) to 2026-09-25 (ADR-0014). All runs on 40
`p6-b200.48xlarge` nodes in `arena-prod-bom-v2`, namespace `arena-tasks`,
GBS 512 (64 prompts x 8 samples), 288 gym pods, `arena_train_segments all`.
r13, r14 and r16 split the nodes 8 actor + 32 engine (TP8 PP4 EP16, DP 2;
32 SGLang engines at TP8). r17 and r18 split 16 + 24 (`r17/BUILD.md`).

### What `train/ppo_kl` measures

`train/ppo_kl` is the mean of `old_log_probs - log_probs` over loss-mask
tokens (`miles/backends/training_utils/loss_hub/losses.py:185`). One flag
selects `old_log_probs` (`losses.py:133-136`):

- `use_rollout_logprobs: true`: the SGLang log-prob of each sampled token.
  `ppo_kl` is then the SGLang-versus-Megatron gap.
- `use_rollout_logprobs: false`: the actor's own no-grad `compute_log_prob`
  pass, which `train_actor` runs only in this mode
  (`miles/backends/megatron_utils/actor.py:574-590`). `ppo_kl` is then the
  gap between two Megatron passes of the same weights.

`use_tis` and `use_rollout_logprobs` exclude each other
(`miles/utils/arguments.py:3195`). The same metric name therefore measures
two different gaps, and the arm table below states which one each run
logged. `train/pg_clipfrac` is the fraction of tokens whose PPO ratio left
`[0.8, 1.28]`. `train/tis` and `train/tis_clipfrac` are the mean of
`clip(pi_train / pi_rollout, 0, 2)` and the clipped fraction (TIS runs only).

| Arm | `use_rollout_routing_replay` | `use_tis` | `use_rollout_logprobs` | `old_log_probs` source | Source |
| --- | --- | --- | --- | --- | --- |
| r13, r14 | false | false | true | SGLang | `r13/miles-config.yaml`, `r14/miles-config.yaml`; W&B config `jnbq9gub`, `tor7hf5u` |
| r16 | true | true | false | actor no-grad pass | `r16/miles-config.yaml:270-271,286` |
| r17 | true | false | true | SGLang | `r17/miles-config.yaml:275,278,293`; W&B config `qp51gz7c` |
| r18 | true | true | false | actor no-grad pass | `r18/miles-config.yaml:285,290,305`; W&B config `fzl7n8vw` |

### Data pull

W&B pulls ran on 2026-09-28 through
`wandb.Api().run("arena/rl-snorkel27/<id>").history(keys=[k], samples=5000)`.
`scan_history` fails from this host with an S3 403 on the parquet export.
Point counts equal the step counts of each run, so the sampled path returned
every step. The `_step` axis is the W&B step (6 W&B steps per train step).
The r16 and r17 metrics share one W&B run because r17 warm-started from the
r16 `iter_0000014` checkpoint and resumed the run id; the 16 points with a
`train/tis` value are r16, the 3 points without are r17.

### The harness

Trainer (this repo): `miles_plugins/arena/nats_arena/message_format.py`
emits `capture_routed_experts` on every task message (T1);
`nats_rollout.py` stashes the `{path, bytes, sha256}` ref on
`Sample.metadata` and materializes a group at drain time (T2, T3);
`miles_plugins/arena/nats_arena/routing_replay.py` holds `decode_routing`,
`materialize_group_routing`, `pad_routing` (`PAD_EXPERT = -1`, lines
310-324), `reap_result_refs` and, since ADR-0014, `resolve_token_arrays`.
`replay_data.fill_replay_data` feeds per-layer replay queues; the Megatron
`TopKRouter` hook lives in `radixark/Megatron-LM` branch `miles-main`
(ADR-0012, "How upstream R3 works"). Gym (AREnATasks ADR-0064): the capture
layer snapshots the routing blob per compaction segment (G1), the Harbor
worker stages it with `stage_routed_experts` under `ARENA_ROUTING_DIR` (G2),
and both pod kinds mount `/mnt/scratch-s3files-rw` (G3;
`r16/gym-worker.yaml:292`, `r16/trainer-pytorchjob.yaml:381`).

## Results

### 1. Mismatch metrics by arm

| Arm | Metric | Value | Source |
| --- | --- | --- | --- |
| r13 | `train/ppo_kl` steps 0-2 | 7.99e-03, 8.09e-03, 8.29e-03; 30 steps: min 7.99e-03, median 9.55e-03, max 1.54e-02 | W&B `jnbq9gub` |
| r13 | `train/pg_clipfrac` | 1.10e-02 at step 0; median 1.22e-02 | W&B `jnbq9gub` |
| r14 | `train/ppo_kl` steps 0-2 | 7.96e-03, 8.24e-03, 7.98e-03; 20 steps: median 8.92e-03, max 1.25e-02 | W&B `tor7hf5u`; memory note `glm53-r3-routing-replay-r16` (0.00796) |
| r14 | `train/pg_clipfrac` | 1.10e-02 at step 0; median 1.16e-02 | W&B `tor7hf5u` |
| r16 | `train/ppo_kl` step 0 | 1.82e-05 | W&B `qp51gz7c` `_step` 7; RUNLOG:1691; ADR-0012:187 |
| r16 | `train/pg_clipfrac` step 0 | 2.57e-04 | W&B `qp51gz7c`; RUNLOG:1691 |
| r16 | `train/tis`, `train/ess_ratio` step 0 | 1.000, 0.999 | W&B `qp51gz7c`; RUNLOG:1699-1700 |
| r16 | `train/ppo_kl` steps 0-15 | min -2.84e-05, max 7.60e-05, 8 of 16 points negative | W&B `qp51gz7c` |
| r16 | `train/pg_clipfrac` steps 0-15 | 2.57e-04 to 3.59e-04 | W&B `qp51gz7c` |
| r16 | `train/tis_clipfrac` steps 0-15 | 1.21e-03 to 1.47e-03 | W&B `qp51gz7c`; `r44/miles-config.yaml:437` ("0.1%") |
| r17 | `train/ppo_kl` steps 15-17 | 6.18e-03, 6.52e-03, 6.11e-03 | W&B `qp51gz7c` `_step` 110, 116, 122; `r18/BUILD.md` (0.0062) |
| r17 | `train/pg_clipfrac` steps 15-17 | 2.03e-02 to 2.11e-02 | W&B `qp51gz7c`; `r18/BUILD.md` (2%) |
| r18 | `train/ppo_kl` 50 steps | first 8.62e-07; min -1.63e-04 (one point); median -4.11e-06; max 1.63e-05 | W&B `fzl7n8vw` |
| r18 | `train/pg_clipfrac` 50 steps | 2.08e-04 to 3.15e-04 | W&B `fzl7n8vw` |
| r18 | `train/tis`, `train/tis_clipfrac` 50 steps | 0.922 to 1.000; 9.9e-04 to 1.57e-03 | W&B `fzl7n8vw` |

### 2. Like-for-like comparisons

| Gap | Before R3 | After R3 | Change | Source |
| --- | --- | --- | --- | --- |
| SGLang versus Megatron (`use_rollout_logprobs: true`) | r13, r14 steps 0-2: 7.96e-03 to 8.29e-03 (medians 8.9e-03 to 9.6e-03 over the run) | r17 steps 15-17: 6.11e-03 to 6.52e-03 | -20% to -35% | W&B `jnbq9gub`, `tor7hf5u`, `qp51gz7c` |
| Megatron no-grad pass versus Megatron training forward (TIS on) | none: r15 (TIS, no R3) was deleted before a comparable step | r16, r18: 1e-05 scale, sign straddles zero | not attributable to R3 | RUNLOG:1656-1658; W&B `qp51gz7c`, `fzl7n8vw` |

### 3. r16 per step

| Step | Logged UTC | `ppo_kl` | `pg_clipfrac` | `grad_norm` | Step wall | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 16:49 | 1.8e-05 | 2.6e-04 | 0.099 | 54 min (first TileLang JIT of `sparse_mla_bwd_kernel`) | RUNLOG:1691, 1705-1706; W&B `qp51gz7c` |
| 1 | 17:24 | 3.5e-06 | 2.9e-04 | 0.082 | 35 min | RUNLOG:1692 |
| 2 | 17:50 | -1.3e-05 | 3.2e-04 | 0.095 | 26 min | RUNLOG:1693 |
| 3 | 18:17 | -9.7e-06 | 3.0e-04 | 0.105 | 26 min | RUNLOG:1694 |
| 4 | 18:46 | -1.4e-05 | 3.0e-04 | 0.135 | 29 min | RUNLOG:1695 |
| 5 | 19:42 | -1.4e-07 | 3.1e-04 | 0.084 | 28 min | RUNLOG:1696 |

Rewards for rollouts 0-7: 0.63, 0.57, 0.50, 0.55, 0.55, 0.55, 0.54, 0.56;
queue 321 groups by rollout 7, so the trainer was the bottleneck
(RUNLOG:1703-1706). r16 spent 22 of a 28 min step in `actor_train`; the TIS
old-log-prob pass cost about 5 min per step (`r17/BUILD.md`).

### 4. Payload sizes

| Quantity | Value | Source |
| --- | --- | --- |
| Routed experts per token | 1,440 B = 45 layers x top-8 x int32; `Sample.rollout_routed_experts` shape `(len(tokens) - 1, num_layers, moe_router_topk)` | ADR-0012:78 and "How upstream R3 works" |
| Per 40k-token episode | 58 MB; base64 adds one third on the wire; gzip does not help | ADR-0012:78-81 |
| Per GBS-512 step | 29 GB into pinned CPU on each actor rank | ADR-0012:78, 182 |
| Observed per 11k-token episode | 16 MB | memory note `glm53-r3-routing-replay-r16`; not re-verified (11k x 1,440 B = 15.8 MB agrees) |
| Indexer top-k per token | 90,112 B; 3.6 GB per episode; 1.8 TB per step; skipped (decision 1) | ADR-0012:79, 85-89 |
| Materialize 64 groups (512 samples) at drain, nfs4 | 74 s / 75 s | ADR-0012:203; RUNLOG:1672 |
| r45 step s42 routing per DP shard | about 49 GB = 36.5M tokens x 1,344 B (`[tokens, 42, 8]` int32); inferred from code shapes, not measured | `wf_545a45a0-6ca` |
| r45 R3 fill on CPU per step | 50-71 s, after a 92-144 s DP-shard object-store fetch on the last PP stage; GPUs idle for 170-241 s between steps | `wf_545a45a0-6ca` |
| NATS `max_payload` r16-r18 | 8 MiB (`8388608`) | `r16/nats.yaml:15` |
| Token file per token (ADR-0014) | 13 B (`float64` log-prob, `int32` id, `uint8` mask); about 1% of the routing bytes | ADR-0014:62, 182 |

The layer count differs between sources: ADR-0012 counts every decoder layer
(45, dense layers included, 1,440 B per token); `wf_545a45a0-6ca` used the
42 MoE layers (1,344 B per token). At 1,440 B the same r45 shard is about
52 GB. Neither figure was measured.

### 5. Payload incidents that the ref protocol handled or forced

| Date | Run | Event | Source |
| --- | --- | --- | --- |
| 2026-09-09 | r16 first attempt | All-zero pad routing selected expert 0 eight times per token; `routing_map` collapsed the duplicates, the dropless dispatcher sized the all-to-all for `tokens * topk` rows: `RuntimeError: Split sizes doesn't match total dim 0 size` in `compute_log_prob`. Fix: -1 rows, which `replay_base._get_replay_result` rewrites to `arange(topk) % num_experts`. Image `20260909b`. | RUNLOG:1670-1680; `70337a951`; `routing_replay.py:310-324`; ADR-0012:190-197 |
| 2026-09-12 | r19 | 100k-token groups gzip to 11-13 MB; 8 MiB `max_payload` plus `max_deliver=-1` gave an endless redelivery loop; 234 of 288 gym pods showed the error, 76% of ack-pending tasks were redeliveries; live patch to 64 MiB and a NATS restart | memory note `nats-max-payload-redelivery-loop`; not re-verified, RUNLOG has no 2026-09-12 entry |
| 2026-09-15 | template v4 | `max_payload` 64 MiB in the deployer (AREnATasksApps `5a0baa1`) | RUNLOG:1853-1855 |
| 2026-09-20 | r27, r28 | The trainer deleted its own routing files: a NATS reconnect redelivered an accepted result, the stale-result path reaped the refs, and the queued group failed hours later with `RoutingReplayError` | `03791ce86` body; ADR-0014 "Cleanup" |
| 2026-09-25 | r34 | 58 of 92 groups lost at the 32 MiB JetStream file-store record cap (`filestore.go` `rlBadThresh`), not a setting; heaviest group 33.5 MB gzip, 109 MB JSON, 63 segments, 3,975,555 tokens; `log_probs` 61%, `token_ids` 20%, `loss_mask` 11%, `messages` 8.5% | ADR-0014:29-40; AREnATasks ADR-0071 context |
| 2026-09-25 | design | Token arrays ride a `.tokens` file ref through the same reader, containment check and reap; a bad file drops one trajectory, not the run; the gym fails closed over 30 MiB gzip | ADR-0014; `dd5e67391`; AREnATasks ADR-0071 |

## Verdict

R3 works as a mechanism and stays on in every run since r16, but the
headline number overstates it. The 400x drop of `train/ppo_kl` from 0.008 to
1.8e-05 (RUNLOG:1698-1699; ADR-0012:187-189; `r44/miles-config.yaml:148`)
compares two different quantities. r13 and r14 ran with
`use_rollout_logprobs: true`, so their `ppo_kl` is the SGLang-versus-Megatron
gap. r16 ran with TIS, so its `ppo_kl` is the gap between two Megatron passes
of the same weights. That number is bf16 forward noise: it straddles zero in
r16 (8 of 16 points negative), in r18 (median -4.1e-06 over 50 steps) and in
r27 (memory note `glm53-ppo-kl-is-numerical-noise`). The only like-for-like
R3 measurement is r17, which put the SGLang log-probs back as
`old_log_probs` with R3 on: 6.1e-03 to 6.5e-03 against 8.0e-03 at r13 and
r14 steps 0-2. R3 removes about one quarter of the SGLang-versus-Megatron
gap. Route flips were therefore not the dominant mismatch term. About 6e-03
remains from sources this study did not separate: the DSA indexer top-k that
R3 does not replay, attention and MLA kernel differences, the engine
`moe_runner_backend` fallback to `auto`, and sampling-time numerics. The
case for R3 beyond that number is the upstream MoE recipes and the mechanism
itself: the actor now trains on the experts that produced the tokens.

TIS corrects little on this path (`train/tis` 0.92 to 1.00,
`train/tis_clipfrac` about 0.1%), but the actor no-grad pass that `use_tis`
forces sets the PPO ratio against the actor's own policy, so `eps_clip` is
inert (`pg_clipfrac` 3e-04). r17 showed the alternative: with the SGLang
log-probs as `old_log_probs` the PPO clip was live (`pg_clipfrac` 2%) and
the update was an importance-corrected step against the SGLang
distribution. The team kept the r18 form, the upstream GLM-5.2 recipe.

The payload cost is real and grows with the token count: 1,440 B per token,
29 GB per GBS-512 step, 50-75 s of CPU fill per step, and an inferred 49 to
52 GB per DP shard on the r45 shape. The pad rule (-1, never 0), the
drain-time materialization and the accepted-tid ownership rule are correct
and necessary. The `{path, bytes, sha256}` ref protocol that R3 forced
became the template that fixed the 32 MiB NATS result cap (ADR-0014).

## Caveats and open items

- The r17 versus r13/r14 comparison is not a clean A/B: r17 started from the
  r16 step-14 weights, used the curriculum manifest with
  `rollout_shuffle: false`, ran on a 16 + 24 node split, and logged 3 points.
  r15 (TIS without R3) was deleted before it produced a comparable step. A
  clean test is one run with `use_rollout_logprobs: true` and R3 off on the
  current stack, or the upstream `train_rollout_kl` metric, which upstream
  computes from trainer-scored log-probs since #3655 (`wf_3ff0709b-51c`).
- The residual 6e-03 is not attributed. Indexer replay is 60x larger than
  routing replay and every upstream recipe marks it debug-only
  (ADR-0012:85-89), so the cheap probe is the `moe_runner_backend`: SGLang
  refuses `--enable-return-routed-experts` on `flashinfer_trtllm` and falls
  back to `auto` (RUNLOG:1716-1718); the upstream GLM launcher forces
  `triton` for R3 (`wf_3ff0709b-51c`). Neither throughput nor numerics of
  that fallback were measured.
- `ppo_kl` under TIS is not a drift metric. NEVER cite it as evidence of
  policy movement; it never references the base model (memory note
  `glm53-ppo-kl-is-numerical-noise`). `train/kl_loss` is not a valid
  on-policy check under R3 because the ref model runs without replay
  (RUNLOG:1700-1701), and `use_kl_loss` is off anyway
  (`r44/miles-config.yaml:459-466`).
- "About 49 GB per DP shard" is inferred from code shapes with 42 layers;
  ADR-0012 uses 45. The Ray object-store size and the head-node transfer
  rate were not measured (`wf_545a45a0-6ca`, open question 2).
- "16 MB per 11k-token episode" and the whole r19 8 MiB incident exist only
  in memory notes; RUNLOG has no 2026-09-12 entry.
- The ADR index still lists 0012 as "(proposed)"
  (`miles_plugins/arena/adr/README.md:41`) while the ADR reads
  `Status: Accepted (2026-09-09)` (ADR-0012:3).
- Orphan routing files: a second run of a task after a lost ack leaves its
  routing files on the mount, and every publish failure, SIGKILL or stream
  purge does the same (ADR-0014 "Cleanup" and "Consequences"). Operators
  clear `ARENA_ROUTING_DIR` between runs. Possible orphans under
  `.../routing/rl-glm53f-adebt-cut-r34` (memory note, inference).
- A corrupt, mismatched or all-zero routing payload stays a fatal
  `RoutingReplayError`; a lost ref drops the whole group and counts in
  `rollout/routing_ref_lost_groups` (`03791ce86`).
- As of 2026-09-25 no gym or trainer image carried ADR-0014 and ADR-0071
  (memory note `nats-token-arrays-by-ref-2026-09-25`). Whether the r44-r47
  images carry it is outside this study; read those run records.
- Cheaper routing bytes are untested: `uint16` expert ids halve the payload
  (288 experts), and a per-PP-stage slice (12 of 42 layers) cuts about 70%
  of the per-rank transfer (`wf_545a45a0-6ca`, items 9 and the idle-gap
  finding).
- DeepEP with R3 on `glm5_next` is unverified (`wf_545a45a0-6ca`).
- Each router keeps its own replay cursor and the recompute reads from it.
  A layer that is always or never recomputed keeps the cursors aligned; a
  per-microbatch toggle misaligns them (`wf_58a8a79c-51c`). This binds any
  future selective-recompute work under R3.

## Actions taken

- ADR-0012 written and accepted (`3903e8f92`); T1-T3 shipped in
  `a9951bb59`; -1 pads in `70337a951`; trainer image
  `arena-slime-dev:miles-glm53-r3-20260909b`, gym image
  `arena-slime-dev:gym-glm53-r3-20260909a` (`r16/BUILD.md`).
- `use_rollout_routing_replay: true` from r16 on (`r16/miles-config.yaml:286`
  through `r44/miles-config.yaml:456`).
- r17 turned TIS off and used the SGLang log-probs (`fc9847ab7`); r18
  restored TIS and the actor pass (`ece3378e7`). Every later run keeps
  `use_tis: true`, `use_rollout_logprobs: false`.
- `03791ce86`: a redelivered result for an accepted task id no longer reaps
  its files; a missing ref raises `RoutingRefLostError` and drops one group.
  r29 moved to the trainer image with that fix (`7cec2a4b8`).
- NATS `max_payload` 8 MiB to 64 MiB: live patch on r19 (memory note), then
  template v4 (AREnATasksApps `5a0baa1`, RUNLOG:1853-1855).
- ADR-0014 with `dd5e67391` (trainer) and AREnATasks ADR-0071 with `f0472e9`
  (gym): token arrays by file reference, `token_arrays_by_ref` on every
  training task, the 30 MiB failed envelope, and the drop categories
  `too_large`, `staging`, `token_ref` (`db3955b70`).
- This study corrects the attribution of the 400x `ppo_kl` drop in
  RUNLOG:1698-1699, ADR-0012:187-189 and the r16 and r18 comments in
  `r44/miles-config.yaml:145-152`.

## Sources

- `training-runs/harbor-rl-glm53-flash/RUNLOG.md`: 1646-1722 (r15 and r16
  entry: launch, crash, results table, incident, notes), 1853-1855 (template
  v4), 1888-1890 (`.pt` files carry `rollout_routed_experts`), 2046-2064
  (AREnATasks ADR number map: 0049 became 0064).
- `training-runs/harbor-rl-glm53-flash/r16/BUILD.md`, `r16/miles-config.yaml`,
  `r16/nats.yaml`, `r16/gym-worker.yaml`, `r16/trainer-pytorchjob.yaml`;
  `r17/BUILD.md`, `r17/miles-config.yaml`; `r18/BUILD.md`,
  `r18/miles-config.yaml`; `r13/miles-config.yaml`, `r14/miles-config.yaml`;
  `r44/miles-config.yaml:145-166, 421-470`.
- `miles_plugins/arena/adr/0012-r3-routing-replay-over-nats.md`,
  `miles_plugins/arena/adr/0014-token-arrays-by-file-reference.md`,
  `miles_plugins/arena/adr/README.md:41-43`.
- AREnATasks `adr/0064-r3-routing-replay-harbor-gym.md`,
  `adr/0071-token-arrays-by-file-reference.md`
  (`/workplace/guparpit/arena-glm53/src/AREnATasks`).
- Code: `miles_plugins/arena/nats_arena/routing_replay.py`,
  `miles/backends/training_utils/loss_hub/losses.py:103-136, 185`,
  `miles/backends/megatron_utils/actor.py:574-590`,
  `miles/utils/arguments.py:3195`.
- Commits: `3903e8f92`, `a9951bb59`, `70337a951`, `969931b9e`, `805d7b382`,
  `fc9847ab7`, `ece3378e7`, `03791ce86`, `7cec2a4b8`, `dd5e67391`,
  `db3955b70`, `cee16d9d0`, `0dd04cc2a`, `4ac3f4b7f`.
- W&B `arena/rl-snorkel27`: `jnbq9gub` (r13), `tor7hf5u` (r14), `ocpsk0ip`
  (r16 first attempt, empty), `qp51gz7c` (r16 and r17), `fzl7n8vw` (r18 and
  its resume). Pulled 2026-09-28 with the key from the live r47 trainer pod.
- Workflow journals: `wf_545a45a0-6ca`, `wf_3ff0709b-51c`,
  `wf_58a8a79c-51c`.
- Memory notes (facts with dates; secondary): `glm53-r3-routing-replay-r16`,
  `nats-max-payload-redelivery-loop`, `nats-token-arrays-by-ref-2026-09-25`,
  `glm53-ppo-kl-is-numerical-noise`.
