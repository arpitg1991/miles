# harbor-rl-27b-snorkel run log

Run history for the `snorkel-general-bash-harbor` Qwen3.5-27B job on miles
(the plain-stack port of the AGISlime snorkel run 5, ADR-0007): what was
staged, what was validated on CPU, every GPU launch derived from this
directory with its identity, image, shape, outcome and root cause, and the
decisions taken along the way. It is the run-history counterpart to
`miles_plugins/arena/adr/`; package-level milestones live in
`miles_plugins/arena/RUNLOG.md`.

## Format

One `## <date PT> - <run or milestone>` section per event, newest last.
Times are Pacific (source logs are UTC). Run identity = `EXPERIMENT_NAME`
(checkpoint dir + W&B group), plus image tag and node shape. Numbers come
from the trainer log, the harness logs or W&B; nothing is inferred without
saying so. Raw evidence referenced by path lives outside the repo under
`/workplace/guparpit/miles/arena-port-artifacts/`.

## 2026-09-01 09:03-09:37 PT - example staged as the plain stack

- Lineage: AGISlime snorkel r1 (2026-08-26, 14 nodes, rbs 32 / GBS 256,
  `train.jsonl` 2,600 tasks) -> r2 (08-27, zero-variance filter) -> r3
  (08-27, rbs 256 / GBS 2048) -> r3b (08-27, "8x pushed to NATS": rbs 128,
  `ARENA_DYN_SAMPLING_MAX_EXAMINE_MULT=8`, survivor norm) -> r4 (08-27,
  removed-sample replacement, commit 637d714) -> run5 (08-29, r3b algorithm
  on 6 nodes = 2 actor + 4 rollout, image `arena-slime-dev:rl-snorkel-r4b-20260827`).
- Decision (user): port the PLAIN stack first (ADR-0007). Kept: r2 filter via
  miles-native `check_reward_nonzero_std`, r1 batch shape (rbs 32, GBS 256,
  n 8, num_rollout 90 ~ 1 epoch), lr 1.5e-6, run5's 6/2 topology, W&B on
  (`arena/rl-snorkel27`). Descoped: survivor norm, `removed_sample_replacement`
  (+ its test, deleted), `ARENA_NATS_REPLACE_REMOVED_SAMPLES`, the examine-mult
  env override, the 8x budget.
- Dataset (user): the `KNOWN_GYMS` lakeFS pin
  `.../snorkel-general-bash-harbor/ecr-20260823/manifest.jsonl` (2,922 tasks;
  bare variant-dir URI 404s, the object name is required). NAMED risk: includes
  run5's 322 held-out dev tasks.
- Files: `snorkel-27b.yaml` (09:35), `trainer-pytorchjob.yaml` (09:26; run5
  trainer with the full EFA env, the 2026-08-27 S3-CSI node exclusion, app
  label normalised, image placeholder), `sglang-svc.yaml` (09:03, verbatim
  run5), `ARGV_PARITY.md` (09:33), `README.md` (09:37).
- Argv parity (both sides executed for real, descoping applied to both): old
  110 vs new 101 flag occurrences, 9-occurrence delta = 15 only-old minus 6
  only-new, every row justified; strict `parse_args` on CPU (sm100 stub)
  `PARSE_OK` with `rollout_batch_size=32`, `global_batch_size=256`,
  `custom_reward_post_process_path=None`, `dynamic_sampling_max_examine_mult=4.0`,
  `loss_mask_type=qwen3_5`, `use_wandb=True`.

## 2026-09-01 09:31 PT - e2e-snorkel harness on the real lakeFS manifest: 145/145

- Harness `arena-port-artifacts/e2e-snorkel/` (`run.sh`, `driver.py`,
  `fake_gym_worker.py`): dockerized NATS JetStream + fake snorkel gym worker
  (binary pytest-verifier rewards, real Qwen3.5-27B tokenizer) + a driver that
  replicates RolloutManager wiring (data source, `generate_rollout`,
  `convert_samples_to_train_data`, DP split). Scaled shape rbs 4 / n 4 /
  GBS 16 (rbs*n == GBS, nothing ragged). Credentials
  `AWS_PROFILE=lakefs-arena-gym` + `LAKECTL_SERVER_ENDPOINT_URL`.
- Pulled the real 2,922-row manifest (payload commit `ca22976f...`); every
  served task carried the pinned `lakefs_uri`/`lakefs_commit_id` (110 served).
- Rounds r0/r1 normal, r2-heavy (2 truncated per kept group): removed rows
  keep zeroed loss masks but participate in the native baseline (+/-0.86602
  on all 4 rows of a `[1,0,0,1]` group; survivor norm would give +/-1.0 / 0.0)
  - the ADR-0007 consequence, asserted. Static asserts: replacement module
  gone, no replacement hook in `nats_rollout`. Resume: offsets/epochs/dedup
  keys/counters restored. JetStream: `ARENA_TASKS` workqueue, `ARENA_RESULTS`
  limits 128 MiB, durable `slime-trainer` explicit ack / max_deliver 3 /
  ack_wait 3600.
- Result: `checks passed: 145` (driver.log 09:31 PT). Logs at
  `e2e-snorkel/logs/{driver,worker}.log`.

## 2026-09-01 11:47-12:08 PT - GPU smoke rl-milesgb1-smoke1, attempt 1: ImportError

- Goal: first GPU run of the ported plugin, derived from this example at
  smoke scale (`smoke-3node/`): 3x p6-b200 (workers 0-1 actor TP4/PP2/CP2,
  worker 2 = 8 single-GPU engines), rbs 8 / GBS 64, num_rollout 4,
  save_interval 4, 1 GPU per engine + dp-attention off (miles' own qwen3.5
  recipes: sglang TP>1 mis-generates Qwen3.5, sgl-project/sglang#21039),
  `pin_rollout_manager_to_head` so the router binds where the Service points,
  TCP fabric (`FI_PROVIDER=tcp`, `NCCL_IB_DISABLE=1`, no EFA resources - the
  miles image has no libfabric/aws-ofi-nccl). Gym side: run5 gym worker
  verbatim renamed `rl-milesgb1-*`, 8 replicas, image
  `arena-tasks-dev:rl-smoke-20260821b`, `ARENA_MAX_TOKENS` 2048 /
  `ARENA_ROLLOUT_CONTEXT_LIMIT` 32768 / `ARENA_NATS_ACK_WAIT` 4500 /
  `ARENA_ROLLOUT_MAX_CONCURRENCY` 8. Manifests written 11:47-11:48 PT.
- 12:07:50 PT ray head up; 12:08:17 job `raysubmit_xDfC2QTy1JtJxbNC`
  submitted; 12:08:29 entrypoint died at import:
  `transformer_engine_torch...so: undefined symbol
  _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE` - the TE
  wheel in the base image was built against a different torch. Same symbol
  already visible as the "bridge shim not applied" warnings at launcher start.
- Root cause: the trainer image was built (`examples/arena/Dockerfile`) on the
  radixark/miles `latest-cu12` nightly of 2026-09-01, whose TE<->torch pairing
  is broken. Decision: rebuild on `radixark/miles:latest` (cu13; B200 nodes run
  driver 580.159, cu13 OK) -> `arena-slime-dev:miles-arena-20260901b`. Avoid
  `latest-cu12`.
- Log: `arena-port-artifacts/glm53/wud-evidence/trainer-0-s3.log` (310 lines,
  tee'd to `/mnt/scratch-s3files-rw/guparpit/logs/rl-milesgb1-smoke1/`).

## 2026-09-01 16:39 PT -> 2026-09-02 00:34 PT - rl-milesgb1-smoke1 relaunch: 4/4 rollouts, ray SUCC

Identity: `EXPERIMENT_NAME=rl-milesgb1-smoke1`, PyTorchJob `rl-milesgb1-trainer`
(arena-tasks, 3x p6-b200, kueue `gpu.p6-b200-48xlarge`), image
`arena-slime-dev:miles-arena-20260901b`, ray job `raysubmit_9w4xgTSkP9E7p54p`,
W&B `mega.wandb.agi.amazon.dev/arena/rl-snorkel27` group `rl-milesgb1-smoke1`
run `j6gr37za`. Trainer manifest updated 16:37 PT with the new image; it
keeps run5's `nodeAffinity NotIn i-09907cd660684a460` (the 2026-08-27 S3-CSI
mount-attach exclusion, inherited).

Bring-up (16:39-16:46 PT): ray head 16:39:43; W&B run 16:40; lakeFS manifest
pulled 16:40:57 (`Pulled lakefs://.../manifest.jsonl`); slime-era TP4 DCP
loaded into radixark Megatron through the empty-`--load` -> `ref_load`
fallback ("(TP, PP) mismatch after resume ((4, 2) vs (1, 1))", RNG ignored,
fine); 8 engines up 16:45:59; first `update_weights` 18.6 s (16:46:01-20);
`NATS dynamic sampling enabled: filter=...check_reward_nonzero_std`.

| rollout | wall | avg reward | filter (kept / examined, cap 32) | removed / 64 (truncated_ratio) | failed |
| --- | --- | --- | --- | --- | --- |
| 0 | 8494.9 s (done 19:07 PT) | 0.391 | 8 / 31, 23 zero-variance dropped | 20 (0.3125) | 0 |
| 1 | 7968.2 s (done 21:20 PT) | 0.156 | 8 / 32 = cap, 27 dropped | 55 (0.859) | 0 |
| 2 | 6730 s | 0.312 | (memory record) | | |
| 3 | 4082 s | 0.172 | (memory record) | | |

- Startup burn: the first 58 groups (g1-g60, 16:47-17:00 PT) came back
  `status=failed` within seconds and were dropped ("all 1 tasks failed");
  attributed to Docker Hub 429s on the gost egress-sidecar base for a fresh
  8-worker fleet (dind caches after the first ~30 min). First real group
  17:53 PT (reward 0.0, 36,255 tokens). Rollout 0 response_len mean 15,245 /
  max 29,670 tokens, episode total mean 31,932; ~30 tok/GPU/s.
- Train step 0 (19:08-19:22 PT): ref_log_probs 240 s + actor_train 601.6 s
  = 842.4 s; `loss -0.0222, pg_clipfrac 2.1e-5, ppo_kl 1.6e-4, ess_ratio
  0.687, grad_norm 0.301`; actor MFU 0.019, `wait_time_ratio 0.91`. Step 1
  (21:21-21:27 PT): 359.8 s (81.5 + 277.7); `loss -0.0284, ess_ratio 0.141,
  grad_norm 0.081`; MFU 0.041. GRPO healthy on the plain stack.
- `update_weights` steady state 4.1 s at 21:20 PT (then 4.37 / 3.5 s per the
  memory record); the 18.6 s was the cold first sync only.
- Terminal (00:34 PT 2026-09-02): ray job SUCC exit 0 after 4 rollouts and 4
  train steps; `iter_0000003` torch_dist checkpoint + `latest_checkpointed_iteration.txt=3`
  on `scratch-s3files-rw` (that fs completes the DCP `.metadata` rename;
  fast-scratch mountpoint-S3 does not).
- Defect found: HF export `hf/rollout_3` wrote 178 weight shards but no
  `config.json` / tokenizer / `model.safetensors.index.json`: rank 0 aborted
  the aux copy on `[Errno 524] ENOTSUPP` at `.gitattributes`
  (`hf_export.py:173` caught it; the job continued). Fix tracked separately
  (hf_export ENOTSUPP tolerance).
- Live log capture (through 21:37 PT, 16,795 lines):
  `arena-port-artifacts/glm53/wud-evidence/trainer-0-live.log`; the full
  tee'd log is under `/mnt/scratch-s3files-rw/guparpit/logs/rl-milesgb1-smoke1/`.

## 2026-09-02 - verdicts and what is versioned from the smoke

- Weight sync is a NON-ISSUE (adversarially verified against the AGISlime
  logs, raw evidence `wud-evidence/`): miles steady-state update 4.07 s over
  TCP vs run5 5.69-8.59 s over EFA (mean 6.6 s over 15 steps); the 18.6 s was
  ~14.5 s of one-time lazy NCCL communicator bootstrap inside the timed span.
  The remembered "updates take as long as rollouts" was `train_wait`
  (`wait_time_ratio` 0.91-0.95 here). Do not spend effort on it; EFA mainly
  speeds the cold bootstrap. Topology confound: run5 pushed to 32 engine GPUs
  on 4 nodes, this smoke to 8 on 1 node.
- Proven on GPU: slime-era TP4 DCP -> radixark Megatron via the `ref_load`
  fallback; NATS <-> run5 gym workers end to end on the bit-identical wire
  contract; zero-variance filter + native group_index normalization; step-0
  GRPO healthy. Gotchas: no EFA in miles images (TCP, MFU 2-4%); trainer W&B
  auth is the `WANDB_API_KEY` secretKeyRef from secret `wandb-env-arena`.
- Follow-ups: EFA image layer (lands with the GLM scaffolding); hf_export
  ENOTSUPP fix; the 4x examine cap bound in rollout 1 (ADR-0007 consequence).
- `smoke-3node/` = the as-applied smoke manifests, versioned as the record of
  this validation: `miles-config.yaml`, `trainer-pytorchjob.yaml`,
  `gym-worker.yaml` (8 replicas), `nats.yaml` (`rl-milesgb1-nats`, nats
  2.11.3, 8 MiB max_payload, emptyDir JetStream), `sglang-svc.yaml`
  (`rl-milesgb1-sglang` -> replica-index 0:30000), `monitor.sh` (5-min poll
  to rollout 0 + first step) and `monitor2.sh` (poll to terminal state). The
  README's gym-side apply/teardown lines point here because the run5 assets
  they cited live in an unversioned AREnATasks staging directory.

## 2026-09-02 22:23 PT — Smoke follow-up closed: HF export tolerates ENOTSUPP on the aux-file copy (integration 44ddf62fb)

Commit: `fix(megatron): tolerate ENOTSUPP when copying HF aux files in hf_export`
(`miles/backends/megatron_utils/hf_export.py`, +8/-1; the worktree file is
byte-identical to the 44ddf62fb blob). Closes the one defect the smoke left
open in the previous entry.

**What the smoke left behind (rl-milesgb1-smoke1, terminal 2026-09-02 00:34 PT)**

| item | state after the run |
|---|---|
| DCP | `iter_0000003` saved, `latest_checkpointed_iteration.txt` = 3 (on `scratch-s3files-rw`, whose fs completes the `.metadata` rename) |
| HF export `hf/rollout_3` | 178 weight shards written; **no `config.json`, no tokenizer files, no `model.safetensors.index.json`** |
| cause line | rank 0 aborted the aux-file copy at `.gitattributes` with `OSError [Errno 524] ENOTSUPP` |
| job outcome | `save_hf_model` caught it (`Failed to save HuggingFace format: ...`), the run continued to ray job SUCC exit 0 |

Mount pair on the smoke: source `hf_checkpoint` `/mnt/models-ro/external/Qwen/Qwen3.5-27B`
(the curated read-only volume, EFS/NFS per the storage notes) -> destination
`/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-milesgb1-smoke1/hf/rollout_3`
(EFS). The archived trainer logs (`arena-port-artifacts/glm53/wud-evidence/trainer-0-{s3,live}.log`)
stop at 2026-09-01 21:37 PT, before the final save, so the raw traceback is recorded only in
the 00:34 PT terminal-state notes.

**Root cause (code reading, `export_hf_model_direct`)**

- The aux loop used `shutil.copy2` = `copyfile` + `copystat`. `copystat`'s xattr step raised
  errno 524 `ENOTSUPP`, the kernel-internal errno that `shutil` does not swallow (it tolerates
  `ENOTSUP` 95 / `ENODATA` / `EINVAL` only). Which side raised it (NFS source or EFS destination)
  was not isolated; the fix is side-agnostic.
- The exception escaped the `for meta_file in base_checkpoint.iterdir()` loop, so the
  `model.safetensors.index.json` write right after it never ran; the `finally: barrier()` kept
  the collective in step, `save_hf_model` logged and returned, and the `.complete` marker
  (touched only after a clean return) was never written. The 178 shards were already on disk.
- Any run of this example would hit the same path: `snorkel-27b.yaml` reads the same
  `/mnt/models-ro` `hf_checkpoint`, `--save-hf <ckpt>/hf/rollout_{rollout_id}` is appended by
  the launcher, and `save_hf_model` runs after every DCP save (`save_interval 20`,
  `num_rollout 90` -> 4 saves + the final one, README "Teardown").

**Fix (landed 22:23 PT, integration 44ddf62fb)**

- `shutil.copyfile` (data only, no `copystat`) per aux file, wrapped in `try/except OSError ->
  logger.warning("HF export: could not copy <name> to <path>: ...")`. One failing aux file no
  longer aborts the export, skips the index write, or suppresses `.complete`.
- Comment in the code names the case the fix was written for: a mountpoint-S3 / FUSE
  `hf_checkpoint` (the GLM-5.3 job's BF16 weights live on fast scratch).

**Alternatives considered**

- The 00:34 PT note proposed "tolerate ENOTSUPP metadata ops + backfill aux files from the
  base model dir". Rejected as the shape of the fix: avoiding the metadata op altogether
  (`copyfile`) makes the data copy succeed, so no backfill pass is needed; a post-hoc backfill
  would also have to run on rank 0 after the barrier and re-derive the file set.
- Catching only errno 524 around `copy2`: not taken; the export must never lose the index
  over a cosmetic file (`.gitattributes`, `README.md`), whatever the errno, and the per-file
  warning keeps the omission visible.
- Plugin-side workaround: none exists; `hf_export.py` is core and has no hook here, so this is
  the ADR-0001 "core edit only where miles has no seam" case (same as the qwen3_5 mask).

**Where it shipped, and what is NOT verified**

- Baked into the GLM trainer image `arena-slime-dev:miles-glm53-20260902c` together with the
  `--arena-mask-clipped-final-turn` flag (5f8925db0) and the r2 relaunch fix set (3098ae1d3);
  `20260903d` is a superset, so GLM r2-r7 all carry it. It is in **no 27B image**: the smoke ran
  `arena-slime-dev:miles-arena-20260901b` (built 2026-09-01, before the fix); a rebuild of
  `examples/arena/Dockerfile` from this tree picks it up.
- No CPU test covers the aux copy (the export is a distributed collective; the only test
  touching `hf_export` is `tests/fast/backends/megatron_utils/test_lora_model_branches.py`).
- **GPU effect unverified.** The GLM runs save at `save_interval 20`: run 1 / run 2 / r2 / r3
  died in step 0 or 1, r4 ended after 13 steps (20:29 PT), so the first run that can have
  exported HF after the fix is r5 (`rl-glm53f-gbash-r5`, reached step 40), then r6 (step 33 by
  2026-09-05 05:15 PT) and r7. No source records a complete export. Before claiming the fix
  works: `ls <ckpt>/hf/rollout_*/` for `config.json`, tokenizer files,
  `model.safetensors.index.json` and `.complete`, and grep the EFS `trainer-0.log` for
  `HF export: could not copy` (warning path) vs `Successfully saved merged HuggingFace model`.
