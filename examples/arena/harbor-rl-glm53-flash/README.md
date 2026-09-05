# Harbor RL: snorkel-general-bash-harbor — GLM-5.3-Flash (miles trainer side)

GLM-5.3-Flash (321B total / 18B active: 45 layers = 3 dense + 42 MoE with 288
experts top-8 sigmoid; 34 KDA linear-attention + 11 DSA layers; NoPE MLA; mHC
hyper-connections; MTP layer 45 dropped for training) GRPO against the
`snorkel-general-bash-harbor` Harbor gym over NATS, 12x p6-b200.48xlarge on
prod-bom (kueue queue `gpu.p6-b200-48xlarge`).

Two parents, merged deliberately:

- **Arena wiring, batch shape, GRPO block** — r5-faithful from
  `examples/arena/harbor-rl-27b-snorkel/` (see its README for reward shape,
  dataset provenance, and descoping lineage; all of it applies unchanged).
- **Model-bound blocks** (parallelism, perf, optimizer, sglang engine) — the
  upstream-validated GLM-5.3-Flash recipe, radixark/miles PR #2786
  (`scripts/run_glm5_3_flash.py`, shape `(8,4)`), minus its colocate/offload
  flags: that recipe was COLOCATED, this job is DISAGGREGATED like every
  arena harbor-rl run.

Topology: workers 0-7 = Megatron actor (64 GPUs: TP8+SP x PP4
[11/11/11/12 layers] x CP1, EP16/ETP1, **DP=2** — the upstream shape was 32
GPUs at DP=1); workers 8-11 = SGLang engines (32 GPUs, **4 engines** at
TP8/EP8 — the 27B run had 8 engines at 4 GPUs; GLM needs a full node per
engine).

## Runbook (order matters)

### 0. Preflight

- Fork/image tree carries the GLM merge (upstream PR #2786:
  `scripts/models/glm5.3-flash.py`, `miles_plugins/models/glm5_next/`).
- kubectl context `arena-prod-bom-v2`, namespace `arena-tasks` (always pass
  `-n arena-tasks`).

### 1. Fetch the BF16 weights (CPU job, ~643 GB, hours)

`fetch-job.yaml` + `fetch_glm53.py` are the repo copies of the already
deployed and admission-validated `/tmp/glm53-deploy/` assets (pinned revision
`61f77a1e…`). Idempotent: re-runs skip size-verified files.

```bash
kubectl create configmap glm53-fetch-script -n arena-tasks \
  --from-file=fetch_glm53.py=fetch_glm53.py
kubectl apply -f fetch-job.yaml
kubectl logs -n arena-tasks -f job/glm53-fetch-20260902   # ends with SUCCESS
```

NOTE: a fetch may already be in flight under a different job name/mechanism —
before applying, check `kubectl get job -n arena-tasks | grep -i fetch` and
whether `/mnt/scratch-fast-1a-rw/durable/model-artifacts/external/zai-org/GLM-5.3-Flash-BF16` is already
complete (120 shards); do not double-fetch.

Result appears (read-only) at
`/mnt/scratch-fast-1a-rw/durable/model-artifacts/external/zai-org/GLM-5.3-Flash-BF16`
in every arena-tasks pod. (Do not confuse this with `/mnt/models-ro`: that is a
separate curated read-only EFS volume, not a view of fast scratch.)

### 2. Build and push the trainer image

From the miles repo root (the whole integration tree ships in the image):

```bash
docker build -f examples/arena/Dockerfile \
  --build-arg MILES_BASE_IMAGE=427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:glm53next-upstream-20260902 \
  -t 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-20260902a .
docker push 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-20260902a
```

The base is the ECR mirror of `docker.io/radixark/miles:glm53next` (SGLang
branch `sglang-miles-glm53next@9a26e749` + radixark/Megatron-LM PR#89
`@e8f57451`). `examples/arena/Dockerfile` now also bakes the EFA userspace +
aws-ofi-nccl layer — without it multi-node NCCL silently falls back to raw IB
and hangs (see the Dockerfile comment). Verify the pushed digest changed
(`aws ecr describe-images --region ap-south-1 --repository-name
arena-slime-dev --image-ids imageTag=miles-glm53-20260902a`).

This step runs BEFORE the convert: convert-job.yaml pins this exact image
(`imagePullPolicy: Always`), so applying it before the push gives
ImagePullBackOff. The fetch takes hours anyway — build while it runs.

### 3. Convert HF -> torch_dist (one 8-GPU node)

```bash
kubectl apply -f convert-job.yaml
kubectl logs -n arena-tasks -f rl-glm53f-convert-worker-0   # ends with SUCCESS
```

Saves to `/dev/shm` first (mountpoint-S3 cannot take DCP's final `.metadata`
rename; node ephemeral allocatable is uncertain at this size), then `cp -r` to
`/mnt/scratch-fast-1a-rw/durable/model-artifacts/dcp/glm5.3-flash_torch_dist`
with a per-file size-manifest diff (NOT `du -sb`: directory inode sizes differ
between tmpfs and mountpoint-S3, so an aggregate byte compare fails after a
successful copy). The trainer reads it at
`/mnt/scratch-s3files-rw/guparpit/checkpoints/dcp/glm5.3-flash_torch_dist` (`ref_load`). It lives on EFS rather than fast scratch because mountpoint-S3 caps single objects at ~78 GiB and the PP8 conversion emits 83 GB `.distcp` shards; a 2-node (16-rank) re-conversion would shrink shards enough for fast scratch if that ever matters.

Sharding: `CONVERT_KEEP_PP1` is deliberately NOT set — the converter
auto-reshapes PP to world_size (PP8, worst rank ~87 GB bf16, fits a 180 GB
B200); setting it would make each of the 8 ranks build the full 321B model
(~642 GB) and OOM at model build. torch_dist checkpoints re-shard at load, so
the PP8 layout loads cleanly under the trainer's TP8/PP4/EP16. Cheap
pre-validation of the exact command: point `--hf-checkpoint` at
`CharyZeng/GLM-5.3-Flash-4layer` first.

### 4. Deploy

Gym-worker side first — take the smoke-proven
`/tmp/miles-smoke-deploy/{nats,gym-worker}.yaml` with every `rl-milesgb1-`
name replaced by `rl-glm53f-` (NATS config/Deployment/Service, gym Deployment,
`--nats-url nats://rl-glm53f-nats:4222`,
`--model-base-url http://rl-glm53f-sglang:30000`) and the gym worker's
`ARENA_MODEL_PATH` pointed at
`/mnt/scratch-fast-1a-rw/durable/model-artifacts/external/zai-org/GLM-5.3-Flash-BF16` (the workers tokenize
against the policy model). Size the worker replica count for rbs 32 like the
snorkel run (the smoke ran 8 replicas for rbs 8).

```bash
kubectl apply -f nats.yaml           # rl-glm53f-nats (renamed smoke asset)
kubectl apply -f gym-worker.yaml     # rl-glm53f gym workers (renamed smoke asset)
```

Trainer side (this directory):

```bash
kubectl apply -f sglang-svc.yaml     # rl-glm53f-sglang -> trainer replica-index 0
kubectl create configmap rl-glm53f-trainer-config -n arena-tasks \
  --from-file=miles-config.yaml=miles-config.yaml
kubectl apply -f trainer-pytorchjob.yaml
kubectl logs -n arena-tasks -f rl-glm53f-trainer-worker-0
```

ADR-0004 reminder: pod env `EXPERIMENT_NAME` (not the YAML) drives the
checkpoint dir passed as both `--load` and `--save`; an existing dir resumes
silently. Bump `EXPERIMENT_NAME`/`PROJECT_NAME` for a fresh run. The job runs
`restartPolicy: Never` / `maxRestarts: 0` (known-good arena configuration): a
mid-run failure means delete + re-apply the PyTorchJob, and the SAME
`EXPERIMENT_NAME` then resumes from the last save intentionally.

### 5. First-run verification

- **EFA is load-bearing**: worker-0 (and any actor worker) NCCL INFO log must
  show the libfabric provider (`NET/OFI Selected provider is efa` /
  `Loaded net plugin Libfabric`). `Using network IB` or `Using network
  Socket` means the ofi-nccl plugin did not load — stop, fix the image/env,
  do not let a 64-rank all_gather "run" over TCP. `NCCL_DEBUG=INFO` +
  `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH` are set in the manifest for exactly this
  check; drop them once verified.
- Rendered `--load` names the new experiment dir; no
  `Loaded slime extra state`/`Restored wandb_run_id` lines (those mean
  resume).
- Data source pulls the lakeFS manifest (2922 rows); engines pass router
  health (allow up to 40 x 15 s — DSA/tilelang startup is slow); gym workers
  log `Completed <task>: n/m real`.
- W&B (`arena/rl-snorkel27`, group `rl-glm53f-gbash-r1`): binary-reward
  signal check as in the snorkel README — if >~80% of groups are
  zero-variance early, stop and revisit.

### Teardown

```bash
kubectl delete pytorchjob rl-glm53f-trainer -n arena-tasks
kubectl delete pytorchjob rl-glm53f-convert -n arena-tasks
kubectl delete configmap rl-glm53f-trainer-config -n arena-tasks
kubectl delete -f sglang-svc.yaml
# gym side: kubectl delete -f gym-worker.yaml -f nats.yaml
# fetch: kubectl delete job glm53-fetch-20260902 -n arena-tasks
#        kubectl delete configmap glm53-fetch-script -n arena-tasks
```

Checkpoints: at 321B a DCP save is roughly the 643 GB weight footprint plus
optimizer state per save point (`save_interval: 20`, `no_save_optim` trims
it); prune `${ARENA_CHECKPOINTS_DIR}/slime_experiments/rl-glm53f-gbash-r1`
when done.

## Deviations

Legend: **forced** = the model swap / platform makes the parent value wrong;
**chosen** = a judgment call, revisit freely.

### vs (a) the r5 snorkel config (`harbor-rl-27b-snorkel/`)

| setting | this config | source/reason |
|---|---|---|
| `experiment_name`/`project_name` | `rl-glm53f-gbash-r1` | forced — new run identity (ADR-0004: never reuse) |
| `agislime_dir` | `/root/miles` | chosen — matches the baked image tree (snorkel YAML carried the legacy `/opt/AGISlime` record; the smoke already used `/root/miles`) |
| `replicas` / `num_trainers` | 12 / 8 (was 6 / 2) | forced — GLM needs 64 actor GPUs (TP8xPP4 = 32 per DP replica, DP=2) + 4 engine nodes; same launcher convention as the smoke (rollout nodes = replicas − num_trainers) |
| `model_arch` | `glm5.3-flash` (was `qwen3.5-27B`) | forced — model swap; resolves `scripts/models/glm5.3-flash.py` |
| `hf_checkpoint` | `/mnt/scratch-fast-1a-rw/durable/model-artifacts/external/zai-org/GLM-5.3-Flash-BF16` | forced — model swap (fetch-job output; fast scratch, NOT the curated /mnt/models-ro EFS) |
| `ref_load` | `/mnt/scratch-s3files-rw/guparpit/checkpoints/dcp/glm5.3-flash_torch_dist` | forced — model swap (convert-job output on EFS; fast scratch rejects the 83 GB shards, see step 3) |
| `loss_mask_type` | **omitted** (was `qwen3_5`) | forced — GLM has no registered mask type (parser default falls back to `qwen`). Safe ONLY because `use_rollout_logprobs: true` removes every tokenizer-fallback sample from training (zero-fill + ABORTED + remove_sample); that key is load-bearing for the omission — see the YAML comment |
| `attention_backend` | **dropped** (was `flash`) | forced — GLM-5.3-Flash uses the custom KDA/DSA spec (`miles_plugins.models.glm5_next`); the upstream recipe sets no attention backend |
| `sglang_enable_dp_attention` | **dropped** (was `true`) | forced — qwen-specific; recipe runs `sglang_dp_size: 1` |
| `sglang_tool_call_parser` / `sglang_reasoning_parser` | **dropped** (were `qwen3_coder`/`qwen3`) | chosen — qwen values would be wrong for GLM; the textual bash agent parses tool calls itself. Unverified for GLM (see Risks) |
| `pin_rollout_manager_to_head` | `true` (absent in snorkel YAML) | chosen — smoke-proven pattern; the `rl-glm53f-sglang` Service selects replica-index 0 |
| TP/PP/CP/EP | 8 / 4 (+ first 11 / last 12 layer split) / 1 / 16 (ETP 1) (was 4/2/2/1) | forced — upstream GLM recipe parallelism |
| `rollout_num_gpus_per_engine` | 8 (was 4) | forced — recipe; bf16 weights need a full TP8 node per engine |
| `sglang_tp_size`/`sglang_ep_size`/`sglang_dp_size` | 8 / 8 / 1 (absent before) | forced — recipe engine shape |
| `sglang_chunked_prefill_size` | 8192 (was 12288) | forced — recipe |
| `sglang_mem_fraction_static` | 0.7 (was 0.85) | forced — recipe; weights alone are ~80 GB/GPU |
| `sglang_disable_radix_cache`, `sglang_dsa_prefill_backend`/`_decode_backend: tilelang`, `sglang_kv_cache_dtype: bfloat16` | added | forced — recipe; DSA layers need the tilelang backends on this SGLang branch |
| `router_health_success_threshold`/`_check_interval_secs`/`_failure_threshold` | 1 / 15 / 40 (absent before) | forced — recipe; slow DSA engine startup |
| `rollout_health_check_interval`/`_timeout` | 300 / 300 (absent before) | forced — recipe |
| `max_tokens_per_gpu` | 8192 (was 65536) | forced — recipe token budget for 321B at full recompute; `use_dynamic_batch_size: true` **kept** (arena packing mechanism), `micro_batch_size: 1` added as the recipe-faithful fallback (ignored under dynamic batching). Reconciliation documented in the YAML |
| `update_weight_buffer_size` | 1073741824 (absent before) | forced — recipe (1 GiB staged weight-update buffer) |
| `train_memory_margin_bytes` | 3221225472 (absent before) | forced — recipe |
| `qkv_format` | `thd` (absent before) | forced — recipe; the tilelang DSA path requires it |
| `model_name` | `glm5_next` (absent before) | forced — recipe; selects the megatron<->sglang weight mapping in the weight updater/HF export (the HF config class name alone does not resolve it) |
| `lr` | 1e-6 (was 15e-7) | chosen — GLM recipe optimizer block taken whole per the port decision |
| trainer manifest: replicas/elastic | 12, `maxRestarts: 0`, `restartPolicy: Never` (was 6, 0, Never) | chosen — replicas forced by GLM; restart semantics kept at the known-good arena values. OnFailure+3 was considered and reverted: recovery is unproven on this ray-based launcher (needs all 11 workers to exit on head-GCS loss or the restarted head hangs in `_wait_for_ray_nodes`); revisit after run 1 |
| `enable_trajectory_tracing` | **dropped** (parent carried `false`) | forced — no miles flag with that name is registered (deploy.sh-era key); `false` was inert, `true` would crash the trainer at argparse |
| `kl_loss_coef` | `0.0` pinned explicitly (parent left it to the default 0.0) | chosen — `use_kl_loss: true` makes with_ref=True (ref checkpoint + full ref forward every step, x0.0 in the loss); kept for r5-parity and documented in the YAML rather than silently inherited |
| `check_weight_update_equal` + `check_weight_update_skip_list: visual.` | added for run 1 (absent in parent) | chosen — first disaggregated run of the glm5_next update-weights path; remove after step 1 passes |
| trainer manifest: image | `arena-slime-dev:miles-glm53-20260902a` | forced — GLM needs the glm53next SGLang/Megatron stack |
| trainer manifest: node-exclusion affinity (`i-09907cd660684a460`) | **dropped** | chosen — point-in-time S3-CSI incident workaround (2026-08-27); at 12 nodes a stale exclusion costs real capacity |
| trainer manifest: `WANDB_API_KEY` secretKeyRef (`wandb-env-arena`) + `WANDB_BASE_URL` | added | chosen — smoke-proven pattern (snorkel manifest predates the secret) |
| trainer manifest: `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH` | added | chosen — bring-up diagnostics for the EFA/ofi-nccl check; drop after the first verified run |
| trainer manifest: `SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1`, `SGLANG_HEALTH_CHECK_TIMEOUT=120`, `PYTHONFAULTHANDLER=1`, `TORCHINDUCTOR_COMPILE_THREADS=1`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` | added | chosen — the upstream recipe's `extra_env_vars`, carried as pod env (inherited by the ray workers) |
| names | everything `rl-glm53f-*` | forced — parallel-run isolation (NATS streams are per-server; services/configmaps must not collide) |

Everything not listed (batch shape `rollout_batch_size 32` /
`n_samples_per_prompt 8` / `global_batch_size 256` / `num_rollout 90`,
over-sampling via `dynamic_sampling_filter_path` at the default 4x examine
cap, GRPO eps 0.4/0.4/c2.0 + low_var_kl at 0, `prompt-data-list` lakeFS pin,
rollout keys, wandb project/team, EFA env/resources/tolerations/hostPaths,
memory 1800Gi + ephemeral 32Gi, kueue queue label) is **unchanged** from the
snorkel parent.

### vs (b) the upstream GLM-5.3-Flash recipe (PR #2786 `run_glm5_3_flash.py`)

| setting | this config | source/reason |
|---|---|---|
| `--colocate`, `--offload-train-target disk`, `--offload-train-disk-dir` | **dropped** | forced — those are colocate-only; the arena path is disaggregated (separate engine nodes) |
| actor shape | 8 nodes x 8 GPUs (DP=2) | chosen — the recipe's validated `(8,4)` = 32 GPUs at DP=1, scaled x2 data-parallel on arena capacity; per-replica parallelism identical |
| rollout/data path | arena NATS gym (`rollout_function_path`, `data_source_path`, lakeFS `prompt-data-list`) | forced — arena training path vs the recipe's dapo-math + `--rm-type math` |
| batch shape | rbs 32, n 8, GBS 256, num_rollout 90, response len 2048, temp 1 | chosen — r5-faithful arena shape (recipe: rbs 4, temp 0.8, response len 4096, num_rollout 5 smoke) |
| GRPO block | eps 0.4/0.4, `eps_clip_c` 2.0, `use_kl_loss` + `kl_coef` 0, `use_rollout_logprobs` | chosen — r5-faithful (recipe: eps 0.2/0.28, `--kl-loss-coef 0`) |
| `use_dynamic_batch_size` | `true` | chosen — arena packing mechanism kept; recipe ran static mbs 1 (its `max_tokens_per_gpu 8192` is honored as the budget) |
| `--check-weight-update-equal --check-weight-update-skip-list visual.` | **kept for run 1** | chosen — first disaggregated (NCCL-bucketed) run of the glm5_next update path; the recipe validated colocated (UpdateWeightFromTensor) only. Per-update cost accepted for run 1; remove after step 1 passes |
| checkpoint flow | `--load`/`--save`/`--save-hf` appended by `run_arena_harbor.py` from `EXPERIMENT_NAME` (ADR-0004), `save_interval 20`, `no_load_optim`/`no_save_optim` | forced — arena run-identity contract (recipe defaulted to `skip_saving`) |
| wandb | config keys (`wandb_host/project/team`) + `WANDB_API_KEY` secret | forced — arena tracking contract vs `U.get_default_wandb_args` |
| `sglang_context_length` / `rollout_max_context_len` | 131072 | chosen — arena parent values (recipe left engine defaults) |
| `sglang_disable_overlap_schedule` | `true` | chosen — arena precedent (AGISlime run5 + smoke); recipe does not set it. Engine-side conservative; candidate to lift later |
| `pin_rollout_manager_to_head`, `sglang_router_port 30000` | kept | forced — arena Service wiring |
| launcher-consumed keys (`user`/`cluster`/`replicas`/...) | present | forced — `run_arena_harbor.py` contract; never reach the trainer argv |

## Risks (accepted, watch on run 1)

1. **GLM chat/tool-call format vs the gym parser — UNVERIFIED.** The snorkel
   textual bash agent worked against qwen output; GLM-5.3's template and
   tool-call conventions differ, and no sglang tool-call/reasoning parser is
   configured. If early trajectories show malformed command extraction or
   zero real completions, this is the first suspect.
2. **EFA pairing untested**: the Dockerfile's aws-efa-installer 1.47.0
   (aws-ofi-nccl 1.18.x) against the glm53next base's NCCL 2.29/cu13 has not
   run multi-node before this job. First-run gate: the NCCL log must show the
   libfabric/efa provider — `Using network IB` or `Socket` means broken
   fallback (hang/crawl), not success.
3. **Engine memory is tight**: bf16 weights ~643 GB / TP8 ≈ 80 GB per B200
   (180 GB) and `mem_fraction_static 0.7` budgets ~126 GB — ~46 GB left for
   KV + activations per GPU. Long multi-turn contexts may hit engine OOM or
   heavy prefill throttling; the bf16 KV dtype (recipe) doubles KV vs fp8.
   First mitigation if KV headroom is needed: add
   `sglang_max_running_requests: 256` — with radix cache disabled this flips
   the KDA/mamba state pool from the ratio autosizer (~21 GB for ~1170 slots
   nobody uses; actual load is ≤64 in-flight per engine) to
   from_max_running_requests (~4.5 GB), raising KV from ~23 GB (~1.7M tokens)
   to ~39 GB (~2.9M tokens) per GPU. Deliberately NOT set for run 1
   (untested knob on this branch; the default is safe).
4. **Mask fallback semantics**: `loss_mask_type` is omitted; the parser then
   defaults to `qwen`, which is reached only on the tokenizer-fallback path,
   and every fallback sample is removed from training because
   `use_rollout_logprobs: true` (zero-fill + ABORTED + remove_sample). That
   key is load-bearing — never disable it without adding a GLM mask type.
   Still verify the first-step token accounting (masked/unmasked counts in
   the trainer log) before trusting the loss.
5. **Capacity**: 12 simultaneous p6-b200.48xlarge via kueue
   (`gpu.p6-b200-48xlarge`) plus 1 more for the convert job while it runs.
   Admission may queue; the trainer is all-or-nothing (min=max=12).
   Fallback-shape note: at 12 actor nodes (96 GPUs) DP=3 is IMPOSSIBLE with
   EP16 (96 % (ETP1 x EP16 x PP4 = 64) != 0 raises in megatron
   parallel_state); EP24 would be needed. At 64 GPUs, PP8 does not reduce
   expert-optimizer memory and EP8 is strictly worse (~130 GB static) —
   EP16/PP4/DP2 is the right 64-GPU shape.
6. **First disaggregated glm5_next weight update**: the recipe validated the
   colocated UpdateWeightFromTensor path; this run exercises the NCCL-bucketed
   UpdateWeightFromDistributed path for the first time. The known cross-bucket
   hazard (fused q_a_proj/kv_a_proj_with_mqa must co-arrive) is provably
   covered by miles' q_lora atomic-update grouping, and
   `check_weight_update_equal` is on for run 1 — additionally grep engine logs
   for `The full weights of the ModelRunner are partially updated` as the
   failure signature.
7. **Token budget vs trajectory length**: `max_tokens_per_gpu 8192` is
   recipe-validated for ≤4k responses; arena full-trajectory samples can
   reach the gym's 32k context limit, and a single over-budget sample forms
   its own oversized micro-batch under dynamic batching. Watch actor memory
   on long-trajectory steps; first mitigation is raising recompute or
   lowering the gym context limit, not raising the budget.
