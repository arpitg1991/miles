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

From the integration tree (`/tmp/glm53-integration`, branch `glm53-arena`;
the whole tree ships in the image). Push to us-east-1 — `arena-ecr-dev` can
only push there and the repo replicates to ap-south-1:

```bash
docker build -f examples/arena/Dockerfile \
  --build-arg MILES_BASE_IMAGE=427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:glm53next-upstream-20260902 \
  -t 427267593057.dkr.ecr.us-east-1.amazonaws.com/arena-slime-dev:miles-glm53-<tag> .
docker push 427267593057.dkr.ecr.us-east-1.amazonaws.com/arena-slime-dev:miles-glm53-<tag>
aws ecr describe-images --profile arena-ecr-dev --region ap-south-1 \
  --repository-name arena-slime-dev --image-ids imageTag=miles-glm53-<tag>   # replica landed
```

Image lineage (`arena-slime-dev:`; each tag is a superset of the previous):

| tag | carries | integration commit |
|---|---|---|
| `miles-glm53-20260902a` | glm53next base + integration tree + EFA layer (run 1) | — |
| `miles-glm53-20260902b` | + fla 0.4.2 KDA kernel patch for triton 3.7.1 (`examples/arena/patches/fla_kda_next_power_of_2.py`, applied by the Dockerfile) | 3c40b4660 |
| `miles-glm53-20260902c` | + `--arena-mask-clipped-final-turn` flag; hf_export ENOTSUPP fix for FUSE/mountpoint-S3 aux files (`miles/backends/megatron_utils/hf_export.py`). **Pinned by `trainer-pytorchjob.yaml` (r2, r3)** | 5f8925db0, 44ddf62fb |
| `miles-glm53-20260903d` | + `--arena-keep-timeout-trajectories` (default off) and the per-rollout `Removal reasons:` summary line (`miles_plugins/arena/nats_arena/nats_rollout.py`) | 32da04357 |
| `miles-glm53-20260908a` | + `--arena-train-segments {final,all}` (plugin ADR-0011) and the all-mode DP alignment pad (zero-loss sibling rows, so `build_dp_schedule` can align singleton micro-batches); drops `rollout/compaction_segments_mean` for the upstream `rollout/num_training_samples` and `rollout/episode_raw_reward`. Built 2026-09-08 on top of the r11 image (`miles-glm53-20260907a`: `--arena-inflight-multiplier`, b209d9b4). **Pinned by `r12/trainer-pytorchjob.yaml`** | d28503cf (`arpit-glm-53`) |

miles argparse is strict: a `miles-config.yaml` key whose flag the image does
not know kills the trainer at startup (`arena_mask_clipped_final_turn` needs
>= c, `arena_keep_timeout_trajectories` needs d, `arena_train_segments` needs
`miles-glm53-20260908a`). Bump the manifest image tag together with any such
key.

The base is the ECR mirror of `docker.io/radixark/miles:glm53next` (SGLang
branch `sglang-miles-glm53next@9a26e749` + radixark/Megatron-LM PR#89
`@e8f57451`). `examples/arena/Dockerfile` also bakes the EFA userspace +
aws-ofi-nccl layer — without it multi-node NCCL silently falls back to raw IB
and hangs (see the Dockerfile comment).

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

Gym-worker side first. `nats.yaml` and `gym-worker.yaml` in this directory
are the smoke-proven assets with every `rl-milesgb1-` name replaced by
`rl-glm53f-` (`--nats-url nats://rl-glm53f-nats:4222`,
`--model-base-url http://rl-glm53f-sglang:30000`), `ARENA_MODEL_PATH` on the
GLM BF16 fast-scratch path (workers tokenize against the policy model),
128 replicas for rbs 32 with dynamic sampling, the 32768/131072 caps from the
run-1 live change, and — since r2 — a required `NotIn` anti-affinity that
keeps gym + dind pods off `p6-b200.48xlarge`/`p6-gpu` nodes (GPU nodes are
untainted and the trainer's 1200Gi/0-CPU request leaves room the scheduler
would otherwise fill; see the YAML comment).

```bash
kubectl apply -f nats.yaml           # rl-glm53f-nats
kubectl apply -f gym-worker.yaml     # rl-glm53f-gym-sgb, 128 replicas
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

Current run identity is **r3** (`EXPERIMENT_NAME`/`PROJECT_NAME` =
`rl-glm53f-gbash-r3` in `trainer-pytorchjob.yaml`; `miles-config.yaml` still
records `r1` — inert, the launcher never reads it). r1 and r2 both died in
train step 0, before `save_interval 20`, so no checkpoint exists and nothing
resumes; the fresh name keeps the W&B group and checkpoint dir separate from
the dead runs.

Every replica tees its stdout to
`/mnt/scratch-s3files-rw/guparpit/logs/${EXPERIMENT_NAME}/trainer-<idx>.log`
and a 60 s host-memory sampler writes `memsample-<idx>.log` next to it. The
kubeflow operator deletes a failed PyTorchJob together with its pods, so that
EFS copy is the only post-mortem source.

**kueue TAS mis-pin check — within ~2 min of every apply.** kueue's
topology-aware scheduler writes a `kubernetes.io/hostname` pin into each pod;
when that node is occupied outside kueue's view the pod stays `Pending` with
`Unschedulable`/`FailedScheduling` (Insufficient `nvidia.com/gpu` /
`vpc.amazonaws.com/efa` / memory), the 12-pod gang never binds, the ~10 min
pods-ready timeout evicts it and the operator records a FAILED job (logs
gone). Reproduced on 2026-09-02 (convert job, run-1 launch) and 2026-09-03
(r3, 12 nodes at once).

```bash
kubectl get pods -n arena-tasks -l app=rl-glm53f -o wide          # all 12 Running?
kubectl describe pod -n arena-tasks <pending-pod> | grep -E 'Unschedulable|FailedScheduling|Insufficient'
kubectl get pod -n arena-tasks <pending-pod> -o jsonpath='{.spec.nodeSelector}'   # the TAS pin
```

Procedure: add every pinned-but-rejected hostname to the
`kubernetes.io/hostname` `NotIn` list in `trainer-pytorchjob.yaml` (16 entries
as of 2026-09-03; integration commits 86d2d0ec5 and earlier), then
`kubectl delete pytorchjob rl-glm53f-trainer -n arena-tasks` and re-apply
under the SAME job name (`rdzvId`, `JOBNAME`, the `rl-glm53f-sglang` Service
selector and the NATS URL are bound to it). Karpenter refuses to provision
for hostname-pinned pods, so the gang binds only to existing nodes. Drop
exclusions as nodes heal — each one costs capacity.

### 5. First-run verification

- **EFA is load-bearing**: verified in runs 1-2 (aws-ofi-nccl 1.18.0 +
  Libfabric 2.4 selected on all ranks), so `NCCL_DEBUG=INFO` /
  `NCCL_DEBUG_SUBSYS` were dropped from the manifest (INFO at 96 ranks
  inflated the driver log to 48k lines). Re-add both only after an image or
  base change, and then require `NET/OFI Selected provider is efa` /
  `Loaded net plugin Libfabric`; `Using network IB` or `Socket` means the
  ofi-nccl plugin did not load — stop, do not let a 64-rank all_gather "run"
  over TCP.
- **Run-2 OOM signature absent**: no `Workers (tasks / actors) killed due to
  memory pressure` in `trainer-0.log`; `memsample-<idx>.log` growing on every
  replica; engine `sglang::scheduler` RSS a few GiB per rank (r2: ~3.4 GiB),
  not ~79 GiB (that would mean the WeightChecker snapshot is back on).
- **r2 signature absent**: no `Stale file handle` in train step 0 — the
  kernel caches must resolve under `/tmp/kernel_cache` (container command),
  never EFS.
- Rendered `--load` names the new experiment dir; no
  `Loaded slime extra state`/`Restored wandb_run_id` lines (those mean
  resume).
- Data source pulls the lakeFS manifest (2922 rows); engines pass router
  health (allow up to 40 x 15 s — DSA/tilelang startup is slow); gym workers
  log `Completed <task>: n/m real`.
- W&B (`arena/rl-snorkel27`, group `rl-glm53f-gbash-r3`): binary-reward
  signal check as in the snorkel README — if >~80% of groups are
  zero-variance early, stop and revisit.
- `rollout/truncated_ratio` ~0.89 is the KNOWN Harbor agent-timeout
  signature (Incident history), not a clipping problem; on image d the
  rollout summary's `Removal reasons: timeout=.., context_error=..,
  length=.. (kept_timeout=N)` line attributes it directly.

### Teardown

```bash
kubectl delete pytorchjob rl-glm53f-trainer -n arena-tasks
kubectl delete pytorchjob rl-glm53f-convert -n arena-tasks
kubectl delete configmap rl-glm53f-trainer-config -n arena-tasks
kubectl delete -f sglang-svc.yaml
# gym side: kubectl delete -f gym-worker.yaml -f nats.yaml
# fetch: kubectl delete job glm53-fetch-20260902 -n arena-tasks
#        kubectl delete configmap glm53-fetch-script -n arena-tasks
# memprobe (see memprobe/README.md): kubectl delete job glm53-memprobe-<suffix> -n arena-tasks
#        kubectl delete configmap glm53-memprobe-scripts -n arena-tasks
```

Checkpoints: at 321B a DCP save is roughly the 643 GB weight footprint plus
optimizer state per save point (`save_interval: 20`, `no_save_optim` trims
it); prune `${ARENA_CHECKPOINTS_DIR}/slime_experiments/rl-glm53f-gbash-r3`
(and any r1/r2 dirs, which should be empty) when done.

## Deviations

Legend: **forced** = the model swap / platform makes the parent value wrong;
**chosen** = a judgment call, revisit freely.

### vs (a) the r5 snorkel config (`harbor-rl-27b-snorkel/`)

| setting | this config | source/reason |
|---|---|---|
| `experiment_name`/`project_name` | `rl-glm53f-gbash-r1` in the YAML (inert record); pod env `EXPERIMENT_NAME`/`PROJECT_NAME` = **`rl-glm53f-gbash-r3`** is authoritative | forced — new run identity (ADR-0004: never reuse; r1/r2 saved nothing) |
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
| `rollout_health_check_interval`/`_timeout` | 300 / **900** (absent before; recipe 300/300) | forced/chosen — recipe knobs; timeout raised for r2 because with FT on the probe competes with ~100 running + ~30 queued 30k-token requests per engine (KV usage 0.97 in run 2) and 300 s risks killing a healthy engine. Still catches a dead engine within one rollout |
| `use_fault_tolerance` | `true` (absent before) | chosen — without it both health knobs are INERT (`rollout_manager.py:115-123`) and one dead engine kills the run (run 2 served 32 min on 3 engines, then died at `update_weights`). With it the monitor stops the dead engine and `recover_updatable_engines` rebuilds it at the next update. UNVALIDATED on the arena NATS path (see Risks) |
| `data_pad_size_multiplier` | 512 (default 128; absent before) | chosen — pads each thd micro-batch to a multiple of TP*512 = 4096 tokens; `tilelang_sparse_mla_bwd` takes S/S_kv as static ints and re-JIT'd for every distinct padded length (~3300 compiles across 64 ranks in run 2, ~12 s each). Distinct lengths ~37 -> ~10 for ~+6.5% pad tokens; verify loss parity |
| `max_tokens_per_gpu` | 8192 (was 65536) | forced — recipe token budget for 321B at full recompute; `use_dynamic_batch_size: true` **kept** (arena packing mechanism), `micro_batch_size: 1` added as the recipe-faithful fallback (ignored under dynamic batching). Reconciliation documented in the YAML |
| `update_weight_buffer_size` | 1073741824 (absent before) | forced — recipe (1 GiB staged weight-update buffer) |
| `train_memory_margin_bytes` | 3221225472 (absent before) | forced — recipe |
| `qkv_format` | `thd` (absent before) | forced — recipe; the tilelang DSA path requires it |
| `model_name` | `glm5_next` (absent before) | forced — recipe; selects the megatron<->sglang weight mapping in the weight updater/HF export (the HF config class name alone does not resolve it) |
| `lr` | 1e-6 (was 15e-7) | chosen — GLM recipe optimizer block taken whole per the port decision |
| trainer manifest: replicas/elastic | 12, `maxRestarts: 0`, `restartPolicy: Never` (was 6, 0, Never) | chosen — replicas forced by GLM; restart semantics kept at the known-good arena values. OnFailure+3 was considered and reverted: recovery is unproven on this ray-based launcher (needs all 11 workers to exit on head-GCS loss or the restarted head hangs in `_wait_for_ray_nodes`); revisit after run 1 |
| `enable_trajectory_tracing` | **dropped** (parent carried `false`) | forced — no miles flag with that name is registered (deploy.sh-era key); `false` was inert, `true` would crash the trainer at argparse |
| `use_kl_loss` / `kl_loss_type` | **removed** since r2 (parent: `true`/`low_var_kl`); `kl_coef`/`kl_loss_coef` stay pinned `0.0` | chosen — with coef 0.0 the gradient is identical, but `use_kl_loss` forced with_ref=True: ref checkpoint load, a second pinned bf16 CPU backup per actor rank, and a full ref forward over the GBS every step = 2388 s of run 2's 7722 s step 0 (31%). Upstream `run_glm5_3_flash.py` runs no ref pass. Only the `train/kl_loss` diagnostic is lost |
| `check_weight_update_equal` (+ `check_weight_update_skip_list`) | `false` since r2 (run 1-2: `true` + `visual.`) | chosen — the flag's cost is NOT a per-update compare but a permanent ~73 GiB anonymous CPU copy of every TP rank's shard (sglang WeightChecker snapshot, `weight_checker.py:114-121`, issued once by `miles/ray/placement_group.py:213-217`, never freed): 8 x ~79 GiB = ~632 GiB of the 1934 GiB Ray counted at the run-2 OOM. Step-0 megatron->sglang equality was verified in run 2. `false` -> the launcher drops the flag; skip list removed with it. Re-enable only in a single-engine smoke |
| `arena_mask_clipped_final_turn` | `true` (flag new in image c; absent in parent) | chosen — run 2: truncated_ratio 0.898, ess_ratio 0.099 because any clipped final turn removed the whole sample (r5-lineage policy); now only that turn's loss-mask is zeroed. REQUIRES image >= `miles-glm53-20260902c`. Inert in r2 (the removals turned out to be agent timeouts, not clips — see Incident history) |
| `arena_keep_timeout_trajectories` | **not set** (default off; flag exists only in image d) | open — keeps clean Harbor agent-timeout trajectories (~89% of removals) as training samples; see Risks/open items before enabling |
| trainer manifest: image | `arena-slime-dev:miles-glm53-20260902c` (run 1: a, run 2: b) | forced — GLM needs the glm53next SGLang/Megatron stack; c adds the fla KDA patch, the mask-clipped flag and the hf_export ENOTSUPP fix (lineage table in step 2) |
| trainer manifest: node-exclusion affinity | parent's `i-09907cd660684a460` **dropped**; a live `kubernetes.io/hostname NotIn` list of 16 kueue-TAS-mispinned nodes **added** (2026-09-02/03) | chosen — the parent entry was a point-in-time S3-CSI workaround; the current list is the only fix for TAS pinning pods to nodes the scheduler rejects (step 4). Prune as nodes heal |
| trainer manifest: `RAY_memory_monitor_refresh_ms=0` | added (pod env; `scripts/run_arena_harbor.py` also `setdefault`s it before `ray start`) | forced — run-2 root cause: with no cgroup limit visible, Ray 2.58's monitor measured the WHOLE node (total = MemTotal) and its by_time policy killed the SGLangEngine + 8 `_HttpPosterActor`s at 95% node-wide usage while the pod held ~633 GiB. `0` installs a NoopMemoryMonitor on every raylet (pod env is inherited by `ray start`; ray-job `runtime_env` would NOT reach it). Same setting as miles' own GLM-5 launchers |
| trainer manifest: `resources.limits.memory` / `/dev/shm` | limit **1800Gi** (request 1200Gi) / emptyDir `sizeLimit: 256Gi` (parent: request 1800Gi, no limit, unbounded shm) | chosen — request lowered after run 1 (kueue counts requests: at 1800Gi only 11/304 p6 nodes were TAS-assignable); limit + shm cap added in r2 as OOM backstops: a real exhaustion OOM-kills inside our cgroup instead of the node. NOTE containerd gives privileged containers no cgroupns, so Ray may still read host totals — `refresh_ms=0` is the fix, this is the backstop. 256Gi keeps the ~186 GiB Ray object store on shm; exceeding the sizeLimit EVICTS the pod |
| trainer manifest: kernel JIT caches | `TILELANG_CACHE_DIR`/`TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR` exported in the container command under local `/tmp/kernel_cache` (parent: Triton/Inductor env entries, default locations) | forced — r2 put them on EFS for persistence and train step 0 died in Triton's autotuner with `OSError [Errno 116] Stale file handle` (8 ranks/node racing on NFS). Local disk only; recompiles per pod start are the accepted cost |
| trainer manifest: host-memory sampler | added — background loop in the container command writing `memsample-<idx>.log` (60 s: node meminfo, own cgroup via `/proc/self/cgroup`, `/dev/shm`, anon/file RSS of every sglang scheduler / actor process) next to `trainer-<idx>.log` on EFS | chosen — run 2's driver log had exactly one host-memory snapshot; `node_used - cgroup_used` over time is the only way to attribute co-tenant vs own growth (open item: ~1.3 TB unattributed) |
| gym-worker manifest: node anti-affinity | required `NotIn` on `node.kubernetes.io/instance-type: p6-b200.48xlarge` and `node-type: p6-gpu` (added r2; run 2 had none) | chosen — p6 nodes are untainted and the trainer's 1200Gi / 0-CPU request leaves ~746 GiB + ~190 vCPU per engine node for the scheduler to fill with our 128 gym+dind pods. Repels only OUR pods; other submitters need a platform taint |
| trainer manifest: `WANDB_API_KEY` secretKeyRef (`wandb-env-arena`) + `WANDB_BASE_URL` | added | chosen — smoke-proven pattern (snorkel manifest predates the secret) |
| trainer manifest: `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH` | added for runs 1-2, **dropped** since r2 | chosen — EFA/libfabric selection verified (aws-ofi-nccl 1.18.0 + Libfabric 2.4 on all ranks); INFO at 96 ranks bloated the driver log and Ray dedup hid per-rank counts. Re-add only for NCCL debugging |
| trainer manifest: `SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1`, `SGLANG_HEALTH_CHECK_TIMEOUT=120`, `PYTHONFAULTHANDLER=1`, `TORCHINDUCTOR_COMPILE_THREADS=1` | added | chosen — the upstream recipe's `extra_env_vars`, carried as pod env (inherited by the ray workers); the cache-dir entries moved into the container command (row above) |
| names | everything `rl-glm53f-*` | forced — parallel-run isolation (NATS streams are per-server; services/configmaps must not collide) |

Everything not listed (batch shape `rollout_batch_size 32` /
`n_samples_per_prompt 8` / `global_batch_size 256` / `num_rollout 90`,
over-sampling via `dynamic_sampling_filter_path` at the default 4x examine
cap, GRPO eps 0.4/0.4/c2.0 + low_var_kl at 0, `prompt-data-list` lakeFS pin,
rollout keys, wandb project/team, EFA env/resources/tolerations/hostPaths,
ephemeral 32Gi, kueue queue label) is **unchanged** from the snorkel parent
(memory request/limit and shm are listed above — they did change).

### vs (b) the upstream GLM-5.3-Flash recipe (PR #2786 `run_glm5_3_flash.py`)

| setting | this config | source/reason |
|---|---|---|
| `--colocate`, `--offload-train-target disk`, `--offload-train-disk-dir` | **dropped** | forced — those are colocate-only; the arena path is disaggregated (separate engine nodes) |
| actor shape | 8 nodes x 8 GPUs (DP=2) | chosen — the recipe's validated `(8,4)` = 32 GPUs at DP=1, scaled x2 data-parallel on arena capacity; per-replica parallelism identical |
| rollout/data path | arena NATS gym (`rollout_function_path`, `data_source_path`, lakeFS `prompt-data-list`) | forced — arena training path vs the recipe's dapo-math + `--rm-type math` |
| batch shape | rbs 32, n 8, GBS 256, num_rollout 90, response len 2048, temp 1 | chosen — r5-faithful arena shape (recipe: rbs 4, temp 0.8, response len 4096, num_rollout 5 smoke) |
| GRPO block | eps 0.4/0.4, `eps_clip_c` 2.0, `kl_coef`/`kl_loss_coef` 0 with `use_kl_loss` removed (since r2), `use_rollout_logprobs` | chosen — r5-faithful clipping (recipe: eps 0.2/0.28); the ref pass now matches the recipe (none) |
| `use_dynamic_batch_size` | `true` | chosen — arena packing mechanism kept; recipe ran static mbs 1 (its `max_tokens_per_gpu 8192` is honored as the budget) |
| `--check-weight-update-equal --check-weight-update-skip-list visual.` | kept for runs 1-2, **dropped since r2** | chosen — the first disaggregated (NCCL-bucketed) glm5_next update was verified equal at run-2 step 0; the flag's permanent ~79 GiB/rank CPU snapshot was a run-2 OOM contributor (table (a)) |
| checkpoint flow | `--load`/`--save`/`--save-hf` appended by `run_arena_harbor.py` from `EXPERIMENT_NAME` (ADR-0004), `save_interval 20`, `no_load_optim`/`no_save_optim` | forced — arena run-identity contract (recipe defaulted to `skip_saving`) |
| wandb | config keys (`wandb_host/project/team`) + `WANDB_API_KEY` secret | forced — arena tracking contract vs `U.get_default_wandb_args` |
| `sglang_context_length` / `rollout_max_context_len` | 131072 | chosen — arena parent values (recipe left engine defaults) |
| `sglang_disable_overlap_schedule` | `true` | chosen — arena precedent (AGISlime run5 + smoke); recipe does not set it. Engine-side conservative; candidate to lift later |
| `pin_rollout_manager_to_head`, `sglang_router_port 30000` | kept | forced — arena Service wiring |
| launcher-consumed keys (`user`/`cluster`/`replicas`/...) | present | forced — `run_arena_harbor.py` contract; never reach the trainer argv |

### Run-1 live change (2026-09-02, user-directed)

First rollout with the r5-parity caps (`ARENA_MAX_TOKENS=2048`,
`ARENA_ROLLOUT_CONTEXT_LIMIT=32768`, `rollout_max_response_len: 2048`) produced
28/28 `truncated`, 0 `success` — GLM-5.3 at its default `max` reasoning effort
spends the whole 2048-token turn thinking. Raised per user instruction to
**32768 per turn / 131072 per episode** (trainer + gym env together; engines
already ran `sglang_context_length: 131072`). Consequences to watch: single
samples can now exceed the 8192 `max_tokens_per_gpu` microbatch budget by ~16x
(allowed — a lone over-budget sample forms its own microbatch; memory analysis
had ~50 GB headroom), rollouts get slower per episode, and a ~131k-token
trajectory pushes the NATS result message toward the 8 MiB `max_payload`.

## Incident history

Dated, oldest first; each fix names the file that carries it. Driver logs
survive only on EFS (`/mnt/scratch-s3files-rw/guparpit/logs/<EXPERIMENT_NAME>/trainer-0.log`)
because the kubeflow operator deletes a failed PyTorchJob with its pods.

- **2026-09-02 run 1 (image a, `rl-glm53f-gbash-r1`)** — first rollout
  28/28 truncated at 2048 tokens/turn; caps raised live (previous section).
  Rollout 0 then returned 272 success / 54 failed / 0 truncated on the wire,
  but train step 0 crashed on every rank: fla 0.4.2's
  `chunk_kda_fwd_kernel_intra_token_parallel` declares
  `BK = triton.next_power_of_2(K)` inside the jit body and triton 3.7.1
  rejects it ("Unsupported function referenced"; the only in-kernel offender
  in fla). Fix: hoist BK to a host-side constexpr launch arg —
  `examples/arena/patches/fla_kda_next_power_of_2.py`, applied as an image
  layer by `examples/arena/Dockerfile` -> image `miles-glm53-20260902b`
  (integration 3c40b4660, 949434185). Also seen: only 23/256 samples carried
  loss (any step with `stop_reason=length` removed the whole sample) —
  misread then as length clipping; see the truncation RCA below.
- **2026-09-02 21:15 -> 09-03 01:05 UTC run 2 (image b, still r1)** — fla
  patch validated (step 0 on 64 ranks: loss -0.040, grad_norm 0.082). At
  00:33:57 the raylet on the engine node `trainer-worker-5` (10.100.176.96)
  logged "9 Workers killed due to memory pressure": node 1934.01 / 1996.03 GB
  (0.969), with `sglang::scheduler_TP0..7` at ~79 GB each (~632 GB) as the
  only large processes. Root causes: (1) Ray 2.58's monitor accounted
  **host-wide** (`/proc/meminfo` MemTotal-MemAvailable, total =
  2143217684480 B) because containerd gives the privileged container no cgroup
  namespace and it had no memory limit, so `/sys/fs/cgroup/memory.max` does
  not exist in-pod and the by_time policy fired on node usage. (2) The
  79 GiB/rank is the sglang **WeightChecker snapshot** from
  `check_weight_update_equal: true` — a permanent anonymous CPU copy of each
  TP rank's shard (`param.data.detach().cpu()`, weight_checker.py:114-121,
  issued once by `miles/ray/placement_group.py:213-217`, never freed).
  `use_fault_tolerance` was unset, so the health knobs were inert; the run
  served 32 min on 3 engines and died at the post-train `update_weights`
  (OutOfMemoryError re-raised for the dead actor). Fix set (integration
  3098ae1d3, image c): `RAY_memory_monitor_refresh_ms=0` pod env (every
  raylet; `scripts/run_arena_harbor.py` also `setdefault`s it),
  `check_weight_update_equal: false`, `limits.memory: 1800Gi` +
  `/dev/shm sizeLimit: 256Gi` as backstops (request stays 1200Gi),
  `use_fault_tolerance: true` with `rollout_health_check_timeout: 900`,
  gym-worker `NotIn` anti-affinity off p6 nodes, plus `use_kl_loss` removed,
  `data_pad_size_multiplier 512`, `arena_mask_clipped_final_turn`, the
  `memsample-<idx>.log` sampler and the `memprobe/` tooling. Run-2 perf
  facts: step 0 = ref_log_probs 2388 s + actor_train 5332 s (7722 s) vs
  steady-state rollout 3393 s (engine-bound); MFU 0.26%; every ~32k-token
  sample exceeds `max_tokens_per_gpu 8192`, so dynamic batching ran 127
  micro-batches of one sample; TileLang re-JIT'd the sparse-MLA backward
  ~3300 times per step.
- **2026-09-03 05:49-07:36 UTC r2 (image c, `rl-glm53f-gbash-r2`)** — all
  OOM fixes held (engines ~3.4 GiB host RSS/rank, no memory-pressure events,
  FT monitor healthy, initial weight sync 38 s, rollout 0 in 4938 s with
  135/20 success/failed). Train step 0 died 7 min in with
  `OSError [Errno 116] Stale file handle` in Triton's autotuner: the r2 patch
  had moved the TRITON/TILELANG/INDUCTOR caches to EFS for persistence, and
  8 ranks per node raced on the same NFS cache files. Reverted to local
  `/tmp/kernel_cache` in the container command (integration 0ee733986);
  recompiles per pod start are the accepted cost. Rule: kernel JIT caches
  never go on EFS/NFS. Also: `arena_mask_clipped_final_turn` salvaged
  nothing (removed 229/256, truncated_ratio 0.8945, zero "masked clipped
  final turn" lines).
- **2026-09-03 truncation RCA (gym-pod trial artifacts + task-events logs)**
  — the ~89% removed/"truncated" samples in r1/r2 are Harbor **agent
  wall-clock timeouts** (`AgentTimeoutError`; per-task `[agent] timeout_sec`
  900-1800 s, no gym override), NOT length clips: 0/1033 generates hit the
  32k per-turn cap (max seen 15k). The gym maps the exception to
  `agent_stop_reason="timeout"`, which `nats_rollout.py` treats as degenerate
  -> TRUNCATED + remove_sample, although these trajectories are clean (every
  turn `finish_reason=stop`, no partial turn, verifier ran) and ~9% carry
  reward 1 — >=19 of the ~46 reward-1 samples per step were discarded.
  Driver: ~14.5 tok/s decode per sample with ~400 concurrent trials on 4
  saturated engines; GLM thinks 8-15k tokens/turn -> ~500 s/turn. Mitigation
  shipped: `--arena-keep-timeout-trajectories` (default off; integration
  32da04357, image `miles-glm53-20260903d`, tests included) plus the
  per-rollout `Removal reasons: timeout=.., context_error=.., length=..
  (kept_timeout=N)` log line. Not yet enabled in `miles-config.yaml`.
- **2026-09-03 08:3x UTC r3 launched** (image c, local caches, identity
  r3): kueue TAS pinned pods to 12 more nodes the scheduler rejected; they
  were added to the `NotIn` list (integration 86d2d0ec5) and the job
  resubmitted under the same name (procedure in step 4).

## Risks / open items (watch on r3)

1. **GLM chat/tool-call format vs the gym parser — UNVERIFIED.** The snorkel
   textual bash agent worked against qwen output; GLM-5.3's template and
   tool-call conventions differ, and no sglang tool-call/reasoning parser is
   configured. If early trajectories show malformed command extraction or
   zero real completions, this is the first suspect.
2. **EFA pairing — VERIFIED in runs 1-2**: aws-efa-installer 1.47.0
   (aws-ofi-nccl 1.18.0, Libfabric 2.4) with the glm53next base's NCCL
   2.29/cu13 selected the libfabric/efa provider on all ranks. Re-verify
   with `NCCL_DEBUG=INFO` after any image/base change (`Using network IB` or
   `Socket` = broken fallback).
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
6. **Disaggregated glm5_next weight update — verified equal at run-2
   step 0** (NCCL-bucketed UpdateWeightFromDistributed; the recipe had only
   validated colocated UpdateWeightFromTensor). `check_weight_update_equal`
   is now OFF (its snapshot was a run-2 OOM contributor), so the remaining
   guard is the engine-log signature `The full weights of the ModelRunner
   are partially updated`; the cross-bucket hazard (fused
   q_a_proj/kv_a_proj_with_mqa) stays covered by miles' q_lora atomic-update
   grouping.
7. **Token budget vs trajectory length — confirmed inert batching**: every
   ~32k-token sample exceeds `max_tokens_per_gpu 8192`, so run 2 trained
   127 micro-batches of one sample each (memory held). Watch actor memory on
   long-trajectory steps; first mitigation is raising recompute or lowering
   the gym context limit, not raising the budget. CP is unsupported (kpool
   indexer), so long samples cannot be split across ranks.
8. **Unattributed ~1.3 TB on run 2's dead engine node.** Ray counted
   1934 GB used while the pod's processes summed to ~633 GB. The dead node
   (`i-099b9859c5430fec8`) hosted only 3 small co-tenant pods and survivors
   hosted more, so the co-tenant theory is refuted; engine USS grew only
   ~5.6 GiB in 3 h, so it is not an engine leak. The remainder is external
   to the pod (mount-s3 CSI page cache, driver pinned pages, kernel —
   unknown). The `memsample-<idx>.log` sampler now records node vs own-cgroup
   vs per-process anon/file every 60 s on the real run, and the isolated
   single-node probe lives in `memprobe/` (hypotheses R1-R5 and how to read
   the CSVs: `memprobe/README.md`; Job `glm53-memprobe-b2` ran rc=0, results
   under `/mnt/scratch-s3files-rw/guparpit/logs/glm53-memprobe/`).
   `refresh_ms=0` makes the run immune to Ray's verdict, but a real node-wide
   exhaustion would still end in the kernel OOM killer.
9. **Timeouts dominate removals (~89%)** — the reward-1 samples lost per
   step are training signal. Options, cheapest first: (a)
   `arena_keep_timeout_trajectories: true` (needs image d; status stays
   TRUNCATED so `truncated_ratio` stays honest; never keeps a trajectory
   with a clipped turn); (b) raise the Harbor per-task `timeout_sec`
   (dataset/gym-side change); (c) wire `reasoning_effort` /
   `max_thinking_tokens` through the gym's arena-sglang path
   (`ArenaSGLangLLM` currently drops them) to cut the 8-15k-token thinking
   per turn; (d) more engine throughput (more engine nodes or fewer
   concurrent trials per engine) — ~14.5 tok/s/sample today.
10. **Actor perf (run 2)**: KDA layers are replicated across TP (upstream
    design), dynamo hits its recompile limit on the mHC hyper-connections,
    CP is unsupported, MFU 0.26%. `data_pad_size_multiplier 512` bounds the
    TileLang backward re-JIT count — verify loss parity of the pad segments
    in a smoke; with local caches every pod start recompiles.
11. **`use_fault_tolerance` is UNVALIDATED on the arena NATS path**: kill one
    engine's sglang in a 5-node smoke first. A 900 s `health_generate` under
    100-deep queues could still false-kill a healthy engine and drop its
    in-flight trajectories.

## r5 gym image: reasoning effort + splice fix (2026-09-03)

`gym-worker.yaml` now runs `arena-tasks-dev:glm53-reasoning-20260903b`
(AREnATasks HEAD 35f7ba7 + an uncommitted `amzn_arena_harbor` patch, copy at
`/workplace/guparpit/miles/arena-port-artifacts/arenatasks-reasoning-effort.patch`).
What changed versus the r1-r4 image `rl-smoke-20260821b`:

- `ARENA_REASONING_EFFORT` (set to `low`) renders GLM-5.3's chat-template
  `reasoning_effort` variable into the first-turn prompt as
  `<|system|>Reasoning Effort: Low`. The template honors only `low`/`high`;
  anything else renders the default `Max` (the client warns once). Motivation:
  at the default effort GLM-5.3 thinks 8-15k tokens/turn and 87-93% of r4
  trials hit the 900-1800 s task.toml agent timeout.
- Splice fix: GLM ends an assistant turn by emitting `<|user|>` as its stop
  token, and the template suffix for the next turn starts with `<|user|>` too.
  r1-r4 token streams therefore carried `<|user|><|user|>` at every turn
  boundary (8/8 boundaries in a sampled live rollout). The client now splices
  the role token once, matching a full re-render.
- HEAD gym CLI contract: `--mode rollout --agent arena-terminus-2` are
  required for parity (HEAD defaults to the Vulcan agent and requires
  `--mode`). Ships harbor 0.22.0 (was 0.21.0); the result envelope now reports
  timed-out groups as status `truncated` rather than `success` (the trainer
  salvages both, so this is telemetry only).
- Build: `brazil-build docker-push-arena <tag>` from the AREnATasks tree (needs
  the mise Python 3.12 on PATH on this host). Push goes to us-east-1 and
  replicates to ap-south-1 within ~1 min.

## r6 (2026-09-04): 16 engines, batch 64, 2x agent timeout

- Trainer 24 replicas (8 actor + 16 engine nodes); `rollout_batch_size` 64 /
  `global_batch_size` 512 lifts the publisher in-flight cap (2x batch) from 64
  to 128 groups so the engines see ~64 requests each (r5: ~130 with 10-60
  queued on KV). `num_rollout` 130 ~= 10 passes over the task list (dynamic
  sampling publishes ~3.5x the kept groups). Gym 160 replicas (>= 128 + slack).
- Gym image `glm53-reasoning-20260904a`: AREnATasks mainline c1a0439 (adds
  `ARENA_AGENT_TIMEOUT_MULTIPLIER`, eval path only) + fix passing it on the
  training path + the r5 reasoning-effort/splice patch rebased on top
  (`arena-port-artifacts/arenatasks-reasoning-effort.patch`).
- `ARENA_AGENT_TIMEOUT_MULTIPLIER=2`, `ARENA_NATS_ACK_WAIT=6000` (deadline math
  in gym-worker.yaml). `sglang_disable_radix_cache` deliberately unchanged.
- r5 reference (4 steps): rollouts 3874/3197/2592/2718 s, reward
  0.273/0.328/0.367/0.398, ~90% of published trials timed out, ess 0.97,
  ppo_kl 0.011, actors ~21% duty (rollout-bound).

## r7 (2026-09-05): r6 shape + 1800 s SGLang call timeout + reasoning effort High

Same 24-node shape, batch 64 / GBS 512, 2x agent timeout and 160 gym workers as
r6. Two gym-side changes, prefix `rl-glm53f7`, experiment `rl-glm53f-gbash-r7`:

- gym image `arena-tasks-dev:glm53-sgltimeout-20260904b` (AREnATasks branch
  guparpit/reasoning-effort + `ARENA_SGLANG_REQUEST_TIMEOUT_SEC`): the /generate
  httpx read timeout was hardcoded at 600 s and r6 lost 33-74 of 512 samples per
  rollout to `httpx.ReadTimeout` ("Unknown Error in LLM interaction: ", empty
  message) on 32k-token turns. r7 sets it to 1800 s.
- `ARENA_REASONING_EFFORT=high` (r5/r6 ran low; template honours only low/high,
  anything else renders Max).

The gym start script keeps the r6 fresh-node fixes (ECR-mirrored alpine probe
image, prebuilt egress sidecar, model-mount guard) -- without them every task is
rejected on nodes that cannot pull from Docker Hub.
