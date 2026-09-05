# Run log: GLM-5.3-Flash on the arena Harbor gym (`harbor-rl-glm53-flash`)

Dated history of the GLM-5.3-Flash (321B MoE, `glm5_next`) GRPO bring-up on
prod-bom against `snorkel-general-bash-harbor` over NATS: what was tried, what
happened (with the numbers the logs actually carried), root causes, what was
decided and why, and the identity of every run (`EXPERIMENT_NAME`, trainer and
gym image tags, W&B group). It is the run-history counterpart of
`miles_plugins/arena/adr/` (decisions) and of this directory's `README.md`
(the current runbook). Driver logs survive only on EFS under
`/mnt/scratch-s3files-rw/guparpit/logs/<EXPERIMENT_NAME>/trainer-<idx>.log`
(the kubeflow operator deletes a failed PyTorchJob with its pods); metrics are
in W&B `arena/rl-snorkel27`, one group per `EXPERIMENT_NAME`.

## Format

- Entries are chronological, oldest first: `## YYYY-MM-DD HH:MM PT - <milestone or run>`.
  All times are Pacific (sources are UTC; PDT = UTC-7).
- Numbers come from trainer/gym logs, W&B, HF API responses or commit messages;
  a number that is in no source is omitted rather than estimated.
- Run identity is the `EXPERIMENT_NAME` pod env (`rl-glm53f-gbash-r<N>`), never the
  YAML. Naming trap: GLM "run 1" and "run 2" are two launches under identity
  `r1` (trainer images a and b); `r2` onwards are distinct identities. "r5-faithful"
  / "run5" in the configs means AGISlime snorkel run 5 (2026-08-29), not GLM r5.
- Integration SHAs (`b981804e7`, `3098ae1d3`, ...) are commits of the image build
  tree (`/tmp/glm53-integration`, branch `glm53-arena`), archived in this fork as
  `refs/archive/glm53-arena`; the upstream PR history is `refs/archive/pr2786`.
- Trainer images are `427267593057.dkr.ecr.<region>.amazonaws.com/arena-slime-dev:miles-glm53-<tag>`
  (pushed us-east-1, replicated to ap-south-1); gym images are `arena-tasks-dev:<tag>`.

## 2026-09-01 evening to 2026-09-02 00:27 PT - Kickoff and recon: model, weights, upstream support

User request (exact time not recorded; the work precedes the 00:27 PT upstream
clone): fetch GLM-5.3-Flash to fast scratch, convert it for training, and build
performant EFA arena configs, taking inspiration from miles' own GLM recipes.

- **Model facts** (HF API, verified 2026-09-02): `zai-org/GLM-5.3-Flash` = 321.3B total /
  ~18B active, `Glm5NextForConditionalGeneration`, `model_type glm5_next`; 45 layers
  (3 dense + 42 MoE, 288 experts, top-8 sigmoid) + an MTP layer 45 that training drops;
  34 KDA linear-attention + 11 DSA sparse layers (indices 3, 7, ..., 43); mHC
  hyper-connections; NoPE MLA (`qk_rope_head_dim = 0`, no `rope_theta` in the config);
  vision tower 563M (skipped for RL); vocab 154880; eos `[154820, 154827, 154829]`;
  `transformers >= 5.16`.
- **Weights: BF16, not the main repo.** The main HF repo is FP8 (328 GB, 62 shards) and the
  official `GLM-5.3-Flash-BF16` repo is 643 GB / 120 shards; both were re-uploaded
  2026-08-31, AFTER the upstream recipe page (`docs/models/glm/glm5-3-flash.md`, #2788,
  2026-08-27) was written with `hf download zai-org/GLM-5.3-Flash` and no cast. Nothing in
  the HF -> torch_dist converter dequantizes FP8. Decision: fetch BF16, pin revision
  `61f77a1e`.
- **Upstream support** is not in `radixark/miles` main. It is PR #2786 (branch
  `zhichen/glm53-flash`, Zhichen Zeng), head `dbbd610e7` (2026-08-29 04:03 PT). The recipe
  page pins `1cd14c00` (2026-08-27 12:16 PT), six commits stale and missing three real
  fixes: `eee9819da` (rope_theta absent from the checkpoint -> inert 10000 default),
  `75ace38a1` (DSA indexer fields resolved via `text_config` for composite HF configs -
  GLM-5.3-Flash's config is composite because of the vision tower), `dbbd610e7` (kpool
  indexer pinned to per-sequence pools in a packed batch). Aside: the page says rotary
  base 800000; the PR head's `scripts/models/glm5.3-flash.py` uses `--rotary-base 10000`
  (inert under NoPE).
- **Stack pins.** The PR needs SGLang branch `sglang-miles-glm53next@9a26e749` and
  radixark/Megatron-LM PR#89 `e8f57451`; all three are baked into
  `docker.io/radixark/miles:glm53next` (multi-arch), with miles at `1cd14c00`. Our
  `examples/arena/Dockerfile` already does `rm -rf /root/miles; COPY . /root/miles`, so a
  tree carrying the PR head + the arena port overlays the image's stale checkout.
- **Recipe facts to inherit** (`scripts/run_glm5_3_flash.py` at the PR head): actor
  TP8+SP / PP4 (11/11/11/12 layers) / CP1 / EP16 / ETP1; engines 8 GPUs at SGLang TP8/EP8,
  DSA prefill+decode backend tilelang, bf16 KV cache, mem-fraction 0.7, chunked prefill
  8192, radix cache off; recompute full/uniform/1, mbs 1, max-tokens-per-gpu 8192; Adam
  lr 1e-6 constant, wd 0.1, betas 0.9/0.98; 1 GiB update-weight buffer. Upstream validated
  it COLOCATED with `--offload-train-target disk` on 16 x 4 GB300 (288 GB); arena runs
  DISAGGREGATED on p6-b200 (8 x B200 180 GB, 8 EFA), so the colocate/offload flags must
  not carry over. Planned arena shape: 8 actor nodes (DP2) + 4 engine nodes. Upstream's
  healthy-run reference: `train_rollout_logprob_abs_diff` 0.0068-0.0106, `raw_reward`
  0.5 -> 0.94 in 10 rollouts, `ppo_kl` ~2.6e-4, `grad_norm` 0.31-0.49 (DAPO-Math-17k).

## 2026-09-02 00:27-00:55 PT - Merge PR #2786 onto the fork base (ADR-0008)

Sequence (all PT): 00:27 clone upstream to `/tmp/glm53-upstream` (main was at
`1e3b7a088`, 2026-09-01 23:33 PT); 00:30 worktree at the PR head `dbbd610e7`; 00:32 trial
merge `55600acfd` of fork base `2799fe38` + `dbbd610e7` -> no conflicts, tree
`84d28154`; 00:54 the real merge `b981804e7` ("Merge radixark/miles PR #2786
(GLM-5.3-Flash) onto arena fork base 2799fe38") on the integration branch
`glm53-arena`, same tree; 00:55 the port's uncommitted working tree committed on top
as `7182346d7` (49 files, +14004/-2; the 4 core edits apply cleanly over the PR).
The fork worktree itself stayed uncommitted; the images were built from the
integration tree.

- Merge base with upstream main is `071fd2a6f` (2026-08-27 12:44 PT, the recipe docs
  commit); the fork base is 25 main commits past it. PR delta = **22 files, +1578/-61**
  (`git diff 2799fe386 b981804e7`); the 168-file raw diff between PR head and fork
  base is main drift, not PR content. Base kept at `2799fe38`, not bumped to main.
- Only file both sides touch: `miles/utils/arguments.py` - fork adds the `qwen3_5`
  loss-mask choice (~L2408), the PR edits `parse_args` (L2713-2725:
  `rollout_indexer_topk_num_streams`, indexer fields read from `text_config`). Disjoint.
- Core hunks that ride along: `hf_config.py` (`glm5_next` config alias on
  `Glm4vMoeConfig`, `is_dsa` accepts `glm5_next`), `sglang_rollout.py` and
  `generate_endpoint_utils.py` (all-zero `routed_experts` and indexer stream-count
  asserts), `train_data_conversion.py` (fail fast when replay payloads are missing),
  `megatron_to_hf/__init__.py` (+`glm5_next.py` converter), `mbridge/__init__.py`
  (+`Glm5NextBridge`), `reloadable_process_group.py` (drop the python collective
  overrides: torch 2.13 `PyWorkHolder` use-after-free). New: `miles_plugins/models/glm5_next/`
  (KDA, kpool DSA indexer, mHC spec), `scripts/models/glm5.3-flash{,-4layer,-8layer}.py`,
  `scripts/run_glm5_3_flash.py`, `tests/fast/test_kpool_indexer_packed.py`,
  `tests/e2e/.../test_glm5_3_flash_4layer_ci.py`.
- Pinned at `dbbd610e7`, not the doc's `1cd14c00` (three fixes above). The PR is
  unmerged upstream and may force-push; its 33 commits are kept as `refs/archive/pr2786`.
- Verified 2026-09-05 on the equivalent tree (fork + PR + port, integration `3a6cbe5ba`):
  `tests/fast/plugins` + `tests/fast/utils/test_loss_mask_qwen3_5.py` 177 passed;
  launcher snapshots (`tests/manual/launch_scripts -k arena`) 4/4; `tests/fast/launch_scripts`
  43 passed; `test_arguments.py` + `test_hf_config.py` + `test_kpool_indexer_packed.py`
  159 passed / 1 skipped; `miles_plugins.models.glm5_next` imports.
- Next: overlay this tree onto `radixark/miles:glm53next`, add the EFA layer, fetch and
  convert the BF16 weights, write the 12-node config (following entries).

## 2026-09-02 01:40 PT — Image-a prerequisite: guard the anthropic SGLang imports (integration 664b2ab98)

Not a run: a two-file fix the first trainer image needed before anything could
launch. Cherry-picked into the fork so the branch equals the image tree file for
file (the PR #2786 merge above was the first image-tree-only change; this is the
second and last).

**Symptom.** With the fork tree overlaid on `/root/miles` of the
`radixark/miles:glm53next` base (ECR mirror
`arena-slime-dev:glm53next-upstream-20260902`), the trainer fails at import:
`miles/rollout/session/sessions.py` and `anthropic_adapter.py` import
`sglang.srt.entrypoints.anthropic.utils` / `.serving` at module load, and the
base's SGLang (`sglang-miles-glm53next@9a26e749`) predates those helpers.

**Why the base and the tree disagree (git ancestry, verified).**

| what | commit | date (PT) |
|---|---|---|
| miles baked into the glm53next base | `1cd14c00` (PR #2786 branch, "ci: set CUDA_DEVICE_MAX_CONNECTIONS=1 ...") | 2026-08-27 12:16 |
| anthropic-messages imports added to sessions.py / anthropic_adapter.py | upstream `11563829a` feat(session): serve Anthropic Messages through SessionCore (#2358) | 2026-08-29 18:34 |
| PR #2786 head merged in the previous commit | `dbbd610e` — does NOT contain 11563829a | 2026-08-29 |
| fork base | `2799fe38` — contains 11563829a | 2026-08-31 |

The base's own miles never imported these helpers, so its pinned SGLang never
needed them; overlaying fork main (which carries 11563829a) brought the
module-load imports in.

**Import chain that trips** (all module-level, nothing lazy):
`miles_plugins.arena.train_async_arena` -> `miles.ray.placement_group` ->
`miles.ray.rollout.rollout_manager` -> `miles.ray.rollout.router_manager`
(`start_session_server`) -> `miles.rollout.session.server` ->
`miles.rollout.session.sessions` -> `sglang.srt.entrypoints.anthropic.{utils,serving}`.
The Dockerfile import smoke (`import miles_plugins.arena.train_async_arena`)
exercises exactly this chain, so the build fails before a pod does.

**Fix (664b2ab98, 2 files, +15/-3).** `try/except ImportError` around the three
names in `sessions.py` (`anthropic_utils`, `convert_response`,
`convert_to_chat_completion_request`) and the one in `anthropic_adapter.py`
(`anthropic_utils`), each falling back to `None`, with an inline comment naming
the pinned branch. `sglang.srt.entrypoints.anthropic.protocol`
(`AnthropicMessagesRequest`, `is_server_tool`) stays unguarded: it resolves on
the pinned branch, only `utils` and `serving` were missing. On a current SGLang
the guards are no-ops.

**Alternatives rejected.**
- Newer SGLang in the image: the glm53next stack exists because mainline SGLang
  lacks the KDA/DSA engine support (Dockerfile header); swapping SGLang is not a
  config change.
- Pin the fork to the image's miles (1cd14c00): loses the 2799fe38 base the
  arena port was built and CPU-tested on.
- Drop the anthropic route from the tree: diverges from upstream and makes the
  re-sync harder once #2786 lands.

**Consequences.** The session server imports and serves as before; on the pinned
branch the anthropic-messages handler would fail at call time (`None` helpers).
The arena NATS path never calls it. No tests added (behaviour is unchanged where
the helpers exist). Chronology: landed between the EFA-layer Dockerfile fixes
(aebcc7ac3 01:17 PT, eda0f8454 01:27 PT) and the verifier pass (5ea810d7f
02:26 PT), i.e. during the image-a build iteration; baked into
`arena-slime-dev:miles-glm53-20260902a` and every later tag (b, c, 20260903d,
each a superset of the previous). Re-check the guards when #2786 or the SGLang
pin moves.

**Verification (2026-09-05, on the equivalent tree fork + PR + port):** 177
arena CPU tests, launcher snapshots 4/4.

## 2026-09-02 01:00-04:00 PT — Staging for the first GLM launch: image a, fetch, convert, verifier pass ("ALL DELIVERED")

Integration-tree commits behind this entry (all 2026-09-02 PT): aebcc7ac3 01:17 (EFA image
layer + `harbor-rl-glm53-flash/` config set, 8 files, +963) -> eda0f8454 01:27 (Dockerfile
apt-get fix) -> [664b2ab98 01:40, previous entry] -> 5ea810d7f 02:26 (verifier fixes) ->
1499bcef4 03:05 (fast-scratch paths, convert 1200Gi + TAS exclusion) -> 34d17a57e 03:35 (ref
DCP on EFS) -> 8660c520a 03:50 ("ALL DELIVERED" state). This commit stages the 8660c520a
versions (Dockerfile at eda0f8454; fetch assets unchanged since 01:04 PT).

### Trainer image a — `arena-slime-dev:miles-glm53-20260902a`

Composition (`examples/arena/Dockerfile`, top to bottom):

| layer | content | why |
|---|---|---|
| base | ECR mirror `arena-slime-dev:glm53next-upstream-20260902` of `docker.io/radixark/miles:glm53next` (SGLang `sglang-miles-glm53next@9a26e749` + radixark/Megatron-LM PR#89 `e8f57451`; its baked miles is pinned at 1cd14c00) | the mainline base has no KDA/DSA engine support |
| arena deps | `pip install nats-py>=2.6.0 kubernetes==35.0.0 lakefs boto3 omegaconf typer` | baked instead of per-pod-start installs; kubernetes 36.x returns 401 on EKS |
| EFA | aws-efa-installer 1.47.0 `--skip-kmod --no-verify`; build gate `test -f /opt/amazon/ofi-nccl/lib/libnccl-net.so` | recipe from AGISlime 47900d5; without the plugin NCCL silently falls back to raw IB/TCP and the first multi-node all_gather hangs (the 3-node smoke ran `FI_PROVIDER=tcp` at ~2% MFU) |
| tree | `rm -rf /root/miles && COPY . /root/miles` = fork + PR #2786 merge + session-import guards + arena port | the base's editable install points at /root/miles, so the copy is the install |
| smoke | arena-plugin imports + `run_arena_harbor.py --help` against the CUDA driver stub | fail the build, not the job |

- First build attempt died inside the EFA installer: its apt deps (pciutils /
  environment-modules / tcl) could not be installed without a prior `apt-get update`. Fix
  eda0f8454 (01:27 PT): `apt-get update` before the installer, `apt-get clean` after.
- Pushed through us-east-1 (the only region `arena-ecr-dev` can push to); the repo
  replicates to ap-south-1, where the cluster pulls. The README push command at this commit
  still names ap-south-1 directly; corrected in a later revision.
- Image lineage (each tag is a superset of the previous; rows b-d are built in the entries
  that introduce them and are listed here only so the lineage lives in one place):

| tag | adds | integration commit |
|---|---|---|
| a `miles-glm53-20260902a` | glm53next base + this tree + EFA layer (run 1) | eda0f8454 / 8660c520a |
| b `miles-glm53-20260902b` | fla 0.4.2 KDA kernel patch for triton 3.7.1 | 3c40b4660 |
| c `miles-glm53-20260902c` | `--arena-mask-clipped-final-turn`; hf_export ENOTSUPP fix | 5f8925db0, 44ddf62fb |
| d `miles-glm53-20260903d` | `--arena-keep-timeout-trajectories`; "Removal reasons:" summary | 32da04357 |

### Fetch — Job `glm53-fetch-20260902` (applied ~00:45-01:00 PT, SUCCESS ~35 min later)

- `fetch_glm53.py` via ConfigMap `glm53-fetch-script`; `general-cpu` node; image
  `arena-slime-dev:miles-arena-20260901b` (the smoke trainer image, has huggingface_hub);
  8 CPU / 24-32Gi / 100Gi ephemeral; 6 download threads; in-cluster pods have direct HF egress.
- Design: each file lands on node-local disk, then stream-copies (64 MiB chunks) to
  `/mnt/scratch-fast-1a-rw/durable/model-artifacts/external/zai-org/GLM-5.3-Flash-BF16`,
  because mountpoint-S3 has no rename (no download-in-place). Idempotent under Job retries
  (size-verified files skipped, short objects removed); `_FETCH_MANIFEST.json` written last.
  Fast scratch is the platform-injected mount; kyverno rejects manifests that declare
  platform PVCs, so none are declared.
- Result: 130 files / 642.7 GB (120 safetensors shards), rev `61f77a1e` (BF16 repo; the FP8
  main repo was rejected, see the opening entry). Destination `durable/` (no expiry) rather
  than `sandboxes/<alias>/` (60-day expiry). The README estimated "hours"; it took ~35 min.

### Verifier pass — 5ea810d7f (02:26 PT)

Three adversarial probes against the real code, CPU only (archived:
arena-port-artifacts/glm53/adv-glm53/probe_{argv,parse,mask}.py + argv.json):

1. `probe_argv.py`: yaml -> argv through the real `run_arena_harbor.py` converter under the
   exact pod env contract (REPLICA 12 / REPLICA_TRAINER 8 / EXPERIMENT_NAME
   `rl-glm53f-gbash-r1` / NATS_URL ...): 244 argv tokens.
2. `probe_parse.py`: strict `miles.utils.arguments.parse_args()` over that argv with the full
   validation chain (miles + megatron + sglang + hf), B200 device stubs (8 x 180 GB), the real
   GLM-5.3-Flash config.json. Passed; actor memory estimate ~130 GB peak of 180 GB per GPU.
3. `probe_mask.py`: what the parser-default `loss_mask_type=qwen` fallback does on the GLM
   tokenizer.

Findings -> fixes in 5ea810d7f:

- convert-job: `CONVERT_KEEP_PP1=1` (the doc-pattern command) would make each of 8 ranks build
  the full 321B model (~642 GB) and OOM; dropped, so the converter's PP -> world_size
  auto-reshape (PP8, worst rank ~87 GB bf16) is what fits. `du -sb` aggregate verify replaced
  by a per-file size-manifest diff (directory inode sizes differ tmpfs vs mountpoint-S3).
  Request 1900Gi -> 1800Gi (platform-proven value).
- trainer: `OnFailure`/`maxRestarts 3` reverted to `Never`/`0`. Restart recovery is unproven
  on the ray-based launcher: if any of the 11 workers does not exit on head-GCS loss the
  restarted head waits forever in `_wait_for_ray_nodes`. Revisit after a clean observation.
- miles-config: the "loss_mask_type omitted = pass-through" comment was wrong. The parser
  default is `qwen` and it is reached on the tokenizer-fallback path; what makes the omission
  safe is `use_rollout_logprobs: true` (fallback samples zero-filled + ABORTED + removed), now
  documented as load-bearing. `enable_trajectory_tracing` removed (no such miles flag; `true`
  would kill argparse). `kl_loss_coef 0.0` pinned and the ref-forward cost of `use_kl_loss`
  documented (kept for r5 parity). `check_weight_update_equal` + skip list `visual.` added for
  run 1 (first disaggregated glm5_next update path). `sglang_tp_size` marked derived;
  bf16 KV / tilelang constraints documented; PP4 split typo fixed (11/11/11/12).
- README: build the image BEFORE convert (convert-job pins it with `imagePullPolicy: Always`);
  deviations and risks tables updated.

### Convert saga — `rl-glm53f-convert` (02:30-03:45 PT)

Goal: HF BF16 -> torch_dist ref checkpoint on one p6-b200 (8 GPUs); `--save` to /dev/shm then
cp, because mountpoint-S3 cannot perform DCP's final `.metadata` rename.

| # | attempt | outcome | lesson / change |
|---|---|---|---|
| 1-2 | PyTorchJob, 1800Gi request, twice | kueue TAS pinned the pod to `i-00cb6ebceda36a4ef`, which the scheduler could not place on; the pods-ready timeout evicted it and `restartPolicy: Never` turned that into a Failed job that the operator deleted with its pods and logs | TAS assigns nodes that are occupied outside kueue's view; a deleted job leaves nothing to debug from |
| 3 | plain kueue-labelled Pod `rl-glm53f-convert-dbg` (02:58 PT; archived arena-port-artifacts/glm53/glm53-deploy/convert-debug-pod.yaml): `nodeAffinity NotIn i-00cb6ebceda36a4ef`, 1200Gi, `tee` to `/mnt/scratch-s3files-rw/guparpit/logs/glm53-convert-dbg.log`, `sleep 7200` after exit | the convert failed: `--hf-checkpoint /mnt/models-ro/external/zai-org/GLM-5.3-Flash-BF16` does not exist there | `/mnt/models-ro` is a SEPARATE curated read-only EFS volume, not a view of fast scratch (a recon pass equated the two because both carry `external/<org>/` trees). 1499bcef4 (03:05 PT) repoints `hf_checkpoint`, `--hf-checkpoint`, `ref_load` and the gym `ARENA_MODEL_PATH` at `/mnt/scratch-fast-1a-rw/...`; convert-job gets 1200Gi + the NotIn |
| 4 | rerun with the fast-scratch path | conversion OK in ~10 min (PP8 auto-reshape, 54-96 GB GPU per rank); the `cp` to `/mnt/scratch-fast-1a-rw/durable/model-artifacts/dcp/` failed EFBIG | mountpoint-S3 caps a single object at ~78 GiB (8 MiB part x 10k parts); the PP8 layout emits 83 GB `.distcp` shards |
| 5 | destination -> EFS `/mnt/scratch-s3files-rw/guparpit/checkpoints/dcp/glm5.3-flash_torch_dist` (34d17a57e, 03:35 PT), parallel per-file cp + manifest diff | 584 GiB: 8 x PP8 `.distcp` + `.metadata` + `latest_checkpointed_iteration.txt`, per-file sizes verified | EFS has rename and no object cap and is the proven ref_load home (smoke DCP, bowenxie qwen DCP) |

Decisions: keep the convert as a 1-replica PyTorchJob (not a batch/v1 Job) because it reuses
the admission wiring proven on these nodes; `CONVERT_KEEP_PP1` stays unset; the ref DCP lives
on EFS; 1200Gi request (~599 GiB checkpoint on tmpfs + 200-300 GiB for the ranks, ~1.5x
headroom, and it schedules where 1800Gi sat Pending). Rejected or deferred: hosting the DCP on
fast scratch (needs a >=16-rank, 2-node conversion so every shard lands under the cap; not
worth it for a one-time artifact); `--save` straight to fast scratch (no rename) or to node
ephemeral disk (allocatable uncertain at ~600 GB) instead of /dev/shm.

### State at "ALL DELIVERED" (8660c520a 03:50 PT; declared ~04:00 PT)

1. HF BF16 weights on fast scratch (130 files, 642.7 GB, manifest).
2. torch_dist ref checkpoint on EFS (584 GiB, verified).
3. Image a in us-east-1 and ap-south-1.
4. Config set as staged here: `miles-config.yaml` (`rl-glm53f-gbash-r1`; 12 nodes = 8 actor
   TP8+SP / PP4 [11/11/11/12] / CP1 / EP16 / ETP1, DP2 + 4 engines TP8/EP8 at 8 GPUs each,
   mem-fraction 0.7, tilelang DSA, bf16 KV, ctx 131072; rbs 32 x n 8 = GBS 256, num_rollout
   90, response len 2048, temp 1; GRPO eps 0.4/0.4/c 2.0, kl coefs 0.0, `use_rollout_logprobs`;
   recompute full/uniform/1, dynamic batch at max_tokens_per_gpu 8192; adam lr 1e-6 constant,
   wd 0.1, betas 0.9/0.98; router health 1/15 s/40; `check_weight_update_equal` on),
   `trainer-pytorchjob.yaml` (12 replicas min=max, `Never`/0, image a, kueue queue
   `gpu.p6-b200-48xlarge`, the full run5 EFA env + `vpc.amazonaws.com/efa: 8` +
   `/dev/infiniband`, `NCCL_DEBUG=INFO` / `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH` for bring-up,
   upstream recipe env, W&B via the `wandb-env-arena` Secret, 1800Gi request, trainer log tee to
   `/mnt/scratch-s3files-rw/guparpit/logs/${EXPERIMENT_NAME}/trainer-${REPLICA_IDX}.log`, the
   stale run5 exclusion `i-09907cd660684a460` dropped, no NotIn list yet), `sglang-svc.yaml`
   (`rl-glm53f-sglang` -> replica-index 0), `convert-job.yaml`, `fetch-job.yaml` +
   `fetch_glm53.py`, README (runbook fetch -> image -> convert -> deploy -> first-run checks;
   deviations vs the snorkel-27b parent and vs `run_glm5_3_flash.py`; risks).

Not yet present: the gym-side `nats.yaml` / `gym-worker.yaml` (README step 4 still says
"rename the smoke's `rl-milesgb1-` assets"); they land with the launch entry.

### Risks accepted going into run 1 (README "Risks" at this commit)

1. GLM chat / tool-call format vs the snorkel textual bash agent parser: unverified (no sglang
   tool-call or reasoning parser configured).
2. EFA pairing untested: aws-ofi-nccl 1.18.x (installer 1.47.0) with the base's NCCL
   2.29/cu13. Gate: the NCCL log must show the libfabric/efa provider; `Using network IB` or
   `Socket` means the broken fallback.
3. Engine memory: ~80 GB bf16 weights per GPU under mem-fraction 0.7 leaves ~46 GB for KV +
   activations; `sglang_max_running_requests: 256` identified as the first KV lever,
   deliberately not set for run 1.
4. Mask fallback semantics (above): verify first-step masked/unmasked token counts.
5. Capacity: a 12-node all-or-nothing gang (+1 node while the convert runs). Shape notes:
   DP=3 is impossible with EP16 at 96 GPUs (EP24 needed); EP8 is strictly worse at 64 GPUs;
   EP16/PP4/DP2 is the right 64-GPU shape.
6. First disaggregated glm5_next update-weights path (the recipe validated colocated only);
   `check_weight_update_equal` on; failure signature `The full weights of the ModelRunner are
   partially updated` in engine logs.
7. `max_tokens_per_gpu 8192` vs full-trajectory samples up to the gym's 32k context: an
   over-budget sample forms its own oversized micro-batch.

Next: launch (r1 entry).
