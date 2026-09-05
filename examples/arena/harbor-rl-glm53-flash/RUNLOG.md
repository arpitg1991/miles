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

## 2026-09-02 ~09:15 PT — r1 launch: NATS + 128 gym workers + trainer TAS exclusion (`rl-glm53f-gbash-r1`, image a)

First GLM-5.3-Flash training launch, on the user's go ("priority high like
r5"), ~5 h after the pre-launch "ALL DELIVERED" state. Manifests are the
integration commit 2f71601fd (2026-09-02 08:57 PT).

**Run identity**

| item | value |
|---|---|
| `EXPERIMENT_NAME` / `PROJECT_NAME` (checkpoint dir + W&B group) | `rl-glm53f-gbash-r1` |
| PyTorchJob / kueue queue / priority | `rl-glm53f-trainer` / `gpu.p6-b200-48xlarge` / `priorityClassName: high` |
| shape | 12x p6-b200.48xlarge: workers 0-7 Megatron actor (64 GPUs, TP8+SP / PP4 [11/11/11/12] / EP16, DP2), workers 8-11 SGLang engines (4 engines, TP8/EP8) |
| trainer image | `arena-slime-dev:miles-glm53-20260902a` (glm53next base + integration tree + EFA layer) |
| gym | Deployment `rl-glm53f-gym-sgb`, 128 replicas, image `arena-tasks-dev:rl-smoke-20260821b` (Harbor 0.21.0; no code change since the smoke, so no rebuild) |
| broker / engine service | `rl-glm53f-nats` (nats 2.11.3, JetStream on a 10Gi emptyDir, `max_payload` 8 MiB) / `rl-glm53f-sglang:30000` |
| batch + caps (r5-parity, unchanged from the snorkel parent) | rbs 32 x n 8 = GBS 256, `num_rollout` 90; `rollout_max_response_len` 2048, `ARENA_MAX_TOKENS` 2048, `ARENA_ROLLOUT_CONTEXT_LIMIT` 32768, `ARENA_NATS_ACK_WAIT` 4500, concurrency 8, temperature 1.0 |
| W&B | mega.wandb.agi.amazon.dev `arena/rl-snorkel27`, group `rl-glm53f-gbash-r1`, run `v57ymk1l` |

**What the three files are**

- `nats.yaml` + `gym-worker.yaml`: the smoke-proven `rl-milesgb1-` assets
  (themselves the AGISlime run5 gym-side manifests, see
  `harbor-rl-27b-snorkel/smoke-3node/`) with every name prefix replaced by
  `rl-glm53f-`, replicas 8 -> 128 (r5 fleet size for rbs 32 with dynamic
  sampling) and `ARENA_MODEL_PATH` -> the GLM BF16 fast-scratch path (workers
  tokenize against the policy model). A separate broker per run is
  mandatory: JetStream stream names (`ARENA_TASKS`/`ARENA_RESULTS`) are
  per-server. The file headers still carry the parents' comments verbatim
  ("27B RL run 2", "Deltas vs run 1: 96 replicas") — lineage, not GLM state.
- `trainer-pytorchjob.yaml`: first `kubernetes.io/hostname NotIn` entry,
  `i-00cb6ebceda36a4ef` — the node the convert job was pinned to twice (see
  the staging entry). Rule written into the yaml: add exclusions only against
  a live, reproduced incident, and drop them as nodes heal (each one costs
  capacity at 12 nodes). Still at the platform-customary `memory: 1800Gi`
  request with no limit; `NCCL_DEBUG=INFO` / `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH`
  left ON for bring-up. `restartPolicy: Never` / `maxRestarts: 0` kept
  (OnFailure restarts are unproven on the ray-based launcher).

**Scheduling: what happened**

- Attempt 1 evicted. Priority high preempted backfill, but kueue TAS pinned
  pods to nodes the scheduler rejected (`Insufficient memory/gpu/efa`); the
  gang partially bound, the ~10-min pods-ready timeout evicted it, and the
  kubeflow operator deleted the job together with its pods (and their logs).
- Attempt 2 bound 12/12 in ~15 min after four nodes were excluded (the three
  added on top of `i-00cb6ebceda36a4ef` are codified in the next entry).
- Karpenter refuses to PROVISION for hostname-affinity pods; binding to
  existing nodes still works, so the gang only ever lands on live capacity.
- The 41 unreaped Failed `arena-backfill` pods fleet-wide (deletion-protected,
  owner-only) are NOT the blocker: running workers coexist with them; the
  blocked nodes are consumed outside arena-tasks' view.

**Bring-up verification (all green)**

- EFA LIVE: aws-ofi-nccl 1.18.0 + Libfabric 2.4 loaded from `/opt/amazon` on
  all ranks alongside the base image's NCCL 2.29/cu13 — the pre-launch
  "ofi-nccl 1.18 vs NCCL 2.29 untested" risk is retired.
- 4 engines TP8/EP8 loaded the 120 BF16 shards from fast scratch at
  ~3 s/shard; rollout 0 collecting, results flowing over NATS.

**Early signal (~10:40 PT)**

- The first ~28 completions were ALL `status=truncated`, 0 success: GLM-5.3
  at its template-default `max` reasoning effort spends the whole 2048-token
  turn thinking. Concern noted at the time: if rewards degenerate the
  zero-variance filter will churn groups. Decision and its effect are in the
  next entry (32k/131k caps, user-directed 10:41 PT).

**Alternatives considered**

- Reuse the smoke's NATS/gym fleet: rejected — stream names collide, and the
  smoke fleet was sized for 8 workers.
- Rebuild the gym image for GLM: not needed — no gym code change; only
  `ARENA_MODEL_PATH` differs.
- OnFailure + restarts for a 12-node job: deferred — recovery would need
  replica-0 AND all 11 ray workers to exit on GCS loss; unproven.

## 2026-09-02, from 10:18 PT — run 1 (`rl-glm53f-gbash-r1`, image a): live changes and findings

Continues the launch entry above; identity unchanged throughout: PyTorchJob
`rl-glm53f-trainer` (12x p6-b200 = 8 actor TP8/PP4/EP16/DP2 + 4 engines
TP8/EP8), trainer image `arena-slime-dev:miles-glm53-20260902a`, gym
`rl-glm53f-gym-sgb` x128 on `arena-tasks-dev:rl-smoke-20260821b` (Harbor
0.21.0), NATS `rl-glm53f-nats`, W&B `arena/rl-snorkel27` group
`rl-glm53f-gbash-r1`. Tree state: trainer `fbc8ddaa8`, gym/config/README
`abd7edf8f`.

### Live change 1 — 10:18 PT: TAS exclusions 1 -> 4 nodes (`fbc8ddaa8`)

- Second kueue TAS incident of the day, same signature as the convert job:
  9 actor pods placed in ~3 min while three TAS-pinned pods sat
  `FailedScheduling` (Insufficient gpu/efa/memory) and stalled the gang into
  the ~10-min pods-ready eviction; the kubeflow operator recorded a FAILED
  job and deleted the pods.
- Fix: `kubernetes.io/hostname NotIn` extended with `i-09913cb3a1edcc270`,
  `i-08c9e00e2a0af830a`, `i-04c8462a637bbf738` (alongside
  `i-00cb6ebceda36a4ef`); job deleted and re-applied under the same name
  (rdzvId, JOBNAME, the sglang Service selector and the NATS URL are bound
  to it).
- Theory vs RCA: the yaml comment blames the unreaped Failed `arena-backfill`
  pod each of the three nodes hosts (41 fleet-wide, deletion-protected to
  their submitter). The launch-day RCA refuted it: running workers coexist
  with those pods; the nodes are consumed outside arena-tasks' view, so TAS
  believes them free while kube-scheduler cannot place. The comment stays as
  written (tree state) — do not wait for a "backfill reaper".
- Rejected: re-applying without exclusions (each miss costs a ~10-min
  eviction plus the job's pod logs); deleting the backfill pods (not ours,
  and not the cause).
- Consequences: Karpenter refuses to provision for hostname-affinity pods, so
  the gang binds only to existing nodes; every exclusion costs capacity —
  prune as nodes heal. The list reaches 16 entries by r3 (later entry).

### Live change 2 — 10:41 PT: caps 2048 -> 32768 per turn, 32768 -> 131072 per episode (`abd7edf8f`, user-directed)

- Observation: the first ~28 completions were 28/28 `status=truncated`, 0
  success — GLM-5.3 at its template-default `max` reasoning effort spends the
  whole 2048-token turn thinking. All-zero rewards also mean the zero-variance
  filter churns through groups without ever filling a batch.
- Change, trainer and gym together (breaks r5 parity deliberately):

  | knob | before (r5 parity) | after |
  |---|---|---|
  | gym `ARENA_MAX_TOKENS` | 2048 | 32768 |
  | gym `ARENA_ROLLOUT_CONTEXT_LIMIT` | 32768 | 131072 |
  | trainer `rollout_max_response_len` | 2048 | 32768 |
  | trainer `rollout_max_context_len` / `sglang_context_length` | 131072 | unchanged |

  Rule written into both files: `ARENA_MAX_TOKENS` and
  `rollout_max_response_len` stay in lockstep. Both sides were redeployed
  under the same identity (the trainer reads `miles-config.yaml` only at
  start) and rollout 0 restarted under the new caps.
- Rejected: keeping the r5 caps (no training signal at all); cutting thinking
  via `reasoning_effort` instead — not available, the gym's `ArenaSGLangLLM`
  dropped that kwarg until the r5 gym image (2026-09-03).
- Effect: rollout 0 wire statuses 272 success / 54 failed / 0 truncated,
  `response_len` median ~20.4k tokens, episode reward 0.18.
- Accepted consequences (README "Run-1 live change"): a ~32k-token sample
  exceeds `max_tokens_per_gpu 8192` by up to ~16x, so each forms its own
  micro-batch (dynamic batching effectively inert; the memory analysis had
  ~50 GB headroom); slower rollouts per episode; a ~131k-token trajectory
  pushes the NATS result message toward the 8 MiB `max_payload`.

### Findings from run 1 (all before the 13:54 PT fla patch)

1. **Train step 0 crashed on every rank** in fla 0.4.2
   `chunk_kda_fwd_kernel_intra_token_parallel`: `BK: tl.constexpr =
   triton.next_power_of_2(K)` is declared inside the `@triton.jit` body and
   triton 3.7.1 rejects it ("Unsupported function referenced"). AST scan of
   fla: the only in-kernel offender. Decision: hoist BK to a host-computed
   constexpr launch arg, shipped as an image layer -> image b (next entry).
2. **91% of samples removed as TRUNCATED — only 23/256 carried loss.** Read
   at the time as `nats_rollout`'s any-step `stop_reason=length` rule removing
   whole samples when long GLM episodes clip mid-trajectory (wire status
   `success` notwithstanding). Recipe decision left PENDING with the options
   loosen the any-step policy / cut thinking via reasoning_effort / larger
   context / accept; the first became `--arena-mask-clipped-final-turn`
   (16:44 PT). Re-diagnosed 2026-09-03: Harbor agent wall-clock timeouts
   (`agent_stop_reason=timeout`, in the degenerate-stop set), 0/1033 generates
   at the 32k cap; the Harbor 0.21 envelope reported timed-out groups as
   `success`, which is why the wire looked clean while the trainer removed
   them.
3. **Post-mortem source.** The kubeflow operator deletes a failed PyTorchJob
   with its pods; the tee to
   `/mnt/scratch-s3files-rw/guparpit/logs/<EXPERIMENT_NAME>/trainer-<idx>.log`
   is the only surviving driver log.
4. **kueue counts REQUESTS against node allocatable.** At `requests.memory:
   1800Gi` only 11/304 p6 nodes were TAS-assignable (287 excluded on memory)
   and the 12-pod gang could not admit for the image-b relaunch; at 1200Gi it
   admitted immediately (actual per-node use is far lower; the convert job
   ran at 1200Gi). Applied by hand to the deployed manifest only — this tree
   still says 1800Gi; codified in the r2 fix set (`3098ae1d3`).
5. Ops: the ECR docker login expires after 12 h — "authorization token has
   expired" means re-login, not missing permissions.

## 2026-09-02 13:55 PT — fla KDA kernel patch for triton 3.7.1, image b, run 2 relaunch (`rl-glm53f-gbash-r1`, `miles-glm53-20260902b`)

Answers finding (2) of the previous entry: run 1's train step 0 crashed on
every rank inside fla. The fix is an image layer (integration 3c40b4660), the
manifests move to image b (949434185), and run 2 — same identity, same
job name — validates it 4 h later. Run 2's death that evening is a different
defect and has its own entry (Ray host-memory OOM).

**Timeline (PT)**

| when | what |
|---|---|
| 11:03 -> 12:23 | run 1 rollout 0 on image a: 32 groups / 256 samples in 4800.6 s, avg_reward 0.180, `truncated_ratio` 0.910 |
| 12:24 | first forward of train step 0 (the ref log-prob pass, 22.7 s in) dies on all 64 actor ranks — traceback below |
| 13:53-13:54 | `examples/arena/patches/fla_kda_next_power_of_2.py` + Dockerfile `RUN` (3c40b4660) |
| 13:55 | image `arena-slime-dev:miles-glm53-20260902b` built and pushed (us-east-1, replicated to ap-south-1) |
| 13:58 | `trainer-pytorchjob.yaml` and `convert-job.yaml` pinned to b (949434185) |
| 14:15 | run 2 trainer up (`rl-glm53f-trainer`, still `EXPERIMENT_NAME` / W&B group `rl-glm53f-gbash-r1`; nothing had been saved by run 1, so it restarted from the ref weights and re-collected rollout 0) |
| 14:34 | engines loaded; initial `update_weights` 28.3 s on 64 ranks (weight-equality check on) |
| 15:57 | rollout 0 done: 4893.6 s, episode reward 0.152, `truncated_ratio` 0.898, response_len median 19361 |
| 15:57 -> 16:36 | `ref_log_probs` 2342.6 s wall-clock (`end (elapsed: ...)` line; the `perf/ref_log_probs_time` metric records 2388 s) — the same KDA forward that killed run 1 now runs through: **patch validated** |
| 16:36 -> 18:05 | `actor_train` 5378 s wall-clock (perf metric 5332 s; `perf/train_time` 7722 s — the fix-set entry quotes the perf metrics); step 0 logged: loss -0.0401, grad_norm 0.0816, ppo_kl 0.0013, ess_ratio 0.099, pg_clipfrac 0.0017 |

**Root cause (from the run-1 driver log, EFS tee — the pods were gone)**

```
fla/ops/kda/chunk_intra_token_parallel.py, line 153, in chunk_kda_fwd_intra_token_parallel
  ...
triton/runtime/jit.py, line 193, in visit_Attribute -> record_reference
RuntimeError: Unsupported function referenced: <function next_power_of_2 at 0x...>
```

fla 0.4.2's `chunk_kda_fwd_kernel_intra_token_parallel` declares
`BK: tl.constexpr = triton.next_power_of_2(K)` INSIDE the `@triton.jit` body.
Triton 3.7.1's JIT dependency walker refuses host functions referenced from a
kernel body, so every GLM-5.3-Flash (KDA) train step fails at its first
forward. The engines were not affected: run 1 served rollout 0 normally (the
fla KDA kernels they JIT-compiled, e.g. `chunk_kda_fwd_kernel_inter_solve_fused`,
are not this one); the offender is reached only on the training forward. An AST
scan over all of fla found no other in-kernel offender. The sibling
`intra_sub_chunk` path already computes `BK` on the host, which is what the
patch copies.

**Fix: `examples/arena/patches/fla_kda_next_power_of_2.py` (+49 lines)**

- Rewrites the installed file at
  `/usr/local/lib/python3.12/dist-packages/fla/ops/kda/chunk_intra_token_parallel.py`:
  deletes the in-body line, adds `BK: tl.constexpr` to the kernel signature
  (after `BH`), passes `BK=triton.next_power_of_2(K)` at the launch site, then
  `ast.parse`s the result. Semantics identical; K=128 is already a power of two.
- Idempotent (`already applied` -> exit 0); exits 0 with a message when fla is
  absent, so the same Dockerfile still builds the non-GLM (27B) images; exits 1
  ("anchor not found — fla changed, refusing to guess") if any of the three
  anchors drifted, so a base-image bump cannot ship an unpatched kernel silently.
- Dockerfile: one `RUN python3 /root/miles/examples/arena/patches/...` after
  `COPY . /root/miles` and before the import smoke. Image layer, no pod-start
  step, no network. The patched source was checked inside the run-2 pods.

**Alternatives considered**

- Different triton/fla pins: both come with the `radixark/miles:glm53next`
  base (the stack that exists because mainline lacks KDA/DSA support) — a base
  change, not a config change.
- Patch at pod start (entrypoint): rejected — a runtime dependency and a drift
  source; the port had just removed AGISlime's pod-start pip installs for the
  same reason.
- A blind in-place edit (`sed`) in the Dockerfile: would not notice fla drift;
  the script anchors on exact text and refuses to guess.
- `FLA_CACHE_RESULTS=0` — found on 2026-09-03 in the precedent hunt
  (ForgeModelEnablement `glm53-flash-sft-strl` sidesteps the same crash with
  it). Not tried here: the image-layer patch was already validated and every
  later tag (c, 20260903d) is a superset of b. Recorded as the fallback if the
  anchors ever drift.

**Manifest pins**

- `trainer-pytorchjob.yaml`: image a -> b, nothing else (the 4-node `NotIn`
  list and the 1800Gi request are as the previous entry left them; the
  1200Gi drop was made in the deployed copy and is codified with the r2 fix set).
- `convert-job.yaml`: a -> b is post hoc — the conversion finished ~04:00 PT on
  image a and never ran on b. The pin keeps the example on one tag (README
  lineage table); a re-apply hits the job's `REFUSING to overwrite` guard.

**Also seen in run 2 (handled elsewhere)**

- 89.8% of samples removed as TRUNCATED again (run 1: 91.0%) — the
  `--arena-mask-clipped-final-turn` flag written at 16:44 PT while this run
  was alive is the next entry; the timeout RCA that re-explains the number is
  two entries on.
- Rollout 1 completed asynchronously at 16:53 PT (3392.6 s, reward 0.258,
  `truncated_ratio` 0.883) while step 0 trained.
- 17:33 PT: Ray's memory monitor killed the SGLang schedulers on engine node
  `trainer-worker-5`; the trainer only noticed at the post-train
  `update_weights` (18:05 PT). Step-0 perf, RCA and fix set: the r2 fix-set
  entry.

## 2026-09-02 22:23 PT — Image c composition: mask-clipped flag + hf_export ENOTSUPP fix (`miles-glm53-20260902c`); the r2 fix set lands alongside

Not a run. Records what the third trainer image carries (README lineage
table: c = b + 5f8925db0 + 44ddf62fb), in the order the pieces landed in the
integration tree, and why a core Megatron fix from the 27B smoke rides in a
GLM image. Run 2's death and the fix set itself are in the next entry.

| landed (PT) | integration commit | content |
|---|---|---|
| 2026-09-02 13:55 | (image b, 3c40b4660 / 949434185) | base for c: glm53next stack + tree + EFA layer + fla KDA patch |
| 2026-09-02 16:44 | 5f8925db0 | `--arena-mask-clipped-final-turn` (default off; `miles-config.yaml` turns it on from r2) |
| 2026-09-02 22:23 | 44ddf62fb | `hf_export.py`: `copyfile` + per-file `OSError` warning instead of `copy2` (this commit) |
| 2026-09-02 22:37 | 3098ae1d3 | r2 relaunch fix set (Ray monitor off, WeightChecker snapshot dropped, FT on, memprobe/, ...) — config, manifests and the launcher's `_pin_raylet_env()`; NOT listed as image-c content by the README lineage table. Whether the c build included the launcher's +19 lines is unrecorded; the pod env carries `RAY_memory_monitor_refresh_ms=0` regardless |

- Build window: the trainer-manifest comment written at 22:20 PT says `20260902c`
  "did not exist yet; newest is 20260902b"; ECR `describe-images` records the push
  (us-east-1, replica ap-south-1) at 22:24 PT, one minute after the hf_export
  commit, digest `sha256:9c963ae7...`; r2 launched on it at 22:49 PT. Pinned by
  `trainer-pytorchjob.yaml` for r2 and r3; `20260903d`
  (built 2026-09-03 01:40 PT) is a superset, so r4-r7 carry everything here.
- Why the hf_export fix is in this image: the 27B smoke (`rl-milesgb1-smoke1`,
  terminal 00:34 PT) left `hf/rollout_3` with 178 weight shards and no
  `config.json` / tokenizer / `model.safetensors.index.json` because `copy2`'s
  metadata step raised `[Errno 524] ENOTSUPP` (RCA in
  `harbor-rl-27b-snorkel/RUNLOG.md`). This job is the sharper case the fix
  comment names: `hf_checkpoint` is the BF16 tree on fast scratch
  (mountpoint-S3, a FUSE mount), the launcher appends
  `--save-hf <ckpt>/hf/rollout_{rollout_id}`, and `save_hf_model` runs after
  every DCP save at `save_interval 20`. An image rebuild was due anyway for
  the fix set, so the one-hunk core change went in at zero extra cost.
- Strict-argparse trap recorded in the manifest: `arena_mask_clipped_final_turn`
  in `miles-config.yaml` needs image >= c; on image b the trainer dies at
  startup. The same rule applies to `arena_keep_timeout_trajectories` (needs d).
- Effect of the hf_export fix on GLM: **unverified**. run 1, run 2, r2 and r3
  never reached a save (died in step 0/1); r4 ended after 13 steps (20:29 PT),
  below `save_interval`. r5 (reached step 40) is the first run that can hold a
  post-fix export (r6, at step 33 by 2026-09-05 05:15 PT, is the second); check its
  `hf/rollout_*/` for `config.json`, the index and `.complete`, and its EFS
  `trainer-0.log` for `HF export: could not copy` warnings, before claiming it.

## 2026-09-02 17:33 PT — Run 2 (`rl-glm53f-gbash-r1`, image b) killed by Ray's memory monitor on engine node worker-5

Run 2 is the image-b relaunch of the previous entry (same identity r1, ray head
up 14:16 PT). It validated the fla patch at train step 0 and then died of a
host-memory verdict that was Ray's, not the kernel's. Only post-mortem source:
the replica-0 tee on EFS (`logs/rl-glm53f-gbash-r1/trainer-0.log`, 48,297
lines; S3 mirror
`s3://arena-scratch-prod-bom-ap-south-1/guparpit/logs/rl-glm53f-gbash-r1/trainer-0.log`;
raw copy archived as `arena-port-artifacts/glm53/glm53-run2/glm53-run2-t0.log`)
— the kubeflow operator deleted the failed job with its pods. Ray placed the 4
engines on workers 1/5/7/10, not on "workers 8-11" as the topology comments say.

**Timeline (PT, 2026-09-02)**

| time | event |
|---|---|
| 14:17-14:24 | 4 engines (TP8/EP8) load the 120 shards in 363 s; per GPU: weights 73.7 GB, KV 25.4 GB (2,144,384 tokens), mamba state 22.8 GB, 48.9 GB free after CUDA graphs |
| 14:34:48 | initial `update_weights`: 28.3 s on 64 ranks; `WeightChecker` compare -> equal tensors (the one-time check `check_weight_update_equal` was kept for) |
| 14:35 -> 15:56 | rollout 0: 4894 s, raw reward 0.152, removed 230/256 (`truncated_ratio` 0.898), failed 0/256 |
| 15:57 -> 16:53 | rollout 1 (async, alongside train step 0): 3393 s, raw reward 0.258, removed 227/256 (0.887) |
| 15:57 -> 18:05:43 | train step 0: `ref_log_probs` 2388 s + `actor_train` 5332 s = 7722 s; loss -0.040, grad_norm 0.082, ess_ratio 0.099, ppo_kl 0.0013 |
| 17:33:40 | engine 0 (`rl-glm53f-trainer-worker-5`, 10.100.176.96) still healthy GPU-side: 89 running / 10 queued requests, token usage 0.98 — not an outlier vs the other three |
| 17:33:57 | raylet on that node: "9 Workers killed due to memory pressure" = the `SGLangEngine` actor + its 8 `_HttpPosterActor`s; node 1934.01 / 1996.03 GB (0.969) vs the 95% threshold of MemTotal 2,143,217,684,480 B. Top-10: `sglang::scheduler_TP0..7` at 78.83-79.00 GB each (~631 GB); ~1300 GB of the counted 1934 GB is outside the pod's processes |
| 17:34 -> 17:43 | router health check fails 40x at 15 s; the worker is never removed (router `log_level=warn`); the other 3 engines keep serving (79 more NATS results, 70 success / 9 failed, vs 399 before) |
| 18:05:44 | post-train `update_weights` -> `_pause_and_prepare_engines` -> `ray.get(pause_generation)` re-raises `OutOfMemoryError` for the dead actor; driver exits 18:05:56 |

Run 1 (same config, 78 min of serving) had zero memory-pressure events: the
snapshot plus ~3 h of serving was what reached 95%. Gym Deployment scaled to 0
at 21:15 PT while the RCA ran.

**RCA (24-agent pass, ~21:20-22:30 PT; notes `oom-context.md`,
`log-forensics-notes.md`; Ray 2.58 monitor sources read for the accounting path)**

1. **Ray measured the whole node.** containerd gives the privileged container no
   cgroup namespace, so `/sys/fs/cgroup/memory.max` does not exist in-pod (the
   pod's cgroup sits under `/sys/fs/cgroup/kubepods.slice/...`); Ray 2.58's
   `ThresholdMemoryMonitor` falls back to `/proc/meminfo` MemTotal-MemAvailable
   and its 2.56+ by_time policy kills at 95% node-wide. A pod memory *limit*
   therefore cannot switch Ray to cgroup accounting here (confirmed by memprobe
   variant -b, next entry).
2. **The 79 GiB/rank is self-inflicted.** `check_weight_update_equal: true`
   makes sglang's `WeightChecker` keep a permanent anonymous CPU copy of every
   TP rank's shard (`param.data.detach().cpu()`, issued once at startup by
   `miles/ray/placement_group.py`, never freed): 8 x ~79 GiB = ~632 GiB per
   engine node, for a check that had already passed at 14:35.
3. **~1300 GiB unattributed.** Co-tenant theory refuted: the dead node
   (`i-099b9859c5430fec8`) hosted only 3 small swebad pods while survivors hosted
   more. Engine USS minus the snapshot grew only ~5.6 GiB in 3 h -> no material
   engine leak. The remainder is external to the pod (mount-s3 CSI page cache,
   driver pinned pages, kernel — unknown); left open, instrumented below.
4. **No survival path.** `use_fault_tolerance` was unset (log: False), so both
   `rollout_health_check_*` knobs were inert and no `RolloutHealthMonitor`
   existed; `SGLangEngine` actors have no `max_restarts`. The run served 32 min
   on 3 engines and died the moment the trainer touched the dead one.
5. **Perf, recorded not fatal:** MFU 0.26% (5.9 TFLOPs/rank, 1521 tok/s); the
   ref pass is 31% of step 0 at kl coef 0; every ~32k-token sample exceeds
   `max_tokens_per_gpu 8192`, so dynamic batching ran 127 micro-batches of one
   sample; TileLang re-JITs the sparse-MLA backward per distinct padded length
   (~3300 compiles across 64 ranks, ~12 s each); dynamo hits its recompile limit
   on the mHC hyper-connections; KDA is replicated across TP (upstream design);
   CP unsupported. Steady state: rollout 3393 s vs train 7722 s.

## 2026-09-02 22:27-22:49 PT — memprobe Jobs, r2 fix set (integration 3098ae1d3, image c) and r2 launch (`rl-glm53f-gbash-r2`)

**memprobe/ (files 22:22-22:27 PT; how to read the CSVs: `memprobe/README.md`).**
A single-node SGLang host-memory experiment to decide hypotheses R1-R5 (Ray
accounting scope; 79 GiB = snapshot; co-tenants; leak under load; own
shmem/kernel) in ~75 min on ONE p6 node instead of a 12-node multi-hour run.
`memprobe.sh` runs the phases boot/loading/idle/snapshot/load/post with the
run's exact ServerArgs; `memsampler.py` writes 30 s CSVs (`mem.csv` node + own
cgroup + a literal read of `/sys/fs/cgroup/memory.max`, `procs.csv` anon/file
split per process, `pods.csv` per-pod `memory.current` from `kubepods.slice`);
`loadgen.py` keeps 96 `/generate` requests in flight with 40k-100k-token
prompts and 10% client aborts; Job `-a` mirrors the run (no memory limit), `-b`
adds `limits.memory: 1800Gi`. Stdlib only; loadgen self-tested against a fake
server before submit (39/39 requests, 0 errors).

- 22:27 PT: `glm53-memprobe-a` / `-b` created (one ConfigMap
  `glm53-memprobe-scripts`); deleted ~22:33 after kueue TAS mis-pins;
  resubmitted 22:34 PT as `a2` / `b2` with two more `NotIn` nodes
  (`i-0e7902b4e3d9f2d0c`, `i-04f9a44ac50ae500d`). 22:36: `b2` Running on
  `i-0c0969769a21a3001`, `a2` Pending. `b2` finished rc=0; artifacts under EFS
  `logs/glm53-memprobe/glm53-memprobe-b2/`. `a2`'s outcome is not recorded.
- Result used for the fix set: a kubelet memory limit is NOT visible at
  `/sys/fs/cgroup/memory.max` inside the privileged container, so
  `limits.memory` alone cannot fix R1; `RAY_memory_monitor_refresh_ms=0` is
  required.
- Ops facts: kyverno rewrites `ttlSecondsAfterFinished` to 0 (the EFS artifacts
  are the only record); kueue admits a `batch/v1 Job` carrying the queue label.
  As-applied Job a: `arena-port-artifacts/glm53/memprobe-cm/job-a-live.yaml`.

**Fix set (3098ae1d3, 22:37 PT; 10 files, +1490/-35)**

| change | file | why |
|---|---|---|
| `RAY_memory_monitor_refresh_ms=0` in the pod env; launcher `_pin_raylet_env()` setdefaults it before `ray start` on both roles | trainer yaml, `scripts/run_arena_harbor.py` | installs Ray's NoopMemoryMonitor on every raylet; the raylet reads `RAY_*` from the env `ray start` inherits, ray-job runtime_env never reaches it. Same setting as miles' own GLM-5 launchers |
| `check_weight_update_equal: false` (+ skip list removed) | miles-config | drops the ~632 GiB/node snapshot; equality already verified at 14:35. Re-enable only in a single-engine smoke |
| `use_kl_loss` / `kl_loss_type` removed; `kl_coef`/`kl_loss_coef` stay 0.0 | miles-config | the ref pass was 2388 s of 7722 s at coef 0; the upstream recipe runs no ref pass. Only the `train/kl_loss` diagnostic is lost |
| `use_fault_tolerance: true`; `rollout_health_check_timeout` 300 -> 900 | miles-config | the health knobs were inert; 900 s so the probe cannot false-kill a healthy engine under ~100-deep queues. UNVALIDATED on the arena NATS path |
| `limits.memory: 1800Gi`, `/dev/shm sizeLimit: 256Gi`; request 1800Gi -> 1200Gi codified | trainer yaml | backstops: a real exhaustion now OOM-kills inside our cgroup; 256Gi keeps the ~186 GiB Ray object store on shm (exceeding it evicts the pod). The deployed run-2 manifest already ran 1200Gi — at 1800Gi only 11/304 p6 nodes were TAS-assignable |
| 60 s `memsample-<idx>.log` host-memory sampler in the container command | trainer yaml | run 2's log had exactly one host-memory snapshot; node vs own-cgroup vs per-process over time is the only way to attribute the ~1300 GiB |
| gym `NotIn` anti-affinity off `p6-b200.48xlarge` / `node-type: p6-gpu` | gym-worker | p6 nodes are untainted; the trainer's 1200Gi / 0-CPU request left ~746 GiB + ~190 vCPU per engine node for our 128 gym+dind pods to land on. Repels only OUR pods |
| `data_pad_size_multiplier: 512` | miles-config | pads to multiples of TP*512 = 4096 tokens: distinct padded lengths ~37 -> ~10 for ~+6.5% pad tokens, bounding the TileLang bwd re-JIT count; verify loss parity |
| `arena_mask_clipped_final_turn: true` (needs image c) | miles-config | ADR-0009; ~90% of run-2 samples carried no loss |
| `NCCL_DEBUG` / `NCCL_DEBUG_SUBSYS` dropped | trainer yaml | EFA verified in runs 1-2; INFO at 96 ranks inflated the log to 48k lines |
| TRITON / TILELANG / INDUCTOR caches -> EFS `kernel_cache/<EXPERIMENT_NAME>/worker-<idx>` | trainer yaml | for persistence across pod restarts (per-replica dirs against cross-NODE writers). **Regression:** the 8 ranks of one node still share the dir and race on NFS -> Triton ESTALE kills r2 step 0; reverted in the next entry |
| `EXPERIMENT_NAME` / `PROJECT_NAME` -> `rl-glm53f-gbash-r2` | trainer yaml | run 2 saved no checkpoint (`save_interval 20`); a fresh name keeps ckpt dir and W&B group separate |

Alternatives rejected: raising `RAY_memory_usage_threshold` alone (documented
to still fail near the edge, slime #1851 at 0.99); relying on `limits.memory`
to switch Ray to cgroup accounting (refuted by R1); keeping
`check_weight_update_equal` as a per-update guard (its cost is the permanent
snapshot, not the compare); a 300 s health timeout (false-kill risk); a
platform taint on the GPU nodes (not ours to set); OnFailure restarts (still
unproven on the ray-based launcher).

**Image and launch.** `arena-slime-dev:miles-glm53-20260902c` = b +
`--arena-mask-clipped-final-turn` + the hf_export ENOTSUPP fix (previous two
entries); pushed to us-east-1 (replica ap-south-1) at 22:24 PT, one minute after
the hf_export commit (ECR `describe-images`; digest `sha256:9c963ae7...`) — the
22:20 PT manifest comment records that the tag did not exist yet. r2 launched
22:49 PT: PyTorchJob `rl-glm53f-trainer`, 12 nodes,
`EXPERIMENT_NAME` / W&B group `rl-glm53f-gbash-r2`, gym `rl-glm53f-gym-sgb`
x128 (image `rl-smoke-20260821b`, unchanged) now kept off the p6 nodes.
Outcome in the next entry.
