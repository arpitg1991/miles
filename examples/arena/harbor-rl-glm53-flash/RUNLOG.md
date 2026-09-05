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

## 2026-09-02 22:49 PT -> 2026-09-03 01:32 PT — r2 outcome (`rl-glm53f-gbash-r2`) and r3 launch (`rl-glm53f-gbash-r3`), both image c

Continues the r2 relaunch entry above. Tree state after this entry: trainer
`86d2d0ec5` (= `0ee733986` caches + `86d2d0ec5` exclusions);
`miles-config.yaml`, `gym-worker.yaml`, `nats.yaml` unchanged from `3098ae1d3`.
Trainer image `arena-slime-dev:miles-glm53-20260902c` for both runs.

### r2 outcome — 22:49 PT (Sep 2) to 00:36 PT (Sep 3): the OOM fix set held, then Triton ESTALE

- **What held** (every run-2 fix validated up to the train step):
  - engine `sglang::scheduler` host RSS ~3.4 GiB per rank (run 2: ~79 GiB
    with the WeightChecker snapshot); no `killed due to memory pressure`
    events with `RAY_memory_monitor_refresh_ms=0`;
  - fault-tolerance monitor healthy, no engine restart needed;
  - initial weight sync 38 s;
  - rollout 0 completed in 4938 s, wire statuses 135 success / 20 failed.
- **What failed:** train step 0 died 7 min in with `OSError [Errno 116] Stale
  file handle` in Triton's autotuner (`compiler.py` `read_text`).
- **Root cause:** the r2 fix set had moved `TRITON_CACHE_DIR` /
  `TILELANG_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` to EFS
  (`/mnt/scratch-s3files-rw/<user>/kernel_cache/<EXPERIMENT_NAME>/worker-<idx>`)
  so the ~3300 TileLang re-JITs per step would survive pod restarts. The
  per-replica dirs kept nodes apart, but the 8 ranks of one node share the
  dir and race on the same NFS cache files -> ESTALE on read.
- **Decision (`0ee733986`, 01:20 PT):** `export KCACHE=/tmp/kernel_cache` in
  the container command (the three `*_CACHE_DIR`s still derive from it); the
  env-block comment rewritten to say why. **RULE: kernel JIT caches live on
  local disk only — never EFS/NFS.** Accepted cost: a cold JIT per pod start
  (`data_pad_size_multiplier 512` bounds the TileLang backward recompiles).
- **Alternatives:** per-rank EFS subdirs would keep NFS in the JIT hot path
  for a persistence gain that was never measured, while r2 showed the cost is
  a dead run — not tried. The parent manifest's default in-pod locations
  (what `/tmp/kernel_cache` amounts to) are the proven configuration.
- **Also observed:** `arena_mask_clipped_final_turn: true` salvaged nothing —
  removed 229/256 samples (`truncated_ratio` 0.8945), zero "masked clipped
  final turn" log lines. The removals are therefore not trailing-final-turn
  clips as the flag assumes; this opened the trajectory-structure
  investigation on the gym pods (truncation RCA, next entry).
- Post-mortem source: EFS tee
  `/mnt/scratch-s3files-rw/guparpit/logs/rl-glm53f-gbash-r2/trainer-<idx>.log`
  (+ `memsample-<idx>.log`).

### r3 launch — 01:32 PT, image c, local caches, identity r3

| item | value |
|---|---|
| `EXPERIMENT_NAME` / `PROJECT_NAME` (ckpt dir + W&B group) | `rl-glm53f-gbash-r3` — r1 and r2 saved no checkpoint (`save_interval` 20, both died in step 0), so nothing resumes; the fresh name keeps its W&B group and ckpt dir separate (ADR-0004) |
| PyTorchJob | `rl-glm53f-trainer` (same name — `rdzvId`, `JOBNAME`, the `rl-glm53f-sglang` Service selector and the NATS URL are bound to it) |
| shape | 12x p6-b200: workers 0-7 actor (TP8/PP4/EP16, DP2) + workers 8-11 SGLang engines (TP8/EP8) — unchanged |
| trainer image / config | `miles-glm53-20260902c` / `miles-config.yaml` as r2 (`3098ae1d3`): FT on, WeightChecker off, `use_kl_loss` removed, mask-clipped on, pad multiplier 512 |
| gym / broker | `rl-glm53f-gym-sgb` x128 on `arena-tasks-dev:rl-smoke-20260821b` (Harbor 0.21.0, p6 anti-affinity) / `rl-glm53f-nats` — unchanged |
| W&B | `arena/rl-snorkel27`, group `rl-glm53f-gbash-r3` |

**Scheduling — third kueue TAS mis-pin, 12 nodes at once**

- The first r3 apply (after `0ee733986`) had TAS pin pods to 12 more p6 nodes
  the scheduler rejected (`Insufficient nvidia.com/gpu` / `vpc.amazonaws.com/efa`
  / memory); same signature as the convert job and the run-1 launch.
- Fix (`86d2d0ec5`, 01:28 PT): the 12 hostnames appended to the
  `kubernetes.io/hostname NotIn` list (4 -> 16 entries, dated comment in the
  yaml), job deleted and re-applied under the same name; r3 running from
  ~01:32 PT.
- Consequence: 16 exclusions on a ~304-node p6 pool, each costing capacity
  (Karpenter will not provision for hostname-affinity pods) — prune as nodes
  heal. The README runbook gains a "TAS mis-pin check within ~2 min of every
  apply" procedure plus the two negative health checks for r3 (no `killed due
  to memory pressure`, no `Stale file handle`; caches must resolve under
  `/tmp/kernel_cache`) in `37164f537` (next entry).

**Ops trap during r3 rollout 0 — NATS restart without a gym restart (~50 min lost)**

- A `kubectl rollout restart deploy/rl-glm53f-nats` replaced the broker pod.
  Effects: (1) every gym worker's NATS client died with `ConnectionRefusedError`
  against the NATS ClusterIP while the pods stayed `Running 2/2` (no restart,
  no readiness signal); (2) JetStream state is an `emptyDir` (`nats.yaml`), so
  the durable consumers vanished with the old pod; (3) the trainer kept
  dispatching into a stream nobody consumed — `Waiting for results: 0/32
  groups` with idle engines — for ~50 min of rollout 0.
- **RULE:** always restart the gym Deployment right after (or together with)
  any NATS restart; the trainer's `Waiting for results: 0/N groups` with idle
  engines is the detection signal, pod status is not.
- r3 outcome (step 0, post-train sync, rollout 1, step-1 CUDA OOM): r4 entry.

## 2026-09-03 00:40-01:47 PT — Truncation RCA: the ~89% "truncated" are Harbor agent timeouts; image d; README to its r3 state

Sits between the r3 launch (01:32 PT, previous entry) and the r3 outcome
(next entry). Not a run: a root-cause analysis done on r2's gym pods while r3
was being relaunched, the trainer-side flag it produced, and the README
catch-up that records runs 1-2, r2 and r3 as incident history. Plugin-side
detail and the decision record: `miles_plugins/arena/RUNLOG.md` (same date)
and ADR-0010.

### Timeline (PT)

| when | what |
|---|---|
| 00:36 | r2 dies in train step 0 (Triton ESTALE, previous entry). Its rollout 0 had removed 229/256 with the mask-clipped flag on and no salvage line - the length-clip reading of run 1 is falsified. |
| 01:23-01:29 | Evidence pulled from the still-running r2 gym deployment (`rl-glm53f-gym-sgb-*`, 128 pods, gym image `arena-tasks-dev:rl-smoke-20260821b` = Harbor 0.21.0 / `amzn_arena_harbor 1.0.763.0`): 96 trials from 12 pods (12 groups x 8; trials run 23:07-00:29 PT), job history of all 128 pods, gym source, task-events logs. Archived: `arena-port-artifacts/glm53/glm53-salvage/{analysis1-3.txt,trials_summary.json,pod_runs.json,r0_kept_map.json,r0_pods.txt,pods.txt,gym-src/}`. |
| 01:32 | r3 launched on image c (flag does not exist yet; r3 runs under the unchanged removal policy). |
| 01:37-01:38 | `--arena-keep-timeout-trajectories` + removal-reason log committed (integration `32da04357`, 19 tests). |
| 01:40 | Trainer image `arena-slime-dev:miles-glm53-20260903d` built and pushed (c + the flag, default OFF), replicated to ap-south-1. Not deployed to r3. |
| 01:47 | README brought to its r2/r3 state (integration `37164f537`, +250/-60) - this commit ships that version. |

### What the gym pods showed

- 91/96 trials: `AgentTimeoutError: Agent execution timed out after N seconds`
  with N = the task's `task.toml` `[agent] timeout_sec` (1800 s x59, 1500 x16,
  1200 x8, 900 x8; no gym override), reached 31-85 s after trial start (env
  build). 5/96 finished normally.
- 0/1033 generates hit the 32768 per-turn cap (max 15033; p50 630, p90 4948
  output tokens). Every trajectory ends cleanly at the model's turn-close
  token; every step `stop_reason=stop`; turn count == `weight_versions`
  count. Median timeout: 8 turns, 34k tokens; max 74k (< 131k episode cap).
- The gym (`gym_worker._EXCEPTION_AGENT_STOPS`) maps the exception to
  `agent_stop_reason="timeout"`, which `nats_rollout.py` treats as a
  degenerate stop -> TRUNCATED + `remove_sample`. The Harbor 0.21 envelope
  still reports the group `success`, so the wire statuses (run 1: 272/54/0)
  never showed it.
- Reward in the discard pile: 8/91 timeouts scored 1 (~9%; 4 of the 5
  finished trials did) - >=19 of the ~46 reward-1 samples per step were lost.
- Driver: GLM at its template-default `max` reasoning effort thinks 8-15k
  tokens/turn; decode p50 14.5 tok/s per sample with ~400 concurrent trials
  on 4 saturated engines -> ~500 s for a >8k-token turn (61 such turns,
  median latency 494 s).

### Decision

- Trainer side (this commit): keep a timeout trajectory only when the timeout
  is its sole defect, default off, status stays TRUNCATED; log
  `Removal reasons: timeout=.., context_error=.., length=.. (kept_timeout=N)`
  every rollout so the next run needs no pod forensics. ADR-0010 lists the
  precedence with `arena_mask_clipped_final_turn` (correct but inert here -
  there are no clips) and the alternatives.
- Gym/dataset side (not this tree): raise `timeout_sec` (Harbor 0.21 already
  has `JobConfig.agent_timeout_multiplier`, unplumbed by
  `gym_worker.build_rollout_job_config`; needs `ARENA_NATS_ACK_WAIT` raised
  in lockstep), cut thinking via `reasoning_effort` (dropped by the gym's
  `ArenaSGLangLLM`), or more engine throughput. All three were taken later
  (r5 gym image, r6 multiplier + 16 engines) and are logged with those runs.
- Rollout plan: image d ships the flag OFF so r4 measures the loss with
  attribution before anything changes (`Removal reasons: timeout=230/256
  (kept_timeout=0)` in its rollout 0); r5 is the first run to set
  `arena_keep_timeout_trajectories: true` (image d required - strict argparse
  kills a trainer on image c).

### README state at `37164f537` (what this commit's README carries)

- Image lineage table a -> b -> c -> d with the integration commit of each
  layer and the rule "bump the image tag together with any new config key"
  (`arena_mask_clipped_final_turn` needs >= c, `arena_keep_timeout_trajectories`
  needs d); push to us-east-1 and check the ap-south-1 replica.
- Run identity r3 (`EXPERIMENT_NAME`/`PROJECT_NAME`, YAML still says r1 -
  inert); EFS tee + 60 s `memsample-<idx>.log` as the only post-mortem
  source; the kueue TAS mis-pin check and `NotIn` procedure (16 nodes as of
  2026-09-03).
- First-run verification: EFA provider line required, run-2 OOM signature
  absent, r2 ESTALE signature absent, `truncated_ratio` ~0.89 explained as
  the timeout signature with the image-d log line.
- Deviations table (a) extended for r2/r3: `use_fault_tolerance` +
  `rollout_health_check_timeout 900`, `data_pad_size_multiplier 512`,
  `use_kl_loss` removed, `check_weight_update_equal false`,
  `arena_mask_clipped_final_turn true`, `arena_keep_timeout_trajectories`
  "not set - open", image c, `RAY_memory_monitor_refresh_ms=0`, 1800Gi limit
  + 256Gi shm, local kernel caches, memsampler, gym anti-affinity off p6
  nodes, NCCL debug dropped, upstream `extra_env_vars` as pod env.
- New "Incident history" (run 1, run 2, r2, truncation RCA, r3 launch) and
  "Risks / open items (watch on r3)" 1-11, incl. #9 timeouts with the four
  options above and #11 fault tolerance unvalidated on the NATS path.
- Not yet in this README: r3's step-1 CUDA OOM and the optimizer offload
  (next entry), r4 results, the r5 gym image (later entries).

## 2026-09-03 01:32-04:58 PT - r3 outcome (`rl-glm53f-gbash-r3`, image c): memory fix set proven, then step-1 CUDA OOM

Continues the r3 launch entry above (identity, image c, NATS-restart trap
unchanged). First GLM run to get past the point that killed run 2.

**What happened (trainer-0.log, all green until step 1)**

| phase | result | note |
|---|---|---|
| train step 0 | 61 min, MFU 0.40% | no ref pass (`use_kl_loss` dropped in the r2 set); r2's step 0 had been 7722 s incl. 2388 s of ref_log_probs |
| post-train `update_weights` | 19.7 s, OK | the exact point where run 2 died (Ray killed an engine 32 min earlier); Ray monitor off + no WeightChecker snapshot + FT on all held |
| rollout 1 | 3724 s, avg reward 0.281 | engines healthy throughout |
| train step 1 | **CUDA OOM after 73 s** on every rank | in fla `chunk_kda_bwd_wy_dqkg_fused` (KDA backward) on ~37k-token samples: "177 GiB in use, 144 GiB allocated, 1 GiB free" |

**Root cause (from the OOM trace + post-step-0 memory readings)**

- Step 0 fits, step 1 does not, because step 1 is the first step that carries the
  Adam states: fp32 master + m + v, ~33 GB/rank even sharded over DP=2 (they
  materialise at the first `optimizer.step`, i.e. at the END of step 0). After step
  0 each rank already held ~109 GB allocated.
- The KDA layers run all 64 heads on every TP rank (upstream `glm5_next` design:
  KDA projections are replicated, not TP-sharded), so the long-sequence backward
  transient of one ~37k-token sample is large and lands on top of the new ~33 GB.
- Structural follow-up if the offload had not sufficed: shard KDA heads across TP in
  the `glm5_next` plugin (~8x less KDA memory/compute per rank). Not attempted.

r3 ended 04:58 PT (kubeflow deleted the failed job; the EFS tee is the record).
No checkpoint (save_interval 20).

## 2026-09-03 05:31-05:50 PT - r4 fix + launch: optimizer CPU offload trio (`rl-glm53f-gbash-r4`, image d, integration 3a6cbe5ba)

**Decision.** Turn on the optimizer-offload trio in `miles-config.yaml` and relaunch
as r4 - the same three flags every upstream large-MoE miles recipe uses
(`scripts/run_deepseek_v32.py`, `amd/run_glm5_2_744b_a40b.py`):

```yaml
optimizer_cpu_offload: true
overlap_cpu_optimizer_d2h_h2d: true
use_precision_aware_optimizer: true
```

All three go together: Megatron requires precision-aware with CPU offload and
miles asserts the reverse implication. Yaml prediction written at the time: ~66 GB
fp32 state per rank -> ~530 GB per actor node, inside the 1800Gi limit; engine
nodes unaffected.

**Alternatives considered and not taken**

- DP4 (16 actor nodes): halves the per-rank optimizer shard but costs 8 more
  p6-b200 nodes on a cluster where the 12-node gang already fights TAS mis-pins;
  kept as the fallback if host RAM under offload proved tight.
- TP-sharded KDA in the `glm5_next` plugin: the structural fix, but a model-code
  change with no CPU test path here; deferred (templates exist upstream, see the
  precedent entry below).
- Shorter samples (`rollout_max_response_len` / context caps back down): removes
  the training signal the 32k/131k caps were raised for in run 1.
- `micro_batch_size` and recompute were already at their floor (1 / full-uniform-1).

**Verification before launch.** Offline strict+full `parse_args` of the
launcher-built argv (`/tmp/glm53-e3`, same harness as the r2 fix set) passed at
05:30 PT with `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer` present. Committed as integration 3a6cbe5ba
05:31 PT - the LAST integration-tree commit; every later change lives only in the
fork.

**Run identity**

| item | value |
|---|---|
| `EXPERIMENT_NAME` / `PROJECT_NAME` / W&B group (`arena/rl-snorkel27`) | `rl-glm53f-gbash-r4` (r1-r3 saved no checkpoint, so nothing resumes) |
| trainer image | `arena-slime-dev:miles-glm53-20260903d` = c + `--arena-keep-timeout-trajectories` (default OFF, NOT set here) + the per-rollout `Removal reasons:` line; built 01:40 PT, i.e. after r3 launched on c |
| shape / batch | unchanged: 12x p6-b200 (8 actor TP8/PP4[11/11/11/12]/EP16, DP2 + 4 engines TP8/EP8); rbs 32 x 8 = GBS 256, num_rollout 90, save_interval 20; 128 gym workers on `arena-tasks-dev:rl-smoke-20260821b` (Harbor 0.21.0) |
| manifest state | NotIn list 16 nodes, request 1200Gi / limit 1800Gi, shm 256Gi, `RAY_memory_monitor_refresh_ms=0`, local `/tmp/kernel_cache`; `arena_mask_clipped_final_turn: true` still on (inert - no clips) |

Launched 05:50 PT.

## 2026-09-03 05:50-19:20 PT - r4: offload trio validated end to end; steady state is rollout-bound and timeout-starved

**Validation (first two steps)**

| phase | r4 | r3 for comparison |
|---|---|---|
| rollout 0 | 4890 s, reward 0.164, `Removal reasons: timeout=230 (kept_timeout=0)` of 256 | - |
| train step 0 | 4273 s (cold JIT caches), loss -0.029 | 61 min |
| post-train sync | 19.7 s | 19.7 s |
| rollout 1 | 3734 s, reward 0.258, timeout=221 | 3724 s, reward 0.281 |
| train step 1 | **584 s, NO OOM**, loss -0.091, ppo_kl 0.0016 | OOM at 73 s |

- GPU after step 0: ~36-40 GB allocated per rank (r3: ~109 GB) - the ~70 GB/rank
  the offload bought.
- Host RAM: the offloaded Adam state materialised at the end of step 0 and
  plateaued within ~2 min at 438-615 GiB pod-anon per actor node (PP-stage
  dependent; ~245 GiB before), max 34% of the 1800Gi limit with >= 1.1 TB
  MemAvailable. Brackets the yaml's ~530 GB/node estimate and sits far below the
  880-990 GB the later 16 B/param model predicted for the PP3 nodes. DP2 is fine;
  no DP4.
- Engine-side "Connection refused" tracebacks during weight sync are benign retries.

**Steady state to the 19:20 PT handoff** (user monitors r4 from another session)

- 12 rollouts + 12 train steps in 13.7 h; no memory / OOM / fault-tolerance event;
  host anon flat at 439-618 GiB per actor node.
- Warm train step 373-449 s; rollouts 3478-4094 s -> rollout-bound ~9x.
- Reward 0.16-0.27 oscillating, no trend. Reward-1 samples per rollout 42-43
  (steps 0-5) -> 51-68 (steps 6-11; step 10 dipped to 43).
- Timeouts remove 222-245/256 samples every step -> only 18-35 trainable samples,
  ess_ratio 0.07-0.13. >= 22 reward-1 trajectories discarded as timeouts per
  rollout (rollout 9: 51 reward-1 vs 29 kept).

**Timeout mechanism, measured** (12 gym pods, ~45 jobs; `jobdelta.py` pipes
`kubectl logs <gym pod> -c gym-worker | grep -aE "Rollout job |AgentTimeoutError
is in exclude"` into per-job start/timeout deltas; archived at
`arena-port-artifacts/glm53/jobdelta.py`):

- Every timed-out job's 8 trials throw `AgentTimeoutError` in a burst at the
  task.toml `[agent] timeout_sec` + 30-80 s (environment build): clusters
  1829-1878 s (1800-s tasks; 8/13 cached tasks), 1555-1562 (1500), 1231-1258
  (1200), 933-987 (900). Typically 8/8 trials of a group.
- The gym's 4200-s job deadline (`ARENA_NATS_ACK_WAIT` 4500 - 300) never fires.
- At 100+ running requests per engine a 1800-s episode is ~40-45k generated
  tokens incl. thinking - why GLM-5.3 at max effort runs out of clock.
- The deployed gym image runs Harbor 0.21.0, which ALREADY has
  `JobConfig.agent_timeout_multiplier` + `AgentConfig.override_timeout_sec` /
  `max_timeout_sec` (`trial.py _compute_agent_timeout_sec` = (override or task
  timeout_sec) x multiplier); the RL path `gym_worker.build_rollout_job_config`
  hard-wires task.toml, while the eval CLI has had `--agent-timeout-multiplier`
  since 2026-08-26. Decision: the fix is gym-side (AREnATasks) - plumb
  `ARENA_AGENT_TIMEOUT_MULTIPLIER` into the rollout job config and raise
  `ARENA_NATS_ACK_WAIT` in lockstep on gym AND trainer (a multiplier > ~2.2 would
  otherwise hit the 4200-s deadline). Trainer-side `--arena-keep-timeout-trajectories`
  (available in image d, OFF in r4) is the complement, first enabled in r5.

r4's `trainer-0.log` ends 2026-09-03 20:29 PT after 13 steps (last completed step
perf 12, reward 0.19); it was torn down 3 min before r5's log opens at 20:32 PT
(next entry). No checkpoint (save_interval 20 > 13 steps).

## 2026-09-03 - Precedent hunt (14-agent workflow) while r4 ran: what others did with this shape

Recorded here because it shaped r5-r7; verified unless marked.

- **AGISlime `origin/v0.3.1/glm52`** (GLM-5.2 744B RL on prod-bom-v2,
  `experiments/k8s/glm52/glm52-chakra-config.yaml`): the same offload trio, 32
  trainer nodes TP4/PP8/CP2/EP8 DP4 + 32 single-node FP8 engines (fp8_e4m3 KV,
  `cuda_graph_max_bs 32`, overlap schedule), `rollout_batch_size 64`. Their engines
  were STARVED; ours are saturated - so batch is not a copy-paste lever. Its "DP2
  hit 1.84 TB host RAM (exp8)" is a comment on an unlimited pod with an uncommitted
  config - not transferable. `partial_rollout` inert there; `enable_pipeline_rl` ==
  miles `pause_generation_mode: in_place`; `arena_instance_timeout` is read only by
  the AREnABase gym, never by the Harbor gym. CP is closed for `glm5_next` (kpool
  indexer raises on CP > 1; KDA has no CP).
- **Host RAM under offload** (Megatron `e8f57451` in image d): 16 B/param of the
  rank's shard (fp32 master + pinned fp32 grad + AdamW m, v), materialised at the
  first `optimizer.step`; expert params are NOT DP-sharded (expert-DP =
  64/(1x16x4) = 1) and KDA projections are replicated per TP rank. Predicted
  ~880-990 GB on the PP3 actor nodes (~50% of 1800Gi); r4 measured 438-615 GiB.
  Precision-aware dtype knobs are inert under offload (miles PR #1592).
  `optimizer_offload_fraction` (default 1.0; AMD DSv4 recipe 0.75) is the config
  lever if host RAM ever binds; DP4 = 16 actor nodes is the fallback.
- **Upstream miles:** PR #1571 (GLM-5.2 on GB300) is the exact "step 0 fits, step 1
  OOMs on fp32 m/v" precedent (their fix: a smaller per-rank optimizer shard).
  TP-sharded KDA already exists as templates: radixark/Megatron-Bridge PR #35
  `glm5_next/kda.py` (bridge mode, used by miles PR #3098) and the miles `kimi-k3`
  branch (PR #1825) `kimi_k3/layers.py` (`local_num_heads = num_heads // tp`);
  fla issue #1155 confirms KDA backward memory is linear in resident heads.
  Upstream `run_glm5_3_flash.py` has NO offload flags, `micro_batch_size 1`, no
  dynamic batching.
- **Internal ForgeModelEnablement `enablements/glm53-flash-sft-strl`** (run
  `zhuoweli-glm53-rl-hmwg`) independently enabled GLM-5.3-Flash RL on p6-b200 and
  MEASURED a `use_dynamic_batch_size` + PP > 1 deadlock in `send_forward` (the
  micro-batch count is all-reduced over DP only). **Live risk for this config**
  (dynamic batching with PP4): harmless while every ~32k sample is its own
  micro-batch, but it can fire once short samples appear. They use
  `FLA_CACHE_RESULTS=0` instead of our fla kernel patch (alternative, not
  adopted), plus the same `RAY_memory_monitor_refresh_ms=0` / local Triton cache /
  rotary-base fixes. `use_rollout_logprobs` makes `train_rollout_logprob_abs_diff`
  tautologically 0 - run one diagnostic step with it off before trusting that
  metric. `log_probs_chunk_size 16384` is hygiene only.
- **FP8 engines are NOT config-only for `glm5_next`** (`quantizer_fp8` lacks
  `modules_to_not_convert` handling; `kv_b_proj`). Not pursued.

## 2026-09-03 20:32 PT — r5: keep-timeout ON + reasoning-effort gym image (`rl-glm53f-gbash-r5`; image d; gym `glm53-reasoning-20260903b`, effort low)

First run that applies both answers to the timeout RCA at once: the trainer
keeps clean Harbor-timeout trajectories (`arena_keep_timeout_trajectories: true`,
image d, previous entries) and the gym asks GLM-5.3 for less thinking
(`ARENA_REASONING_EFFORT=low` via a new gym image). Shape, trainer image and
optimizer recipe are r4's (12 nodes = 8 actor + 4 engines, rbs 32 / GBS 256 /
`num_rollout` 90, `miles-glm53-20260903d`). The r5 manifests were edited in place
and never saved; this entry reconstructs them from the memory notes, the yaml
comments and the archived gym-pod evidence.

**Identity**

- `EXPERIMENT_NAME` / W&B group `rl-glm53f-gbash-r5`, W&B run `43f0vjwx` —
  confirmed from the EFS log dir `/mnt/scratch-s3files-rw/guparpit/logs/rl-glm53f-gbash-r5/`
  (`trainer-0.log` opens 2026-09-03 20:32 PT, ends 2026-09-05 00:12 PT).
- Kubernetes names unchanged from r1-r4 (`rl-glm53f-trainer`, `rl-glm53f-nats`,
  `rl-glm53f-sglang`, gym `rl-glm53f-gym-sgb` x128 — ReplicaSet hash `64d99dc846`
  in the archived pod lists), so r4 had been torn down before r5 came up; r4's
  `trainer-0.log` ends 20:29 PT, 3 min before r5's opens (previous entry).
- Trainer image `miles-glm53-20260903d` — the lineage table ends at d, the r4 and
  r7 manifests both pin it, and d (pushed 2026-09-03 01:40 PT) is the newest
  trainer tag in ECR.
- Gym image `arena-tasks-dev:glm53-reasoning-20260903b` (harbor 0.22.0,
  amzn-arena-harbor 1.0.1061.0): AREnATasks HEAD `35f7ba7` + the then-uncommitted
  patch (archived as `arena-port-artifacts/glm53/glm53-reasoning/arenatasks-reasoning-effort.patch`,
  later commit `0fb3e54`): `ArenaSGLangLLM(reasoning_effort=)` renders the chat
  template's `reasoning_effort` on turn 1 only (template honours `low`/`high`,
  anything else renders `Reasoning Effort: Max` — the client warns once, and
  raises if the template never reads the variable); `ARENA_REASONING_EFFORT` ->
  `llm_kwargs` for both agents; and the pre-existing `<|user|><|user|>` splice
  fixed (GLM stops on `<|user|>`, the template suffix opens with it — 8/8 turn
  boundaries doubled in a live r4 `rollout.json`). First cut `20260903a` raised
  on `max` and kept the splice bug; b is the one deployed. Built 19:30-20:10 PT,
  verified with the tokenizer copied out of the image
  (`arena-port-artifacts/glm53/glm53-tok/e2e.py`).
- Gym CLI contract of HEAD: `--mode rollout --agent arena-terminus-2` (HEAD
  requires `--mode` and defaults to Vulcan). HEAD-vs-0.21 drift checked: result
  envelope reports timed-out groups as `truncated` instead of `success` (the
  trainer salvages both — telemetry only); harbor 0.22 task-root
  `trajectory.json` seeding inert for snorkel; lakeFS materializer 8 threads.

**Timeline (PT)**

| when | what |
|---|---|
| 20:22 | offline strict + full argv parse of the r5 `miles-config.yaml` (glm53-e3 harness; `parse_strict.log`, `strict.exit`/`full.exit` = 0) |
| ~20:31 | 128 gym pods up on image b, polling for the NATS stream ("NATS not ready", attempt 1/30) — trainer not yet up |
| 20:32 | `trainer-0.log` opens (r4's ended 20:29) |
| 20:46 | workers attached (durable `gym-worker-snorkel-general-bash-harbor`); first rollout-0 jobs dispatched (e.g. `resume-replay-ledger.g22` 20:46:21) |
| 21:19 | timeout sample from the gym logs (`r5_jobs.txt`, 45 jobs): 8/8 trials time out at 947-957 s on 900 s tasks and 1841-1887 s on 1800 s tasks — r4's signature, unchanged |
| 09-04 01:10 | r6 submitted alongside (next entry); r5 kept running — 4 steps done at that point |
| 09-05 00:12 | killed on the user's order at step 40 (41 steps done, reward 0.395; `trainer-0.log` ends 00:12) to free its 12 nodes for r7 |

**Results**

- First 4 steps: rollouts 3874 / 3197 / 2592 / 2718 s, reward 0.273 / 0.328 /
  0.367 / 0.398, ess_ratio 0.97, ppo_kl 0.011, actors ~21% duty (rollout-bound).
- Steady state to step 40: ~50 min/rollout, ~10 min/step, reward 0.30-0.40 (0.39
  at the kill), `kept_timeout` 183-221/256, `truncated_ratio` 0.71-0.86; dynamic
  sampling published ~3.5x the kept groups (README).
- The keep-timeout flag did its job: the 183-221 timeout trajectories per
  rollout now train instead of being removed (r4 trained on 18-35 samples per
  step after removing 222-245/256).
- **Low effort did NOT cut timeouts.** ~90% of published trials still time out
  at `task.toml timeout_sec` + ~40-90 s after ~13 turns of 60-120 s each. Cause:
  the run is engine-KV-bound, not think-length-bound — the 4 engines sat at KV
  usage 0.97 with ~100 running + 10-60 queued requests each, so per-request
  decode stayed throughput-limited and less thinking per turn did not get
  episodes to a natural stop inside the wall clock.
- r5 is the first GLM run past `save_interval 20`; its checkpoint dir
  (`slime_experiments/rl-glm53f-gbash-r5`, saves at steps 20/40) is where
  `hf_export`'s behaviour on GLM can be inspected (not done).

**Decision -> r6.** Attack the wall clock from both ends: 4x the engines and
halve the per-engine queue (rbs 64 lifts the in-flight cap to 128 groups), and
double the Harbor agent timeout gym-side (`ARENA_AGENT_TIMEOUT_MULTIPLIER`, which
AREnATasks mainline `c1a0439` had just added for the eval path — authored
19:01 PT, committed 20:39 PT) with
`ARENA_NATS_ACK_WAIT` raised in lockstep. Rejected: raising `timeout_sec` per
task (a dataset change); pushing effort lower (only `low`/`high` exist, and low
was just shown ineffective); a multiplier > ~2.2 without the ack_wait change
(the group deadline `ack_wait - 300 s` would cancel whole groups); staying at
12 nodes (~5x rollout-bound at 50 min rollout vs 10 min step).

Evidence archived under `arena-port-artifacts/glm53/glm53-reasoning/`
(`r5_jobs*.txt`, `gym_pods.txt`, `gym_pod_nodes.txt`, `mount_check.txt` — model
mount readable on 125 of the 128 gym pods, 3 BROKEN —, `onepod.log`, `tmo_pod.log`,
`think_stats*.py`, `decode_head.py`, `apply_patch*.py`) and `glm53-tok/`. The
think-token statistics those scripts printed were not archived.

## 2026-09-04 01:10 PT — r6: 16 engines, rbs 64 / GBS 512, 2x agent timeout (prefix `rl-glm53f6`; gym `glm53-reasoning-20260904a`)

Side-by-side with r5 (separate NATS: stream names are per-server). Submitted
01:10 PT, kueue-admitted 01:37 PT; `trainer-0.log` opens 01:21 PT. `EXPERIMENT_NAME` /
W&B group `rl-glm53f-gbash-r6`, W&B run `gweq9pme` — confirmed from the EFS log dir
`/mnt/scratch-s3files-rw/guparpit/logs/rl-glm53f-gbash-r6/` (the r6 manifest itself
was not saved).

**Shape and levers (all now in the committed files, see the table at the end)**

- Trainer 24 replicas = 8 actor (unchanged TP8/PP4/EP16/DP2) + 16 engine nodes
  (TP8/EP8); `REPLICA=24`. Trainer image d; optimizer offload trio as r4.
- `rollout_batch_size` 32 -> 64, `global_batch_size` 256 -> 512: the publisher's
  in-flight cap (2x batch) goes 64 -> 128 groups, so each of 16 engines sees ~64
  requests instead of r5's ~130. `num_rollout` 130 ~= 10 passes over the
  2922-task list (~3.5x publish factor -> ~224 prompts/rollout -> ~13
  rollouts/pass). Gym 128 -> 160 replicas (>= 128 in flight + slack).
- Gym image `glm53-reasoning-20260904a` = AREnATasks mainline `c1a0439`
  (`ARENA_AGENT_TIMEOUT_MULTIPLIER`, eval path only) + `0fb3e54` (effort
  plumbing + splice fix rebased, 00:52 PT) + `bae6a6b` (pass the multiplier on
  the training/rollout path too, 00:58 PT).
- `ARENA_AGENT_TIMEOUT_MULTIPLIER=2` (task `timeout_sec` 900-1800 s -> 1800-3600 s),
  `ARENA_NATS_ACK_WAIT` 4500 -> 6000 (per-message deadline `ack_wait - 300 s`
  bounds all 8 trials of a group; trainer `NATS_TASK_DEADLINE_SECS` default 12600
  still covers it). `ARENA_REASONING_EFFORT=low` kept. `sglang_disable_radix_cache`
  deliberately unchanged.

**Findings**

1. **kueue priority is already maxed.** Only two WorkloadPriorityClasses exist:
   `inference` = 1000 and `arena-backfill-low` = -10; every training job runs at
   1000 through `priorityClassName: high`. "Submit at higher priority" cannot
   jump other training jobs. (r7 adds the explicit
   `kueue.x-k8s.io/priority-class: inference` label — visibility only.)
2. **Fresh-node Docker-Hub trap, three layers deep** (gym pods scheduled onto
   nodes with nothing cached; r5's pods had everything cached from launch):
   - harbor 0.22 `DockerEnvironment` probes kernel nftables once per trial with
     `docker run alpine:3.23.4@sha256:5b10f4...`; the anonymous Docker Hub pull
     is rejected -> probe returns False -> egress control silently disabled ->
     every `network_mode='no-network'` task rejected at `Trial.create`
     ("network_mode='no-network' is not supported by EnvironmentType.DOCKER").
     270 groups burned on this alone.
   - with the probe passing, harbor builds its egress sidecar `FROM
     docker.io/gogost/gost@sha256:afc0...` -> "Failed to build Docker image
     harbor-prebuilt:harbor-docker-egress-control-sidecar--b1e6760734a9e6f6".
   - mountpoint-s3 fast scratch (`ARENA_MODEL_PATH`) unreadable for minutes after
     pod start -> `AutoTokenizer` falls through to the HF Hub ("Repo id must be
     in the form 'repo_name'"); one pod failed every trial this way.
   Fix, in the `gym-worker.yaml` start script: pull the ECR mirror
   `arena-tasks-dev:alpine-probe-3.23.4` (made with `buildx imagetools create`,
   index digest byte-identical; dind's containerd store resolves digest refs by
   target digest) and `docker tag` it `alpine:3.23.4`; pull
   `arena-tasks-dev:egress-sidecar-b1e6760734a9e6f6` (`docker save` from an r5
   pod — identical sidecar context in 0.22.0, identical hash) and tag it under
   harbor's content-addressed name (harbor skips the build when `docker image
   inspect` succeeds); `until [ -f $ARENA_MODEL_PATH/tokenizer_config.json ]`.
   Cost before the fix landed: ~1000 dropped groups, about 1/3 of a task-list
   pass.
3. **NATS after a gym scale-to-0** (needed for the start-script rollout):
   delivered messages sit ack-pending for `ACK_WAIT` (now 6000 s). Reset from the
   trainer pod with nats-py: `js.delete_consumer("ARENA_TASKS",
   "gym-worker-snorkel-general-bash-harbor")` — the gym recreates the durable on
   start. Monitor at `http://<nats-svc>:8222/jsz?streams=true&consumers=true`.
4. Startup burn: the trainer publishes its 128-group in-flight window before the
   engines are up; those groups fail "All connection attempts failed" (~4% of a
   pass, bounded). Same in r7.

**Results (rollout 0, steps 0-1)**

| metric | r5 (12 nodes, rbs 32) | r6 (24 nodes, rbs 64) |
|---|---|---|
| kept_timeout share | 72-86% (183-221/256) | 45% (228/512) |
| `truncated_ratio` | 0.71-0.86 | 0.45 |
| reward | 0.30-0.40 | 0.40 |
| mean response | — | 33k tokens (2x r5); episode total ~58k |
| train step 0 | — | 6749 s (cold JIT) |
| warm train step | ~10 min | 1380 s at 140 TFLOPs |

Training is now shorter than the ~50 min rollout by ~2x (r5: ~5x), and half the
kept samples reach a natural stop instead of a quarter or less.

**Residual defect -> r7.** 33-74 of 512 samples per rollout (~7-14%) came back
`failed` with `httpx.ReadTimeout` ("Unknown Error in LLM interaction: ", empty
message): `amzn_arena_harbor/sglang_rollout.py::_get_client` hard-codes
`httpx.Timeout(600.0)` and a 32k-token response at ~50 tok/s per request on a
loaded engine runs past it. Fix written the same day on the local AREnATasks
branch `guparpit/reasoning-effort`: env `ARENA_SGLANG_REQUEST_TIMEOUT_SEC`
(default 600, validated > 0) + `TestSglangRequestTimeout`, `brazil-build release`
green; not deployed into r6 — it needs a new gym image and a rolling restart of
160 pods, to be done right after a rollout completes. r6 was still running at
2026-09-05 05:15 PT, side by side with r7: latest completed step 33, reward 0.469
on 512 samples (summary entry).

## 2026-09-04 18:38 PT -> 2026-09-05 00:30 PT — r7: r6 shape + 1800 s SGLang call timeout + effort high (`rl-glm53f-gbash-r7`, prefix `rl-glm53f7`) — IN PROGRESS

Identity (all in the committed manifests): `EXPERIMENT_NAME` / `PROJECT_NAME` /
W&B group `rl-glm53f-gbash-r7`; PyTorchJob `rl-glm53f7-trainer`, NATS
`rl-glm53f7-nats`, Service `rl-glm53f7-sglang`, gym `rl-glm53f7-gym-sgb` x160;
trainer image `miles-glm53-20260903d`; gym image
`arena-tasks-dev:glm53-sgltimeout-20260904b` = `0fb3e54` + `bae6a6b` + the
uncommitted `ARENA_SGLANG_REQUEST_TIMEOUT_SEC` change (archived diff
`arena-port-artifacts/glm53/arenatasks-patches/arenatasks-glm53-sgltimeout.patch`
== `git diff c1a0439` at 18:59 PT; it is the change this plan commits in
AREnATasks).

Changes vs r6, both gym-side: `ARENA_SGLANG_REQUEST_TIMEOUT_SEC=1800` (new; the
image reads it) and `ARENA_REASONING_EFFORT` low -> high (low did not reduce
timeouts in r5; the template honours only these two values). Trainer side: only
the rename and the `kueue.x-k8s.io/priority-class: inference` label.

**Timeline (PT)**

| when | what |
|---|---|
| 09-04 18:38 | gym-worker / miles-config / nats / sglang-svc renamed to `rl-glm53f7`, gym image + the two env changes, README r7 section |
| 19:03 | first `kubectl apply`; kueue leaves the PyTorchJob Suspended (pending on quota) |
| by 23:13 | the Suspended job is GONE — no events, no audit trail visible from arena-tasks; the NATS Deployment/Service/ConfigMaps survive |
| 23:14 | second submission (the trainer yaml's last edit is stamped 23:14 PT; the committed copy carries the priority-class label) -> gone ~23:27 |
| 23:41 | third submission -> gone ~00:10 |
| 09-05 00:12 | r5 killed on the user's order (step 40, reward 0.39) |
| 00:13 | fourth submission; 00:14 kueue admits it |
| 00:30 | engines up, first `update_weights`; rollout 0 collecting |

**Findings so far**

- A PyTorchJob left Suspended in arena-tasks for ~25-30 min gets deleted by
  something outside our RBAC view (3 for 3). Root cause unknown. Workaround that
  worked: submit only when the quota gap is small, or within a minute of freeing
  capacity.
- Priority cannot help (r6 finding 1); the label only makes the 1000 visible on
  the Workload.
- Startup burn of the first 128 groups ("All connection attempts failed", window
  published before engines are up) — ~4% of a pass, as in r6.
- SGLang startup tracebacks (`cpu_ids` `int('\n')`, `sock.connect`) are benign
  noise. Gym env checked with `kubectl exec ... env`: effort `high` is set (the
  gym does not log the rendered effort).
- Read back from `trainer-0.log` (opens 00:14 PT) at 05:15 PT, steps 0-3
  complete: step 0 reward 0.33, `truncated_ratio` 0.69, `failed` 15,
  `kept_timeout` 352/512, rollout 5944 s, train 6784 s (cold JIT, as r6's
  step 0); step 1 reward 0.39, truncated 0.58, failed 43, kept_timeout 296,
  rollout 3302 s, train 2054 s; step 2 reward 0.39, truncated 0.64, failed 16,
  kept_timeout 326; step 3 reward 0.40.
- Gym logs: zero `httpx` timeouts since the engines came up — the 1800 s
  lever works. ~3-14% of calls hit `Context length exceeded` at ~99-101k
  prompt tokens: the conversation may grow to
  `ARENA_ROLLOUT_CONTEXT_LIMIT=131072` while each call still asks for
  `ARENA_MAX_TOKENS=32768` new tokens; these end the episode as `truncated`,
  not `failed`.

**Pending (recorded in the summary entry's placeholder row, filled by a
follow-up docs commit):** healthy-step count (the commit gate is >= 5, user
instruction 2026-09-05 02:45 PT); `kept_timeout` share at effort high; whether
1800 s removed r6's 7-14% `failed` samples; the context-ceiling /
`context_error` observation against the 131072 `rollout_max_context_len` /
`ARENA_ROLLOUT_CONTEXT_LIMIT`; W&B run id.

**Manifest state committed with this entry** — the r7 files, because the r5 and
r6 intermediates were never saved (reconstructing them by hand was rejected as
fabricated history). Deltas vs the r4 tree (`3a6cbe5ba`):

| file | delta | first used in |
|---|---|---|
| `miles-config.yaml` | `arena_keep_timeout_trajectories: true` | r5 |
| | `replicas` 12 -> 24, `rollout_batch_size` 32 -> 64, `global_batch_size` 256 -> 512, `num_rollout` 90 -> 130 | r6 |
| | `experiment_name`/`project_name` record r1 -> r7 (launcher never reads them) | r7 |
| `trainer-pytorchjob.yaml` | 24 replicas (min/max/Worker), `REPLICA=24` | r6 |
| | names `rl-glm53f` -> `rl-glm53f7`, `EXPERIMENT_NAME`/`PROJECT_NAME` r4 -> r7, `NATS_URL`, configmap name, priority-class label; image d unchanged | r7 |
| `gym-worker.yaml` | image `rl-smoke-20260821b` -> `glm53-reasoning-20260903b` (now `glm53-sgltimeout-20260904b`), `--mode rollout --agent arena-terminus-2`, `ARENA_REASONING_EFFORT` | r5 |
| | `replicas` 128 -> 160, `ARENA_NATS_ACK_WAIT` 4500 -> 6000, `ARENA_AGENT_TIMEOUT_MULTIPLIER=2`, pre-seed start script (probe + sidecar mirrors, tokenizer gate) | r6 |
| | `ARENA_SGLANG_REQUEST_TIMEOUT_SEC=1800`, effort low -> high, names -> `rl-glm53f7` | r7 |
| `nats.yaml`, `sglang-svc.yaml` | rename `rl-glm53f` -> `rl-glm53f7` only | r7 |
| `README.md` | + sections "r5 gym image: reasoning effort + splice fix", "r6", "r7" | r5-r7 |

Stale text deliberately left as history (fix in a separate docs commit if
wanted): README "Current run identity is **r3**" and its 16-entry `NotIn`
procedure, `nats.yaml` header "27B RL run 2", the `miles-config.yaml`
"wandb group = ... r1" comment, the `gym-worker.yaml` header "replicas 8 -> 128".
The README preflight claim that the tree carries the GLM merge is true on this
branch from the PR #2786 merge commit onward.
