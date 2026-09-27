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
