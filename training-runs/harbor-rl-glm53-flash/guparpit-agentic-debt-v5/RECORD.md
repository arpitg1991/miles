# Run record: guparpit-agentic-debt-v5 (r52) — r49 with EP8 and the FlashMLA DSA forward

**Status:** Prepared
<!-- gen-workflow:begin -->
**Date:** 2026-10-02
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v5-`
**Experiment name:** `guparpit-agentic-debt-v5`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v5`)
**Dataset:** `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480`
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-recon-20261002-flashmla@sha256:6a0b7b8a6021ab743d9a0104d46ec2affd63056b44cc4f91641083edb92a83dd`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v3`
<!-- gen-workflow:end -->
**Argo workflow:** not submitted
**W&B run:** not created
**Task pin:** `3cadc6b0`
**Image digests:** gym `sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d`, trainer `sha256:6a0b7b8a6021ab743d9a0104d46ec2affd63056b44cc4f91641083edb92a83dd`
**Trainer config deltas vs base:** `miles_dsa_sparse_attention_forward_backend` unset (`tilelang`) -> `flash_mla`; image `miles-glm53-recon-20261001b` -> `miles-glm53-recon-20261002-flashmla` (PR #3608 kernels); `experiment_name`, `project_name`, `arena_sample_summary_dir` name the run
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v5`, seeded from r47 `iter_0000059` (2026-10-01 02:51Z)
**Outcome:** Prepared

Starts from r47 checkpoint `iter_0000059`: a copy that was made before this record, with a sidecar without `wandb_run_id`. miles resumes at rollout 60. This record copies no checkpoint. Sibling arms: r50 (`guparpit-agentic-debt-v3`, r49 plus EP8) and r51 (`guparpit-agentic-debt-v4`, r50 plus an output queue capped at 64 groups).

## Goal

Find out if the DSA kernels of upstream PR #3608 with the FlashMLA forward make the live trainer faster at the same learning. The run ends when the gates below pass or fail at rollout 75.

## Setup

r52 is r50 (r49 plus EP8) with the FlashMLA forward selected on a new image.

| Item | Base r50 (`guparpit-agentic-debt-v3`) | This run |
| --- | --- | --- |
| trainer image | `miles-glm53-recon-20261001b` (miles `da93977ba8`) | `miles-glm53-recon-20261002-flashmla` (miles `a515a167b6`, branch `arpit-r52-flashmla`) |
| DSA sparse-attention forward | old TileLang kernel (`glm5/ops/tilelang_sparse_mla_fwd.py`) | FlashMLA sparse prefill (`flash_mla` 1.0.0+b7643bd), 8 TP-local heads padded to 64, top-k 2112 padded to 2176 |
| DSA sparse-attention backward | old TileLang kernel, static lengths | PR #3608 TileLang kernel: indices and masks in shared memory, dynamic batch and sequence lengths |
| `miles_dsa_sparse_attention_forward_backend` | not a flag (`tilelang`) | `flash_mla` |
| `expert_model_parallel_size` | 8 | 8 |
| `data_pad_size_multiplier` | 512 | 512 (kept, so the kernel is the only change) |

The image commit `a515a167b6` is the r50 image commit `da93977ba8` plus 6 commits:

| Commit | Source | Change |
| --- | --- | --- |
| `fc38eb6dcb` | upstream #3605 `ff3de16927`, picked with `-x` | quant, MoE, dense-attention backward and activation kernels move to `miles/kernels` |
| `460a10c7b5` | upstream #3606 `817ed9e805` | attention kernels move to `miles/kernels/attention`; kpool Triton part splits into `dsa/kpool.py` |
| `8062607fe1` | upstream #3607 `c6221495d5` | one DSA kernel pair for GLM-5 and DeepSeek-V4 in `miles/kernels/attention/dsa/tilelang` |
| `5e23812d0f` | upstream #3608 `557fb097c2` (the PR head) | FlashMLA forward, TileLang shared-memory fix, dynamic backward shapes |
| `5c38daf8cf` | fork only | `--miles-dsa-sparse-attention-forward-backend` (`tilelang` or `flash_mla`, default `tilelang`) |
| `a515a167b6` | fork only | the kdatp DSA harness imports the kpool helpers from their new module |

The four upstream commits apply without a conflict. Their net patch is equal to the PR net patch `8f409c7c49..557fb097c2` line for line. All `miles/kernels` files and the PR tests are byte-equal to the PR head.

Diffs between the PR head and this port:

- The PR makes FlashMLA the default forward when `flash_mla` imports on SM90 or SM100 and the shape fits. The base image carries `flash_mla`, so the PR alone would move every GLM-5, GLM-5.3-Flash and DeepSeek-V4 run on this image to FlashMLA. The fork flag keeps TileLang as the default. All three model call sites (`glm5/glm5.py`, `glm5_next/dsa.py`, `deepseek_v4/deepseek_v4.py`) pass the flag value as `forward_backend`, the selector that the PR defines. The GLM attention raises at construction when the flag selects `flash_mla` and `flash_mla` does not import.
- `miles/utils/arguments.py` and `docs/user-guide/training-backend.md` keep the lines that the fork and 3 upstream commits after the PR base (`8f409c7c49..d7f1a42109`) changed. The PR hunks in them are the same.
- `examples/arena/harbor-rl-glm53-flash/kdatp/dsa/t1_dsa.py` imports `build_pooled_keys` and `pool_boundaries` from `miles/kernels/attention/dsa/kpool.py`. The earlier port (`arpit-dsa-3608`) kept them in `kpool_indexer.py`.

Against the earlier port `arpit-dsa-3608` (`a60c648e61`), which the kdatp T2 arms measured, the TileLang kernel files and the FlashMLA wrapper logic are the same. The PR head adds lazy TileLang imports (so the package imports on CPU) and the kpool split. The other differences come from the newer upstream base, for example the GLM-5 indexer query norm.

## Upstream check

- radixark/miles PR #3608 at `557fb097c2`: OPEN, `REVIEW_REQUIRED`, not draft, last update 2026-10-01 08:04Z (`gh pr view 3608 -R radixark/miles`, read 2026-10-02). It is stacked on #3605 to #3607 and rebased on main `8f409c7c49`. The recon base `d7f1a42109` is 3 commits past that.
- The PR selects the backend in `_default_forward_backend`, and the caller overrides it with `forward_backend` (PR body: "TileLang stays as the fallback, selectable with `forward_backend`"). No upstream launch flag exists. The fork flag follows the `--miles-dsa-topk-backend` pattern: module spec params for GLM, the config for DeepSeek-V4.
- The upstream base image installs the FlashMLA wheel on CUDA 13 (`docker/Dockerfile` l.124-128). `miles-base-d7f1a42-20261001a` has `flash_mla` 1.0.0+b7643bd and TileLang 0.1.14. The PR ran FlashMLA `b7643bd` and TileLang 0.1.12 and 0.1.14.
- PR numbers, GLM-5.3-Flash with the 64-wide zero tail, TP8, fwd+bwd per layer: GB300 64K 552.9 -> 337.7 ms (1.6x); H200 64K 744.2 -> 351.9 ms (2.1x). The PR has no B200 number.
- The SGLang rollout keeps `sglang_dsa_prefill_backend` and `sglang_dsa_decode_backend` at `tilelang`, as r50.

Measured on the earlier port (round 20260930 of the trainer-core profile; image `miles-glm53-r18dsa-20260930a`, miles `a60c648e61`, the same FlashMLA, TileLang 0.1.9, the r17 base, not the recon base):

| Arm (T2 rows, warm steps 41 to 43) | Step | vs base 776.9 s | Step-40 `grad_norm` vs base 0.126965 | Memory |
| --- | --- | --- | --- | --- |
| TP8, EP16, FlashMLA forward | 588.9 s | -24.2% | 0.127806 (+0.66%) | +2.5 GiB allocated, +1.4 to +8.7 GiB reserved per stage |
| TP8, EP16, TileLang forward | 672.1 s | -13.5% | 0.127002 (+0.03%) | same as base |
| TP8, EP8, FlashMLA forward | 470.7 s (469.7 / 472.7 / 469.8) | -39.4% | 0.126487 (-0.38%) | torch reserved max 116.37 GiB |
| TP8, EP8 (r50 layout), old kernel, job `kdatp-prof-20260929d` | 672.6 s | -13.4% | +7.2% (cause UNVERIFIED, study verdict) | torch reserved max 109.7 GiB |

The 1-GPU kernel job `kdatp-dsa1-20260930a` passed all 7 harness gates: TileLang out, lse and dq bitwise equal to the old kernel at width 576, and a FlashMLA relative gap of at most 2.62e-6. The PR unit tests gave 58 passed and 2 failed at TileLang 0.1.9. Both failures are the TileLang forward at 16 heads with a real 64-wide tail, which is not the r52 shape.

## Measurement plan

Compare r52 with r50 at the same rollout ids, and with r49. Start the speed gates at rollout 62, after the first train step compiled the kernels.

| Question | Signal | Gate |
| --- | --- | --- |
| Kernel on | the Megatron argument dump in the trainer log; the model build | `miles_dsa_sparse_attention_forward_backend` reads `flash_mla` in the dump; no `ImportError` at model build |
| Faster per token | `perf/actor_train_tok_per_s`, `perf/actor_train_time` | at least 15% above r50 at the same rollout ids, and at least 30% above r49 (25,300 to 27,900 tok/s). The T2 arms give -30% step time vs EP8 alone (ESTIMATE for live) |
| No recompile in steady steps | TileLang compile lines in the trainer logs (method of `reports/live.md` section 3 of the study) | none after rollout 62 |
| Memory | `memsample-*.log` peak, no OOM | peak below 170 GiB per GPU, and at most 10 GiB above r50 |
| Same learning | `train/train_rollout_logprob_abs_diff`, `train/grad_norm`, `rollout/group_metrics/reward.mean` | within r50's range at the same rollout ids. T2 moved abs_diff by -1.1e-5 and `grad_norm` by +0.66% (EP16) and -0.38% (EP8) |
| Engine path | `perf/update_weights_time` | near r50 (r49: 31 to 37 s) |

Before launch (not done, needs a GPU and a user go): run `tests/fast-gpu/kernels/attention/dsa` and the kdatp DSA harness (`kdatp/dsa`, job `kdatp-dsa1-<stamp>`) on this image, on 1 B200. Gate: all harness gates pass, and the 2 TileLang 0.1.9 failures pass or stay outside the r52 shape.

## Checks done before launch

| Check | Result |
| --- | --- |
| Port | the 4 upstream commits pick without a conflict; the net patch equals the PR net patch line for line; `miles/kernels` and the PR tests equal the PR head |
| New CPU test `tests/fast/test_dsa_sparse_attention_forward_backend.py`, stubfull | 8 passed: default `tilelang`, every model call passes `forward_backend`, each spec builder reads the flag, and with stub kernels on CPU the value selects the kernel and its top-k and head padding. It fails 7 of 8 on the plain PR port |
| CPU tests near the moved code, stubfull | `test_glm5_indexer_query_norm.py`, `test_dsv4_thd.py`: 104 passed |
| Tests, plain stub (`tests/fast/plugins/arena`, `tests/fast/launch_scripts`) | 409 passed, 1 known upstream failure (`test_workplace_backend`) |
| Tests, stubfull, conftests, local Ray (`tests/fast/plugins/arena`, `test_run_arena_harbor.py`, the new test, `test_glm5_indexer_query_norm.py`, `test_dsv4_thd.py`) | 467 passed, 1 known stubfull failure (`test_register_nova_reasoning_parser`) |
| `tests/fast/utils/test_arguments.py` and `test_megatron_cli_flags.py`, stubfull | 335 passed, 2 failed, 5 skipped; the same 2 stubfull failures on the base `da93977ba8` |
| Launcher snapshots (`tests/manual/launch_scripts/test_py_launch_scripts.py -k "run_arena_harbor or run_glm5_3_flash"`) | 6 passed |
| `tests/fast` sweep by entry, plain stub, `--noconftest`, against the base `da93977ba8` | base: 6962 passed, 131 failed, 692 errors, 368 skipped; this branch: the same outcome for every test on both trees. The one new outcome is a collection error of the new test, which needs the stubfull `sglang.srt.server_args` |
| GPU tests | not run here: the test venv has no CUDA, TileLang, Triton or Megatron. `tests/fast/test_kpool_indexer_packed.py` needs Triton and errors at import on this branch and on the base |
| Image `miles-glm53-recon-20261002-flashmla` | `git rev-parse HEAD` = `a515a167b63fcceb262af68b649c729a177734ad`, clean tree, `opentelemetry-api` 1.44.0 = `opentelemetry-sdk` 1.44.0, `flash_mla` 1.0.0+b7643bd imports, TileLang 0.1.14; `glm5`, `glm5_next` and `deepseek_v4` import with the CUDA driver stub. Pushed to us-east-1; the ap-south-1 replica has the same digest |
| r52 argv | the launcher builds `--miles-dsa-sparse-attention-forward-backend flash_mla`; the miles parser resolves `flash_mla` in the venv and in the image. Against the r50 config, the argv differs only in that flag and `arena_sample_summary_dir` |
| `workflow.yaml` | against r50, only `generateName`, `experiment-name`, `trainer-image`, and `miles-config` change; `miles-config` is this `miles-config.yaml` verbatim |

## Timeline

| UTC | Event |
| --- | --- |
| 2026-10-02 04:35Z | trainer image pushed |

## Results

| Metric | Value | Source |
| --- | --- | --- |
| none yet | | |

## Issues

- None yet.

## Follow-ups

- When upstream merges #3605 to #3608, decide if the fork flag stays. Without it, a base image with `flash_mla` turns FlashMLA on for every DSA run.
- `data_pad_size_multiplier` 512 exists to bound the static-shape backward recompiles. With the dynamic backward, a run can test the default (128) for fewer pad tokens.
- Study step 7 ("no zero tail", #3607 `D_tail = 0`) is the next DSA lever on top of this kernel.

## Sources

- miles `arpit-r52-flashmla` `a515a167b6`; upstream `refs/pull/3608/head` `557fb097c2`; earlier port `arpit-dsa-3608` `a60c648e61`.
- Base run files: `arpit-recon-20261001:training-runs/harbor-rl-glm53-flash/guparpit-agentic-debt-v3/`.
- T2 and L2 numbers: jobs `kdatp-prof-20260930f` (FlashMLA image), `kdatp-prof-20260930t` (TileLang-only image) and `kdatp-dsa1-20260930a`; their `results.json` files and the change inventory of 2026-09-30 (`B2`). The study `STUDY.md` on `arpit-glm-53` `6ae5099fca` does not hold these numbers yet.
