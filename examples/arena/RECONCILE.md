# Reconcile record: `arpit-glm-53` onto radixark/miles `main` (2026-09-27)

This file records how branch `arpit-reconcile-upstream` was made. It lists each
dropped commit, each partial keep, and each conflict resolution. The decisions
are in plugin ADR-0018 (`miles_plugins/arena/adr/0018-reconcile-with-upstream-main.md`;
ADR-0016 until 2026-10-01). Section "Refresh 2026-10-01" at the end records
branch `arpit-recon-20261001`: the same branch on a newer upstream `main`, with
the later fork and R3 data-path commits.

## Inputs

| item | value |
| --- | --- |
| fork tip | `origin/arpit-glm-53` = `3028bc358` (local ref `refs/recon/fork-3028bc358`) |
| upstream tip | `upstream/main` = `23d41d711` (2026-09-26, "Drop the empty-batch timeout only multi-LoRA v1 ever raised (#3723)") |
| merge base | `2799fe386` (2026-08-31) |
| fork side | 171 commits: 167 non-merge commits and 4 merge commits |
| upstream side | 750 commits |
| result | 139 kept fork commits, 1 restore commit, and the new commits in "New commits after the rebase", on top of `23d41d711` |

## Method

1. `git rebase -i --onto upstream/main 2799fe386` with a scripted todo list. The
   todo list has the order of a dry run with `git merge-tree`: the PR branch
   commits first, then the first-parent fork commits.
2. `drop` for the 28 PR-copy commits. `pick` plus `break` for the three partial
   keeps. `exec` after `f01b8201f` for the restore commit.
3. Each conflict was resolved by hand. The rebase keeps the original author and
   author date of each commit. The committer is Arpit Gupta
   <arpitg1991@gmail.com>.
4. `git cherry -v HEAD refs/recon/fork-3028bc358` after the rebase: 131 fork
   commits have the same patch-id on the new branch. The other 36 are the 28
   drops, the 3 partial keeps, the 4 conflict picks, and `8989cfb49` (a clean
   merge whose context changed, so its patch-id changed).

## Dropped commits

### The fork copy of PR #2786 (28 commits)

Upstream merged PR #2786 as the squash commit `cc76e2391` on 2026-09-24. The
fork carried the PR branch head `dbbd610e7` (2026-08-29). The core hunks of the
PR copy (`megatron_to_hf/__init__.py`, `train_data_conversion.py`,
`generate_endpoint_utils.py`, `sglang_rollout.py`, `arguments.py`,
`mbridge/__init__.py`, and `hf_config.py`, now `hf_utils/config.py`) are
identical to the hunks of `cc76e2391`. The model files of `cc76e2391` are the
final PR version (see "Upstream changes that alter a run" below). Each file that a
dropped commit touches is in `cc76e2391`, or the PR itself reverted the change
before its head (`tests/glm5_next/*`, `hf_compat.py`, `sglang_engine.py`, the
8-layer CI test), with one exception:
`scripts/models/glm5.3-flash-8layer.py` (5 lines, the 8-layer CI cut). The
upstream squash leaves it out. No arena config, script, or test uses it, so the
rebase does not restore it.

Cover for each row: upstream `cc76e2391` (radixark/miles#2786).

| # | fork commit | subject |
| --- | --- | --- |
| 1 | `5e986f1d9` | feat(glm5_next): GLM-5.3-Flash training plugin (KDA + kpool DSA + mHC spec) |
| 2 | `f5acb61d2` | feat(glm5_next): mbridge bridge for GLM-5.3-Flash |
| 3 | `b30002aa7` | feat(glm5_next): raw megatron-to-hf weight-sync converter |
| 4 | `f9ec413d1` | feat(glm5_next): GLM-5.3-Flash model scripts and DAPO launcher |
| 5 | `c235457a1` | test(glm5_next): scratch KDA-wrapper and bridge-name audits |
| 7 | `3d134203e` | test(glm5_next): bf16-floor tolerance for the KDA gate gradient check |
| 8 | `beb136ac6` | glm5.3: disable rope fusion (qk_rope_head_dim=0, MLA fusion requires yarn) |
| 9 | `2b9950ed1` | glm5_next: register a transformers compat config for model_type glm5_next |
| 10 | `823102057` | glm5_next bridge: inert rope_theta default when qk_rope_head_dim is 0 |
| 11 | `6e6653d2d` | glm5_next bridge: nest language-tower HF names under model.language_model |
| 12 | `a7ebc7e56` | sglang engine: skip the no-op host bracket strip |
| 13 | `3067f473f` | glm5.3 launcher: tilelang DSA backends + bf16 kv cache |
| 14 | `0dd04cc2a` | R3: fail fast on an all-zero routed_experts rollout payload |
| 15 | `ea55560a6` | glm5.3 mHC: match the reference proj-RMS convention (eps inside, rms_norm_eps) |
| 16 | `cee16d9d0` | glm5.3 R3 indexer replay: stream-count guard + launcher toggle |
| 17 | `4ac3f4b7f` | R3 replay: fail fast at conversion when rollout samples lack replay payloads |
| 18 | `fb4dc718f` | glm5.3 launcher: 6x4 full-model topology (TP8/PP3/EP8, 15 layers per stage) |
| 19 | `e8ff73a3e` | glm5.3 launcher: 16x4 full-model topology (TP8/PP4/EP16, DP2) |
| 21 | `04a287790` | glm5_next: triton kpool indexer selection hot path |
| 22 | `70d5e33ea` | Add GLM-5.3-Flash 8layer e2e CI test (disabled until HF cut + CI image land) |
| 23 | `cf0d3b38b` | Clean glm5_next code: fla fused KDA gate, idiomatic hf_compat, strip comments |
| 24 | `b5f2863ff` | Replace glm5_next hf_compat with a config alias entry |
| 25 | `a227e23ac` | Drop glm5_next unit test files, keep only the e2e CI test |
| 26 | `39f0c59ef` | Switch GLM-5.3-Flash CI to a 4layer cut (KDA+dense, DSA+MoE, KDA+MoE, DSA+MoE) |
| 27 | `1cd14c000` | ci: set CUDA_DEVICE_MAX_CONNECTIONS=1 for the TP2 4layer conversion |
| 28 | `eee9819da` | glm5_next: rope_theta is absent from the checkpoint (NoPE); use the inert 10000 default |
| 29 | `75ace38a1` | Resolve DSA indexer fields via the text config (composite hf configs) |
| 30 | `dbbd610e7` | test(glm5.3): pin the kpool indexer to per-sequence pools in a packed batch |

The number is the position of the commit in the rebase order: the 30 PR-branch commits first, then the first-parent fork commits, oldest first. The appendix uses the same numbers.

### Merge commits (4)

A linear rebase removes merge commits.

| merge | content | action |
| --- | --- | --- |
| `58a9f94d7` | merges upstream `8b642a6d4` into the PR branch | `8b642a6d4` is an ancestor of `2799fe386`; nothing to keep |
| `4b61e01d3` | merges upstream `071fd2a6f` into the PR branch | `071fd2a6f` is an ancestor of `2799fe386`; nothing to keep |
| `dc9ff3983` | joins two lines of the PR branch | no own content |
| `38921ae45` | merges the PR head into the fork | the code part is in `cc76e2391`. The merge alone added three files; see "Restored merge-only content" |

## Partial keeps

| fork commit | rebased | kept | removed, with the upstream cover |
| --- | --- | --- | --- |
| `bd8843d29` glm5_next: production cleanup sweep | `8ca11e758` | the `miles/utils/reloadable_process_group.py` hunk (class docstring, shorter `_adopt_inner_backends` docstring, comment block removed) | the other 13 files: in `cc76e2391`, or deleted by it |
| `8ceca603d` feat(arena): asyncio training driver with checkpoint-sidecar W&B resume | `9842f5cb6` | the driver, `checkpoint_extras.py`, ADR-0005, the RUNLOG hunk, the index row | the `init_wandb_primary` hunk of `wandb_utils.py`: upstream #3030 (`8defbefa6`) passes `id` and `resume="allow"` when `args.wandb_run_id is not None`. The hunk merges without a conflict, so the rebase removed it by hand. Two copies ran the same step twice |
| `f687e598e` fix(rollout): guard anthropic sglang imports absent on glm53next base | `e2c5a657a` | the 66-line hunk of `examples/arena/harbor-rl-glm53-flash/RUNLOG.md` | the guards in `anthropic_adapter.py` and `sessions.py`: upstream #3114 (`d2fc97ce5`) guards the same imports and also the protocol import |

Each partial keep has a "Reconcile note" paragraph at the end of its commit
message.

`b90c21ed4` (reloadable pg: drop python collective overrides) is a PR-branch
commit that upstream did not take. The rebase keeps it whole as `3793358db`.

## Conflict resolutions

| fork commit | rebased | file | resolution |
| --- | --- | --- | --- |
| `6bec84d14` feat(arena): add arena plugin skeleton | `8bd2593a2` | `setup.py` | keep the upstream `e2b` and `modal` extras and version 0.1.1; add the `arena` extra after them |
| `000746bea` fix(megatron): tolerate ENOTSUPP when copying HF aux files | `8f37e9b05` | `miles/backends/megatron_utils/hf_export.py` | take the upstream file. Upstream #3198 (`3cdaad102`) moved the aux-file copy to `SnapshotPublisher.write_model` in `miles/backends/training_utils/weight_update/snapshot_publisher.py`, which still used `shutil.copy2`. Apply the same `shutil.copyfile` and per-file `OSError` warning there. The fix is still necessary: upstream `write_checkpoint_dir` deletes the whole export directory when a write raises |
| `aca2e9520` feat(arena): truncated-turn shift rule | `c98c3d129` | `miles/backends/megatron_utils/model.py` | keep the upstream `get_batch` keys (`loss_weights`, `target_tokens`, `sample_indices`, `*sampling_mask_keys`); add `truncated_turn_shifted` after `rollout_mask_sums` |
| `b04f433e7` refactor(arena): train every verified episode | `4a7b41cc9` | `miles/backends/megatron_utils/model.py` | take the upstream file. The commit removes `truncated_turn_shifted`, so the net fork change to this file is zero, as on `arpit-glm-53` |

`8989cfb49` (`57f08ca59`) merged without a conflict into the upstream
`debug_data.py`. The rebased change is the same: `sample_index` is `Int64`,
and a failed dashboard dump logs a warning and does not stop the run.

## Restored merge-only content

`6c3ac0dd7` docs(arena): restore the GLM-5.3 RUNLOG, ADR-0008, and index row.
It restores three files from `38921ae45` without change, right after
`cd3f405e1` (the rebased `f01b8201f`):

- `examples/arena/harbor-rl-glm53-flash/RUNLOG.md` (the first 111 lines)
- `miles_plugins/arena/adr/0008-glm5-next-support-via-pr2786-merge.md`
- `miles_plugins/arena/adr/README.md` (the ADR-0008 index row)

## New commits after the rebase

| commit | change |
| --- | --- |
| `a18a23903` | ADR-0018 (numbered 0016 then) and the amendment links in ADR-0005, ADR-0006, ADR-0008, ADR-0011 |
| `bc0c843c6` | `train_async_arena.py` rebuilt on the upstream `train_async.py`. The old driver imported `create_rollout_manager`, `MainProcessIdentity`, and `ft_utils.control_server`, which upstream removed. The ADR-0005 check stays at 0 deleted lines. The final drain and the heartbeat removal are one disposer callback, `_finish_arena` |
| `f32dd8838` | `scripts/run_arena_harbor.py` calls `args.create_backend()`; upstream #2432 removed `U.exec_command_cpu` and `U.execute_train`. The snapshot adds `--deploy-component all` |
| `e8d4ae9a1` | the gym weight versions move to `Sample.metadata["arena_weight_versions"]`; `Sample.weight_versions` stays empty. Upstream #1891 made that field a list of span objects |
| `aed3522de` | the advantage-scale tests set a one-rank parallel state (upstream #3125) |
| `7466bc9cc` | the `run_glm5_3_flash` launcher snapshot regenerated for the merged #2786 script |
| `9a5b6216d` | the plugin RUNLOG entry |
| `77cd880d4` | this file |
| `c5b138782` | `patches/fla_kda_next_power_of_2.py` finds fla with `importlib.util.find_spec` and accepts the upstream form of the patch (see "Test images") |
| `4d6c39a7d` | `patches/fla_conv_int64_offsets.py`: the int64 lines of fla PR #1082 for `causal_conv1d` (see "GPU validation") |
| `4316e4957` | `miles/utils/ray_utils.py` finds the head node with `ray.nodes()` (see "GPU validation") |
| this commit | the test images, the GPU validation, and the open items in this file; the image lineage rows; the plugin RUNLOG entry |

## Upstream changes that alter a run on this branch

| area | change | effect on an r44, r45, or r46 config |
| --- | --- | --- |
| GLM-5.3 DSA indexer | the final #2786 applies RMSNorm to the indexer query. The fork copy fed the raw query; SGLang `9a26e749` feeds the normalized query | the live runs on `arpit-glm-53` have this train/rollout mismatch in all 11 DSA layers. The fix changes the top-k selection and the log-probs |
| GLM-5.3 MLP | `scripts/models/glm5.3-flash.py` adds `--activation-func-clamp-value 10`. The checkpoint sets `swiglu_limit` 10, and SGLang clamps | a second train/rollout mismatch goes away |
| GLM-5.3 mHC | the mean output contraction comes from a Megatron spec, not a monkeypatch | no checkpoint format change is expected. A GPU load of an r45 checkpoint must prove it |
| fault tolerance | with `use_fault_tolerance`, upstream turns on the mini fault-tolerance controller on `api_server_port` 18080. `--control-server-port` is gone | new auto-heal of failed engines. No r44, r45, or r46 config sets `mini_ft_controller_enable`, so a relaunch from these files turns the controller on. Keep the key unset. On this branch the controller is the only code that restarts a failed engine: it does the work of the fork `recover_updatable_engines`. A YAML `false` does nothing, because the launcher drops a false value (corrected 2026-10-01; see "Fault tolerance against r48" in the refresh). In the train-only GPU jobs the controller started and polled an empty cell list |
| event logger | `save_debug_event_data` defaults to `<save>/events` (#2505), and no flag turns it off. The event logger is then on in every process. `update_weights` runs `check_weights(action="checksum")` on every engine at each weight update (`miles/ray/placement_group.py`), and each save copies the whole events dir into the checkpoint | with `update_weights_interval: 1`, one checksum pass on each engine per rollout, and a copy that grows at each save. The GPU jobs of this file have no engines, so they do not measure the checksum time. The Megatron DCP format, the data-source state path, and the single-policy dir layout do not change |
| metrics | `train_rollout_logprob_abs_diff` and `train_rollout_kl` come from trainer-scored log-probs (#3655) | values before and after the rebase are not directly comparable |
| weight-version metrics | the upstream `weight_version/*` metrics read span objects | arena samples carry no spans, so these metrics are absent. `rollout/off_policy_round/*` does not change |
| TIS, PPO, R3 | the math does not change. R3 still sets `enable_return_routed_experts` and reads `rollout_routed_experts` | low. The upstream GLM launcher forces `--sglang-moe-runner-backend triton` for R3; the arena configs keep `auto`, which r16 validated |
| dependencies | the upstream Dockerfile moves SGLang v0.5.18 to v0.5.20, TileLang 0.1.8 to 0.1.14, Megatron-Bridge `7f0fb345` to `8cd3466d`, and adds a patch for fla 0.5.2. Upstream starts each SGLang engine with `--gated-launch-port` (#2096), and since #3031 the launch stops when SGLang does not serve it | SGLang `9a26e749` of `glm53next-upstream-20260902` does not serve `--gated-launch-port`, so this branch needs a new base. The test images use `recon-miles-base-20260927a` (see "Test images") |
| fla 0.5.2 | `causal_conv1d` computes token offsets in int32 (fla PR #1062). fla fixed it after 0.5.2 in PR #1082, with no release yet | a GLM-5.3 row longer than 87,381 tokens stops the train step with an illegal memory access. `4d6c39a7d` patches the kernel at image build |
| head pinning | `compute_ray_pin_head_options()` runs in the RayWorkerManager actor, not in the driver. It read the head node from the dashboard at 127.0.0.1:8265 | with `pin_rollout_manager_to_head: true` (r44, r45, r46), the launch stopped with `ServerUnavailable` when Ray placed that actor on a worker node. `4316e4957` reads the GCS node table |

## Validation (2026-09-27)

| check | result |
| --- | --- |
| arena fast tests: `PYTHONPATH=$PWD:/tmp/sglang-stub /tmp/miles-test-venv/bin/python -m pytest tests/fast/plugins/arena -q -p no:cacheprovider --noconftest` | 325 passed |
| launcher tests: `tests/fast/launch_scripts` in the same venv | 49 passed, 1 failed. `test_workplace_backend.py::test_workplace_launch_uses_the_configured_backend` also fails on upstream `main` in the same venv |
| launcher snapshots: `tests/manual/launch_scripts/test_py_launch_scripts.py -k "run_arena_harbor or run_glm5_3_flash"` | 6 passed. The whole manual suite has no failure that upstream `main` does not also have |
| `tests/fast` sweep in the venv, by directory, against upstream `main` | 6545 passed on this branch, 6205 on upstream. 341 tests pass only on this branch; they are fork tests. One test changes from pass to fail: `tests/fast/doc/test_sync_example_docs.py::test_docs_examples_matches_the_readmes`, because `examples/README.md` does not list `examples/arena`. The fork tip fails the same test. All other outcomes are identical |
| image `arena-slime-dev:miles-glm53-r15-20260927a` (SGLang `9a26e749`, Megatron `e8f57451`), tree mounted on `PYTHONPATH` | the `examples/arena/Dockerfile` import smoke passes, including `miles_plugins.arena.train_async_arena`. `run_arena_harbor.py --help` passes. The arena, launcher, loss, `test_train_data_conversion`, `test_samples`, `test_loss_mask_qwen3_5`, and `test_types` tests: 574 passed, 13 failed; upstream `main` has the same 13 failures in that image (12 loss snapshots that clone an artifact repository, and `test_workplace_backend`). With network access, `test_loss_snapshot.py --compare` passes (12) |
| fast tests at `4316e4957`, same venv | arena 325 passed, `test_run_arena_harbor.py` 6 passed, `tests/fast/utils/test_ray_utils.py` 4 passed. In the recon test image, `test_ray_utils.py` and `tests/fast/utils/workers/test_ray_worker_manager.py`: 135 passed |
| argv parse: `_build_train_args` of the launcher, the model args of `scripts/models/glm5.3-flash.py`, and `--deploy-component all`, then `miles.utils.arguments.parse_args()` in the r15 image | all five configs parse: `r44/miles-config.yaml`, `r44/miles-config-r0.yaml`, `r45/miles-config.yaml`, `r46/miles-config.yaml`, `r46/miles-config-r0.yaml`. Against the fork tip, the argv adds `--activation-func-clamp-value 10` and `--deploy-component all` only. The resolved values change for `activation_func_clamp_value` (None to 10.0), `mini_ft_controller_enable` (False to True), `save_debug_event_data` (None to `<save>/events`), `session_sample_picker_path` (renamed default), and `rollout_health_check_first_wait` (0 to 0.0) |

## Test images (2026-09-27)

TEST ONLY. NEVER use these tags for a live run.

| tag | tree | ap-south-1 digest | result |
| --- | --- | --- | --- |
| base `recon-miles-base-20260927a` | upstream `docker/build.py --variant cu13-x86` at `23d41d711` | `sha256:f83c7828a756` | the base of the three test tags |
| `miles-glm53-recon-test-20260927a` | `c5b138782` | `sha256:8f5fe3dc5715` | T1 passes. The warm-up train step fails (fla int32 offsets) |
| `miles-glm53-recon-test-20260927b` | `4d6c39a7d` | `sha256:429d4ebffec4` | the warm-up passes. T2 fails at the launch (head pinning) |
| `miles-glm53-recon-test-20260927c` | `4316e4957` | `sha256:2cd2c030c567` | T2 passes |

The base comes from the upstream recipe only:
`python3 docker/build.py --variant cu13-x86 --image-tag custom --custom-tag recon-miles-base-20260927a --build-arg SGLANG_COMMIT=571212b636baca45e10fa3b4da11a289123f3235 --build-arg MEGATRON_COMMIT=f148a32b4385b758b66a77c9c3ad1641f1295d4b --build-arg MILES_COMMIT=23d41d711f3b80544fda655898ed4f051ed644fe`.
Each test tag is `docker build -f examples/arena/Dockerfile --build-arg MILES_BASE_IMAGE=<base>` on a clean clone of this branch.

| part | `glm53next-upstream-20260902` (live runs) | `recon-miles-base-20260927a` |
| --- | --- | --- |
| SGLang | `9a26e749` | `sglang-miles` `571212b6` (0.5.21.dev63, on `lmsysorg/sglang:v0.5.20`) |
| Megatron-LM | `e8f57451` | `miles-main` `f148a32b` (the merge commit of PR #89) |
| fla | 0.4.2 | 0.5.2, with the upstream KDA patch and `4d6c39a7d` |
| Megatron-Bridge | `7f0fb345` | `8cd3466d` |
| TileLang | 0.1.9 | 0.1.14 |
| torch / TE | 2.13.0+cu130 / 2.17.0 | the same |

## GPU validation (2026-09-27, prod-bom `recon-*` jobs)

The jobs use the queue, the priorities, and the excluded nodes of the live
runs. They write only under `/mnt/scratch-s3files-rw/guparpit/recon/`. They
read the r45 `iter_0000039` DCP (links to r43), the r43 `hf/rollout_39`
export, and the kdatp T2 rows. No job saves a checkpoint. No job uses W&B.
The harness is outside the repo: ConfigMap `recon-harness-20260927a` (the
kdatp harness pattern). The reference image is `miles-glm53-r15-20260927a`
(`bc31f88ac`, the same code as the fork tip).

T1, one node (TP 8 with SP, EP 8, PP 1; jobs `recon-t1-{ref,new}-20260927b`,
image a). Each image loads the r45 `iter_0000039` DCP with its own
`load_checkpoint`, runs its own `forward_only` log-prob pass on 4 real r45
rows (3,509 to 6,538 tokens), and runs its own HF weight iterator.

| check | ref (r15) | new (image a) |
| --- | --- | --- |
| load | iteration 39, 528 s, 81.1 GiB and 43,490,708,478 parameters per GPU | the same values, 509 s |
| HF gather (weight sync and HF export) | 37,534 tensors, SHA-256 equal to r43 `hf/rollout_39` | 37,534 tensors, SHA-256 equal to r43 `hf/rollout_39` and to ref |

Log-prob differences on the 6,578 trainable tokens. The noise floor is
pass 1 (one row per micro-batch) against pass 2 (the 4 rows packed in one
micro-batch) in the same image:

| comparison | mean abs | p99 | max |
| --- | --- | --- | --- |
| ref pass 1 against ref pass 2 (floor) | 0.0458 | 0.491 | 1.53 |
| new pass 1 against new pass 2 (floor) | 0.0448 | 0.466 | 1.29 |
| new, clamp off, against ref | 0.0373 | 0.366 | 1.03 |
| new, clamp 10 (default), against ref | 0.0476 | 0.486 | 1.51 |
| ref against SGLang rollout log-probs | 0.0506 | 0.534 | 1.84 |
| new, clamp 10, against SGLang rollout log-probs | 0.0472 | 0.517 | 1.69 |

The SGLang log-probs come from the r43 policy at or before iteration 39, so
they are an approximate reference. The clamp (`--activation-func-clamp-value
10`, the SGLang `swiglu_limit`) moves the trainer 7% closer to them.

Logits over all 18,829 positions and the full vocabulary: new (clamp off)
against ref gives a max abs difference of 25.9, a max of (max abs
difference / max abs logit) of 0.557, a worst-row relative L2 error of
0.192, and a top-1 agreement of 92.1%. The same-image floor has a
relative L2 error of 0.22 and a max abs difference of 26.7. In 3 of the 4
rows the first 6 to 11 positions are bitwise equal; the difference grows
with the position. Gate (the kdatp rule): relative L2 error at most 2 x floor
+ 2**-8, and mean log-prob difference at most 2 x floor + 1e-3. Both
new runs pass. Images b and c change the `causal_conv1d` offsets (the output
is bitwise equal below 87,381 tokens, see the fla check) and the head
lookup only, so the T1 values also apply to them.

fla check (`recon-convchk-20260927a`, one GPU of image a): ShortConvolution
with 24,576 channels, forward and backward. The installed fla 0.5.2 passes
at 80,000 tokens and fails at 131,070 tokens (illegal memory access). With
`fla_conv_int64_offsets.py` it passes at 131,070 tokens (relative L2 error
1.7e-3 against an fp32 reference, for y, dx, and dw). At 80,000 tokens the
patched output is bitwise equal to the installed output.

Warm-up (`recon-warm-20260927{a,b}`, one node): the 5-layer slice trains one
step on the T2 rows. Image a stops in the first log-prob pass
(`causal_conv1d_fwd_kernel`, illegal memory access). Image b passes in
46 min and writes the kernel cache for T2.

T2, eight nodes (the r45 layout: TP 8 with SP, PP 4, EP 16, DP 2; R3 and
TIS on; job `recon-t2-20260927c`, image c): train only, rollouts 40 to 42 of
the kdatp T2 rows (314 rows, 22.3M tokens, up to 131,070 tokens per row),
from the r45 `iter_0000039` DCP. The reference is the kdatp T2 `baseline`
arm (`kdatp-t2-20260927e`, the r15 code with the KDA split off). Image b
stopped at the launch (head pinning); image c has the fix.

| rollout | metric | kdatp baseline | recon |
| --- | --- | --- | --- |
| 40 | grad_norm | 0.1274 | 0.1263 |
| 40 | loss | -3.6e-6 | -4.8e-5 |
| 40 | ppo_kl | -8.5e-6 | 1.8e-5 |
| 40 | train_rollout_logprob_abs_diff | 0.02721 | 0.02576 |
| 40 | train_rollout_kl | 0.00212 | 0.00197 |
| 40 | step time (first step, kernel compile) | 1,687 s | 2,097 s |
| 41 | grad_norm | 0.1233 | 0.1244 |
| 41 | loss | -5.30e-4 | -5.45e-4 |
| 41 | step time | 1,278 s | 1,306 s |
| 41 | log-prob time / actor train time | 219 s / 989 s | 265 s / 1,015 s |
| 42 | grad_norm | 0.1215 | 0.1358 |
| 42 | loss | -1.86e-3 | -1.90e-3 |
| 42 | step time | 1,284 s | 1,315 s |
| 42 | log-prob time / actor train time | 229 s / 993 s | 270 s / 1,019 s |
| 40, 41 | peak allocated GiB, PP stages 0 / 1 / 2 / 3 | 128.8 / 129.7 / 124.7 / 124.9 | 136.8 / 137.7 / 132.7 / 132.9 |

At one optimizer step per rollout, `train_rollout_logprob_abs_diff` of both
trees compares the trainer log-probs of the same weights with the SGLang
log-probs, so #3655 does not change its value here. The recon peak window
starts at the log-prob pass (hook `recon_hooks.py`); the kdatp window starts
at the `train` call. The +8.0 GiB is the same on each stage; its source is
not isolated.

## Open items

1. The owner accepts or changes ADR-0018 decision 4 (the weight versions in
   `Sample.metadata`, no spans).
2. Test the engine path before a relaunch from r44, r45, or r46: SGLang
   `sglang-miles` `571212b6` with the r45 engine flags, one weight update, the
   event-logger checksum on each engine, and the mini fault-tolerance
   controller. The GPU jobs of this file have no engines.
3. The steady log-prob pass is 18% to 21% slower (219 to 229 s against 265 to
   270 s), and the peak allocated memory is 8.0 GiB higher on each stage.
   Isolate the source before a long run.
4. Move the base to a fla release that has PR #1082. Then
   `fla_conv_int64_offsets.py` reports "already applied" and can go.
5. Offer `4316e4957` (head lookup from the GCS) to upstream.

## Appendix: kept commits, fork to rebased

| # | fork commit | rebased commit | note | subject |
| --- | --- | --- | --- | --- |
| 6 | `b90c21ed4` | `3793358db` |  | reloadable pg: drop python collective overrides (torch 2.13 PyWorkHolder UAF) |
| 20 | `bd8843d29` | `8ca11e758` | partial | glm5_next: production cleanup sweep |
| 31 | `6bec84d14` | `8bd2593a2` | conflict resolved | feat(arena): add arena plugin skeleton and torch-free support modules |
| 32 | `667deaf3f` | `7ed69d84d` |  | feat(utils): add qwen3_5 multi-turn loss-mask type |
| 33 | `ac493db98` | `2986bcd8b` |  | feat(arena): port NATS wire format, controllers, data source, rewards |
| 34 | `565cadbe4` | `77835089c` |  | feat(arena): port the NATS rollout hot path (generate_rollout) |
| 35 | `fac599e66` | `0c73fa9f0` |  | feat(arena): port the hosted eval path (coordinator, trigger, rollout) |
| 36 | `8ceca603d` | `9842f5cb6` | partial | feat(arena): asyncio training driver with checkpoint-sidecar W&B resume |
| 37 | `d28f0acf0` | `f5b8082fe` |  | feat(arena): run_arena_harbor launcher, 27B smoke config, snapshots |
| 38 | `8685b1064` | `cf2785b61` |  | test(arena): port the AGISlime CPU test suite (145 tests) |
| 39 | `bbfaf52a2` | `a7def3bb6` |  | test(arena): guard group identity; record verification fix-ups |
| 40 | `0b94255ad` | `0026e72c4` |  | feat(arena): harbor-rl-27b financeagent smoke example and trainer image |
| 41 | `f01b8201f` | `cd3f405e1` |  | feat(arena): harbor-rl-27b-snorkel plain-stack example |
| 42 | `f687e598e` | `e2c5a657a` | partial | fix(rollout): guard anthropic sglang imports absent on glm53next base |
| 43 | `57b4cdb44` | `744d5de83` |  | feat(arena): GLM-5.3-Flash example scaffolding and EFA image layer |
| 44 | `3f015ee2b` | `e06a14fe6` |  | feat(arena): glm53 r1 launch manifests (NATS, 128 gym workers, TAS) |
| 45 | `adc8772f1` | `9775787d4` |  | fix(arena): glm53 run-1 live changes - 32k/131k caps, TAS exclusions |
| 46 | `93d480886` | `fa3bf1215` |  | fix(arena): patch fla KDA kernel for triton 3.7.1; relaunch on image b |
| 47 | `fa2d48d92` | `d789ccec4` |  | feat(arena): --arena-mask-clipped-final-turn salvage flag |
| 48 | `000746bea` | `8f37e9b05` | conflict resolved | fix(megatron): tolerate ENOTSUPP when copying HF aux files in hf_export |
| 49 | `986d57758` | `eeb0ce799` |  | fix(arena): glm53 r2 relaunch fix set - Ray monitor off, FT on, memprobe |
| 50 | `28fe99513` | `72b00c421` |  | fix(arena): glm53 r3 - kernel caches on local disk, TAS exclusions |
| 51 | `00a894764` | `1f3ae812f` |  | feat(arena): --arena-keep-timeout-trajectories and removal-reason log |
| 52 | `8c77bb63e` | `32e522ba1` |  | fix(arena): glm53 r4 - optimizer CPU offload after step-1 CUDA OOM |
| 53 | `a346bc7e0` | `0599c75e9` |  | feat(arena): glm53 r5-r7 - keep-timeout on, effort gym image, 16 engines |
| 54 | `9346bdf02` | `f04b7236f` |  | docs(arena): RUNLOG final state, r7 in progress, examples index |
| 55 | `412877bcf` | `30b28e7ae` |  | test(launch): record the run_glm5_3_flash launcher snapshot |
| 56 | `6df8edc55` | `19dff9836` |  | docs(arena): glm53 r7 read-out at the 5-step gate |
| 57 | `88f0572f1` | `18450d06c` |  | docs(arena): glm53 truncation RCA - wall clock, not length or context; fix r8 note |
| 58 | `d879b5a91` | `05cd1274b` |  | feat(arena): glm53 r8 - keep context-error trajectories, 8x agent timeouts |
| 59 | `777c3a970` | `098c99c2b` |  | fix(arena): bound NATS results stream and re-publish in-flight tasks on reconnect |
| 60 | `684cc8bcc` | `7305e0f3b` |  | chore(glm53-flash): add r9 manifests and experiment backlog |
| 61 | `3c8c1deef` | `7b55bc5d8` |  | feat(glm53-rl): add r10 — double engines (16->32) + per-turn cap 8192 |
| 62 | `f56def206` | `36735cc3f` |  | fix(glm53f-r10): pull preseed probe+sidecar from ap-south-1 |
| 63 | `1238dd44d` | `f9d7d8ab0` |  | feat(nats-rollout): make publisher in-flight multiplier configurable |
| 64 | `b209d9b4a` | `fbd8c448f` |  | feat(glm53f-r11): 32 engines, 32k cap, inflight x4, save_interval 5 |
| 65 | `25ddb2146` | `4c91e1e6a` |  | chore(glm53f-r11): pin trainer image miles-glm53-20260907a |
| 66 | `436a17281` | `1acee8be9` |  | fix(glm53f-r11): pull gym + preseed images from arena-slime-dev |
| 67 | `34db7fc6e` | `206ea53a5` |  | feat(nats_arena): train on multi-segment episodes behind --arena-train-segments |
| 68 | `a4ec8b908` | `4cd8aedf9` |  | fix(nats_arena): count episode telemetry across every segment |
| 69 | `d28503cf8` | `03fbf654e` |  | fix(nats_arena): pad all-mode rows to the DP alignment; use upstream sample metrics |
| 70 | `941e6e3ef` | `014f5e91c` |  | docs(arena): add the r12 manifests and record the launch |
| 71 | `50835f079` | `08fe2a37d` |  | feat(arena): add harbor-rl-glm53-flash r13 launch files |
| 72 | `02f0c75c8` | `85eb91acb` |  | docs(arena): pin AREnATasks 0d98f0a in r13 BUILD.md |
| 73 | `43d695f6b` | `b9bdd15b4` |  | feat(glm53): add r14 recipe, r13 with lr 1.5e-6 |
| 74 | `ff19a83f2` | `db7302164` |  | feat(arena-glm53): r15 = r13 + upstream TIS recipe (eps 0.2/0.28, tis-clip 2.0) |
| 75 | `3903e8f92` | `ab6d3b1c1` |  | docs(arena-adr): ADR-0012 R3 routing replay over NATS, investigation and plan |
| 76 | `9508f3b52` | `19600a566` |  | docs(arena-adr): fix ADR-0012 index row format |
| 77 | `a9951bb59` | `f1ed35132` |  | feat(nats_arena): R3 rollout routing replay over NATS (ADR-0012) |
| 78 | `805d7b382` | `ad7912f33` |  | chore(arena-glm53): finalize r16 manifests: R3 images, shared ARENA_ROUTING_DIR |
| 79 | `735dd3e2a` | `ab6ac1133` |  | chore(arena-glm53): r16 smoke manifests (5 nodes, GBS 64, 3 rollouts) |
| 80 | `61bce8b6a` | `a2a6ec350` |  | chore(arena-glm53): drop r16 smoke manifests; r16 runs at the r15 shape |
| 81 | `ee190d8d5` | `9b44f8a90` |  | chore(arena-glm53): r16 exclude two TAS-mispinned nodes |
| 82 | `70337a951` | `e093b3763` |  | fix(nats_arena): pad routing rows with -1, not zeros (r16 step-0 crash) |
| 83 | `969931b9e` | `7edefbe71` |  | chore(arena-glm53): r16 trainer image 20260909b (-1 pad routing) |
| 84 | `3fa14dd3f` | `e3a4a2495` |  | chore(arena-glm53): r16 header names image 20260909b |
| 85 | `f734bb1ff` | `cc9637956` |  | chore(r16): exclude NVLink-faulted node i-0f2f7d13b4d31644d |
| 86 | `589dc62fb` | `8400cf1df` |  | docs(arena): r15/r16 RUNLOG entry and ADR-0012 outcome |
| 87 | `fc9847ab7` | `c0c68e51a` |  | feat(r17): 16+24 node split, rollout logprobs, GPT-5.6 curriculum |
| 88 | `bd5b1a010` | `034ac0353` |  | chore(r17): exclude two TAS-mispinned nodes |
| 89 | `ed45c7eaa` | `f340037db` |  | docs(r17): record apply order; sglang Service is mandatory before the gym |
| 90 | `ece3378e7` | `12c4a3dd0` |  | feat(harbor-rl-glm53-flash): r18 TIS back on, fresh run, Argo miles-deployer launch |
| 91 | `29825a497` | `cfc3d96c8` |  | docs(harbor-rl-glm53-flash): r18 Argo attempt findings, live manual path, node exclusion |
| 92 | `53c4db675` | `61f8c6cc7` |  | feat(arena-data-source): shuffle after first epoch flag |
| 93 | `5ed4d072f` | `c1a1bd302` |  | Revert "feat(arena-data-source): shuffle after first epoch flag" |
| 94 | `71eff6f9f` | `754b656b8` |  | feat(nats-arena): log per-group reward variance |
| 95 | `09f477834` | `0ff39b1b0` |  | feat(arena-data-source): skip prompts above reward threshold after epoch 0 |
| 96 | `33423298a` | `c728f5056` |  | feat(harbor-rl-glm53-flash): r19 skip1000 curriculum + reward-gated prompt skip |
| 97 | `8530442f8` | `9d6681aff` |  | docs(harbor-rl-glm53-flash): r18 resume workflow |
| 98 | `6cd81d714` | `c288cd029` |  | docs(harbor-rl-glm53-flash): record r18 resume admission and first rollout |
| 99 | `37e428e8e` | `c88147d7a` |  | feat(harbor-rl-glm53-flash): add r20 and r21 configs with partial rewards |
| 100 | `f025b3ec6` | `a888be711` |  | docs(harbor-rl-glm53-flash): point r20/r21 at guparpit-miles-deployer-v2 |
| 101 | `c6869bc08` | `acef86c7d` |  | docs(harbor-rl-glm53-flash): record r19 retire and r20/r21 submission |
| 102 | `5e2370032` | `47ab24e35` |  | docs(harbor-rl-glm53-flash): record r20/r21 admission and first rollouts |
| 103 | `7418166ba` | `4d8439089` |  | docs(harbor-rl-glm53-flash): fix r20/r21 admission timestamps |
| 104 | `db036926c` | `af9acff7d` |  | docs(harbor-rl-glm53): r22 run on the robust-20260913 variant |
| 105 | `1c911d8f0` | `aee2fbf96` |  | docs(harbor-rl-glm53): gym priorityClassName high; template v3 |
| 106 | `79c23905c` | `eab32795d` |  | docs(harbor-rl-glm53): NATS restart needs a gym restart; 2026-09-14 recovery log |
| 107 | `8d1887197` | `670e9f904` |  | feat(harbor-rl-glm53): r23/r24 without epoch-2 prompt skip; retire r20-r22 |
| 108 | `c94c98f15` | `ef31749ae` |  | feat(arena): per-token advantage_scale for continued clipped turns |
| 109 | `e247e5e80` | `cebd847f8` |  | feat(harbor-rl-glm53): r25 run dir with 2x in-flight, gbs 256, 64 MiB NATS |
| 110 | `d67214c71` | `e12b3a4cb` |  | chore(harbor-rl-glm53): r25 image tags gym-glm53-r5-20260914a and miles-glm53-r5-20260914a |
| 111 | `73223f264` | `614988acc` |  | feat(harbor-rl-glm53): r25 on the r24 base with arena_truncated_turn_rule mask |
| 112 | `2e8a7e087` | `b17080345` |  | docs(harbor-rl-glm53): r25 launch and r23 watcher restart on 2026-09-15 |
| 113 | `aca2e9520` | `c98c3d129` | conflict resolved | feat(arena): truncated-turn shift rule with lambda and floor; advantage mass metrics |
| 114 | `b4c3cbb8c` | `ccc5b7919` |  | feat(harbor-rl-glm53): r26 run dir with the shift rule |
| 115 | `e6d45866c` | `6fb7cdcda` |  | chore(harbor-rl-glm53): drop the stray TODO line in r25/gym-worker.yaml |
| 116 | `7091473fa` | `9aa9d3ade` |  | docs(harbor-rl-glm53): r23 retired, r26 launched on 2026-09-15 |
| 117 | `737fc586b` | `9c21582e3` |  | docs(harbor-rl-glm53): r23 end was a post-export trainer crash, not a watcher misfire |
| 118 | `8989cfb49` | `57f08ca59` | context merge | feat(arena): per-sample summary jsonl and a non-fatal dashboard dump |
| 119 | `f60ead4cc` | `3531d56e2` |  | chore(harbor-rl-glm53): keep the trainer image on regeneration; r26 r7 probe key |
| 120 | `8fa0e43f9` | `0a6f9558c` |  | chore(harbor-rl-glm53): r26 resume on trainer image r7 with the sample summary |
| 121 | `5e672da35` | `717bfb6ad` |  | docs(harbor-rl-glm53): r26 sample-summary probe, two lost rollouts, image r7 |
| 122 | `33697784f` | `71cb82f3e` |  | feat(arena): group-relative token-efficiency reward term |
| 123 | `28d8fd75d` | `e21748dc0` |  | test(arena): resolve the length-reward check path from __file__ |
| 124 | `99aaa9f05` | `3fbabf04a` |  | feat(glm53): r27 run config for the token-efficiency reward |
| 125 | `91a7f456c` | `4d7409a53` |  | chore(glm53): exclude the node that evicted r27, log the relaunch |
| 126 | `df10ccdcc` | `b7e7ff08d` |  | chore(glm53): exclude all drifted p6 nodes from r27 placement |
| 127 | `4a0d99bb3` | `254130f90` |  | docs(glm53-r27): record drift exclusion, admission path, resume watcher |
| 128 | `b5a6590e0` | `f5d755cd7` |  | feat(arena): r28 multi-step agentic-debt run config |
| 129 | `04b52b095` | `f7cb4c1a0` |  | chore(glm53-r28): set the ADR-0064 gym image gym-glm53-adebt-r28-20260920a |
| 130 | `e08b60d5e` | `7ba685185` |  | docs(glm53-r28): record the v5 template creation and the gym image |
| 131 | `6452ffeea` | `cfb7665a2` |  | chore(glm53-r28): trim excluded-nodes to the 83 nodes still drifted |
| 132 | `5aaa9f47d` | `cdf81bb37` |  | docs(glm53-r28): record the r28 launch and the watcher restart |
| 133 | `14aa62e70` | `03897666d` |  | feat(arena): r29 run config on the hardened gym image and template v6 |
| 134 | `03791ce86` | `143b700e7` |  | fix(arena): keep routing refs alive for buffered samples and drop a group on a lost ref |
| 135 | `7cec2a4b8` | `977c1344a` |  | chore(glm53): point r29 at trainer image r9 with the routing-ref fix |
| 136 | `d34c20c51` | `b832843e5` |  | fix(glm53): set r29 wandb project to rl-glm53f-adebt |
| 137 | `e6b1c370a` | `833cea774` |  | feat(harbor): r30 auctioneer-caponly on rl-glm53f-auct-cap |
| 138 | `68c2e5e71` | `c369a0012` |  | chore(harbor-rl-glm53): add r31 agentic-debt cut-at-first-failure config |
| 139 | `dc850c406` | `d5fcb98ec` |  | feat(glm53): r32 batch/update-cadence experiment (gbs 64, update_weights_interval 4) |
| 140 | `aa9202dea` | `5a2bb1594` |  | fix(glm53): regenerate r32 workflow with corrected embedded config (gbs 64, update_weights_interval 4) |
| 141 | `35395834b` | `bde34930c` |  | feat(harbor-rl-glm53): add r33 (4x agent timeout) and r34 (radix prefix cache) |
| 142 | `fe452e168` | `3e7ba74b4` |  | feat(harbor-rl-glm53): r34 combine radix prefix cache with 4x agent timeout |
| 143 | `57c82a535` | `4e82072e1` |  | feat(harbor-rl): r35 agentic-debt cut trained on >5-segment chains |
| 144 | `beaff500c` | `43315795b` |  | docs(rollout): record truncated_turn_shifted reporting rule |
| 145 | `b04f433e7` | `4a7b41cc9` | conflict resolved | refactor(arena): train every verified episode, drop clipped-turn rules |
| 146 | `cdafefc05` | `681deb7a2` |  | docs(arena): cite the renumbered AREnATasks ADRs |
| 147 | `a639ea445` | `d4908f2be` |  | feat(arena): default --arena-train-segments to all |
| 148 | `d6ef6ccbe` | `193566ea6` |  | chore(arena): r38 auctioneer-caponly redo on the rebuilt stack |
| 149 | `3df922f3a` | `52145c68a` |  | chore(arena): drop the partial-reward flag and CTRF reward text |
| 150 | `db3955b70` | `5cc9f928f` |  | feat(arena): log and count why the gym dropped a trajectory |
| 151 | `dd5e67391` | `024cd066f` |  | feat(arena): read token arrays from staged files (ADR-0014) |
| 152 | `82f4a288c` | `895983104` |  | chore(arena): r39 auctioneer caponly-1034 on the adr69 gym and r12 trainer |
| 153 | `91e6b115a` | `f08deae37` |  | chore(arena): r41 agentic-debt v3-locked on the adr71 gym and r13 trainer |
| 154 | `ed3dbdc9c` | `c22a23681` |  | chore(arena): r42 r39 with radix cache, overlap schedule, timeout x2 |
| 155 | `a65c95eb9` | `509ea0743` |  | chore(arena): record the r41b and r43 agentic-debt v3 runs |
| 156 | `4c9e97b0f` | `de20c95dd` |  | feat(arena): send the output cap, window, and sampling in the task message |
| 157 | `e0987aed6` | `46d0df956` |  | fix(arena): map compaction-max to agent-kwargs and reject top-k |
| 158 | `10a5242d1` | `38247af95` |  | docs(arena): record the miles-glm53-r14 trainer image |
| 159 | `0bd43a4ef` | `8bfcae379` |  | docs(arena): record the gym-glm53-adr72 image |
| 160 | `e949f2cdd` | `2e3971b0b` |  | docs(arena): prepare r44 auctioneer and r45 agentic-debt resume |
| 161 | `aa8cd8e70` | `abd4b4196` |  | docs(arena): prepare r46 auctioneer with token-level loss average |
| 162 | `f54990ee4` | `e9bc69c9d` |  | docs(arena): launch r44 and retire r39 and r42 |
| 163 | `0feaf4f4d` | `b63c6cf6d` |  | docs(arena): resume agentic debt as r45 from r43 step 39 |
| 164 | `662e0cc95` | `aad1fca18` |  | docs(arena): launch r46 |
| 165 | `bc31f88ac` | `41d45863f` |  | feat(arena): make the per-save HF export opt-in |
| 166 | `c10201fc5` | `d7a4ded25` |  | docs(arena): record the miles-glm53-r15 trainer image |
| 167 | `3028bc358` | `64882ca5f` |  | docs(arena): prepare r44 and r46 resumes with 8 engines |

## Refresh 2026-10-01

Branch `arpit-recon-20261001` is `arpit-reconcile-upstream` moved onto a newer
upstream `main`, plus the fork commits after the first reconcile and the R3
data-path branch. The goal is a trainer image with the r48 configuration on
this code.

### Inputs

| item | value |
| --- | --- |
| start point | `fc93b6f23` (`arpit-reconcile-upstream`), saved as `refs/recon/pre-refresh-fc93b6f23` |
| old base | upstream `23d41d711` (2026-09-26) |
| new base | upstream `main` `d7f1a4210` (2026-09-30 PT, "feat(fsdp): load module-level compute kernels from the Hub (#2829)"), 13 commits after `23d41d711` |
| fork | `arpit-glm-53` `be9d68a3f`: 37 non-merge commits after `3028bc358` (the source tip of the first reconcile), no merge commits |
| r48 code | the live r48 image (`guparpit-agentic-debt-v1`) runs `arpit-glm-53` `4716a367a`. After `4716a367a` the fork changes only docs, run records, `gen-workflow.py` and tests |
| R3 data path | `arpit-r3-datapath` `e193eae73`: 8 commits after `4716a367a` |

### Method

1. `git update-ref refs/recon/pre-refresh-fc93b6f23 HEAD`.
2. `git rebase --onto upstream/main 23d41d711`: 152 commits, no conflict. The
   diff of the old branch against its base and the diff of the new branch
   against its base differ only in blob ids and hunk line numbers.
   `git cherry` gives the same patch-id for all 152 commits.
3. `git cherry-pick -x` of each commit of
   `git rev-list --reverse --no-merges 3028bc358..arpit-glm-53`, in order.
4. `git cherry-pick -x` of each commit of `4716a367a..arpit-r3-datapath`, in
   order.
5. New commits for the weight-version metrics and for the test findings.

The picks keep the original author and author date. The committer is Arpit
Gupta <arpitg1991@gmail.com>. Each pick message ends with
`(cherry picked from commit <sha>)`.

### Upstream commits in the refresh

| upstream commit | change | effect on this branch |
| --- | --- | --- |
| `833d5bf78` #3718 | the command worker runs `exec <launch cmd>`, so the actor death frees its ports | no fork change in these files |
| `f84b1496e` #3660, `55ec3d870` #3699 | e2e CI on H200 | tests only |
| `9e4260de0` #3738 | the base image keeps the Mooncake of `lmsysorg/sglang`; no dev wheel | base image only. `examples/arena/Dockerfile` installs no Mooncake. The arena runs use the Ray object store |
| `3439ec751` #3689 | DSv4 CSA indexer balanced across CP ranks | `get_batch` also builds `cu_seqlens_host`, and `get_packed_seq_params` returns `PackedSeqParamsWithHostCuSeqlens` for each thd batch, GLM-5.3 included. Merged without a conflict next to the fork `get_batch` keys |
| `5ebc06f8d` #2541, `874b3e275` #3873, `077fb5922` #3157 | ROCm CI, ROCm images, AMD LoRA launcher | not used |
| `bb35ecc49` #3872 | two megatron rpc fast tests | tests only |
| `f602ca207` #3867 | a Claude rule for launch and request args | docs only |
| `79ef601d6` #3869 | each trainer pins its NUMA node from the PCI bus id of its GPU (`miles/ray/train_actor.py`) | a different file from `4316e4957` (the head node from the GCS node table, `miles/utils/ray_utils.py`). Both stay: upstream still reads the head from the dashboard |
| `8f409c7c4` #3647 | the base image patches the TileLang fp16 256-bit store on Blackwell | base image only |
| `d7f1a4210` #2829 | FSDP kernels from the Hub | FSDP only. The only change to `miles/utils/arguments.py` in the refresh is the FSDP `validate_kernel_backend_args` call |

No upstream commit does the work of a fork commit, so the refresh drops no
fork commit.

### Pick conflicts and resolutions

| source commit | new commit | file | resolution |
| --- | --- | --- | --- |
| `a9207dd7d` feat(glm5-next): shard KDA across tensor-parallel ranks | `8fe7f9b46` | `miles/backends/megatron_utils/update_weight/common.py` (deleted upstream) | upstream #2754 (`0e539b795`, already in `23d41d711`) moved `_check_and_fix_partition` and `_gather_with_stride` to `update_weight/hf_weight_iterator_direct.py`. The stride-3 branch for `self_attention.kda.conv1d.weight` goes there. `_gather_with_stride` has the same interleave on both sides. `test_glm5_next_kda_tp.py` imports from the new module. The other four files merged; their added and removed lines are the same as in the source commit |
| `3ad0037e8` docs(arena): record the miles-glm53-r17 trainer image | `ea78cb3c4` | `examples/arena/README.md` | keep both table additions: the three recon test rows (2026-09-27), then the r17 row (2026-09-28) and its note |
| `12fae498b` docs(arena): record the KDA tensor-parallel sharding decision (ADR-0016) | `ed3df8943` | `miles_plugins/arena/adr/README.md` | number collision; see "ADR numbers" |
| `f3c7ea1ce` chore(arena): add training-runs records and move the GLM-5.3 run records | `a0d1fa188` | `miles_plugins/arena/RUNLOG.md` | take the new `training-runs/` path in the old line; keep the two reconcile entries after it. The 195 renames applied; the file set under `examples/arena/harbor-rl-glm53-flash/` and `training-runs/` equals the fork commit |
| `d93d7eff7` docs(arena): record the training-runs convention (ADR-0017) | `b964d9a27` | `miles_plugins/arena/adr/README.md` | index rows in number order: 0016, 0017, 0018 |
| `2587b1d55` feat(r3): carry rollout routing as int16 and log the data path | `d0c58651b` | `miles/ray/rollout/rollout_manager.py` (deleted upstream), `miles/utils/data.py`, `miles/utils/object_store.py` | upstream moved `RolloutManager.get` to `RolloutExecutor.get` (`rollout_executor.py`): the load and convert timing and the `rollout_id` of `split_train_data_by_dp` go there. Upstream refs are pydantic models with a `ray.ObjectRef` `payload`, not `Box`, so `locate()` and two new tests read `ref.payload`. `data.py` keeps the upstream `RolloutDataPack` and `remove_rollout_data_refs` |
| `fce93b637` perf(r3): copy only the local rows in the thd replay fill | `238a3b9c4` | `miles/backends/megatron_utils/actor.py` | import block only: add the `r3_log` import; upstream removed the `Box` import |
| `23423ad6a` feat(r3): prefetch the next rollout shard behind --prefetch-rollout-data | `a323dcbab` | `miles/ray/train/actor_factory.py` (deleted upstream by #2162), `tests/fast/ray/test_actor_factory.py`, `miles/backends/megatron_utils/model.py`, `miles/utils/data.py`, `miles_plugins/arena/train_async_arena.py` | ported to the upstream worker structure; see "Prefetch port" |
| `d47c7951c` fix(r3): add prefetch_rollout_data to the v1 train group | `8a1f304c5` | `miles/ray/actor_group.py` (deleted upstream) | upstream has no v1 `RayTrainGroup` and no `_select_train_group_class`; the driver always calls `TrainerController`. The v1 hunk and `tests/fast/ray/test_train_group_prefetch.py` do not apply. The intent stays as a test of `TrainerController.prefetch_rollout_data` with a failed cell |

Skipped: `1c981e69d` test(kdatp): copy the prof harness of `arpit-glm-53`
`c2e17e5870`. The five files are byte-equal on the branch after the fork picks
(they come from `e7a0d35ba` to `c2e17e587`), so the pick is empty.

`git cherry` after the picks: 32 of the 37 fork commits and 3 of the 8 R3
commits have the same patch-id on the branch. The others are the conflict
picks above and the skipped commit.

### ADR numbers

`arpit-glm-53` gave ADR-0016 to the KDA sharding decision and ADR-0017 to the
training-runs convention, and its run records cite both numbers. The reconcile
ADR was Proposed and existed only on the reconcile branches. `30321858a` moves
it to ADR-0018 with its index row, the amendment lines in ADR-0005, ADR-0006,
ADR-0008 and ADR-0011, the plugin RUNLOG entry, this file, and the code
comments. The Apps ADR-0016 citations stay. `83a59c5d4` closes the numbering
notes in ADR-0016 and ADR-0017.

### Prefetch port

- `TrainRayActor.prefetch_rollout_data` has
  `@rpc(concurrency_group="rollout_prefetch")` and full type hints (the RPC
  surface needs them). `TRAINER_CONCURRENCY_GROUPS` declares
  `rollout_prefetch`.
- Upstream threads the trainer actor only with `--use-fault-tolerance`
  (a threaded non-FT actor deadlocked in NCCL setup). So
  `--prefetch-rollout-data` now also requires `--use-fault-tolerance`. Each
  GLM-5.3 config sets it.
- `TrainerController.prefetch_rollout_data(rollout_id, rollout_data_pack)`
  sends the call to each alive cell with `kill_on_failure=False`. The driver
  holds a worker handle, so `prefetch_when_generated` awaits the generate task
  and then calls the controller. A failure logs a warning.
- `RayObjectStore.pull` reads `ref.payload`.
- The upstream `train_one_step` has no multi-LoRA branch, so only
  `optimizer.step()` gets the optimizer timing.

Upstream bug found by the port: `_route_method_to_concurrency_group` wrapped the
method with `functools.wraps`, which sets `__wrapped__`. Ray 2.58 reads
`__ray_concurrency_group__` from `inspect.unwrap(method)`, so each routed
method ran in the default group, behind `train()`. A local Ray probe in the
test venv showed this. The upstream tests use a fake Ray cluster. `1b5d2978c`
drops `__wrapped__` and keeps the signature. With `--use-fault-tolerance` and
the Ray comm backend, `get_heartbeat_status`, `inject_fault`, `kill_self` and
`prefetch_rollout_data` now run in their own groups, as the static
`@ray.method` groups of the fork actor factory did. A trainer with one cell
(r48) starts no heartbeat checker.

### Weight-version metrics

`arpit-glm-53` logs `rollout/weight_version/{min,mean,median,max}` and
`rollout/weight_version/mixed_version_ratio` from `Sample.weight_versions`
strings (`miles/ray/rollout/metrics.py`). Upstream #1891 computes the same keys
from `WeightVersionSpan` objects. The NATS path keeps the versions in
`Sample.metadata["arena_weight_versions"]` (ADR-0018 decision 4), so the keys
were absent.

Filling the upstream spans was the first choice. A span needs the token range
of each turn, and the gym sends none. A made-up range also reaches
`assert_samples_weight_version_sane`, the token-weighted staleness filter,
`train_data["weight_versions"]` on every rank, and the dashboard. ADR-0018
rejected that. So `3f9010e73` adds
`rollout_metrics.compute_weight_version_metrics`: it uses the upstream
`compute_statistics` and the two upstream expressions (the oldest numeric
version of each sample; the share of all samples with more than one distinct
version). The keys cover every training row of the final batch, pads
included. A compact batch (`arena_train_segments: all`) is not trimmed, so this
is the list that upstream `log_rollout_data` gets, as on `arpit-glm-53`. A test
checks the values against upstream `Sample` spans with the same versions.
ADR-0018 has an amendment.

`3f9010e73` put the keys only in the plugin W&B dict. That dict exists only
with `use_wandb`, inside the `try` block that logs "Failed to log queue
metrics". The r48 gate reads the keys from the `perf <rollout_id>` line
(r48 `RECORD.md`, "Measurement plan"), and that line had none. `dda652e05`
returns the keys in `RolloutFnTrainOutput.metrics`, the channel of the upstream
rollout functions. `LegacyRolloutFnAdapter` passes it through, and
`log_rollout_data` writes the keys to the perf line and to `tracking.log`, as
on `arpit-glm-53`. The test runs `final` mode with W&B, and `all` mode without
W&B and with two DP pads (mean 4.375 over 8 rows), then feeds the output to
upstream `log_rollout_data`. Both cases fail on `3f9010e73`.

### Weight-version spans (replaces the metadata key)

The owner accepted ADR-0018 with a change to decision 4: the gym sends the
token range of each model call, and the NATS path fills the upstream
`Sample.weight_versions`. The commit after `0b09a6ab9` does this and deletes
the shim of `3f9010e73` and `dda652e05`. The gym side is AREnATasks ADR-0074
(branch `arpit-weight-version-spans`).

- Wire: each step keeps its data and adds `weight_version_spans`, one
  `{"version", "start", "end"}` per model call with an SGLang version.
  `[start, end)` indexes the cumulative `token_ids` of the step.
- Conversion: `_step_to_sample` keeps the step `token_ids` as
  `Sample.tokens`, so the offset is 0.
  `test_span_offset_is_zero_against_tokens_and_loss_mask` checks that each
  range holds the call output in `Sample.tokens` and loss mask 1. Each entry
  becomes one `WeightVersionsPerCall` with one `WeightVersionSpan`. A hard
  context overflow cuts the spans as `Sample.strip_last_output_tokens` does.
- Old gym images: a step with token arrays and without `weight_version_spans`
  stops the run when the trajectory reports a weight version. The error
  (`WeightVersionSpansError`, fatal like `RoutingReplayError`) names the field
  and the minimum gym image. A trajectory without versions gives empty
  `Sample.weight_versions`.
- Upstream code on arena samples: `assert_samples_weight_version_sane`,
  `Sample.validate`, `rollout/weight_version/*` of `log_rollout_data`,
  `train_data["weight_versions"]`, and the dashboard. On rows with the same
  versions, the upstream keys give the numbers of the deleted shim: 5.0, 5.5,
  6, 3 and 0.25 in `final` mode, 4.375, 4.0, 6, 3 and 0.5 in `all` mode with
  two DP pads (`test_rollout_logs_the_upstream_weight_version_metrics`), and
  14/3, 5.0, 6, 3 and 0.5 on four rows
  (`TestUpstreamWeightVersionMetrics`).
- Meaning change: a segment row reads the spans of its own calls, not the
  trajectory list. `rollout/off_policy_round/*` stays per episode and reads
  every segment, without the DP pads.
- Consume-time staleness: `generate_rollout` applies the upstream
  `DefaultDataBuffer.get` rule with the upstream `group_staleness` helper and
  `--max-weight-staleness` (default off). It returns
  `rollout/fully_async/{stale_groups_filtered,avg_staleness,max_staleness}`
  and logs one `Weight staleness:` line per rollout. The engine version comes
  only through the class seam, so `NatsRolloutFn` wraps `generate_rollout`.
  The run configs still name the function, which gets no version: then only
  the drop count (0) is logged, and `--max-weight-staleness` stops the run
  with an error.

Validation (CPU, the test venv and the stand-ins of "Validation (CPU,
2026-10-01)"):

| check | `0b09a6ab9` | this commit |
| --- | --- | --- |
| arena fast tests, `stub` | 344 passed | 362 passed |
| arena tests and `test_run_arena_harbor.py`, `stubfull`, conftests, local Ray | 355 passed, 1 failed (`test_register_nova_reasoning_parser`) | 373 passed, the same failure |
| launcher tests, `stub` | 65 passed, 1 failed (`test_workplace_launch_uses_the_configured_backend`, also on upstream) | the same |
| launcher snapshots, `-k "run_arena_harbor or run_glm5_3_flash"` | 6 passed | 6 passed |
| upstream tests of the reused code: `utils/test_types.py`, `utils/test_weight_version.py`, `rollout/test_filters.py`, `rollout/test_fully_async_rollout.py`, `ray/rollout/test_metrics.py`, `rollout/inference_rollout/test_compatibility.py`, `ray/rollout/test_train_data_conversion.py`, `ray/rollout/test_rollout_executor.py`, `dashboard/test_dump_reader_views.py`, `dashboard/test_trajectory_sink.py`; `stubfull`, conftests, local Ray | 347 passed, 1 skipped | the same |
| ADR-0005 check, `diff train_async.py miles_plugins/arena/train_async_arena.py \| grep -c '^<'` | 0 | 0 |

### Other new commits

| commit | change |
| --- | --- |
| `8268c4991` | `kdatp/t1_parity.py` imports `all_gather_params_async` from `hf_weight_iterator_direct.py` (upstream #2754) |
| `4f743c5bf` | `kdatp-prof-run.sh` and `dsa-run.sh` take the checkout only from `AGISLIME_DIR`, which both job manifests set. Upstream `test_shell_script_hygiene` rejects `/root/miles` in a script |
| `87db4caa4` | `test_rollout_prefetch.py` and `test_r3_log_parse.py` use the session fixture `ray_local_mode`. Upstream `test_ray_cluster_bootstrap` allows only `tests/conftest.py` to call `ray.init` |
| `2c172d505` | the two `split_train_data_by_dp` fakes in `test_rollout_executor_multi_policy.py` accept the `rollout_id` keyword of the R3 port |
| `25ed41e1f` | this section, first version |
| `dda652e05` | `rollout/weight_version/*` through `log_rollout_data` (see "Weight-version metrics"). Five tests read `.samples` of the return value |
| `2cb91b309` | the mini FT controller advice (see "Fault tolerance against r48"), and `test_yaml_false_emits_no_flag` |
| `a43aa80f2` | `guparpit-agentic-debt-v2/miles-config.yaml` and `test_training_run_identity.py` (see "Launch from the r48 files") |
| `000f49d2c` | open item 6, the KDA tensor-parallel checkpoint load check |
| `b167425ce` | "Review findings (2026-10-01)" and the validation numbers |
| `ab0958b17`, `4a78e4b6d`, `0b09a6ab9` | the r48 summary-dir record, the `guparpit-agentic-debt-v2` workflow, and its image `recon-20261001b` |
| `da93977ba` | `examples/arena/Dockerfile` pins `opentelemetry-api` to the installed sdk, so `ray start --head` works in the image |
| this commit | the weight-version spans (see "Weight-version spans") |

### Validation (CPU, 2026-10-01)

The test venv is `/workplace/guparpit/kdfast/testvenv` (python 3.12, torch
2.14 CPU, Ray 2.58, no sglang, no megatron). Two sglang stand-ins, both outside
the repo:

- `stub`: the files of `/tmp/sglang-stub` (the first reconcile), with
  `--noconftest`.
- `stubfull`: `stub` plus an `sglang.srt.server_args.ServerArgs` that
  registers the 90 `--sglang-*` flags the repo uses, and placeholder modules
  for every other `sglang.*` import. With it the conftests load, so the
  fixtures exist. The placeholders also make some tests wait for servers, so
  the `stubfull` sweep skips `rollout` and `router`.

Ray sockets must be shorter than 108 bytes, so a run with a local Ray cluster
starts pytest in the scratch dir with `RAY_TMPDIR=/proc/self/cwd/r` and
`--rootdir` on the repo. The upstream reference is a detached worktree of
`d7f1a4210` in the same venv.

| check | result |
| --- | --- |
| arena fast tests: `tests/fast/plugins/arena`, `stub` | 343 passed, 1 failed (`test_training_runs_index`, see below). `arpit-glm-53` `be9d68a3f`: 335 passed, the same failure. The 8 new tests: 5 weight-version tests and 3 run-identity tests. At `25ed41e1f`: 339 passed, the same failure |
| launcher tests: `tests/fast/launch_scripts`, `stub` | 65 passed, 1 failed: `test_workplace_backend.py::test_workplace_launch_uses_the_configured_backend`, which also fails on upstream (53 passed, 1 failed). The 3 new tests are `test_yaml_false_emits_no_flag` |
| arena tests and `test_run_arena_harbor.py`, `stubfull`, conftests, local Ray | 354 passed, 2 failed: `test_training_runs_index` and `test_register_nova_reasoning_parser` (see below) |
| launcher snapshots: `tests/manual/launch_scripts/test_py_launch_scripts.py -k "run_arena_harbor or run_glm5_3_flash"` | 6 passed, with each stub |
| KDA TP: `tests/fast/backends/megatron_utils/test_glm5_next_kda_tp.py` | skipped: the module needs `megatron.core`. The stride-3 gather check of that file, run on the two functions lifted from `hf_weight_iterator_direct.py`, passes: the gather of 8 TP shards of the packed [q; k; v] conv equals the full tensor, and stride 1 raises |
| R3, replay, and prefetch tests, `stubfull`, conftests, local Ray | `test_rollout_prefetch.py` 8 passed (with the real-Ray test that the prefetch runs next to `train()` and gives `prefetch=hit`), `test_r3_log.py` and `test_r3_log_parse.py` 8, `test_route_method_to_concurrency_group.py` 2, `test_replay_base.py` 6, `test_replay_data.py` 291, `test_object_store.py` 66 (2 Mooncake skips), `test_train_data_conversion.py` 60, `test_rollout_metrics.py` 17, `test_rollout_executor_multi_policy.py` 11, `test_train_actor_rpc_surface.py` 15 (4 megatron skips), `test_ray_cluster_bootstrap.py` 10, `test_ray_utils.py` 4. `test_shared_ppo_lifecycle.py` errors on both trees (no megatron) |
| `tests/fast` sweep by directory, `stub`, `--noconftest` | this branch: 7041 passed, 130 failed, 692 errors, 368 skipped. Upstream: 6384 passed, 128 failed, 689 errors, 367 skipped. Of the tests on both trees, one changes outcome: `test_docs_examples_matches_the_readmes` (pass to fail). The 663 tests only on this branch: 658 passed, 1 skipped, 1 failed, 3 errors (see below). No test exists only on upstream |
| sweep of `plugins`, `ray`, `utils`, `backends`, `launch_scripts`, `miles_plugins`, `scripts`, `tools` and the top-level files, `stubfull`, conftests, local Ray | this branch: 10642 passed, 461 failed, 81 errors, 127 skipped. Upstream: 9970 passed, 459 failed, 81 errors, 126 skipped. No test on both trees changes outcome. The 675 tests only on this branch: 672 passed, 1 skipped, 2 failed: `test_training_runs_index` and `test_register_nova_reasoning_parser` (the placeholder `ReasoningParser` of `stubfull` has no `DetectorMap`; the test passes with `stub`) |
| r48 argv | parses on both trees with the miles parser; see "Behavior changes against r48" |
| `git cherry` | all 152 commits of the first reconcile keep their patch-id; see "Pick conflicts and resolutions" for the picks |

Upstream-baseline failures (`stub` sweep, the same tests and outcomes on both
trees): 128 failed and 689 errors. The causes: modules that the venv lacks
(`sglang.srt.server_args` 144, `megatron` 123, `sglang.srt.constants` 40, other
`sglang.srt.*` 23, `tinker`, `msgspec`, `triton`), conftest fixtures that
`--noconftest` removes (`handle`, `service`, `ray_local_mode`, `raw`, and
others), and 49 `encoding_dsv4` attribute errors of the empty stub module.

Test outcomes that exist only on this branch:

- `tests/fast/doc/test_sync_example_docs.py::test_docs_examples_matches_the_readmes`
  passes on upstream and fails here: `examples/README.md` lists no
  `examples/arena` directory. `arpit-glm-53` and the first reconcile fail the
  same way. The fix adds 8 fork-only bullets to an upstream index, so it waits
  for the owner.
- `tests/fast/plugins/arena/test_training_runs_index.py::test_committed_index_is_current`
  fails here and on `arpit-glm-53` `be9d68a3f`: the r48 commit did not
  regenerate `training-runs/INDEX.md`, and the r48 `RECORD.md` header lacks the
  date, image, and outcome fields. A fix belongs in the r48 record on
  `arpit-glm-53`.
- Under `stub` with `--noconftest`, `test_r3_log_parse.py` and
  `test_rollout_prefetch.py` do not collect (`sglang.srt.server_args`,
  `sglang.srt.constants`), and `test_locate_reports_size_right_after_put` has
  no `ray_local_mode` fixture. All three pass in the `stubfull` run with the
  conftests and a local Ray cluster.

### Behavior changes against r48 (`arpit-glm-53` `4716a367a`)

Method: the launcher of each tree builds the argv of
`training-runs/harbor-rl-glm53-flash/guparpit-agentic-debt-v1/miles-config.yaml`
(`_build_train_args`, the model args of `scripts/models/glm5.3-flash.py`, and
`--deploy-component all` on this branch). Then the miles parser of each tree
parses it (`get_miles_extra_args_provider`, `parse_known_args`, the same
`stubfull` sglang flags on both). The venv has no megatron, so the Megatron
defaults and `miles_validate_args` are a static reading, not a parse.

| area | r48 | this branch |
| --- | --- | --- |
| argv | the r48 argv | the same plus `--activation-func-clamp-value 10` (model script) and `--deploy-component all` (backend). No other flag or value differs |
| miles defaults | | changed: `mini_ft_controller_enable` False to None, then resolved True by `use_fault_tolerance` (the controller on `api_server_port` 18080); `session_sample_picker_path` `drop_retries` to `drop_same_prompt_retries`. Removed: `control_server_port`, `rollout_external`, `sglang_gated_launch_port`, `eval_sglang_gated_launch_port`, the 9 `multi_lora_*` flags. New, inert for r48: `api_server_host`, `cluster_backend` ray, `worker_comm_backend` None (resolves to ray), `deploy_component`, `deploy_instance_id`, `run_uuid`, `trainer_controller_addrs`, `inference_controller_addr`, `init_expected_num_cells`, `megatron_config`, `env_report_interval_seconds` 3600, `rollout_health_check_failure_threshold` 1 (r48 also stops an engine at the first failed check), `rollout_external_router_pd`, `session_server_external_host`, `custom_inference_engine_provider_path`, `debug_unified_grad_fused_logprob`, three `ci_*` flags, and `prefetch_rollout_data` False |
| resolved at validation (static) | | `activation_func_clamp_value` None to 10.0; `mini_ft_controller_enable` False to True; `save_debug_event_data` None to `<save>/events`; `rollout_health_check_first_wait` 0 to 0.0 (the list of the first reconcile; the refresh adds no change for this config) |
| GLM-5.3 model | fork copy of PR #2786 | upstream #2786: RMSNorm on the DSA indexer query, the clamp 10, the mHC spec. KDA TP (`glm5_next_kda_tp: true`) is the same code on the upstream module |
| events and checksums | | the event logger is on, `update_weights` runs `check_weights(action="checksum")` on each engine at each update, and each save copies the events dir |
| driver | `rollout_manager.generate` | the upstream loop: `inference_controller.prepare_rollout` before each generate, the weight version published to the rollout executor after each update, the disposer teardown, the API server and the mini FT controller |
| fault tolerance | a failed engine stops at the first failed check and restarts at the next weight update | the mini FT controller restarts it at any time. Keep `mini_ft_controller_enable` unset; see "Fault tolerance against r48" |
| metrics | | `train_rollout_logprob_abs_diff` and `train_rollout_kl` from trainer-scored log-probs (#3655). `rollout/weight_version/*` have the r48 keys and meaning (`3f9010e73`), on the perf line and in `tracking.log` as on r48 (`dda652e05`). After `0b09a6ab9` upstream computes them from the spans, and a segment row reads only its own calls (see "Weight-version spans") |
| R3 data path | int32 routing | routing as int16 in the shards and the pinned replay buffers, widened to int32 at the router (the same replayed indices); the thd replay fill copies only the local rows (byte-equal to the old fill in the tests) |
| logs | | always-on `[r3-timing]`, `[r3-mem]` and `[r3-digest]` lines. `[r3-digest]` hashes the replay buffers of each rank once per step and reports its own `seconds` |
| Ray actors | static `@ray.method` FT groups | the upstream groups through `_route_method_to_concurrency_group` with `1b5d2978c`, plus `kill_self` and `rollout_prefetch` (idle with the flag off). Each trainer sets its NUMA affinity from the PCI bus id of its GPU (#3869) |
| head node | the driver on the head | `RayWorkerManager` actor, head from the GCS node table (`4316e4957`) |
| thd batches | `PackedSeqParams` | `PackedSeqParamsWithHostCuSeqlens`, a subclass with the host copy of `cu_seqlens` (#3689) |
| command workers | `<launch cmd>` under a shell | `exec <launch cmd>`, so the death of the actor frees the ports (#3718) |
| base image | `glm53next-upstream-20260902` (SGLang `9a26e749`, Megatron-LM `e8f57451`, fla 0.4.2, TileLang 0.1.9) | needs a base from the upstream `docker/build.py` at `d7f1a4210`: SGLang `sglang-miles` with `--gated-launch-port`, Megatron-LM `miles-main`, fla 0.5.2 with the upstream KDA patch and `fla_conv_int64_offsets.py`, TileLang 0.1.14 with the #3647 Blackwell patch, Mooncake from `lmsysorg/sglang` |

### Fault tolerance against r48

On `arpit-glm-53`, `RolloutHealthMonitor._check_engine_health`
(`miles/utils/health_monitor.py`) stops an engine at the first failed check.
`RayTrainGroup.update_weights` (`miles/ray/actor_group.py`) calls
`rollout_manager.recover_updatable_engines` before each weight update, so the
engine restarts then.

On this branch the mini fault-tolerance controller does that work. The r48
config sets `use_fault_tolerance`, so `ft_components` resolves to `rollout`,
`api_server_port` to 18080, and `mini_ft_controller_enable` to True. The
controller polls the cells on the API server. It suspends and resumes each
unhealthy cell, through `stop_cell_between_weight_updates` and
`RayWorkerManager.start_cells`. A cell that updates weights joins the router
only in `mark_weights_ready`, after the next weight update, so it never
serves old weights. The difference is the time: the controller heals at
any time, not only at the next weight update. No other code restarts an
engine. `SimpleHealthChecker` only sets the cell status, and
`start_update_weights` only waits for the cells to be ready.

Keep `mini_ft_controller_enable` unset. The two settings that turn the
controller off leave a failed engine stopped for the rest of the run:

| config line | argv | resolved |
| --- | --- | --- |
| none (r48) | none | True, port 18080 |
| `mini_ft_controller_enable: false` | none: the launcher drops a false value | True, port 18080 |
| `no_mini_ft_controller_enable: true` | `--no-mini-ft-controller-enable` | False |
| `api_server_port: 0` | `--api-server-port 0` | False, port 0 |

The probe builds the argv with the launcher and resolves it with the miles
parser and the three upstream resolvers (`stubfull`).
`test_run_arena_harbor.py::test_yaml_false_emits_no_flag` checks the argv
column.

### Launch from the r48 files

`training-runs/harbor-rl-glm53-flash/guparpit-agentic-debt-v2/miles-config.yaml`
is the r48 config with a new run identity. Only `experiment_name`,
`project_name` (both launcher keys) and `arena_sample_summary_dir` differ.
The launcher builds the same 190-token argv from both files, except for the
sample-summary dir. r48 writes into the r47 sample-summary dir, and
`_write_sample_summary` replaces `rollout_<id>.jsonl`, so a third run there
would replace the files of both. `test_training_run_identity.py` checks the
key set, and that no two run folders share an experiment name or a
sample-summary dir (r47 and r48 are the one recorded exception). The three
tests fail when the v2 folder holds copies of the r48 `miles-config.yaml` and
`workflow.yaml`.

The launch needs three more steps that no committed file holds:

1. Set the workflow parameter `experiment-name` to
   `guparpit-agentic-debt-v2`. The launcher sets `--load` and `--save` to
   `<ARENA_CHECKPOINTS_DIR>/slime_experiments/<experiment-name>` and ignores
   the config `experiment_name`. With the r48 value, the run resumes from
   the latest r48 checkpoint and saves into the r48 dir. The W&B group and
   run name come from the same parameter.
2. Seed `slime_experiments/guparpit-agentic-debt-v2/` with a copy of a
   checkpoint saved with `glm5_next_kda_tp: true`: r47 `iter_0000059` (the
   r48 start, for a comparison at the same rollout ids) or a later r48 save.
   Copy `rollout/arena_data_source_state_<N>.pt` too, and write
   `latest_checkpointed_iteration.txt` last. Write the
   `slime_extra_state.json` of the copy without `wandb_run_id`.
   `_load_extra_state` restores that id, and the run then resumes the W&B
   run of the source (r48 `631mke7d`, r47 `0ix3m75e`) at the same
   `rollout/step` values.
3. Keep `mini_ft_controller_enable` unset (see "Fault tolerance against
   r48").

Before the long run, do the KDA tensor-parallel load check (open item 6).

### Review findings (2026-10-01)

Four review findings against `25ed41e1f`. Each was checked again before a
change.

| finding | check | result |
| --- | --- | --- |
| The advice `mini_ft_controller_enable: false` has no effect, and turning the controller off is wrong | the launcher drops a false value; the probe table; the restart paths on both trees | real. `2cb91b309` |
| `rollout/weight_version/*` go only to W&B, but the r48 gate reads the perf line | the keys are only in the `use_wandb` dict; upstream `log_rollout_data` merges `RolloutFnTrainOutput.metrics` | real. `dda652e05` |
| A copy of the r48 files reuses the r47 and r48 checkpoints, the W&B run, and the sample summaries | `ckpt_dir` from `experiment-name`; the r47 sample-summary dir in the r48 config; `_load_extra_state` | real. `a43aa80f2` prepares the v2 config and its test. The experiment name and the checkpoint seed are launch steps ("Launch from the r48 files") |
| No job has loaded a KDA-TP checkpoint on the new base | r47 config `glm5_next_kda_tp: true`; the GPU jobs of the first reconcile ran with KDA TP off; the CPU-only refresh check | real, open: it needs a GPU job. `000f49d2c` adds open item 6. One detail of the finding is not correct: upstream `kimi_k3` and `inkling` use the same Megatron checkpoint APIs as `kda.py` |

### Open items of the refresh

1. Build a test image on the new base and run the r48 argv through
   `parse_args()` in it. That check covers the Megatron defaults
   (Megatron-LM `e8f57451` to `miles-main`) and `miles_validate_args`.
2. The open items of the first reconcile still apply: the engine path with
   SGLang 0.5.21 and the event-logger checksum, the 18-21% slower log-prob pass
   and the 8 GiB higher peak memory, the fla release with PR #1082.
3. Offer `1b5d2978c` (the routed-method fix) and `4316e4957` to upstream.
4. On `arpit-glm-53`: complete the r48 `RECORD.md` header and regenerate
   `training-runs/INDEX.md`.
5. `--prefetch-rollout-data` ran on GPU only on `arpit-r3-datapath`. The port
   to the upstream worker structure passes the CPU tests with a local Ray
   cluster, but it has no GPU run.
6. No job has loaded a checkpoint saved with `glm5_next_kda_tp: true` on the
   new base. r47 saved `iter_0000059` with KDA TP on (image r17,
   `4716a367a`), and an r48-like run starts there. The GPU jobs of the
   first reconcile loaded the r45 DCP with KDA TP off. The refresh checked
   the stride-3 gather on CPU only. The KDA module differs from `4716a367a`
   by one import line. Upstream `kimi_k3/layers.py` calls
   `ensure_metadata_has_dp_cp_group` and `make_sharded_tensors_for_checkpoint`
   the same way, and upstream `inkling/layers.py` builds a custom
   `ShardedTensorFactory` with the same arguments. These three APIs and
   the `tp_group` and `dp_cp_group` keywords of
   `make_tp_sharded_tensor_for_checkpoint` are the same in the Megatron tree
   of the 20260929c profile pods and in NVIDIA Megatron-LM `b4d72b79`. No
   tree of `miles-main` `f148a32b` was on disk, so that check is only a
   static hint. Before the long run, run one T1-style job on the new image
   with `glm5_next_kda_tp: true`:
   1. Load r47 `iter_0000059` with TP 8, as in the run.
   2. Compare the SHA-256 of the HF gather (the raw-mode weight sync path,
      with the stride-3 branch of `hf_weight_iterator_direct.py`) with the
      r17 image at the same iteration.
   3. Compare the log-probs with the r17 image on a few r48 rows, with the
      kdatp gate of "GPU validation (2026-09-27)": relative L2 error at
      most 2 x floor + 2**-8, and mean log-prob difference at most
      2 x floor + 1e-3.

   Open item 1 (`parse_args()` in the image) does not cover this.
7. The weight-version spans (see "Weight-version spans") have CPU tests
   only.
   1. Build a gym image from the AREnATasks ADR-0074 change. Put its tag in
      `_MIN_GYM_IMAGE_FOR_WEIGHT_VERSION_SPANS` (`nats_rollout.py`). A
      trainer image from this commit stops on the first message of an older
      gym image, for example `gym-glm53-adr72-20260927a` of
      `guparpit-agentic-debt-v2`. The v2 trainer image `recon-20261001b` is
      older than this commit.
   2. That gym image MUST send the spans from both training runtimes: Harbor
      and Inspect streaming (`amzn_arena_streaming`, `sglang_full` and
      `sglang_perstep`). An Inspect streaming gym without the spans stops
      this trainer at the first message. The error names the gym class of
      each runtime (`test_inspect_streaming_without_spans_names_both_gym_runtimes`).
      `test_inspect_streaming_spans_have_offset_zero` checks the offset 0 of
      both Inspect providers.
   3. For the staleness filter or the staleness keys, set
      `rollout_function_path` to
      `miles_plugins.arena.nats_arena.nats_rollout.NatsRolloutFn`.
   4. Run one smoke job on the new gym and trainer images. Check
      `rollout/weight_version/*` in the `perf <rollout_id>` line, the
      `Weight staleness:` line, and that `assert_samples_weight_version_sane`
      passes.
