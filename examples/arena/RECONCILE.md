# Reconcile record: `arpit-glm-53` onto radixark/miles `main` (2026-09-27)

This file records how branch `arpit-reconcile-upstream` was made. It lists each
dropped commit, each partial keep, and each conflict resolution. The decisions
are in plugin ADR-0016 (`miles_plugins/arena/adr/0016-reconcile-with-upstream-main.md`).

## Inputs

| item | value |
| --- | --- |
| fork tip | `origin/arpit-glm-53` = `3028bc358` (local ref `refs/recon/fork-3028bc358`) |
| upstream tip | `upstream/main` = `23d41d711` (2026-09-26, "Drop the empty-batch timeout only multi-LoRA v1 ever raised (#3723)") |
| merge base | `2799fe386` (2026-08-31) |
| fork side | 171 commits: 167 non-merge commits and 4 merge commits |
| upstream side | 750 commits |
| result | 139 kept fork commits, 1 restore commit, and 8 new commits on top of `23d41d711`. The commit that adds this file is the branch tip |

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
| `a18a23903` | ADR-0016 and the amendment links in ADR-0005, ADR-0006, ADR-0008, ADR-0011 |
| `bc0c843c6` | `train_async_arena.py` rebuilt on the upstream `train_async.py`. The old driver imported `create_rollout_manager`, `MainProcessIdentity`, and `ft_utils.control_server`, which upstream removed. The ADR-0005 check stays at 0 deleted lines. The final drain and the heartbeat removal are one disposer callback, `_finish_arena` |
| `f32dd8838` | `scripts/run_arena_harbor.py` calls `args.create_backend()`; upstream #2432 removed `U.exec_command_cpu` and `U.execute_train`. The snapshot adds `--deploy-component all` |
| `e8d4ae9a1` | the gym weight versions move to `Sample.metadata["arena_weight_versions"]`; `Sample.weight_versions` stays empty. Upstream #1891 made that field a list of span objects |
| `aed3522de` | the advantage-scale tests set a one-rank parallel state (upstream #3125) |
| `7466bc9cc` | the `run_glm5_3_flash` launcher snapshot regenerated for the merged #2786 script |
| `9a5b6216d` | the plugin RUNLOG entry |
| this commit | this file |

## Upstream changes that alter a run on this branch

| area | change | effect on an r44, r45, or r46 config |
| --- | --- | --- |
| GLM-5.3 DSA indexer | the final #2786 applies RMSNorm to the indexer query. The fork copy fed the raw query; SGLang `9a26e749` feeds the normalized query | the live runs on `arpit-glm-53` have this train/rollout mismatch in all 11 DSA layers. The fix changes the top-k selection and the log-probs |
| GLM-5.3 MLP | `scripts/models/glm5.3-flash.py` adds `--activation-func-clamp-value 10`. The checkpoint sets `swiglu_limit` 10, and SGLang clamps | a second train/rollout mismatch goes away |
| GLM-5.3 mHC | the mean output contraction comes from a Megatron spec, not a monkeypatch | no checkpoint format change is expected. A GPU load of an r45 checkpoint must prove it |
| fault tolerance | with `use_fault_tolerance`, upstream turns on the mini fault-tolerance controller on `api_server_port` 18080. `--control-server-port` is gone | new auto-heal of failed engines. Set `mini_ft_controller_enable: false` for a like-for-like test |
| checkpoint dir | `save_debug_event_data` defaults to `<save>/events` (#2505); each checkpoint gets a copy of the event logs | extra files in the checkpoint dir. The Megatron DCP format, the data-source state path, and the single-policy dir layout do not change |
| metrics | `train_rollout_logprob_abs_diff` and `train_rollout_kl` come from trainer-scored log-probs (#3655) | values before and after the rebase are not directly comparable |
| weight-version metrics | the upstream `weight_version/*` metrics read span objects | arena samples carry no spans, so these metrics are absent. `rollout/off_policy_round/*` does not change |
| TIS, PPO, R3 | the math does not change. R3 still sets `enable_return_routed_experts` and reads `rollout_routed_experts` | low. The upstream GLM launcher forces `--sglang-moe-runner-backend triton` for R3; the arena configs keep `auto`, which r16 validated |
| dependencies | the upstream Dockerfile moves SGLang v0.5.18 to v0.5.20, TileLang 0.1.8 to 0.1.14, Megatron-Bridge `7f0fb345` to `8cd3466d`, and adds a patch for fla 0.5.2 | the arena base `glm53next-upstream-20260902` keeps SGLang `9a26e749`, Megatron `e8f57451`, fla 0.4.2, and TileLang 0.1.9. The upstream GLM-5.3 docs pin the same SGLang and Megatron pair. No new base image is necessary for the CPU checks |

## Validation (2026-09-27)

| check | result |
| --- | --- |
| arena fast tests: `PYTHONPATH=$PWD:/tmp/sglang-stub /tmp/miles-test-venv/bin/python -m pytest tests/fast/plugins/arena -q -p no:cacheprovider --noconftest` | 325 passed |
| launcher tests: `tests/fast/launch_scripts` in the same venv | 49 passed, 1 failed. `test_workplace_backend.py::test_workplace_launch_uses_the_configured_backend` also fails on upstream `main` in the same venv |
| launcher snapshots: `tests/manual/launch_scripts/test_py_launch_scripts.py -k "run_arena_harbor or run_glm5_3_flash"` | 6 passed. The whole manual suite has no failure that upstream `main` does not also have |
| `tests/fast` sweep in the venv, by directory, against upstream `main` | 6545 passed on this branch, 6205 on upstream. 341 tests pass only on this branch; they are fork tests. One test changes from pass to fail: `tests/fast/doc/test_sync_example_docs.py::test_docs_examples_matches_the_readmes`, because `examples/README.md` does not list `examples/arena`. The fork tip fails the same test. All other outcomes are identical |
| image `arena-slime-dev:miles-glm53-r15-20260927a` (SGLang `9a26e749`, Megatron `e8f57451`), tree mounted on `PYTHONPATH` | the `examples/arena/Dockerfile` import smoke passes, including `miles_plugins.arena.train_async_arena`. `run_arena_harbor.py --help` passes. The arena, launcher, loss, `test_train_data_conversion`, `test_samples`, `test_loss_mask_qwen3_5`, and `test_types` tests: 574 passed, 13 failed; upstream `main` has the same 13 failures in that image (12 loss snapshots that clone an artifact repository, and `test_workplace_backend`). With network access, `test_loss_snapshot.py --compare` passes (12) |
| argv parse: `_build_train_args` of the launcher, the model args of `scripts/models/glm5.3-flash.py`, and `--deploy-component all`, then `miles.utils.arguments.parse_args()` in the r15 image | all five configs parse: `r44/miles-config.yaml`, `r44/miles-config-r0.yaml`, `r45/miles-config.yaml`, `r46/miles-config.yaml`, `r46/miles-config-r0.yaml`. Against the fork tip, the argv adds `--activation-func-clamp-value 10` and `--deploy-component all` only. The resolved values change for `activation_func_clamp_value` (None to 10.0), `mini_ft_controller_enable` (False to True), `save_debug_event_data` (None to `<save>/events`), `session_sample_picker_path` (renamed default), and `rollout_health_check_first_wait` (0 to 0.0) |

## Open items

1. The owner accepts or changes ADR-0016 decision 4 (the weight versions in
   `Sample.metadata`, no spans).
2. Build the test trainer image `arena-slime-dev:miles-glm53-recon-test-20260927a`
   with `examples/arena/Dockerfile` on `glm53next-upstream-20260902`. Add the
   lineage row in `examples/arena/README.md` when the image exists.
3. GPU test jobs (`recon-*`): the import smoke, a load of an r45-lineage
   checkpoint copy, and one train step with R3 and TIS. Compare the step-0
   log-prob gap and `ppo_kl` with r45.

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
