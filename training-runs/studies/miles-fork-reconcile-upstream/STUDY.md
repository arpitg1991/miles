# Study: Rebase of the miles fork onto upstream `main` (`arpit-reconcile-upstream`)

**Date:** 2026-09-28
**Status:** Open
**Question:** Can the fork `arpit-glm-53` move onto radixark/miles `main` with no loss of arena behavior, and which gates stand between the result and a live run?
**Runs and data used:** r45 `iter_0000039` DCP (read-only links to r43), r43 `hf/rollout_39` export, the kdatp T2 rows (314 rows, 22.3M tokens, up to 131,070 tokens per row), the kdatp T2 `baseline` arm `kdatp-t2-20260927e`, the r44, r45, r46 configs (argv parse only). Results under `s3://arena-scratch-prod-bom-ap-south-1/guparpit/recon/` (`t1/`, `t2/`, `convchk/`). No W&B run. No checkpoint written.
**Code SHAs:** fork tip `3028bc358` (`origin/arpit-glm-53` on 2026-09-27); upstream tip `23d41d711` (2026-09-26); merge base `2799fe386` (2026-08-31); result `fc93b6f23` = `origin/arpit-reconcile-upstream`; upstream squash of PR #2786 `cc76e2391`; new commits `a18a23903`, `bc0c843c6`, `f32dd8838`, `e8d4ae9a1`, `aed3522de`, `7466bc9cc`, `9a5b6216d`, `77cd880d4`, `c5b138782`, `4d6c39a7d`, `4316e4957`, `fc93b6f23`; KDA ADR `12fae498b` on `arpit-glm-53`.
**Workflow ids:** `wf_3ff0709b-51c` (six agent reports: inventory, rebase, review, image, test, report)

## Method

The work ran in the worktree `/workplace/guparpit/miles-reconcile` on the
branch `arpit-reconcile-upstream`. The full record is
`examples/arena/RECONCILE.md` on that branch (453 lines). The decisions are
in the branch ADR `miles_plugins/arena/adr/0016-reconcile-with-upstream-main.md`.

1. **Inventory.** `git cherry` against upstream (0 of 167 fork commits
   already upstream by patch-id), a `git merge-tree` dry run of the rebase,
   the CPU tests of the simulated tree in the `miles-glm53-r15-20260927a`
   image, and a parse of the full r45 argv against upstream plus the arena
   plugin (journal report 1).
2. **Rebase.** `git rebase -i --onto upstream/main 2799fe386` with a
   scripted todo list: `drop` for the 28 PR-copy commits, `pick` plus
   `break` for three partial keeps, one `exec` for a restore commit. Each
   conflict resolved by hand. Author and author date kept; committer
   Arpit Gupta (RECONCILE.md:18-31).
3. **Review.** A read-only adversarial pass: drop coverage over all 335
   fork-changed files, conflict resolutions, behavior survival of TIS,
   `advantage_scale`, token arrays by reference, and the SGLang seam
   (journal report 3).
4. **Images.** Base from the upstream recipe only
   (`docker/build.py --variant cu13-x86` at `23d41d711`); trainer from
   `examples/arena/Dockerfile` on a clean clone (RECONCILE.md:190-192).
5. **GPU tests** on prod-bom nodes (8 GPUs per node), with the queue,
   priorities, and node exclusions of the live runs. Harness: ConfigMap
   `recon-harness-20260927a` (the kdatp pattern), files at
   `/local/home/guparpit/recon-work/gpu/harness/` with copies in each S3 job
   dir (`recon-run.sh`, `recon_t1_forward.py`, `compare_t1.py`,
   `compare_t2.py`, `conv_overflow_check.py`, `recon_hooks.py`).
   - **T1**, one node, TP 8 with SP, EP 8, PP 1: load the r45 `iter_0000039`
     DCP, one `forward_only` log-prob pass on 4 real r45 rows (3,509 to
     6,538 tokens), one HF weight gather. Reference image
     `miles-glm53-r15-20260927a` (fork tip code).
   - **fla check**, one GPU: `ShortConvolution` with 24,576 channels,
     forward and backward, at 80,000 and 131,070 tokens.
   - **Warm-up**, one node: a 5-layer slice trains one step on the T2 rows
     and writes the kernel cache.
   - **T2**, eight nodes, the r45 layout (TP 8 with SP, PP 4, EP 16, DP 2;
     R3 and TIS on), train only, rollouts 40 to 42 from the r45
     `iter_0000039` DCP. Reference: the kdatp `baseline` arm.
   - Gate (the kdatp rule): relative L2 error at most 2 x floor + 2^-8, and
     mean log-prob difference at most 2 x floor + 1e-3. The floor is pass 1
     (one row per micro-batch) against pass 2 (4 rows packed) in the same
     image.

## Results

### Commit accounting

| Item | Value | Source |
| --- | --- | --- |
| Fork side over the merge base | 171 commits = 167 non-merge + 4 merge | `git rev-list --count 2799fe386..3028bc358` (re-verified) |
| Upstream side over the merge base | 750 commits | `git rev-list --count 2799fe386..23d41d711` (re-verified) |
| Result over `23d41d711` | 152 commits = 139 kept + 1 restore + 12 new | `git rev-list --count 23d41d711..fc93b6f23` (re-verified); RECONCILE.md:16 |
| Same patch-id as the fork | 131 | `git cherry`, RECONCILE.md:28-31 |
| Different patch-id | 36 = 28 drops + 3 partial keeps + 4 conflict picks + 1 context merge (`8989cfb49`) | RECONCILE.md:28-31 |
| Core code delta against upstream, outside the arena plugin, examples, tests, and `scripts/run_arena_harbor.py` | 15 files, +249 / -93 | `git diff --shortstat` (re-verified; the launcher adds one file, +398) |
| Pushed | `origin/arpit-reconcile-upstream` = `fc93b6f23`, 0 behind upstream | `git rev-parse` (re-verified) |

### Dropped commits and why

| Group | Count | Reason | Cover | Source |
| --- | --- | --- | --- | --- |
| Fork copy of PR #2786 (`5e986f1d9` to `dbbd610e7`) | 28 | Upstream merged the PR as squash `cc76e2391` on 2026-09-24. The core hunks are identical; the model files are the final PR version | `cc76e2391` | RECONCILE.md:35-83 |
| Merge commits `58a9f94d7`, `4b61e01d3`, `dc9ff3983`, `38921ae45` | 4 | A linear rebase removes merges. Only `38921ae45` added files of its own | restore commit `6c3ac0dd7` (GLM RUNLOG first 111 lines, ADR-0008, index row) | RECONCILE.md:86-95, 124-132 |
| File not restored: `scripts/models/glm5.3-flash-8layer.py` | 1 file | Upstream left it out; no arena config, script, or test uses it | none | RECONCILE.md:46-49 |

Partial keeps (the removed part exists upstream):

| Fork commit | Rebased | Kept | Removed | Source |
| --- | --- | --- | --- | --- |
| `bd8843d29` production cleanup sweep | `8ca11e758` | the `reloadable_process_group.py` hunk | 13 files, in or deleted by `cc76e2391` | RECONCILE.md:101 |
| `8ceca603d` asyncio driver + sidecar W&B resume | `9842f5cb6` | driver, `checkpoint_extras.py`, ADR-0005 | the `init_wandb_primary` hunk; upstream #3030 `8defbefa6` does the same and two copies ran the step twice | RECONCILE.md:102 |
| `f687e598e` anthropic import guards | `e2c5a657a` | the 66-line RUNLOG hunk | the guards; upstream #3114 `d2fc97ce5` | RECONCILE.md:103 |

Conflict resolutions: `setup.py` (keep upstream `e2b`, `modal` extras and
version 0.1.1, add `arena`); `hf_export.py` (take upstream #3198, move the
ENOTSUPP `shutil.copyfile` fix to `snapshot_publisher.py:92-102`, still
needed because `write_checkpoint_dir` deletes the export dir on one failed
copy); `megatron_utils/model.py` twice, net change zero after `b04f433e7`
(RECONCILE.md:113-122; journal report 3).

### New commits after the rebase

| Commit | Change | Source |
| --- | --- | --- |
| `a18a23903` | ADR-0016 (branch numbering) and amendment links in ADR-0005, 0006, 0008, 0011 | RECONCILE.md:138 |
| `bc0c843c6` | `train_async_arena.py` rebuilt on the upstream `train_async.py`; the removed imports were `create_rollout_manager`, `MainProcessIdentity`, `ft_utils.control_server`. ADR-0005 check stays at 0 deleted lines. Drain and heartbeat removal are one disposer callback `_finish_arena` | RECONCILE.md:139 |
| `f32dd8838` | `scripts/run_arena_harbor.py` calls `args.create_backend()` (#2432); argv gains `--deploy-component all` | RECONCILE.md:140 |
| `e8d4ae9a1` | gym weight versions move to `Sample.metadata["arena_weight_versions"]`; `Sample.weight_versions` stays empty (#1891 made it span objects) | RECONCILE.md:141 |
| `aed3522de` | advantage-scale tests set a one-rank parallel state (#3125) | RECONCILE.md:142 |
| `7466bc9cc` | `run_glm5_3_flash` launcher snapshot regenerated | RECONCILE.md:143 |
| `9a5b6216d`, `77cd880d4`, `fc93b6f23` | plugin RUNLOG entries, RECONCILE.md, test images and GPU validation | RECONCILE.md:144-149 |
| `c5b138782` | `patches/fla_kda_next_power_of_2.py` finds fla with `importlib.util.find_spec` and accepts the upstream patch form; tested on 4 file states | RECONCILE.md:146; journal report 4 |
| `4d6c39a7d` | `patches/fla_conv_int64_offsets.py`: the int64 lines of fla PR #1082 for `causal_conv1d` | RECONCILE.md:147 |
| `4316e4957` | `miles/utils/ray_utils.py` finds the head node with `ray.nodes()`, not the dashboard at 127.0.0.1:8265; `tests/fast/utils/test_ray_utils.py` (4 tests) | RECONCILE.md:148, 165 |

### Base image pins

Upstream starts each SGLang engine with `--gated-launch-port` (#2096) and
since #3031 (2026-09-26) stops when SGLang does not serve it. SGLang
`9a26e749` of `glm53next-upstream-20260902` does not serve it, so the branch
needs a new base (RECONCILE.md:163; journal report 4). The old-base build
failed at `_assert_launch_gate_served()` and was not pushed.

| Part | `glm53next-upstream-20260902` (live runs) | `recon-miles-base-20260927a` | Source |
| --- | --- | --- | --- |
| SGLang | `9a26e749` | `sglang-miles` `571212b6` (0.5.21.dev63, on `lmsysorg/sglang:v0.5.20`) | RECONCILE.md:196 |
| Megatron-LM | `e8f57451` | `miles-main` `f148a32b` (merge commit of PR #89) | RECONCILE.md:197 |
| fla | 0.4.2 | 0.5.2 + upstream KDA patch + `4d6c39a7d` | RECONCILE.md:198 |
| Megatron-Bridge | `7f0fb345` | `8cd3466d` | RECONCILE.md:199 |
| TileLang | 0.1.9 | 0.1.14 | RECONCILE.md:200 |
| torch / TE | 2.13.0+cu130 / 2.17.0 | the same | RECONCILE.md:201 |

Base recipe: `python3 docker/build.py --variant cu13-x86 --image-tag custom --custom-tag recon-miles-base-20260927a --build-arg SGLANG_COMMIT=571212b636baca45e10fa3b4da11a289123f3235 --build-arg MEGATRON_COMMIT=f148a32b4385b758b66a77c9c3ad1641f1295d4b --build-arg MILES_COMMIT=23d41d711f3b80544fda655898ed4f051ed644fe` (RECONCILE.md:191).

| Tag (`arena-slime-dev`, same digest in us-east-1 and ap-south-1) | Tree | Digest | Result | Source |
| --- | --- | --- | --- | --- |
| `recon-miles-base-20260927a` | upstream `23d41d711` | `sha256:f83c7828a75653e946ed3e2eb00a2ae6e8c85e511ef8a24a96d9b76e66140d13` | base of the three test tags | RECONCILE.md:185; journal report 4 |
| `miles-glm53-recon-test-20260927a` | `c5b138782` | `sha256:8f5fe3dc5715433d4a30dd87054882beff6c4b6d431651ef356103700363145e` | T1 pass; warm-up train step fails (fla int32) | RECONCILE.md:186 |
| `miles-glm53-recon-test-20260927b` | `4d6c39a7d` | `sha256:429d4ebffec4` (short) | warm-up pass; T2 fails at launch (head pinning) | RECONCILE.md:187 |
| `miles-glm53-recon-test-20260927c` | `4316e4957` | `sha256:2cd2c030c567` (short) | T2 pass | RECONCILE.md:188 |

All four tags are TEST ONLY. NEVER use them for a live run
(RECONCILE.md:181; `examples/arena/README.md` lineage rows 62-64 on the
branch).

### Upstream changes that alter a run

| Area | Change | Effect on an r44, r45, r46 config | Source |
| --- | --- | --- | --- |
| GLM-5.3 DSA indexer | final #2786 applies RMSNorm to the indexer query; the fork copy fed the raw query; SGLang `9a26e749` feeds the normalized query | the live runs carry this train/rollout mismatch in all 11 DSA layers; the fix changes top-k selection and log-probs | RECONCILE.md:155 |
| GLM-5.3 MLP | `scripts/models/glm5.3-flash.py` adds `--activation-func-clamp-value 10`; the checkpoint sets `swiglu_limit` 10 and SGLang clamps | a second mismatch goes away; trainer moves 7% closer to SGLang (T1 below) | RECONCILE.md:156, 236-238 |
| GLM-5.3 mHC | mean output contraction from a Megatron spec, not a monkeypatch | no checkpoint format change; T1 load proves it | RECONCILE.md:157 |
| Fault tolerance | with `use_fault_tolerance`, upstream turns on the mini FT controller on `api_server_port` 18080 (`arguments.py:3166-3169`); `--control-server-port` is gone | r44:362, r45:338, r46:370 set `use_fault_tolerance: true` and no config sets `mini_ft_controller_enable`, so a relaunch turns auto-heal on in silence. Set `mini_ft_controller_enable: false` for a like-for-like run | RECONCILE.md:158; journal report 3 |
| Event logger | `save_debug_event_data` defaults to `<save>/events` (#2505), no off switch. `update_weights` runs `check_weights(action="checksum")` on every engine at each update (`placement_group.py:316-338`); each save copies the events dir into the checkpoint (`rollout_executor.py:302`) | with `update_weights_interval: 1`, one checksum pass per rollout on every engine, and a copy that grows at each save. Not measured: the GPU jobs had no engines. Evidence that it is on: the T2 job wrote `events/main.jsonl` (172,453 B) and `events/worker_manager.jsonl` (161,187 B) | RECONCILE.md:159; journal report 3; S3 `recon/checkpoints/slime_experiments/recon-t2-baseline-20260927b/events/` |
| Metrics | `train_rollout_logprob_abs_diff` and `train_rollout_kl` come from trainer-scored log-probs (#3655) | values before and after the rebase are not directly comparable across a resume | RECONCILE.md:160 |
| Weight-version metrics | upstream `weight_version/*` reads span objects | arena samples carry no spans; these metrics are blank. `rollout/off_policy_round/*` does not change | RECONCILE.md:161; ADR-0016 decision 4 |
| TIS, PPO, R3 | math unchanged; R3 still sets `enable_return_routed_experts` | low; arena keeps `--sglang-moe-runner-backend auto` (r16 validated) | RECONCILE.md:162 |
| fla 0.5.2 | `causal_conv1d` offsets in int32 (fla PR #1062); fixed after 0.5.2 in PR #1082, no release | a row longer than 87,381 tokens stops the step with an illegal memory access; `4d6c39a7d` patches at image build | RECONCILE.md:164; fla check below |
| Head pinning | `compute_ray_pin_head_options()` runs in the `RayWorkerManager` actor and read the dashboard at 127.0.0.1:8265 | with `pin_rollout_manager_to_head: true` the 8-node launch stopped with `ServerUnavailable`; `4316e4957` reads the GCS node table | RECONCILE.md:165 |

### Resolved-argument changes (all five configs parse)

The argv gains only `--activation-func-clamp-value 10` and
`--deploy-component all` against the fork tip (RECONCILE.md:177).

| Setting | Fork tip | Reconciled branch | Source |
| --- | --- | --- | --- |
| `activation_func_clamp_value` | None | 10.0 | RECONCILE.md:177 |
| `mini_ft_controller_enable` | False | True | RECONCILE.md:177 |
| `save_debug_event_data` | None | `<save>/events` | RECONCILE.md:177 |
| Megatron `activation_func_clamp_shared_expert` | not present | True | journal report 4 (new base only) |
| `sglang_mamba_full_memory_ratio` | 0.9 | None | journal report 4 (new base only) |
| `sglang_swa_full_tokens_ratio` | 0.8 | None | journal report 4 (new base only) |

### CPU tests

| Check | Result | Source |
| --- | --- | --- |
| `tests/fast/plugins/arena` | 325 passed | RECONCILE.md:171 |
| `tests/fast/launch_scripts` | 49 passed, 1 failed (`test_workplace_backend`, also fails on upstream `main`) | RECONCILE.md:172 |
| manual launcher snapshots (`run_arena_harbor`, `run_glm5_3_flash`) | 6 passed | RECONCILE.md:173 |
| `tests/fast` sweep, branch vs upstream `main`, same venv | 6,545 vs 6,205 passed; 341 fork-only tests; one pass-to-fail (`test_docs_examples_matches_the_readmes`, fails on the fork tip too) | RECONCILE.md:174 |
| r15 image, tree on `PYTHONPATH` | Dockerfile import smoke and `run_arena_harbor.py --help` pass; 574 passed, 13 failed, the same 13 on upstream (12 need network, 1 `test_workplace_backend`) | RECONCILE.md:175 |
| at `4316e4957` | arena 325, `test_run_arena_harbor.py` 6, `test_ray_utils.py` 4; in image c 135 passed with the worker-manager tests | RECONCILE.md:176 |
| new recon image, network off | 22 `miles_plugins.arena.*` modules import; `glm5_next`, `dsa`, `kda`, `kpool_indexer` import; 211 engine imports, 7 fail inside `try`/`except` (old base: 10) | journal report 4 |

### GPU T1: one node, r45 `iter_0000039`

| Check | ref (r15 image) | new (image a) | Source |
| --- | --- | --- | --- |
| load | iteration 39, 528 s, 81.1 GiB and 43,490,708,478 parameters per GPU | the same values, 509 s | RECONCILE.md:220 |
| HF gather (weight sync and HF export) | 37,534 tensors, SHA-256 equal to r43 `hf/rollout_39` | 37,534 tensors, equal to r43 and to ref (`bitwise_equal_to_ref: true`, `gather_vs_ref_run.bitwise_equal: true`) | S3 `recon/t1/20260927b-compare.json` (re-verified) |

Log-prob differences on the 6,578 trainable tokens
(S3 `recon/t1/20260927b-lp_trainable.json`, re-verified):

| Comparison | mean abs | p99 | max |
| --- | --- | --- | --- |
| ref pass 1 vs ref pass 2 (floor) | 0.04584 | 0.4908 | 1.533 |
| new pass 1 vs new pass 2 (floor, clamp 10) | 0.04478 | 0.4663 | 1.294 |
| new, clamp off, vs ref | 0.03729 | 0.3658 | 1.028 |
| new, clamp 10 (default), vs ref | 0.04762 | 0.4862 | 1.506 |
| ref vs SGLang rollout log-probs | 0.05056 | 0.5337 | 1.842 |
| new, clamp 10, vs SGLang rollout log-probs | 0.04718 | 0.5173 | 1.692 |

The clamp moves the trainer 6.7% closer to SGLang
((0.05056 - 0.04718) / 0.05056). The SGLang log-probs come from the r43
policy at or before iteration 39, so they are an approximate reference
(RECONCILE.md:236-238).

Logits over 18,829 positions and the full vocabulary
(S3 `recon/t1/20260927b-compare.json`, re-verified):

| Arm | max abs diff | max rel diff | worst-row rel L2 | top-1 agreement | gate limit rel L2 | pass |
| --- | --- | --- | --- | --- | --- | --- |
| same-image floor (ref) | 26.66 | - | 0.2193 | - | - | - |
| new, clamp off, vs ref | 25.91 | 0.557 | 0.1917 | 92.1% | 0.4425 | yes |
| new, clamp 10, vs ref | 23.03 | 0.487 | 0.2283 | 89.5% | 0.4491 | yes |

Images b and c change only the `causal_conv1d` offsets and the head
lookup, so the T1 values also apply to them (RECONCILE.md:248-250).

### fla `causal_conv1d` overflow check (one GPU, image a)

`ShortConvolution`, 24,576 channels, forward and backward
(S3 `recon/convchk/20260927a-new/convchk.log` and `summary.json`, re-verified):

| Kernel | tokens | `offset_elems` | rc | rel L2 vs fp32 (y / dx / dw) | y SHA-256 |
| --- | --- | --- | --- | --- | --- |
| installed 0.5.2 | 80,000 | 1,966,080,000 | 0 | 1.67e-3 / 1.75e-3 / 1.74e-3 | `bec01fec...` |
| installed 0.5.2 | 131,070 | 3,221,176,320 | 1, `CUDA error: an illegal memory access` | - | - |
| patched (`fla_conv_int64_offsets.py`) | 80,000 | 1,966,080,000 | 0 | the same values | `bec01fec...` (bitwise equal to installed) |
| patched | 131,070 | 3,221,176,320 | 0 | 1.67e-3 / 1.75e-3 / 1.75e-3 | `161dcb38...` |

2^31 - 1 = 2,147,483,647; 87,381 x 24,576 = 2,147,475,456 is the last row
length below it. The patched kernel file is byte-equal to fla `31d15f75`
(journal report 5; `examples/arena/README.md` lineage row for image c).

Warm-up: image a stops in the first log-prob pass; image b passes in
46 min and writes the kernel cache (RECONCILE.md:259-262; not re-verified).

### GPU T2: eight nodes, r45 layout, train only, 3 steps

Both runs rc 0, no OOM (S3 `recon/t2/20260927c-compare.json`, re-verified).
`rollout/rollout_log_probs` is -0.8729434 in both, so both trained the same
rows. Reference: kdatp `baseline` arm (`kdatp-t2-20260927e`;
`examples/arena/harbor-rl-glm53-flash/kdatp/RESULTS.md:49-105` on
`arpit-glm-53`; S3 `recon/t2/kdatp-ref/results-20260927e-baseline.json`).

| Rollout | Metric | kdatp baseline | recon (image c) | Delta |
| --- | --- | --- | --- | --- |
| 40 | `grad_norm` | 0.12740 | 0.12634 | -0.8% |
| 40 | `loss` | -3.59e-6 | -4.77e-5 | - |
| 40 | `ppo_kl` | -8.5e-6 | 1.8e-5 | noise |
| 40 | `train_rollout_logprob_abs_diff` | 0.02721 | 0.02576 | -5.3% |
| 40 | step time (first step, kernel compile) | 1,687 s | 2,097 s | +24% |
| 41 | `grad_norm` | 0.12331 | 0.12435 | +0.8% |
| 41 | `loss` | -5.30e-4 | -5.45e-4 | - |
| 41 | `train_rollout_logprob_abs_diff` | 0.02741 | 0.02603 | -5.0% |
| 41 | step time | 1,278 s | 1,306 s | +2.3% |
| 41 | log-prob time / actor train time | 219 s / 989 s | 265 s / 1,015 s | +20.9% / +2.6% |
| 42 | `grad_norm` | 0.12154 | 0.13581 | +11.7% |
| 42 | `loss` | -1.86e-3 | -1.90e-3 | - |
| 42 | `train_rollout_logprob_abs_diff` | 0.02812 | 0.02683 | -4.6% |
| 42 | step time | 1,284 s | 1,315 s | +2.4% |
| 42 | log-prob time / actor train time | 229 s / 993 s | 270 s / 1,019 s | +17.9% / +2.6% |
| 40, 41 | peak allocated GiB, PP stages 0 / 1 / 2 / 3 | 128.79 / 129.72 / 124.72 / 124.90 | 136.79 / 137.72 / 132.7 / 132.9 | +8.0 each |

At one optimizer step per rollout, #3655 does not change
`train_rollout_logprob_abs_diff`, so the two trees are comparable. The recon
peak window starts at the log-prob pass; the kdatp window starts at the
`train` call (RECONCILE.md:289-294).

### Costs

| Cost | Value | Source |
| --- | --- | --- |
| Steady log-prob pass | +18% to +21% (219-229 s to 265-270 s) | T2 table |
| Steady step | about +2% | T2 table |
| Peak allocated memory | +8.0 GiB on each PP stage, 36 GiB still free | T2 table; journal report 5 |
| Source of the slowdown and the memory | not isolated | RECONCILE.md:304-306 |
| GPU test wall time | T1 about 19 min on 2 nodes (23:04Z-23:23Z); fla check about 3 min on 1 GPU; warm-up 46 min on 1 node; T2 image c about 87 min on 8 nodes (00:58Z-02:25Z) | S3 object times; RECONCILE.md:261 |

### ADR-0016 numbering collision

| Branch | File | Status | Date | Source |
| --- | --- | --- | --- | --- |
| `arpit-reconcile-upstream` | `miles_plugins/arena/adr/0016-reconcile-with-upstream-main.md` | Proposed | 2026-09-27 | worktree (re-verified) |
| `arpit-glm-53` | `miles_plugins/arena/adr/0016-shard-kda-across-tensor-parallel-ranks.md` | Accepted | 2026-09-28, commit `12fae498b` | main checkout (re-verified) |

The KDA ADR carries a "Numbering note" that names the collision and says:
renumber one of the two at the merge. Proposal of this study: the reconcile
ADR is Proposed and lives only on the unmerged branch, so it becomes 0017
when the branch rebases onto `arpit-glm-53`. The owner decides.

### KDA branches against the reconciled base

Every KDA branch sits on the old lineage. `git merge-base` with the
reconcile tip `fc93b6f23` is the old merge base `2799fe386` for all five
(re-verified 2026-09-28):

| Branch | Tip | Merge base with `3028bc358` | Note | Source |
| --- | --- | --- | --- | --- |
| `origin/arpit-kda-shared-layer` | `ab6069ac9` | `3028bc358` | 12 commits on the fork tip, 39 files, +4,309 / -182; the bottom 3 are `arpit-kda-tp-recompute` | journal report 6 |
| `origin/arpit-kda-shared-layer-rebased` | `389d26303` | `3028bc358` | on `arpit-glm-53` after the fork tip, not on the reconcile | `git merge-base` |
| `origin/arpit-kda-tp-recompute-rebased` | `4716a367a` | `3028bc358` | also merged into `arpit-glm-53` | `git log` |

A `git merge-tree` trial of the 12 shared-layer commits on the reconcile tip
finds 2 text conflicts (`kda.py`: upstream moved the `build_gdn_cp_context`
import to `miles_plugins.models.cp_utils`; `megatron_to_hf/glm5_next.py`:
final #2786) plus one at `3803da20d` on `update_weight/common.py`, which
upstream deleted and later KDA commits undo (journal report 6; not
re-verified). Both branches fix the fla int32 limit in a different place:
#3609 `short_conv` splits channels at 2^31 - 1; the reconcile patches fla.
Keep both until the shared layer passes 131,070-token rows on the new base.

## Verdict

The rebase is complete and sound as code. All 139 kept fork commits, the
drops, and the conflicts are accounted for; the CPU suites show no failure
that upstream does not share; T1 shows a bitwise-equal weight gather and a
log-prob difference inside the micro-batch packing floor; T2 trains three
steps with `grad_norm`, `loss`, and `train_rollout_logprob_abs_diff` within
the noise of the kdatp baseline and 5% lower on the log-prob gap. The
numbers support: GO for `arpit-reconcile-upstream` as the base for new
development and for more test-only jobs. The numbers do not support a live
run: no test ran an SGLang engine on `571212b6`, two defaults
(`mini_ft_controller_enable`, `save_debug_event_data`) change in silence,
the log-prob pass is 18% to 21% slower with 8 GiB more per stage, and
ADR-0016 (branch) stays Proposed. The DSA RMSNorm and clamp fixes change
log-probs and top-k, so the move belongs at a fresh run, not a resume
(journal report 6). r47 launched on `miles-glm53-r17-20260928a` (old
lineage, first-port KDA; `training-runs/harbor-rl-glm53-flash/r47/BUILD.md:5,16`),
so no live run uses the reconciled branch on 2026-09-28.

## Caveats and open items

Gates before live use (RECONCILE.md:296-309; journal report 6):

1. Engine-path GPU test: SGLang `571212b6` with the r45 engine flags, one
   weight update, the per-engine checksum time, the mini FT controller with
   real engines, and GPU memory with `sglang_mamba_full_memory_ratio` and
   `sglang_swa_full_tokens_ratio` now None.
2. Set `mini_ft_controller_enable` in every run config (r44, r45, r46, r47
   set none).
3. The owner accepts or changes ADR-0016 decision 4 (weight versions in
   `Sample.metadata`, `weight_version/*` blank).
4. Isolate or accept the log-prob slowdown and the +8 GiB.
5. Build a release (not `recon-`) base and trainer tag with the same recipe.
6. Fold in the `arpit-glm-53` commits after `3028bc358`: 13 on origin
   (`12fae498b`), 14 with the local layout commit `f3c7ea1ce`; the first,
   `41efa9d0a`, applied cleanly in the trial (journal report 6). Rebase the
   KDA shared-layer branch onto the result and re-run its T1 and T2 on
   fla 0.5.2 and TileLang 0.1.14.

Other open items:

- Move the base to a fla release that has PR #1082; then
  `fla_conv_int64_offsets.py` reports "already applied" and can go.
- Offer `4316e4957` (head lookup from the GCS) to upstream.
- Stale text on the branch (re-verified at `fc93b6f23`):
  `examples/arena/Dockerfile:16-18` still names the `glm53next` base;
  `examples/arena/harbor-rl-glm53-flash/RUNLOG.md:100` still names
  `glm5.3-flash-8layer.py`, which the branch does not ship.
- The events-dir copy into each checkpoint grows with the run
  (`rollout_executor.py:302`); the size over a 300-rollout run is not
  measured (journal report 3).
- Not re-verified: the warm-up times, the `git merge-tree` KDA trial, the
  upstream PR #3609 state (open, stacked on #3608, 367 commits behind
  `main`, last update 2026-09-24), the build logs, and the r45 gym
  `OOMKilled` restarts since 07:00Z 2026-09-27 (pre-existing, unrelated;
  journal report 6).
- The T1 SGLang reference log-probs come from an older policy, so the 6.7%
  clamp figure is approximate.

## Actions taken

- Branch `arpit-reconcile-upstream` at `fc93b6f23`, pushed to origin;
  `arpit-glm-53` not rewritten.
- ADR-0016 (branch), `examples/arena/RECONCILE.md`, two plugin RUNLOG entries
  (`miles_plugins/arena/RUNLOG.md:1081-1140` on the branch), and three
  TEST-ONLY lineage rows in `examples/arena/README.md` (rows 62-64).
- Base image `recon-miles-base-20260927a` and test images
  `miles-glm53-recon-test-20260927{a,b,c}` in `arena-slime-dev`, both
  regions.
- Core fixes `4d6c39a7d` (fla int64 offsets) and `4316e4957` (GCS head
  lookup), each with a runnable check.
- All `recon-*` jobs, the two helper jobs, and the ConfigMap deleted; a
  read-only scan found no leftover object (journal report 6).
- No live run changed. r47 launched on the old lineage.

## Sources

- Journal `wf_3ff0709b-51c`: reports 1 (inventory), 2 (rebase), 3 (review),
  4 (image), 5 (test), 6 (report).
- `examples/arena/RECONCILE.md` and
  `miles_plugins/arena/adr/0016-reconcile-with-upstream-main.md` at
  `fc93b6f23` in `/workplace/guparpit/miles-reconcile` (line numbers above).
- `miles_plugins/arena/adr/0016-shard-kda-across-tensor-parallel-ranks.md`
  (`12fae498b`) and `examples/arena/harbor-rl-glm53-flash/kdatp/RESULTS.md`
  on `arpit-glm-53`.
- S3 `s3://arena-scratch-prod-bom-ap-south-1/guparpit/recon/`:
  `t1/20260927b-compare.json`, `t1/20260927b-lp_trainable.json`,
  `t2/20260927c-compare.json`, `t2/kdatp-ref/results-20260927e-baseline.json`,
  `convchk/20260927a-new/{convchk.log,summary.json}`, `convchk/20260927b-new/`,
  `checkpoints/slime_experiments/recon-t2-baseline-20260927b/events/`.
- Git on both checkouts, re-verified 2026-09-28: `rev-list --count`,
  `merge-base`, `diff --shortstat`, `log`.
