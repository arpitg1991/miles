# miles_plugins/arena — run log

Chronological record of how the arena RL training path (AGISlime's
`amzn_agi_slime`, an overlay on vendored THUDM/slime 0.3.0) was ported onto
miles and then exercised: what was tried, what happened, what was decided and
why. It exists so the git history reads as an investigation rather than a code
drop. Per-job run logs (launches, images, rewards, RCAs) live next to each
example in `examples/arena/<job>/RUNLOG.md`; decisions with lasting effect are
ADRs in `adr/`. Evidence too large for the repo is archived under
`/workplace/guparpit/miles/arena-port-artifacts/` (analysis, impl and verify
reports; the e2e harnesses).

## Format

- One `## <date> (PT) — <milestone or run>` section per event, appended in
  the order the code landed (commit order, which is chronological for the
  milestones themselves); a later fix-up to a module already logged is a
  `###` subsection of that module's milestone. Earlier sections are never
  rewritten; a correction is a new section that references the old one.
- All times are Pacific Time. Sources (file mtimes, pod logs, W&B) are UTC;
  PDT = UTC-7.
- Each section states what was tried, what happened (numbers only when a
  source has them), the root cause where known, the decision with the
  alternatives rejected, and the run identity (EXPERIMENT_NAME, image tag,
  W&B group) whenever something was launched.
- Source pointers: `arena-port-artifacts/reports/<name>.md` for the archived
  reports, `adr/NNNN` for decisions.

## 2026-08-31 → 2026-09-01 (PT) — Analysis phase: what has to move, where miles differs

Goal: run the arena/harbor RL training path — first target the Qwen3.5-27B
financeagent smoke (`tmp_staging/smoke/harbor-rl-27b` in AGISlime) — on the
miles fork instead of AGISlime. Fork HEAD 2799fe38 (= upstream radixark/miles,
2026-08-31 20:17 PT). Source studied: `/workplace/guparpit/miles/src/AGISlime`
at `origin/mainline-0.3.0` 90ce39f (2026-08-28 13:36 PT), package
`src/amzn_agi_slime/amzn_agi_slime`: 7,667 lines + 2,139 test lines, 10
top-level modules plus `nats_arena/` (11 modules). The snorkel-run customs on
the harbor-workspace branch `rl-snorkel-r4-removed-sample-replacement`
(637d714, b8aae83) were not part of this analysis. The reports carry no
timestamps; the phase is bounded by the fork HEAD date and the first plugin
file (00:56 PT, next section).

Six reports, archived at `arena-port-artifacts/reports/`:

| Report | What it established |
| --- | --- |
| `harbor.md` | The wire contract's single source of truth is AREnATasks `amzn_arena_contract.rl`; the trainer side is a declared mirror pinned by value in `tests/harbor_runtime/test_cross_runtime_parity.py` (TestTrainerMirrorPins). Streams `ARENA_TASKS`/`ARENA_RESULTS`, subjects `arena.tasks.<gym>`/`arena.results`, durable `slime-trainer` (ack_wait 3600, max_deliver 3), gzip-JSON task/result envelopes, salvage statuses `success`/`truncated`. |
| `overlay.md` | The overlay is strictly additive over pristine slime 0.3.0 and plugs in through five seams (driver module, `--rollout-function-path`, `--data-source-path`, `--eval-function-path`, `--custom-reward-post-process-path`); its driver is stock `train_async.py` plus four insertions. CRITICAL GAP: `eval_metrics_drain.drain()` is defined but never called trainer-side. |
| `milesApi.md` | All 14 slime APIs the overlay imports exist in miles, with drift: asyncio driver; `Sample.rollout_id` is readable where vendored 0.3.0 had renamed it `group_id` (read raises); tracking API moved; rollout-fn v2 protocol with an `add_arguments` hook; `init_wandb_primary` ignores a pre-set `wandb_run_id`. HARD BLOCKER: `--loss-mask-type` choices lack `qwen3_5` and `MultiTurnLossMaskGenerator` has no `qwen3_5` generator; the smoke YAML sets it. |
| `runtime.md` | Launch chain is `entrypoint.sh` + hydra_converter → `ray job submit ... python3 -m amzn_agi_slime.train_async_arena`; slime's 3-phase parse silently DROPS unknown flags where miles errors; pod-start pip installs include `kubernetes==35.0.0` because 36.x returns 401 on EKS; `--use-gated-attention` exists only in AGISlime's patched Megatron. |
| `tests.md` | Two test layers: 10 build-gated stdlib self-checks + 135 manually-run pytest tests (102+19+6+5+3) = 145; no `sys.modules` stubs; `pytest.importorskip("slime")` gates the torch-touching classes. |
| `testEnv.md` | A CPU-only miles test environment on this GPU-less host (`/tmp/miles-venv`, three-way PYTHONPATH mirroring the hosted CPU CI): `tests/fast/utils` 3210 passed / 16 skipped, `test_arguments.py` 148 passed; `parse_args` runs end-to-end on CPU, so argv validation is possible offline. |

Blockers carried into the port and where each was resolved:

| Blocker | Resolution |
| --- | --- |
| `loss_mask_type: qwen3_5` rejected by miles argparse; no generator | Core edit to `mask_utils.py` / `arguments.py` (next milestone). |
| The sync `ray.get` driver cannot be mechanically ported (`async_train` gone, `generate` returns a dict pack, `update_weights` needs rollout-manager cooperation) | Driver rebuilt on miles `train_async.py`'s asyncio loop (driver milestone, adr/0005). |
| Group identity: vendored `group_id` vs miles `rollout_id` — a naive rename sets a dead attribute silently | adr/0003 (nats_rollout milestone). |
| Eval-metrics drain never wired | Wired in the driver milestone; drain semantics fixed below. |
| kubernetes 36.x → 401 on EKS | Pinned `==35.0.0` in the `arena` extra (this milestone). |

Decision taken here: port as one plugin package with a fixed module mapping and
minimal core edits — adr/0001.

## 2026-09-01 00:56–01:07 PT — Milestone: package skeleton and torch-free support modules

What: `miles_plugins/arena/__init__.py` and `nats_arena/__init__.py` (00:56),
`rewards.py`, `parsers.py`, `s3_artifact.py` and the `setup.py` `arena` extra
(00:57), `rollout_metrics.py` (01:06), `logging_extensions.py` (01:07), and
the first copy of `eval_metrics_drain.py` (rewritten at 02:42, fix-up below).
`checkpoint_extras.py` and `wandb_extensions.py` were written in the same pass
(the impl report lists 8 support modules) but are not in this commit:
`checkpoint_extras` lands with the driver, `wandb_extensions` was deleted in
verification (driver milestone). The package README drafted in this pass lands
with the first example because its usage section points at `examples/arena/`.

Fidelity vs 90ce39f after normalising `amzn_agi_slime` → `miles_plugins.arena`
and `slime.utils.types` → `miles.utils.types` (re-verified with `diff` while
preparing this commit):

| Module | Changed lines | Nature |
| --- | --- | --- |
| `parsers.py`, `s3_artifact.py`, `logging_extensions.py` | 0 | byte-identical |
| `__init__.py` | 4 | docstring; "hosts without torch (which rewards requires)" → "without the full training stack" — the original claim was stale, `rewards` is torch-free by design |
| `rewards.py` | 6 | docstrings only; executable code identical |
| `rollout_metrics.py` | 2 | docstring only |
| `eval_metrics_drain.py` | 68 (191 → 173 lines) | the one rewrite, next section |

`setup.py`: `extras_require["arena"] = ["nats-py>=2.6.0", "kubernetes==35.0.0",
"lakefs", "boto3"]`, `install_requires` untouched; each is imported lazily so
the plugin imports without them. `find_packages(include=["miles*",
"miles_plugins*"])` already discovers the new subpackages; miles plugins are
referenced by dotted path, no entry-point registration.

Verified (impl-support report, `/tmp/miles-venv`):

- all modules import; PEP 562 lazy access resolves `binarize_reward`,
  `register_nova_reasoning_parser`, `split_reasoning` without importing
  `rewards` up front; an unknown attribute raises `AttributeError`;
- `binarize_reward` on 4 real `miles.utils.types.Sample` objects (rewards
  1.0/0.5/0.2/0.1, `n_samples_per_prompt=2`, std normalisation) → binary
  `[1,0,0,0]`, processed `[+0.7071, -0.7071, 0.0, 0.0]`, matching hand-computed
  GRPO normalisation;
- `register_nova_reasoning_parser()` against the real sglang tree returns True
  and `nova` appears in `ReasoningParser.DetectorMap`;
- `get_eval_metrics_dir` anchors on `args.save` and returns None for an
  `s3://` `save_hf` fallback; `is_s3_uri` / `parse_s3_uri` edge cases pass;
- `setup.py` parses and the extra follows the existing extras style.

Open at this point: `drain()` still had no caller anywhere (the overlay.md
gap) — assigned to the driver owner; `examples/arena/` did not exist yet.

### 2026-09-01 02:42 PT — Fix-up: `eval_metrics_drain` commits every step file

Finding (verify-redundancy report, severity major): the byte-faithful copy kept
AGISlime's protocol — flush every queued `step_*.json` with `commit=True`
except the last, staged with `commit=False` so its keys "piggyback onto the
trainer's very next commit". That held in gym-evals, where the drain ran at the
top of `log_perf_data_raw` in the process that logs train metrics. In miles
the drain runs in the driver, and the driver makes no other `wandb.log` calls
(train/rollout/perf metrics come from the megatron actors and the rollout
manager as shared-mode secondary writers). Consecutive staged files share keys
(`eval/step`, `eval/<dataset>/...`), so later ones would overwrite earlier ones
and a single merged row would land at shutdown — every intermediate eval point
lost.

Fix: `drain()` now logs every file with `wandb.log(payload, commit=True)`;
`final=True` is a log annotation only. Row placement on wandb's `_step` axis is
irrelevant because `eval/*` plots use `eval/step` as their x-axis
(`wandb.define_metric("eval/*", step_metric="eval/step")`). 191 → 173 lines;
the `.trainer_alive` heartbeat, queue-file naming and the WARNING-on-skip
behaviour are unchanged.

Rejected for now and recorded as the follow-up in the module docstring: retire
the file queue and heartbeat protocol entirely by making the eval coordinator a
wandb shared-mode secondary writer (`init_wandb_secondary` pattern) that logs
`eval/*` directly. Also noted: `remove_trainer_alive()` has no caller — it had
none in AGISlime either; the coordinator's flush keys off 600 s heartbeat
staleness, so the missing call only delays a flush.

## 2026-09-01 00:56 PT — Milestone: `qwen3_5` loss mask ported into miles core

**Scope:** `miles/utils/mask_utils.py`, `miles/utils/arguments.py`, `tests/fast/utils/test_loss_mask_qwen3_5.py`
**Source of truth:** AGISlime `vendor/slime/0.3.0` (vendored 2026-06-23 PT, commit 5cc7905 "Vendor THUDM/slime v0.3.0"); the tree already carried `gen_multi_turn_loss_mask_qwen3_5` at vendoring time, so "0.3.0" is post-0.3.0 here (milesApi.md "baseline confusion" note: diff against this tree, never the sibling upstream slime clone)
**Env:** `/tmp/miles-venv`, CPU-only host, the 3-way PYTHONPATH recipe from testEnv.md (miles repo root + sglang `sglang-miles` python/ + Megatron-LM `miles-main`)

### Why (the blocker)

- The analysis pass (milesApi.md, 2026-08-31/09-01 PT) listed one **HARD BLOCKER** for the harbor-rl-27b smoke job: `financeagent-27b-smoke.yaml` sets `loss_mask_type: qwen3_5`, but at the fork point (upstream 2799fe38, 2026-08-31 20:17 PT) miles' `--loss-mask-type` choices were `['qwen', 'qwen3', 'distill_qwen']` (arguments.py:2404-2410) and `MultiTurnLossMaskGenerator.get_loss_mask` had no `qwen3_5` branch (mask_utils.py:133-144). Strict argparse rejects the value before any plugin hook runs.
- Upstream miles had not touched `mask_utils.py` since 2026-04-09 (ef228e64, transformers>=5.0 chat-template fix); no ref in the fork carries a `qwen3_5` generator.
- Consumers that need the generator: the NATS rollout slow path (`nats_rollout.py` `MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=getattr(args, "loss_mask_type", None))`, reached when a trajectory's last step lacks `has_generate_tokens`), miles' own `miles/rollout/sft_rollout.py:42`, and both 27B configs (`harbor-rl-27b/financeagent-27b-smoke.yaml:39`, `harbor-rl-27b-snorkel/snorkel-27b.yaml:69`).

### What was done (files written 00:56-00:57 PT)

| Change | Detail |
|---|---|
| `mask_utils.py` (+77/-1) | `gen_multi_turn_loss_mask_qwen3_5` ported verbatim: render with `tokenize=False`, tokenize with `return_offsets_mapping`, char-mask every `<|im_start|>assistant\n` ... `<|im_end|>`(+`\n`) span, skip the `<think>\n` prefix, honour `step_loss_mask`, cross-check against `apply_chat_template(tokenize=True)`; `qwen3_5` branch in `get_loss_mask` |
| `mask_utils.py` `__init__` | vendored guard adopted: `system_message_length`/`gen_token_length` default 0, `get_system_message_length()` only for `tokenizer_type in ("qwen", "qwen3")` |
| `arguments.py` (1 line) | `choices=["qwen", "qwen3", "qwen3_5", "distill_qwen"]` — vendored order |
| new test (3 tests) | port of vendored `tests/utils/test_loss_mask_type_qwen35.py`; self-contained char-level `FakeQwen35Tokenizer` (no HF download, runs for real); `register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])` prepended, import switched to `miles.utils.mask_utils`; bodies otherwise identical |

### Decisions and alternatives

- **Patch core, not the plugin.** milesApi.md offered two routes: patch miles (choice + generator) or route masking through the plugin. The choice list is enforced by core argparse at parse time and `sft_rollout.py` also constructs the generator, so a plugin-only route could not make `loss_mask_type: qwen3_5` parse; this is the "core edit only where miles has no seam" case of ADR-0001.
- **Byte-faithful port.** The only textual deviation inside the new method is joining the black-wrapped two-literal `ValueError` message into one literal (identical runtime string). Miles' pre-existing kwarg ordering in `gen_multi_turn_loss_mask_qwen/qwen3` (`return_dict=False, tools=tools` vs vendored `tools=tools, return_dict=False`) was deliberately left alone: keyword args, no semantic difference, out of scope.
- **Why the `__init__` guard.** `get_system_message_length()` does a two-match unpack (`idx_1, idx_2 = find_all_sublist_indices(...)`) that is not guaranteed under the Qwen3.5 template (it injects `<think>` into the generation prompt), so computing it unconditionally could raise at construction for a `qwen3_5` tokenizer. Named behaviour change vs current miles: `distill_qwen` (and any non-qwen/qwen3 type) no longer computes it; nothing outside `mask_utils.py` reads those attributes (grep over `miles/`, `miles_plugins/`, `scripts/`), and the `distill_qwen` path never did, so mask outputs are unchanged.
- **Explicit CI registration** with `est_time=5` follows the `test_processing_utils.py` convention; the neighbouring `test_mask_utils.py` relies on implicit directory registration.

### Verification (2026-09-01 PT, `/tmp/miles-venv`)

- `pytest tests/fast/utils/test_loss_mask_qwen3_5.py tests/fast/utils/test_mask_utils.py` -> 5 passed in 1.6 s (3 new + 2 existing).
- `pytest tests/fast/utils` (whole tree) -> **3213 passed, 16 skipped** in 105 s; baseline from testEnv.md was 3210 passed / 16 skipped, so exactly +3 and no regressions. Re-confirmed unchanged by the later regression sweep (verify-regression.md).
- `pytest tests/fast/utils/test_arguments.py` -> **148 passed** (the choices edit breaks no argument test).
- `pytest tests/ci/test` -> 678 passed, 1 skipped (CI registration policy accepts the new `register_cpu_ci` call); `run_suite.py --hw cpu --suite stage-a-cpu --list-only` discovers the new file at est 5.0 s.
- Parse probe: `parse_args([... '--train-backend', 'megatron', ..., '--loss-mask-type', 'qwen3_5'])` -> `loss_mask_type == 'qwen3_5'`; `import miles.utils.mask_utils` exposes `gen_multi_turn_loss_mask_qwen3_5`.
- `ruff check` clean on all three files; `black --check --line-length 119` reports `mask_utils.py` and the test unchanged.

### Open / follow-on

- Not exercised against a real Qwen3.5 HF tokenizer on this host (only the fake tokenizer plus the existing Qwen/Qwen3-8B `qwen3` tests). The path requires a fast tokenizer (`return_offsets_mapping`) and raises a clear `ValueError` otherwise, same as vendored. The 2026-09-02 GPU smoke (`rl-milesgb1-smoke1`, snorkel-27b config with `loss_mask_type: qwen3_5`) parsed the flag on a real run; whether the slow re-tokenize path fired there is not recorded (the fast path takes gym-provided masks).
- The multi-turn test documents why `qwen3` is wrong for Qwen3.5 full-text renderings: it rebuilds each assistant turn in isolation and fabricates a `<think>` block for earlier assistant answers; `qwen3_5` supervises every assistant turn as rendered.
- No GLM mask type exists. The later GLM-5.3-Flash config omits `loss_mask_type` and relies on `use_rollout_logprobs: true` to remove tokenizer-fallback samples; that dependency is recorded where the config lands.

## 2026-09-01 00:59-01:08 PT — Milestone: nats_arena wire format, controllers, rewards, data source

Commit `feat(arena): port NATS wire format, controllers, data source, rewards`.
Baseline: AGISlime clone `/workplace/guparpit/miles/src/AGISlime`, branch
`mainline-0.3.0` @ 90ce39f (2026-08-28 13:36 PT), cloned 00:13 PT. Reports:
`arena-port-artifacts/reports/{impl-dataSource,impl-natsRollout,verify-dataEval,verify-natsGrpo}.md`.

| File (lines) | Written PT | Delta vs original |
| --- | --- | --- |
| `nats_arena/mixture_controller.py` (209) | 00:59 | byte-identical (stdlib only) |
| `nats_arena/rollout_timing_tracker.py` (226) | 00:59 | byte-identical (stdlib only) |
| `nats_arena/gym_autoscaler.py` (667) | 01:00 | import path; `kubernetes==35.0.0` rationale (36.x returns 401 on EKS) moved from AGISlime `entrypoint.sh` into the `_get_k8s_apps_client` docstring |
| `nats_arena/reward_binary.py` (219) | 01:00 | `slime.utils.types` -> `miles.utils.types`; two docstring words |
| `nats_arena/message_format.py` (182) | 01:06 | 16 diff lines, all docstring; every wire value unchanged (ADR-0002) |
| `nats_arena/data_source.py` (557) | this window; fix-up 02:37 | import swaps (`read_file` monkeypatch retargeted to `miles.utils.data.read_file`); `_MultiGymDatasetView` + `self.dataset`; `chat_template_path` pass-through |

Decisions

- lakeFS pin URI must end in `manifest.jsonl`: `_resolve_manifest_path` pulls
  exactly the object a `lakefs://` URI names and appends `manifest.jsonl` only
  for LOCAL directories, so the bare `KNOWN_GYMS` variant dir 404s at trainer
  startup. Established on the real-lakeFS snorkel e2e (09:31 PT; 2,922
  prompts from `lakefs://arena-inspect/dev/internal/snorkel-general-bash-harbor/ecr-20260823/manifest.jsonl`,
  `AWS_PROFILE=lakefs-arena-gym`); baked into the snorkel/glm53 configs and
  README. Resolver kept as-is (parity) rather than auto-appending.
- `_MultiGymDatasetView` exposed as `.dataset`: miles
  `RolloutManager.get_num_rollout_per_epoch` reads `len(data_source.dataset)`
  when `--num-rollout` is unset (slime 0.3.0 used `len(data_source)`). Sums
  prompts across gyms, buffer excluded; verified through the real unbound
  method (2-gym / 6-prompt manifest -> 3). Rejected: mandatory
  `--num-rollout`. No `close()` added, `__len__` kept.
- Buffer NOT persisted (parity): state keys are exactly `{per_gym, weights,
  sample_group_index, sample_index, metadata, timing_tracker}`, zero hunks in
  save/load vs original; groups buffered at checkpoint time are lost on
  restart. Recorded as a known limitation, not fixed.
- `chat_template_path` pass-through (only behavioural fix-up here, 02:37 PT,
  from the verify-dataEval note): stock `RolloutDataSource` passes it to
  `load_tokenizer`, the original did not. No harbor-rl config sets it.
- `graded_reward` / `graded_renormalize_after_mask` carried: added to
  `mainline-0.3.0` by 90ce39f; absent from the harbor-workspace line
  (637d714/b8aae83, 2026-08-27 PT) that carried `removed_sample_replacement`.
  No harbor-rl config (27B smoke, snorkel, glm53) sets
  `custom_reward_post_process_path`, so miles-native group normalisation runs
  in every run; kept as the tested menu.
- Autoscaler / mixture / timing tracker inert in every run (`gym_autoscale:
  false` in all three example configs, `gym_mixture_targets` never set).
  Kept: miles has no weighted multi-dataset source or backlog autoscaler
  (verify-redundancy KEEP). Parity warts preserved: `--gym-autoscale-auto-tune`
  is `store_true` default True (YAML `false` is a no-op); `GymAutoscaler`
  headroom default 1.25 vs `nats_rollout` getattr 2.0 (getattr wins).
- Wire values byte-identical -> ADR-0002. impl-natsRollout claimed the
  `slime-trainer` durable is pinned by the AREnATasks parity test;
  verify-natsGrpo showed only worker-side values are. The durable stays for
  JetStream consumer-state continuity, not because of a test.

Verification (CPU, `/tmp/miles-venv`, 2026-09-01)

- Imports + `compileall` clean; `reward_binary` numeric parity vs the original
  on vendored slime 0.3.0 (`binarize_reward`, `renormalize_after_mask`,
  `graded_reward`, `metadata['binary_reward']`): bit-identical.
- Data source: 3-row manifest + Qwen3-0.6B tokenizer — 3:1 weights ->
  `{financeagent: 3, othergym: 1}`; dedup_key `<iid>-e<epoch>`; epoch wrap +
  reshuffle; `save(7)`/`load(7)` restores offsets/epochs/indices/consumed
  ids/timing state/weights exactly. Controllers: gating/EMA/clamp,
  percentile/winsorize edges, dry-run scale-up 2 -> 20 clamped, RBAC and
  active-work guards, state round-trips.
- e2e against real NATS 2.14.6 + fake gym worker: financeagent harness 49/50
  (the 1 fail is a teardown ack artifact identical in AGISlime); snorkel
  plain-stack harness 145/0 (resume restored offset 110 + 12 dedup keys;
  durable `slime-trainer` explicit/max_deliver 3/ack_wait 3600 asserted).

Left as-is (inherited): resume dedup guard mostly vacuous (publisher runs
ahead — epochs `{financeagent: 6}` after 2 trained e2e rollouts); ruff UP028
at `data_source.py:160` exists identically upstream (miles ruff excludes
`miles_plugins/**`); `prompt_data_list` list-vs-JSON concern is moot — the
launcher remaps `prompt-data-list` to `--prompt-data` (str), `json.loads`'d
by `_resolve_gym_configs`.

## 2026-09-01 ~01:00 PT — Milestone: NATS rollout hot path ported (`nats_arena/nats_rollout.py`)

Source: AGISlime `src/amzn_agi_slime/amzn_agi_slime/nats_arena/nats_rollout.py`
from the `mainline-0.3.0` checkout at `/workplace/guparpit/miles/src/AGISlime`
(1698 lines; last changed there in `5f7cf5f`, 2026-08-19). Sibling files of
the same porting task carry 01:06-01:07 PT mtimes (`message_format.py`,
`rollout_metrics.py`, `logging_extensions.py`). The file lands at its
verified 2026-09-01 state = integration-tree snapshot `7182346d7`
(1819 lines, committed 2026-09-02 00:55 PT); the two salvage flags added on
2026-09-02/03 follow in their own commits.

- Ported: `NATSRolloutWorker` (daemon asyncio thread, JetStream publish /
  collect / dedup / DLQ sweep), `_result_to_samples_full_trajectory` (fast
  path from gym `token_ids`/`loss_mask`/`log_probs`, slow path via
  `MultiTurnLossMaskGenerator`), `_process_group`, `generate_rollout`.
- Import swaps only: `slime.utils.{types,mask_utils,misc,metric_utils}`,
  `slime.rollout.filter_hub.base_types` -> `miles.*`;
  `slime.utils.logging_utils.log` -> `miles.utils.tracking_utils.tracking.log`
  (same `(args, metrics, step_key)` shape, still gated on `args.use_wandb`);
  `amzn_agi_slime.*` -> `miles_plugins.arena.*`.
- Wire values untouched (ADR-0002): `ARENA_TASKS`/`ARENA_RESULTS`, durable
  `slime-trainer`, `ack_wait 3600` / `max_deliver 3`, `max_msg_size 134217728`,
  `SALVAGEABLE_RESULT_STATUSES = ('success', 'truncated')`, `.g<counter>.`
  task-id suffix, degenerate agent-stop set, synthetic-slot skip + sibling pad.
- The one semantic adaptation: `s.group_id = gid` -> shared
  `s.group_index = gid` + unique `s.index = gid * n_per_prompt + i`,
  `rollout_id` left `None` (ADR-0003). The mechanical `rollout_id = gid`
  mapping was tried first and raised
  `all samples in rollout 99 must share one reward` on rewards `[1, 0, 1, 0]`.
  Loss weighting thereby becomes per-trajectory (ADR-0004).
- New `_add_arena_arguments` attached as `generate_rollout.add_arguments`
  (the hook miles' strict `parse_args` auto-invokes for
  `--rollout-function-path`): `--arena-sample-mode`,
  `--dynamic-sampling-max-examine-mult` (4.0), `--gym-mixture-targets`,
  `--mixture-adjustment-interval` (60) / `--mixture-smoothing` (0.3),
  `--k8s-namespace`, the `--gym-autoscale*` family (interval 30, cooldown 180,
  warmup 1800, headroom 2.0, deficit-alpha 0.5, growth 1.3 / 1.25,
  profile-duration 1800, min-samples 50, windows 100 / 1000, `auto-tune`
  default `True`). A `parse_args` probe confirmed every default equals its
  `getattr` fallback in value and type; the `getattr` + env fallbacks stay for
  `MILES_USE_LEGACY_ROLLOUT_V1=1`, where the hook is skipped. Under the
  vendored megatron parser these knobs were silently dropped and the
  defaults always won; on miles a YAML value now takes effect.
- Not carried: the lazy `removed_sample_replacement` import
  (`ARENA_NATS_REPLACE_REMOVED_SAMPLES`; AGISlime harbor-workspace revision
  `b8aae83`, 1800 lines). The r3b-r5 survivor-normalisation custom is
  descoped with the plain-stack decision (ADR-0007); only stale `__pycache__`
  entries of the module and its test remain in the working tree.
- Kept as-is (parity, not port defects): ruff's 4 findings (3x F841
  import-guard fallbacks, 1x B905 `zip`) match the original;
  `--gym-autoscale-auto-tune` cannot be switched off from YAML; the blocking
  `output_queue.put` inside the event loop.
- Checks at port time (CPU venv): imports in normal and legacy mode;
  hunk-by-hunk diff vs the original; `_result_to_samples_full_trajectory`
  cases (fast path, synthetic skip, length truncation, degenerate stop,
  bad logprobs, context overflow); `_process_group` pad + stamping; two
  groups through `postprocess_rollout_data` + `convert_samples_to_train_data`
  (per-group normalised rewards `[0.866, -0.866, ...]`, per-trajectory
  `rollout_mask_sums`); `generate_rollout` through `LegacyRolloutFnAdapter`
  with a stubbed worker queue (4 groups x 2 samples); full `parse_args`.

### 2026-09-01 02:34-02:58 PT — verification fix-ups folded into `nats_rollout.py`

Two findings of the adversarial verification wave were fixed in place (the
wave itself is logged with the verification-hardening commit):

- **`rollout/truncated_ratio` -> `rollout/truncated_ratio_prefilter`**
  (verify-redundancy, major). The plugin logged the ratio over the
  pre-filter population (`all_data`, incl. groups dropped by the
  dynamic-sampling filter) while miles-native `log_rollout_data` logs a
  post-filter `rollout/truncated_ratio` at the same `rollout/step` — the only
  colliding key across both metric sets; it would zigzag the chart on any
  wandb-enabled run. The collision pre-dated the port (slime 0.3.0
  `rollout.py:1243` vs AGISlime `nats_rollout.py:1616`) but fires identically
  on miles. Decision: rename the plugin key, keep both populations, comment
  at the log site.
- **Slow-path zero-fill under `use_rollout_logprobs`** (verify-e2e, minor).
  Messages-only trajectories left `rollout_log_probs = None`; a mixed
  fast/slow group produced rows `[20, 20, 20, None]` and a `TypeError` at
  tensorisation (slow-first: field silently dropped for the batch).
  Decision: zero-fill to `response_length`, `ABORTED`, `remove_sample`,
  reason `no_logprobs` (ADR-0004).

Left open: blocking `output_queue.put` (e2e teardown showed `ack_pending=4`,
identical to AGISlime; production drains every step and
`ack_wait 3600` / `max_deliver 3` redeliver); per-trajectory weighting
sign-off (ADR-0004); no regression test yet for the stamping (added in the
verification-hardening commit). No GPU run had exercised this file yet: the
first exercise was the CPU e2e harness (real NATS 2.14.6 JetStream + fake
financeagent gym worker, 49/50 checks), then `rl-milesgb1-smoke1` on
2026-09-01 (see the snorkel example run log).

## 2026-09-01 01:04-01:08 PT - Milestone: hosted eval path ported (eval_rollout, eval_coordinator, argo_eval_trigger)

`argo_eval_trigger.py` written 01:04:53 PT, `eval_coordinator.py` 01:08:40 PT, `eval_rollout.py`
in the same window and re-saved 02:37:16 PT with the fix-up below. Source: AGISlime
`src/amzn_agi_slime/amzn_agi_slime/nats_arena/` at the port baseline 90ce39f
(`mainline-0.3.0`; the three files are byte-identical at the harbor-workspace
b8aae83, 2026-08-27 PT), mapping rule `amzn_agi_slime.X -> miles_plugins.arena.X` (ADR-0001). Records: `arena-port-artifacts/reports/
impl-evalPath.md`, `verify-dataEval.md`, `verify-redundancy.md`, `verify-driverLauncher.md`.

| Module | Role | Lines (orig) | Changed vs AGISlime |
|---|---|---|---|
| `nats_arena/eval_rollout.py` | `--eval-function-path` target: resolve the step's HF ckpt (S3-aware, DCP->HF fallback), render `eval-coordinator-job.yaml`, apply via kubernetes SDK, return `RolloutFnEvalOutput(data={})` at once | 643 (613) | `_arena_source_dir()` = MILES_ARENA_DIR > AGISLIME_DIR (set-but-empty falls through); `_find_convert_script()` prefers miles `tools/convert_torch_dist_to_hf.py` (same `--input-dir/--output-dir/--origin-hf-dir/--force` CLI), AGISlime `convert_torch_dist_to_hf_parallel.py` kept as fallback; ruff F401; template lookup moved inside the guard (fix-up) |
| `nats_arena/eval_coordinator.py` | `python -m` entrypoint of the K8s Job pod: owner-ref discovery, eval streams, eval-sglang + eval-gym Deployments, bounded-inflight publish, dedup/DLQ collect, pass@k, `step_*.json` file queue | 2133 (2134) | import swaps; `_find_templates_dir` prefers MILES_ARENA_PATH over AGISLIME_PATH; ruff F401 `subprocess`, F841 `sglang_url`, B904 `from exc` |
| `nats_arena/argo_eval_trigger.py` | opt-in (`ARENA_EVAL_TASKS`) per-checkpoint `arena-hosted-eval` Argo Workflow submitter, called from the driver save block, never raises | 194 (194) | docstring wording only |

Kept byte-identical (externally visible state, not module paths): streams `ARENA_EVAL_TASKS` /
`ARENA_EVAL_RESULTS`, subjects `arena.eval.tasks.<step>.<gym>` / `arena.eval.results.<step>`,
`max_age` 14400, `max_msg_size` 134217728, `ack_wait` 3600 + `max_deliver` 3, durable
`slime-eval-coord-step-<N>`, OTel tracer `slime.eval_coordinator`, the `.trainer_alive` /
`.flush_lock` protocol (600 s heartbeat staleness, 1800 s lock break), wandb `resume="must"` ->
`"allow"` fallback, the rendered template key `AGISLIME_PATH` (runtime templates reference
`${AGISLIME_PATH}`), Argo constants (template `arena-hosted-eval`, profile `qwen3-5-b200-tp2`,
engine `sglang`, `--limit 32`, priority `preemptible`, label `arena.agif.amazon.dev/submitter`).
The 2133-line coordinator was not split: semantics-preserving port, and the style rule's path
scope does not cover `miles_plugins/**`.

**Status: ported, CPU-verified, never exercised by any run.** `financeagent-27b-smoke.yaml`,
`snorkel-27b.yaml` and the GLM `miles-config.yaml` carry only `skip_eval_before_train: true` (no
`eval_datasets` / `hallmark_benchmarks` / `eval_interval` / `--eval-function-path`) and no trainer
manifest sets `ARENA_EVAL_TASKS`, so `submit_eval` returns at its opt-in check and
`eval_rollout.generate_rollout` is never called - in rl-milesgb1-smoke1 (2026-09-01/02) and in
GLM r1-r7 alike. CPU evidence (2026-09-01): all three modules import in `/tmp/miles-venv`; the
throwaway `/tmp/evalport/verify.py` passed on the pass@k estimator (8 samples / 3 correct:
pass@1 0.375, pass@2 1-10/28, pass@8 1.0), metrics payload, HF/DCP checkpoint resolution on a
fake tree, env-alias precedence, template resolution against a fake tree, the 40-key coordinator
env, the flush-lock protocol (O_EXCL acquire, 600/1800 s staleness) and the Argo workflow build;
ruff clean with the repo config. The ported tests (`test_argo_eval_trigger.py` 6,
`test_eval_coordinator_batched.py` 3, and the `TestRowToTask` / `TestEnvVarParsingGuards` /
`TestHandleResultMsgDedup` / `TestTaskIdFromLakefsUri` / `TestEstimatePassAtK` classes of
`test_nats_arena.py`) land with the test-suite commit.

**Templates live outside this tree; fix-up 02:37 PT (verify-dataEval minor).**
`eval-coordinator-job.yaml`, `eval-sglang-server.yaml`, `eval-gym-workers.yaml` are looked up
under `$MILES_ARENA_DIR|$AGISLIME_DIR/experiments/k8s/templates` and exist in neither the miles
tree nor the AGISlime git tree (they sit on the runtime-mounted deployment volume). The manifests
bake `AGISLIME_DIR=/root/miles`, which satisfies converter resolution only (`/root/miles/tools/`
is in the image). `_find_template_path()` was called outside `generate_rollout`'s try/except, so
the first eval trigger of any eval-enabled job would have raised `RuntimeError` into the trainer
instead of log-and-skip; the lookup now sits inside the guard, and the 27B README / trainer
manifest comments state the rule: an eval-enabled job MUST set `MILES_ARENA_DIR` to an external
tree carrying the templates, whose coordinator command must be re-pointed from
`python -m amzn_agi_slime.nats_arena.eval_coordinator` to `python -m
miles_plugins.arena.nats_arena.eval_coordinator` at deploy time (no code here renders it).
Alternative (porting the templates into `examples/arena/`) deferred: no job needed eval and the
templates were in no git tree the port was taken from.

**Why keep a path nothing runs.** The driver (next commit) hooks into it twice
(`_maybe_trigger_eval` after `save_model` -> `submit_eval`; `_maybe_drain_eval_metrics` per
iteration -> `eval_metrics_drain`, already in the skeleton commit), the launcher appends
`--disable-wandb-random-suffix` whenever `ARENA_EVAL_TASKS` is set because the Argo hook resolves
the W&B run by display name (entrypoint.sh parity), and the `.trainer_alive` / `.flush_lock` /
`step_*.json` queue has its trainer-side consumer in the tree. Dropping the modules would orphan
those seams and the `arena-hosted-eval` WorkflowTemplate contract. Two trigger mechanisms
coexist as in AGISlime and must stay mutually exclusive as configured: `eval_rollout` rides
`--eval-function-path` + `--eval-interval` (miles `arguments.py:3097` asserts `eval_datasets`),
`argo_eval_trigger` fires from the save block and deliberately bypasses that assert.

**Follow-ups recorded (verify-redundancy; not fixed here).**
- [major] The file-queue premise is obsolete on miles: `init_wandb_secondary` already provides
  cross-process writers (`id=wandb_run_id, mode="shared", x_primary=False,
  x_update_finish_state=False`). Make the coordinator a shared-mode secondary writer logging
  `eval/*` directly and retire `.trainer_alive` / `.flush_lock` / `resume="must"`. Interim: the
  drain commits every file (skeleton commit).
- [note] The Argo trigger bypasses miles' `CheckpointEvalFn` / `EvalDispatcher` seam
  (`eval_uses_snapshots`, `eval_max_in_flight`, `EvalSkip` attribution). Wrap submit+poll in a
  `CheckpointEvalFn` behind `--eval-function-path` + `--eval-interval` with a stub
  `eval_datasets` adapter; that also removes the display-name run resolution.
- The 1800 s conversion subprocess timeout was tuned for AGISlime's parallel converter; miles
  ships only the serial one, and the `parents[3]/tools` fallback exists only in a source tree.
- boto3 is imported at call time by the S3-checkpoint / hosted-eval paths; the example
  Dockerfile bakes it (verify-driverLauncher finding, fixed there).

## 2026-09-01 (PT) — Milestone: asyncio training driver + checkpoint-sidecar W&B resume

Files: `train_async_arena.py`, `checkpoint_extras.py`, core `miles/utils/tracking_utils/wandb_utils.py`. Decision record: ADR-0005.

- **~01:10 PT, driver written.** The AGISlime `train_async_arena.py` (183-line
  sync `ray.get` loop) could not be ported mechanically: `async_train`,
  `update_tracking_open_metrics`, `get_metrics_router_addr` and
  `finish_tracking(args)` do not exist in miles (`milesApi.md` hard blocker
  no. 2). Rebuilt as a copy of miles `train_async.py` (136 lines) plus
  insertions only: `diff | grep -c '^<'` = 0, 160 lines added, 296 total; hook
  bodies are the AGISlime ones after the import-path swap. Import OK in
  `/tmp/miles-venv`; `--help` engages the full miles+Megatron argparse (exit 0)
  as a script and via `python -m`; never-raise smoke on every hook passed.
  The open-metrics/router-addr block is gone (native in `RolloutManager` under
  `--sglang-enable-metrics`); `finish_tracking()` moves to the `finally`.
- **New vs AGISlime:** `_maybe_drain_eval_metrics` per iteration + `final=True`
  in the `__main__` `finally` closes the `overlay.md` CRITICAL GAP (drain was
  defined but never called trainer-side, so eval `step_*.json` piled up until
  the trainer died). No-op when wandb is off (27B smoke).
- **02:36-02:44 PT fix-ups** (from the adversarial verification wave; the full
  findings list is in the verification entry below):
  - 02:36 `wandb_utils.py` +6 lines: `init_wandb_primary` passes
    `id=args.wandb_run_id, resume='allow'` when the id is pre-set. Before this
    the sidecar restore and `--wandb-run-id` were both dead for the primary
    writer (it overwrote `args.wandb_run_id` after `wandb.init`).
  - 02:36 `checkpoint_extras.py` docstring corrected (resume is real now;
    `rollout_id` is log-only; filename keeps the `slime` prefix — AGISlime
    <-> miles sidecar interop verified both directions, byte-identical file).
  - 02:44 `train_async_arena.py`: `_register_wandb_metrics` hook and
    `wandb_extensions.py` (113 lines) deleted — 100 % redundant with miles
    `_init_wandb_common`; `_remove_trainer_alive` wired before
    `finish_tracking()` (never called in AGISlime either).
- **Deferred:** the package README (02:37 PT) lands with the first example
  directory because its usage section points readers at `examples/arena/`.
- **GPU evidence.** Every arena job since the smoke relaunch (2026-09-01
  16:39 PT) runs `python3 /root/miles/miles_plugins/arena/train_async_arena.py`
  as its ray entrypoint: `rl-milesgb1-smoke1` (3x p6-b200, image
  `arena-slime-dev:miles-arena-20260901b`, W&B `arena/rl-snorkel27` group
  `rl-milesgb1-smoke1` run `j6gr37za`) ran 4/4 rollouts and 4 train steps,
  saved `iter_0000003`, ray job SUCC 2026-09-02 00:34 PT (details in the
  snorkel example RUNLOG); GLM r1-r7 followed.
- **Observed on every fresh start** (smoke bring-up 2026-09-01 16:39-16:40 PT;
  run 2 under identity `rl-glm53f-gbash-r1`, 2026-09-02 14:15-14:16 PT): `Failed to load slime extra
  state: invalid literal for int() with base 10: 'release'`. Cause: miles
  rewrites `args.load` to `--ref-load` when the experiment dir has no
  `latest_checkpointed_iteration.txt`, and the converted reference DCP's marker
  reads `release`. Warn-only, training unaffected; follow-up: skip non-integer
  markers.
- **Not yet exercised on GPU:** a restart into an existing checkpoint dir with
  a sidecar present (r1/r2 died before `save_interval 20`; later runs used
  fresh `EXPERIMENT_NAME`s). Resume is CPU-verified only.

## 2026-09-01 01:21 PT — Milestone: launcher `scripts/run_arena_harbor.py`, 27B smoke config, argv parity, snapshots

**Scope:** `scripts/run_arena_harbor.py` (354 lines, the state archived as integration
commit 7182346d7 on 2026-09-02 00:55 PT), `examples/arena/harbor-rl-27b/financeagent-27b-smoke.yaml`,
`examples/arena/harbor-rl-27b/ARGV_PARITY.md`,
`tests/snapshots/launch_scripts/py/scripts/run_arena_harbor.py/{train,worker}.txt`,
`tests/fast/launch_scripts/py_harness.py` (`CLEARED_ENV`).
**Source:** AGISlime `scripts/custom/entrypoint.sh` + `scripts/custom/hydra_runner/hydra_converter.py`
(port baseline 90ce39f; both files byte-identical at the harbor-workspace b8aae83, 2026-08-27 PT) and `tmp_staging/smoke/harbor-rl-27b/financeagent-27b-smoke.yaml`;
the run-1 pod env of `rl-smoke27-financeagent` (`REPLICA=3`, `REPLICA_TRAINER=2`,
`GPUS_PER_NODE` 8; `WANDB_RUN_ID` / `ARENA_EVAL_TASKS` / `S3_ARTIFACT_BASE` / `USE_MLFLOW_AMZN`
unset). Records: `arena-port-artifacts/reports/impl-launcher.md` (implementation),
`verify-driverLauncher.md` + `verify-redundancy.md` (adversarial verification). Decision: ADR-0006.
**Env:** `/tmp/miles-venv`, CPU-only host, 3-way PYTHONPATH (testEnv.md).

### What was built

- Typer launcher with `train` / `worker` commands; `ScriptArgs(U.ExecuteTrainConfig)` reads
  the pod-env contract through `default_factory`; `_flatten` + `_load_experiment_config`
  reproduce hydra_converter; launcher-consumed keys (`user`, `cluster`, `experiment_name`,
  `project_name`, `agislime_dir`, `replicas`, `num_trainers`, `model_arch`) popped; entrypoint.sh
  appends re-created; submission via `U.execute_train(train_script="miles_plugins/arena/
  train_async_arena.py", megatron_model_type="qwen3.5-27B")`; bounded 120 x 5 s node wait.
- `financeagent-27b-smoke.yaml`: Qwen3.5-27B, TP4/PP2/CP2 on 2 actor nodes + 1 engine node,
  `rollout_batch_size 8 x n_samples_per_prompt 8 = global_batch_size 64`, `num_rollout 30`,
  `lr 2e-6`, `loss_mask_type qwen3_5`, `use_wandb false`. Exactly two deltas from the AGISlime
  original (header comment): the two dotted paths -> `miles_plugins.arena.nats_arena.*`, and
  `rollout_global_dataset: true` removed. `partial_rollout: true` kept although inert on this
  path (verify-redundancy), for config parity.
- Snapshots recorded 01:21:41 PT: `MILES_UPDATE_LAUNCH_SCRIPT_SNAPSHOTS=1 pytest
  tests/manual/launch_scripts -k arena` -> `train.txt` 963 lines (mkdir, preamble, ray head,
  all 120 `ray status`/`sleep 5` polls, `nvidia-smi` NVLink probe, then the `ray job submit`
  with the full runtime env and argv as command 245), `worker.txt` 13 lines (cleanup, `nc -z`,
  `ray start --block`).

### What failed on the way

- First snapshot recording: direct `os.makedirs` of the ckpt/log/tb/wandb dirs raised
  `PermissionError` on `/root` inside the harness. Fixed by routing the mkdir through one
  `U.exec_command_cpu("mkdir -p ...")` (also what the rule wants: shell only via
  `command_utils`; mirrors entrypoint.sh; records deterministically). Re-recorded.
- The full `tests/manual/launch_scripts/test_py_launch_scripts.py` on this non-root host:
  6 failed + 40 errors in *other* launchers (`run_deepseek*`, `run_kimi*`, `amd/*`, ...), all
  `PermissionError` writing `/root/models` / `/root/shared_data` — the harness expects the
  root-writable CI container; pre-existing, unrelated; every arena case and every
  `TestDiscovery` case passed.

### Verification (impl 01:xx PT, re-derived by the verifier the same night)

- Converter exactness: token streams byte-identical to the original `hydra_converter.py` on
  BOTH original YAMLs — run 1 (137 vs 121 tokens) and run 2 `financeagent-27b-run2.yaml`
  (144 vs 128); after the 8 consumed keys the multiset diff is empty both ways
  (`2e-6 -> 2e-06`, `15e-7 -> 1.5e-06`, `prompt-data-list` moved last).
- Argv parity (executed, not transcribed): **old 106 vs new 97 flag occurrences**; only-in-old
  = 7 consumed echo flags + `--rollout-function-path` x2 + `--data-source-path` + `--spec` +
  `--rollout-global-dataset` + `--use-gated-attention`; only-in-new = 3 renamed dotted paths +
  `--num-gpus-per-node 8`. The verifier recomputed the table independently: row-for-row and
  count-for-count match with `ARGV_PARITY.md`.
- Strict parse: the complete new argv through `miles.utils.arguments.parse_args` (argparse +
  plugin `add_arguments` hook + miles/megatron/sglang validate) -> `PARSE_OK` with
  `loss_mask_type=qwen3_5`, `attention_output_gate=True`, `rollout_global_dataset=True`,
  `arena_sample_mode=full_trajectory`, actor 2x8, rollout 8 GPUs, `lr=2e-06`. Host
  substitutions: `--hf-checkpoint` dropped, `--ref-load` -> empty temp dir,
  `torch.cuda.get_device_properties` stubbed to sm100. Control: an added `--bogus-flag` exits
  `SystemExit(2)`, so strictness is engaged before the checkpoint probe.
- Quoting proven end-to-end: the assembled `execute_train` command run through `bash -c` into
  a fake `ray` binary replicating ray 2.58's `shlex.join` + `shell=True`; the `--prompt-data`
  JSON arrives as one byte-identical token, `--save-hf` keeps the literal `{rollout_id}`.
- Megatron gate probes against `/tmp/miles-deps/Megatron-LM`: `--attention-output-gate` parses
  (`attention_output_gate=True`), `--use-gated-attention` is an argparse usage error; grep of
  AGISlime's patches: v0.5.9 registers the flag with no consumer (only v0.5.5/v0.5.6 wired it).
- `pytest tests/fast/launch_scripts` -> **43 passed**; `pytest tests/manual/launch_scripts
  -k arena` (without the update flag) -> **4 passed**, snapshots stable.

### Verification findings -> fix-ups (02:3x PT; YAML re-saved 02:37, ARGV_PARITY.md 02:38 PT)

- [note, verify-driverLauncher] `USE_MLFLOW_AMZN=true` appended `--use-mlflow-amzn
  --mlflow-project --mlflow-group`, registered by no parser in the tree (nor in AGISlime /
  slime 0.3.0 — silently-dropped no-ops there): the trainer would die at argparse inside the
  ray job. Fix: the launcher raises `typer.BadParameter` up front; both 27B jobs leave it unset.
- [verify-redundancy] with `S3_ARTIFACT_BASE` set the launcher passed an `s3://` URI as
  `--save-hf`; miles (like vendored slime) does `Path(args.save_hf.format(...))`, collapsing it
  to a pod-local `s3:/...` dir holding a ~55 GB HF export per save, never uploaded. Fix:
  `hf_save_dir` is always the local `<ckpt>/hf/rollout_{rollout_id}`; the AWS region vars are
  still exported for the eval path's `s3_artifact` upload.
- [major] the 27B README pointed at the wrong AGISlime `sglang-svc.yaml` variant (selector
  `rl-smoke-trainer`, not `rl-smoke27-trainer`) and [minor] the trainer Dockerfile lacked boto3
  — both fixed in the example/image files, recorded with that commit.
- impl-launcher concern left open: the arena pod-contract env vars were not in the harness
  `CLEARED_ENV` ("not mine to grow") — a developer with `EXPERIMENT_NAME` exported would see a
  local snapshot mismatch.
- The fix-ups did not touch the recorded commands: both opt-ins are unset by default, so the
  snapshot files kept their 01:21 PT state and are byte-identical to the copies archived in
  integration commit 7182346d7.

### Commit-time hardening (2026-09-05)

- `CLEARED_ENV` gains the launcher's 15 `default_factory` knobs, alphabetically:
  `ARENA_CHECKPOINTS_DIR`, `ARENA_DATA_DIR`, `ARENA_DEFAULT_GYM`, `ARENA_EVAL_TASKS`, `CFG_NAME`,
  `EXPERIMENT_NAME`, `GPUS_PER_NODE`, `NATS_URL`, `PROJECT_NAME`, `REPLICA`, `REPLICA_TRAINER`,
  `S3_ARTIFACT_BASE`, `S3_ARTIFACT_REGION`, `USE_MLFLOW_AMZN`, `WANDB_RUN_ID` (rule
  `launch-and-model-scripts.md:48`). The snapshots were recorded with the defaults
  (`EXPERIMENT_NAME`/`PROJECT_NAME` `arena-smoke`, `REPLICA` 3, `REPLICA_TRAINER` 2,
  `GPUS_PER_NODE` 8, default config = the 27B smoke YAML), so they stay byte-identical;
  re-run `tests/fast/launch_scripts` (expect 43) and `tests/manual/launch_scripts -k arena`
  (expect 4) after the edit — the 2026-09-05 replay of this commit on the pre-edit harness
  returned exactly those counts.
- The launcher is committed at its 7182346d7 state; the raylet
  `RAY_memory_monitor_refresh_ms=0` pin (+19 lines, worktree 373 lines) belongs to the GLM r2
  relaunch fix set and is logged there.

### Open at this milestone

- Never executed on GPU yet; the first real launch is the 27B smoke on prod-bom (next
  example commit). Watch argparse/validation output there because of the CPU-parse substitutions.
- YAML keys consumed by AGISlime modules other than `nats_rollout` (eval-pod `eval_*`,
  `wandb_disable_system_metrics`, `hallmark_benchmarks`) hard-fail on miles; none in the 27B configs.
- Eval-enabled jobs need `MILES_ARENA_DIR` pointing at a tree with `experiments/k8s/templates/`.
- miles' own Qwen3.5 recipes force 1 GPU per SGLang engine (TP>1 mis-generation on the pinned
  sglang, sgl-project/sglang#21039); the smoke config keeps the original
  `rollout_num_gpus_per_engine: 4` + dp-attention as a semantics-preserving choice — verify
  the arena image's sglang build before trusting rollout outputs.

## 2026-09-01 01:43-01:51 PT - Milestone: AGISlime CPU test suite ported (145 tests, 0 skipped)

**What.** Both AGISlime test layers moved to `tests/fast/plugins/arena/` (package markers 01:43 PT; `test_eval_coordinator_batched.py` 01:49; `test_nats_arena.py` 01:51). Sources: `AGISlime/src/amzn_agi_slime/tests/test_plugins.py` (layer 1 - the only python test `build.sh run_check` ever ran, with plain python) and `amzn_agi_slime/nats_arena/tests/*.py` (layer 2 - manual `pytest -v` only, never build-gated). On miles both layers run under pytest from `tests/fast/`, which CPU CI discovers and registers into stage-a-cpu.

| File | Tests | Covers |
|---|---|---|
| test_nats_arena.py | 102 (14 classes) | message_format, MixtureController, RolloutTimingTracker, reward_binary, data_source, eval_coordinator helpers + dedup/DLQ state machine, GymAutoscaler decisions (dry_run, no kubernetes), SALVAGEABLE_RESULT_STATUSES |
| test_rollout_metrics.py | 19 | truncated_ratio / binary_reward inline extractions; real compute_group_metrics_from_samples / compute_off_policy_metrics |
| test_plugins.py | 10 | parsers.split_reasoning, register_nova_reasoning_parser, rewards.binarize_reward / normalize_grouped_rewards; keeps the framework-free `__main__` runner |
| test_argo_eval_trigger.py | 6 | submit_eval opt-in gating, Argo workflow shape, W&B merge coords, never-raises guarantee |
| test_weight_versions.py | 5 | `_result_to_samples_full_trajectory` fast path: weight_versions precedence/coercion, group_metrics on first sample only |
| test_eval_coordinator_batched.py | 3 | publish-side wire contract: n_samples on the task, no `-s<idx>` suffix, gzip serialize/parse round-trip |

**Result.** `pytest tests/fast/plugins/arena/ -q -rs` -> 145 passed, 0 failed, 0 skipped (0.63 s; verification re-runs 0.58 s / 0.31 s). `python tests/fast/plugins/arena/test_plugins.py` -> 10/10 PASS, exit 0 (register_nova_reasoning_parser really registers against the sglang source tree instead of early-returning). `tests/fast/utils` unaffected (3229 collected; 3213 passed / 16 skipped). Per-file test counts identical to the originals; assert count 187 -> 188 (the std fix only). Env: `/tmp/miles-venv` (CPython 3.12, torch 2.11.0+cpu) + 3-way PYTHONPATH from the repo root (arena-port-artifacts/reports/testEnv.md).

**Decisions.**
- Every `pytest.importorskip("slime")` removed (TestReadManifest, TestTaskIdFromUri, TestArenaDataSourceWeightedSampling): `slime` does not exist on miles, so the guard would skip forever and the suite would look green while executing nothing. `importorskip("torch")` kept where the original had it; torch is a hard miles dependency so it never fires (audited with `-rs`). test_weight_versions keeps its unguarded module-level nats_rollout import for the same reason.
- `TestBinarizeReward::test_grpo_std_normalization` expected +/-1.0; corrected to 0.5/(sqrt(1/3)+1e-6) ~= 0.866. torch `Tensor.std` is the unbiased (n-1) estimator; the upstream comment assumed population std. Proof: the ORIGINAL test run against the ORIGINAL AGISlime code (PYTHONPATH=src/amzn_agi_slime:vendor/slime/0.3.0) fails with processed[0]=0.8660238981246948 (1 failed, 5 passed) - an upstream test bug, not a port regression. Production `reward_binary.py` untouched, bit-faithful. Rejected: switching production to population std (would silently change GRPO normalization relative to every AGISlime run).
- Inline copies kept for the three blocks that cannot be imported (the `_handle_result_msg` closure inside eval_coordinator's collection loop; the truncated_ratio and binary_reward blocks inside generate_rollout's metrics section) but resynced to the PORTED source: the dedup state machine gained the `tid.rsplit("-s", 1)[0]` task-registry fallback and now calls the real `message_format.extract_trajectories` (verify-driverLauncher confirmed the ported closure is identical to AGISlime's, i.e. the original test's copy was already stale upstream). `_BINARY_PATH` -> `miles_plugins.arena.nats_arena.reward_binary.binarize_reward`. Rejected for this commit: extracting the blocks into module-level functions (production code owned by other commits); recorded as a follow-up because these three groups still cannot catch drift in the real code.
- `sys.path.insert(parents[N])` bootstrap hacks and the imports they needed dropped; `miles_plugins.arena.*` resolves from the repo-root PYTHONPATH. Placement under `tests/fast/plugins/arena/` with `__init__.py` markers mirrors every other tests/fast subtree. Rejected: in-package `miles_plugins/arena/tests/` as in AGISlime (outside miles' testpaths, not CPU-CI discovered).
- Mechanical fidelity otherwise: test_nats_arena's mid-file `if __name__ == "__main__"` block stays where it was; per-file standalone runners kept; docstrings renamed slime -> miles and re-anchored to ported line numbers.
- ruff (repo config): the findings in test_nats_arena.py (F841 pass_threshold_per_dataset, UP037/F821 quoted GymAutoscaler annotation) and test_rollout_metrics.py (UP037) are byte-identical in the AGISlime originals and left alone; the port removed three original F401s (torch, GymAutoscaler, typing.Any).

**Fix-ups from the verification wave (02:34 PT / 02:44 PT).**
- verify-regression [major]: four files carried `register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])`, the semantic-default form banned by `tests/ci/test/test_ci_register.py::TestNoSemanticDefaultRegisterCpuCi` because `collect_tests` already synthesises an implicit registry for every tests/fast file (policy suite: 1 failed / 677 passed / 1 skipped; offenders test_argo_eval_trigger.py:20, test_plugins.py:9, test_rollout_metrics.py:20, test_weight_versions.py:18). Call and orphan import removed from test_argo_eval_trigger / test_plugins / test_weight_versions (02:34 PT) and test_rollout_metrics (02:44 PT); test_nats_arena (est_time 30) and test_eval_coordinator_batched (est_time 5) keep explicit registration. The whole suite runs in well under a second, so est_time only affects scheduling.
- test_rollout_metrics.py (mtime 02:44 PT): the two extraction docstrings re-anchored to the fixed-up nats_rollout - the plugin metric is now `rollout/truncated_ratio_prefilter` (verify-redundancy double-log finding) and the cited ranges moved from ~1478-1485 / ~1672-1677 to ~1498-1505 / ~1694-1699. Logic unchanged.

**Coverage gap carried forward.** verify-natsGrpo [minor]: none of the 102 test_nats_arena tests assert the group_index / index / rollout_id stamping in `_process_group` (only publish-side task metadata); closed by test_group_identity.py in the next milestone. Port-added CPU tests after this commit: 145 here + 3 (test_loss_mask_qwen3_5, already landed) = 148; 153 once test_group_identity lands.
