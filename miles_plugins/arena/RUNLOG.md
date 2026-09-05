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
