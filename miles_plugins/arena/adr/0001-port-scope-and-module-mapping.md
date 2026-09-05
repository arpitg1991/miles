# ADR-0001: Port scope and module mapping for the arena plugin

**Status:** Accepted
**Date:** 2026-09-01

**Builds on:** AGISlime ADR 0001 (vendor pristine slime, ship first-party code as a separate wheel), ADR 0002 (`nats_arena` plugin placement), ADR 0003 (training utilities as standalone modules) — the overlay's own rules are what made a mechanical mapping possible.

## Summary

Bring the arena RL training path onto miles as one plugin package,
`miles_plugins.arena`, produced from AGISlime's `amzn_agi_slime` by a fixed 1:1
module mapping. Edit miles core only where miles offers no seam. Keep `slime`
in identifiers that are on-disk or on-wire contracts; rename everything else.

## Context

- AGISlime is Amazon's overlay on a pristine vendored THUDM/slime 0.3.0
  (`src/amzn_agi_slime`, 7,667 lines + 2,139 test lines at
  `origin/mainline-0.3.0` 90ce39f, 2026-08-28 13:36 PT). It plugs into slime
  through five official seams — a driver module
  (`python3 -m amzn_agi_slime.train_async_arena`), `--rollout-function-path`,
  `--data-source-path`, `--eval-function-path`,
  `--custom-reward-post-process-path` — and never patches vendor code (one
  scoped `read_file` rebind aside).
- The user is moving arena/harbor training onto miles (radixark/miles; fork
  HEAD 2799fe38, 2026-08-31 20:17 PT). miles descends from slime: the same
  seams exist, `slime.*` became `miles.*`, and `miles_plugins/` already ships
  `mbridge`, `models` and `optimizers` plugins in the wheel
  (`find_packages(include=["miles*", "miles_plugins*"])`), referenced by
  dotted path with no entry-point registration.
- The analysis (`arena-port-artifacts/reports/milesApi.md`) found every one of
  the 14 slime APIs the overlay imports in miles, but with semantic drift: the
  driver loop is asyncio end-to-end; `Sample.rollout_id` is a readable field
  where vendored 0.3.0 had renamed it `group_id` (read raises);
  `DataSource.__len__` is gone; tracking became `init_tracking` / `log` /
  `finish_tracking()`; `parse_args` is strict and rejects
  `--loss-mask-type qwen3_5`; `init_wandb_primary` ignores a pre-set
  `wandb_run_id`. slime's parser silently dropped unknown flags; miles errors.
- Hard constraints: the deployed gym workers (AREnATasks `amzn_arena_harbor`)
  speak the `amzn_arena_contract` wire format, pinned by value in
  `tests/harbor_runtime/test_cross_runtime_parity.py`; checkpoints written by
  AGISlime runs carry an `iter_%07d/slime_extra_state.json` sidecar that
  resume must read; miles core should stay rebaseable onto upstream.

## Decision

1. **One package, fixed mapping.** `amzn_agi_slime.X → miles_plugins.arena.X`,
   including the `nats_arena` subpackage; config function paths change only by
   prefix (`miles_plugins.arena.nats_arena.nats_rollout.generate_rollout`,
   `...data_source.ArenaDataSourceWithBuffer`,
   `...reward_binary.binarize_reward`). Every module is ported from the same
   revision and diffed against it after import normalisation; each remaining
   difference is named in the impl report for that module.
2. **Core edits only where miles has no seam.** Arena-specific arguments
   register through the rollout function's `add_arguments` hook; the data
   source and reward post-processor ride their function-path seams; the driver
   is a separate module. Core is edited for the `qwen3_5` loss-mask type
   (argparse `choices` cannot be extended from a plugin), for
   `init_wandb_primary` honouring a pre-set run id (resume), and for the
   `arena` extra in `setup.py`. Any further core edit needs a `RUNLOG.md`
   entry naming the missing seam.
3. **`slime` stays where it is a contract, goes everywhere else.** Kept: the
   NATS durable consumer `slime-trainer` (`NATS_RESULTS_CONSUMER` default —
   JetStream consumer state and the gym-side contract); the sidecar
   `iter_%07d/slime_extra_state.json` with `save_/load_slime_extra_state`
   (checkpoint interop); the checkpoint root
   `<checkpoints>/slime_experiments/<EXPERIMENT_NAME>` (path parity with
   `entrypoint.sh`). Renamed: package, docstrings, log wording. Provenance
   docstrings that cite AGISlime source paths stay as true provenance notes.
4. **Dependencies as an extra.** `pip install -e ".[arena]"` = `nats-py>=2.6.0`,
   `kubernetes==35.0.0` (36.x returns 401 on EKS and silently breaks the gym
   autoscaler and eval-Job creation), `lakefs`, `boto3`; all imported lazily so
   non-arena runs never need them; `install_requires` untouched.
5. **Fidelity is verified, not assumed.** Import-normalised diffs, the ported
   CPU test suite under `tests/fast/plugins/arena`, argv parity for the
   launcher, and an e2e harness against a real NATS JetStream server are the
   acceptance evidence, recorded in `RUNLOG.md` as they land.

### Alternatives considered

- **Vendor slime 0.3.0 into the miles tree and run `amzn_agi_slime` unchanged.**
  Rejected: the training stack would still be slime 0.3.0, nothing is gained
  from miles (rollout v2 protocol, object store, model plugins, fault
  tolerance), and the repo would carry two trainers.
- **A `slime` compatibility shim inside the plugin**, re-exporting miles APIs
  under slime names so the overlay drops in. Rejected: the drift is semantic,
  not nominal — asyncio driver, `Sample` identity inversion, dict data packs,
  changed tracking signatures. A shim would hide exactly the `group_id`
  inversion that `milesApi.md` warned turns into a silent dead attribute.
- **Rename everything, including the durable and the sidecar.** Rejected:
  `slime-trainer` is the JetStream durable the gym contract and existing
  streams know; renaming it breaks consumer-state continuity for no functional
  gain. `slime_extra_state.json` must stay readable from AGISlime-written
  checkpoints for resume.
- **Fold the NATS path into miles core** (`miles/rollout/...`). Rejected: miles
  has no NATS, lakeFS or S3 support anywhere (`verify-redundancy.md`), so it
  would be a large core divergence to carry on every upstream rebase; the
  plugin namespace is how miles already ships optional code.
- **Edit core wherever convenient.** Rejected: each core hunk is rebase cost;
  the "no seam" rule keeps the core diff to a handful of files.

## Consequences

### Easier

- Upstream rebases: the plugin is additive and the core diff is small.
- Gym workers and NATS deployments need no change (wire values untouched;
  formalised in ADR-0002).
- AGISlime-era checkpoints resume on miles (sidecar filename preserved).
- Review by construction: each module can be diffed against 90ce39f after
  import normalisation; three support modules come out byte-identical.

### Harder / open

- Two behaviours could not be preserved by mapping alone and became explicit
  decisions of their own: group identity (ADR-0003) and per-trajectory vs
  per-group loss weighting (ADR-0004).
- `slime` strings in a miles tree surprise readers; every kept name carries an
  interop note at its definition.
- The snorkel-run customs on AGISlime's local branch
  `rl-snorkel-r4-removed-sample-replacement` (637d714, b8aae83 — not ancestors
  of 90ce39f) are outside this port's source revision; their fate is decided
  separately (ADR-0007).
- Ruff findings inherited byte-for-byte are kept (miles' ruff config excludes
  `miles_plugins/**`); cleaning them is a separate chore so diffs against the
  source stay readable.
- Lazy imports mean a missing extra surfaces at first use, not at import; the
  package README lists which module needs which package.
