# ADR-0017: Training runs get one record per run and one record per study under `training-runs/`

**Status:** Accepted
**Date:** 2026-09-28

**Amends:** nothing. The run folders that `gen-workflow.py` writes move
from `examples/arena/<family>/r<N>/` to `training-runs/<family>/r<N>/`;
no earlier ADR fixed that path.
**Pairs with:** AREnATasks `eval-runs/` (the eval counterpart, same shape:
`README.md`, `TEMPLATE.md`, one record per run, one index).
**Numbering note:** the `arpit-reconcile-upstream` branch held its own
ADR-0016 (reconcile the fork with upstream `main`), so its number collided
with ADR-0016 on this branch (KDA sharding). Branch `arpit-recon-20261001`
moved the reconcile ADR to ADR-0018 on 2026-10-01; this ADR keeps 0017.

## Summary

Every training run gets `training-runs/<family>/r<N>/RECORD.md` with a
required header. `gen-workflow.py` writes the machine-known part of the
header and refuses to run when the config names no dataset. Every
investigation that spans runs gets `training-runs/studies/<slug>/STUDY.md`.
`training-runs/INDEX.md` lists both; `build_index.py` regenerates its Runs
table. The chronological `RUNLOG.md` of a family is closed.

## Context

- **One log, 2,400 lines, no fixed fields.** `RUNLOG.md` of
  `harbor-rl-glm53-flash` grew to 2,419 lines over r9 to r47. An entry
  named the dataset, the images, the checkpoints, or the retire reason
  only when the writer thought of it. `r41/BUILD.md` names the
  `manifest-le5.jsonl` subset; `r43/BUILD.md` and `r41b/BUILD.md` do not.
- **A dataset mix-up went unnoticed for four runs.** r41 to r45 trained
  on the 551-chain `K <= 5` subset while the user meant `agentic-debt-766`.
  The user caught it on 2026-09-27; the all-pass audit (`wf_e6084655-6ed`,
  2026-09-28) measured the effect. A required, script-written dataset line
  makes that omission impossible.
- **Investigations lived in journals and memory notes.** The MFU, R3,
  radix-cache, per-token-loss, KDA, and overflow studies existed as
  workflow journals under a home directory, `/tmp` scratch, and memory
  notes. None of those is in git, and `/tmp` is a tmpfs.
- **A precedent exists.** AREnATasks `eval-runs/` holds one dated record
  per eval run with a `TEMPLATE.md`, a `README.md`, and an index table.

## Decision

- `training-runs/<family>/r<N>/` holds `RECORD.md`, `miles-config.yaml`,
  `workflow.yaml`, each `workflow-resume<K>.yaml`, and the legacy
  `BUILD.md`. One folder per run identity (`EXPERIMENT_NAME`). Recipe and
  code stay in `examples/arena/<family>/`.
- `RECORD.md` starts with the header of `training-runs/TEMPLATE.md`:
  Status, then the `gen-workflow` block (Date, Family, Argo generateName,
  Experiment name, W&B project, Dataset, Manifest commit, Gym image,
  Trainer image, Template, Base), then the operator fields (Argo workflow,
  W&B run, Task pin, Image digests, Trainer config deltas vs base,
  Checkpoints, Outcome). Sections: Goal, Setup, Timeline, Results, Issues,
  Follow-ups, Sources.
- `gen-workflow.py` writes the block between `<!-- gen-workflow:begin -->`
  and `<!-- gen-workflow:end -->` on every run and keeps the text outside
  the markers byte for byte. It reads the dataset from `prompt-data-list`
  of `r<N>/miles-config.yaml` and exits with `FAIL` before it writes
  `workflow.yaml` when no row has a `path`.
- `training-runs/studies/<slug>/STUDY.md` follows `STUDY-TEMPLATE.md`:
  Date, Status, Question, Runs and data used, Code SHAs, Workflow ids;
  Method, Results, Verdict, Caveats and open items, Actions taken,
  Sources. Each number without a primary source reads "not re-verified".
- `training-runs/INDEX.md` holds the Runs table (written by
  `python training-runs/build_index.py` from the record headers) and the
  hand-written Studies table. `tests/fast/plugins/arena/` checks the
  script and that the committed table is current.
- `<family>/RUNLOG.md` and `experiment-list.md` stay as history and take
  no new entries.

### Alternatives considered

- **Keep `RUNLOG.md` and add a checklist.** Rejected: a checklist in prose
  has no check. The script-written header has one.
- **A catalog registry entry per run.** Rejected for now: the runs are
  research runs of one team; a Markdown record in the repo is the smallest
  thing that reviewers read in a CR.
- **Run records inside `examples/arena/<family>/r<N>/`.** Rejected: the
  recipe directory then mixes code with 37 run folders, and studies that
  span families have no home.

## Consequences

- A new run needs `r<N>/miles-config.yaml` and one `gen-workflow.py`
  call; the header is then complete except for the operator fields.
- A run without a dataset row cannot get a `workflow.yaml`.
- The runs index is a build artifact of the records; a stale table fails
  `test_committed_index_is_current`.
- Studies carry their sources, so a deleted `/tmp` no longer erases an
  investigation. Numbers that only a journal or a memory note holds stay
  marked "not re-verified".
- r39 to r47 are the first records; r9 to r38 keep only their files and
  the `RUNLOG.md` entries. A record for an older run is welcome but not
  required.
