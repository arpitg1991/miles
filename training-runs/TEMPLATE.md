# Run record: r<N> — <one-line title>

**Status:** Prepared | Running | Complete | Retired | Invalid
<!-- gen-workflow:begin -->
**Date:** YYYY-MM-DD
**Family:** `<family>`
**Argo generateName:** `rl-glm53f<N>-`
**Experiment name:** `<EXPERIMENT_NAME>`
**W&B project:** `<wandb_project>` (group `<EXPERIMENT_NAME>`)
**Dataset:** `<prompt-data-list path>` (gym `<gym_name>`)
**Manifest commit:** `<lakeFS ref of the dataset URI>`
**Gym image:** `<repository:tag>`
**Trainer image:** `<repository:tag>`
**Template:** `<WorkflowTemplate name>`
**Base:** `r<M>`
<!-- gen-workflow:end -->
**Argo workflow:** `rl-glm53f<N>-<suffix>`
**W&B run:** `<run id>`
**Task pin:** `<lakefs_commit_id of the manifest rows>`
**Image digests:** gym `sha256:<...>`, trainer `sha256:<...>`
**Trainer config deltas vs base:** <keys that differ from the base run, with both values>
**Checkpoints:** `<ARENA_CHECKPOINTS_DIR>/slime_experiments/<EXPERIMENT_NAME>`, last save `iter_<NNNNNNN>` (YYYY-MM-DD HH:MMZ)
**Outcome:** <result, or the retire reason>

## Goal

<One or two sentences: what this run tests, and what result ends it.>

## Setup

<What differs from the base run and why. Link `BUILD.md` when it exists.>

| Item | Base r<M> | This run |
| --- | --- | --- |
| `<key>` | `<value>` | `<value>` |

## Timeline

| UTC | Event |
| --- | --- |
| YYYY-MM-DD HH:MMZ | <launch, admission, save, resume, retire> |

## Results

| Metric | Value | Source |
| --- | --- | --- |
| `<metric>` | 0.000 | <W&B run, log line, S3 file> |

## Issues

- <What broke, the root cause, and the fix or workaround.>

## Follow-ups

- <Open item, owner, and the record or ADR that tracks it.>

## Sources

- <W&B run URL, S3 prefix, log path, commit, ADR, workflow id.>
