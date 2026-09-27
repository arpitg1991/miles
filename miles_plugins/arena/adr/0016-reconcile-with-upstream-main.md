# ADR-0016: Reconcile the fork with upstream `main` (2026-09-27)

**Status:** Proposed
**Date:** 2026-09-27

**Amends:** ADR-0005 (the driver anchors), ADR-0006 (the launcher helper calls),
ADR-0011 (where the per-turn weight versions live)
**Supersedes in part:** ADR-0008 (the fork no longer carries its own copy of PR #2786)

## Context

- Branch `arpit-glm-53` forked from radixark/miles `2799fe386` (2026-08-31).
  Upstream `main` has 750 commits after that point. The rebase target is
  `23d41d711` (2026-09-26). Branch `arpit-reconcile-upstream` holds the result.
- Upstream merged PR #2786 (GLM-5.3-Flash, `glm5_next`) as the squash commit
  `cc76e2391` on 2026-09-24. The fork carried the PR branch head `dbbd610e7`
  (2026-08-29) through the merge commit `38921ae45` (ADR-0008).
- Four fork contracts break on upstream `main`:
  1. `train_async_arena.py` imports `create_rollout_manager`,
     `MainProcessIdentity`, and `ft_utils.control_server`. Upstream removed all
     three. Upstream split the rollout manager into an inference controller and
     a rollout executor. Upstream also moved the tracking setup and teardown
     into `init_orchestration_script` and a `Disposer` stack.
  2. `scripts/run_arena_harbor.py` calls `U.exec_command_cpu` and
     `U.execute_train`. Upstream moved both to a backend object (#2432).
  3. `Sample.weight_versions` is now a list of `WeightVersionsPerCall` span
     objects (#1891). The NATS path stored one string per turn in that field.
     `convert_samples_to_train_data` calls `to_dicts()` on each entry, so a
     string stops the conversion with `AttributeError`.
  4. `compute_advantages_and_returns` now reads the parallel state and returns
     early off the last pipeline stage (#3125).

## Decision

1. **Take the upstream GLM-5.3 plugin.** The rebase drops the 28 PR commits of
   the fork. The core hunks of the PR copy are identical to the core hunks of
   `cc76e2391`. The model files of `cc76e2391` are the final PR version. The
   fork keeps two changes that upstream does not have: `b90c21ed4` and the
   `reloadable_process_group.py` hunk of `bd8843d29`.
2. **Rebuild the driver on the upstream `train_async.py`.** The ADR-0005
   invariant stays: `diff train_async.py miles_plugins/arena/train_async_arena.py | grep -c '^<'`
   is 0. The hooks stay at the same anchors, with one change. The final
   eval-metrics drain and the trainer-alive removal are one disposer callback,
   `_finish_arena`. The driver registers it right after
   `init_orchestration_script`, which registers `finish_tracking`. The stack
   runs its callbacks in reverse order. Thus `_finish_arena` runs after the
   eval dispatcher, the models, and the rollout components stop, and before
   `finish_tracking`. This is the same order as the old `finally` block.
3. **The launcher calls the backend object.** `train` and `worker` call
   `args.create_backend()`, then `backend.exec_command_cpu` and
   `backend.execute_train`. The backend appends `--deploy-component all` to the
   train argv. The launcher argv does not change in any other way.
4. **The gym weight versions move to `Sample.metadata["arena_weight_versions"]`.**
   The value is the same list as before: one string per turn, the trajectory
   list on each segment, with no per-segment slice. `Sample.weight_versions`
   stays empty on the NATS path. `rollout/off_policy_round/*` reads the
   metadata key and gives the same values as before.
5. **The advantage-scale tests set a one-rank parallel state.** The
   `advantage_scale` code path does not change.

### Alternatives considered

| Alternative | Why not |
| --- | --- |
| Build real `WeightVersionSpan` objects on the NATS path | A span needs the absolute token range of each turn output. The gym sends only the cumulative token arrays and one version per turn. The loss mask cannot give the ranges, because the gym sets 0 on a clipped or empty turn. A span from the loss mask puts a wrong range into the upstream checks and metrics. |
| Keep strings in `Sample.weight_versions` and patch the upstream conversion | This is a core edit where an upstream contract exists. ADR-0001 allows a core edit only where miles has no seam. |
| Merge upstream `main` into `arpit-glm-53` | The merge keeps the PR copy and the upstream squash as two histories of the same files. It also hides the drop list inside a merge commit. |
| Cherry-pick the arena commits onto upstream `main` | A rebase keeps the commit order and the patch-id mapping to the fork, so `git cherry` can prove which commits changed. |

## Consequences

### Easier

- The GLM-5.3 plugin is upstream code. An upstream fix to the plugin applies
  without a fork patch.
- The driver keeps the pure-insertion check, so the next upstream change to
  `train_async.py` is again a three-way merge of a copied file.

### Harder / open

- The upstream `weight_version/*` metrics (`miles/ray/rollout/metrics.py`) and
  the upstream staleness filters see no spans for arena samples. The fork logged
  `weight_version/*` from the strings before. Upgrade path: the gym sends the
  output token range of each turn, and the NATS path builds one
  `WeightVersionsPerCall` per turn.
- Upstream changes alter a run on this branch. `examples/arena/RECONCILE.md`
  lists them. The main items:
  - The DSA indexer applies RMSNorm to the indexer query (final #2786). The
    fork copy feeds the raw query. SGLang `9a26e749` feeds the normalized query.
  - The GLM-5.3 model args add `--activation-func-clamp-value 10`. The
    checkpoint sets `swiglu_limit` 10, and SGLang clamps.
  - With `use_fault_tolerance`, upstream turns on the mini fault-tolerance
    controller on `api_server_port` 18080. `--control-server-port` is gone.
  - `save_debug_event_data` defaults to `<save>/events`. Each checkpoint gets a
    copy of the event logs.
  - `train_rollout_logprob_abs_diff` and `train_rollout_kl` come from
    trainer-scored log-probs (#3655). Values before and after the rebase are
    not directly comparable.
- The status stays Proposed until two conditions are true. The owner accepts
  decision 4. A GPU test job on a trainer image from this branch loads an
  r45-lineage checkpoint and runs one step with R3 and TIS.
