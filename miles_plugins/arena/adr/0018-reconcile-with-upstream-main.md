# ADR-0018: Reconcile the fork with upstream `main` (2026-09-27)

**Status:** Accepted (2026-10-01, with the span design of decision 4)
**Date:** 2026-09-27
**Number:** ADR-0016 until 2026-10-01. The fork branch `arpit-glm-53` gave
ADR-0016 and ADR-0017 to other decisions, so this record moved to ADR-0018.

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
4. **The gym sends one weight-version span per model call, and the NATS path
   fills the upstream `Sample.weight_versions`** (2026-10-01). Until
   2026-10-01 the versions stayed in `Sample.metadata["arena_weight_versions"]`
   and `Sample.weight_versions` stayed empty. The contract:
   - Each step (segment) that the training gym ships keeps its
     `weight_versions` data and adds `weight_version_spans`. That list has one
     `{"version", "start", "end"}` entry per model call of the segment, in
     call order. `version` is the SGLang `weight_version` of the call (an
     int). `[start, end)` is the output of the call in the cumulative
     `token_ids` of the same segment, the index space of the retired
     `truncated_spans`. The ranges do not overlap, they increase, and
     `end <= len(token_ids)`. A call without an SGLang version gives no entry.
     The gym side is AREnATasks ADR-0074.
   - `_step_to_sample` keeps the step `token_ids` as `Sample.tokens`, so the
     offset is 0. Each entry becomes one
     `WeightVersionsPerCall(spans=[WeightVersionSpan(str(version), start, end)])`.
     A hard context overflow cuts `Sample.tokens`. The spans are then cut the
     same way as upstream `Sample.strip_last_output_tokens` cuts them.
   - The trainer checks each entry: `int` values,
     `previous end <= start < end <= len(token_ids)`. A step with token arrays
     and without `weight_version_spans` stops the run when the trajectory
     reports a weight version (`weight_versions`, or `weight_version` on a
     step). The error names the field and the minimum gym image. A bad entry
     stops the run too. Both raise `WeightVersionSpansError`, which the NATS
     worker treats as fatal, as a `RoutingReplayError`. A trajectory without
     any weight version (eval and no-SGLang paths) gives empty
     `Sample.weight_versions`, as upstream allows. The messages-only path
     makes its own tokens, so it gives empty `Sample.weight_versions` too.
   - A DP pad copies the spans of its source row, because it copies the
     tokens. A failed-slot pad gets no spans.
   - Upstream code then reads arena samples with no plugin copy:
     `assert_samples_weight_version_sane`, `Sample.validate`,
     `Sample.oldest_weight_version`, `rollout/weight_version/*` of
     `log_rollout_data`, `train_data["weight_versions"]`, and the dashboard.
     `rollout_metrics.compute_weight_version_metrics` and the metadata key are
     gone.
   - `rollout/off_policy_round/*` stays per episode. It reads the span versions
     of every segment of the episode. A DP pad adds none.
   - The consume-time staleness filter of upstream `DefaultDataBuffer.get`
     applies where `generate_rollout` takes groups from the queue: the same
     `group_staleness` helper, the same `--max-weight-staleness` flag, and the
     same default (off). A stale group is dropped, and its routing files are
     deleted. `generate_rollout` returns the upstream keys
     `rollout/fully_async/stale_groups_filtered`,
     `rollout/fully_async/avg_staleness` and
     `rollout/fully_async/max_staleness` in `RolloutFnTrainOutput.metrics` and
     logs one `Weight staleness:` line per rollout. The filter needs the engine
     weight version. Only the class-based seam passes it
     (`RolloutFnTrainInput.weight_version`), so `NatsRolloutFn` wraps
     `generate_rollout`. The function path gets no version: the staleness keys
     other than the drop count stay absent, and `--max-weight-staleness` stops
     the run with an error that names `NatsRolloutFn`.
5. **The advantage-scale tests set a one-rank parallel state.** The
   `advantage_scale` code path does not change.

### Alternatives considered

| Alternative | Why not |
| --- | --- |
| Keep the versions in `Sample.metadata["arena_weight_versions"]` (decision 4 until 2026-10-01) | The upstream checks, the staleness filter and the dashboard see no versions, and the plugin keeps a copy of the upstream `weight_version/*` code. The gym can send the ranges, so the limit is not necessary. |
| Build the spans from the loss mask | The gym sets 0 on a clipped or empty turn, so the mask cannot give the ranges. A wrong range goes into the upstream checks and metrics. |
| Train on with empty `Sample.weight_versions` when an old gym sends no spans | Two code paths, and the run loses its staleness numbers without a trace. One error at the first message is the cheaper failure. |
| Take the engine version from `rollout_id` | The trainer weight version starts at 1 again after each restart. Only the rollout executor knows the published version. |
| Pass the version to legacy rollout functions in `LegacyRolloutFnAdapter` | This is a core edit where an upstream seam exists (the class-based rollout function). |
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

- Lockstep order for decision 4: gym first, trainer second. A trainer from
  this decision stops on the first message of an older gym image. A trainer
  image built before the decision ignores `weight_version_spans`.
- Each segment now carries the versions of its own calls, not the trajectory
  list. On rows with the same versions, `rollout/weight_version/*` gives the
  values of the deleted plugin copy (the tests pin the numbers). Under
  `--arena-train-segments all` a segment row and its DP pads now read only
  that segment, and under `final` the off-policy round reads the final
  segment only. Values of a compaction episode before and after the change
  are thus not directly comparable.
- The staleness filter and the staleness keys need
  `--rollout-function-path miles_plugins.arena.nats_arena.nats_rollout.NatsRolloutFn`.
  The run configs still name `generate_rollout`. A stale group is always
  dropped. The NATS path has no prompt recycle, so
  `--async-unused-samples-handler retry` has no effect.
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
- Acceptance (2026-10-01). The two conditions of the proposal are true. The
  owner accepted decision 4 with the span design. The GPU job
  `recon-t2-20260927c` loaded the r45 `iter_0000039` checkpoint and ran three
  steps with R3 and TIS (`examples/arena/RECONCILE.md`, "GPU validation").
  The span design has CPU tests only. No gym image with
  `weight_version_spans` exists yet, and no job has run the span path.
