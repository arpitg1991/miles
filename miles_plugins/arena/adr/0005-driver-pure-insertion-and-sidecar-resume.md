# ADR-0005: Driver as pure insertion over `train_async.py`; checkpoint-sidecar W&B resume via a 6-line core change

**Status:** Accepted
**Date:** 2026-09-01

**Builds on:** ADR-0001 (port scope: core edits only where miles has no seam; keep `slime` names that are on-disk/wire compat)

## Summary

Rebuild AGISlime's `train_async_arena.py` on miles' own `train_async.py`
asyncio loop as a pure insertion (0 deleted lines, 160 inserted, 136 -> 296
lines) with never-raise hooks at four anchor points. Keep the
`iter_%07d/slime_extra_state.json` checkpoint sidecar and make its W&B resume
real with a 6-line change to `init_wandb_primary` (`id=args.wandb_run_id`,
`resume='allow'` when the id is pre-set). Drop `wandb_extensions.py`.

## Context

- The AGISlime driver (183 lines) is a synchronous `ray.get` loop over slime
  0.3.0 APIs that no longer exist in miles (`actor_model.async_train`,
  `update_tracking_open_metrics`, `get_metrics_router_addr`,
  `finish_tracking(args)`). In miles the model calls are coroutines, `generate`
  returns a `dict(sample_indices, data_ref)` pack that must be released with
  `remove_rollout_data_refs`, `object_store.init_instance` is mandatory on the
  driver, and a missed `await` is fatal (strict unawaited-coroutine hook).
  `milesApi.md` listed this as one of the two hard blockers of the port.
- `overlay.md` CRITICAL GAP: AGISlime defined `eval_metrics_drain.drain()` but
  nothing trainer-side ever called it, so eval `step_*.json` files accumulated
  until the trainer died. `remove_trainer_alive` was likewise never called.
- The sidecar's `wandb_run_id` restore was dead code in both stacks: vendored
  slime and miles `init_wandb_primary` build `init_kwargs` without `id`/`resume`
  and then overwrite `args.wandb_run_id = wandb.run.id`; run continuity rode
  solely on the `WANDB_RUN_ID` pod env that `wandb.init` reads itself, and the
  parsed `--wandb-run-id` flag was ignored too (verify-redundancy, 2026-09-01).
- `wandb_extensions.register_arena_wandb_metrics` (113 lines as ported)
  re-declared step bindings miles' `_init_wandb_common` already declares
  (`train/*`, `rollout/*`, `multi_turn/*`, `passrate/*`, `perf/*`, `eval/*`);
  wandb 0.29.0 glob matching is prefix-based across `/`, so every arena
  declaration was a no-op duplicate.

## Decision

1. **Driver = miles `train_async.py` + insertions only.** `diff train_async.py
   miles_plugins/arena/train_async_arena.py | grep -c '^<'` is 0. Hooks are
   lazy-import, warn-only, and sit at four anchors: `_load_extra_state` before
   `init_tracking`; `_save_extra_state` + `_maybe_trigger_eval` right after the
   awaited actor `save_model`; `_maybe_drain_eval_metrics` once per iteration
   and `final=True` in the `__main__` `finally`; `_remove_trainer_alive`
   between the final drain and `finish_tracking()`. The drain wrapper is a
   strict no-op unless `use_wandb` and the queue dir exist.
2. **Sidecar kept, resume made real in core.** `checkpoint_extras.py` writes
   `{rollout_id, wandb_run_id}` on rank 0, best effort, under the historical
   filename (AGISlime interop verified both directions, byte-identical). The
   restored id is honoured by `init_wandb_primary` passing `id` +
   `resume='allow'` when `args.wandb_run_id` is pre-set (+6 lines,
   `wandb_utils.py:81-86`). `rollout_id` in the sidecar is log-only; miles
   derives the resume position from the Megatron checkpoint.
3. **`wandb_extensions.py` and the `_register_wandb_metrics` hook deleted.**

### Alternatives considered

| Alternative | Why not |
|-------------|---------|
| Port the sync `ray.get` driver mechanically | The APIs it calls are gone; the miles loop also adds hard requirements (object store init, data-ref release, `EvalDispatcher`, `save_trigger_sentinel`) that a hand-written loop would have to track forever. |
| Bespoke arena driver / subclassed loop | Loses the property that upstream `train_async.py` changes re-apply as a three-way merge of a copied file; the 0-deleted-lines invariant is checkable in one command. |
| Delete the sidecar, rely on `WANDB_RUN_ID` env | Works today but ties run identity to pod env instead of the checkpoint, and drops interop with AGISlime-written checkpoints. The env path stays supported alongside. |
| Plugin-side tracker (wrap `init_tracking` / monkeypatch `wandb.init`) | ADR-0001 allows a core edit where miles has no seam; 6 additive lines in the primary init beat a monkeypatch of shared tracking code. |
| Keep `wandb_extensions.py` | 100 % redundant with `_init_wandb_common`; keeping it would steer contributors away from the native tracking API. |
| Wire the drain into `log_perf_data_raw` as gym-evals did | On miles that runs inside the train actors, not the driver; miles has no patch point there. Per-iteration drain on the driver is the seam that exists. |

## Consequences

### Easier

- Every GPU run since 2026-09-01 16:39 PT (smoke `rl-milesgb1-smoke1`, GLM
  r1-r7) runs this driver; the loop is miles' own, so miles fixes (FT
  controller, control server, snapshot eval) apply unchanged.
- Eval metrics reach W&B while the trainer is alive (one committed row per
  queued file); `--wandb-run-id` now works for the primary writer.

### Harder / open

- Upstream edits to `train_async.py` must be re-applied by hand; keep the
  0-deleted-lines check in review. Hooks fire after every actor save, including
  miles-only `save_trigger_sentinel` saves (AGISlime gated them on the critic
  schedule).
- On a fresh start miles substitutes `--ref-load` for the not-yet-existing
  `--load` dir; the converted reference DCP's `latest_checkpointed_iteration.txt`
  reads `release`, so `_load_extra_state` logs
  `Failed to load slime extra state: invalid literal for int() ... 'release'`
  on every fresh run (observed at the smoke bring-up 2026-09-01 16:39-16:40 PT
  and at GLM run 2, 2026-09-02 14:15-14:16 PT).
  Harmless (warn-only); follow-up: skip when the marker is not an integer.
- The sidecar resume has not yet been exercised on GPU: r1/r2 died before
  `save_interval 20` and later runs used fresh `EXPERIMENT_NAME`s.
- The file-queue drain is a stopgap; the recorded follow-up is an eval
  coordinator that writes `eval/*` as a wandb shared-mode secondary writer.
