"""Arena-wrapped async training loop.

Thin wrapper around miles' train_async.py that adds arena-specific hooks
at the right points in the training loop WITHOUT modifying miles core code:

1. Checkpoint sidecar (wandb_run_id + counters) after save_model
2. Fire-and-forget per-checkpoint hosted eval (argo_eval_trigger) after save_model
3. Trainer-side eval-metrics queue drain (eval_metrics_drain) once per loop
   iteration and once, with final=True, at teardown before finish_tracking
4. Trainer-alive heartbeat removal (eval_metrics_drain.remove_trainer_alive)
   between the final drain and finish_tracking, so a post-training eval
   coordinator flushes its queue immediately instead of waiting out the
   heartbeat-staleness window

The final drain and the heartbeat removal are one disposer callback. It is
registered right after init_orchestration_script, which registers
finish_tracking, so the disposer stack runs it after every later teardown
(eval dispatcher, models, rollout components) and before finish_tracking.

Arena W&B metric step bindings need no plugin-side declarations: miles'
_init_wandb_common already declares train/*, rollout/*, multi_turn/*,
passrate/*, perf/* and eval/*, and wandb glob matching is prefix-based
(crossing '/'), so those cover every arena key.

Off-policy staleness (rollout/off_policy_round/*) and rollout/binary_reward are
logged inside the rollout function (nats_arena/nats_rollout.py) where the Sample
objects live. The gym-reported per-turn weight versions ride in
Sample.metadata["arena_weight_versions"] (ADR-0018). miles' update_weights
publishes each new weight version to the rollout executor, so no
wrapper-side propagation is needed here.

Usage (in entrypoint.sh):
    python3 -m miles_plugins.arena.train_async_arena
    # instead of: python3 train_async.py
"""

import asyncio
import logging
import os
from functools import partial

from miles.ray.placement_group import (
    create_rollout_components,
    create_training_models,
    maybe_start_api_server,
    update_weights,
)
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.utils.arguments import parse_args, validate_async_off_policy_correction
from miles.utils.async_utils import Disposer, eager_create_task, with_disposer
from miles.utils.data import remove_rollout_data_refs, remove_train_output_refs
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.misc import should_run_periodic_action
from miles.utils.orchestration_utils import init_orchestration_script

logger = logging.getLogger(__name__)


def _save_extra_state(args, rollout_id: int):
    """Save checkpoint sidecar with wandb_run_id and rollout counters."""
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        return
    try:
        from miles_plugins.arena.checkpoint_extras import save_slime_extra_state

        extra = {"rollout_id": rollout_id}
        try:
            import wandb

            if wandb.run is not None:
                extra["wandb_run_id"] = wandb.run.id
        except Exception:
            pass
        save_dir = getattr(args, "save", None)
        if save_dir:
            save_slime_extra_state(save_dir, rollout_id, extra)
    except Exception as e:
        logger.warning("Failed to save slime extra state: %s", e)


def _maybe_trigger_eval(args, rollout_id: int):
    """Fire-and-forget per-checkpoint eval (no-op unless ARENA_EVAL_TASKS set).

    Runs on the driver right after the (blocking) save_model, so the HF
    checkpoint is flushed. Never raises — a submit failure must not break
    training.
    """
    try:
        from miles_plugins.arena.nats_arena.argo_eval_trigger import submit_eval

        submit_eval(args, rollout_id)
    except Exception as e:  # noqa: BLE001 - never interrupt training
        logger.warning("Eval trigger failed for rollout_id=%d: %s", rollout_id, e)


def _load_extra_state(args):
    """Load checkpoint sidecar and restore wandb_run_id if available.

    A restored args.wandb_run_id makes miles' primary wandb init resume that
    run (id=args.wandb_run_id, resume='allow'); the WANDB_RUN_ID pod env
    (read by wandb.init itself) is the other supported resume path. The
    sidecar's rollout_id is log-only — miles derives the resume position
    from the Megatron checkpoint natively.
    """
    try:
        from miles_plugins.arena.checkpoint_extras import load_slime_extra_state

        load_dir = getattr(args, "load", None)
        if not load_dir:
            return
        import re
        from pathlib import Path

        latest_file = Path(load_dir) / "latest_checkpointed_iteration.txt"
        if latest_file.is_file():
            iteration = int(latest_file.read_text().strip())
        else:
            dirs = [d for d in Path(load_dir).iterdir() if d.is_dir() and re.match(r"iter_\d+", d.name)]
            if not dirs:
                return
            iteration = max(int(d.name.split("_")[1]) for d in dirs)
        extra = load_slime_extra_state(load_dir, iteration)
        if extra.get("wandb_run_id") and not getattr(args, "wandb_run_id", None):
            args.wandb_run_id = extra["wandb_run_id"]
            logger.info("Restored wandb_run_id=%s from checkpoint sidecar.", args.wandb_run_id)
        if extra.get("rollout_id"):
            logger.info("Checkpoint sidecar rollout_id=%d.", extra["rollout_id"])
    except Exception as e:
        logger.warning("Failed to load slime extra state: %s", e)


def _maybe_drain_eval_metrics(args, final: bool = False):
    """Drain queued eval-metrics files (step_*.json) into the trainer's wandb run.

    The eval coordinator writes per-step result files to <save>/eval_metrics
    and the driver commits each one as its own wandb row (see
    eval_metrics_drain; the recommended follow-up is for the coordinator to
    log eval/* directly as a wandb shared-mode secondary writer, which miles
    supports natively, and retire this queue). Called once per training-loop
    iteration and once, with final=True, before finish_tracking. Never
    raises — a drain failure must not break training. No-op unless wandb is
    on and the queue dir exists (the harbor-rl-27b smoke job runs with wandb
    off).
    """
    if not getattr(args, "use_wandb", False):
        return
    try:
        from miles_plugins.arena.eval_metrics_drain import drain, get_eval_metrics_dir

        metrics_dir = get_eval_metrics_dir(args)
        if metrics_dir is None or not os.path.isdir(metrics_dir):
            return
        drain(args, final=final)
    except Exception as e:  # noqa: BLE001 - never interrupt training
        logger.warning("Eval-metrics drain failed: %s", e)


def _remove_trainer_alive(args):
    """Clear the trainer-alive heartbeat so a post-training eval coordinator
    flushes the metrics queue itself instead of waiting for mtime staleness.
    Never raises — shutdown must proceed regardless.
    """
    try:
        from miles_plugins.arena.eval_metrics_drain import remove_trainer_alive

        remove_trainer_alive(args)
    except Exception as e:  # noqa: BLE001 - never interrupt shutdown
        logger.warning("Failed to remove trainer-alive sentinel: %s", e)


def _finish_arena(args):
    """Run the final eval-metrics drain, then clear the trainer-alive heartbeat.

    The disposer runs this after the eval dispatcher, the models, and the
    rollout components are torn down, and before finish_tracking. Never
    raises: both steps are warn-only.
    """
    _maybe_drain_eval_metrics(args, final=True)
    _remove_trainer_alive(args)


# The framework supports other asynchronous approaches such as fully async (see miles/rollout/fully_async_rollout.py).
async def train(args, *, disposer: Disposer):
    assert not args.colocate, "Colocation is not supported for async training."
    validate_async_off_policy_correction(args)
    # Load checkpoint sidecar BEFORE init_tracking so wandb_run_id is set
    _load_extra_state(args)
    _worker_manager = init_orchestration_script(args, disposer=disposer)
    disposer.add(partial(_finish_arena, args))

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)
    disposer.add(inference_controller, rollout_executor)

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, rollout_executor)
    disposer.add(critic_model, actor_model)

    maybe_start_api_server(args, trainer_models={"actor": actor_model}, inference_controller=inference_controller)
    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await update_weights(args, actor_model, rollout_executor, inference_controller)

    if args.check_weight_update_equal:
        await inference_controller.check_weights(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_executor)
    disposer.add(eval_dispatcher.drain)

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        await inference_controller.prepare_eval()
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def save_training_model(model, rollout_id, force_sync):
        if args.use_critic and args.offload_train:
            await model.onload()
        await model.save_model(rollout_id, force_sync=force_sync)
        if args.use_critic and args.offload_train:
            await model.offload()

    async def prepare_and_generate(rollout_id):
        await inference_controller.prepare_rollout(rollout_id)
        return await rollout_executor.get(rollout_id)

    async def prefetch_when_generated(rollout_id, rollout_data_future):
        try:
            await actor_model.prefetch_rollout_data(rollout_id, await rollout_data_future)
        except Exception as e:  # noqa: BLE001 - a failed prefetch only makes train() fetch the shard itself
            logger.warning("Prefetch of rollout %d failed, train() fetches it itself: %r", rollout_id, e)

    # async train loop.
    rollout_data_next_future = await eager_create_task(prepare_and_generate(args.start_rollout_id))
    prefetch_tasks: set[asyncio.Task] = set()  # the event loop keeps only weak references to tasks
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = await rollout_data_next_future

        has_next_rollout = rollout_id + 1 < args.num_rollout
        weight_update_due = (rollout_id + 1) % args.update_weights_interval == 0

        # A fully-async producer keeps generating without a pending get(). When
        # weights will change, defer the next drain so it uses the new version.
        defer_next_drain = args.fully_async and has_next_rollout and weight_update_due
        if has_next_rollout and not defer_next_drain:
            rollout_data_next_future = await eager_create_task(prepare_and_generate(rollout_id + 1))
            if args.prefetch_rollout_data:
                # When generate ends, the train nodes pull the next shards while this step trains.
                task = asyncio.create_task(prefetch_when_generated(rollout_id + 1, rollout_data_next_future))
                prefetch_tasks.add(task)
                task.add_done_callback(prefetch_tasks.discard)

        if args.use_critic:
            values = await critic_model.train(rollout_id, rollout_data_curr_ref)
            if args.offload_train:
                await critic_model.offload()
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_curr_ref, external_data=values)
                if args.offload_train:
                    await actor_model.offload()
            remove_train_output_refs(values)
        else:
            await actor_model.train(rollout_id, rollout_data_curr_ref)
        remove_rollout_data_refs(args, rollout_data_curr_ref)

        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        if external_save or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = external_save or rollout_id == args.num_rollout - 1
            await save_training_model(actor_model, rollout_id, force_sync)
            _save_extra_state(args, rollout_id)
            _maybe_trigger_eval(args, rollout_id)
            if args.use_critic:
                await save_training_model(critic_model, rollout_id, force_sync)
            await rollout_executor.save(rollout_id)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        if weight_update_due:
            if not args.fully_async:
                # sync generate before update weights to prevent update weight in the middle of generation
                rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None
                rollout_data_next_future = None
            await update_weights(args, actor_model, rollout_executor, inference_controller, rollout_id=rollout_id)
            if defer_next_drain:
                rollout_data_next_future = await eager_create_task(prepare_and_generate(rollout_id + 1))

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout):
            await inference_controller.prepare_eval()
            await eval_dispatcher.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)

        # Forward any eval-coordinator metrics queued since the last iteration.
        _maybe_drain_eval_metrics(args)

        if (
            args.debug_exit_after_rollout is not None
            and (rollout_id - args.start_rollout_id + 1) >= args.debug_exit_after_rollout
        ):
            logger.info(
                "debug_exit_after_rollout=%d reached at rollout_id=%d, exiting",
                args.debug_exit_after_rollout,
                rollout_id,
            )
            break


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(with_disposer(train, args))
