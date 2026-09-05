"""Arena-wrapped async training loop.

Thin wrapper around miles' train_async.py that adds arena-specific hooks
at the right points in the training loop WITHOUT modifying miles core code:

1. Checkpoint sidecar (wandb_run_id + counters) after save_model
2. Fire-and-forget per-checkpoint hosted eval (argo_eval_trigger) after save_model
3. Trainer-side eval-metrics queue drain (eval_metrics_drain) once per loop
   iteration and once, with final=True, before finish_tracking
4. Trainer-alive heartbeat removal (eval_metrics_drain.remove_trainer_alive)
   between the final drain and finish_tracking, so a post-training eval
   coordinator flushes its queue immediately instead of waiting out the
   heartbeat-staleness window

Arena W&B metric step bindings need no plugin-side declarations: miles'
_init_wandb_common already declares train/*, rollout/*, multi_turn/*,
passrate/*, perf/* and eval/*, and wandb glob matching is prefix-based
(crossing '/'), so those cover every arena key.

Off-policy staleness (rollout/off_policy_round/*) and rollout/binary_reward are
logged inside the rollout function (nats_arena/nats_rollout.py) where the Sample
objects — and their SGLang-tagged weight_versions / rewards — actually live.
Weight versions are propagated to the rollout engines natively by miles'
update-weight backend (the trainer calls rollout_manager.set_weight_version),
so no wrapper-side propagation is needed here.

Usage (in entrypoint.sh):
    python3 -m miles_plugins.arena.train_async_arena
    # instead of: python3 train_async.py
"""

import asyncio
import logging
import os

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.utils import object_store
from miles.utils.arguments import parse_args, validate_async_off_policy_correction
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.ft_utils.control_server.server import start_control_server
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

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
        from pathlib import Path
        import re

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


# The framework supports other asynchronous approaches such as fully async (see miles/rollout/fully_async_rollout.py).
async def train(args):
    """Arena-wrapped async training loop."""
    assert not args.colocate, "Colocation is not supported for async training."
    validate_async_off_policy_correction(args)
    configure_logger(args, source=MainProcessIdentity())
    maybe_start_periodic_pyspy_dump()

    # Load checkpoint sidecar BEFORE init_tracking so wandb_run_id is set
    _load_extra_state(args)

    # allocate the GPUs
    pgs = create_placement_groups(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    if args.control_server_port:
        start_control_server(
            actor_model=actor_model,
            rollout_manager=rollout_manager,
            port=args.control_server_port,
            ft_components=args.ft_components,
        )

    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_manager)

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def save_training_model(model, rollout_id, force_sync):
        if args.use_critic and args.offload_train:
            await model.onload()
        await model.save_model(rollout_id, force_sync=force_sync)
        if args.use_critic and args.offload_train:
            await model.offload()

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = await rollout_data_next_future

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            values = await critic_model.train(rollout_id, rollout_data_curr_ref)
            if args.offload_train:
                await critic_model.offload()
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_curr_ref, external_data=values)
                if args.offload_train:
                    await actor_model.offload()
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
            await rollout_manager.save.remote(rollout_id)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            await actor_model.update_weights(rollout_id=rollout_id)

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout):
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

    await eval_dispatcher.drain()
    await rollout_manager.dispose.remote()


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


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        _maybe_drain_eval_metrics(args, final=True)
        _remove_trainer_alive(args)
        finish_tracking()
