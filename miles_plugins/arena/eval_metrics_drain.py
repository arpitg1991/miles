"""Trainer-side helpers for the eval-metrics file queue.

The eval coordinator writes per-step result files to a shared directory; the
trainer driver reads them and commits each one to wandb. The queue exists so
the coordinator itself stays wandb-free — not because a second wandb writer
would be unsafe: miles natively supports shared-mode secondary writers
(miles.utils.tracking_utils.wandb_utils.init_wandb_secondary; the rollout
manager and train actors already log this way). The recommended follow-up is
to have the coordinator log eval/* directly as a shared-mode secondary writer
and retire this file queue plus its heartbeat/lock protocol.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_TRAINER_ALIVE_NAME = ".trainer_alive"


def get_eval_metrics_dir(args) -> str | None:
    """Derive the shared eval-metrics queue path, mirroring eval_rollout.

    This dir is a LOCAL inter-pod IPC control-plane (mtime heartbeats,
    flock, os.listdir) shared between the trainer and the eval-coordinator
    pod — so it must live on a POSIX mount, never S3. It is anchored on
    ``args.save`` (the DCP checkpoint dir, which stays on /scratch), at
    ``<save>/eval_metrics``. Historically it was derived from ``args.save_hf``
    (``save_hf.parent.parent`` == ``args.save``), so this yields the SAME
    path; anchoring on ``save`` keeps it POSIX even when ``save_hf`` is an
    ``s3://`` URI. Returns None if no save path is configured (e.g. dry-runs).
    """
    save_dir = getattr(args, "save", None)
    if save_dir:
        return str(Path(save_dir) / "eval_metrics")
    # Fallback: derive from save_hf when --save is unset but --save-hf is a
    # local path (never for s3:// save_hf — that isn't a valid POSIX queue).
    save_hf = getattr(args, "save_hf", None)
    if not save_hf or save_hf.startswith("s3://"):
        return None
    try:
        rollout_0 = save_hf.format(
            rollout_id=0,
            experiment_name=getattr(args, "experiment_name", ""),
        )
        return str(Path(rollout_0).parent.parent / "eval_metrics")
    except (KeyError, IndexError):
        return None


def touch_trainer_alive(args) -> None:
    """Mark the trainer as alive. Idempotent — call from wandb init and from
    each per-iter drain to refresh the heartbeat mtime."""
    metrics_dir = get_eval_metrics_dir(args)
    if metrics_dir is None:
        return
    try:
        os.makedirs(metrics_dir, exist_ok=True)
        Path(metrics_dir, _TRAINER_ALIVE_NAME).touch()
    except OSError as e:
        logger.warning("Failed to touch trainer-alive sentinel: %s", e)


def remove_trainer_alive(args) -> None:
    """Call before wandb.finish() so any post-training coordinator knows to
    flush the queue itself."""
    metrics_dir = get_eval_metrics_dir(args)
    if metrics_dir is None:
        return
    try:
        os.unlink(os.path.join(metrics_dir, _TRAINER_ALIVE_NAME))
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("Failed to remove trainer-alive sentinel: %s", e)


def drain(args, final: bool = False) -> int:
    """Read every queued step_*.json, commit each to wandb, delete the file.
    Returns the number of files drained.

    Every file is committed as its own wandb row. The driver process this
    runs in makes no other wandb commits (train/rollout/perf metrics are
    logged from the megatron-actor and rollout-manager processes), so a
    ``commit=False`` payload would just sit staged: consecutive staged
    files share keys (``eval/step``, ``eval/<dataset>/...``), later ones
    would overwrite earlier ones, and only a single merged row would land
    at shutdown — losing every intermediate eval point. Committing each
    file preserves one row per eval step; eval line plots use
    ``eval/step`` as their x-axis via
    ``wandb.define_metric("eval/*", step_metric="eval/step")``, so where
    those rows fall on wandb's internal ``_step`` axis is irrelevant.

    ``final=True`` marks the shutdown drain (log annotation only — the
    flush behavior is identical).
    """
    metrics_dir = get_eval_metrics_dir(args)
    if metrics_dir is None or not os.path.isdir(metrics_dir):
        # Quietly bail: this is the common no-eval case (no save_hf set
        # for ad-hoc runs, or queue dir not created yet).
        return 0

    # Anything that prevents the drain when files exist is a real bug —
    # silently returning 0 then was how a recent regression hid behind
    # a missing log line. Surface those at WARNING so they're visible.
    pending_count = sum(
        1 for f in os.listdir(metrics_dir)
        if f.startswith("step_") and f.endswith(".json")
    )
    if not getattr(args, "use_wandb", False):
        if pending_count:
            logger.warning(
                "drain skipped: args.use_wandb is False but %d eval-metrics "
                "files are queued in %s. Files will accumulate.",
                pending_count, metrics_dir,
            )
        return 0
    try:
        import wandb
    except ImportError:
        if pending_count:
            logger.warning(
                "drain skipped: wandb not importable; %d files queued in %s",
                pending_count, metrics_dir,
            )
        return 0
    if wandb.run is None:
        if pending_count:
            logger.warning(
                "drain skipped: wandb.run is None in this process; %d files "
                "queued in %s. The trainer rank that calls drain must have "
                "an active wandb session.",
                pending_count, metrics_dir,
            )
        return 0

    # Refresh heartbeat so a slow eval doesn't think we died.
    try:
        Path(metrics_dir, _TRAINER_ALIVE_NAME).touch()
    except OSError:
        pass

    pending = sorted(
        f for f in os.listdir(metrics_dir)
        if f.startswith("step_") and f.endswith(".json")
    )
    if not pending:
        return 0

    drained = 0
    for fname in pending:
        fpath = os.path.join(metrics_dir, fname)
        try:
            payload = json.load(open(fpath))
        except Exception as e:
            logger.error("Failed to read %s: %s; leaving in queue", fpath, e)
            continue
        try:
            wandb.log(payload, commit=True)
            os.unlink(fpath)
            drained += 1
        except Exception as e:
            logger.error("wandb.log failed for %s: %s", fpath, e)
            break  # don't keep trying if wandb is broken
    if drained:
        logger.info(
            "Drained %d eval-metrics file(s) into wandb (final=%s)",
            drained, final,
        )
    return drained
