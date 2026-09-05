"""Trainer checkpoint sidecar: persist/restore wandb_run_id and rollout counters.

Saves a JSON sidecar alongside Megatron's iter_%07d checkpoint directory.
On restart the driver restores ``args.wandb_run_id`` from the sidecar before
init_tracking (train_async_arena._load_extra_state), and miles'
init_wandb_primary resumes that run (``id=args.wandb_run_id,
resume='allow'``); setting the ``WANDB_RUN_ID`` pod env (read by wandb.init
itself) is the other supported resume path. The rollout_id entry is
informational only — miles derives the resume position from the Megatron
checkpoint natively. Standalone — importable without Megatron on PATH.

The filename and function names keep the historical ``slime`` prefix so the
sidecar stays byte-compatible with checkpoints written by the AGISlime
(slime 0.3.0) trainer — do not rename.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SLIME_EXTRA_STATE_FILENAME = "slime_extra_state.json"


def _extra_state_path(checkpoints_path: str | Path, iteration: int) -> Path:
    return Path(checkpoints_path) / f"iter_{int(iteration):07d}" / _SLIME_EXTRA_STATE_FILENAME


def save_slime_extra_state(
    save_dir: str | Path, iteration: int, extra_state: dict, rank: int = 0
) -> None:
    """Persist trainer-specific state alongside the checkpoint.

    Only rank 0 should write. Best-effort: failures log a warning, never abort.
    """
    if rank != 0:
        return
    path = _extra_state_path(save_dir, iteration)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(extra_state, f)
        logger.info("Saved slime extra state to %s: %s", path, extra_state)
    except Exception as e:
        logger.warning("Failed to save slime extra state to %s: %s", path, e)


def load_slime_extra_state(load_dir: str | Path, iteration: int) -> dict:
    """Load the extra-state sidecar for ``iteration``, or ``{}``.

    Returns empty dict when the sidecar is absent (checkpoint predates this
    feature, or HF cold start).
    """
    path = _extra_state_path(load_dir, iteration)
    if not path.is_file():
        return {}
    try:
        with open(path) as f:
            extra_state = json.load(f)
        logger.info("Loaded slime extra state from %s: %s", path, extra_state)
        return extra_state
    except Exception as e:
        logger.warning("Failed to load slime extra state from %s: %s", path, e)
        return {}
