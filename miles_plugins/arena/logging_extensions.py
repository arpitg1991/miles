"""Arena logging extensions: eval-metrics drain integration and heartbeat.

Thin wrappers around eval_metrics_drain that the trainer's logging loop calls.
All path resolution is delegated to eval_metrics_drain to avoid inconsistencies.
"""

from __future__ import annotations

from typing import Any


def touch_trainer_alive(args: Any) -> None:
    """Touch a heartbeat file so the eval coordinator knows the trainer is alive."""
    from miles_plugins.arena.eval_metrics_drain import touch_trainer_alive as _touch

    _touch(args)


def is_trainer_alive(args: Any) -> bool:
    """Check if the trainer heartbeat is recent enough (delegates to drain module)."""
    from miles_plugins.arena.eval_metrics_drain import get_eval_metrics_dir

    import time
    from pathlib import Path

    metrics_dir = get_eval_metrics_dir(args)
    if metrics_dir is None:
        return False
    heartbeat = Path(metrics_dir) / ".trainer_alive"
    if not heartbeat.is_file():
        return False
    try:
        return (time.time() - heartbeat.stat().st_mtime) < 600.0
    except Exception:
        return False


def flush_eval_metrics(args: Any) -> None:
    """Flush pending eval metric files from the drain directory into W&B."""
    from miles_plugins.arena.eval_metrics_drain import drain

    drain(args)
