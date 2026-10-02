"""Snapshot/restore the event directory alongside model checkpoints."""

import logging
import shutil
import time
import uuid
from argparse import Namespace
from pathlib import Path

from miles.backends.megatron_utils.checkpoint_tracker import read_checkpoint_tracker_iteration
from miles.backends.megatron_utils.megatron_config import compute_trainer_checkpoint_dir, resolve_megatron_config

logger = logging.getLogger(__name__)


def snapshot(args: Namespace, iteration: int) -> None:
    if args.save_debug_event_data is None or args.save is None:
        return

    src = Path(args.save_debug_event_data)
    if not src.is_dir():
        return

    dst = _snapshot_dir(Path(args.save), iteration)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _copy_file_contents(src, dst)
    logger.info("Snapshotted event dir %s -> %s", src, dst)


def restore(args: Namespace) -> None:
    if args.save_debug_event_data is None or args.requested_load is None:
        return

    requested_load = Path(args.requested_load)
    iteration = _read_checkpoint_iteration(args)
    if iteration is None:
        return

    src = _snapshot_dir(requested_load, iteration)
    if not src.is_dir():
        return

    dst = Path(args.save_debug_event_data)
    if dst.exists():
        trash = _move_aside(dst)
        logger.info("Moved pre-restore event dir %s -> %s", dst, trash)
    _copy_file_contents(src, dst)
    logger.info("Restored event dir %s <- %s", dst, src)


def discard(args: Namespace) -> None:
    if args.save_debug_event_data is None:
        return

    dst = Path(args.save_debug_event_data)
    assert not dst.is_symlink(), f"a take-over moves the event log aside, and {dst} is a symlink"
    if not dst.is_dir() or not any(dst.iterdir()):
        return

    # TODO: startup events the new incarnation already wrote go into the trash with the abandoned log
    trash = _move_aside(dst)
    dst.mkdir(parents=True)
    logger.info("Moved the log of the run a hot restart takes over %s -> %s", dst, trash)


def _read_checkpoint_iteration(args: Namespace) -> int | None:
    leader = resolve_megatron_config(args).trainers[0]
    load_dir = (
        compute_trainer_checkpoint_dir(base_dir=args.requested_load, trainer_id=leader.trainer_id)
        if leader.model_id is not None
        else args.requested_load
    )
    return read_checkpoint_tracker_iteration(load_dir)


def _move_aside(dst: Path) -> Path:
    trash = dst.parent / f".trash_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    dst.rename(trash)
    return trash


def _copy_file_contents(src: Path, dst: Path) -> None:
    # Copy bytes only. shutil.copytree also copies the xattrs of every file and directory, and S3 Files
    # lists a `user.s3files.status` xattr on each of them that it refuses to set with errno 524 (ENOTSUPP).
    dst.mkdir(parents=True)
    for entry in src.iterdir():
        if entry.is_dir():
            _copy_file_contents(entry, dst / entry.name)
        else:
            shutil.copyfile(entry, dst / entry.name)


def _snapshot_dir(checkpoint_root: Path, iteration: int) -> Path:
    return checkpoint_root / f"iter_{iteration:07d}" / "debug_events"
