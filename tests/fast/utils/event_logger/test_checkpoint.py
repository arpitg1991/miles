"""Tests for miles.utils.audit_utils.event_logger.checkpoint."""

import os
import shutil
from argparse import Namespace
from pathlib import Path

import pytest

from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint


def _args(
    *,
    event_dir: Path | None,
    save: Path | None = None,
    load: Path | None = None,
    requested_load: Path | None = None,
    megatron_config: str | None = None,
) -> Namespace:
    return Namespace(
        save_debug_event_data=str(event_dir) if event_dir else None,
        save=str(save) if save else None,
        load=str(load) if load else None,
        requested_load=str(one) if (one := requested_load or load) else None,
        megatron_config=megatron_config,
    )


def _write_tracker(ckpt: Path, content: str) -> None:
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "latest_checkpointed_iteration.txt").write_text(content)


class TestSnapshotRestoreRoundtrip:
    def test_restore_replaces_live_dir_with_snapshot(self, tmp_path: Path) -> None:
        """A resumed run sees exactly the snapshotted events, not the live dir's leftovers."""
        ckpt = tmp_path / "ckpt"
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("committed\n")
        event_logger_checkpoint.snapshot(_args(event_dir=events, save=ckpt), iteration=3)

        # Events written after the save (would be re-executed by the resumed run).
        (events / "main.jsonl").write_text("committed\nrewound-future\n")
        (events / "straggler.jsonl").write_text("late\n")
        _write_tracker(ckpt, "3")
        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt))

        assert (events / "main.jsonl").read_text() == "committed\n"
        assert not (events / "straggler.jsonl").exists()

    def test_a_named_trainer_restores_from_the_tracker_in_its_own_namespace(self, tmp_path: Path) -> None:
        """Such a run writes every trainer's checkpoints, tracker included, under `<save>/trainers/<trainer_id>/`."""
        ckpt = tmp_path / "ckpt"
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("committed\n")
        config = encode_megatron_config("policy")
        event_logger_checkpoint.snapshot(_args(event_dir=events, save=ckpt, megatron_config=config), iteration=3)

        (events / "main.jsonl").write_text("committed\nrewound-future\n")
        _write_tracker(ckpt / "trainers" / "policy-actor", "3")
        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt, megatron_config=config))

        assert (events / "main.jsonl").read_text() == "committed\n"

    def test_snapshot_overwrites_previous_snapshot_of_same_iteration(self, tmp_path: Path) -> None:
        """Re-saving the same iteration replaces its snapshot."""
        ckpt = tmp_path / "ckpt"
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("v1\n")
        event_logger_checkpoint.snapshot(_args(event_dir=events, save=ckpt), iteration=1)
        (events / "main.jsonl").write_text("v2\n")
        event_logger_checkpoint.snapshot(_args(event_dir=events, save=ckpt), iteration=1)

        assert (ckpt / "iter_0000001" / "debug_events" / "main.jsonl").read_text() == "v2\n"


# Linux kernel ENOTSUPP; the errno module has no name for it.
ENOTSUPP = 524
S3_FILES_XATTR = "user.s3files.status;<arbitrary>"


def _write_event_tree(root: Path) -> None:
    files = {
        "main.jsonl": b'{"rollout_id": 68}\n',
        "actor_cell00000_rank00040.jsonl": b'{"rank": 40}\n' * 1000,
        "nested/worker_manager.jsonl": b"",
    }
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)
    (root / "empty").mkdir()


def _read_tree(root: Path) -> dict[str, bytes | None]:
    return {str(p.relative_to(root)): None if p.is_dir() else p.read_bytes() for p in root.rglob("*")}


class TestS3FilesMount:
    """S3 Files lists an xattr on every file and directory and rejects a copy of it with errno 524 (ENOTSUPP)."""

    @pytest.fixture
    def s3_files_xattr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _reject(path: object, *args: object, **kwargs: object) -> None:
            raise OSError(ENOTSUPP, os.strerror(ENOTSUPP), path)

        monkeypatch.setattr(os, "listxattr", lambda *args, **kwargs: [S3_FILES_XATTR])
        monkeypatch.setattr(os, "getxattr", lambda *args, **kwargs: b"")
        monkeypatch.setattr(os, "setxattr", _reject)

    def test_snapshot_copies_every_file(self, tmp_path: Path, s3_files_xattr: None) -> None:
        """The run must not die at its first checkpoint save, as four runs did at iteration 69."""
        events = tmp_path / "events"
        events.mkdir()
        _write_event_tree(events)
        with pytest.raises(shutil.Error, match="Errno 524"):
            shutil.copytree(events, tmp_path / "copytree")

        event_logger_checkpoint.snapshot(_args(event_dir=events, save=tmp_path / "ckpt"), iteration=69)

        snapshot = tmp_path / "ckpt" / "iter_0000069" / "debug_events"
        assert _read_tree(snapshot) == _read_tree(events)

    def test_restore_copies_every_file(self, tmp_path: Path, s3_files_xattr: None) -> None:
        """A resumed run gets back exactly the snapshotted events."""
        ckpt = tmp_path / "ckpt"
        snapshot = ckpt / "iter_0000069" / "debug_events"
        snapshot.mkdir(parents=True)
        _write_event_tree(snapshot)
        _write_tracker(ckpt, "69")
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_bytes(b"rewound-future\n")

        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt))

        assert _read_tree(events) == _read_tree(snapshot)


class TestNoOpCases:
    def test_restore_skips_when_not_resuming(self, tmp_path: Path) -> None:
        """No --load means no restore."""
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("keep\n")

        event_logger_checkpoint.restore(_args(event_dir=events))

        assert (events / "main.jsonl").read_text() == "keep\n"

    def test_restore_skips_when_checkpoint_has_no_snapshot(self, tmp_path: Path) -> None:
        """Checkpoints predating event snapshots leave the live dir untouched."""
        ckpt = tmp_path / "ckpt"
        _write_tracker(ckpt, "2")
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("keep\n")

        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt))

        assert (events / "main.jsonl").read_text() == "keep\n"

    def test_restore_skips_a_reference_checkpoint_a_fresh_run_fell_back_to(self, tmp_path: Path) -> None:
        """A run whose --load holds nothing resumable must not import the reference events."""
        ref = tmp_path / "ref"
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("fresh\n")
        event_logger_checkpoint.snapshot(_args(event_dir=events, save=ref), iteration=7)
        _write_tracker(ref, "7")

        event_logger_checkpoint.restore(_args(event_dir=events, load=ref, requested_load=tmp_path / "run"))

        assert (events / "main.jsonl").read_text() == "fresh\n"

    def test_restore_skips_release_tracker(self, tmp_path: Path) -> None:
        """A non-numeric tracker (e.g. 'release') is not a resumable iteration."""
        ckpt = tmp_path / "ckpt"
        _write_tracker(ckpt, "release")
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("keep\n")

        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt))

        assert (events / "main.jsonl").read_text() == "keep\n"

    def test_restore_of_a_named_trainer_ignores_a_tracker_left_at_the_root(self, tmp_path: Path) -> None:
        """The cursor such a run rewinds to is its own, so a root tracker is some other layout's."""
        ckpt = tmp_path / "ckpt"
        _write_tracker(ckpt, "3")
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("keep\n")
        (ckpt / "iter_0000003" / "debug_events").mkdir(parents=True)

        event_logger_checkpoint.restore(
            _args(event_dir=events, load=ckpt, megatron_config=encode_megatron_config("policy"))
        )

        assert (events / "main.jsonl").read_text() == "keep\n"

    def test_snapshot_skips_when_events_disabled_or_no_save(self, tmp_path: Path) -> None:
        """Disabled events or no save dir means no snapshot."""
        events = tmp_path / "events"
        events.mkdir()

        event_logger_checkpoint.snapshot(_args(event_dir=None, save=tmp_path / "ckpt"), iteration=1)
        event_logger_checkpoint.snapshot(_args(event_dir=events), iteration=1)

        assert not (tmp_path / "ckpt").exists()


class TestDiscardEventLog:
    def test_moves_the_log_aside_and_leaves_an_empty_directory(self, tmp_path: Path) -> None:
        """The live loggers reopen this path on every write, so a directory that vanished crashes them."""
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("from the run being taken over\n")

        event_logger_checkpoint.discard(_args(event_dir=events))

        [trash] = list(tmp_path.glob(".trash_*"))
        assert (trash / "main.jsonl").read_text() == "from the run being taken over\n"
        assert events.is_dir() and list(events.iterdir()) == []

    def test_leaves_an_empty_directory_alone(self, tmp_path: Path) -> None:
        """A take-over of a run that logged nothing has nothing to throw away, and leaves no empty trash."""
        events = tmp_path / "events"
        events.mkdir()

        event_logger_checkpoint.discard(_args(event_dir=events))

        assert list(tmp_path.glob(".trash_*")) == []
        assert events.is_dir()

    def test_leaves_a_missing_directory_alone(self, tmp_path: Path) -> None:
        """The run may be taken over before it opened its log at all."""
        events = tmp_path / "events"

        event_logger_checkpoint.discard(_args(event_dir=events))

        assert list(tmp_path.glob(".trash_*")) == []
        assert not events.exists()

    def test_refuses_a_live_symlink(self, tmp_path: Path) -> None:
        """Moving the link aside would leave the loggers writing into a directory nobody reads."""
        target = tmp_path / "elsewhere"
        target.mkdir()
        (target / "main.jsonl").write_text("not this run's to move\n")
        events = tmp_path / "events"
        events.symlink_to(target)

        with pytest.raises(AssertionError, match="is a symlink"):
            event_logger_checkpoint.discard(_args(event_dir=events))

        assert events.is_symlink()
        assert (target / "main.jsonl").read_text() == "not this run's to move\n"

    def test_refuses_a_dangling_symlink(self, tmp_path: Path) -> None:
        """A dangling link is not a directory, so the emptiness check alone would pass it silently."""
        events = tmp_path / "events"
        events.symlink_to(tmp_path / "gone")

        with pytest.raises(AssertionError, match="is a symlink"):
            event_logger_checkpoint.discard(_args(event_dir=events))

        assert events.is_symlink()

    def test_leaves_a_run_that_logs_nowhere_alone(self, tmp_path: Path) -> None:
        """Without --save-debug-event-data there is no log to move, and nothing may be created."""
        event_logger_checkpoint.discard(_args(event_dir=None))

        assert list(tmp_path.iterdir()) == []
