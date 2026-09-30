import hashlib
import logging

import torch

from miles.utils.r3_log import log_replay_digest, replay_digest


def test_replay_digest_hashes_the_stored_bytes_in_order():
    """The digest is sha256 of the raw buffer bytes in order, so the in-job and offline digests compare."""
    a = torch.arange(24, dtype=torch.int16).reshape(3, 8)
    b = torch.full((2, 8), -1, dtype=torch.int16)
    expected = hashlib.sha256(a.numpy().tobytes() + b.numpy().tobytes()).hexdigest()
    assert replay_digest([a, b]) == expected
    assert replay_digest([b, a]) != expected
    # A strided view of one layer hashes like its contiguous copy.
    routing = torch.arange(3 * 45 * 8, dtype=torch.int16).reshape(3, 45, 8)
    assert replay_digest([routing[:, 5]]) == replay_digest([routing[:, 5].contiguous()])


def test_log_replay_digest_line(caplog):
    buffers = [torch.zeros((4, 8), dtype=torch.int16)]
    with caplog.at_level(logging.INFO):
        log_replay_digest(logging.getLogger("test"), rollout=40, buffers=buffers)
    (line,) = [r.getMessage() for r in caplog.records]
    assert line.startswith("[r3-digest] rank=-1 node=")
    assert f"rollout=40 sha256={replay_digest(buffers)} buffers=1 bytes=64 dtype=int16 seconds=" in line
