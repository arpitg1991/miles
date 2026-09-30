"""Always-on log lines of the rollout routing replay (R3) data path.

Each event logs one ``[r3-<kind>] key=value ...`` line at INFO, with no spaces
inside a value:

- ``[r3-timing]``: one phase of the path from the rollout to the replay
  buffers (``load``, ``convert``, ``put``, ``fetch``, ``fill``,
  ``prefetch_start``, ``prefetch_done``, ``optimizer``).
- ``[r3-mem]``: host memory of one train rank.
- ``[r3-digest]``: sha256 of the replay buffers of one train rank.

``examples/arena/harbor-rl-glm53-flash/kdatp/prof/parse_r3_timing.py`` reads
them. ``rank`` is the global torch rank, or ``rm`` on the RolloutManager.
"""

import hashlib
import logging
import socket
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist


def train_rank() -> int:
    """The global torch rank, or -1 before the process group exists."""
    return dist.get_rank() if dist.is_initialized() else -1


def dtype_name(arrays: Sequence) -> str:
    """``int16`` for numpy or torch arrays of that dtype, ``-`` for no arrays."""
    return str(arrays[0].dtype).removeprefix("torch.") if len(arrays) else "-"


def r3_line(kind: str, *, rank: int | str, rollout: int | None, **fields: object) -> str:
    """``[r3-<kind>] rank=<rank> node=<host> rollout=<rollout> ...``; floats get 3 decimals."""
    items = {"rank": rank, "node": socket.gethostname(), "rollout": rollout, **fields}
    return f"[r3-{kind}] " + " ".join(
        f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}" for key, value in items.items()
    )


def log_r3_timing(
    log: logging.Logger,
    *,
    rank: int | str,
    rollout: int | None,
    phase: str,
    nbytes: int,
    t0: float,
    t1: float,
    **fields: object,
) -> None:
    log.info(
        r3_line(
            "timing", rank=rank, rollout=rollout, phase=phase, bytes=nbytes, seconds=t1 - t0, t0=t0, t1=t1, **fields
        )
    )


@contextmanager
def r3_timing(
    log: logging.Logger, *, rank: int | str, rollout: int | None, phase: str, **fields: object
) -> Iterator[dict]:
    """Time the body and log one ``[r3-timing]`` line.

    The body puts ``bytes`` and any extra keys into the dict that it gets. A
    body that raises logs no line.
    """
    extra: dict = {"bytes": 0}
    t0 = time.time()
    yield extra
    t1 = time.time()
    log_r3_timing(
        log, rank=rank, rollout=rollout, phase=phase, nbytes=extra.pop("bytes"), t0=t0, t1=t1, **fields, **extra
    )


def replay_digest(buffers: Sequence[torch.Tensor]) -> str:
    """sha256 over the raw bytes of ``buffers`` as stored, in order.

    The kdatp offline check (check_r3_fill.py) calls this function too, so the
    in-job digest and the reference digest hash the same bytes.
    """
    digest = hashlib.sha256()
    for buf in buffers:
        digest.update(memoryview(buf.contiguous().numpy()))
    return digest.hexdigest()


def log_replay_digest(log: logging.Logger, *, rollout: int, buffers: Sequence[torch.Tensor]) -> None:
    """One ``[r3-digest]`` line; ``seconds`` is the hash time, so the fill time does not include it."""
    t0 = time.time()
    digest = replay_digest(buffers)
    log.info(
        r3_line(
            "digest",
            rank=train_rank(),
            rollout=rollout,
            sha256=digest,
            buffers=len(buffers),
            bytes=sum(buf.nbytes for buf in buffers),
            dtype=dtype_name(buffers),
            seconds=time.time() - t0,
        )
    )


def log_r3_mem(log: logging.Logger, *, rollout: int, at: str) -> None:
    """One ``[r3-mem]`` line of this process and its node, in GiB, -1 where the file is missing.

    RssAnon, not VmRSS: VmRSS also counts the touched pages of the shared
    object store, and a private copy of a shard shows only in RssAnon.
    """
    status, meminfo = _kib_fields("/proc/self/status"), _kib_fields("/proc/meminfo")
    try:
        cgroup_gib = f"{int(Path('/sys/fs/cgroup/memory.current').read_text()) / 2**30:.2f}"
    except (OSError, ValueError):
        cgroup_gib = "-1"
    log.info(
        r3_line(
            "mem",
            rank=train_rank(),
            rollout=rollout,
            at=at,
            rss_gib=_gib(status.get("RssAnon")),
            rss_shmem_gib=_gib(status.get("RssShmem")),
            avail_gib=_gib(meminfo.get("MemAvailable")),
            shmem_gib=_gib(meminfo.get("Shmem")),
            cgroup_gib=cgroup_gib,
        )
    )


def _kib_fields(path: str) -> dict[str, int]:
    """The ``Name: <n> kB`` lines of a /proc file."""
    try:
        text = Path(path).read_text()
    except OSError:
        return {}
    fields = {}
    for line in text.splitlines():
        name, _, value = line.partition(":")
        parts = value.split()
        if len(parts) == 2 and parts[1] == "kB":
            fields[name] = int(parts[0])
    return fields


def _gib(kib: int | None) -> str:
    return "-1" if kib is None else f"{kib / 2**20:.2f}"
