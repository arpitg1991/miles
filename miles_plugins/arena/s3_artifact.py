"""S3 artifact helpers for writing checkpoints/traces directly to S3.

Training artifacts (HF/DCP checkpoints, OTel traces) normally land on the
``/scratch`` NFS mount, which is itself the ``arena-scratch-*`` S3 bucket
exposed via AWS S3 Files. Going through NFS is a poor fit for multi-GB
checkpoint shards: S3 Files does not auto-import objects >= 5 GB, expires its
cache after a day, and versions every overwrite. These helpers let a writer
target an ``s3://bucket/key/prefix`` URI and PUT via the AWS API directly,
preserving the exact sub-path layout used under ``/scratch/<alias>/``.

Credentials come from the default boto3 chain — in-cluster that is EKS Pod
Identity (service account ``arena-s3-access`` -> role
``arena-workload-*``), which already has read/write on the scratch bucket.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def is_s3_uri(uri: str | os.PathLike | None) -> bool:
    """True when ``uri`` is an ``s3://`` URI (a plain string, not a Path)."""
    return isinstance(uri, str) and uri.startswith("s3://")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key/prefix`` into ``(bucket, key_prefix)``.

    ``key_prefix`` has no leading or trailing slash.
    """
    rest = uri[len("s3://") :]
    bucket, _, key_prefix = rest.partition("/")
    return bucket, key_prefix.strip("/")


_CLIENT = None


def _client():
    """Return a cached S3 client pinned to the artifact bucket's real region.

    The trainer/eval images bake ``AWS_REGION=us-east-1``, but the scratch
    artifact bucket lives in ``ap-south-1``. A client with the wrong region
    fails cross-region writes, so resolve the region from the env override
    (``S3_ARTIFACT_REGION``, set by entrypoint.sh) and fall back to the
    default chain. Cached because clients are cheap to reuse but not free
    to build per call.
    """
    global _CLIENT
    if _CLIENT is None:
        import boto3  # local import so non-S3 runs never require boto3

        region = os.environ.get("S3_ARTIFACT_REGION") or os.environ.get("AWS_REGION")
        _CLIENT = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    return _CLIENT


def upload_file(local_path: str | os.PathLike, s3_uri: str) -> None:
    """Upload a single local file to a fully-qualified ``s3://bucket/key``.

    boto3's ``upload_file`` does multipart automatically for large objects,
    so multi-GB safetensors shards stream up in parallel parts.
    """
    bucket, key = parse_s3_uri(s3_uri)
    _client().upload_file(str(local_path), bucket, key)


def upload_dir(local_dir: str | os.PathLike, s3_prefix_uri: str) -> int:
    """Recursively upload every file under ``local_dir`` to ``s3_prefix_uri``.

    Mirrors the local tree under the S3 key prefix. Returns the number of
    files uploaded. Empty dirs are skipped (S3 has no directories).
    """
    root = Path(local_dir)
    bucket, prefix = parse_s3_uri(s3_prefix_uri)
    client = _client()
    count = 0
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(root).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
        client.upload_file(str(f), bucket, key)
        count += 1
    return count
