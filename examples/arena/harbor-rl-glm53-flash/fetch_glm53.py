"""Fetch zai-org/GLM-5.3-Flash-BF16 onto arena fast scratch (durable/).

Mountpoint-S3 has no rename support, so each file is downloaded to node-local
disk first, then stream-copied to its final path and size-verified. Re-runs
skip files that already match the pinned size (idempotent under Job retries).
"""
import concurrent.futures
import json
import os
import shutil
import sys
import time

from huggingface_hub import HfApi, hf_hub_download

REPO = "zai-org/GLM-5.3-Flash-BF16"
REV = "61f77a1e1a67c410650ce5017411337da0dcd11a"  # pinned 2026-08-31 upload
DEST = "/mnt/scratch-fast-1a-rw/durable/model-artifacts/external/zai-org/GLM-5.3-Flash-BF16"
TMP = "/tmp/hfdl"
WORKERS = 6

def fetch_one(entry):
    rel, want_size = entry
    final = os.path.join(DEST, rel)
    st = None
    try:
        st = os.stat(final)
    except FileNotFoundError:
        pass
    if st is not None:
        if st.st_size == want_size:
            return (rel, "skip", want_size)
        os.remove(final)  # short/corrupt object from an interrupted run
    for attempt in range(4):
        try:
            local = hf_hub_download(REPO, rel, revision=REV, local_dir=os.path.join(TMP, f"w{os.getpid()}-{abs(hash(rel)) % 8}"))
            got = os.path.getsize(local)
            if got != want_size:
                raise RuntimeError(f"local size {got} != expected {want_size}")
            os.makedirs(os.path.dirname(final), exist_ok=True)
            with open(local, "rb") as src, open(final, "wb") as dst:
                shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
            os.remove(local)
            wrote = os.stat(final).st_size
            if wrote != want_size:
                raise RuntimeError(f"dest size {wrote} != expected {want_size}")
            return (rel, "done", want_size)
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            print(f"[retry {attempt + 1}/4] {rel}: {exc}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"failed after retries: {rel}")

def main():
    os.makedirs(DEST, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    api = HfApi()
    tree = api.list_repo_tree(REPO, revision=REV, recursive=True)
    files = [(f.path, f.size) for f in tree if hasattr(f, "size") and f.size is not None]
    files = [f for f in files if f[0] != ".gitattributes"]
    total = sum(s for _, s in files)
    print(f"{len(files)} files, {total / 1e9:.1f} GB -> {DEST}", flush=True)
    done_bytes = 0
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for rel, state, size in pool.map(fetch_one, sorted(files, key=lambda x: -x[1])):
            done_bytes += size
            results.append((rel, size))
            print(f"[{state}] {rel} ({size / 1e9:.2f} GB) cumulative {done_bytes / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)
    manifest = {"repo": REPO, "revision": REV, "files": {r: s for r, s in results}, "total_bytes": total}
    with open(os.path.join(DEST, "_FETCH_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("SUCCESS", flush=True)

if __name__ == "__main__":
    sys.exit(main())
