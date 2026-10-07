"""Merge the /tmp/kernel_cache tarballs of several warm pods into one seed tar (kernel_cache.tar).

Units are never mixed across pods: a Triton kernel directory (kernel_cache/triton/<hash>/) and a
TileLang kernel directory (kernel_cache/tilelang/<ver>/kernels/<hash>/) come whole from the first
pod that has them; an inductor file (each keyed by its own hash) is one unit. The same key built on
two pods can differ in autotune timings or debug info, and mixing the files of one Triton kernel
from two builds could pair a cubin with another build's metadata.

Usage: python3 merge_kcache.py <out.tar> <kc-wN.tgz> [<kc-wM.tgz> ...]   (first tarball wins)
"""

import os
import shutil
import sys
import tarfile
import tempfile


def main() -> None:
    out, tgzs = sys.argv[1], sys.argv[2:]
    work = tempfile.mkdtemp()
    merged = f"{work}/merged/kernel_cache"
    taken: set[str] = set()
    counts = {"triton": 0, "tilelang": 0, "inductor": 0, "skipped": 0}
    for tgz in tgzs:
        x = f"{work}/x"
        shutil.rmtree(x, ignore_errors=True)
        os.makedirs(x)
        with tarfile.open(tgz) as t:
            t.extractall(x, filter="data")
        kc = f"{x}/kernel_cache"
        units = []
        if os.path.isdir(f"{kc}/triton"):
            units += [("triton", f"triton/{d}") for d in os.listdir(f"{kc}/triton")]
        for ver in os.listdir(f"{kc}/tilelang") if os.path.isdir(f"{kc}/tilelang") else []:
            kdir = f"{kc}/tilelang/{ver}/kernels"
            if os.path.isdir(kdir):
                units += [("tilelang", f"tilelang/{ver}/kernels/{d}") for d in os.listdir(kdir)]
        for dp, _, files in os.walk(f"{kc}/inductor"):
            units += [("inductor", os.path.relpath(f"{dp}/{f}", kc)) for f in files]
        for kind, rel in units:
            if rel in taken:
                counts["skipped"] += 1
                continue
            src, dst = f"{kc}/{rel}", f"{merged}/{rel}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
            taken.add(rel)
            counts[kind] += 1
    with tarfile.open(out, "w") as t:
        t.add(merged, arcname="kernel_cache")
    shutil.rmtree(work)
    print(counts)


if __name__ == "__main__":
    main()
