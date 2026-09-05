#!/usr/bin/env python3
"""Host/cgroup/process memory sampler for the glm53 memprobe (stdlib only).

Every --interval seconds writes one row to <out>/mem.csv (node + own-cgroup view),
up to --top rows to <out>/procs.csv (per-process anon/file split from
/proc/<pid>/smaps_rollup and /proc/<pid>/status) and, when the host cgroup tree
is visible (privileged container without a cgroup namespace), one row per pod
slice to <out>/pods.csv (co-tenant attribution straight from kubepods.slice).

Why these columns (see README "Reading the CSVs"):
  * host_used_kb = MemTotal - MemAvailable is exactly what Ray 2.58 counted
    against the 95% threshold when no cgroup limit is visible
    (ray memory_monitor_utils.cc:261-335).
  * cg_ray_used_kb = memory.current - inactive_file - active_file is the value
    Ray would use if it saw a cgroup limit (memory_monitor_utils.cc:144-197).
  * ray_root_memory_max is the literal read of /sys/fs/cgroup/memory.max, the
    hard-coded path Ray reads; ENOENT/max => Ray falls back to host accounting.
  * procs.csv Private_Clean+Private_Dirty is Ray's per-process USS (the number in
    the Top-10 table, log 47868-47878); Anonymous vs Rss-Anonymous separates a CPU
    weight copy (anon) from resident mmapped safetensors (file-backed).
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

MEMINFO_FIELDS = [
    "MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "Shmem", "AnonPages",
    "Mapped", "Unevictable", "Mlocked", "Slab", "SReclaimable", "SUnreclaim",
    "KernelStack", "PageTables", "HugePages_Total",
]
CG_STAT_FIELDS = ["anon", "file", "file_mapped", "shmem", "unevictable", "active_file", "inactive_file", "kernel", "slab"]
SMAPS_FIELDS = [
    "Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Shared_Clean", "Shared_Dirty",
    "Private_Clean", "Private_Dirty", "Anonymous", "LazyFree", "AnonHugePages",
    "Shared_Hugetlb", "Private_Hugetlb", "Swap", "Locked",
]
STATUS_FIELDS = ["VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap"]


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_text(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError as e:
        return None


def read_kv_kb(path, fields):
    """Parse 'Key:   123 kB' files (/proc/meminfo, status, smaps_rollup)."""
    out = {k: "" for k in fields}
    txt = read_text(path)
    if txt is None:
        return out
    for line in txt.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        if k in out:
            out[k] = v.strip().split()[0] if v.strip() else ""
    return out


def own_cgroup_dir():
    """Resolve this process's cgroup directory as seen from /sys/fs/cgroup.

    cgroup v2 + private cgroupns  -> /proc/self/cgroup says '0::/'            -> /sys/fs/cgroup
    cgroup v2 + host cgroupns     -> '0::/kubepods.slice/.../cri-containerd-X.scope' -> /sys/fs/cgroup/<that>
    cgroup v1                     -> lines like '4:memory:/kubepods/...'       -> /sys/fs/cgroup/memory/<that>
    """
    txt = read_text("/proc/self/cgroup") or ""
    v2 = None
    v1_mem = None
    for line in txt.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hid, ctrls, path = parts
        if hid == "0" and ctrls == "":
            v2 = path
        elif "memory" in ctrls.split(","):
            v1_mem = path
    if v2 is not None:
        d = os.path.normpath("/sys/fs/cgroup/" + v2.lstrip("/"))
        if os.path.exists(os.path.join(d, "memory.current")):
            return "v2", d, v2
        if os.path.exists("/sys/fs/cgroup/memory.current"):
            return "v2", "/sys/fs/cgroup", v2
    if v1_mem is not None:
        d = os.path.normpath("/sys/fs/cgroup/memory/" + v1_mem.lstrip("/"))
        if os.path.exists(os.path.join(d, "memory.usage_in_bytes")):
            return "v1", d, v1_mem
        if os.path.exists("/sys/fs/cgroup/memory/memory.usage_in_bytes"):
            return "v1", "/sys/fs/cgroup/memory", v1_mem
    return "none", "", v2 or v1_mem or ""


def read_cgroup(mode, d):
    out = {"cg_max": "", "cg_current_kb": "", "cg_ray_used_kb": ""}
    for k in CG_STAT_FIELDS:
        out["cg_" + k + "_kb"] = ""
    if mode == "v2":
        mx = (read_text(os.path.join(d, "memory.max")) or "").strip()
        cur = (read_text(os.path.join(d, "memory.current")) or "").strip()
        stat = read_text(os.path.join(d, "memory.stat")) or ""
        st = {}
        for line in stat.splitlines():
            p = line.split()
            if len(p) == 2:
                st[p[0]] = p[1]
        out["cg_max"] = mx
        if cur.isdigit():
            out["cg_current_kb"] = str(int(cur) // 1024)
            act = int(st.get("active_file", 0)); inact = int(st.get("inactive_file", 0))
            out["cg_ray_used_kb"] = str((int(cur) - act - inact) // 1024)
        for k in CG_STAT_FIELDS:
            if k in st:
                out["cg_" + k + "_kb"] = str(int(st[k]) // 1024)
    elif mode == "v1":
        mx = (read_text(os.path.join(d, "memory.limit_in_bytes")) or "").strip()
        cur = (read_text(os.path.join(d, "memory.usage_in_bytes")) or "").strip()
        stat = read_text(os.path.join(d, "memory.stat")) or ""
        st = {}
        for line in stat.splitlines():
            p = line.split()
            if len(p) == 2:
                st[p[0]] = p[1]
        out["cg_max"] = mx
        if cur.isdigit():
            out["cg_current_kb"] = str(int(cur) // 1024)
            act = int(st.get("total_active_file", 0)); inact = int(st.get("total_inactive_file", 0))
            out["cg_ray_used_kb"] = str((int(cur) - act - inact) // 1024)
        v1map = {"anon": "total_rss", "file": "total_cache", "file_mapped": "total_mapped_file",
                 "shmem": "total_shmem", "unevictable": "total_unevictable",
                 "active_file": "total_active_file", "inactive_file": "total_inactive_file"}
        for k, v1k in v1map.items():
            if v1k in st:
                out["cg_" + k + "_kb"] = str(int(st[v1k]) // 1024)
    return out


def ray_root_memory_max():
    """What Ray 2.58's monitor literally reads (kDefaultCgroupPath=/sys/fs/cgroup)."""
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(p) as f:
                return p + "=" + f.read().strip()
        except FileNotFoundError:
            continue
        except OSError as e:
            return p + "=ERR:" + e.strerror
    return "ENOENT"


def shm_usage():
    try:
        st = os.statvfs("/dev/shm")
        size = st.f_blocks * st.f_frsize
        used = (st.f_blocks - st.f_bfree) * st.f_frsize
        return str(size // 1024), str(used // 1024)
    except OSError:
        return "", ""


def gpu_mem():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        vals = [int(x.strip()) for x in r.stdout.split() if x.strip().isdigit()]
        return str(sum(vals)), "|".join(str(v) for v in vals)
    except Exception:
        return "", ""


def list_procs(top):
    rows = []
    for pid_dir in glob.glob("/proc/[0-9]*"):
        pid = pid_dir[6:]
        st = read_kv_kb(pid_dir + "/status", STATUS_FIELDS + ["Name"])
        rss = st.get("VmRSS", "")
        if not rss or not rss.isdigit():
            continue
        rows.append((int(rss), pid, st))
    rows.sort(reverse=True)
    out = []
    for rss, pid, st in rows[:top]:
        cmd = (read_text("/proc/%s/cmdline" % pid) or "").replace("\0", " ").strip()[:120]
        sm = read_kv_kb("/proc/%s/smaps_rollup" % pid, SMAPS_FIELDS)
        rec = {"pid": pid, "name": st.get("Name", ""), "cmdline": cmd}
        for k in STATUS_FIELDS:
            rec[k + "_kb"] = st.get(k, "")
        for k in SMAPS_FIELDS:
            rec[k + "_kb"] = sm.get(k, "")
        try:
            uss = int(sm.get("Private_Clean") or 0) + int(sm.get("Private_Dirty") or 0)
            rec["USS_kb"] = str(uss)
        except ValueError:
            rec["USS_kb"] = ""
        out.append(rec)
    return out


def list_pod_slices(own_path):
    """Per-pod memory from the host cgroup tree, if visible. own_path is our v2 path."""
    res = []
    root = "/sys/fs/cgroup/kubepods.slice"
    if not os.path.isdir(root):
        return res
    pats = [root + "/kubepods-pod*.slice", root + "/kubepods-*.slice/kubepods-*-pod*.slice"]
    own_pod = ""
    for seg in own_path.split("/"):
        if seg.startswith("kubepods-") and "-pod" in seg:
            own_pod = seg
    for pat in pats:
        for d in glob.glob(pat):
            cur = (read_text(os.path.join(d, "memory.current")) or "").strip()
            if not cur.isdigit():
                continue
            st = {}
            for line in (read_text(os.path.join(d, "memory.stat")) or "").splitlines():
                p = line.split()
                if len(p) == 2:
                    st[p[0]] = p[1]
            slice_name = os.path.basename(d)
            res.append({
                "slice": slice_name,
                "is_self": "1" if slice_name == own_pod else "0",
                "current_kb": str(int(cur) // 1024),
                "anon_kb": str(int(st.get("anon", 0)) // 1024),
                "file_kb": str(int(st.get("file", 0)) // 1024),
                "shmem_kb": str(int(st.get("shmem", 0)) // 1024),
                "max": (read_text(os.path.join(d, "memory.max")) or "").strip(),
            })
    res.sort(key=lambda r: -int(r["current_kb"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mode, cgdir, cgpath = own_cgroup_dir()
    with open(os.path.join(a.out, "sampler-info.txt"), "a") as f:
        f.write("%s sampler start pid=%d cgroup_mode=%s own_cgroup_dir=%s /proc/self/cgroup=%s ray_root=%s\n"
                % (now(), os.getpid(), mode, cgdir, cgpath.replace("\n", " "), ray_root_memory_max()))

    mem_cols = (["ts", "phase", "host_used_kb"] + [k + "_kb" for k in MEMINFO_FIELDS if k != "HugePages_Total"]
                + ["HugePages_Total", "cg_mode", "cg_dir", "cg_max", "cg_current_kb", "cg_ray_used_kb"]
                + ["cg_" + k + "_kb" for k in CG_STAT_FIELDS]
                + ["ray_root_memory_max", "shm_size_kb", "shm_used_kb", "gpu_used_mib_sum", "gpu_used_mib_each",
                   "kubepods_visible", "other_pods_current_kb", "n_other_pods"])
    proc_cols = (["ts", "phase", "pid", "name", "USS_kb"] + [k + "_kb" for k in STATUS_FIELDS]
                 + [k + "_kb" for k in SMAPS_FIELDS] + ["cmdline"])
    pod_cols = ["ts", "phase", "slice", "is_self", "current_kb", "anon_kb", "file_kb", "shmem_kb", "max"]

    def opener(name, cols):
        path = os.path.join(a.out, name)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        f = open(path, "a", newline="")
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        return f, w

    fm, wm = opener("mem.csv", mem_cols)
    fp, wp = opener("procs.csv", proc_cols)
    fpod, wpod = opener("pods.csv", pod_cols)
    phase_file = os.path.join(a.out, "phase.current")

    while True:
        ts = now()
        phase = (read_text(phase_file) or "").strip()
        mi = read_kv_kb("/proc/meminfo", MEMINFO_FIELDS)
        row = {"ts": ts, "phase": phase}
        for k in MEMINFO_FIELDS:
            row[(k + "_kb") if k != "HugePages_Total" else k] = mi.get(k, "")
        try:
            row["host_used_kb"] = str(int(mi["MemTotal"]) - int(mi["MemAvailable"]))
        except (KeyError, ValueError):
            row["host_used_kb"] = ""
        row["cg_mode"] = mode
        row["cg_dir"] = cgdir
        row.update(read_cgroup(mode, cgdir) if mode != "none" else {})
        row["ray_root_memory_max"] = ray_root_memory_max()
        row["shm_size_kb"], row["shm_used_kb"] = shm_usage()
        row["gpu_used_mib_sum"], row["gpu_used_mib_each"] = gpu_mem()
        pods = list_pod_slices(cgpath)
        row["kubepods_visible"] = "1" if pods else "0"
        others = [int(p["current_kb"]) for p in pods if p["is_self"] == "0"]
        row["other_pods_current_kb"] = str(sum(others)) if pods else ""
        row["n_other_pods"] = str(len(others)) if pods else ""
        wm.writerow(row); fm.flush()
        for rec in list_procs(a.top):
            rec.update({"ts": ts, "phase": phase})
            wp.writerow(rec)
        fp.flush()
        for p in pods:
            p.update({"ts": ts, "phase": phase})
            wpod.writerow(p)
        fpod.flush()
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
