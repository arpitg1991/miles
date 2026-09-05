# glm53 memprobe: single-node SGLang host-memory experiment

Decides, in ~75 min on ONE p6-b200.48xlarge, the open questions from the
rl-glm53f engine-node OOM (run 2, `/tmp/glm53-run2-t0.log:42541`, raylet:
"9 Workers killed due to memory pressure", node used 1934.01 / 1996.03 GB,
of which the pod's top-10 processes were only ~633 GiB):

| # | Hypothesis | What the probe measures |
|---|------------|-------------------------|
| R1 | Ray measured the WHOLE node (MemTotal-MemAvailable) because no cgroup limit is visible at `/sys/fs/cgroup/memory.max` | `info.txt` + `mem.csv:ray_root_memory_max` (literal read of the path Ray 2.58 hard-codes) in variant **-a** (no limit) and **-b** (limits.memory 1800Gi) |
| R2 | ~79 GiB per `sglang::scheduler` = the WeightChecker CPU snapshot (`check_weight_update_equal: true`), not page cache / pinned buffers | `procs.csv` RssAnon / Anonymous / USS per scheduler before vs after the `snapshot` phase |
| R3 | The other ~1300 GiB was OTHER pods on the node | `pods.csv` (per-pod `memory.current` straight from the host `kubepods.slice`, visible from the privileged container) and `mem.csv:host_used_kb - cg_ray_used_kb` |
| R4 | Engine host memory grows with served / aborted requests | slope of scheduler `Private_Dirty`/`RssAnon` and `cg_anon_kb` across `load_start..load_end..done` |
| R5 | Our own shmem / kernel / unevictable memory is large | `mem.csv` Shmem, Unevictable, Mlocked, SUnreclaim, `shm_used_kb`, `cg_shmem_kb` |

Files (all in this dir; ship to the pod via ONE ConfigMap):

* `memprobe-job.yaml` — Job `glm53-memprobe-a`, mirrors the run (NO memory limit)
* `memprobe-job-limited.yaml` — Job `glm53-memprobe-b`, identical + `limits.memory: 1800Gi`
* `memprobe.sh` — driver (sampler, SGLang with the run's exact ServerArgs, /health wait, snapshot, load, post-load)
* `memsampler.py` — 30 s sampler -> `mem.csv`, `procs.csv`, `pods.csv`
* `loadgen.py` — 96-concurrent `/generate` load, 40k-100k-token `input_ids` prompts, 10% client aborts via `/abort_request`

## Submit

```bash
export KUBECONFIG=~/.kube/config
K="kubectl --context arena-prod-bom-v2 -n arena-tasks"
cd /workplace/guparpit/miles/src/miles/examples/arena/harbor-rl-glm53-flash/memprobe

# preflight (read-only)
aws ecr describe-images --profile arena-ecr-dev --region ap-south-1 \
  --repository-name arena-slime-dev --image-ids imageTag=miles-glm53-20260902b --query 'imageDetails[0].imagePushedAt'
$K auth can-i create jobs && $K auth can-i create configmaps

# scripts -> ConfigMap (unique name; re-create if you edit the scripts)
$K create configmap glm53-memprobe-scripts \
  --from-file=memprobe.sh --from-file=memsampler.py --from-file=loadgen.py

# variant a (mirrors the run: no memory limit) -- run this FIRST
$K create -f memprobe-job.yaml
# variant b (1800Gi limit) -- run after a, or in parallel if two p6 nodes are free
$K create -f memprobe-job-limited.yaml
```

Names are unique (`glm53-memprobe-a`, `glm53-memprobe-b`, ConfigMap
`glm53-memprobe-scripts`) -- but check first: `$K get jobs,workloads | grep memprobe`.
History: `-a`/`-b` were submitted 2026-09-03 05:27 UTC (identical ConfigMap
content), deleted by ~05:33, and re-submitted as `glm53-memprobe-a2` / `-b2` at
05:34 (b2 Running on i-0c0969769a21a3001 at 05:36, a2 Pending). Their EFS dirs
are `.../glm53-memprobe/glm53-memprobe-a2/` and `-b2/`. Any further re-run must
use a fresh suffix in BOTH `metadata.name` and `env MEMPROBE_NAME` (the EFS
dir), e.g. `sed 's/glm53-memprobe-a/glm53-memprobe-a3/g' memprobe-job.yaml | $K create -f -`.
Note: the platform (kyverno) rewrites `ttlSecondsAfterFinished` to 0 and adds
EFA limits/requests + secret volumes to the pod; a finished or failed Job is
therefore deleted immediately -- the EFS artifacts below are the only record.

### Fallback if the Job never gets a kueue Workload

No GPU `batch/v1 Job` currently runs in arena-tasks (GPU work is PyTorchJob, or
ReplicaSet/StatefulSet pods carrying the `kueue.x-k8s.io/queue-name` label).
The yaml carries the queue label on BOTH the Job and the pod template so either
the Job integration or the pod integration admits it. If after ~5 min
`$K get job glm53-memprobe-a -o yaml` shows `spec.suspend: true` and
`$K get workloads | grep memprobe` is empty, submit the pod template as a plain
Pod (the proven "debug pod" pattern from the convert saga):

```bash
python3 - <<'EOF2'
import yaml
j = yaml.safe_load(open("memprobe-job.yaml"))
t = j["spec"]["template"]
pod = {"apiVersion": "v1", "kind": "Pod",
       "metadata": {"name": "glm53-memprobe-a-pod", "namespace": "arena-tasks", **t["metadata"]},
       "spec": {**t["spec"], "activeDeadlineSeconds": 14400}}
yaml.safe_dump(pod, open("/tmp/memprobe-pod-a.yaml", "w"), sort_keys=False)
EOF2
$K create -f /tmp/memprobe-pod-a.yaml
```

## Watch

```bash
$K get job glm53-memprobe-a; $K get pods -l app=glm53-memprobe -o wide
POD=$($K get pods -l job-name=glm53-memprobe-a -o jsonpath='{.items[0].metadata.name}')
$K logs -f $POD                       # driver log (phases, top procs after load / snapshot)
$K describe pod $POD | tail -20       # scheduling: kueue admission, Insufficient memory, LimitRange rejection (variant b)

# node co-tenants (R3, operator side): every 5 min while the probe runs
NODE=$($K get pod $POD -o jsonpath='{.spec.nodeName}')
while sleep 300; do date -u; $K get pods -o wide --field-selector spec.nodeName=$NODE \
  -o custom-columns='NAME:.metadata.name,PHASE:.status.phase,MEMREQ:.spec.containers[*].resources.requests.memory,MEMLIM:.spec.containers[*].resources.limits.memory'; done \
  | tee /tmp/glm53-memprobe-a-cotenants.txt
```

All artifacts land on EFS and survive pod deletion:

```
/mnt/scratch-s3files-rw/guparpit/logs/glm53-memprobe/glm53-memprobe-a/
  info.txt            static facts: /proc/self/cgroup, ls /sys/fs/cgroup, cat /sys/fs/cgroup/memory.max, versions (ray, sglang, flashinfer), env
  phases.csv          ts,phase   (boot loading loaded idle snapshot_start snapshot_done snapshot_idle load_start load_end post done exit)
  mem.csv             one row / 30 s: node meminfo, own cgroup, ray-style used, /dev/shm, GPU, other-pods total
  procs.csv           top-12 processes / 30 s: USS, VmRSS, RssAnon, RssFile, RssShmem, smaps_rollup fields
  pods.csv            per-pod cgroup memory.current/anon/file/shmem from kubepods.slice (empty if the host tree is not visible)
  sglang-server.log   full server log (Load weight end, cuda graph capture, any OOM)
  loadgen.log/.csv    throughput per minute; one row per request (status ok/aborted/error, latency)
  memprobe.log        driver log
```

Read from the workstation via the cluster only (EFS is not desktop-mounted):
`$K exec $POD -- tail -5 /mnt/scratch-s3files-rw/guparpit/logs/glm53-memprobe/glm53-memprobe-a/mem.csv`
while the pod lives, or from any other arena-tasks pod afterwards (the mount is
platform-injected into every pod).

## Reading the CSVs: which pattern decides what

Units: `*_kb` columns are KiB (1 GiB = 1048576 KiB). Ray's Top-10 "MEM(GB)"
is USS (= `Private_Clean + Private_Dirty`, `procs.csv:USS_kb`) in GiB.

**R1 (Ray accounting scope)**
* `info.txt` "what Ray 2.58 reads: /sys/fs/cgroup/memory.max"
  * `max` -> private cgroupns, no limit: Ray uses host MemTotal-MemAvailable (the run's behaviour).
  * `No such file or directory` -> the privileged container sees the HOST cgroup root (containerd skips cgroupns for privileged containers). Ray ALSO falls back to host accounting, AND a kubelet memory limit (variant -b) is invisible to Ray: `memory.max` only exists at `mem.csv:cg_dir` (the resolved `kubepods.slice/.../cri-containerd-*.scope`). Then `limits.memory` alone does not fix R1; `RAY_memory_monitor_refresh_ms=0` is required.
  * a number (variant -b) -> the limit is visible where Ray looks; Ray would report total = that number and used = `cg_ray_used_kb`.
* Compare variant -b `mem.csv:cg_max` (own cgroup; expect 1932735283200) with `ray_root_memory_max`.

**R2 (79 GiB/rank = WeightChecker snapshot)**
* `procs.csv` rows with `name` = `sglang::scheduler_TP*` in phase `idle` vs `snapshot_idle`:
  * `RssAnon_kb` / `Anonymous_kb` / `USS_kb` jump by ~73-75 GiB per rank (8 ranks, ~590 GiB total) exactly at `snapshot_done`, `RssFile_kb` unchanged -> CONFIRMED: it is an anonymous CPU weight copy, not page cache. Expect the pre-snapshot scheduler USS to be ~4-6 GiB.
  * `RssFile_kb` large (tens of GiB) and `RssAnon_kb` small before the snapshot -> resident mmapped safetensors (page cache attributed to the process) instead.
  * No jump -> the snapshot is not resident (would contradict weight_checker.py:114-121); check `sglang-server.log` for the `[WeightChecker] handle action=snapshot` line.
* Set `MEMPROBE_SNAPSHOT=0` in a re-run to see the steady-state footprint without the snapshot (what the fixed relaunch will have).

**R3 (co-tenant pods)**
* `mem.csv`: `host_used_kb` rising while `cg_ray_used_kb` (own cgroup minus file cache) is flat -> memory outside our pod (co-tenants, host daemons, kernel).
* `pods.csv`: rows with `is_self=0` — their `current_kb` sum (`mem.csv:other_pods_current_kb`) IS the co-tenant number, per pod slice, no Grafana needed. Cross-check with the `kubectl get pods --field-selector spec.nodeName=` loop. Empty `pods.csv` means the host cgroup tree is not visible; fall back to the kubectl loop + Grafana.
* Expected identity per row: `host_used ~= own(anon+shmem+kernel) + other_pods_current + kernel/driver residue`. A residue of hundreds of GiB that is neither ours nor other pods' points at host-level consumers (mount-s3 CSI page cache, driver pinned pages) — see `Unevictable_kb`, `Mlocked_kb`, `SUnreclaim_kb`.

**R4 (leak under load)**
* `procs.csv` scheduler `Private_Dirty_kb`/`RssAnon_kb` and `mem.csv:cg_anon_kb` from `load_start` to `done`: slope < 1 GiB/h/rank and flat after `load_end` -> REFUTED (no material engine-side leak). A monotone rise that tracks `loadgen.log` gen_tokens/aborts -> leak; correlate with `loadgen.csv` aborted count.
* `mem.csv:cg_file_kb` growing is the page cache of the mmapped checkpoint / logs and is reclaimable — not a leak.

**R5 (own shmem/kernel)**
* `mem.csv`: `Shmem_kb + Unevictable_kb + SUnreclaim_kb` < ~50 GiB and `shm_used_kb` ~0 -> REFUTED. `cg_shmem_kb` large -> something in our pod writes /dev/shm (in the real run the Ray object store lives there; here there is no Ray).

**Variant -b specifics (1800Gi limit)**
* Admission: `$K describe pod` shows a LimitRange/quota rejection -> record it; the relaunch then relies on `RAY_memory_monitor_refresh_ms=0` alone.
* If the kernel OOM-kills a scheduler at the limit (`sglang-server.log` / `dmesg` not readable, but the driver logs "sglang died"), the limit is too tight for snapshot + load; the fixed relaunch removes the snapshot.

## STOP condition

* Stop early (`$K delete job glm53-memprobe-a` — you may delete your own Job) when BOTH of these are true after the `snapshot_idle` phase: scheduler `USS_kb` before/after snapshot is recorded (R2 decided) AND `ray_root_memory_max` + `cg_max` are recorded for the variant (R1 decided). The load phase only adds R4/R5 evidence and co-tenant history; 20 min of it (`MEMPROBE_LOAD_MIN=20`) is enough for a first slope.
* Stop immediately if `mem.csv:host_used_kb` exceeds 1.85e9 KiB (~1765 GiB, 88% of MemTotal): the probe itself must not push a shared node into the kernel OOM killer. `memprobe.sh` does not enforce this automatically (no cluster writes from inside the pod); watch `$K logs -f`.
* The Job self-terminates at `activeDeadlineSeconds` 4 h; a normal run ends in ~75 min (load 6-7 min, idle 3, snapshot + 4, load-test 40, post 20).

## Knobs (pod env)

`MEMPROBE_IDLE_S` (180), `MEMPROBE_SNAPSHOT` (1), `MEMPROBE_LOAD_MIN` (40),
`MEMPROBE_POST_S` (1200), `MEMPROBE_SAMPLE_S` (30), `MEMPROBE_CONCURRENCY` (96),
`MEMPROBE_PROMPT_MIN`/`MAX` (40000/100000 -- lower to 20000/40000 for more
request churn per minute; 96 x 70k tokens is prefill-bound like the run),
`MEMPROBE_MAX_NEW_TOKENS` (2048), `MEMPROBE_ABORT_FRAC` (0.10).

## Fidelity notes / limitations

* SGLang is launched with `python3 -m sglang.launch_server` and the run's ServerArgs
  (`/tmp/glm53-facts/sglang_server_args.txt`, log line 2120) translated 1:1 to
  CLI flags; miles launches the same `launch_server()` in a spawned process
  (miles/backends/sglang_utils/sglang_engine.py:73-77). No Ray here: the raylet,
  object store and `_HttpPosterActor`s are absent, so `host_used` in the probe is
  a lower bound for the real engine node; the sglang-side numbers are like-for-like.
* Prompts are random token ids, not chat text; logprobs are returned like the run
  (`return_logprob`), decoding runs to `max_new_tokens` (`ignore_eos`).
* `/dev/shm` is unbounded in both variants (as in the run); the relaunch patch adds `sizeLimit: 256Gi`.
* Node identity: `NODE_NAME` is in `info.txt`/`memprobe.log`; use it for the co-tenant kubectl loop.
