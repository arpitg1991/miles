# kdatp-prof: train-only arms and profiled steps on the r47 layout

One 8-node train-only job that runs one or more arms in one Ray cluster. Each
arm starts from the `kdatp` T2 arm of the in-image `gen_arm_configs.py` (the
r47 trainer layout: `glm5_next_kda_tp`, full uniform recompute, TP8 SP, PP4
11/11/11/12, EP16, `micro_batch_size` 1, `max_tokens_per_gpu` 8192, r47 flip
A `skip_actor_forward_only`). An arm sets its data, the torch profiler on or
off (`miles/utils/profile_utils.py`, target `train_overall`), and any config
override. An optional pre-step measures the all-to-all bandwidth of the EP
groups. All arms train from the r43 `iter_0000039` seed and never save.

Data: whole groups of the r45 sample summary of rollout 43, with the token
data of the r43 staged token files (`../build_rollout_data.py`). These jobs
are train-only timing tests on replayed T2 rows. The agentic-debt training
dataset of record is
`lakefs://arena-inspect/main/internal/agentic-debt-r3/agentic-debt-766/`. A
launch record MUST state both facts.

| File | Use |
| --- | --- |
| `kdatp-prof-job.yaml` | PyTorchJob `kdatp-prof-<stamp>`: `t2-job.yaml` plus the driver on the mount, `KDATP_KCACHE`, a 5 h deadline and a 200 GiB ephemeral request. Placeholders `__IMAGE__`, `__STAMP__` only. |
| `kdatp-prof-run.sh` | Pod driver, the t2 branch of `../kdatp-run.sh`. It sources `job.env`, writes the Kineto config, runs the optional a2a bench, then runs the arms in order. |
| `patch_prof_arm.py` | Reads the arms file, checks it, and writes `arms/<name>.yaml` and `arms/plan.tsv` from the in-image `kdatp` arm. |
| `a2a_bench.py` | All-to-all bandwidth of the EP16 and EP8 layouts (`PROF_A2A_BENCH=1`). |
| `job.env` | Per-job settings. You write it for each job. |
| `<arms file>` | The arms of the job (`PROF_ARMS_FILE`). You write it for each job. |

The pods read these files from `kdatp/prof/<stamp>/harness/` on the scratch
mount. The pods use the in-image `gen_arm_configs.py`,
`build_rollout_data.py` and `parse_logs.py` (miles `4716a367a`). New logic
MUST live in this directory.

## job.env

`KEY=VALUE` lines in bash syntax; `#` starts a comment. Put quotes around a
value with spaces. Every pod sources the file with `set -a` before the
defaults of the driver, so a `job.env` value wins over the manifest env, and
`ray start` passes it to the Ray workers. The file MUST exist: the driver
waits 10 minutes for it and then stops, so that a late mount import never
runs a job on the defaults. An empty file keeps all defaults.

| Key | Default | Use |
| --- | --- | --- |
| `PROF_ARMS_FILE` | empty | File name of the arms file in `harness/`. Empty: one arm, `kdatp-prof`, profiled, `PROF_GROUPS`, DP 2 (the arm of runs `20260929a` and `20260929ab`). |
| `PROF_A2A_BENCH` | `0` | `1` runs `a2a_bench.py` on all pods before Ray starts. |
| `PROF_GROUPS` | `239,258` | Default `data.groups`. |
| `ARM_TIMEOUT` | `12600` | Default arm `timeout` (s). |
| `PROF_SKIP_ACTOR_FORWARD_ONLY` | `1` | Default arm `skip_actor_forward_only`. |
| `KDATP_KCACHE` | manifest value | Space-separated kernel cache tarballs. |
| `PROF_KINETO_BUFFER_MB` | `4096` | Kineto GPU buffer cap per rank. |
| `KINETO_LOG_LEVEL` | `1` | Kineto INFO lines in the trainer log. |

## Arms file

A YAML list. The arms run in this order. A failed arm (data, seed, out of
memory, `rc` other than 0, timeout) does not stop the next arm.

| Key | Default | Use |
| --- | --- | --- |
| `name` | required | Lowercase letters, digits and `-`. It names `<run>/<name>/`, `arms/<name>.yaml` and the checkpoint dir `kdatp-prof-<stamp>-<name>`. |
| `profile` | required | `true` adds the profiler keys: step 0 warms up, step 1 is recorded, step 2 runs clean, `debug_exit_after_rollout` 3. Traces go to `<run>/<name>/tb/`. |
| `skip_actor_forward_only` | `PROF_SKIP_ACTOR_FORWARD_ONLY` | r47 flip A. |
| `set` | `{}` | Config overrides, applied last, so they win over every other key. |
| `data.groups` | `PROF_GROUPS` | Comma-separated `group_index` values of the r45 summary. Put quotes around the value. |
| `data.dp_size` | `2` | The actor DP of the arm. The DP pad of the data depends on it. The check stops an arm whose DP (`8 x num_trainers / (TP x PP x CP)`) differs. |
| `data.replay_rollout` | none | A rollout id (40 to 43). Every step loads `rollout_<id>.pt`, so the arm can run any number of steps. |
| `data.dir` | `prof-<groups>` for DP 2, `prof-<groups>-dp<N>` otherwise | The data dir under `kdatp/data/`. Use `t2` for the T2 groups at DP 2: that dir holds the rows of the 778.6 s baseline (run `20260929a`). |
| `timeout` | `ARM_TIMEOUT` | Seconds for the launcher. |

Replay. `load_debug_rollout_data` without `{rollout_id}` loads the same file
at every step: the in-image `miles/ray/rollout/debug_data.py` calls
`args.load_debug_rollout_data.format(rollout_id=...)`, and `str.format`
leaves a path without the placeholder unchanged. The harness uses that path
and no link dir. Note: `build_rollout_data.py` already writes one file,
`rollout_40.pt`, and links the other ids to it (`data/t2/manifest.json`:
`links` 41 to 44). So every step of every arm trains the same rows. Replay
removes the step limit only. Without replay, the data dir holds rollouts 40
to 43, and the train loop prefetches the next id
(`train_async_arena.py`), so the arm can run at most 3 steps. The check
stops an arm with more steps and no replay.

Data dirs. The driver builds a missing dir once (CPU and disk only: 112 s
for `prof-239-258` in run `20260929ab`; the 8 T2 groups made 35.8 GB in
`data/t2`). Two jobs can need the same dir at the same time. The first job
builds it under the lock dir `data/<dir>.lock`; the other job waits up to
20 minutes for the manifest. If a job dies during a build, remove the stale
lock dir before the next job. The driver checks that the manifest groups
equal the arm groups and that the row count is a multiple of `dp_size`.

Check an arms file on the desktop before the upload:

```bash
W=/workplace/guparpit/kdfast/scratch/prof/<stamp>/harness-check   # never /tmp
PY=/workplace/guparpit/arena/src/AREnATasks/.venv/bin/python
mkdir -p $W/img/kdatp $W/img/r45
git show 4716a367a:examples/arena/harbor-rl-glm53-flash/kdatp/gen_arm_configs.py > $W/img/kdatp/gen_arm_configs.py
git show 4716a367a:examples/arena/harbor-rl-glm53-flash/r45/miles-config.yaml > $W/img/r45/miles-config.yaml
$PY $W/img/kdatp/gen_arm_configs.py t2 --data-dir /mnt/scratch-s3files-rw/guparpit/kdatp/data --out $W/run/arms --arms kdatp
$PY patch_prof_arm.py $W/run/arms/kdatp.yaml $W/run --data-root /mnt/scratch-s3files-rw/guparpit/kdatp/data \
  --arms-file arms.yaml --skip-actor-forward-only   # prints the key diff of each arm and the timeout sum
```

Time budget. The job deadline is 18,000 s. Wall time per arm in the earlier
runs: T2 rows, 3 steps: 3,508 s (run `20260929a`: `actor_train` 1,049, 800
and 779 s, and 364 s of data wait at the first step); slice 239,258, 3
steps: 1,212 s (run `20260929ab`). Add 60 s per arm for the Ray job stop,
the a2a bench (at most 450 s), and the data builds. The kernel cache
tarball holds TP8 kernels. An arm that changes the KDA or DSA shapes (for
example TP4) compiles its kernels cold at low GPU power. The idle-GPU reaper
deletes a workload whose 60-minute mean GPU power is below 10%
(`kdatp-t2-20260927b` was lost that way). Put such an arm after a warm arm.
The driver does not save `/tmp/kernel_cache` at the arm end, so the kernels
of a new shape are lost with the pods: the TP4 arm of `20260929e` compiled
768 TileLang kernels in step 0 (+1,599 s) and kept none.

## Kineto buffer cap fix (tested live in job 20260929c)

Runs `20260929a` and `20260929ab` wrote stub traces. Kineto keeps at most
`1 + cap / 4 MiB` CUPTI buffers per rank (default cap 128 MiB, 33 buffers).
In run `20260929ab`, all 64 ranks logged `Exceeded max GPU buffer count (33
>= 33)` at 09:23:37, in the gradient sync of the warmup step 0. torch 2.13
then read the Kineto stop flag, entered `DEVICE_STOPPED`, and wrote stubs.
A smaller slice does not avoid the cut; details in the study record
`training-runs/studies/trainer-core-profile-glm53-flash/STUDY.md`.

The fix: before the `REPLICA_IDX` branch, every pod writes
`/tmp/kineto.conf` with `ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB=<PROF_KINETO_BUFFER_MB>`
and exports `KINETO_CONFIG=/tmp/kineto.conf`. `/tmp` in the pod is the
overlay disk. The Ray workers inherit the env of `ray start`, the same path
that took `KINETO_LOG_LEVEL=1` to the trainer ranks in run `20260929ab`.

Live result, job `kdatp-prof-20260929c` (groups 239,258, profiled step 1,
harness `dabe8a8972`). Source: `kdatp/prof/20260929c/kdatp-prof/trainer-0.log`
and `results.json`.

| Signal | Value |
| --- | --- |
| `Max GPU buffer size: 4096MB` | 64 ranks (0 at `128MB`) |
| `Exceeded max GPU buffer count` | 0 |
| torch `activity collection was stopped early` | 0 |
| `Processed N GPU records` | 7.36M to 8.93M per rank |
| Traces in `tb/` | 64 files, 393.5 to 460.0 MB each (gz), 27.21 GB in total. The 8 local copies (ranks 0, 8, ..., 56) hold 5.7 to 6.7 GB each after `gunzip` |
| Trace export after the step | 338 s on rank 0, 405 s on the slowest rank (`train_time` 594.4 s minus `actor_train` 189.4 s) |
| Host memory while tracing | about 30 GiB more RSS per trainer rank (node 0: 113.7 to 145.0 GiB). The pod limit is 1,800 Gi |
| Profiler cost in the step | `actor_train` 189.4 s (traced) against 186.3 s (clean step 2): at most +3.0 s (1.6%). Run `20260929ab` shows +3.9 s with no trace, so the true cost is near 0 (UNVERIFIED) |

Keep a profiled arm on the 2-group slice. A traced full T2 step (156
micro-batches) holds about 4.7x the records and can pass the pod memory
limit (UNVERIFIED).

Evidence. The pod torch is `2.13.0+cu130`, `torch.version.git_version`
`cf30153c4c131c8164ee7798e5022d810682e2cb`, which is the `v2.13.0` tag
commit. That tag pins `third_party/kineto` at
`094d3c1d072362d0a919a77299459eee94f97931`. In that Kineto source:

- `libkineto/src/ActivityProfilerProxy.cpp` `prepareTrace` (the torch
  `torch.profiler` path: `kineto_shim.cpp` `prepareTrace` calls it with the
  trace-id string) parses the `configStr` from torch, then
  `configLoader_.getConfString()`, then calls `setClientDefaults()` and
  `controller_->prepareTrace(config)`. So the file is read at each
  `prepareTrace`, that is, at each profiler start (`on_init_end` in miles),
  and its keys win over the torch string.
- `libkineto/src/ConfigLoader.cpp`: `getConfString()` =
  `readConfigFromConfigFile(configFileName(), false)`. `configFileName()`
  reads `getenv("KINETO_CONFIG")` once per process (a function-local
  static), else `/etc/libkineto.conf`. A missing file gives an empty string
  and no log line. The env MUST be set before the trainer process starts.
- `libkineto/src/AbstractConfig.cpp` `parse`: one `KEY=VALUE` per line, `#`
  comments, blank lines skipped. An unknown key only logs `Unrecognized
  config line`. A line with more or less than one `=` logs `Invalid config
  line` and stops the parse.
- `libkineto/src/Config.cpp`: `ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB` sets
  `activitiesMaxGpuBufferSize_ = toInt32(val) * 1024 * 1024` (default
  `128 * 1024 * 1024`). `setClientDefaults()` and `validate()` do not touch
  it.
- `libkineto/src/GenericActivityProfiler.cpp` `configure`: prints the config
  (`Max GPU buffer size: <n>MB`) when INFO logs are on, logs `Enabling GPU
  tracing with max buffer size <n>MB)`, and calls
  `setMaxGpuBufferSize(config_->activitiesMaxGpuBufferSize())` for every GPU
  trace. No other key is necessary. `CuptiActivityApi::setMaxBufferSize`:
  `maxGpuBufferCount_ = 1 + size / kBufSize`, `kBufSize` 4 MiB. 4096 MB
  gives 1,025 buffers. Kineto allocates the buffers on demand.
- The `_ExperimentalConfig(custom_profiler_config=...)` route does not
  work: `kineto_shim.cpp` wraps the string as `CUSTOM_CONFIG=<string>`.

Binary check, read-only, in the r47 trainer pod
(`/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so`,
`grep -a -o -F`): `KINETO_CONFIG` 1, `/etc/libkineto.conf` 1,
`ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB` 1, `Enabling GPU tracing with max buffer
size` 1, `Max GPU buffer size: ` 1, `Exceeded max GPU buffer count (` 1,
`Unrecognized config line: ` 1, and the symbol
`_ZN9libkineto12ConfigLoader13getConfStringB5cxx11Ev` 2. The pod has no
`/etc/libkineto.conf` and no `KINETO_CONFIG`.

Success signal of a profiled arm, in `driver.log`: `arm <name> kineto: 64
ranks at 4096MB, 0 Exceeded, 0 stopped early`, then 64 trace files far
above the 13 to 33 KB of the stubs. The `stopped early` count is the torch
warning only. The Kineto summary line `GPU stopped early? = 0` is on every
rank, and it read 0 also in the cut run `20260929ab`, so the driver does
not count it. The driver of `dabe8a8972` counted it and logged `64 stopped
early` for the clean job `20260929c`. Early check, about 10 minutes after
`arm <name>: start`: `<name>/trainer-0.log` on S3 holds 64 `Max GPU buffer
size: 4096MB` lines. If it shows `128MB`, the file was not read: delete the
job and use the fallback.

Fallback (a separate job): `set: {profile_step_start: 0, profile_step_end:
1}`. Step 0 is then recorded from `on_init_end`, and the natural
record-to-none transition saves the trace up to the cut at the stock cap: the
cold microbatch loop of step 0 without the last gradient sync and optimizer
step. This comes from a reading of the torch 2.13 `profiler.py` state
machine; it is not tested live. Job `20260929c2` did not run, because the
fix worked.

## a2a bench

`PROF_A2A_BENCH=1` runs `a2a_bench.py` on all 8 pods before the
`REPLICA_IDX` branch, with `python3 -m torch.distributed.run --nnodes 8
--nproc-per-node 8 --node-rank $REPLICA_IDX --rdzv-backend static
--rdzv-endpoint <job>-worker-0:23456`. NCCL uses the pod env as it is.

- Head address: the PyTorchJob has only `Worker` replicas and an
  `elasticPolicy`, so the operator sets `PET_RDZV_ENDPOINT=<job>-worker-0:23456`,
  `PET_NNODES`, `PET_NPROC_PER_NODE`, and no `MASTER_ADDR`, `RANK` or
  `WORLD_SIZE` (the r47 trainer pod env). The yaml sets `REPLICA_IDX`. The
  driver takes the host `<job>-worker-0` from the pod hostname, as
  `run_arena_harbor.py` does for the Ray head.
- Port 23456 is the rendezvous port of the operator. Ray uses 6379, 8265
  and ports that it picks at `ray start`, after the bench.
- Static rendezvous gives node rank = `REPLICA_IDX`, so global ranks
  16k..16k+15 are on nodes 2k and 2k+1, as the r47 stage expert groups.
- Bounds: `timeout -k 30 420` on every pod, `--rdzv-conf timeout=240`, and a
  180 s NCCL group timeout in the bench. The Ray workers wait at most 600 s
  for the Ray head (`_wait_for_head_port`, 120 x 5 s) and then fail the job,
  so the bench MUST end well before 600 s on the head. A failed or hung bench
  loses only `a2a/`.
- Layouts, one after the other, all groups at the same time: `ep16` (4
  groups of 16 ranks on 2 nodes, the r47 stage groups) and `ep8` (8 groups,
  one node each). bf16 `all_to_all_single`, equal splits, per-rank send sizes
  32, 64, 128, 256 MiB, 536,870,912 B (8192 tokens x top-8 x 4096 x 2 B, the
  r47 dispatch estimate) and 1 GiB; 3 warmup and 10 timed calls.
- Output: `a2a/a2a.json` and `a2a/a2a.md` from rank 0 (median time, per-rank
  algbw, cross-node GB/s per rank for `ep16`, range of the per-group
  medians), and `a2a/node-<n>.log` per pod. `driver.log` records `a2a_rc`.

## Jobs of round 20260929c

All three jobs ran to `rc 0`: `20260929c` (profile and a2a bench),
`20260929d` (arms `ep8`, `base`, `pack`) and `20260929e` (arms `ep8pack`,
`tp4ep8pack`). The results are in the study record
`training-runs/studies/trainer-core-profile-glm53-flash/STUDY.md`. The arm
lists of `20260929d` and `20260929e` come from the feasibility study. The
lists below show the format.

`20260929c`, the profile job: one profiled arm on groups 239,258 (the default
arm) plus the a2a bench. `job.env`:

```bash
# kdatp-prof-20260929c: one profiled arm on groups 239,258, plus the a2a bench.
PROF_A2A_BENCH=1
```

`20260929c2`, only if the Kineto fix fails in `20260929c`. `job.env`:
`PROF_ARMS_FILE=arms.yaml`. `arms.yaml`:

```yaml
- name: kdatp-prof
  profile: true
  data: {groups: "239,258", dp_size: 2}
  set: {profile_step_start: 0, profile_step_end: 1}
```

`20260929d` and `20260929e`, the arms jobs: T2 groups with replay, profile
off. `job.env`: `PROF_ARMS_FILE=arms.yaml`. Example `arms.yaml`:

```yaml
- name: base
  profile: false
  data: {groups: "243,284,307,258,239,306,276,309", dp_size: 2, replay_rollout: 40, dir: t2}
  set: {debug_exit_after_rollout: 3}
  timeout: 4500
- name: ep8
  profile: false
  data: {groups: "243,284,307,258,239,306,276,309", dp_size: 2, replay_rollout: 40, dir: t2}
  set: {expert_model_parallel_size: 8, debug_exit_after_rollout: 3}
  timeout: 4500
- name: tp4-ep8-dp4
  profile: false
  data: {groups: "243,284,307,258,239,306,276,309", dp_size: 4, replay_rollout: 40}
  set: {tensor_model_parallel_size: 4, expert_model_parallel_size: 8, debug_exit_after_rollout: 3}
  timeout: 7200
```

## Run

The image reference comes from the r47 trainer pod spec. Upload every
harness file before the create. The driver waits for `job.env`, the arms
file and `a2a_bench.py`, because the mount imports the files in no fixed
order.

```bash
K="kubectl --context arena-prod-bom-v2 -n arena-tasks"
IMAGE=427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r17-20260928a
STAMP=20260929c
H=s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/prof/$STAMP/harness/
for f in kdatp-prof-run.sh patch_prof_arm.py a2a_bench.py; do aws --profile arena-prod-bom-user s3 cp $f $H; done
aws --profile arena-prod-bom-user s3 cp <job.env of the job> ${H}job.env
aws --profile arena-prod-bom-user s3 cp <arms file of the job> ${H}arms.yaml   # only with PROF_ARMS_FILE
aws --profile arena-prod-bom-user s3 ls $H
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" kdatp-prof-job.yaml | $K create --dry-run=server -f -
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" kdatp-prof-job.yaml | $K create -f -
# When driver.log says "done" or the head pod has ended:
$K delete pytorchjob kdatp-prof-$STAMP --ignore-not-found
$K get pods -l app=kdatp-prof-$STAMP   # MUST show no pods
```

Zero-second TTL. The cluster adds `ttlSecondsAfterFinished: 0` to the job
`runPolicy` (our yaml does not set it). When a job succeeds, the operator
removes the job, its pods and its services at once (jobs `20260929c`, `d`
and `e`: gone within seconds of `done`). So `kubectl logs` of a finished
job is lost. Read the live logs with a read-only `kubectl exec` into the
head pod while the job runs, and keep every output on the scratch mount.
The S3 view of the mount lags by some minutes. The TTL effect on a failed
job is UNVERIFIED, so the delete command stays.

Results in `kdatp/prof/<stamp>/`: `driver.log`, `arms/<name>.yaml`,
`arms/plan.tsv`, `<name>/trainer-0.log` (`perf N:` and `[peak-memory]`
lines), `<name>/rc`, `<name>/tb/train_overall_rank_<rank>.<ns>.pt.trace.json.gz`
for ranks 0 to 63 of a profiled arm, `a2a/`, `data-<dir>.log` for each data
build, and `results.json` (in-image `parse_logs.py`, rewritten after each
arm: per arm `rc`, `out_of_memory`, metrics, `steady_mean` over the steps
after the first one, peak memory per stage).

Read a trace with these limits:

- Step 1 carries the profiler overhead (`record_shapes`, `with_stack`,
  `profile_memory`, `with_flops`). Compare its `perf 41` times with `perf 42`.
- The slice gives 33 rows per DP rank, so the pipeline bubble (3 of 33
  microbatch slots per stage) is larger than in r47, which trains 256
  episodes (about 1,250 rows) per step. Read the per-microbatch kernel mix,
  not the bubble share.
- Rank to stage: ranks 0 to 15 are pipeline stage 0 (ranks 0 to 7 DP 0, 8 to
  15 DP 1), 16 to 31 stage 1, 32 to 47 stage 2, 48 to 63 stage 3. Confirm
  from the `[peak-memory] ... rank=N pp=P tp=T` lines of the trainer log.
- A kernel cache miss shows as a long step 0 in `driver.log`; the profiled
  step 1 is warm in both cases.
- With `skip_actor_forward_only`, the step has no separate log-prob pass, so
  `perf/log_probs_time` is absent, as in r47.

Rules: `../README.md` (never touch a live run; MUST delete the job when it
ends).
