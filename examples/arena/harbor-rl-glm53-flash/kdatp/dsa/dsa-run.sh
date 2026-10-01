#!/usr/bin/env bash
# kdatp-dsa driver: the 1-node DSA kernel test of miles PR #3608 (README.md). The pod of the
# kdatp-dsa1-<stamp> PyTorchJob runs it from the scratch mount, so the image code does not change.
#
# 1. Waits for the harness files, copies them to the pod disk and logs their sha256.
# 2. Starts one t1_dsa.py process per GPU; GPU g runs group g. After round 1, the processes on
#    the GPUs other than the pytest GPU repeat their timed calls until the stop file exists, so
#    that the GPUs stay busy (the idle-GPU reaper deletes a workload whose 60-minute mean GPU
#    power is below 10%).
# 3. When every process has ended round 1 (or DSA_ROUND1_TIMEOUT), runs the PR unit tests on the
#    pytest GPU: cd "$AGISLIME_DIR" && python3 -m pytest -q tests/fast-gpu/kernels/attention/dsa
# 4. Stops the busy rounds, merges the case files into results.json and results.md, writes done.
#
# Writes only under $KDATP_DIR/dsa/$KDATP_STAMP (default
# /mnt/scratch-s3files-rw/guparpit/kdatp/dsa/<stamp>), plus the pod disk under /tmp/kdatp-dsa
# (the harness copy, the kernel caches, the sync files). Synthetic kernel inputs only; no dataset.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KD=${KDATP_DIR:-/mnt/scratch-s3files-rw/guparpit/kdatp}
STAMP=${KDATP_STAMP:?the manifest sets KDATP_STAMP}
REPO=${AGISLIME_DIR:?the job manifest sets AGISLIME_DIR to the image checkout}
RUN=$KD/dsa/$STAMP
LOCAL=${DSA_LOCAL:-/tmp/kdatp-dsa}  # the pod disk (overlay), not the scratch mount
NGPU=8
PYTEST_GPU=0
# The job deadline (activeDeadlineSeconds) is 5,400 s. The last 300 s stay free for the stop,
# the merge and the done file.
DEADLINE=${DSA_DEADLINE:-5400}
ROUND1_TIMEOUT=${DSA_ROUND1_TIMEOUT:-2700}
PYTEST_TIMEOUT=${DSA_PYTEST_TIMEOUT:-2100}
T0=$(date +%s)
FILES="t1_dsa.py old_dsa/__init__.py old_dsa/sparse_mla.py old_dsa/tilelang_sparse_mla_fwd.py old_dsa/tilelang_sparse_mla_bwd.py"

wait_file() {  # the mount imports the uploaded harness files within minutes, in no fixed order
  local i
  for i in $(seq 120); do
    [ -f "$1" ] && return 0
    echo "[kdatp-dsa] waiting for $1"
    sleep 5
  done
  echo "[kdatp-dsa] missing $1"
  return 1
}
for f in $FILES; do
  wait_file "$HERE/$f" || exit 1
done

mkdir -p "$RUN" "$LOCAL/sync" "$LOCAL/harness"
exec > >(tee -a "$RUN/driver.log") 2>&1
log() { echo "[kdatp-dsa $(date -u +%FT%TZ)] $*"; }
elapsed() { echo $(($(date +%s) - T0)); }
log "stamp=$STAMP image_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) image_dirty_files=$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)"
log "data: synthetic kernel inputs only (random q, kv and upstream gradients; the PR bench causal_indices;" \
  "kpool_select_topk on random index tensors). The agentic-debt dataset of record" \
  "lakefs://arena-inspect/main/internal/agentic-debt-r3/agentic-debt-766/ is not used."
# A frozen copy on the pod disk: a later upload to the mount cannot change a running test.
cp "$HERE/t1_dsa.py" "$LOCAL/harness/" && cp -r "$HERE/old_dsa" "$LOCAL/harness/" || { log "harness copy FAILED"; exit 1; }
(cd "$LOCAL/harness" && sha256sum t1_dsa.py old_dsa/*.py) | tee "$RUN/harness.sha256"
python3 - <<'PY'
import importlib.metadata as m

import torch

for name in ("torch", "tilelang", "triton", "flash_mla", "sglang-kernel"):
    try:
        print(name, m.version(name), flush=True)
    except m.PackageNotFoundError:
        print(name, "not installed", flush=True)
print("cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
PY
nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv,noheader
# GPU power and use every 60 s, the input of the idle-GPU reaper.
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used --format=csv,noheader -l 60 > "$RUN/gpu-power.csv" 2>&1 &
SMI_PID=$!

export TILELANG_PRINT_ON_COMPILATION=1 PYTHONDONTWRITEBYTECODE=1
BUSY_UNTIL=$((T0 + DEADLINE - 300))
pids=()
for g in $(seq 0 $((NGPU - 1))); do
  mkdir -p "$RUN/gpu$g" "$LOCAL/kcache/gpu$g"
  busy=()
  [ "$g" != "$PYTEST_GPU" ] && busy=(--busy --busy-until "$BUSY_UNTIL")
  # One kernel cache per process: a fresh cache gives a true compile count, and no two processes
  # write the same cache entry.
  CUDA_VISIBLE_DEVICES=$g TILELANG_CACHE_DIR=$LOCAL/kcache/gpu$g/tilelang TRITON_CACHE_DIR=$LOCAL/kcache/gpu$g/triton \
    TORCHINDUCTOR_CACHE_DIR=$LOCAL/kcache/gpu$g/inductor \
    python3 "$LOCAL/harness/t1_dsa.py" --group "$g" --out "$RUN/gpu$g" --marker "$LOCAL/sync/gpu$g.round1" \
    --stop-file "$LOCAL/sync/stop" "${busy[@]}" > "$RUN/gpu$g/log.txt" 2>&1 &
  pids+=($!)
  log "gpu $g: pid ${pids[$g]} ${busy[*]}"
done

# Round 1 ends for a GPU when its marker exists or its process has exited.
while :; do
  n=0
  for g in $(seq 0 $((NGPU - 1))); do
    if [ -f "$LOCAL/sync/gpu$g.round1" ] || ! kill -0 "${pids[$g]}" 2>/dev/null; then
      n=$((n + 1))
    fi
  done
  [ "$n" = "$NGPU" ] && break
  if [ "$(elapsed)" -ge "$ROUND1_TIMEOUT" ]; then
    log "round 1: timeout after $(elapsed) s, $n of $NGPU ended"
    break
  fi
  sleep 30
done
log "round 1: $(ls "$LOCAL"/sync/*.round1 2>/dev/null | wc -l) of $NGPU markers at $(elapsed) s"

# The pytest GPU process has no busy rounds, so it exits after round 1. Free its GPU for pytest.
p=${pids[$PYTEST_GPU]}
for i in $(seq 24); do
  kill -0 "$p" 2>/dev/null || break
  sleep 5
done
if kill -0 "$p" 2>/dev/null; then
  log "gpu $PYTEST_GPU: process $p still runs; stop it for pytest"
  kill "$p"
fi

left=$((DEADLINE - $(elapsed) - 300))
limit=$((left < PYTEST_TIMEOUT ? left : PYTEST_TIMEOUT))
if [ "$limit" -ge 120 ]; then
  log "pytest: start on GPU $PYTEST_GPU, timeout $limit s"
  (cd "$REPO" && CUDA_VISIBLE_DEVICES=$PYTEST_GPU TILELANG_CACHE_DIR=$LOCAL/kcache/pytest/tilelang \
    TRITON_CACHE_DIR=$LOCAL/kcache/pytest/triton TORCHINDUCTOR_CACHE_DIR=$LOCAL/kcache/pytest/inductor \
    timeout -k 30 "$limit" python3 -m pytest -q tests/fast-gpu/kernels/attention/dsa) > "$RUN/pytest.log" 2>&1
  echo $? > "$RUN/pytest.rc"
else
  log "pytest: skipped, only $left s left"
  echo skipped > "$RUN/pytest.rc"
fi
log "pytest: rc=$(cat "$RUN/pytest.rc") at $(elapsed) s: $(grep -E '[0-9]+ (passed|failed|error)' "$RUN/pytest.log" 2>/dev/null | tail -1)"

# Stop the busy rounds. A process ends its current timed calls first (at most about a minute).
touch "$LOCAL/sync/stop"
for i in $(seq 60); do
  alive=0
  for p in "${pids[@]}"; do
    kill -0 "$p" 2>/dev/null && alive=1
  done
  [ "$alive" = 0 ] && break
  sleep 5
done
for g in $(seq 0 $((NGPU - 1))); do
  p=${pids[$g]}
  if kill -0 "$p" 2>/dev/null; then
    log "gpu $g: process $p still runs after the stop; kill it"
    kill "$p"
  fi
  wait "$p"
  rc=$?
  echo "$rc" > "$RUN/gpu$g/rc"
  rounds=0
  [ -f "$RUN/gpu$g/rounds.jsonl" ] && rounds=$(wc -l < "$RUN/gpu$g/rounds.jsonl")
  log "gpu $g: rc=$rc busy-round rows=$rounds"
done

python3 "$LOCAL/harness/t1_dsa.py" --merge "$RUN" > "$RUN/merge.log" 2>&1
merge_rc=$?
log "merge: rc=$merge_rc $(tail -1 "$RUN/merge.log")"
kill "$SMI_PID" 2>/dev/null
echo "done $(date -u +%FT%TZ) elapsed_s=$(elapsed) pytest_rc=$(cat "$RUN/pytest.rc") merge_rc=$merge_rc" > "$RUN/done"
log "done: $RUN/results.json $RUN/results.md"
sync
exit 0
