#!/usr/bin/env bash
# glm53 memprobe driver: one p6-b200 node, the run's SGLang engine config, a
# 30 s host/cgroup/process memory sampler, the WeightChecker snapshot the run
# issued (check_weight_update_equal: true), a rollout-like /generate load with
# client aborts, then 20 more minutes of sampling. Everything is written to EFS
# (/mnt/scratch-s3files-rw) because the Job/pod is the only other copy and a
# failed pod's logs disappear.
#
# Phases (phases.csv):  boot -> loading -> loaded -> idle -> snapshot_start ->
#   snapshot_done -> snapshot_idle -> load_start -> load_end -> post -> done
set -uo pipefail

NAME="${MEMPROBE_NAME:-glm53-memprobe}"
USER_NAME="${USER:-guparpit}"
OUT="/mnt/scratch-s3files-rw/${USER_NAME}/logs/glm53-memprobe/${NAME}"
MODEL="${MEMPROBE_MODEL:-/mnt/scratch-fast-1a-rw/durable/model-artifacts/external/zai-org/GLM-5.3-Flash-BF16}"
PORT="${MEMPROBE_PORT:-30000}"
BASE="http://127.0.0.1:${PORT}"
IDLE_S="${MEMPROBE_IDLE_S:-180}"
SNAPSHOT="${MEMPROBE_SNAPSHOT:-1}"
LOAD_MIN="${MEMPROBE_LOAD_MIN:-40}"
POST_S="${MEMPROBE_POST_S:-1200}"
SAMPLE_S="${MEMPROBE_SAMPLE_S:-30}"
LOAD_WAIT_MAX_S="${MEMPROBE_LOAD_WAIT_MAX_S:-2700}"   # 45 min: run 2 loaded in 363 s + CUDA graphs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT"
# full driver log to EFS (the Job's stdout is tee'd separately by the pod command)
exec > >(tee -a "$OUT/memprobe.log") 2>&1

echo "=== memprobe ${NAME} start $(date -u +%FT%TZ) pod=${POD_NAME:-?} node=${NODE_NAME:-?} variant=${MEMPROBE_VARIANT:-?} out=${OUT}"

phase() {
  echo "$(date -u +%FT%TZ),$1" >> "$OUT/phases.csv"
  printf '%s' "$1" > "$OUT/phase.current"
  echo "=== PHASE $1 $(date -u +%FT%TZ)"
}

# tiny stdlib HTTP helpers (curl is not guaranteed in the image)
http_get() {  # url -> prints status code
  python3 - "$1" <<'PY'
import sys, urllib.request, urllib.error
try:
    r = urllib.request.urlopen(sys.argv[1], timeout=30); print(r.status)
except urllib.error.HTTPError as e:
    print(e.code)
except Exception as e:
    print("ERR", e)
PY
}
http_post_json() {  # url json -> prints status + body
  python3 - "$1" "$2" <<'PY'
import sys, urllib.request, urllib.error
req = urllib.request.Request(sys.argv[1], data=sys.argv[2].encode(), headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=3600); print(r.status, r.read()[:2000].decode(errors="replace"))
except urllib.error.HTTPError as e:
    print(e.code, e.read()[:2000].decode(errors="replace"))
except Exception as e:
    print("ERR", e)
PY
}

sgl_alive() { [ -n "${SGL_PID:-}" ] && kill -0 "$SGL_PID" 2>/dev/null; }

# sleep N seconds in 10 s steps, bail out if the server died
sleep_watch() {
  local n=$1 t=0
  while [ "$t" -lt "$n" ]; do
    if ! sgl_alive; then echo "!!! sglang server (pid ${SGL_PID:-?}) died during wait"; return 1; fi
    sleep 10; t=$((t + 10))
  done
}

cleanup() {
  local rc=$?
  phase "exit rc=${rc}"
  echo "=== cleanup: killing sglang (${SGL_PID:-none}) and sampler (${SAMPLER_PID:-none})"
  if [ -n "${SGL_PID:-}" ]; then kill -TERM "$SGL_PID" 2>/dev/null; sleep 15; pkill -9 -f "sglang" 2>/dev/null; fi
  # one last sample after the server is gone (baseline: what is NOT the engine)
  python3 "$SCRIPT_DIR/memsampler.py" --out "$OUT" --once --top 12 || true
  [ -n "${SAMPLER_PID:-}" ] && kill "$SAMPLER_PID" 2>/dev/null
  echo "=== memprobe ${NAME} end $(date -u +%FT%TZ) rc=${rc}; artifacts in ${OUT}"
  sync || true
}
trap cleanup EXIT

# ---------------------------------------------------------------- 0. static facts
{
  echo "date=$(date -u +%FT%TZ) pod=${POD_NAME:-?} node=${NODE_NAME:-?} variant=${MEMPROBE_VARIANT:-?} hostname=$(hostname)"
  echo "--- /proc/self/cgroup"; cat /proc/self/cgroup
  echo "--- ls /sys/fs/cgroup (first 60)"; ls /sys/fs/cgroup 2>&1 | head -60
  echo "--- what Ray 2.58 reads: /sys/fs/cgroup/memory.max"; cat /sys/fs/cgroup/memory.max 2>&1
  echo "--- /sys/fs/cgroup/cgroup.controllers"; cat /sys/fs/cgroup/cgroup.controllers 2>&1
  echo "--- kubepods.slice visible?"; ls -d /sys/fs/cgroup/kubepods.slice 2>&1
  echo "--- /proc/meminfo"; cat /proc/meminfo
  echo "--- df /dev/shm"; df -h /dev/shm 2>&1
  echo "--- mounts of interest"; grep -E ' /mnt/| /dev/shm | cgroup' /proc/mounts
  echo "--- nvidia-smi -L"; nvidia-smi -L 2>&1
  echo "--- versions"; python3 -c 'import ray, sglang, torch; print("ray", ray.__version__); print("sglang", sglang.__version__); print("torch", torch.__version__)' 2>&1
  python3 -m pip show flashinfer-python tilelang 2>/dev/null | grep -E '^(Name|Version)'
  echo "--- env"; env | grep -E '^(SGLANG|RAY|CUDA|NCCL|PYTORCH|TRITON|TORCHINDUCTOR|FI_|MEMPROBE|LD_LIBRARY)' | sort
  echo "--- ulimit -a"; ulimit -a
} > "$OUT/info.txt" 2>&1
echo "static facts -> $OUT/info.txt"; grep -E "memory.max|kubepods|^ray|^sglang" "$OUT/info.txt" | head -8

# ---------------------------------------------------------------- 1. sampler
python3 "$SCRIPT_DIR/memsampler.py" --out "$OUT" --interval "$SAMPLE_S" --top 12 &
SAMPLER_PID=$!
phase boot
sleep 5

# ---------------------------------------------------------------- 2. SGLang
# EXACTLY the run's ServerArgs (/tmp/glm53-facts/sglang_server_args.txt, log
# line 2120), translated to CLI flags (auto-derived from ServerArgs field names
# by add_cli_args_from_dataclass, sglang server_args.py:8751-8754). Flags the
# run had at their defaults are omitted; --disable-shared-experts-fusion is
# auto-set by flashinfer_trtllm (log 2118) and not passed. miles used host=<pod
# IP>, port 15000, nccl_port 15001 -- irrelevant here (single node, loopback).
SGL_ARGS=(
  --model-path "$MODEL"
  --tp-size 8
  --ep-size 8
  --dp-size 1
  --mem-fraction-static 0.7
  --context-length 131072
  --chunked-prefill-size 8192
  --max-prefill-tokens 16384
  --page-size 64
  --disable-radix-cache
  --kv-cache-dtype bfloat16
  --attention-backend dsa
  --dsa-prefill-backend tilelang
  --dsa-decode-backend tilelang
  --mamba-backend triton
  --moe-runner-backend flashinfer_trtllm
  --disable-overlap-schedule
  --trust-remote-code
  --skip-server-warmup
  --enable-metrics
  --random-seed 1234
  --watchdog-timeout 300
  --log-level info
  --host 0.0.0.0
  --port "$PORT"
)
echo "launching: python3 -m sglang.launch_server ${SGL_ARGS[*]}"
python3 -m sglang.launch_server "${SGL_ARGS[@]}" > "$OUT/sglang-server.log" 2>&1 &
SGL_PID=$!
phase loading

# ---------------------------------------------------------------- 3. wait for /health
t=0
while :; do
  if ! sgl_alive; then echo "!!! sglang exited during load; tail of server log:"; tail -50 "$OUT/sglang-server.log"; exit 2; fi
  code=$(http_get "$BASE/health")
  if [ "$code" = "200" ]; then break; fi
  if [ "$t" -ge "$LOAD_WAIT_MAX_S" ]; then echo "!!! /health not 200 after ${t}s (last=$code)"; tail -50 "$OUT/sglang-server.log"; exit 3; fi
  sleep 15; t=$((t + 15))
done
phase loaded
echo "loaded after ~${t}s; scheduler pids: $(pgrep -f 'sglang::scheduler' | tr '\n' ' ')"
grep -E "Load weight end|Capture cuda graph end|avail mem|mem usage" "$OUT/sglang-server.log" | tail -12
echo "--- weight-load host baseline (top procs by RSS):"; python3 "$SCRIPT_DIR/memsampler.py" --out "$OUT" --once --top 12 >/dev/null; tail -12 "$OUT/procs.csv" | cut -d, -f3-9

# one tiny real request (health_generate does the same with input_ids=[0])
http_post_json "$BASE/generate" '{"input_ids":[1000,1001,1002,1003],"sampling_params":{"max_new_tokens":8,"temperature":1.0},"return_logprob":true}' | cut -c1-300

phase idle
sleep_watch "$IDLE_S" || exit 4

# ---------------------------------------------------------------- 4. WeightChecker snapshot
# This is what miles/ray/placement_group.py:213-217 issues at startup when
# check_weight_update_equal is true: a CPU copy of every parameter+buffer of
# each TP rank (sglang weight_checker.py:114-121), never released. Hypothesis R2
# predicts scheduler RssAnon/USS jumps by ~73 GiB per rank right here.
if [ "$SNAPSHOT" = "1" ]; then
  phase snapshot_start
  http_post_json "$BASE/weights_checker" '{"action":"snapshot"}' | cut -c1-400
  phase snapshot_done
  sleep 60
  echo "--- post-snapshot top procs:"; tail -12 "$OUT/procs.csv" | cut -d, -f3-9
  phase snapshot_idle
  sleep_watch "$IDLE_S" || exit 5
else
  echo "snapshot skipped (MEMPROBE_SNAPSHOT=$SNAPSHOT)"
fi

# ---------------------------------------------------------------- 5. load
phase load_start
python3 "$SCRIPT_DIR/loadgen.py" \
  --base-url "$BASE" \
  --minutes "$LOAD_MIN" \
  --concurrency "${MEMPROBE_CONCURRENCY:-96}" \
  --prompt-min "${MEMPROBE_PROMPT_MIN:-40000}" \
  --prompt-max "${MEMPROBE_PROMPT_MAX:-100000}" \
  --max-new-tokens "${MEMPROBE_MAX_NEW_TOKENS:-2048}" \
  --abort-frac "${MEMPROBE_ABORT_FRAC:-0.10}" \
  --csv "$OUT/loadgen.csv" \
  --log "$OUT/loadgen.log"
LG_RC=$?
phase "load_end rc=${LG_RC}"
sgl_alive || { echo "!!! sglang died during load"; tail -80 "$OUT/sglang-server.log"; exit 6; }

# ---------------------------------------------------------------- 6. post-load sampling
phase post
sleep_watch "$POST_S" || exit 7
phase done
echo "=== summary"
echo "phases:"; cat "$OUT/phases.csv"
echo "last mem row:"; tail -1 "$OUT/mem.csv"
echo "server log grep (memory pressure / OOM / abort):"; grep -ciE "out of memory|OOM|killed" "$OUT/sglang-server.log"
exit 0
