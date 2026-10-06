#!/usr/bin/env bash
# sgl driver: the 1-node SGLang engine benchmark of the DSA attention backends (sgl_bench.py). The pod of
# the kdatp-sgl-<stamp> PyTorchJob runs it from the scratch mount. The engine code is the image code; the
# harness files come from the mount so that they can change without a build.
#
# Writes only under $SGL_DIR/<stamp> (default /mnt/scratch-s3files-rw/acuadron/kdatp/sgl/<stamp>), plus
# the pod disk under /tmp/sgl. Inputs: the model on the fast scratch mount and synthetic prompts.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP=${SGL_STAMP:?the manifest sets SGL_STAMP}
RUN=${SGL_DIR:-/mnt/scratch-s3files-rw/acuadron/kdatp/sgl}/$STAMP
LOCAL=/tmp/sgl
T0=$(date +%s)

for i in $(seq 120); do [ -f "$HERE/sgl_bench.py" ] && break; echo "[sgl] waiting for $HERE/sgl_bench.py"; sleep 5; done
mkdir -p "$RUN" "$LOCAL/harness" "$LOCAL/kcache"
exec > >(tee -a "$RUN/driver.log") 2>&1
log() { echo "[sgl $(date -u +%FT%TZ)] $*"; }
log "stamp=$STAMP image_head=$(git -C "${AGISLIME_DIR:-/root/miles}" rev-parse HEAD 2>/dev/null)"
cp "$HERE/sgl_bench.py" "$LOCAL/harness/" || { log "harness copy FAILED"; exit 1; }
sha256sum "$LOCAL/harness/sgl_bench.py" | tee "$RUN/harness.sha256"
python3 - <<'PY'
import importlib.metadata as m
import torch
for name in ("sglang", "flashinfer-python", "flash_mla", "tilelang", "nvidia-cutlass-dsl", "torch"):
    try:
        print(name, m.version(name), flush=True)
    except m.PackageNotFoundError:
        print(name, "not installed", flush=True)
print("cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)
PY
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used --format=csv,noheader -l 30 > "$RUN/gpu-power.csv" 2>&1 &
SMI_PID=$!

export PYTHONDONTWRITEBYTECODE=1
export TILELANG_CACHE_DIR=$LOCAL/kcache/tilelang TRITON_CACHE_DIR=$LOCAL/kcache/triton TORCHINDUCTOR_CACHE_DIR=$LOCAL/kcache/inductor
log "bench: start arms=[${SGL_ARMS:-default}]"
# shellcheck disable=SC2086
timeout -k 60 "${SGL_TIMEOUT:-10000}" python3 "$LOCAL/harness/sgl_bench.py" --out "$RUN" ${SGL_ARMS:+--arms $SGL_ARMS} ${SGL_BENCH_ARGS:-} > "$RUN/bench.log" 2>&1
rc=$?
echo "$rc" > "$RUN/bench.rc"
log "bench: rc=$rc at $(( $(date +%s) - T0 )) s"
grep -E "^\[sgl-bench\]" "$RUN/bench.log" | tail -40
kill "$SMI_PID" 2>/dev/null
echo "done $(date -u +%FT%TZ) elapsed_s=$(( $(date +%s) - T0 )) bench_rc=$rc" > "$RUN/done"
sync
exit 0
