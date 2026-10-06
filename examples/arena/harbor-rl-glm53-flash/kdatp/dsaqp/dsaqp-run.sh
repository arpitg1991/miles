#!/usr/bin/env bash
# dsaqp driver: the 1-node parity and timing test of --glm5-next-dsa-qp (../dsa/t1_dsa_qp.py). The pod
# of the kdatp-dsaqp-<stamp> PyTorchJob runs it from the scratch mount. The code under test is the
# image code (/root/miles); the harness file comes from the mount so that it can change without a build.
#
# Writes only under $DSAQP_DIR/<stamp> (default /mnt/scratch-s3files-rw/guparpit/kdatp/dsaqp/<stamp>),
# plus the pod disk under /tmp/dsaqp. Synthetic inputs only (random hidden states and weights); no dataset.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP=${DSAQP_STAMP:?the manifest sets DSAQP_STAMP}
RUN=${DSAQP_DIR:-/mnt/scratch-s3files-rw/guparpit/kdatp/dsaqp}/$STAMP
REPO=${AGISLIME_DIR:-/root/miles}
LOCAL=/tmp/dsaqp
T0=$(date +%s)

for i in $(seq 120); do [ -f "$HERE/t1_dsa_qp.py" ] && break; echo "[dsaqp] waiting for $HERE/t1_dsa_qp.py"; sleep 5; done
mkdir -p "$RUN" "$LOCAL/harness" "$LOCAL/kcache"
exec > >(tee -a "$RUN/driver.log") 2>&1
log() { echo "[dsaqp $(date -u +%FT%TZ)] $*"; }
log "stamp=$STAMP image_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) image_dirty_files=$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)"
log "data: synthetic (random hidden states, Megatron-initialized weights). No dataset."
cp "$HERE/t1_dsa_qp.py" "$LOCAL/harness/" || { log "harness copy FAILED"; exit 1; }
sha256sum "$LOCAL/harness/t1_dsa_qp.py" | tee "$RUN/harness.sha256"
python3 - <<'PY'
import importlib.metadata as m
import torch
for name in ("torch", "tilelang", "triton", "flash_mla", "megatron-core", "transformer_engine"):
    try:
        print(name, m.version(name), flush=True)
    except m.PackageNotFoundError:
        print(name, "not installed", flush=True)
print("cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)
PY
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used --format=csv,noheader -l 60 > "$RUN/gpu-power.csv" 2>&1 &
SMI_PID=$!

export TILELANG_PRINT_ON_COMPILATION=1 PYTHONDONTWRITEBYTECODE=1
# The flashinfer top-k is deterministic only with this set; the parity test compares two calls on the same logits.
export SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=1
export TILELANG_CACHE_DIR=$LOCAL/kcache/tilelang TRITON_CACHE_DIR=$LOCAL/kcache/triton TORCHINDUCTOR_CACHE_DIR=$LOCAL/kcache/inductor
# Fast tests that need libcuda (the megatron bridge import, a GPU Ray fixture) and so cannot run on a
# CPU-only build host: DSAQP_PYTEST lists them, space-separated, as paths under $REPO.
if [ -n "${DSAQP_PYTEST:-}" ]; then
  log "pytest: start ($DSAQP_PYTEST)"
  # shellcheck disable=SC2086
  (cd "$REPO" && timeout -k 60 1800 python3 -m pytest -q --no-header -p no:cacheprovider $DSAQP_PYTEST) > "$RUN/pytest.log" 2>&1
  echo $? > "$RUN/pytest.rc"
  log "pytest: rc=$(cat "$RUN/pytest.rc") $(grep -E '[0-9]+ (passed|failed)' "$RUN/pytest.log" | tail -1)"
fi
log "parity: start"
(cd "$REPO" && timeout -k 60 "${DSAQP_TIMEOUT:-3600}" python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
  --rdzv-backend c10d --rdzv-endpoint localhost:29511 "$LOCAL/harness/t1_dsa_qp.py" --out "$RUN/parity" ${DSAQP_ARGS:-}) \
  > "$RUN/parity.log" 2>&1
rc=$?
echo "$rc" > "$RUN/parity/rc" 2>/dev/null || echo "$rc" > "$RUN/parity.rc"
log "parity: rc=$rc at $(( $(date +%s) - T0 )) s"
grep -E "^\[t1-qp\]" "$RUN/parity.log" | tail -40
kill "$SMI_PID" 2>/dev/null
echo "done $(date -u +%FT%TZ) elapsed_s=$(( $(date +%s) - T0 )) parity_rc=$rc" > "$RUN/done"
sync
exit 0
