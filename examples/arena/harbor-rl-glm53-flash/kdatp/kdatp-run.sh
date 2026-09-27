#!/usr/bin/env bash
# kdatp test driver. Each pod of a kdatp-* PyTorchJob runs it (README.md).
#
#   kdatp-run.sh t1   # 1 node: T1 parity, then the T1c 5-layer train-only arms
#   kdatp-run.sh t2   # 8 nodes: the T2 train-only arms on the r45 layout
#
# Writes only under $KDATP_DIR (default /mnt/scratch-s3files-rw/guparpit/kdatp).
# Reads the base DCP, the r43 iter_0000039 DCP and HF export, the r43 staged
# token files and the r45 sample summary. T2 links the r43 DCP files into its
# own checkpoint dirs and never saves (save_interval 100000).
set -uo pipefail

TEST=${1:?usage: kdatp-run.sh t1|t2}
KD=${KDATP_DIR:-/mnt/scratch-s3files-rw/guparpit/kdatp}
STAMP=${KDATP_STAMP:?the manifest sets one KDATP_STAMP for all pods}
HERE=/root/miles/examples/arena/harbor-rl-glm53-flash/kdatp
LAUNCHER=/root/miles/scripts/run_arena_harbor.py
HF=/mnt/scratch-fast-1a-rw/durable/model-artifacts/external/zai-org/GLM-5.3-Flash-BF16
BASE_DCP=/mnt/scratch-s3files-rw/guparpit/checkpoints/dcp/glm5.3-flash_torch_dist
R43=/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-adebt-v3-r43
R43_ROUTING=/mnt/scratch-s3files-rw/guparpit/routing/rl-glm53f-adebt-v3-r43
R45_SUMMARY=/mnt/scratch-s3files-rw/guparpit/debug/rl-glm53f-adebt-v3-r45/sample_summary/rollout_43.jsonl
T1C_GROUPS=${T1C_GROUPS:-243}
T2_GROUPS=${T2_GROUPS:-243,284,307,258,239,306,276,309}
T1C_ARMS=${T1C_ARMS:-baseline kdatp selective kdatp-selective none kdatp-none}
T2_ARMS=${T2_ARMS:-baseline kdatp kdatp-block10 baseline2 kdatp-selective}
ARM_TIMEOUT=${ARM_TIMEOUT:-7200}
RUN=$KD/$TEST/$STAMP

export ARENA_CHECKPOINTS_DIR=$KD/checkpoints ARENA_DATA_DIR=$KD/run
export MILES_LOG_PEAK_MEMORY=1
# Kernel JIT caches on local disk (r45 pattern). All arms of a job share them, so only arm 1 compiles cold.
export KCACHE=/tmp/kernel_cache
export TILELANG_CACHE_DIR=$KCACHE/tilelang TRITON_CACHE_DIR=$KCACHE/triton TORCHINDUCTOR_CACHE_DIR=$KCACHE/inductor
mkdir -p "$TILELANG_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
ulimit -n 1000000

if [ "${REPLICA_IDX:-0}" != "0" ]; then
  # T2 workers join the head's ray cluster and block until the head stops it.
  exec python3 "$LAUNCHER" worker
fi

mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
log() { echo "[kdatp $(date -u +%FT%TZ)] $*"; }
log "test=$TEST stamp=$STAMP image_head=$(git -C /root/miles rev-parse HEAD 2>/dev/null)"
cp "$HERE"/*.py "$HERE"/kdatp-run.sh "$RUN/" 2>/dev/null

build_data() {  # name groups rollout_ids dp_size
  local out=$KD/data/$1
  if [ -f "$out/manifest.json" ]; then
    log "data $1 exists: $out/manifest.json"
    return 0
  fi
  log "building data $1 (groups $2)"
  python3 "$HERE/build_rollout_data.py" --summary "$R45_SUMMARY" --groups "$2" --routing-dir "$R43_ROUTING" \
    --out-dir "$out" --rollout-ids "$3" --dp-size "$4" > "$RUN/data-$1.log" 2>&1
}

run_arm() {  # test arm
  local dir=$RUN/$2
  mkdir -p "$dir"
  log "arm $2: start"
  EXPERIMENT_NAME=kdatp-$1-$2-$STAMP PROJECT_NAME=kdatp-$1 CFG_NAME=$RUN/arms/$2.yaml \
    timeout --signal=TERM "$ARM_TIMEOUT" python3 "$LAUNCHER" train > "$dir/trainer-0.log" 2>&1
  echo $? > "$dir/rc"
  log "arm $2: rc=$(cat "$dir/rc")"
}

stop_ray_jobs() {  # a timed-out launcher leaves its ray job running; free the GPUs for the next arm
  python3 - <<'PY'
from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://127.0.0.1:8265")
for job in client.list_jobs():
    if str(job.status) in ("PENDING", "RUNNING", "JobStatus.PENDING", "JobStatus.RUNNING"):
        print("stopping ray job", job.submission_id, flush=True)
        client.stop_job(job.submission_id)
PY
  sleep 60
}

seed_t2_ckpt() {  # experiment name: link r43 iter_0000039 file by file (r45/BUILD.md pattern)
  local dst=$ARENA_CHECKPOINTS_DIR/slime_experiments/$1 it=iter_0000039
  if [ -e "$dst" ]; then
    log "refusing to reuse $dst"
    return 1
  fi
  mkdir -p "$dst/$it"
  for f in "$R43/$it/.metadata" "$R43/$it/metadata.json" "$R43/$it"/__*_0.distcp; do
    ln -s "$f" "$dst/$it/${f##*/}"
  done
  printf '{"rollout_id": 39}' > "$dst/$it/slime_extra_state.json"  # no wandb_run_id: never the r43 W&B run
  printf 39 > "$dst/latest_checkpointed_iteration.txt"
  test "$(ls -A "$dst/$it" | wc -l)" = "$(ls -A "$R43/$it" | wc -l)"
}

case "$TEST" in
t1)
  export REPLICA=1 REPLICA_TRAINER=1
  [ -f "$KD/data/hf/GLM-5.3-Flash-5layer/config.json" ] ||
    python3 "$HERE/make_slice_hf.py" --src "$HF" --out "$KD/data/hf/GLM-5.3-Flash-5layer" --layers 5
  build_data t1c "$T1C_GROUPS" 0,1 1 || log "data t1c: FAILED (see data-t1c.log)"
  # The T2 data build is CPU and disk only; it runs while the GPUs do T1.
  build_data t2 "$T2_GROUPS" 40,41,42,43 2 &
  T2_DATA_PID=$!

  log "T1 parity"
  mkdir -p "$RUN/parity"
  python3 -m torch.distributed.run --nproc-per-node 8 "$HERE/t1_parity.py" \
    --hf-checkpoint "$HF" \
    --ckpt "base=$BASE_DCP=$HF" \
    --ckpt "r43=$R43/iter_0000039=$R43/hf/rollout_39" \
    --layers 0,44 --timing-lengths 8192,32768,131072 \
    --out "$RUN/parity" > "$RUN/parity/log.txt" 2>&1
  echo $? > "$RUN/parity/rc"
  log "T1 parity rc=$(cat "$RUN/parity/rc")"

  python3 "$HERE/gen_arm_configs.py" t1c --data-dir "$KD/data" --out "$RUN/arms"
  for arm in $T1C_ARMS; do
    run_arm t1c "$arm"
  done
  wait "$T2_DATA_PID"
  log "data t2: rc=$?"
  ;;
t2)
  export REPLICA=${REPLICA:-8} REPLICA_TRAINER=${REPLICA_TRAINER:-8}
  test -f "$KD/data/t2/manifest.json" || { log "no T2 data: run t1 first"; exit 1; }
  HEAD=$(hostname)
  export MILES_SCRIPT_EXTERNAL_RAY=1
  ray start --head --node-ip-address "$HEAD" --num-gpus 8 --disable-usage-stats
  python3 "$HERE/gen_arm_configs.py" t2 --data-dir "$KD/data" --out "$RUN/arms"
  for arm in $T2_ARMS; do
    if seed_t2_ckpt "kdatp-t2-$arm-$STAMP"; then
      run_arm t2 "$arm"
    else
      log "arm $arm: seed FAILED"
    fi
    stop_ray_jobs
  done
  ;;
*)
  log "unknown test $TEST"
  exit 2
  ;;
esac

python3 "$HERE/parse_logs.py" "$RUN" > "$RUN/results.json" 2> "$RUN/parse_logs.err"
log "done: $RUN/results.json"
if [ "$TEST" = t2 ]; then
  ray stop --force  # the workers exit when the head goes away
fi
