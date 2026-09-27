#!/usr/bin/env bash
# kdash test driver. Each pod of a kdash-* PyTorchJob runs it (README.md).
#
#   kdash-run.sh t1   # 1 node: GPU unit tests, T1 parity, then the T1c 5-layer train-only arms
#   kdash-run.sh t2   # 8 nodes: the T2 train-only arms on the r45 layout
#
# Writes only under $KDASH_DIR (default /mnt/scratch-s3files-rw/guparpit/kdash).
# Reads the base DCP, the r43 iter_0000039 DCP and HF export, and the train-only
# data of the kdatp tests ($KDASH_DATA, read only). When that data is missing, t1
# builds it under $KDASH_DIR/data from the r43 staged token files and the r45
# sample summary. T2 links the r43 DCP files into its own checkpoint dirs and
# never saves (save_interval 100000).
set -uo pipefail

TEST=${1:?usage: kdash-run.sh t1|t2}
KD=${KDASH_DIR:-/mnt/scratch-s3files-rw/guparpit/kdash}
STAMP=${KDASH_STAMP:?the manifest sets one KDASH_STAMP for all pods}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
LAUNCHER=$REPO/scripts/run_arena_harbor.py
HF=/mnt/scratch-fast-1a-rw/durable/model-artifacts/external/zai-org/GLM-5.3-Flash-BF16
BASE_DCP=/mnt/scratch-s3files-rw/guparpit/checkpoints/dcp/glm5.3-flash_torch_dist
R43=/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-adebt-v3-r43
R43_ROUTING=/mnt/scratch-s3files-rw/guparpit/routing/rl-glm53f-adebt-v3-r43
R45_SUMMARY=/mnt/scratch-s3files-rw/guparpit/debug/rl-glm53f-adebt-v3-r45/sample_summary/rollout_43.jsonl
KDATP_DATA=/mnt/scratch-s3files-rw/guparpit/kdatp/data
T1C_GROUPS=${T1C_GROUPS:-243}
T2_GROUPS=${T2_GROUPS:-243,284,307,258,239,306,276,309}
T1C_ARMS=${T1C_ARMS:-shared selective none}
T2_ARMS=${T2_ARMS:-shared block10}
ARM_TIMEOUT=${ARM_TIMEOUT:-7200}
UNIT_TIMEOUT=${UNIT_TIMEOUT:-1800}
RUN=$KD/$TEST/$STAMP

# The kdatp data holds the same rows as a fresh build (same groups, same r43 files). Read it in place.
if [ -z "${KDASH_DATA:-}" ]; then
  if [ -f "$KDATP_DATA/t1c/manifest.json" ] && [ -f "$KDATP_DATA/t2/manifest.json" ] &&
    [ -f "$KDATP_DATA/hf/GLM-5.3-Flash-5layer/config.json" ]; then
    KDASH_DATA=$KDATP_DATA
  else
    KDASH_DATA=$KD/data
  fi
fi
export ARENA_CHECKPOINTS_DIR=$KD/checkpoints ARENA_DATA_DIR=$KD/run
export MILES_LOG_PEAK_MEMORY=1
# The ray driver folds log lines that differ only in numbers ("[repeated Nx]"), so it hides
# the per-rank [peak-memory] lines (r11/trainer-pytorchjob.yaml saw the same). ray start and
# the job driver inherit this.
export RAY_DEDUP_LOGS=0
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
log() { echo "[kdash $(date -u +%FT%TZ)] $*"; }
log "test=$TEST stamp=$STAMP image_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) data=$KDASH_DATA"
cp "$HERE"/*.py "$HERE"/kdash-run.sh "$RUN/" 2>/dev/null

build_data() {  # name groups rollout_ids dp_size
  local out=$KDASH_DATA/$1
  if [ -f "$out/manifest.json" ]; then
    log "data $1 exists: $out/manifest.json"
    return 0
  fi
  case "$out" in
  "$KD"/*) ;;
  *)
    log "data $1: missing in $KDASH_DATA, and kdash builds only under $KD"
    return 1
    ;;
  esac
  log "building data $1 (groups $2)"
  python3 "$HERE/build_rollout_data.py" --summary "$R45_SUMMARY" --groups "$2" --routing-dir "$R43_ROUTING" \
    --out-dir "$out" --rollout-ids "$3" --dp-size "$4" > "$RUN/data-$1.log" 2>&1
}

run_unit() {  # name command...: one GPU unit test, its log and its exit code under $RUN/unit
  local name=$1
  shift
  mkdir -p "$RUN/unit"
  log "unit $name: start"
  (cd "$REPO" && PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" timeout "$UNIT_TIMEOUT" "$@") > "$RUN/unit/$name.log" 2>&1
  echo $? > "$RUN/unit/$name.rc"
  log "unit $name: rc=$(cat "$RUN/unit/$name.rc")"
}

run_arm() {  # test arm
  local dir=$RUN/$2
  mkdir -p "$dir"
  log "arm $2: start"
  EXPERIMENT_NAME=kdash-$1-$2-$STAMP PROJECT_NAME=kdash-$1 CFG_NAME=$RUN/arms/$2.yaml \
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
  case "$1" in kdash-*) ;; *)
    log "refusing to seed $1"
    return 1
    ;;
  esac
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
  [ -f "$KDASH_DATA/hf/GLM-5.3-Flash-5layer/config.json" ] ||
    python3 "$HERE/make_slice_hf.py" --src "$HF" --out "$KD/data/hf/GLM-5.3-Flash-5layer" --layers 5
  build_data t1c "$T1C_GROUPS" 0,1 1 || log "data t1c: FAILED (see data-t1c.log)"
  # The T2 data build is CPU and disk only; it runs while the GPUs do T1.
  # Rollouts 40..43 train; the loop prefetches the next id, so 44 must exist too.
  build_data t2 "$T2_GROUPS" 40,41,42,43,44 2 &
  T2_DATA_PID=$!

  # The upstream delta-rule tests (PR #3609 stack, ported) and the GLM-5.3 checkpoint and weight-sync tests.
  run_unit head_sharded_tp8 python3 -m torch.distributed.run --nproc-per-node 8 tests/fast-gpu/test_delta_rule_head_sharded.py
  run_unit head_sharded_tp2 python3 -m torch.distributed.run --nproc-per-node 2 tests/fast-gpu/test_delta_rule_head_sharded.py
  run_unit linear_attn_layer python3 -m torch.distributed.run --nproc-per-node 2 tests/fast-gpu/test_linear_attn_layer.py
  run_unit conv_chunking python3 -m pytest -q -p no:cacheprovider tests/fast-gpu/test_delta_rule_conv_chunking.py
  run_unit cpu_layout python3 -m pytest -q -p no:cacheprovider \
    tests/fast/backends/megatron_utils/test_glm5_next_kda_shared_layer.py tests/fast/models

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

  python3 "$HERE/gen_arm_configs.py" t1c --data-dir "$KDASH_DATA" --out "$RUN/arms" --arms "$T1C_ARMS"
  for arm in $T1C_ARMS; do
    run_arm t1c "$arm"
  done
  wait "$T2_DATA_PID"
  log "data t2: rc=$?"
  ;;
t2)
  export REPLICA=${REPLICA:-8} REPLICA_TRAINER=${REPLICA_TRAINER:-8}
  test -f "$KDASH_DATA/t2/manifest.json" || { log "no T2 data in $KDASH_DATA: run t1 first"; exit 1; }
  HEAD=$(hostname)
  export MILES_SCRIPT_EXTERNAL_RAY=1
  ray start --head --node-ip-address "$HEAD" --num-gpus 8 --disable-usage-stats
  python3 "$HERE/gen_arm_configs.py" t2 --data-dir "$KDASH_DATA" --out "$RUN/arms" --arms "$T2_ARMS" ||
    { log "arm configs FAILED"; ray stop --force; exit 1; }  # the workers exit when the head goes away
  for arm in $T2_ARMS; do
    if seed_t2_ckpt "kdash-t2-$arm-$STAMP"; then
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
