#!/usr/bin/env bash
# kdatp-prof driver: train-only GLM-5.3-Flash arms on the r47 trainer layout
# (8 nodes), with the torch profiler on the arms that ask for it. Each pod of
# the kdatp-prof-<stamp> PyTorchJob runs it from the scratch mount, so the
# image code does not change. It mirrors the t2 branch of ../kdatp-run.sh:
# same env, same worker branch, same ray head, the in-image gen_arm_configs.py
# 'kdatp' arm patched by patch_prof_arm.py.
#
# Per-job settings come from harness/job.env (KEY=VALUE lines, bash syntax),
# which every pod sources before the defaults below. The arms come from
# harness/$PROF_ARMS_FILE (a YAML list, README.md). Without an arms file, the
# job runs one profiled arm, 'kdatp-prof', on PROF_GROUPS with DP 2. The arms
# run in order in one ray cluster; a failed arm does not stop the next one.
# Each arm trains whole groups of the r45 summary from data/<dir>, built once
# by the in-image build_rollout_data.py.
#
# Writes only under $KDATP_DIR/prof/$KDATP_STAMP (default
# /mnt/scratch-s3files-rw/guparpit/kdatp/prof/<stamp>): driver.log, arms/,
# <arm>/trainer-0.log, <arm>/rc, <arm>/tb/ (the traces of a profiled arm),
# a2a/ (PROF_A2A_BENCH=1), results.json; and data/<dir> when it is missing.
# It links the r43 iter_0000039 DCP into $KDATP_DIR/checkpoints for each arm
# and never saves.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wait_file() {  # the mount imports the uploaded harness files within minutes, in no fixed order
  local i
  for i in $(seq 120); do
    [ -f "$1" ] && return 0
    echo "[kdatp-prof] waiting for $1"
    sleep 5
  done
  echo "[kdatp-prof] missing $1"
  return 1
}
# job.env MUST exist (an empty file keeps the defaults), so that a late import
# never runs the job on the defaults. set -a exports each value to ray start,
# so the ray workers get it too.
wait_file "$HERE/job.env" || exit 1
set -a
. "$HERE/job.env"
set +a

KD=${KDATP_DIR:-/mnt/scratch-s3files-rw/guparpit/kdatp}
STAMP=${KDATP_STAMP:?the manifest sets one KDATP_STAMP for all pods}
REPO=${AGISLIME_DIR:-/root/miles}
KDATP=$REPO/examples/arena/harbor-rl-glm53-flash/kdatp
LAUNCHER=$REPO/scripts/run_arena_harbor.py
R43=/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-adebt-v3-r43
R43_ROUTING=/mnt/scratch-s3files-rw/guparpit/routing/rl-glm53f-adebt-v3-r43
R45_SUMMARY=/mnt/scratch-s3files-rw/guparpit/debug/rl-glm53f-adebt-v3-r45/sample_summary/rollout_43.jsonl
# Comma-separated group_index values of the r45 summary; the default of an arm without data.groups.
PROF_GROUPS=${PROF_GROUPS:-239,258}
# A YAML list in the harness dir (README.md); empty: one profiled arm on PROF_GROUPS.
PROF_ARMS_FILE=${PROF_ARMS_FILE:-}
PROF_A2A_BENCH=${PROF_A2A_BENCH:-0}
# Space-separated kernel cache tarballs (kdatp-run.sh warm) to unpack before the start.
KDATP_KCACHE=${KDATP_KCACHE:-}
# The default of an arm without timeout (s).
ARM_TIMEOUT=${ARM_TIMEOUT:-12600}
# r47 flip A, the default of an arm without skip_actor_forward_only. The T2 rows carry
# rollout_log_probs and no actor log_probs, and global_batch_size 64 == rollout_batch_size 8
# x n_samples_per_prompt 8.
PROF_SKIP_ACTOR_FORWARD_ONLY=${PROF_SKIP_ACTOR_FORWARD_ONLY:-1}
RUN=$KD/prof/$STAMP

export ARENA_CHECKPOINTS_DIR=$KD/checkpoints ARENA_DATA_DIR=$KD/run
export MILES_LOG_PEAK_MEMORY=1
# The ray driver folds log lines that differ only in numbers, so it hides the per-rank
# [peak-memory] lines. ray start and the job driver inherit this.
export RAY_DEDUP_LOGS=0
# Kernel JIT caches on local disk (r45 pattern). All arms of a job share them.
export KCACHE=/tmp/kernel_cache
export TILELANG_CACHE_DIR=$KCACHE/tilelang TRITON_CACHE_DIR=$KCACHE/triton TORCHINDUCTOR_CACHE_DIR=$KCACHE/inductor
mkdir -p "$TILELANG_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
for tarball in $KDATP_KCACHE; do
  if tar -C "${KCACHE%/*}" -xf "$tarball"; then
    echo "[kdatp-prof] kernel cache unpacked: $tarball"
  else
    echo "[kdatp-prof] kernel cache FAILED: $tarball"
  fi
done
ulimit -n 1000000
# Kineto INFO lines (buffer size, stop reasons) in the actor stderr; ray workers inherit the raylet env.
export KINETO_LOG_LEVEL=${KINETO_LOG_LEVEL:-1}
# Kineto keeps at most 1 + MB/4 CUPTI buffers of 4 MiB per rank. The default 128 MiB
# (33 buffers) ran out in step 0 on all 64 ranks (runs 20260929a, 20260929ab). Kineto
# reads the KINETO_CONFIG file in each prepareTrace (README.md). The file is pod-local.
PROF_KINETO_BUFFER_MB=${PROF_KINETO_BUFFER_MB:-4096}
printf 'ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB=%s\n' "$PROF_KINETO_BUFFER_MB" > /tmp/kineto.conf
export KINETO_CONFIG=/tmp/kineto.conf

# All-to-all bandwidth (a2a_bench.py) on all pods before ray starts. Static rendezvous
# on replica 0 with node rank REPLICA_IDX, so ranks 16k..16k+15 are on nodes 2k and 2k+1,
# as the r47 stage expert groups. Port 23456 is the PyTorchJob rendezvous port
# (PET_RDZV_ENDPOINT); ray and the launcher do not use it. The workers wait at most
# 600 s for the ray head (run_arena_harbor.py _wait_for_head_port), so the bench stops
# by 420 s on every pod, and a failed bench loses only a2a/.
A2A_RC=off
if [ "$PROF_A2A_BENCH" = 1 ]; then
  mkdir -p "$RUN/a2a"
  if wait_file "$HERE/a2a_bench.py"; then
    timeout -k 30 420 python3 -m torch.distributed.run --nnodes "${REPLICA:-8}" --nproc-per-node 8 \
      --node-rank "${REPLICA_IDX:-0}" --rdzv-backend static --rdzv-endpoint "${HOSTNAME%-*}-0:23456" \
      --rdzv-conf timeout=240 --max-restarts 0 "$HERE/a2a_bench.py" --out "$RUN/a2a" \
      > "$RUN/a2a/node-${REPLICA_IDX:-0}.log" 2>&1
    A2A_RC=$?
  fi
  echo "[kdatp-prof] a2a bench rc=$A2A_RC"
fi

if [ "${REPLICA_IDX:-0}" != "0" ]; then
  # Workers join the head's ray cluster and block until the head stops it.
  exec python3 "$LAUNCHER" worker
fi

mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
log() { echo "[kdatp-prof $(date -u +%FT%TZ)] $*"; }
log "stamp=$STAMP image_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) arms_file=${PROF_ARMS_FILE:-none} a2a_rc=$A2A_RC"
log "job.env: $(grep -v '^ *#' "$HERE/job.env" | tr '\n' ' ')"
log "kcache=$KDATP_KCACHE tilelang_entries=$(ls "$TILELANG_CACHE_DIR" 2>/dev/null | wc -l) triton_entries=$(ls "$TRITON_CACHE_DIR" 2>/dev/null | wc -l)"
# The in-image copies; harness/ holds the rest.
cp "$KDATP"/gen_arm_configs.py "$KDATP"/build_rollout_data.py "$KDATP"/parse_logs.py "$RUN/" 2>/dev/null

export REPLICA=${REPLICA:-8} REPLICA_TRAINER=${REPLICA_TRAINER:-8}
HEAD=$(hostname)
export MILES_SCRIPT_EXTERNAL_RAY=1
# Before any data build: the workers wait at most 600 s for the head.
ray start --head --node-ip-address "$HEAD" --num-gpus 8 --disable-usage-stats

python3 "$KDATP/gen_arm_configs.py" t2 --data-dir "$KD/data" --out "$RUN/arms" --arms kdatp ||
  { log "arm configs FAILED"; ray stop --force; exit 1; }
arms_flag=()
if [ -n "$PROF_ARMS_FILE" ]; then
  wait_file "$HERE/$PROF_ARMS_FILE" || { log "no arms file"; ray stop --force; exit 1; }
  arms_flag=(--arms-file "$HERE/$PROF_ARMS_FILE")
fi
skip_flag=()
[ "$PROF_SKIP_ACTOR_FORWARD_ONLY" = 1 ] && skip_flag=(--skip-actor-forward-only)
wait_file "$HERE/patch_prof_arm.py" &&
  python3 "$HERE/patch_prof_arm.py" "$RUN/arms/kdatp.yaml" "$RUN" --data-root "$KD/data" "${arms_flag[@]}" \
    --groups "$PROF_GROUPS" --timeout "$ARM_TIMEOUT" "${skip_flag[@]}" ||
  { log "arm plan FAILED"; ray stop --force; exit 1; }

build_data() {  # dir groups dp_size
  local out=$KD/data/$1 i
  # CPU and disk only (data/prof-239-258: 112 s in run 20260929ab). Two jobs can
  # need the same dir: the first one builds it under a lock dir, the other one waits.
  if [ ! -f "$out/manifest.json" ] && mkdir "$out.lock" 2>/dev/null; then
    log "building data $1 (groups $2, dp $3)"
    python3 "$KDATP/build_rollout_data.py" --summary "$R45_SUMMARY" --groups "$2" --routing-dir "$R43_ROUTING" \
      --out-dir "$out" --rollout-ids 40,41,42,43 --dp-size "$3" > "$RUN/data-$1.log" 2>&1
    rmdir "$out.lock"
  fi
  for i in $(seq 40); do
    [ -f "$out/manifest.json" ] && break
    [ -d "$out.lock" ] || break
    log "data $1: waiting for the build of another job ($out.lock)"
    sleep 30
  done
  # The manifest comes last, so it marks a complete dir. Its groups and row count MUST fit the arm.
  [ -f "$out/manifest.json" ] && python3 - "$out/manifest.json" "$2" "$3" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
print({k: manifest[k] for k in ("groups", "rows", "dp_pads", "episodes", "tokens", "max_length")}, flush=True)
groups, dp_size = [int(g) for g in sys.argv[2].split(",")], int(sys.argv[3])
sys.exit(manifest["groups"] != groups or manifest["rows"] % dp_size != 0)
PY
}

seed_ckpt() {  # experiment name: link r43 iter_0000039 file by file (kdatp-run.sh seed_t2_ckpt)
  local dst=$ARENA_CHECKPOINTS_DIR/slime_experiments/$1 it=iter_0000039 f
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

mapfile -t PLAN < "$RUN/arms/plan.tsv"
for row in "${PLAN[@]}"; do
  IFS=$'\t' read -r name groups dp data limit profile <<< "$row"
  if ! build_data "$data" "$groups" "$dp"; then
    log "arm $name: data $data FAILED (see data-$data.log)"
  elif ! seed_ckpt "kdatp-prof-$STAMP-$name"; then
    log "arm $name: seed FAILED"
  else
    mkdir -p "$RUN/$name"
    log "arm $name: start (data $data, timeout $limit s)"
    EXPERIMENT_NAME=kdatp-prof-$STAMP-$name PROJECT_NAME=kdatp-prof CFG_NAME=$RUN/arms/$name.yaml \
      timeout --signal=TERM "$limit" python3 "$LAUNCHER" train > "$RUN/$name/trainer-0.log" 2>&1
    echo $? > "$RUN/$name/rc"
    log "arm $name: rc=$(cat "$RUN/$name/rc")"
    if [ "$profile" = 1 ]; then
      # Success: 64 ranks at the buffer size, 0 Exceeded, 0 stopped early.
      # Count only the torch warning "Device profiling activity collection was stopped
      # early at step N". The Kineto summary "GPU stopped early? = 0" is on every rank
      # of a clean run (20260929c), and it also read 0 on the cut run 20260929ab.
      tlog=$RUN/$name/trainer-0.log
      log "arm $name kineto: $(grep -c "Max GPU buffer size: ${PROF_KINETO_BUFFER_MB}MB" "$tlog")" \
        "ranks at ${PROF_KINETO_BUFFER_MB}MB, $(grep -c 'Exceeded max GPU buffer count' "$tlog") Exceeded," \
        "$(grep -c 'activity collection was stopped early' "$tlog") stopped early"
      log "arm $name traces: $(ls "$RUN/$name/tb" 2>/dev/null | wc -l) files"
      ls -la "$RUN/$name/tb" 2>/dev/null | tail -n +2 | awk '{print "[kdatp-prof trace] " $5 " " $9}'
    fi
    stop_ray_jobs
  fi
  python3 "$KDATP/parse_logs.py" "$RUN" > "$RUN/results.json" 2> "$RUN/parse_logs.err"
done
log "done: $RUN/results.json"
ray stop --force  # the workers exit when the head goes away
