#!/usr/bin/env bash
# kdatp-prof driver: one profiled GLM-5.3-Flash train step on the r47 trainer
# layout (8 nodes). Each pod of the kdatp-prof-<stamp> PyTorchJob runs it from
# the scratch mount, so the image code does not change. It mirrors the t2
# branch of ../kdatp-run.sh: same env, same worker branch, same ray head, the
# in-image gen_arm_configs.py 'kdatp' arm patched by patch_prof_arm.py.
#
# Writes only under $KDATP_DIR/prof/$KDATP_STAMP (default
# /mnt/scratch-s3files-rw/guparpit/kdatp/prof/<stamp>): driver.log, arms/,
# kdatp-prof/trainer-0.log, tb/ (the traces), results.json. It links the r43
# iter_0000039 DCP into $KDATP_DIR/checkpoints and never saves.
set -uo pipefail

KD=${KDATP_DIR:-/mnt/scratch-s3files-rw/guparpit/kdatp}
STAMP=${KDATP_STAMP:?the manifest sets one KDATP_STAMP for all pods}
REPO=${AGISLIME_DIR:-/root/miles}
KDATP=$REPO/examples/arena/harbor-rl-glm53-flash/kdatp
LAUNCHER=$REPO/scripts/run_arena_harbor.py
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R43=/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-adebt-v3-r43
ARM=kdatp-prof
# Space-separated kernel cache tarballs (kdatp-run.sh warm) to unpack before the start.
KDATP_KCACHE=${KDATP_KCACHE:-}
ARM_TIMEOUT=${ARM_TIMEOUT:-12600}
# r47 flip A. The T2 rows carry rollout_log_probs and no actor log_probs, and
# global_batch_size 64 == rollout_batch_size 8 x n_samples_per_prompt 8.
PROF_SKIP_ACTOR_FORWARD_ONLY=${PROF_SKIP_ACTOR_FORWARD_ONLY:-1}
RUN=$KD/prof/$STAMP
TB=$RUN/tb

export ARENA_CHECKPOINTS_DIR=$KD/checkpoints ARENA_DATA_DIR=$KD/run
export MILES_LOG_PEAK_MEMORY=1
# The ray driver folds log lines that differ only in numbers, so it hides the per-rank
# [peak-memory] lines. ray start and the job driver inherit this.
export RAY_DEDUP_LOGS=0
# Kernel JIT caches on local disk (r45 pattern).
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

if [ "${REPLICA_IDX:-0}" != "0" ]; then
  # Workers join the head's ray cluster and block until the head stops it.
  exec python3 "$LAUNCHER" worker
fi

mkdir -p "$RUN" "$TB"
exec > >(tee -a "$RUN/driver.log") 2>&1
log() { echo "[kdatp-prof $(date -u +%FT%TZ)] $*"; }
log "stamp=$STAMP image_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) skip_actor_forward_only=$PROF_SKIP_ACTOR_FORWARD_ONLY"
log "kcache=$KDATP_KCACHE tilelang_entries=$(ls "$TILELANG_CACHE_DIR" 2>/dev/null | wc -l) triton_entries=$(ls "$TRITON_CACHE_DIR" 2>/dev/null | wc -l)"
cp "$HERE"/kdatp-prof-run.sh "$HERE"/patch_prof_arm.py "$KDATP"/gen_arm_configs.py "$KDATP"/parse_logs.py "$RUN/" 2>/dev/null

export REPLICA=${REPLICA:-8} REPLICA_TRAINER=${REPLICA_TRAINER:-8}
test -f "$KD/data/t2/manifest.json" || { log "no T2 data: $KD/data/t2/manifest.json"; exit 1; }
HEAD=$(hostname)
export MILES_SCRIPT_EXTERNAL_RAY=1
ray start --head --node-ip-address "$HEAD" --num-gpus 8 --disable-usage-stats

python3 "$KDATP/gen_arm_configs.py" t2 --data-dir "$KD/data" --out "$RUN/arms" --arms kdatp ||
  { log "arm configs FAILED"; ray stop --force; exit 1; }
skip_flag=()
[ "$PROF_SKIP_ACTOR_FORWARD_ONLY" = 1 ] && skip_flag=(--skip-actor-forward-only)
python3 "$HERE/patch_prof_arm.py" "$RUN/arms/kdatp.yaml" "$RUN/arms/$ARM.yaml" \
  --tensorboard-dir "$TB" --sample-summary-dir "$RUN/$ARM/sample_summary" "${skip_flag[@]}" ||
  { log "arm patch FAILED"; ray stop --force; exit 1; }

# Link r43 iter_0000039 file by file (kdatp-run.sh seed_t2_ckpt).
EXP=kdatp-prof-$STAMP
dst=$ARENA_CHECKPOINTS_DIR/slime_experiments/$EXP it=iter_0000039
if [ -e "$dst" ]; then
  log "refusing to reuse $dst"; ray stop --force; exit 1
fi
mkdir -p "$dst/$it"
for f in "$R43/$it/.metadata" "$R43/$it/metadata.json" "$R43/$it"/__*_0.distcp; do
  ln -s "$f" "$dst/$it/${f##*/}"
done
printf '{"rollout_id": 39}' > "$dst/$it/slime_extra_state.json"  # no wandb_run_id: never the r43 W&B run
printf 39 > "$dst/latest_checkpointed_iteration.txt"
test "$(ls -A "$dst/$it" | wc -l)" = "$(ls -A "$R43/$it" | wc -l)" || { log "seed FAILED"; ray stop --force; exit 1; }

mkdir -p "$RUN/$ARM"
log "arm $ARM: start"
EXPERIMENT_NAME=$EXP PROJECT_NAME=kdatp-prof CFG_NAME=$RUN/arms/$ARM.yaml \
  timeout --signal=TERM "$ARM_TIMEOUT" python3 "$LAUNCHER" train > "$RUN/$ARM/trainer-0.log" 2>&1
echo $? > "$RUN/$ARM/rc"
log "arm $ARM: rc=$(cat "$RUN/$ARM/rc")"
log "traces: $(ls "$TB" 2>/dev/null | wc -l) files"
ls -la "$TB" 2>/dev/null | tail -n +2 | awk '{print "[kdatp-prof trace] " $5 " " $9}'

# A timed-out launcher leaves its ray job running; stop it before the ray cluster goes down.
python3 - <<'PY'
from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://127.0.0.1:8265")
for job in client.list_jobs():
    if str(job.status) in ("PENDING", "RUNNING", "JobStatus.PENDING", "JobStatus.RUNNING"):
        print("stopping ray job", job.submission_id, flush=True)
        client.stop_job(job.submission_id)
PY
sleep 60
python3 "$KDATP/parse_logs.py" "$RUN" > "$RUN/results.json" 2> "$RUN/parse_logs.err"
log "done: $RUN/results.json"
ray stop --force  # the workers exit when the head goes away
