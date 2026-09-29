#!/usr/bin/env bash
# kdatp-prof driver: one profiled GLM-5.3-Flash train step on the r47 trainer
# layout (8 nodes). Each pod of the kdatp-prof-<stamp> PyTorchJob runs it from
# the scratch mount, so the image code does not change. It mirrors the t2
# branch of ../kdatp-run.sh: same env, same worker branch, same ray head, the
# in-image gen_arm_configs.py 'kdatp' arm patched by patch_prof_arm.py.
#
# The step is a slice of the T2 rows: whole groups of the same r45 summary
# (PROF_GROUPS, default 239,258: 66 rows, 4.72M tokens, mean row 71.5K tokens
# as T2), built once by the in-image build_rollout_data.py with DP 2 padding.
# Kineto holds at most 128 MiB of GPU activity records, and torch 2.13 skips
# the record phase when the warmup step alone fills it: a full T2 step did
# (run 20260929a, "Device profiling activity collection was stopped early").
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
R43_ROUTING=/mnt/scratch-s3files-rw/guparpit/routing/rl-glm53f-adebt-v3-r43
R45_SUMMARY=/mnt/scratch-s3files-rw/guparpit/debug/rl-glm53f-adebt-v3-r45/sample_summary/rollout_43.jsonl
ARM=kdatp-prof
# Comma-separated group_index values of the r45 summary; the data dir is data/prof-<groups>.
PROF_GROUPS=${PROF_GROUPS:-239,258}
PROF_DATA=$KD/data/prof-${PROF_GROUPS//,/-}
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
# Kineto INFO lines (buffer size, stop reasons) in the actor stderr; ray workers inherit the raylet env.
export KINETO_LOG_LEVEL=${KINETO_LOG_LEVEL:-1}

if [ "${REPLICA_IDX:-0}" != "0" ]; then
  # Workers join the head's ray cluster and block until the head stops it.
  exec python3 "$LAUNCHER" worker
fi

mkdir -p "$RUN" "$TB"
exec > >(tee -a "$RUN/driver.log") 2>&1
log() { echo "[kdatp-prof $(date -u +%FT%TZ)] $*"; }
log "stamp=$STAMP image_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) skip_actor_forward_only=$PROF_SKIP_ACTOR_FORWARD_ONLY groups=$PROF_GROUPS"
log "kcache=$KDATP_KCACHE tilelang_entries=$(ls "$TILELANG_CACHE_DIR" 2>/dev/null | wc -l) triton_entries=$(ls "$TRITON_CACHE_DIR" 2>/dev/null | wc -l)"
cp "$HERE"/kdatp-prof-run.sh "$HERE"/patch_prof_arm.py "$KDATP"/gen_arm_configs.py "$KDATP"/parse_logs.py "$RUN/" 2>/dev/null

export REPLICA=${REPLICA:-8} REPLICA_TRAINER=${REPLICA_TRAINER:-8}
if [ ! -f "$PROF_DATA/manifest.json" ]; then
  # CPU and disk only (T1c, one group of 43 rows: 94 s). Rollouts 40..42 train; the loop prefetches 43.
  log "building data $PROF_DATA (groups $PROF_GROUPS)"
  python3 "$KDATP/build_rollout_data.py" --summary "$R45_SUMMARY" --groups "$PROF_GROUPS" --routing-dir "$R43_ROUTING" \
    --out-dir "$PROF_DATA" --rollout-ids 40,41,42,43 --dp-size 2 > "$RUN/data-build.log" 2>&1 ||
    { log "data build FAILED (see data-build.log)"; exit 1; }
fi
log "data: $(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print({k: m[k] for k in ('groups','rows','dp_pads','episodes','tokens','max_length')})" "$PROF_DATA/manifest.json")"
NUM_GROUPS=$(echo "$PROF_GROUPS" | tr ',' '\n' | grep -c .)
HEAD=$(hostname)
export MILES_SCRIPT_EXTERNAL_RAY=1
ray start --head --node-ip-address "$HEAD" --num-gpus 8 --disable-usage-stats

python3 "$KDATP/gen_arm_configs.py" t2 --data-dir "$KD/data" --out "$RUN/arms" --arms kdatp ||
  { log "arm configs FAILED"; ray stop --force; exit 1; }
skip_flag=()
[ "$PROF_SKIP_ACTOR_FORWARD_ONLY" = 1 ] && skip_flag=(--skip-actor-forward-only)
# One optimizer step: global_batch_size = groups x n_samples_per_prompt (8), rollout_batch_size = groups.
python3 "$HERE/patch_prof_arm.py" "$RUN/arms/kdatp.yaml" "$RUN/arms/$ARM.yaml" \
  --tensorboard-dir "$TB" --sample-summary-dir "$RUN/$ARM/sample_summary" "${skip_flag[@]}" \
  --set "load_debug_rollout_data=$PROF_DATA/rollout_{rollout_id}.pt" \
  --set "rollout_batch_size=$NUM_GROUPS" --set "global_batch_size=$((NUM_GROUPS * 8))" ||
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
