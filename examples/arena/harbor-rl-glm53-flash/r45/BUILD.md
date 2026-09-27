# r45: r43 continued from its latest checkpoint on the ADR-0015 images

Prepared 2026-09-27. Not launched. r43 (`rl-glm53f43-qztzt`) was still
Running after its rollout 39 save when this record was written. Generated
with:

```
.venv/bin/python gen-workflow.py 45 --base r43 --template guparpit-miles-deployer-v10 \
  --experiment-name rl-glm53f-adebt-v3-r45 \
  --gym-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a \
  --trainer-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r14-20260927a \
  --param 'agent-kwargs={}' --param 'publish-jobs-dir='
```

| Delta vs r43 | r43 | r45 |
| --- | --- | --- |
| Start point | base DCP (`ref_load`) | r43 `iter_0000039`, seeded into the r45 dir |
| Rollout ids | 0..39 | 40.. |
| Data position | row 0 | r43 offset and epoch at 39 |
| W&B run | `2z0599lb` | a new run in project `rl-glm53f-adebt-v3` |
| `rollout_max_response_len` | 32768 (gym copy `ARENA_MAX_TOKENS` 32768) | 16384, sent in the task message |
| Window | gym env `ARENA_ROLLOUT_CONTEXT_LIMIT` 131072 | task message `max_seq_len` 131072 |
| Output cap near the window end | fixed 32768 | clamped to the room left in the window |
| Vulcan `max_compactions` | 2 (`compaction-max` 2) | 4 (`agent-kwargs` `{}`, Vulcan default) |
| Template | `guparpit-miles-deployer-v9` | `guparpit-miles-deployer-v10` |
| Trainer image | `miles-glm53-r13-20260925a` | `miles-glm53-r14-20260927a` (miles `e0987aed6`) |
| Gym image | `gym-glm53-adr71-20260925a` | `gym-glm53-adr72-20260927a` (AREnATasks mainline `538bc63`, tree `ab266688`) |
| Trajectory publication | off (v9 has no publish env) | off (`publish-jobs-dir` `''`) |
| `step-cut-on-fail` argument | `false` (v9 declares none) | removed (`gen-workflow.py` drops it) |
| Names | `rl-glm53f-adebt-v3-r43` | `rl-glm53f-adebt-v3-r45` |

Same as r43: dataset `20260923-v3-locked-oracle/manifest-le5.jsonl`,
`agent-timeout-multiplier` 4, `ack-wait` 86000, trainer deadline 90000,
radix on, overlap off, `save_interval` 10, `excluded-nodes`, W&B project
`rl-glm53f-adebt-v3`. The images and template v10 are in `r44/BUILD.md`.

Why: the same overflow evidence as r44 (`r44/BUILD.md`). r45 keeps the
r43 policy, data position, and RNG state. The curve has these breaks
between rollout 39 and rollout 40:

- the new trainer and gym images
- the 16384 output cap
- the output cap clamp to the room left in the window
- Vulcan `max_compactions` 2 -> 4
- a new W&B run, with `rollout/step` from 40
- Adam moments restart at zero (as at every earlier resume)

NEVER fit one trend across rollout 39 and rollout 40. A resume boundary
changes the episode length and the reward level.

## How miles resumes

- `scripts/run_arena_harbor.py` appends `--load` and `--save` =
  `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/<EXPERIMENT_NAME>`
  after the config flags. argparse keeps the last value, so a `load` key in
  `miles-config.yaml` has no effect. r43 ran with
  `--load` = `--save` = `.../slime_experiments/rl-glm53f-adebt-v3-r43`
  (trainer-worker-0 `/proc/<pid>/cmdline`, 2026-09-27).
- When `--load` holds `latest_checkpointed_iteration.txt` = N, Megatron
  loads `iter_%07d` of N. The actor returns N and rollout ids start at
  N + 1. The rollout manager then loads
  `rollout/arena_data_source_state_N.pt` (per-gym offset and epoch, prompt
  rewards). The RNG state loads (`no_load_rng` unset).
- `no_load_optim: true` skips the optimizer state. It MUST stay true: r43
  sets `no_save_optim: true`, so its checkpoints hold no optimizer state.
  `lr` is constant (1.5e-6), so the scheduler state has no effect.
- When `--load` has no tracker file, miles sets `finetune`,
  `no_load_optim`, `no_load_rng`, `load = ref_load`, and
  `start_rollout_id = 0`. That path restarts rollout ids and the data
  position, so it is not a continuation.
- `train_async_arena.py` reads `iter_N/slime_extra_state.json` from the
  load dir. It resumes the `wandb_run_id` of that file when no
  `WANDB_RUN_ID` is set, and v10 sets none. The `rollout_id` of that file
  goes to the log only. The r43 sidecar holds
  `{"rollout_id": 39, "wandb_run_id": "2z0599lb"}`. The r45 seed writes
  its own sidecar with `rollout_id` only, so r45 opens a new W&B run.

## Seed the r45 checkpoint dir

The r17 warm start is the precedent (`r17/BUILD.md`). r45 links each r43
file one by one, so the r45 iteration dir can hold its own sidecar. It also
copies the data source state, so the data position continues.

The r43 `iter_0000039` holds 64 `__*_0.distcp` shards, `.metadata`,
`metadata.json`, and `slime_extra_state.json`. The step-39 save was
complete at 04:31:37Z: sidecar, `rollout/arena_data_source_state_39.pt`,
and 671 files in `hf/rollout_39`. The next r43 save is step 49, about 10 h
later.

Run the commands in a pod that mounts `/mnt/scratch-s3files-rw`. The Stop
removes the r43 pods, so use r43 trainer-worker-0 before the Stop. If a
check fails, stop and update this record.

```
R43=/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-adebt-v3-r43
R45=/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-adebt-v3-r45
N=$(cat $R43/latest_checkpointed_iteration.txt); I=$(printf 'iter_%07d' "$N")
test "$N" = 39
test ! -e $R45
test -f $R43/$I/slime_extra_state.json && test -f $R43/rollout/arena_data_source_state_$N.pt
mkdir -p $R45/$I $R45/rollout
for f in $R43/$I/.metadata $R43/$I/metadata.json $R43/$I/__*_0.distcp; do
  ln -s ../../rl-glm53f-adebt-v3-r43/$I/${f##*/} $R45/$I/${f##*/}
done
printf '{"rollout_id": %d}' "$N" > $R45/$I/slime_extra_state.json
cp $R43/rollout/arena_data_source_state_$N.pt $R45/rollout/
printf '%s' "$N" > $R45/latest_checkpointed_iteration.txt
# Checks: the same entry count (67), every link resolves, no r43 run id.
test "$(ls -A $R45/$I | wc -l)" = "$(ls -A $R43/$I | wc -l)"
for f in $R45/$I/.metadata $R45/$I/metadata.json $R45/$I/__*_0.distcp; do test -f $f; done
cat $R45/$I/slime_extra_state.json $R45/latest_checkpointed_iteration.txt
```

- The `for` list names `.metadata` explicitly. A `*` glob skips dot files,
  and Megatron cannot load the dir without `.metadata`.
- The iteration dir MUST be a real dir. A symlink to the r43 dir carries the
  r43 sidecar, and r45 then continues W&B run `2z0599lb`.
- The tracker MUST be a file copy, not a symlink. r45 rewrites it at each
  save.
- The trainer writes the tracker before the sidecar and the data source
  state. If the tracker reads 49 but a step-49 file is missing, r43
  stopped inside a save. Then set N = 39 by hand.

## Launch checklist

1. Done: the gym image and template v10 exist (`r44/BUILD.md`).
2. Seed the r45 dir (above) from r43 trainer-worker-0. Record the time.
3. Stop r43 on the user's go. First check that no `/tmp/r43-resume.sh`
   watcher runs. After the Stop, check that the r43 tracker still reads 39.
4. `excluded-nodes` is the r43 list. On 2026-09-27 no listed instance
   existed in the cluster, so the list excludes no node.
5. `kubectl create -f r45/workflow.yaml`.
6. In `trainer-0.log`, look for:
   - the load of `iter_0000039` from the r45 dir
   - `Checkpoint sidecar rollout_id=39.` and no `Restored wandb_run_id` line
   - `Loading arena data source state from .../arena_data_source_state_39.pt`
   - a first rollout id of 40
   - a W&B run id other than `2z0599lb`
7. On one gym pod, read `HARBOR_AGENT_KWARGS` (`{}`), `DOCKER_CONFIG`
   (`/root/.docker`), and `ARENA_PUBLISH_JOBS_DIR` (empty).
