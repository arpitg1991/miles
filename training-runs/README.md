# Training runs

Run records and investigation records for the miles RL training runs. This
tree is the training counterpart of AREnATasks `eval-runs/` and the
run-history counterpart of `miles_plugins/arena/adr/` (decisions). It
records what ran, on which dataset and images, what changed, what came out,
and why a run ended.

```text
training-runs/
  README.md                       this convention
  INDEX.md                        the index: Runs table (generated) and Studies table
  build_index.py                  rewrites the Runs table of INDEX.md
  TEMPLATE.md                     run record (r<N>/RECORD.md)
  STUDY-TEMPLATE.md               investigation record (studies/<slug>/STUDY.md)
  <family>/
    RUNLOG.md                     legacy chronological log, closed
    experiment-list.md            legacy experiment backlog
    r<N>/
      RECORD.md                   the run record (required header + narrative)
      miles-config.yaml           the trainer config the run parsed
      workflow.yaml               the Argo Workflow that gen-workflow.py wrote
      workflow-resume<K>.yaml     each resume, when one happened
      BUILD.md                    preparation notes (legacy, r11 to r47)
  studies/
    <slug>/
      STUDY.md                    the investigation record
```

Recipe and code stay in `examples/arena/<family>/`: the runbook `README.md`,
the base `miles-config.yaml` and manifests, `gen-workflow.py`, and the
harness directories (`kdatp/`, `memprobe/`). A run folder holds only the
files of that run.

## What a run record is

One folder per run identity (`EXPERIMENT_NAME`). The folder name is the run
number, `r<N>`. `RECORD.md` starts with the required header, then the
sections Goal, Setup, Timeline, Results, Issues, Follow-ups, and Sources.
[TEMPLATE.md](TEMPLATE.md) holds the shape. Keep the record short: numbers
with their source, decisions with their date, and the reason the run ended.

### Required header

| Field | Meaning | Source |
| --- | --- | --- |
| Status | `Prepared`, `Running`, `Complete`, `Retired`, or `Invalid` | operator |
| Date | the day `gen-workflow.py` first wrote the record (UTC); a re-run keeps it | `gen-workflow.py` |
| Family | the `training-runs/<family>` directory | `gen-workflow.py` |
| Argo generateName | `metadata.generateName` of `workflow.yaml` | `gen-workflow.py` |
| Experiment name | the `experiment-name` parameter; the pod env `EXPERIMENT_NAME`, the checkpoint dir name, and the W&B group | `gen-workflow.py` |
| W&B project | `wandb_project` of `miles-config.yaml`, with the group | `gen-workflow.py` |
| Dataset | each `prompt-data-list` row: path and `gym_name` | `gen-workflow.py`; a config without a row is a hard FAIL |
| Manifest commit | the ref segment of the lakeFS URI. When the ref is a branch (`main`, `dev`), add the commit id the trainer pulled (log line `Pulled lakefs://...`) | `gen-workflow.py`, then operator |
| Gym image | the `gym-image` parameter | `gen-workflow.py` |
| Trainer image | the `trainer-image` parameter | `gen-workflow.py` |
| Template | `workflowTemplateRef.name` | `gen-workflow.py` |
| Base | the `--base` run the workflow parameters came from | `gen-workflow.py` |
| Argo workflow | the created Workflow name, `rl-glm53f<N>-<suffix>` | operator, after `kubectl create` |
| W&B run | the run id, one per launch or resume | operator |
| Task pin | the `lakefs_commit_id` of the manifest rows | operator, from the manifest |
| Image digests | `sha256:` of the gym and trainer images (ECR `describe-images`) | operator |
| Trainer config deltas vs base | each key that differs from the base run, with both values | operator |
| Checkpoints | `<ARENA_CHECKPOINTS_DIR>/slime_experiments/<EXPERIMENT_NAME>` and the last `iter_<N>` with its time | operator |
| Outcome | the result, or the reason the run was retired | operator |

## How gen-workflow.py fills the header

`examples/arena/<family>/gen-workflow.py <N>` writes
`training-runs/<family>/r<N>/workflow.yaml`. Then it creates
`r<N>/RECORD.md` from [TEMPLATE.md](TEMPLATE.md), or updates it. The lines
between `<!-- gen-workflow:begin -->` and `<!-- gen-workflow:end -->` belong
to the script: it rewrites them on every run. Text outside the markers
belongs to the operator: the script keeps it byte for byte. NEVER remove the
markers; without them the script stops with a FAIL.

The dataset comes from the `prompt-data-list` key of `r<N>/miles-config.yaml`.
When the key is absent or has no `path`, the script exits with `FAIL` before
it writes `workflow.yaml`. Thus a record cannot omit the dataset line.

`tests/fast/plugins/arena/test_gen_workflow.py` is the runnable check. Run it
with a Python that has `pytest` and `pyyaml`; `--confcutdir` keeps the root
`tests/conftest.py` (it imports `miles.rollout`, which needs `sglang`) out:

```bash
PYTHONPATH=$PWD python -m pytest --confcutdir=tests/fast/plugins/arena \
  tests/fast/plugins/arena/test_gen_workflow.py \
  tests/fast/plugins/arena/test_training_runs_index.py -q
```

## How to add a run

1. Create `training-runs/<family>/r<N>/miles-config.yaml` from the base run.
2. Run `gen-workflow.py <N> --base r<M> --template <name> --experiment-name
   <name> --gym-image <image> --trainer-image <image>` from
   `examples/arena/<family>/`. Read its docstring for `--param`.
3. Fill the operator fields of the header and the Goal and Setup sections
   before the launch. Add the Argo workflow name after `kubectl create`.
4. Add a Timeline row for each launch, admission, save, resume, and retire
   event. Add the resume workflow as `workflow-resume<K>.yaml`.
5. At the end, fill Results, Outcome, and the last save. Set Status.
6. Run `python training-runs/build_index.py` to refresh the Runs table of
   [INDEX.md](INDEX.md). `test_committed_index_is_current` fails on a stale
   table.

## How to add a study

An investigation that spans runs (a kernel port, a loss change, an engine
setting, a throughput gate) gets `training-runs/studies/<slug>/STUDY.md`
from [STUDY-TEMPLATE.md](STUDY-TEMPLATE.md). Name the runs, the data, the
code SHAs, and the workflow journal ids it used. Mark each number you cannot
re-verify against a primary source as "not re-verified". Add a row to the
Studies table of [INDEX.md](INDEX.md).

## Legacy logs

`<family>/RUNLOG.md` is the chronological log of a family up to 2026-09-28.
It is closed: new entries go into `r<N>/RECORD.md`. `experiment-list.md` is
the experiment backlog of the same period. Both stay as history. The
families `harbor-rl-27b` and `harbor-rl-27b-snorkel` keep their `RUNLOG.md`
under `examples/arena/`; they have no run folders.

## What not to track

Raw trainer logs, W&B exports, checkpoints, token dumps, and manifests stay
out of git. Record their S3 prefix or EFS path instead. NEVER commit a
secret, a Midway cookie, or a W&B key.

## Index

[INDEX.md](INDEX.md) holds the Runs table (one row per `RECORD.md`, written
by `python training-runs/build_index.py` from the record headers) and the
Studies table (one row per `studies/<slug>/STUDY.md`, hand-written).

### Families

| Family | Recipe and code | Legacy logs | Run folders |
| --- | --- | --- | --- |
| `harbor-rl-glm53-flash` | `examples/arena/harbor-rl-glm53-flash/` | [RUNLOG.md](harbor-rl-glm53-flash/RUNLOG.md), [experiment-list.md](harbor-rl-glm53-flash/experiment-list.md) | `r9` to `r47` (`r36`, `r37`, `r40` were never prepared; `r41b` is the r41 relaunch) |
