# ADR-0006: Launcher argv parity — `scripts/run_arena_harbor.py` replaces entrypoint.sh + hydra_converter

**Status:** Accepted
**Date:** 2026-09-01

**Builds on:** ADR-0001 (module mapping `amzn_agi_slime.X -> miles_plugins.arena.X`),
ADR-0005 (the driver this launcher submits). Honours AGISlime ADR-0004 (trainer run
identity: the `EXPERIMENT_NAME` pod env keys `--load`/`--save` and the W&B group; an
existing checkpoint dir resumes silently).

## Context

- AGISlime launched arena Harbor RL as a kubeflow PyTorchJob whose identical pods ran
  `scripts/custom/entrypoint.sh`: pod-start `pip install`s, EFA/NCCL exports, rank from the
  hostname suffix, rank>0 `ray start --block`, rank 0 starts the head, waits *forever* for
  `REPLICA` active nodes, converts the ConfigMap YAML with the 41-line
  `hydra_runner/hydra_converter.py`, appends `--wandb-group` / node math /
  `--load --save --save-hf`, greps `--model-arch` to `source scripts/models/qwen3.5-27B.sh`,
  then `ray job submit ... python3 -m amzn_agi_slime.train_async_arena` inside a
  `MAX_RETRIES=1` loop with a driver-side `tee` (`arena-port-artifacts/reports/runtime.md`).
- miles' launcher rule (`.claude/rules/launch-and-model-scripts.md`): launchers and model
  definitions are `.py` (`load_model_args` imports `scripts/models/<type>.py`); "porting a
  shell recipe is a semantics-preserving move — compare the ray runtime env plus the argv as
  a flag -> values multiset; every intended difference gets named"; every public entrypoint
  has a snapshot under `tests/snapshots/launch_scripts/py/`; any environment knob a script
  reads must be in `CLEARED_ENV` (line 48) because the snapshots pin the expanded argv.
- miles argparse is strict. slime 0.3.0 rode Megatron's `ignore_unknown_args=True`, so the
  YAML's launcher-addressed keys (`user`, `cluster`, `experiment_name`, `project_name`,
  `agislime_dir`, `replicas`, `num_trainers`) reached argv and were silently dropped; on
  miles they would kill the trainer deep inside the ray job.
- Stack deltas found while porting: miles registers only `--disable-rollout-global-dataset`
  (store_false, default True; `arguments.py:1037-1039`); radixark Megatron-LM (`miles-main`)
  does not register `--use-gated-attention` but auto-registers `attention_output_gate`
  (`transformer_config.py:265`); `save_hf_model` consumes `--save-hf` strictly as a local
  `Path` (`hf_export.py:132,152`).

## Decision

**One typer launcher, `scripts/run_arena_harbor.py`, whose YAML -> argv conversion is
hydra-exact and whose acceptance is a flag -> values multiset parity record.**

1. **Two commands, role chosen by the pod command.** `worker` (replicas 1..N-1): kill
   stale sglang/ray/miles, `nc -z <head> 6379` probe, `ray start --address ... --num-gpus
   --block`. `train` (replica 0): `U.execute_train` starts the head; `before_ray_job_submit`
   runs entrypoint.sh's `ray status | sed Active..Pending | grep -c node_` wait, bounded at
   120 x 5 s (then submits anyway so the job's own resource error names the shortfall); the
   job is `miles_plugins/arena/train_async_arena.py`. Rank/head come from `REPLICA_IDX` or
   the kubeflow `<job>-worker-<idx>` hostname (CLI `--rank`/`--head-addr` override).
2. **`ScriptArgs(U.ExecuteTrainConfig)` defaults follow the pod-env contract** through
   `default_factory`: `CFG_NAME`, `REPLICA`, `REPLICA_TRAINER`, `GPUS_PER_NODE`,
   `EXPERIMENT_NAME`, `PROJECT_NAME`, `ARENA_CHECKPOINTS_DIR`, `ARENA_DATA_DIR`, `NATS_URL`,
   `ARENA_DEFAULT_GYM`, `WANDB_RUN_ID`, `ARENA_EVAL_TASKS`, `S3_ARTIFACT_BASE`,
   `S3_ARTIFACT_REGION`, `USE_MLFLOW_AMZN`; machine-neutral `/root/shared_data/arena/...`
   defaults replace the `/scratch/$USER` per-cluster fallbacks so no snapshot embeds the
   recording user.
3. **Hydra-exact conversion.** `_flatten` is hydra_converter's flatten verbatim (true ->
   bare flag, false/None -> dropped, list -> flag + one token per item, scalars `str()`'d so
   OmegaConf re-emits `2e-6` as `2e-06`); `prompt-data-list` becomes `--prompt-data` and
   moves last; the eight launcher-consumed keys are popped first (only `model_arch` has a
   consumer: `scripts/models/<model_arch>.py`). The YAML schema stays AGISlime's, so a
   harbor config ports by editing two dotted paths. Every token is `shlex.quote`d so the
   `--prompt-data` JSON survives `bash -c`. entrypoint.sh's appends keep their conditions
   (`--wandb-group`, `--wandb-run-id`, `--disable-wandb-random-suffix`, node math,
   `--load` = `--save` = `<ARENA_CHECKPOINTS_DIR>/slime_experiments/<EXPERIMENT_NAME>`,
   `--save-hf <ckpt>/hf/rollout_{rollout_id}`); model args are prepended by `execute_train`.
4. **Parity is the acceptance artifact** (`examples/arena/harbor-rl-27b/ARGV_PARITY.md`):
   (a) the old effective run-1 argv of `rl-smoke27-financeagent`, produced by *executing*
   hydra_converter + the entrypoint.sh assembly with the run-1 pod env; (b) the new argv from
   `_build_train_args`; (c) the multiset diff — **old 106 vs new 97 flag occurrences**, the
   9-occurrence gap fully covered by 12 only-in-old rows (13 occurrences) and 4 only-in-new
   rows, each justified:

   | Delta | Why |
   | --- | --- |
   | 7 launcher-consumed echo flags gone | unregistered no-ops on slime 0.3.0; hard-fail on miles |
   | `--rollout-function-path` 2 -> 1, `--data-source-path`, `--spec` renamed | `amzn_agi_slime`/`slime_plugins` -> `miles_plugins` (ADR-0001); the duplicate was entrypoint.sh's unconditional re-append, an argparse last-wins no-op |
   | `--rollout-global-dataset` dropped | no positive flag in miles; effective value stays True (parse probe) |
   | `--use-gated-attention` dropped | dead: unregistered in radixark Megatron-LM; AGISlime's own v0.5.9 megatron.patch registers it but wires it to nothing (only v0.5.5/v0.5.6 mapped it). Qwen3.5's output gate rides `--attention-output-gate`, present in both argvs |
   | `--num-gpus-per-node 8` added | launcher rule ("always pass it"); 8 is the default |

   The runtime-env diff (PYTHONPATH `/opt/AGISlime` -> the miles checkout, trailing-colon
   cwd hack dropped; `TENSORBOARD_DIR` stamp -> `U.create_run_id()`; `NATS_URL` /
   `ARENA_DEFAULT_GYM` mirrored in when set; AWS region vars now reach the driver) and nine
   non-argv differences (pip installs baked into the image, bounded wait, role split in the
   pod command, retry loop and driver tee dropped, NCCL/EFA exports moved to the PyTorchJob,
   `MEGAT_REQUIRE_*` / `EXTRA_PYTHONPATH` dropped) are named there too.
5. **Two opt-in branches change on purpose.** `USE_MLFLOW_AMZN=true` no longer appends
   `--use-mlflow-amzn --mlflow-project --mlflow-group` (no parser registers them here, nor
   did any in AGISlime/slime 0.3.0): the launcher raises `typer.BadParameter` up front.
   `--save-hf` always takes the local `<ckpt>/hf/rollout_{rollout_id}` form even when
   `S3_ARTIFACT_BASE` is set: the `s3://` URI entrypoint.sh emitted was collapsed by `Path()`
   into a pod-local `s3:/...` directory holding a full ~55 GB HF export per save, never
   uploaded (verify-redundancy finding, inherited verbatim from AGISlime). Eval checkpoints
   still upload through `eval_rollout` + `s3_artifact`, which is why the region vars stay.
6. **Snapshots and `CLEARED_ENV`.** `train.txt` / `worker.txt` are recorded with
   `MILES_UPDATE_LAUNCH_SCRIPT_SNAPSHOTS=1` against the launcher's default config
   `examples/arena/harbor-rl-27b/financeagent-27b-smoke.yaml` (the harness rglobs
   `scripts/run_*.py`, so the YAML ships with the launcher). The 15 env knobs above go into
   `tests/fast/launch_scripts/py_harness.py` `CLEARED_ENV`: the snapshot pins the expanded
   argv, so an exported `EXPERIMENT_NAME` or `REPLICA` would otherwise flip it. The
   snapshots were recorded with the defaults and stay byte-identical.

### Alternatives considered

| Alternative | Why not |
| --- | --- |
| Keep `entrypoint.sh`, call it from the pod | Violates the rule (no shell launchers; `scripts/models/*.sh` unresolvable by `load_model_args`); its pip installs and `/opt/AGISlime` paths do not exist in a miles image; its argv hard-fails miles argparse (7 unregistered flags, `--use-gated-attention`, `--rollout-global-dataset`). |
| Port `hydra_converter.py` verbatim behind a thin shell wrapper | Same rule violation; the 41 lines are inlined as `_flatten` + `_load_experiment_config`, byte-identical output on both original YAMLs (run 1 and run 2). |
| Register the seven launcher keys as no-op trainer flags in the plugin hook | Keeps YAML -> argv 1:1 but adds dead flags to the trainer; popping them in the launcher is the honest boundary. |
| Drop the launcher keys from the YAML schema | Existing harbor configs would need rewriting; only `rollout_global_dataset: true` is removed, forced by miles. |
| Register the mlflow-amzn flags | Nothing consumes them; deferred until a consumer exists. |
| Keep the `s3://` `--save-hf` and teach `save_hf_model` S3 | Larger core change; the eval path already uploads via `s3_artifact`; trainer-side HF exports stay local by design. |
| Hand-roll `ray start` / `ray job submit` (AGISlime style) | Rule: reach the shell only through `command_utils`; pre-formed clusters use `MILES_SCRIPT_EXTERNAL_RAY=1`. |

## Consequences

Easier: a snapshot-covered miles launcher (`tests/fast/launch_scripts` 43,
`tests/manual/launch_scripts -k arena` 4); the recorded `train.txt` (963 lines) shows all
120 bounded polls and then the submit, so the wait bound is visible in the review artifact;
the full new argv strict-parses through `miles.utils.arguments.parse_args` on a CPU host
(`PARSE_OK`: `loss_mask_type=qwen3_5`, `attention_output_gate=True`,
`rollout_global_dataset=True`, `arena_sample_mode=full_trajectory`); a harbor config ports
by editing two dotted paths.

Harder / open: a YAML key read by AGISlime modules other than `nats_rollout` (eval-pod
`eval_*` knobs, `wandb_disable_system_metrics`, `hallmark_benchmarks`) is not registered by
the plugin hook and hard-fails on miles where it silently worked on 0.3.0 — the 27B configs
carry none. Eval-enabled jobs must set `MILES_ARENA_DIR` (templates are outside this tree).
`REPLICA_IDX` / `HOSTNAME` feed only rank/head detection and are not cleared; the harness's
frozen `MASTER_ADDR=127.0.0.1` is the fallback the snapshot records. The CPU parse used host
substitutions (`--hf-checkpoint` removed, `--ref-load` -> empty dir, CUDA arch stubbed to
sm100), so the first on-cluster run had to watch argparse/validation output. mlflow-amzn is
unsupported on this path until someone registers the flags.
