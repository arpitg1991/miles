# Argv parity: AGISlime entrypoint.sh vs scripts/run_arena_harbor.py

Acceptance record for the launcher port (per the porting rule in
`.claude/rules/launch-and-model-scripts.md`): the old effective train argv of
the harbor-rl-27b smoke run 1 (`rl-smoke27-financeagent`), the new argv the
miles launcher emits for the adapted YAML, and a flag -> values multiset diff
with every difference named.

How each side was generated (both executed for real, not transcribed):

- **Old**: `AGISlime/scripts/custom/hydra_runner/hydra_converter.py` run on
  `AGISlime/tmp_staging/smoke/harbor-rl-27b/financeagent-27b-smoke.yaml`, then
  the `entrypoint.sh` assembly applied with run-1 pod env (`EXPERIMENT_NAME=
  rl-smoke27-financeagent`, `REPLICA=3`, `REPLICA_TRAINER=2`, `GPUS_PER_NODE`
  default 8, `WANDB_RUN_ID`/`ARENA_EVAL_TASKS`/`S3_ARTIFACT_BASE`/
  `USE_MLFLOW_AMZN` unset): `--wandb-group` append, `--model-arch` strip, node
  math, `--rollout-function-path` re-append, `--load/--save/--save-hf`, and
  `scripts/models/qwen3.5-27B.sh` MODEL_ARGS last. Matches the run-1
  reconstruction in the port analysis (`/tmp/arena-port/runtime.md`).
- **New**: `run_arena_harbor._build_train_args(ScriptArgs())` run against
  `examples/arena/harbor-rl-27b/financeagent-27b-smoke.yaml` with the same pod
  env, prefixed with `shell_safe_model_args("qwen3.5-27B")` exactly as
  `execute_train` prepends it.

The new argv was additionally fed through miles' strict
`miles.utils.arguments.parse_args()` end to end (argparse + miles_validate_args
+ megatron_validate_args + sglang_validate_args) on a CPU host, with only
host substitutions: `--hf-checkpoint` removed (its config dir is a cluster
mount), `--ref-load` pointed at an existing empty dir (existence check), and
the CUDA arch probe stubbed to sm100 (B200). Result: `PARSE_OK`, with
`arena_sample_mode=full_trajectory` (registered by the plugin's
`add_arguments` hook), `loss_mask_type=qwen3_5`, `attention_output_gate=True`,
`rollout_global_dataset=True`.

## (a) Old effective argv (run 1)

```
python3 -m amzn_agi_slime.train_async_arena
--user guparpit
--cluster bom
--experiment-name rl-smoke27-financeagent
--project-name rl-smoke27-financeagent
--agislime-dir /opt/AGISlime
--replicas 3
--num-trainers 2
--hf-checkpoint /mnt/models-ro/external/Qwen/Qwen3.5-27B
--ref-load /mnt/scratch-s3files-rw/bowenxie/checkpoints/dcp/tp4/Qwen/Qwen3.5-27B
--save-interval 10
--no-load-optim
--no-save-optim
--rollout-function-path amzn_agi_slime.nats_arena.nats_rollout.generate_rollout
--data-source-path amzn_agi_slime.nats_arena.data_source.ArenaDataSourceWithBuffer
--sglang-enable-dp-attention
--sglang-disable-overlap-schedule
--input-key messages
--metadata-key metadata
--loss-mask-type qwen3_5
--rollout-global-dataset
--rollout-shuffle
--rollout-skip-special-tokens
--num-rollout 30
--rollout-batch-size 8
--n-samples-per-prompt 8
--rollout-max-response-len 2048
--rollout-temperature 1
--global-batch-size 64
--balance-data
--rollout-num-gpus-per-engine 4
--rollout-max-context-len 131072
--sglang-context-length 131072
--sglang-chunked-prefill-size 12288
--sglang-router-port 30000
--sglang-tool-call-parser qwen3_coder
--sglang-reasoning-parser qwen3
--sglang-mem-fraction-static 0.85
--update-weights-interval 1
--partial-rollout
--skip-eval-before-train
--distributed-timeout-minutes 60
--tensor-model-parallel-size 4
--sequence-parallel
--pipeline-model-parallel-size 2
--context-parallel-size 2
--expert-model-parallel-size 1
--expert-tensor-parallel-size 1
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
--use-dynamic-batch-size
--max-tokens-per-gpu 65536
--attention-dropout 0.0
--hidden-dropout 0.0
--accumulate-allreduce-grads-in-fp32
--attention-softmax-in-fp32
--attention-backend flash
--advantage-estimator grpo
--use-rollout-logprobs
--use-kl-loss
--kl-coef 0.0
--kl-loss-type low_var_kl
--entropy-coef 0.0
--eps-clip 0.4
--eps-clip-high 0.4
--eps-clip-c 2.0
--optimizer adam
--lr 2e-06
--lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9
--adam-beta2 0.98
--use-distributed-optimizer
--use-distributed-post
--arena-sample-mode full_trajectory
--log-passrate
--prompt-data [{"path":"/mnt/scratch-s3files-rw/guparpit/rl-smoke27/datasets/financeagent/manifest.jsonl","gym_name":"financeagent","weight":1.0}]
--wandb-group rl-smoke27-financeagent
--actor-num-nodes 2
--actor-num-gpus-per-node 8
--rollout-num-gpus 8
--rollout-function-path amzn_agi_slime.nats_arena.nats_rollout.generate_rollout
--load /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-smoke27-financeagent
--save /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-smoke27-financeagent
--save-hf /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-smoke27-financeagent/hf/rollout_{rollout_id}
--spec slime_plugins.models.qwen3_5 get_qwen3_5_spec
--disable-bias-linear
--qk-layernorm
--group-query-attention
--num-attention-heads 24
--num-query-groups 4
--kv-channels 256
--num-layers 64
--hidden-size 5120
--ffn-hidden-size 17408
--use-gated-attention
--normalization RMSNorm
--apply-layernorm-1p
--position-embedding-type rope
--norm-epsilon 1e-6
--rotary-percent 0.25
--swiglu
--untie-embeddings-and-output-weights
--vocab-size 248320
--rotary-base 10000000
--attention-output-gate
```

## (b) New argv (miles launcher, adapted YAML, same pod env)

```
python3 <repo>/miles_plugins/arena/train_async_arena.py
--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec
--disable-bias-linear
--qk-layernorm
--group-query-attention
--num-attention-heads 24
--num-query-groups 4
--kv-channels 256
--num-layers 64
--hidden-size 5120
--ffn-hidden-size 17408
--normalization RMSNorm
--apply-layernorm-1p
--position-embedding-type rope
--norm-epsilon 1e-6
--rotary-percent 0.25
--swiglu
--untie-embeddings-and-output-weights
--vocab-size 248320
--rotary-base 10000000
--attention-output-gate
--hf-checkpoint /mnt/models-ro/external/Qwen/Qwen3.5-27B
--ref-load /mnt/scratch-s3files-rw/bowenxie/checkpoints/dcp/tp4/Qwen/Qwen3.5-27B
--save-interval 10
--no-load-optim
--no-save-optim
--rollout-function-path miles_plugins.arena.nats_arena.nats_rollout.generate_rollout
--data-source-path miles_plugins.arena.nats_arena.data_source.ArenaDataSourceWithBuffer
--sglang-enable-dp-attention
--sglang-disable-overlap-schedule
--input-key messages
--metadata-key metadata
--loss-mask-type qwen3_5
--rollout-shuffle
--rollout-skip-special-tokens
--num-rollout 30
--rollout-batch-size 8
--n-samples-per-prompt 8
--rollout-max-response-len 2048
--rollout-temperature 1
--global-batch-size 64
--balance-data
--rollout-num-gpus-per-engine 4
--rollout-max-context-len 131072
--sglang-context-length 131072
--sglang-chunked-prefill-size 12288
--sglang-router-port 30000
--sglang-tool-call-parser qwen3_coder
--sglang-reasoning-parser qwen3
--sglang-mem-fraction-static 0.85
--update-weights-interval 1
--partial-rollout
--skip-eval-before-train
--distributed-timeout-minutes 60
--tensor-model-parallel-size 4
--sequence-parallel
--pipeline-model-parallel-size 2
--context-parallel-size 2
--expert-model-parallel-size 1
--expert-tensor-parallel-size 1
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
--use-dynamic-batch-size
--max-tokens-per-gpu 65536
--attention-dropout 0.0
--hidden-dropout 0.0
--accumulate-allreduce-grads-in-fp32
--attention-softmax-in-fp32
--attention-backend flash
--advantage-estimator grpo
--use-rollout-logprobs
--use-kl-loss
--kl-coef 0.0
--kl-loss-type low_var_kl
--entropy-coef 0.0
--eps-clip 0.4
--eps-clip-high 0.4
--eps-clip-c 2.0
--optimizer adam
--lr 2e-06
--lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9
--adam-beta2 0.98
--use-distributed-optimizer
--use-distributed-post
--arena-sample-mode full_trajectory
--log-passrate
--prompt-data [{"path":"/mnt/scratch-s3files-rw/guparpit/rl-smoke27/datasets/financeagent/manifest.jsonl","gym_name":"financeagent","weight":1.0}]
--wandb-group rl-smoke27-financeagent
--actor-num-nodes 2
--actor-num-gpus-per-node 8
--num-gpus-per-node 8
--rollout-num-gpus 8
--load /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-smoke27-financeagent
--save /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-smoke27-financeagent
--save-hf /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-smoke27-financeagent/hf/rollout_{rollout_id}
```

The single-token `--prompt-data` value is `shlex.quote`d by the launcher so its
literal double quotes survive `bash -c` — the same bytes reach argv as in the
original (which relied on unquoted shell expansion not re-parsing quotes).

## (c) Flag -> values multiset diff

Everything not listed below is identical, including duplicity counts
(old 106 flag occurrences vs new 97; the 9-occurrence delta is exactly the
rows here).

### Only in old

| flag | count | justification |
| --- | --- | --- |
| `--user guparpit` | 1 | Launcher-consumed key. Unregistered in slime 0.3.0 (megatron `ignore_unknown_args=True` silently dropped it — it never had an effect); miles argparse is strict and would hard-fail, so the launcher pops it from the YAML. |
| `--cluster bom` | 1 | Same. The `CLUSTER` pod env, not this flag, was ever the consumer. |
| `--experiment-name rl-smoke27-financeagent` | 1 | Same. Run identity comes from the `EXPERIMENT_NAME` pod env (ADR-0004). |
| `--project-name rl-smoke27-financeagent` | 1 | Same, `PROJECT_NAME` pod env. |
| `--agislime-dir /opt/AGISlime` | 1 | Same, `AGISLIME_DIR` pod env was the real consumer. |
| `--replicas 3` | 1 | Same. Node math uses the `REPLICA` pod env (launcher `--replicas` field). |
| `--num-trainers 2` | 1 | Same, `REPLICA_TRAINER` pod env (launcher `--num-trainers` field). |
| `--rollout-function-path amzn_agi_slime.nats_arena.nats_rollout.generate_rollout` | 2 | Dotted path becomes `miles_plugins.arena.…` (module mapping rule). The second copy was entrypoint.sh's unconditional re-append of the value it grepped from the YAML — a no-op under argparse last-wins that existed only to apply a fallback default; the launcher emits it once. |
| `--rollout-global-dataset` | 1 | miles has no positive flag: `rollout_global_dataset` defaults to True and only `--disable-rollout-global-dataset` (store_false) exists. The key is removed from the adapted YAML; effective value is unchanged (True), verified in the parse probe. |
| `--data-source-path amzn_agi_slime.nats_arena.data_source.ArenaDataSourceWithBuffer` | 1 | Dotted path becomes `miles_plugins.arena.…`. |
| `--spec slime_plugins.models.qwen3_5 get_qwen3_5_spec` | 1 | The layer spec ships as `miles_plugins.models.qwen3_5` in miles (`scripts/models/qwen3.5-27B.py`). Same `get_qwen3_5_spec` provider. |
| `--use-gated-attention` | 1 | Dead flag on the deployed stack, dropped by miles' `scripts/models/qwen3.5-27B.py`. Investigation: radixark Megatron-LM (`miles-main`, `/tmp/miles-deps/Megatron-LM`) does NOT register `--use-gated-attention` (a parse probe hard-fails) and has no `use_gated_attention` config field; it DOES support `attention_output_gate` (TransformerConfig field, `transformer_config.py:265`, auto-registered in argparse — a parse probe sets `attention_output_gate=True`). In AGISlime's own v0.5.9 megatron.patch, `--use-gated-attention` is argparse-registered (patch line 778) but wired to nothing — only the older v0.5.5/v0.5.6 patches mapped it to a config field. The functional Qwen3.5 output gate rides on `--attention-output-gate`, which both argvs carry. |

### Only in new

| flag | count | justification |
| --- | --- | --- |
| `--rollout-function-path miles_plugins.arena.nats_arena.nats_rollout.generate_rollout` | 1 | Ported dotted path (see above; single occurrence by design). |
| `--data-source-path miles_plugins.arena.nats_arena.data_source.ArenaDataSourceWithBuffer` | 1 | Ported dotted path. |
| `--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec` | 1 | Ported spec module. |
| `--num-gpus-per-node 8` | 1 | miles launcher convention (`.claude/rules/launch-and-model-scripts.md`: always pass it). 8 is the miles default, so behavior is unchanged. |

Ordering differences (multiset-irrelevant, argparse-insensitive): MODEL_ARGS
were last in the old command and are first in the new one (`execute_train`
prepends `shell_safe_model_args`); the launcher-appended flags keep their
relative order. `--train-backend` is not passed; miles defaults to `megatron`.

Opt-in branches not exercised by either run — both are named intended
differences from entrypoint.sh:

- `USE_MLFLOW_AMZN=true`: entrypoint.sh appended `--use-mlflow-amzn
  --mlflow-project <p> --mlflow-group <g>`. No consumer registers those flags
  in this tree (none existed in the AGISlime overlay or vendored slime 0.3.0
  either — they were silently-dropped no-ops), so on miles' strict parser they
  would kill the trainer at argparse deep inside the ray job. The launcher no
  longer appends them: it fails fast with a clear error stating mlflow-amzn is
  not supported on the miles arena path yet.
- `S3_ARTIFACT_BASE=<s3://…>`: entrypoint.sh emitted an s3:// URI as
  `--save-hf`, but miles' `save_hf_model` consumes `--save-hf` strictly as a
  local Path — the old behavior wrote each ~55 GB HF export into a mangled
  pod-local `s3:/…` directory under the actor cwd and never uploaded it
  (verify-redundancy.md finding 5; inherited verbatim from
  AGISlime/vendored slime 0.3.0). The launcher now always points `--save-hf`
  at the local `<ckpt_dir>/hf/rollout_{rollout_id}` form. Trainer-side HF
  exports stay local by design; the S3 upload of eval checkpoints happens in
  `eval_rollout` via `s3_artifact`, which is why the AWS region vars still go
  into the runtime env when `S3_ARTIFACT_BASE` is set.

## Ray runtime env comparison

Old (`entrypoint.sh:175,195`):

```
PYTHONPATH=/root/Megatron-LM/:/opt/AGISlime:   CUDA_DEVICE_MAX_CONNECTIONS=1
NCCL_NVLS_ENABLE=<nvidia-smi NVLink probe>     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TENSORBOARD_DIR=<data>/tb_logs/<proj>/<exp>_<MMDD>_<secs>   WANDB_DIR=<data>/logs/wandb/<proj>/<exp>
RAY_enable_open_telemetry_metrics=false
```

New (`execute_train` runtime env + launcher `extra_env_vars`, from the recorded
snapshot `tests/snapshots/launch_scripts/py/scripts/run_arena_harbor.py/train.txt`):

```
PYTHONUNBUFFERED=1                             CUDA_DEVICE_MAX_CONNECTIONS=1
NCCL_NVLS_ENABLE=<same nvidia-smi NVLink probe, or env override>
no_proxy=127.0.0.1,<head>                      MASTER_ADDR=<head>
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TENSORBOARD_DIR=<data>/tb_logs/<proj>/<exp>_<run_id>        WANDB_DIR=<data>/logs/wandb/<proj>/<exp>
RAY_enable_open_telemetry_metrics=false
NATS_URL=<pod env, when set>                   ARENA_DEFAULT_GYM=<pod env, when set>
PYTHONPATH=<miles repo>:/root/Megatron-LM[:ambient PYTHONPATH]
NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME passthrough when set in the pod env
```

Named runtime-env differences:

1. `PYTHONPATH`: `/opt/AGISlime` (scripts-only dir in the old image) becomes the
   miles checkout; the old trailing `:` (cwd on sys.path via empty
   `EXTRA_PYTHONPATH`) is dropped — imports resolve from the checkout, nothing
   relies on cwd.
2. `PYTHONUNBUFFERED`, `no_proxy`, `MASTER_ADDR`, `NCCL_SOCKET_IFNAME`/
   `GLOO_SOCKET_IFNAME` passthrough are additions from `execute_train`'s
   standard preamble (the pod env already carried equivalents).
3. `NATS_URL`/`ARENA_DEFAULT_GYM` are now mirrored into the runtime env when
   set (defensive; the original relied purely on pod-env inheritance, which
   still holds).
4. `TENSORBOARD_DIR` timestamp: `_<MMDD>_<secsSinceMidnight>` becomes
   `_<U.create_run_id()>` (miles convention; per-run uniqueness preserved).
5. When `S3_ARTIFACT_BASE` is set, the AWS region vars go into the runtime env
   (the old S3 branch exported them into the entrypoint shell after
   `ray start`, where they could not reach the driver); they serve the eval
   path's DCP->HF-convert-and-upload flow — `--save-hf` itself stays local
   (see the opt-in branches above).

## Other named launcher-behavior differences (not argv)

1. Pod-start `pip install`s (transformers@git, cudnn, numpy<2, nats-py, lakefs,
   kubernetes==35.0.0) are gone: the arena deps are baked into the image
   (`examples/arena/Dockerfile`, kubernetes pin preserved); the ML stack comes
   from the miles base image.
2. Ray head is started by `execute_train`'s preamble (no
   `--dashboard-host=0.0.0.0`; job submission is local). A pre-formed cluster
   is expressed with `MILES_SCRIPT_EXTERNAL_RAY=1` instead of editing the
   launcher.
3. The wait-for-N-active-nodes loop (rank 0) moved into the launcher as
   `before_ray_job_submit` and is bounded at 120 x 5 s (the original waited
   forever), then submits anyway so the job's own resource error names the
   shortfall. The node count is the same `ray status` Active-section parse.
4. Head/rank detection requires the kubeflow `-worker-<idx>` hostname segment
   (or `REPLICA_IDX`); the original stripped the last `-<suffix>`
   indiscriminately. CLI overrides: `--head-addr` / `--rank`.
5. The role split (rank 0 = train, others = worker) moved from inside
   entrypoint.sh to the pod command in `trainer-pytorchjob.yaml`, selecting the
   typer command from `REPLICA_IDX`.
6. The `MAX_RETRIES=1` retry loop (which never retried) and the driver-side
   `tee` to `<data>/logs/slime_terminal_log_<exp>.log` are dropped; the
   pod-level tee in the PyTorchJob args still captures everything and the exit
   code propagates from `ray job submit`.
7. `CFG_NAME` relative paths resolve under the miles checkout instead of
   `$AGISLIME_DIR/experiments` (absolute paths, as used by the ConfigMap mount,
   behave identically).
8. The `/scratch/$USER` / `/checkpoints/$USER-sandbox` per-cluster fallback
   path layouts are dropped: `ARENA_CHECKPOINTS_DIR`/`ARENA_DATA_DIR` (set on
   both 27B jobs) are the contract, with machine-neutral `/root/shared_data`
   defaults.
9. entrypoint.sh's shell-level exports `NCCL_IB_DISABLE=0`,
   `NCCL_NET_GDR_LEVEL=2`, `NCCL_P2P_DISABLE=0`, `FI_EFA_FORK_SAFE=1` and
   `ulimit -n 1000000` moved to the PyTorchJob env / command;
   `PYTHONBUFFERED=16` (a typo for PYTHONUNBUFFERED) is dropped —
   `PYTHONUNBUFFERED=1` stays in the pod env. `MEGAT_REQUIRE_TE/DATAKIT/MM`
   pod env is dropped (consumed only by the internal Megatron fork in the old
   base image); `EXTRA_PYTHONPATH` is dropped (ambient `PYTHONPATH` passes
   through `execute_train`).
