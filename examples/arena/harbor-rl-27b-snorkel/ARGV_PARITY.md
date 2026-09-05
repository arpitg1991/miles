# Argv parity: AGISlime snorkel run5 (minus descoped customs) vs miles run_arena_harbor.py

Acceptance record for the snorkel-general-bash-harbor job (per the porting
rule in `.claude/rules/launch-and-model-scripts.md`). This port deliberately
descopes run5's custom training-path pieces (the group-keyed survivor
normalization via `custom_reward_post_process_path`, the removed-sample
replacement module, and the `ARENA_DYN_SAMPLING_MAX_EXAMINE_MULT` env
override — see README.md, "Lineage and descoping"), so the old side of this
record is the AGISlime assembly over a **run5-minus-customs** YAML: the
descoped keys expressed in the original AGISlime shape, everything else
verbatim run5. The diff below therefore names only porting differences, not
the descoping (which is a config-content decision recorded in the README and
the YAML header).

Launcher-level behavior and ray-runtime-env differences are identical to the
financeagent port and are recorded once in
`examples/arena/harbor-rl-27b/ARGV_PARITY.md` — this file covers what this
job adds on top (dynamic sampling, wandb, the dataset pin).

How each side was generated (both executed for real, not transcribed — same
method as the financeagent ARGV_PARITY):

- **Old**: a run5-minus-customs YAML derived from
  `AGISlime/tmp_staging/smoke/harbor-rl-27b-snorkel/run5/snorkel-27b-r5.yaml`
  (drop `custom_reward_post_process_path`, `rollout_batch_size` 128 -> 32,
  `num_rollout` 100000 -> 90; the `amzn_agi_slime.*` dotted paths, the
  `slime.rollout.filter_hub.*` filter path, `rollout_global_dataset: true`
  and the scratch-staged `train.jsonl` dataset pin all kept), run through
  `AGISlime/scripts/custom/hydra_runner/hydra_converter.py`, then the
  `entrypoint.sh` assembly applied with run5 pod env (`EXPERIMENT_NAME=
  rl-snork27b-gbash-r5`, `REPLICA=6`, `REPLICA_TRAINER=2`, `GPUS_PER_NODE`
  default 8, `WANDB_RUN_ID`/`ARENA_EVAL_TASKS`/`S3_ARTIFACT_BASE`/
  `USE_MLFLOW_AMZN` unset, and no `ARENA_DYN_SAMPLING_MAX_EXAMINE_MULT` —
  descoped; it was pod-env only and never touched the argv): `--wandb-group`
  append, `--model-arch` strip, node math, `--rollout-function-path`
  re-append, `--load/--save/--save-hf`, and `scripts/models/qwen3.5-27B.sh`
  MODEL_ARGS last.
- **New**: `run_arena_harbor._build_train_args(ScriptArgs())` run against
  `examples/arena/harbor-rl-27b-snorkel/snorkel-27b.yaml` with the same pod
  env (run5-shaped, minus the examine-mult env), prefixed with
  `shell_safe_model_args("qwen3.5-27B")` exactly as `execute_train`
  prepends it.

The new argv was additionally fed through miles' strict
`miles.utils.arguments.parse_args()` end to end (argparse incl. the arena
plugin's `add_arguments` hook + miles_validate_args + megatron_validate_args +
sglang_validate_args) on a CPU host, with only host substitutions:
`--hf-checkpoint` removed (its config dir is a cluster mount), `--ref-load`
pointed at an existing empty dir (existence check), and
`torch.cuda.get_device_properties` stubbed to sm100 (B200). Result:
`PARSE_OK` with `rollout_batch_size=32`, `num_rollout=90`,
`global_batch_size=256` (rbs * n_samples_per_prompt == GBS again),
`dynamic_sampling_filter_path` landing verbatim,
`custom_reward_post_process_path=None` (asserted — the key appears nowhere;
miles' native group_index-keyed `_normalize_rewards_by_rollout` applies),
`dynamic_sampling_max_examine_mult=4.0` (the registered flag default; no env
override), `arena_sample_mode=full_trajectory`, `loss_mask_type=qwen3_5`,
`rollout_global_dataset=True`, `actor 2x8`, `rollout_num_gpus=32`,
`lr=1.5e-06`, `use_wandb=True` with host/project/team, `log_passrate=False`.
The filter path was then proven live by actually loading AND calling it in
the venv:

```
load_function(miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std) -> OK
  called on a mixed-reward group   (rewards 0.0/1.0) -> keep=True
  called on a zero-variance group  (rewards 1.0/1.0) -> keep=False, reason=zero_std_1.0
```

## (a) Old effective argv (run5 minus descoped customs)

```
python3 -m amzn_agi_slime.train_async_arena
--user guparpit
--cluster bom
--experiment-name rl-snork27b-gbash-r5
--project-name rl-snork27b-gbash-r5
--agislime-dir /opt/AGISlime
--replicas 6
--num-trainers 2
--hf-checkpoint /mnt/models-ro/external/Qwen/Qwen3.5-27B
--ref-load /mnt/scratch-s3files-rw/bowenxie/checkpoints/dcp/tp4/Qwen/Qwen3.5-27B
--save-interval 20
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
--num-rollout 90
--rollout-batch-size 32
--n-samples-per-prompt 8
--rollout-max-response-len 2048
--rollout-temperature 1
--global-batch-size 256
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
--lr 1.5e-06
--lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9
--adam-beta2 0.98
--use-distributed-optimizer
--use-wandb
--use-distributed-post
--arena-sample-mode full_trajectory
--wandb-host https://mega.wandb.agi.amazon.dev
--wandb-project rl-snorkel27
--wandb-team arena
--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
--prompt-data [{"path":"/mnt/scratch-s3files-rw/guparpit/rl-snorkel/datasets/train.jsonl","gym_name":"snorkel-general-bash-harbor","weight":1.0}]
--wandb-group rl-snork27b-gbash-r5
--actor-num-nodes 2
--actor-num-gpus-per-node 8
--rollout-num-gpus 32
--rollout-function-path amzn_agi_slime.nats_arena.nats_rollout.generate_rollout
--load /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-snork27b-gbash-r5
--save /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-snork27b-gbash-r5
--save-hf /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-snork27b-gbash-r5/hf/rollout_{rollout_id}
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

## (b) New argv (miles launcher, reworked YAML, same pod env)

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
--save-interval 20
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
--num-rollout 90
--rollout-batch-size 32
--n-samples-per-prompt 8
--rollout-max-response-len 2048
--rollout-temperature 1
--global-batch-size 256
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
--lr 1.5e-06
--lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9
--adam-beta2 0.98
--use-distributed-optimizer
--use-wandb
--use-distributed-post
--arena-sample-mode full_trajectory
--wandb-host https://mega.wandb.agi.amazon.dev
--wandb-project rl-snorkel27
--wandb-team arena
--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
--prompt-data [{"path":"lakefs://arena-inspect/dev/internal/snorkel-general-bash-harbor/ecr-20260823/manifest.jsonl","gym_name":"snorkel-general-bash-harbor","weight":1.0}]
--wandb-group rl-snork27b-gbash-r5
--actor-num-nodes 2
--actor-num-gpus-per-node 8
--num-gpus-per-node 8
--rollout-num-gpus 32
--load /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-snork27b-gbash-r5
--save /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-snork27b-gbash-r5
--save-hf /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-snork27b-gbash-r5/hf/rollout_{rollout_id}
```

The single-token `--prompt-data` value is `shlex.quote`d by the launcher so
its literal double quotes survive `bash -c` — the same bytes reach argv as in
the original (which relied on unquoted shell expansion not re-parsing quotes).

## (c) Flag -> values multiset diff

Everything not listed below is identical, including duplicity counts
(old 110 flag occurrences vs new 101; the 9-occurrence delta is exactly the
rows here: 15 only-old occurrences minus 6 only-new). Both sides carry
`--rollout-batch-size 32`, `--num-rollout 90` and NO
`--custom-reward-post-process-path` (the descoping is applied to both inputs,
so it cannot hide a porting difference here).

### Only in old

| flag | count | justification |
| --- | --- | --- |
| `--user guparpit` | 1 | Launcher-consumed key. Unregistered in slime 0.3.0 (megatron `ignore_unknown_args=True` silently dropped it); miles argparse is strict, so the launcher pops it from the YAML. |
| `--cluster bom` | 1 | Same; the `CLUSTER` pod env is the real consumer. |
| `--experiment-name rl-snork27b-gbash-r5` | 1 | Same; run identity comes from the `EXPERIMENT_NAME` pod env (ADR-0004). |
| `--project-name rl-snork27b-gbash-r5` | 1 | Same, `PROJECT_NAME` pod env. |
| `--agislime-dir /opt/AGISlime` | 1 | Same, `AGISLIME_DIR` pod env. |
| `--replicas 6` | 1 | Same; node math uses the `REPLICA` pod env. |
| `--num-trainers 2` | 1 | Same, `REPLICA_TRAINER` pod env. |
| `--rollout-function-path amzn_agi_slime.nats_arena.nats_rollout.generate_rollout` | 2 | Dotted path becomes `miles_plugins.arena.…`. The second copy was entrypoint.sh's unconditional re-append (a no-op under argparse last-wins); the launcher emits it once. |
| `--data-source-path amzn_agi_slime.nats_arena.data_source.ArenaDataSourceWithBuffer` | 1 | Dotted path becomes `miles_plugins.arena.…`. |
| `--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` | 1 | miles ships `check_reward_nonzero_std` natively at `miles.rollout.filter_hub.dynamic_sampling_filters` (same semantics); the module root renames `slime.` -> `miles.`. Proven live by loading AND calling it (see above). |
| `--prompt-data [{"path":"/mnt/scratch-s3files-rw/guparpit/rl-snorkel/datasets/train.jsonl",…}]` | 1 | Dataset pin change (per instruction): run5's scratch-staged `train.jsonl` (2600-task training subset) -> the KNOWN_GYMS lakeFS pin. NAMED behavioral difference: trains on all 2922 validated tasks incl. run5's 322 held-out dev tasks (README, "Dataset"). The URI carries an explicit `manifest.jsonl` because the data source pulls exactly the object the URI names; the pin was verified working end to end (the e2e pulled the real 2,922-row manifest via the prod lakeFS endpoint). |
| `--rollout-global-dataset` | 1 | miles has no positive flag: `rollout_global_dataset` defaults to True and only `--disable-rollout-global-dataset` exists. Key removed from the adapted YAML; effective value unchanged (True, verified in the parse probe). |
| `--spec slime_plugins.models.qwen3_5 get_qwen3_5_spec` | 1 | The layer spec ships as `miles_plugins.models.qwen3_5` in miles (`scripts/models/qwen3.5-27B.py`). Same provider function. |
| `--use-gated-attention` | 1 | Dead flag on the deployed stack, dropped by miles' `scripts/models/qwen3.5-27B.py` (full investigation in the financeagent ARGV_PARITY); the functional Qwen3.5 output gate rides on `--attention-output-gate`, which both argvs carry. |

### Only in new

| flag | count | justification |
| --- | --- | --- |
| `--rollout-function-path miles_plugins.arena.nats_arena.nats_rollout.generate_rollout` | 1 | Ported dotted path (single occurrence by design). |
| `--data-source-path miles_plugins.arena.nats_arena.data_source.ArenaDataSourceWithBuffer` | 1 | Ported dotted path. |
| `--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` | 1 | miles-native filter path (see above). |
| `--prompt-data [{"path":"lakefs://arena-inspect/dev/internal/snorkel-general-bash-harbor/ecr-20260823/manifest.jsonl",…}]` | 1 | KNOWN_GYMS dataset pin (see above). |
| `--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec` | 1 | Ported spec module. |
| `--num-gpus-per-node 8` | 1 | miles launcher convention (always pass it). 8 is the miles default; behavior unchanged. |

Ordering differences (multiset-irrelevant, argparse-insensitive): MODEL_ARGS
were last in the old command and are first in the new one (`execute_train`
prepends `shell_safe_model_args`); the launcher-appended flags keep their
relative order.

## Descoped vs run5 proper (config content, not porting)

Named here once so the record is complete against the literal run5 argv —
these are input-YAML differences applied to BOTH sides of this parity, per
the descoping decision (README.md, "Lineage and descoping"):

- `--custom-reward-post-process-path …normalize_by_group_over_survivors`:
  gone. miles' native `_normalize_rewards_by_rollout` keys reward groups on
  `Sample.group_index`, so filter survivors normalize per-group without a
  custom hook; the parse probe asserts the namespace value is `None`. The
  reference implementation lives in the harbor AGISlime commits
  637d714/b8aae83.
- `--rollout-batch-size 128` -> `32`, `--num-rollout 100000` -> `90`: the r3b
  "8x pushed to NATS" budget reverted to the r1 baseline batch shape
  (rbs*n == GBS) with a ~1-epoch cap (2922 tasks / 32).

## Pod-env-only knobs (never argv on either side)

- `ARENA_DYN_SAMPLING_MAX_EXAMINE_MULT`: NOT set (descoped; run5 set 8). The
  filter's examine cap comes from the registered
  `--dynamic-sampling-max-examine-mult` flag, default 4.0 — confirmed as the
  parsed value in the probe with the flag unset in the YAML.
- `ARENA_NATS_REPLACE_REMOVED_SAMPLES`: NOT set (run4's variable; the
  replacement module is not ported).
- `WANDB_BASE_URL`, `NATS_URL`, `ARENA_DEFAULT_GYM`, `LAKECTL_*`: pod env
  contract preserved in `trainer-pytorchjob.yaml`; the launcher additionally
  mirrors `NATS_URL`/`ARENA_DEFAULT_GYM` into the ray runtime env (a named
  difference of the launcher port, recorded in the financeagent
  ARGV_PARITY).

Launcher-behavior and ray-runtime-env differences (pip installs removed, ray
head handling, bounded node wait, role split in the pod command, tee/retry
removal, `NCCL_*`/`FI_EFA_FORK_SAFE`/`ulimit` moved to pod env/command,
`MEGAT_REQUIRE_*`/`EXTRA_PYTHONPATH` dropped): identical to the financeagent
port — see `examples/arena/harbor-rl-27b/ARGV_PARITY.md`, "Ray runtime env
comparison" and "Other named launcher-behavior differences".
