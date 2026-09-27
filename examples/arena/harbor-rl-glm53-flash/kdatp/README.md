# kdatp: isolated tests for GLM-5.3 KDA tensor parallelism and recompute

Two changes from the r45 MFU review (2026-09-27), tested away from the live runs:

- **Item 3, KDA tensor parallelism.** `--glm5-next-kda-tp` (YAML
  `glm5_next_kda_tp: true`) splits the 64 KDA heads across the 8
  tensor-parallel ranks. The default keeps all heads on each rank, so a run
  without the flag does not change. Port of upstream miles Kimi-K3
  `_init_kda` (PR #1825).
- **Item 4, partial recompute.** Config only: selective recompute of the
  Megatron modules that reach GLM-5.3 (`mhc`, `moe_act`, `layernorm`), and
  block recompute on top of item 3. `mhc` needs `enable_hyper_connections:
  true` in the YAML. Megatron checks it before the glm5_next spec turns
  mHC on.

## Code under test

| File | Change |
| --- | --- |
| `miles_plugins/models/glm5_next/kda.py` | `Glm5NextKDATensorParallel`: TE column linears (q, k, v, b, f_b, g_b), duplicated f_a and g_a, row-parallel o_proj, head-split A_log, dt_bias and packed conv (partition stride 3), o_norm gradient summed by the sequence-parallel all-reduce. `tp_strided_sharded_factory` keeps each checkpoint key and global shape. `Glm5NextKDA` (the default) shares the same forward code. |
| `miles/backends/megatron_utils/update_weight/common.py` | The weight-sync gather accepts stride 3 for `kda.conv1d.weight`. |
| `miles_plugins/mbridge/glm5_next.py` | TP split and merge of the packed conv (HF to DCP conversion, bridge mode). |
| `miles/utils/arguments.py` | `--glm5-next-kda-tp`, default off. |
| `miles/backends/megatron_utils/actor.py` | With `MILES_LOG_PEAK_MEMORY=1`, each rank logs its CUDA peak per step (`[peak-memory]`). The peak window starts at the start of each `train` call, so the model load is not in it. |
| `scripts/run_arena_harbor.py` | A train-only YAML (`load_debug_rollout_data`) runs with no rollout node. |
| `scripts/models/glm5.3-flash-5layer.py` | The real layers 0 to 4: KDA + dense x3, DSA + MoE, KDA + MoE. |

The CPU tests are
`tests/fast/backends/megatron_utils/test_glm5_next_kda_tp.py` (the conv
layout in the DCP factory, the weight-sync gather and the mbridge split and
merge; the default module names and shapes) and
`tests/fast/launch_scripts/test_run_arena_harbor.py` (train only).

## Test image

`arena-slime-dev:miles-glm53-kdatp-test-20260927a`, built 2026-09-27 from
`arpit-kda-tp-recompute` `2340ced4d` with `docker build -f
examples/arena/Dockerfile --build-arg
MILES_BASE_IMAGE=<glm53next-upstream-20260902>` at the repo root. Digest
`sha256:0fe78530f35cfa39da7198e91b0286ab71b091987e08514ef14fffb3aa064b6d`
in us-east-1 and ap-south-1. Image checks: `git rev-parse HEAD` =
`2340ced4d`, a clean tree, and the 6 CPU tests of
`test_glm5_next_kda_tp.py` pass in the image. It is r15 plus this branch,
so it is a test image only: NEVER use it for a live run.

## Files

| File | Use |
| --- | --- |
| `kdatp-run.sh` | Pod driver: `t1` or `t2`. It writes only under `/mnt/scratch-s3files-rw/guparpit/kdatp/`. |
| `t1-job.yaml`, `t2-job.yaml` | PyTorchJob `kdatp-t1-<stamp>` (1 node) and `kdatp-t2-<stamp>` (8 nodes). Placeholders `__IMAGE__`, `__STAMP__`. |
| `t1_parity.py` | T1 module parity, 8 GPUs (`torchrun`). |
| `build_rollout_data.py` | Train-only rollout files from r45 groups and r43 staged tokens and routing. |
| `make_slice_hf.py` | Config-only HF dir of the 5-layer slice. |
| `gen_arm_configs.py` | Arm YAMLs from `../r45/miles-config.yaml`. |
| `parse_logs.py` | Per-arm metrics, peak memory per stage, and exit codes to JSON. |

## Rules

- NEVER touch a live run (r44, r45, r46, and the runs of other users). The
  tests read the base DCP, the r43 `iter_0000039` DCP and `hf/rollout_39`,
  the r43 staged files and the r45 sample summary. They write only under
  `kdatp/`.
- T2 links the r43 DCP files into `kdatp/checkpoints/`. No arm saves:
  `save_interval` is 100000 and each arm stops after 4 rollouts or fewer.
- The jobs use the queue, the workload priority, the pod priority and the
  excluded-nodes list of the live runs. They set no W&B key and
  `use_wandb: false`.
- `cleanPodPolicy: All` stops the pods when the job fails or reaches its
  deadline. The default keeps them, so they keep the GPUs.
- `kdatp-run.sh` sets `RAY_DEDUP_LOGS=0`. Without it, the ray driver folds
  the per-rank `[peak-memory]` lines into `[repeated Nx]`.
- MUST delete each job when it ends. The idle-GPU reaper deletes a job whose
  GPUs idle for 60 minutes, so start a job only when it can run.

## T1: one node

```bash
K="kubectl --context arena-prod-bom-v2 -n arena-tasks"
IMAGE=427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-kdatp-test-20260927a
STAMP=<fresh suffix, for example 20260927a>
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" t1-job.yaml | $K create --dry-run=server -f -
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" t1-job.yaml | $K create -f -
$K logs -f kdatp-t1-$STAMP-worker-0
$K delete pytorchjob kdatp-t1-$STAMP   # when driver.log says "done"
```

Phases (results in `kdatp/t1/<stamp>/`):

1. Data: the 5-layer HF config, `data/t1c` (group 243, 43 rows plus
   rollout 1 as a link), and `data/t2` in the background (8 groups, about
   27 GB, 20 to 40 min).
2. Parity (`parity/SUMMARY.json`, `parity/rc`), layers 0 and 44 of the base
   DCP and of r43 `iter_0000039`:
   - L1: each B shard equals its slice of the A tensor, bit for bit.
   - L2: B asks for no new key except the TE `_extra_state` objects. The
     production strictness (`assume_ok_unexpected`) loads B.
   - E1: the weight-sync gather and `convert_glm5_next_to_hf` give the same
     HF tensors for A and B, equal to the HF safetensors bit for bit.
   - F1, B1: A and B are compared with an fp32 reference (A with fp32
     weights and inputs). The output, the input gradient and each weight
     gradient of B pass when the relative L2 error of B is at most 2 x the
     error of A plus 2**-8 (one bf16 rounding). B is not bitwise equal to A:
     its row-parallel `o_proj` sums 8 bf16 partial outputs. A weight
     gradient norm ratio B/A outside 0.99 to 1.01 fails B1. A ratio near 8,
     1/8 or 2.83 is a TP reduction bug. The JSON also holds the B-A errors
     and the max abs errors.
   - N1: a contiguous conv slice moves the output by more than 10%.
   - R1: B saves a DCP. The old module and B load it bit for bit (rollback).
   - T7: forward plus backward time of A and B at 8K, 32K and 131K tokens.
3. T1c arms on the 5-layer slice (`results.json`): `baseline`, `kdatp`,
   `selective`, `kdatp-selective`, `none`, `kdatp-none`. Each loads the base
   DCP and trains one step. An out-of-memory error is a result, not a
   failure. The peak memory difference against `none` gives the saved
   activations per recompute setting.

## T2: eight nodes, the r45 layout

Run T2 after T1 passes. It needs `data/t2/manifest.json`.

```bash
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" t2-job.yaml | $K create -f -
$K logs -f kdatp-t2-$STAMP-worker-0
$K delete pytorchjob kdatp-t2-$STAMP
```

Arms, in order: `baseline`, `kdatp`, `kdatp-block10`, `baseline2`,
`kdatp-selective` (the design predicts an out-of-memory error on stage 0).
Each arm starts from r43 `iter_0000039` and trains rollouts 40 to 43 (one
warm-up step and 3 timed steps) on the same 64 episodes (313 rows plus 1
DP pad, 22.3M tokens, up to 131,070 tokens per row). `baseline2` trains
rollouts 40 and 41 only. The kernel caches stay on each pod, so only the first arm
compiles cold. Budget: about 1.5 h per arm.

Compare in `results.json`:

- Step 1 parity against `baseline`: `train/train_rollout_logprob_abs_diff`,
  `train/ppo_kl`, `train/grad_norm`, `rollout/log_probs`. Use
  `baseline2` against `baseline` as the noise floor.
- Timing: `steady_mean` (steps 2 and 3) of `perf/actor_train_time` and
  `perf/log_probs_time` against `baseline` (r45 step 42: 3,416 s and
  806 s at 73M tokens).
- Memory: `peak_memory_gib` per pipeline stage. Keep at least 8 GiB free
  on a 178 GiB GPU.
