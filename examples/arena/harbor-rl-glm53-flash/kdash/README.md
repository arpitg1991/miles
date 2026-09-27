# kdash: isolated tests for GLM-5.3 KDA on the shared head-sharded delta-rule layer

The GLM-5.3-Flash KDA layers run on the shared head-sharded delta-rule layer of
upstream miles (radixark/miles PR #3609, #3634, stack tip `ff6193f26`). Tensor
parallelism splits the 64 KDA heads, 8 heads per rank at TP 8. This is the only
code path: the replicated layer and the `--glm5-next-kda-tp` switch of the
kdatp branch are gone. These tests run away from the live runs.

## Code under test

| File | Change |
| --- | --- |
| `miles/kernels/attention/delta_rule/` | The fla kernel calls, from upstream (`PROVENANCE.md`). |
| `miles_plugins/models/linear_attn.py` | `LinearAttentionLayer`, `DeltaRuleAttention`, `KimiDeltaAttention`, from upstream. |
| `miles_plugins/models/glm5_next/kda.py` | `Glm5NextDeltaAttention` (KDA with the GLM low-rank output gate and bf16 convs) and `Glm5NextKDAAttention` (the `self_attention` layer). `sharded_state_dict` keeps the checkpoint keys `self_attention.kda.*`, `o_norm`, `o_proj` and the packed `conv1d.weight`. |
| `miles/backends/megatron_utils/megatron_to_hf/glm5_next.py` | Weight sync: the runtime names `self_attention.linear_attn.*` to the HF names. |
| `miles_plugins/mbridge/glm5_next.py` | The same names for HF to DCP conversion. The base TP split and merge are correct now. |

The kernel call changes too. The old trainer computed the gate with
`fused_kda_gate` and then called `chunk_kda`. The shared layer
(`KimiDeltaRule`) computes the gate inside `chunk_kda` with the safe-gate
path, as SGLang runs GLM-5.3 prefill. Both compute the same function. T1 measures
the difference (S against A).

## Test image

`arena-slime-dev:miles-glm53-kdash-test-<date><letter>`, built from the
`arpit-kda-shared-layer` branch with `docker build -f examples/arena/Dockerfile
--build-arg MILES_BASE_IMAGE=<glm53next-upstream-20260902>` at the repo root.
It is a test image only: NEVER use it for a live run.

## Files

| File | Use |
| --- | --- |
| `kdash-run.sh` | Pod driver: `t1`, `t2` or `warm`. It writes only under `/mnt/scratch-s3files-rw/guparpit/kdash/`. |
| `t1-job.yaml`, `t2-job.yaml` | PyTorchJob `kdash-t1-<stamp>` (1 node) and `kdash-t2-<stamp>` (8 nodes). Placeholders `__IMAGE__`, `__STAMP__`. |
| `t1_parity.py` | T1 checkpoint-level parity, 8 GPUs (`torchrun`). |
| `build_rollout_data.py` | Train-only rollout files from r45 groups and r43 staged tokens and routing. |
| `make_slice_hf.py` | Config-only HF dir of the 5-layer slice. |
| `gen_arm_configs.py` | Arm YAMLs from `../r45/miles-config.yaml`. |
| `parse_logs.py` | Per-arm metrics, peak memory per stage, and exit codes to JSON. |

## Rules

- NEVER touch a live run (r44, r45, r46, and the runs of other users) or the
  kdatp jobs. The tests read the base DCP, the r43 `iter_0000039` DCP and
  `hf/rollout_39`, and the kdatp train-only data (`kdatp/data`, read only).
  They write only under `kdash/`.
- When `kdatp/data` holds `t1c`, `t2` and the 5-layer HF dir, the tests read
  it in place. Otherwise T1 builds the data under `kdash/data`.
- T2 links the r43 DCP files into `kdash/checkpoints/`. No arm saves:
  `save_interval` is 100000 and each arm stops after 4 rollouts or fewer.
- The jobs use the queue, the workload priority, the pod priority and the
  excluded-nodes list of the live runs. They set no W&B key and
  `use_wandb: false`.
- `cleanPodPolicy: All` stops the pods when the job fails or reaches its
  deadline. The default keeps them, so they keep the GPUs.
- `kdash-run.sh` sets `RAY_DEDUP_LOGS=0`. Without it, the ray driver folds
  the per-rank `[peak-memory]` lines into `[repeated Nx]`.
- MUST delete each job when it ends. The idle-GPU reaper deletes a job whose
  GPUs idle for 60 minutes, so start a job only when it can run.

## T1: one node

```bash
K="kubectl --context arena-prod-bom-v2 -n arena-tasks"
IMAGE=427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-kdash-test-20260927a
STAMP=<fresh suffix, for example 20260927a>
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" t1-job.yaml | $K create --dry-run=server -f -
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" t1-job.yaml | $K create -f -
$K logs -f kdash-t1-$STAMP-worker-0
$K delete pytorchjob kdash-t1-$STAMP   # when driver.log says "done"
```

Phases (results in `kdash/t1/<stamp>/`):

1. Data: reuse `kdatp/data`, or build `data/t1c`, the 5-layer HF config and
   `data/t2` (in the background).
2. GPU unit tests (`unit/<name>.log` and `unit/<name>.rc`):
   - `head_sharded_tp8`, `head_sharded_tp2`:
     `tests/fast-gpu/test_delta_rule_head_sharded.py` at TP 8 and TP 2. The
     layer against its replicated references (Kimi-K3 KDA, GLM-5.3 on the same
     kernel call, GLM-5.3 as the r45 trainer ran it). Relative error at most
     2e-2 for the output, the input gradient and every weight gradient.
   - `linear_attn_layer`: `tests/fast-gpu/test_linear_attn_layer.py`, the
     sequence-parallel and context-parallel paths and a TP 2 checkpoint
     round trip.
   - `conv_chunking`: `tests/fast-gpu/test_delta_rule_conv_chunking.py`, the
     short conv against a float64 reference and the channel chunking past
     2**31 elements.
   - `cpu_layout`: `tests/fast/backends/megatron_utils/test_glm5_next_kda_shared_layer.py`
     and `tests/fast/models` (the checkpoint round trip, the weight-sync gather
     and mbridge, on gloo).
3. Parity (`parity/SUMMARY.json`, `parity/rc`), layers 0 and 44 of the base
   DCP and of r43 `iter_0000039`. A is the r45 replicated layer, S is A on the
   kernel call of the shared layer, B is the production layer:
   - L1: each B shard equals its slice of the A tensor, bit for bit.
   - L2: B asks for exactly the 13 checkpoint keys of A. No key is
     unexpected, and `load_state_dict(strict=True)` passes. The production
     strictness (`assume_ok_unexpected`) gives the same weights.
   - E1: the weight-sync gather and `convert_glm5_next_to_hf` give the HF
     tensors of A, equal to the HF safetensors bit for bit.
   - B1 marks: no B parameter has the `sequence_parallel` mark, and exactly
     the head-sharded parameters have `tensor_model_parallel`.
   - F1, B1: A, S and B are compared with an fp32 reference (A with fp32
     weights and inputs). The output, the input gradient and each weight
     gradient of B pass when the relative L2 error of B is at most 2 x the
     error of A plus 2**-8 (one bf16 rounding). A weight gradient norm ratio
     B/A outside 0.99 to 1.01 fails B1. A ratio near 8, 1/8 or 2.83 is a TP
     reduction bug. The JSON also holds S against A (the kernel call) and B
     against S (the TP split).
   - N1: a contiguous slice of the packed conv moves the output by more than 10%.
   - R1: B saves a DCP. The old layer and B load it bit for bit (rollback).
   - T7: forward plus backward time and peak memory of A and B at 8K, 32K
     and 131K tokens.
4. T1c arms on the 5-layer slice (`results.json`): `shared`, `selective`,
   `none` (`T1C_ARMS`). Each loads the base DCP and trains one step. An
   out-of-memory error is a result, not a failure. Compare with the kdatp
   T1c `baseline` arm (`kdatp/t1/20260927b/results.json`, the same data):
   `rollout/log_probs`, `train/grad_norm`, `perf/actor_train_time`, peak memory.

## Warm-up: one node before T2

A cold T2 job compiles kernels for about an hour, and most GPUs wait at about
250 W. The idle-GPU reaper (AREnAThanatos) deletes a job whose 60-minute mean
GPU power stays below 10%. It deleted `kdatp-t2-20260927b` 65 minutes after
the start, in the first step. MUST warm the kernel cache before T2:

```bash
W=${STAMP}w
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$W#g" -e "s#kdash-t1-#kdash-w1-#g" \
    -e "s#kdash-run.sh t1#kdash-run.sh warm#" t1-job.yaml | $K create -f -
```

The job trains one step of the T2 rows on the 5-layer slice and writes
`kdash/kcache/<stamp>.tar`. The reaper can delete the warm-up job too, so it
also writes `kdash/kcache/<stamp>.snap.tar` every 10 minutes. Give the T2 job
the tarball in `KDASH_KCACHE`. Each pod unpacks it before the start.

## T2: eight nodes, the r45 layout

T2 needs `t2/manifest.json` in the data dir (`kdatp/data` has it). It does
not need a finished T1. Run the warm-up first.

```bash
KC=/mnt/scratch-s3files-rw/guparpit/kdash/kcache/${STAMP}w.tar
ARMS="shared selective"
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" \
    -e "s#- {name: KDASH_STAMP, value: $STAMP}#&\n            - {name: KDASH_KCACHE, value: $KC}\n            - {name: T2_ARMS, value: $ARMS}#" \
    t2-job.yaml | $K create -f -
$K logs -f kdash-t2-$STAMP-worker-0
$K delete pytorchjob kdash-t2-$STAMP
```

Arms (`T2_ARMS`, default `shared block10`): `shared` (the r45 config),
`block<N>` (full recompute of the first N layers of each stage), `shared2`
(the noise floor), `selective`, `none`. Each arm starts from r43
`iter_0000039` and trains rollouts 40 to 43 (one warm-up step and 3 timed
steps) on the same 64 episodes as the kdatp T2 (313 rows plus 1 DP pad, 22.3M
tokens, up to 131,070 tokens per row). Budget: about 1.5 h per arm.

Compare `results.json` with the kdatp T2 (`kdatp/t2/20260927b/results.json`):

- Step 1 parity against the kdatp `baseline` arm (the old replicated layer):
  `train/train_rollout_logprob_abs_diff`, `train/ppo_kl`, `train/grad_norm`,
  `rollout/log_probs`. Use kdatp `baseline2` against `baseline` as the noise
  floor. MUST pass this gate before a live resume on the shared layer.
- Timing: `steady_mean` (steps 2 and 3) of `perf/actor_train_time` and
  `perf/log_probs_time` against kdatp `baseline` and `kdatp`.
- Memory: `peak_memory_gib` per pipeline stage. Keep at least 8 GiB free
  on a 178 GiB GPU.
