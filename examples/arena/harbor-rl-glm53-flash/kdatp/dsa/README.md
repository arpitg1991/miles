# kdatp-dsa: 1-node kernel test of the DSA kernels of miles PR #3608

The test compares the DSA sparse attention kernels of miles PR #3608
(branch `arpit-dsa-3608`, `a60c648e61`) with the current kernel of the r17
image (miles `4716a367a`) on one p6-b200 node. It measures the speed, the
memory and the numerics of each kernel at the GLM-5.3-Flash shapes. It also
runs the PR unit tests on TileLang 0.1.9. It is step (a) of section 6 of
`kdfast/scratch/prof/20260929c/datapath/dsa-3608-prep.md`.

Data: synthetic kernel inputs only (random q, kv and upstream gradients, the
PR bench `causal_indices`, and `kpool_select_topk` on random index tensors).
The agentic-debt dataset of record,
`lakefs://arena-inspect/main/internal/agentic-debt-r3/agentic-debt-766/`, is
not used. A launch record MUST state both facts.

| File | Use |
| --- | --- |
| `dsa-job.yaml` | PyTorchJob `kdatp-dsa1-<stamp>`: `../t1-job.yaml` with the driver on the mount and a 5,400 s deadline. Placeholders `__IMAGE__`, `__STAMP__`. |
| `dsa-run.sh` | Pod driver. It writes only under `kdatp/dsa/<stamp>/` on the scratch mount. |
| `t1_dsa.py` | One process per GPU: the cases of one group. `--merge` writes `results.json` and `results.md`. `--plan` prints the plan and checks the imports. |
| `old_dsa/` | The current kernel: `git show 4716a367a:miles_plugins/models/glm5/ops/<file>` for `sparse_mla.py`, `tilelang_sparse_mla_fwd.py`, `tilelang_sparse_mla_bwd.py`, byte for byte (the PR image deletes that directory). |

## Cases

GPU `g` runs group `g`, three lengths each: 8,192 (with the fp32 dense
reference), 65,536 and 131,072 tokens. All cases: batch 1, one KV group,
index width 2,112, `d_v` 512, and the model softmax scale `256**-0.5`
(`glm5_next/dsa.py`).

| GPU | Heads per rank | q/kv width | Indices |
| --- | --- | --- | --- |
| 0 | 8 (TP8) | 576 (zero tail, as today) | `causal` |
| 1 | 8 | 576 | `kpool` |
| 2 | 16 (TP4) | 576 | `causal` |
| 3 | 16 | 576 | `kpool` |
| 4 to 7 | 8, 8, 16, 16 | 512 (no zero tail, optional) | `causal`, `kpool`, `causal`, `kpool` |

- `causal`: `causal_indices` of `tests/manual/bench_dsa.py`, to compare with
  the PR tables.
- `kpool`: `build_pooled_keys` of `miles/kernels/attention/dsa/kpool.py` and
  `kpool_select_topk` of `glm5_next/ops/kpool_indexer.py` (index top-k 2,048,
  kpool 4, 32 index heads of 128), one sequence per row. This is the real pool
  layout.
- The seed depends on the indices and the length only. So the 512 and 576
  cases get the same values, and `results.md` shows if the zero tail changes
  any bit.

Kernels: `old` (`old_dsa.SparseMLA`, width 576 only), `tl`
(`sparse_attention(..., forward_backend="tilelang")`) and `fmla`
(`forward_backend="flash_mla"`, only when `flash_mla` imports). Per case and
kernel: 2 warm-up and 5 timed calls. CUDA events give the forward, backward
and fwd+bwd time of each call. The forward runs with autograd on, as in
training. The host wall time of each call is a cross-check. One more call gives the peak memory above the inputs
(`max_memory_allocated` minus `memory_allocated`) for the forward and for
fwd+bwd. A handler on the `tilelang` logger counts the TileLang compiles and
their time. Each process has its own empty kernel cache.

## Gates

`rel_diff` is the PR metric, `dsa_reference.rel_diff` =
1 - 2xy / (x^2 + y^2). The PR tables print this metric, not a relative L2
error (they show negative values). The JSON also holds the relative L2 and
the max abs error.

| Gate | Cases | Pass |
| --- | --- | --- |
| `tl_vs_old` | width 576 | out, lse and dq bitwise equal; dkv `rel_diff` <= 1e-6 (PR: 4e-7, fp32 atomic order) |
| `fmla_vs_old` | width 576 | out, dq and dkv `rel_diff` <= 1e-5. A wrong LSE base of the wheel shows as a large dq and dkv error. |
| `fmla_vs_tl` | width 512 | out, dq and dkv `rel_diff` <= 1e-5 (PR test tolerance) |
| `ref` | 8,192 tokens | `tl` and `fmla` out `rel_diff` to the fp32 reference <= 2.5e-6. The PR prints "2e-06" with the `.0e` format, which covers values below 2.5e-6. `old` is reported. |
| `finite` | all | no NaN and no inf in out, lse, dq, dkv |
| `no_error` | all | every kernel of the case ran (no out-of-memory error, no exception) |
| `tl_bwd_one_compile` | per GPU | the new backward compiles at most once for the three lengths (the old backward compiles once per length) |
| pytest | GPU 0 | `cd /root/miles && python3 -m pytest -q tests/fast-gpu/kernels/attention/dsa` passes (PR: 60 passed on GB300) |

An out-of-memory error or a failed gate is a result, not a reason for a
relaunch.

## Driver timeline

1. Wait for the harness files, copy them to the pod disk, and log their
   sha256, the package versions and the GPUs.
2. Start 8 processes, one per GPU. Round 1 runs all cases of the group.
3. When all processes end round 1 (at most 2,700 s), run pytest on GPU 0
   (at most 2,100 s). GPUs 1 to 7 repeat their timed calls (rounds 2, 3, ...)
   meanwhile. The idle-GPU reaper deletes a workload whose 60-minute mean GPU
   power is below 10%, so the GPUs MUST stay busy.
4. Stop the rounds, merge, and write `done`.

Results in `kdatp/dsa/<stamp>/`: `driver.log`, `harness.sha256`,
`gpu-power.csv`, `gpu<g>/case-<case>.json`, `gpu<g>/group.json` (versions,
all compile events, `tl_bwd_one_compile`), `gpu<g>/rounds.jsonl`,
`gpu<g>/log.txt`, `gpu<g>/rc`, `pytest.log`, `pytest.rc`, `results.json`,
`results.md`, `merge.log` and `done`.

## Run

Upload the committed files, not a working tree. Upload `dsa-run.sh` last.

```bash
K="kubectl --context arena-prod-bom-v2 -n arena-tasks"
STAMP=<fresh suffix, for example 20260930a>
SHA=<the miles commit of this directory>
IMAGE=427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r18dsa-20260930a
SRC=/workplace/guparpit/kdfast/scratch/prof/20260929c/dsa-run/upload-$STAMP   # never /tmp
H=s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/dsa/$STAMP/harness/
mkdir -p $SRC
git -C /workplace/guparpit/arena/src/miles archive $SHA examples/arena/harbor-rl-glm53-flash/kdatp/dsa \
  | tar -x -C $SRC --strip-components=5
for f in t1_dsa.py old_dsa/__init__.py old_dsa/sparse_mla.py old_dsa/tilelang_sparse_mla_fwd.py \
         old_dsa/tilelang_sparse_mla_bwd.py dsa-run.sh; do
  aws --profile arena-prod-bom-user s3 cp $SRC/$f $H$f
done
aws --profile arena-prod-bom-user s3 ls --recursive $H
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" $SRC/dsa-job.yaml | $K create --dry-run=server -f -
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" $SRC/dsa-job.yaml | $K create -f -
# When kdatp/dsa/$STAMP/done exists or the pod has ended:
$K delete pytorchjob kdatp-dsa1-$STAMP --ignore-not-found
$K get pods -l app=kdatp-dsa1-$STAMP   # MUST show no pods
```

The cluster sets `ttlSecondsAfterFinished: 0`, so a finished job and its pod
vanish at once. Read the results on S3
(`s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/dsa/<stamp>/`), not
from the pod. Rules: `../README.md` and `../prof/README.md`.
