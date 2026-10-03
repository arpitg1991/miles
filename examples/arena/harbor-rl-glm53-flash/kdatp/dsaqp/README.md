# kdatp-dsaqp: 1-node parity and timing of the query-parallel DSA core

`--glm5-next-dsa-qp` (`miles_plugins/models/glm5_next/dsa.py`) moves the DSA
sparse-attention core and the indexer from the head split (each TP rank: all
queries, 8 heads) to the sequence split (each rank: its sequence-parallel
chunk of queries, 64 heads) with two all-to-alls over the TP group. The
parameters, their sharding and the checkpoint keys do not change.

| File | Use |
| --- | --- |
| `dsaqp-job.yaml` | PyTorchJob `kdatp-dsaqp-<stamp>`, one p6-b200 node. Placeholders `__IMAGE__`, `__STAMP__`. |
| `dsaqp-run.sh` | Pod driver: runs `../dsa/t1_dsa_qp.py` under `torchrun --nproc-per-node 8`. |
| `../dsa/t1_dsa_qp.py` | The test: I1 indices bitwise, F1 output, B1 gradients and norm ratios, N1 negative control, T7 timing, kernels alone. |

The image code is under test; the pods read only the test script from the
mount (`/mnt/scratch-s3files-rw/acuadron/kdatp/dsaqp/<stamp>/harness/`).

## Run

```bash
K="kubectl --context arena-prod-bom-v2 -n arena-tasks"
STAMP=<fresh suffix>; IMAGE=<trainer image with the flag and the FlashMLA wheel>
H=s3://arena-scratch-prod-bom-ap-south-1/acuadron/kdatp/dsaqp/$STAMP/harness/
aws --profile arena-prod-bom-user s3 cp ../dsa/t1_dsa_qp.py ${H}t1_dsa_qp.py
aws --profile arena-prod-bom-user s3 cp dsaqp-run.sh ${H}dsaqp-run.sh
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" dsaqp-job.yaml | $K create -f -
# results: s3://arena-scratch-prod-bom-ap-south-1/acuadron/kdatp/dsaqp/$STAMP/parity/SUMMARY.md
```

The cluster deletes a finished job at once (TTL 0); read the results on S3.

## Result 20261003a (image `miles-glm53-r19-20261003a`, miles `a56d9a9c`)

PASS. Top-k indices bitwise equal in all four packed cases; the layer output
bitwise equal (both cores run the FlashMLA forward); input and weight
gradients within 5.7e-3 relative L2 (the dKV atomic order), gradient norm
ratios within 2e-4; the reversed-head-group control moves the output by 0.76.

| tokens | head-parallel fwd+bwd ms | query-parallel fwd+bwd ms | speedup | peak GiB above inputs |
| --- | --- | --- | --- | --- |
| 8,192 | 55.3 | 19.1 | 2.90x | 1.50 -> 0.40 |
| 32,768 | 188.9 | 43.0 | 4.40x | 6.11 -> 1.72 |
| 65,536 | 406.2 | 79.1 | 5.14x | 12.26 -> 3.48 |
| 131,072 | 936.0 | 168.5 | 5.55x | 24.56 -> 7.00 |

Kernels alone at 65,536 tokens, one rank: today (8 heads, all queries, width
576) TileLang fwd 101.3 ms, FlashMLA fwd 19.9 ms, TileLang bwd 322.6 ms;
query-parallel (64 heads, 8,192 queries, width 512) FlashMLA fwd 2.3 ms,
TileLang bwd 42.3 ms. The r17 image (the old kernels) measured 102.7 / 496.3 ms
for the same today's case (`../dsa/README.md`, run 20260930a).
