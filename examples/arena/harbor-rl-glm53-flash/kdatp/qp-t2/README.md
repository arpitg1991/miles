# kdatp-qp: 8-node train-only timing of --glm5-next-dsa-qp on the T2 rows

The prof harness (`../prof/`) with `KDATP_DIR` under `acuadron` and the data
dir linked to the `guparpit` T2 rows. Arms (`arms.yaml`): `qp` (the flag on
the r47 layout), `qp-ep8` (plus `expert_model_parallel_size` 8),
`qp-ep8-pp12` (plus the 12/11/11/11 pipeline split) and `base` (the flag
off, the r18dsa path). Each arm trains rollouts 40 to 43 of `data/t2`
(replayed rollout 40: 314 rows, 22.3M tokens, 64 episodes) from the r43
`iter_0000039` seed; step 0 is cold, steps 1 to 3 time.

```bash
K="kubectl --context arena-prod-bom-v2 -n arena-tasks"
STAMP=<fresh suffix>; IMAGE=<trainer image>
H=s3://arena-scratch-prod-bom-ap-south-1/acuadron/kdatp/prof/$STAMP/harness/
for f in patch_prof_arm.py job.env arms.yaml kdatp-prof-run.sh; do aws --profile arena-prod-bom-user s3 cp $f $H$f; done
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" kdatp-qp-job.yaml | $K create -f -
# results: s3://arena-scratch-prod-bom-ap-south-1/acuadron/kdatp/prof/$STAMP/results.json (and <arm>/trainer-0.log)
```

`kdatp-prof-run.sh` and `patch_prof_arm.py` are verbatim copies of `../prof/`.
The job yaml adds the data link (`$KDATP_DIR/data -> /mnt/scratch-s3files-rw/guparpit/kdatp/data`)
before the driver starts. Rules: `../README.md`.

## Result 20261003q (image `miles-glm53-r19-20261003a`, miles `a56d9a9c`)

All four arms `rc 0`, no out-of-memory error. `actor_train` is the warm mean of steps 41 to 43 (steps 41 to 42 for
`qp-ep8-pp12` and `base`, read before the arm ended). Reference controls on the same rows: r17 (`20260929d` base)
776.9 s, r17 EP8 672.6 s, r18dsa (`20260930f`) 588.9 s, r18dsa EP8 470.7 s.

| Arm | Layout | `actor_train` (s) | vs r17 776.9 s | tok/s per 64 GPUs | Torch max allocated pp0 / pp1 / pp2 / pp3 (GiB) | step-40 `grad_norm` | step-40 log-prob gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `base` | flag off, TP8 PP4 11/11/11/12 EP16 | 581.5 | -25% | 38,400 | 76.6 / 80.2 / 74.2 / 73.2 | 0.129643 | 0.027283 |
| `qp` | `glm5_next_dsa_qp` | 376.5 (377.3 / 380.9 / 371.2) | -51.5% | 59,300 | 65.3 / 69.5 / 63.1 / 66.4 | 0.130988 | 0.027294 |
| `qp-ep8` | plus EP8 | 262.4 (259.7 / 259.2 / 268.4) | -66.2% | 85,100 | 85.6 / 97.6 / 92.1 / 96.8 | 0.129978 | 0.027273 |
| `qp-ep8-pp12` | plus PP 12/11/11/11 | 242.0 (242.6 / 241.3) | -68.8% | 92,300 | 93.6 / 97.4 / 91.9 / 90.9 | 0.128203 | 0.027267 |

The cold step 40 (kernel compiles, allocator warm-up): base 702 s, qp 602 s, qp-ep8 354 s, qp-ep8-pp12 339 s. The
query-parallel backward compiles once (dynamic shapes), so no per-length recompiles follow. Steps 41 to 43 of every
arm track the control step by step in `train_rollout_logprob_abs_diff` (within 0.3%) and `grad_norm` (within 2.3%).
