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
