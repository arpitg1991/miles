# kdatp-qp-t2: 8-node train-only timing of `--glm5-next-dsa-qp` on the ns stack

The arms of `arms.yaml` run in one `kdatp-prof-<stamp>` job (`../prof/kdatp-prof-job.yaml`,
`../prof/kdatp-prof-run.sh`, `../prof/patch_prof_arm.py`; `../prof/README.md` holds the
procedure) on the shared T2 rows. Every arm is the ns layout, TP8 SP, PP4, EP8, FlashMLA forward;
the same-job `control` arm is the flag off. The arms add the query-parallel core (`qp`), the
flashinfer top-k in the kpool indexer (`qp-fi`), the 12/11/11/11 stage split with and without the
core (`qp-split`, `split`). Results: `s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/prof/<stamp>/results.json`.

Reference numbers on the same rows: `20260930f` ep8 (FlashMLA + EP8, earlier kernel port) 470.7 s
`actor_train`; acuadron `20261003q` qp-ep8 259.5 s, qp-ep8-pp12 242.0 s, same-job control 581.5 s
(their ADR-0018, fork numbering).

```bash
STAMP=2026MMDDx; IMAGE=<qp trainer image by digest>
H=s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/prof/$STAMP/harness/
for f in ../prof/kdatp-prof-run.sh ../prof/patch_prof_arm.py ../prof/a2a_bench.py; do aws --profile arena-prod-bom-user s3 cp $f $H; done
aws --profile arena-prod-bom-user s3 cp job.env ${H}job.env; aws --profile arena-prod-bom-user s3 cp arms.yaml ${H}arms.yaml
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" ../prof/kdatp-prof-job.yaml | kubectl --context arena-prod-bom-v2 -n arena-tasks create -f -
```
