# kcache — the kernel-cache seed of the GLM-5.3 trainer image

A fresh trainer pod compiles and autotunes its TileLang, Triton and inductor kernels in train
step 0. final-v7 (2026-10-06, r20, 16 actor nodes) spent 2,782 s in step 0 for 33.6M tokens, about
41 min more than the warm rate (~110k tok/s) needs, and step 1 another ~6 min. The deployer's
trainer command keeps the caches on local disk only (`/tmp/kernel_cache/{tilelang,triton,inductor}`;
an NFS cache raced 8 ranks into Triton ESTALE in r2), so every pod start paid it again.

The image now carries a seed at that path (`examples/arena/Dockerfile`, build context
`kernel-cache`). A kernel whose source, compiler or GPU changed misses the seed and compiles as
before, so the seed is safe to reuse; re-capture it when the trainer kernels change.

## Capture (from a warm run, step 2 or later)

Trainer pods are the 16 with ~18k Triton files and 20 TileLang files; engine pods have ~2k Triton
and 92 TileLang files. Take every trainer pod (their autotuned shapes differ by rank) and two engine
pods. Files younger than 10 minutes are left out (a kernel that is still being written).

```bash
K="kubectl --context arena-prod-bom-v2 -n arena-tasks"; WF=<workflow>; D=/mnt/scratch-s3files-rw/acuadron/kcache/<stamp>
for w in <trainer pod indices> <2 engine pod indices>; do
  $K exec $WF-trainer-worker-$w -c pytorch -- bash -c \
    "cd /tmp && find kernel_cache -type f -mmin +10 | tar czf $D/kc-w$w.tgz.part -T - && mv $D/kc-w$w.tgz.part $D/kc-w$w.tgz" &
done; wait
aws --profile arena-prod-bom-user s3 sync s3://arena-scratch-prod-bom-ap-south-1/acuadron/kcache/<stamp>/ tgz/ --exclude "*.part"
python3 merge_kcache.py kcache-ctx/kernel_cache.tar tgz/kc-w*.tgz
docker build -f examples/arena/Dockerfile --build-context kernel-cache=kcache-ctx ... .
```

## Seeds

| stamp | from | units (triton dirs / tilelang dirs / inductor files) | tar | used by |
|---|---|---|---|---|
| `v7-20261007` | `acuadron-agentic-debt-final-v7-x44fz` at step 8 (r20 code: dsa-qp, FlashMLA fwd, TileLang bwd, KDA), 16 trainer + 2 engine pods | 3,669 / 28 / 20,387 | 4.45 GB (`s3://arena-scratch-prod-bom-ap-south-1/acuadron/kcache/v7-20261007/kernel_cache.tar`) | r22 |
