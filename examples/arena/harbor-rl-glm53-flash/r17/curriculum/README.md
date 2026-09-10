# r17 curriculum manifest (2026-09-10)

`/mnt/scratch-s3files-rw/guparpit/data/snorkel-general-bash-harbor/ecr-20260823-curriculum-gpt56/manifest.jsonl`
holds the 2922 rows of
`lakefs://arena-inspect/dev/internal/snorkel-general-bash-harbor/ecr-20260823/manifest.jsonl`
(byte-identical row set, md5 of the sorted rows `2ef2103b40f27bc3c99cd8065e1f921a`),
ordered easiest first. Task name = last path component of `lakefs_uri`.

Sort key, in order:

1. GPT-5.6 solve rate, descending. Harbor job
   `guparpit-snorkel-gpt56sol-wqbvs` (8 trials per task, all 2922 tasks),
   read from `/api/jobs/<id>/tasks?page=N&page_size=100`. Distribution:
   1234 tasks at 1.0, 290 at 0.0, mean 0.70.
2. GLM-5.2 base verifier mean, descending, as the tie-break. Source: the
   registered Inspect eval `snorkel-glm52-gptoss120b-8x-merged` (2917 of 2922
   tasks). Spearman with GPT-5.6 is 0.30 per task but monotone per bucket
   (GPT 0.0 -> GLM 0.26, GPT 1.0 -> GLM 0.59).
3. Original manifest order.

r16 per-task rewards were not recoverable: the trainer logs only group
status, the gym deletes trial dirs after publish, and no reward reaches the
task-events log. `scores.csv` carries both signals per task.
