# r35: agentic-debt cut-at-first-failure, chains with MORE than 5 segments

r33's recipe and config verbatim, retrained on the complement dataset: the
agentic-debt chains with more than 5 segments (segment count > 5). r33 trained
the <=5 subset (`manifest-adebt-le5.jsonl`, 556 chains). r35 trains the other
278 chains. Everything else matches r33.

## Data

`manifest-adebt-gt5.jsonl`: the 278 chains of the 834-chain manifest with more
than 5 segments. Exact set difference `manifest-adebt-834.jsonl` - le5, because
le5 is the byte-exact subset of 834 with segment count <= 5. No segcounts file
is needed; kept rows keep their 834 order and bytes.

| Item | Value |
| --- | --- |
| Builder | `r35/mkman-gt5.py` over `manifest-adebt-834.jsonl` + `manifest-adebt-le5.jsonl` |
| Object mount | `/mnt/scratch-s3files-rw/guparpit/data/agentic-debt/20260916-v1/manifest-adebt-gt5.jsonl` |
| Rows, md5 | 278 chains, `b82cf755013c2037e06dda2c00709f2c` |
| Disjoint check | le5 (556) + gt5 (278) = 834, no overlap |

## Deltas vs r33 (config)

- experiment_name / project_name: rl-glm53f-adebt-cut-r33 -> -r35
- arena_sample_summary_dir -> .../rl-glm53f-adebt-cut-r35/sample_summary
- prompt-data-list path -> manifest-adebt-gt5.jsonl
- Every other knob identical: global_batch_size 256, update_weights_interval 1,
  sglang radix cache disabled, partial-reward off, step-cut-on-fail true, same
  trainer + gym image, same v7 template, same deadlines
  (ack-wait 86000, agent-timeout-multiplier 4, trainer-task-deadline-secs 90000).

## Caveat

The gt5 chains hold 6+ segments. r33's deadlines were sized for <=5. Step-cut
aborts a chain at the first failed step, so most chains stop early, but a chain
that keeps passing to segment 6 or 7 can exceed ack-wait 86000. Watch the
`kept_timeout` and chain-abort rates; raise the deadlines if long full-pass
chains time out.

## Launch

```
.venv/bin/python gen-workflow.py 35 --base r33 --template guparpit-miles-deployer-v7 \
  --partial-reward off --experiment-name rl-glm53f-adebt-cut-r35 \
  --gym-image .../arena-slime-dev:gym-glm53-adebt-cut-r31-20260921a \
  --trainer-image .../arena-slime-dev:miles-glm53-r9-20260920a \
  --param gym=agentic-debt --param ack-wait=86000 --param agent-timeout-multiplier=4 \
  --param trainer-task-deadline-secs=90000 --param compaction-max=2 \
  --param step-cut-on-fail=true
kubectl create -f r35/workflow.yaml
```

Launched 2026-09-22 as `rl-glm53f35-nvkh2` (W&B `rl-glm53f-adebt-cut-r35`).
Fresh run, no resume watcher.
