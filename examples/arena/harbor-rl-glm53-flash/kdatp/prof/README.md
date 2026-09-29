# kdatp-prof: one profiled train step on the r47 layout

One 8-node train-only job that runs the `kdatp` T2 arm (the r47 trainer
layout: `glm5_next_kda_tp`, full uniform recompute, TP8 SP, PP4 11/11/11/12,
EP16, `micro_batch_size` 1, `max_tokens_per_gpu` 8192) with the miles torch
profiler (`miles/utils/profile_utils.py`, target `train_overall`) on the T2
rows (314 rows, 64 episodes, 22.3M tokens, up to 131,070 tokens per row).
Three steps run from the r43 `iter_0000039` seed: rollout 40 warms up and
compiles, rollout 41 is profiled on all 64 ranks, rollout 42 runs clean to
measure the profiler overhead. It answers the r45 MFU review question: where
does the forward and backward time go at about 3% of peak.

| File | Use |
| --- | --- |
| `kdatp-prof-job.yaml` | PyTorchJob `kdatp-prof-<stamp>`, `t2-job.yaml` plus the driver on the mount, `KDATP_KCACHE`, a 5 h deadline and a 200 GiB ephemeral request. Placeholders `__IMAGE__`, `__STAMP__`. |
| `kdatp-prof-run.sh` | Pod driver, the t2 branch of `../kdatp-run.sh` for one arm. The pods read it from `kdatp/prof/<stamp>/harness/` on the scratch mount, so the image code does not change. |
| `patch_prof_arm.py` | Writes the `kdatp-prof` arm from the in-image `gen_arm_configs.py` `kdatp` arm: profiler keys, `debug_exit_after_rollout` 3, `tensorboard_dir`, and r47 flip A (`skip_actor_forward_only`). |

Run (the image reference comes from the r47 trainer pod spec):

```bash
K="kubectl --context arena-prod-bom-v2 -n arena-tasks"
IMAGE=427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r17-20260928a
STAMP=20260929a
aws --profile arena-prod-bom-user s3 cp kdatp-prof-run.sh s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/prof/$STAMP/harness/
aws --profile arena-prod-bom-user s3 cp patch_prof_arm.py s3://arena-scratch-prod-bom-ap-south-1/guparpit/kdatp/prof/$STAMP/harness/
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" kdatp-prof-job.yaml | $K create --dry-run=server -f -
sed -e "s#__IMAGE__#$IMAGE#" -e "s#__STAMP__#$STAMP#g" kdatp-prof-job.yaml | $K create -f -
$K delete pytorchjob kdatp-prof-$STAMP   # when driver.log says "done"
```

Results in `kdatp/prof/<stamp>/`: `driver.log`, `arms/kdatp-prof.yaml`,
`kdatp-prof/trainer-0.log` (`perf N:` and `[peak-memory]` lines),
`tb/train_overall_rank_<rank>.<ns>.pt.trace.json.gz` for ranks 0 to 63, and
`results.json` (`parse_logs.py`).

Read the trace with these limits:

- Step 1 carries the profiler overhead (`record_shapes`, `with_stack`,
  `profile_memory`, `with_flops`). Compare its `perf 41` times with `perf 42`.
- The T2 rows give 157 rows and 156 microbatches per DP rank, so the pipeline
  bubble is smaller than in r47, which trains 256 episodes per step.
- Rank to stage: ranks 0 to 15 are pipeline stage 0 (ranks 0 to 7 DP 0, 8 to
  15 DP 1), 16 to 31 stage 1, 32 to 47 stage 2, 48 to 63 stage 3. Confirm
  from the `[peak-memory] ... rank=N pp=P tp=T` lines of the trainer log.
- The kernel cache tarball comes from the kdatp test image. A miss shows as a
  long step 0 in `driver.log`; the profiled step 1 is warm in both cases.
- With `skip_actor_forward_only`, the step has no separate log-prob pass, so
  `perf/log_probs_time` is absent, as in r47.

Rules: `../README.md` (never touch a live run; MUST delete the job when it
ends).
