# kdatp-prof: one profiled train step on the r47 layout

One 8-node train-only job that runs the `kdatp` T2 arm (the r47 trainer
layout: `glm5_next_kda_tp`, full uniform recompute, TP8 SP, PP4 11/11/11/12,
EP16, `micro_batch_size` 1, `max_tokens_per_gpu` 8192, r47 flip A
`skip_actor_forward_only`) with the miles torch profiler
(`miles/utils/profile_utils.py`, target `train_overall`) on a slice of the T2
rows: whole groups of the same r45 summary (`PROF_GROUPS`, default `239,258`:
66 rows, 16 episodes, 4.72M tokens, mean row 71.5K tokens as T2, up to
131,070). Three steps run from the r43 `iter_0000039` seed: rollout 40 warms
up and compiles, rollout 41 is profiled on all 64 ranks, rollout 42 runs clean
to measure the profiler overhead. It answers the r45 MFU review question:
where does the forward and backward time go at about 3% of peak.

## Status 2026-09-29: no trace yet, two runs, one cause

Kineto keeps at most 128 MiB of GPU activity records per trace (33 buffers
of 4 MiB; `KINETO_LOG_LEVEL=1` prints `Max GPU buffer size: 128MB`). The
miles hook schedules `warmup=1` when `profile_step_start > 0`, so the whole
step 0 runs under warmup with GPU collection on. torch 2.13 reads the Kineto
"collection stopped" flag at the warmup-to-record transition and, when set,
enters `DEVICE_STOPPED`: it fires `start_trace, stop_trace, _trace_ready` at
once, so every rank writes a stub trace (14 to 20 KB, under 10 ms, about 600
events) and records nothing in step 1. Both runs hit this:

| Run | Data | Microbatches per DP rank | Step 0 | Kineto | Traces |
| --- | --- | --- | --- | --- | --- |
| `20260929a` | T2, 314 rows, 22.3M tokens | 156 | 1,049 s | `Device profiling activity collection was stopped early at step 1` on 64 ranks (Kineto log off) | 64 stubs, 13 to 17 KB |
| `20260929ab` | groups 239,258: 66 rows, 4.72M tokens | 33 | 401 s | `Exceeded max GPU buffer count (33 >= 33) - terminating tracing` at 09:23:37, then `Processed 2782134 GPU records (196977624 bytes)` per rank | 64 stubs, 15 to 18 KB |

One 33-microbatch step (with the cold TileLang compile of step 0) produces
2.78M GPU records, about 197 MB, per rank: about 84K records per microbatch
(kernel, memcpy, memset and runtime API records). The cap holds about 1.8M
records. A slice under about 15 microbatches per DP rank (one group of 30
rows, for example 258) fits with the stock hook; the profiled step 1 then
runs warm. The code fix is one line in `miles/utils/profile_utils.py`:
`experimental_config=torch._C._profiler._ExperimentalConfig(custom_profiler_config="ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB=2048")`
(a new image), or `profile_step_start: 0`, which records step 0 with no
warmup and saves the partial GPU trace at the cap (the compile is in it).
Both runs gave clean r47-layout step times, in `kdatp/prof/<stamp>/kdatp-prof/trainer-0.log`.

| File | Use |
| --- | --- |
| `kdatp-prof-job.yaml` | PyTorchJob `kdatp-prof-<stamp>`, `t2-job.yaml` plus the driver on the mount, `KDATP_KCACHE`, a 5 h deadline and a 200 GiB ephemeral request. Placeholders `__IMAGE__`, `__STAMP__`. |
| `kdatp-prof-run.sh` | Pod driver, the t2 branch of `../kdatp-run.sh` for one arm. It builds `data/prof-<groups>/` once with the in-image `build_rollout_data.py` (DP 2 padding). The pods read it from `kdatp/prof/<stamp>/harness/` on the scratch mount, so the image code does not change. |
| `patch_prof_arm.py` | Writes the `kdatp-prof` arm from the in-image `gen_arm_configs.py` `kdatp` arm: profiler keys, `debug_exit_after_rollout` 3, `tensorboard_dir`, r47 flip A (`skip_actor_forward_only`), and `--set` overrides for the data file and the batch shape (`rollout_batch_size` = groups, `global_batch_size` = 8 x groups). |

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
- The slice gives 33 rows per DP rank, so the pipeline bubble (3 of 33
  microbatch slots per stage) is larger than in r47, which trains 256
  episodes (about 1,250 rows) per step. Read the per-microbatch kernel mix,
  not the bubble share.
- Rank to stage: ranks 0 to 15 are pipeline stage 0 (ranks 0 to 7 DP 0, 8 to
  15 DP 1), 16 to 31 stage 1, 32 to 47 stage 2, 48 to 63 stage 3. Confirm
  from the `[peak-memory] ... rank=N pp=P tp=T` lines of the trainer log.
- The kernel cache tarball comes from the kdatp test image. A miss shows as a
  long step 0 in `driver.log`; the profiled step 1 is warm in both cases.
- With `skip_actor_forward_only`, the step has no separate log-prob pass, so
  `perf/log_probs_time` is absent, as in r47.

Rules: `../README.md` (never touch a live run; MUST delete the job when it
ends).
