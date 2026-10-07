# Run record: guparpit-agentic-debt-v23 (qp-fi-final-v2) — the v20 recipe on agentic-final-v2, fresh from the base model

**Status:** Prepared
<!-- gen-workflow:begin -->
**Date:** 2026-10-07
**Family:** `harbor-rl-glm53-flash`
**Argo generateName:** `guparpit-agentic-debt-v23-`
**Experiment name:** `guparpit-agentic-debt-v23`
**W&B project:** `agentic-debt` (group `guparpit-agentic-debt-v23`)
**Dataset:** `lakefs://arena-inspect/6e4b517ce5355126d1ec65e7da35a2b85dee083cd6bc5adc02943d1ab5b2c2ac/internal/agentic-debt-r3/agentic-final-v2/manifest.jsonl` (gym `agentic-debt`)
**Manifest commit:** `6e4b517ce5355126d1ec65e7da35a2b85dee083cd6bc5adc02943d1ab5b2c2ac` (lakeFS `main` on 2026-10-07 01:40 UTC)
**Gym image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-wvspans-20261001a@sha256:e29ba91a3392fd250e78f551fca6f33061520a2b456134693d419e1bf5c9eab9`
**Trainer image:** `427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-qp-20261006a@sha256:5e540e4b3f04e93480d992154da0686783439114780b9a41ca4affb194a3c667`
**Template:** `guparpit-miles-deployer-v10`
**Base:** `guparpit-agentic-debt-v20`
<!-- gen-workflow:end -->
**Argo workflow:** (set at launch)
**W&B run:** (set at launch)
**Task pin:** `a7b647a2df19` (every manifest row)
**Trainer config deltas vs base:** `prompt-data-list` agentic-debt-766@327e057c -> agentic-final-v2@6e4b517c; `experiment_name`, `project_name`, `arena_sample_summary_dir` -> v23. No checkpoint seed: the checkpoint dir is empty, so the trainer loads `ref_load` (the base GLM-5.3-Flash DCP) at rollout 0. Workflow: `experiment-name` and the refreshed node exclusions only.
**Checkpoints:** `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/guparpit-agentic-debt-v23` (fresh; starts from `ref_load`)
**Outcome:** (set at the end)

## Goal

The owner asked for the v20 run on `agentic-final-v2` as a new run (2026-10-07). This run keeps every v20 setting and changes only the dataset, and it starts from the base model, not from a trained checkpoint.

Comparisons:

- against Alex Cuadron's runs on the same reward family (`acuadron-agentic-debt-final-v6`, W&B `jzfxq3v2`: live-baseline vault reward, fresh from the base, lr 1e-5, GBS 512);
- against v20 (`guparpit-agentic-debt-v20`) on agentic-debt-766 for trainer throughput, trainer wait and log-prob gap on the same trainer image.

This run keeps lr 1.5e-6 and GBS 256 (32 x 8) from v20. Alex's final-v4 and final-v6 used lr 1e-5 and GBS 512.

## Dataset

- `agentic-final-v2` on lakeFS `main` was republished on 2026-10-07 01:40 UTC (commit message "feat(agentic-debt): expand final-v2 to 121 chains with unchanged rewards"). README: 121 chains, 901 checkpoints.
- Every manifest row pins task commit `a7b647a2df19a1b846424a2cc1d716668212c5669bd78aed52a227e7e0980ef9`. Rows hold only `lakefs_commit_id` and `lakefs_uri`; the gym name comes from the `prompt-data-list` entry (`agentic-debt`).
- All 121 chain ids are also ids in agentic-debt-766 (117 repositories). The reward is different: a live baseline, stop on regression, and two gradings per checkpoint.
- Examples: `realdiff_googlefonts_ufomerge_s0` (3 segments), `realdiff_iamjackg_md2cf_s0`, `realdiff_ipnet-mesh_meshcore-mqtt_s0`.

### The final-v4 payload bug, and why it does not apply to this pin

`acuadron-agentic-debt-final-v4` ran `agentic-final-v2` at commit `c519f32c…` for 15.5 h with 0 optimizer steps: every trajectory dropped `no_reward`. Cause (its RECORD): `steps/segment-NN/workdir/setup.sh` ran `rm -rf … /logs/artifacts/handoff …` as the agent user before every agent phase. That deleted the live-baseline snapshot, which the healthcheck or the previous collect hook had written into `/logs/artifacts` (Harbor makes it 0777). The collect hook then found no `handoff/`, `test.sh` reported `grading_setup_failed:prestep_missing`, and no `reward.txt` was written. Payload defect, not a trainer or gym defect.

The republished pin `a7b647a2` moves the baseline out of `/logs/artifacts`:

- the `[environment.healthcheck]` lockdown writes the segment-01 snapshot to `/var/lib/ad-vault/baseline.tar` (root, 0700) and clears `/logs/artifacts/handoff` and `/logs/artifacts/prestep`;
- the root `[[verifier.collect]]` hook builds `prestep/` from the vault and `handoff/` from the submitted tree after the agent stops, then rotates `next.tar` to `baseline.tar` for the next step;
- `setup.sh` (line 92 in `realdiff_googlefonts_ufomerge_s0` segment-01) still runs `rm -rf … /logs/artifacts/handoff …`, but before the agent phase, and the collect hook creates `handoff/` after it. The deletion no longer removes the baseline.

This is the vault scheme of ADR-0076 that `acuadron-agentic-debt-final-v6` trains on (agentic-debt-final@249fc194; `no_reward` about 2.7% of attempts, from gym dind OOMs and the lakeFS startup herd, not from the payload). No harbor nop/oracle gate ran on this exact pin before launch; the first rollout is the gate (see the plan).

Known gym-side risk, from final-v6 on the same template v10: dind is capped at 64 GiB per gym pod with 8 trials and 16 GiB verifier containers, so a few trials die of dind OOM (about 3% of rollout-0 attempts in final-v6).

## Upstream check

- `--glm5-next-dsa-qp` is a fork port of `acuadron/dsa-qp` (ADR-0019 on `arpit-qp-20261006`); upstream `main` has no query-parallel DSA (checked 2026-10-06).
- `--miles-dsa-topk-backend flashinfer` is an upstream choice; the port routes it through the kpool indexer.
- `--max-weight-staleness` is the upstream fully-async flag, ported to the NATS path on 2026-10-01.
- A fresh start uses the upstream path: `resolve_args_checkpoint_load` sets `load = ref_load`, `start_rollout_id = 0`, `finetune`, `no_load_optim`, `no_load_rng` when the `--load` dir holds no Megatron checkpoint.

## Measurement plan

| Question | Signal | Gate |
| --- | --- | --- |
| Does the payload grade? | `Received result` lines and `no_reward` drops in rollout 0 | Stop the run if every result in the first 30 min of results is `no_reward` or `prestep_missing`. Expect a few dind OOM drops. |
| Reward from step 0 | batch `avg_reward`, `rollout/group_metrics/reward.mean`, `raw_reward` | record; compare with final-v6 rollouts 0 to 7 (0.66 to 0.79 kept, 0.72 to 0.83 raw) |
| Episode shape | `rollout/episode_response_length/mean`, graded steps per episode | record; final-v6 grew 61K to 314K tokens per episode in 7 steps |
| Trainer speed | `perf/actor_train_tok_per_s`, `perf/actor_train_time` | within 10% of v20 (69K tokens/s, 748 s) on comparable token counts |
| Loop balance | `perf/train_wait_time`, `rollout/num_old_age_dropped` | record; expect rollout-bound (two gradings per checkpoint) |
| Off-policy drift | `train/train_rollout_logprob_abs_diff` | record; v20 and v5 near 0.03 |
| Faults | `WeightVersionSpansError`, `CUDA error: misaligned address` | none |

## Launch

| Time (UTC) | Event |
| --- | --- |
| 2026-10-07 06:3x | Dataset resolved and checked (above). `v23` free on the cluster, in S3 and on every pushed branch. |
| 06:4x | In-image parse (`recon2/parse/fullparse11.py`, `CFG=/work/v23.yaml` with `hf_checkpoint` remapped to `/work/hf`, `WORLD_SIZE` 320, `RANK` 0, CUDA stub on `LD_LIBRARY_PATH`; `parse-v23.out`): exit 0, 274 argv tokens. `glm5_next_dsa_qp True`, `miles_dsa_topk_backend flashinfer`, `rollout_function_path …NatsRolloutFn`, `max_weight_staleness 8`, multiplier 8, queue cap 64, `prefetch_rollout_data True`, `flash_mla`, `triton`, EP 8, lr 1.5e-6, `calculate_per_token_loss True`, `grpo_std_normalization False`, length coef 0.0, R3 on. `load` resolves to `ref_load` (the empty v23 dir triggers the fresh-start fallback); `save` is the v23 dir; `prompt_data` is the pinned agentic-final-v2 manifest. |
| 06:4x | Node exclusions refreshed: 304 B200 nodes, 1 tainted; 347 excluded (v20's 346, the tainted node, and `i-06818d629c2521eb7`, the node of v19's CUDA fault). |
