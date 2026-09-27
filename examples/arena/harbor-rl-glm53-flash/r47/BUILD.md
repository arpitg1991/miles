# r47: agentic debt on agentic-debt-766, fresh from the base model

Prepared 2026-09-27. Not launched. The trainer image is a placeholder. r45
(`rl-glm53f45-vvqhg`) keeps its nodes until the r47 launch (user decision,
2026-09-27). Generated with:

```
/workplace/guparpit/arena/src/AREnATasks/.venv/bin/python gen-workflow.py 47 --base r45 --template guparpit-miles-deployer-v10 \
  --experiment-name rl-glm53f-adebt-766-r47 \
  --gym-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a \
  --trainer-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:TODO-KDA-RELEASE-IMAGE \
  --param 'agent-kwargs={}' --param 'publish-jobs-dir='
```

## Why

The user gave this dataset for the training:
`lakefs://arena-inspect/main/internal/agentic-debt-r3/agentic-debt-766/`.
r41 to r45 trained on a different set. That set is v3-locked commit
`77c2239b`, filtered to the 551 chains with 5 or fewer steps
(`r41/BUILD.md`). 203 of those 551 chains are not in agentic-debt-766. The
418 agentic-debt-766 chains with more than 5 steps were never trained.

The user also asked for a fresh start from the base model on the latest
configuration. The configuration includes the KDA work, the 16384 output
cap, and the radix cache.

## Delta vs r45

| Item | r45 | r47 |
| --- | --- | --- |
| Dataset | v3-locked `77c2239b`, `manifest-le5.jsonl`: 551 chains, 1,809 segments, K 2..5 | agentic-debt-766: 766 chains, 6,070 segments, K 2..66, mean K 7.92 |
| `prompt-data-list` path | `/mnt/scratch-s3files-rw/guparpit/data/agentic-debt/20260923-v3-locked-oracle/manifest-le5.jsonl` | `lakefs://arena-inspect/327e057c2db03bb4115be89dced633f75fe36dbf8679b4bf29f43c548fbb7480/internal/agentic-debt-r3/agentic-debt-766/manifest.jsonl` |
| Task pin in each row | `77c2239b` | `3cadc6b0a9a6c71bd8e7485e2d9da1cb04e2e608bfa1b9b531cb1df37cd82140` |
| Start point | r43 `iter_0000039`, seeded into the r45 dir | base DCP (`ref_load`) |
| Rollout ids, data position | 40.., r43 offset | 0.., row 0 |
| Epochs in 300 rollouts | 17.4 | 12.5 (23.9 rollouts per epoch) |
| W&B | project `rl-glm53f-adebt-v3` | project `rl-glm53f-adebt-766`, a new run |
| `sglang_disable_overlap_schedule` | `true` | `false` (as r42, r44, r46) |
| Trainer image | `miles-glm53-r14-20260927a` | `TODO-KDA-RELEASE-IMAGE`; fallback `miles-glm53-r15-20260927a` |
| HF export per save | yes (the r14 launcher always adds `--save-hf`) | no (r15 and later add it only on `arena_save_hf: true`) |
| Names | `rl-glm53f-adebt-v3-r45`, `rl-glm53f45-` | `rl-glm53f-adebt-766-r47`, `rl-glm53f47-` |
| Sample summary dir | `debug/rl-glm53f-adebt-v3-r45/sample_summary` | `debug/rl-glm53f-adebt-766-r47/sample_summary` |

Same as r45: gym image `gym-glm53-adr72-20260927a`, template
`guparpit-miles-deployer-v10` (uid `16ddd530-501f-4734-82ae-88500671bed8`,
generation 1), `agent-kwargs` `{}` (Vulcan `max_compactions` 4),
`publish-jobs-dir` `''`, `rollout_max_response_len` 16384, window 131072,
radix cache on, `replicas` 40 (8 actor nodes and 32 SGLang engines),
`gym-replicas` 288, `gym` `agentic-debt`, `agent-timeout-multiplier` 4,
`ack-wait` 86000, trainer deadline 90000, GBS 256, `rollout_batch_size` 64,
`arena_inflight_multiplier` 4, `lr` 1.5e-6, TIS, R3, `save_interval` 10,
`num_rollout` 300, and `excluded-nodes`.

Diff check:

- The workflow parameters are those of `r45/workflow.yaml`, in the same
  order. Only `experiment-name`, `trainer-image`, and `miles-config`
  change. `generateName` is the only other change.
- A local copy of the launcher `_flatten` gives 173 argv tokens for r47 and
  174 for r45. r47 has no `--sglang-disable-overlap-schedule`. The W&B
  project, the sample summary path, and `--prompt-data` are the only other
  token changes.
- No `slime_experiments/rl-glm53f-adebt-766-r47` dir exists, and no
  `logs/`, `debug/`, or `routing/` path with that name exists (S3 list of
  `arena-scratch-prod-bom-ap-south-1/guparpit/`, 2026-09-27 23:41Z). Thus
  miles loads `ref_load` with `finetune` and starts at rollout 0.
- No workflow `rl-glm53f47-*` and no r47 PyTorchJob exists in
  `arena-tasks`.
- `excluded-nodes` is the r45 list (83 instance ids). On 2026-09-27 23:45Z
  none of them was among the 450 cluster nodes, so the list excludes no
  node. A non-empty list selects the `deploy-trainer-pinned` step, as on
  r45.

Images in ap-south-1 `arena-slime-dev` (ECR read, 2026-09-27):

- Gym `gym-glm53-adr72-20260927a`:
  `sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d`.
- Fallback trainer `miles-glm53-r15-20260927a`:
  `sha256:67b57cdf92695b4ec1e3c065cb803349745bc83858b28ef046d59da4cf960247`
  (miles `bc31f88ac`).
- KDA test image `miles-glm53-kdash-test-20260927a`:
  `sha256:58b984912e90c53da6cfbbf3763106f1a859102c952d8c6f1569e8f8105981f5`.
  It is a test image, not the release image.

## Dataset

The dataset check of 2026-09-27 found no blocker. The row format and the
task format are those that r43 and r45 run on the adr72 gym and template
v10. Thus the gym needs no code change.

- The URI resolves to commit `327e057c` (2026-09-25 09:50:46Z, "re-pin
  manifest.jsonl at 3cadc6b0a9a6"). That is the last change under the path.
  Main HEAD `6f4c3f48` holds the same manifest bytes.
- `manifest.jsonl`: 766 rows with the keys `lakefs_commit_id` and
  `lakefs_uri`, md5 `cc78c1ca74df94480a2239402e7fcce0`. Each row names its
  task under `main` and pins commit `3cadc6b0`. The gym puts the commit in
  place of the ref before the download (`gym_worker.py`).
- `prompt-data-list` names commit `327e057c`, not `main`, so a later change
  on `main` cannot change the run.
- Trainer read path: `data_source.py` `_pull_lakefs_file` passes the ref
  segment of the URI to `repository.branch()`. lakeFS accepts a commit id
  there. A local run of the same code (lakeFS SDK 1.86, `aws_iam`) read 766
  rows with the md5 above, for the commit URI and for the `main` URI. The
  trainer pods get `LAKECTL_SERVER_ENDPOINT_URL` and
  `LAKECTL_CREDENTIALS_PROVIDER_TYPE` from template v10. On the r15 image,
  r44 pulled its `lakefs://` manifest at 14:22:14Z.
- The scratch copy of the manifest is not necessary. It needs an S3 write,
  and nobody approved one.
- Task format: schema 1.3, `multi_step_reward_strategy` `final`, step gate
  `min_reward` `{segment_pass = 1.0}`, reward k/K. No agent timeout.
  Verifier 2400 s. `HOME=/home/agent`, so the gym needs
  `DOCKER_CONFIG=/root/.docker` (v10 sets it).
- The verifier now has a memory cap: 16 GiB, and 32 GiB and 48 GiB on
  `MP_PyTorch_s0` and `flamedisx_s0`. r45 ran its verifiers with no cap.
- Images: 766 digests on
  `427267593057.dkr.ecr.us-east-1.amazonaws.com/arena-sandbox/swe-bad-testbeds`,
  also in ap-south-1. The v10 gym logs in to us-east-1.
- Size: 93,225 objects, 26.3 GB. The largest task is `kymatio_s0`
  (13.9 GB), which r45 already runs.

Frontier reference on the same 766 tasks (Harbor viewer, OpenSandbox):
`acuadron-ad-766-pass8-gpt56-xhigh` (6,112 trials, reward 0.4730,
`segment_pass` 0.2403) and `acuadron-ad-766-pass8-opus48-high`. Both jobs
ran commit `7a51be40`. That commit is older than the spec fixes on 33
chains and the verifier memory fix. 95 of its attempts on 11 chains ran out
of memory and have no score.

## Capacity

| Measure | r45 | r47 (estimate) |
| --- | --- | --- |
| Mean steps reached per trial | 2.77 | about 5.1 (Opus depth on all 766; GLM equals Opus depth on the 348 shared chains) |
| Training rows per episode | 1x | about 1.8x |
| Group wall time | mean 11,661 s, p90 20,902 s, max 44,058 s | mean 16,000-24,000 s |
| Groups over the gym deadline (85,700 s) | 0 of 1,173 | 0 to 34 of 766 (0-4.4%), all long chains |
| Train step | log-prob pass 916-986 s, actor train 3,676-4,022 s (rollouts 48-50) | about 1.8x on the r15 image |

The r45 output queue holds 262-320 finished groups (rollouts 48-50), so the
trainer sets the pace. The KDA image and flip A (below) cut the train step.

## Deadlines (open decision)

r47 keeps the r45 values: `ack-wait` 86000 and `trainer-task-deadline-secs`
90000.

- `agent-timeout-multiplier` 4 has no effect. The dataset declares no agent
  timeout.
- The verifier limit 2400 s is enough. The frontier maximum was 1,848 s.
- The gym deadline (`ack-wait` - 300 s) is the only clock of a chain. About
  96% or more of the groups finish in it. The risk is the tail of chains
  with K 15-19 or more.
- A cut turns the unfinished trials into removed pads. Those are the
  deepest trials, which are usually the best. Thus a cut biases the group
  reward low.
- Alternative: `--param ack-wait=172000 --param trainer-task-deadline-secs=176000`.
  The cost is more off-policy lag on the long groups, and a stuck worker
  holds its slot for 48 h, not 24 h.
- If r47 keeps 86000, count `rollout deadline exceeded` in the gym logs
  after each epoch.

## Pending items

| Item | State | How to turn it on | Gate before the launch |
| --- | --- | --- | --- |
| KDA across TP, shared head-sharded layer | Code on miles `arpit-kda-shared-layer` (`8770fcbab`, arpit-glm-53 plus 10 commits). Test image `miles-glm53-kdash-test-20260927a`. kdash T1 and T2 in workflow `wf_cce1530b-ce9`. | Put the release tag in place of `TODO-KDA-RELEASE-IMAGE`. No config key changes. | T1 parity and T2 pass. Release image from the branch head, in ap-south-1 and us-east-1. |
| Flip A: `--skip-actor-forward-only` | Prepared. Upstream miles `23ec9e534`, in r15 and later. | The three `# r47 flip A:` lines: `skip_actor_forward_only: true`, `rollout_batch_size: 32`, `arena_inflight_multiplier: 8`. | User decision. |
| Flip B: selective recompute | Candidate. | The five `# r47 flip B:` lines. | kdash T2 at the r45 layout with no OOM. |
| Token-level loss average (r46) | OFF. | `calculate_per_token_loss: true` | r46 shows a clear win over r44. |
| PP split 12/11/11/11 | Follow-up, not prepared. | `decoder_first_pipeline_num_layers: 12`, `decoder_last_pipeline_num_layers: 11` | A memory and time test. |
| Persistent TileLang and Triton caches | Follow-up, not prepared. | `TILELANG_CACHE_DIR` and `TRITON_CACHE_DIR` on shared storage | A template env change. |

To apply a flip group, remove the `# r47 flip X: ` prefix from each line of
the group. When the next line sets the same key, delete that next line.
OmegaConf stops on a duplicate key. Then run `gen-workflow.py` again with
the final `--trainer-image`, so that the `miles-config` block changes too.

### Flip A: skip the actor forward-only pass

- The arena rollout collects `global_batch_size / n_samples_per_prompt` =
  32 groups (256 episodes) per rollout (`nats_rollout.py`
  `generate_rollout`). `build_dp_schedule` counts episodes, not rows. Thus
  each rollout is already one optimizer step. The trainer logs of r44 and
  r45 show 32 groups and 256 samples per rollout.
- With one step, the old policy is the current policy. The flag reuses the
  detached log-probs of the train forward. The loss is the same, and the
  step saves the log-prob pass (r45: 916-986 s).
- `validate_skip_actor_forward_only` (`miles/utils/arguments.py`) asserts
  `global_batch_size == rollout_batch_size x n_samples_per_prompt`. 256 is
  not 64 x 8, so `rollout_batch_size` goes to 32. `actor.py` also asserts
  one optimizer step.
- In the arena path, `rollout_batch_size` sets only the publisher in-flight
  cap (`arena_inflight_multiplier` x `rollout_batch_size`). Multiplier 8
  keeps the cap at 256 groups. `num_rollout` is set, so no epoch count
  reads `rollout_batch_size`.
- Local check: the flipped config has no duplicate key, and the miles
  validator passes. Without the two batch lines the validator fails with
  "requires exactly one optimizer step for 512 rollout samples".
- The other asserts pass: `keep_old_actor` false, `kl_coef` 0, dropout 0,
  no OPD. R3 (`use_rollout_routing_replay`) is allowed. Upstream tests
  cover R3 with the flag (`test_deepseek_v32_5layer_fp8.py`,
  `test_shared_ppo_lifecycle.py`).
- Metrics: `train/ppo_kl` and `train/pg_clipfrac` read 0, and W&B has no
  `rollout/log_probs`. The drift check moves to
  `train/train_rollout_logprob_abs_diff` and `train/train_rollout_kl`.

### Flip B: selective recompute

T1 runs, 1 node, 5-layer slice, PP1:

| Run | Recompute | Step | Peak allocated |
| --- | --- | --- | --- |
| kdatp T1 | full (uniform, 1 layer) | 391 s | 41 GiB |
| kdatp T1 | selective | 177 s | 82 GiB |
| kdash T1 (shared layer) | selective | 174 s | 77 GiB |
| kdash T1 (shared layer) | none | 161 s | 84 GiB |

- The kdash arms agree on `grad_norm` to 5 digits.
- T1 times one step per arm, so kernel JIT can make an arm slow. The kdash
  full-recompute arm ran first: its log-prob pass took 240 s against 32 s
  for the other arms, and its step took 673 s. Thus it is not in the table.
  Read T1 for memory. kdash T2 times steps 2-4.
- The r45 layout holds 11-12 layers per stage, not 5. NEVER flip B before
  kdash T2 shows no OOM.
- `mhc` needs `enable_hyper_connections` on the command line, because
  Megatron checks it before the `glm5_next` spec turns mHC on.

## Risks

1. The deadline tail on long chains (above).
2. `flamedisx_s0` (K 17, new) has a 48 GiB verifier cap. Its 8 trials share
   one 64 GiB Docker-in-Docker sidecar, so one group can stop the sidecar.
3. The verifiers have a 16 GiB cap. r45 had none. The dataset authors set
   the value.
4. The 418 chains with K > 5 never ran on the Docker gym path. The first
   rollouts are their smoke test. They passed batch validation on
   OpenSandbox.
5. Deep chains (K up to 66, up to 5 segments per step, 8 trials) can pass
   the 30 MiB result cap. Watch `rollout/dropped_groups/too_large`.
6. Two oracles do not pass every step: `dakk_qlasskit_s0` (16/17) and
   `jupyterlab-translate_s0` (2/3). A segment can be impossible.
7. With `arena_train_segments` `all`, a long chain adds more rows than a
   short chain. Its loss weight depends on the loss average.
8. A later move from the r15 image to the KDA image changes the KDA layer
   code. The kdash branch tests a checkpoint round trip. Do a resume
   across the change only after that test passes.

## Launch checklist

1. Decide the open items: the trainer image (KDA release or the r15
   fallback), the deadlines, and flips A and B.
2. Put the image tag in place of `TODO-KDA-RELEASE-IMAGE`, and apply the
   chosen flips. Run `gen-workflow.py` again with the final
   `--trainer-image`. Commit and push.
3. `kubectl create --dry-run=server -f r47/workflow.yaml`. On 2026-09-27 a
   copy with the r15 image was accepted (`rl-glm53f47-k24cr`, not
   created).
4. Check for an r45 watcher. On 2026-09-27 23:40Z no `/tmp/r45-*` script
   and no r45 watcher process ran on this host. If one runs, stop it by
   PID, never with `pkill -f`.
5. Record the r45 tracker and its latest `iter_*` dir, so that r45 can
   resume later.
6. Stop r45 (`rl-glm53f45-vvqhg`) with `shutdown: Stop`, on the user's go.
   Wait until its PyTorchJob and pods are gone. Do not touch r44, r46,
   other users' runs, or the `kdatp-*`, `kdash-*`, and `recon-*` jobs.
7. `kubectl create -f r47/workflow.yaml`. Record the workflow name.
8. trainer-worker-0 argv:
   - the chosen trainer image
   - `--prompt-data` with the `327e057c` URI
   - no `--sglang-disable-overlap-schedule` and no `--save-hf`
   - `--load` = `--save` = `.../slime_experiments/rl-glm53f-adebt-766-r47`
   - with flip A: `--skip-actor-forward-only`, `--rollout-batch-size 32`,
     `--arena-inflight-multiplier 8`
9. `trainer-0.log`:
   - `Pulled lakefs://arena-inspect/327e057c.../agentic-debt-766/manifest.jsonl`
   - the base DCP load from `ref_load` and a first rollout id of 0
   - a new W&B run in project `rl-glm53f-adebt-766`
   - `NATS worker started: ... max_in_flight=256`
10. SGLang server args: `disable_radix_cache=False`,
    `disable_overlap_schedule=False`, and
    `mamba_radix_cache_strategy='extra_buffer'`.
11. On one gym pod, read `HARBOR_AGENT_KWARGS` (`{}`), `DOCKER_CONFIG`
    (`/root/.docker`), and `ARENA_PUBLISH_JOBS_DIR` (empty).
12. In the gym logs, each task download MUST read
    `Downloading lakefs://arena-inspect/3cadc6b0a9a6c71bd8e7485e2d9da1cb04e2e608bfa1b9b531cb1df37cd82140/internal/agentic-debt-r3/agentic-debt-766/tasks/...`.
    A `77c2239b` commit or a `20260923-v3-locked` path means the wrong
    dataset. Also look for `max_new_tokens` 16384 and `max_seq_len` 131072.
13. First rollouts:
    - `Rollout 0 complete: 32 groups (agentic-debt=32), 256 samples`
    - no `pull access denied`
    - count verifier out-of-memory exits, `rollout/dropped_groups/too_large`,
      and `rollout deadline exceeded`
    - chains with K > 5 reach step 6 or more
14. First train step: `train_rollout_logprob_abs_diff`, `train_rollout_kl`,
    and `grad_norm` near the r45 values. `ppo_kl` near 0, or exactly 0 with
    flip A. Record `perf/actor_train_time`, `perf/log_probs_time`, and the
    peak memory.
