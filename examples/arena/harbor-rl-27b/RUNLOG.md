# harbor-rl-27b run log

Run history and decision record for `examples/arena/harbor-rl-27b/`, the
trainer side of the Qwen3.5-27B + `financeagent` Harbor RL smoke ported from
AGISlime to miles. This is the first arena example directory and the origin
of `examples/arena/Dockerfile`; the snorkel example and the GLM-5.3-Flash
example keep their own run logs and only link back here. Entries record what
was ported, how it was validated, which findings changed the files, and what
was deliberately left as-is. All times are Pacific (PT).

## Format

One `## <date PT> — <milestone or run>` section per event, newest last.
Each section names the source assets, the identity of any run
(`EXPERIMENT_NAME`, image tag, W&B group), the numbers that were actually
measured, the decision taken and the alternatives not taken. Facts that were
inferred rather than observed are marked as such. Never rewrite an old entry;
append a correction.

## 2026-09-01 01:21-02:38 PT — Port of the ADR-0039 27B financeagent smoke (trainer side)

Lineage. AREnATasks ADR-0039 (Harbor NATS gym worker, 2026-08-18) ran its 27B
smoke with an AGISlime trainer: `AGISlime/tmp_staging/smoke/harbor-rl-27b/`
(`financeagent-27b-smoke.yaml`, `trainer-pytorchjob.yaml`, `sglang-svc.yaml`)
plus the gym side in `AREnATasks/tmp_staging/harbor-rl-smoke/`
(`bom-27b/gym-worker.yaml`, `nats.yaml`, `stage_dataset.sh`; tracked in git).
AGISlime run 1 = `rl-smoke27-financeagent`: image
`arena-slime-dev:rl-smoke-20260821` (AGISlime build on bowenxie's
`arena-slime:latest` B200 base), 3x p6-b200.48xlarge (workers 0-1 Megatron TP4/PP2/CP2 = 16 GPUs, DP1;
worker 2 = 2 SGLang engines x 4 GPUs with dp-attention), rbs 8 x n 8,
GBS 64, 30 rollouts, lr 2e-6, 2048 tokens/turn, 10 gym workers
(`arena-tasks-dev:rl-smoke-20260821`, ARENA_MAX_TOKENS 2048,
ARENA_ROLLOUT_CONTEXT_LIMIT 32768, ARENA_NATS_ACK_WAIT 4500). Reward
quarter means 0.173 / 0.242 / 0.320 / 0.331 are the passing baseline quoted
in the README. ADR-0039 records that any-generate `length` stamping marked
67% of the live 27B smoke's samples for removal. Run 2 (`rl-smoke27b-financeagent-r3`,
14 nodes, rbs 32 / GBS 256, lr 1.5e-6, W&B on) was used only as a converter
test input; this directory ports the run-1 shape.

What landed here (mtimes): `sglang-svc.yaml` 02:34, `README.md` 02:35,
`trainer-pytorchjob.yaml` and `financeagent-27b-smoke.yaml` 02:37,
`ARGV_PARITY.md` 02:38 (the config and the parity record are committed with
the launcher). First cut of the launcher + config + PyTorchJob + README +
Dockerfile was ~01:21 PT (snapshots recorded then); the files were finalised
in the 02:34-02:58 PT fix-up pass after the verification wave.

Validation was CPU-only; nothing in this directory has been launched on GPU
on miles (see the next entry):

- Argv parity (`ARGV_PARITY.md`): old effective run-1 argv (executed
  `hydra_converter.py` + `entrypoint.sh` assembly with the run-1 pod env) vs
  the launcher's argv for the adapted YAML: 106 vs 97 flag occurrences, the
  9-occurrence delta fully named. Only-in-old: 7 launcher-consumed echo
  flags (`--user`, `--cluster`, `--experiment-name`, `--project-name`,
  `--agislime-dir`, `--replicas`, `--num-trainers`; unregistered no-ops on
  slime 0.3.0, fatal on miles' strict parser), the duplicate
  `--rollout-function-path` re-append, `--rollout-global-dataset` (miles has
  only `--disable-rollout-global-dataset`; default True), `--use-gated-attention`
  (not registered in radixark Megatron-LM; unwired even in AGISlime's v0.5.9
  patch; the Qwen3.5 gate rides `--attention-output-gate`, present on both
  sides), and the three `amzn_agi_slime`/`slime_plugins` dotted paths.
  Only-in-new: the three `miles_plugins` paths and `--num-gpus-per-node 8`.
- Strict parse: the complete argv through `miles.utils.arguments.parse_args`
  (argparse + miles/megatron/sglang validate) -> `PARSE_OK` with
  `arena_sample_mode=full_trajectory`, `loss_mask_type=qwen3_5`,
  `attention_output_gate=True`, `rollout_global_dataset=True`. Host
  substitutions: `--hf-checkpoint` removed (cluster mount), `--ref-load` ->
  empty dir, CUDA arch probe stubbed to sm100.
- Launcher snapshots `train.txt` 963 lines / `worker.txt` 13 lines;
  `tests/fast/launch_scripts` 43 passed; arena snapshot cases 4 passed.
- e2e harness (`arena-port-artifacts/e2e-harness/`): real NATS 2.14.6
  JetStream in docker + a fake financeagent gym worker speaking the
  `amzn_arena_contract` wire format, driver args = this YAML scaled to
  rbs 2 / n 4 / GBS 8 / 2 rollouts: 49/50 checks. The one failure ("results
  all acked") is a teardown artifact of the blocking `output_queue.put`,
  byte-identical to AGISlime. Adversarial stale-session and garbage
  injections handled. Tokenizer substitute: Qwen/Qwen3-0.6B (venv
  transformers 5.12.1 < the 5.15 Qwen3.5 floor); the fast path never
  re-tokenizes.
- 145 ported arena tests + 3 qwen3_5 mask tests green; YAML sanity (config
  83 keys, PyTorchJob bash shim exercised for REPLICA_IDX 0 and 1).

Verification findings that changed this directory (wave ~01:50-02:30 PT):

| finding | severity | fix |
| --- | --- | --- |
| `sglang-svc.yaml` missing; README pointed at the non-27B `harbor-rl` Service (`rl-smoke-sglang` selecting `rl-smoke-trainer`) -> zero endpoints, wrong DNS name | major | copied `AGISlime/tmp_staging/smoke/harbor-rl-27b/sglang-svc.yaml` verbatim (Service `rl-smoke27-sglang`, selector job-name `rl-smoke27-trainer` + replica-index "0", port 30000); README warns against substituting the other variant |
| Dockerfile omitted `boto3` although `setup.py`'s `arena` extra declares it and `s3_artifact` / `eval_rollout` / `eval_coordinator` import it; the miles base has none | minor | added to the pip line, with `typer` (launcher CLI) |
| `USE_MLFLOW_AMZN=true` would hard-fail argparse (no consumer registers the three mlflow-amzn flags anywhere) | note | launcher fails fast with a clear message instead of appending them |
| `AGISLIME_DIR=/root/miles` satisfies DCP->HF converter resolution only; miles ships no `experiments/k8s/templates/`, so an eval trigger would raise instead of log-and-skip | verify-dataEval | README + yaml comment: eval-enabled jobs MUST set `MILES_ARENA_DIR`; the smoke configures no eval |

Decisions:

- Trainer side only. Gym workers, NATS and dataset staging stay in
  AREnATasks unchanged; the wire contract (streams `ARENA_TASKS` /
  `ARENA_RESULTS`, subjects `arena.tasks.<gym>` / `arena.results`, durable
  `slime-trainer`) is bit-identical (plugin ADR-0002).
- Trainer image (`examples/arena/Dockerfile` v1, 56 lines): `FROM
  ${MILES_BASE_IMAGE}` (a `docker/build.py` image, placeholder
  `radixark/miles:latest`); bake `nats-py>=2.6.0 kubernetes==35.0.0 lakefs
  boto3 omegaconf typer` instead of AGISlime's per-pod-start pip installs
  (transformers@git, cudnn, numpy<2, nats-py, lakefs, kubernetes) - a network
  dependency and drift source; keep the `kubernetes==35.0.0` pin (36.x returns
  401 on EKS); `rm -rf /root/miles && COPY . /root/miles` over the base's
  editable checkout (the AGISlime delete-upstream-source pattern); CPU import
  smoke through a `libcuda.so.1` stub asserting
  `SALVAGEABLE_RESULT_STATUSES == ('success', 'truncated')` and
  `run_arena_harbor.py --help`. The hand-listed pip line was kept over
  `pip install -e '/root/miles[arena]'` so omegaconf/typer are documented as
  launcher deps.
- Config semantics preserved: `rollout_num_gpus_per_engine: 4` + dp-attention
  as in the original, although miles' own Qwen3.5 recipes force 1 GPU per
  engine (sglang#21039, SGLang TP>1 mis-generates Qwen3.5 on the pinned
  build). Recorded as a concern, not changed; the snorkel-derived GPU smoke
  later ran 1 GPU/engine with dp-attention off.
- PyTorchJob: same env contract as the original (CLUSTER, REPLICA/
  REPLICA_TRAINER, EXPERIMENT_NAME/PROJECT_NAME, CFG_NAME, JOBNAME, USER,
  NATS_URL, ARENA_DEFAULT_GYM, ARENA_CHECKPOINTS_DIR/DATA_DIR, LAKECTL_*,
  REPLICA_IDX, POD_NAME); `entrypoint.sh`'s shell exports
  (`NCCL_IB_DISABLE=0`, `NCCL_NET_GDR_LEVEL=2`, `NCCL_P2P_DISABLE=0`,
  `FI_EFA_FORK_SAFE=1`, `ulimit -n 1000000`) moved into pod env / command;
  `MEGAT_REQUIRE_*`, `EXTRA_PYTHONPATH` and the `PYTHONBUFFERED=16` typo
  dropped; role picked from `REPLICA_IDX` in the pod command (0 = `train`,
  else `worker`); `AGISLIME_DIR` repointed to `/root/miles`. The
  `elasticPolicy`/torchrun block is vestigial (the command is overridden),
  as in the original.

## 2026-09-05 — Status: never GPU-launched on miles; caveats for anyone who does

- No run of this example exists on miles. The only GPU validation of the
  port used a 3-node derivative of the snorkel example
  (`rl-milesgb1-smoke1`, 2026-09-01 16:39 -> 2026-09-02 00:34 PT); see
  `examples/arena/harbor-rl-27b-snorkel/RUNLOG.md`. That job was the first
  real build of this Dockerfile (base `radixark/miles:latest`, cu13; the
  `latest-cu12` nightly of 2026-09-01 failed on a TE<->torch ImportError).
- `EXPERIMENT_NAME=rl-smoke27-financeagent` is the AGISlime run-1 identity,
  and `--load`/`--save` resolve to the same
  `/mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-smoke27-financeagent`
  the AGISlime run wrote. Launching as-is would silently resume that
  checkpoint dir if it still exists (ADR-0004). Bump `EXPERIMENT_NAME` and
  `PROJECT_NAME` first.
- EFA env is aspirational at this commit. The manifest carries the
  original's fabric wiring (`FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1`,
  `LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:...`,
  `vpc.amazonaws.com/efa: 8`, `/dev/infiniband`), but Dockerfile v1 has no
  EFA layer and the miles base ships no aws-ofi-nccl; without
  `/opt/amazon/ofi-nccl/lib/libnccl-net.so` NCCL silently falls back to raw
  IB / TCP and the first multi-node all_gather hangs. The layer
  (aws-efa-installer 1.47.0 + build gate) was added 2026-09-02 01:17 PT and
  fixed at 01:27 PT (apt-get update first) with the GLM-5.3-Flash
  scaffolding; the snorkel smoke ran on `FI_PROVIDER=tcp` (MFU ~2%).
- Image naming drift. README and manifest say
  `<registry>/miles-arena-dev:rl-smoke-<date>` /
  `<account>.dkr.ecr.ap-south-1.amazonaws.com/miles-arena-dev:rl-smoke-latest`
  (placeholder). Every image actually built from this Dockerfile was pushed
  to the `arena-slime-dev` ECR repository (us-east-1, replicated to
  ap-south-1): `miles-arena-20260901b` for the smoke, then the
  `miles-glm53-2026090x*` series. Treat the README names as illustrative.
- `memory: 1800Gi` was inherited from the original. On 2026-09-02 the GLM
  run-1 launch found kueue counts requests against allocatable and at
  1800Gi only 11/304 p6 nodes were TAS-assignable; the GLM manifest dropped
  to 1200Gi. This manifest still says 1800Gi.
- Unverified on miles: 2 engines x TP4 + dp-attention for Qwen3.5 (see the
  sglang#21039 concern above), the `qwen3_5` slow-path loss mask on real
  gym traffic (Harbor workers always emit `has_generate_tokens` steps, so the
  fast path is the one exercised), and the ~328 GB per-experiment checkpoint
  footprint quoted from the AREnATasks staging README.
