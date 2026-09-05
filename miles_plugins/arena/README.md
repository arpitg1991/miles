# Arena plugin

Port of the `amzn_agi_slime` package from AGISlime (Amazon's overlay on
vendored THUDM/slime 0.3.0) onto miles. It provides the arena RL training
path: rollouts are executed by external gym workers (e.g. financeagent) that
receive tasks and return trajectories over NATS JetStream, while miles owns
training, weight updates, and SGLang serving.

## Module mapping

Fixed 1:1 rule used by the whole port:

```
amzn_agi_slime.X  ->  miles_plugins.arena.X
```

including the `nats_arena/` subpackage, e.g.
`amzn_agi_slime.nats_arena.nats_rollout.generate_rollout` ->
`miles_plugins.arena.nats_arena.nats_rollout.generate_rollout`.

Modules in this package:

- `nats_arena/` — NATS rollout worker, data source, wire format
  (`message_format`), gym autoscaler, mixture controller, eval
  coordinator/rollout, Argo eval trigger, binary reward post-processors.
- `train_async_arena.py` — arena training driver (asyncio, mirrors
  `train_async.py` with arena hooks).
- `rewards.py` — torch-free `binarize_reward` (also re-exported lazily from
  `miles_plugins.arena`).
- `parsers.py` — Nova reasoning parser for SGLang (`register_nova_reasoning_parser`).
- `checkpoint_extras.py` — JSON sidecar (`iter_%07d/slime_extra_state.json`)
  persisting wandb run id / rollout counters across resumes. Filename is kept
  byte-identical for interop with AGISlime-written checkpoints.
- `eval_metrics_drain.py` — trainer-side drain of the eval-metrics file queue.
- `s3_artifact.py` — direct-to-S3 checkpoint/trace upload helpers.
- `logging_extensions.py`, `rollout_metrics.py` — logging and rollout metric
  helpers used by the NATS rollout path.

## Configuration knobs

Wire the plugin into miles through the standard function-path arguments:

```
--rollout-function-path miles_plugins.arena.nats_arena.nats_rollout.generate_rollout
--data-source-path      miles_plugins.arena.nats_arena.data_source.ArenaDataSourceWithBuffer
```

Optional:

```
--custom-reward-post-process-path miles_plugins.arena.nats_arena.reward_binary.binarize_reward
```

NATS connectivity comes from the environment (`NATS_URL`, plus optional
`NATS_TASKS_STREAM` / `NATS_TASKS_SUBJECT_PREFIX` / `NATS_RESULTS_STREAM` /
`NATS_RESULTS_SUBJECT` / `NATS_RESULTS_CONSUMER` / `ARENA_DEFAULT_GYM`
overrides). Defaults match the AGISlime wire contract: streams
`ARENA_TASKS` / `ARENA_RESULTS`, subjects `arena.tasks.<gym>` /
`arena.results`, durable consumer `slime-trainer`. The wire format is
bit-identical to the AGISlime original, so existing gym workers need no
changes.

## harbor-rl-27b usage

The Qwen3.5-27B + financeagent smoke job (gym workers over NATS) lives under
`examples/arena/` — see that directory for the launch script, training config,
and Kubernetes manifests corresponding to AGISlime's
`tmp_staging/smoke/harbor-rl-27b` setup.

## Installation

Runtime extras for this plugin (NATS, Kubernetes, lakeFS, S3):

```
pip install -e ".[arena]"
```

The core rollout path needs `nats-py`; `kubernetes` is used by the gym
autoscaler and eval/Argo integrations, `lakefs` by the manifest data source,
and `boto3` by the S3 artifact helpers — all are imported lazily, so
non-arena runs never require them.

## Known limitations and follow-ups

Findings from the adversarial verification of the port (see the verify
reports for full evidence):

- **GRPO loss weighting is per-trajectory, not per-group.** The vendored
  AGISlime overlay intended per-group token-weighted loss means; the port
  uses miles-standard per-trajectory means (a plugin cannot change this).
  Needs explicit job-owner sign-off — mitigating fact: the vendored baseline
  provably could not complete a training step on this smoke config (its
  dp-schedule assert fires with 8 groups < GBS 64), so per-group weighting
  was never demonstrated behavior.
- **Data-source buffer is not persisted across restarts** (parity with the
  original): groups sitting in `ArenaDataSourceWithBuffer`'s buffer at
  checkpoint time are silently dropped on restart.
- **Resume dedup guard is mostly vacuous** (parity): `save()` persists the
  publisher's offsets/epochs, which run ahead of training, so restored dedup
  keys only match consumed keys when the publisher position overlaps the
  trained epochs — queued-but-untrained work is skipped over on restart.
- **`--gym-autoscale-auto-tune` cannot be switched off via YAML** (parity):
  it is a `store_true` flag with default `True`, so `false` in the config
  emits no flag and auto-tune stays on.

Recommended follow-ups:

- Make the eval coordinator a wandb **shared-mode secondary writer**
  (miles' `init_wandb_secondary` pattern) logging `eval/*` directly,
  replacing the `eval_metrics_drain` file-queue/heartbeat protocol.
- Ride miles' native `CheckpointEvalFn`/`EvalDispatcher` seam for the
  per-checkpoint Argo eval instead of the fire-and-forget trigger, removing
  the display-name-based wandb run resolution.
- Convert `generate_rollout` into a class-based `RolloutFn` so it receives
  the authoritative `weight_version` from `RolloutFnTrainInput` instead of
  inferring off-policy staleness from `rollout_id`.
