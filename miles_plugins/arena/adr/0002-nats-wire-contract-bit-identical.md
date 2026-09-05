# ADR-0002: NATS wire contract stays bit-identical to `amzn_arena_contract`

**Status:** Accepted
**Date:** 2026-09-01

**Builds on:** ADR-0001 (port scope and module mapping); AREnATasks
ADR-0039 (Harbor NATS gym worker — the worker side of the same contract)

## Summary

Every value that crosses NATS between this plugin and the arena gym workers
is carried over from `amzn_agi_slime.nats_arena` unchanged: stream names,
subjects, the trainer durable, the gzip-JSON task and result envelopes, the
salvage rule and the degenerate agent-stop vocabulary. The port changes
import paths and docstrings only, so deployed gym workers (AREnATasks
`amzn_arena_harbor.gym_worker`, `amzn_arena_streaming.cli`) serve a miles
trainer without a new image.

## Context

- The RL loop spans two repos. Gym workers live in AREnATasks and import
  the wire glue from `amzn_arena_contract.rl` (subjects `arena.tasks.<gym>`
  / `arena.results`, durable `gym-worker-<gym>`, `ARENA_NATS_ACK_WAIT`
  default 4500 s, task/result schema, `DEGENERATE_AGENT_STOPS`,
  `synthetic_trajectory`). The trainer side was `amzn_agi_slime.nats_arena`
  (`message_format.py` + `nats_rollout.py`), which AREnATasks cannot import;
  it pins the trainer literals BY VALUE in
  `tests/harbor_runtime/test_cross_runtime_parity.py::TestTrainerMirrorPins`
  (task-message field set incl. the `.g<counter>.` id suffix, result
  envelope keys, status vocabulary success/truncated/failed, the six
  degenerate stops, subjects + gym durable, ack-wait 4500, the
  synthetic-trajectory skip shape).
- NOT pinned by that test (verify-natsGrpo finding, 2026-09-01): the stream
  names `ARENA_TASKS`/`ARENA_RESULTS`, the trainer durable `slime-trainer`
  and its sizing (ack_wait 3600 s, max_deliver 3, max_msg_size 128 MiB).
  They still matter: the gym's stream-wait loop waits for trainer-created
  streams, and a JetStream durable carries delivery/ack state across
  trainer restarts.
- Port baseline: AGISlime `mainline-0.3.0` @ 90ce39f (2026-08-28 13:36 PT),
  cloned to `/workplace/guparpit/miles/src/AGISlime` 2026-09-01 00:13 PT.

## Decision

Keep every wire value bit-identical to the AGISlime original, including the
names that say "slime". Env overrides keep their names too.

| Value | Setting kept | Where (env override) |
| --- | --- | --- |
| Streams | `ARENA_TASKS` (work-queue, subjects `arena.tasks.*`, purged on fresh trainer start) / `ARENA_RESULTS` (limits, max_msg_size 134217728) | `nats_rollout._ensure_nats` (`NATS_TASKS_STREAM`, `NATS_RESULTS_STREAM`) |
| Subjects | `arena.tasks.<gym>` / `arena.results` | `NATS_TASKS_SUBJECT_PREFIX`, `NATS_RESULTS_SUBJECT`, `ARENA_DEFAULT_GYM` |
| Trainer durable | `slime-trainer`, explicit ack, max_deliver 3, ack_wait 3600 s | `NATS_RESULTS_CONSUMER` |
| Gym durable polled by the autoscaler | `gym-worker-<gym>` | `NATS_GYM_CONSUMER_PREFIX` |
| Task envelope | gzip JSON `{id, lakefs_uri, lakefs_commit_id, n_samples, metadata{rollout_id, group_index, gym_name}, session?}`; id published as `<instance_id>.g<counter>.` | `message_format.build_task_message` / `serialize_task` / `sample_to_task` |
| Result envelope | gzip-transparent JSON `{task_id, gym_name, status, trajectories, group_metrics, session?, error?}` | `message_format.parse_result` / `extract_trajectories` |
| Salvage rule | `SALVAGEABLE_RESULT_STATUSES = ("success", "truncated")`; `failed` dropped, only `error` logged | `message_format` |
| Degenerate stops | `{context_error, context_churn, empty_response, max_budget, timeout, no_choices}` -> TRUNCATED + remove_sample | `nats_rollout._DEGENERATE_AGENT_STOP` |
| DLQ deadline | `NATS_TASK_DEADLINE_SECS` default 12600 s | `nats_rollout` |

Evidence: `message_format.py` differs from the original in 16 lines, all
docstring text; the ported `sample_to_task` output parses losslessly through
`amzn_arena_contract.rl.parse_task_message`, `assemble_result` output parses
through `parse_result`, and the degenerate set equals
`DEGENERATE_AGENT_STOPS` (executed live against the AREnATasks source,
verify-natsGrpo, 2026-09-01).

### Alternatives considered

| Alternative | Why not |
| --- | --- |
| Rename the durable to `miles-trainer` | Orphans the live consumer state on the first miles start against an existing NATS (in-flight deliveries stay bound to the old durable until ack_wait expires). No worker or test references the name, so the rename buys nothing and — because no AREnATasks test pins it — a later "fix" would go unnoticed. |
| Version the envelope (schema field or a `trainer: miles` marker) | Workers parse the exact field set; a new field forces a coordinated AREnATasks release plus parity-literal update for zero functional gain. Trainer identity already travels out of band (session token, `EXPERIMENT_NAME`). |
| Import `amzn_arena_contract` from the plugin instead of mirroring | Runtime dependency on an internal Brazil package absent from the miles images; AREnATasks itself keeps the trainer as a by-value mirror ("the AGISlime mirror is cross-repo and stays a mirror"). |
| Drop NATS for an in-process rollout function | Out of scope for the port (ADR-0001); abandons the per-gym Deployment / autoscaling architecture (AREnATasks ADR-0039 alternatives). |

## Consequences

### Easier

- Existing gym images serve the miles trainer unchanged. Proven CPU-side on
  2026-09-01 against a real NATS 2.14.6 JetStream server with a fake worker
  written from the contract: financeagent harness 49/50 checks (the one
  failure is a teardown ack artifact identical in AGISlime); snorkel
  plain-stack harness 145/145, asserting the topology (`ARENA_TASKS`
  workqueue + `arena.tasks.*`, `ARENA_RESULTS` limits + 128 MiB, durable
  `slime-trainer` explicit/3/3600, gym durable
  `gym-worker-snorkel-general-bash-harbor` 110/110 acked). GPU proof
  (`rl-milesgb1-smoke1`, 2026-09-01 16:39 -> 09-02 00:34 PT, unchanged run5
  gym workers) is in the harbor-rl-27b-snorkel run log.
- Three-way pin against drift: `amzn_arena_contract` (source), the
  AREnATasks parity literals, and this plugin's CPU suite
  (`tests/fast/plugins/arena/test_nats_arena.py`: TestBuildTaskMessage,
  TestSerializeParseRoundTrip, TestSampleToTask, TestSalvageableStatuses).

### Harder / open

- `slime-trainer` is a permanent misnomer inside miles; documented here and
  in the plugin README so nobody "fixes" it.
- A contract change needs three coordinated edits (contract package, parity
  literals, this plugin).
- Inherited: the trainer purges both streams on every fresh start, so two
  trainers must never share a NATS deployment — every run gets its own
  (`nats://rl-<prefix>-nats:4222`).
- Trainer and gym size their windows independently: trainer ack_wait 3600 s
  / max_deliver 3 / DLQ 12600 s vs gym `ARENA_NATS_ACK_WAIT` 4500 s. A
  larger gym window (6000 s in the GLM r6 shape) must stay under the
  trainer's `NATS_TASK_DEADLINE_SECS`, or the trainer expires groups that
  are still running.
- Two ops lessons recorded later in the GLM run log follow from the durable
  design kept here: restarting the NATS Deployment drops the gym durables
  with the emptyDir store (r3, 2026-09-03), and scaling gyms to zero strands
  ack-pending deliveries for a full ack_wait (r6, 2026-09-04).
