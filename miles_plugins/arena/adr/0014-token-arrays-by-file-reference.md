# ADR-0014: Token arrays ride a file reference; the trainer reads both shapes

**Status:** Accepted
**Date:** 2026-09-25

**Amends:** ADR-0002 (the task envelope gains an optional top-level
`token_arrays_by_ref`; a step can carry `token_arrays_ref` in place of
`token_ids`, `loss_mask` and `log_probs`; a trajectory can omit `messages`),
ADR-0012 (the ref reader, the `ARENA_ROUTING_DIR` containment check and
`reap_result_refs` also serve `.tokens` files; token refs load eagerly at
conversion, routing refs stay lazy)
**Builds on:** ADR-0007 (a removed sample's reward enters the group
baseline), ADR-0011 (`_training_steps` and `--arena-train-segments`),
commit `db3955b70` ("feat(arena): log and count why the gym dropped a
trajectory", the drop-reason categories; no ADR records them)
**Pairs with:** AREnATasks ADR-0071 (the gym side of this contract, same date)

## Summary

The trainer sets `token_arrays_by_ref: true` on every training task message.
A gym that knows the flag stages the three per-step token arrays of a
training trajectory as one raw binary file under `ARENA_ROUTING_DIR` and
sends a small ref. It also leaves out `messages` when the trainer cannot
read them. The trainer reads, checks and deletes each file when it converts
the result. A step without the ref goes through the inline path unchanged,
so every gym image works with this trainer.

## Context

- **The result store rejects large records.** `ARENA_RESULTS` is a
  file-backed JetStream stream. nats-server 2.11.3 rejects every stored
  record over 32 MiB (`filestore.go` `rlBadThresh`). The limit is not
  configurable. `max_payload` (64 MiB) and the stream `max_msg_size`
  (128 MiB) never apply below it.
- **r34 lost most of its groups.** r34 (agentic-debt v1 chains) lost 58 of
  92 groups to that limit. One heavy group was 33.5 MB gzip and 109 MB JSON:
  63 segments and 3,975,555 tokens. The per-segment arrays dominate:
  `log_probs` 61 percent, `token_ids` 20 percent, `loss_mask` 11 percent,
  chat `messages` 8.5 percent. v3-locked chains (mean about 7 steps, up to
  66) are larger.
- **The precedent is routing replay.** ADR-0012 already moves the R3
  routing blob through a file on the shared nfs4 mount
  (`/mnt/scratch-s3files-rw`) with a `{path, bytes, sha256}` ref. The same
  reader, the same checks and the same reap fit the token arrays.

## Decision

### 1. Wire

| Direction | Key | Shape | When |
| --- | --- | --- | --- |
| trainer to gym | `token_arrays_by_ref` (top level) | `true` | every training task; `build_task_message` emits it only when True, so an eval task (`eval_coordinator._row_to_task`) is byte-identical to before |
| gym to trainer | `steps[j].token_arrays_ref` | `{path, bytes, sha256, n_tokens}` | a training step of a new gym with `ARENA_ROUTING_DIR` set; it replaces `token_ids`, `loss_mask` and `log_probs` |
| gym to trainer | trajectory `messages` | absent | only when `steps` is non-empty and `steps[-1]["has_generate_tokens"]` is truthy |

Every other step key stays inline: `routed_experts_ref` or
`routed_experts`, `has_generate_tokens`, `stop_reason`, `segment_end`,
`weight_versions` and the counters. The envelope keys do not change.

File layout, one file per step, no header, little-endian:
`log_probs` float64 at offset 0, `token_ids` int32 at `8n`, `loss_mask`
uint8 at `12n`. `bytes == 13 * n_tokens`. float64 keeps every value
bit-identical to the double that `json.loads` gives for the inline list.

### 2. Reader (`routing_replay.resolve_token_arrays`)

1. The ref must have a non-empty string path with the `.tokens` suffix. When
   `ARENA_ROUTING_DIR` is set, the resolved path must be inside it. A ref
   that fails this check is neither read nor deleted.
2. The byte count must equal `bytes` and the sha256 must equal `sha256`.
3. `n_tokens` must be an int (not a bool), at least 1, and `13 * n_tokens`
   must equal the byte count.
4. After check 1 passes, the file is always deleted in a `finally` block,
   whether the read is clean or not (the ADR-0012 rule).
5. A clean read replaces the ref with three Python lists. The existing
   `_step_to_sample` checks then run unchanged.

`_load_token_arrays` in `nats_rollout.py` calls the reader for the steps that
`_training_steps` selects, just before `_result_to_episodes_full_trajectory`.
The read is eager because the Sample build reads `loss_mask`. The RAM cost
equals the JSON decode of the inline arrays that it replaces.

### 3. A bad file drops one trajectory, never the run

Every reader failure raises `TokenArraysRefError`. That class is NOT a
`RoutingReplayError` subclass: the worker loop stores a `RoutingReplayError`
as `fatal_error` and stops the run, and `_process_group` re-raises it. A lost
token file loses the tokens of one trajectory only. `_load_token_arrays`
reaps the routing and unread token files of that trajectory and puts a
synthetic pad in its slot with `error = "token_arrays_ref unusable: ..."`.
`_process_group` then pads the slot with a removed sibling copy, and the
siblings train. The pad has reward 0.0 inside the group baseline (ADR-0007),
the same as every other gym-side synthetic slot.

### 4. Why the trainer never needs the dropped `messages`

The trainer reads `messages` in two places only:

- `_is_synthetic_trajectory`: `stop_reason == "errored"` and no `messages`
  and no `steps`. The gym drops `messages` only when `steps` is non-empty,
  so a real trajectory never looks synthetic.
- The slow path of `_result_to_episodes_full_trajectory`, which runs only
  when `training[-1]` lacks `has_generate_tokens`.

With `steps[-1]["has_generate_tokens"]` truthy, the fast path runs in both
`--arena-train-segments` modes. Under `all` with a `segment_end` marker,
`training` keeps every step with `has_generate_tokens`, so its last item is
`steps[-1]`. Under `final`, or without a marker, `training` is `steps[-1:]`.
The slow path therefore cannot run for a trajectory that the gym stripped.
`test_chain_without_messages_takes_the_token_path_in_both_modes` pins it.

### 5. Cleanup

`reap_result_refs` deletes the files of both step keys, so every existing
drop path covers `.tokens` files: the unknown-tid stale result, the prior
session, the DLQ sweep, the failed status, the conversion error, and the
untrained steps under `final`. `_load_token_arrays` adds the per-trajectory
drop.

A result for an accepted tid reaps its token files only
(`reap_result_refs(keys=(TOKEN_REF_KEY,))`). Two cases reach that path:

- A redelivery of the same message. The conversion already read and deleted
  its token files, so the reap finds nothing.
- A second run of the task, for example when a NATS restart or a pod kill
  loses the task ack after the gym published the result. Its token files
  have new names, and nothing else reads or deletes them.

Its routing files stay, because the queued group of the accepted copy reads
them at drain time (the r27 and r28 rule of ADR-0012). The trainer cannot
tell the two cases apart, so the routing files of a second run stay on the
mount, an ADR-0012 orphan path.

### 6. Telemetry

Three drop-reason categories join `_FAILED_REASON_CATEGORIES` and give the
W&B keys `rollout/failed/<category>` and `rollout/dropped_groups/<category>`:

| Category | Gym or trainer text | Source |
| --- | --- | --- |
| `too_large` | `result too large: <N> bytes > 31457280` | the gym's 30 MiB failed envelope (AREnATasks ADR-0071) |
| `staging` | `result staging failed: <OSError>` | the gym's failed envelope for a write fault on the shared mount (AREnATasks ADR-0071) |
| `token_ref` | `token_arrays_ref unusable: <reason>` | `_load_token_arrays` |

The three patterns match first. The full phrase "result too large" keeps a
Docker "request entity too large" text in `other`. Commit `db3955b70` added
the category set, and no other ADR records it. The full set is now
`deadline`, `no_reward`, `missing_rollout`, `chain_abort`, `too_large`,
`staging`, `token_ref`, `other` and `unknown`.

### Alternatives considered

| Alternative | Why not |
| --- | --- |
| A `--arena-token-arrays-by-ref` flag or env knob | The trainer reads both shapes and an old gym ignores the key, so an off switch buys nothing. |
| Lazy load at drain time, like routing | The Sample build needs `loss_mask` for `prompt_len` and `response_length`, and pads copy `tokens`. |
| `TokenArraysRefError` as a `RoutingReplayError` subclass | A lost 13-byte-per-token file would stop the run. |
| Drop the whole group on a bad file | One result is one group; a single lost file would lose all 8 trajectories. |
| A second reader module | Duplicate containment, checks and reap code. The reader takes the step key as a parameter. |

## Consequences

- Every gym and trainer pairing works. An old gym sends inline arrays and
  `messages`; this trainer takes today's path. A new gym without the flag
  (an old trainer) sends today's JSON, except that a result over 30 MiB
  becomes a failed envelope.
- An old gym still loses every group over 32 MiB. This change cannot fix
  that on the trainer side.
- If the gym and trainer `ARENA_ROUTING_DIR` differ, every trajectory
  becomes a `token_ref` drop, so every group drops as `token_ref`. If the
  gym cannot write to the mount, every group becomes a `staging` drop.
  Nothing fails fast in either case. The live smoke MUST check that
  `rollout/failed/token_ref`, `rollout/dropped_groups/token_ref` and
  `rollout/dropped_groups/staging` stay 0, and that the directory drains
  after each step.
- `_process_group` runs on the NATS event loop. The token reads replace a
  larger gunzip and JSON decode, but an NFS stall now blocks the loop.
  ponytail: the upgrade path is `asyncio.to_thread` for `_process_group`
  after a lock review of `pending_*` and `output_queue`.
- Orphan paths stay the ADR-0012 ones (publish failure, SIGKILL, a crash
  between ack and processing, stream purge or eviction). Token files add
  about 1 percent to the routing bytes. Operators clear
  `ARENA_ROUTING_DIR` between runs.
- Tests: `tests/fast/plugins/arena/test_routing_replay.py` (flag, decode,
  checks, reap paths, per-trajectory drop) and
  `test_nats_arena.py::TestFailedReasonTelemetry` (categories and the
  `too_large` group drop).
