# ADR-0015: The task message carries the output cap, the window, and the sampling values

**Status:** Accepted
**Date:** 2026-09-27

**Amends:** ADR-0002 (the task envelope gains the optional top-level keys
`sampling_params` and `max_seq_len`)
**Pairs with:** AREnATasks ADR-0072 (the gym side of this contract, same
date) and the 2026-09-27 amendment of AREnATasks ADR-0063 (the gym clamps
`max_new_tokens` to the room left in the window)
**Reverses:** the r8 rejections R2 and R3 in
`examples/arena/harbor-rl-glm53-flash/experiment-list.md`

## Summary

The trainer puts the per-call output cap, the sampling values, and the
episode window into every Harbor training task message. The training gym
reads them from the message and keeps no env copy. Thus the miles config is
the only source of each value.

## Context

- **Two copies, no check.** The training gym read `ARENA_MAX_TOKENS`,
  `ARENA_ROLLOUT_CONTEXT_LIMIT`, `ARENA_TEMPERATURE` and `ARENA_TOP_P` from
  its env. The trainer holds the same values in `rollout_max_response_len`,
  `rollout_max_context_len` and `sglang_context_length`,
  `rollout_temperature`, and `rollout_top_p`. The gym manifest hardcoded
  `ARENA_MAX_TOKENS` "32768" next to a "keep in lockstep" comment. A drift
  between the two copies gave no error.
- **The fixed budget overflowed the window.** The gym sent
  `max_new_tokens` 32768 on every call, whatever the prompt length. On
  2026-09-26 and 2026-09-27 the r39 and r42 auctioneer runs sent about 500
  SGLang requests per 90 minutes with 98K to 131K input tokens plus the
  32768 budget. That total is over the 131072 window. About 99 percent of
  these requests were Vulcan summary calls. 38 to 48 percent of the
  compactions lost their handoff note.
- **Upstream precedent.** Upstream
  `examples/swe-agent-harbor-docker/swe_agent_function.py` sends
  `sampling_params` and `max_seq_len` to its Harbor agent server. Upstream
  `compute_request_payload` (`miles/rollout/generate_utils/generate_endpoint_utils.py`)
  clamps `max_new_tokens` to `rollout_max_context_len - len(input_ids)`.

## Decision

### 1. Wire

| Key | Shape | Source |
| --- | --- | --- |
| `sampling_params.max_new_tokens` | int | `rollout_max_response_len` |
| `sampling_params.temperature` | float | `rollout_temperature` |
| `sampling_params.top_p` | float | `rollout_top_p` |
| `max_seq_len` (top level) | int | the smaller of `rollout_max_context_len` and `sglang_context_length`, over the values that are set |

The key names copy upstream `swe_agent_function.py`. `build_task_message`
emits each key only when its argument is set. Thus an eval task
(`eval_coordinator._row_to_task`) stays byte-identical to before. An old gym
ignores both keys.

`max_seq_len` takes the smaller value, so the gym never plans a turn past
the SGLang window or past the trainer sample limit. Upstream clamps on
`rollout_max_context_len` only.

### 2. Publisher

`NATSRolloutWorker.__init__` computes the window once with
`_rollout_max_seq_len(args)`. The Harbor training publish in `_worker_loop`
passes the four values to `sample_to_task`. The eval path
(`eval_rollout.py`, `eval_coordinator.py`) does not change.

### 3. Fail at startup

`NATSRolloutWorker.__init__` raises `ValueError` before the tokenizer load
and before any publish in two cases:

- `rollout_max_response_len` is not set.
- Neither `rollout_max_context_len` nor `sglang_context_length` is set.

A gym that follows AREnATasks ADR-0072 fails every group whose message lacks
a value. One startup error is easier to read than one failure per group.

### Alternatives considered

| Alternative | Why not |
| --- | --- |
| Keep the gym env values and add a startup parity check | Two sources stay. The check needs the trainer config inside the gym pod. |
| Send `rollout_max_context_len` only, as upstream does | A run with a smaller `sglang_context_length` then overflows the engine. |
| New key names (for example `max_tokens`, `context_limit`) | Upstream already names the same values. New names add a second vocabulary. |
| A flag to turn the keys off | An old gym ignores the keys, so an off switch buys nothing. |

## Consequences

- One miles config value sets each limit and each sampling value. The next
  GLM run sets `rollout_max_response_len: 16384` in its `miles-config.yaml`
  and nowhere else.
- Deploy order: MUST deploy the trainer image with this change before a gym
  image with AREnATasks ADR-0072. An older trainer sends no keys, so that
  gym fails every group with a `ValueError` that names this ADR.
- A new trainer with an old gym image works as before. The old gym ignores
  the keys and reads its env values.
- NEVER run an old gym image on a template without the gym env entries.
  That gym falls back to the `ArenaSGLangLLM` defaults of 8192 output
  tokens and a 32768 window and gives no error.
- The W&B chart `raw_response_length/response_length_clip_ratio`
  (`log_utils.py`) compares the whole multi-turn response with the per-call
  cap. At 16384 it reads near 1. Only the chart is wrong.
- Tests: `tests/fast/plugins/arena/test_nats_arena.py`
  (`TestBuildTaskMessage`, `TestSampleToTask`, `TestRolloutMaxSeqLen`). No
  test drives the async publish loop, so the publish call site has no test.
