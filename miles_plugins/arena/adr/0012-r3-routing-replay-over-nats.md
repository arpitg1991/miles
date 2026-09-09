# ADR-0012: R3 rollout routing replay over the arena NATS path

Status: Proposed (2026-09-09). Investigation complete, no code yet.

## Context

Every GLM-5.3-Flash run r9-r14 (`examples/arena/harbor-rl-glm53-flash/`)
logs `train/ppo_kl` about 0.008 and `train/pg_clipfrac` about 1.1 percent at
step 0 and a flat reward. Upstream `ci_utils.check_kl` asserts both are below
1e-8 on-policy. The residual is train/inference mismatch. For a MoE model the
dominant term is route flips: SGLang and Megatron pick different top-k
experts for the same token. R3 (arXiv 2510.11370) removes it by replaying the
rollout engine's routing in the trainer forward and backward.

r15 (2026-09-09) takes the cheap fix first: `use_tis` with eps 0.2/0.28, the
upstream GLM-5.2 recipe. This ADR records what R3 needs on our path so r16 can
add it without a second investigation. Sources: our fork at `ff19a83f`,
upstream `radixark/miles` main at `4f8e0b86f` (279 commits ahead), AREnATasks
`arpit-glm-53`, AGISlime `origin/mainline-0.3.1`.

## How upstream R3 works

Flags (`miles/utils/arguments.py:1600-1624`):

| Flag | Effect |
| --- | --- |
| `--use-rollout-routing-replay` | R3. Sets `use_routing_replay`. Adds SGLang `enable_return_routed_experts` (`sglang_engine.py:824`). Each `/generate` sends `return_routed_experts: true`. |
| `--use-rollout-indexer-replay` | Replays the DSA indexer top-k. Adds `enable_return_indexer_topk`. Asserts `context_parallel_size == 1`. |

Data contract (`miles/utils/types.py:50-55`):

- `Sample.rollout_routed_experts`: int32 `(len(tokens) - 1, num_layers, moe_router_topk)`.
  `num_layers` counts every decoder layer, dense layers included. Row i is the
  routing that produced token i+1. The payload covers prompt and response,
  including prefix-cached tokens.
- `Sample.rollout_indexer_topk`: int32 `(len(tokens) - 1, num_dsa_layers, index_topk)`,
  0-based KV positions, -1 pads.

Wire format: `meta_info.routed_experts` is base64 of a flat little-endian
int32 buffer. Decode with `_decode_topk_buffer` in
`miles/rollout/generate_utils/generate_endpoint_utils.py:114-121`. SGLang can
return one extra stop-edge row; `miles/rollout/sglang_rollout.py:258-263`
trims it.

Multi-turn: every turn returns the full prefix, so `merge_samples` keeps the
last turn's array (`generate_utils/sample_utils.py:141-169`). The session
server requests deltas with `routed_experts_start_len`; the NATS path does not
use the session server.

Trainer: `train_data_conversion.py:136-149` copies the field into
`rollout_data` and (fork only, commit `4ac3f4b7f`) raises when the flag is on
and the payload is missing. `replay_data.fill_replay_data` pads one -1 row,
slices by CP and SP, and feeds per-layer `Replay` queues.
`register_replay_list_moe` maps global layer index to local MoE layers. The
Megatron `TopKRouter` hook lives in `radixark/Megatron-LM` branch
`miles-main`, not in this repo. The ref model runs without replay, so
`train/kl_loss` is not zero under R3 and `check_kl` skips it.

Upstream recipes: `run_glm47_flash.py`, `run_deepseek_v4.py`, `run_kimi_k25.py`
turn R3 on. `docs/advanced/miles-router.md:53-60`: for MoE plus GRPO
"current recipes turn R3 on". `docs/developer/debug.md:161`: MoE collapse
after tens of steps means check `--use-rollout-routing-replay`.

Upstream deltas we lack: only `73e0f87b8` (session-server incremental R3 for
all non-retract pause modes). It does not touch the NATS path. Upstream has no
`scripts/run_glm5_3_flash.py` and no `glm5_next` plugin; both came from PR
#2786 merged in the fork only.

## Sizing for GLM-5.3-Flash

From `GLM-5.3-Flash-BF16/config.json`: 45 layers, `first_k_dense_replace` 3,
288 routed experts, `num_experts_per_tok` 8, 11 DSA layers, `index_topk` 2048,
`index_kpool` 4.

| Payload | Bytes per token | 40k-token episode | GBS 512 |
| --- | --- | --- | --- |
| routed experts (45 x 8 x 4) | 1,440 | 58 MB | 29 GB |
| indexer top-k (11 x 2048 x 4) | 90,112 | 3.6 GB | 1.8 TB |

Base64 adds one third on the wire. gzip does not help.

## Decision

1. Implement routing replay only. Indexer replay is 60x larger, every upstream
   recipe marks it debug-only ("78-128 GB/rank host buffer OOMs",
   `run_glm5_2_744b_a40b_lora.py:240`), and sgl-router strips
   `return_indexer_topk`. Accept the residual indexer mismatch. TIS stays on
   to cover it.
2. Carry the payload out of band. 58 MB per episode exceeds the 8 MiB NATS
   broker `max_payload`. Use the shared-filesystem ref protocol that
   `amzn_arena_streaming.routing.prepare_routed_experts_for_transport`
   already defines: `steps[-1]["routed_experts_ref"] = {path, bytes, sha256}`
   under `ARENA_ROUTING_DIR`, on a mount the gym and trainer share.
3. Keep `--arena-train-segments all`. Each compaction segment is its own
   `Sample` with its own token stream, so each archived segment needs its own
   cumulative blob.

## Gaps and where each change goes

Gym side (AREnATasks, new image):

- G1 `sglang_rollout.py:136-171` `RolloutState.begin_segment`: snapshot
  `self.routed_experts` into the archived segment. Today only the final
  trajectory-level blob survives (ADR-0048 in AREnATasks, line 358).
- G2 `gym_worker.py:320-333` `trial_to_trajectory`: after `_rollout_steps`,
  call `prepare_routed_experts_for_transport` per step when
  `ARENA_ROUTING_DIR` is set. Port from the streaming worker; the Harbor
  worker never imports `amzn_arena_streaming.routing`.
- G3 `gym-worker.yaml`: mount the shared scratch path and set
  `ARENA_ROUTING_DIR`. Trainer pods must see the same path.

Trainer side (miles fork, new image):

- T1 `miles_plugins/arena/nats_arena/message_format.py:24-77`
  `build_task_message`: emit `capture_routed_experts: true` when
  `args.use_rollout_routing_replay`. The gym already honors it
  (`amzn_arena_contract/rl/messages.py:83`, `gym_worker.py:606,615`).
- T2 `nats_rollout.py` `_step_to_sample` (~L410-600): read
  `step["routed_experts"]` or `step["routed_experts_ref"]`, decode with
  `_decode_topk_buffer`, reshape `(len(tokens) - 1, args.num_layers,
  args.moe_router_topk)`, trim the stop-edge row, apply after the hard
  overflow clip at L459-475. Missing payload on a trainable sample is fatal
  (`RoutingReplayError`), not a swallowed `n_failed`
  (`nats_rollout.py:1806-1809`).
- T3 Pads: `_process_group` (L1821-1850) and dp pads (L1995-2020) must carry
  a zero array of matching shape, because conversion gates on `samples[0]`
  and `fill_replay_data` asserts every packed sample. AGISlime
  `_materialize_group_routing` (mainline-0.3.1 commit `56501fb`) is the
  reference: decode refs at drain time, unlink the file, share a zeroed
  sibling array for pads.
- T4 `miles-config.yaml`: `use_rollout_routing_replay: true`. Combine with
  `use_tis: true` (the two are independent; both recommended in
  `docs/advanced/low-precision.md:111`).
- T5 Verify the engines start with `enable_return_routed_experts` and that
  the MoE runner backend is not top-k bypassing. Fork commit `0dd04cc2a`
  asserts a non-zero payload.

## Validation

- Unit: fake a `steps[-1].routed_experts` base64 buffer with
  `pybase64.b64encode(np.arange(...).astype(np.int32).tobytes())` and assert
  the decoded shape (pattern in `tests/fast/rollout/generate_utils/test_indexer_replay.py:15`).
- Smoke: 2x4 shape, `--limit 10`, one step. Success is `train/ppo_kl` at
  step 0 dropping from about 0.008 toward 1e-3 or lower, and
  `train/pg_clipfrac` near 0. `train/kl_loss` is not a valid check under R3.
- Cost: 29 GB per step of int32 through the shared mount and into pinned CPU
  on each actor rank. Watch `train_memory_margin` and the scratch mount.

## References

- AGISlime `origin/mainline-0.3.1` `nats_rollout.py` L66-263 (`_decode_routed_experts`, `_materialize_group_routing`).
- AREnATasks `0bd5e57` (streaming capture, `routing.py`), ADR-0036, ADR-0039 L204 (offload deferred), ADR-0048 L358.
- miles fork commits `cee16d9d0`, `0dd04cc2a`, `4ac3f4b7f`; upstream `73e0f87b8`.
