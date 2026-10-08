# Run record: acuadron-agentic-debt-final-v10

**Status:** Running — `acuadron-agentic-debt-final-v10-62jdk` from `workflow-512-noarms.yaml` (created 21:24 UTC 2026-10-08).
**Family:** `harbor-rl-glm53-flash`. **Base:** final-v9 recipe (workflow `dx8b7`); fresh start from the base DCP (no `load`).
**Template:** `acuadron-miles-deployer-v13`. **Trainer image:** `arena-github/miles:miles-glm53-r24-20261008a` (290940d7).
**Dataset:** 110-chain subset, manifest commit `f75dbe74…` (as v9).

## Config deltas vs final-v9 (the live 62jdk run)

| key | v9 | v10 |
| --- | --- | --- |
| replicas / replica-trainer | 64 / 16 | 64 / 16 (48 engines) |
| gym-replicas | 288 | 576 |
| rollout_batch_size / global_batch_size | 64 / 512 | 64 / 512 |
| arena_inflight_multiplier | 4 | 8 (512 groups in flight, 1.125 gyms per group) |
| rollout_function_path | generate_rollout | NatsRolloutFn |
| max_weight_staleness | — | 16 |
| calculate_per_token_loss | — | true |
| disable_grpo_std_normalization | — | true |
| arena_routing_workers | — | 16 |
| trainer-image | r22 | r24 |

Engines unchanged from v9 (trtllm DSA backends, flashinfer_cutlass, allreduce fusion, NEXTN 3/4, bf16 KV, mem 0.7).
lr 1.5e-6, save_interval 10, arena_length_reward_coef 0 unchanged.

## 2026-10-08 launch sequence

1. `gxsnq` 19:34 UTC — batch 256 variant (80 nodes, multiplier 12, 432 gyms). Admitted, all pods ready; stopped at
   20:22 before step 0 when the user asked for batch 512 (`workflow-512.yaml`, ARMS-20261008.md step 3).
2. `4hcn6` 20:38 UTC — `workflow-512.yaml` with every engine arm (DP attention, dp 8, deepep, TBO, spec 5/6, fp8 KV,
   mem 0.85); steps 1-2 of the hand-off skipped on request. Engines died at scheduler init:
   `NotImplementedError: Runner backend MoeRunnerBackend.FLASHINFER_CUTLASS requires a fused func for a2a backend
   deepep, but none is registered` — the same failure bench 20261008b had recorded for all three `live-dpa*` arms.
   Stopped 20:49.
3. `b98fw` 20:53 UTC — `moe_a2a_backend: flashinfer` (the fused pair this SGLang registers for `flashinfer_cutlass`;
   it requires dp_size == tp_size with DP attention), TBO dropped, the rest kept. 48/48 engines served, weight push
   42.5 s, 576 gyms, rollout 0 started. Speculation collapsed: accept len 1.21 -> 1.45, accept rate 0.04 -> 0.09
   over 10 minutes (v9: 2.62 / 0.54); decode 184 tok/s per DP rank = ~1,470 per engine at 48 running against v9's
   ~2,450 while decoding. The arm fails the hand-off's speculation gate and is slower than v9's engines. Stopped
   21:19 before rollout 0 completed; its ~2,000 published trials sit under `wv0000` of this experiment name.
4. `62jdk` 21:24 UTC — `workflow-512-noarms.yaml`: the v9 engines with every trainer-side change above.

Bench follow-ups: the other session's job `kdatp-sgl-20261008c` (`live-dpa-nodeep`, `live-dpa-humming`,
`live-fp8w-triton`); mine `kdatp-sgl-20261008d` (`live`, `live-dpa-fi`, `live-dpa-fi-fp8kv`, `live-dpa-fi-spec5-6`)
to separate fp8 KV, speculation depth and DP attention as the cause of the collapse. Results:
`s3://arena-scratch-prod-bom-ap-south-1/acuadron/kdatp/sgl/20261008{c,d}/SUMMARY.md`.

## Expected

~20 min steps with v9's engines at batch 512 (ARMS-20261008.md); ~13 min if a working engine arm is found and the
run is relaunched on it at a checkpoint boundary. ~55 min to step 0. W&B carries `rollout/population/*`.

## 2026-10-08 23:35 — 16 + 96 requested

`workflow-512-96.yaml`: replicas 112 (16 actor + 96 engines), `arena_inflight_multiplier` 9 (576 groups in
flight, 48 trials per engine), `gym-replicas` 648, otherwise `workflow-512-noarms.yaml`. Server dry run
accepted. The quota holds 303 nodes: guparpit 160, lijiahu 40, this run 64, engine-ab-v3 16, backfill 11,
free 14. 112 nodes therefore need a run of another owner to end. To avoid idling the 64 nodes, the swap is
conditional (`/tmp/glm53f/swap_v10_96.py`, log `/tmp/glm53f/swap-v10-96.log`): when a checkpoint exists
(iter >= 9) and >= 48 nodes are free, it stops the 64-node run and creates the 112-node workflow (same
experiment name, resumes from the checkpoint); if kueue does not admit it within 20 min, it falls back to the
64-node workflow. Steps lost to the swap are bounded to 3 after a checkpoint unless capacity has been free for
30 min.
