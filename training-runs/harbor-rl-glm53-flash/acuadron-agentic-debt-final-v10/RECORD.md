# Run record: acuadron-agentic-debt-final-v10 — prepared, not submitted

**Status:** Prepared 2026-10-08 (hand-off: `examples/arena/harbor-rl-glm53-flash/kdatp/sgl/ARMS-20261008.md`).
**Base:** `acuadron-agentic-debt-final-v9` (workflow `dx8b7`); fresh start (no `load`).
**Files:** `workflow-512.yaml` (batch 512, 16 + 48 nodes; the engine arms are commented at the end of `miles-config`, fill in the A/B winner), `workflow-256.yaml` (batch 256, 16 + 64 nodes), `miles-config.yaml` (the 512 variant).
**Config deltas vs final-v9 (512 variant):** image r22 -> r24; `calculate_per_token_loss: true`; `disable_grpo_std_normalization: true`; `rollout_function_path` -> `NatsRolloutFn`; `max_weight_staleness: 16`; `arena_inflight_multiplier` 4 -> 8; `gym-replicas` 288 -> 576; `arena_routing_workers: 16`; run names.
**Expected step:** ~13 min at batch 512 if the engines reach 2x usable groups per engine-hour (ARMS-20261008.md); ~20 min with final-v9's engines. ~55 min to step 0.
