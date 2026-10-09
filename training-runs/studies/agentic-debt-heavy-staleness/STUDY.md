# Study: training under heavy weight staleness (agentic-debt, GLM-5.3-Flash)

**Status:** done (arms admitted 06:34 UTC after 5 h in the queue, stopped 08:46 UTC at the user's request for the nodes; 12-15 steps per arm)
**Question:** how far can the rollout policy lag the trainer before the update degrades, and what makes the lag safe?
The lag sets the step time: with 70-min mean episodes (160 min p90), a 13-min step forces a mean staleness
of ~5 and a 5-min step a mean of ~14 (p90 32). A shorter step is the next efficiency lever once the engines
and the trainer are both busy (`agentic-debt-v9-sample-length-growth`, `engine-ab-v3`), and it is only
available if training is sound at that staleness.

## What the production runs already say (W&B, final-v9 `i8ufpnjc` and v23 `3tirhnqh`)

Train-rollout mismatch against weight age at lr 1.5e-6, TIS cap 2.0, routing replay on:

| run | step | mean age (max) | gap `train_rollout_logprob_abs_diff` | KL | `tis_clipfrac` |
|---|---|---|---|---|---|
| v9 (no cap) | 0 | 0 | 0.032 | 0.0045 | 0.0008 |
| v9 | 10 | 5.2 (9) | 0.034 | 0.0044 | 0.0009 |
| v9 | 25 | 7.3 (12) | 0.031 | 0.0031 | 0.0004 |
| v9 | 30 | 7.3 (27) | 0.040 | 0.0046 | 0.0007 |
| v9 | 40 | 6.3 (27) | 0.042 | 0.0045 | 0.0006 |
| v23 (cap 8) | 0 | 0 | 0.029 | 0.0039 | 0.0007 |
| v23 | 30 | 4.7 (8) | 0.037 | 0.0050 | 0.0007 |
| v23 | 60 | 4.3 (8) | 0.043 | 0.0089 | 0.0017 |

The gap does not move with age. Groups 27 versions stale (v9, steps 30-40) sit on the same 0.03-0.04 as
fresh groups; the slow upward drift over the run is the same in v23 at a fixed age ~4.4, so it comes from
the trained weights, not from staleness. At this learning rate the policy moves less in 27 updates than the
bf16 inference-training numerics floor (KL ~0.0045 at age 0). The clip fraction (ratio above 2) stays
under 0.1% of tokens. engine-ab-v2 (lr 1e-5, mean age ~6) is the counterexample: gap 0.34 -> 3.0 -> 5.6,
grad_norm 0.12 -> 56, collapse by step 20. The product lr x age decides, not age alone.

## Literature (fetched 2026-10-09)

- **M2PO** (Zheng, Zhao, Chen, arXiv:2510.01161): "prosperity before collapse" — stale data (256 updates)
  trains as well as fresh data until the trust region fails; the fix masks, largest first, the
  trust-region tokens (A>0 & r>1 or A<0 & r<1) until the batch mean of `(log r)^2` is at most 0.04, and
  keeps the unclipped IS weight elsewhere. Clip fraction 1.22% -> 0.06%, six models 1.7B-32B, lr 1e-6,
  batch 512, 8 samples per prompt — the same scale as ours. GSPO is the only baseline that stays stable at
  256 (with a loss); GRPO, TOPR, CISPO, AReaL's decoupled PPO and GPPO degrade or collapse there.
- **AReaL** (Fu et al., arXiv:2505.24298): fully async; staleness bounded by a max age; decoupled PPO
  objective (clip against the proximal policy, IS weight against the behavior policy) is what keeps up to
  ~8 steps of staleness harmless. miles' one-epoch loss plus TIS is exactly that split.
- **PipelineRL** (Piché et al., arXiv:2509.19128): in-flight weight updates — the engine takes new weights
  mid-sequence, so staleness per token is bounded by one step even with long generations. miles'
  `--pause-generation-mode retract` (our default) does this: running requests re-prefill under the new
  weights and continue; the behavior log-probs stay per token, so the IS ratio stays exact.
- **Asynchronous RLHF** (Noukhovitch et al., arXiv:2410.18252): off-policyness of 1 step is free; beyond,
  loss choice matters more than the system.
- **ROLL Flash** (arXiv:2510.11345), **StreamRL** (arXiv:2504.15930), **RhymeRL** (arXiv:2508.18588):
  the systems side: queue scheduling, disaggregated generation, replay of history; all bound staleness.
- **Ring-1T IcePop** (arXiv:2510.18855): token-level masking of the train/inference ratio outside a band
  (miles `icepop_function`). **MiniMax-M1 CISPO** (arXiv:2506.13585): clip the IS weight, never the
  gradient. **TOPR** (arXiv:2503.14286): asymmetric taper for negatives.

## Arms (5 x 16 nodes: 4 actor DP 1 + 12 engines; batch 64 = 8 groups x 8; inflight x8; no staleness cap)

Common: image r25, final-v11 recipe (per-token loss, no std normalization, spec 5/6, fp8 KV, 64 running),
`NatsRolloutFn` (the train weight version reaches the drain), `--custom-tis-function-path` so the per-age
metrics exist in every arm. `--update-weights-interval K` holds the engine weights for K steps: the
sawtooth gives every age from 0 to K-1 inside one run, on top of the natural lag (~8 = the multiplier).

| arm | K | lr | correction | tests |
|---|---|---|---|---|
| a | 1 | 1.5e-6 | TIS 2.0 | control: natural staleness ~8 at the production lr |
| b | 32 | 1.5e-6 | TIS 2.0 | heavy staleness at the production lr: lr x age up to 48 (ab-v2 collapsed at 60) |
| c | 32 | 5e-6 | TIS 2.0 | stress: lr x age up to 160 |
| d | 32 | 5e-6 | M2PO tau 0.04 | does the second-moment mask hold where c fails |
| e | 1 | 5e-6 | TIS 2.0 | lr control for c and d |

Read-outs per step: `train/age_b*_abs_log_ratio / train/age_b*_tokens` (gap by age), `train/m2`,
`train/tis_clipfrac` (= masked fraction for d), `train/grad_norm`, `rollout/raw_reward`,
`rollout/total_lengths`, per-task paired reward from the Harbor `result.json` files.

## Results (steps 0-13, 2026-10-09 06:41 -> 08:46 UTC)

W&B: a `450m9kxj`, b `kwozrdfe`, c `6ydkwpsz`, d `sj36b1dp`, e `yncx3c1s`. Step time 5-10 min (generation-bound at 12
engines; first batch 20-24 min). Arm e exited after step 2 (3 W&B rows; not investigated). In arms b/c/d the engines held
`weight_version` 1 for the whole run, so every sample at train step `s` is exactly `s` updates stale: the gap at step `s`
*is* the drift of the trainer policy after `s` updates on frozen data, on top of the numerics floor.

Numerics floor (steps 0-1, all arms, fp8 KV + spec 5/6 engines): gap 0.0217, KL 0.0019, `m2` 0.0042.

| arm | lr | data | s=4 gap / m2 | s=8 gap / m2 | s=13 gap / m2 | KL at 13 | clip or mask at 13 | grad_norm |
|---|---|---|---|---|---|---|---|---|
| b | 1.5e-6 | frozen θ0, TIS | 0.025 / 0.0045 | 0.028 / 0.0057 | **0.032 / 0.0063** | 0.0030 | 0.04% | 0.017-0.05 flat |
| c | 5e-6 | frozen θ0, TIS | 0.055 / 0.018 | 0.095 / 0.062 | **0.151 / 0.153** | 0.054 | 1.06% | 0.03 -> 0.066, rising |
| d | 5e-6 | frozen θ0, M2PO | 0.047 / 0.014 | 0.086 / 0.053 | **0.096 / 0.075** | 0.025 | 0.25-0.7% masked (from s=6) | 0.03-0.06, spikes 0.12 |
| a | 1.5e-6 | fresh, K=1, mean age 3.9 / 6.1 / 7.6 | 0.031 / 0.0081 | 0.035 / 0.0102 | 0.038 / 0.0108 (step 11) | 0.0050 | 0.09% | 0.024-0.05 flat |

Power-law fits of the excess over the floor, `excess = A * s^p` (s >= 3):

| arm | gap: A, p | m2: A, p | m2 reaches the M2PO threshold 0.04 at | predicted m2 excess at s=32 |
|---|---|---|---|---|
| b (1.5e-6) | 3.9e-4, 1.27 | 1.3e-4, 1.15 | **s ~ 150** | 0.007 |
| c (5e-6, TIS) | 4.3e-3, 1.37 | 6.4e-4, 2.16 | **s ~ 7** | 1.13 |
| d (5e-6, M2PO) | 5.0e-3, 1.11 | 7.6e-4, 1.86 | s ~ 8 | 0.48 |

### Reading

1. **lr sets the staleness budget, and steeply.** 3.3x the learning rate gave 13x the gap drift and 70x the `m2`
   drift per update (`m2` grows ~s^2 at 5e-6: the policy moves in one direction; ~s^1.15 at 1.5e-6: closer to a
   random walk). At 1.5e-6, 13 updates of staleness add 45% to the gap and 50% to `m2`; the second moment stays
   under M2PO's 0.04 trust threshold for ~150 updates. At 5e-6 it crosses that threshold after 7 updates and the
   clipped fraction is 1% by 13 — the regime in which M2PO reports GRPO collapsing.
2. **M2PO acts as a soft trust region, not a fix.** At 5e-6 it halved the KL and `m2` drift and cut the gap by
   43% while masking 0.25-0.7% of tokens, but `m2` still reached 0.075 at s=13 (the mask is per micro-batch and
   covers trust-region tokens only; the metric covers all tokens). The unclipped IS weights showed up as two
   grad_norm spikes (0.12 vs 0.03-0.06). Whether the slower drift is less learning or less noise is not
   observable here: the engines ran θ0 throughout, so the rollout reward of b/c/d cannot show learning, and no
   checkpoint was saved (`save_interval` 100).
3. **Fresh data moves the policy faster than frozen data.** Arm a (K=1, batch 64) reached gap 0.038 / `m2` 0.011
   at mean age 5.75 (max 10) — about 5x arm b's excess at the same age. On frozen θ0 data the gradient fades as
   the policy learns it; with new data every step it does not. Arm a is the production-like case, and it drifts
   ~2x faster per unit age than final-v9 did at batch 512 (gap excess 0.016 at mean age 5.8 vs <= 0.01 at 7.3):
   the smaller batch is a noisier direction at the same Adam step size.
4. **Extrapolation for production at lr 1.5e-6, fresh data (arm a's slope, sublinear ~age^0.75):** mean age 23
   (a 3-min step) -> gap ~0.06 (2.7x floor), `m2` ~0.02, clip ~0.3%; mean age 50 -> gap ~0.09, `m2` ~0.03,
   clip ~0.5%. Both under the 0.04 threshold where M2PO would start masking, and under final-v9's observed
   history (age up to 27, no effect). Arm b (lr 1.5e-6) showed no instability of any kind through s=13.

### What this licenses, and what it does not

- **Yes:** lr <= 1.5e-6, no staleness cap below ~48 updates (or none), batch 128 / inflight x32 for a ~3-min
  step at the v11 fleet (4x the updates per hour of batch 512), TIS as today, M2PO (`arena_m2po_function`,
  tau 0.04) as the guard — it masks nothing until `m2` exceeds 0.04, so it is free in the expected regime.
  Drop the 14 groups per step that v11 discards at cap 12.
- **No:** raising the learning rate to buy faster learning in the async setting. 5e-6 spends the whole staleness
  budget in 7 updates; final-v6/v7/engine-ab-v2 at 1e-5 collapsed at age 2-6.
- **Open:** whether 4x the updates at batch 128 learns faster per hour than batch 512 (gradient noise vs step
  count). It needs a K=1 run at batch 128 with checkpoints evaluated against v11 at equal wall-clock, ~50 steps.
- **Bug found:** the per-age buckets `train/age_b*` stayed silent in arm a: `_package_shards` did not ship
  `metadata` to the DP shards (fixed in d4dd5d34, needs the next image). `train/m2` and `train/abs_log_ratio`
  worked.
