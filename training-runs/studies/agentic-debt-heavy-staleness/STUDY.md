# Study: training under heavy weight staleness (agentic-debt, GLM-5.3-Flash)

**Status:** arms running (launched 2026-10-09 01:38 UTC)
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

## Results

(pending)
