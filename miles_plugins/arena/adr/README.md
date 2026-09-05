# Architecture Decision Records — miles_plugins/arena

Decisions behind the arena plugin (the AGISlime → miles port of the NATS-gym RL
training path) and the jobs built on it under `examples/arena/`. Format and
rules mirror AREnATasks `adr/`: ADRs are numbered in decision order; do not
delete accepted records; mark a replaced decision as superseded and link its
replacement; read amendment and supersession links before changing a recorded
contract. Run history (what was launched, what happened, what was tried) lives
in `../RUNLOG.md` and `examples/arena/<job>/RUNLOG.md`, not here.

## Format

```text
# ADR-NNNN: Title

**Status:** Proposed | Accepted | Rejected | Superseded by ADR-XXXX | Deprecated
**Date:** YYYY-MM-DD

## Context
## Decision
### Alternatives considered
## Consequences
```

## Index

| ADR | Title |
| --- | --- |
| [0001](0001-port-scope-and-module-mapping.md) | Port scope and module mapping: `amzn_agi_slime` → `miles_plugins.arena` |
| [0002](0002-nats-wire-contract-bit-identical.md) | NATS wire contract stays bit-identical to `amzn_arena_contract` |
| [0003](0003-group-identity-group-index-not-rollout-id.md) | Group identity is a shared `group_index` plus a unique `index`, never `rollout_id` |
| [0004](0004-per-trajectory-loss-weighting-and-rollout-logprobs.md) | Accept per-trajectory loss weighting; zero-fill slow-path rollout log-probs |
| [0005](0005-driver-pure-insertion-and-sidecar-resume.md) | Driver as pure insertion over `train_async.py`; checkpoint-sidecar W&B resume via `init_wandb_primary` |
| [0006](0006-launcher-argv-parity.md) | Launcher argv parity: `scripts/run_arena_harbor.py` replaces entrypoint.sh + hydra_converter |
| [0007](0007-plain-stack-descoping-for-snorkel.md) | Plain-stack descoping for the snorkel job |
| [0008](0008-glm5-next-support-via-pr2786-merge.md) | GLM-5.3-Flash (glm5_next) support via the PR #2786 merge |
