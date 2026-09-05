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
