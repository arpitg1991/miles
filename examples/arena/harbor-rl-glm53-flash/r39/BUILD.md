# r39: auctioneer-caponly on caponly-1034, ADR-0069 gym, r12 trainer

Launched 2026-09-25T06:21:07Z as `rl-glm53f39-25pdn` (template
`guparpit-miles-deployer-v7`, generated with `gen-workflow.py 39 --base r38`).
r38 (`rl-glm53f38-htf6t`) keeps running; r39 runs on separate nodes.

| Delta vs r38 | r38 | r39 |
| --- | --- | --- |
| Dataset | `lakefs://arena-inspect/main/internal/auctioneer/caponly/20260916-v1` (1,047 tasks) | `lakefs://arena-inspect/dev/internal/auctioneer/caponly/caponly-1034` (1,034 tasks; rows pin commit `e5ef91b0`; dev head at launch `4476af5e`) |
| Gym image | `gym-glm53-adr68-20260924a` (AREnATasks `2befd21`) | `gym-glm53-adr69-20260925a` (AREnATasks `4d3ecdf`, digest `sha256:6b18f411`) |
| Trainer image | `miles-glm53-r11-20260924a` (miles `a639ea445`) | `miles-glm53-r12-20260925a` (miles `db3955b70`, digest `sha256:fa9a1252`) |
| `partial-reward` parameter | `off` (explicit) | template default `off` (gen-workflow drops it) |
| Run names | `rl-glm53f-auct-cap-r38` | `rl-glm53f-auct-cap-r39` |

Dataset versus 20260916-v1: 13 seeds removed (93, 155, 241, 462, 493, 494,
505, 516, 526, 564, 720, 906, 968); only `instruction.md` changed on all
1,034 tasks (documents `unlist`, states the day-30 control window, removes
the stale `set_rental_terms` step). `task.toml`, the task image digest, the
verifier, and `tests/oracle.json` are byte-identical, so the reward is the
same function as r38.

Gym change: training trains on Harbor's trial reward (ADR-0069); for a flat
task the output is byte-identical to adr68. Trainer change: new W&B keys
`rollout/failed/<reason>` and `rollout/dropped_groups/<reason>`.
Everything else (40 replicas, 8 trainer nodes, 288 gyms, 83 excluded nodes,
ack-wait 36000, agent-timeout-multiplier 1, trainer-task-deadline-secs 39600,
compaction-max 2) is r38.
