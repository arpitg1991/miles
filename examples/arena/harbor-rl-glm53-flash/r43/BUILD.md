# r43: r41b on template v9 (DOCKER_CONFIG in the gym)

Launched 2026-09-25T21:26:30Z as `rl-glm53f43-qztzt` from
`gen-workflow.py 43 --base r41b --template guparpit-miles-deployer-v9
--experiment-name rl-glm53f-adebt-v3-r43 --gym-image ...gym-glm53-adr71-20260925a`.

| Delta vs r41b | r41b | r43 |
| --- | --- | --- |
| Template | `guparpit-miles-deployer-v8` | `guparpit-miles-deployer-v9` |
| Gym env `DOCKER_CONFIG` | unset | `/root/.docker` |
| Names | `rl-glm53f-adebt-v3-r41b` | `rl-glm53f-adebt-v3-r43` |

Why: Harbor 0.22.0 merges the task `[environment].env` into the
`docker compose` process env. agentic-debt v3-locked sets
`HOME=/home/agent`, so the Docker CLI reads `/home/agent/.docker`, finds no
registry login, and every pull fails with "pull access denied". A local
repro with the real validator (`amzn_arena_validate programmatic --up-to
zero_test`) fails with the task `HOME` and passes with
`DOCKER_CONFIG=/root/.docker`.

v9 is a cluster-only template. The `DOCKER_CONFIG` line is not in the Apps
source and is not for mainline: it unblocks this run only. The durable fix
belongs in Harbor (task env must not reach the Docker CLI env) or in the
dataset.

Same images as r41b: gym `gym-glm53-adr71-20260925a` (AREnATasks ff880da),
trainer `miles-glm53-r13-20260925a` (miles dd5e67391). This run is the live
check for CR-307888266 (token arrays as `.tokens` files).
