# r41: agentic-debt v3-locked (oracle branch, <= 5 steps), ADR-0069 reward, token-array file refs

Launched 2026-09-25T08:13:08Z as `rl-glm53f41-kfjhv`; admitted 08:25Z after r33 was retired.
Generated with `gen-workflow.py 41 --base r34` on template
`guparpit-miles-deployer-v7`. Named r41, not r40, because another user's run
is `rl-glm53f40-q75bm`.

| Delta vs r34 | r34 | r41 |
| --- | --- | --- |
| Dataset | `20260916-v1` le5 (556 chains, `mean`, no gate, 0/1 per step) | `acuadron-v3-oracle-validation-20260924` / `20260923-v3-locked`, commit `77c2239b`, 551 chains with <= 5 steps (182 x 2, 139 x 3, 122 x 4, 108 x 5); `final`, gate `{segment_pass = 1.0}`, prefix reward k/N |
| Manifest | `/mnt/scratch-s3files-rw/guparpit/data/agentic-debt/20260916-v1/manifest-adebt-le5.jsonl` | `/mnt/scratch-s3files-rw/guparpit/data/agentic-debt/20260923-v3-locked-oracle/manifest-le5.jsonl` |
| Gym image | `gym-glm53-adebt-cut-r31-20260921a` (AREnATasks `7ea168d`, reward = passed prefix / K, runtime step cut) | `gym-glm53-adr71-20260925a` (AREnATasks `ff880da`: reward = Harbor trial reward, ADR-0069/0070; token arrays as `.tokens` files and a 30 MiB result cap, ADR-0071) |
| Trainer image | `miles-glm53-r9-20260920a` (miles `03791ce86`) | `miles-glm53-r13-20260925a` (miles `dd5e67391`: token-file reader, ADR-0014; drop-reason telemetry) |
| Six clipped-turn keys | present (r9 accepted them) | removed (the current argparse rejects them), as in r38 |
| `step-cut-on-fail` | `true` (old gym runtime cut) | `false` (the dataset's `min_reward` gate stops the chain in Harbor) |
| Names | `rl-glm53f-adebt-cut-r34`, W&B project `rl-glm53f-adebt-cut` | `rl-glm53f-adebt-v3-r41`, W&B project `rl-glm53f-adebt-v3` |

Unchanged from r34: 40 replicas, 8 trainer nodes, 288 gyms, the excluded-node
list, ack-wait 86000 (gym deadline 85700 s), trainer-task-deadline-secs 90000,
agent-timeout-multiplier 4 (no effect: v3 declares no agent timeout),
compaction-max 2, routing replay on, radix cache on.

Live checks this run provides for AREnATasks CR-307888266 (ADR-0050): results
arrive with `token_arrays_ref` steps and no NATS size rejection;
`rollout/failed/token_ref`, `rollout/dropped_groups/{token_ref,staging,too_large}`
stay 0; the routing dir holds `.tokens` files only transiently.

Images: gym `gym-glm53-adr71-20260925a` digest `sha256:42c5bb8e`, trainer
`miles-glm53-r13-20260925a` digest `sha256:f5e24d9e`, both built 2026-09-25.
