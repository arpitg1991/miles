# r42: r39 with the radix cache and overlap scheduling on, agent timeout x2

Launched 2026-09-25T18:06:16Z as `rl-glm53f42-5cthp` from
`gen-workflow.py 42 --base r39 --param agent-timeout-multiplier=2` on template
`guparpit-miles-deployer-v7`. r39 (`rl-glm53f39-25pdn`) keeps running for the
rollout-time comparison.

| Delta vs r39 | r39 | r42 |
| --- | --- | --- |
| `sglang_disable_radix_cache` | true | false |
| `sglang_disable_overlap_schedule` | true | false |
| `agent-timeout-multiplier` | 1 (2 h agent limit) | 2 (4 h) |
| Names | `rl-glm53f-auct-cap-r39` | `rl-glm53f-auct-cap-r42` |

Same dataset (caponly-1034), gym image `gym-glm53-adr69-20260925a`, trainer image
`miles-glm53-r12-20260925a`, W&B project `rl-glm53f-auct-cap`.

Why: r39 engines ran at KV 97% with about 17 tok/s per request, and about 10%
of recent episodes ended by AgentTimeoutError. The r33/r34 A/B showed radix
ON halves rollout time (1130 s -> 535 s). With overlap on, SGLang's `auto`
Mamba cache strategy selects `extra_buffer` for GLM-5.3-Flash (triton
linear-attention backend), which supports overlap with the radix cache.
Checks: SGLang starts with `extra_buffer`; no stall at weight sync; step-0
policy drift stays near zero (R3); compare `perf/rollout_time` and
`rollout/stop/timeout` with r39 at the same rollout indices.
