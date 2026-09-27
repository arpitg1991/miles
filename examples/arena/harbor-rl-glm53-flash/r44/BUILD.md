# r44: r42 with a 16384 output cap from the task message (ADR-0015)

Prepared 2026-09-27. Not launched. Generated with:

```
.venv/bin/python gen-workflow.py 44 --base r42 --template guparpit-miles-deployer-v10 \
  --experiment-name rl-glm53f-auct-cap-r44 \
  --gym-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adr72-20260927a \
  --trainer-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r14-20260927a \
  --param 'agent-kwargs={}' --param 'publish-jobs-dir='
```

| Delta vs r42 | r42 | r44 |
| --- | --- | --- |
| `rollout_max_response_len` | 32768 (gym copy `ARENA_MAX_TOKENS` 32768) | 16384, sent in the task message |
| Window | gym env `ARENA_ROLLOUT_CONTEXT_LIMIT` 131072 | task message `max_seq_len` 131072 |
| Output cap near the window end | fixed 32768 | clamped to the room left in the window |
| Vulcan `max_compactions` | 2 (`compaction-max` 2) | 4 (`agent-kwargs` `{}`, Vulcan default) |
| Template | `guparpit-miles-deployer-v7` | `guparpit-miles-deployer-v10` |
| Trainer image | `miles-glm53-r12-20260925a` | `miles-glm53-r14-20260927a` (miles `e0987aed6`) |
| Gym image | `gym-glm53-adr69-20260925a` | `gym-glm53-adr72-20260927a` (AREnATasks mainline `538bc63`, tree `ab266688`) |
| Trajectory publication | off (v7 has no publish env) | off (`publish-jobs-dir` `''`) |
| Names | `rl-glm53f-auct-cap-r42` | `rl-glm53f-auct-cap-r44` |

Same as r42: dataset caponly-1034, radix cache and overlap schedule on,
`agent-timeout-multiplier` 2, `ack-wait` 36000, trainer deadline 39600,
`excluded-nodes`, W&B project `rl-glm53f-auct-cap`. Fresh run: no
checkpoint dir under `slime_experiments/rl-glm53f-auct-cap-r44` (checked
2026-09-27 04:50Z).

Why:

- r39 and r42 sent about 500 SGLang requests per 90 minutes that SGLang
  rejected with HTTP 400. Each had 98K-131K input tokens plus the fixed
  32768 output budget, more than the 131072 window.
- About 99% of those requests were Vulcan summary calls. 38-48% of the
  compactions lost their handoff note.
- Healthy runs show a per-call P99 of 17K-20K output tokens. The collapsed
  calls sit at P90 = 32768, the cap. A 16384 cap keeps the healthy range
  and leaves 16K more room for the input.
- The trainer is now the only source of the cap, the window, and the
  sampling values. The gym copies drifted with no error.

Code reviews, all merged: AREnATasks CR-308323817 (clamp to the room left
in the window, `0b12642`) and CR-308323860 (task message limits, ADR-0072,
`538bc63`); Apps CR-308323875 (gym env copies removed, `agent-kwargs`,
mainline `e4bf764`). The trainer side is miles ADR-0015 (`4c9e97b0f`,
`e0987aed6`).

Images:

- Trainer `miles-glm53-r14-20260927a`: `sha256:48d52a6a431257f9f7b40ce00debe9b2a10b61f94cee85d83291b21a2439e60c`
  in us-east-1 and ap-south-1.
- Gym `gym-glm53-adr72-20260927a`: `sha256:57f0ed616892c50f727247a7b6eedf972bd35b98e70bd0ab4283f570554cdd0d`
  in us-east-1 and ap-south-1. Label `org.opencontainers.image.revision`
  is `538bc6378870a31a7e29451c163dca1d6c161676`.

## Template v10

`guparpit-miles-deployer-v10.yaml` in this directory is the Apps mainline
`e4bf764` generated template (CR-308323875, same tree as `07e4970`) with
`__AWS_REGION__` set to `ap-south-1`. It adds the two cluster-only gym
additions of v8 and v9: the us-east-1 ECR login and
`DOCKER_CONFIG=/root/.docker`. The default `trainer-image` and
`gym-image` are the two new images. RBAC allows create but not patch, so
each revision has a new name.

Created 2026-09-27T04:48:19Z on `arena-prod-bom-v2`, namespace
`arena-tasks`, uid `16ddd530-501f-4734-82ae-88500671bed8`. The live spec
equals this file.

The template default of `publish-jobs-dir` is
`/mnt/scratch-s3files-rw/harbor-training`. r44 sets `''`, so the gym
publishes no trajectories.

## Launch checklist

1. Done: the gym image exists in ap-south-1 ECR (digest above).
2. Done: template `guparpit-miles-deployer-v10` exists (uid above).
3. Pair check: the r14 trainer with an ADR-0072 gym only. An older trainer
   fails every group. An older gym on v10 falls back to 8192/32768 with no
   error.
4. `excluded-nodes` is the r42 list. On 2026-09-27 no listed instance
   existed in the cluster, so the list excludes no node. A non-empty list
   selects the `deploy-trainer-pinned` step, as on r42.
5. Stop r42 (`rl-glm53f42-5cthp`) to free its 40 nodes.
6. `kubectl create -f r44/workflow.yaml`.
7. On one gym pod, read `HARBOR_AGENT_KWARGS` (`{}`), `DOCKER_CONFIG`
   (`/root/.docker`), and `ARENA_PUBLISH_JOBS_DIR` (empty). In the gym
   log, look for `max_new_tokens` 16384 and `max_seq_len` 131072.
8. Watch the SGLang 400 count and the Vulcan compaction note loss against
   r42 at the same rollout indices. Record `max_compactions` 4.
