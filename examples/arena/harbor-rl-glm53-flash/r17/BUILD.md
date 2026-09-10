# r17 (2026-09-10): r16 images, three config changes

No image change. Trainer `arena-slime-dev:miles-glm53-r3-20260909b`, gym
`arena-slime-dev:gym-glm53-r3-20260909a` (see `r16/BUILD.md`).

Deltas vs r16, all config:

1. Node split 8 + 32 -> 16 + 24 (`REPLICA_TRAINER` 16). r16 spent 22 of a
   28 min step in `actor_train` while 32 engines drained a rollout in 5 min.
2. `use_tis: false`, `use_rollout_logprobs: true`. r16 logged `train/tis`
   1.000 under R3; the old-logprob pass cost 5 min per step.
3. Curriculum manifest, `rollout_shuffle: false` (`r17/curriculum/`).

Warm start: `slime_experiments/rl-glm53f-gbash-r17/` holds a symlink
`iter_0000014 -> ../rl-glm53f-gbash-r16/iter_0000014` and
`latest_checkpointed_iteration.txt` = 14, no `rollout/` state. Weights resume
from r16 step 14; the data source starts at row 0 of the curriculum;
rollout ids continue from 15. `lr` stays 1e-6.
