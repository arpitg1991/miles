# r24: r22 at lr 1.5e-6 without the epoch-2 prompt skip

Same as r22 (robust-20260913 `manifest-849.jsonl`, gym image
`gym-glm53-r4-20260913a`, trainer image `miles-glm53-r3-20260911b`) with
`lr: 1.5e-6` (the r21 value) and `arena_skip_prompt_above_reward` removed.
Under the skip, r22 dropped 369 of 849 prompts (43%) at its epoch-2 boundary.

## Launch

1. `cp -r r22 r24`, rename, set lr, drop the skip key (see the header of
   `miles-config.yaml`).
2. `.venv/bin/python gen-workflow.py 24` (template `guparpit-miles-deployer-v3`).
3. `kubectl create -f r24/workflow.yaml` -> `rl-glm53f24-2nl5b` at 09:56Z on
   2026-09-14. kueue held it Suspended for ~2 min, then admitted it; 40/40
   pods Running by 10:03Z, before r20 was retired.
4. Resume watcher `/tmp/r24-resume.sh` started at 10:11Z (same fixed script as
   r23).
