# r23: r21 without the epoch-2 prompt skip

Same as r21 (lr 1.5e-6, ecr-20260823 `manifest-871.jsonl`, gym image
`gym-glm53-r4-20260913a`, trainer image `miles-glm53-r3-20260911b`) with
`arena_skip_prompt_above_reward` removed. Under the skip, epoch 2 dropped 202
of 871 prompts (23%) on r21 and the reward curve tracked pool difficulty, not
the policy. r23 keeps every prompt in every epoch.

## Launch

1. `cp -r r21 r23`, rename, drop the skip key (see the header of
   `miles-config.yaml`).
2. `.venv/bin/python gen-workflow.py 23` (template `guparpit-miles-deployer-v3`).
3. `kubectl create -f r23/workflow.yaml` -> `rl-glm53f23-fkmgl` at 09:56Z on
   2026-09-14. kueue admitted the trainer at once; 40/40 pods Running by 10:00Z.
4. Resume watcher `/tmp/r23-resume.sh` started at 10:11Z, after the PyTorchJob
   existed. It selects the Running workflow by phase and skips workflows
   younger than 1800 s.
