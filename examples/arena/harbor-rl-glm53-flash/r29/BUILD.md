# r29: r28 recipe on the hardened gym image and template v6

r28 (`rl-glm53f28-gs4d4`, launched 2026-09-20 02:46Z) validated the
multi-step path: rollout 0 gave 832 rows for 256 episodes (one optimizer
step, 369 micro-batches), episode reward 0.761; rollout 1 gave 752 rows,
reward 0.684. Two infrastructure faults appeared in the first two hours:

| Fault | Evidence | Fix |
| --- | --- | --- |
| DinD sidecar OOM (32Gi cgroup) | 9 gym pods OOMKilled in 60 min; each kill destroyed its group of 8 attempts; Vulcan kept calling the model against the dead environment (up to 109 turns, 1.5M tokens per attempt) | gym: `ARENA_TASK_MEMORY_MB` 6144 per task container (harbor `override_memory_mb` -> compose `mem_limit`); Vulcan aborts after 5 consecutive dead-environment errors; template v6: sidecar 64Gi |
| Pod eviction on ephemeral storage | "usage exceeds the total limit of containers 100Gi"; `/var/lib/docker` 20-39 GB per pod after 1-2 groups (each group pulls a different 2-6 GB image, nothing pruned) | gym: `docker image prune -af` + `docker builder prune -af` when usage > `ARENA_DOCKER_PRUNE_GB` (40) between groups; template v6: dind ephemeral-storage 100Gi/250Gi so the pod limit sum exceeds the emptyDir sizes |

Everything else is r28: le5 manifest, `global_batch_size` 256, v5 deadline
parameters (ack-wait 36000, multiplier 1, trainer deadline 39600,
compaction-max 2), inflight 4, length coefficient 0, excluded-nodes as
trimmed for r28.

## Regenerate

```
.venv/bin/python gen-workflow.py 29 --base r28 --template guparpit-miles-deployer-v6 \
  --partial-reward off --experiment-name rl-glm53f-adebt-r29 \
  --gym-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-adebt-r29-20260920a \
  --trainer-image 427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-r9-20260920a \
  --param gym=agentic-debt --param ack-wait=36000 --param agent-timeout-multiplier=1 \
  --param trainer-task-deadline-secs=39600 --param compaction-max=2
```

## Launch checklist

1. Gym image `gym-glm53-adebt-r29-20260920a` and trainer image
   `miles-glm53-r9-20260920a` (arpit-glm-53 03791ce86, routing-ref fix)
   visible in ap-south-1.
2. WorkflowTemplate `guparpit-miles-deployer-v6` created from
   `~/glm53-prep/guparpit-miles-deployer-v6.yaml`.
3. Retire r28 only on an explicit user yes that names `rl-glm53f28-gs4d4`:
   kill `/tmp/r28-resume.sh` by PID first, then `shutdown: Stop`, wait
   4 min, check `deploy,svc,pytorchjob` leftovers.
4. `kubectl create -f r29/workflow.yaml`; start `/tmp/r29-resume.sh` after
   the PyTorchJob exists.
