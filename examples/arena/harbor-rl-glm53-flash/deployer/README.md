# deployer — the Argo WorkflowTemplate variants of this family

`acuadron-miles-deployer-v4.yaml` is `guparpit-miles-deployer-v10` (the template that AREnATasksApps
renders, fetched from the cluster on 2026-10-07) with one change: the gym fan-out starts with the
trainer pod, not after the trainer's NATS handshake.

- v10: `deploy-gym-workers` depends on `wait-trainer-nats`, which waits for worker-0 to log
  `NATS connected (initial)`. final-v7: the trainer connected at 20:24, the gyms were deployed at
  20:25 and ready at about 20:26, and the 64 tasks of rollout 0 were published at 20:23:44.
- v4: `deploy-gym-workers` depends on `wait-trainer-running`, which waits for worker-0 to log
  `Ray runtime started` (the PyTorchJob is admitted and the trainer pod is up, so a Kueue wait of
  hours parks no idle gyms) and gives up when the PyTorchJob fails, as v10 does. The gym workers
  wait for the trainer's streams themselves (`amzn_arena_contract.rl.nats`: one retry per 60 s,
  `ARENA_NATS_STREAM_WAIT_ATTEMPTS`); they cannot take a task or publish a result before the
  trainer creates the streams and its results consumer. v4 sets that budget to 90 attempts.
  `wait-trainer-nats` stays as a timing marker; nothing depends on it.

Create or update it with `kubectl --context arena-prod-bom-v2 -n arena-tasks apply -f acuadron-miles-deployer-v4.yaml`
and select it with `workflowTemplateRef.name: acuadron-miles-deployer-v4`. The change belongs in the
AREnATasksApps source of the deployer; this file records what runs until then.
