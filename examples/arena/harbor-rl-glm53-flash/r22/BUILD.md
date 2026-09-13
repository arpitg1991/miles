# r22 (2026-09-13): r20 settings and task ids, tasks from robust-20260913

Trainer image `arena-slime-dev:miles-glm53-r3-20260911b`, gym image
`arena-slime-dev:gym-glm53-r4-20260913a`, WorkflowTemplate
`guparpit-miles-deployer-v2`. All unchanged from r20.

## Deltas vs r20

1. Manifest `robust-20260913-r19signal/manifest-849.jsonl` (849 rows, md5
   `00bf484f59552ddb5a55b6b3f50ddf55`). The 871 r20 task ids re-pinned to
   `lakefs://arena-inspect/dev/internal/snorkel-general-bash-harbor/robust-20260913/`
   (variant manifest 2877 tasks, one commit `f6d04286ce2a...`). 22 of the 871
   ids (2.5%) are absent from that variant and dropped. Kept and missing
   lists: `r22-derivation-849.json` next to the manifest.
2. `experiment_name` / `project_name` `rl-glm53f-gbash-r22`.

Kept from r20: `lr` 1e-6, `ARENA_PARTIAL_REWARD=ctrf`,
`arena_skip_prompt_above_reward` 0.9, `rollout_shuffle` true, GBS 512,
`rollout_batch_size` 64, `n_samples_per_prompt` 8, `arena_inflight_multiplier`
4, `gym-replicas` 288, `REPLICA_TRAINER` 8, the 22-node `NotIn` list.

## Missing task ids (22)

relay-yard-rsyslog-failover, acoustic-fwi-phase-fix, curling-draw-scheduler,
merkle-bundle-admission, minesweeper-field-campaign, site-enrollment-clawbacks,
chess-king-race, boolean-expression-evaluator, helios-tpm-breakglass,
rune-tide-router, sealed-tile-solver, filter-wheel-scheduler,
trial-site-activation, dispatch-vm-recovery, annealing-tsp-report-fix, and 7
more in the derivation file.

## Launch

1. Manifest written from the r20 worker-0 pod with
   `kubectl exec -i ... -- sh -c 'cat > .../manifest-849.jsonl' < local`,
   md5 verified on both sides.
2. `python gen-workflow.py 22`; parameter diff vs `r20/workflow.yaml` is
   `experiment-name` only.
3. `kubectl create -f r22/workflow.yaml` -> `rl-glm53f22-6f2sp` at 23:10Z.
   **Killed by my own watcher 25 s later**: `/tmp/r22-resume.sh` was started
   before the PyTorchJob existed and read "no trainer" as a crash. Leftover
   `nats-svc` deleted; `rl-glm53f22-6f2sp-config` ConfigMap stays (owner
   policy). Resubmitted -> `rl-glm53f22-fv54h` at 23:13Z.
4. Watcher restarted with a 1800 s workflow-age guard (PID 460435). Kill it
   by PID before any retire.

## Checks after admission

- worker-0 logs `Loaded gym snorkel-general-bash-harbor: 849 prompts`.
- First rollouts: reward in the r20 range (0.78-0.82) or note the shift; the
  robust variant changes task content, not the id list.
