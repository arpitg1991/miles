# r20 (2026-09-13): partial rewards, 871-prompt r19signal list, shuffle on

Trainer image unchanged: `arena-slime-dev:miles-glm53-r3-20260911b`. Gym image
`arena-slime-dev:gym-glm53-r4-20260913a` (AREnATasks `arpit-glm-53`, CTRF
partial reward mode; see ADR-0039). Launched through the user-owned
WorkflowTemplate `guparpit-miles-deployer-v2`, created 2026-09-13 from AREnATasksApps
`arpit-glm-53` with the `partial-reward` parameter and the gym tokenizer-cache
and mount-fallback fixes (`32589b3`, `2e19fe6`).

## Deltas vs r19

1. Gym `ARENA_PARTIAL_REWARD=ctrf` (workflow parameter `partial-reward`).
   When `reward.txt` is 0 the worker reports `passed/tests` from the pytest
   CTRF report, or from the `-rA` summary line when a task has no CTRF file.
   Full credit stays verifier-only. `grader_metadata.reward_binary` keeps
   the original 0/1. Default GRPO group normalization takes the float; no
   `custom_reward_post_process_path`.
2. Manifest `ecr-20260823-curriculum-r19signal/manifest-871.jsonl`
   (871 rows, md5 `0ed6fdefa911536d8b2d24ca48692bcd`): the r19 prompts with
   training signal. 712 = epoch-2 eligible prompts solved at least once in
   r19 steps 0-39 minus the 52 prompts that scored above 0.5 in epoch 2;
   plus the 159 prompts never solved. Derivation lists in
   `r19-derivation-871.json` next to the manifest. GPT-5.6 pass@1 of the
   871: 47.0% at or below 25%, 23.5% between, 29.5% above 62.5%.
3. `rollout_shuffle: true`. The list is a filtered set, not a curriculum.
4. `arena_skip_prompt_above_reward` 0.5 -> 0.9. With fractional rewards 0.5
   would drop a prompt that only ever scores 0.6.
5. `gen-workflow.py` (parent dir) emits `workflow.yaml` from
   `r19/workflow.yaml` plus this run's config, gym image, and mode.

Kept from r19: `REPLICA_TRAINER` 8, `use_tis` true, `use_rollout_logprobs`
false, `lr` 1e-6, GBS 512, `rollout_batch_size` 64, `n_samples_per_prompt` 8,
`arena_inflight_multiplier` 4, `gym-replicas` 288, the 22-node `NotIn` list.

r21 is this run at `lr` 1.5e-6 (`r21/`). Submitted as `rl-glm53f20-x9tvq` (2026-09-13 06:08Z).

## Not changed, still open

- NATS `max_payload` stays 8 MiB (`ARENA_COMPACTION_MAX=2` keeps results
  under it; the r19 relaunch logged 0 `MaxPayloadError` in a 60-pod sample).
- Resume still resets `prompt_reward`, `offsets`, and `epochs`.
- The reward gate still reads the last group's mean only.

## Launch

Context `arena-prod-bom-v2`, namespace `arena-tasks`, `kubectl` only.

1. `kubectl exec <live trainer pod> -c pytorch -- ls /mnt/scratch-s3files-rw/guparpit/checkpoints/slime_experiments/rl-glm53f-gbash-r20`
   must fail with `No such file or directory`.
2. Create the regenerated template: `kubectl create -f /tmp/guparpit-miles-deployer-v2.yaml`
   (generated file with `__AWS_REGION__` -> `ap-south-1` and the name
   `guparpit-miles-deployer-v2`; RBAC grants create, not patch, on
   workflowtemplates). A running Workflow keeps its stored copy.
3. `python gen-workflow.py 20`, then `kubectl create --dry-run=client -f r20/workflow.yaml`.
4. `kubectl create -f r20/workflow.yaml`. Watch the DAG as in `r19/BUILD.md`.
5. Kueue holds the PyTorchJob until 40 `p6-b200` nodes are free.

## Checks after admission

- worker-0 logs `Loaded gym snorkel-general-bash-harbor: 871 prompts` and
  the parsed `arena_skip_prompt_above_reward` `0.9`.
- Gym logs: `grep -c "partial reward"` per pod; a fraction between 0 and 1
  appears in `rollout_tasks`-adjacent group reward lines.
- `rollout/zero_std/all_zero_percentage` falls versus r19 (partial passes now
  carry variance).

## Admission record

- Admitted 2026-09-13 18:30Z after the kimi scale-down; NATS connected
  18:55Z; gym 288/288 ready.
- Rollout 0 avg_reward 0.785, rollout 1
  0.806; `all_zero_percentage` 0.0.
- Resume watcher `/tmp/r20-resume.sh` started 20:10Z. Kill it by PID before a retire.
