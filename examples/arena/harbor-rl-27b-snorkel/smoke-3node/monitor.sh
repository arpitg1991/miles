#!/bin/bash
# Poll the smoke run until rollout 0 completes + first train step, or failure.
CTX="--context arena-prod-bom-v2 -n arena-tasks"
for i in $(seq 1 40); do
  sleep 300
  TS=$(date +%H:%M:%S)
  J=$(kubectl $CTX get pytorchjob rl-milesgb1-trainer -o jsonpath='{.status.conditions[-1].type}' 2>/dev/null)
  L=$(kubectl $CTX logs rl-milesgb1-trainer-worker-0 --tail=400 2>/dev/null)
  COLLECTED=$(echo "$L" | grep -E "groups collected" | tail -1 | grep -oE "[0-9]+/8 groups" | tail -1)
  DROPS=$(echo "$L" | grep -c "dyn-sampling drop")
  TRAIN=$(echo "$L" | grep -cE "train_wait end|Timer train start|rollout 1|Rollout 1:")
  echo "$TS job=$J collected=$COLLECTED drops_recent=$DROPS train_markers=$TRAIN"
  if [[ "$J" == "Failed" || "$J" == "Succeeded" ]]; then echo "JOB_TERMINAL: $J"; break; fi
  if [[ "$TRAIN" -gt 0 ]]; then echo "ROLLOUT0_DONE_TRAINING_STARTED"; break; fi
done
echo MONITOR_EXIT
