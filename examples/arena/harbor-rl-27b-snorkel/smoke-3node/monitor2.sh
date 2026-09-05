#!/bin/bash
# Poll until the smoke job reaches a terminal state (all 4 rollouts + save), max ~5h.
CTX="--context arena-prod-bom-v2 -n arena-tasks"
for i in $(seq 1 60); do
  sleep 300
  TS=$(date +%H:%M:%S)
  J=$(kubectl $CTX get pytorchjob rl-milesgb1-trainer -o jsonpath='{.status.conditions[-1].type}' 2>/dev/null)
  L=$(kubectl $CTX logs rl-milesgb1-trainer-worker-0 --tail=300 2>/dev/null)
  STEP=$(echo "$L" | grep -oE "'train/step': [0-9]+" | tail -1)
  COLLECTED=$(echo "$L" | grep -E "groups collected" | tail -1 | grep -oE "[0-9]+/8 groups")
  ROLLOUT=$(echo "$L" | grep -oE "Rollout [0-9]+ complete" | tail -1)
  SAVE=$(echo "$L" | grep -cE "saving checkpoint|save_model|Saving.*checkpoint|save-hf|hf_export")
  echo "$TS job=$J $STEP collected=$COLLECTED last=$ROLLOUT save_lines=$SAVE"
  if [[ "$J" == "Failed" || "$J" == "Succeeded" ]]; then echo "JOB_TERMINAL: $J"; break; fi
done
echo MONITOR2_EXIT
