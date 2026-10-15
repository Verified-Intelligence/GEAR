#!/bin/bash
# resumable wrapper: retries the campaign (which resumes from logs) up to 5 times
# usage: harness/run_campaign.sh [solver ...]      (env: GEAR_BASELINE_RUNS, GEAR_BASELINE_LOCK)
cd "$(dirname "$0")"
for i in 1 2 3 4 5; do
  python campaign.py "$@" && break
  echo "campaign exited with error, retry $i" ; sleep 60
done
