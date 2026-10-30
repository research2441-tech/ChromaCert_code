#!/bin/bash
# Run each line of a job file sequentially; log to logs/<jobfile>.log
cd "$(dirname "$0")"
export CC_DATA=$PWD/data
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  echo "[$(date +%H:%M:%S)] START $cmd"
  eval "$cmd" || echo "FAILED $cmd"
  echo "[$(date +%H:%M:%S)] END"
done < "$1"
echo QUEUEDONE
