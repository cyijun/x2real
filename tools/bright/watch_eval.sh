#!/bin/bash
# usage: tools/bright/watch_eval.sh <task> <log> [eval args]
# Runs run_bright_eval.sh; if Kit hangs at startup (log idle >100 s right after
# 'Using Mana headless camera experience') it kills the run and retries (max 3).
task=$1; log=$2; shift 2
cd "$(dirname "$0")/../.."
for attempt in 1 2 3; do
  tools/bright/run_bright_eval.sh "$task" "$log" "$@" &
  pid=$!
  hung=0
  while kill -0 $pid 2>/dev/null; do
    sleep 15
    age=$(( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || date +%s) ))
    if [ $age -gt 100 ] && grep -q "Using Mana headless camera experience" "$log" && ! grep -q "episode\|Episode\|STEP\|assembl" "$log"; then
      echo "HANG detected attempt $attempt"; hung=1
      pkill -P $pid; kill $pid; sleep 5
      for p in $(pgrep -f "x2real_eval.py.*$task"); do kill -9 $p; done
      break
    fi
    if [ $age -gt 600 ]; then echo "IDLE600 attempt $attempt"; hung=1; pkill -P $pid; kill $pid; sleep 5; for p in $(pgrep -f "x2real_eval.py.*$task"); do kill -9 $p; done; break; fi
  done
  [ $hung = 0 ] && break
done
echo "WATCH_DONE $task"
