#!/usr/bin/env bash
# External watchdog for the bounded protocol-v2 training trial.
#
# The clock starts before python does, so the limit covers model loading and
# every operation already running -- an in-process deadline can only stop
# between operations, and a single generation is the longest one there is.
# TERM first so the pilot can label its own partial record, then KILL.
set -u
LIMIT="${1:-1200}"
OUT="evidence/rl-v2-trial"
mkdir -p "$OUT"

PYTHONPATH=src .venv-rl/bin/python -m cerl_rl.pilot_v2 \
  --updates 1 --group-size 4 --lr 1e-5 --lora-rank 8 \
  --deadline-minutes 18 --training-only \
  --out "$OUT" --allow-existing > "$OUT/stdout.log" 2>&1 &
PID=$!
echo "trial pid $PID, external hard limit ${LIMIT}s"

( sleep "$LIMIT"
  if kill -0 "$PID" 2>/dev/null; then
      echo "WATCHDOG: ${LIMIT}s reached; terminating $PID" | tee -a "$OUT/stdout.log"
      kill -TERM "$PID" 2>/dev/null; sleep 15; kill -KILL "$PID" 2>/dev/null
  fi ) &
WD=$!
wait "$PID"; STATUS=$?
kill "$WD" 2>/dev/null
echo "trial exited with status $STATUS"
exit "$STATUS"
