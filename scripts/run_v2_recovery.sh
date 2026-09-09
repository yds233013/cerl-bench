#!/usr/bin/env bash
# External watchdog for the recovery update. Covers model loading and every
# model computation; the clock starts before python does.
set -u
BUDGET="${1:-1200}"
OUT="evidence/rl-v2-recovery"
mkdir -p "$OUT"
PYTHONPATH=src .venv-rl/bin/python scripts/recover_v2_update.py \
  > "$OUT/stdout.log" 2> "$OUT/stderr.log" &
PID=$!
echo "recovery pid $PID, external hard limit ${BUDGET}s"
( sleep "$BUDGET"
  if kill -0 "$PID" 2>/dev/null; then
      echo "WATCHDOG: ${BUDGET}s reached; terminating $PID" | tee -a "$OUT/stderr.log"
      kill -TERM "$PID" 2>/dev/null; sleep 15; kill -KILL "$PID" 2>/dev/null
  fi ) &
WD=$!
wait "$PID"; STATUS=$?
kill "$WD" 2>/dev/null
echo "recovery exited with status $STATUS"
exit "$STATUS"
