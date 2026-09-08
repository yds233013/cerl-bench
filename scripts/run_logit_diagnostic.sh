#!/usr/bin/env bash
# External watchdog for the logit diagnostic.
#
# The limit covers ALL model work, including loading: the clock starts before
# the python process does. macOS ships no coreutils `timeout`, so the watchdog
# is a separate supervising process here -- it is genuinely external to the
# model process, which is the point. TERM first so the script can label its own
# partial record, then KILL if it does not go.
set -u
LIMIT="${1:-600}"
OUT="evidence/rl-logit-diagnostic"
mkdir -p "$OUT"

PYTHONPATH=src .venv-rl/bin/python scripts/rl_logit_diagnostic.py > "$OUT/stdout.log" 2>&1 &
PID=$!
echo "diagnostic pid $PID, hard limit ${LIMIT}s"

( sleep "$LIMIT"
  if kill -0 "$PID" 2>/dev/null; then
      echo "WATCHDOG: ${LIMIT}s limit reached; terminating $PID" | tee -a "$OUT/stdout.log"
      kill -TERM "$PID" 2>/dev/null
      sleep 10
      kill -KILL "$PID" 2>/dev/null
  fi ) &
WATCHDOG=$!

wait "$PID"; STATUS=$?
kill "$WATCHDOG" 2>/dev/null
echo "diagnostic exited with status $STATUS"
exit "$STATUS"
