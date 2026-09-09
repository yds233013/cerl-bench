#!/usr/bin/env bash
set -u
BUDGET="${1:-1200}"
OUT="evidence/rl-v2-smoke-2rollout"
mkdir -p "$OUT"
PYTHONPATH=src .venv-rl/bin/python scripts/v2_two_rollout_smoke.py \
  > "$OUT/stdout.log" 2> "$OUT/stderr.log" &
PID=$!
echo "smoke pid $PID, external hard limit ${BUDGET}s (covers loading)"
( sleep "$BUDGET"
  if kill -0 "$PID" 2>/dev/null; then
      echo "WATCHDOG: ${BUDGET}s reached; terminating $PID" | tee -a "$OUT/stderr.log"
      kill -TERM "$PID" 2>/dev/null; sleep 15; kill -KILL "$PID" 2>/dev/null
  fi ) &
WD=$!
wait "$PID"; STATUS=$?
kill "$WD" 2>/dev/null
echo "smoke exited with status $STATUS"
exit "$STATUS"
