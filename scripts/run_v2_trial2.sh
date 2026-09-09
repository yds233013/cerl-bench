#!/usr/bin/env bash
# Corrected v2 training trial, from 94ab6eb.
#
# One shared budget covering model loading, preflight, generation and backward
# work, started before any child runs. Preflight and trial are separate
# processes: a Metal assertion aborts rather than raising, so running them apart
# keeps an abort in one from touching the other's record. The trial starts only
# if preflight exits 0.
set -u
BUDGET="${1:-1200}"
OUT="evidence/rl-v2-trial2"
mkdir -p "$OUT"
START=$(date +%s)
: > "$OUT/supervisor.log"
log() { echo "$@" | tee -a "$OUT/supervisor.log"; }
remaining() { echo $(( BUDGET - ( $(date +%s) - START ) )); }

run_child() {
  local label="$1"; shift
  local left; left=$(remaining)
  if [ "$left" -le 30 ]; then log "$label: SKIPPED, only ${left}s left"; return 99; fi
  log "--- $label (${left}s of the shared ${BUDGET}s budget left) ---"
  "$@" > "$OUT/${label}.stdout" 2> "$OUT/${label}.stderr" &
  local pid=$!
  ( sleep "$left"; kill -TERM "$pid" 2>/dev/null; sleep 10; kill -KILL "$pid" 2>/dev/null ) &
  local wd=$!
  wait "$pid"; local status=$?
  kill "$wd" 2>/dev/null
  log "$label exit status $status"
  [ -s "$OUT/${label}.stderr" ] && { log "$label stderr:"; sed 's/^/    /' "$OUT/${label}.stderr" | tee -a "$OUT/supervisor.log"; }
  return "$status"
}

run_child preflight env PYTHONPATH=src .venv-rl/bin/python scripts/v2_preflight.py
PRE=$?
echo "preflight:$PRE" >> "$OUT/status.txt"
if [ "$PRE" -ne 0 ]; then
  log "preflight did not pass; the trial will NOT start. Evidence preserved in $OUT."
  exit "$PRE"
fi

# Whatever is left of the shared budget, minus a margin for the final write.
LEFT=$(remaining)
INNER=$(( (LEFT - 60) / 60 ))
log "preflight passed; running the trial with an in-process deadline of ${INNER} minutes"
run_child trial env PYTHONPATH=src .venv-rl/bin/python -m cerl_rl.pilot_v2 \
  --updates 1 --group-size 4 --lr 1e-5 --lora-rank 8 \
  --deadline-minutes "$INNER" --training-only --out "$OUT" --allow-existing
TRIAL=$?
echo "trial:$TRIAL" >> "$OUT/status.txt"
log "trial exit status $TRIAL; $(remaining)s of budget left"
exit "$TRIAL"
