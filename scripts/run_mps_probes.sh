#!/usr/bin/env bash
# Supervisor for the MPS generation probes.
#
# One shared budget across ALL probes, model loading included, started before
# any child runs. Each probe is a separate process, so a Metal assertion --
# which aborts rather than raising -- takes down only that probe and this
# supervisor keeps going, preserving its stderr, exit status and whatever
# markers it managed to fsync.
#
# Probe 4 is chosen by the earlier results, not run blindly.
set -u
BUDGET="${1:-600}"
OUT="evidence/mps-probe"
mkdir -p "$OUT"
START=$(date +%s)
: > "$OUT/supervisor.log"

remaining() { echo $(( BUDGET - ( $(date +%s) - START ) )); }

run_probe() {
  local n="$1"
  local left; left=$(remaining)
  if [ "$left" -le 20 ]; then
      echo "probe $n: SKIPPED, only ${left}s of the shared budget left" | tee -a "$OUT/supervisor.log"
      return 99
  fi
  echo "--- probe $n (${left}s of budget left) ---" | tee -a "$OUT/supervisor.log"
  PYTHONPATH=src .venv-rl/bin/python scripts/mps_generation_probe.py --probe "$n" \
      > "$OUT/probe_${n}.stdout" 2> "$OUT/probe_${n}.stderr" &
  local pid=$!
  ( sleep "$left"; kill -KILL "$pid" 2>/dev/null ) & local wd=$!
  wait "$pid"; local status=$?
  kill "$wd" 2>/dev/null
  echo "probe $n exit status $status" | tee -a "$OUT/supervisor.log"
  if [ -s "$OUT/probe_${n}.stderr" ]; then
      echo "probe $n stderr:" | tee -a "$OUT/supervisor.log"
      sed 's/^/    /' "$OUT/probe_${n}.stderr" | tee -a "$OUT/supervisor.log"
  fi
  echo "probe $n last marker: $(tail -1 "$OUT/probe_${n}.jsonl" 2>/dev/null || echo none)" \
      | tee -a "$OUT/supervisor.log"
  return "$status"
}

for n in 1 2 3; do
  run_probe "$n"; echo "$n:$?" >> "$OUT/status.txt"
done
echo "probes 1-3 done, $(remaining)s left; probe 4 is chosen from these results" \
    | tee -a "$OUT/supervisor.log"
