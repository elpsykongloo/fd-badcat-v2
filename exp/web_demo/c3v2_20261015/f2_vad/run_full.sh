#!/bin/bash
# f2_vad full run (NOT yet executed). One repeat per invocation so a partial run is usable;
# every invocation resumes (finished units are skipped, cached Omni results are reused).
# Omni concurrency = procs * conc = 24 (offline cap). Do not overlap with any acceptance run
# that uses the production 2 s decision budget, nor with other offline Omni batches.
set -e
cd "$(dirname "$0")"
PY=/root/miniconda3/envs/fd-sds/bin/python
OUT=${OUT:-full}
mkdir -p "$OUT"
for REP in 0 1 2; do
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    $PY run_f2.py --out "$OUT" --sets H,P,Q,L --holes 0,0.3,0.4,0.5 --reps $REP \
    --arms arms.json --procs ${PROCS:-8} --conc ${CONC:-3} >> "$OUT/run.log" 2>&1
done
$PY analyze_f2.py "$OUT" --prereg prereg.json --out "$OUT/summary.json" >> "$OUT/run.log" 2>&1
