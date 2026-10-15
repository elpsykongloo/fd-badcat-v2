#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
if [[ ${1:-} != --foreground ]]; then
  setsid nohup bash "$0" --foreground > driver_fp8.log 2>&1 < /dev/null &
  echo $! > driver_fp8.pid
  echo "Started PID $!"
  exit 0
fi
exec 9>/tmp/c3v2_gpu.lock
flock 9
export NO_PROXY=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
PY=/root/miniconda3/envs/fd-sds/bin/python
export F2_POOL=$HERE/problem_size/all/latency_pool_fp8.json
OUT=full_fp8; mkdir -p $OUT
printf 'running %s\n' "$(date -u +%FT%TZ)" > progress_fp8.txt
trap 'rc=$?; printf "finished rc=%s %s\n" "$rc" "$(date -u +%FT%TZ)" > progress_fp8.txt' EXIT
for REP in 0 1 2; do
  $PY run_f2.py --out $OUT --sets H,P,Q,L --holes 0,0.3,0.4,0.5 --reps $REP --arms arms_fp8.json --procs 8 --conc 3 >> $OUT/run.log 2>&1
  printf 'rep %s done %s\n' "$REP" "$(date -u +%FT%TZ)" >> progress_fp8.txt
done
$PY analyze_f2.py $OUT --prereg prereg.json --out $OUT/summary.json >> $OUT/run.log 2>&1
