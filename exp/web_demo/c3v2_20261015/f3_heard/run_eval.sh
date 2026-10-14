#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
if [[ ${1:-} != --foreground ]]; then
  setsid nohup bash "$0" --foreground > driver.log 2>&1 < /dev/null &
  echo $! > driver.pid
  echo "Started PID $!; progress: $HERE/progress.txt; log: $HERE/driver.log"
  exit 0
fi
exec 9>/tmp/c3v2_gpu.lock
flock -n 9 || { echo "Another c3v2 GPU job owns the lock"; exit 1; }
export NO_PROXY=127.0.0.1,localhost FDBC_QWEN_TIMEOUT=60
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
PY=/root/miniconda3/envs/fd-sds/bin/python
printf 'running %s\n' "$(date -u +%FT%TZ)" > progress.txt
trap 'rc=$?; printf "finished rc=%s %s\n" "$rc" "$(date -u +%FT%TZ)" > progress.txt' EXIT
$PY run_f3.py --arms off,on050,on100,on050_history,cf_trueprefix,cf_fullsent --repeats 0,1,2 --concurrency "${CONC:-2}" --item-timeout 60 --results results
$PY run_f3.py --manifest supplement/manifest.jsonl --arms off,on050,on050_history,cf_trueprefix,cf_fullsent --repeats 0,1,2 --concurrency "${CONC:-2}" --item-timeout 60 --results results_sup
$PY analyze_f3.py --results results/live --arms off,on050,on100,on050_history,cf_trueprefix,cf_fullsent --out results/summary.json
$PY analyze_sup.py --results results_sup/live --arms off,on050,on050_history,cf_trueprefix,cf_fullsent --out results_sup/summary.json
$PY e2e_answers.py --results results/live --out e2e/answers.jsonl --concurrency "${CONC:-2}"
