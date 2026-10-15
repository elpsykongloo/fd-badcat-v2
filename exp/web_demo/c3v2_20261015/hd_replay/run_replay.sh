#!/bin/bash
E=/root/autodl-tmp/fd-badcat-c3v2/exp/web_demo/c3v2_20261015; P=$E/hd_replay
cd $P
export F2_POOL=$E/f2_vad/problem_size/all/latency_pool_fp8.json NO_PROXY=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
PY=/root/miniconda3/envs/fd-sds/bin/python
echo "start $(date -u +%FT%TZ)" >> progress.txt
$PY full_harness.py --replay sessions.json --out $P/run --procs 4 --conc 3 >> run.log 2>&1
$PY full_harness.py --replay sessions.json --out $P/run --procs 4 --conc 3 >> run.log 2>&1
echo "done $(date -u +%FT%TZ)" >> progress.txt
