#!/bin/bash
cd /root/autodl-tmp/fd-badcat-c3v2/exp/web_demo/c3v2_20261015/f3_heard
export NO_PROXY=127.0.0.1,localhost FDBC_QWEN_TIMEOUT=60
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
PY=/root/miniconda3/envs/fd-sds/bin/python
echo "q start $(date -u +%FT%TZ)"
$PY run_f3.py --arms on050_q,on050_q_history --repeats 0,1,2 --concurrency 4 --item-timeout 60 --results results
echo "q main rc=$? $(date -u +%FT%TZ)"
$PY run_f3.py --manifest supplement/manifest.jsonl --arms on050_q,on050_q_history --repeats 0,1,2 --concurrency 4 --item-timeout 60 --results results_sup
echo "q sup rc=$? $(date -u +%FT%TZ)"
$PY analyze_f3.py --results results/live --arms off,on050,on050_history,on050_q,on050_q_history --out results/summary_q.json
$PY analyze_sup.py --results results_sup/live --arms off,on050,on050_history,on050_q,on050_q_history --out results_sup/summary_q.json
echo "q done $(date -u +%FT%TZ)"
