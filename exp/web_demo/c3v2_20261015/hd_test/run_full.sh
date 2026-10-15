#!/bin/bash
# HumDial test (noisy, cn+en, 4999) whole-dialogue, arms base/combo/combo_W; resumable
E=/root/autodl-tmp/fd-badcat-c3v2/exp/web_demo/c3v2_20261015; T=$E/hd_test
cd $T
export F2_POOL=$E/f2_vad/problem_size/all/latency_pool_fp8.json NO_PROXY=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
PY=/root/miniconda3/envs/fd-sds/bin/python
echo "start $(date -u +%FT%TZ)" >> full_progress.txt
$PY full_harness.py --out $T/full --procs 8 --conc 3 >> full_run.log 2>&1
echo "pass1 done $(date -u +%FT%TZ)" >> full_progress.txt
# second pass picks up units that failed (transport errors)
$PY full_harness.py --out $T/full --procs 8 --conc 3 >> full_run.log 2>&1
echo "done $(date -u +%FT%TZ)" >> full_progress.txt
$PY hdt_analyze.py full --out full/summary.json --diffs full/diffs.jsonl > full/summary.txt 2>&1
echo "analyzed $(date -u +%FT%TZ)" >> full_progress.txt
