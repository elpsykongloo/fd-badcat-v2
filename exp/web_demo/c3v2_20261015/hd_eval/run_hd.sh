#!/bin/bash
# usage: run_hd.sh SPLIT  (train|dev); resumable
E=/root/autodl-tmp/fd-badcat-c3v2/exp/web_demo/c3v2_20261015
S=$1
cd $E/hd_eval/$S
export F2_POOL=$E/f2_vad/problem_size/all/latency_pool_fp8.json NO_PROXY=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
PY=/root/miniconda3/envs/fd-sds/bin/python
for REP in 0 1; do
  $PY run_f2.py --out $E/hd_eval/$S/full --sets HD,HDP --holes 0 --reps $REP --arms $E/hd_eval/arms.json --procs 8 --conc 3 >> full_run.log 2>&1
  echo "$S rep $REP done $(date -u +%FT%TZ)" >> $E/hd_eval/progress.txt
done
cd $E/hd_eval && $PY hd_analyze.py $S/full --manifest $S/manifest.json --out $S/summary.json > $S/summary.txt 2>&1
