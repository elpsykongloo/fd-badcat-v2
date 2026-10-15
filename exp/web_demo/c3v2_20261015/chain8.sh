#!/bin/bash
# chain8: after chain7, test input_wait_release_s (2 = production) at 3/5/8 and 0 (never) on
# HumDial train+dev (virtual clock, pooled latency) and the 11 real replay sessions.
E=/root/autodl-tmp/fd-badcat-c3v2/exp/web_demo/c3v2_20261015
PY=/root/miniconda3/envs/fd-sds/bin/python
log() { echo "[chain8 $(date -u +%FT%TZ)] $*"; }
for k in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do unset $k; done
export NO_PROXY=127.0.0.1,localhost F2_POOL=$E/f2_vad/problem_size/all/latency_pool_fp8.json
until grep -q ALL_DONE $E/chain7.log; do sleep 60; done
exec 9>/tmp/c3v2_gpu.lock; flock 9; log "gpu lock acquired"
cat > $E/hd_eval/arms_rel.json <<'J'
{"base": {"engine": {}, "latency": "pool", "note": "production release 2.0"},
 "rel3": {"engine": {"input_wait_release_s": 3.0}, "latency": "pool"},
 "rel5": {"engine": {"input_wait_release_s": 5.0}, "latency": "pool"},
 "rel8": {"engine": {"input_wait_release_s": 8.0}, "latency": "pool"},
 "rel0": {"engine": {"input_wait_release_s": 0}, "latency": "pool", "note": "never release (pre 10-12)"}}
J
sed 's/"latency": "pool", //; s/, "latency": "pool"//' $E/hd_eval/arms_rel.json > $E/hd_replay/arms_rel.json
for S in dev train; do
  cd $E/hd_eval/$S
  for REP in 0 1; do
    $PY run_f2.py --out $E/hd_eval/$S/rel --sets HD,HDP --holes 0 --reps $REP --arms $E/hd_eval/arms_rel.json --procs 8 --conc 3 >> rel_run.log 2>&1
    log "$S rel rep $REP rc=$?"
  done
  cd $E/hd_eval && $PY hd_analyze.py $S/rel --manifest $S/manifest.json --out $S/rel_summary.json > $S/rel_summary.txt 2>&1; log "$S analyze rc=$?"
done
cd $E/hd_replay
for i in 1 2; do $PY full_harness.py --replay sessions.json --arms arms_rel.json --out $E/hd_replay/run_rel --procs 4 --conc 3 >> run_rel.log 2>&1; done
log "replay rc=$?"
$PY replay_cmp.py $E/hd_replay/run_rel > cmp_rel.txt 2>&1; log "replay cmp rc=$?"
log "ALL_DONE"