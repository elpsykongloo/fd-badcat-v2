#!/bin/bash
# Rerun the trials that overlapped live demo sessions (08:56-09:06 UTC) during the r2_wr3 latency bench.
exec 8>/tmp/r2rr.lock; flock -n 8 || { echo "r2rr already running"; exit 0; }
W=/root/autodl-tmp/fd-badcat-r2; D=$W/exp/web_demo/r2_20261015; PY=/root/miniconda3/envs/fd-sds/bin/python
log(){ echo "[r2rr $(date -u +%FT%TZ)] $*"; }
for k in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do unset $k; done
export NO_PROXY=127.0.0.1,localhost PYTHONUTF8=1 OMP_NUM_THREADS=2
cd $W
for p in $(ss -Hltnp '( sport = :18011 )' | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do kill $p; done; sleep 2
[ -e $W/exp/lat5/bench ] || ln -s /root/autodl-tmp/fd-badcat-lat/exp/lat5/bench $W/exp/lat5/bench
O=/root/autodl-tmp/fd-badcat-lat/exp/lat5/bench; L=$D/lat_bench; rm -rf $L/r2_wr3_rerun
C=(--backend-root $W --fixtures $O/fixtures --episodes $O/real_episodes/episodes.json --out $L --port 18011
   --net-delay-ms 40 --reply-stop-s 4 --repeats 2 --suites synthetic real
   --set engine.case_capture_dir=exp/web_demo/r2_20261015/lat_bench/demo_cases)
A=(--set engine.input_route_prefill=false --set engine.input_route_early_close=false
   --set engine.input_vad_silence_ms=100 --set engine.input_vad_silence_ms_playing=100
   --set engine.stream_startup_ms=300 --set engine.stream_resume_lead_ms=80 --set engine.client_local_hold=true
   --set "engine.end_hold_tiers={low: 0.40, default: 0.48, keep: 0.64}" --set engine.hold_tier_rules=true
   --set engine.hold_tier_shadow=false --set engine.hold_min_rtt_ms=50 --set engine.hold_low_max_s=2.2
   --set engine.hold_not_chinese_tier=default)
log "rerun start"
$PY scripts/bench_guarded_latency.py run --profile r2_wr3_rerun "${C[@]}" "${A[@]}" \
  --only pause_zh_b pause_zh_c pause_en_a cont_ follow_ d90e22 > $L/r2_wr3_rerun.log 2>&1; log "rerun rc=$?"
rm -f $W/exp/lat5/bench
log "ALL_DONE"
