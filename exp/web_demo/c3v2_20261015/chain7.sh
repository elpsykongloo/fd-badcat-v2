#!/bin/bash
# chain7: stop f1 after main comparison (skip max2/rel3 ablations), analyze, then latency bench base vs combo.
W=/root/autodl-tmp/fd-badcat-c3v2; E=$W/exp/web_demo/c3v2_20261015; F=$E/f1_wait
PY=/root/miniconda3/envs/fd-sds/bin/python
log() { echo "[chain7 $(date -u +%FT%TZ)] $*"; }
for k in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do unset $k; done
export NO_PROXY=127.0.0.1,localhost PYTHONUTF8=1 OMP_NUM_THREADS=2
log "waiting for f1 main pass2"
while true; do
  grep -q "main pass2(retry) exit" $F/driver.log && break
  grep -q "^finished" $F/progress.txt && break
  sleep 30
done
if ! grep -q "^finished" $F/progress.txt; then
  log "main done; stopping ablations (pgid 740294)"
  kill -TERM -740294 2>/dev/null; sleep 15
  for p in $(ss -Hltnp '( sport = :18131 )' | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do log "kill leftover backend $p"; kill $p; done
  kill $(cat $F/load_monitor.pid) 2>/dev/null
  printf 'stopped_after_main %s\n' "$(date -u +%FT%TZ)" > $F/progress.txt
fi
cd $F
$PY f1_analyze.py runs/full --out runs/full/summary.json > runs/full/summary.txt 2>&1; log "f1_analyze rc=$?"
if [ -f audit/audit_f1.py ]; then $PY audit/audit_f1.py runs/full > runs/full/audit.txt 2>&1; log "audit rc=$?"; fi
exec 9>/tmp/c3v2_gpu.lock; flock 9; log "gpu lock acquired"
cd $W
O=/root/autodl-tmp/fd-badcat-lat/exp/lat5/bench; L=$E/lat_bench; mkdir -p $L
C=(--backend-root $W --fixtures $O/fixtures --episodes $O/real_episodes/episodes.json --out $L --port 18011
   --net-delay-ms 40 --reply-stop-s 4 --repeats 2 --suites synthetic real
   --set engine.case_capture_dir=exp/web_demo/c3v2_20261015/lat_bench/demo_cases)
A=(--set engine.input_route_prefill=false --set engine.input_route_early_close=false
   --set engine.input_vad_silence_ms=100 --set engine.input_vad_silence_ms_playing=100
   --set engine.stream_startup_ms=300 --set engine.stream_resume_lead_ms=80 --set engine.client_local_hold=true
   --set "engine.end_hold_tiers={low: 0.40, default: 0.48, keep: 0.64}" --set engine.hold_tier_rules=true
   --set engine.hold_tier_shadow=false --set engine.hold_min_rtt_ms=50 --set engine.hold_low_max_s=2.2
   --set engine.hold_not_chinese_tier=default)
K=(--set engine.input_wait_uncertain_recover=true --set engine.input_recover_max=1
   --set engine.input_reply_partial_heard_s=0.5 --set engine.input_reply_partial_heard_history=true
   --set engine.input_reply_partial_heard_questions_only=true
   --set engine.input_vad_end_ms_playing=400 --set engine.input_vad_end_ms_playing_aec_only=true)
log "bench c3_base"; $PY scripts/bench_guarded_latency.py run --profile c3_base "${C[@]}" "${A[@]}" > $L/c3_base.log 2>&1; log "c3_base rc=$?"
log "bench c3_combo"; $PY scripts/bench_guarded_latency.py run --profile c3_combo "${C[@]}" "${A[@]}" "${K[@]}" > $L/c3_combo.log 2>&1; log "c3_combo rc=$?"
$PY scripts/bench_guarded_latency.py report $L/c3_base $L/c3_combo /root/autodl-tmp/fd-badcat-lat/exp/lat5/bench/cand/lat_final --out $L/report.md > $L/report.log 2>&1; log "report rc=$?"
log "ALL_DONE"