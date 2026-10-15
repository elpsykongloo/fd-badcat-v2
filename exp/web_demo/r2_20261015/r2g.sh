#!/bin/bash
# r2 chain G: after chain F, write wr3 into configs/demo_chat.yaml, pytest, guarded-turns canary, latency bench.
W=/root/autodl-tmp/fd-badcat-r2; D=$W/exp/web_demo/r2_20261015; C3=/root/autodl-tmp/fd-badcat-c3v2/exp/web_demo/c3v2_20261015
PY=/root/miniconda3/envs/fd-sds/bin/python
log(){ echo "[r2g $(date -u +%FT%TZ)] $*"; }
for k in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do unset $k; done
export NO_PROXY=127.0.0.1,localhost PYTHONUTF8=1 OMP_NUM_THREADS=2
until grep -q ALL_DONE $D/r2f.log; do sleep 20; done
log "start"
cd $W
$PY - <<'EOF'
import json, yaml
from pathlib import Path
W = Path('/root/autodl-tmp/fd-badcat-r2'); D = W / 'exp/web_demo/r2_20261015'
p = W / 'configs/demo_chat.yaml'
t = p.read_text(encoding='utf-8')
A = "停止本身不需要回答。\n"
add = "用户说明自己要暂时离开、先去做别的事（如“稍等，我去开个门”“等一下，我接个电话”“Hold on, someone's at the door”），也是stop_only：用户回来后会再开口。"
line = "    stop_only：明确要求停止或别说了，没有同时提出新的问题、纠正或要求。停止本身不需要回答。\n"
assert t.count(line) == 1
t = t.replace(line, line[:-1] + add + "\n")
p.write_text(t, encoding='utf-8')
got = yaml.safe_load(t)['prompts']['input_route']
want = json.loads((D / 'hd_eval/arms_wr34.json').read_text())['wr3']['prompts']['input_route']
assert got == want, 'prompt mismatch'
print('config ok')
EOF
log "config rc=$?"
git -C $W diff --stat
timeout 1800 $PY -m pytest -q -x tests > $D/pytest.log 2>&1; log "pytest rc=$? $(tail -1 $D/pytest.log)"
rm -rf $D/guarded_turns_wr3
timeout 900 $PY scripts/check_guarded_turns.py --output $D/guarded_turns_wr3 > $D/guarded_turns_wr3.log 2>&1; log "guarded_turns rc=$?"
[ -e $W/exp/lat5/bench ] || ln -s /root/autodl-tmp/fd-badcat-lat/exp/lat5/bench $W/exp/lat5/bench
O=/root/autodl-tmp/fd-badcat-lat/exp/lat5/bench; L=$D/lat_bench; mkdir -p $L
C=(--backend-root $W --fixtures $O/fixtures --episodes $O/real_episodes/episodes.json --out $L --port 18011
   --net-delay-ms 40 --reply-stop-s 4 --repeats 2 --suites synthetic real
   --set engine.case_capture_dir=exp/web_demo/r2_20261015/lat_bench/demo_cases)
A=(--set engine.input_route_prefill=false --set engine.input_route_early_close=false
   --set engine.input_vad_silence_ms=100 --set engine.input_vad_silence_ms_playing=100
   --set engine.stream_startup_ms=300 --set engine.stream_resume_lead_ms=80 --set engine.client_local_hold=true
   --set "engine.end_hold_tiers={low: 0.40, default: 0.48, keep: 0.64}" --set engine.hold_tier_rules=true
   --set engine.hold_tier_shadow=false --set engine.hold_min_rtt_ms=50 --set engine.hold_low_max_s=2.2
   --set engine.hold_not_chinese_tier=default)
log "bench r2_wr3"; $PY scripts/bench_guarded_latency.py run --profile r2_wr3 "${C[@]}" "${A[@]}" > $L/r2_wr3.log 2>&1; log "r2_wr3 rc=$?"
$PY scripts/bench_guarded_latency.py report $L/r2_wr3 $C3/lat_bench/c3_combo $O/cand/lat_final --out $L/report.md > $L/report.log 2>&1; log "report rc=$?"
rm $W/exp/lat5/bench
log "ALL_DONE"
