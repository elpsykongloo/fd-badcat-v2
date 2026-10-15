#!/bin/bash
# r2 chain B: after chain A, same flow for wr2 (base/wr1 served from seed cache).
D=/root/autodl-tmp/fd-badcat-r2/exp/web_demo/r2_20261015; H=$D/hd_eval; L=/root/autodl-tmp/fd-badcat-lat
C3=/root/autodl-tmp/fd-badcat-c3v2/exp/web_demo/c3v2_20261015
PY=/root/miniconda3/envs/fd-sds/bin/python
log(){ echo "[r2b $(date -u +%FT%TZ)] $*"; }
omni_up(){ curl --noproxy '*' -fsS -m 5 http://127.0.0.1:10003/v1/models > /dev/null 2>&1 && curl --noproxy '*' -fsS -m 5 http://127.0.0.1:18000/api/demo/info > /dev/null 2>&1; }
restart_omni(){
  local cfg=$1 ts=$(date -u +%Y%m%dT%H%M%SZ) envcfg="-u QWEN_DEPLOY_CONFIG"
  [ -n "$cfg" ] && envcfg="QWEN_DEPLOY_CONFIG=$cfg"
  tmux send-keys -t lat-live C-c
  for i in $(seq 1 60); do ss -ltn | grep -q ':10003 ' || break; sleep 3; done
  tmux kill-session -t lat-live 2>/dev/null
  for i in $(seq 1 60); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$m" -lt 4000 ] && break; sleep 3; done
  tmux new-session -d -s lat-live "cd $L && exec env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u QWEN_EXTRA_ARGS -u FDBC_DEMO_RESIDUAL_OPT $envcfg bash setup/start_demo_pro6000.sh --port 18000 > $L/exp/web_demo/lat-live-launch-$ts.log 2>&1"
  for i in $(seq 1 300); do omni_up && break; sleep 5; done
  omni_up && log "omni up cfg=${cfg:-production}" || log "OMNI FAILED cfg=${cfg:-production}"
}
for k in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do unset $k; done
export NO_PROXY=127.0.0.1,localhost F2_POOL=$C3/f2_vad/problem_size/all/latency_pool_fp8.json
log "lock"
restart_omni $C3/omni_eval_seqs32.yaml
for S in train dev; do
  cd $H/$S
  $PY run_f2.py --out $H/$S/wr2 --sets HD,HDP --holes 0 --reps 0 --arms $H/arms_wr.json --procs 8 --conc 3 >> wr2_run.log 2>&1
  log "$S rep0 rc=$?"
  cd $H && $PY hd_analyze.py $S/wr2 --manifest $S/manifest.json --out $S/wr2_summary.json > $S/wr2_summary.txt 2>&1; log "$S analyze rc=$?"
done
restart_omni ""
log "ALL_DONE"