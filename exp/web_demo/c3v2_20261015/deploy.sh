#!/bin/bash
# deploy 1376ebd9 (f1+f3 on) to lat-live after chain10 bench, then restore the public gateway.
E=/root/autodl-tmp/fd-badcat-c3v2/exp/web_demo/c3v2_20261015; L=/root/autodl-tmp/fd-badcat-lat
log() { echo "[deploy $(date -u +%FT%TZ)] $*"; }
omni_up(){ curl --noproxy '*' -fsS -m 5 http://127.0.0.1:10003/v1/models > /dev/null 2>&1 && curl --noproxy '*' -fsS -m 5 http://127.0.0.1:18000/api/demo/info > /dev/null 2>&1; }
until grep -q ALL_DONE $E/chain10.log; do sleep 20; done
log "bench done"
cd $L
git fetch -q /root/autodl-tmp/fd-badcat-c3v2 c3v2 && git merge --ff-only FETCH_HEAD 2>&1 | tail -1
log "lat tree at $(git log --oneline -1)"
tmux send-keys -t lat-live C-c; sleep 15
tmux kill-session -t lat-live 2>/dev/null
for i in $(seq 1 30); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 2; done
ts=$(date -u +%Y%m%dT%H%M%SZ)
tmux new-session -d -s lat-live "cd $L && exec env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u QWEN_EXTRA_ARGS -u QWEN_DEPLOY_CONFIG -u FDBC_DEMO_RESIDUAL_OPT bash setup/start_demo_pro6000.sh --port 18000 > $L/exp/web_demo/lat-live-launch-$ts.log 2>&1"
for i in $(seq 1 300); do omni_up && break; sleep 5; done
omni_up && log "live up" || { log "LIVE FAILED"; exit 1; }
curl --noproxy '*' -fsS http://127.0.0.1:18000/api/demo/info | python3 -c "import sys,json; d=json.load(sys.stdin); print({k:d.get(k) for k in ('reply_context','reply_partial_heard_s','reply_partial_heard_history','reply_partial_heard_questions_only','browser_auto_gain_control')})"
grep -n "input_wait_uncertain_recover" $L/configs/demo_chat.yaml
systemctl start casecade-share-gateway; sleep 3; log "gateway $(systemctl is-active casecade-share-gateway)"
log "DEPLOY_DONE"