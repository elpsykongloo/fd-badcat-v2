#!/bin/bash
# f1_wait full pre-registered run (round 2): main comparison (off/on_max1 x 135 x 2 reps,
# prereg alternating order), one retry pass, then max2 and release-3.0 ablations.
# Serial, concurrency 1. Hard cap 12000 Omni requests across the three run dirs.
set -eu
cd /root/autodl-tmp/fd-badcat-c3v2
EXP=/root/autodl-tmp/fd-badcat-c3v2/exp/web_demo/c3v2_20261015/f1_wait
P="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY /root/miniconda3/envs/fd-sds/bin/python"
TOTAL=12000
B_MAIN=${B_MAIN:-9600}
if (( B_MAIN < 1 || B_MAIN > TOTAL )); then echo "B_MAIN must be in [1,12000]"; exit 1; fi
log() { echo "[driver $(date -u +%FT%TZ)] $*"; }
used() { [ -f "$1/calls.jsonl" ] && grep -c '"omni": true' "$1/calls.jsonl" || echo 0; }
# read-only load monitor (harness_fixes.md H7)
( while true; do
    echo "$(date -u +%FT%TZ) gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ') prod_conn=$(ss -Htn state established '( sport = :18000 )' | wc -l)"
    sleep 30
  done ) >> $EXP/load_monitor.log 2>&1 &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
echo $MON > $EXP/load_monitor.pid
log "monitor pid=$MON"
log "start main budget=$B_MAIN"
$P $EXP/f1_run.py --run $EXP/runs/full --arms off,on_max1 --reps 2 --budget $B_MAIN
log "main pass1 exit=$?"
$P $EXP/f1_run.py --run $EXP/runs/full --arms off,on_max1 --reps 2 --budget $B_MAIN
log "main pass2(retry) exit=$?"
U_MAIN=$(used $EXP/runs/full)
REM=$((TOTAL - U_MAIN))
if (( REM <= 0 )); then echo "Total request budget exhausted"; exit 1; fi
B_MAX2=$(( REM * 35 / 100 )); B_REL3=$(( REM - B_MAX2 ))
log "main used=$U_MAIN remaining=$REM -> max2=$B_MAX2 rel3=$B_REL3"
$P $EXP/f1_run.py --run $EXP/runs/abl_max2 --arms on_max2 --reps 2 --scenarios P_timeout,S_double_cut --budget $B_MAX2
log "abl_max2 exit=$?"
$P $EXP/f1_run.py --run $EXP/runs/abl_max2 --arms on_max2 --reps 2 --scenarios P_timeout,S_double_cut --budget $B_MAX2
log "abl_max2 retry exit=$?"
U_MAX2=$(used $EXP/runs/abl_max2)
B_REL3=$(( TOTAL - U_MAIN - U_MAX2 ))
if (( B_REL3 <= 0 )); then echo "Total request budget exhausted"; exit 1; fi
log "rel3 budget=$B_REL3"
$P $EXP/f1_run.py --run $EXP/runs/abl_rel3 --arms off_rel3,on_max1_rel3 --reps 1 --scenarios P_timeout,N1_resume,D_resume_late,N5_wait_natural --budget $B_REL3
log "abl_rel3 exit=$?"
$P $EXP/f1_run.py --run $EXP/runs/abl_rel3 --arms off_rel3,on_max1_rel3 --reps 1 --scenarios P_timeout,N1_resume,D_resume_late,N5_wait_natural --budget $B_REL3
log "abl_rel3 retry exit=$?"
log "omni used main=$(used $EXP/runs/full) max2=$(used $EXP/runs/abl_max2) rel3=$(used $EXP/runs/abl_rel3)"
kill $MON
log "ALL_DONE"
