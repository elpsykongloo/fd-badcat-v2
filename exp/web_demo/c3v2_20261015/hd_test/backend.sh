#!/bin/bash
# usage: backend.sh ARM  -> starts private backend for the arm, pid in hd_test/ARM.backend.pid
R=/root/autodl-tmp/fd-badcat-c3v2; T=$R/exp/web_demo/c3v2_20261015/hd_test; A=$1
cd $R
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=127.0.0.1,localhost FDBC_QWEN_URL=http://127.0.0.1:10003/v1/chat/completions FDBC_OMNI_TTS_URL=http://127.0.0.1:10003/v1/chat/completions FDBC_DEMO_VOICE_ADAPTER=1 FDBC_QWEN_TIMEOUT=60 OMP_NUM_THREADS=2
setsid nohup /root/miniconda3/envs/fd-sds/bin/python src/backend.py --config $T/$A.yaml --streaming --host 127.0.0.1 --port $(grep -A3 '^server:' $T/$A.yaml | grep port | awk '{print $2}') > $T/$A.backend.log 2>&1 < /dev/null &
echo $! > $T/$A.backend.pid
