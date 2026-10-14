#!/bin/bash
cd /root/autodl-tmp/fd-badcat-c3v2/exp/web_demo/c3v2_20261015/f3_heard
source /tmp/f3_e2e_env.sh
echo "e2e start $(date -u +%FT%TZ)"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY /root/miniconda3/envs/fd-sds/bin/python e2e_answers.py --concurrency 2
echo "e2e rc=$? end $(date -u +%FT%TZ)"
