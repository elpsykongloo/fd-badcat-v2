#!/bin/bash
cd /root/autodl-tmp/fd-badcat-c3v2/exp/web_demo/c3v2_20261015/f3_heard
P=/root/miniconda3/envs/fd-sds/bin/python
E="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY"
echo "main start $(date -u +%FT%TZ)"
$E $P run_f3.py --arms off,on050,on100,cf_trueprefix,cf_fullsent --repeats 0,1,2 --concurrency 2
echo "main rc=$? end $(date -u +%FT%TZ)"
echo "sup start $(date -u +%FT%TZ)"
$E $P run_f3.py --manifest supplement/manifest.jsonl --arms off,on050,cf_trueprefix,cf_fullsent --repeats 0,1,2 --concurrency 2 --results results_sup
echo "sup rc=$? end $(date -u +%FT%TZ)"
