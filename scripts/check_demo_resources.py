#!/usr/bin/env python3
"""Read-only disk, GPU, memory and demo health receipt; never reads credentials."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request


def main(args):
    disk = shutil.disk_usage(args.root)
    row = {'wall_s':time.time(), 'disk_free_gib':disk.free/2**30,
           'disk_total_gib':disk.total/2**30, 'issues':[], 'services':{}}
    if row['disk_free_gib'] < args.min_disk_gib:
        row['issues'].append('disk_free_below_target')
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], text=True)
    row['gpus']=[]
    for line in output.strip().splitlines():
        values = [v.strip() for v in line.split(',')]
        gpu = dict(name=values[0],memory_used_mib=float(values[1]),memory_total_mib=float(values[2]),
                   utilization_pct=float(values[3]),temperature_c=float(values[4]),power_w=float(values[5]))
        gpu['free_gib'] = (gpu['memory_total_mib']-gpu['memory_used_mib'])/1024
        row['gpus'].append(gpu)
        if gpu['free_gib'] < args.min_gpu_gib:
            row['issues'].append('gpu_free_below_target')
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    row['memory_available_gib']=mem['MemAvailable']/2**20
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for name,url in [('omni','http://127.0.0.1:10003/v1/models'),('demo','http://127.0.0.1:18000/api/demo/info')]:
        try:
            with opener.open(url,timeout=5) as response:
                data=json.load(response)
                row['services'][name]={'healthy':response.status==200}
                if name=='demo':
                    row['demo_protocols']={key:data.get(key) for key in ('protocol','route_protocol','input_protocol','input_continuation','speech_scheduling','response_completion')}
                    row['capture_budget']=data.get('case_capture',{}).get('max_bytes')
        except Exception as exc:
            row['services'][name]={'healthy':False,'error_type':type(exc).__name__}
            if not args.allow_stopped:
                row['issues'].append(name+'_unhealthy')
    row['passed']=not row['issues']
    text=json.dumps(row,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text,end='')
    return 0 if row['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/fd-badcat'))
    parser.add_argument('--min-disk-gib',type=float,default=40)
    parser.add_argument('--min-gpu-gib',type=float,default=4)
    parser.add_argument('--allow-stopped',action='store_true')
    parser.add_argument('--output',type=Path)
    raise SystemExit(main(parser.parse_args()))
