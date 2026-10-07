#!/usr/bin/env python3
"""Four direct long-context requests, then four TTS streams; capacity, not latency."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

import aiohttp
import numpy as np

ROOT=Path('/root/autodl-tmp/fd-badcat')
sys.path.insert(0,str(ROOT/'src'))
import module
from p0_pro6000_baseline import sample_resources


async def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    sampler=asyncio.create_task(sample_resources(args.output))
    receipt={'scope':'direct-service-capacity-not-actor-admission-or-paper-latency','requests':[], 'tts':[]}
    text=('The forest has a river, trees, birds and a small house. '*500)+'\nDescribe the scene in one short sentence.'
    async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=180)) as client:
        async with client.post('http://127.0.0.1:10003/tokenize',json={'model':module.QWEN_MODEL,'prompt':text}) as response:
            tokens=await response.json()
            if response.status!=200:
                raise RuntimeError('Tokenization endpoint failed: '+str(tokens)[:200])
        # Use the tokenizer's own API to bound the test at the production limit.
        token_ids=tokens['tokens'][:3500]
        async with client.post('http://127.0.0.1:10003/detokenize',json={'model':module.QWEN_MODEL,'tokens':token_ids}) as response:
            decoded=await response.json()
            if response.status!=200: raise RuntimeError('Detokenization endpoint failed')
        async def request(index):
            start=time.monotonic()
            async with client.post('http://127.0.0.1:10003/v1/chat/completions',json={
                'model':module.QWEN_MODEL,'messages':[{'role':'user','content':decoded['prompt']}],
                'modalities':['text'],'max_tokens':32,'temperature':0,'seed':42}) as response:
                value=await response.json()
                receipt['requests'].append({'index':index,'http_status':response.status,
                    'elapsed_ms':(time.monotonic()-start)*1000,'usage':value.get('usage'),
                    'finish_reasons':[c.get('finish_reason') for c in value.get('choices',[])],
                    'passed':response.status==200 and bool(value.get('choices'))})
        await asyncio.gather(*(request(i) for i in range(4)))
    async def tts(index):
        samples=0;energy=0;rate=None
        try:
            async for chunk in module.tts_omni_stream('你好，很高兴认识你。今天我们一起探索森林里的小秘密。',voice_control={'speaker':'chelsie','seed':42}):
                pcm=np.frombuffer(chunk.pcm,dtype='<i2').astype(np.float64)/32768
                samples+=len(pcm);energy+=float(np.dot(pcm,pcm));rate=chunk.sample_rate
            receipt['tts'].append({'index':index,'audio_ms':samples/rate*1000,
                                   'rms':(energy/samples)**.5,'passed':samples>0 and energy>0})
        except Exception as exc:
            receipt['tts'].append({'index':index,'passed':False,'error_type':type(exc).__name__,'error':str(exc)[:200]})
    await asyncio.gather(*(tts(i) for i in range(4)))
    sampler.cancel()
    try: await sampler
    except asyncio.CancelledError: pass
    resources=[json.loads(line) for line in (args.output/'resources.jsonl').read_text().splitlines()]
    receipt['resources']={'sample_interval_s':1,'samples':len(resources),
        'peak_gpu_memory_mib':max(r['gpu_memory_mib'] for r in resources),
        'minimum_gpu_free_mib':min(r['gpu_total_mib']-r['gpu_memory_mib'] for r in resources),
        'peak_system_used_mib':max(r['system_used_mib'] for r in resources)}
    receipt['passed']=len(receipt['requests'])==4 and len(receipt['tts'])==4 and all(r['passed'] for r in receipt['requests']+receipt['tts'])
    (args.output/'summary.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    return 0 if receipt['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
