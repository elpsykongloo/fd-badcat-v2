#!/usr/bin/env python3
"""Serial, self-authored TTS + real Chromium continuity diagnostics.

Uses production SpeechPipeline, SpeechPlayer, telemetry and TTS adapters. Browser
fixtures supply fixed reply text (not the live perception/Actor decision path).
No physical device or listening claim; seq4 engineering data, not serial-eval.
"""
import argparse
import asyncio
from contextlib import aclosing
import json
from pathlib import Path
import sys
import time
import unicodedata

import numpy as np
import soundfile as sf
from aiohttp import web, WSMsgType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import module
from speech_stream import SpeechPipeline
from request_capacity import RequestCapacity, NORMAL, CONTROL

VOICE = {'speaker': 'chelsie', 'seed': 42}
TEXTS = [
    '你好，我们接着聊。',
    '窗外的小鸟停在树枝上，正在等待雨停。',
    '请先把蓝色杯子放在桌上，然后慢慢打开窗户。',
    '你希望我先解释原理，还是先举一个例子？\n',
    'The little bird waited beside the window until the rain stopped.',
    '今天我们学习一个词，hello，它表示你好。',
]
LONG = ('今天我们来讲一个关于森林图书馆的小故事。'
        '清晨，小松鼠背着装满书的背包，沿着山坡上的小路慢慢走进森林。'
        '它在一棵大树下摆好桌子，把故事书和画册整整齐齐地放在上面。'
        '小兔子先到了，它挑了一本讲星星的书，坐在树荫下认真地读起来。'
        '接着，小鹿带来了温热的茶水，小鸟负责给每一本书贴上漂亮的标签。'
        '到了下午，大家轮流分享自己最喜欢的一页，也把不明白的问题记在纸上。'
        '太阳快要落山的时候，它们约好明天继续相聚，一起把这座小小的图书馆照顾好。')


def normalized(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', value).casefold() if c.isalnum())


async def raw_probe(output, repeats):
    rows = []
    for index, text in enumerate(TEXTS):
        for repeat in range(repeats):
            started = time.perf_counter()
            row = {'text_index': index, 'repeat': repeat, 'text': text, 'chunks': []}
            pcm = []
            try:
                async for c in module.tts_omni_stream(text, voice_control=VOICE, timing=True):
                    row['chunks'].append({'at_ms': (time.perf_counter()-started)*1000,
                        'samples': len(c.pcm)//2, 'rate': c.sample_rate, **(c.timing or {})})
                    pcm.append(c.pcm)
                row['elapsed_ms'] = (time.perf_counter()-started)*1000
                rate = row['chunks'][0]['rate']
                audio = np.frombuffer(b''.join(pcm), dtype='<i2')
                row['audio_ms'] = len(audio)/rate*1000
                row['rtf'] = row['elapsed_ms']/row['audio_ms']
                path = output / f'raw-{index}-{repeat}.wav'
                sf.write(path, audio, rate, subtype='PCM_16')
                heard = await asyncio.to_thread(module.asr, path)
                row.update(status='completed', recognized=heard,
                           asr_match=normalized(heard)==normalized(text))
                prior_ms = 0
                lead = 0
                for c in row['chunks']:
                    lead = max(lead, c['at_ms']-row['chunks'][0]['at_ms']-prior_ms)
                    prior_ms += c['samples']/c['rate']*1000
                row['minimum_continuous_lead_ms'] = lead
            except Exception as exc:
                row.update(status='error', error=f'{type(exc).__name__}: {exc}')
            rows.append(row)
            (output/'raw.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in {'chunks','text','recognized'}},ensure_ascii=False),flush=True)
    return rows


HTML = '''<!doctype html><meta charset="utf-8"><button id="start">Start</button>
<script type="module">
import {SpeechPlayer} from '/static/speech-player.js';
import {DemoTelemetry} from '/static/demo-telemetry.js';
window.events=[]; window.done=false;
document.querySelector('button').onclick=async()=>{
 const context=new AudioContext(); await context.resume(); window.context=context;
 const ws=new WebSocket(`ws://${location.host}/ws`); ws.binaryType='arraybuffer';
 const send=(event,data)=>{window.events.push({event,data}); if(ws.readyState===1)ws.send(JSON.stringify({event,data}));};
 const telemetry=new DemoTelemetry(send,context,()=>ws.bufferedAmount); telemetry.enabled=true;
 const player=new SpeechPlayer(context,send,s=>telemetry.playback(s)); window.player=player;
 ws.onmessage=({data})=>{
  if(data instanceof ArrayBuffer){player.packet(data);return;}
  const m=JSON.parse(data);
  if(m.event==='speech_start')player.start(m.data);
  if(m.event==='speech_audio_end')player.finish(m.data);
  if(m.event==='speech_cancelled'){
   telemetry.snapshot('cancel',player.speech);player.cancel();
  }
  if(m.event==='complete'){window.done=true;window.result=m.data;}
 };
 window.timer=setInterval(()=>player.pollStart(),10);
};
</script>'''


async def browser_case(browser, name, text, startup, prefetch, route_audio=None, cancel=False):
    result = {'name':name, 'startup_ms':startup, 'prefetch_ms':prefetch,
              'text':text, 'events':[], 'telemetry':[], 'controls':[], 'errors':[]}
    capacity = RequestCapacity()
    route_messages = None
    if route_audio is not None:
        # Prepare the fixture before timing; lazy resampler imports must not
        # block the WebSocket event loop during the first TTS request.
        import yaml
        from messages import build_audio_content
        config = yaml.safe_load((ROOT/'configs/demo_chat.yaml').read_text())
        audio, rate = sf.read(route_audio,dtype='float32')
        content = build_audio_content(module._mono_16k(audio,rate),16000,'audio_url')
        route_messages = [{'role':'system','content':config['prompts']['input_route']},
                          {'role':'user','content':[{'type':'text','text':'{"assistant_playing":true}'},content]}]
    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        queue, done = asyncio.Queue(), asyncio.Event()
        began = time.perf_counter()
        async def text_fn(_):
            yield text
        async def tts_fn(sentence):
            async with capacity.slot(NORMAL):
                async with aclosing(module.tts_omni_stream(sentence, voice_control=VOICE, timing=True)) as source:
                    async for chunk in source:
                        yield chunk
        p = SpeechPipeline(1, queue, [], text_fn, tts_fn, startup_ms=startup,
                           prefetch_ms=prefetch, diagnostics=True, track_sentences=True)
        await ws.send_json({'event':'speech_start','data':{
            'utterance_id':1,'buffer_ms':600,'startup_ms':startup}})
        async def receive():
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                value = json.loads(msg.data)
                event, data = value['event'], value['data']
                if event == 'playback_progress':
                    p.progress(data['played_samples'])
                    result['max_outstanding_ms'] = max(result.get('max_outstanding_ms',0),
                        (p.sent-data['played_samples'])/(p.rate or 24000)*1000)
                    if data.get('ended'):
                        result['played'] = data
                        done.set()
                elif event == 'demo_telemetry':
                    result['telemetry'].append(data)
                elif event == 'playback_stopped':
                    result['stopped'] = data
                    done.set()
        async def route():
            if route_audio is None:
                return
            for _ in range(3):
                await asyncio.sleep(2)
                call_started=time.perf_counter()
                async with capacity.slot(CONTROL):
                    answer=''.join([x async for x in module.llm_qwen3o_stream(route_messages,route=True)])
                from control_labels import parse_route
                if parse_route(answer) is None:
                    raise RuntimeError('Invalid route result during overlap probe')
                result['controls'].append({'elapsed_ms':(time.perf_counter()-call_started)*1000,
                                           'answer':answer})
        async def cancel_later():
            await asyncio.sleep(3)
            p.cancel()
            await ws.send_json({'event':'speech_cancelled','data':{'utterance_id':1}})
            # Deliberate old packet: the production player must ignore it.
            from speech_stream import PCM_HEADER
            await ws.send_bytes(PCM_HEADER.pack(b'FDS1',1,999,24000)+bytes(1920))
        tasks=[asyncio.create_task(receive()),asyncio.create_task(route())]
        if cancel:
            tasks.append(asyncio.create_task(cancel_later()))
        try:
            while True:
                ev=await asyncio.wait_for(queue.get(),90)
                if ev.kind=='audio':
                    await ws.send_bytes(ev.data['wire'])
                else:
                    result['events'].append({'at_ms':(time.perf_counter()-began)*1000,
                                             'kind':ev.kind, 'data':ev.data})
                    if ev.kind=='audio_end':
                        await ws.send_json({'event':'speech_audio_end','data':{'utterance_id':1,**ev.data}})
                    elif ev.kind=='error':
                        raise RuntimeError(ev.data)
                if ev.delivered is not None and not ev.delivered.done():
                    ev.delivered.set_result(None)
                if ev.kind=='finished':
                    break
            await asyncio.wait_for(done.wait(),90)
            await asyncio.sleep(.05)
            if route_audio is not None:
                await asyncio.wait_for(asyncio.shield(tasks[1]), 10)
                if len(result['controls']) != 3:
                    raise RuntimeError('Control overlap probe incomplete')
            result['capacity'] = capacity.snapshot()
            result['status']='cancelled-as-planned' if cancel else 'completed'
        except Exception as exc:
            result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        finally:
            p.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(p.task,*tasks,return_exceptions=True)
            await ws.send_json({'event':'complete','data':{'status':result['status']}})
            await ws.close()
        return ws
    app=web.Application()
    app.router.add_get('/', lambda _: web.Response(text=HTML,content_type='text/html'))
    app.router.add_static('/static/',ROOT/'src/static')
    app.router.add_get('/ws',ws_handler)
    runner=web.AppRunner(app)
    await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0)
    await site.start()
    port=site._server.sockets[0].getsockname()[1]
    page=await browser.new_page()
    page.on('pageerror',lambda err:result['errors'].append(str(err)))
    try:
        await page.goto(f'http://127.0.0.1:{port}/')
        await page.click('#start')
        await page.wait_for_function('window.done',timeout=150000)
        result['final_player']=await page.evaluate('window.player.speech && ({received:player.speech.received, played:player.speech.played, underruns:player.speech.underruns, underrun_ms:player.speech.underrunMs})')
        if cancel:
            assert result['final_player'] is None, 'cancelled playback resurrected'
        assert not result['errors'], result['errors']
    finally:
        await page.close()
        await runner.cleanup()
    return result


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',required=True)
    parser.add_argument('--raw-only',action='store_true')
    parser.add_argument('--browser-only',action='store_true')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--chromium',type=Path)
    parser.add_argument('--route-audio',type=Path)
    parser.add_argument('--cases',nargs='+',choices=['short_legacy','short_candidate','long_legacy',
                                                  'long_candidate','cancel_candidate'])
    args=parser.parse_args()
    if args.raw_only and args.browser_only:
        parser.error('--raw-only and --browser-only are mutually exclusive')
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.browser_only and (not args.cases or 'long_candidate' in args.cases) and not args.route_audio:
        parser.error('--browser-only long_candidate requires --route-audio from a self-authored fixture')
    args.output.mkdir(parents=True,exist_ok=False)
    module.configure_asr({'backend':'sensevoice','provider':'cpu','num_threads':2})
    receipt={'profile':args.profile,'scope':__doc__,'human_audio':False,'physical_device':False,
             'serial_cases':True,'browser':[],'raw':[], 'exclusions':[]}
    def save():
        (args.output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    try:
        if not args.browser_only:
            receipt['raw']=await raw_probe(args.output,args.repeats)
            save()
        if not args.raw_only:
            from playwright.async_api import async_playwright
            async with async_playwright() as pw:
                options={'headless':True,'args':['--no-sandbox','--autoplay-policy=no-user-gesture-required']}
                if args.chromium:
                    options['executable_path']=str(args.chromium)
                browser=await pw.chromium.launch(**options)
                try:
                    route_audio=args.route_audio or (args.output/'raw-0-0.wav')
                    for name,text,startup,prefetch,control,cancel in [
                        ('short_legacy',TEXTS[1],80,0,False,False),
                        ('short_candidate',TEXTS[1],350,2000,False,False),
                        ('long_legacy',LONG,80,0,False,False),
                        ('long_candidate',LONG,350,2000,True,False),
                        ('cancel_candidate',LONG,350,2000,False,True)]:
                        if args.cases and name not in args.cases:
                            continue
                        row=await browser_case(browser,name,text,startup,prefetch,
                            route_audio if control else None,cancel)
                        receipt['browser'].append(row)
                        save()
                        print(json.dumps({'browser':name,'status':row['status'],
                            'final':row.get('final_player'),'errors':row['errors']},ensure_ascii=False),flush=True)
                finally:
                    await browser.close()
    finally:
        save()
    if any(r['status']=='error' for r in receipt['raw']+receipt['browser']):
        raise RuntimeError('One or more diagnostics failed; see the retained receipt')


if __name__=='__main__':
    asyncio.run(main())
