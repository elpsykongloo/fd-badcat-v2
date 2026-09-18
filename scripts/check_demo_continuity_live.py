#!/usr/bin/env python3
"""Live Actor/Omni barge-in and recovery via real Chromium and synthetic mic.

Self-authored inputs only. Real WebAudio/AudioWorklet and model calls, no human
recordings or physical audio device. All model decisions remain audio grounded.
"""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import module

INPUTS=[
    '请给我讲一个长一点的森林童话故事，至少讲三十秒。',
    '请停止说话，现在先不要继续讲了。',
    '现在请只说一句，你好，很高兴认识你。',
]

MIC='''(() => {
 window.__tracks=[];
 navigator.mediaDevices.getUserMedia=async()=>{
  const context=new AudioContext({sampleRate:16000}); await context.resume();
  const destination=context.createMediaStreamDestination();
  window.__micContext=context;window.__micDestination=destination;
  window.__tracks.push(...destination.stream.getTracks());
  return destination.stream;
 };
 window.__inject=async(encoded)=>{
  const data=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
  const buffer=await window.__micContext.decodeAudioData(data.buffer);
  const source=window.__micContext.createBufferSource(); source.buffer=buffer;
  source.connect(window.__micDestination);
  await window.__micContext.resume();
  source.start(window.__micContext.currentTime+.1);
  return buffer.duration;
 };
})();'''


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:18000')
    parser.add_argument('--chromium',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    receipt={'scope':__doc__,'inputs':INPUTS,'events':[],'client':[],'errors':[], 'passed':False}
    audios=[]
    for index,text in enumerate(INPUTS):
        chunks=[c async for c in module.tts_omni_stream(text,voice_control={'speaker':'chelsie','seed':42})]
        path=args.output/f'input-{index}.wav'
        sf.write(path,np.frombuffer(b''.join(c.pcm for c in chunks),dtype='<i2'),chunks[0].sample_rate,subtype='PCM_16')
        audios.append(base64.b64encode(path.read_bytes()).decode())
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,executable_path=str(args.chromium),args=['--no-sandbox','--autoplay-policy=no-user-gesture-required','--no-proxy-server'])
        context=await browser.new_context(permissions=['microphone'])
        await context.add_init_script(MIC)
        page=await context.new_page()
        page.on('pageerror',lambda error:receipt['errors'].append(str(error)))
        def connected(ws):
            def received(payload):
                if isinstance(payload,str):
                    receipt['events'].append(json.loads(payload))
            def sent(payload):
                if isinstance(payload,str):
                    value=json.loads(payload)
                    if value.get('event') in ('demo_telemetry','playback_stopped','playback_progress'):
                        receipt['client'].append(value)
            ws.on('framereceived',received)
            ws.on('framesent',sent)
        page.on('websocket',connected)
        async def until(predicate, seconds=90):
            async def poll():
                while not predicate():
                    if receipt['errors']:
                        raise RuntimeError(receipt['errors'])
                    await asyncio.sleep(.05)
            await asyncio.wait_for(poll(),seconds)
        def events(kind):
            return [r['data'] for r in receipt['events'] if r['event']==kind]
        try:
            await page.goto(args.url+'/demo/')
            await page.click('#start')
            await until(lambda:bool(events('demo_ready')),30)
            receipt['session_id']=events('demo_ready')[0]['session_id']
            await page.evaluate('s=>window.__inject(s)',audios[0])
            await until(lambda:bool(events('speech_playback_started')))
            first_sid=events('speech_playback_started')[0]['utterance_id']
            await asyncio.sleep(1)
            await page.evaluate('s=>window.__inject(s)',audios[1])
            await until(lambda:any(r['utterance_id']==first_sid for r in events('speech_cancelled')),30)
            await until(lambda:any(r.get('reason')=='stop_only' for r in events('input_waiting')),10)
            before=len(events('speech_start'))
            await asyncio.sleep(3)
            assert len(events('speech_start'))==before,'STOP_ONLY generated an unsolicited reply'
            assert any(r['event']=='playback_stopped' and r['data']['utterance_id']==first_sid for r in receipt['client'])
            await page.evaluate('s=>window.__inject(s)',audios[2])
            await until(lambda:any(r['utterance_id']!=first_sid for r in events('speech_played')))
            completed=[r for r in events('speech_played') if r['utterance_id']!=first_sid]
            assert all(r.get('underruns',0)==0 for r in completed),completed
            assert all(r['data'].get('underruns',0)==0 for r in receipt['client'] if r['event']=='demo_telemetry' and r['data'].get('kind') in ('cancel','playback_end'))
            await page.click('#stop')
            await page.wait_for_function("window.__tracks.every(t=>t.readyState==='ended')")
            await page.evaluate('window.__micContext.close()')
            receipt['passed']=True
            receipt['completed']=completed
        except Exception as exc:
            receipt['failure']=f'{type(exc).__name__}: {exc}'
            receipt['ui']=await page.locator('body').inner_text()
            raise
        finally:
            (args.output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
            await context.close()
            await browser.close()
    print(json.dumps({'passed':receipt['passed'],'session_id':receipt['session_id'],'completed':receipt['completed']},ensure_ascii=False))


if __name__=='__main__':
    asyncio.run(main())
