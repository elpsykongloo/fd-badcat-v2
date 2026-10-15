"""Inventory every web-demo session on this box: timing, RTT, capture, turns. Output jsonl for human/automation triage."""
import glob, json, os, statistics, sys
from pathlib import Path

OUT = Path(sys.argv[1])
rows = []
for ev in glob.glob('/root/autodl-tmp/*/exp/web-demo-*/*/events.jsonl'):
    d = Path(ev).parent
    r = {'tree': ev.split('/')[3], 'session': d.parent.name, 'sub': d.name, 'dir': str(d)}
    rtt, first, last, n_vad, n_turn_done, settings, utc, pings, client_ms0 = [], None, 0.0, 0, 0, None, None, 0, None
    asr = []
    try:
        with open(ev, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                k = e.get('event'); dt = e.get('data') or {}
                last = max(last, float(e.get('server_ms') or 0))
                if k == 'session_start':
                    utc = dt.get('utc')
                elif k == 'client_rtt' and isinstance(dt.get('rtt_ms'), (int, float)):
                    rtt.append(dt['rtt_ms'])
                elif k == 'vad_start':
                    n_vad += 1
                elif k == 'input_settings':
                    settings = dt
                elif k == 'asr_done':
                    t = dt.get('text') or dt.get('transcript')
                    if t:
                        asr.append(str(t)[:60])
                elif k == 'turn_finished':
                    n_turn_done += 1
    except OSError:
        continue
    cap = d / 'diagnostics/capture/browser_mic.wav'
    r.update(utc=utc, dur_s=round(last / 1000, 1), rtt_med=round(statistics.median(rtt), 1) if rtt else None,
             n_rtt=len(rtt), n_vad=n_vad, turns=n_turn_done, settings=settings,
             mic=cap.exists(), mic_mb=round(cap.stat().st_size / 1e6, 1) if cap.exists() else 0,
             turn_wavs=len(glob.glob(str(d / 'stream_turn*_input.wav'))), asr=asr[:4])
    rows.append(r)
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', encoding='utf-8') as fh:
    for r in sorted(rows, key=lambda x: x['utc'] or ''):
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')
print(len(rows))
