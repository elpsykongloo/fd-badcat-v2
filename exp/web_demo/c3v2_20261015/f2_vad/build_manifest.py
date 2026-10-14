# Migrated to c3v2; no artifact integrity hashing.
"""Build and freeze the f2_vad experiment manifest (labels + synthesis parameters, no audio).

usage: python build_manifest.py OUT_DIR

Sets (all audio is existing dev-set / self-authored TTS audio, read in place):
  H  playing, one browser-AEC hole per utterance (none / 0.3 / 0.4 / 0.5 s), labelled route
     sources: P1 dev bases (p1_input_diagnosis_20261008/base_manifest.json, no greeting)
              route_addressee_20261009 TEST split (HumDial dev + self TTS), minus P1 overlap
  P  playing, HumDial dev "Pause Handling" (natural 1.5-3 s pause), unused by P1, expect yield_ready
  Q  playing, two self-TTS acts joined by a REAL pause (0.3 / 0.45 / 0.6 s): side-effect set
  L  listening (no playback), hole 0.3: the end window must change nothing

Holes and voiced spans come from an offline Silero labelling pass (get_speech_timestamps);
the runner's VAD is the production engine path. The mic is synthesised at run time from
these parameters, so no new audio is written. Freezes manifest sha256, item counts and
every source file's size + sha256.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from silero_vad import load_silero_vad, get_speech_timestamps
MAIN = Path('/root/autodl-tmp/fd-badcat')
P1 = MAIN / 'exp/web_demo/p1_input_diagnosis_20261008'
RA = MAIN / 'exp/web_demo/route_addressee_20261009'
HD = Path('/root/autodl-tmp/HumDial_train/HD-Track2/HD-Track2-dev')
ECHO = Path('/root/autodl-tmp/fd-badcat-c3v2/exp/web_demo/p0_pro6000_20261007/fixtures/long.wav')
HOLES = [0.3, 0.4, 0.5]
PAIR_GAPS = [0.3, 0.45, 0.6]
SR = 16000
torch.set_num_threads(4)
MODEL = load_silero_vad()
GENERIC = {'zh': '海水淡化主要有两种方法。第一种是反渗透，用高压把海水压过一层很细的膜。第二种是蒸馏，把海水加热成水蒸气，再冷却成淡水。反渗透更省电，所以现在用得更多。蒸馏更适合有大量废热的工厂。不过两种方法都要妥善处理剩下的浓盐水。浓盐水直接排回海里，会影响附近的海洋生物。所以很多工厂会先稀释，再分散排放。', 'en': 'There are two main ways to make seawater drinkable. The first is reverse osmosis, which pushes water through a very fine membrane. The second is distillation, which boils the water and condenses the steam. Reverse osmosis uses less energy, so it is more common today. Distillation still makes sense where waste heat is cheap. Both methods leave a salty brine that must be handled carefully. Dumping it straight back into the sea can harm marine life nearby. Many plants dilute it first and spread it out.'}
SOURCES = {}

def source(path):
    path = str(path)
    if path not in SOURCES:
        data = Path(path).read_bytes()
        SOURCES[path] = {'bytes': len(data)}
    return {'path': path, **SOURCES[path]}

def read(path, seg=None):
    (x, sr) = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype('float32')
    if seg is not None:
        x = x[max(0, int(seg[0] * SR)):min(len(x), int(seg[1] * SR))]
    return x

def speech(x, min_silence):
    ts = get_speech_timestamps(torch.from_numpy(x), MODEL, threshold=0.5, sampling_rate=SR, min_silence_duration_ms=min_silence, speech_pad_ms=0)
    return [(t['start'] / SR, t['end'] / SR) for t in ts]

def gain_for(x, voiced):
    (a, b) = (int(voiced[0] * SR), int(voiced[1] * SR))
    rms = float(np.sqrt(np.mean(x[a:b] ** 2))) if b > a else 0.0
    if rms < 1e-06:
        return None
    g = 10 ** (-24 / 20) / rms
    return round(min(g, 0.95 / max(float(np.max(np.abs(x))), 1e-06)), 6)

def holes(uid, x, voiced):
    fine = speech(x, 100)
    out = {}
    (vs, ve) = voiced
    for h in HOLES:
        rng = np.random.default_rng(int(hashlib.sha256(f'{uid}|{h}'.encode()).hexdigest()[:8], 16))
        (lo, hi) = (vs + 0.25, ve - 0.25 - h)
        pos = None
        if hi > lo:
            for _ in range(50):
                s = float(rng.uniform(lo, hi))
                cover = sum((max(0.0, min(s + h, b) - max(s, a)) for (a, b) in fine)) / h
                if cover >= 0.6:
                    pos = round(s, 4)
                    break
        out[str(h)] = pos
    return out

def item(uid, set_, src, expected, *, category, lang, text, root, ctx, seg=None, extra=None, want_holes=True):
    x = read(src, seg)
    spans = speech(x, 300)
    if not spans:
        return None
    voiced = (round(spans[0][0], 4), round(spans[-1][1], 4))
    g = gain_for(x, voiced)
    if g is None:
        return None
    rec = {'uid': uid, 'set': set_, 'category': category, 'expected': expected, 'lang': lang, 'text': text, 'root': root, 'context': ctx, 'gain': g, 'parts': [{**source(src), 'seg': seg}], 'duration_s': round(len(x) / SR, 4), 'voiced': voiced, 'playing': set_ != 'L', **(extra or {})}
    rec['holes'] = holes(uid, x, voiced) if want_holes else {}
    return rec

def p1_context(b):
    lang = b['language']
    previous = b.get('previous_text') or ''
    if b['category'] == 'answer':
        rid = b['root_id']
        q = '你想听童话还是科幻故事？' if lang == 'zh' else 'Would you prefer a fairy tale or science fiction?'
        if 'answer_2' in rid or 'answer_3' in rid:
            q = '我现在可以继续吗？'
        if 'answer_4' in rid or 'answer_5' in rid:
            q = 'May I continue now?'
        return {'previous_user': previous, 'assistant_text': q + (' ' if lang == 'en' else '') + GENERIC[lang]}
    return {'previous_user': previous, 'assistant_text': GENERIC[lang]}

def ra_context(it):
    hist = it.get('history') or []
    previous = hist[-1][0] if hist else ''
    answer = re.sub('[*#`>]+', '', hist[-1][1]) if hist else ''
    answer = re.sub('\\s*\\n+\\s*', ' ' if it['lang'] == 'en' else '', answer).strip()
    text = answer if len(answer) >= 60 else (answer + (' ' if it['lang'] == 'en' else '') + GENERIC[it['lang']]).strip()
    return {'previous_user': previous, 'assistant_text': text}

def main():
    out = Path(sys.argv[1])
    items = []
    bases = json.loads((P1 / 'base_manifest.json').read_text())
    p1_followups = set()
    for b in bases:
        if b['category'] == 'greeting':
            continue
        if b.get('source_file') and b['category'] == 'followup':
            stem = Path(b['source_file']).stem
            p1_followups.add(f"hd_{b['language']}_followup_{stem}")
        r = item('p1:' + b['id'], 'H', b['path'], b['expected'], category=b['category'], lang=b['language'], text=b['text'], root='p1:' + b['root_id'], ctx=p1_context(b), extra={'source': 'p1_' + b['source']})
        if r:
            items.append(r)
    ra = [x for x in json.loads((RA / 'items.json').read_text()) if x['split'] == 'test']
    ra_dropped = 0
    for it in ra:
        if it['id'] in p1_followups:
            ra_dropped += 1
            continue
        r = item('ra:' + it['id'], 'H', it['path'], it['expected'], category='ra_' + it['kind'], lang=it['lang'], text=it['text'], root='ra:' + it['id'], ctx=ra_context(it), extra={'source': 'ra_test_' + it['source']})
        if r:
            items.append(r)
    used = {json.dumps(Path(p['source_file']).parts[-3:]) for p in json.loads((P1 / 'pause_manifest.json').read_text())}
    for lang in ('zh', 'en'):
        folder = HD / f'HD-Track2-dev-{lang}' / 'Pause Handling'
        for meta in sorted(folder.glob('*_sentence.json')):
            wav = meta.with_name(meta.name.replace('_sentence.json', '.wav'))
            if not wav.exists() or json.dumps(wav.parts[-3:]) in used:
                continue
            seg = json.loads(meta.read_text())['speech_segments'][-1]
            uid = f'pause:{lang}:{wav.stem}'
            r = item(uid, 'P', str(wav), 'yield_ready', category='natural_pause', lang=lang, text=seg['text'], root=uid, ctx={'previous_user': '', 'assistant_text': GENERIC[lang]}, seg=[max(0.0, seg['xmin'] - 0.1), seg['xmax'] + 0.1], want_holes=False, extra={'source': 'humdial_dev_pause'})
            if r:
                items.append(r)
    tts = collections.defaultdict(list)
    for b in bases:
        if b['source'] == 'tts_expansion':
            tts[b['category'], b['language'], b['tts_seed']].append(b)
    types = {'bc_q': ('backchannel', ('short_question', 'short_request'), {'cancel': True, 'respond': True}), 'bc_bc': ('backchannel', ('backchannel',), {'cancel': False, 'respond': False}), 'stop_q': ('stop', ('short_question', 'short_request'), {'cancel': True, 'respond': True}), 'stop_bc': ('stop', ('backchannel',), {'cancel': True, 'respond': False}), 'bc_tp': ('backchannel', ('third_party',), {'cancel': False, 'respond': False})}
    for (name, (first, seconds, expect)) in types.items():
        for lang in ('zh', 'en'):
            combos = []
            for seed in (42, 101, 202):
                for a in tts[first, lang, seed]:
                    for cat in seconds:
                        for b in tts[cat, lang, seed]:
                            if a['id'] != b['id'] and a['text'] != b['text']:
                                combos.append((a, b))
            rng = np.random.default_rng(int(hashlib.sha256(f'{name}|{lang}'.encode()).hexdigest()[:8], 16))
            pick = [combos[i] for i in sorted(rng.permutation(len(combos))[:8])]
            for (a, b) in pick:
                (xa, xb) = (read(a['path']), read(b['path']))
                (va, vb) = (speech(xa, 300), speech(xb, 300))
                if not va or not vb:
                    continue
                for gap in PAIR_GAPS:
                    uid = f"pair:{name}:{a['id']}+{b['id']}:{gap}"
                    items.append({'uid': uid, 'set': 'Q', 'category': name, 'expected': expect, 'lang': lang, 'text': a['text'] + ' | ' + b['text'], 'root': f"pair:{a['id']}+{b['id']}", 'context': {'previous_user': '', 'assistant_text': GENERIC[lang]}, 'parts': [{**source(a['path']), 'seg': None, 'cut': [va[0][0], va[-1][1]]}, {**source(b['path']), 'seg': None, 'cut': [vb[0][0], vb[-1][1]]}], 'gap_s': gap, 'gain': None, 'holes': {}, 'playing': True, 'source': 'p1_tts_pairs'})
    eligible = [r for r in items if r['set'] == 'H' and r['holes'].get('0.3') is not None]
    rng = np.random.default_rng(20261010)
    for r in [eligible[i] for i in sorted(rng.permutation(len(eligible))[:40])]:
        items.append({**r, 'uid': 'listen:' + r['uid'], 'set': 'L', 'playing': False, 'holes': {'0.3': r['holes']['0.3']}})
    header = {'version': 'f2-vad-prereg-v1', 'created': '2026-10-10', 'sample_rate': SR, 'holes_s': HOLES, 'pair_gaps_s': PAIR_GAPS, 'synthesis': {'hole_depth_db': -60, 'hole_ramp_ms': 5, 'voiced_rms_dbfs': -24, 'floor': {'kind': 'pink', 'dbfs': -63, 'seed': 'sha256(unit)'}, 'echo_residual': {**source(ECHO), 'dbfs': None, 'delay_ms': 181, 'reference_rms_dbfs': -20}, 'pair_fade_ms': 10, 'aec_basis': '2026-10-09 web-demo-8f06b71c: holes 0.3-0.5 s at about -60 dB, playback floor -62..-65 dBFS'}, 'labelling': 'silero get_speech_timestamps threshold .5, min_silence 300 ms (voiced span) and 100 ms (hole speech cover >= 0.6), speech_pad 0', 'ra_dropped_p1_overlap': ra_dropped}
    manifest = {'header': header, 'items': items}
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=0)
    (out / 'manifest.json').write_text(text, encoding='utf-8')
    counts = collections.Counter(((r['set'], r['category'], json.dumps(r['expected'], sort_keys=True)) for r in items))
    hole_counts = collections.Counter(((r['set'], h) for r in items for (h, v) in r['holes'].items() if v is not None))
    freeze = {'manifest_bytes': len(text.encode()), 'items': len(items), 'by_set': dict(collections.Counter((r['set'] for r in items))), 'by_set_category_expected': {' / '.join(k): v for (k, v) in sorted(counts.items())}, 'hole_positions': {f'{s}:{h}': v for ((s, h), v) in sorted(hole_counts.items())}, 'sources': len(SOURCES), 'source_bytes': sum((v['bytes'] for v in SOURCES.values())), 'frozen_before_inference': True}
    (out / 'manifest_freeze.json').write_text(json.dumps(freeze, ensure_ascii=False, indent=1))
    print(json.dumps(freeze, ensure_ascii=False, indent=1))
main()
