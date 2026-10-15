"""Build a HumDial manifest for the f2 audio-clock harness (copied runner in this directory).

usage: python hd_build.py train [--per-cell 200]
       python hd_build.py dev            (needs dev_context.json from hd_dev_context.py)

train: each TextGrid is sequential: user question, model reply, user target utterance
(interruption / backchannel / third-party), [model reply]. The target is injected while the
model reply (text from the TextGrid) is playing, 0.3 s after its first sentence.
Pause Handling: the question itself (with a natural 1.5-3 s pause) in listening mode.
Expected route per scenario (HumDial definitions): interruption scenarios -> yield_ready,
Silence or Termination -> stop_only, backchannel / third-party -> keep, pause -> yield_ready.
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
from silero_vad import get_speech_timestamps, load_silero_vad

HERE = Path(__file__).resolve().parent
HD = Path('/root/autodl-tmp/HumDial_train/HD-Track2')
F2_HEADER = HERE.parent / 'f2_vad' / 'manifest.json'
SR = 16000
HOLES = [0.3, 0.4, 0.5]
torch.set_num_threads(4)
MODEL = load_silero_vad()

EXPECTED = {
    'Follow-up Questions': 'yield_ready', 'Negation or Dissatisfaction': 'yield_ready',
    'Repetition Requests': 'yield_ready', 'Topic Switching': 'yield_ready',
    'Silence or Termination': 'stop_only', 'User Real-time Backchannels': 'keep',
    'Third-party Speech': 'keep', 'Third-party Speech_after': 'keep',
    'Speech Directed at Others': 'keep', 'Pause Handling': 'yield_ready',
    'Third-party Speech_before': 'yield_ready',
}


def read(path, seg=None):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype('float32')
    if seg is not None:
        x = x[max(0, int(seg[0] * SR)):min(len(x), int(seg[1] * SR))]
    return x


def speech(x, min_silence):
    ts = get_speech_timestamps(torch.from_numpy(x), MODEL, threshold=0.5, sampling_rate=SR,
                               min_silence_duration_ms=min_silence, speech_pad_ms=0)
    return [(t['start'] / SR, t['end'] / SR) for t in ts]


def holes(uid, x, voiced):
    fine = speech(x, 100)
    out = {}
    vs, ve = voiced
    for h in HOLES:
        rng = np.random.default_rng(int(hashlib.sha256(f'{uid}|{h}'.encode()).hexdigest()[:8], 16))
        lo, hi = vs + 0.25, ve - 0.25 - h
        pos = None
        if hi > lo:
            for _ in range(50):
                s = float(rng.uniform(lo, hi))
                cover = sum(max(0.0, min(s + h, b) - max(s, a)) for a, b in fine) / h
                if cover >= 0.6:
                    pos = round(s, 4)
                    break
        out[str(h)] = pos
    return out


def item(uid, set_, src, expected, *, category, lang, text, ctx, seg, playing, source):
    x = read(src, seg)
    spans = speech(x, 300)
    if not spans:
        return None
    voiced = (round(spans[0][0], 4), round(spans[-1][1], 4))
    a, b = int(voiced[0] * SR), int(voiced[1] * SR)
    rms = float(np.sqrt(np.mean(x[a:b] ** 2))) if b > a else 0.0
    if rms < 1e-6:
        return None
    gain = round(min(10 ** (-24 / 20) / rms, 0.95 / max(float(np.max(np.abs(x))), 1e-6)), 6)
    return {'uid': uid, 'set': set_, 'category': category, 'expected': expected, 'lang': lang,
            'text': text, 'root': uid, 'context': ctx, 'gain': gain,
            'parts': [{'path': str(src), 'bytes': Path(src).stat().st_size, 'seg': seg}],
            'duration_s': round(len(x) / SR, 4), 'voiced': voiced, 'playing': playing,
            'source': source, 'holes': holes(uid, x, voiced) if playing else {}}


def textgrid(path):
    """Non-empty intervals of the first tier: [(xmin, xmax, text)]."""
    s = Path(path).read_text(encoding='utf-8', errors='replace')
    tier = s.split('item [2]:')[0]
    out = []
    for m in re.finditer(r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "(.*)"', tier):
        if m.group(3).strip():
            out.append((float(m.group(1)), float(m.group(2)), m.group(3).strip()))
    return out


def clean_reply(text, lang):
    text = re.sub(r'(\.\.\.|…)+\s*$', '', text).strip()
    if text and text[-1] not in '。？！.?!':
        text += '。' if lang == 'zh' else '.'
    return text


def build_train(per_cell):
    items, skipped = [], collections.Counter()
    for lang in ('zh', 'en'):
        root = HD / 'HD-Track2-train' / f'HD-Track2-train-{lang}'
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            cat = folder.name
            grids = sorted(folder.glob('*.TextGrid'))
            rng = np.random.default_rng(int(hashlib.sha256(f'train|{lang}|{cat}'.encode()).hexdigest()[:8], 16))
            order = [grids[i] for i in rng.permutation(len(grids))]
            n = 0
            for g in order:
                if n >= per_cell:
                    break
                wav = g.with_suffix('.wav')
                segs = textgrid(g)
                if not wav.exists() or len(segs) < (2 if cat == 'Pause Handling' else 3):
                    skipped[(lang, cat, 'segments')] += 1
                    continue
                uid = f'hdtr:{lang}:{cat}:{g.stem}'
                if cat == 'Pause Handling':
                    q = segs[0]
                    r = item(uid, 'HDP', wav, 'yield_ready', category=cat, lang=lang, text=q[2],
                             ctx={'previous_user': '', 'assistant_text': ''},
                             seg=[max(0.0, q[0] - 0.1), q[1] + 0.1], playing=False, source='humdial_train')
                else:
                    q, reply, target = segs[0], segs[1], segs[2]
                    ctx = {'previous_user': q[2], 'assistant_text': clean_reply(reply[2], lang)}
                    r = item(uid, 'HD', wav, EXPECTED[cat], category=cat, lang=lang, text=target[2], ctx=ctx,
                             seg=[max(0.0, target[0] - 0.1), target[1] + 0.1], playing=True, source='humdial_train')
                if r is None:
                    skipped[(lang, cat, 'no_speech')] += 1
                    continue
                items.append(r)
                n += 1
    return items, skipped


def build_dev():
    ctxs = json.loads((HERE / 'dev_context.json').read_text(encoding='utf-8'))
    items, skipped = [], collections.Counter()
    for key, c in sorted(ctxs.items()):
        lang, cat, stem = c['lang'], c['category'], c['stem']
        wav = Path(c['wav'])
        uid = f'hddev:{lang}:{cat}:{stem}'
        tseg = c['target_seg']
        playing = c['playing']
        r = item(uid, 'HD' if playing else 'HDP', wav, EXPECTED[cat], category=cat, lang=lang, text=c['target_text'],
                 ctx={'previous_user': c['previous_user'], 'assistant_text': c['assistant_text']},
                 seg=[max(0.0, tseg[0] - 0.1), tseg[1] + 0.1], playing=playing, source='humdial_dev')
        if r is None:
            skipped[(lang, cat)] += 1
            continue
        if 'second_start_rel' in c:
            # Utterance-relative start of the user's own question (audio starts at seg - 0.1 s).
            r['second_start_rel'] = round(c['second_start_rel'] + min(0.1, tseg[0]), 4)
        items.append(r)
    return items, skipped


def main():
    split = sys.argv[1]
    per_cell = int(sys.argv[sys.argv.index('--per-cell') + 1]) if '--per-cell' in sys.argv else 200
    items, skipped = build_train(per_cell) if split == 'train' else build_dev()
    header = json.loads(F2_HEADER.read_text())['header']
    header = {**header, 'version': f'hd-{split}-v1', 'created': '2026-10-15', 'note': __doc__.strip().splitlines()[0]}
    out = HERE / split
    out.mkdir(exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps({'header': header, 'items': items}, ensure_ascii=False, indent=0), encoding='utf-8')
    counts = collections.Counter((r['set'], r['lang'], r['category'], r['expected']) for r in items)
    for k, v in sorted(counts.items()):
        print(k, v)
    print('items', len(items), 'skipped', dict(skipped))


if __name__ == '__main__':
    main()
