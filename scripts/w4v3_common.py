#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w4v3_common.py — shared loader/lexicon/stats layer for the W4-V3 Phase-1 batch
(HumDial HD-Track2 census + text probe + Omni readout; docs/w4v3_design.md).

Data layout, parser quirks and edge cases are FROZEN from the byte-level format
probe report (2026-07-15, HD-Track2.zip sha256 6555F6F2...):
  * train samples = <id>.wav + <id>.TextGrid (9,988); dev samples are indexed by
    *_sentence.json (1,800; Third-party dirs have NO TextGrid — never enumerate
    dev by TextGrid).
  * TextGrid: Praat long form, tiers 文本/事件/情绪; only 文本 carries text; nine
    train files contain unescaped double quotes inside text = "..." lines — the
    line-anchored regex below keeps them (validated in the probe report §13).
    CRLF endings are neutralized by universal-newline text reads.
  * [break] marks an in-utterance pause (Pause Handling); 64 train files of that
    scene lack the literal marker — scene identity comes from the directory.
  * Annotation timelines can overrun the real WAV tail (§11.8) — durations must
    come from WAV headers; clamp interval ends when computing silences.

COMPLIANCE RED LINE (Data_Protocol.md; see w4v3_design.md §3): HumDial transcript
text must never be written into any artifact file (repo history is published to
a public branch). Everything under exp/w4v3/ carries statistics, paths, times,
hashes and labels only. `assert_no_text()` is the enforcement hook — call it on
every JSON-serializable payload before writing.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import wave
from pathlib import Path

SR = 16000
BREAK = "[break]"

# ---------------------------------------------------------------------------
# roots & enumeration
# ---------------------------------------------------------------------------


def resolve_root(root):
    """Accept the dir that contains Data_Protocol.md, or any ancestor holding
    the HD-Track2 wrapper (the zip ships one wrapper level)."""
    root = Path(root)
    for cand in (root, root / "HD-Track2"):
        if (cand / "Data_Protocol.md").exists() or (cand / "HD-Track2-train").exists():
            return cand
    raise SystemExit(f"HD-Track2 root not found under {root} "
                     "(expect Data_Protocol.md or HD-Track2-train/)")


# role templates per train scene (probe report §12.1; derived, not annotated)
TRAIN_ROLES = {
    "Follow-up Questions":         ["user", "assistant", "user", "assistant"],
    "Negation or Dissatisfaction": ["user", "assistant", "user", "assistant"],
    "Repetition Requests":         ["user", "assistant", "user", "assistant"],
    "Silence or Termination":      ["user", "assistant", "user", "assistant"],
    "Topic Switching":             ["user", "assistant", "user", "assistant"],
    "User Real-time Backchannels": ["user", "assistant", "user"],
    "Pause Handling":              ["user", "assistant"],
    "Third-party Speech":          ["user", "assistant", "third"],
}
INTERRUPT_SCENES = ("Follow-up Questions", "Negation or Dissatisfaction",
                    "Repetition Requests", "Silence or Termination",
                    "Topic Switching")

_INTERVAL_RE = re.compile(
    r"intervals \[\d+\]:\s*"
    r"xmin\s*=\s*([^\s]+)\s*"
    r"xmax\s*=\s*([^\s]+)\s*"
    r'text\s*=\s*"(.*)"[ \t]*$',
    re.MULTILINE,
)


def read_textgrid_segments(path):
    """Non-empty 文本-tier intervals -> [{xmin,xmax,text}]; None on parse failure.
    Relies on universal-newline text reads (CRLF train files decode to \\n)."""
    try:
        content = Path(path).read_text(encoding="utf-8")
        t0 = content.index('name = "文本"')
        t1 = content.find("item [2]:", t0)
        tier = content[t0:t1] if t1 != -1 else content[t0:]
        segs = [{"xmin": float(a), "xmax": float(b), "text": t}
                for a, b, t in _INTERVAL_RE.findall(tier) if t]
        return segs or None
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def wav_duration(path):
    try:
        with wave.open(str(path), "rb") as r:
            return r.getnframes() / r.getframerate()
    except (OSError, wave.Error, EOFError):
        return None


def sample_key(split, lang, scene, sample_id):
    return f"{split}/{lang}/{scene}/{sample_id}"


def iter_train_samples(root, langs=("zh", "en")):
    for lang in langs:
        base = root / "HD-Track2-train" / f"HD-Track2-train-{lang}"
        if not base.is_dir():
            continue
        for wav in sorted(base.rglob("*.wav")):
            grid = wav.with_suffix(".TextGrid")
            scene = wav.parent.name
            yield {"split": "train", "language": lang, "scene": scene,
                   "sample_id": wav.stem,
                   "key": sample_key("train", lang, scene, wav.stem),
                   "wav": wav, "textgrid": grid if grid.exists() else None,
                   "rel_wav": str(wav.relative_to(root)),
                   "rel_grid": str(grid.relative_to(root)) if grid.exists() else None}


def iter_dev_samples(root, langs=("zh", "en")):
    """Dev is JSON-indexed (Third-party dirs have no TextGrid; clean_* wavs are
    paired variants without their own annotation)."""
    for lang in langs:
        base = root / "HD-Track2-dev" / f"HD-Track2-dev-{lang}"
        if not base.is_dir():
            continue
        for jp in sorted(base.rglob("*_sentence.json")):
            sid = jp.name[:-len("_sentence.json")]
            wav = jp.with_name(f"{sid}.wav")
            scene = jp.parent.name
            try:
                ann = json.loads(jp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                ann = None
            yield {"split": "dev", "language": lang, "scene": scene,
                   "sample_id": sid, "key": sample_key("dev", lang, scene, sid),
                   "wav": wav if wav.exists() else None, "json": jp,
                   "annotation": ann,
                   "rel_wav": str(wav.relative_to(root)) if wav.exists() else None}


def derive_roles(scene, n):
    tpl = TRAIN_ROLES.get(scene, [])
    return [tpl[i] if i < len(tpl) else "unknown" for i in range(n)]


# ---------------------------------------------------------------------------
# text handling & lexicons (zh/en). Terminal punctuation is annotator-injected
# (leakage-suspect) => stripped before S/T1/T2 featurization; punctuation-only
# features live in the separate P family, reported outside the probe gate.
# ---------------------------------------------------------------------------
_TERM_PUNCT = "。？！?!.,，、；;：:…~— \t　"
_WORD_RE = re.compile(r"[A-Za-z']+$")

FILLER_ZH = {"嗯", "呃", "啊", "那个", "这个", "就是", "然后", "反正", "其实"}
CONN_ZH = {"和", "或", "跟", "与", "及", "但", "而", "并", "或者", "以及", "还是",
           "但是", "因为", "所以", "如果", "而且", "并且", "还有", "然后", "要么"}
PREP_ZH = {"在", "从", "到", "给", "对", "把", "被", "比", "向", "往", "让",
           "帮", "用", "按", "照", "关于", "对于"}
PART_ZH = {"的", "得", "地"}
NUM_CH = set("0123456789零一二两三四五六七八九十百千万亿点")

FILLER_EN = {"um", "uh", "er", "erm", "hmm", "like", "well"}
CONN_EN = {"and", "or", "but", "because", "if", "when", "while", "that",
           "which", "who", "also", "plus", "then", "so", "although", "unless"}
PREP_EN = {"of", "to", "in", "on", "at", "for", "with", "from", "by", "about",
           "into", "over", "under", "between", "before", "after", "during",
           "per", "around", "near", "without", "within"}
DET_EN = {"the", "a", "an", "my", "your", "his", "her", "their", "our", "its",
          "this", "these", "those", "some", "any", "each", "every", "no"}
AUX_EN = {"is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
          "can", "could", "will", "would", "should", "shall", "may", "might",
          "must", "have", "has", "had"}

T1_NAMES = ("ends_filler", "ends_conn", "ends_prep", "ends_det_or_part",
            "ends_aux", "ends_num", "single_token")
P_NAMES = ("p_question", "p_terminal", "p_comma_tail")


def strip_break(text):
    return text.replace(BREAK, " ")


def strip_term_punct(text):
    return text.rstrip(_TERM_PUNCT)


def _last_tokens(text, lang):
    """-> (last_word, last1, last2) on punctuation-stripped text."""
    t = strip_term_punct(text.strip())
    if not t:
        return "", "", ""
    m = _WORD_RE.search(t)
    word = m.group(0).lower() if m else ""
    return word, t[-1:], t[-2:]


def t1_features(text, lang):
    word, c1, c2 = _last_tokens(text, lang)
    if lang == "zh":
        f = {"ends_filler": c1 in FILLER_ZH or c2 in FILLER_ZH,
             "ends_conn": c1 in CONN_ZH or c2 in CONN_ZH,
             "ends_prep": c1 in PREP_ZH or c2 in PREP_ZH,
             "ends_det_or_part": c1 in PART_ZH,
             "ends_aux": False,
             "ends_num": c1 in NUM_CH}
    else:
        f = {"ends_filler": word in FILLER_EN,
             "ends_conn": word in CONN_EN,
             "ends_prep": word in PREP_EN,
             "ends_det_or_part": word in DET_EN,
             "ends_aux": word in AUX_EN,
             "ends_num": bool(c1.isdigit())}
    body = strip_term_punct(strip_break(text).strip())
    f["single_token"] = (len(body) <= 2) if lang == "zh" else (len(body.split()) <= 1)
    return {k: float(v) for k, v in f.items()}


def p_features(text):
    """Annotator punctuation family — leakage-suspect, NEVER in the probe gate."""
    t = text.strip()
    return {"p_question": float(t.endswith(("?", "？"))),
            "p_terminal": float(t.endswith(("?", "？", "。", ".", "!", "！"))),
            "p_comma_tail": float(t.endswith((",", "，", "、")))}


def t2_features(text, lang, n_buckets=256, tail=12):
    """Char n-gram hash counts over the last `tail` chars of the
    punctuation-stripped, [break]-stripped, lowercased text (the 'poor man's
    learned representation' tier of the probe)."""
    t = strip_term_punct(strip_break(text).strip()).lower()[-tail:]
    v = [0.0] * n_buckets
    for n in (2, 3):
        for i in range(max(0, len(t) - n + 1)):
            g = t[i:i + n]
            v[int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % n_buckets] += 1.0
    s = sum(v)                    # L1-normalize: kill the count-sum length
    return [x / s for x in v] if s else v  # channel (length lives in S only)


def text_hash(text):
    norm = re.sub(r"\s+", "", strip_break(text)).lower()
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# small stats (stdlib only — the census must run in the 2GB no-GPU container)
# ---------------------------------------------------------------------------
def quantiles(xs, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    out = {"n": n, "mean": round(sum(xs) / n, 4),
           "min": round(xs[0], 4), "max": round(xs[-1], 4)}
    for p in ps:
        k = (n - 1) * p
        f = math.floor(k)
        c = min(f + 1, n - 1)
        out[f"p{int(p * 100)}"] = round(xs[f] + (xs[c] - xs[f]) * (k - f), 4)
    return out


def assert_no_text(obj, path="$"):
    """Compliance guard: refuse payloads carrying a 'text' field or any string
    long enough to plausibly be a transcript (>80 chars) outside path fields."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "text":
                raise SystemExit(f"compliance: raw 'text' field at {path}.{k}")
            assert_no_text(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_text(v, f"{path}[{i}]")
    elif isinstance(obj, str) and len(obj) > 80 and "/" not in obj:
        raise SystemExit(f"compliance: suspicious long string at {path}")


# ---------------------------------------------------------------------------
# numpy model layer (probe only; import guarded so census stays stdlib-pure)
# ---------------------------------------------------------------------------
def fit_lr(X, y, epochs=400, lr=0.5, l2=1e-3):
    import numpy as np
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Xn = (X - mu) / sd
    w_pos = max(1.0, (y == 0).sum() / max(1, (y == 1).sum()))
    sw = np.where(y == 1, w_pos, 1.0)
    sw = sw / sw.mean()
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-np.clip(Xn @ w + b, -30, 30)))
        g = (p - y) * sw
        w -= lr * (Xn.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b, mu, sd


def predict_lr(model, X):
    import numpy as np
    w, b, mu, sd = model
    return 1 / (1 + np.exp(-np.clip(((X - mu) / sd) @ w + b, -30, 30)))


def auc(y, p):
    import numpy as np
    order = np.argsort(p)
    r = np.empty(len(p))
    r[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not (n1 and n0):
        return None
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _midrank(x):
    import numpy as np
    order = np.argsort(x)
    sx = x[order]
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        r[i:j] = (i + j - 1) / 2.0 + 1
        i = j
    out = np.empty(len(x))
    out[order] = r
    return out


def delong_test(y, p1, p2):
    """DeLong test for two correlated AUCs on the same instances (pooled CV OOF
    scores; nested-model comparison). -> (auc1, auc2, z, p_one_sided_auc1_gt)."""
    import numpy as np
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    v01, v10, aucs = [], [], []
    for p in (np.asarray(p1, float), np.asarray(p2, float)):
        tx, ty = p[pos], p[neg]
        tz = _midrank(np.concatenate([tx, ty]))
        txr, tyr = _midrank(tx), _midrank(ty)
        v01.append((tz[:m] - txr) / n)
        v10.append(1.0 - (tz[m:] - tyr) / m)
        aucs.append(float(v01[-1].mean()))
    v01, v10 = np.array(v01), np.array(v10)
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    var = s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]
    var = var / m + (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n
    if var <= 0:
        return aucs[0], aucs[1], 0.0, 0.5
    z = (aucs[0] - aucs[1]) / math.sqrt(var)
    p_one = 0.5 * math.erfc(z / math.sqrt(2))
    return aucs[0], aucs[1], float(z), float(p_one)


def group_fold(group_id, k=5):
    """Deterministic fold assignment by group hash (no RNG state)."""
    return int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % k
