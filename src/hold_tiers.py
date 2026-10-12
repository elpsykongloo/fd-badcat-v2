"""Per-candidate END hold tiers (hold-tier-v1): pure, table-driven, no engine state.

The guarded path releases a private reply at VAD END + max(hold, R + G). A single
hold for every input over-protects short, complete questions and under-protects
inputs that predict a continuation. This module only CLASSIFIES; the engine owns
timing and applies the result to one candidate.

Tiers (seconds, from configuration; without `end_hold_tiers` all three equal the
legacy `end_hold_frame`, so nothing changes):

- keep (never lowered): the transcript ends with a function word / connective,
  the whole input is >= 4 s, the candidate comes from a playback-time input,
  the text is not mainly Chinese, or (optional) the session RTT median is below
  `hold_min_rtt_ms` (client pre-playback hold cannot add budget there).
- low: route `yield_ready` AND whole input < 1.5 s AND (a question OR the first
  turn of the session) AND mainly Chinese AND not starting with an opener phrase.
- default: everything else.

Every feature is computed on the FULL admitted input: the transcript of all
accepted-unanswered audio merged into the reply, and the duration of that whole
concatenated audio, never on the newest segment alone.

Timing rule (`resolve`): a new candidate starts at default. Raising to keep is
always applied. Lowering to low applies only while (t - anchor) < H_low, i.e.
the transcript arrived before the low hold would already have expired; a late
transcript degrades to default, never to an immediate release.
"""
import math
import re

HOLD_TIER_PROTOCOL = "hold-tier-v1"
TIERS = ("low", "default", "keep")
LOW_MAX_S = 1.5
KEEP_MIN_S = 4.0

# Sentence-final function words / connectives: the user is likely to continue.
# Whole-word match at the end of the punctuation-stripped transcript; a longer
# complete word that merely ends with one of these characters is excluded below,
# and a single character only counts when it is attached to preceding speech.
TAIL_WORDS = (
    "就是说", "比如说", "的时候", "然后", "还有", "因为", "所以", "但是", "而且", "或者",
    "如果", "比如", "还是", "不过", "可是", "并且", "以及", "另外", "包括", "关于", "对于",
    "就是", "那个", "这个", "那些", "这些", "一个", "的话", "那么", "接着", "首先", "其次",
    "的", "是", "在", "和", "跟", "与", "把", "让", "给", "被", "从", "向", "往", "比",
    "呃", "嗯", "额", "就",
)
# Complete words that end with a tail word; matched first, longest wins.
TAIL_EXCEPTIONS = (
    "是的", "好的", "对的", "真的", "行的", "会的", "有的", "可以的", "没事的", "当然的",
    "是不是", "不是", "也是", "总是", "老是", "都是", "可不是", "于是", "凡是",
    "现在", "正在", "存在", "实在", "自在", "不在", "所在", "在不在",
    "暖和", "温和", "平和", "总和", "缓和", "随和", "附和", "讲和",
    "成就", "早就", "造就", "迁就", "将就",
    "目的", "似的", "怎么样的",
)
# Question cues. Deliberately NO bare 么/几/哪: they match 那么/茶几/哪怕.
QUESTION_WORDS = (
    "什么", "怎么", "为什么", "为啥", "干嘛", "干什么", "哪里", "哪儿", "哪个", "哪些", "哪位",
    "哪种", "哪天", "哪年", "谁", "几点", "几个", "几岁", "几号", "几天", "几年", "几次", "多少",
    "多久", "多大", "多长", "多远", "是不是", "有没有", "能不能", "可不可以", "会不会", "对不对",
    "好不好", "行不行", "要不要", "是否", "啥",
)
QUESTION_FINALS = ("吗", "呢", "嘛", "吧")
# Opener phrases announce a longer utterance; such inputs never take the low tier.
OPENERS = (
    "我想问一下", "我问一下", "问一下", "我想问", "我跟你说", "跟你说", "我想", "请问", "就是",
    "那么", "那个", "然后", "还有", "我觉得", "你知道", "对了", "话说", "我说", "嗯", "呃",
)
_PUNCT_END = "。.！!？?，,、；;：:…~～ \t\r\n\"'”’）)】》"
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_WORD = re.compile(r"[A-Za-z]+")
_A_NOT_A = re.compile(r"([\u4e00-\u9fff])不\1")  # 在不在 / 去不去 / 好不好


def language(text):
    """'zh' when Chinese characters dominate, 'en' when Latin words do, else ''."""
    cjk = len(_CJK.findall(text or ""))
    latin = len(_LATIN_WORD.findall(text or ""))
    if not cjk and not latin:
        return ""
    return "zh" if cjk >= latin else "en"


def _core(text):
    return (text or "").strip().rstrip(_PUNCT_END)


def tail_word(text):
    """The dangling function word/connective ending `text`, or ''. Whole words only."""
    core = _core(text)
    if not core:
        return ""
    if any(core.endswith(word) for word in TAIL_EXCEPTIONS):
        return ""
    for word in sorted(TAIL_WORDS, key=len, reverse=True):
        if core.endswith(word):
            # A lone character ("是。") is a complete answer, and a single-character
            # ending under an ASR question mark ("这是谁的？") is a complete question.
            if len(word) == 1 and (core == word or (text or "").rstrip().endswith(("？", "?"))):
                return ""
            return word
    return ""


def trailing_comma(text):
    """An ASR comma/ellipsis at the very end also marks an unfinished clause."""
    stripped = (text or "").rstrip()
    return bool(stripped) and stripped[-1] in "，,、…；;："


def is_question(text):
    stripped = (text or "").strip()
    if stripped.rstrip("”’\"') ）").endswith(("？", "?")):
        return True
    core = _core(stripped)
    return (core.endswith(QUESTION_FINALS) or any(word in core for word in QUESTION_WORDS)
            or bool(_A_NOT_A.search(core)))


def opener(text):
    core = (text or "").strip().lstrip("，,。. ")
    for word in sorted(OPENERS, key=len, reverse=True):
        if core.startswith(word):
            return word
    return ""


def classify(text, *, duration_s, label, playing, first_turn, rtt_ms=None, min_rtt_ms=0):
    """Rule tier and reason for one admitted input; timing is applied by `resolve`.

    text: full transcript of everything merged into the reply; duration_s: the
    whole concatenated input; label: the final route label (or a release reason).
    """
    text = text or ""
    lang = language(text)
    if playing:
        return "keep", "playing"
    if duration_s >= KEEP_MIN_S:
        return "keep", "long_input"
    if lang == "en":
        return "keep", "not_chinese"
    word = tail_word(text)
    if word:
        return "keep", "tail_word:" + word
    if trailing_comma(text):
        return "keep", "tail_comma"
    if min_rtt_ms and rtt_ms is not None and rtt_ms < min_rtt_ms:
        return "keep", "low_rtt"
    if label != "yield_ready":
        return "default", "not_ready"
    if not text.strip() or lang != "zh":
        return "default", "no_transcript"
    if duration_s >= LOW_MAX_S:
        return "default", "not_short"
    question = is_question(text)
    if not (question or first_turn):
        return "default", "not_question"
    word = opener(text)
    if word:
        return "default", "opener:" + word
    return "low", "short_question" if question else "short_first_turn"


def resolve(tiers, tier, reason, *, elapsed_s):
    """Apply the timing rule to a rule tier. Returns (tier, reason, hold_s)."""
    if tier == "low" and not elapsed_s < tiers["low"]:
        return "default", "low_late", tiers["default"]
    return tier, reason, tiers[tier]


def hold_config(engine_cfg, end_hold):
    """Validated tier configuration; absent keys reproduce the single legacy hold."""
    cfg = engine_cfg or {}
    raw = cfg.get("end_hold_tiers")
    if raw is None:
        tiers = {name: float(end_hold) for name in TIERS}
    else:
        if not isinstance(raw, dict) or set(raw) != set(TIERS):
            raise ValueError("end_hold_tiers must map exactly low, default and keep to seconds")
        tiers = {}
        for name in TIERS:
            value = raw[name]
            if type(value) not in (int, float) or not math.isfinite(value) or not .1 <= value <= 2:
                raise ValueError("end_hold_tiers values must be seconds in [0.1, 2]")
            tiers[name] = float(value)
        if not tiers["low"] <= tiers["default"] <= tiers["keep"]:
            raise ValueError("end_hold_tiers must satisfy low <= default <= keep")
    rules, shadow = cfg.get("hold_tier_rules", False), cfg.get("hold_tier_shadow", False)
    if type(rules) is not bool or type(shadow) is not bool:
        raise ValueError("hold_tier_rules and hold_tier_shadow must be booleans")
    min_rtt = cfg.get("hold_min_rtt_ms", 0)
    if type(min_rtt) not in (int, float) or not math.isfinite(min_rtt) or not 0 <= min_rtt <= 1000:
        raise ValueError("hold_min_rtt_ms must be in [0, 1000] (0 disables)")
    # Shadow wins: rules are evaluated and recorded but never change the hold.
    return {**tiers, "rules": rules, "shadow": shadow, "min_rtt_ms": min_rtt,
            "evaluate": rules or shadow, "apply": rules and not shadow}
