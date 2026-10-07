"""Canonical demo speech, including adversarial token boundaries; no models."""
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from spoken_text import SpokenTextBuffer, SpeechTextError
from speech_stream import SpeechPipeline
from stream_transport import PCMChunk
from test_speech_stream import collect


@pytest.mark.parametrize("raw,expected", [
    ("你好。今天晴天！", "你好。今天晴天！"),
    ("下面是两个要点。\n\n1. **时态**：时态表示动作发生的时间。\n2. **语序**：语序影响意思。",
     "下面是两个要点。\n时态：时态表示动作发生的时间。\n语序：语序影响意思。"),
    ("## 标题\n- 清洗杯子。\n- 擦干。", "标题\n清洗杯子。\n擦干。"),
    ("> 引用文字。\n> **重点**。", "引用文字。\n重点。"),
    ("**Hello. World.** Next.", "Hello. World. Next."),
    ("***重点***与 _强调_、~~旧版~~。", "重点与 强调、旧版。"),
    ("[文档](https://example.com/a) 可以查看。", "文档 可以查看。"),
    ("![一只小鸟](https://example.com/bird.png)", "一只小鸟"),
    ("Read `a_b` and 3.14. Next!", "Read a_b and 3.14. Next!"),
    ("```python\nprint(\"a_b\")\n```\n完成。", 'print("a_b")\n完成。'),
    ("~~~text\n2 * 3 = 6\n~~~\n完成。", "2 * 3 = 6\n完成。"),
    ("3.14、1,000.5、12:30。", "3.14、1,000.5、12:30。"),
    ("2 * 3 = 6; foo_bar is a name.", "2 * 3 = 6; foo_bar is a name."),
    ("2*3*4 = 24.", "2*3*4 = 24."),
    ("2➕3=5，6➗2=3。", "2+3=5，6/2=3。"),
    ("第一行\r\n第二行", "第一行\n第二行"),
    ("1.", "1."),  # A legitimate numeric answer must not disappear.
    ("A story. 🌫️💡\nThe end.", "A story.\nThe end."),
    ("今天☀️，出门走走吧😊。", "今天，出门走走吧。"),
    ("👩🏽‍💻", ""),
    ("| 姓名 | 数值 |\n| --- | ---: |\n| 小猫 | 3 |", "姓名；数值\n小猫；3"),
    ("**", ""), ("---\n", ""),
])
def test_canonical_text_independent_of_chunk_boundaries(raw, expected):
    for width in range(1, max(2, len(raw) + 1)):
        normalizer = SpokenTextBuffer()
        result = "".join(normalizer.feed(raw[i:i + width]) for i in range(0, len(raw), width))
        result += normalizer.flush()
        assert result == expected, (width, repr(result))


def test_plain_first_sentence_streams_without_waiting_for_paragraph():
    n = SpokenTextBuffer()
    assert n.feed("你好。接") == "你好。"
    assert n.feed("下来解释。好") == "接下来解释。"
    assert n.flush() == "好"


def test_open_markup_is_not_published_then_discarded():
    n = SpokenTextBuffer()
    assert n.feed("**First. ") == ""
    assert "**" not in n.feed("Second.** Next.") + n.flush()
    with pytest.raises(SpeechTextError):
        SpokenTextBuffer(max_buffer=64).feed("**" + "a" * 65)


async def test_pipeline_tts_ui_and_done_share_canonical_text():
    raw = "要点如下。\n1. **时态**：时态表示动作发生的时间。\n2. **语序**：语序影响意思。"
    async def text(_):
        for char in raw:
            yield char
    spoken = []
    async def tts(sentence):
        spoken.append(sentence)
        yield PCMChunk(b"\0\0" * 960, 24000)
    queue, events = asyncio.Queue(), []
    pipeline = SpeechPipeline(1, queue, [], text, tts, spoken_text=True, track_sentences=True)
    await collect(pipeline, queue, events)
    assert not any(e.kind == "error" for e in events)
    public = "".join(e.data["text"] for e in events if e.kind == "text_delta")
    done = next(e.data["text"] for e in events if e.kind == "text_done")
    assert public == done
    assert "**" not in public and "1." not in public and "2." not in public
    assert not any(s.strip() in {"1.", "2.", "**"} for s in spoken)
    assert "".join(spoken).replace("\n", "") == public.replace("\n", "")


async def test_literal_pipeline_unchanged_when_flag_off():
    raw = "**literal**。"
    async def text(_):
        yield raw
    spoken = []
    async def tts(sentence):
        spoken.append(sentence)
        yield PCMChunk(b"\0\0" * 960, 24000)
    q, events = asyncio.Queue(), []
    pipeline = SpeechPipeline(1, q, [], text, tts)
    await collect(pipeline, q, events)
    assert spoken == [raw]


async def test_formatting_only_is_explicit_failure_not_false_success():
    async def text(_):
        yield "**\n---\n"
    async def tts(_):
        raise AssertionError("Formatting is not a TTS sentence")
        yield
    q, events = asyncio.Queue(), []
    p = SpeechPipeline(1, q, [], text, tts, spoken_text=True)
    await collect(p, q, events)
    assert any(e.kind == "error" for e in events)
    assert not any(e.kind in {"text_done", "audio_end"} for e in events)
