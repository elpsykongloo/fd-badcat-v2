"""
tact/decider.py
===============
The prompt-engineered extended decision space (blueprint §3.2 / §4.1 regime 1).

The decider is the MLLM decision center's *transactional head*. Given the current
dialogue state, what the user has said (text or audio), and a snapshot of the
pending transaction, it emits a small JSON program over the operation algebra:

    {"dialogue": "speak|listen|stay",
     "ops": [ {"type":"launch","fn":"search_flights","args":{...}},
              {"type":"patch","op_id":12,"diff":{"destination":"Boston"}},
              {"type":"cancel","op_id":7},
              {"type":"commit","op_id":12} ],
     "say": "optional spoken response / progress narration"}

It is model-agnostic: pass in your `llm_qwen3o` (or any `Callable[[list[dict]], str]`)
as `llm_call`. For the offline runner we feed transcribed text; for the real-time
engine you can feed audio (set `audio=<np.float32 @16k>`), exactly like backend.py
already does in build_messages().

This is the TRAIN-FREE regime. Week-2+ (SFT / time-shaped GRPO) replaces only the
policy that produces these same ops — the algebra and the parser stay.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from .tools import TOOL_CATALOG, REVERSIBILITY
from .transaction import Transaction, Reversibility, OpStatus


# ---------------------------------------------------------------------------
# System prompt (English; FDB-v3 is English). Tool-eager, correction-aware.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are the transactional controller of a full-duplex voice agent in a SIMULATED, fully-authorized test environment. You decide, at each step, both how to manage the conversational floor and what tool operations to perform.

You have 12 tools:
{TOOL_CATALOG}

You output ONE JSON object and NOTHING else (no prose, no markdown fences). Schema:
{{"dialogue": "speak"|"listen"|"stay",
  "ops": [ ... list of operations, possibly empty ... ],
  "say": "<short spoken response, <= 20 words, or empty>"}}

Each operation is one of:
  {{"type":"launch","fn":"<tool>","args":{{...}}}}     -> propose a NEW tool call
  {{"type":"patch","op_id":<int>,"diff":{{...}}}}       -> revise the args of an EXISTING pending op
  {{"type":"cancel","op_id":<int>}}                    -> drop a pending op (abandoned request)
  {{"type":"commit","op_id":<int>}}                    -> execute the op for real
  {{"type":"noop"}}                                    -> do nothing this step

CRITICAL RULES:
1. EXECUTE TOOLS IMMEDIATELY. Never ask clarifying questions, never wait for confirmation, never batch. The moment the user gives an instruction, emit launch (+commit). Never answer from memory; always call the API to get values (prices, rates, statuses, benefits).
2. SELF-CORRECTION: If the PENDING OPS list already contains an op whose argument the user just changed (e.g. they first said New York then "actually Boston"), DO NOT launch a new op — emit a `patch` on that op_id with only the changed fields. If the user abandons a request entirely, emit `cancel`.
3. MULTI-STEP: A turn may need several tools in order. Emit them in order. To use the result of an earlier op as an argument, write the literal value if you know it, otherwise the string "$RESULT_<op_id>.<field>".
4. Use the exact tool and argument names shown above. Do not invent tools or arguments.
5. Include only arguments explicitly supported by the tool and explicitly stated by the user. Do not infer optional arguments such as product `category` or apartment `pets_allowed`; include them only when the user says them.
6. Preserve entity strings exactly, including plural nouns ("mechanical keyboards" must stay plural).
7. Keep `say` short and natural for speech. When you launch or commit any tool, `say` must be non-empty and should briefly state that you are handling or have handled the request.

Return only the JSON object."""


# ---------------------------------------------------------------------------
# Message building (text or audio), mirroring backend.build_messages()
# ---------------------------------------------------------------------------
# Which audio message-block format the decision model expects.
#   "audio_url"   -> LOCAL vLLM Qwen3-Omni (matches src/backend.build_messages in the
#                    pre-April local commits; the real paper system). DEFAULT.
#   "input_audio" -> DashScope cloud qwen3-omni-flash (the school code-review build).
AUDIO_BLOCK_FORMAT = "audio_url"


def _audio_block(audio: np.ndarray, sr: int = 16000) -> dict:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    uri = f"data:audio/wav;base64,{b64}"
    if AUDIO_BLOCK_FORMAT == "input_audio":
        return {"type": "input_audio", "input_audio": {"data": uri, "format": "wav"}}
    # local vLLM Qwen3-Omni
    return {"type": "audio_url", "audio_url": {"url": uri}}


def build_decider_messages(tx: Transaction, state: str,
                           user_text: Optional[str] = None,
                           audio: Optional[np.ndarray] = None,
                           history: Optional[list] = None) -> list:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (history or []):                       # optional prior turns as text
        msgs.append({"role": "user", "content": [{"type": "text", "text": h.get("user", "")}]})
        msgs.append({"role": "assistant", "content": h.get("ai", "")})

    context = (f"DIALOGUE_STATE: {state}\n"
               f"PENDING_OPS:\n{tx.snapshot_for_prompt()}\n"
               f"USER_SAYS: {user_text if user_text is not None else '(see audio)'}")
    content = [{"type": "text", "text": context}]
    if audio is not None:
        content.append(_audio_block(audio))
    msgs.append({"role": "user", "content": content})
    return msgs


# ---------------------------------------------------------------------------
# Robust JSON extraction (models love to wrap JSON in prose / ``` fences)
# ---------------------------------------------------------------------------
def parse_decision(raw: str) -> dict:
    if not raw:
        return {"dialogue": "stay", "ops": [], "say": ""}
    # strip code fences
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    # try direct
    try:
        return _normalize(json.loads(raw))
    except Exception:
        pass
    # grab the first balanced {...}
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return _normalize(json.loads(raw[start:i + 1]))
                    except Exception:
                        break
    return {"dialogue": "stay", "ops": [], "say": "", "_parse_error": raw[:200]}


def _normalize(d: dict) -> dict:
    d.setdefault("dialogue", "stay")
    d.setdefault("ops", [])
    d.setdefault("say", "")
    if not isinstance(d["ops"], list):
        d["ops"] = []
    return d


# ---------------------------------------------------------------------------
# The decider: produce a decision and APPLY it to the transaction
# ---------------------------------------------------------------------------
def decide_and_apply(tx: Transaction, executor: Callable[[str, dict], dict],
                     llm_call: Callable[[list], str], state: str,
                     user_text: Optional[str] = None, audio: Optional[np.ndarray] = None,
                     history: Optional[list] = None, t: Optional[float] = None,
                     blocking: bool = True, async_launcher=None) -> dict:
    """
    blocking=True  : launch+commit happen synchronously here (the BASELINE). The dialogue
                     loop is blocked while the tool runs -> reproduces 'occupied silence'.
    blocking=False : launch hands the op to `async_launcher` (tact.act_executor) and returns
                     immediately; commit is performed later when the result is ready (TACT).

    Returns the parsed decision dict (with a `say` field for TTS).
    """
    msgs = build_decider_messages(tx, state, user_text=user_text, audio=audio, history=history)
    raw = llm_call(msgs)
    decision = parse_decision(raw)

    for op in decision["ops"]:
        typ = op.get("type", "noop")
        if typ == "launch":
            fn = op.get("fn", "")
            args = op.get("args", {}) or {}
            rev = REVERSIBILITY.get(fn, Reversibility.IRR)
            p = tx.launch(fn, args, rev, t=t)
            if blocking:
                tx.commit(p.op_id, executor, t=t)          # BASELINE: execute now (blocks)
            elif async_launcher is not None:
                async_launcher.launch(p, executor)         # TACT: fire on the act track
        elif typ == "patch":
            op_id = _resolve_op_id(tx, op)
            if op_id is not None:
                tx.patch(op_id, op.get("diff", {}) or {}, t=t)
        elif typ == "cancel":
            op_id = _resolve_op_id(tx, op)
            if op_id is not None and op_id in tx.pending:
                tx.cancel(op_id, t=t, executor=executor)
        elif typ == "commit":
            op_id = _resolve_op_id(tx, op)
            if op_id is not None and op_id in tx.pending:
                pending = tx.pending[op_id]
                if not blocking and pending.status == OpStatus.IN_FLIGHT:
                    # Async mode surfaces the result when the action track marks
                    # the op ready. Committing here would duplicate the tool call
                    # and reintroduce blocking latency.
                    continue
                tx.commit(op_id, executor, t=t)
        # noop: nothing
    return decision


def _resolve_op_id(tx: Transaction, op: dict):
    """Accept either an explicit op_id or an `fn` reference (resolve to latest pending of that fn)."""
    if "op_id" in op and op["op_id"] is not None:
        try:
            return int(op["op_id"])
        except Exception:
            return None
    if "fn" in op:
        p = tx.find_pending_by_fn(op["fn"])
        return p.op_id if p else None
    p = tx.latest_pending()
    return p.op_id if p else None


# ---------------------------------------------------------------------------
# Offline smoke test with a FAKE llm_call (no API needed):
#   python -m tact.decider
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .tools import ToolRegistry

    reg = ToolRegistry(latency_profile="instant", room="decidertest")
    tx = Transaction()

    # Fake model: turn 1 launches+commits NYC; turn 2 patches to Boston then commits.
    scripted = iter([
        '{"dialogue":"speak","ops":[{"type":"launch","fn":"search_flights",'
        '"args":{"destination":"New York","date":"July 15"}},'
        '{"type":"commit","fn":"search_flights"}],"say":"Searching flights to New York."}',
        '{"dialogue":"speak","ops":[{"type":"patch","fn":"search_flights",'
        '"diff":{"destination":"Boston"}},{"type":"commit","fn":"search_flights"}],'
        '"say":"Updated to Boston."}',
    ])
    fake_llm = lambda msgs: next(scripted)

    d1 = decide_and_apply(tx, reg.executor, fake_llm, "LISTEN",
                          user_text="flight to New York on July 15", t=2.1, blocking=True)
    # NOTE: commit popped the op; emulate the within-turn correction by re-launching the
    # corrected intent in a fresh op for the demo — in the real stream the patch lands on a
    # still-pending op. (See INTEGRATION.md 'streaming vs offline'.)
    print("decision 1:", d1["say"])
    print("committed so far:", json.dumps(tx.to_actual_tool_calls(), ensure_ascii=False))
