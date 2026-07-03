"""
src/decider_b.py
================
Phase-B decision center with extended operation algebra.

Adapted from /root/autodl-tmp/tact/decider.py for fd-badcat integration.
This is the prompt-engineered extended decision space (blueprint §3.2 / §4.1 regime 1).

The decider is the MLLM decision center's *transactional head*. Given the current
dialogue state, what the user has said (audio), and a snapshot of the pending
transaction, it emits a small JSON program over the operation algebra:

    {"dialogue": "speak|listen|stay",
     "ops": [ {"type":"launch","fn":"search_flights","args":{...}},
              {"type":"patch","op_id":12,"diff":{"destination":"Boston"}},
              {"type":"cancel","op_id":7},
              {"type":"commit","op_id":12} ],
     "say": "optional spoken response / progress narration"}

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

from transaction import Transaction, Reversibility, OpStatus


# ---------------------------------------------------------------------------
# Tool catalog & reversibility map (FDB-v3 tools)
# ---------------------------------------------------------------------------
# Reversibility lattice: READ ⪯ REV ⪯ COMP ⪯ IRR
REVERSIBILITY = {
    # Travel & Identity
    "search_flights":       Reversibility.READ,
    "book_flight":          Reversibility.COMP,   # compensator: cancel_booking
    "update_identity_doc":  Reversibility.IRR,
    # Finance & Billing
    "get_card_benefits":    Reversibility.READ,
    "get_exchange_rate":    Reversibility.READ,
    "modify_autopay":       Reversibility.COMP,   # compensator: revert_autopay
    # Housing & Location
    "search_apartments":    Reversibility.READ,
    "calculate_commute":    Reversibility.READ,
    "update_search_filter": Reversibility.REV,    # inverse: set previous value
    # E-Commerce
    "track_order":          Reversibility.READ,
    "search_products":      Reversibility.READ,
    "add_to_cart":          Reversibility.REV,    # inverse: remove_from_cart
}

COMPENSATORS = {
    "book_flight": "cancel_booking",
    "modify_autopay": "revert_autopay",
}

TOOL_CATALOG = """\
Travel & Identity:
  search_flights(destination, date)
  book_flight(passenger_name)
  update_identity_doc(doc_type, doc_number)
Finance & Billing:
  get_card_benefits(card_type)
  get_exchange_rate(amount, from_currency, to_currency)
  modify_autopay(bill_type, source_account)
Housing & Location:
  search_apartments(city, bedrooms, max_price, pets_allowed optional)
  calculate_commute(origin_address, destination_address, mode)
  update_search_filter(filter_name, value)
E-Commerce:
  track_order(order_id)
  search_products(query, max_price optional, category optional)
  add_to_cart(product_id, quantity)"""


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
# Message building (audio blocks), mirroring engine.build_messages()
# ---------------------------------------------------------------------------
# Which audio message-block format the decision model expects.
#   "audio_url"   -> LOCAL vLLM Qwen3-Omni (fd-badcat default)
#   "input_audio_datauri" -> DashScope cloud qwen3-omni-flash
AUDIO_BLOCK_FORMAT = "audio_url"


def _audio_block(audio: np.ndarray, sr: int = 16000) -> dict:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    uri = f"data:audio/wav;base64,{b64}"
    if AUDIO_BLOCK_FORMAT == "input_audio_datauri":
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
               f"USER_SAYS: {user_text if user_text else '(audio)'}")

    if audio is not None:
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": context},
            _audio_block(audio),
        ]})
    else:
        msgs.append({"role": "user", "content": [{"type": "text", "text": context}]})
    return msgs


# ---------------------------------------------------------------------------
# Decision parser & applier
# ---------------------------------------------------------------------------
def decide_and_apply(tx: Transaction, executor: Callable[[str, dict], dict],
                     llm_call: Callable[[list], str],
                     state: str, user_text: Optional[str] = None,
                     audio: Optional[np.ndarray] = None,
                     t: Optional[float] = None,
                     blocking: bool = True,
                     history: Optional[list] = None) -> dict:
    """
    Invoke the decision model, parse its JSON, apply ops to `tx`.

    Returns:
        {"dialogue": "...", "say": "...", "ops_applied": [...], "raw": "..."}

    blocking=True: launch+commit in one step (Phase-B v0).
    blocking=False: launch only, commit requires explicit op (async speculation, W3+).
    """
    msgs = build_decider_messages(tx, state, user_text, audio, history)
    raw = llm_call(msgs)

    # Parse JSON (strip markdown fences if present)
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    try:
        decision = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Fallback: no ops, stay silent
        return {"dialogue": "stay", "say": "", "ops_applied": [],
                "raw": raw, "parse_error": str(e)}

    dialogue = decision.get("dialogue", "stay")
    say = decision.get("say", "")
    ops = decision.get("ops", [])
    ops_applied = []

    for op in ops:
        op_type = op.get("type", "noop")
        if op_type == "noop":
            continue
        elif op_type == "launch":
            fn = op["fn"]
            args = op.get("args", {})
            rev = REVERSIBILITY.get(fn, Reversibility.IRR)
            pending_op = tx.launch(fn, args, rev, t=t)
            ops_applied.append({"type": "launch", "op_id": pending_op.op_id, "fn": fn})
            # blocking mode: commit immediately
            if blocking:
                tx.commit(pending_op.op_id, executor, t=t)
                ops_applied.append({"type": "commit", "op_id": pending_op.op_id})
        elif op_type == "patch":
            op_id = _resolve_op_id(op, tx)
            if op_id and op_id in tx.pending:
                diff = op.get("diff", {})
                tx.patch(op_id, diff, t=t)
                ops_applied.append({"type": "patch", "op_id": op_id, "diff": diff})
                # blocking mode: commit after patch
                if blocking and op_id in tx.pending:
                    tx.commit(op_id, executor, t=t)
                    ops_applied.append({"type": "commit", "op_id": op_id})
        elif op_type == "cancel":
            op_id = _resolve_op_id(op, tx)
            if op_id and op_id in tx.pending:
                tx.cancel(op_id, t=t)
                ops_applied.append({"type": "cancel", "op_id": op_id})
        elif op_type == "commit":
            op_id = _resolve_op_id(op, tx)
            if op_id and op_id in tx.pending:
                tx.commit(op_id, executor, t=t)
                ops_applied.append({"type": "commit", "op_id": op_id})

    return {"dialogue": dialogue, "say": say, "ops_applied": ops_applied, "raw": raw}


def _resolve_op_id(op: dict, tx: Transaction) -> Optional[int]:
    """Resolve op_id from op dict. If missing, fall back to fn name or latest pending."""
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
#   python -m src.decider_b
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Fake executor
    def _exec(fn, args):
        return {"status": "success", "fn": fn, "args": args}

    tx = Transaction()

    # Fake model: turn 1 launches+commits NYC; turn 2 patches to Boston then commits.
    scripted = iter([
        '{"dialogue":"speak","ops":[{"type":"launch","fn":"search_flights",'
        '"args":{"destination":"New York","date":"July 15"}}],"say":"Searching flights to New York."}',
        '{"dialogue":"speak","ops":[{"type":"patch","fn":"search_flights",'
        '"diff":{"destination":"Boston"}},{"type":"commit","fn":"search_flights"}],'
        '"say":"Updated to Boston."}',
    ])
    fake_llm = lambda msgs: next(scripted)

    d1 = decide_and_apply(tx, _exec, fake_llm, "LISTEN",
                          user_text="flight to New York on July 15", t=2.1, blocking=True)
    print("decision 1:", d1["say"])
    print("committed so far:", json.dumps(tx.to_actual_tool_calls(), ensure_ascii=False))
    print("\nOK: self-test passed.")
