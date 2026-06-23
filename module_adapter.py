"""
tact/module_adapter.py
======================
Thin binding between TACT and your existing model interface in src/module.py.

The decider only needs a callable `llm_call(messages: list[dict]) -> str`. Your
`llm_qwen3o` already has exactly that signature, so this is a one-line re-export
that also fixes up the import path whether you run from the repo root or from src/.

If you later want the decider to use a *different* model than the dialogue head
(e.g. a cheaper one for the transactional head), swap the binding here only.
"""

from __future__ import annotations

import json
import os

import requests

from .paths import add_to_syspath, fd_badcat_src


_HTTP = requests.Session()
_HTTP.trust_env = False


def _resolve_llm():
    add_to_syspath(fd_badcat_src())
    try:
        from module import llm_qwen3o

        return llm_qwen3o
    except Exception:
        return _http_llm_qwen3o


def _http_llm_qwen3o(messages: list, response_format: dict | None = None) -> str:
    """OpenAI-compatible fallback matching fd-badcat/src/module.py."""
    payload = {
        "model": os.getenv("FDBC_QWEN_MODEL", "Qwen3-Omni-30B-A3B-Instruct"),
        "temperature": float(os.getenv("FDBC_QWEN_TEMPERATURE", "0")),
        "top_p": float(os.getenv("FDBC_QWEN_TOP_P", "0.7")),
        "top_k": int(os.getenv("FDBC_QWEN_TOP_K", "40")),
        "presence_penalty": float(os.getenv("FDBC_QWEN_PRESENCE_PENALTY", "1.2")),
        "frequency_penalty": float(os.getenv("FDBC_QWEN_FREQUENCY_PENALTY", "0.8")),
        "max_tokens": int(os.getenv("FDBC_QWEN_MAX_TOKENS", "768")),
        "seed": int(os.getenv("FDBC_QWEN_SEED", "42")),
        "modalities": ["text"],
        "messages": messages,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    response = _HTTP.post(
        os.getenv("FDBC_QWEN_URL", "http://127.0.0.1:10004/v1/chat/completions"),
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=int(os.getenv("FDBC_QWEN_TIMEOUT", "600")),
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# the callable the decider expects: messages -> text
def llm_text(messages: list) -> str:
    return _resolve_llm()(messages)
