"""Demo response completion policy; no model, history or engine state writes."""


class ResponseIncompleteError(RuntimeError):
    code = "response_incomplete"

    def __init__(self, reason):
        self.reason = reason
        super().__init__(f"Response incomplete: {reason}")


def response_length_config(engine_cfg):
    if not (engine_cfg.get("chat_demo") and engine_cfg.get("response_length_repair")):
        return None
    tokens = engine_cfg.get("response_max_tokens", 512)
    continuations = engine_cfg.get("response_max_continuations", 3)
    if type(tokens) is not int or not 64 <= tokens <= 1024:
        raise ValueError("response_max_tokens must be an integer in [64, 1024]")
    if type(continuations) is not int or not 0 <= continuations <= 8:
        raise ValueError("response_max_continuations must be an integer in [0, 8]")
    return {"version": "response-completion-v2", "max_tokens": tokens,
            "max_continuations": continuations, "audio_grounded": True}


def continuation_messages(messages, prefix, budget):
    """Keep the frozen system/current audio and exact prefix; evict oldest pairs.

    UTF-8 bytes + role allowance conservatively bound text tokens, as in the
    Actor's initial history window. Never crop a word, current audio or prefix.
    If even the prefix does not fit, fail explicitly instead of asking the
    model to continue from a silently shortened version of its own answer.
    """
    cost = len(prefix.encode("utf-8")) + 16
    if not prefix or cost > budget:
        raise ResponseIncompleteError("continuation_context_budget")
    base = list(messages)
    # The last user message holds this turn's original audio. Only preceding
    # complete user/assistant history pairs are eligible for eviction.
    if len(base) < 2 or base[0].get("role") != "system" or base[-1].get("role") != "user":
        raise ResponseIncompleteError("invalid_continuation_context")
    history = base[1:-1]
    if len(history) % 2 or any(m.get("role") != ("user" if i % 2 == 0 else "assistant")
                              for i, m in enumerate(history)):
        raise ResponseIncompleteError("invalid_continuation_history")

    def text_cost(message):
        content = message.get("content", "")
        if isinstance(content, list):
            if any(block.get("type") != "text" for block in content):
                raise ResponseIncompleteError("nontext_continuation_history")
            content = "".join(block.get("text", "") for block in content)
        return len(str(content).encode("utf-8")) + 16

    costs = [text_cost(m) for m in history]
    first = 0
    while cost + sum(costs[first:]) > budget:
        first += 2
    return [base[0], *history[first:], base[-1], {"role": "assistant", "content": prefix}], first // 2
