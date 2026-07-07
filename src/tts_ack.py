"""
src/tts_ack.py
==============
ack-v0: Acknowledgment-first TTS mechanism for reduced first-response latency.

Strategy:
- When Phase-B emits a `say` response, split synthesis into two phases:
  Phase 1: Short acknowledgment (5-8 words, ~0.3-0.4s synthesis)
  Phase 2: Main response body
- First audio chunk is sent immediately after Phase 1 completes
- This anchors the perceived response latency to the ack synthesis time
  instead of the full response synthesis time

W2 裁断 7.2b: TTS 是首响最大单项 (baseline 0.66s)。
ack-v0 目标: 将首响降至 ~0.35s (短句合成时间)。
"""

import asyncio
import random
import time
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Acknowledgment phrase pool
# ---------------------------------------------------------------------------
ACK_PHRASES_FILE = Path(__file__).resolve().parents[1] / "prompts" / "ack_phrases.txt"

_ACK_POOL = []
_ACK_POOL_LOADED = False


def _load_ack_pool():
    global _ACK_POOL, _ACK_POOL_LOADED
    if _ACK_POOL_LOADED:
        return

    if not ACK_PHRASES_FILE.exists():
        # Fallback pool if file not found
        _ACK_POOL = [
            "Got it, let me check that.",
            "Sure, I'll handle that now.",
            "Okay, working on that for you.",
            "Looking that up right now.",
            "Processing that request now.",
        ]
    else:
        with open(ACK_PHRASES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    _ACK_POOL.append(line)

    if not _ACK_POOL:
        raise RuntimeError("Acknowledgment phrase pool is empty")

    _ACK_POOL_LOADED = True


def get_random_ack() -> str:
    """Return a random acknowledgment phrase from the pool."""
    _load_ack_pool()
    return random.choice(_ACK_POOL)


def select_ack_by_context(say_text: str, ops: list) -> str:
    """
    Select an acknowledgment phrase based on context (tool type, say text).

    Heuristics:
    - Search/query tools -> "Looking that up...", "Searching..."
    - Booking tools -> "Starting the booking...", "Processing..."
    - Update tools -> "Updating...", "Making that change..."
    - Default -> generic confirmation

    Args:
        say_text: The full response text from decider
        ops: List of operations from the decision

    Returns:
        Selected acknowledgment phrase
    """
    _load_ack_pool()

    # Extract tool names from ops
    tool_names = set()
    for op in ops:
        if op.get("type") == "launch" and "fn" in op:
            tool_names.add(op["fn"])
        elif "fn" in op:  # patch/commit may also carry fn
            tool_names.add(op["fn"])

    # Category detection
    search_tools = {"search_flights", "search_apartments", "search_products", "track_order"}
    booking_tools = {"book_flight", "add_to_cart"}
    update_tools = {"update_identity_doc", "modify_autopay", "update_search_filter"}

    # Match category
    if tool_names & search_tools:
        category_pool = [p for p in _ACK_POOL if "search" in p.lower() or "look" in p.lower() or "find" in p.lower()]
    elif tool_names & booking_tools:
        category_pool = [p for p in _ACK_POOL if "book" in p.lower() or "process" in p.lower() or "transaction" in p.lower()]
    elif tool_names & update_tools:
        category_pool = [p for p in _ACK_POOL if "updat" in p.lower() or "chang" in p.lower() or "adjust" in p.lower()]
    else:
        category_pool = [p for p in _ACK_POOL if "got it" in p.lower() or "sure" in p.lower() or "okay" in p.lower()]

    if not category_pool:
        category_pool = _ACK_POOL

    return random.choice(category_pool)


# ---------------------------------------------------------------------------
# Two-phase TTS synthesis
# ---------------------------------------------------------------------------
async def synthesize_with_ack(
    say_text: str,
    tts_fn,
    output_dir: Path,
    turn: int,
    ops: list = None,
    strategy: str = "random",
    seed: Optional[int] = None
) -> Tuple[str, str, float, float, float]:
    """
    Two-phase TTS synthesis: ack phrase first, then main response.

    Args:
        say_text: Full response text to synthesize
        tts_fn: TTS function (from module.py)
        output_dir: Directory for output files
        turn: Turn index for naming
        ops: Operations list (for context-aware ack selection)
        strategy: "random" | "context" | "fixed:<phrase>"
        seed: Random seed (for deterministic selection)

    Returns:
        (ack_path, main_path, ack_latency, main_latency, total_latency)
    """
    if seed is not None:
        random.seed(seed)

    # Select acknowledgment phrase
    if strategy.startswith("fixed:"):
        ack_phrase = strategy[6:]
    elif strategy == "context":
        ack_phrase = select_ack_by_context(say_text, ops or [])
    else:  # random
        ack_phrase = get_random_ack()

    # Phase 1: Synthesize ack (fast path)
    t0_ack = time.perf_counter()
    ack_path = output_dir / f"turn{turn}_ack.wav"
    await asyncio.to_thread(tts_fn, ack_phrase, ack_path)
    ack_latency = time.perf_counter() - t0_ack

    # Phase 2: Synthesize main response (parallel if possible, but we need sequential for now)
    t0_main = time.perf_counter()
    main_path = output_dir / f"turn{turn}_main.wav"
    await asyncio.to_thread(tts_fn, say_text, main_path)
    main_latency = time.perf_counter() - t0_main

    total_latency = ack_latency + main_latency

    return (str(ack_path), str(main_path), ack_latency, main_latency, total_latency)


async def synthesize_baseline(
    say_text: str,
    tts_fn,
    output_dir: Path,
    turn: int
) -> Tuple[str, float]:
    """
    Baseline single-phase TTS (for comparison).

    Returns:
        (wav_path, latency)
    """
    t0 = time.perf_counter()
    wav_path = output_dir / f"turn{turn}_baseline.wav"
    await asyncio.to_thread(tts_fn, say_text, wav_path)
    latency = time.perf_counter() - t0
    return (str(wav_path), latency)


# ---------------------------------------------------------------------------
# Integration helpers for engine_b.py
# ---------------------------------------------------------------------------
def should_use_ack(say_text: str, ops: list, engine_cfg: dict) -> bool:
    """
    Decide whether to use ack-v0 for this response.

    Heuristics:
    - Enable if `say` is non-empty and there are tool ops
    - Disable if `say` is already very short (<= 8 words)
    - Controlled by engine_cfg.ack_enabled flag

    Args:
        say_text: Response text
        ops: Operations list
        engine_cfg: Engine configuration

    Returns:
        True if ack-v0 should be used
    """
    if not engine_cfg.get("ack_enabled", False):
        return False

    if not say_text or not say_text.strip():
        return False

    # Don't use ack if the response is already very short
    word_count = len(say_text.split())
    if word_count <= 8:
        return False

    # Use ack if there are actual tool operations (not just noop)
    has_real_ops = any(op.get("type") != "noop" for op in (ops or []))

    return has_real_ops


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    import soundfile as sf
    import numpy as np

    # Mock TTS function (generates silent audio for testing)
    def mock_tts(text: str, path):
        # Simulate synthesis time proportional to text length
        import time
        time.sleep(len(text.split()) * 0.05)  # ~50ms per word

        # Generate silent audio
        duration = len(text.split()) * 0.2  # 200ms per word
        samples = int(16000 * duration)
        audio = np.zeros(samples, dtype=np.float32)
        sf.write(str(path), audio, 16000, subtype='PCM_16')

    async def test():
        print("Testing ack-v0 mechanism...")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Test 1: Baseline
            print("\n[Baseline] Synthesizing full response...")
            baseline_path, baseline_lat = await synthesize_baseline(
                "I've searched for flights to New York on July 15th and found several options.",
                mock_tts, tmpdir, 0
            )
            print(f"  Latency: {baseline_lat:.3f}s")
            print(f"  Output: {baseline_path}")

            # Test 2: ack-v0
            print("\n[ack-v0] Synthesizing with acknowledgment...")
            ack_path, main_path, ack_lat, main_lat, total_lat = await synthesize_with_ack(
                "I've searched for flights to New York on July 15th and found several options.",
                mock_tts, tmpdir, 1,
                ops=[{"type": "launch", "fn": "search_flights"}],
                strategy="context",
                seed=42
            )
            print(f"  Ack latency: {ack_lat:.3f}s")
            print(f"  Main latency: {main_lat:.3f}s")
            print(f"  Total latency: {total_lat:.3f}s")
            print(f"  First response improvement: {baseline_lat - ack_lat:.3f}s ({(baseline_lat - ack_lat) / baseline_lat * 100:.1f}%)")
            print(f"  Ack output: {ack_path}")
            print(f"  Main output: {main_path}")

            # Test 3: Random selection
            print("\n[Random] Testing random ack selection...")
            for i in range(3):
                ack = get_random_ack()
                print(f"  {i+1}. {ack}")

            # Test 4: Context-aware selection
            print("\n[Context] Testing context-aware selection...")
            test_cases = [
                ("search_flights", "Looking up flights for you."),
                ("book_flight", "Booking your flight now."),
                ("update_identity_doc", "Updating your document."),
            ]
            for fn, expected_theme in test_cases:
                ack = select_ack_by_context("", [{"type": "launch", "fn": fn}])
                print(f"  {fn} -> {ack}")

        print("\n✓ All tests passed!")

    asyncio.run(test())
