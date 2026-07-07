# -*- coding: utf-8 -*-
"""
src/engine_b_ack.py — ack-v0 (two-phase TTS) layer over the Phase-B v1 TactEngine.

W3 D1 rewrite against the unified engine. ack-v0 is a PURE TTS-SIDE optimization
(W2 D4 measurement: ack 0.429s vs full-sentence 0.933s, first audio 53.8% earlier):
when a decision both speaks and acts, synthesize a short acknowledgment first and
the full response second. It plugs into TactEngine via the `_emit_say` hook ONLY —
transaction semantics, objection windows and the commit barrier live in tact_core
and are untouched by this class.

Config (engine_cfg):
    ack_enabled: bool (default False)      gate, see tts_ack.should_use_ack
    ack_strategy: "random" | "context" | "fixed:<phrase>"
    ack_seed: int (default 42)             deterministic ack selection

Event flow (realtime): dispatch_tts_with_ack -> worker synthesizes both phases ->
two ModelDone events (kind tts_ack / tts_main) on the queue -> handlers send audio
and trace `tts_ack_done` / `tts_main_done`. _inflight is incremented by 2 at
dispatch — ActorEngine auto-decrements once per consumed ModelDone.
"""

import asyncio

from engine import ModelDone
from engine_b import TactEngine
from tts_ack import should_use_ack, synthesize_with_ack


class TactEngineWithAck(TactEngine):
    """TactEngine with ack-v0 first-audio optimization (say-emission override)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ack_strategy = self.engine_cfg.get("ack_strategy", "context")
        self.ack_seed = self.engine_cfg.get("ack_seed", 42)

    # ------------------------------------------------------------------
    # the ONLY integration point: TactEngine calls this when tts_enabled
    # ------------------------------------------------------------------
    def _emit_say(self, say, applied, turn):
        if should_use_ack(say, applied, self.engine_cfg):
            self.dispatch_tts_with_ack(say, applied, turn)
        else:
            super()._emit_say(say, applied, turn)

    # ------------------------------------------------------------------
    def dispatch_tts_with_ack(self, say: str, ops: list, turn: int):
        """Two-phase TTS: ack phrase immediately, main response follows."""
        gen, epoch = self.session_gen, self.seg_epoch

        if self.replay_mode != "realtime":
            res = self.decision_script("tts_ack", {"turn": turn, "say": say,
                                                   "t_audio": self.t_audio})
            ack_infer = float(res.get("ack_infer", 0.3))
            main_infer = float(res.get("main_infer", 0.9))
            self._schedule(self.t_audio + ack_infer,
                           ModelDone(kind="tts_ack", gen=gen, epoch=epoch, turn=turn,
                                     text=say, infer=ack_infer,
                                     dur_audio=float(res.get("ack_dur", 1.0))))
            self._schedule(self.t_audio + ack_infer + main_infer,
                           ModelDone(kind="tts_main", gen=gen, epoch=epoch, turn=turn,
                                     text=say, infer=main_infer,
                                     dur_audio=float(res.get("main_dur", 3.0))))
            return

        async def _run():
            try:
                ack_path, main_path, ack_lat, main_lat, _total = \
                    await synthesize_with_ack(say, self.tts_fn, self.output_dir,
                                              turn, ops=ops,
                                              strategy=self.ack_strategy,
                                              seed=self.ack_seed)
                import soundfile as sf
                for kind, path, lat in (("tts_ack", ack_path, ack_lat),
                                        ("tts_main", main_path, main_lat)):
                    data, sr = sf.read(path, dtype="float32")
                    with open(path, "rb") as f:
                        raw = f.read()
                    self.q.put_nowait(ModelDone(
                        kind=kind, gen=gen, epoch=epoch, turn=turn, text=say,
                        infer=round(lat, 3), wav_path=str(path),
                        dur_audio=len(data) / sr, audio_bytes=raw))
            except Exception as exc:
                print(f"[TTS ACK ERROR] {exc}")
                # keep _inflight accounting balanced, then fall back to baseline
                for kind in ("tts_ack", "tts_main"):
                    self.q.put_nowait(ModelDone(kind=kind, gen=gen, epoch=epoch,
                                                turn=turn, text="", infer=0.0,
                                                error=str(exc)))
                self.dispatch_tts(say, turn)

        self._inflight += 2   # one per queued ModelDone (auto-decremented on consume)
        asyncio.create_task(_run())

    # ------------------------------------------------------------------
    # result routing: intercept ack kinds BEFORE ActorEngine's handler dict
    # ------------------------------------------------------------------
    async def _on_model_done(self, ev: ModelDone):
        if ev.kind == "tts_ack":
            await self._on_tts_ack_done(ev)
            return
        if ev.kind == "tts_main":
            await self._on_tts_main_done(ev)
            return
        await super()._on_model_done(ev)

    async def _on_tts_ack_done(self, ev: ModelDone):
        if ev.gen != self.session_gen:
            return
        if self.websocket is not None and ev.audio_bytes:
            await self.websocket.send_bytes(ev.audio_bytes)
        await self.send_control("tts_ack_done", {
            "timestamp": self._wall_ts(), "turn": ev.turn,
            "infer_time": ev.infer, "dur_audio": round(ev.dur_audio, 3),
            "first_response_latency": ev.infer})
        if not ev.error and ev.dur_audio:
            self.STATE = "SPEAK"
            self.playback_end_audio = self.t_audio + ev.dur_audio

    async def _on_tts_main_done(self, ev: ModelDone):
        if ev.gen != self.session_gen:
            return
        if self.websocket is not None and ev.audio_bytes:
            await self.websocket.send_bytes(ev.audio_bytes)
        await self.send_control("tts_main_done", {
            "timestamp": self._wall_ts(), "turn": ev.turn,
            "infer_time": ev.infer, "dur_audio": round(ev.dur_audio, 3)})
        if not ev.error and ev.dur_audio:
            self.STATE = "SPEAK"
            self.playback_end_audio = self.t_audio + ev.dur_audio


def create_tact_engine_with_ack(**kwargs):
    """Factory function for creating TactEngine with ack-v0 support."""
    return TactEngineWithAck(**kwargs)
