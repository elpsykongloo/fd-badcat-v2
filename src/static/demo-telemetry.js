// Diagnostic-only. No turn decisions, audio mutation, clock synchronization,
// or claims about physical speaker onset. Legacy servers simply don't enable it.
export class DemoTelemetry {
  constructor(send, context, buffered, rtt = () => {}, now = () => performance.now()) {
    this.send = send; this.context = context; this.buffered = buffered;
    this.rtt = rtt; this.now = now; this.seq = 0;
    this.enabled = false; this.pending = null;
    this.lastPing = -Infinity;
  }
  emit(kind, data = {}) {
    if (this.enabled) this.send("demo_telemetry", {kind, client_ms: this.now(),
      upload_buffer_bytes: this.buffered(), ...data});
  }
  tick() {
    if (!this.enabled || this.now() - this.lastPing < 5000) return;
    this.lastPing = this.now();
    this.pending = {seq: ++this.seq, at: this.lastPing};
    this.emit("ping", {seq: this.seq});
  }
  pong(data) {
    if (!this.pending || data.seq !== this.pending.seq) return;
    const ms = this.now() - this.pending.at;
    this.pending = null;
    this.rtt(ms);
    this.emit("rtt", {seq: data.seq, rtt_ms: ms});
  }
  playback(s) {
    if (!this.enabled) return;
    if (s.firstAudio !== undefined && !s.telemetryFirst) {
      s.telemetryFirst = true;
      this.emit("first_audio", {utterance_id: s.id, first_audio_ms: s.firstAudio,
        scheduled_lead_ms: s.scheduledLeadMs,
        base_latency_ms: this.context.baseLatency * 1000,
        output_latency_ms: this.context.outputLatency * 1000});
    }
    if (s.eof && s.received === s.played && !s.telemetryEnd) {
      s.telemetryEnd = true;
      this.snapshot("playback_end", s);
    }
  }
  snapshot(kind, s) {
    this.emit(kind, s ? {utterance_id: s.id, played_samples: s.played,
      received_samples: s.received, underruns: s.underruns} : {});
  }
}
