// Arrival is never a playback acknowledgement. Cancellation also fences
// packets already on the wire. The PCM contract is independent of the UI.
export class SpeechPlayer {
  constructor(context, send, update = () => {}, destination = context.destination) {
    this.context = context;
    this.send = send;
    this.update = update;
    this.nodes = new Set();
    this.speech = null;
    this.destination = destination;
  }
  cancel() {
    const s = this.speech;
    let stopped;
    if (s) {
      // Account the partial node using the same WebAudio clock as scheduling.
      let played = s.played;
      for (const node of this.nodes) {
        if (node.fdStart !== undefined && this.context.currentTime >= node.fdStart) played = Math.max(played, node.fdOffset + Math.max(0,
          Math.min(node.fdCount, Math.floor((this.context.currentTime - node.fdStart) * s.rate))));
      }
      stopped = {utterance_id: s.id, played_samples: Math.min(played, s.received)};
    }
    this.speech = null;
    for (const node of this.nodes) {
      node.onended = null;
      node.stop();
      node.disconnect();
    }
    this.nodes.clear();
    if (stopped) this.send("playback_stopped", stopped);
  }
  start(data) {
    this.cancel();
    if (!Number.isInteger(data.utterance_id) || !Number.isFinite(data.buffer_ms)
        || data.buffer_ms <= 0 || data.buffer_ms > 2000) throw Error("无效播放配置");
    this.speech = {id: data.utterance_id, seq: 0, received: 0, played: 0, next: 0,
      limit: data.buffer_ms, start: performance.now(), underruns: 0, eof: false};
    this.speech.pending = [];
    this.update(this.speech);
  }
  text() {
    if (!this.speech) return;
    this.speech.firstText ??= Math.round(performance.now() - this.speech.start);
    this.update(this.speech);
  }
  finish(data) {
    const s = this.speech;
    if (!s || s.id !== data.utterance_id) return;
    if (data.samples !== s.received) throw Error("音频结束计数不一致");
    s.eof = true;
    this.progress(s);
  }
  progress(s) {
    if (this.speech !== s) return;
    s.started ||= s.played > 0 || (s.received > 0 && this.context.state === "running"
      && this.context.currentTime >= s.playAt);
    this.send("playback_progress", {utterance_id: s.id, played_samples: s.played,
      started: !!s.started, ended: s.eof && s.played === s.received, underruns: s.underruns});
    this.update(s);
  }
  pollStart() {
    const s = this.speech;
    if (s && !s.started && s.received && this.context.currentTime >= s.playAt) this.progress(s);
  }
  hold(held) {
    const s = this.speech;
    if (!s) return;
    this.pollStart();
    if (s.started) return; // Never pause already-playing speech on raw VAD.
    if (held && !s.held) {
      s.held = true;
      for (const node of this.nodes) {
        node.onended = null; node.stop(); node.disconnect();
        s.pending.push({buffer: node.buffer, offset: node.fdOffset, count: node.fdCount});
      }
      this.nodes.clear();
      s.pending.sort((a, b) => a.offset - b.offset);
      s.playAt = Infinity;
    } else if (!held && s.held) {
      s.held = false;
      s.next = this.context.currentTime + .08;
      s.playAt = s.next;
      for (const chunk of s.pending) this.schedule(s, chunk.buffer, chunk.offset, chunk.count);
      s.pending = [];
    }
  }
  schedule(s, buffer, offset, count) {
    const node = this.context.createBufferSource();
    node.buffer = buffer; node.connect(this.destination); this.nodes.add(node);
    node.fdStart = s.next; node.fdOffset = offset; node.fdCount = count;
    node.onended = () => {
      this.nodes.delete(node); node.disconnect();
      if (this.speech !== s) return;
      s.played = Math.max(s.played, offset + count);
      this.progress(s);
    };
    node.start(s.next); s.next += count / s.rate;
  }
  packet(raw) {
    if (raw.byteLength < 18 || (raw.byteLength - 16) % 2) throw Error("无效 PCM 包");
    const view = new DataView(raw);
    if (view.getUint32(0, true) !== 0x31534446) throw Error("未知音频协议"); // FDS1
    const s = this.speech, id = view.getUint32(4, true), seq = view.getUint32(8, true);
    if (!s || id !== s.id) return;
    const rate = view.getUint32(12, true);
    if (seq !== s.seq++ || (s.rate && rate !== s.rate)) throw Error("音频顺序或采样率错误");
    if (rate < 8000 || rate > 96000) throw Error("无效音频采样率");
    if (s.eof) throw Error("音频结束后收到额外数据");
    s.rate = rate;
    const count = (raw.byteLength - 16) / 2;
    s.received += count;
    if ((s.received - s.played) / rate * 1000 > s.limit + 1) throw Error("播放缓冲超限");
    const buffer = this.context.createBuffer(1, count, rate), values = buffer.getChannelData(0);
    for (let i = 0; i < count; i++) values[i] = view.getInt16(16 + 2 * i, true) / 32768;
    if (s.firstAudio === undefined) {
      s.firstAudio = Math.round(performance.now() - s.start);
      s.next = this.context.currentTime + 0.08;
      s.playAt = s.next;
      s.scheduledLeadMs = (s.next - this.context.currentTime) * 1000;
    } else if (!s.held && s.next < this.context.currentTime) {
      s.underruns++;
      s.next = this.context.currentTime + 0.08;
    }
    if (s.held) {
      s.pending.push({buffer, offset: s.received - count, count});
      s.playAt = Infinity;
    } else this.schedule(s, buffer, s.received - count, count);
    this.progress(s);
  }
}
