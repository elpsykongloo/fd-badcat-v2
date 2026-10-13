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
    // Lead after a pre-playback hold is released; undefined keeps startupMs.
    this.resumeLeadMs = undefined;
    // AudioContext time until which earlier output actually sounded (echo gate).
    this.soundedUntil = -Infinity;
  }
  setResumeLead(ms) {
    this.resumeLeadMs = Number.isFinite(ms) && ms >= 0 && ms <= 1000 ? ms : undefined;
  }
  // True while output started in the render clock is playing, or ended less
  // than windowSec ago. Scheduled-but-unstarted audio never counts.
  recentOutput(windowSec) {
    const now = this.context.currentTime;
    let until = this.soundedUntil;
    for (const node of this.nodes) if (node.fdStart !== undefined && node.fdStart <= now) until = Math.max(until, now);
    return now < until + windowSec;
  }
  stopNodes() {
    const now = this.context.currentTime;
    for (const node of this.nodes) {
      if (node.fdStart !== undefined && node.fdStart <= now) this.soundedUntil = Math.max(this.soundedUntil, now);
      node.onended = null; node.stop(); node.disconnect();
    }
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
    this.stopNodes();
    this.nodes.clear();
    if (stopped) this.send("playback_stopped", stopped);
  }
  start(data) {
    this.cancel();
    const startup = data.startup_ms ?? 80;
    if (!Number.isInteger(data.utterance_id) || !Number.isFinite(data.buffer_ms)
        || data.buffer_ms <= 0 || data.buffer_ms > 2000
        || !Number.isFinite(startup) || startup < 0 || startup > 1000) throw Error("无效播放配置");
    this.speech = {id: data.utterance_id, seq: 0, received: 0, played: 0, next: 0,
      limit: data.buffer_ms, startupMs: startup, start: performance.now(),
      underruns: 0, underrunMs: 0, maxUnderrunMs: 0, eof: false};
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
      for (const node of this.nodes) s.pending.push({buffer: node.buffer, offset: node.fdOffset, count: node.fdCount});
      this.stopNodes();
      this.nodes.clear();
      s.pending.sort((a, b) => a.offset - b.offset);
      s.playAt = Infinity;
    } else if (!held && s.held) {
      s.held = false;
      // Buffered audio is already local: re-wait only resumeLeadMs (never more than startup).
      const lead = Math.min(s.startupMs, this.resumeLeadMs ?? s.startupMs);
      s.next = this.context.currentTime + lead / 1000;
      s.playAt = s.next;
      s.scheduledLeadMs = lead;
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
      this.soundedUntil = Math.max(this.soundedUntil, node.fdStart + count / s.rate);
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
      s.next = this.context.currentTime + s.startupMs / 1000;
      s.playAt = s.next;
      s.scheduledLeadMs = (s.next - this.context.currentTime) * 1000;
    } else if (!s.held && s.next < this.context.currentTime) {
      s.underruns++;
      const now = this.context.currentTime, gap = (now - s.next + .08) * 1000;
      s.underrunMs += gap;
      s.maxUnderrunMs = Math.max(s.maxUnderrunMs, gap);
      s.lastUnderrun = {underruns: s.underruns, gap_ms: gap,
        late_ms: (now - s.next) * 1000, audio_context_ms: now * 1000,
        previous_end_ms: s.next * 1000, packet_seq: seq,
        sample_offset: s.received - count,
        arrival_interval_ms: performance.now() - s.lastArrival};
      // Recovery remains 80 ms; the larger startup lead is not added to a gap.
      s.next = this.context.currentTime + 0.08;
    }
    s.lastArrival = performance.now();
    if (s.held) {
      s.pending.push({buffer, offset: s.received - count, count});
      s.playAt = Infinity;
    } else this.schedule(s, buffer, s.received - count, count);
    this.progress(s);
  }
}

/*
 * client-hold-v1: local pre-playback hold (HOLD_PLAN.md 2.3). Python mirrors
 * (e.g. scripts/bench_guarded_latency.py) must reproduce exactly this:
 *
 * Input. One call per uploaded 16 kHz frame of 256 samples (16 ms), AFTER the
 *   mute zeroing and BEFORE the WebSocket send. Microphone channel only: in
 *   FDM1 (pcm16.ref.v1) mic is int16 at byte 16+4*i and the render reference
 *   at 18+4*i, both /32768; legacy float32 frames have no reference (ref=0).
 *   rms = sqrt(mean(x^2)) over the 256 samples; db(x) = 20*log10(max(x, 1e-9)).
 *   Not the 80 ms throttled UI meter. Muted frames are skipped entirely.
 * Echo gate (echo = true blocks onset counting and freezes the floor):
 *   echo = db(ref_rms) > refGateDb
 *       || frames since the last such reference frame <= refHangFrames
 *       || output started on the render clock is playing, or ended less than
 *          outputLatency (context.outputLatency || baseLatency || 0) + echoTailMs ago.
 * Detector, per frame, with floor_db starting at floorInitDb:
 *   threshold_db = max(floor_db + marginDb, absMinDb)        (uses the old floor)
 *   above = db(mic_rms) > threshold_db
 *   if not echo:
 *     rate = fallRate if db < floor_db
 *            else riseVoicedRate if above and run < sustainFrames   (run = previous frame's)
 *            else riseRate      (a step in steady noise is learned after ~2 s)
 *     floor_db = clamp(floor_db + (db - floor_db) * rate, floorMinDb, floorMaxDb)
 *   run = run + 1 if (above and not echo) else 0
 *   onset = run >= onsetFrames     (stays true while the run continues)
 * Action. On onset, only if the current utterance has received PCM, has not
 *   started on the render clock, is not held (by anyone) and was never locally
 *   held before (at most one local hold per utterance): player.hold(true), then
 *   a release timer of max(timerMinMs, timerRttFactor * rtt + timerPadMs) where
 *   rtt is the latest telemetry RTT or fallbackRttMs. Timer expiry releases with
 *   lead = min(startupMs, resumeLeadMs). Any server speech_hold for that
 *   utterance clears the timer and the server's held value wins. Started speech
 *   is never paused (SpeechPlayer.hold invariant). "shadow" mode evaluates the
 *   same rule and only reports what it would have done.
 * Defaults. marginDb 15 / absMinDb -45 / onsetFrames 2 were calibrated offline on
 *   the lat5 real episodes (exp/lat5/localhold/eval_localhold.py, replaying the
 *   bench feed through the Python twin): with 12 / -50 the real recordings
 *   (noise floor -45..-50 dB, floor still unlearned after an item's leading
 *   digital silence) mis-held 15/91 replies on -49..-34 dB non-speech sounds;
 *   15 / -45 leaves 4/91, keeps every continuation the live run caught, and
 *   costs continuation detection lag P50/P90 30/71 ms (was 30/42). Sounds under
 *   -45 dBFS never count as an onset; speech after AGC sits around -25 dBFS.
 */
export const LOCAL_HOLD_DEFAULTS = Object.freeze({
  floorInitDb: -60, floorMinDb: -80, floorMaxDb: -35,
  fallRate: 0.2, riseRate: 0.02, riseVoicedRate: 0.0005,
  marginDb: 15, absMinDb: -45, onsetFrames: 2, sustainFrames: 125,
  refGateDb: -60, refHangFrames: 10, echoTailMs: 150,
  timerMinMs: 300, timerRttFactor: 2, timerPadMs: 200, fallbackRttMs: 150,
});
const toDb = value => 20 * Math.log10(Math.max(value, 1e-9));
const fromDb = value => Math.pow(10, value / 20);
export class OnsetDetector {
  constructor(params = {}) {
    this.p = {...LOCAL_HOLD_DEFAULTS};
    for (const key of Object.keys(LOCAL_HOLD_DEFAULTS)) {
      if (Number.isFinite(params?.[key])) this.p[key] = params[key];
    }
    this.floorDb = this.p.floorInitDb;
    this.run = 0;
    this.sinceRef = Infinity;
  }
  frame(micRms, refRms = 0, outputRecent = false) {
    const p = this.p, db = toDb(micRms), refDb = toDb(refRms);
    this.sinceRef = refDb > p.refGateDb ? 0 : this.sinceRef + 1;
    const echo = outputRecent || this.sinceRef <= p.refHangFrames;
    const thresholdDb = Math.max(this.floorDb + p.marginDb, p.absMinDb), floorDb = this.floorDb;
    const above = db > thresholdDb;
    if (!echo) {
      const rate = db < this.floorDb ? p.fallRate
        : above && this.run < p.sustainFrames ? p.riseVoicedRate : p.riseRate;
      this.floorDb = Math.min(p.floorMaxDb, Math.max(p.floorMinDb, this.floorDb + (db - this.floorDb) * rate));
    }
    this.run = above && !echo ? this.run + 1 : 0;
    return {onset: this.run >= p.onsetFrames, echo, db, refDb, floorDb, thresholdDb};
  }
}
export class LocalHold {
  constructor(player, {mode = "off", params = {}, emit = () => {}, rtt = () => undefined,
    outputLatency = () => 0, setTimer = (fn, ms) => setTimeout(fn, ms),
    clearTimer = id => clearTimeout(id), now = () => performance.now()} = {}) {
    this.player = player;
    this.mode = mode === true || mode === "on" ? "on" : mode === "shadow" ? "shadow" : "off";
    this.detector = new OnsetDetector(params);
    Object.assign(this, {emit, rtt, outputLatency, setTimer, clearTimer, now});
  }
  report(s, action, data) {
    this.emit("client_hold", {utterance_id: s.id, action, ...data});
  }
  frame(micRms, refRms = 0) {
    const p = this.detector.p, player = this.player;
    const latency = Math.max(0, Number(this.outputLatency()) || 0);
    const r = this.detector.frame(micRms, refRms, player.recentOutput(latency + p.echoTailMs / 1000));
    if (this.mode === "off" || !r.onset) return r;
    const s = player.speech;
    if (!s || !s.received || s.held || s.localHold) return r;
    player.pollStart();
    if (s.started) return r;
    const rtt = this.rtt(), rttMs = Number.isFinite(rtt) && rtt >= 0 ? rtt : p.fallbackRttMs;
    const timerMs = Math.max(p.timerMinMs, p.timerRttFactor * rttMs + p.timerPadMs);
    const energy = {frame_rms: micRms, ref_rms: refRms, noise_floor_rms: fromDb(r.floorDb),
      threshold_rms: fromDb(r.thresholdDb), timer_ms: timerMs, rtt_ms: rttMs,
      output_latency_ms: latency * 1000};
    if (this.mode === "shadow") {
      s.localHold = {at: this.now(), done: true, shadow: true};
      this.report(s, "shadow", energy);
      return r;
    }
    player.hold(true);
    if (!s.held) return r;
    const hold = {at: this.now(), timerMs, done: false};
    s.localHold = hold;
    hold.timer = this.setTimer(() => this.expire(s, hold), timerMs);
    this.report(s, "hold", energy);
    return r;
  }
  expire(s, hold) {
    if (hold.done) return;
    hold.done = true;
    if (this.player.speech !== s || !s.held) return;
    this.player.hold(false);
    this.report(s, "timeout", {hold_ms: this.now() - hold.at, timer_ms: hold.timerMs,
      scheduled_lead_ms: s.scheduledLeadMs});
  }
  // Server speech_hold for the current utterance: the server owns the decision.
  server(held) {
    const s = this.player.speech, hold = s?.localHold;
    if (hold && !hold.done) {
      hold.done = true;
      this.clearTimer(hold.timer);
      this.report(s, "release", {hold_ms: this.now() - hold.at, timer_ms: hold.timerMs,
        server_held: held ? 1 : 0});
    }
    this.player.hold(held);
  }
  // Before the player's utterance is replaced or cancelled.
  cancel() {
    const hold = this.player.speech?.localHold;
    if (hold && !hold.done) {
      hold.done = true;
      this.clearTimer(hold.timer);
    }
  }
}
