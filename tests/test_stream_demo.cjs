// Node-only playback/worklet contract tests. These do NOT replace listening in
// a browser: they exercise the production JS with mocked WebAudio/WebSocket.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const root = path.resolve(__dirname, "../src/static");
const sent = [], sources = [];
const elements = new Map();
class AudioContext {
  constructor() { this.currentTime = 0; this.state = "running"; }
  createBuffer(channels, count, rate) {
    const data = new Float32Array(count);
    return {getChannelData: () => data, duration: count / rate};
  }
  createBufferSource() {
    const node = {connect() {}, start(time) { this.at = time; }, stop() { this.stopped = true; }};
    sources.push(node); return node;
  }
  async close() { this.state = "closed"; }
}
const scope = vm.createContext({
  document: {getElementById(id) {
    if (!elements.has(id)) elements.set(id, {textContent: ""});
    return elements.get(id);
  }},
  window: {addEventListener() {}}, AudioContext, WebSocket: {OPEN: 1},
  performance: {now: () => 100}, location: {protocol: "http:", host: "localhost"},
  URL, Float32Array, DataView, sent,
});
const source = fs.readFileSync(path.join(root, "demo.js"), "utf8")
  .replaceAll("import.meta.url", '"http://localhost/demo/demo.js"');
vm.runInContext(source, scope);
vm.runInContext(`context = new AudioContext(); ws = {
  readyState: 1, send(value) { sent.push(JSON.parse(value)); }, close() {}
};`, scope);
function control(event, data) { scope.control({event, data}); }
function packet(id, seq, samples = 960) {
  const raw = new ArrayBuffer(16 + samples * 2), view = new DataView(raw);
  view.setUint32(0, 0x31534446, true);
  view.setUint32(4, id, true); view.setUint32(8, seq, true); view.setUint32(12, 24000, true);
  for (let i = 0; i < samples; i++) view.setInt16(16 + i * 2, 8192, true);
  return raw;
}
control("speech_start", {utterance_id: 1, buffer_ms: 600});
scope.audioPacket(packet(1, 0)); scope.audioPacket(packet(1, 1));
assert.equal(sources[0].at, .08);
assert.ok(Math.abs(sources[1].at - .12) < 1e-9);
assert.equal(sources[0].buffer.getChannelData()[0], .25);
assert.equal(sent.at(-1).data.played_samples, 0, "arrival is not playback acknowledgement");
sources[0].onended();
assert.equal(sent.at(-1).data.played_samples, 960);
control("speech_audio_end", {utterance_id: 1, samples: 1920});
assert.equal(sent.at(-1).data.ended, false);
sources[1].onended();
assert.equal(sent.at(-1).data.ended, true);
control("speech_start", {utterance_id: 2, buffer_ms: 600});
scope.audioPacket(packet(2, 0));
control("speech_cancelled", {utterance_id: 2, reason: "shot_interrupt"});
assert.equal(sources.at(-1).stopped, true);
const before = sources.length;
scope.audioPacket(packet(2, 1));
assert.equal(sources.length, before, "late old packets must not resurrect playback");
control("speech_start", {utterance_id: 3, buffer_ms: 600});
assert.throws(() => scope.audioPacket(packet(3, 1)), /顺序/);
control("speech_start", {utterance_id: 4, buffer_ms: 600});
for (let i = 0; i < 15; i++) scope.audioPacket(packet(4, i));
assert.throws(() => scope.audioPacket(packet(4, 15)), /超限/);
console.log("playback: ordering, 80 ms startup, acknowledgements, cancellation, buffer cap PASS");

for (const rate of [16000, 44100, 48000]) {
  const frames = [];
  let Processor;
  const worklet = vm.createContext({
    sampleRate: rate, Float32Array,
    AudioWorkletProcessor: class { constructor() { this.port = {postMessage(buffer) { frames.push(new Float32Array(buffer)); }}; } },
    registerProcessor(name, value) { Processor = value; },
  });
  vm.runInContext(fs.readFileSync(path.join(root, "mic-worklet.js"), "utf8"), worklet);
  const processor = new Processor();
  let total = 0;
  for (const size of [128, 127, 513, 1024, 255, 4096, 11025]) {
    processor.process([[new Float32Array(size).fill(.5)]]); total += size;
  }
  assert.equal(frames.length, Math.floor(total * 16000 / rate / 256));
  assert.ok(frames.every(frame => frame.length === 256 && frame.every(x => x === .5)));
}
console.log("microphone: 16k/44.1k/48k input, fractional carry, fixed 256-sample frames PASS");
