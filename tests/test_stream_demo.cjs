// Node-only tests of the production player/worklet; no GPU or browser required.
// Real browser coverage lives in scripts/check_web_demo.py.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const root = path.resolve(__dirname, "../src/static");
const sent = [], sources = [];
class AudioContext {
  constructor() { this.currentTime = 0; this.state = "running"; }
  createBuffer(channels, count, rate) {
    const data = new Float32Array(count);
    return {getChannelData: () => data, duration: count / rate};
  }
  createBufferSource() {
    const node = {connect() {}, disconnect() { this.disconnected = true; },
      start(time) { this.at = time; }, stop() { this.stopped = true; }};
    sources.push(node); return node;
  }
}
const scope = vm.createContext({performance: {now: () => 100}, Float32Array, DataView});
const source = fs.readFileSync(path.join(root, "speech-player.js"), "utf8").replace("export class", "class");
vm.runInContext(source, scope);
const SpeechPlayer = vm.runInContext("SpeechPlayer", scope);
const context = new AudioContext();
const player = new SpeechPlayer(context, (event, data) => sent.push({event, data}));
function start(id) { player.start({utterance_id: id, buffer_ms: 600}); }
function packet(id, seq, samples = 960) {
  const raw = new ArrayBuffer(16 + samples * 2), view = new DataView(raw);
  view.setUint32(0, 0x31534446, true);
  view.setUint32(4, id, true); view.setUint32(8, seq, true); view.setUint32(12, 24000, true);
  for (let i = 0; i < samples; i++) view.setInt16(16 + i * 2, 8192, true);
  return raw;
}
start(1);
player.packet(packet(1, 0)); player.packet(packet(1, 1));
assert.equal(sources[0].at, .08);
assert.ok(Math.abs(sources[1].at - .12) < 1e-9);
assert.equal(sources[0].buffer.getChannelData()[0], .25);
assert.equal(sent.at(-1).data.played_samples, 0, "arrival is not playback acknowledgement");
assert.equal(sent.at(-1).data.started, false, "queued audio has not started");
context.currentTime = .081;
player.pollStart();
assert.equal(sent.at(-1).data.started, true);
assert.equal(sent.at(-1).data.played_samples, 0, "start is distinct from completed samples");
context.currentTime = 0;
sources[0].onended();
assert.equal(sent.at(-1).data.played_samples, 960);
assert.equal(sources[0].disconnected, true);
player.finish({utterance_id: 1, samples: 1920});
assert.equal(sent.at(-1).data.ended, false);
sources[1].onended();
assert.equal(sent.at(-1).data.ended, true);
assert.throws(() => player.packet(packet(1, 2)), /额外/);
start(2);
player.packet(packet(2, 0));
player.cancel();
assert.equal(sources.at(-1).stopped, true);
assert.equal(sources.at(-1).disconnected, true);
const before = sources.length;
player.packet(packet(2, 1));
assert.equal(sources.length, before, "late old packets must not resurrect playback");
start(3);
assert.throws(() => player.packet(packet(3, 1)), /顺序/);
start(4);
for (let i = 0; i < 15; i++) player.packet(packet(4, i));
assert.throws(() => player.packet(packet(4, 15)), /超限/);
start(5);
assert.throws(() => player.finish({utterance_id: 5, samples: 1}), /计数/);
const badRate = packet(5, 0);
new DataView(badRate).setUint32(12, 0, true);
assert.throws(() => player.packet(badRate), /采样率/);
assert.throws(() => player.start({utterance_id: 6, buffer_ms: Infinity}), /配置/);
start(7);
player.packet(packet(7, 0));
context.currentTime = .5;
player.packet(packet(7, 1));
assert.equal(player.speech.underruns, 1);
assert.equal(sources.at(-1).at, .58);
console.log("playback: 80 ms startup, ordering, native rate, acknowledgements, cancel, cap, EOF, underruns PASS");

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

for (const rate of [16000, 44100, 48000]) {
  const packets = [];
  let Processor;
  const worklet = vm.createContext({sampleRate: rate, Float32Array, ArrayBuffer, DataView,
    AudioWorkletProcessor: class { constructor() { this.port = {postMessage(raw) { packets.push(raw); }}; } },
    registerProcessor(name, value) { Processor = value; }});
  vm.runInContext(fs.readFileSync(path.join(root, "mic-worklet.js"), "utf8"), worklet);
  const processor = new Processor({processorOptions: {inputReference: true}});
  for (let i = 0; i < 24; i++) processor.process([[new Float32Array(128).fill(.5)], [new Float32Array(128).fill(-.25)]]);
  assert.equal(packets.length, Math.floor(3072 * 16000 / rate / 256));
  packets.forEach((raw, index) => {
    assert.equal(raw.byteLength, 1040);
    const d = new DataView(raw);
    assert.equal(d.getUint32(0, true), 0x314d4446);
    assert.equal(d.getUint32(4, true), index);
    assert.equal(d.getUint32(8, true), 256); assert.equal(d.getUint32(12, true), 16000);
    for (let j = 0; j < 256; j++) {
      assert.equal(d.getInt16(16 + j * 4, true), 16384);
      assert.equal(d.getInt16(18 + j * 4, true), -8192);
    }
  });
}
context.currentTime = 0;
start(8); player.packet(packet(8, 0)); player.packet(packet(8, 1));
player.hold(true);
context.currentTime = .2;
player.packet(packet(8, 2)); player.pollStart();
assert.ok(!player.speech.started && player.speech.held);
assert.equal(player.nodes.size, 0);
player.hold(false);
assert.equal(player.nodes.size, 3);
context.currentTime = .3;
player.cancel();
const stopped = sent.filter(x => x.event === "playback_stopped").at(-1).data;
assert.equal(stopped.utterance_id, 8);
assert.ok(stopped.played_samples >= 479 && stopped.played_samples <= 480,
  "future queued nodes cannot count as played; partial node uses render clock");
console.log("guarded playback/input: paired clock/rates/sequence, preplay hold/release, partial stop ACK PASS");

vm.runInContext(fs.readFileSync(path.join(root, "demo-telemetry.js"), "utf8").replace("export class", "class"), scope);
const DemoTelemetry = vm.runInContext("DemoTelemetry", scope);
const measured = [], rtts = [];
let now = 100;
const telemetry = new DemoTelemetry((event, data) => measured.push({event, data}), context,
  () => 1024, ms => rtts.push(ms), () => now);
telemetry.tick();
assert.equal(measured.length, 0, "old backend must not receive telemetry");
telemetry.enabled = true;
telemetry.tick(); now += 80;
telemetry.pong({seq: 9}); assert.equal(rtts.length, 0);
telemetry.pong({seq: 1}); assert.equal(rtts[0], 80);
assert.equal(measured.at(-1).data.rtt_ms, 80);
telemetry.tick(); assert.equal(measured.length, 2, "ping interval bound");
const diagnosticSpeech = {id: 99, firstAudio: 449, scheduledLeadMs: 80,
  received: 960, played: 0, underruns: 0, eof: false};
telemetry.playback(diagnosticSpeech); telemetry.playback(diagnosticSpeech);
assert.equal(measured.filter(m => m.data.kind === "first_audio").length, 1);
diagnosticSpeech.eof = true; diagnosticSpeech.played = 960;
telemetry.playback(diagnosticSpeech); telemetry.playback(diagnosticSpeech);
assert.equal(measured.filter(m => m.data.kind === "playback_end").length, 1);
assert.equal(measured.at(-1).data.played_samples, 960);
assert.equal(measured.at(-1).data.upload_buffer_bytes, 1024);
console.log("telemetry: opt-in, same-clock RTT, bounded pings, first/played milestones exactly once PASS");
