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
const source = fs.readFileSync(path.join(root, "speech-player.js"), "utf8").replace(/^export /gm, "");
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

context.currentTime = 0;
player.start({utterance_id: 100, buffer_ms: 600, startup_ms: 350});
player.packet(packet(100, 0));
assert.equal(sources.at(-1).at, .35);
player.hold(true); context.currentTime = .1; player.hold(false);
assert.ok(Math.abs(sources.at(-1).at - .45) < 1e-9);
context.currentTime = .8;
player.packet(packet(100, 1));
assert.ok(Math.abs(sources.at(-1).at - .88) < 1e-9, "recovery stays 80 ms");
assert.ok(Math.abs(player.speech.underrunMs - 390) < 1e-8);
assert.equal(player.speech.lastUnderrun.sample_offset, 960);
assert.equal(player.speech.lastUnderrun.packet_seq, 1);
telemetry.playback(player.speech); telemetry.playback(player.speech);
assert.equal(measured.filter(m => m.data.kind === "underrun").length, 1);
telemetry.snapshot("cancel", player.speech);
assert.ok(Math.abs(measured.at(-1).data.underrun_ms - 390) < 1e-8);
player.cancel(); player.packet(packet(100, 2));
assert.equal(player.speech, null);
for (const startup_ms of [-1, 1001, Infinity, "350"]) {
  assert.throws(() => player.start({utterance_id: 101, buffer_ms: 600, startup_ms}), /配置/);
}
console.log("continuity: configured startup/hold, unchanged recovery, same-clock gap telemetry, cancellation PASS");

// client-hold-v1: local pre-playback hold, resume lead, echo gate, server handover.
const LocalHold = vm.runInContext("LocalHold", scope);
const OnsetDetector = vm.runInContext("OnsetDetector", scope);
function holdRig({mode = "on", resumeLead = 80, rtt = 100, latency = 0.02, params} = {}) {
  const ctx = new AudioContext(), events = [], reports = [], timers = [], cleared = [];
  const p = new SpeechPlayer(ctx, (event, data) => events.push({event, data}));
  p.setResumeLead(resumeLead);
  const local = new LocalHold(p, {mode, params, emit: (kind, data) => reports.push({kind, ...data}),
    rtt: () => rtt, outputLatency: () => latency,
    setTimer: (fn, ms) => { timers.push({fn, ms}); return timers.length; },
    clearTimer: id => cleared.push(id), now: () => ctx.currentTime * 1000});
  const feed = (n, mic, ref = 0) => { for (let i = 0; i < n; i++) local.frame(mic, ref); };
  return {ctx, p, local, events, reports, timers, cleared, feed};
}
{
  const r = holdRig();
  r.feed(40, 1e-4);                          // quiet room: floor settles low
  r.p.start({utterance_id: 200, buffer_ms: 600, startup_ms: 300});
  r.p.packet(packet(200, 0)); r.p.packet(packet(200, 1));
  assert.ok(Math.abs(sources.at(-1).fdStart - .34) < 1e-9);
  r.ctx.currentTime = .1;
  r.feed(1, .05);
  assert.ok(!r.p.speech.held, "one loud frame is not an onset (onsetFrames=2)");
  r.feed(1, .05);
  assert.ok(r.p.speech.held && !r.p.speech.started && r.p.nodes.size === 0, "onset holds unstarted speech");
  const hold = r.reports.at(-1);
  assert.equal(hold.action, "hold"); assert.equal(hold.utterance_id, 200);
  assert.equal(hold.frame_rms, .05); assert.equal(hold.timer_ms, 400, "max(300, 2*RTT+200)");
  assert.ok(hold.threshold_rms < .05 && hold.noise_floor_rms < hold.threshold_rms);
  assert.equal(r.timers.length, 1);
  r.p.packet(packet(200, 2));                // arrivals while held stay pending
  assert.equal(r.p.nodes.size, 0);
  r.ctx.currentTime = .5;
  r.timers[0].fn();
  assert.ok(!r.p.speech.held && r.p.nodes.size === 3);
  assert.ok(Math.abs(sources.at(-3).fdStart - .58) < 1e-9, "release re-waits resumeLead 80, not startup 300");
  assert.equal(r.reports.at(-1).action, "timeout"); assert.equal(r.reports.at(-1).scheduled_lead_ms, 80);
  assert.ok(Math.abs(r.reports.at(-1).hold_ms - 400) < 1e-6);
  r.feed(5, .05);
  assert.ok(!r.p.speech.held && r.reports.length === 2, "at most one local hold per utterance");
  r.timers[0].fn();
  assert.equal(r.reports.length, 2, "a stale timer is inert");
}
{
  const r = holdRig({resumeLead: 500});
  r.feed(40, 1e-4);
  r.p.start({utterance_id: 201, buffer_ms: 600, startup_ms: 300});
  r.p.packet(packet(201, 0)); r.feed(2, .05);
  r.ctx.currentTime = .4; r.timers[0].fn();
  assert.ok(Math.abs(sources.at(-1).fdStart - .7) < 1e-9, "resume lead never exceeds startup");
  const legacy = new SpeechPlayer(new AudioContext(), () => {});
  legacy.setResumeLead(undefined); assert.equal(legacy.resumeLeadMs, undefined);
  legacy.setResumeLead(-1); assert.equal(legacy.resumeLeadMs, undefined);
}
{
  const r = holdRig();
  r.feed(40, 1e-4);
  r.p.start({utterance_id: 202, buffer_ms: 600, startup_ms: 300});
  r.p.packet(packet(202, 0));
  r.ctx.currentTime = .31;                   // already started on the render clock
  r.feed(10, .05);
  assert.ok(!r.p.speech.held && !sources.at(-1).stopped, "started speech is never paused");
  assert.equal(r.reports.length, 0);
  r.p.hold(true);
  assert.ok(!r.p.speech.held, "player invariant unchanged");
  r.ctx.currentTime = .32;
  r.p.start({utterance_id: 203, buffer_ms: 600, startup_ms: 300});   // previous cut at .32
  r.p.packet(packet(203, 0));
  r.ctx.currentTime = .32 + .02 + .149;      // inside outputLatency + 150 ms of local output
  r.feed(10, .05);
  assert.ok(!r.p.speech.held && r.reports.length === 0, "recent local output gates onset");
  r.ctx.currentTime = .32 + .02 + .151;
  r.feed(1, 1e-4); r.feed(2, .05);
  assert.ok(r.p.speech.held, "gate opens after the echo tail");
}
{
  const r = holdRig();
  r.feed(40, 1e-4);
  r.p.start({utterance_id: 204, buffer_ms: 600, startup_ms: 300});
  r.p.packet(packet(204, 0));
  r.feed(20, .05, .01);                      // render reference carries energy: echo
  assert.ok(!r.p.speech.held && r.reports.length === 0, "reference energy never triggers");
  r.feed(10, .05, 0);                        // 160 ms reference hang-over
  assert.ok(!r.p.speech.held);
  r.feed(2, .05, 0);
  assert.ok(r.p.speech.held, "after the hang-over a real onset still holds");
}
{
  const r = holdRig();
  r.feed(40, 1e-4);
  r.p.start({utterance_id: 205, buffer_ms: 600, startup_ms: 300});
  r.p.packet(packet(205, 0)); r.feed(2, .05);
  assert.ok(r.p.speech.held);
  r.local.server(true);                      // server VAD confirms: server owns the hold
  assert.deepEqual(r.cleared, [1]);
  assert.equal(r.reports.at(-1).action, "release"); assert.equal(r.reports.at(-1).server_held, 1);
  assert.ok(r.p.speech.held);
  r.ctx.currentTime = .45; r.timers[0].fn();
  assert.ok(r.p.speech.held && r.reports.length === 2, "cleared timer cannot release a server hold");
  r.local.server(false);
  assert.ok(!r.p.speech.held && Math.abs(sources.at(-1).fdStart - .53) < 1e-9);
  r.local.server(true); r.local.server(false);   // later server holds do not report
  assert.equal(r.reports.length, 2);
  r.p.start({utterance_id: 206, buffer_ms: 600, startup_ms: 300});
  r.p.packet(packet(206, 0)); r.ctx.currentTime = .46; r.feed(1, 1e-4); r.feed(2, .05);
  assert.ok(r.p.speech.held);
  r.local.cancel(); r.p.cancel();
  assert.deepEqual(r.cleared, [1, 2], "cancel clears the pending timer");
}
{
  const r = holdRig({mode: "shadow"});
  r.feed(40, 1e-4);
  r.p.start({utterance_id: 207, buffer_ms: 600, startup_ms: 300});
  r.p.packet(packet(207, 0)); r.feed(6, .05);
  assert.ok(!r.p.speech.held && r.timers.length === 0);
  assert.equal(r.reports.length, 1); assert.equal(r.reports[0].action, "shadow");
  const off = holdRig({mode: "off"});
  off.p.start({utterance_id: 208, buffer_ms: 600, startup_ms: 300});
  off.p.packet(packet(208, 0)); off.feed(6, .5);
  assert.ok(!off.p.speech.held && off.reports.length === 0, "default off keeps old behavior");
  assert.equal(new LocalHold(off.p, {mode: undefined}).mode, "off");
  assert.equal(new LocalHold(off.p, {mode: true}).mode, "on");
}
{
  const d = new OnsetDetector({marginDb: 6, onsetFrames: 1, bogus: 1, absMinDb: "x"});
  assert.equal(d.p.marginDb, 6); assert.equal(d.p.absMinDb, -45); assert.equal(d.p.bogus, undefined);
  for (let i = 0; i < 124; i++) d.frame(.01);     // a step in steady noise looks like voice first
  assert.ok(d.floorDb < -58, String(d.floorDb));
  for (let i = 0; i < 300; i++) d.frame(.01);     // after sustainFrames the floor learns it
  assert.ok(d.floorDb > -42 && d.floorDb <= -35, String(d.floorDb));
  assert.equal(d.frame(.01).onset, false, "adapted noise is not an onset");
  assert.equal(d.frame(.1).onset, true);
  const v = new OnsetDetector();
  for (let k = 0; k < 10; k++) {                   // syllables with dips: floor barely moves
    for (let i = 0; i < 30; i++) v.frame(.05);
    for (let i = 0; i < 3; i++) v.frame(1e-4);
  }
  assert.ok(v.floorDb < -55, String(v.floorDb));
  const z = new OnsetDetector();
  for (let i = 0; i < 100; i++) z.frame(0);
  assert.equal(z.floorDb, -80, "digital silence clamps at floorMinDb");
}
{
  const r = holdRig({rtt: null});
  r.feed(40, 1e-4);
  r.p.start({utterance_id: 209, buffer_ms: 600, startup_ms: 300});
  r.p.packet(packet(209, 0)); r.feed(2, .05);
  assert.equal(r.timers[0].ms, 500, "unknown RTT uses fallbackRttMs 150");
}
console.log("client hold: local onset hold, timed release with resume lead, started never paused, ref/output echo gate, server handover PASS");
