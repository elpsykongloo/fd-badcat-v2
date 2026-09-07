const $ = id => document.getElementById(id);
let ws, context, mic, capture, media, speech, connecting = false;
const nodes = new Set();
function send(event, data) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({event, data}));
}
function clearAudio() {
  speech = null;
  for (const node of nodes) { node.onended = null; node.stop(); }
  nodes.clear();
}
async function stop() {
  clearAudio();
  const connection = ws, audioContext = context;
  ws = null; context = null;
  if (connection) {
    connection.onclose = connection.onmessage = connection.onerror = null;
    connection.close();
  }
  if (capture) capture.port.onmessage = null;
  capture?.disconnect(); mic?.disconnect();
  media?.getTracks().forEach(track => track.stop());
  capture = mic = media = null;
  if (audioContext && audioContext.state !== "closed") await audioContext.close();
  $("start").disabled = false; $("stop").disabled = true;
}
function progress(s) {
  if (speech !== s) return;
  send("playback_progress", {utterance_id: s.id, played_samples: s.played,
    ended: s.eof && s.played === s.received, underruns: s.underruns});
  $("metrics").textContent = `首文本 ${s.firstText ?? "—"} ms · 首音频到达 ${s.firstAudio ?? "—"} ms\n` +
    `未播放缓冲 ${Math.round((s.received - s.played) / (s.rate || 24000) * 1000)} ms · 断流 ${s.underruns} 次`;
}
function audioPacket(raw) {
  if (raw.byteLength < 18 || (raw.byteLength - 16) % 2) throw Error("无效 PCM 包");
  const view = new DataView(raw);
  if (view.getUint32(0, true) !== 0x31534446) throw Error("未知音频协议"); // FDS1
  const s = speech, id = view.getUint32(4, true), seq = view.getUint32(8, true);
  if (!s || id !== s.id) return; // cancellation fence, including packets already on wire
  const rate = view.getUint32(12, true);
  if (seq !== s.seq++ || (s.rate && rate !== s.rate)) throw Error("音频顺序或采样率错误");
  s.rate = rate;
  const count = (raw.byteLength - 16) / 2;
  s.received += count;
  if ((s.received - s.played) / rate * 1000 > s.limit + 1) throw Error("播放缓冲超限");
  const buffer = context.createBuffer(1, count, rate), values = buffer.getChannelData(0);
  for (let i = 0; i < count; i++) values[i] = view.getInt16(16 + 2 * i, true) / 32768;
  const node = context.createBufferSource();
  node.buffer = buffer; node.connect(context.destination); nodes.add(node);
  if (s.firstAudio === undefined) {
    s.firstAudio = Math.round(performance.now() - s.start);
    s.next = context.currentTime + 0.08;
  } else if (s.next < context.currentTime) {
    s.underruns++;
    s.next = context.currentTime + 0.08;
  }
  const endSamples = s.received;
  node.onended = () => {
    nodes.delete(node);
    if (speech !== s) return;
    s.played = Math.max(s.played, endSamples);
    progress(s);
  };
  node.start(s.next); s.next += count / rate;
  progress(s);
}
function control(msg) {
  const d = msg.data || {};
  if (msg.event === "error") throw Error(d.message);
  if (msg.event === "speech_start") {
    clearAudio();
    speech = {id: d.utterance_id, seq: 0, received: 0, played: 0, next: 0,
      limit: d.buffer_ms, start: performance.now(), underruns: 0, eof: false};
    $("text").textContent = "";
  } else if (msg.event === "speech_cancelled" && d.utterance_id === speech?.id) {
    clearAudio();
    if (d.reason !== "stream_error") $("status").textContent = `已停止旧回复（${d.reason}）`;
  } else if (d.utterance_id === speech?.id) {
    if (msg.event === "speech_text_delta") {
      speech.firstText ??= Math.round(performance.now() - speech.start);
      $("text").textContent += d.text;
    } else if (msg.event === "speech_audio_end") {
      if (d.samples !== speech.received) throw Error("音频结束计数不一致");
      speech.eof = true; progress(speech);
    } else if (msg.event === "speech_error") {
      $("status").textContent = `生成失败：${d.error}`;
    }
  }
}
$("stop").onclick = () => { $("status").textContent = "已断开"; void stop(); };
$("start").onclick = async () => {
  if (connecting) return;
  connecting = true; $("start").disabled = true;
  try {
    context = new AudioContext({sampleRate: 16000, latencyHint: "interactive"});
    await context.resume();
    media = await navigator.mediaDevices.getUserMedia({audio: {
      channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true}});
    await context.audioWorklet.addModule(new URL("./mic-worklet.js", import.meta.url));
    mic = context.createMediaStreamSource(media);
    capture = new AudioWorkletNode(context, "mic-frames");
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/realtime`);
    ws.binaryType = "arraybuffer";
    await new Promise((resolve, reject) => {
      ws.onopen = resolve; ws.onerror = () => reject(Error("WebSocket 连接失败"));
    });
    send("config", {exp: `stream-demo-${crypto.randomUUID()}`, lang: "live", audio_protocol: "pcm16.v1"});
    ws.onmessage = event => {
      try { typeof event.data === "string" ? control(JSON.parse(event.data)) : audioPacket(event.data); }
      catch (error) { $("status").textContent = error.message; void stop(); }
    };
    ws.onclose = () => { $("status").textContent = "连接已关闭"; void stop(); };
    capture.port.onmessage = event => {
      if (ws?.readyState !== WebSocket.OPEN) return;
      if (ws.bufferedAmount > 1024 * 1024) {
        $("status").textContent = "上传网络拥塞，已断开"; void stop(); return;
      }
      ws.send(event.data);
    };
    mic.connect(capture); capture.connect(context.destination); // worklet output is silent
    $("stop").disabled = false;
    $("status").textContent = "麦克风已连接，可以说话了。";
  } catch (error) {
    $("status").textContent = error.message;
    await stop();
  } finally { connecting = false; }
};
window.addEventListener("pagehide", () => { void stop(); });
