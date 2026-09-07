import {SpeechPlayer} from "./speech-player.js";
import {DemoTelemetry} from "./demo-telemetry.js";

const $ = id => document.getElementById(id);
const socketURL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/realtime";
let active = null;
const messages = new Map();
const STATES = {
  idle: ["准备好，聊两句？", "戴上耳机，点击下方按钮开始。"],
  connecting: ["正在连接", "请允许浏览器使用麦克风；可以随时点击结束。"],
  listening: ["我在听，你说。", "直接开口就好，不需要按住任何按钮。"],
  hearing: ["听到你的声音了", "按自己的节奏说，也可以在回复时继续开口。"],
  thinking: ["正在准备回应", "文本和语音正在接力生成，你仍然可以继续说话。"],
  speaking: ["正在回应你", "你可以继续说话；模型判断需要打断时会停止旧回复。"],
  muted: ["麦克风已静音", "不上传麦克风声音；仍发送静音帧，已有回复会继续播放。"],
  error: ["连接需要检查", "处理下方提示后，点击开始对话重试。"],
  ended: ["本次对话已结束", "麦克风已释放。再次开始会建立一段全新的对话。"],
};
function mode(name, detail) {
  document.body.dataset.mode = name;
  $("activity").textContent = STATES[name][0];
  $("status").textContent = detail || STATES[name][1];
}
function activity(s, value) {
  s.mode = value;
  if (active === s) mode(s.muted ? "muted" : value);
}
function notice(message = "") {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}
function controls() {
  const busy = !!active;
  $("start").disabled = busy;
  $("stop").disabled = !busy;
  $("mute").disabled = !active?.ready;
  $("microphone").disabled = busy;
  $("clear").disabled = busy || messages.size === 0;
  $("connection").dataset.connected = String(!!active?.ready);
  $("connection-label").textContent = active?.ready ? "实时连接已建立" : busy ? "连接中…" : "尚未连接";
  $("mute").setAttribute("aria-pressed", String(!!active?.muted));
  $("mute").textContent = active?.muted ? "恢复麦克风" : "静音麦克风";
}
function friendly(error) {
  const known = {
    NotAllowedError: "麦克风权限未获准。请在地址栏的网站权限中允许麦克风，并检查系统隐私设置。",
    NotFoundError: "未找到可用麦克风。请接入耳机或麦克风后重试。",
    NotReadableError: "麦克风无法打开，可能被其他程序占用。请关闭占用设备的程序后重试。",
    NotSupportedError: "当前浏览器或音频设备不支持此录音方式。请换用桌面 Chrome/Edge，或切换系统默认麦克风。",
    OverconstrainedError: "所选麦克风已不可用，请重新选择系统默认麦克风。",
    SecurityError: "浏览器禁止了麦克风访问。请用 localhost 或 HTTPS 打开本页。",
  };
  return known[error.name] || error.message || "连接失败，请检查服务器和 SSH 转发。";
}
function send(s, event, data) {
  if (active === s && s.ws?.readyState === WebSocket.OPEN) {
    s.ws.send(JSON.stringify({event, data}));
  }
}
function setLevel(value) {
  $("level-fill").style.width = Math.round(value * 100) + "%";
  $("mic-level").setAttribute("aria-valuenow", String(Math.round(value * 100)));
  document.body.style.setProperty("--level", value.toFixed(3));
}
function resetView() {
  messages.clear();
  $("messages").replaceChildren();
  $("empty-state").hidden = false;
  $("first-text").textContent = $("first-audio").textContent = "—";
  $("buffer").textContent = "0 ms";
  $("interrupts").textContent = "0 / 0";
  $("last-event").textContent = "—";
  for (const id of ["stage-hold", "stage-decision", "stage-generation", "stage-total", "socket-rtt"]) $(id).textContent = "—";
  $("duration").textContent = "00:00";
  $("transcript-note").textContent = "转写可能晚于回复到达；未播完的回复会保留并标注。";
}
function record(key, role, text, pending = false, beforeKey = null) {
  let row = messages.get(key);
  const list = $("conversation");
  const nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 90;
  if (!row) {
    const element = document.createElement("article");
    element.className = "message " + role;
    const heading = document.createElement("div");
    heading.className = "message-heading";
    const avatar = document.createElement("span");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "你" : "H";
    const name = document.createElement("span");
    name.textContent = role === "user" ? "你 · 语音转写" : "HumDial";
    const tag = document.createElement("span");
    tag.className = "message-tag";
    const body = document.createElement("p");
    body.className = "message-text";
    heading.append(avatar, name, tag);
    element.append(heading, body);
    $("messages").insertBefore(element, messages.get(beforeKey)?.element || null);
    row = {element, body, tag, text: "", role};
    messages.set(key, row);
    // UI-only retention bound; this does not modify the engine's history.
    if (messages.size > 120) {
      const oldest = messages.keys().next().value;
      messages.get(oldest).element.remove();
      messages.delete(oldest);
    }
  }
  row.text = text;
  row.body.textContent = text || (pending ? "等待转写…" : "正在生成…");
  row.body.classList.toggle("pending", pending || !text);
  $("empty-state").hidden = true;
  if (nearBottom) list.scrollTop = list.scrollHeight;
  return row;
}
function userRecord(turn) {
  const key = "user-" + turn;
  if (!messages.has(key)) record(key, "user", "", true);
}
function tagSpeech(s, label, cancelled = false) {
  const row = messages.get("assistant-" + s.player?.speech?.id);
  if (row) {
    row.tag.textContent = label;
    row.tag.dataset.cancelled = String(cancelled);
  }
}
function playback(s, state) {
  if (active !== s) return;
  s.telemetry?.playback(state);
  $("first-text").textContent = state.firstText === undefined ? "—" : state.firstText + " ms";
  $("first-audio").textContent = state.firstAudio === undefined ? "—" : state.firstAudio + " ms";
  $("buffer").textContent = Math.round((state.received - state.played) / (state.rate || 24000) * 1000) + " ms";
  $("interrupts").textContent = (s.underruns + state.underruns) + " / " + s.interrupts;
  const ended = state.eof && state.played === state.received;
  if (ended) {
    tagSpeech(s, "播放完成");
    activity(s, s.hearing ? "hearing" : "listening");
  } else if (state.received > state.played) {
    tagSpeech(s, "正在播放");
    activity(s, "speaking");
  }
}
function control(s, msg) {
  const d = msg.data || {};
  $("last-event").textContent = msg.event;
  if (msg.event === "error") throw Error(d.message || "服务器拒绝了本次连接。");
  if (msg.event === "demo_ready") {
    if (d.protocol !== "pcm16.v1") throw Error("服务器音频协议不兼容，请更新后端。");
    $("session-id").textContent = d.session_id;
    s.telemetry.enabled = d.observability === "demo-trace-v1";
    $("trace-status").textContent = s.telemetry.enabled ? "逐轮记录已启用 · demo-trace-v1" : "旧后端：逐轮记录未启用";
    s.accept?.();
    return;
  }
  if (msg.event === "demo_pong") {
    s.telemetry.pong(d);
    return;
  }
  if (msg.event === "demo_latency") {
    for (const [id, field] of [["stage-hold", "hold_ms"], ["stage-decision", "decision_ms"],
      ["stage-generation", "generation_ms"], ["stage-total", "vad_to_audio_ms"]]) {
      $(id).textContent = Number.isFinite(d[field]) ? Math.round(d[field]) + " ms" : "—（无匹配锚点）";
    }
    return;
  }
  if (msg.event === "vad_start") {
    s.hearing = true;
    if (s.mode !== "speaking") activity(s, "hearing");
  } else if (msg.event === "vad_done" || msg.event === "vad_640_done") {
    s.hearing = false;
    if (s.mode !== "speaking") activity(s, "thinking");
  } else if (msg.event === "asr_done") {
    const text = String(d.content || "").trim();
    if (text) record("user-" + d.turn, "user", text, false, s.replyTurns.get(d.turn));
  } else if (msg.event === "speech_start") {
    if (s.player.speech) {
      if (!s.player.speech.eof || s.player.speech.played < s.player.speech.received) tagSpeech(s, "已替换，可能未播完", true);
      s.underruns += s.player.speech.underruns;
    }
    userRecord(d.turn);
    s.replyTurns.set(d.turn, "assistant-" + d.utterance_id);
    record("assistant-" + d.utterance_id, "assistant", "");
    s.player.start(d);
    activity(s, "thinking");
  } else if (msg.event === "speech_cancelled" && d.utterance_id === s.player.speech?.id) {
    s.telemetry.snapshot("cancel", s.player.speech);
    const interrupted = ["shot_interrupt", "long_interrupt"].includes(d.reason);
    if (interrupted) s.interrupts++;
    const failed = d.reason === "stream_error";
    tagSpeech(s, failed ? "生成失败，可能未播完" : interrupted ? "已打断，可能未播完" : "已停止，可能未播完", true);
    s.underruns += s.player.speech.underruns;
    s.player.cancel();
    $("buffer").textContent = "0 ms";
    $("interrupts").textContent = s.underruns + " / " + s.interrupts;
    activity(s, s.hearing ? "hearing" : "listening");
    if (interrupted) $("status").textContent = s.muted ? STATES.muted[1] : "旧回复已停止，继续说就好。";
  } else if (s.player.speech && d.utterance_id === s.player.speech.id) {
    if (msg.event === "speech_text_delta") {
      s.player.text();
      const key = "assistant-" + d.utterance_id;
      record(key, "assistant", (messages.get(key)?.text || "") + String(d.text || ""));
    } else if (msg.event === "speech_audio_end") {
      s.player.finish(d);
    } else if (msg.event === "speech_error") {
      tagSpeech(s, "生成失败", true);
      notice("本次回复生成失败。可以继续说话；如持续失败，请检查服务器日志。");
    }
  }
}
async function devices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const available = await navigator.mediaDevices.enumerateDevices();
    const current = $("microphone").value;
    const options = [new Option("系统默认麦克风", "")];
    for (const device of available.filter(d => d.kind === "audioinput" && d.deviceId && d.deviceId !== "default")) {
      options.push(new Option(device.label || "麦克风 " + options.length, device.deviceId));
    }
    $("microphone").replaceChildren(...options);
    if (options.some(o => o.value === current)) $("microphone").value = current;
  } catch { /* Device labels are optional before permission. */ }
}
function alive(s) {
  if (active !== s) throw new DOMException("连接已取消", "AbortError");
}
async function release(s) {
  clearInterval(s.timer);
  clearTimeout(s.openTimer);
  s.abort.abort();
  s.reject?.(new DOMException("连接已取消", "AbortError"));
  s.player?.cancel();
  if (s.capture) {
    s.capture.port.onmessage = null;
    s.capture.port.close();
    s.capture.disconnect();
  }
  s.mic?.disconnect();
  s.media?.getTracks().forEach(track => { track.onended = null; track.stop(); });
  if (s.ws) {
    s.ws.onopen = s.ws.onclose = s.ws.onmessage = s.ws.onerror = null;
    s.ws.close();
  }
  if (s.context && s.context.state !== "closed") {
    s.context.onstatechange = null;
    await s.context.close().catch(() => {});
  }
}
async function stop(s = active, reason = "", failed = false) {
  if (!s || active !== s) return;
  s.telemetry?.snapshot("stop", s.player?.speech);
  if (s.player?.speech && (!s.player.speech.eof || s.player.speech.played !== s.player.speech.received)) {
    tagSpeech(s, "连接结束，可能未播完", true);
  }
  active = null; // Fence old promises and WebSocket callbacks before any await.
  controls();
  setLevel(0);
  $("buffer").textContent = "0 ms";
  mode(failed ? "error" : "ended");
  if (reason) notice(reason);
  await release(s);
}
async function connect() {
  if (active) return;
  notice();
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    notice("麦克风需要安全连接。请通过 SSH 转发后打开 http://localhost:18000/demo/，或使用 HTTPS。");
    mode("error"); return;
  }
  if (!window.AudioContext || !window.AudioWorkletNode || !window.WebSocket) {
    notice("当前浏览器缺少实时音频接口，请使用支持 AudioWorklet 的桌面浏览器。");
    mode("error"); return;
  }
  const selectedDevice = $("microphone").value;
  const s = {abort: new AbortController(), muted: false, ready: false, mode: "connecting",
    replyTurns: new Map(), interrupts: 0, underruns: 0, hearing: false};
  active = s;
  controls(); mode("connecting");
  try {
    // Resume in the click gesture, before network/permission awaits.
    s.context = new AudioContext({latencyHint: "interactive"});
    await s.context.resume();
    alive(s);
    const timeout = setTimeout(() => s.abort.abort(), 10000);
    let info;
    try {
      const response = await fetch("/api/demo/info", {cache: "no-store", signal: s.abort.signal});
      if (!response.ok) throw Error("无法读取演示配置。请确认连接的是新版 backend，而不是单独的静态文件服务。");
      info = await response.json();
    } finally { clearTimeout(timeout); }
    alive(s);
    if (!info.streaming) throw Error("服务器未启用流式 ActorEngine。请用 bash setup/start_demo.sh 启动，或给 backend 加 --streaming。");
    s.media = await navigator.mediaDevices.getUserMedia({audio: {
      channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
      ...(selectedDevice ? {deviceId: {exact: selectedDevice}} : {}),
    }});
    alive(s);
    void devices();
    await s.context.audioWorklet.addModule(new URL("./mic-worklet.js", import.meta.url));
    alive(s);
    s.mic = s.context.createMediaStreamSource(s.media);
    s.capture = new AudioWorkletNode(s.context, "mic-frames");
    s.player = new SpeechPlayer(s.context, (event, data) => send(s, event, data), state => playback(s, state));
    s.telemetry = new DemoTelemetry((event, data) => send(s, event, data), s.context,
      () => s.ws?.bufferedAmount || 0, ms => { $("socket-rtt").textContent = Math.round(ms) + " ms"; });
    s.ws = new WebSocket(socketURL);
    s.ws.binaryType = "arraybuffer";
    await new Promise((resolve, reject) => {
      s.reject = reject;
      s.accept = () => { clearTimeout(s.openTimer); s.reject = null; s.accept = null; resolve(); };
      s.openTimer = setTimeout(() => reject(Error("服务握手超时。请检查 SSH 转发与服务器日志后重试。")), 30000);
      s.ws.onopen = () => send(s, "config", {client: "humdial-web", audio_protocol: "pcm16.v1"});
      s.ws.onmessage = event => {
        if (active !== s) return;
        try {
          if (typeof event.data === "string") control(s, JSON.parse(event.data));
          else s.player.packet(event.data);
        } catch (error) {
          if (s.reject) s.reject(error);
          else void stop(s, friendly(error), true);
        }
      };
      s.ws.onerror = () => {
        const error = Error("语音连接失败。请检查 SSH 转发窗口和服务器是否仍在运行。");
        if (s.reject) s.reject(error);
        else void stop(s, error.message, true);
      };
      s.ws.onclose = () => {
        const error = Error("语音连接已断开，麦克风和播放已停止。恢复网络后请重新开始。");
        if (s.reject) s.reject(error);
        else void stop(s, error.message, true);
      };
    });
    alive(s);
    resetView();
    s.ready = true;
    s.started = performance.now();
    s.telemetry.tick();
    s.timer = setInterval(() => {
      s.telemetry.tick();
      const seconds = Math.floor((performance.now() - s.started) / 1000);
      $("duration").textContent = String(Math.floor(seconds / 60)).padStart(2, "0") + ":" + String(seconds % 60).padStart(2, "0");
    }, 1000);
    const track = s.media.getAudioTracks()[0];
    track.onended = () => { void stop(s, "麦克风已断开或权限被撤销，请重新连接设备。", true); };
    const settings = track.getSettings();
    $("audio-format").textContent = (settings.sampleRate || "未知") + " Hz 设备 → 16 kHz 上传 / 256 样本";
    let lastLevel = 0;
    s.capture.port.onmessage = event => {
      if (active !== s || s.ws.readyState !== WebSocket.OPEN) return;
      // Keep the engine's audio clock advancing during explicit user mute.
      if (s.muted) new Float32Array(event.data).fill(0);
      if (s.ws.bufferedAmount > 128 * 1024) {
        void stop(s, "上传网络拥塞，已停止对话以避免继续积压音频。请检查网络后重连。", true);
        return;
      }
      if (performance.now() - lastLevel > 80) {
        const frame = new Float32Array(event.data);
        const rms = Math.sqrt(frame.reduce((sum, value) => sum + value * value, 0) / frame.length);
        setLevel(Math.min(1, rms * 6));
        lastLevel = performance.now();
      }
      s.ws.send(event.data);
    };
    s.context.onstatechange = () => {
      if (active === s && s.ready && s.context.state !== "running") {
        void stop(s, "浏览器暂停了音频设备。请保持页面在前台，再次开始对话。", true);
      }
    };
    if (s.context.state !== "running") throw Error("浏览器已暂停音频。请回到页面后重新开始。");
    s.mic.connect(s.capture); s.capture.connect(s.context.destination); // Worklet output is silent.
    controls(); activity(s, "listening");
  } catch (error) {
    if (active === s) await stop(s, error.name === "AbortError" ? "读取服务配置超时，请检查 SSH 转发后重试。" : friendly(error), true);
    else await release(s); // A late permission result must release its tracks too.
  }
}
$("start").addEventListener("click", () => { void connect(); });
$("stop").addEventListener("click", () => { notice(); void stop(); });
$("mute").addEventListener("click", () => {
  const s = active;
  if (!s?.ready) return;
  s.muted = !s.muted;
  s.media.getAudioTracks().forEach(track => { track.enabled = !s.muted; });
  controls();
  activity(s, s.mode);
  if (s.muted) setLevel(0);
});
$("clear").addEventListener("click", () => {
  if (active) return;
  resetView(); controls();
  $("transcript-note").textContent = "页面记录已清空；服务器上的输入音频与日志未删除。";
});
$("fullscreen").addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch { notice("当前浏览器不支持页面全屏，可以使用浏览器自带的全屏功能。"); }
});
document.addEventListener("fullscreenchange", () => { $("fullscreen").textContent = document.fullscreenElement ? "退出全屏 ↙" : "全屏展示 ↗"; });
window.addEventListener("pagehide", () => { void stop(); });
window.addEventListener("offline", () => { if (active) void stop(active, "网络已离线，麦克风和播放已停止。", true); });
navigator.mediaDevices?.addEventListener("devicechange", () => { void devices(); });
$("endpoint").textContent = socketURL;
controls();
void devices();
