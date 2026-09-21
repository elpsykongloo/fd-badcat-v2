const $ = id => document.getElementById(id);
const LABELS = {
  acceptable:"可接受", vad_or_truncation:"VAD／截断", transcription:"转写", route:"路由",
  third_party_or_echo:"第三方／回声", response_content:"回复内容", tts_fidelity:"TTS 忠实度",
  voice:"音色", stutter_or_underrun:"卡顿／断流", cancellation:"取消", history_pollution:"历史污染",
  latency:"延迟", cannot_determine:"无法判断"
};
let selected = null;
async function json(url, options) {
  const response = await fetch(url, {cache:"no-store", ...options});
  if (!response.ok) throw Error((await response.json().catch(()=>({}))).detail || `HTTP ${response.status}`);
  return response.json();
}
function node(tag, text, cls) { const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(cls)n.className=cls; return n; }
async function sessions() {
  const data = await json(`/api/demo/diagnostics/sessions?anomalies_only=${$("anomalies").checked}`);
  $("sessions").replaceChildren(...data.sessions.map(s => {
    const b=node("button",undefined,"session"+(selected===s.session_id?" selected":"")); b.type="button";
    b.append(node("strong",s.session_id));
    const badge=node("span",s.status,"badge "+s.status); b.firstChild.append(badge);
    b.append(node("small",`${s.created_utc || ""} · ${s.turns ?? "进行中"} 轮 · ${s.anomaly_count} 异常 · ${s.reviews} review`));
    b.onclick=()=>load(s.session_id); return b;
  }));
}
function rows(target, values) { target.replaceChildren(...values.map(v=>node("div",JSON.stringify(v,null,2),"row"))); }
function renderCase(item) {
  const box=node("div",undefined,"case");
  box.append(node("strong",`${item.kind} · ${item.status} · ${Math.round(item.elapsed_ms||0)} ms`));
  if(item.tts_expected_text) box.append(node("p","期望文本："+item.tts_expected_text));
  if(item.text) box.append(node("p","模型输出："+item.text));
  for(const name of item.audio||[]) { const audio=document.createElement("audio"); audio.controls=true; audio.preload="none"; audio.src=`/api/demo/diagnostics/cases/${encodeURIComponent(item.case_id)}/audio/${encodeURIComponent(name)}`; box.append(node("small",name),audio); }
  return box;
}
function renderReview(review) { return node("div",`修订 ${review.revision} · ${review.reviewer} · ${review.labels.join(", ")}\n${review.note}`,"row"); }
function renderTurn(turn,data) {
  const fragment=$("turn-template").content.cloneNode(true), article=fragment.querySelector(".turn"), body=fragment.querySelector(".turn-body");
  fragment.querySelector(".turn-name").textContent=turn.turn_id;
  fragment.querySelector(".turn-meta").textContent=`${turn.call_ids.length} 调用 · ${turn.case_ids.length} 案例 · ${turn.outcomes.length} 结果`;
  fragment.querySelector(".turn-head").onclick=()=>{body.hidden=!body.hidden;};
  rows(fragment.querySelector(".route-list"),turn.routes);
  rows(fragment.querySelector(".outcome-list"),turn.outcomes);
  rows(fragment.querySelector(".history-list"),turn.history||[]);
  const utteranceIds=new Set((turn.utterances||[]).map(item=>item.utterance_id));
  rows(fragment.querySelector(".span-list"),data.spans.filter(span=>(
    span.span_type==="model"&&turn.call_ids.includes(span.call_id))||(
    span.span_type==="utterance"&&utteranceIds.has(span.utterance_id))));
  fragment.querySelector(".case-list").replaceChildren(...turn.case_ids.filter(id=>data.cases[id]).map(id=>renderCase(data.cases[id])));
  const labels=fragment.querySelector(".labels");
  for(const [value,label] of Object.entries(LABELS)){const l=node("label");const c=document.createElement("input");c.type="checkbox";c.name="labels";c.value=value;l.append(c," "+label);labels.append(l);}
  const reviewList=fragment.querySelector(".review-list");
  const reviews=data.reviews.filter(r=>r.turn_id===turn.turn_id); reviewList.replaceChildren(...reviews.map(renderReview));
  const form=fragment.querySelector("form"),state=fragment.querySelector(".save-state");
  form.onsubmit=async event=>{event.preventDefault();state.textContent="保存中…";const f=new FormData(form);try{const review=await json(`/api/demo/diagnostics/sessions/${encodeURIComponent(selected)}/turns/${encodeURIComponent(turn.turn_id)}/reviews`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({labels:f.getAll("labels"),reviewer:f.get("reviewer"),note:f.get("note")})});reviewList.append(renderReview(review));state.textContent="已保存新修订";}catch(error){state.textContent=error.message;}};
  return article;
}
async function load(id) {
  selected=id; await sessions(); const data=await json(`/api/demo/diagnostics/sessions/${encodeURIComponent(id)}`);
  $("empty").hidden=true;$("detail").hidden=false;$("session-id").textContent=id;$("session-status").textContent=data.summary.status;$("case-count").textContent=Object.keys(data.cases).length;$("review-count").textContent=data.reviews.length;$("capture-status").textContent=data.manifest.capture?.enabled?(data.manifest.capture.truncated?"已开启／截断":"已开启"):"关闭";$("manifest").textContent=JSON.stringify(data.manifest,null,2);
  $("findings").replaceChildren(...(data.summary.findings||[]).map(f=>node("div",`${f.code}：${f.message}`,"finding "+f.severity)));
  $("turns").replaceChildren(...data.turns.map(t=>renderTurn(t,data)));
}
$("refresh").onclick=()=>sessions(); $("anomalies").onchange=()=>sessions();
sessions().catch(error=>{$("sessions").textContent=error.message;});
