// Frankenstein v2 page: talk to the engine (microphone), or play a demo dialogue (the engine speaks a caller's lines
// with another voice; the page plays them and streams them in as the session's input, like a microphone), and a
// Traces tab over the engine's SQLite store. The page talks to the bridge (webui/server.py), which relays to the engine.
"use strict";
const $ = (id) => document.getElementById(id);
const S = {
  cfg: null, ws: null, mode: null, inCtx: null, outCtx: null, mic: null, nextPlay: 0, sources: [],
  t0: 0, timer: null, turns: 0, tools: 0, tTurn: null, firstMs: [], agentLoud: 0, agentEver: false,
  pendingTools: 0, lastAsr: 0, lastDone: 0, agentMsg: null, demo: null, closing: false,
};
const IN_RATE = 16000, OUT_RATE = 22050, STEP = 1280;     // the engine's input chunk: 80 ms at 16 kHz

// ------------------------------------------------------------------ boot, tabs, engine status
async function boot() {
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("orb").addEventListener("click", () => (S.ws ? stop("user") : start("mic")));
  $("demoBtn").addEventListener("click", () => (S.ws ? stop("user") : start("demo")));
  $("logToggle").addEventListener("change", () => { $("log").hidden = !$("logToggle").checked; remember("f2.log", $("logToggle").checked); });
  $("logToggle").checked = recall("f2.log") === "true"; $("log").hidden = !$("logToggle").checked;
  $("refresh").addEventListener("click", loadSessions);
  $("lang").addEventListener("change", () => remember("f2.lang", $("lang").value));
  $("voice").addEventListener("change", () => remember("f2.voice", $("voice").value));
  if (location.hash === "#traces") showTab("traces");
  await refreshConfig();
}
function remember(k, v) { try { localStorage.setItem(k, String(v)); } catch (_) { /* storage unavailable */ } }
function recall(k) { try { return localStorage.getItem(k); } catch (_) { return null; } }

function showTab(name) {
  document.querySelectorAll(".tab").forEach((b) => { const on = b.dataset.tab === name; b.classList.toggle("active", on); b.setAttribute("aria-selected", on); });
  $("view-talk").hidden = name !== "talk"; $("view-traces").hidden = name !== "traces";
  history.replaceState(null, "", name === "traces" ? "#traces" : "#");
  if (name === "traces") loadSessions();
}

async function refreshConfig() {
  try {
    const c = await fetch("/api/config").then((r) => r.json());
    S.cfg = c;
    const ok = c.gpu_configured && c.v2_reachable;
    $("engine").className = "engine " + (ok ? "ok" : "bad");
    $("engineLabel").textContent = ok ? `engine online · ${c.languages.length} languages` : (c.gpu_configured ? "engine offline" : "no engine configured");
    fill($("lang"), c.languages.slice().sort((a, b) => (b.tested - a.tested) || a.name.localeCompare(b.name))
      .map((l) => [l.code, `${l.name}${l.tested ? "" : " · preview"}`]), recall("f2.lang") || "it");
    fill($("voice"), (c.voices || []).map((v) => [v, v]), recall("f2.voice"));
    $("voice").closest("label").hidden = (c.voices || []).length < 2;
    $("demoBtn").disabled = !ok || !c.demo_dialogue; $("orb").disabled = !ok;
    if (!c.demo_dialogue) $("demoNote").textContent = "The demo dialogue needs an engine with a talker.";
    if (!ok) setOrb("idle", "Engine offline", "Start it with: make f2-webui");
  } catch (e) {
    $("engine").className = "engine bad"; $("engineLabel").textContent = "web UI server unreachable";
  }
}
function fill(sel, items, keep) {
  sel.innerHTML = "";
  items.forEach(([v, t]) => { const o = document.createElement("option"); o.value = v; o.textContent = t; sel.append(o); });
  if (keep && items.some(([v]) => v === keep)) sel.value = keep;
}

// ------------------------------------------------------------------ session
function setOrb(state, title, sub) {
  $("orb").dataset.s = state; $("orbState").textContent = title; if (sub !== undefined) $("orbSub").textContent = sub;
  $("orbIcon").textContent = S.ws ? "■" : "●";
}

async function start(mode) {
  if (S.ws) return;
  resetView();
  S.mode = mode; S.closing = false;
  setOrb("connecting", "Connecting…", mode === "demo" ? "Preparing the demo dialogue." : "Allow the microphone when asked.");
  try {
    S.inCtx = new AudioContext({ sampleRate: IN_RATE });
  } catch (_) { S.inCtx = new AudioContext(); }
  S.outCtx = new AudioContext({ sampleRate: OUT_RATE });
  await S.outCtx.resume();
  if (mode === "mic") {
    if (!window.isSecureContext) { fail("The microphone needs localhost or HTTPS (see webui/README.md)."); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      S.mic = { stream };
    } catch (e) { fail(`Microphone unavailable: ${e.name || e}`); return; }
  }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.hostname}:${S.cfg.ws_port}/ws`);
  ws.binaryType = "arraybuffer"; S.ws = ws;
  ws.onopen = () => ws.send(JSON.stringify({
    type: "session.start", transport: "gpu", language: $("lang").value, voice: $("voice").value || undefined,
    system_prompt: $("prompt").value.trim() || undefined, input_rate: mode === "demo" ? IN_RATE : S.inCtx.sampleRate,
  }));
  ws.onmessage = (e) => (typeof e.data === "string" ? onJson(JSON.parse(e.data)) : onAudio(e.data));
  ws.onerror = () => { if (S.ws === ws) fail("Connection error: is the engine running?"); };
  ws.onclose = () => { if (S.ws === ws) teardown(S.closing ? "Session ended" : "Disconnected"); };
}

function onSessionCreated(m) {
  S.t0 = performance.now();
  S.timer = setInterval(() => { $("mTime").textContent = fmtS((performance.now() - S.t0) / 1000); }, 500);
  act("ok", "Session open", `${m.language}${m.voice ? " · voice " + m.voice : ""}`);
  $("demoBtn").textContent = "■ End"; $("demoBtn").disabled = false;
  if (S.mode === "mic") startMic(); else startDemo();
}

function startMic() {
  const ctx = S.inCtx, src = ctx.createMediaStreamSource(S.mic.stream);
  const proc = ctx.createScriptProcessor(2048, 1, 1), mute = ctx.createGain(); mute.gain.value = 0;
  proc.onaudioprocess = (e) => sendPcm(e.inputBuffer.getChannelData(0));
  src.connect(proc); proc.connect(mute); mute.connect(ctx.destination);
  S.mic.proc = proc; S.mic.src = src;
  setOrb("listening", "Listening", "Speak naturally; interrupt the agent whenever you like. Tap to end.");
}

function sendPcm(f32) {
  if (!S.ws || S.ws.readyState !== 1) return;
  const pcm = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) pcm[i] = Math.max(-1, Math.min(1, f32[i])) * 32767;
  S.ws.send(pcm.buffer);
}

function stop(reason) {
  const ws = S.ws; if (!ws) return;
  S.closing = true;
  if (S.demo) S.demo.cancelled = true;
  try { ws.send(JSON.stringify({ type: "session.end", reason })); } catch (_) { /* closed */ }
  setOrb("idle", "Ending…", "");
  setTimeout(() => { if (ws.readyState < 2) ws.close(1000, reason); }, 1500);
}

function teardown(label) {
  if (S.demo) S.demo.cancelled = true;
  if (S.mic) { S.mic.stream.getTracks().forEach((t) => t.stop()); S.mic = null; }
  if (S.pump) { clearInterval(S.pump); S.pump = null; }
  clearInterval(S.timer); S.timer = null;
  [S.inCtx, S.outCtx].forEach((c) => c && c.close().catch(() => {}));
  S.inCtx = S.outCtx = null; S.ws = null; S.sources = []; S.nextPlay = 0;
  $("demoBtn").textContent = "▶ Play a demo dialogue"; $("demoBtn").disabled = !(S.cfg && S.cfg.demo_dialogue);
  setOrb("idle", "Tap to talk", label === "Session ended" ? "Session ended. Its trace is in the Traces tab." : label);
}
function fail(msg) { act("bad", "Error", msg); if (S.ws) { S.closing = true; try { S.ws.close(); } catch (_) {} } teardown(msg); }

// ------------------------------------------------------------------ engine events
function onJson(m) {
  if (m.type === "session.created") return onSessionCreated(m);
  if (m.type === "session.closed") { act("", "Session closed", `${m.steps || 0} steps`); return; }
  if (m.type === "error") { act("bad", "Engine error", m.error || ""); if (m.error === "busy") fail("The engine is busy (all sessions in use)."); return; }
  if (m.type === "demo.audio" || m.type === "demo.done") return S.demo && S.demo.onLine(m);
  if (m.type !== "event") return;
  if (m.log) log(m.log);
  const ev = m.ev;
  if (ev === "asr.turn") {
    S.turns++; $("mTurns").textContent = S.turns; S.tTurn = performance.now(); S.lastAsr = S.turns; S.tAsr = S.tTurn; S.agentMsg = null;
    say("user", m.text, S.mode === "demo" ? "caller · heard by the ASR" : "you · heard by the ASR");
    act("", "Your turn ended", `"${cut(m.text, 60)}"`);
    if (S.mode === "mic") setOrb("thinking", "Thinking…");
  } else if (ev === "llm.request") {
    setOrb("thinking", "Thinking…", "");
  } else if (ev === "tool.started") {
    S.tools++; S.pendingTools++; $("mTools").textContent = S.tools;
    toolChip(m); act("tool", `Tool: ${m.name}`, args(m.args)); setOrb("tool", "Looking it up…", `${m.name}(${args(m.args)})`);
  } else if (ev === "tool.waiting") {
    act("warn", "Tool still running", `${m.s} s`);
  } else if (ev === "tool.completed") {
    S.pendingTools = Math.max(0, S.pendingTools - 1); S.tToolDone = performance.now(); toolDone(m);
    act(m.status === "completed" ? "ok" : "bad", `Tool ${m.status || "completed"}`, `${m.name} · ${m.ms ?? "?"} ms`);
  } else if (ev === "tts.segment") {
    agentText(m.text, m.holding, m.lane === "bc");
    if (m.holding) act("", "Holding phrase", cut(m.text, 60));
  } else if (ev === "llm.done") {
    S.lastDone = S.turns;
  } else if (ev === "audio.flush") {
    flushAudio(); if (S.agentMsg) S.agentMsg.classList.add("cut");
    act("warn", "Barge-in", "the agent stopped talking");
  } else if (ev === "engine.teardown") {
    act("", "Engine closed the session", "");
  } else if (ev === "tts.error" || ev === "ears.error") {
    act("bad", ev, m.error || "");
  }
}

// ------------------------------------------------------------------ agent audio
function onAudio(buf) {
  const ctx = S.outCtx; if (!ctx) return;
  const pcm = new Int16Array(buf); let peak = 0;
  const b = ctx.createBuffer(1, pcm.length, OUT_RATE), ch = b.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) { ch[i] = pcm[i] / 32768; peak = Math.max(peak, Math.abs(pcm[i])); }
  if (peak > 300) {                                    // the engine sends silence between replies too
    S.agentLoud = performance.now(); S.agentEver = true;
    if (S.tTurn !== null) { const ms = Math.round(performance.now() - S.tTurn); S.firstMs.push(ms); $("mFirst").textContent = `${ms} ms`; act("ok", "Agent speaking", `first audio ${ms} ms after your turn`); S.tTurn = null; }
    if ($("orb").dataset.s !== "speaking") setOrb("speaking", "Speaking", S.mode === "mic" ? "Talk over it to interrupt." : "");
  } else if ($("orb").dataset.s === "speaking" && performance.now() - S.agentLoud > 700) {
    setOrb("listening", S.mode === "demo" ? "Demo dialogue" : "Listening", S.mode === "demo" ? "" : "Your turn.");
  }
  play(b);
}
function play(b) {
  const ctx = S.outCtx, src = ctx.createBufferSource(); src.buffer = b; src.connect(ctx.destination);
  S.nextPlay = Math.max(S.nextPlay, ctx.currentTime + 0.05); src.start(S.nextPlay); S.nextPlay += b.duration;
  S.sources.push(src); src.onended = () => { S.sources = S.sources.filter((x) => x !== src); };
}
function flushAudio() { S.sources.forEach((s) => { try { s.stop(); } catch (_) {} }); S.sources = []; if (S.outCtx) S.nextPlay = S.outCtx.currentTime; }

// ------------------------------------------------------------------ demo dialogue
// The caller's lines (per language) are spoken by the engine with another voice (demo.say -> demo.audio). The page
// plays each line and streams it, resampled to 16 kHz, as the session's input; between lines it streams silence, so
// the engine hears exactly what a microphone would send. The next line waits until the agent has answered.
async function startDemo() {
  const lang = $("lang").value;
  const script = await fetch(`/api/demo-script?lang=${encodeURIComponent(lang)}`).then((r) => r.json());
  const d = S.demo = { cancelled: false, queue: [], lines: {}, onLine: null };
  d.onLine = (m) => {
    const L = d.lines[m.id] || (d.lines[m.id] = { chunks: [], done: false });
    if (m.type === "demo.audio") L.chunks.push({ sr: m.sr, pcm: b64pcm(m.pcm) }); else { L.done = true; L.resolve && L.resolve(); }
  };
  S.pump = setInterval(pump, 80);                      // real-time input: one 80 ms chunk per tick (silence or caller)
  setOrb("listening", "Demo dialogue", "A simulated caller is talking to the engine.");
  act("ok", "Demo dialogue", `${script.lines.length} caller lines${script.fallback ? " (English: no script for this language)" : ""}`);
  await sleep(1200);
  for (let i = 0; i < script.lines.length && !d.cancelled; i++) {
    const text = script.lines[i], before = S.lastAsr;
    S.ws.send(JSON.stringify({ type: "demo.say", id: i, text }));
    const L = d.lines[i] || (d.lines[i] = { chunks: [], done: false });
    if (!L.done) await new Promise((r) => { L.resolve = r; setTimeout(r, 30000); });
    if (d.cancelled) break;
    const pcm24 = concat(L.chunks.map((c) => c.pcm)), sr = L.chunks[0]?.sr || 24000;
    if (!pcm24.length) { act("bad", "Demo", "the caller line came back empty"); continue; }
    callerLine(text);
    const pcm16 = await resample(pcm24, sr, IN_RATE);
    playCaller(pcm24, sr);
    d.queue.push(pcm16);
    await sleep((pcm16.length / IN_RATE) * 1000 + 300);
    await waitAgent(d, before, 25000);
  }
  if (!d.cancelled) { await sleep(1500); act("ok", "Demo finished", "the session stays open: tap End, or talk"); }
}
function pump() {
  const d = S.demo; if (!d || !S.ws || S.ws.readyState !== 1) return;
  let out = new Float32Array(STEP), off = 0;
  while (off < STEP && d.queue.length) {
    const head = d.queue[0], n = Math.min(STEP - off, head.length);
    out.set(head.subarray(0, n), off); off += n;
    d.queue[0] = head.subarray(n); if (!d.queue[0].length) d.queue.shift();
  }
  sendPcm(out);
}
// The agent has answered the caller's line: a new user turn was heard, the LLM finished it, no tool is pending, the
// agent spoke after that turn (and after the last tool result) and has been quiet for 1.4 s.
async function waitAgent(d, before, timeoutMs) {
  const t0 = performance.now();
  while (!d.cancelled && performance.now() - t0 < timeoutMs) {
    await sleep(200);
    const heard = S.lastAsr > before;
    const spoke = S.agentLoud > (S.tAsr || 0) && S.agentLoud > (S.tToolDone || 0);
    const answered = S.lastDone >= S.lastAsr && S.pendingTools === 0 && spoke;
    if (heard && answered && performance.now() - S.agentLoud > 1400) return;
  }
}
function playCaller(pcm, sr) {
  const ctx = S.outCtx; if (!ctx) return;
  const b = ctx.createBuffer(1, pcm.length, sr); b.getChannelData(0).set(pcm);
  const src = ctx.createBufferSource(); src.buffer = b; src.connect(ctx.destination); src.start();
}
async function resample(pcm, from, to) {
  if (from === to) return pcm;
  const off = new OfflineAudioContext(1, Math.ceil(pcm.length * to / from), to);
  const b = off.createBuffer(1, pcm.length, from); b.getChannelData(0).set(pcm);
  const s = off.createBufferSource(); s.buffer = b; s.connect(off.destination); s.start();
  return (await off.startRendering()).getChannelData(0);
}
function b64pcm(s) { const bin = atob(s), n = bin.length / 2, out = new Float32Array(n), v = new DataView(new ArrayBuffer(bin.length)); for (let i = 0; i < bin.length; i++) v.setUint8(i, bin.charCodeAt(i)); for (let i = 0; i < n; i++) out[i] = v.getInt16(2 * i, true) / 32768; return out; }
function concat(arrs) { const n = arrs.reduce((a, x) => a + x.length, 0), out = new Float32Array(n); let o = 0; arrs.forEach((x) => { out.set(x, o); o += x.length; }); return out; }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ conversation + activity views
function resetView() {
  $("chat").innerHTML = ""; $("act").innerHTML = ""; $("log").textContent = "";
  Object.assign(S, { turns: 0, tools: 0, tTurn: null, tAsr: 0, tToolDone: 0, firstMs: [], agentLoud: 0, agentEver: false, pendingTools: 0, lastAsr: 0, lastDone: 0, agentMsg: null, demo: null });
  ["mTurns", "mTools"].forEach((k) => ($(k).textContent = "0")); $("mFirst").textContent = "—"; $("mTime").textContent = "0 s";
}
function say(role, text, who) {
  const el = document.createElement("div"); el.className = `msg ${role}`;
  el.innerHTML = '<span class="who"></span><span class="text"></span>';
  el.querySelector(".who").textContent = who; el.querySelector(".text").textContent = text || "";
  $("chat").append(el); scroll($("chat")); return el;
}
function agentText(text, holding, backchannel) {
  if (backchannel) return;
  if (!S.agentMsg) S.agentMsg = say("agent", "", "agent");
  const t = S.agentMsg.querySelector(".text"), span = document.createElement("span");
  span.textContent = (t.textContent ? " " : "") + text; if (holding) span.className = "hold";
  t.append(span); scroll($("chat"));
}
function callerLine(text) { const el = document.createElement("div"); el.className = "callerline"; el.textContent = `simulated caller says: “${text}”`; $("chat").append(el); scroll($("chat")); }
function toolChip(m) {
  const el = document.createElement("div"); el.className = "chip"; el.dataset.call = m.call_id || "";
  el.innerHTML = '<span class="fn"></span><span class="st">running…</span>';
  el.querySelector(".fn").textContent = `🔧 ${m.name}(${args(m.args)})`; $("chat").append(el); scroll($("chat"));
  S.agentMsg = null;                                 // the answer after the tool is a new bubble
}
function toolDone(m) {
  const el = [...document.querySelectorAll(".chip")].reverse().find((x) => !m.call_id || x.dataset.call === String(m.call_id));
  if (!el) return; const ok = (m.status || "completed") === "completed";
  el.classList.add(ok ? "ok" : "bad"); el.querySelector(".st").textContent = `${ok ? "✓" : "✗ " + m.status} ${m.ms ?? "?"} ms`;
  el.title = typeof m.result === "string" ? m.result : JSON.stringify(m.result);
}
function act(kind, title, detail) {
  const li = document.createElement("li"); li.className = kind;
  const t = S.t0 ? fmtS((performance.now() - S.t0) / 1000) : "";
  li.innerHTML = '<span class="t"></span><b></b> <span class="d"></span>';
  li.querySelector(".t").textContent = t; li.querySelector("b").textContent = title; li.querySelector(".d").textContent = detail || "";
  $("act").append(li); scroll($("act"));
}
function log(line) { const el = $("log"); el.textContent += line + "\n"; if (el.textContent.length > 60000) el.textContent = el.textContent.slice(-50000); if (!el.hidden) scroll(el); }
function scroll(el) { el.scrollTop = el.scrollHeight; }
function args(a) { if (!a) return ""; if (typeof a === "string") return cut(a, 80); return cut(Object.entries(a).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", "), 80); }
function cut(s, n) { s = String(s ?? ""); return s.length > n ? s.slice(0, n - 1) + "…" : s; }
function fmtS(s) { return s < 60 ? `${s.toFixed(0)} s` : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }

// ------------------------------------------------------------------ traces tab
async function loadSessions() {
  const box = $("sessions");
  if (S.cfg && !S.cfg.traces) { box.innerHTML = '<div class="empty">No trace store configured (start the web UI with --trace-db; make f2-webui does).</div>'; return; }
  const { sessions } = await fetch("/api/traces/sessions").then((r) => r.json());
  box.innerHTML = sessions.length ? "" : '<div class="empty">No sessions recorded yet.</div>';
  sessions.forEach((s) => {
    const el = document.createElement("div"); el.className = "sess";
    const dur = s.ended_wall && s.started_wall ? fmtS(s.ended_wall - s.started_wall) : "open";
    el.innerHTML = '<div class="row1"><span class="lang"></span><span class="when"></span><span class="id"></span></div><div class="row2"></div>';
    el.querySelector(".lang").textContent = s.language || "?";
    el.querySelector(".when").textContent = new Date(s.started_wall * 1000).toLocaleString();
    el.querySelector(".id").textContent = s.session_id;
    el.querySelector(".row2").innerHTML = `${dur} · ${s.turns} turns · ${s.tools} tool calls${s.first_ms ? ` · first audio ~${Math.round(s.first_ms)} ms` : ""}${s.errors ? ` · <span class="err">${s.errors} errors</span>` : ""}`;
    el.addEventListener("click", () => { box.querySelectorAll(".sess").forEach((x) => x.classList.remove("sel")); el.classList.add("sel"); loadSession(s.session_id); });
    box.append(el);
  });
}
async function loadSession(id) {
  const det = $("detail"); det.innerHTML = '<div class="empty">Loading…</div>';
  const r = await fetch(`/api/traces/session/${encodeURIComponent(id)}`);
  if (!r.ok) { det.innerHTML = '<div class="empty">Session not found.</div>'; return; }
  const { session, events, spans } = await r.json();
  const t0 = session.started_wall, t1 = Math.max(session.ended_wall || 0, ...spans.map((x) => x.end_wall || x.start_wall), ...events.map((e) => e.wall)), span = Math.max(1, t1 - t0);
  det.innerHTML = "";
  const h = document.createElement("h3"); h.textContent = `${session.language || "?"} · ${session.session_id}`; det.append(h);
  const sub = document.createElement("p"); sub.className = "sub";
  sub.textContent = `${new Date(t0 * 1000).toLocaleString()} · ${fmtS(span)} · voice ${session.voice || "default"}`; det.append(sub);
  const wf = document.createElement("div"); wf.className = "wf";
  ["turn", "llm", "tool", "first_audio"].forEach((kind) => {
    const lane = document.createElement("div"); lane.className = "lane";
    lane.innerHTML = `<span>${{ turn: "turns", llm: "LLM requests", tool: "tool calls", first_audio: "first audio" }[kind]}</span><div class="track"></div>`;
    spans.filter((x) => x.kind === kind).forEach((x) => {
      const b = document.createElement("div"); b.className = `bar ${kind}${x.status && x.status !== "ok" ? " bad" : ""}`;
      b.style.left = `${((x.start_wall - t0) / span) * 100}%`; b.style.width = `${(((x.end_wall || x.start_wall) - x.start_wall) / span) * 100}%`;
      b.title = `${x.name || kind} · ${x.dur_ms ? Math.round(x.dur_ms) + " ms" : ""} · ${x.status || ""}`;
      lane.querySelector(".track").append(b);
    });
    wf.append(lane);
  });
  det.append(wf);
  const fl = document.createElement("div"); fl.className = "filters";
  fl.innerHTML = '<label><input type="checkbox" id="evAll"> every event (not only the ones with a log line)</label>'; det.append(fl);
  const list = document.createElement("div"); list.className = "evlist"; det.append(list);
  const render = () => {
    list.innerHTML = "";
    const all = $("evAll").checked;
    events.filter((e) => all || e.line).forEach((e) => {
      const row = document.createElement("div");
      const cls = e.ev.startsWith("tool") ? "tool" : e.ev.startsWith("asr") ? "asr" : e.ev.startsWith("tts") ? "tts" : e.ev.includes("error") ? "err" : "";
      row.innerHTML = '<span class="t"></span><span></span>'; row.firstChild.textContent = `+${(e.wall - t0).toFixed(2)}s`;
      row.lastChild.className = cls; row.lastChild.textContent = e.line || `${e.ev} ${e.data ? cut(e.data, 160) : ""}`;
      if (e.data) row.title = e.data;
      list.append(row);
    });
  };
  $("evAll").addEventListener("change", render); render();
}

boot();
