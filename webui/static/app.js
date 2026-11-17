const state = { scenarios: {}, ws: null, transport: "demo", gpuConfigured: false, connected: false, mic: null, audioIn: null, audioOut: null, nextPlayback: 0, events: 0, t0: 0, seenText: "" };
const $ = (id) => document.getElementById(id);

async function boot() {
  const data = await fetch("/api/scenarios").then((r) => r.json());
  data.scenarios.forEach((s) => state.scenarios[s.id] = s);
  const select = $("scenarioSelect");
  data.scenarios.forEach((s) => { const option = document.createElement("option"); option.value = s.id; option.textContent = s.name; select.append(option); });
  select.addEventListener("change", () => { if (state.ws && state.ws.readyState < 2) resetSession(); updateScenario(); });
  $("transportSelect").addEventListener("change", () => { if (state.ws && state.ws.readyState < 2) resetSession(); state.transport = $("transportSelect").value; setConnection("idle", state.transport === "gpu" ? "V2 GPU bridge offline" : "Local demo offline"); setAgent("idle", "Voice agent offline"); updateMicNote(); updateMicButton(); });
  $("connectButton").addEventListener("click", connect);
  $("micButton").addEventListener("click", toggleMic);
  $("trySuggestion").addEventListener("click", trySuggestion);
  $("resetButton").addEventListener("click", resetSession);
  $("promptToggle").addEventListener("change", () => $("promptEditor").classList.toggle("visible", $("promptToggle").checked));
  updateScenario();
  updateMicButton();
  updateMicNote();
  const config = await fetch("/api/config").then((r) => r.json());
  state.gpuConfigured = Boolean(config.gpu_configured && config.v2_reachable !== false);
  state.wsPort = config.ws_port ? String(config.ws_port) : null;
  if (!config.gpu_configured) { [...$("transportSelect").options].find((o) => o.value === "gpu").textContent = "GPU · not configured"; }
  else if (!state.gpuConfigured) { [...$("transportSelect").options].find((o) => o.value === "gpu").textContent = "GPU · offline"; }
}

function currentScenario() { return state.scenarios[$("scenarioSelect").value]; }
function updateScenario() {
  const s = currentScenario(); if (!s) return;
  $("suggestionText").textContent = s.suggestion; $("scenarioDescription").textContent = s.description; $("promptEditor").value = s.prompt;
  $("toolName").textContent = s.tool.name; $("stageLabel").textContent = "Connect, then choose how to start";
}
function setConnection(kind, label) { const dot = $("connectionDot"); dot.className = "connection-dot " + (kind === "ok" ? "ok" : kind === "error" ? "error" : ""); $("connectionLabel").textContent = label; }
function setAgent(kind, label) { const dot = $("agentDot"); dot.className = "connection-dot " + (kind === "ok" ? "ok" : kind === "error" ? "error" : ""); $("agentLabel").textContent = label; }
function setSession(label, active = false) { $("sessionState").textContent = label; $("sessionState").classList.toggle("active", active); $("stageLabel").textContent = label; }
function setMicNote(label) { $("micNote").textContent = label; }
function updateMicButton() { $("micButton").innerHTML = state.transport === "demo" ? "Run guided example" : '<span class="mic-icon">●</span> Start microphone'; }
function updateMicNote() {
  setMicNote(state.transport === "gpu"
    ? "Connect to V2, then start microphone. Mic input needs localhost or HTTPS."
    : "Connect first, then run the guided example. No microphone permission is needed for the demo.");
}

function connect() {
  if (state.ws && state.ws.readyState < 2) { state.ws.close(); return; }
  state.transport = $("transportSelect").value;
  if (state.transport === "gpu" && !state.gpuConfigured) { setConnection("error", "V2 bridge not configured"); setAgent("error", "Voice agent unavailable"); setSession("GPU not configured"); setMicNote("This page is not connected to a GPU bridge. Open the SSH-tunnel URL, then refresh."); $("micButton").disabled = true; return; }
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  // Ask the browser for a 16 kHz capture context, matching the V2 client
  // contract. If a browser chooses another rate, advertise that exact rate
  // and let the V2 transport resample it server-side.
  if (state.transport === "gpu" && !state.audioIn) { try { state.audioIn = new AudioContext({sampleRate:16000}); } catch (_) { state.audioIn = new AudioContext(); } }
  state.ws = new WebSocket(`${protocol}://${location.hostname}:${state.wsPort || (location.port === "8787" ? "8788" : location.port)}/ws`);
  state.ws.binaryType = "arraybuffer";
  state.ws.onopen = () => { setConnection("ok", state.transport === "gpu" ? "V2 bridge connected" : "Demo bridge connected"); $("connectButton").textContent = "Disconnect"; $("micButton").disabled = false; setSession("Opening agent…", true); state.t0 = performance.now(); const rate = state.audioIn?.sampleRate || 16000; state.ws.send(JSON.stringify({type:"session.start", transport:state.transport, scenario:currentScenario().id, system_prompt:$("promptEditor").value, input_rate:rate, language:$("languageSelect")?.value || undefined, voice:$("voiceSelect")?.value || undefined})); };
  state.ws.onmessage = (e) => { if (typeof e.data === "string") handleJson(JSON.parse(e.data)); else playPcm(e.data); };
  state.ws.onerror = () => { setConnection("error", state.transport === "gpu" ? "V2 bridge unreachable" : "Demo bridge unreachable"); setAgent("error", "Voice agent offline"); setSession("Connection error"); $("micButton").disabled = true; };
  state.ws.onclose = () => { state.connected = false; $("connectButton").textContent = "Reconnect"; $("micButton").disabled = true; stopMic(); setAgent("idle", "Voice agent offline"); setConnection("idle", state.transport === "gpu" ? "V2 bridge offline" : "Demo bridge offline"); setSession("Disconnected"); updateMicNote(); };
}

function handleJson(msg) {
  if (msg.type === "error") { setAgent("error", "Voice agent error"); setSession(msg.error || "Connection error"); setMicNote(msg.error || "The bridge returned an error."); return; }
  if (msg.type === "session.created") { state.connected = true; setConnection("ok", msg.mode === "demo" ? "Demo bridge connected" : "V2 GPU connected"); setAgent("ok", msg.mode === "demo" ? "Voice agent demo ready" : "V2 voice agent online"); setSession("Ready to talk", true); updateMicNote(); return; }
  if (msg.type === "session.closed") { setAgent("idle", "Voice agent offline"); setConnection("idle", "Bridge disconnected"); setSession("Session closed"); stopMic(); updateMicNote(); return; }
  if (msg.type !== "event") return;
  state.events++; $("eventCount").textContent = state.events;
  const ev = msg.ev || "event";
  if (ev === "audio.flush") flushPlayback();                 // Frankenstein engine: the reply was cut (barge-in)
  if (ev === "engine.session" && msg.voices) fillVoices(msg.voices);
  if (ev === "asr.generated") addMessage("user", msg.text, "ASR");
  if (ev === "text.generated") { addMessage("agent", msg.text, "AGENT"); speakDemo(msg.text); setSession("Agent speaking", true); }
  if (ev === "thinker.step" || ev === "function.started") setSession("Agent thinking", true);
  if (ev.startsWith("tool.")) setSession(ev === "tool.completed" ? "Tool completed" : "Tool call running", true);
  addActivity(msg);
}

function addMessage(role, text, meta) {
  const box = $("transcript"); box.querySelector(".empty-state")?.remove();
  const row = document.createElement("div"); row.className = `message ${role}`;
  row.innerHTML = `<div class="message-avatar">${role === "agent" ? "✦" : "U"}</div><div><div class="message-body"></div><div class="message-meta">${meta}</div></div>`;
  row.querySelector(".message-body").textContent = text || ""; box.append(row); box.scrollTop = box.scrollHeight;
}

function addActivity(msg) {
  const list = $("activityList"); list.querySelector(".empty-state")?.remove();
  const ev = msg.ev; const isTool = ev.startsWith("tool."); const isThink = ev === "thinker.step" || ev === "function.started";
  const row = document.createElement("div"); row.className = `activity ${isTool ? "tool" : isThink ? "think" : ""}`;
  const labels = {"thinker.step":"Think", "function.started":"Decision", "asr.generated":"Listen", "text.generated":"Speak", "tool.pending":"Tool pending", "tool.running":"Tool running", "tool.completed":"Tool result", "tool.result.injected":"Grounding", "tool.result.consumed":"Consumed"};
  const icon = isTool ? "⌘" : isThink ? "◌" : ev === "text.generated" ? "↗" : "·";
  const detail = msg.args || msg.result || msg.text || (msg.phase ? `${msg.phase} · step ${msg.model_step || "—"}` : "");
  row.innerHTML = `<div class="activity-icon">${icon}</div><div class="activity-copy"><b>${labels[ev] || ev}</b><small>${msg.tool || msg.status || msg.phase || "event"}</small></div><div class="activity-time">${formatTime(msg)}</div>${detail ? `<div class="activity-detail"></div>` : ""}`;
  if (detail) row.querySelector(".activity-detail").textContent = typeof detail === "string" ? detail : JSON.stringify(detail, null, 2);
  list.append(row); list.scrollTop = list.scrollHeight; if (state.t0) $("latencyValue").textContent = `${Math.round(performance.now() - state.t0)} ms`;
}
function formatTime(msg) { return msg.t_ns ? `${((msg.t_ns % 60000000000) / 1000000000).toFixed(1)}s` : "now"; }

async function toggleMic() {
  if (state.transport === "demo") { trySuggestion(); return; }
  if (state.mic) { stopMic(); setMicNote("Microphone paused. Press Start microphone to resume."); return; }
  if (!state.ws || state.ws.readyState !== 1) { setMicNote("Connect to the V2 bridge first."); return; }
  if (!window.isSecureContext && !["localhost", "127.0.0.1", "::1"].includes(location.hostname)) { setMicNote("Microphone needs a secure context. Use the SSH localhost tunnel described in README."); setSession("Secure context required"); return; }
  if (!navigator.mediaDevices?.getUserMedia) { setMicNote("This browser does not expose microphone capture on this page."); return; }
  try { const stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1, echoCancellation:true, noiseSuppression:true}}); const ctx = state.audioIn || new AudioContext({sampleRate:16000}); await ctx.resume(); const src = ctx.createMediaStreamSource(stream); const processor = ctx.createScriptProcessor(2048,1,1); const mute = ctx.createGain(); mute.gain.value = 0; processor.onaudioprocess = (e) => { if (!state.ws || state.ws.readyState !== 1) return; const input = e.inputBuffer.getChannelData(0); const pcm = new Int16Array(input.length); for (let i=0;i<input.length;i++) pcm[i] = Math.max(-1, Math.min(1, input[i])) * 32767; state.ws.send(pcm.buffer); }; src.connect(processor); processor.connect(mute); mute.connect(ctx.destination); state.audioIn = ctx; state.mic = {stream,ctx,processor}; $("micButton").innerHTML = '<span class="mic-icon">●</span> Stop microphone'; $("waveform").classList.add("active"); setMicNote("Microphone active — speak naturally."); setSession("Listening", true); } catch (e) { setMicNote(`Microphone unavailable: ${e.name || "permission denied"}. Check browser permissions and use localhost/HTTPS.`); setSession("Microphone unavailable"); }
}
function stopMic() { if (!state.mic) return; state.mic.stream.getTracks().forEach((t) => t.stop()); state.mic.ctx.close(); state.mic = null; state.audioIn = null; updateMicButton(); $("waveform").classList.remove("active"); }
function playPcm(buffer) { if (!state.audioOut) state.audioOut = new AudioContext({sampleRate:22050}); const ctx = state.audioOut; if (ctx.state === "suspended") ctx.resume(); const pcm = new Int16Array(buffer); const data = ctx.createBuffer(1, pcm.length, 22050); const channel = data.getChannelData(0); for (let i=0;i<pcm.length;i++) channel[i] = pcm[i] / 32768; const src = ctx.createBufferSource(); src.buffer = data; src.connect(ctx.destination); state.nextPlayback = Math.max(state.nextPlayback, ctx.currentTime) + data.duration; src.start(Math.max(ctx.currentTime, state.nextPlayback - data.duration)); (state.sources = state.sources || []).push(src); src.onended = () => { state.sources = (state.sources || []).filter((x) => x !== src); }; }
// Frankenstein engine: a barge-in cancels the reply server-side and sends "audio.flush": drop what is still queued
function flushPlayback() { (state.sources || []).forEach((src) => { try { src.stop(); } catch (_) {} }); state.sources = []; if (state.audioOut) state.nextPlayback = state.audioOut.currentTime; }
function fillVoices(list) { const sel = $("voiceSelect"); if (!sel || !Array.isArray(list)) return; const keep = sel.value; sel.innerHTML = '<option value="">Default</option>'; list.forEach((v) => { const o = document.createElement("option"); o.value = v; o.textContent = v; sel.append(o); }); sel.value = keep; }
function speakDemo(text) { if (state.transport !== "demo" || !window.speechSynthesis || !text) return; window.speechSynthesis.cancel(); const voices = window.speechSynthesis.getVoices(); const voice = voices.find((v) => /^en-US$/i.test(v.lang) && /Samantha|Alex|Google US English/i.test(v.name)) || voices.find((v) => /^en(-|_)/i.test(v.lang)); const utterance = new SpeechSynthesisUtterance(text); utterance.lang = "en-US"; if (voice) utterance.voice = voice; utterance.rate = .96; utterance.pitch = 1; window.speechSynthesis.speak(utterance); }
function trySuggestion() { if (!state.ws || state.ws.readyState !== 1) { setMicNote("Connect first, then run the guided example."); setSession("Connect first"); return; } if (state.transport !== "demo") { setMicNote("This example runs in Local simulation. For GPU, speak through the microphone."); return; } state.ws.send(JSON.stringify({type:"demo.request", scenario:currentScenario().id})); setMicNote("Running guided example — watch the activity stream."); setSession("Agent working", true); }
function resetSession() { const ws = state.ws; state.ws = null; if (ws) { ws.onclose = null; ws.close(); } stopMic(); state.events = 0; $("eventCount").textContent = "0"; $("latencyValue").textContent = "—"; $("activityList").innerHTML = '<div class="empty-state compact"><span class="empty-icon">⌁</span><p>No events yet</p><small>Tool calls and model phases will appear here.</small></div>'; $("transcript").innerHTML = '<div class="empty-state"><span class="empty-icon">◌</span><p>Your conversation will appear here.</p><small>Connect, then start the microphone or run the guided example.</small></div>'; setConnection("idle", state.transport === "gpu" ? "V2 GPU bridge offline" : "Local demo offline"); setAgent("idle", "Voice agent offline"); $("connectButton").textContent = "Connect"; $("micButton").disabled = true; setSession("Ready"); updateMicNote(); }
boot();
