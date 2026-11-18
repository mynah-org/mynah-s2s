const state = { scenarios: {}, ws: null, transport: "demo", profile: "demo", gpuConfigured: false, connected: false, closing: false, sessionId: null, callId: null, mic: null, sampleRun: null, audioIn: null, audioOut: null, nextPlayback: 0, events: 0, t0: 0, seenText: "" };
const $ = (id) => document.getElementById(id);

async function boot() {
  const data = await fetch("/api/scenarios").then((r) => r.json());
  data.scenarios.forEach((s) => state.scenarios[s.id] = s);
  const select = $("scenarioSelect");
  data.scenarios.forEach((s) => { const option = document.createElement("option"); option.value = s.id; option.textContent = s.name; select.append(option); });
  select.addEventListener("change", () => { if (state.ws && state.ws.readyState < 2) resetSession(); updateScenario(); });
  $("transportSelect").addEventListener("change", () => { if (state.ws && state.ws.readyState < 2) resetSession(); state.transport = $("transportSelect").value; updateProfile(state.transport === "gpu" ? state.profile : "demo", false); setConnection("idle", state.transport === "gpu" ? "V2 GPU bridge offline" : "Local demo offline"); setAgent("idle", "Voice agent offline"); updateScenario(); updateMicNote(); updateMicButton(); });
  $("inputSelect").addEventListener("change", () => { if (state.sampleRun) stopSample(); updateScenario(); updateMicNote(); updateMicButton(); });
  $("connectButton").addEventListener("click", connect);
  $("micButton").addEventListener("click", toggleMic);
  $("trySuggestion").addEventListener("click", trySuggestion);
  $("resetButton").addEventListener("click", resetSession);
  $("promptToggle").addEventListener("change", () => $("promptEditor").classList.toggle("visible", $("promptToggle").checked));
  updateScenario();
  updateMicButton();
  updateMicNote();
  const config = await fetch("/api/config").then((r) => r.json());
  state.profile = config.profile || "demo";
  updateProfile(state.transport === "gpu" ? state.profile : "demo", false);
  state.gpuConfigured = Boolean(config.gpu_configured && config.v2_reachable !== false);
  state.wsPort = config.ws_port ? String(config.ws_port) : null;
  if (!config.gpu_configured) { [...$("transportSelect").options].find((o) => o.value === "gpu").textContent = "GPU · not configured"; }
  else if (!state.gpuConfigured) { [...$("transportSelect").options].find((o) => o.value === "gpu").textContent = "GPU · offline"; }
  updateScenario();
}

function currentScenario() { return state.scenarios[$("scenarioSelect").value]; }
function updateScenario() {
  const s = currentScenario(); if (!s) return;
  $("suggestionText").textContent = s.suggestion; $("scenarioDescription").textContent = s.description; $("promptEditor").value = s.prompt;
  $("toolName").textContent = s.tool.name; $("stageLabel").textContent = "Connect, then choose how to start";
  const input = $("inputSelect"); const previous = input.value;
  input.innerHTML = '<option value="mic">Microphone</option>';
  (s.samples || []).filter((sample) => sample.available).forEach((sample) => {
    const option = document.createElement("option"); option.value = `sample:${sample.id}`; option.textContent = `Guided · ${sample.label}`; input.append(option);
  });
  if ([...input.options].some((option) => option.value === previous)) input.value = previous;
  else if (s.samples?.some((sample) => sample.available)) input.value = `sample:${s.samples.find((sample) => sample.available).id}`;
  const sample = selectedSample();
  $("sampleNote").hidden = !sample;
  if (sample) $("sampleText").textContent = `${sample.label} — “${sample.text}”`;
  updateMicButton(); updateMicNote();
}
function selectedSample() { const value = $("inputSelect").value; const id = value.startsWith("sample:") ? value.slice(7) : null; return (currentScenario()?.samples || []).find((sample) => sample.id === id); }
function updateProfile(profile, online) { $("profileLabel").textContent = ({int4:"INT4", q8:"Q8", bf16:"BF16", demo:"LOCAL"}[profile] || profile.toUpperCase()); $("profileDetail").textContent = profile === "demo" ? "guided simulation" : online ? "server session active" : "configured server profile"; $("profileCard").classList.toggle("online", online); }
function setConnection(kind, label) { const dot = $("connectionDot"); dot.className = "connection-dot " + (kind === "ok" ? "ok" : kind === "error" ? "error" : ""); $("connectionLabel").textContent = label; }
function setAgent(kind, label) { const dot = $("agentDot"); dot.className = "connection-dot " + (kind === "ok" ? "ok" : kind === "error" ? "error" : ""); $("agentLabel").textContent = label; }
function setSession(label, active = false) { $("sessionState").textContent = label; $("sessionState").classList.toggle("active", active); $("stageLabel").textContent = label; }
function setIds() { $("sessionId").textContent = state.sessionId ? `session ${state.sessionId}` : "session —"; $("callId").textContent = state.callId ? `call ${state.callId}` : "call —"; }
function setMicNote(label) { $("micNote").textContent = label; }
function updateMicButton() {
  if (state.transport === "demo") $("micButton").innerHTML = "Run guided example";
  else if (selectedSample()) $("micButton").innerHTML = state.sampleRun ? '<span class="mic-icon">■</span> Stop sample' : '<span class="mic-icon">▶</span> Play guided sample';
  else $("micButton").innerHTML = state.mic ? '<span class="mic-icon">●</span> Stop microphone' : '<span class="mic-icon">●</span> Start microphone';
}
function updateMicNote() {
  setMicNote(state.transport === "gpu"
    ? selectedSample() ? "Real corpus audio will be streamed to V2; its ASR, tools, and TTS remain live." : "Connect to V2, then start microphone. Mic input needs localhost or HTTPS."
    : "Connect first, then run the guided example. No microphone permission is needed for the demo.");
}

function requestSocketClose(reason = "user") {
  const ws = state.ws;
  if (!ws) return;
  state.closing = true;
  ws.onmessage = null;
  if (ws.readyState === WebSocket.OPEN) {
    try { ws.send(JSON.stringify({type:"session.cancel", reason})); } catch (_) {}
    // Let the cancel frame leave before closing the browser side of the tunnel.
    window.setTimeout(() => { if (ws.readyState < 2) ws.close(1000, reason); }, 80);
  } else if (ws.readyState < 2) ws.close(1000, reason);
}
function stopAudioOutput() { if (state.audioOut) { state.audioOut.close().catch(() => {}); state.audioOut = null; } state.nextPlayback = 0; }
function connect() {
  if (state.ws && state.ws.readyState < 2) { requestSocketClose("user_disconnect"); $("connectButton").textContent = "Disconnecting…"; return; }
  state.transport = $("transportSelect").value;
  if (state.transport === "gpu" && !state.gpuConfigured) { setConnection("error", "V2 bridge not configured"); setAgent("error", "Voice agent unavailable"); setSession("GPU not configured"); setMicNote("This page is not connected to a GPU bridge. Open the SSH-tunnel URL, then refresh."); $("micButton").disabled = true; return; }
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  // Ask the browser for a 16 kHz capture context, matching the V2 client
  // contract. If a browser chooses another rate, advertise that exact rate
  // and let the V2 transport resample it server-side.
  if (state.transport === "gpu" && !state.audioIn) { try { state.audioIn = new AudioContext({sampleRate:16000}); } catch (_) { state.audioIn = new AudioContext(); } }
  const ws = new WebSocket(`${protocol}://${location.hostname}:${state.wsPort || (location.port === "8787" ? "8788" : location.port)}/ws`);
  state.ws = ws; state.closing = false; ws.binaryType = "arraybuffer";
  ws.onopen = () => { setConnection("ok", state.transport === "gpu" ? "V2 bridge connected" : "Demo bridge connected"); $("connectButton").textContent = "Disconnect"; $("micButton").disabled = false; setSession("Opening agent…", true); state.t0 = performance.now(); const rate = state.audioIn?.sampleRate || 16000; ws.send(JSON.stringify({type:"session.start", transport:state.transport, scenario:currentScenario().id, system_prompt:$("promptEditor").value, input_rate:rate, language:$("languageSelect")?.value || undefined, voice:$("voiceSelect")?.value || undefined})); };
  ws.onmessage = (e) => { if (typeof e.data === "string") { try { handleJson(JSON.parse(e.data)); } catch (_) { setSession("Invalid bridge message"); } } else playPcm(e.data); };
  ws.onerror = () => { if (state.ws !== ws) return; setConnection("error", state.transport === "gpu" ? "V2 bridge unreachable" : "Demo bridge unreachable"); setAgent("error", "Voice agent offline"); setSession("Connection error"); $("micButton").disabled = true; };
  ws.onclose = () => { if (state.ws !== ws) return; state.ws = null; state.connected = false; state.closing = false; $("connectButton").textContent = "Reconnect"; $("micButton").disabled = true; stopMic(); stopSample(); stopAudioOutput(); updateProfile(state.transport === "gpu" ? state.profile : "demo", false); setAgent("idle", "Voice agent offline"); setConnection("idle", state.transport === "gpu" ? "V2 bridge offline" : "Demo bridge offline"); setSession("Disconnected"); updateMicNote(); };
}

function handleJson(msg) {
  if (msg.type === "error") { setAgent("error", "Voice agent error"); setSession(msg.error || "Connection error"); setMicNote(msg.error || "The bridge returned an error."); return; }
  if (msg.type === "session.created") { state.connected = true; state.sessionId = msg.session_id || null; state.callId = null; setIds(); if (msg.backend_profile) state.profile = msg.backend_profile; updateProfile(msg.mode === "demo" ? "demo" : state.profile, true); setConnection("ok", msg.mode === "demo" ? "Demo bridge connected" : `V2 ${msg.backend_profile_label || state.profile.toUpperCase()} connected`); setAgent("ok", msg.mode === "demo" ? "Voice agent demo ready" : "V2 voice agent online"); setSession("Ready to talk", true); updateMicNote(); return; }
  if (msg.type === "session.closed") { setAgent("idle", "Voice agent offline"); setConnection("idle", "Bridge disconnected"); setSession("Session closed"); stopMic(); stopSample(); stopAudioOutput(); updateProfile(state.transport === "gpu" ? state.profile : "demo", false); updateMicNote(); return; }
  if (msg.type !== "event") return;
  if (msg.session_id) state.sessionId = msg.session_id;
  if (msg.call_id) state.callId = msg.call_id;
  setIds();
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
  const trace = msg.call_id ? `call ${msg.call_id}` : msg.tool || msg.status || msg.phase || "event";
  row.innerHTML = `<div class="activity-icon"></div><div class="activity-copy"><b></b><small></small></div><div class="activity-time"></div>${detail ? `<div class="activity-detail"></div>` : ""}`;
  row.querySelector(".activity-icon").textContent = icon;
  row.querySelector("b").textContent = labels[ev] || ev;
  row.querySelector("small").textContent = trace;
  row.querySelector(".activity-time").textContent = formatTime(msg);
  if (detail) row.querySelector(".activity-detail").textContent = typeof detail === "string" ? detail : JSON.stringify(detail, null, 2);
  list.append(row); list.scrollTop = list.scrollHeight; if (state.t0) $("latencyValue").textContent = `${Math.round(performance.now() - state.t0)} ms`;
}
function formatTime(msg) { return msg.t_ns ? `${((msg.t_ns % 60000000000) / 1000000000).toFixed(1)}s` : "now"; }

async function toggleMic() {
  if (state.transport === "demo") { trySuggestion(); return; }
  if (selectedSample()) { if (state.sampleRun) stopSample(); else await playSample(); return; }
  if (state.mic) { stopMic(); setMicNote("Microphone paused. Press Start microphone to resume."); return; }
  if (!state.ws || state.ws.readyState !== 1) { setMicNote("Connect to the V2 bridge first."); return; }
  if (!window.isSecureContext && !["localhost", "127.0.0.1", "::1"].includes(location.hostname)) { setMicNote("Microphone needs a secure context. Use the SSH localhost tunnel described in README."); setSession("Secure context required"); return; }
  if (!navigator.mediaDevices?.getUserMedia) { setMicNote("This browser does not expose microphone capture on this page."); return; }
  try { const stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1, echoCancellation:true, noiseSuppression:true}}); const ctx = state.audioIn || new AudioContext({sampleRate:16000}); await ctx.resume(); const src = ctx.createMediaStreamSource(stream); const processor = ctx.createScriptProcessor(2048,1,1); const mute = ctx.createGain(); mute.gain.value = 0; processor.onaudioprocess = (e) => { if (!state.ws || state.ws.readyState !== 1) return; const input = e.inputBuffer.getChannelData(0); const pcm = new Int16Array(input.length); for (let i=0;i<input.length;i++) pcm[i] = Math.max(-1, Math.min(1, input[i])) * 32767; state.ws.send(pcm.buffer); }; src.connect(processor); processor.connect(mute); mute.connect(ctx.destination); state.audioIn = ctx; state.mic = {stream,ctx,processor}; $("micButton").innerHTML = '<span class="mic-icon">●</span> Stop microphone'; $("waveform").classList.add("active"); setMicNote("Microphone active — speak naturally."); setSession("Listening", true); } catch (e) { setMicNote(`Microphone unavailable: ${e.name || "permission denied"}. Check browser permissions and use localhost/HTTPS.`); setSession("Microphone unavailable"); }
}
function stopMic() { if (!state.mic) return; state.mic.stream.getTracks().forEach((t) => t.stop()); state.mic.ctx.close(); state.mic = null; state.audioIn = null; updateMicButton(); $("waveform").classList.remove("active"); }
async function playSample() {
  const sample = selectedSample();
  if (!sample || !sample.available || !state.ws || state.ws.readyState !== 1) { setMicNote("Connect to the GPU server and choose an available guided sample first."); return; }
  const run = {cancelled:false}; state.sampleRun = run; updateMicButton(); $("waveform").classList.add("active"); setSession("Streaming guided input", true); setMicNote(`Streaming ${sample.label} to the real V2 server…`);
  let ctx;
  try {
    const response = await fetch(`/api/samples/${encodeURIComponent(currentScenario().id)}/${encodeURIComponent(sample.id)}`);
    if (!response.ok) throw new Error("sample unavailable");
    const data = await response.arrayBuffer();
    ctx = new AudioContext();
    const decoded = await ctx.decodeAudioData(data.slice(0));
    const input = decoded.getChannelData(0);
    const rate = decoded.sampleRate;
    const chunk = Math.max(1, Math.round(rate * 0.08));
    const started = performance.now();
    state.ws.send(JSON.stringify({type:"guided.input", sample_id:sample.id, sample_text:sample.text, input_rate:rate}));
    for (let offset = 0; offset < input.length && !run.cancelled; offset += chunk) {
      const target = started + (offset / rate) * 1000;
      const wait = target - performance.now();
      if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
      const end = Math.min(offset + chunk, input.length); const pcm = new Int16Array(end - offset);
      for (let i = offset; i < end; i++) pcm[i - offset] = Math.max(-1, Math.min(1, input[i])) * 32767;
      if (state.ws?.readyState === 1) state.ws.send(pcm.buffer);
    }
    if (!run.cancelled) { setSession("Waiting for agent", true); setMicNote("Sample complete — the agent is processing and speaking back."); }
  } catch (error) {
    if (!run.cancelled) { setSession("Sample unavailable"); setMicNote(`Guided sample failed: ${error.message || error}`); }
  } finally {
    if (ctx) ctx.close();
    if (state.sampleRun === run) state.sampleRun = null;
    $("waveform").classList.remove("active"); updateMicButton();
  }
}
function stopSample() { if (state.sampleRun) state.sampleRun.cancelled = true; state.sampleRun = null; $("waveform").classList.remove("active"); updateMicButton(); }
function playPcm(buffer) { if (!state.audioOut) state.audioOut = new AudioContext({sampleRate:22050}); const ctx = state.audioOut; if (ctx.state === "suspended") ctx.resume(); const pcm = new Int16Array(buffer); const data = ctx.createBuffer(1, pcm.length, 22050); const channel = data.getChannelData(0); for (let i=0;i<pcm.length;i++) channel[i] = pcm[i] / 32768; const src = ctx.createBufferSource(); src.buffer = data; src.connect(ctx.destination); state.nextPlayback = Math.max(state.nextPlayback, ctx.currentTime) + data.duration; src.start(Math.max(ctx.currentTime, state.nextPlayback - data.duration)); (state.sources = state.sources || []).push(src); src.onended = () => { state.sources = (state.sources || []).filter((x) => x !== src); }; }
// Frankenstein engine: a barge-in cancels the reply server-side and sends "audio.flush": drop what is still queued
function flushPlayback() { (state.sources || []).forEach((src) => { try { src.stop(); } catch (_) {} }); state.sources = []; if (state.audioOut) state.nextPlayback = state.audioOut.currentTime; }
function fillVoices(list) { const sel = $("voiceSelect"); if (!sel || !Array.isArray(list)) return; const keep = sel.value; sel.innerHTML = '<option value="">Default</option>'; list.forEach((v) => { const o = document.createElement("option"); o.value = v; o.textContent = v; sel.append(o); }); sel.value = keep; }
function speakDemo(text) { if (state.transport !== "demo" || !window.speechSynthesis || !text) return; window.speechSynthesis.cancel(); const voices = window.speechSynthesis.getVoices(); const voice = voices.find((v) => /^en-US$/i.test(v.lang) && /Samantha|Alex|Google US English/i.test(v.name)) || voices.find((v) => /^en(-|_)/i.test(v.lang)); const utterance = new SpeechSynthesisUtterance(text); utterance.lang = "en-US"; if (voice) utterance.voice = voice; utterance.rate = .96; utterance.pitch = 1; window.speechSynthesis.speak(utterance); }
function trySuggestion() { if (!state.ws || state.ws.readyState !== 1) { setMicNote("Connect first, then run the guided example."); setSession("Connect first"); return; } if (state.transport !== "demo") { if (selectedSample()) playSample(); else setMicNote("Choose a guided sample in INPUT, or start the microphone."); return; } state.ws.send(JSON.stringify({type:"demo.request", scenario:currentScenario().id})); setMicNote("Running guided example — watch the activity stream."); setSession("Agent working", true); }
function resetSession() { if (state.ws) requestSocketClose("reset"); stopMic(); stopSample(); stopAudioOutput(); state.sessionId = null; state.callId = null; setIds(); state.events = 0; $("eventCount").textContent = "0"; $("latencyValue").textContent = "—"; $("activityList").innerHTML = '<div class="empty-state compact"><span class="empty-icon">⌁</span><p>No events yet</p><small>Tool calls and model phases will appear here.</small></div>'; $("transcript").innerHTML = '<div class="empty-state"><span class="empty-icon">◌</span><p>Your conversation will appear here.</p><small>Connect, then start the microphone or run a guided sample.</small></div>'; setConnection("idle", state.transport === "gpu" ? "V2 GPU bridge offline" : "Local demo offline"); setAgent("idle", "Voice agent offline"); updateProfile(state.transport === "gpu" ? state.profile : "demo", false); $("connectButton").textContent = "Connect"; $("micButton").disabled = true; setSession("Ready"); updateMicNote(); }
setIds();
boot();
