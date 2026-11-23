import os
import sys
import time
import json
import re
import base64
import requests
import pyttsx3
import webbrowser
from threading import Timer
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

app = FastAPI(title="JARVIS OMEGA // HOLOGRAPHIC CORE")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AGENTS = {
    "Commander": {
        "title": "👑 Commander Agent",
        "role": "Master Orchestrator & Strategic Planning",
        "system": "You are JARVIS OMEGA Commander Agent. Speak authoritatively, calmly, and concisely with maximum strategic precision. Address the user as Sir."
    },
    "Researcher": {
        "title": "🔬 Researcher Agent",
        "role": "Deep Information Synthesis & Knowledge Retrieval",
        "system": "You are JARVIS OMEGA Researcher Agent. You specialize in deep scientific and technical synthesis. Be concise, factual, and analytical."
    },
    "Executor": {
        "title": "⚡ Executor Agent",
        "role": "Code Generation & System Execution",
        "system": "You are JARVIS OMEGA Executor Agent. You write robust, production-grade code with zero fluff."
    },
    "Guardian": {
        "title": "🛡️ Guardian Agent",
        "role": "Zero-Knowledge Security & Threat Detection",
        "system": "You are JARVIS OMEGA Guardian Agent. You audit privacy, network threats, and safeguard systems."
    },
    "Creative": {
        "title": "🎨 Creative Agent",
        "role": "Multimodal Synthesis & Generative UI",
        "system": "You are JARVIS OMEGA Creative Agent. You design futuristic interfaces and imaginative solutions."
    },
    "Toolsmith": {
        "title": "🛠️ Toolsmith Agent",
        "role": "Dynamic Tool Creation & API Integration",
        "system": "You are JARVIS OMEGA Toolsmith Agent. You automate workflows and integrate tools on the fly."
    },
    "Monetizer": {
        "title": "💰 Monetizer Agent",
        "role": "Economic Optimization & Workflows",
        "system": "You are JARVIS OMEGA Monetizer Agent. You analyze market viable revenue and optimize business logic."
    }
}

def tts_b64(text: str) -> str:
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        voices = engine.getProperty("voices")
        for v in voices:
            if "hazel" in v.name.lower() or "david" in v.name.lower() or "great britain" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        clean = re.sub(r'[*_#`~]', '', text)
        clean = re.sub(r'\[.*?\]\(.*?\)', '', clean)[:350]
        fname = f"temp_{int(time.time()*1000)}.wav"
        engine.save_to_file(clean, fname)
        engine.runAndWait()
        b64 = ""
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            os.remove(fname)
        return b64
    except Exception as e:
        print("TTS Engine Error:", e)
        return ""

@app.get("/api/models")
def get_models():
    models = []
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                models.append(m["name"])
    except Exception:
        pass
    if not models:
        models = ["llama3.2:1b"]
    return {"models": models}

@app.post("/api/chat")
async def chat_endpoint(req: Request):
    data = await req.json()
    msg = data.get("message", "").strip()
    agent_name = data.get("agent", "Commander")
    model = data.get("model", "llama3.2:1b")
    voice_enabled = data.get("voice_enabled", True)
    history = data.get("history", [])

    if not msg:
        return JSONResponse({"response": "I am standing by, Sir. Please state your directive.", "audio": ""})

    agent_info = AGENTS.get(agent_name, AGENTS["Commander"])
    system_prompt = agent_info["system"]

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-6:]:
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": msg})

    answer = ""
    try:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7}
        }
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=45)
        if resp.status_code == 200:
            answer = resp.json().get("message", {}).get("content", "")
        else:
            answer = f"Error from Ollama ({resp.status_code}): {resp.text}"
    except Exception as e:
        answer = f"Sir, neural link connection error: {str(e)}. Please verify Ollama is active."

    audio_b64 = tts_b64(answer) if voice_enabled else ""
    return JSONResponse({
        "response": answer,
        "audio": audio_b64,
        "agent": agent_name
    })

@app.post("/api/vision-analyze")
async def vision_analyze_endpoint(req: Request):
    data = await req.json()
    directive = data.get("directive", "Describe what is visible in front of the camera.")
    agent_name = data.get("agent", "Commander")
    
    prompt = f"[Optical Sensor Telemetry Feed from HP HD Camera]: Analyze the optical environment. User Directive: {directive}"
    
    try:
        payload = {
            "model": "llama3.2:1b",
            "messages": [
                {"role": "system", "content": AGENTS[agent_name]["system"]},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=30)
        answer = resp.json().get("message", {}).get("content", "") if resp.status_code == 200 else "Optical analysis error."
    except Exception as e:
        answer = f"Optical processing offline: {str(e)}"
        
    audio_b64 = tts_b64(answer)
    return JSONResponse({"response": answer, "audio": audio_b64})

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JARVIS OMEGA // 3D HOLOGRAPHIC QUANTUM MATRIX</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
  body {
    background: #02050b;
    color: #00f0ff;
    font-family: 'Consolas', 'Courier New', monospace;
    overflow: hidden;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  
  /* Scanline Overlay */
  body::before {
    content: " ";
    position: absolute;
    top: 0; left: 0; bottom: 0; right: 0;
    background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.03), rgba(0, 255, 0, 0.01), rgba(0, 0, 255, 0.03));
    z-index: 99;
    background-size: 100% 3px, 6px 100%;
    pointer-events: none;
  }

  /* Header */
  header {
    height: 60px;
    background: rgba(3, 8, 18, 0.95);
    border-bottom: 1px solid rgba(0, 240, 255, 0.4);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.2);
    z-index: 10;
  }
  .hud-title {
    font-size: 18px;
    font-weight: 900;
    letter-spacing: 3px;
    text-shadow: 0 0 10px #00f0ff;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .hud-badge {
    font-size: 10px;
    padding: 4px 8px;
    border: 1px solid #00ffaa;
    color: #00ffaa;
    border-radius: 4px;
    background: rgba(0, 255, 170, 0.1);
    letter-spacing: 1px;
  }
  
  /* Agent Pills */
  .agent-bar {
    display: flex;
    gap: 8px;
    overflow-x: auto;
  }
  .agent-pill {
    padding: 5px 12px;
    border: 1px solid rgba(0, 240, 255, 0.3);
    border-radius: 20px;
    font-size: 11px;
    cursor: pointer;
    background: rgba(0, 240, 255, 0.05);
    transition: all 0.2s;
    color: #8ba2c4;
    white-space: nowrap;
  }
  .agent-pill.active {
    border-color: #00f0ff;
    color: #00f0ff;
    background: rgba(0, 240, 255, 0.2);
    box-shadow: 0 0 12px rgba(0, 240, 255, 0.5);
  }

  /* Main Viewport */
  main {
    flex: 1;
    display: flex;
    position: relative;
    overflow: hidden;
  }

  /* 3D Arc Reactor Background */
  #three-canvas {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 1;
  }

  /* Left Sensor HUD */
  .left-hud {
    width: 320px;
    background: rgba(2, 6, 15, 0.82);
    border-right: 1px solid rgba(0, 240, 255, 0.25);
    z-index: 5;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    backdrop-filter: blur(8px);
  }
  .panel-box {
    border: 1px solid rgba(0, 240, 255, 0.3);
    border-radius: 8px;
    padding: 12px;
    background: rgba(4, 10, 22, 0.85);
    position: relative;
  }
  .panel-label {
    font-size: 10px;
    letter-spacing: 2px;
    color: #00f0ff;
    margin-bottom: 8px;
    text-transform: uppercase;
    display: flex;
    justify-content: space-between;
  }
  
  /* Camera Viewfinder */
  #cam-container {
    position: relative;
    width: 100%;
    height: 160px;
    background: #000;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid rgba(0, 240, 255, 0.5);
  }
  #webcam-video {
    width: 100%; height: 100%;
    object-fit: cover;
  }
  .cam-reticle {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 50px; height: 50px;
    border: 1px dashed rgba(0, 240, 255, 0.6);
    border-radius: 50%;
    pointer-events: none;
  }
  
  /* Audio Spectrum */
  .spectrum-container {
    display: flex;
    align-items: flex-end;
    gap: 4px;
    height: 40px;
  }
  .spectrum-bar {
    flex: 1;
    background: #00f0ff;
    height: 15%;
    border-radius: 2px;
    transition: height 0.05s ease;
  }

  /* Center-Right Dialogue HUD */
  .right-hud {
    flex: 1;
    z-index: 5;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 20px 28px;
    pointer-events: none;
  }
  .dialogue-stream {
    max-height: calc(100vh - 220px);
    overflow-y: auto;
    pointer-events: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding-right: 12px;
  }
  .dialogue-stream::-webkit-scrollbar { width: 4px; }
  .dialogue-stream::-webkit-scrollbar-thumb { background: rgba(0, 240, 255, 0.4); border-radius: 2px; }

  .msg-card {
    padding: 14px 18px;
    border-radius: 10px;
    font-size: 13px;
    line-height: 1.5;
    max-width: 80%;
    backdrop-filter: blur(10px);
  }
  .msg-user {
    align-self: flex-end;
    background: rgba(0, 112, 243, 0.25);
    border: 1px solid #0070f3;
    color: #e0f0ff;
  }
  .msg-jarvis {
    align-self: flex-start;
    background: rgba(0, 240, 255, 0.12);
    border: 1px solid #00f0ff;
    color: #00f0ff;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.15);
  }

  /* Bottom Controls Bar */
  footer {
    height: 80px;
    background: rgba(3, 8, 18, 0.95);
    border-top: 1px solid rgba(0, 240, 255, 0.35);
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 0 28px;
    z-index: 10;
  }

  /* Holographic Mic Button */
  .hologram-mic-btn {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0, 240, 255, 0.4) 0%, rgba(3, 8, 18, 0.9) 80%);
    border: 2px solid #00f0ff;
    color: #00f0ff;
    font-size: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.4);
    transition: all 0.25s ease;
    flex-shrink: 0;
  }
  .hologram-mic-btn:hover {
    transform: scale(1.08);
    box-shadow: 0 0 35px rgba(0, 240, 255, 0.8);
  }
  .hologram-mic-btn.listening {
    border-color: #ff3366;
    color: #ff3366;
    background: radial-gradient(circle, rgba(255, 51, 102, 0.4) 0%, rgba(3, 8, 18, 0.9) 80%);
    animation: micPulse 1.2s infinite alternate;
  }
  @keyframes micPulse {
    from { box-shadow: 0 0 10px #ff3366; }
    to { box-shadow: 0 0 35px #ff3366, 0 0 50px #ff0055; }
  }

  .input-box {
    flex: 1;
    height: 48px;
    background: rgba(2, 6, 16, 0.8);
    border: 1px solid rgba(0, 240, 255, 0.4);
    border-radius: 8px;
    padding: 0 18px;
    color: #00f0ff;
    font-family: inherit;
    font-size: 14px;
    outline: none;
    transition: all 0.2s;
  }
  .input-box:focus {
    border-color: #00f0ff;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
  }

  .cyber-btn {
    padding: 12px 24px;
    background: rgba(0, 240, 255, 0.18);
    border: 1px solid #00f0ff;
    color: #00f0ff;
    border-radius: 8px;
    font-family: inherit;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 1.5px;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.25);
  }
  .cyber-btn:hover {
    background: rgba(0, 240, 255, 0.35);
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.6);
  }
  
  .quick-chip {
    padding: 4px 10px;
    background: rgba(0, 240, 255, 0.08);
    border: 1px solid rgba(0, 240, 255, 0.3);
    border-radius: 12px;
    font-size: 11px;
    color: #8ba2c4;
    cursor: pointer;
    transition: all 0.2s;
  }
  .quick-chip:hover {
    border-color: #00f0ff;
    color: #00f0ff;
    background: rgba(0, 240, 255, 0.2);
  }
</style>
</head>
<body>

<header>
  <div class="hud-title">
    <span>⚡ JARVIS OMEGA</span>
    <span class="hud-badge">● QUANTUM MATRIX ONLINE</span>
  </div>
  
  <div class="agent-bar">
    <div class="agent-pill active" onclick="selectAgent('Commander', this)">👑 Commander</div>
    <div class="agent-pill" onclick="selectAgent('Researcher', this)">🔬 Researcher</div>
    <div class="agent-pill" onclick="selectAgent('Executor', this)">⚡ Executor</div>
    <div class="agent-pill" onclick="selectAgent('Guardian', this)">🛡️ Guardian</div>
    <div class="agent-pill" onclick="selectAgent('Creative', this)">🎨 Creative</div>
    <div class="agent-pill" onclick="selectAgent('Toolsmith', this)">🛠️ Toolsmith</div>
    <div class="agent-pill" onclick="selectAgent('Monetizer', this)">💰 Monetizer</div>
  </div>

  <div style="display: flex; gap: 10px; align-items: center;">
    <span id="handsfree-badge" class="hud-badge" style="border-color: #00ffaa; color: #00ffaa; background: rgba(0,255,170,0.15); box-shadow: 0 0 10px rgba(0,255,170,0.3);">🟢 HANDS-FREE: ALWAYS LISTENING</span>
    <button class="quick-chip" id="handsfree-toggle-btn" onclick="toggleHandsFree()" style="color: #00ffaa; border-color: #00ffaa; font-weight: bold;">⚡ HANDS-FREE: ON</button>
    <button class="quick-chip" id="voice-toggle-btn" onclick="toggleVoice()">🔊 VOICE: ON</button>
    <span id="time-display" style="font-size: 12px; letter-spacing: 1px; color: #8ba2c4;">--:--:--</span>
  </div>
</header>

<main>
  <!-- 3D WebGL Holographic Arc Reactor Core -->
  <canvas id="three-canvas"></canvas>

  <!-- Left Sensor Panel (Camera & Audio) -->
  <div class="left-hud">
    <div class="panel-box">
      <div class="panel-label">
        <span>👁️ OPTICAL FEED</span>
        <span id="cam-status" style="color: #ffaa00;">STANDBY</span>
      </div>
      <div id="cam-container">
        <video id="webcam-video" autoplay playsinline muted></video>
        <div class="cam-reticle"></div>
      </div>
      <button class="cyber-btn" onclick="activateCamera()" style="width: 100%; margin-top: 10px; font-size: 11px; padding: 8px;">
        ⚡ ACTIVATE CAMERA (HP HD)
      </button>
      <button class="quick-chip" onclick="analyzeCameraScene()" style="width: 100%; margin-top: 6px; text-align: center;">
        🔍 Scan & Analyze Visual Scene
      </button>
    </div>

    <div class="panel-box">
      <div class="panel-label">
        <span>🎙️ AUDIO SPECTRUM (REALTEK)</span>
        <span id="mic-status" style="color: #00ffaa;">ACTIVE</span>
      </div>
      <div class="spectrum-container">
        <div class="spectrum-bar"></div><div class="spectrum-bar"></div>
        <div class="spectrum-bar"></div><div class="spectrum-bar"></div>
        <div class="spectrum-bar"></div><div class="spectrum-bar"></div>
        <div class="spectrum-bar"></div><div class="spectrum-bar"></div>
      </div>
    </div>

    <div class="panel-box" style="flex: 1; font-size: 11px; line-height: 1.6; color: #8ba2c4;">
      <div class="panel-label"><span>📊 QUANTUM VITALS</span></div>
      <div>CORE: <b>OLLAMA 100% OFFLINE</b></div>
      <div>MODEL: <b style="color: #00ffaa;">LLaMA 3.2 (1B)</b></div>
      <div>ENGINE: <b>FASTAPI + THREE.JS</b></div>
      <div>TTS: <b style="color: #00f0ff;">SAPI5 BRITISH AI</b></div>
      <div>GPU: <b>AMD RADEON VEGA 8</b></div>
      <div>STATUS: <b style="color: #00ffaa;">NOMINAL [ACTIVE]</b></div>
    </div>
  </div>

  <!-- Center-Right Dialogue Stream -->
  <div class="right-hud">
    <div class="dialogue-stream" id="dialogue-box">
      <div class="msg-card msg-jarvis">
        <b>[JARVIS OMEGA CORE]</b><br>
        All quantum subsystems operational, Sir. The 3D Arc Reactor, offline neural engine, and biometric sensors are synchronized. State your directive.
      </div>
    </div>

    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 6px;">
      <div class="quick-chip" onclick="quickPrompt('System status report')">⚡ System Status</div>
      <div class="quick-chip" onclick="quickPrompt('Who are you and what are your capabilities?')">❓ Capabilities</div>
      <div class="quick-chip" onclick="quickPrompt('Scan optical room and report findings')">👁️ Room Scan</div>
      <div class="quick-chip" onclick="quickPrompt('Plan an automated high-yield business workflow')">💰 Monetize</div>
      <div class="quick-chip" onclick="quickPrompt('Perform cybersecurity threat audit')">🛡️ Security Audit</div>
    </div>
  </div>
</main>

<footer>
  <button class="hologram-mic-btn" id="mic-btn" onclick="toggleMic()" title="Click to Speak">🎙️</button>
  <input type="text" class="input-box" id="user-input" placeholder="Speak or state your directive to JARVIS..." onkeydown="handleKey(event)">
  <button class="cyber-btn" onclick="sendDirective()">TRANSMIT 🚀</button>
</footer>

<script>
// Live Clock
setInterval(() => {
  const d = new Date();
  document.getElementById('time-display').innerText = d.toTimeString().split(' ')[0];
}, 1000);

let currentAgent = "Commander";
let voiceEnabled = true;
let conversationHistory = [];

function selectAgent(name, el) {
  currentAgent = name;
  document.querySelectorAll('.agent-pill').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  appendMessage('jarvis', `Switched active cognitive profile to: <b>${name} Agent</b>.`);
  speakText(`Profile switched to ${name} Agent.`);
}

function toggleVoice() {
  voiceEnabled = !voiceEnabled;
  const btn = document.getElementById('voice-toggle-btn');
  btn.innerText = voiceEnabled ? "🔊 VOICE: ON" : "🔇 VOICE: MUTED";
  btn.style.color = voiceEnabled ? "#00f0ff" : "#8ba2c4";
}

function appendMessage(role, text) {
  const box = document.getElementById('dialogue-box');
  const card = document.createElement('div');
  card.className = 'msg-card ' + (role === 'user' ? 'msg-user' : 'msg-jarvis');
  card.innerHTML = (role === 'user' ? '<b>[USER]</b><br>' : '<b>[JARVIS // ' + currentAgent.toUpperCase() + ']</b><br>') + text;
  box.appendChild(card);
  box.scrollTop = box.scrollHeight;
}

function handleKey(e) {
  if (e.key === 'Enter') sendDirective();
}

function quickPrompt(txt) {
  document.getElementById('user-input').value = txt;
  sendDirective();
}

// Hands-Free Mode & Audio State
let handsFreeMode = true;
let isSpeakingOrPlaying = false;
let silenceTimer = null;
let recognition = null;
let isListening = false;
let lastResponseText = "";

function toggleHandsFree() {
  handsFreeMode = !handsFreeMode;
  const btn = document.getElementById('handsfree-toggle-btn');
  const badge = document.getElementById('handsfree-badge');
  if (handsFreeMode) {
    btn.innerText = "⚡ HANDS-FREE: ON";
    btn.style.color = "#00ffaa";
    btn.style.borderColor = "#00ffaa";
    badge.innerText = "🟢 HANDS-FREE: ALWAYS LISTENING";
    badge.style.color = "#00ffaa";
    badge.style.borderColor = "#00ffaa";
    startContinuousListening();
  } else {
    btn.innerText = "⏸️ HANDS-FREE: OFF";
    btn.style.color = "#8ba2c4";
    btn.style.borderColor = "#8ba2c4";
    badge.innerText = "⚪ MANUAL CLICK MODE";
    badge.style.color = "#8ba2c4";
    badge.style.borderColor = "#8ba2c4";
    if (recognition && isListening) recognition.stop();
  }
}

function startContinuousListening() {
  if (!handsFreeMode || isSpeakingOrPlaying) return;
  if (!recognition) initRecognition();
  if (!recognition) return;
  try {
    recognition.start();
  } catch(e) {
    // already listening or restarting
  }
}

// Guaranteed Audio Player with Echo Prevention
function playAudioB64(b64) {
  if (!voiceEnabled || !b64) {
    isSpeakingOrPlaying = false;
    setTimeout(startContinuousListening, 300);
    return;
  }
  try {
    isSpeakingOrPlaying = true;
    if (recognition && isListening) {
      try { recognition.stop(); } catch(e){}
    }
    document.getElementById('mic-status').innerText = "JARVIS SPEAKING...";
    document.getElementById('mic-status').style.color = "#00f0ff";

    const audio = new Audio("data:audio/wav;base64," + b64);
    coreSpeedMultiplier = 2.5;

    audio.onended = () => {
      coreSpeedMultiplier = 1.0;
      isSpeakingOrPlaying = false;
      document.getElementById('mic-status').innerText = "LISTENING [HANDS-FREE]";
      document.getElementById('mic-status').style.color = "#00ffaa";
      setTimeout(startContinuousListening, 350);
    };

    audio.onerror = () => {
      coreSpeedMultiplier = 1.0;
      isSpeakingOrPlaying = false;
      setTimeout(startContinuousListening, 350);
    };

    audio.play().catch(e => {
      console.warn("Audio autoplay blocked, using fallback speech:", e);
      speakText(lastResponseText);
    });
  } catch(e) {
    console.warn("Audio error:", e);
    isSpeakingOrPlaying = false;
    setTimeout(startContinuousListening, 300);
  }
}

// SpeechSynthesis Fallback
function speakText(text) {
  if (!voiceEnabled || !window.speechSynthesis) {
    isSpeakingOrPlaying = false;
    setTimeout(startContinuousListening, 300);
    return;
  }
  isSpeakingOrPlaying = true;
  if (recognition && isListening) {
    try { recognition.stop(); } catch(e){}
  }
  window.speechSynthesis.cancel();
  const clean = text.replace(/[*_#`~]/g, "").slice(0, 300);
  const utt = new SpeechSynthesisUtterance(clean);
  utt.rate = 1.05;
  utt.pitch = 0.95;
  const voices = window.speechSynthesis.getVoices();
  const v = voices.find(v => v.name.includes('David') || v.name.includes('George') || v.lang.includes('en-GB') || v.lang.includes('hi'));
  if (v) utt.voice = v;
  utt.onend = () => {
    isSpeakingOrPlaying = false;
    setTimeout(startContinuousListening, 350);
  };
  window.speechSynthesis.speak(utt);
}

// Send Directive to Backend
async function sendDirective() {
  const input = document.getElementById('user-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  
  appendMessage('user', text);
  conversationHistory.push({ role: 'user', content: text });
  
  coreSpeedMultiplier = 2.0;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        agent: currentAgent,
        model: 'llama3.2:1b',
        voice_enabled: voiceEnabled,
        history: conversationHistory
      })
    });
    const data = await res.json();
    lastResponseText = data.response;
    appendMessage('jarvis', data.response);
    conversationHistory.push({ role: 'assistant', content: data.response });
    
    if (data.audio) {
      playAudioB64(data.audio);
    } else {
      speakText(data.response);
    }
  } catch(err) {
    appendMessage('jarvis', `⚠️ Neural connection error: ${err.message}`);
    speakText("Sir, connection error encountered.");
  } finally {
    setTimeout(() => { coreSpeedMultiplier = 1.0; }, 1500);
  }
}

// Camera Activation
async function activateCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    const video = document.getElementById('webcam-video');
    video.srcObject = stream;
    document.getElementById('cam-status').innerText = "ONLINE";
    document.getElementById('cam-status').style.color = "#00ffaa";
    setupAudioSpectrum(stream);
  } catch(err) {
    console.warn("Camera permission pending user gesture:", err);
  }
}

async function analyzeCameraScene() {
  appendMessage('user', '[DIRECTIVE: Scan & Analyze Optical Environment]');
  try {
    const res = await fetch('/api/vision-analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ directive: 'Scan the room and report person presence, devices, and surroundings.', agent: currentAgent })
    });
    const data = await res.json();
    appendMessage('jarvis', data.response);
    if (data.audio) playAudioB64(data.audio);
    else speakText(data.response);
  } catch(e) {
    appendMessage('jarvis', `Optical scan error: ${e.message}`);
  }
}

// Audio Spectrum
function setupAudioSpectrum(stream) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 32;
    src.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    const bars = document.querySelectorAll('.spectrum-bar');
    function update() {
      requestAnimationFrame(update);
      analyser.getByteFrequencyData(data);
      bars.forEach((b, i) => {
        b.style.height = Math.max(12, (data[i % data.length] / 255) * 100) + '%';
      });
    }
    update();
  } catch(e){}
}

function initRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) return;
  recognition = new SpeechRec();
  recognition.lang = 'hi-IN';
  recognition.continuous = true;
  recognition.interimResults = true;

  recognition.onstart = () => {
    isListening = true;
    document.getElementById('mic-btn').classList.add('listening');
    document.getElementById('mic-status').innerText = "LISTENING [HANDS-FREE]";
    document.getElementById('mic-status').style.color = "#00ffaa";
  };

  recognition.onresult = (e) => {
    if (isSpeakingOrPlaying) return;
    let t = "";
    for (let i = e.resultIndex; i < e.results.length; ++i) {
      t += e.results[i][0].transcript;
    }
    const input = document.getElementById('user-input');
    if (t.trim().length > 0) {
      input.value = t;
      document.getElementById('mic-status').innerText = "HEARING: " + t.slice(0, 20) + "...";
      
      clearTimeout(silenceTimer);
      silenceTimer = setTimeout(() => {
        if (input.value.trim().length > 0 && !isSpeakingOrPlaying) {
          sendDirective();
        }
      }, 1100);
    }
  };

  recognition.onerror = (e) => {
    console.warn("Speech recognition error:", e.error);
  };

  recognition.onend = () => {
    isListening = false;
    document.getElementById('mic-btn').classList.remove('listening');
    if (handsFreeMode && !isSpeakingOrPlaying) {
      setTimeout(() => {
        startContinuousListening();
      }, 250);
    }
  };
}

function toggleMic() {
  if (!recognition) initRecognition();
  if (isListening) {
    if (recognition) recognition.stop();
  } else {
    startContinuousListening();
  }
}

// Three.js 3D Holographic Arc Reactor Core
let coreSpeedMultiplier = 1.0;
const canvas = document.getElementById('three-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio || 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 4.2;

// Core Icosahedron
const coreGeo = new THREE.IcosahedronGeometry(0.8, 2);
const coreMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, wireframe: true, transparent: true, opacity: 0.85 });
const core = new THREE.Mesh(coreGeo, coreMat);
scene.add(core);

// Inner Nucleus
const nucGeo = new THREE.SphereGeometry(0.35, 16, 16);
const nucMat = new THREE.MeshBasicMaterial({ color: 0x00ffaa, wireframe: true, transparent: true, opacity: 0.9 });
const nucleus = new THREE.Mesh(nucGeo, nucMat);
scene.add(nucleus);

// Concentric Orbit Rings
const ring1 = new THREE.Mesh(new THREE.TorusGeometry(1.4, 0.02, 16, 100), new THREE.MeshBasicMaterial({ color: 0x00ffaa, transparent: true, opacity: 0.8 }));
scene.add(ring1);
const ring2 = new THREE.Mesh(new THREE.TorusGeometry(1.75, 0.02, 16, 100), new THREE.MeshBasicMaterial({ color: 0x0070f3, transparent: true, opacity: 0.6 }));
scene.add(ring2);
const ring3 = new THREE.Mesh(new THREE.TorusGeometry(2.1, 0.015, 16, 100), new THREE.MeshBasicMaterial({ color: 0xffaa00, transparent: true, opacity: 0.5 }));
ring3.rotation.x = Math.PI / 4;
scene.add(ring3);

// 400 Star Particles
const partGeo = new THREE.BufferGeometry();
const pPos = new Float32Array(400 * 3);
for(let i=0; i<400*3; i+=3) {
  pPos[i] = (Math.random() - 0.5) * 8;
  pPos[i+1] = (Math.random() - 0.5) * 5;
  pPos[i+2] = (Math.random() - 0.5) * 6;
}
partGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
const particles = new THREE.Points(partGeo, new THREE.PointsMaterial({ color: 0x00f0ff, size: 0.04, transparent: true, opacity: 0.7 }));
scene.add(particles);

// Parallax
let mouseX = 0, mouseY = 0;
window.addEventListener('mousemove', (e) => {
  mouseX = (e.clientX - window.innerWidth / 2) * 0.0006;
  mouseY = (e.clientY - window.innerHeight / 2) * 0.0006;
});
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

let t = 0;
function animate() {
  requestAnimationFrame(animate);
  t += 0.02 * coreSpeedMultiplier;
  core.rotation.x += (0.008 + mouseY * 0.1) * coreSpeedMultiplier;
  core.rotation.y += (0.012 + mouseX * 0.1) * coreSpeedMultiplier;
  nucleus.rotation.y -= 0.015 * coreSpeedMultiplier;
  const pulse = 1 + Math.sin(t * 2) * 0.08;
  nucleus.scale.set(pulse, pulse, pulse);
  ring1.rotation.x += 0.01 * coreSpeedMultiplier;
  ring1.rotation.y -= 0.007 * coreSpeedMultiplier;
  ring2.rotation.z += 0.006 * coreSpeedMultiplier;
  ring2.rotation.x -= 0.004 * coreSpeedMultiplier;
  ring3.rotation.y += 0.005 * coreSpeedMultiplier;
  particles.rotation.y += 0.001 * coreSpeedMultiplier;
  renderer.render(scene, camera);
}
animate();

// Auto-start Hands-Free Listening & Camera
window.addEventListener('DOMContentLoaded', () => {
  initRecognition();
  const unlock = () => {
    activateCamera();
    startContinuousListening();
  };
  window.addEventListener('click', unlock, { once: true });
  window.addEventListener('keydown', unlock, { once: true });
  setTimeout(() => {
    activateCamera();
    startContinuousListening();
  }, 1000);
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_TEMPLATE

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:7860")

if __name__ == "__main__":
    Timer(1.5, open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=7860)
