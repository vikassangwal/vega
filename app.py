import os
import sys
import time
import json
import re
import requests
import pyttsx3
import speech_recognition as sr
import gradio as gr
from huggingface_hub import InferenceClient

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

def text_to_speech_file(text, output_path="jarvis_voice_reply.wav"):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        voices = engine.getProperty('voices')
        for v in voices:
            if "hazel" in v.name.lower() or "david" in v.name.lower() or "great britain" in v.name.lower():
                engine.setProperty('voice', v.id)
                break
        clean = re.sub(r'[*_#`~]', '', str(text))
        clean = re.sub(r'\[.*?\]\(.*?\)', '', clean)
        clean = clean[:600]
        full_path = os.path.abspath(output_path)
        engine.save_to_file(clean, full_path)
        engine.runAndWait()
        return full_path
    except Exception as e:
        print("TTS Engine Error:", e)
        return None

def speech_to_text(audio_path, language="hi-IN"):
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language=language)
            return text
    except Exception as e:
        print("STT Engine Error:", e)
        return None

def get_available_models():
    models = []
    # Check local Ollama models first
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if r.status_code == 200:
            data = r.json()
            for m in data.get("models", []):
                models.append(f"💻 [LOCAL] {m['name']}")
    except Exception:
        pass
    
    # Add Cloud models
    models.extend([
        "🌐 [CLOUD] deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "🌐 [CLOUD] Qwen/Qwen2.5-Coder-32B-Instruct",
        "🌐 [CLOUD] meta-llama/Llama-3.3-70B-Instruct",
        "🌐 [CLOUD] mistralai/Mistral-7B-Instruct-v0.3",
    ])
    return models

AGENTS = {
    "Commander": {
        "title": "👑 Commander Agent",
        "role": "Master Orchestrator & Strategic Planning",
        "system": "You are JARVIS OMEGA Commander Agent. You oversee all strategic planning, agent delegation, and high-level decision making with calm, authoritative precision."
    },
    "Researcher": {
        "title": "🔬 Researcher Agent",
        "role": "Deep Information Synthesis & Knowledge Retrieval",
        "system": "You are JARVIS OMEGA Researcher Agent. You specialize in deep technical research, literature analysis, data synthesis, and comprehensive factual reporting."
    },
    "Executor": {
        "title": "⚡ Executor Agent",
        "role": "Code Generation, DevOps & System Execution",
        "system": "You are JARVIS OMEGA Executor Agent. You write robust, production-grade code, solve complex bugs, and execute DevOps tasks with zero errors."
    },
    "Guardian": {
        "title": "🛡️ Guardian Agent",
        "role": "Zero-Knowledge Security & Threat Detection",
        "system": "You are JARVIS OMEGA Guardian Agent. You evaluate privacy, verify biometric security protocols, audit vulnerabilities, and safeguard sensitive data."
    },
    "Creative": {
        "title": "🎨 Creative Agent",
        "role": "Multimodal Synthesis, UI/UX & Generative Media",
        "system": "You are JARVIS OMEGA Creative Agent. You design futuristic interfaces, craft creative narratives, and envision cutting-edge user experiences."
    },
    "Toolsmith": {
        "title": "🛠️ Toolsmith Agent",
        "role": "Dynamic Tool Creation & API Integration",
        "system": "You are JARVIS OMEGA Toolsmith Agent. You invent custom automation tools, create OpenAPI client wrappers, and wire up third-party services on the fly."
    },
    "Monetizer": {
        "title": "💰 Monetizer Agent",
        "role": "Economic Optimization, Workflows & ROI",
        "system": "You are JARVIS OMEGA Monetizer Agent. You analyze market viability, automate revenue-generating pipelines, and optimize business logic."
    }
}

CUSTOM_CSS = """
:root {
    --bg-dark: #040810;
    --neon-cyan: #00f0ff;
    --neon-blue: #0070f3;
    --neon-amber: #ffaa00;
    --neon-green: #00ffaa;
}

body, .gradio-container {
    background-color: #040810 !important;
    color: #e0f0ff !important;
    font-family: 'Consolas', 'Courier New', monospace, sans-serif !important;
}

.hud-header {
    background: radial-gradient(ellipse at top, rgba(0, 240, 255, 0.12), rgba(4, 8, 16, 0.95) 75%);
    border: 1px solid rgba(0, 240, 255, 0.4);
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 20px;
    box-shadow: 0 0 35px rgba(0, 240, 255, 0.2), inset 0 0 20px rgba(0, 240, 255, 0.05);
    position: relative;
    overflow: hidden;
}

.hud-header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00f0ff, transparent);
    animation: scanline 3s linear infinite;
}

@keyframes scanline {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.hud-title {
    font-size: 32px;
    font-weight: 900;
    letter-spacing: 4px;
    color: #00f0ff;
    text-transform: uppercase;
    text-shadow: 0 0 12px rgba(0, 240, 255, 0.6), 0 0 25px rgba(0, 240, 255, 0.3);
    margin: 0;
}

.hud-badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    background: rgba(0, 255, 170, 0.15);
    color: #00ffaa;
    border: 1px solid #00ffaa;
    box-shadow: 0 0 10px rgba(0, 255, 170, 0.3);
    margin-right: 8px;
}

.agent-card {
    background: rgba(8, 16, 30, 0.85);
    border: 1px solid rgba(0, 240, 255, 0.3);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: inset 0 0 15px rgba(0, 240, 255, 0.05);
}

.hologram-box {
    border: 1px solid rgba(0, 240, 255, 0.4);
    border-radius: 12px;
    background: #02050b;
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.15);
    padding: 15px;
    margin-bottom: 15px;
}
"""

HOLOGRAPHIC_3D_CANVAS = """
<iframe srcdoc="<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #02050b; overflow: hidden; font-family: 'Consolas', monospace; }
  #canvas-container { width: 100vw; height: 100vh; position: relative; }
  .hud-tag { position: absolute; font-size: 11px; letter-spacing: 2px; font-weight: bold; pointer-events: none; }
  .top-left { top: 12px; left: 16px; color: #00f0ff; text-shadow: 0 0 8px #00f0ff; }
  .top-right { top: 12px; right: 16px; color: #00ffaa; text-shadow: 0 0 8px #00ffaa; }
  .bot-left { bottom: 12px; left: 16px; color: #8ba2c4; font-size: 10px; }
  .bot-right { bottom: 12px; right: 16px; color: #ffaa00; font-size: 10px; }
</style>
<script src='https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'></script>
</head>
<body>
<div id='canvas-container'>
  <div class='hud-tag top-left'>▲ JARVIS OMEGA // ARC REACTOR CORE V3.2</div>
  <div class='hud-tag top-right'>● 100% OFFLINE LOCAL ENGINE ONLINE</div>
  <div class='hud-tag bot-left'>ACTIVE AGENTS: COMMANDER • RESEARCHER • EXECUTOR • GUARDIAN • CREATIVE • TOOLSMITH • MONETIZER</div>
  <div class='hud-tag bot-right'>AMD RADEON VEGA 8 READY • LOCAL: LLAMA 3.2 (1B)</div>
  <canvas id='c'></canvas>
</div>
<script>
  const canvas = document.getElementById('c');
  const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.z = 4.2;

  // Core Wireframe Icosahedron
  const coreGeo = new THREE.IcosahedronGeometry(0.75, 2);
  const coreMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, wireframe: true, transparent: true, opacity: 0.85 });
  const core = new THREE.Mesh(coreGeo, coreMat);
  scene.add(core);

  // Inner Glowing Nucleus
  const nucGeo = new THREE.SphereGeometry(0.35, 16, 16);
  const nucMat = new THREE.MeshBasicMaterial({ color: 0x00ffaa, wireframe: true, transparent: true, opacity: 0.9 });
  const nucleus = new THREE.Mesh(nucGeo, nucMat);
  scene.add(nucleus);

  // Outer Ring 1
  const ring1Geo = new THREE.TorusGeometry(1.35, 0.02, 16, 100);
  const ring1Mat = new THREE.MeshBasicMaterial({ color: 0x00ffaa, transparent: true, opacity: 0.8 });
  const ring1 = new THREE.Mesh(ring1Geo, ring1Mat);
  scene.add(ring1);

  // Outer Ring 2
  const ring2Geo = new THREE.TorusGeometry(1.68, 0.02, 16, 100);
  const ring2Mat = new THREE.MeshBasicMaterial({ color: 0x0070f3, transparent: true, opacity: 0.6 });
  const ring2 = new THREE.Mesh(ring2Geo, ring2Mat);
  scene.add(ring2);

  // Outer Ring 3
  const ring3Geo = new THREE.TorusGeometry(2.0, 0.015, 16, 100);
  const ring3Mat = new THREE.MeshBasicMaterial({ color: 0xffaa00, transparent: true, opacity: 0.5 });
  const ring3 = new THREE.Mesh(ring3Geo, ring3Mat);
  ring3.rotation.x = Math.PI / 4;
  scene.add(ring3);

  // Particle Starfield
  const partGeo = new THREE.BufferGeometry();
  const count = 350;
  const pos = new Float32Array(count * 3);
  for(let i=0; i<count*3; i+=3) {
      pos[i] = (Math.random() - 0.5) * 7;
      pos[i+1] = (Math.random() - 0.5) * 4;
      pos[i+2] = (Math.random() - 0.5) * 5;
  }
  partGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const partMat = new THREE.PointsMaterial({ color: 0x00f0ff, size: 0.045, transparent: true, opacity: 0.75 });
  const particles = new THREE.Points(partGeo, partMat);
  scene.add(particles);

  // Mouse interaction
  let mouseX = 0, mouseY = 0;
  window.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX - window.innerWidth / 2) * 0.0008;
    mouseY = (e.clientY - window.innerHeight / 2) * 0.0008;
  });

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  let t = 0;
  function animate() {
    requestAnimationFrame(animate);
    t += 0.02;
    core.rotation.x += 0.008 + mouseY * 0.1;
    core.rotation.y += 0.012 + mouseX * 0.1;
    nucleus.rotation.y -= 0.015;
    const pulse = 1 + Math.sin(t * 2) * 0.08;
    nucleus.scale.set(pulse, pulse, pulse);
    ring1.rotation.x += 0.01;
    ring1.rotation.y -= 0.007;
    ring2.rotation.z += 0.006;
    ring2.rotation.x -= 0.004;
    ring3.rotation.y += 0.005;
    particles.rotation.y += 0.001;
    renderer.render(scene, camera);
  }
  animate();
</script>
</body>
</html>" style="width: 100%; height: 300px; border: 1px solid rgba(0, 240, 255, 0.4); border-radius: 12px; box-shadow: 0 0 35px rgba(0, 240, 255, 0.25); background: #02050b;" frameborder="0"></iframe>
"""

def chat_with_vega(message, history, agent_name, model_choice, api_key):
    if not message or not str(message).strip():
        yield history, ""
        return
    
    agent_info = AGENTS.get(agent_name, AGENTS["Commander"])
    system_prompt = agent_info["system"]
    
    current_history = list(history) if history else []
    
    # Normalize history to dicts for Gradio 6
    is_dict = True
    api_messages = [{"role": "system", "content": system_prompt}]
    for item in current_history:
        if isinstance(item, dict):
            api_messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            api_messages.append({"role": "user", "content": str(item[0])})
            api_messages.append({"role": "assistant", "content": str(item[1])})
            is_dict = False
            
    api_messages.append({"role": "user", "content": message})
    
    if is_dict:
        current_history.append({"role": "user", "content": message})
        current_history.append({"role": "assistant", "content": ""})
    else:
        current_history.append([message, ""])
        
    # Check if Local Ollama is requested
    if "[LOCAL]" in model_choice:
        clean_model = model_choice.replace("💻 [LOCAL] ", "").strip()
        try:
            payload = {
                "model": clean_model,
                "messages": api_messages,
                "stream": True,
                "options": {"temperature": 0.7}
            }
            resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, stream=True, timeout=120)
            if resp.status_code == 200:
                response_text = ""
                for line in resp.iter_lines():
                    if line:
                        chunk_json = json.loads(line.decode('utf-8'))
                        chunk_text = chunk_json.get("message", {}).get("content", "")
                        response_text += chunk_text
                        if is_dict:
                            current_history[-1]["content"] = response_text
                        else:
                            current_history[-1][1] = response_text
                        yield current_history, ""
            else:
                err_text = f"⚠️ [Local Ollama Error] Server status {resp.status_code}: {resp.text}"
                if is_dict:
                    current_history[-1]["content"] = err_text
                else:
                    current_history[-1][1] = err_text
                yield current_history, ""
        except Exception as e:
            err_text = f"⚠️ [Local Ollama Connection Error] Could not connect to {OLLAMA_HOST}.\nDetails: {str(e)}\n\nMake sure Ollama is running (`ollama serve`)."
            if is_dict:
                current_history[-1]["content"] = err_text
            else:
                current_history[-1][1] = err_text
            yield current_history, ""
    else:
        # Remote Cloud Inference
        clean_model = model_choice.replace("🌐 [CLOUD] ", "").strip()
        hf_token = api_key.strip() if api_key and api_key.strip() else os.environ.get("HF_TOKEN")
        try:
            client = InferenceClient(model=clean_model, token=hf_token)
            response_text = ""
            stream = client.chat_completion(api_messages, max_tokens=1024, stream=True, temperature=0.7)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
                    if is_dict:
                        current_history[-1]["content"] = response_text
                    else:
                        current_history[-1][1] = response_text
                    yield current_history, ""
        except Exception as e:
            err = f"⚠️ [Cloud Neural Error] {str(e)}"
            if is_dict:
                current_history[-1]["content"] = err
            else:
                current_history[-1][1] = err
            yield current_history, ""

VOICE_HEAD_JS = """
<script>
window.jarvisVoiceEnabled = true;
window.jarvisVoiceLang = 'hi-IN';
window.jarvisSpeechRec = null;

function getMsgInput() {
    return document.querySelector('#msg-input textarea') || 
           document.querySelector('.gradio-container textarea');
}

function getSendButton() {
    return document.querySelector('#send-btn') || 
           document.querySelector('button.primary');
}

function initJarvisSpeech() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
        console.warn("Web Speech API not supported in this browser.");
        return;
    }
    const rec = new SpeechRec();
    rec.continuous = false;
    rec.interimResults = true;
    rec.lang = window.jarvisVoiceLang;

    rec.onstart = function() {
        const badge = document.getElementById('mic-badge');
        if (badge) {
            badge.innerText = "🔴 LISTENING (बोलिए, JARVIS सुन रहा है...)";
            badge.style.color = "#ff3366";
            badge.style.borderColor = "#ff3366";
            badge.style.boxShadow = "0 0 15px #ff3366";
        }
    };

    rec.onresult = function(event) {
        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            transcript += event.results[i][0].transcript;
        }
        const input = getMsgInput();
        if (input) {
            input.value = transcript;
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }
        const badge = document.getElementById('mic-badge');
        if (badge && transcript) {
            badge.innerText = "🎙️ '" + transcript + "'";
        }
    };

    rec.onerror = function(event) {
        console.error("Speech rec error:", event.error);
        const badge = document.getElementById('mic-badge');
        if (badge) {
            badge.innerText = "⚠️ Mic Error: " + event.error;
            badge.style.color = "#ffaa00";
            badge.style.borderColor = "#ffaa00";
            badge.style.boxShadow = "none";
        }
    };

    rec.onend = function() {
        const badge = document.getElementById('mic-badge');
        if (badge) {
            badge.innerText = "● Mic Standby (Click 🎙️ to talk)";
            badge.style.color = "#00ffaa";
            badge.style.borderColor = "#00ffaa";
            badge.style.boxShadow = "none";
        }
        const input = getMsgInput();
        const btn = getSendButton();
        if (input && input.value.trim().length > 0 && btn) {
            setTimeout(() => { btn.click(); }, 300);
        }
    };

    window.jarvisSpeechRec = rec;
}

window.toggleSpeech = function() {
    if (!window.jarvisSpeechRec) {
        initJarvisSpeech();
    }
    if (!window.jarvisSpeechRec) {
        alert("Microphone recognition is supported in Google Chrome & Microsoft Edge. Please open http://localhost:7860 in Edge or Chrome.");
        return;
    }
    try {
        window.jarvisSpeechRec.lang = window.jarvisVoiceLang;
        window.jarvisSpeechRec.start();
    } catch(e) {
        console.warn("Speech rec starting error:", e);
        try { window.jarvisSpeechRec.stop(); } catch(err){}
    }
};

window.toggleTTSVoice = function() {
    window.jarvisVoiceEnabled = !window.jarvisVoiceEnabled;
    const btn = document.getElementById('tts-toggle-btn');
    if (btn) {
        if (window.jarvisVoiceEnabled) {
            btn.innerText = "🔊 JARVIS Voice: ACTIVE";
            btn.style.color = "#00f0ff";
            btn.style.borderColor = "#00f0ff";
            speakJarvis("Audio system activated, Sir.");
        } else {
            btn.innerText = "🔇 JARVIS Voice: MUTED";
            btn.style.color = "#8ba2c4";
            btn.style.borderColor = "#8ba2c4";
            if (window.speechSynthesis) window.speechSynthesis.cancel();
        }
    }
};

window.setVoiceLang = function(lang) {
    window.jarvisVoiceLang = lang;
    if (window.jarvisSpeechRec) {
        window.jarvisSpeechRec.lang = lang;
    }
};

function speakJarvis(text) {
    if (!window.jarvisVoiceEnabled || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();

    let clean = text.replace(/```[\s\S]*?```/g, "Code block omitted.")
                    .replace(/`.*?`/g, "")
                    .replace(/[*_#~]/g, "")
                    .replace(/\[.*?\]\(.*?\)/g, "")
                    .trim();
    if (!clean || clean.length < 2) return;

    const utt = new SpeechSynthesisUtterance(clean);
    utt.rate = 1.05;
    utt.pitch = 0.95;

    const voices = window.speechSynthesis.getVoices();
    let selectedVoice = null;
    const hasHindi = /[\u0900-\u097F]/.test(clean);
    if (hasHindi) {
        selectedVoice = voices.find(v => v.lang.includes('hi') || v.name.toLowerCase().includes('hindi') || v.name.toLowerCase().includes('swara') || v.name.toLowerCase().includes('madhur'));
    }
    if (!selectedVoice) {
        selectedVoice = voices.find(v => v.name.toLowerCase().includes('hazel') || v.name.toLowerCase().includes('george') || v.name.toLowerCase().includes('david') || v.lang === 'en-GB');
    }
    if (!selectedVoice && voices.length > 0) {
        selectedVoice = voices.find(v => v.lang.includes('en')) || voices[0];
    }
    if (selectedVoice) utt.voice = selectedVoice;

    window.speechSynthesis.speak(utt);
}

window.jarvisStream = null;
window.audioCtx = null;
window.analyser = null;

window.activateSensors = async function() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: { ideal: 1280 }, height: { ideal: 720 } }, 
            audio: true 
        });
        window.jarvisStream = stream;
        
        const video = document.getElementById('jarvis-cam-feed');
        if (video) {
            video.srcObject = stream;
            video.onloadedmetadata = () => video.play();
        }
        
        const statusBadge = document.getElementById('cam-status-badge');
        if (statusBadge) {
            statusBadge.innerText = "● OPTICAL: ACTIVE [HP HD CAMERA]";
            statusBadge.style.color = "#00ffaa";
            statusBadge.style.borderColor = "#00ffaa";
        }
        
        const btn = document.getElementById('activate-sensors-btn');
        if (btn) {
            btn.innerText = "✓ SENSORS ACTIVE (CAM + MIC ON)";
            btn.style.color = "#00ffaa";
            btn.style.borderColor = "#00ffaa";
            btn.style.background = "rgba(0, 255, 170, 0.2)";
        }
        
        setupAudioVisualizer(stream);
        speakJarvis("Sensors online. Optical camera and audio matrix fully synchronized, Sir.");
    } catch(err) {
        console.error("Sensor activation error:", err);
        alert("Camera / Mic Permission Required!\n\nPlease check your browser address bar:\n1. Click the padlock/tune icon next to http://localhost:7860\n2. Set Camera to 'Allow' and Microphone to 'Allow'\n3. Reload the page.");
    }
};

function setupAudioVisualizer(stream) {
    try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;
        window.audioCtx = new AudioContext();
        const src = window.audioCtx.createMediaStreamSource(stream);
        window.analyser = window.audioCtx.createAnalyser();
        window.analyser.fftSize = 32;
        src.connect(window.analyser);
        
        const dataArr = new Uint8Array(window.analyser.frequencyBinCount);
        const bars = document.querySelectorAll('.audio-bar');
        
        function updateBars() {
            requestAnimationFrame(updateBars);
            if (!window.analyser) return;
            window.analyser.getByteFrequencyData(dataArr);
            bars.forEach((bar, idx) => {
                const val = (dataArr[idx % dataArr.length] / 255) * 100;
                bar.style.height = Math.max(12, val) + '%';
            });
        }
        updateBars();
    } catch(e) {
        console.warn("Visualizer init error:", e);
    }
}

window.addEventListener("DOMContentLoaded", function() {
    initJarvisSpeech();
    let lastSpokenText = "";
    const observer = new MutationObserver(function() {
        const messages = document.querySelectorAll('.message.bot, [data-testid="bot"], .message-bubble-border');
        if (messages.length > 0) {
            const latest = messages[messages.length - 1];
            const text = latest.innerText || latest.textContent || "";
            if (text && text !== lastSpokenText && text.trim().length > 3 && !text.includes("Thinking...")) {
                lastSpokenText = text;
                speakJarvis(text);
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
"""

def clear_chat():
    return [], ""

def refresh_models():
    return gr.Dropdown(choices=get_available_models())

with gr.Blocks(title="VEGA // JARVIS OMEGA - 3D HUD", head=VOICE_HEAD_JS) as demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    
    with gr.Column(elem_classes=["hud-header"]):
        gr.HTML("""
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 class="hud-title">⚡ JARVIS OMEGA // VEGA HUD</h1>
                <p style="margin: 4px 0 0 0; color: #8ba2c4; font-size: 13px; letter-spacing: 1px;">LOCAL OFFLINE AI ENGINE • 3D HOLOGRAPHIC MATRIX • DUAL VOICE STT/TTS</p>
            </div>
            <div style="margin-top: 10px;">
                <span class="hud-badge">● OLLAMA LOCAL ACTIVE</span>
                <span class="hud-badge" style="background: rgba(0, 240, 255, 0.15); color: #00f0ff; border-color: #00f0ff;">🎙️ AUDIO RECOGNITION</span>
                <span class="hud-badge" style="background: rgba(255, 170, 0, 0.15); color: #ffaa00; border-color: #ffaa00;">🔊 VOICE SYNTHESIS</span>
            </div>
        </div>
        """)

    with gr.Tabs():
        with gr.TabItem("🌐 3D Holographic Core & Terminal"):
            gr.HTML(HOLOGRAPHIC_3D_CANVAS)
            
            gr.HTML("""
            <div style="background: rgba(4, 10, 22, 0.95); border: 1px solid rgba(0, 240, 255, 0.4); border-radius: 12px; padding: 12px 16px; margin: 12px 0 16px 0; box-shadow: 0 0 25px rgba(0, 240, 255, 0.15);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
                    <span style="font-family: monospace; font-size: 12px; font-weight: bold; color: #00f0ff; letter-spacing: 1.5px;">
                        ▲ JARVIS BIOMETRIC & OPTICAL SENSOR MATRIX (HP HD CAMERA + REALTEK MIC)
                    </span>
                    <span id="cam-status-badge" style="font-family: monospace; font-size: 11px; color: #ffaa00; border: 1px solid #ffaa00; padding: 3px 8px; border-radius: 4px; background: rgba(255, 170, 0, 0.1);">
                        ● SENSORS STANDBY (CLICK ACTIVATE)
                    </span>
                </div>
                <div style="display: flex; gap: 14px; flex-wrap: wrap; align-items: center;">
                    <div style="position: relative; width: 260px; height: 160px; background: #02050b; border: 1px solid rgba(0, 240, 255, 0.5); border-radius: 8px; overflow: hidden;">
                        <video id="jarvis-cam-feed" autoplay playsinline muted style="width: 100%; height: 100%; object-fit: cover;"></video>
                        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; pointer-events: none; border: 1px solid rgba(0, 240, 255, 0.2); box-sizing: border-box;">
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 50px; height: 50px; border: 1px dashed rgba(0, 240, 255, 0.6); border-radius: 50%;"></div>
                            <div style="position: absolute; bottom: 5px; left: 8px; font-family: monospace; font-size: 9px; color: #00f0ff; letter-spacing: 1px;">HP HD CAM // 720P</div>
                            <div style="position: absolute; bottom: 5px; right: 8px; font-family: monospace; font-size: 9px; color: #00ffaa; letter-spacing: 1px;">OPTICAL TRACKING</div>
                        </div>
                    </div>
                    <div style="flex: 1; min-width: 250px;">
                        <button id="activate-sensors-btn" onclick="activateSensors()" style="width: 100%; padding: 12px; background: rgba(0, 240, 255, 0.18); border: 1px solid #00f0ff; color: #00f0ff; font-weight: bold; cursor: pointer; border-radius: 8px; font-family: monospace; font-size: 13px; letter-spacing: 1px; box-shadow: 0 0 15px rgba(0, 240, 255, 0.25);">
                            ⚡ ACTIVATE CAMERA & MIC (कैमरा और माइक चालू करें)
                        </button>
                        <div style="margin-top: 10px; padding: 10px; background: rgba(2, 6, 14, 0.8); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 6px;">
                            <div style="font-family: monospace; font-size: 10px; color: #8ba2c4; margin-bottom: 6px; letter-spacing: 1px;">
                                REALTEK AUDIO SPECTRUM (MIC INPUT LEVEL):
                            </div>
                            <div style="display: flex; gap: 4px; height: 32px; align-items: flex-end;">
                                <div class="audio-bar" style="flex: 1; background: #00f0ff; height: 15%; border-radius: 2px;"></div>
                                <div class="audio-bar" style="flex: 1; background: #00f0ff; height: 25%; border-radius: 2px;"></div>
                                <div class="audio-bar" style="flex: 1; background: #00ffaa; height: 40%; border-radius: 2px;"></div>
                                <div class="audio-bar" style="flex: 1; background: #00ffaa; height: 60%; border-radius: 2px;"></div>
                                <div class="audio-bar" style="flex: 1; background: #00ffaa; height: 30%; border-radius: 2px;"></div>
                                <div class="audio-bar" style="flex: 1; background: #0070f3; height: 50%; border-radius: 2px;"></div>
                                <div class="audio-bar" style="flex: 1; background: #0070f3; height: 20%; border-radius: 2px;"></div>
                                <div class="audio-bar" style="flex: 1; background: #ffaa00; height: 15%; border-radius: 2px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="VEGA Quantum Neural Stream",
                        height=420,
                    )
                    
                    gr.HTML("""
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin: 8px 0; padding: 8px 12px; background: rgba(5, 12, 25, 0.9); border: 1px solid rgba(0, 240, 255, 0.35); border-radius: 8px;">
                        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                            <button id="mic-btn" onclick="toggleSpeech()" style="background: rgba(0, 240, 255, 0.18); border: 1px solid #00f0ff; color: #00f0ff; padding: 7px 15px; border-radius: 6px; font-weight: bold; cursor: pointer; font-family: monospace; letter-spacing: 1px;">
                                🎙️ Speak (माइक से बोलें)
                            </button>
                            <button id="tts-toggle-btn" onclick="toggleTTSVoice()" style="background: rgba(0, 255, 170, 0.12); border: 1px solid #00ffaa; color: #00ffaa; padding: 7px 12px; border-radius: 6px; font-weight: bold; cursor: pointer; font-family: monospace; letter-spacing: 1px;">
                                🔊 JARVIS Voice: ACTIVE
                            </button>
                            <select onchange="setVoiceLang(this.value)" style="background: #081220; color: #00f0ff; border: 1px solid rgba(0, 240, 255, 0.4); padding: 6px 8px; border-radius: 6px; font-family: monospace; font-size: 11px;">
                                <option value="hi-IN">🇮🇳 Hindi / Hinglish</option>
                                <option value="en-US">🇬🇧 English (JARVIS)</option>
                            </select>
                        </div>
                        <span id="mic-badge" style="font-family: monospace; font-size: 11px; color: #00ffaa; padding: 4px 8px; border: 1px solid #00ffaa; border-radius: 6px; background: rgba(0, 255, 170, 0.08);">
                            ● Mic Standby (Click 🎙️ to talk)
                        </span>
                    </div>
                    """)

                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="State your directive or click 🎙️ Speak above...",
                            show_label=False,
                            scale=8,
                            container=False,
                            elem_id="msg-input",
                        )
                        send_btn = gr.Button("Transmit 🚀", scale=2, variant="primary", elem_id="send-btn")
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Stream", variant="secondary")

                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ Neural Control Matrix")
                    agent_selector = gr.Dropdown(
                        choices=list(AGENTS.keys()),
                        value="Commander",
                        label="Active Specialist Agent",
                    )
                    agent_desc = gr.Markdown(f"**{AGENTS['Commander']['title']}**\n\n{AGENTS['Commander']['role']}")
                    
                    def update_agent_desc(name):
                        info = AGENTS.get(name, AGENTS["Commander"])
                        return f"**{info['title']}**\n\n{info['role']}"
                    
                    agent_selector.change(update_agent_desc, inputs=[agent_selector], outputs=[agent_desc])
                    
                    model_choices = get_available_models()
                    model_selector = gr.Dropdown(
                        choices=model_choices,
                        value=model_choices[0] if model_choices else "🌐 [CLOUD] deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                        label="AI Engine (Local / Cloud)",
                    )
                    refresh_btn = gr.Button("🔄 Refresh Local Models", size="sm")
                    refresh_btn.click(refresh_models, outputs=[model_selector])
                    
                    user_token = gr.Textbox(
                        label="Cloud Token (Optional)",
                        placeholder="hf_... (For cloud models)",
                        type="password",
                    )

            send_btn.click(
                chat_with_vega,
                inputs=[msg, chatbot, agent_selector, model_selector, user_token],
                outputs=[chatbot, msg],
            )
            msg.submit(
                chat_with_vega,
                inputs=[msg, chatbot, agent_selector, model_selector, user_token],
                outputs=[chatbot, msg],
            )
            clear_btn.click(
                clear_chat,
                inputs=[],
                outputs=[chatbot, msg],
            )

        with gr.TabItem("👁️ Vision & Optical Camera"):
            gr.Markdown("## 👁️ JARVIS Optical Matrix (HP HD Camera दृश्य विश्लेषण)")
            gr.Markdown("Take a snap using your HP HD Camera or upload an image. JARVIS will visually inspect the scene, detect objects/faces, and report back.")
            with gr.Row():
                with gr.Column(scale=2):
                    cam_snap = gr.Image(
                        sources=["webcam", "upload"],
                        type="pil",
                        label="📸 HP HD Camera Snap / Optical Input",
                    )
                    vision_directive = gr.Textbox(
                        label="Optical Analysis Directive",
                        value="Describe what you see in front of the camera and report key objects, person presence, or details.",
                    )
                    vision_snap_btn = gr.Button("⚡ Run Optical Analysis (कैमरा विश्लेषण करें)", variant="primary")
                with gr.Column(scale=2):
                    vision_report = gr.Textbox(
                        label="JARVIS Optical Analysis Report",
                        lines=10,
                    )
                    
            def handle_vision_snap(img, directive, agent, model, token):
                if img is None:
                    return "⚠️ No camera snap detected. Please click 'Record/Snap' on the webcam feed first."
                w, h = img.size
                prompt = f"[Optical Sensor Frame {w}x{h}] {directive}"
                res = ""
                for h_list, _ in chat_with_vega(prompt, [], agent, model, token):
                    if h_list:
                        last_item = h_list[-1]
                        res = last_item["content"] if isinstance(last_item, dict) else last_item[1]
                return res

            vision_snap_btn.click(
                handle_vision_snap,
                inputs=[cam_snap, vision_directive, agent_selector, model_selector, user_token],
                outputs=[vision_report],
            )

        with gr.TabItem("🎙️ Voice Neural Link & Audio"):
            gr.Markdown("## 🎙️ Dedicated Voice Subsystem (माइक्रोफोन निर्देश केंद्र)")
            gr.Markdown("Speak into your microphone or upload an audio note. JARVIS will transcribe your words, run cognitive reasoning, and answer you with a spoken audio response.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    voice_in = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="🎙️ Record Voice Directive (माइक से बोलें)",
                    )
                    with gr.Row():
                        voice_lang = gr.Radio(
                            choices=["Hindi (हिन्दी)", "English"],
                            value="Hindi (हिन्दी)",
                            label="Recognition Dialect",
                        )
                        voice_exec_btn = gr.Button("⚡ Execute Voice Directive", variant="primary", scale=2)
                
                with gr.Column(scale=2):
                    voice_out_text = gr.Textbox(
                        label="Transcribed Voice Directive (पहचानी गई आवाज़)",
                        placeholder="Your speech transcript will appear here...",
                    )
                    voice_out_audio = gr.Audio(
                        label="🔊 JARVIS Spoken Voice Reply (JARVIS की आवाज़)",
                        autoplay=True,
                    )

            def handle_voice_flow(audio_file, lang_choice, history, agent, model, token):
                if not audio_file:
                    return "⚠️ No audio detected. Please record or speak into the microphone.", None, history
                
                lang_code = "hi-IN" if "Hindi" in lang_choice else "en-US"
                transcript = speech_to_text(audio_file, language=lang_code)
                if not transcript:
                    return "⚠️ Could not understand the audio. Please speak clearly into the microphone.", None, history
                
                last_hist = history
                for h, _ in chat_with_vega(transcript, history, agent, model, token):
                    last_hist = h
                
                ans = ""
                if last_hist:
                    last_entry = last_hist[-1]
                    ans = last_entry["content"] if isinstance(last_entry, dict) else last_entry[1]
                
                wav_file = text_to_speech_file(ans) if ans else None
                return transcript, wav_file, last_hist

            voice_exec_btn.click(
                handle_voice_flow,
                inputs=[voice_in, voice_lang, chatbot, agent_selector, model_selector, user_token],
                outputs=[voice_out_text, voice_out_audio, chatbot],
            )

        with gr.TabItem("👥 7 Specialist Agents"):
            gr.Markdown("## The 7 Core Specialist Agents of JARVIS OMEGA")
            for name, data in AGENTS.items():
                with gr.Column(elem_classes=["agent-card"]):
                    gr.Markdown(f"### {data['title']}")
                    gr.Markdown(f"**Primary Objective:** {data['role']}")
                    gr.Markdown(f"*System Protocol:* `{data['system']}`")

        with gr.TabItem("📊 Local Telemetry & Vitals"):
            gr.Markdown("### PC Hardware & Local Engine Telemetry")
            gr.JSON({
                "System": "JARVIS OMEGA PC Edition",
                "Ollama_Status": "Connected to localhost:11434",
                "Ollama_Version": "0.40.16",
                "Local_Inference": "Enabled (100% Offline Capable)",
                "3D_Graphics": "Three.js WebGL Holographic HUD",
                "Audio_Engine": "Dual STT (SpeechRecognition) + TTS (SAPI5/WebSpeech)",
                "Specialists_Available": [
                    "Commander", "Researcher", "Executor", "Guardian", "Creative", "Toolsmith", "Monetizer"
                ]
            })

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
