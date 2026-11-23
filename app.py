import os
import sys
import time
import json
import requests
import gradio as gr
from huggingface_hub import InferenceClient

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

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

def clear_chat():
    return [], ""

def refresh_models():
    return gr.Dropdown(choices=get_available_models())

with gr.Blocks(title="VEGA // JARVIS OMEGA - 3D HUD") as demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    
    with gr.Column(elem_classes=["hud-header"]):
        gr.HTML("""
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 class="hud-title">⚡ JARVIS OMEGA // VEGA HUD</h1>
                <p style="margin: 4px 0 0 0; color: #8ba2c4; font-size: 13px; letter-spacing: 1px;">LOCAL OFFLINE AI ENGINE • 3D HOLOGRAPHIC MATRIX • 7 SPECIALISTS</p>
            </div>
            <div style="margin-top: 10px;">
                <span class="hud-badge">● OLLAMA LOCAL ACTIVE</span>
                <span class="hud-badge" style="background: rgba(0, 240, 255, 0.15); color: #00f0ff; border-color: #00f0ff;">THREE.JS 3D HUD</span>
                <span class="hud-badge" style="background: rgba(255, 170, 0, 0.15); color: #ffaa00; border-color: #ffaa00;">100% PRIVATE</span>
            </div>
        </div>
        """)

    with gr.Tabs():
        with gr.TabItem("🌐 3D Holographic Core & Terminal"):
            gr.HTML(HOLOGRAPHIC_3D_CANVAS)
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="VEGA Quantum Neural Stream",
                        height=420,
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="State your directive to JARVIS / VEGA...",
                            show_label=False,
                            scale=8,
                            container=False,
                        )
                        send_btn = gr.Button("Transmit 🚀", scale=2, variant="primary")
                    
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
                "Specialists_Available": [
                    "Commander", "Researcher", "Executor", "Guardian", "Creative", "Toolsmith", "Monetizer"
                ]
            })

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
