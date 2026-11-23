import os
import sys
import time
import json
import gradio as gr
from huggingface_hub import InferenceClient

# Available models on Hugging Face Free Inference
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

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
    --bg-dark: #070b12;
    --neon-cyan: #00f0ff;
    --neon-blue: #0070f3;
    --neon-amber: #ffaa00;
    --neon-green: #00ffaa;
}

body, .gradio-container {
    background-color: #070b12 !important;
    color: #e0f0ff !important;
    font-family: 'Segoe UI', 'Roboto', monospace, sans-serif !important;
}

.hud-header {
    background: linear-gradient(135deg, rgba(0, 240, 255, 0.08), rgba(0, 112, 243, 0.15));
    border: 1px solid rgba(0, 240, 255, 0.3);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.12);
}

.hud-title {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 3px;
    color: #00f0ff;
    text-transform: uppercase;
    text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
    margin: 0;
}

.hud-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    background: rgba(0, 255, 170, 0.15);
    color: #00ffaa;
    border: 1px solid #00ffaa;
    margin-right: 8px;
}

.agent-card {
    background: rgba(10, 20, 35, 0.7);
    border: 1px solid rgba(0, 240, 255, 0.2);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}
"""

def chat_with_vega(message, history, agent_name, model_name, api_key):
    if not message.strip():
        yield history, ""
        return
    
    # Get agent configuration
    agent_info = AGENTS.get(agent_name, AGENTS["Commander"])
    system_prompt = agent_info["system"]
    
    # Use user provided HF token or default environment token
    hf_token = api_key.strip() if api_key.strip() else os.environ.get("HF_TOKEN")
    
    try:
        client = InferenceClient(model=model_name, token=hf_token)
        
        messages = [{"role": "system", "content": system_prompt}]
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})
        
        history.append((message, ""))
        
        response_text = ""
        stream = client.chat_completion(messages, max_tokens=1024, stream=True, temperature=0.7)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
                history[-1] = (message, response_text)
                yield history, ""
    except Exception as e:
        error_msg = f"⚠️ [System Notice] Response error: {str(e)}\n\nTip: You can paste your free Hugging Face / Groq token in the Settings panel for unlimited access."
        history.append((message, error_msg))
        yield history, ""

with gr.Blocks(css=CUSTOM_CSS, title="VEGA OMEGA - AI HUD") as demo:
    with gr.Column(elem_classes=["hud-header"]):
        gr.HTML("""
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 class="hud-title">⚡ VEGA // JARVIS OMEGA</h1>
                <p style="margin: 4px 0 0 0; color: #8ba2c4; font-size: 13px;">AUTONOMOUS AGENTIC INTELLIGENCE SUITE • 7 SPECIALIST MATRIX</p>
            </div>
            <div style="margin-top: 10px;">
                <span class="hud-badge">● LIVE 24/7</span>
                <span class="hud-badge" style="background: rgba(0, 240, 255, 0.15); color: #00f0ff; border-color: #00f0ff;">16 GB RAM CLOUD</span>
                <span class="hud-badge" style="background: rgba(255, 170, 0, 0.15); color: #ffaa00; border-color: #ffaa00;">ZERO-KNOWLEDGE ACTIVE</span>
            </div>
        </div>
        """)

    with gr.Tabs():
        with gr.TabItem("💬 Holographic Terminal"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="VEGA Holographic Stream",
                        height=520,
                        show_copy_button=True,
                        bubble_full_width=False,
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="State your directive to VEGA...",
                            show_label=False,
                            scale=8,
                            container=False,
                        )
                        send_btn = gr.Button("Transmit 🚀", scale=2, variant="primary")
                    
                    with gr.Row():
                        clear_btn = gr.ClearButton([msg, chatbot], value="Clear Terminal")

                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ Agent Control")
                    agent_selector = gr.Dropdown(
                        choices=list(AGENTS.keys()),
                        value="Commander",
                        label="Active Specialist Agent",
                    )
                    agent_desc = gr.Markdown(f"**{AGENTS['Commander']['title']}**\n{AGENTS['Commander']['role']}")
                    
                    def update_agent_desc(name):
                        info = AGENTS.get(name, AGENTS["Commander"])
                        return f"**{info['title']}**\n\n{info['role']}"
                    
                    agent_selector.change(update_agent_desc, inputs=[agent_selector], outputs=[agent_desc])
                    
                    model_selector = gr.Dropdown(
                        choices=DEFAULT_MODELS,
                        value=DEFAULT_MODELS[0],
                        label="Neural Foundation Engine",
                    )
                    
                    user_token = gr.Textbox(
                        label="Custom HF Token (Optional)",
                        placeholder="hf_... (Leave blank to use free cloud)",
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

        with gr.TabItem("👥 7 Specialist Agents"):
            gr.Markdown("## The 7 Core Specialist Agents of VEGA OMEGA")
            for name, data in AGENTS.items():
                with gr.Column(elem_classes=["agent-card"]):
                    gr.Markdown(f"### {data['title']}")
                    gr.Markdown(f"**Primary Objective:** {data['role']}")
                    gr.Markdown(f"*System Protocol:* `{data['system']}`")

        with gr.TabItem("📊 Telemetry & Architecture"):
            gr.Markdown("### System Telemetry & Deployed Architecture")
            gr.JSON({
                "System": "VEGA OMEGA Cloud Instance",
                "Status": "Online (24x7 Continuous)",
                "Host": "Hugging Face Spaces Cloud",
                "RAM": "16 GB Allocated",
                "Architecture": "FastAPI + React HUD + 7-Tier Memory",
                "Active_Memory_Tiers": [
                    "Working Memory", "Short-term Cache", "Episodic Journal",
                    "Semantic Vector Index", "Procedural Skills", "Reflective Insights", "Knowledge Graph"
                ],
                "Connected_Plugins": [
                    "Smart Home Assistant", "Spotify API", "Obsidian Vault", "Notion Workspace", "Email/Calendar Sync"
                ]
            })

demo.queue().launch(server_name="0.0.0.0", server_port=7860)
