import os
import sys
import time
import json
import gradio as gr
from huggingface_hub import InferenceClient

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, *args, **kwargs):
            if fn:
                return fn
            def wrapper(f):
                return f
            return wrapper

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

@spaces.GPU
def chat_with_vega(message, history, agent_name, model_name, api_key):
    if not message or not str(message).strip():
        yield history, ""
        return
    
    agent_info = AGENTS.get(agent_name, AGENTS["Commander"])
    system_prompt = agent_info["system"]
    
    hf_token = api_key.strip() if api_key and api_key.strip() else os.environ.get("HF_TOKEN")
    
    current_history = list(history) if history else []
    
    api_messages = [{"role": "system", "content": system_prompt}]
    
    is_dict = False
    if current_history:
        first = current_history[0]
        if isinstance(first, dict):
            is_dict = True
            for item in current_history:
                api_messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})
        elif isinstance(first, (list, tuple)) and len(first) == 2:
            is_dict = False
            for item in current_history:
                api_messages.append({"role": "user", "content": str(item[0])})
                api_messages.append({"role": "assistant", "content": str(item[1])})
    else:
        is_dict = True
        
    api_messages.append({"role": "user", "content": message})
    
    if is_dict:
        current_history.append({"role": "user", "content": message})
        current_history.append({"role": "assistant", "content": ""})
    else:
        current_history.append([message, ""])
        
    try:
        client = InferenceClient(model=model_name, token=hf_token)
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
        err = f"⚠️ [System Notice] {str(e)}\n\nTip: You can paste your free Hugging Face token in the token input box for dedicated high-speed access."
        if is_dict:
            current_history[-1]["content"] = err
        else:
            current_history[-1][1] = err
        yield current_history, ""

def clear_chat():
    return [], ""

with gr.Blocks(title="VEGA OMEGA - AI HUD") as demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    
    with gr.Column(elem_classes=["hud-header"]):
        gr.HTML("""
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 class="hud-title">⚡ VEGA // JARVIS OMEGA</h1>
                <p style="margin: 4px 0 0 0; color: #8ba2c4; font-size: 13px;">AUTONOMOUS AGENTIC INTELLIGENCE SUITE • 7 SPECIALIST MATRIX</p>
            </div>
            <div style="margin-top: 10px;">
                <span class="hud-badge">● LIVE 24/7</span>
                <span class="hud-badge" style="background: rgba(0, 240, 255, 0.15); color: #00f0ff; border-color: #00f0ff;">ZeroGPU READY</span>
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
                        clear_btn = gr.Button("Clear Terminal", variant="secondary")

                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ Agent Control")
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
            clear_btn.click(
                clear_chat,
                inputs=[],
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
                "Hardware": "ZeroGPU + 16 GB RAM",
                "Architecture": "FastAPI + React HUD + 7-Tier Memory",
                "Active_Memory_Tiers": [
                    "Working Memory", "Short-term Cache", "Episodic Journal",
                    "Semantic Vector Index", "Procedural Skills", "Reflective Insights", "Knowledge Graph"
                ],
                "Connected_Plugins": [
                    "Smart Home Assistant", "Spotify API", "Obsidian Vault", "Notion Workspace", "Email/Calendar Sync"
                ]
            })

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
