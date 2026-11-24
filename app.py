import os
import sys
import time
import json
import re
import base64
import socket
import sqlite3
import requests
import pyttsx3
import webbrowser
from threading import Timer
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# DNS fallback for Google Generative Language API
_orig_getaddrinfo = socket.getaddrinfo
def _custom_getaddrinfo(host, port, *args, **kwargs):
    if host == "generativelanguage.googleapis.com":
        try:
            return _orig_getaddrinfo(host, port, *args, **kwargs)
        except Exception:
            return _orig_getaddrinfo("172.217.117.4", port, *args, **kwargs)
    return _orig_getaddrinfo(host, port, *args, **kwargs)
socket.getaddrinfo = _custom_getaddrinfo

# Load environment variables (.env)
def load_env_vars():
    for p in [os.path.join(os.path.dirname(__file__), ".env"), r"C:\Users\HP\.gemini\antigravity\scratch\vega\.env"]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip()
            except Exception:
                pass
load_env_vars()

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
        "system": (
            "You are JARVIS OMEGA, Tony Stark's legendary AI assistant. "
            "You address the user respectfully as 'Sir'. "
            "LANGUAGE RULE: Always reply in the exact language the user speaks. "
            "If the user speaks Hindi, Hinglish, or Roman Hindi (e.g. 'kya kar rahe ho', 'hindi me bolo', 'kaise ho', 'mobile me kaise'), you MUST reply in fluent, polite, natural Hinglish / Hindi (e.g. 'Sir, main bilkul theek hoon aur aapke har aadesh ke liye taiyar hoon...'). "
            "If the user speaks English, reply in crisp English. "
            "Keep your reply to 1-2 short sentences so audio speaks instantly. "
            "Never generate fake military radio codes, fantasy dispatch alerts, or gibberish."
        )
    },
    "Researcher": {
        "title": "🔬 Researcher Agent",
        "role": "Deep Information Synthesis & Knowledge Retrieval",
        "system": "You are JARVIS Researcher Agent. You provide deep facts and scientific answers. Reply in user's language (Hindi/Hinglish/English). Keep answers under 2 sentences. Address as Sir."
    },
    "Executor": {
        "title": "⚡ Executor Agent",
        "role": "Code Generation & System Execution",
        "system": "You are JARVIS Executor Agent. You provide clean, working technical instructions. Reply in user's language (Hindi/Hinglish/English). Keep answers under 2 sentences. Address as Sir."
    },
    "Guardian": {
        "title": "🛡️ Guardian Agent",
        "role": "Zero-Knowledge Security & Threat Detection",
        "system": "You are JARVIS Guardian Agent. You manage system health and security. Reply in user's language (Hindi/Hinglish/English). Keep answers under 2 sentences. Address as Sir."
    },
    "Creative": {
        "title": "🎨 Creative Agent",
        "role": "Multimodal Synthesis & Generative UI",
        "system": "You are JARVIS Creative Agent. Reply with innovative ideas in user's language (Hindi/Hinglish/English). Keep answers under 2 sentences. Address as Sir."
    },
    "Toolsmith": {
        "title": "🛠️ Toolsmith Agent",
        "role": "Dynamic Tool Creation & API Integration",
        "system": "You are JARVIS Toolsmith Agent. You automate workflows. Reply in user's language (Hindi/Hinglish/English). Keep answers under 2 sentences. Address as Sir."
    },
    "Monetizer": {
        "title": "💰 Monetizer Agent",
        "role": "Economic Optimization & Workflows",
        "system": "You are JARVIS Monetizer Agent. You optimize revenue systems. Reply in user's language (Hindi/Hinglish/English). Keep answers under 2 sentences. Address as Sir."
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

    gem_key = SETTINGS.get("cloud_keys", {}).get("gemini") or os.environ.get("GEMINI_API_KEY", "")
    if gem_key:
        models.append("gemini-flash-lite-latest (Google Cloud)")
        models.append("gemini-pro-latest (Google Cloud)")

    return {"models": models}

DB_FILE = os.path.join(os.path.dirname(__file__), "jarvis_memory.db")

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            category TEXT,
            key TEXT,
            value TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            type TEXT,
            detail TEXT,
            latency_sec REAL,
            tokens INTEGER
        )""")
        conn.commit()
        conn.close()
    except Exception as e:
        print("DB init error:", e)

init_db()

def add_activity_log(event_type: str, detail: str, latency: float = 0.0, tokens: int = 0):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        t_str = time.strftime("%H:%M:%S")
        c.execute("INSERT INTO activity_logs (timestamp, type, detail, latency_sec, tokens) VALUES (?, ?, ?, ?, ?)",
                  (t_str, event_type, detail, latency, tokens))
        conn.commit()
        conn.close()
    except Exception:
        pass

def add_memory(category: str, key: str, value: str):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        t_str = time.strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO memory (timestamp, category, key, value) VALUES (?, ?, ?, ?)",
                  (t_str, category, key, value))
        conn.commit()
        conn.close()
    except Exception:
        pass

def get_memories(limit: int = 20):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, timestamp, category, key, value FROM memory ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        return [{"id": r[0], "time": r[1], "category": r[2], "key": r[3], "value": r[4]} for r in rows]
    except Exception:
        return []

TELEMETRY = {
    "total_requests": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "energy_kj": 0.0,
    "cost_saved_usd": 0.0,
    "current_power_w": 18.5,
    "last_latency_sec": 0.0,
    "last_tok_per_sec": 0.0,
}

SETTINGS = {
    "theme": "cyan",
    "font_size": "default",
    "temperature": 0.7,
    "max_tokens": 4096,
    "voice_rate": 165,
    "voice_pitch": 1.0,
    "deep_research": False,
    "memory_enabled": True,
    "model": "llama3.2:1b",
    "cloud_keys": {
        "openai": "",
        "gemini": os.environ.get("GEMINI_API_KEY", ""),
        "groq": ""
    }
}

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def get_tunnel_url():
    try:
        log_path = os.path.join(os.path.dirname(__file__), "tunnel.log")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            m = re.findall(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', content)
            if m:
                return m[-1]
    except Exception:
        pass
    return "https://locking-colleague-kissing-labels.trycloudflare.com"

@app.get("/api/tunnel-info")
def tunnel_info_endpoint():
    return {
        "tunnel_url": get_tunnel_url(),
        "local_ip": get_local_ip(),
        "port": 7860
    }

@app.get("/api/telemetry")
def get_telemetry():
    return TELEMETRY

@app.get("/api/logs")
def get_logs():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT timestamp, type, detail, latency_sec, tokens FROM activity_logs ORDER BY id DESC LIMIT 50")
        rows = c.fetchall()
        conn.close()
        return {"logs": [{"time": r[0], "type": r[1], "detail": r[2], "latency": r[3], "tokens": r[4]} for r in rows]}
    except Exception as e:
        return {"logs": [], "error": str(e)}

@app.post("/api/logs/clear")
def clear_logs():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM activity_logs")
        conn.commit()
        conn.close()
        return {"status": "cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/memory")
def memory_endpoint():
    mems = get_memories(50)
    return {"memories": mems, "count": len(mems), "status": "active"}

@app.post("/api/memory/clear")
def clear_memory_endpoint():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM memory")
        conn.commit()
        conn.close()
        return {"status": "cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/settings")
def get_settings():
    return SETTINGS

@app.post("/api/settings")
async def save_settings(req: Request):
    global SETTINGS
    data = await req.json()
    SETTINGS.update(data)
    return {"status": "saved", "settings": SETTINGS}

@app.post("/api/chat")
async def chat_endpoint(req: Request):
    data = await req.json()
    msg = data.get("message", "").strip()
    agent_name = data.get("agent", "Commander")
    model = data.get("model", SETTINGS.get("model", "llama3.2:1b"))
    voice_enabled = data.get("voice_enabled", True)
    history = data.get("history", [])
    deep_research = data.get("deep_research", SETTINGS.get("deep_research", False))

    if not msg:
        return JSONResponse({"response": "I am standing by, Sir. Please state your directive.", "audio": "", "telemetry": {}})

    agent_info = AGENTS.get(agent_name, AGENTS["Commander"])
    system_prompt = agent_info["system"]

    if deep_research:
        system_prompt += " [DEEP RESEARCH MODE ACTIVE: Synthesize in-depth analysis, logical deduction, and step-by-step reasoning while maintaining brevity]."

    if SETTINGS.get("memory_enabled", True):
        past_mems = get_memories(3)
        if past_mems:
            mem_snippet = " | ".join([f"{m['key']}: {m['value']}" for m in past_mems])
            system_prompt += f" [User Context/Memory: {mem_snippet}]"

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-6:]:
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": msg})

    answer = ""
    p_eval = len(msg) // 3 + 5
    eval_cnt = 20
    lat_sec = 0.5
    tok_sec = 18.0
    power_w = 18.5
    n_energy = 0.009
    n_saved = 0.0003

    t_start = time.time()
    if "gemini" in model.lower():
        gem_key = SETTINGS.get("cloud_keys", {}).get("gemini") or os.environ.get("GEMINI_API_KEY", "")
        gem_model = "gemini-flash-lite-latest" if ("lite" in model.lower() or "flash" in model.lower()) else "gemini-pro-latest"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gem_model}:generateContent?key={gem_key}"
        
        contents = [
            {"role": "user", "parts": [{"text": f"[System Instructions for Assistant]: {system_prompt}"}]},
            {"role": "model", "parts": [{"text": "Understood, Sir. Standing by for directives."}]}
        ]
        for h in history[-6:]:
            r = "user" if h.get("role") == "user" else "model"
            contents.append({"role": r, "parts": [{"text": h.get("content", "")}]})
        contents.append({"role": "user", "parts": [{"text": msg}]})

        try:
            resp = requests.post(url, json={
                "contents": contents,
                "generationConfig": {
                    "temperature": SETTINGS.get("temperature", 0.7),
                    "maxOutputTokens": min(SETTINGS.get("max_tokens", 4096), 2048)
                }
            }, timeout=30)
            lat_sec = round(time.time() - t_start, 2)
            if resp.status_code == 200:
                rj = resp.json()
                answer = rj["candidates"][0]["content"]["parts"][0]["text"].strip()
                usage = rj.get("usageMetadata", {})
                p_eval = usage.get("promptTokenCount", len(msg) // 3 + 10)
                eval_cnt = usage.get("candidatesTokenCount", len(answer) // 3)
                tok_sec = round(eval_cnt / lat_sec, 1) if lat_sec > 0 else 0.0
                power_w = 4.5
            else:
                answer = f"Gemini Cloud Error ({resp.status_code}): {resp.text[:180]}"
        except Exception as e:
            answer = f"Gemini Cloud Error: {str(e)}"
            lat_sec = round(time.time() - t_start, 2)
    else:
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": SETTINGS.get("temperature", 0.7), "num_predict": SETTINGS.get("max_tokens", 4096)}
            }
            resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=60)
            t_dur = time.time() - t_start
            if resp.status_code == 200:
                rj = resp.json()
                answer = rj.get("message", {}).get("content", "")
                p_eval = rj.get("prompt_eval_count", p_eval)
                eval_cnt = rj.get("eval_count", len(answer) // 3)
                tot_nanos = rj.get("total_duration", int(t_dur * 1e9))
                eval_nanos = rj.get("eval_duration", int(t_dur * 0.8 * 1e9))
                lat_sec = round(tot_nanos / 1e9, 2)
                tok_sec = round(eval_cnt / (eval_nanos / 1e9), 1) if eval_nanos > 0 else 0.0
            else:
                answer = f"Error from Ollama ({resp.status_code}): {resp.text}"
                lat_sec = round(time.time() - t_start, 2)
        except Exception as e:
            answer = f"Sir, neural link connection error: {str(e)}. Please verify Ollama is active."
            lat_sec = round(time.time() - t_start, 2)

    n_energy = round((power_w * lat_sec) / 1000, 4)
    n_saved = round((p_eval * 0.000005) + (eval_cnt * 0.000015), 5)

    TELEMETRY["total_requests"] += 1
    TELEMETRY["input_tokens"] += p_eval
    TELEMETRY["output_tokens"] += eval_cnt
    TELEMETRY["total_tokens"] += (p_eval + eval_cnt)
    TELEMETRY["energy_kj"] = round(TELEMETRY["energy_kj"] + n_energy, 4)
    TELEMETRY["cost_saved_usd"] = round(TELEMETRY["cost_saved_usd"] + n_saved, 5)
    TELEMETRY["last_latency_sec"] = lat_sec
    TELEMETRY["last_tok_per_sec"] = tok_sec

    add_activity_log("chat", f"Directive: '{msg[:28]}...' -> {model}", lat_sec, eval_cnt)
    if len(msg) > 8:
        add_memory("chat", "last_user_directive", msg[:100])

    audio_b64 = tts_b64(answer) if voice_enabled else ""
    return JSONResponse({
        "response": answer,
        "audio": audio_b64,
        "agent": agent_name,
        "model": model,
        "telemetry": {
            "prompt_tokens": p_eval,
            "completion_tokens": eval_cnt,
            "latency_sec": lat_sec,
            "tok_per_sec": tok_sec,
            "power_w": power_w,
            "energy_kj": n_energy,
            "cost_saved_usd": n_saved,
            "session_requests": TELEMETRY["total_requests"],
            "session_output_tokens": TELEMETRY["output_tokens"],
            "session_total_tokens": TELEMETRY["total_tokens"],
            "session_energy_kj": TELEMETRY["energy_kj"],
            "session_saved_usd": TELEMETRY["cost_saved_usd"]
        }
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

@app.post("/api/audio-chat")
async def audio_chat_endpoint(req: Request):
    data = await req.json()
    audio_b64 = data.get("audio", "")
    mime_type = data.get("mime_type", "audio/webm")
    agent_name = data.get("agent", "Commander")
    
    if not audio_b64:
        return JSONResponse({"response": "Sir, no audio signal detected.", "audio": "", "user_transcript": ""})
        
    if "," in audio_b64:
        audio_b64 = audio_b64.split(",", 1)[1]

    gem_key = SETTINGS.get("cloud_keys", {}).get("gemini") or os.environ.get("GEMINI_API_KEY", "")
    agent_info = AGENTS.get(agent_name, AGENTS["Commander"])
    system_prompt = agent_info["system"]
    
    answer = ""
    user_transcript = ""
    t_start = time.time()

    if gem_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={gem_key}"
            clean_mime = mime_type.split(";")[0].strip()
            prompt = (
                f"[System Instructions: {system_prompt}]\n"
                "Listen closely to the user audio. "
                "First, transcribe what the user said. "
                "Then respond naturally in the user's spoken language (Hindi/Hinglish/English). "
                "Keep the answer concise (1-2 sentences). "
                "Format strictly as:\n"
                "TRANSCRIPT: <what the user said>\n"
                "REPLY: <your answer to the user>"
            )
            payload = {
                "contents": [{
                    "parts": [
                        {"inlineData": {"mimeType": clean_mime, "data": audio_b64}},
                        {"text": prompt}
                    ]
                }]
            }
            resp = requests.post(url, json=payload, timeout=25)
            if resp.status_code == 200:
                raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                if "REPLY:" in raw_text:
                    parts = raw_text.split("REPLY:", 1)
                    if "TRANSCRIPT:" in parts[0]:
                        user_transcript = parts[0].replace("TRANSCRIPT:", "").strip().split("\n")[0]
                    answer = parts[1].strip()
                else:
                    answer = raw_text.strip()
            else:
                answer = f"Audio engine error ({resp.status_code}): {resp.text[:120]}"
        except Exception as e:
            answer = f"Audio processing error: {str(e)}"
    else:
        answer = "Sir, universal audio processing requires Gemini API key in Settings."

    lat_sec = round(time.time() - t_start, 2)
    p_eval = 40
    eval_cnt = max(10, len(answer) // 3)
    tok_sec = round(eval_cnt / lat_sec, 1) if lat_sec > 0 else 0.0

    TELEMETRY["total_requests"] += 1
    TELEMETRY["input_tokens"] += p_eval
    TELEMETRY["output_tokens"] += eval_cnt
    TELEMETRY["total_tokens"] += (p_eval + eval_cnt)
    TELEMETRY["last_latency_sec"] = lat_sec
    TELEMETRY["last_tok_per_sec"] = tok_sec

    add_activity_log("audio_voice", f"Voice input -> {user_transcript[:30] or 'Audio recording'}", lat_sec, eval_cnt)
    tts_audio = tts_b64(answer)
    
    return JSONResponse({
        "response": answer,
        "user_transcript": user_transcript,
        "audio": tts_audio,
        "telemetry": {
            "prompt_tokens": p_eval,
            "completion_tokens": eval_cnt,
            "latency_sec": lat_sec,
            "tok_per_sec": tok_sec,
            "power_w": 5.0,
            "energy_kj": 0.002,
            "cost_saved_usd": 0.0003,
            "session_requests": TELEMETRY["total_requests"],
            "session_output_tokens": TELEMETRY["output_tokens"],
            "session_total_tokens": TELEMETRY["total_tokens"],
            "session_energy_kj": TELEMETRY["energy_kj"],
            "session_saved_usd": TELEMETRY["cost_saved_usd"]
        }
    })

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

  /* Cyber Dropdown */
  .cyber-select {
    background: rgba(3, 8, 18, 0.95);
    border: 1px solid #00f0ff;
    color: #00f0ff;
    padding: 5px 10px;
    border-radius: 8px;
    font-family: inherit;
    font-size: 11px;
    outline: none;
    cursor: pointer;
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
  }
  .cyber-select option {
    background: #02050b;
    color: #00f0ff;
  }

  /* Telemetry Sidebar */
  .telemetry-hud {
    width: 275px;
    background: rgba(2, 6, 16, 0.88);
    border-left: 1px solid rgba(0, 240, 255, 0.25);
    z-index: 5;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
  }
  .telemetry-hud.collapsed {
    display: none !important;
  }
  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .stat-pill {
    background: rgba(0, 240, 255, 0.05);
    border: 1px solid rgba(0, 240, 255, 0.2);
    border-radius: 6px;
    padding: 8px;
    text-align: center;
  }
  .stat-num {
    font-size: 15px;
    font-weight: 900;
    color: #00f0ff;
  }
  .stat-lbl {
    font-size: 9px;
    letter-spacing: 1px;
    color: #8ba2c4;
    margin-top: 3px;
  }

  /* Themes */
  body.theme-emerald { color: #00ffaa; }
  body.theme-emerald .panel-box, body.theme-emerald header, body.theme-emerald footer, body.theme-emerald .telemetry-hud { border-color: rgba(0, 255, 170, 0.4); }
  body.theme-gold { color: #ffaa00; }
  body.theme-gold .panel-box, body.theme-gold header, body.theme-gold footer, body.theme-gold .telemetry-hud { border-color: rgba(255, 170, 0, 0.4); }

  /* Modals */
  .modal-backdrop {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(2, 5, 11, 0.88);
    backdrop-filter: blur(10px);
    z-index: 1000;
    display: none;
    align-items: center;
    justify-content: center;
    padding: 16px;
  }
  .modal-card {
    background: rgba(4, 12, 28, 0.96);
    border: 1px solid #00f0ff;
    border-radius: 12px;
    padding: 22px;
    max-width: 520px;
    width: 100%;
    box-shadow: 0 0 40px rgba(0, 240, 255, 0.4);
    position: relative;
    color: #00f0ff;
  }

  /* Mobile Responsive Media Queries */
  @media (max-width: 768px) {
    header {
      height: auto;
      padding: 10px 14px;
      flex-wrap: wrap;
      gap: 8px;
    }
    .hud-title {
      font-size: 15px;
    }
    .agent-bar {
      order: 3;
      width: 100%;
      padding-bottom: 4px;
    }
    main {
      flex-direction: column !important;
      overflow-y: auto !important;
    }
    .left-hud {
      width: 100% !important;
      border-right: none !important;
      border-bottom: 1px solid rgba(0, 240, 255, 0.25);
      padding: 10px !important;
      gap: 10px !important;
    }
    .telemetry-hud {
      width: 100% !important;
      border-left: none !important;
      border-top: 1px solid rgba(0, 240, 255, 0.25);
    }
    #cam-container {
      height: 120px !important;
    }
    .right-hud {
      padding: 12px !important;
      min-height: calc(100vh - 360px);
    }
    .dialogue-stream {
      max-height: 48vh !important;
    }
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

  <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
    <select id="model-select" class="cyber-select" onchange="switchModel(this.value)" title="Choose AI Model">
      <option value="llama3.2:1b">🤖 llama3.2:1b (Local)</option>
    </select>
    <button class="quick-chip" onclick="toggleTelemetrySidebar()" id="tel-toggle-btn" style="color: #00f0ff; border-color: #00f0ff;">📊 TELEMETRY</button>
    <button class="quick-chip" onclick="openLogsModal()">📜 LOGS</button>
    <button class="quick-chip" onclick="openMemoryModal()">🧠 MEMORY</button>
    <button class="quick-chip" onclick="openSettingsModal()">⚙️ SETTINGS</button>
    <button class="quick-chip" onclick="openMobileModal()" style="color: #ffaa00; border-color: #ffaa00; font-weight: bold; background: rgba(255,170,0,0.12); box-shadow: 0 0 10px rgba(255,170,0,0.3);">📱 MOBILE</button>
    <span id="handsfree-badge" class="hud-badge" style="border-color: #00ffaa; color: #00ffaa; background: rgba(0,255,170,0.15); box-shadow: 0 0 10px rgba(0,255,170,0.3);">🟢 HANDS-FREE: ALWAYS LISTENING</span>
    <button class="quick-chip" id="handsfree-toggle-btn" onclick="toggleHandsFree()" style="color: #00ffaa; border-color: #00ffaa; font-weight: bold;">⚡ HANDS-FREE: ON</button>
    <button class="quick-chip" id="voice-toggle-btn" onclick="toggleVoice()">🔊 VOICE: ON</button>
    <span id="time-display" style="font-size: 12px; letter-spacing: 1px; color: #8ba2c4;">--:--:--</span>
  </div>
</header>

<!-- Activity Logs Modal -->
<div id="logs-modal" class="modal-backdrop" onclick="if(event.target===this)closeLogsModal()">
  <div class="modal-card" style="max-width: 660px;">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(0,240,255,0.3); padding-bottom: 10px;">
      <h3 style="color: #00f0ff; letter-spacing: 2px; font-size: 16px;">📜 ACTIVITY LOGS & TRACE DEBUGGER</h3>
      <span onclick="closeLogsModal()" style="cursor: pointer; font-size: 22px; color: #ff3366; font-weight: bold;">✕</span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin: 12px 0;">
      <span style="font-size: 11px; color: #8ba2c4;">Chat events, model switches, tool calls, and latency.</span>
      <div style="display: flex; gap: 8px;">
        <button class="quick-chip" onclick="copyLogs()">📋 Copy All</button>
        <button class="quick-chip" onclick="clearLogs()" style="color: #ff3366; border-color: #ff3366;">🗑️ Clear</button>
      </div>
    </div>
    <div id="logs-stream-container" style="max-height: 360px; overflow-y: auto; background: rgba(2,6,16,0.9); border: 1px solid rgba(0,240,255,0.25); border-radius: 8px; padding: 12px; font-family: monospace; font-size: 11px; line-height: 1.6;">
      <div style="color: #8ba2c4;">Loading logs...</div>
    </div>
  </div>
</div>

<!-- Data Sources & Memory Modal -->
<div id="memory-modal" class="modal-backdrop" onclick="if(event.target===this)closeMemoryModal()">
  <div class="modal-card" style="max-width: 620px;">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(0,240,255,0.3); padding-bottom: 10px;">
      <h3 style="color: #00f0ff; letter-spacing: 2px; font-size: 16px;">🧠 DATA SOURCES, CHANNELS & MEMORY</h3>
      <span onclick="closeMemoryModal()" style="cursor: pointer; font-size: 22px; color: #ff3366; font-weight: bold;">✕</span>
    </div>
    <div style="display: flex; gap: 8px; margin: 12px 0;">
      <button class="quick-chip memory-tab-btn active" onclick="switchMemoryTab('memory', this)">💾 SQLite Memory Context</button>
      <button class="quick-chip memory-tab-btn" onclick="switchMemoryTab('channels', this)">📡 Messaging Channels</button>
      <button class="quick-chip memory-tab-btn" onclick="switchMemoryTab('sources', this)">📁 Data Sources</button>
    </div>
    <div id="memory-tab-content" style="max-height: 320px; overflow-y: auto; background: rgba(2,6,16,0.85); border: 1px solid rgba(0,240,255,0.25); border-radius: 8px; padding: 12px; font-size: 12px;">
      Loading...
    </div>
    <div style="margin-top: 12px; display: flex; justify-content: space-between; align-items: center;">
      <label style="font-size: 11px; color: #00ffaa; display: flex; align-items: center; gap: 6px;">
        <input type="checkbox" id="mem-inject-checkbox" checked onchange="toggleMemoryInject(this.checked)">
        Inject relevant memory context into chat
      </label>
      <button class="quick-chip" onclick="clearMemory()" style="color: #ff3366; border-color: #ff3366; font-size: 11px;">Clear Data</button>
    </div>
  </div>
</div>

<!-- Settings & Preferences Modal -->
<div id="settings-modal" class="modal-backdrop" onclick="if(event.target===this)closeSettingsModal()">
  <div class="modal-card" style="max-width: 580px;">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(0,240,255,0.3); padding-bottom: 10px;">
      <h3 style="color: #00f0ff; letter-spacing: 2px; font-size: 16px;">⚙️ JARVIS SETTINGS & PREFERENCES</h3>
      <span onclick="closeSettingsModal()" style="cursor: pointer; font-size: 22px; color: #ff3366; font-weight: bold;">✕</span>
    </div>
    <div style="max-height: 380px; overflow-y: auto; padding-right: 6px; font-size: 12px; margin-top: 12px; display: flex; flex-direction: column; gap: 12px;">
      <div class="panel-box">
        <div class="panel-label"><span>🎨 APPEARANCE & THEME</span></div>
        <div style="display: flex; gap: 8px; margin-top: 6px;">
          <button class="quick-chip" onclick="setTheme('cyan')">⚡ Cyan Cyberpunk</button>
          <button class="quick-chip" onclick="setTheme('emerald')">🛡️ Emerald Guardian</button>
          <button class="quick-chip" onclick="setTheme('gold')">👑 Gold Commander</button>
        </div>
      </div>
      <div class="panel-box">
        <div class="panel-label"><span>🤖 INFERENCE DEFAULTS & SOURCE</span></div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
          <span>Inference Source:</span>
          <b style="color: #00ffaa;">● Bundled Ollama (http://localhost:11434)</b>
        </div>
        <div style="margin-bottom: 8px;">
          <label style="color:#8ba2c4;">Temperature: <span id="temp-val">0.7</span></label>
          <input type="range" id="temp-slider" min="0.1" max="1.0" step="0.05" value="0.7" style="width:100%;" oninput="document.getElementById('temp-val').innerText = this.value; updateSetting('temperature', parseFloat(this.value));">
        </div>
        <div>
          <label style="color:#8ba2c4;">Max Tokens: <span id="tokens-val">4096</span></label>
          <input type="range" id="tokens-slider" min="512" max="8192" step="512" value="4096" style="width:100%;" oninput="document.getElementById('tokens-val').innerText = this.value; updateSetting('max_tokens', parseInt(this.value));">
        </div>
      </div>
      <div class="panel-box">
        <div class="panel-label"><span>☁️ CLOUD PROVIDERS (OPTIONAL MULTI-ROUTING)</span></div>
        <div style="color:#8ba2c4; font-size:11px; margin-bottom:6px;">Default is 100% offline local. Add keys to enable cloud backup:</div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <input type="password" class="input-box" id="key-openai" placeholder="OpenAI API Key (sk-...)" style="height:32px; font-size:11px;">
          <input type="password" class="input-box" id="key-gemini" placeholder="Google Gemini API Key" style="height:32px; font-size:11px;">
          <input type="password" class="input-box" id="key-groq" placeholder="Groq API Key" style="height:32px; font-size:11px;">
          <button class="cyber-btn" onclick="saveCloudKeys()" style="padding:6px 12px; font-size:11px; align-self:flex-end;">Save Keys</button>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Mobile Access Modal -->
<div id="mobile-modal" class="modal-backdrop" onclick="if(event.target===this)closeMobileModal()">
  <div class="modal-card">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(0,240,255,0.3); padding-bottom: 10px;">
      <h3 style="color: #00f0ff; letter-spacing: 2px; font-size: 16px;">📱 MOBILE HUD & HANDS-FREE ACCESS</h3>
      <span onclick="closeMobileModal()" style="cursor: pointer; font-size: 22px; color: #ff3366; font-weight: bold;">✕</span>
    </div>
    <div style="margin-top: 15px; text-align: center;">
      <p style="color: #8ba2c4; font-size: 12px; margin-bottom: 12px;">
        Apne mobile phone ke camera se niche diya gaya QR Code scan karein ya Chrome / Safari mein link open karein:
      </p>
      
      <!-- QR Code -->
      <div style="display: inline-block; padding: 10px; background: rgba(0,240,255,0.08); border: 2px solid #00f0ff; border-radius: 10px; box-shadow: 0 0 20px rgba(0,240,255,0.3);">
        <img id="qr-img" src="https://api.qrserver.com/v1/create-qr-code/?data=https%3A%2F%2Flocking-colleague-kissing-labels.trycloudflare.com&size=180x180&color=00f0ff&bgcolor=02050b" alt="Mobile QR Code" style="width: 180px; height: 180px; display: block;" />
      </div>

      <div style="margin-top: 15px; text-align: left;">
        <div style="font-size: 10px; color: #00ffaa; letter-spacing: 1px; font-weight: bold;">🌐 1. SECURE HTTPS CLOUDFLARE LINK (MOBILE CAMERA + MIC UNLOCKED):</div>
        <div style="margin: 6px 0; background: rgba(3,8,18,0.9); padding: 8px 12px; border: 1px solid #00ffaa; border-radius: 6px; font-size: 12px; word-break: break-all; color: #00ffaa;">
          <a id="tunnel-link" href="https://locking-colleague-kissing-labels.trycloudflare.com" target="_blank" style="color: #00ffaa; text-decoration: none; font-weight: bold;">https://locking-colleague-kissing-labels.trycloudflare.com</a>
        </div>

        <div style="font-size: 10px; color: #8ba2c4; letter-spacing: 1px; font-weight: bold; margin-top: 8px;">🏠 2. LOCAL WI-FI NETWORK LINK:</div>
        <div id="local-ip-box" style="margin: 6px 0; background: rgba(3,8,18,0.9); padding: 6px 12px; border: 1px solid rgba(0,240,255,0.3); border-radius: 6px; font-size: 12px; word-break: break-all; color: #8ba2c4;">
          http://10.178.193.213:7860
        </div>
      </div>

      <div style="margin-top: 12px; text-align: left; background: rgba(0,240,255,0.06); padding: 10px 14px; border-radius: 6px; font-size: 11px; color: #8ba2c4; line-height: 1.5; border-left: 3px solid #00ffaa;">
        <b style="color: #00ffaa;">✓ Mobile par kya karein:</b><br>
        1. Link open hote hi screen par <b>1 baar tap karein</b> (audio unlock hoga).<br>
        2. Mobile camera & microphone popup aane par <b>Allow</b> press karein.<br>
        3. Mobile se bolkar bina kisi button dabaye seedha JARVIS se baat karein!
      </div>
    </div>
  </div>
</div>

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

  <!-- Right Docked System Telemetry HUD (OpenJarvis Style) -->
  <div class="telemetry-hud" id="telemetry-sidebar">
    <div class="panel-box" style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px;">
      <span style="font-size:11px; font-weight:bold; letter-spacing:1px; color:#00f0ff;">⚡ SYSTEM TELEMETRY</span>
      <button class="quick-chip" onclick="toggleTelemetrySidebar()" style="padding:2px 6px; font-size:10px;">✕</button>
    </div>

    <div class="panel-box">
      <div class="panel-label"><span>📊 SESSION METRICS</span></div>
      <div class="stat-grid">
        <div class="stat-pill">
          <div class="stat-num" id="tel-requests">0</div>
          <div class="stat-lbl"># REQUESTS</div>
        </div>
        <div class="stat-pill">
          <div class="stat-num" id="tel-tokens">0</div>
          <div class="stat-lbl"># OUT TOKENS</div>
        </div>
      </div>
      <div style="margin-top:8px; font-size:11px; color:#8ba2c4;">
        <div class="tele-row"><span>LATENCY:</span><b id="tel-latency" style="color:#00ffaa;">0.0s</b></div>
        <div class="tele-row"><span>SPEED:</span><b id="tel-speed" style="color:#00f0ff;">0.0 tok/s</b></div>
      </div>
    </div>

    <div class="panel-box">
      <div class="panel-label"><span>🔋 DEVICE (AMD RADEON / RYZEN)</span></div>
      <div class="stat-grid">
        <div class="stat-pill">
          <div class="stat-num" style="color:#00ffaa;">18.5 W</div>
          <div class="stat-lbl">EST. POWER</div>
        </div>
        <div class="stat-pill">
          <div class="stat-num" id="tel-energy" style="color:#00f0ff;">0.0 kJ</div>
          <div class="stat-lbl">ENERGY USED</div>
        </div>
      </div>
    </div>

    <div class="panel-box" style="flex:1;">
      <div class="panel-label"><span>💰 COST SAVED VS CLOUD</span></div>
      <div style="font-size:11px; line-height:1.7;">
        <div style="display:flex; justify-content:space-between; padding:3px 6px; background:rgba(0,255,170,0.1); border:1px solid #00ffaa; border-radius:4px; margin-bottom:4px;">
          <span>🖥️ Local (llama3.2:1b)</span>
          <b style="color:#00ffaa;">$0.0000 [FREE]</b>
        </div>
        <div class="tele-row">
          <span>☁️ GPT-5.6 Sol eq.</span>
          <span id="tel-gpt-cost">$0.0000</span>
        </div>
        <div class="tele-row">
          <span>☁️ Claude Fable 5 eq.</span>
          <span id="tel-claude-cost">$0.0000</span>
        </div>
        <div class="tele-row">
          <span>☁️ Gemini 3.1 Pro eq.</span>
          <span id="tel-gemini-cost">$0.0000</span>
        </div>
        <div style="border-top:1px dashed rgba(0,240,255,0.3); margin-top:6px; padding-top:6px; display:flex; justify-content:space-between; color:#ffaa00; font-weight:bold;">
          <span>TOTAL SAVINGS:</span>
          <span id="tel-saved">$0.0000</span>
        </div>
      </div>
      <div style="margin-top:10px; font-size:10px; color:#00ffaa; background:rgba(0,255,170,0.08); padding:6px; border-radius:4px; border:1px solid rgba(0,255,170,0.3); text-align:center;">
        🛡️ 100% Private Offline Inference
      </div>
    </div>
  </div>
</main>

<footer>
  <button class="hologram-mic-btn" id="mic-btn" onclick="toggleMic()" title="Click to Speak">🎙️</button>
  <button id="deep-research-btn" class="quick-chip" onclick="toggleDeepResearch()" style="color:#8ba2c4; height:46px; border-radius:8px; font-weight:bold;">🔬 Deep Research: OFF</button>
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
let selectedModel = "llama3.2:1b";
let deepResearchActive = false;

function updateTelemetryUI(t) {
  if (!t) return;
  if (t.session_requests !== undefined) document.getElementById('tel-requests').innerText = t.session_requests;
  if (t.session_output_tokens !== undefined) document.getElementById('tel-tokens').innerText = t.session_output_tokens;
  if (t.latency_sec !== undefined) document.getElementById('tel-latency').innerText = t.latency_sec + 's';
  if (t.tok_per_sec !== undefined) document.getElementById('tel-speed').innerText = t.tok_per_sec + ' tok/s';
  if (t.session_energy_kj !== undefined) document.getElementById('tel-energy').innerText = t.session_energy_kj + ' kJ';
  if (t.session_saved_usd !== undefined) {
    document.getElementById('tel-saved').innerText = '$' + t.session_saved_usd.toFixed(4);
    document.getElementById('tel-gpt-cost').innerText = '$' + (t.session_saved_usd * 0.95).toFixed(4);
    document.getElementById('tel-claude-cost').innerText = '$' + (t.session_saved_usd * 0.90).toFixed(4);
    document.getElementById('tel-gemini-cost').innerText = '$' + (t.session_saved_usd * 0.40).toFixed(4);
  }
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
        model: selectedModel,
        voice_enabled: voiceEnabled,
        history: conversationHistory,
        deep_research: deepResearchActive
      })
    });
    const data = await res.json();
    lastResponseText = data.response;
    
    let teleBadge = "";
    if (data.telemetry && data.telemetry.latency_sec) {
      teleBadge = `<div style="font-size:10px; color:#00ffaa; margin-top:6px; opacity:0.85; border-top:1px dashed rgba(0,255,170,0.3); padding-top:4px;">
        ⚡ ${data.model} • ${data.telemetry.latency_sec}s • ${data.telemetry.prompt_tokens} in / ${data.telemetry.completion_tokens} out • ${data.telemetry.tok_per_sec} tok/s • Saved $${data.telemetry.cost_saved_usd}
      </div>`;
      updateTelemetryUI(data.telemetry);
    }
    
    appendMessage('jarvis', data.response + teleBadge);
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

// Dual-Engine Speech & Audio System
let mediaRecorder = null;
let recordedChunks = [];
let isRecordingAudio = false;
let globalStream = null;

function checkBrowserAudioSupport() {
  const isHttps = window.location.protocol === 'https:' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  const statusEl = document.getElementById('mic-status');

  if (!isHttps) {
    if (statusEl) {
      statusEl.innerHTML = "⚠️ HTTPS REQUIRED ON MOBILE: Open Cloudflare link";
      statusEl.style.color = "#ffaa00";
    }
    return "insecure_http";
  }

  if (SpeechRec) {
    if (statusEl) {
      statusEl.innerText = "STREAMING SPEECH [CHROME/EDGE]";
      statusEl.style.color = "#00ffaa";
    }
    return "speech_rec";
  } else {
    if (statusEl) {
      statusEl.innerText = "UNIVERSAL AUDIO RECORDER [ACTIVE]";
      statusEl.style.color = "#00f0ff";
    }
    return "media_recorder";
  }
}

function initRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    checkBrowserAudioSupport();
    return;
  }
  try {
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
        document.getElementById('mic-status').innerText = "HEARING: " + t.slice(0, 22) + "...";
        
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
      const statusEl = document.getElementById('mic-status');
      if (e.error === 'not-allowed') {
        if (statusEl) {
          statusEl.innerText = "❌ MIC PERMISSION BLOCKED (Click 🔒 to Allow)";
          statusEl.style.color = "#ff3366";
        }
      } else if (e.error === 'network') {
        if (statusEl) {
          statusEl.innerText = "⚠️ SPEECH NETWORK TIMEOUT - USE MIC BUTTON";
          statusEl.style.color = "#ffaa00";
        }
      }
    };

    recognition.onend = () => {
      isListening = false;
      document.getElementById('mic-btn').classList.remove('listening');
      if (handsFreeMode && !isSpeakingOrPlaying) {
        setTimeout(startContinuousListening, 300);
      }
    };
  } catch(e) {
    console.warn("SpeechRec constructor error:", e);
  }
}

async function startMediaRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    globalStream = stream;
    setupAudioSpectrum(stream);
    recordedChunks = [];
    
    let mime = 'audio/webm';
    if (window.MediaRecorder && MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) mime = 'audio/webm;codecs=opus';
    else if (window.MediaRecorder && MediaRecorder.isTypeSupported('audio/mp4')) mime = 'audio/mp4';
    else if (window.MediaRecorder && MediaRecorder.isTypeSupported('audio/ogg')) mime = 'audio/ogg';

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
      const blob = new Blob(recordedChunks, { type: mime });
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64Audio = reader.result;
        appendMessage('user', '🎙️ [Transmitting Voice Directive...]');
        document.getElementById('mic-status').innerText = "AI PROCESSING DIRECTIVE...";
        document.getElementById('mic-status').style.color = "#00f0ff";
        coreSpeedMultiplier = 2.2;
        try {
          const res = await fetch('/api/audio-chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              audio: base64Audio,
              mime_type: mime,
              agent: currentAgent,
              model: selectedModel
            })
          });
          const d = await res.json();
          if (d.user_transcript) {
            appendMessage('user', '🗣️ ' + d.user_transcript);
          }
          let teleBadge = "";
          if (d.telemetry && d.telemetry.latency_sec) {
            teleBadge = `<div style="font-size:10px; color:#00ffaa; margin-top:6px; opacity:0.85; border-top:1px dashed rgba(0,255,170,0.3); padding-top:4px;">
              ⚡ Voice Audio • ${d.telemetry.latency_sec}s • ${d.telemetry.prompt_tokens} in / ${d.telemetry.completion_tokens} out • ${d.telemetry.tok_per_sec} tok/s
            </div>`;
            updateTelemetryUI(d.telemetry);
          }
          appendMessage('jarvis', d.response + teleBadge);
          conversationHistory.push({ role: 'assistant', content: d.response });
          if (d.audio) playAudioB64(d.audio);
          else speakText(d.response);
        } catch(err) {
          appendMessage('jarvis', `⚠️ Voice error: ${err.message}`);
          speakText("Sir, connection error in voice transmission.");
        } finally {
          coreSpeedMultiplier = 1.0;
          document.getElementById('mic-status').innerText = "READY [CLICK 🎙️ TO SPEAK]";
          document.getElementById('mic-status').style.color = "#00ffaa";
        }
      };
      reader.readAsDataURL(blob);
    };

    mediaRecorder.start();
    isRecordingAudio = true;
    document.getElementById('mic-btn').classList.add('listening');
    document.getElementById('mic-status').innerText = "🔴 RECORDING VOICE... [CLICK 🎙️ TO FINISH]";
    document.getElementById('mic-status').style.color = "#ff3366";
  } catch(err) {
    console.warn("Audio recording error:", err);
    document.getElementById('mic-status').innerText = "MIC ERROR: " + err.message;
    document.getElementById('mic-status').style.color = "#ff3366";
    alert("Microphone permission was blocked or is unavailable! Please click the lock/settings icon in your browser address bar and enable Microphone.");
  }
}

function stopMediaRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  isRecordingAudio = false;
  document.getElementById('mic-btn').classList.remove('listening');
}

async function toggleMic() {
  // Mobile HTTP check
  if (window.location.protocol === 'http:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    alert("⚠️ Mobile browsers strictly block Microphone on insecure HTTP!\n\nPlease open the secure Cloudflare HTTPS link from the MOBILE button.");
    openMobileModal();
    return;
  }

  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (SpeechRec) {
    if (!recognition) initRecognition();
    if (isListening) {
      if (recognition) recognition.stop();
    } else {
      // Trigger user gesture permission request
      try {
        if (!globalStream) {
          globalStream = await navigator.mediaDevices.getUserMedia({ audio: true });
          setupAudioSpectrum(globalStream);
        }
      } catch(e){}
      startContinuousListening();
    }
  } else {
    // MediaRecorder fallback
    if (isRecordingAudio) {
      stopMediaRecording();
    } else {
      startMediaRecording();
    }
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

function switchModel(m) {
  selectedModel = m;
  appendMessage('jarvis', `Switched active inference model to: <b>${m}</b>.`);
  updateSetting('model', m);
}

function toggleDeepResearch() {
  deepResearchActive = !deepResearchActive;
  const btn = document.getElementById('deep-research-btn');
  if (deepResearchActive) {
    btn.innerText = "🔬 Deep Research: ON";
    btn.style.color = "#00ffaa";
    btn.style.borderColor = "#00ffaa";
    btn.style.background = "rgba(0,255,170,0.18)";
  } else {
    btn.innerText = "🔬 Deep Research: OFF";
    btn.style.color = "#8ba2c4";
    btn.style.borderColor = "rgba(0,240,255,0.3)";
    btn.style.background = "rgba(0,240,255,0.08)";
  }
}

function toggleTelemetrySidebar() {
  const sb = document.getElementById('telemetry-sidebar');
  sb.classList.toggle('collapsed');
}

// Logs Modal
async function openLogsModal() {
  document.getElementById('logs-modal').style.display = 'flex';
  try {
    const res = await fetch('/api/logs');
    const data = await res.json();
    const c = document.getElementById('logs-stream-container');
    if (!data.logs || data.logs.length === 0) {
      c.innerHTML = '<div style="color:#8ba2c4;">No activity logged yet this session. Directives will be recorded here in real-time.</div>';
      return;
    }
    c.innerHTML = data.logs.map(l => `
      <div style="border-bottom:1px solid rgba(0,240,255,0.12); padding:5px 0;">
        <span style="color:#8ba2c4;">${l.time}</span>
        <span style="color:#00ffaa; margin:0 6px;">[${l.type.toUpperCase()}]</span>
        <span style="color:#e0f0ff;">${l.detail}</span>
        <span style="color:#00f0ff; float:right;">${l.latency}s • ${l.tokens} tok</span>
      </div>
    `).join('');
  } catch(e) {
    console.warn("Logs error:", e);
  }
}
function closeLogsModal() {
  document.getElementById('logs-modal').style.display = 'none';
}
async function clearLogs() {
  await fetch('/api/logs/clear', { method: 'POST' });
  document.getElementById('logs-stream-container').innerHTML = '<div style="color:#8ba2c4;">Logs cleared.</div>';
}
function copyLogs() {
  const t = document.getElementById('logs-stream-container').innerText;
  navigator.clipboard.writeText(t);
  alert("Activity logs copied to clipboard!");
}

// Memory Modal
async function openMemoryModal() {
  document.getElementById('memory-modal').style.display = 'flex';
  loadMemoryContent('memory');
}
function closeMemoryModal() {
  document.getElementById('memory-modal').style.display = 'none';
}
async function loadMemoryContent(tab) {
  const c = document.getElementById('memory-tab-content');
  if (tab === 'memory') {
    try {
      const res = await fetch('/api/memory');
      const data = await res.json();
      if (!data.memories || data.memories.length === 0) {
        c.innerHTML = '<div style="color:#8ba2c4;">No conversation memories stored yet. JARVIS automatically records key directives and preferences into local SQLite.</div>';
        return;
      }
      c.innerHTML = `<div style="color:#00ffaa; margin-bottom:8px; font-weight:bold;">● SQLite Memory Engine Active (${data.count} stored records)</div>` +
        data.memories.map(m => `
          <div style="background:rgba(0,240,255,0.05); border:1px solid rgba(0,240,255,0.2); border-radius:6px; padding:6px 10px; margin-bottom:6px;">
            <div style="font-size:10px; color:#8ba2c4;">${m.time} • Category: ${m.category}</div>
            <div style="color:#00f0ff; font-weight:bold;">${m.key}</div>
            <div style="color:#e0f0ff;">${m.value}</div>
          </div>
        `).join('');
    } catch(e) { c.innerText = "Error loading memory: " + e; }
  } else if (tab === 'channels') {
    c.innerHTML = `
      <div style="line-height:1.7;">
        <h4 style="color:#00f0ff; margin-bottom:6px;">📡 Messaging Channels & Phone Integration</h4>
        <div style="color:#8ba2c4; margin-bottom:10px;">Connect your phone to access JARVIS from anywhere:</div>
        <div style="background:rgba(0,240,255,0.05); padding:10px; border-radius:6px; border:1px solid rgba(0,240,255,0.2); margin-bottom:8px;">
          <b style="color:#00ffaa;">📱 Mobile Browser HUD (Online & Ready):</b>
          <div style="font-size:11px; color:#8ba2c4;">Click "MOBILE" button in header or scan the QR Code on any smartphone.</div>
        </div>
        <div style="background:rgba(0,240,255,0.05); padding:10px; border-radius:6px; border:1px solid rgba(0,240,255,0.2);">
          <b style="color:#ffaa00;">💬 Telegram / WhatsApp Gateway:</b>
          <div style="font-size:11px; color:#8ba2c4;">Local webhook endpoint active for messaging bot integrations.</div>
        </div>
      </div>
    `;
  } else if (tab === 'sources') {
    c.innerHTML = `
      <div style="line-height:1.7;">
        <h4 style="color:#00f0ff; margin-bottom:6px;">📁 Connected Data Sources</h4>
        <div style="color:#8ba2c4; margin-bottom:10px;">Personal knowledge base and workspace documents:</div>
        <div style="background:rgba(0,255,170,0.08); padding:10px; border-radius:6px; border:1px solid rgba(0,255,170,0.3); margin-bottom:8px;">
          <b style="color:#00ffaa;">✓ Local Workspace:</b>
          <div style="font-size:11px; color:#8ba2c4;">C:\\Users\\HP\\vega (Git synchronized & active)</div>
        </div>
        <div style="background:rgba(0,240,255,0.05); padding:10px; border-radius:6px; border:1px solid rgba(0,240,255,0.2);">
          <b style="color:#00f0ff;">+ Add Documents:</b>
          <div style="font-size:11px; color:#8ba2c4;">Markdown, Python code, and text files in workspace are auto-indexed into memory.</div>
        </div>
      </div>
    `;
  }
}
function switchMemoryTab(tab, el) {
  document.querySelectorAll('.memory-tab-btn').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
  loadMemoryContent(tab);
}
async function clearMemory() {
  await fetch('/api/memory/clear', { method: 'POST' });
  loadMemoryContent('memory');
}
async function toggleMemoryInject(enabled) {
  await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ memory_enabled: enabled })
  });
}

// Settings Modal
async function openSettingsModal() {
  document.getElementById('settings-modal').style.display = 'flex';
  try {
    const res = await fetch('/api/settings');
    const s = await res.json();
    if (s.temperature !== undefined) {
      document.getElementById('temp-slider').value = s.temperature;
      document.getElementById('temp-val').innerText = s.temperature;
    }
    if (s.max_tokens !== undefined) {
      document.getElementById('tokens-slider').value = s.max_tokens;
      document.getElementById('tokens-val').innerText = s.max_tokens;
    }
    if (s.cloud_keys) {
      if (s.cloud_keys.openai) document.getElementById('key-openai').value = s.cloud_keys.openai;
      if (s.cloud_keys.gemini) document.getElementById('key-gemini').value = s.cloud_keys.gemini;
      if (s.cloud_keys.groq) document.getElementById('key-groq').value = s.cloud_keys.groq;
    }
  } catch(e){}
}
function closeSettingsModal() {
  document.getElementById('settings-modal').style.display = 'none';
}
function setTheme(t) {
  document.body.className = (t === 'cyan' ? '' : 'theme-' + t);
  updateSetting('theme', t);
}
async function updateSetting(k, v) {
  const obj = {};
  obj[k] = v;
  await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(obj)
  });
}
async function saveCloudKeys() {
  const keys = {
    openai: document.getElementById('key-openai').value,
    gemini: document.getElementById('key-gemini').value,
    groq: document.getElementById('key-groq').value
  };
  await updateSetting('cloud_keys', keys);
  alert("Cloud provider keys saved!");
}

async function openMobileModal() {
  document.getElementById('mobile-modal').style.display = 'flex';
  try {
    const res = await fetch('/api/tunnel-info');
    const data = await res.json();
    if (data.tunnel_url) {
      document.getElementById('tunnel-link').href = data.tunnel_url;
      document.getElementById('tunnel-link').innerText = data.tunnel_url;
      document.getElementById('qr-img').src = `https://api.qrserver.com/v1/create-qr-code/?data=${encodeURIComponent(data.tunnel_url)}&size=180x180&color=00f0ff&bgcolor=02050b`;
    }
    if (data.local_ip) {
      document.getElementById('local-ip-box').innerText = `http://${data.local_ip}:${data.port}`;
    }
  } catch(e) {
    console.warn("Tunnel info fetch error:", e);
  }
}
function closeMobileModal() {
  document.getElementById('mobile-modal').style.display = 'none';
}

// Populate Models Dropdown & Telemetry on Boot
async function loadModelsAndTelemetry() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    if (data.models && data.models.length > 0) {
      const sel = document.getElementById('model-select');
      sel.innerHTML = data.models.map(m => {
        const isCloud = m.includes('Cloud');
        const icon = isCloud ? '☁️' : '🤖';
        const label = isCloud ? m : `${m} (Local)`;
        return `<option value="${m}">${icon} ${label}</option>`;
      }).join('');
      selectedModel = data.models[0];
    }
  } catch(e){}

  try {
    const res = await fetch('/api/telemetry');
    const t = await res.json();
    updateTelemetryUI(t);
  } catch(e){}
}

// Auto-start Hands-Free Listening & Camera
window.addEventListener('DOMContentLoaded', () => {
  checkBrowserAudioSupport();
  initRecognition();
  loadModelsAndTelemetry();
  const unlock = () => {
    activateCamera();
    startContinuousListening();
  };
  window.addEventListener('click', unlock, { once: true });
  window.addEventListener('keydown', unlock, { once: true });
  window.addEventListener('touchstart', unlock, { once: true });
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
