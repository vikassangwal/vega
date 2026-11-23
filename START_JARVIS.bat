@echo off
title JARVIS OMEGA - AI Agent Control Center
color 0b
echo ========================================================
echo       INITIALIZING JARVIS OMEGA (VEGA 3D HUD)
echo ========================================================
echo.
echo [*] Working Directory: C:\Users\HP\.gemini\antigravity\scratch\vega
echo [*] Checking Local AI Neural Engine (Ollama)...

curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [*] Starting Ollama Local AI Service...
    start /B "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    timeout /t 2 /nobreak >nul
) else (
    echo [?] Ollama Local AI Engine Online!
)

echo [*] Model Configured: llama3.2:1b (100%% Offline Local)
echo [*] Launching Holographic 3D Web HUD...
echo.

cd /d "C:\Users\HP\.gemini\antigravity\scratch\vega"

timeout /t 2 /nobreak >nul
start "" http://localhost:7860

py -3.11 app.py

pause
