"""
JARVIS OMEGA - Media Tools
Built-in tools for media control and playback.
"""
from __future__ import annotations

import asyncio
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.tool_registry import ToolRegistry, ToolMetadata, ToolParameter
from tools.builtin.file_tools import _workspace

logger = logging.getLogger("vega.tools.media")

SYSTEM = platform.system().lower()
IS_WINDOWS = SYSTEM == "windows"
IS_MACOS = SYSTEM == "darwin"
IS_LINUX = SYSTEM == "linux"


async def media_play(params: dict) -> dict:
    path = params.get("path", "")
    player = params.get("player")

    if not path:
        return {"success": False, "error": "Path required"}

    try:
        media_path = _workspace.resolve_path(path)

        if IS_MACOS:
            cmd = ["open", "-a", player or "QuickTime Player", str(media_path)]
        elif IS_WINDOWS:
            cmd = ["start", "", str(media_path)]
        else:
            cmd = [player or "mpv", str(media_path)]

        proc = await asyncio.create_subprocess_exec(*cmd)
        return {"success": True, "pid": proc.pid, "message": "Playing media"}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def media_pause(params: dict) -> dict:
    try:
        if IS_MACOS:
            subprocess.run(["osascript", "-e", "tell application \"Music\" to pause"])
            subprocess.run(["osascript", "-e", "tell application \"Spotify\" to pause"])
            subprocess.run(["osascript", "-e", "tell application \"QuickTime Player\" to pause"])
        elif IS_LINUX:
            subprocess.run(["playerctl", "pause"])
        return {"success": True, "message": "Media paused"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def media_resume(params: dict) -> dict:
    try:
        if IS_MACOS:
            subprocess.run(["osascript", "-e", "tell application \"Music\" to play"])
            subprocess.run(["osascript", "-e", "tell application \"Spotify\" to play"])
            subprocess.run(["osascript", "-e", "tell application \"QuickTime Player\" to play"])
        elif IS_LINUX:
            subprocess.run(["playerctl", "play"])
        return {"success": True, "message": "Media resumed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def media_next(params: dict) -> dict:
    try:
        if IS_MACOS:
            subprocess.run(["osascript", "-e", "tell application \"Music\" to next track"])
            subprocess.run(["osascript", "-e", "tell application \"Spotify\" to next track"])
        elif IS_LINUX:
            subprocess.run(["playerctl", "next"])
        return {"success": True, "message": "Next track"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def media_previous(params: dict) -> dict:
    try:
        if IS_MACOS:
            subprocess.run(["osascript", "-e", "tell application \"Music\" to previous track"])
            subprocess.run(["osascript", "-e", "tell application \"Spotify\" to previous track"])
        elif IS_LINUX:
            subprocess.run(["playerctl", "previous"])
        return {"success": True, "message": "Previous track"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def media_seek(params: dict) -> dict:
    position = params.get("position", 0)

    try:
        if IS_MACOS:
            subprocess.run(["osascript", "-e", f"tell application \"Music\" to set player position to {position}"])
            subprocess.run(["osascript", "-e", f"tell application \"Spotify\" to set player position to {position}"])
        elif IS_LINUX:
            subprocess.run(["playerctl", "position", str(position)])
        return {"success": True, "position": position}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def media_volume(params: dict) -> dict:
    app = params.get("app", "Music")
    volume = params.get("volume", 50)

    try:
        if IS_MACOS:
            subprocess.run(["osascript", "-e", f'tell application "{app}" to set sound volume to {volume}'])
        elif IS_LINUX:
            subprocess.run(["playerctl", "volume", str(volume / 100.0)])
        return {"success": True, "volume": volume}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def media_info(params: dict) -> dict:
    app = params.get("app")

    try:
        if IS_MACOS:
            script = 'tell application "Music" to {name, artist, album, duration, player position}'
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {
                    "success": True,
                    "title": parts[0] if len(parts) > 0 else "",
                    "artist": parts[1] if len(parts) > 1 else "",
                    "album": parts[2] if len(parts) > 2 else "",
                    "duration": parts[3] if len(parts) > 3 else "",
                    "position": parts[4] if len(parts) > 4 else "",
                }

        elif IS_LINUX:
            result = subprocess.run(
                ["playerctl", "metadata", "--format", "{{title}}|{{artist}}|{{album}}|{{duration}}|{{position}}"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split("|")
                return {
                    "success": True,
                    "title": parts[0] if len(parts) > 0 else "",
                    "artist": parts[1] if len(parts) > 1 else "",
                    "album": parts[2] if len(parts) > 2 else "",
                    "duration": parts[3] if len(parts) > 3 else "",
                    "position": parts[4] if len(parts) > 4 else "",
                }

        return {"success": False, "error": "Could not get media info"}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def media_playlist(params: dict) -> dict:
    action = params.get("action", "list")
    name = params.get("name", "default")
    tracks = params.get("tracks", [])

    return {
        "success": True,
        "action": action,
        "playlist": name,
        "message": f"Playlist {action} not fully implemented",
    }


def register_media_tools(registry: ToolRegistry) -> None:
    tools = [
        ("media_play", media_play, "Play media file", [
            ToolParameter("path", "string", "Media file path", required=True),
            ToolParameter("player", "string", "Player application"),
        ], {"requires_approval": False, "sandboxed": False}),

        ("media_pause", media_pause, "Pause media playback", [], {"requires_approval": False, "sandboxed": False}),

        ("media_resume", media_resume, "Resume media playback", [], {"requires_approval": False, "sandboxed": False}),

        ("media_next", media_next, "Next track", [], {"requires_approval": False, "sandboxed": False}),

        ("media_previous", media_previous, "Previous track", [], {"requires_approval": False, "sandboxed": False}),

        ("media_seek", media_seek, "Seek to position", [
            ToolParameter("position", "integer", "Position in seconds", required=True),
        ], {"requires_approval": False, "sandboxed": False}),

        ("media_volume", media_volume, "Set media volume", [
            ToolParameter("app", "string", "Application name", default="Music"),
            ToolParameter("volume", "integer", "Volume 0-100", default=50),
        ], {"requires_approval": False, "sandboxed": False}),

        ("media_info", media_info, "Get current media info", [
            ToolParameter("app", "string", "Application to query"),
        ], {"requires_approval": False, "sandboxed": False}),

        ("media_playlist", media_playlist, "Playlist operations", [
            ToolParameter("action", "string", "Action: list, create, add, remove, play", default="list"),
            ToolParameter("name", "string", "Playlist name", default="default"),
            ToolParameter("tracks", "array", "Track paths/URIs"),
        ], {"requires_approval": False, "sandboxed": False}),
    ]

    for name, func, desc, params, extra in tools:
        metadata = ToolMetadata(
            name=name,
            description=desc,
            category="media",
            parameters=params,
            returns={"type": "object"},
            sandboxed=extra.get("sandboxed", False),
            requires_approval=extra.get("requires_approval", False),
            tags=["media", "music", "video", "playback"],
        )
        registry.register(name, func, metadata)

    logger.info("Registered media tools")


__all__ = [
    "register_media_tools",
]