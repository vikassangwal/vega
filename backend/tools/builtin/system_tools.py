"""
JARVIS OMEGA - System Tools
Built-in tools for system control and monitoring.
"""
from __future__ import annotations

import asyncio
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

from tools.tool_registry import ToolRegistry, ToolMetadata, ToolParameter
from tools.builtin.file_tools import _workspace

logger = logging.getLogger("vega.tools.system")

SYSTEM = platform.system().lower()
IS_WINDOWS = SYSTEM == "windows"
IS_MACOS = SYSTEM == "darwin"
IS_LINUX = SYSTEM == "linux"


async def system_info(params: dict) -> dict:
    detailed = params.get("detailed", True)

    try:
        info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "boot_time": psutil.boot_time(),
            "uptime": time.time() - psutil.boot_time(),
        }

        if detailed:
            info["cpu"] = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "usage_percent": psutil.cpu_percent(interval=1),
                "per_core": psutil.cpu_percent(interval=1, percpu=True),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            }

            mem = psutil.virtual_memory()
            info["memory"] = {
                "total": mem.total,
                "available": mem.available,
                "used": mem.used,
                "percent": mem.percent,
            }

            swap = psutil.swap_memory()
            info["swap"] = {
                "total": swap.total,
                "used": swap.used,
                "free": swap.free,
                "percent": swap.percent,
            }

            disk = psutil.disk_usage("/")
            info["disk"] = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            }

            info["partitions"] = [
                {"device": p.device, "mountpoint": p.mountpoint, "fstype": p.fstype, "opts": p.opts}
                for p in psutil.disk_partitions()
            ]

            net = psutil.net_io_counters()
            info["network"] = {
                "bytes_sent": net.bytes_sent,
                "bytes_recv": net.bytes_recv,
                "packets_sent": net.packets_sent,
                "packets_recv": net.packets_recv,
            }

            info["interfaces"] = {}
            for name, addrs in psutil.net_if_addrs().items():
                info["interfaces"][name] = [
                    {"family": str(addr.family), "address": addr.address, "netmask": addr.netmask, "broadcast": addr.broadcast}
                    for addr in addrs
                ]

            battery = psutil.sensors_battery()
            if battery:
                info["battery"] = {
                    "percent": battery.percent,
                    "power_plugged": battery.power_plugged,
                    "time_left": battery.secsleft if battery.secsleft != -1 else None,
                }

            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    info["temperatures"] = {
                        name: [{"label": t.label, "current": t.current, "high": t.high, "critical": t.critical}
                               for t in sensors]
                        for name, sensors in temps.items()
                    }
            except Exception:
                pass

        return {"success": True, "info": info}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def process_list(params: dict) -> dict:
    sort_by = params.get("sort_by", "cpu")
    limit = params.get("limit", 50)
    filter_name = params.get("filter", "").lower()

    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'status', 'create_time', 'cmdline']):
            try:
                info = proc.info
                if filter_name and filter_name not in (info['name'] or '').lower():
                    continue

                processes.append({
                    "pid": info['pid'],
                    "name": info['name'],
                    "username": info['username'],
                    "cpu_percent": info['cpu_percent'] or 0,
                    "memory_percent": info['memory_percent'] or 0,
                    "status": info['status'],
                    "started": info['create_time'],
                    "cmdline": info['cmdline'][:3] if info['cmdline'] else [],
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        sort_key = {"cpu": "cpu_percent", "memory": "memory_percent", "name": "name", "pid": "pid"}.get(sort_by, "cpu_percent")
        processes.sort(key=lambda p: p.get(sort_key, 0), reverse=True)

        return {"success": True, "processes": processes[:limit], "total": len(processes)}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def process_kill(params: dict) -> dict:
    pid = params.get("pid")
    force = params.get("force", False)

    if not pid:
        return {"success": False, "error": "PID required"}

    try:
        proc = psutil.Process(pid)
        if force:
            proc.kill()
        else:
            proc.terminate()

        try:
            proc.wait(timeout=5)
        except psutil.TimeoutExpired:
            if not force:
                proc.kill()
                proc.wait(timeout=5)

        return {"success": True, "pid": pid, "message": "Process terminated"}

    except psutil.NoSuchProcess:
        return {"success": False, "error": "Process not found"}
    except psutil.AccessDenied:
        return {"success": False, "error": "Permission denied"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def app_open(params: dict) -> dict:
    app = params.get("app", "")
    args = params.get("args", [])

    if not app:
        return {"success": False, "error": "Application name required"}

    try:
        if IS_MACOS:
            cmd = ["open", "-a", app] + args
        elif IS_WINDOWS:
            cmd = ["start", "", app] + args
        else:
            cmd = [app] + args

        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        return {"success": True, "pid": proc.pid, "message": f"Started {app}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def app_close(params: dict) -> dict:
    app = params.get("app", "")

    if not app:
        return {"success": False, "error": "Application name required"}

    try:
        killed = 0
        for proc in psutil.process_iter(['name']):
            if app.lower() in (proc.info['name'] or '').lower():
                proc.terminate()
                killed += 1

        await asyncio.sleep(1)

        for proc in psutil.process_iter(['name']):
            if app.lower() in (proc.info['name'] or '').lower():
                try:
                    proc.kill()
                    killed += 1
                except Exception:
                    pass

        return {"success": True, "killed": killed, "message": f"Closed {app}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def volume_control(params: dict) -> dict:
    action = params.get("action", "get")
    value = params.get("value", 50)
    step = params.get("step", 10)

    try:
        if IS_MACOS:
            return await _volume_macos(action, value, step)
        elif IS_WINDOWS:
            return await _volume_windows(action, value, step)
        elif IS_LINUX:
            return await _volume_linux(action, value, step)
        else:
            return {"success": False, "error": f"Unsupported platform: {SYSTEM}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def _volume_macos(action: str, value: int, step: int) -> dict:
    import subprocess

    if action == "get":
        result = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True, text=True
        )
        return {"success": True, "volume": int(result.stdout.strip()), "muted": False}

    elif action == "set":
        subprocess.run(["osascript", "-e", f"set volume output volume {max(0, min(100, value))}"])
        return {"success": True, "volume": value}

    elif action in ["up", "down"]:
        delta = step if action == "up" else -step
        subprocess.run(["osascript", "-e", f"set volume output volume (output volume of (get volume settings) + {delta})"])
        result = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True, text=True
        )
        return {"success": True, "volume": int(result.stdout.strip())}

    elif action == "mute":
        subprocess.run(["osascript", "-e", "set volume output muted true"])
        return {"success": True, "muted": True}

    elif action == "unmute":
        subprocess.run(["osascript", "-e", "set volume output muted false"])
        return {"success": True, "muted": False}

    elif action == "toggle_mute":
        subprocess.run(["osascript", "-e", "set volume output muted not (output muted of (get volume settings))"])
        return {"success": True, "message": "Toggled mute"}

    return {"success": False, "error": f"Unknown action: {action}"}


async def _volume_windows(action: str, value: int, step: int) -> dict:
    return {"success": False, "error": "Windows volume control requires pycaw/nircmd"}


async def _volume_linux(action: str, value: int, step: int) -> dict:
    import subprocess

    if action == "get":
        result = subprocess.run(
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
            capture_output=True, text=True
        )
        return {"success": True, "raw": result.stdout.strip()}

    elif action == "set":
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{max(0, min(100, value))}%"])
        return {"success": True, "volume": value}

    elif action in ["up", "down"]:
        delta = f"+{step}%" if action == "up" else f"-{step}%"
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", delta])
        return {"success": True, "message": f"Volume {action}"}

    elif action == "mute":
        subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"])
        return {"success": True, "muted": True}

    elif action == "unmute":
        subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"])
        return {"success": True, "muted": False}

    return {"success": False, "error": f"Unknown action: {action}"}


async def screen_capture(params: dict) -> dict:
    output = params.get("output", "screenshot.png")
    region = params.get("region")
    format = params.get("format", "png")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import mss

        with mss.mss() as sct:
            if region:
                monitor = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
            else:
                monitor = sct.monitors[0]

            screenshot = sct.grab(monitor)

            import cv2
            import numpy as np
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            if format == "jpg":
                cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            else:
                cv2.imwrite(str(output_path), img)

        return {
            "success": True,
            "path": str(output_path),
            "size": output_path.stat().st_size,
        }

    except ImportError:
        return {"success": False, "error": "mss not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def power_control(params: dict) -> dict:
    action = params.get("action", "")
    delay = params.get("delay", 0)

    if delay > 0:
        await asyncio.sleep(delay)

    try:
        if action == "sleep":
            if IS_MACOS:
                subprocess.run(["pmset", "sleepnow"])
            elif IS_WINDOWS:
                subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
            elif IS_LINUX:
                subprocess.run(["systemctl", "suspend"])
            return {"success": True, "action": "sleep"}

        elif action == "hibernate":
            if IS_MACOS:
                subprocess.run(["pmset", "hibernatemode", "25"])
                subprocess.run(["pmset", "sleepnow"])
            elif IS_WINDOWS:
                subprocess.run(["shutdown", "/h"])
            elif IS_LINUX:
                subprocess.run(["systemctl", "hibernate"])
            return {"success": True, "action": "hibernate"}

        elif action == "shutdown":
            if IS_MACOS:
                subprocess.run(["osascript", "-e", 'tell app "System Events" to shut down'])
            elif IS_WINDOWS:
                subprocess.run(["shutdown", "/s", "/t", "0"])
            elif IS_LINUX:
                subprocess.run(["systemctl", "poweroff"])
            return {"success": True, "action": "shutdown"}

        elif action == "restart":
            if IS_MACOS:
                subprocess.run(["osascript", "-e", 'tell app "System Events" to restart'])
            elif IS_WINDOWS:
                subprocess.run(["shutdown", "/r", "/t", "0"])
            elif IS_LINUX:
                subprocess.run(["systemctl", "reboot"])
            return {"success": True, "action": "restart"}

        elif action == "lock":
            if IS_MACOS:
                subprocess.run(["pmset", "displaysleepnow"])
            elif IS_WINDOWS:
                subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
            elif IS_LINUX:
                subprocess.run(["loginctl", "lock-session"])
            return {"success": True, "action": "lock"}

        return {"success": False, "error": f"Unknown action: {action}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def service_control(params: dict) -> dict:
    service = params.get("service", "")
    action = params.get("action", "status")

    if not service:
        return {"success": False, "error": "Service name required"}

    if IS_WINDOWS:
        return {"success": False, "error": "Use Windows Services manager"}

    try:
        actions = {
            "start": ["systemctl", "start", service],
            "stop": ["systemctl", "stop", service],
            "restart": ["systemctl", "restart", service],
            "status": ["systemctl", "status", service],
            "enable": ["systemctl", "enable", service],
            "disable": ["systemctl", "disable", service],
        }

        if action not in actions:
            return {"success": False, "error": f"Unknown action: {action}"}

        result = subprocess.run(actions[action], capture_output=True, text=True, timeout=30)

        return {
            "success": result.returncode == 0,
            "service": service,
            "action": action,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def network_info(params: dict) -> dict:
    detailed = params.get("detailed", True)

    try:
        info = {}

        interfaces = {}
        for name, addrs in psutil.net_if_addrs().items():
            interfaces[name] = [
                {"family": str(addr.family), "address": addr.address, "netmask": addr.netmask, "broadcast": addr.broadcast}
                for addr in addrs
            ]
        info["interfaces"] = interfaces

        net_io = psutil.net_io_counters()
        info["io"] = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "errin": net_io.errin,
            "errout": net_io.errout,
            "dropin": net_io.dropin,
            "dropout": net_io.dropout,
        }

        if detailed:
            per_nic = psutil.net_io_counters(pernic=True)
            info["per_interface"] = {
                name: stat._asdict() for name, stat in per_nic.items()
            }

            connections = psutil.net_connections(kind="inet")
            info["connections"] = [
                {
                    "fd": c.fd,
                    "family": str(c.family),
                    "type": str(c.type),
                    "laddr": f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else None,
                    "raddr": f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else None,
                    "status": c.status,
                    "pid": c.pid,
                }
                for c in connections[:50]
            ]

        return {"success": True, "info": info}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def disk_usage(params: dict) -> dict:
    path = params.get("path", "/")

    try:
        usage = psutil.disk_usage(path)
        return {
            "success": True,
            "path": path,
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
            "percent": usage.percent,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def register_system_tools(registry: ToolRegistry) -> None:
    tools = [
        ("system_info", system_info, "Get comprehensive system information", [
            ToolParameter("detailed", "boolean", "Include detailed metrics", default=True),
        ], {"requires_approval": False, "sandboxed": False}),

        ("process_list", process_list, "List running processes", [
            ToolParameter("sort_by", "string", "Sort by: cpu, memory, name, pid", default="cpu"),
            ToolParameter("limit", "integer", "Maximum results", default=50),
            ToolParameter("filter", "string", "Filter by name substring"),
        ], {"requires_approval": False, "sandboxed": False}),

        ("process_kill", process_kill, "Kill process by PID", [
            ToolParameter("pid", "integer", "Process ID", required=True),
            ToolParameter("force", "boolean", "Force kill (SIGKILL)", default=False),
        ], {"requires_approval": True, "sandboxed": False}),

        ("app_open", app_open, "Open application", [
            ToolParameter("app", "string", "Application name or path", required=True),
            ToolParameter("args", "array", "Command line arguments"),
        ], {"requires_approval": False, "sandboxed": False}),

        ("app_close", app_close, "Close application by name", [
            ToolParameter("app", "string", "Application name", required=True),
        ], {"requires_approval": False, "sandboxed": False}),

        ("volume_control", volume_control, "Control system volume", [
            ToolParameter("action", "string", "Action: get, set, up, down, mute, unmute, toggle_mute", required=True),
            ToolParameter("value", "integer", "Volume level 0-100", default=50),
            ToolParameter("step", "integer", "Step for up/down", default=10),
        ], {"requires_approval": False, "sandboxed": False}),

        ("screen_capture", screen_capture, "Capture screenshot", [
            ToolParameter("output", "string", "Output path", default="screenshot.png"),
            ToolParameter("region", "array", "Region: [x, y, width, height]"),
            ToolParameter("format", "string", "Image format: png, jpg", default="png"),
        ], {"requires_approval": False, "sandboxed": False}),

        ("power_control", power_control, "Power management", [
            ToolParameter("action", "string", "Action: sleep, hibernate, shutdown, restart, lock", required=True),
            ToolParameter("delay", "integer", "Delay in seconds", default=0),
        ], {"requires_approval": True, "sandboxed": False}),

        ("service_control", service_control, "System service control", [
            ToolParameter("service", "string", "Service name", required=True),
            ToolParameter("action", "string", "Action: start, stop, restart, status, enable, disable", default="status"),
        ], {"requires_approval": True, "sandboxed": False}),

        ("network_info", network_info, "Get network information", [
            ToolParameter("detailed", "boolean", "Include detailed stats", default=True),
        ], {"requires_approval": False, "sandboxed": False}),

        ("disk_usage", disk_usage, "Get disk usage", [
            ToolParameter("path", "string", "Path to check", default="/"),
        ], {"requires_approval": False, "sandboxed": False}),
    ]

    for name, func, desc, params, extra in tools:
        metadata = ToolMetadata(
            name=name,
            description=desc,
            category="system",
            parameters=params,
            returns={"type": "object"},
            sandboxed=extra.get("sandboxed", False),
            requires_approval=extra.get("requires_approval", False),
            tags=["system", "os", "monitoring"],
        )
        registry.register(name, func, metadata)

    logger.info("Registered system tools")


__all__ = [
    "register_system_tools",
]