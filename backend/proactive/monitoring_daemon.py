"""
JARVIS OMEGA - Monitoring Daemon
Continuous system/environment observation with health assessment.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import platform
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import psutil

from config import get_settings, settings

logger = logging.getLogger("vega.proactive.monitoring")


# =============================================================================
# DATA MODELS
# =============================================================================

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SystemMetrics:
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    cpu_per_core: list[float] = field(default_factory=list)
    cpu_freq_mhz: float = 0.0
    load_avg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    memory_total: int = 0
    memory_available: int = 0
    memory_used: int = 0
    memory_percent: float = 0.0
    swap_percent: float = 0.0
    disk_total: int = 0
    disk_used: int = 0
    disk_free: int = 0
    disk_percent: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    net_bytes_sent: int = 0
    net_bytes_recv: int = 0
    net_packets_sent: int = 0
    net_packets_recv: int = 0
    net_errors: int = 0
    net_drop: int = 0
    interfaces: dict[str, dict] = field(default_factory=dict)
    battery_percent: float | None = None
    battery_plugged: bool | None = None
    battery_time_left: int | None = None
    process_count: int = 0
    top_processes: list[dict] = field(default_factory=list)
    gpu_percent: float | None = None
    gpu_memory_percent: float | None = None
    gpu_temp_c: float | None = None
    custom_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "cpu_percent": self.cpu_percent,
            "cpu_per_core": self.cpu_per_core,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "load_avg": self.load_avg,
            "memory_total": self.memory_total,
            "memory_available": self.memory_available,
            "memory_used": self.memory_used,
            "memory_percent": self.memory_percent,
            "swap_percent": self.swap_percent,
            "disk_total": self.disk_total,
            "disk_used": self.disk_used,
            "disk_free": self.disk_free,
            "disk_percent": self.disk_percent,
            "disk_io_read_mb": self.disk_io_read_mb,
            "disk_io_write_mb": self.disk_io_write_mb,
            "net_bytes_sent": self.net_bytes_sent,
            "net_bytes_recv": self.net_bytes_recv,
            "net_packets_sent": self.net_packets_sent,
            "net_packets_recv": self.net_packets_recv,
            "net_errors": self.net_errors,
            "net_drop": self.net_drop,
            "interfaces": self.interfaces,
            "battery_percent": self.battery_percent,
            "battery_plugged": self.battery_plugged,
            "battery_time_left": self.battery_time_left,
            "process_count": self.process_count,
            "top_processes": self.top_processes,
            "gpu_percent": self.gpu_percent,
            "gpu_memory_percent": self.gpu_memory_percent,
            "gpu_temp_c": self.gpu_temp_c,
            "custom_metrics": self.custom_metrics,
        }

    def get_health_status(self) -> "HealthStatus":
        critical = 0
        warning = 0

        if self.cpu_percent > 95: critical += 1
        elif self.cpu_percent > 80: warning += 1
        if self.memory_percent > 95: critical += 1
        elif self.memory_percent > 85: warning += 1
        if self.disk_percent > 95: critical += 1
        elif self.disk_percent > 85: warning += 1
        if self.battery_percent is not None:
            if self.battery_percent < 10 and not self.battery_plugged: critical += 1
            elif self.battery_percent < 20 and not self.battery_plugged: warning += 1
        if self.gpu_temp_c and self.gpu_temp_c > 85: critical += 1
        elif self.gpu_temp_c and self.gpu_temp_c > 75: warning += 1

        if critical > 0: return HealthStatus.CRITICAL
        elif warning > 0: return HealthStatus.WARNING
        return HealthStatus.HEALTHY


@dataclass
class MonitoringConfig:
    system_interval: int = 5
    process_interval: int = 10
    network_interval: int = 5
    disk_interval: int = 30
    battery_interval: int = 60
    gpu_interval: int = 10
    history_size: int = 3600
    cpu_warning: float = 80.0
    cpu_critical: float = 95.0
    memory_warning: float = 85.0
    memory_critical: float = 95.0
    disk_warning: float = 85.0
    disk_critical: float = 95.0
    battery_warning: float = 20.0
    battery_critical: float = 10.0
    gpu_temp_warning: float = 75.0
    gpu_temp_critical: float = 85.0
    top_processes_count: int = 10
    custom_checks: dict[str, Callable] = field(default_factory=dict)
    enable_gpu: bool = True


# =============================================================================
# MONITORING DAEMON
# =============================================================================

class MonitoringDaemon:
    def __init__(
        self,
        event_bus: Any = None,
        websocket_manager: Any = None,
        config: MonitoringConfig | None = None,
    ):
        self.event_bus = event_bus
        self.websocket_manager = websocket_manager
        self.config = config or MonitoringConfig()
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._last_metrics: SystemMetrics | None = None
        self._metrics_history: deque = deque(maxlen=self.config.history_size)
        self._cpu_history: deque = deque(maxlen=100)
        self._memory_history: deque = deque(maxlen=100)
        self._disk_history: deque = deque(maxlen=100)
        self._prev_disk_io = None
        self._prev_net_io = None
        self._on_metrics: Callable[[SystemMetrics], Any] | None = None
        self._on_health_change: Callable[[HealthStatus, HealthStatus], Any] | None = None

    async def start(self) -> bool:
        if self._running: return True
        self._running = True
        self._tasks = [
            asyncio.create_task(self._system_loop()),
            asyncio.create_task(self._process_loop()),
            asyncio.create_task(self._network_loop()),
            asyncio.create_task(self._disk_loop()),
            asyncio.create_task(self._battery_loop()),
        ]
        if self.config.enable_gpu:
            self._tasks.append(asyncio.create_task(self._gpu_loop()))
        for name, check in self.config.custom_checks.items():
            self._tasks.append(asyncio.create_task(self._custom_check_loop(name, check)))
        logger.info("MonitoringDaemon started")
        return True

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("MonitoringDaemon stopped")

    async def _system_loop(self) -> None:
        while self._running:
            try:
                metrics = await self._collect_system_metrics()
                await self._process_metrics(metrics)
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
            await asyncio.sleep(self.config.system_interval)

    async def _collect_system_metrics(self) -> SystemMetrics:
        metrics = SystemMetrics()
        metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        freq = psutil.cpu_freq()
        if freq: metrics.cpu_freq_mhz = freq.current
        try: metrics.load_avg = psutil.getloadavg()
        except: metrics.load_avg = (0.0, 0.0, 0.0)

        mem = psutil.virtual_memory()
        metrics.memory_total = mem.total; metrics.memory_available = mem.available
        metrics.memory_used = mem.used; metrics.memory_percent = mem.percent
        swap = psutil.swap_memory(); metrics.swap_percent = swap.percent

        disk = psutil.disk_usage("/")
        metrics.disk_total = disk.total; metrics.disk_used = disk.used
        metrics.disk_free = disk.free; metrics.disk_percent = disk.percent

        disk_io = psutil.disk_io_counters()
        if disk_io:
            if self._prev_disk_io:
                metrics.disk_io_read_mb = (disk_io.read_bytes - self._prev_disk_io.read_bytes) / (1024*1024)
                metrics.disk_io_write_mb = (disk_io.write_bytes - self._prev_disk_io.write_bytes) / (1024*1024)
            self._prev_disk_io = disk_io

        net_io = psutil.net_io_counters()
        if net_io:
            metrics.net_bytes_sent = net_io.bytes_sent
            metrics.net_bytes_recv = net_io.bytes_recv
            metrics.net_packets_sent = net_io.packets_sent
            metrics.net_packets_recv = net_io.packets_recv
            metrics.net_errors = net_io.errin + net_io.errout
            metrics.net_drop = net_io.dropin + net_io.dropout
            if self._prev_net_io:
                pass
            self._prev_net_io = net_io

        for name, addrs in psutil.net_if_addrs().items():
            stats = psutil.net_if_stats().get(name)
            metrics.interfaces[name] = {
                "addresses": [{"family": str(a.family), "address": a.address} for a in addrs],
                "is_up": stats.isup if stats else False,
                "speed": stats.speed if stats else 0,
            }

        battery = psutil.sensors_battery()
        if battery:
            metrics.battery_percent = battery.percent
            metrics.battery_plugged = battery.power_plugged
            metrics.battery_time_left = battery.secsleft if battery.secsleft != -1 else None

        metrics.process_count = len(psutil.pids())
        self._cpu_history.append(metrics.cpu_percent)
        self._memory_history.append(metrics.memory_percent)
        self._disk_history.append(metrics.disk_percent)
        return metrics

    async def _process_loop(self) -> None:
        while self._running:
            try:
                processes = []
                for proc in psutil.process_iter(['pid','name','username','cpu_percent','memory_percent','status','create_time','cmdline']):
                    try:
                        info = proc.info
                        if info['cpu_percent'] and info['cpu_percent'] > 0.1:
                            processes.append({"pid": info['pid'], "name": info['name'], "username": info['username'],
                                "cpu_percent": info['cpu_percent'], "memory_percent": info['memory_percent'] or 0,
                                "status": info['status'], "started": info['create_time'],
                                "cmdline": info['cmdline'][:3] if info['cmdline'] else []})
                    except (psutil.NoSuchProcess, psutil.AccessDenied): continue
                processes.sort(key=lambda p: p.get('cpu_percent',0), reverse=True)
                if self._last_metrics:
                    self._last_metrics.top_processes = processes[:self.config.top_processes_count]
            except Exception as e: logger.error(f"Process collection error: {e}")
            await asyncio.sleep(self.config.process_interval)

    async def _network_loop(self) -> None:
        while self._running:
            try: pass
            except Exception as e: logger.error(f"Network monitoring error: {e}")
            await asyncio.sleep(self.config.network_interval)

    async def _disk_loop(self) -> None:
        while self._running:
            try:
                for part in psutil.disk_partitions():
                    try: usage = psutil.disk_usage(part.mountpoint)
                    except: pass
            except Exception as e: logger.error(f"Disk monitoring error: {e}")
            await asyncio.sleep(self.config.disk_interval)

    async def _battery_loop(self) -> None:
        while self._running:
            try:
                battery = psutil.sensors_battery()
                if battery and self._last_metrics:
                    if battery.percent < self.config.battery_critical and not battery.power_plugged:
                        if self.event_bus:
                            await self.event_bus.emit("battery.critical", {"percent": battery.percent, "plugged": battery.power_plugged})
            except Exception as e: logger.error(f"Battery monitoring error: {e}")
            await asyncio.sleep(self.config.battery_interval)

    async def _gpu_loop(self) -> None:
        while self._running:
            try:
                gpu_data = await self._get_nvidia_gpu()
                if gpu_data and self._last_metrics:
                    self._last_metrics.gpu_percent = gpu_data.get("utilization")
                    self._last_metrics.gpu_memory_percent = gpu_data.get("memory_percent")
                    self._last_metrics.gpu_temp_c = gpu_data.get("temperature")
                    temp = gpu_data.get("temperature", 0)
                    if temp > self.config.gpu_temp_critical:
                        if self.event_bus: await self.event_bus.emit("gpu.thermal_critical", {"temp_c": temp})
                    elif temp > self.config.gpu_temp_warning:
                        if self.event_bus: await self.event_bus.emit("gpu.thermal_warning", {"temp_c": temp})
            except Exception as e: logger.debug(f"GPU monitoring: {e}")
            await asyncio.sleep(self.config.gpu_interval)

    async def _get_nvidia_gpu(self) -> dict | None:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return {"utilization": util.gpu, "memory_percent": (mem.used/mem.total)*100, "temperature": temp}
        except: return None

    async def _custom_check_loop(self, name: str, check: Callable) -> None:
        while self._running:
            try:
                result = await check() if asyncio.iscoroutinefunction(check) else check()
                if self.event_bus: await self.event_bus.emit(f"custom_check.{name}", {"result": result})
            except Exception as e: logger.error(f"Custom check '{name}' error: {e}")
            await asyncio.sleep(60)

    async def _process_metrics(self, metrics: SystemMetrics) -> None:
        self._metrics_history.append(metrics)
        self._last_metrics = metrics
        self._cpu_history.append(metrics.cpu_percent)
        self._memory_history.append(metrics.memory_percent)
        self._disk_history.append(metrics.disk_percent)

        new_status = metrics.get_health_status()
        old_status = self._last_metrics.get_health_status() if self._last_metrics else HealthStatus.UNKNOWN
        if new_status != old_status:
            if self._on_health_change: await self._on_health_change(old_status, new_status)
            if self.event_bus:
                await self.event_bus.emit("system.health_changed", {"old": old_status.value, "new": new_status.value, "metrics": metrics.to_dict()})

        if self.event_bus: await self.event_bus.emit("system.metrics", metrics.to_dict())
        if self.websocket_manager:
            await self.websocket_manager.broadcast({"type": "system_metrics", "payload": metrics.to_dict()}, room="hud")
        if self._on_metrics: await self._on_metrics(metrics)

    async def _process_loop(self) -> None:
        while self._running:
            try:
                processes = []
                for proc in psutil.process_iter(['pid','name','username','cpu_percent','memory_percent','status','create_time']):
                    try:
                        info = proc.info
                        if info['cpu_percent'] and info['cpu_percent'] > 0.1:
                            processes.append({"pid": info['pid'], "name": info['name'], "username": info['username'],
                                "cpu_percent": info['cpu_percent'], "memory_percent": info['memory_percent'] or 0,
                                "status": info['status'], "started": info['create_time']})
                    except (psutil.NoSuchProcess, psutil.AccessDenied): continue
                processes.sort(key=lambda p: p.get('cpu_percent',0), reverse=True)
                if self._last_metrics:
                    self._last_metrics.top_processes = processes[:self.config.top_processes_count]
            except Exception as e: logger.error(f"Process collection error: {e}")
            await asyncio.sleep(self.config.process_interval)

    async def _network_loop(self) -> None:
        while self._running:
            try: pass
            except Exception as e: logger.error(f"Network monitoring error: {e}")
            await asyncio.sleep(self.config.network_interval)

    async def _disk_loop(self) -> None:
        while self._running:
            try:
                for part in psutil.disk_partitions():
                    try: usage = psutil.disk_usage(part.mountpoint)
                    except: pass
            except Exception as e: logger.error(f"Disk monitoring error: {e}")
            await asyncio.sleep(self.config.disk_interval)

    async def _battery_loop(self) -> None:
        while self._running:
            try:
                battery = psutil.sensors_battery()
                if battery and self._last_metrics:
                    if battery.percent < self.config.battery_critical and not battery.power_plugged:
                        if self.event_bus: await self.event_bus.emit("battery.critical", {"percent": battery.percent, "plugged": battery.power_plugged})
            except Exception as e: logger.error(f"Battery monitoring error: {e}")
            await asyncio.sleep(self.config.battery_interval)

    async def _get_nvidia_gpu(self) -> dict | None:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return {"utilization": util.gpu, "memory_percent": (mem.used/mem.total)*100, "temperature": temp}
        except: return None

    async def _custom_check_loop(self, name: str, check: Callable) -> None:
        while self._running:
            try:
                result = await check() if asyncio.iscoroutinefunction(check) else check()
                if self.event_bus: await self.event_bus.emit(f"custom_check.{name}", {"result": result})
            except Exception as e: logger.error(f"Custom check '{name}' error: {e}")
            await asyncio.sleep(60)

    async def _process_metrics(self, metrics: SystemMetrics) -> None:
        self._metrics_history.append(metrics)
        self._last_metrics = metrics
        self._cpu_history.append(metrics.cpu_percent)
        self._memory_history.append(metrics.memory_percent)
        self._disk_history.append(metrics.disk_percent)

        new_status = metrics.get_health_status()
        old_status = self._last_metrics.get_health_status() if self._last_metrics else HealthStatus.UNKNOWN
        if new_status != old_status:
            if self._on_health_change: await self._on_health_change(old_status, new_status)
            if self.event_bus: await self.event_bus.emit("system.health_changed", {"old": old_status.value, "new": new_status.value, "metrics": metrics.to_dict()})

        if self.event_bus: await self.event_bus.emit("system.metrics", metrics.to_dict())
        if self.websocket_manager:
            await self.websocket_manager.broadcast({"type": "system_metrics", "payload": metrics.to_dict()}, room="hud")
        if self._on_metrics: await self._on_metrics(metrics)

    def get_current_metrics(self) -> SystemMetrics | None: return self._last_metrics
    def get_history(self, seconds: int = 60) -> list[SystemMetrics]:
        cutoff = time.time() - seconds
        return [m for m in self._metrics_history if m.timestamp >= cutoff]
    def get_baselines(self) -> dict[str, float]:
        if not self._cpu_history: return {}
        import numpy as np
        return {"cpu_mean": float(np.mean(self._cpu_history)), "cpu_std": float(np.std(self._cpu_history)),
            "memory_mean": float(np.mean(self._memory_history)), "memory_std": float(np.std(self._memory_history)),
            "disk_mean": float(np.mean(self._disk_history)), "disk_std": float(np.std(self._disk_history))}
    def get_current_health(self) -> HealthStatus:
        return self._last_metrics.get_health_status() if self._last_metrics else HealthStatus.UNKNOWN

    def on_metrics(self, callback: Callable[[SystemMetrics], Any]) -> None: self._on_metrics = callback
    def on_health_change(self, callback: Callable[[HealthStatus, HealthStatus], Any]) -> None: self._on_health_change = callback

    async def health_check(self) -> dict:
        return {"healthy": self._running, "collecting": len(self._tasks) > 0,
            "history_size": len(self._metrics_history),
            "current_health": self.get_current_health().value,
            "last_metrics_age": time.time() - self._last_metrics.timestamp if self._last_metrics else None}

    async def shutdown(self) -> None:
        self._running = False
        for task in self._tasks: task.cancel()
        if self._tasks: await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("MonitoringDaemon stopped")


async def create_monitoring_daemon(event_bus: Any = None, websocket_manager: Any = None, config: MonitoringConfig | None = None) -> MonitoringDaemon:
    daemon = MonitoringDaemon(event_bus=event_bus, websocket_manager=websocket_manager, config=config)
    await daemon.start()
    return daemon


__all__ = ["MonitoringDaemon", "SystemMetrics", "HealthStatus", "MonitoringConfig", "create_monitoring_daemon"]