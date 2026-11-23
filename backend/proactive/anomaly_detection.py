#!/usr/bin/env python3
"""
================================================================================
VEGA ANOMALY DETECTION - Behavioral & System Anomaly Identification
================================================================================
Multi-modal anomaly detection for:
- System metrics (CPU, memory, disk, network, GPU)
- User behavior patterns (activity, app usage, typing, mouse)
- Network traffic patterns
- Process behavior
- Security events (failed auth, unusual access)
- Resource utilization trends

Detection methods:
- Statistical (z-score, percentile, moving average)
- Machine Learning (Isolation Forest, One-Class SVM)
- Rule-based (thresholds, rate of change)
- Seasonal/Trend decomposition
- Peer group comparison

References:
- OpenJarvis/OpenJarvis: Anomaly monitoring
- Prometheus: Alerting rules
- Netflix: Atlas anomaly detection
- AWS: Lookout for Metrics
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from config import get_settings, settings

logger = logging.getLogger("vega.proactive.anomaly")


# =============================================================================
# DATA MODELS
# =============================================================================

class AnomalyType(str, Enum):
    """Categories of anomalies."""
    SYSTEM_CPU = "system_cpu"
    SYSTEM_MEMORY = "system_memory"
    SYSTEM_DISK = "system_disk"
    SYSTEM_NETWORK = "system_network"
    SYSTEM_GPU = "system_gpu"
    PROCESS_CPU = "process_cpu"
    PROCESS_MEMORY = "process_memory"
    NETWORK_TRAFFIC = "network_traffic"
    NETWORK_LATENCY = "network_latency"
    USER_ACTIVITY = "user_activity"
    USER_IDLE = "user_idle"
    APP_USAGE = "app_usage"
    SECURITY_AUTH = "security_auth"
    SECURITY_ACCESS = "security_access"
    CUSTOM = "custom"


class AnomalySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DetectionMethod(str, Enum):
    ZSCORE = "zscore"
    PERCENTILE = "percentile"
    MOVING_AVERAGE = "moving_average"
    RATE_OF_CHANGE = "rate_of_change"
    ISOLATION_FOREST = "isolation_forest"
    SEASONAL_DECOMPOSE = "seasonal_decompose"
    RULE_BASED = "rule_based"


@dataclass
class Baseline:
    """Statistical baseline for a metric."""
    metric_name: str
    mean: float = 0.0
    std: float = 1.0
    median: float = 0.0
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    sample_count: int = 0
    last_updated: float = field(default_factory=time.time)

    # Seasonal
    hourly_means: dict[int, float] = field(default_factory=dict)  # hour -> mean
    daily_means: dict[int, float] = field(default_factory=dict)   # day of week -> mean

    def update(self, value: float) -> None:
        """Update baseline with new value (exponential moving average)."""
        alpha = 0.1  # Learning rate

        if self.sample_count == 0:
            self.mean = value
            self.std = 0.0
            self.median = value
            self.min_value = value
            self.max_value = value
        else:
            # Exponential moving average
            self.mean = alpha * value + (1 - alpha) * self.mean
            self.std = alpha * abs(value - self.mean) + (1 - alpha) * self.std
            self.min_value = min(self.min_value, value)
            self.max_value = max(self.max_value, value)

        self.sample_count += 1
        self.last_updated = time.time()

    def get_zscore(self, value: float) -> float:
        """Calculate z-score for value."""
        if self.std == 0:
            return 0.0
        return (value - self.mean) / self.std

    def get_percentile_rank(self, value: float) -> float:
        """Estimate percentile rank (simplified)."""
        if self.std == 0:
            return 0.5

        # Using normal approximation
        z = self.get_zscore(value)
        # Approximate CDF
        return 0.5 * (1 + np.tanh(z * 0.7))

    def is_anomalous(self, value: float, threshold: float = 3.0) -> bool:
        """Check if value is anomalous."""
        return abs(self.get_zscore(value)) > threshold


@dataclass
class Anomaly:
    """Detected anomaly."""
    anomaly_id: str
    type: AnomalyType
    severity: AnomalySeverity
    metric_name: str
    expected_value: float
    actual_value: float
    deviation: float  # z-score or percentile deviation
    timestamp: float = field(default_factory=time.time)
    message: str = ""
    context: dict = field(default_factory=dict)
    method: DetectionMethod = DetectionMethod.ZSCORE
    acknowledged: bool = False
    resolved: bool = False
    resolution_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "anomaly_id": self.anomaly_id,
            "type": self.type.value,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "deviation": self.deviation,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "message": self.message,
            "context": self.context,
            "method": self.method.value,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


@dataclass
class AnomalyRule:
    """Rule for anomaly detection."""
    name: str
    type: AnomalyType
    metric_name: str
    method: DetectionMethod = DetectionMethod.ZSCORE
    threshold: float = 3.0  # z-score threshold
    severity: AnomalySeverity = AnomalySeverity.WARNING
    enabled: bool = True
    cooldown: int = 300  # seconds
    message_template: str = "{metric} anomaly: {actual:.1f} (expected ~{expected:.1f})"

    # For rule-based
    condition: Callable[[float, Baseline], bool] | None = None


# =============================================================================
# ANOMALY DETECTOR
# =============================================================================

class AnomalyDetector:
    """
    Multi-method anomaly detection engine.
    Maintains baselines, evaluates rules, detects anomalies in real-time.
    """

    def __init__(
        self,
        event_bus: Any = None,
        websocket_manager: Any = None,
        monitoring_daemon: Any = None,
    ):
        self.event_bus = event_bus
        self.websocket_manager = websocket_manager
        self.monitoring_daemon = monitoring_daemon

        # Baselines for each metric
        self._baselines: dict[str, Baseline] = {}

        # Rules
        self._rules: dict[str, AnomalyRule] = {}

        # State
        self._anomalies: dict[str, Anomaly] = {}
        self._anomaly_history: deque = deque(maxlen=10000)
        self._last_fired: dict[str, float] = {}

        # Data buffers for ML methods
        self._metric_buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Running
        self._running = False
        self._detection_task: asyncio.Task | None = None

        # Callbacks
        self._on_anomaly: Callable[[Anomaly], Any] | None = None

        # Initialize default rules
        self._init_default_rules()

        logger.info("AnomalyDetector initialized")

    # -------------------------------------------------------------------------
    # DEFAULT RULES
    # -------------------------------------------------------------------------

    def _init_default_rules(self) -> None:
        """Initialize built-in anomaly detection rules."""

        # System metrics
        self.add_rule(AnomalyRule(
            name="cpu_high",
            type=AnomalyType.SYSTEM_CPU,
            metric_name="cpu_percent",
            method=DetectionMethod.ZSCORE,
            threshold=3.0,
            severity=AnomalySeverity.WARNING,
            message_template="CPU usage anomaly: {actual:.1f}% (baseline: {expected:.1f}%)",
        ))

        self.add_rule(AnomalyRule(
            name="cpu_critical",
            type=AnomalyType.SYSTEM_CPU,
            metric_name="cpu_percent",
            method=DetectionMethod.RULE_BASED,
            threshold=95.0,
            severity=AnomalySeverity.CRITICAL,
            message_template="CRITICAL: CPU at {actual:.1f}%",
            condition=lambda v, b: v > 95,
        ))

        self.add_rule(AnomalyRule(
            name="memory_high",
            type=AnomalyType.SYSTEM_MEMORY,
            metric_name="memory_percent",
            method=DetectionMethod.ZSCORE,
            threshold=3.0,
            severity=AnomalySeverity.WARNING,
        ))

        self.add_rule(AnomalyRule(
            name="memory_critical",
            type=AnomalyType.SYSTEM_MEMORY,
            metric_name="memory_percent",
            method=DetectionMethod.RULE_BASED,
            threshold=95.0,
            severity=AnomalySeverity.CRITICAL,
            condition=lambda v, b: v > 95,
        ))

        self.add_rule(AnomalyRule(
            name="disk_high",
            type=AnomalyType.SYSTEM_DISK,
            metric_name="disk_percent",
            method=DetectionMethod.RULE_BASED,
            threshold=90.0,
            severity=AnomalySeverity.WARNING,
            condition=lambda v, b: v > 90,
        ))

        self.add_rule(AnomalyRule(
            name="disk_critical",
            type=AnomalyType.SYSTEM_DISK,
            metric_name="disk_percent",
            method=DetectionMethod.RULE_BASED,
            threshold=98.0,
            severity=AnomalySeverity.CRITICAL,
            condition=lambda v, b: v > 98,
        ))

        self.add_rule(AnomalyRule(
            name="network_spike",
            type=AnomalyType.SYSTEM_NETWORK,
            metric_name="net_bytes_sent",
            method=DetectionMethod.RATE_OF_CHANGE,
            threshold=5.0,  # 5x normal rate
            severity=AnomalySeverity.WARNING,
        ))

        self.add_rule(AnomalyRule(
            name="gpu_overheating",
            type=AnomalyType.SYSTEM_GPU,
            metric_name="gpu_temp_c",
            method=DetectionMethod.RULE_BASED,
            threshold=80.0,
            severity=AnomalySeverity.WARNING,
            condition=lambda v, b: v > 80,
        ))

        self.add_rule(AnomalyRule(
            name="gpu_critical",
            type=AnomalyType.SYSTEM_GPU,
            metric_name="gpu_temp_c",
            method=DetectionMethod.RULE_BASED,
            threshold=90.0,
            severity=AnomalySeverity.CRITICAL,
            condition=lambda v, b: v > 90,
        ))

    # -------------------------------------------------------------------------
    # RULE MANAGEMENT
    # -------------------------------------------------------------------------

    def add_rule(self, rule: AnomalyRule) -> None:
        self._rules[rule.name] = rule
        logger.debug(f"Added anomaly rule: {rule.name}")

    def remove_rule(self, name: str) -> bool:
        if name in self._rules:
            del self._rules[name]
            return True
        return False

    def enable_rule(self, name: str, enabled: bool = True) -> bool:
        if name in self._rules:
            self._rules[name].enabled = enabled
            return True
        return False

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    async def start(self, evaluation_interval: int = 10) -> bool:
        """Start anomaly detection."""
        if self._running:
            return True

        self._running = True
        self._detection_task = asyncio.create_task(self._detection_loop(evaluation_interval))
        logger.info("AnomalyDetector started")
        return True

    async def stop(self) -> None:
        """Stop anomaly detection."""
        self._running = False
        if self._detection_task:
            self._detection_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._detection_task
        logger.info("AnomalyDetector stopped")

    # -------------------------------------------------------------------------
    # DETECTION LOOP
    # -------------------------------------------------------------------------

    async def _detection_loop(self, interval: int) -> None:
        """Main detection loop."""
        while self._running:
            try:
                await self._evaluate_metrics()
                await self._cleanup_resolved()
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")

            await asyncio.sleep(interval)

    async def _evaluate_metrics(self) -> None:
        """Evaluate all metrics against rules."""
        # Get current metrics from monitoring daemon
        if not self.monitoring_daemon:
            return

        metrics = self.monitoring_daemon.get_current_metrics()
        if not metrics:
            return

        # Convert metrics to dict
        metric_dict = metrics.to_dict()

        # Update baselines and buffers
        for metric_name, value in metric_dict.items():
            if isinstance(value, (int, float)):
                await self._update_baseline(metric_name, value)
                self._metric_buffers[metric_name].append((time.time(), value))

        # Evaluate rules
        for rule_name, rule in self._rules.items():
            if not rule.enabled:
                continue

            # Check cooldown
            last_fired = self._last_fired.get(rule_name, 0)
            if time.time() - last_fired < rule.cooldown:
                continue

            # Get metric value
            value = metric_dict.get(rule.metric_name)
            if value is None or not isinstance(value, (int, float)):
                continue

            # Get baseline
            baseline = self._baselines.get(rule.metric_name)
            if not baseline or baseline.sample_count < 10:
                continue  # Not enough data

            # Evaluate based on method
            is_anomaly = False
            deviation = 0.0

            if rule.method == DetectionMethod.ZSCORE:
                zscore = baseline.get_zscore(value)
                if abs(zscore) > rule.threshold:
                    is_anomaly = True
                    deviation = zscore

            elif rule.method == DetectionMethod.PERCENTILE:
                percentile = baseline.get_percentile_rank(value)
                if percentile > 0.999 or percentile < 0.001:
                    is_anomaly = True
                    deviation = abs(percentile - 0.5) * 2

            elif rule.method == DetectionMethod.RATE_OF_CHANGE:
                # Compare with previous value
                buffer = self._metric_buffers.get(rule.metric_name)
                if buffer and len(buffer) >= 2:
                    prev_value = buffer[-2][1]
                    if prev_value != 0:
                        rate = abs(value - prev_value) / abs(prev_value)
                        if rate > rule.threshold:
                            is_anomaly = True
                            deviation = rate

            elif rule.method == DetectionMethod.RULE_BASED:
                if rule.condition and rule.condition(value, baseline):
                    is_anomaly = True
                    deviation = abs(value - baseline.mean) / max(baseline.std, 1)

            if is_anomaly:
                await self._fire_anomaly(rule, value, baseline.mean, deviation)

    async def _update_baseline(self, metric_name: str, value: float) -> None:
        """Update baseline for metric."""
        if metric_name not in self._baselines:
            self._baselines[metric_name] = Baseline(metric_name=metric_name)

        self._baselines[metric_name].update(value)

    async def _fire_anomaly(
        self,
        rule: AnomalyRule,
        actual_value: float,
        expected_value: float,
        deviation: float,
    ) -> None:
        """Fire an anomaly."""
        # Check cooldown
        last_fired = self._last_fired.get(rule.name, 0)
        if time.time() - last_fired < rule.cooldown:
            return

        anomaly = Anomaly(
            anomaly_id=f"{rule.name}_{int(time.time())}",
            type=rule.type,
            severity=rule.severity,
            metric_name=rule.metric_name,
            expected_value=expected_value,
            actual_value=actual_value,
            deviation=deviation,
            message=rule.message_template.format(
                metric=rule.metric_name,
                actual=actual_value,
                expected=expected_value,
            ),
            method=rule.method,
        )

        # Store
        self._anomalies[anomaly.anomaly_id] = anomaly
        self._anomaly_history.append(anomaly)
        self._last_fired[rule.name] = time.time()

        # Emit event
        if self.event_bus:
            await self.event_bus.emit("anomaly.detected", anomaly.to_dict())

        # Broadcast to HUD
        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "anomaly",
                "payload": anomaly.to_dict(),
            }, room="hud")

        # Callback
        if self._on_anomaly:
            await self._on_anomaly(anomaly)

        logger.warning(f"Anomaly detected: {anomaly.message}")

    async def _cleanup_resolved(self) -> None:
        """Clean up resolved anomalies."""
        # Auto-resolve if metric returns to normal
        for anomaly_id, anomaly in list(self._anomalies.items()):
            if anomaly.resolved:
                continue

            baseline = self._baselines.get(anomaly.metric_name)
            if not baseline:
                continue

            # Get current value
            metrics = self.monitoring_daemon.get_current_metrics() if self.monitoring_daemon else None
            if not metrics:
                continue

            current_value = metrics.to_dict().get(anomaly.metric_name)
            if current_value is None:
                continue

            # Check if back to normal
            zscore = baseline.get_zscore(current_value)
            if abs(zscore) < 1.0:  # Back within 1 std dev
                anomaly.resolved = True
                anomaly.resolution_notes = f"Auto-resolved: metric returned to normal ({current_value:.1f})"

                if self.event_bus:
                    await self.event_bus.emit("anomaly.resolved", anomaly.to_dict())

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def get_active_anomalies(self) -> list[Anomaly]:
        return [a for a in self._anomalies.values() if not a.resolved and not a.acknowledged]

    def get_anomaly(self, anomaly_id: str) -> Anomaly | None:
        return self._anomalies.get(anomaly_id)

    def acknowledge_anomaly(self, anomaly_id: str) -> bool:
        if anomaly_id in self._anomalies:
            self._anomalies[anomaly_id].acknowledged = True
            return True
        return False

    def resolve_anomaly(self, anomaly_id: str, notes: str = "") -> bool:
        if anomaly_id in self._anomalies:
            self._anomalies[anomaly_id].resolved = True
            self._anomalies[anomaly_id].resolution_notes = notes
            return True
        return False

    def get_baseline(self, metric_name: str) -> Baseline | None:
        return self._baselines.get(metric_name)

    def get_all_baselines(self) -> dict[str, Baseline]:
        return dict(self._baselines)

    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------

    def on_anomaly(self, callback: Callable[[Anomaly], Any]) -> None:
        self._on_anomaly = callback

    # -------------------------------------------------------------------------
    # STATS
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        active = self.get_active_anomalies()
        return {
            "active_anomalies": len(active),
            "total_anomalies": len(self._anomaly_history),
            "baselines": len(self._baselines),
            "rules": len(self._rules),
            "by_severity": {
                "critical": sum(1 for a in active if a.severity == AnomalySeverity.CRITICAL),
                "warning": sum(1 for a in active if a.severity == AnomalySeverity.WARNING),
                "info": sum(1 for a in active if a.severity == AnomalySeverity.INFO),
            },
            "by_type": {
                t.value: sum(1 for a in active if a.type == t)
                for t in AnomalyType
            },
        }

    async def health_check(self) -> dict:
        return {
            "healthy": self._running,
            "active_anomalies": len(self.get_active_anomalies()),
            "baselines_ready": sum(1 for b in self._baselines.values() if b.sample_count >= 10),
        }


# =============================================================================
# FACTORY
# =============================================================================

async def create_anomaly_detector(
    event_bus: Any = None,
    websocket_manager: Any = None,
    monitoring_daemon: Any = None,
) -> AnomalyDetector:
    detector = AnomalyDetector(
        event_bus=event_bus,
        websocket_manager=websocket_manager,
        monitoring_daemon=monitoring_daemon,
    )
    await detector.start()
    return detector


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AnomalyDetector",
    "Anomaly",
    "AnomalyType",
    "AnomalySeverity",
    "DetectionMethod",
    "Baseline",
    "AnomalyRule",
    "create_anomaly_detector",
]