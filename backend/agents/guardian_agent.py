"""
JARVIS OMEGA - Guardian Agent
Specialist agent for security, authentication, anomaly detection, and monitoring.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import cv2
import numpy as np

from agents.base_agent import BaseAgent, AgentConfig, AgentContext
from tools.tool_registry import ToolRegistry, ToolMetadata, ToolParameter
from config import get_settings, settings

logger = logging.getLogger("jarvis.agents.guardian")


# =============================================================================
# DATA MODELS
# =============================================================================

class SecurityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    INTRUDER = "intruder"
    UNKNOWN_FACE = "unknown_face"
    FAILED_AUTH = "failed_auth"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SYSTEM_ANOMALY = "system_anomaly"
    PERMISSION_DENIED = "permission_denied"
    DATA_ACCESS = "data_access"
    NETWORK_ANOMALY = "network_anomaly"


@dataclass
class FaceIdentity:
    """Known face identity."""
    identity_id: str
    name: str
    embeddings: list[list[float]]  # Multiple embeddings for robustness
    access_level: SecurityLevel = SecurityLevel.MEDIUM
    permissions: list[str] = field(default_factory=list)
    enrolled_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class SecurityAlert:
    """Security alert."""
    alert_id: str
    alert_type: AlertType
    level: SecurityLevel
    title: str
    description: str
    source: str  # camera, system, network, etc.
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    resolved: bool = False
    metadata: dict = field(default_factory=dict)
    # For face alerts
    face_image_path: str | None = None
    confidence: float = 0.0


@dataclass
class AccessRequest:
    """Access control request."""
    request_id: str
    requestor: str  # agent, user, service
    resource: str
    action: str  # read, write, execute, delete, admin
    context: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    approved: bool | None = None
    approved_by: str | None = None


@dataclass
class SystemBaseline:
    """System behavior baseline for anomaly detection."""
    metric: str
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
            self.mean = alpha * value + (1 - alpha) * self.mean
            self.std = alpha * abs(value - self.mean) + (1 - alpha) * self.std
            self.min_value = min(self.min_value, value)
            self.max_value = max(self.max_value, value)

        self.sample_count += 1
        self.last_updated = time.time()

    def get_zscore(self, value: float) -> float:
        if self.std == 0:
            return 0.0
        return (value - self.mean) / self.std

    def get_percentile_rank(self, value: float) -> float:
        if self.std == 0:
            return 0.5
        z = self.get_zscore(value)
        return 0.5 * (1 + np.tanh(z * 0.7))

    def is_anomalous(self, value: float, threshold: float = 3.0) -> bool:
        return abs(self.get_zscore(value)) > threshold


# =============================================================================
# GUARDIAN AGENT
# =============================================================================

class GuardianAgent(BaseAgent):
    """
    Guardian Agent - Security, authentication, monitoring, anomaly detection.
    The "shield" of VEGA.
    """

    def __init__(
        self,
        event_bus: Any = None,
        memory_manager: Any = None,
        tool_registry: Any = None,
        llm_client: Any = None,
        vision_engine: Any = None,
    ):
        agent_config = AgentConfig(
            name="guardian",
            role="guardian",
            description="Face authentication, intrusion detection, access control, anomaly detection, security monitoring",
            model="guardian",
            temperature=0.1,  # Very low for security decisions
            max_tokens=4096,
            system_prompt=self._get_system_prompt(),
            tools=[
                "enroll_face", "verify_face", "list_identities", "revoke_identity",
                "check_access", "grant_access", "revoke_access", "audit_permissions",
                "monitor_system", "detect_anomalies", "get_alerts", "acknowledge_alert",
                "start_sentry", "stop_sentry", "security_audit",
            ],
            can_delegate=False,
            can_handoff=True,
            stream_response=False,  # Security responses should be complete
            timeout=30.0,
        )

        super().__init__(
            config=agent_config,
            event_bus=event_bus,
            memory_manager=memory_manager,
            tool_registry=tool_registry,
            llm_client=llm_client,
        )

        self.vision_engine = vision_engine

        # Face database
        self._face_db_path = Path.home() / ".vega" / "face_db"
        self._face_db_path.mkdir(parents=True, exist_ok=True)
        self._identities: dict[str, FaceIdentity] = {}

        # Alerts
        self._alerts: dict[str, SecurityAlert] = {}
        self._alert_history: deque = deque(maxlen=1000)

        # Access control
        self._access_policies: dict[str, dict] = {}
        self._pending_requests: dict[str, AccessRequest] = {}

        # Anomaly detection
        self._baselines: dict[str, SystemBaseline] = {}
        self._metric_history: dict[str, deque] = {}
        self._sentry_active = False
        self._sentry_task: asyncio.Task | None = None

        # Thresholds
        self._face_threshold = 0.6
        self._liveness_threshold = 0.7
        self._anomaly_sensitivity = 0.8

        # Register tools
        if self.tool_registry:
            self._register_tools()

    def _get_system_prompt(self) -> str:
        return """You are the Guardian Agent - VEGA's security specialist. You protect the system, authenticate users, detect threats, and enforce policies.

Responsibilities:
1. FACE AUTHENTICATION: Enroll, verify, manage identities with liveness detection
2. INTRUSION DETECTION: Sentry mode - alert on unknown faces, suspicious activity
3. ACCESS CONTROL: Grant/revoke permissions, audit access, enforce least privilege
4. ANOMALY DETECTION: Monitor system metrics, detect behavioral anomalies
5. SECURITY AUDITING: Log all security events, generate compliance reports
6. INCIDENT RESPONSE: Alert, contain, investigate, remediate

Guidelines:
- DEFAULT DENY: Require explicit authorization for sensitive operations
- MINIMUM PRIVILEGE: Grant only necessary permissions
- AUDIT EVERYTHING: Log all auth decisions, access attempts, anomalies
- REAL-TIME: Sentry mode runs continuously, alerts instantly
- LIVENESS: Require liveness checks for face auth (prevent photos/videos)
- ESCALATION: Critical alerts require human acknowledgment

Never compromise security for convenience. When in doubt, deny and alert.""".

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    async def _agent_initialize(self) -> None:
        """Initialize guardian components."""
        await self._load_identities()
        await self._load_access_policies()
        await self._initialize_baselines()

        # Register with vision engine for face detection callbacks
        if self.vision_engine:
            self.vision_engine.pipeline.on_detection(self._on_face_detected)

        logger.info("GuardianAgent initialized")

    async def _agent_shutdown(self) -> None:
        """Shutdown guardian."""
        await self.stop_sentry()

        # Save identities
        await self._save_identities()

    # -------------------------------------------------------------------------
    # TOOL REGISTRATION
    # -------------------------------------------------------------------------

    def _register_tools(self) -> None:
        tools = [
            # Face management
            ("enroll_face", self._tool_enroll_face, "Enroll a new face for authentication", [
                ToolParameter("name", "string", "Person's name", required=True),
            ]),
            ("verify_face", self._tool_verify_face, "Verify face against enrolled identities", []),
            ("list_identities", self._tool_list_identities, "List all enrolled identities", []),
            ("revoke_identity", self._tool_revoke_identity, "Revoke a face identity", [
                ToolParameter("identity_id", "string", "Identity ID to revoke", required=True),
            ]),
            # Access control
            ("check_access", self._tool_check_access, "Check if requestor has access to resource", [
                ToolParameter("resource", "string", "Resource to access", required=True),
                ToolParameter("action", "string", "Action: read, write, execute, delete, admin", required=True),
                ToolParameter("requestor", "string", "Requestor identity"),
            ]),
            ("grant_access", self._tool_grant_access, "Grant permission to identity", [
                ToolParameter("identity_id", "string", "Identity ID", required=True),
                ToolParameter("permission", "string", "Permission to grant", required=True),
            ]),
            ("revoke_access", self._tool_revoke_access, "Revoke permission from identity", [
                ToolParameter("identity_id", "string", "Identity ID", required=True),
                ToolParameter("permission", "string", "Permission to revoke", required=True),
            ]),
            ("audit_permissions", self._tool_audit_permissions, "Audit all access permissions", []),
            # Monitoring
            ("monitor_system", self._tool_monitor_system, "Get current system metrics", []),
            ("detect_anomalies", self._tool_detect_anomalies, "Run anomaly detection check", []),
            ("get_alerts", self._tool_get_alerts, "Get security alerts", [
                ToolParameter("limit", "integer", "Max alerts", default=50),
            ]),
            ("acknowledge_alert", self._tool_acknowledge_alert, "Acknowledge an alert", [
                ToolParameter("alert_id", "string", "Alert ID", required=True),
            ]),
            # Sentry
            ("start_sentry", self._tool_start_sentry, "Start sentry mode (continuous monitoring)", []),
            ("stop_sentry", self._tool_stop_sentry, "Stop sentry mode", []),
            # Audit
            ("security_audit", self._tool_security_audit, "Run full security audit", []),
        ]

        for name, func, desc, params in tools:
            metadata = ToolMetadata(
                name=name,
                description=desc,
                category="security",
                parameters=params,
                returns={"type": "object"},
                sandboxed=True,
                requires_approval=name in ["revoke_identity", "grant_access", "revoke_access", "start_sentry", "stop_sentry"],
                tags=["security", "guardian"],
            )
            self.tool_registry.register(name, func, metadata)

    # -------------------------------------------------------------------------
    # FACE AUTHENTICATION
    # -------------------------------------------------------------------------

    async def _load_identities(self) -> None:
        """Load face identities from disk."""
        try:
            for file in self._face_db_path.glob("*.json"):
                with open(file) as f:
                    data = json.load(f)
                identity = FaceIdentity(**data)
                self._identities[identity.identity_id] = identity
            logger.info(f"Loaded {len(self._identities)} face identities")
        except Exception as e:
            logger.error(f"Failed to load identities: {e}")

    async def _save_identities(self) -> None:
        """Save face identities to disk."""
        try:
            for identity in self._identities.values():
                file_path = self._face_db_path / f"{identity.identity_id}.json"
                with open(file_path, "w") as f:
                    json.dump(identity.__dict__, f, default=str)
        except Exception as e:
            logger.error(f"Failed to save identities: {e}")

    async def _handle_enroll(self, task: str, context: AgentContext) -> dict:
        """Handle face enrollment."""
        name = self._extract_name(task) or "Unknown"

        if not self.vision_engine:
            return {"success": False, "error": "Vision engine not available"}

        face_data = await self._capture_face_for_enrollment()
        if not face_data:
            return {"success": False, "error": "No face detected. Please face the camera."}

        identity_id = str(uuid.uuid4())
        identity = FaceIdentity(
            identity_id=identity_id,
            name=name,
            embeddings=[face_data["embedding"]],
            access_level=SecurityLevel.MEDIUM,
            permissions=["user"],
        )

        self._identities[identity_id] = identity
        await self._save_identities()

        if face_data.get("image") is not None:
            img_path = self._face_db_path / f"{identity_id}.jpg"
            cv2.imwrite(str(img_path), face_data["image"])

        if self.event_bus:
            await self.event_bus.emit("security.face_enrolled", {
                "identity_id": identity_id,
                "name": name,
            })

        return {
            "success": True,
            "identity_id": identity_id,
            "name": name,
            "message": f"Enrolled {name} with ID {identity_id[:8]}",
        }

    async def _handle_verify(self, task: str, context: AgentContext) -> dict:
        """Handle face verification."""
        if not self.vision_engine:
            return {"success": False, "error": "Vision engine not available"}

        face_data = await self._capture_face_for_verification()
        if not face_data:
            return {"success": False, "error": "No face detected"}

        best_match = None
        best_score = 0.0

        for identity in self._identities.values():
            for enrolled_emb in identity.embeddings:
                score = self._compare_embeddings(face_data["embedding"], enrolled_emb)
                if score > best_score:
                    best_score = score
                    best_match = identity

        verified = best_score >= self._face_threshold and best_match is not None

        liveness = await self._check_liveness(face_data)

        result = {
            "verified": verified,
            "identity": best_match.name if best_match else None,
            "identity_id": best_match.identity_id if best_match else None,
            "confidence": best_score,
            "liveness_passed": liveness,
            "access_level": best_match.access_level.value if best_match else None,
            "permissions": best_match.permissions if best_match else [],
        }

        await self._log_auth_attempt(verified, best_match, best_score, liveness)

        if not verified and best_score > 0.4:
            await self._create_alert(AlertType.UNKNOWN_FACE, SecurityLevel.MEDIUM,
                "Unrecognized face detected",
                f"Face similarity: {best_score:.2f} (threshold: {self._face_threshold})",
                confidence=best_score)

        return result

    def _compare_embeddings(self, emb1: list[float], emb2: list[float]) -> float:
        """Compare face embeddings (cosine similarity)."""
        try:
            a = np.array(emb1)
            b = np.array(emb2)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except Exception:
            return 0.0

    async def _check_liveness(self, face_data: dict) -> bool:
        """Check if face is live (not photo/video)."""
        return face_data.get("liveness_score", 1.0) > self._liveness_threshold

    async def _capture_face_for_enrollment(self) -> dict | None:
        """Capture high-quality face for enrollment."""
        return None  # Placeholder

    async def _capture_face_for_verification(self) -> dict | None:
        """Capture face for verification."""
        return None  # Placeholder

    async def _log_auth_attempt(
        self,
        verified: bool,
        identity: FaceIdentity | None,
        confidence: float,
        liveness: bool,
    ) -> None:
        """Log authentication attempt."""
        if self.memory_manager:
            await self.memory_manager.ingest(
                content=f"Auth attempt: {'SUCCESS' if verified else 'FAILED'} - "
                        f"User: {identity.name if identity else 'Unknown'} - "
                        f"Confidence: {confidence:.2f} - Liveness: {liveness}",
                metadata={
                    "type": "auth_log",
                    "verified": verified,
                    "identity_id": identity.identity_id if identity else None,
                    "confidence": confidence,
                    "liveness": liveness,
                },
                tier="episodic",
            )

    # -------------------------------------------------------------------------
    # ACCESS CONTROL
    # -------------------------------------------------------------------------

    async def _load_access_policies(self) -> None:
        self._access_policies = {
            "system_control": {"level": SecurityLevel.HIGH, "permissions": ["admin"]},
            "financial": {"level": SecurityLevel.CRITICAL, "permissions": ["finance_admin"]},
            "security_config": {"level": SecurityLevel.CRITICAL, "permissions": ["security_admin"]},
            "user_data": {"level": SecurityLevel.HIGH, "permissions": ["data_admin"]},
            "creative": {"level": SecurityLevel.LOW, "permissions": ["user"]},
            "research": {"level": SecurityLevel.LOW, "permissions": ["user"]},
            "executor": {"level": SecurityLevel.MEDIUM, "permissions": ["executor"]},
        }

    async def _handle_access(self, task: str, context: AgentContext) -> dict:
        request = AccessRequest(
            request_id=str(uuid.uuid4()),
            requestor=context.user_id or "unknown",
            resource=self._extract_resource(task),
            action=self._extract_action(task),
            context=context.metadata,
        )

        policy = self._access_policies.get(request.resource, {})
        required_level = policy.get("level", SecurityLevel.MEDIUM)
        required_perms = policy.get("permissions", [])

        identity = await self._get_requestor_identity(request.requestor)

        if not identity:
            request.approved = False
            await self._create_alert(AlertType.PERMISSION_DENIED, SecurityLevel.HIGH,
                "Access denied: No identity",
                f"Requestor {request.requestor} not authenticated")
            return {"approved": False, "reason": "Not authenticated"}

        has_perm = any(p in identity.permissions for p in required_perms)
        level_ok = self._compare_security_level(identity.access_level, required_level)

        approved = has_perm and level_ok
        request.approved = approved

        if not approved:
            await self._create_alert(AlertType.PERMISSION_DENIED, SecurityLevel.HIGH,
                "Access denied: Insufficient privileges",
                f"Resource: {request.resource}, Action: {request.action}")

        return {
            "approved": approved,
            "resource": request.resource,
            "action": request.action,
            "required_level": required_level.value,
            "identity_level": identity.access_level.value,
        }

    def _compare_security_level(self, user_level: SecurityLevel, required: SecurityLevel) -> bool:
        levels = [SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH, SecurityLevel.CRITICAL]
        return levels.index(user_level) >= levels.index(required)

    async def _get_requestor_identity(self, requestor: str) -> FaceIdentity | None:
        for identity in self._identities.values():
            if "admin" in identity.permissions:
                return identity
        return None

    # -------------------------------------------------------------------------
    # SENTRY MODE
    # -------------------------------------------------------------------------

    async def _handle_sentry(self, task: str, context: AgentContext) -> dict:
        task_lower = task.lower()

        if "start" in task_lower or "enable" in task_lower:
            return await self.start_sentry()
        elif "stop" in task_lower or "disable" in task_lower:
            return await self.stop_sentry()
        elif "status" in task_lower:
            return {"sentry_active": self._sentry_active}

        return {"sentry_active": self._sentry_active}

    async def start_sentry(self) -> dict:
        if self._sentry_active:
            return {"status": "already_active"}

        self._sentry_active = True
        self._sentry_task = asyncio.create_task(self._sentry_loop())

        if self.event_bus:
            await self.event_bus.emit("security.sentry_started", {})

        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "security_status",
                "payload": {"sentry": True},
            }, room="hud")

        return {"status": "started", "message": "Sentry mode active - monitoring for intruders"}

    async def stop_sentry(self) -> dict:
        if not self._sentry_active:
            return {"status": "already_inactive"}

        self._sentry_active = False

        if self._sentry_task:
            self._sentry_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sentry_task

        if self.event_bus:
            await self.event_bus.emit("security.sentry_stopped", {})

        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "security_status",
                "payload": {"sentry": False},
            }, room="hud")

        return {"status": "stopped", "message": "Sentry mode deactivated"}

    async def _sentry_loop(self) -> None:
        while self._sentry_active:
            try:
                await self._check_system_anomalies()
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sentry loop error: {e}")
                await asyncio.sleep(10)

    async def _on_face_detected(self, detection) -> None:
        if not self._sentry_active:
            return

        if detection.identity == "unknown" and detection.confidence > 0.5:
            await self._create_alert(
                AlertType.INTRUDER,
                SecurityLevel.HIGH,
                "Intruder Alert",
                f"Unknown face detected with {detection.confidence:.0%} confidence",
                confidence=detection.confidence,
            )

    # -------------------------------------------------------------------------
    # ANOMALY DETECTION
    # -------------------------------------------------------------------------

    async def _handle_alerts(self, task: str, context: AgentContext) -> dict:
        return await self._get_recent_alerts()

    async def _get_recent_alerts(self, limit: int = 50) -> dict:
        alerts = sorted(
            self._alert_history,
            key=lambda a: a.timestamp,
            reverse=True,
        )[:limit]

        return {
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "type": a.alert_type.value,
                    "level": a.level.value,
                    "title": a.title,
                    "description": a.description,
                    "timestamp": a.timestamp,
                    "acknowledged": a.acknowledged,
                    "resolved": a.resolved,
                }
                for a in alerts
            ],
            "total": len(self._alert_history),
            "unacknowledged": sum(1 for a in self._alert_history if not a.acknowledged),
        }

    async def _create_alert(
        self,
        alert_type: AlertType,
        level: SecurityLevel,
        title: str,
        description: str,
        confidence: float = 0.0,
    ) -> SecurityAlert:
        alert = SecurityAlert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            level=level,
            title=title,
            description=description,
            source="guardian",
            confidence=confidence,
        )

        self._alerts[alert.alert_id] = alert
        self._alert_history.append(alert)

        if self.event_bus:
            await self.event_bus.emit("security.alert", {
                "alert_id": alert.alert_id,
                "type": alert_type.value,
                "level": level.value,
                "title": title,
                "description": description,
            })

        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "security_alert",
                "payload": {
                    "alert_id": alert.alert_id,
                    "type": alert_type.value,
                    "level": level.value,
                    "title": title,
                    "description": description,
                },
            }, room="hud")

        logger.warning(f"SECURITY ALERT [{level.value}]: {title} - {description}")

        return alert

    async def _check_system_anomalies(self) -> None:
        import psutil

        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "process_count": len(psutil.pids()),
        }

        for metric, value in metrics.items():
            history = self._metric_history.get(metric)
            if history:
                history.append(value)

                if len(history) >= 10:
                    baseline = self._baselines.get(metric)
                    if baseline:
                        baseline.mean = float(np.mean(history))
                        baseline.std = float(np.std(history))
                        baseline.min_val = float(np.min(history))
                        baseline.max_val = float(np.max(history))
                        baseline.sample_count = len(history)
                        baseline.updated_at = time.time()

                        if baseline.std > 0:
                            z_score = abs(value - baseline.mean) / baseline.std
                            if z_score > 3.0 * self._anomaly_sensitivity:
                                await self._create_alert(
                                    AlertType.SYSTEM_ANOMALY,
                                    SecurityLevel.MEDIUM,
                                    f"System Anomaly: {metric}",
                                    f"{metric} = {value:.1f}% (z-score: {z_score:.1f}, baseline: {baseline.mean:.1f}±{baseline.std:.1f})",
                                )

    # -------------------------------------------------------------------------
    # TOOL IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    async def _tool_enroll_face(self, params: dict) -> dict:
        name = params.get("name", "Unknown")
        return await self._handle_enroll(f"enroll {name}", AgentContext(conversation_id="tool"))

    async def _tool_verify_face(self, params: dict) -> dict:
        return await self._handle_verify("verify face", AgentContext(conversation_id="tool"))

    async def _tool_list_identities(self, params: dict) -> dict:
        return {
            "identities": [
                {
                    "identity_id": i.identity_id,
                    "name": i.name,
                    "access_level": i.access_level.value,
                    "permissions": i.permissions,
                    "enrolled_at": i.enrolled_at,
                    "last_seen": i.last_seen,
                }
                for i in self._identities.values()
            ]
        }

    async def _tool_revoke_identity(self, params: dict) -> dict:
        identity_id = params.get("identity_id")
        if identity_id in self._identities:
            del self._identities[identity_id]
            await self._save_identities()
            return {"success": True}
        return {"success": False, "error": "Identity not found"}

    async def _tool_check_access(self, params: dict) -> dict:
        resource = params.get("resource")
        action = params.get("action", "read")
        requestor = params.get("requestor", "unknown")

        request = AccessRequest(
            request_id=str(uuid.uuid4()),
            requestor=requestor,
            resource=resource,
            action=action,
        )

        return await self._handle_access(f"access {resource} {action}", AgentContext(
            conversation_id="tool",
            user_id=requestor,
        ))

    async def _tool_grant_access(self, params: dict) -> dict:
        identity_id = params.get("identity_id")
        permission = params.get("permission")

        if identity_id in self._identities:
            identity = self._identities[identity_id]
            if permission not in identity.permissions:
                identity.permissions.append(permission)
                await self._save_identities()
                return {"success": True}
        return {"success": False, "error": "Identity not found"}

    async def _tool_revoke_access(self, params: dict) -> dict:
        identity_id = params.get("identity_id")
        permission = params.get("permission")

        if identity_id in self._identities:
            identity = self._identities[identity_id]
            if permission in identity.permissions:
                identity.permissions.remove(permission)
                await self._save_identities()
                return {"success": True}
        return {"success": False, "error": "Identity not found"}

    async def _tool_audit_permissions(self, params: dict) -> dict:
        return {
            "policies": {
                k: {"level": v["level"].value, "permissions": v["permissions"]}
                for k, v in self._access_policies.items()
            },
            "identities": [
                {"name": i.name, "level": i.access_level.value, "permissions": i.permissions}
                for i in self._identities.values()
            ],
        }

    async def _tool_monitor_system(self, params: dict) -> dict:
        import psutil
        return {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage("/").percent,
            "processes": len(psutil.pids()),
            "timestamp": time.time(),
        }

    async def _tool_detect_anomalies(self, params: dict) -> dict:
        await self._check_system_anomalies()
        return {"status": "checked", "baselines": {k: v.__dict__ for k, v in self._baselines.items()}}

    async def _tool_get_alerts(self, params: dict) -> dict:
        limit = params.get("limit", 50)
        return await self._get_recent_alerts(limit)

    async def _tool_acknowledge_alert(self, params: dict) -> dict:
        alert_id = params.get("alert_id")
        if alert_id in self._alerts:
            self._alerts[alert_id].acknowledged = True
            return {"success": True}
        return {"success": False, "error": "Alert not found"}

    async def _tool_start_sentry(self, params: dict) -> dict:
        return await self.start_sentry()

    async def _tool_stop_sentry(self, params: dict) -> dict:
        return await self.stop_sentry()

    async def _tool_security_audit(self, params: dict) -> dict:
        return {
            "timestamp": time.time(),
            "identities": len(self._identities),
            "policies": len(self._access_policies),
            "alerts_total": len(self._alert_history),
            "alerts_unacknowledged": sum(1 for a in self._alert_history if not a.acknowledged),
            "sentry_active": self._sentry_active,
            "baselines": {k: v.__dict__ for k, v in self._baselines.items()},
        }

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _extract_name(self, task: str) -> str | None:
        import re
        patterns = [
            r"(?:enroll|register|add)\s+(\w+)",
            r"(\w+)\s*(?:face|identity)",
        ]
        for pattern in patterns:
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_resource(self, task: str) -> str:
        return "general"

    def _extract_action(self, task: str) -> str:
        return "read"

    async def _get_security_status(self, context: AgentContext) -> dict:
        return {
            "sentry_active": self._sentry_active,
            "identities_enrolled": len(self._identities),
            "alerts_pending": sum(1 for a in self._alert_history if not a.acknowledged),
            "access_policies": len(self._access_policies),
            "system_status": "secure" if not self._sentry_active else "monitoring",
        }

    # -------------------------------------------------------------------------
    # EXPORTS
    # -------------------------------------------------------------------------

__all__ = [
    "GuardianAgent",
    "SecurityLevel",
    "AlertType",
    "FaceIdentity",
    "SecurityAlert",
    "AccessRequest",
    "SystemBaseline",
]