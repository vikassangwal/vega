"""
JARVIS OMEGA - Main Application Entry Point
FastAPI application with WebSocket support, middleware, and lifecycle management.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client.core import CollectorRegistry

from config import settings, get_settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOGS_DIR / "jarvis.log"),
    ],
)
logger = logging.getLogger("jarvis")

# Prometheus metrics
REGISTRY = CollectorRegistry()
HTTP_REQUESTS_TOTAL = Counter(
    "jarvis_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=REGISTRY,
)
HTTP_REQUEST_DURATION = Histogram(
    "jarvis_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    registry=REGISTRY,
)
WS_CONNECTIONS = Counter(
    "jarvis_ws_connections_total",
    "Total WebSocket connections",
    ["status"],
    registry=REGISTRY,
)
WS_MESSAGES = Counter(
    "jarvis_ws_messages_total",
    "Total WebSocket messages",
    ["direction", "type"],
    registry=REGISTRY,
)
AGENT_TASKS = Counter(
    "jarvis_agent_tasks_total",
    "Total agent tasks executed",
    ["agent", "status"],
    registry=REGISTRY,
)

# Application state
class AppState:
    def __init__(self):
        self.settings = settings
        self.start_time = time.time()
        self.websocket_manager = None
        self.orchestrator = None
        self.task_queue = None
        self.event_bus = None
        self.state_manager = None
        self.memory_manager = None
        self.voice_engine = None
        self.vision_engine = None
        self.background_tasks: set[asyncio.Task] = set()
        self.shutdown_event = asyncio.Event()

    @property
    def uptime(self) -> float:
        return time.time() - self.start_time


app_state = AppState()

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.connection_metadata: dict[str, dict] = {}
        self.rooms: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str, metadata: dict | None = None) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections[client_id] = websocket
            self.connection_metadata[client_id] = metadata or {}
            WS_CONNECTIONS.labels(status="connected").inc()
            logger.info(f"WebSocket connected: {client_id} (total: {len(self.active_connections)})")

    async def disconnect(self, client_id: str, code: int = 1000, reason: str = "Normal closure") -> None:
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            if client_id in self.connection_metadata:
                del self.connection_metadata[client_id]
            # Remove from all rooms
            for room in self.rooms.values():
                room.discard(client_id)
            WS_CONNECTIONS.labels(status="disconnected").inc()
            logger.info(f"WebSocket disconnected: {client_id} (code: {code}, reason: {reason})")

    async def send_personal_message(self, client_id: str, message: dict) -> bool:
        async with self._lock:
            websocket = self.active_connections.get(client_id)
            if websocket:
                try:
                    await websocket.send_json(message)
                    WS_MESSAGES.labels(direction="out", type=message.get("type", "unknown")).inc()
                    return True
                except Exception as e:
                    logger.error(f"Failed to send personal message to {client_id}: {e}")
                    await self.disconnect(client_id, code=1011, reason=f"Send failed: {e}")
        return False

    async def broadcast(self, message: dict, room: str | None = None, exclude: str | None = None) -> int:
        sent = 0
        async with self._lock:
            if room:
                targets = self.rooms.get(room, set())
            else:
                targets = set(self.active_connections.keys())

            if exclude:
                targets.discard(exclude)

            for client_id in list(targets):
                websocket = self.active_connections.get(client_id)
                if websocket:
                    try:
                        await websocket.send_json(message)
                        sent += 1
                    except Exception:
                        await self.disconnect(client_id, code=1011, reason="Broadcast failed")

        if sent > 0:
            WS_MESSAGES.labels(direction="out", type=message.get("type", "broadcast")).inc()
        return sent

    async def join_room(self, client_id: str, room: str) -> None:
        async with self._lock:
            if room not in self.rooms:
                self.rooms[room] = set()
            self.rooms[room].add(client_id)

    async def leave_room(self, client_id: str, room: str) -> None:
        async with self._lock:
            if room in self.rooms:
                self.rooms[room].discard(client_id)

    def get_connection_count(self) -> int:
        return len(self.active_connections)

    def get_room_members(self, room: str) -> list[str]:
        return list(self.rooms.get(room, set()))


class WebSocketManager:
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.message_handlers: dict[str, callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        self.message_handlers.update({
            "ping": self._handle_ping,
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe,
            "agent_command": self._handle_agent_command,
            "voice_input": self._handle_voice_input,
            "vision_frame": self._handle_vision_frame,
        })

    def register_handler(self, message_type: str, handler: callable) -> None:
        self.message_handlers[message_type] = handler

    async def handle_connection(self, websocket: WebSocket, client_id: str) -> None:
        metadata = {"connected_at": time.time(), "client_id": client_id}
        await self.connection_manager.connect(websocket, client_id, metadata)

        try:
            while True:
                data = await websocket.receive_json()
                WS_MESSAGES.labels(direction="in", type=data.get("type", "unknown")).inc()
                await self._route_message(client_id, data)
        except WebSocketDisconnect as e:
            await self.connection_manager.disconnect(client_id, code=e.code, reason=e.reason)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await self.connection_manager.disconnect(client_id, code=1011, reason=str(e))

    async def _route_message(self, client_id: str, message: dict) -> None:
        message_type = message.get("type")
        payload = message.get("payload", {})
        request_id = message.get("request_id")

        handler = self.message_handlers.get(message_type)
        if handler:
            try:
                response = await handler(client_id, payload)
                if response and request_id:
                    await self.connection_manager.send_personal_message(client_id, {
                        "type": "response",
                        "request_id": request_id,
                        "payload": response,
                    })
            except Exception as e:
                logger.error(f"Handler error for {message_type}: {e}")
                if request_id:
                    await self.connection_manager.send_personal_message(client_id, {
                        "type": "error",
                        "request_id": request_id,
                        "payload": {"message": str(e), "code": "HANDLER_ERROR"},
                    })
        else:
            logger.warning(f"Unknown message type: {message_type}")
            if request_id:
                await self.connection_manager.send_personal_message(client_id, {
                    "type": "error",
                    "request_id": request_id,
                    "payload": {"message": f"Unknown message type: {message_type}", "code": "UNKNOWN_TYPE"},
                })

    async def _handle_ping(self, client_id: str, payload: dict) -> dict:
        return {"pong": True, "timestamp": time.time()}

    async def _handle_subscribe(self, client_id: str, payload: dict) -> dict:
        room = payload.get("room")
        if room:
            await self.connection_manager.join_room(client_id, room)
            return {"subscribed": room}
        return {"error": "No room specified"}

    async def _handle_unsubscribe(self, client_id: str, payload: dict) -> dict:
        room = payload.get("room")
        if room:
            await self.connection_manager.leave_room(client_id, room)
            return {"unsubscribed": room}
        return {"error": "No room specified"}

    async def _handle_agent_command(self, client_id: str, payload: dict) -> dict:
        if app_state.orchestrator:
            task = payload.get("task")
            agent = payload.get("agent", "orchestrator")
            context = payload.get("context", {})
            return {"status": "queued", "task": task, "agent": agent}
        return {"error": "Orchestrator not initialized"}

    async def _handle_voice_input(self, client_id: str, payload: dict) -> dict:
        if app_state.voice_engine:
            return await app_state.voice_engine.process_audio(payload)
        return {"error": "Voice engine not initialized"}

    async def _handle_vision_frame(self, client_id: str, payload: dict) -> dict:
        if app_state.vision_engine:
            return await app_state.vision_engine.process_frame(payload)
        return {"error": "Vision engine not initialized"}

    async def broadcast_agent_status(self, agent: str, status: str, data: dict | None = None) -> None:
        await self.connection_manager.broadcast({
            "type": "agent_status",
            "payload": {"agent": agent, "status": status, "data": data or {}, "timestamp": time.time()},
        }, room="hud")

    async def broadcast_system_metrics(self, metrics: dict) -> None:
        await self.connection_manager.broadcast({
            "type": "system_metrics",
            "payload": metrics,
        }, room="hud")

    async def broadcast_notification(self, notification: dict) -> None:
        await self.connection_manager.broadcast({
            "type": "notification",
            "payload": notification,
        })


# Middleware
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    HTTP_REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    return response


async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", "")

    logger.info(
        "Request started",
        extra={
            "method": request.method,
            "path": request.url.path,
            "request_id": request_id,
            "client": request.client.host if request.client else "unknown",
        }
    )

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(
        "Request completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "request_id": request_id,
        }
    )

    return response


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 JARVIS OMEGA starting up...")

    try:
        # Initialize core components
        await startup()

        yield

    finally:
        logger.info("🛑 JARVIS OMEGA shutting down...")
        await shutdown()


async def startup() -> None:
    """Initialize all subsystems."""
    try:
        # Initialize WebSocket manager
        app_state.websocket_manager = WebSocketManager()
        logger.info("✅ WebSocket manager initialized")

        # Initialize event bus
        from core.event_bus import EventBus
        app_state.event_bus = EventBus()
        await app_state.event_bus.start()
        logger.info("✅ Event bus started")

        # Initialize state manager
        from core.state_manager import StateManager
        state_manager = StateManager()
        await state_manager.initialize()
        app_state.state_manager = state_manager
        logger.info("✅ State manager initialized")

        # Initialize task queue
        from core.task_queue import TaskQueue
        app_state.task_queue = TaskQueue()
        await app_state.task_queue.start()
        logger.info("✅ Task queue started")

        # Initialize memory manager
        from memory.memory_manager import MemoryManager
        app_state.memory_manager = MemoryManager()
        await app_state.memory_manager.initialize()
        logger.info("✅ Memory manager initialized")

        # Initialize orchestrator (multi-agent brain)
        from agents.orchestrator import Orchestrator
        app_state.orchestrator = Orchestrator(
            event_bus=app_state.event_bus,
            memory_manager=app_state.memory_manager,
            task_queue=app_state.task_queue,
            websocket_manager=app_state.websocket_manager,
        )
        await app_state.orchestrator.initialize()
        logger.info("✅ Orchestrator initialized")

        # Initialize voice engine
        if settings.STT_PROVIDER != "none":
            from senses.voice_engine import VoiceEngine
            app_state.voice_engine = VoiceEngine(
                websocket_manager=app_state.websocket_manager,
                event_bus=app_state.event_bus,
            )
            await app_state.voice_engine.start()
            logger.info("✅ Voice engine started")

        # Initialize vision engine
        if settings.YOLO_MODEL:
            from senses.vision_engine import VisionEngine
            app_state.vision_engine = VisionEngine(
                websocket_manager=app_state.websocket_manager,
                event_bus=app_state.event_bus,
            )
            await app_state.vision_engine.start()
            logger.info("✅ Vision engine started")

        # Start background monitoring
        if settings.PROMETHEUS_ENABLED:
            task = asyncio.create_task(proactive_monitor())
            app_state.background_tasks.add(task)
            task.add_done_callback(app_state.background_tasks.discard)
            logger.info("✅ Proactive monitor started")

        # Register WebSocket handlers with orchestrator
        ws_manager = app_state.websocket_manager
        ws_manager.register_handler("agent_command", app_state.orchestrator.handle_command)

        logger.info("🎉 JARVIS OMEGA startup complete!")

    except Exception as e:
        logger.exception("Startup failed")
        raise


async def shutdown() -> None:
    """Graceful shutdown of all subsystems."""
    # Signal shutdown
    app_state.shutdown_event.set()

    # Cancel background tasks
    for task in app_state.background_tasks:
        task.cancel()
    if app_state.background_tasks:
        await asyncio.gather(*app_state.background_tasks, return_exceptions=True)

    # Shutdown components in reverse order
    if app_state.vision_engine:
        await app_state.vision_engine.stop()
        logger.info("✅ Vision engine stopped")

    if app_state.voice_engine:
        await app_state.voice_engine.stop()
        logger.info("✅ Voice engine stopped")

    if app_state.orchestrator:
        await app_state.orchestrator.shutdown()
        logger.info("✅ Orchestrator stopped")

    if app_state.task_queue:
        await app_state.task_queue.stop()
        logger.info("✅ Task queue stopped")

    if app_state.memory_manager:
        await app_state.memory_manager.shutdown()
        logger.info("✅ Memory manager stopped")

    if app_state.event_bus:
        await app_state.event_bus.stop()
        logger.info("✅ Event bus stopped")

    logger.info("👋 JARVIS OMEGA shutdown complete")


async def proactive_monitor() -> None:
    """Background proactive monitoring task."""
    while not app_state.shutdown_event.is_set():
        try:
            # Collect system metrics
            import psutil
            metrics = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "battery_percent": psutil.sensors_battery().percent if psutil.sensors_battery() else None,
                "timestamp": time.time(),
            }

            # Broadcast to HUD
            if app_state.websocket_manager:
                await app_state.websocket_manager.broadcast_system_metrics(metrics)

            # Check anomalies
            await check_anomalies(metrics)

            # Sleep until next interval
            await asyncio.sleep(settings.PROMETHEUS_ENABLED and 30 or 60)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Proactive monitor error: {e}")
            await asyncio.sleep(5)


async def check_anomalies(metrics: dict) -> None:
    """Check for system anomalies and trigger alerts."""
    thresholds = {
        "battery_percent": 20,
        "cpu_percent": 90,
        "memory_percent": 90,
        "disk_percent": 90,
    }

    alerts = []

    if metrics.get("battery_percent") is not None and metrics["battery_percent"] < thresholds["battery_percent"]:
        alerts.append({
            "type": "battery_low",
            "value": metrics["battery_percent"],
            "threshold": thresholds["battery_percent"],
            "severity": "warning" if metrics["battery_percent"] > 10 else "critical",
        })

    if metrics["cpu_percent"] > thresholds["cpu_percent"]:
        alerts.append({
            "type": "cpu_high",
            "value": metrics["cpu_percent"],
            "threshold": thresholds["cpu_percent"],
            "severity": "warning",
        })

    if metrics["memory_percent"] > thresholds["memory_percent"]:
        alerts.append({
            "type": "memory_high",
            "value": metrics["memory_percent"],
            "threshold": thresholds["memory_percent"],
            "severity": "warning",
        })

    if metrics["disk_percent"] > thresholds["disk_percent"]:
        alerts.append({
            "type": "disk_high",
            "value": metrics["disk_percent"],
            "threshold": thresholds["disk_percent"],
            "severity": "warning",
        })

    for alert in alerts:
        if app_state.websocket_manager:
            await app_state.websocket_manager.broadcast_notification({
                "id": f"anomaly_{alert['type']}_{int(time.time())}",
                "level": alert["severity"],
                "title": f"System Alert: {alert['type'].replace('_', ' ').title()}",
                "message": f"{alert['type'].replace('_', ' ').title()} at {alert['value']:.1f}% (threshold: {alert['threshold']}%)",
                "timestamp": time.time(),
                "data": alert,
            })


# FastAPI Application
app = FastAPI(
    title="JARVIS OMEGA",
    description="Advanced Autonomous AI Assistant - 1000x JARVIS",
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan,
)

# Middleware stack (order matters - outer to inner)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["jarvis.local", "localhost", "127.0.0.1"],
    )
app.middleware("http")(metrics_middleware)
app.middleware("http")(logging_middleware)

# Health & Metrics Endpoints
@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    """Comprehensive health check."""
    checks = {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT.value,
        "uptime_seconds": round(app_state.uptime, 2),
        "components": {},
    }

    # Check database
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
        engine = create_async_engine(settings.DATABASE_URL)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["components"]["database"] = "healthy"
    except Exception as e:
        checks["components"]["database"] = f"unhealthy: {e}"
        checks["status"] = "degraded"

    # Check Redis
    try:
        import redis.asyncio as redis
        r = redis.from_url(settings.REDIS_URL)
        await r.ping()
        await r.close()
        checks["components"]["redis"] = "healthy"
    except Exception as e:
        checks["components"]["redis"] = f"unhealthy: {e}"
        checks["status"] = "degraded"

    # Check Neo4j
    try:
        from neo4j import AsyncGraphDatabase
        driver = AsyncGraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        async with driver.session() as session:
            await session.run("RETURN 1")
        await driver.close()
        checks["components"]["neo4j"] = "healthy"
    except Exception as e:
        checks["components"]["neo4j"] = f"unhealthy: {e}"
        checks["status"] = "degraded"

    # Check ChromaDB
    try:
        import chromadb
        client = chromadb.HttpClient(host=settings.CHROMADB_HOST, port=settings.CHROMADB_PORT)
        client.heartbeat()
        checks["components"]["chromadb"] = "healthy"
    except Exception as e:
        checks["components"]["chromadb"] = f"unhealthy: {e}"
        checks["status"] = "degraded"

    # Check Ollama
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.OLLAMA_HOST}/api/tags")
            resp.raise_for_status()
        checks["components"]["ollama"] = "healthy"
    except Exception as e:
        checks["components"]["ollama"] = f"unhealthy: {e}"

    # WebSocket connections
    if app_state.websocket_manager:
        checks["components"]["websocket_connections"] = app_state.websocket_manager.connection_manager.get_connection_count()

    return checks


@app.get("/health/live", tags=["Health"])
async def liveness_check() -> PlainTextResponse:
    """Kubernetes liveness probe."""
    return PlainTextResponse("OK")


@app.get("/health/ready", tags=["Health"])
async def readiness_check() -> dict[str, Any]:
    """Kubernetes readiness probe."""
    ready = (
        app_state.orchestrator is not None and
        app_state.memory_manager is not None and
        app_state.task_queue is not None
    )
    return {"ready": ready, "orchestrator": app_state.orchestrator is not None}


@app.get("/metrics", tags=["Monitoring"])
async def metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest(REGISTRY), media_type="text/plain")


# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Main WebSocket endpoint for real-time communication."""
    client_id = websocket.query_params.get("client_id", f"client_{id(websocket)}")
    await app_state.websocket_manager.handle_connection(websocket, client_id)


# REST API Endpoints
@app.get("/api/v1/status", tags=["Status"])
async def get_status() -> dict[str, Any]:
    """Get system status."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT.value,
        "uptime_seconds": round(app_state.uptime, 2),
        "features": {
            "voice": settings.STT_PROVIDER != "none",
            "vision": bool(settings.YOLO_MODEL),
            "memory": settings.MEM0_API_KEY is not None or settings.MEM0_LOCAL_MODE,
            "proactive": settings.PROMETHEUS_ENABLED,
            "tools": True,
            "plugins": True,
        },
        "agents": list(settings.AGENTS.keys()) if hasattr(settings, "AGENTS") else [
            "orchestrator", "executor", "researcher", "creative", "guardian", "planner", "toolsmith"
        ],
        "websocket_connections": app_state.websocket_manager.connection_manager.get_connection_count() if app_state.websocket_manager else 0,
    }


@app.get("/api/v1/agents", tags=["Agents"])
async def list_agents() -> dict[str, Any]:
    """List all available agents and their status."""
    agents_info = {}
    for name, config in settings.AGENTS.items() if hasattr(settings, "AGENTS") else {}:
        agents_info[name] = {
            "name": name,
            "model": config.get("model"),
            "temperature": config.get("temperature"),
            "enabled": True,
        }
    return {"agents": agents_info}


@app.post("/api/v1/agents/{agent_name}/execute", tags=["Agents"])
async def execute_agent(agent_name: str, request: Request) -> dict[str, Any]:
    """Execute a task on a specific agent."""
    if agent_name not in (settings.AGENTS.keys() if hasattr(settings, "AGENTS") else {}):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": f"Agent '{agent_name}' not found"},
        )

    body = await request.json()
    task = body.get("task")
    context = body.get("context", {})

    if not task:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Task is required"},
        )

    if not app_state.orchestrator:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "Orchestrator not initialized"},
        )

    result = await app_state.orchestrator.execute_task(agent_name, task, context)
    return result


@app.post("/api/v1/chat", tags=["Chat"])
async def chat(request: Request) -> dict[str, Any]:
    """Main chat endpoint - routes to orchestrator."""
    body = await request.json()
    message = body.get("message")
    context = body.get("context", {})
    stream = body.get("stream", False)

    if not message:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Message is required"},
        )

    if not app_state.orchestrator:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "Orchestrator not initialized"},
        )

    result = await app_state.orchestrator.chat(message, context, stream)
    return result


@app.get("/api/v1/memory/search", tags=["Memory"])
async def search_memory(q: str, limit: int = 10, tier: str | None = None) -> dict[str, Any]:
    """Search memory across tiers."""
    if not app_state.memory_manager:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "Memory manager not initialized"},
        )

    results = await app_state.memory_manager.search(q, limit=limit, tier=tier)
    return {"results": results, "query": q, "limit": limit}


@app.post("/api/v1/memory/ingest", tags=["Memory"])
async def ingest_memory(request: Request) -> dict[str, Any]:
    """Ingest content into memory."""
    if not app_state.memory_manager:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "Memory manager not initialized"},
        )

    body = await request.json()
    content = body.get("content")
    metadata = body.get("metadata", {})
    tier = body.get("tier", "episodic")

    if not content:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Content is required"},
        )

    result = await app_state.memory_manager.ingest(content, metadata, tier)
    return result


@app.get("/api/v1/tools", tags=["Tools"])
async def list_tools() -> dict[str, Any]:
    """List all available tools."""
    from tools.tool_registry import ToolRegistry
    registry = ToolRegistry()
    tools = await registry.list_tools()
    return {"tools": tools}


@app.post("/api/v1/tools/execute", tags=["Tools"])
async def execute_tool(request: Request) -> dict[str, Any]:
    """Execute a tool by name."""
    from tools.tool_registry import ToolRegistry
    registry = ToolRegistry()

    body = await request.json()
    tool_name = body.get("tool")
    params = body.get("params", {})

    if not tool_name:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Tool name is required"},
        )

    result = await registry.execute(tool_name, params)
    return result


# Static Files (Frontend)
frontend_build = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_build.exists():
    app.mount("/", StaticFiles(directory=str(frontend_build), html=True), name="frontend")


# Signal Handlers
def setup_signal_handlers() -> None:
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        app_state.shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# Main Entry Point
if __name__ == "__main__":
    setup_signal_handlers()

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS if settings.ENVIRONMENT == "production" else 1,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.ENVIRONMENT == "development",
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )