"""
JARVIS OMEGA - Base Agent
Abstract agent class with LangGraph-compatible state management, tool calling, and memory integration.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional

from config import get_settings, settings

logger = logging.getLogger("jarvis.agents.base")


# =============================================================================
# DATA MODELS
# =============================================================================

class AgentState(str, Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"
    ERROR = "error"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    HANDOFF = "handoff"
    STATUS = "status"


@dataclass
class AgentConfig:
    """Agent configuration."""
    name: str
    role: str
    description: str
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = ""
    # Capabilities
    tools: list[str] = field(default_factory=list)
    can_delegate: bool = False
    can_handoff: bool = False
    max_retries: int = 3
    timeout: float = 60.0
    # Streaming
    stream_response: bool = True
    # Memory
    use_memory: bool = True
    memory_tiers: list[str] = field(default_factory=lambda: ["episodic", "semantic"])


@dataclass
class AgentMessage:
    """Structured message between agents."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.REQUEST
    source: str = ""
    target: str = ""
    payload: dict = field(default_factory=dict)
    # For streaming
    stream_id: str | None = None
    is_final: bool = True
    # Context
    conversation_id: str | None = None
    parent_message_id: str | None = None
    # Metadata
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "stream_id": self.stream_id,
            "is_final": self.is_final,
            "conversation_id": self.conversation_id,
            "parent_message_id": self.parent_message_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def create_request(
        cls,
        source: str,
        target: str,
        task: str,
        context: dict | None = None,
        conversation_id: str | None = None,
    ) -> "AgentMessage":
        return cls(
            message_type=MessageType.REQUEST,
            source=source,
            target=target,
            payload={"task": task, "context": context or {}},
            conversation_id=conversation_id,
        )

    @classmethod
    def create_response(
        cls,
        source: str,
        target: str,
        result: Any,
        parent_message_id: str,
        conversation_id: str | None = None,
    ) -> "AgentMessage":
        return cls(
            message_type=MessageType.RESPONSE,
            source=source,
            target=target,
            payload={"result": result},
            parent_message_id=parent_message_id,
            conversation_id=conversation_id,
        )

    @classmethod
    def create_stream_chunk(
        cls,
        source: str,
        target: str,
        chunk: str,
        stream_id: str,
        is_final: bool = False,
        conversation_id: str | None = None,
    ) -> "AgentMessage":
        return cls(
            message_type=MessageType.STREAM_CHUNK,
            source=source,
            target=target,
            payload={"chunk": chunk},
            stream_id=stream_id,
            is_final=is_final,
            conversation_id=conversation_id,
        )

    @classmethod
    def create_handoff(
        cls,
        source: str,
        target: str,
        task: str,
        context: dict,
        conversation_id: str | None = None,
    ) -> "AgentMessage":
        return cls(
            message_type=MessageType.HANDOFF,
            source=source,
            target=target,
            payload={"task": task, "context": context},
            conversation_id=conversation_id,
        )


@dataclass
class ToolCall:
    """Tool invocation request."""
    tool: str
    parameters: dict
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requestor: str = ""

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "parameters": self.parameters,
            "call_id": self.call_id,
            "requestor": self.requestor,
        }


@dataclass
class ToolResult:
    """Tool execution result."""
    call_id: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time: float = 0.0


# =============================================================================
# AGENT CONTEXT
# =============================================================================

@dataclass
class AgentContext:
    """Execution context passed to agent methods."""
    conversation_id: str
    user_id: str | None = None
    session_id: str | None = None
    # Memory
    short_term_memory: list[dict] = field(default_factory=list)
    working_memory: dict = field(default_factory=dict)
    # State
    variables: dict = field(default_factory=dict)
    # Callbacks
    emit_event: Callable[[str, dict], Any] | None = None
    send_message: Callable[[AgentMessage], Any] | None = None
    call_tool: Callable[[ToolCall], Any] | None = None
    # Metadata
    metadata: dict = field(default_factory=dict)


# =============================================================================
# BASE AGENT
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all VEGA agents.
    Provides common infrastructure for LLM-powered agents with tools.
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: Any = None,
        memory_manager: Any = None,
        tool_registry: Any = None,
        llm_client: Any = None,
    ):
        self.config = config
        self.event_bus = event_bus
        self.memory_manager = memory_manager
        self.tool_registry = tool_registry
        self.llm_client = llm_client

        # State
        self.state = AgentState.IDLE
        self._running = False
        self._current_task: asyncio.Task | None = None

        # Message handling
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._pending_streams: dict[str, asyncio.Queue] = {}

        # History
        self._conversation_history: deque = deque(maxlen=50)
        self._tool_history: deque = deque(maxlen=100)

        # Stats
        self._stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_tokens": 0,
            "total_tool_calls": 0,
            "avg_response_time": 0.0,
            "handoffs_initiated": 0,
            "handoffs_received": 0,
        }

        # Callbacks
        self._on_state_change: Callable[[AgentState], Any] | None = None
        self._on_message: Callable[[AgentMessage], Any] | None = None

        logger.info(f"Agent '{config.name}' initialized")

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    async def initialize(self) -> bool:
        """Initialize agent resources."""
        try:
            self._set_state(AgentState.INITIALIZING)

            # Initialize LLM client if not provided
            if not self.llm_client:
                await self._init_llm()

            # Initialize tools
            if self.config.tools and self.tool_registry:
                await self._verify_tools()

            # Agent-specific initialization
            await self._agent_initialize()

            self._set_state(AgentState.IDLE)
            logger.info(f"Agent '{self.config.name}' initialized")
            return True

        except Exception as e:
            logger.exception(f"Agent '{self.config.name}' initialization failed: {e}")
            self._set_state(AgentState.ERROR)
            return False

    async def shutdown(self) -> None:
        """Shutdown agent gracefully."""
        self._set_state(AgentState.SHUTTING_DOWN)
        self._running = False

        # Cancel current task
        if self._current_task:
            self._current_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._current_task

        # Clear queues
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Agent-specific cleanup
        await self._agent_shutdown()

        self._set_state(AgentState.STOPPED)
        logger.info(f"Agent '{self.config.name}' shutdown")

    @abstractmethod
    async def _agent_initialize(self) -> None:
        """Agent-specific initialization."""
        pass

    @abstractmethod
    async def _agent_shutdown(self) -> None:
        """Agent-specific cleanup."""
        pass

    async def _init_llm(self) -> None:
        """Initialize LLM client from config."""
        # This will be overridden by orchestrator with proper client
        pass

    async def _verify_tools(self) -> None:
        """Verify all configured tools are available."""
        if not self.tool_registry:
            return

        available = await self.tool_registry.list_tools()
        available_names = {t["name"] for t in available}

        for tool_name in self.config.tools:
            if tool_name not in available_names:
                logger.warning(f"Tool '{tool_name}' not found in registry")

    def _set_state(self, state: AgentState) -> None:
        """Update agent state and notify."""
        old_state = self.state
        self.state = state

        if old_state != state and self._on_state_change:
            try:
                self._on_state_change(state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

        # Emit event
        if self.event_bus:
            asyncio.create_task(self.event_bus.emit("agent.state_change", {
                "agent": self.config.name,
                "old_state": old_state.value,
                "new_state": state.value,
            }))

    # -------------------------------------------------------------------------
    # MESSAGE PROCESSING
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start message processing loop."""
        if self._running:
            return

        self._running = True
        self._current_task = asyncio.create_task(self._process_loop())
        logger.debug(f"Agent '{self.config.name}' message loop started")

    async def stop(self) -> None:
        """Stop message processing."""
        self._running = False
        if self._current_task:
            self._current_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._current_task

    async def send_message(self, message: AgentMessage) -> bool:
        """Send message to this agent."""
        try:
            await self._message_queue.put(message)
            return True
        except asyncio.QueueFull:
            logger.warning(f"Agent '{self.config.name}' message queue full")
            return False

    async def _process_loop(self) -> None:
        """Main message processing loop."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Agent '{self.config.name}' process loop error: {e}")

    async def _handle_message(self, message: AgentMessage) -> None:
        """Route message to appropriate handler."""
        if self._on_message:
            await self._on_message(message)

        if message.message_type == MessageType.REQUEST:
            await self._handle_request(message)
        elif message.message_type == MessageType.HANDOFF:
            await self._handle_handoff(message)
        elif message.message_type == MessageType.TOOL_RESULT:
            await self._handle_tool_result(message)

    async def _handle_request(self, message: AgentMessage) -> None:
        """Handle incoming request."""
        self._set_state(AgentState.PROCESSING)
        start_time = time.time()

        try:
            task = message.payload.get("task", "")
            context = message.payload.get("context", {})

            # Build agent context
            agent_context = AgentContext(
                conversation_id=message.conversation_id or str(uuid.uuid4()),
                session_id=context.get("session_id"),
                user_id=context.get("user_id"),
                emit_event=self._emit_event,
                send_message=self._send_message_to_agent,
                call_tool=self._call_tool,
            )

            # Load relevant memory
            if self.config.use_memory and self.memory_manager:
                agent_context.short_term_memory = await self._load_memory(task)

            # Process task
            result = await self.process_task(task, agent_context)

            # Save to memory
            if self.config.use_memory and self.memory_manager:
                await self._save_memory(task, result, agent_context)

            # Send response
            response = AgentMessage.create_response(
                source=self.config.name,
                target=message.source,
                result=result,
                parent_message_id=message.message_id,
                conversation_id=message.conversation_id,
            )
            await self._send_message(response)

            # Update stats
            elapsed = time.time() - start_time
            self._stats["tasks_completed"] += 1
            self._stats["avg_response_time"] = (
                (self._stats["avg_response_time"] * (self._stats["tasks_completed"] - 1) + elapsed)
                / self._stats["tasks_completed"]
            )

        except Exception as e:
            logger.exception(f"Request handling error: {e}")
            self._stats["tasks_failed"] += 1

            error_response = AgentMessage(
                message_type=MessageType.ERROR,
                source=self.config.name,
                target=message.source,
                payload={"error": str(e)},
                parent_message_id=message.message_id,
                conversation_id=message.conversation_id,
            )
            await self._send_message(error_response)

        finally:
            self._set_state(AgentState.IDLE)

    async def _handle_handoff(self, message: AgentMessage) -> None:
        """Handle agent handoff."""
        self._stats["handoffs_received"] += 1
        # Treat as request with handoff context
        message.message_type = MessageType.REQUEST
        message.payload["is_handoff"] = True
        await self._handle_request(message)

    async def _handle_tool_result(self, message: AgentMessage) -> None:
        """Handle tool execution result."""
        # This would be used for async tool calls
        pass

    # -------------------------------------------------------------------------
    # TASK PROCESSING (Override in subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def process_task(
        self,
        task: str,
        context: AgentContext,
    ) -> Any:
        """
        Process a task and return result.
        Override in subclasses for specific behavior.
        """
        pass

    # Streaming version
    async def process_task_stream(
        self,
        task: str,
        context: AgentContext,
    ) -> AsyncGenerator[str, None]:
        """
        Process task with streaming response.
        Default implementation yields full result at once.
        """
        result = await self.process_task(task, context)
        if isinstance(result, str):
            yield result
        else:
            yield str(result)

    # -------------------------------------------------------------------------
    # TOOL CALLING
    # -------------------------------------------------------------------------

    async def _call_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        if not self.tool_registry:
            return ToolResult(
                call_id=tool_call.call_id,
                success=False,
                error="No tool registry available",
            )

        self._stats["total_tool_calls"] += 1
        start_time = time.time()

        try:
            result = await self.tool_registry.execute(
                tool_call.tool,
                tool_call.parameters,
                requestor=self.config.name,
            )

            return ToolResult(
                call_id=tool_call.call_id,
                success=True,
                result=result,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Tool '{tool_call.tool}' error: {e}")
            return ToolResult(
                call_id=tool_call.call_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def call_tool_sync(self, tool: str, parameters: dict) -> ToolResult:
        """Synchronous tool call for non-async contexts."""
        tool_call = ToolCall(tool=tool, parameters=parameters, requestor=self.config.name)
        return asyncio.run(self._call_tool(tool_call))

    # -------------------------------------------------------------------------
    # MEMORY INTEGRATION
    # -------------------------------------------------------------------------

    async def _load_memory(self, query: str) -> list[dict]:
        """Load relevant memory for task."""
        if not self.memory_manager:
            return []

        try:
            results = await self.memory_manager.search(
                query,
                limit=5,
                tier="episodic",
            )
            return results.get("results", [])
        except Exception as e:
            logger.error(f"Memory load error: {e}")
            return []

    async def _save_memory(
        self,
        task: str,
        result: Any,
        context: AgentContext,
    ) -> None:
        """Save task result to memory."""
        if not self.memory_manager:
            return

        try:
            content = f"Task: {task}\nResult: {result}"
            await self.memory_manager.ingest(
                content=content,
                metadata={
                    "agent": self.config.name,
                    "conversation_id": context.conversation_id,
                    "task_type": self._classify_task(task),
                },
                tier="episodic",
            )
        except Exception as e:
            logger.error(f"Memory save error: {e}")

    def _classify_task(self, task: str) -> str:
        """Classify task type for memory organization."""
        task_lower = task.lower()
        if any(w in task_lower for w in ["code", "script", "program", "debug"]):
            return "coding"
        elif any(w in task_lower for w in ["search", "find", "research", "look up"]):
            return "research"
        elif any(w in task_lower for w in ["create", "generate", "write", "design"]):
            return "creative"
        elif any(w in task_lower for w in ["plan", "schedule", "organize", "remind"]):
            return "planning"
        elif any(w in task_lower for w in ["security", "auth", "protect", "monitor"]):
            return "security"
        return "general"

    # -------------------------------------------------------------------------
    # COMMUNICATION HELPERS
    # -------------------------------------------------------------------------

    async def _send_message(self, message: AgentMessage) -> None:
        """Send message via event bus or direct callback."""
        if self.event_bus:
            await self.event_bus.emit("agent.message", message.to_dict())

    async def _send_message_to_agent(self, target: str, message: AgentMessage) -> bool:
        """Send message to another agent via event bus."""
        if self.event_bus:
            await self.event_bus.emit("agent.route_message", {
                "target": target,
                "message": message.to_dict(),
            })
            return True
        return False

    async def _emit_event(self, event_type: str, data: dict) -> None:
        """Emit event to event bus."""
        if self.event_bus:
            await self.event_bus.emit(event_type, {
                "agent": self.config.name,
                **data,
            })

    async def delegate_task(
        self,
        target_agent: str,
        task: str,
        context: dict | None = None,
        conversation_id: str | None = None,
    ) -> Any:
        """Delegate task to another agent."""
        if not self.config.can_delegate:
            raise PermissionError(f"Agent '{self.config.name}' cannot delegate")

        self._stats["handoffs_initiated"] += 1

        handoff = AgentMessage.create_handoff(
            source=self.config.name,
            target=target_agent,
            task=task,
            context=context or {},
            conversation_id=conversation_id,
        )

        await self._send_message(handoff)

        # Wait for response (simplified - would use proper async pattern)
        return {"status": "delegated", "target": target_agent}

    # -------------------------------------------------------------------------
    # LLM INTERACTION
    # -------------------------------------------------------------------------

    async def _call_llm(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Call LLM with messages."""
        if not self.llm_client:
            raise RuntimeError("LLM client not initialized")

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        if stream:
            return self._call_llm_stream(messages, temp, max_tok)
        else:
            return await self._call_llm_complete(messages, temp, max_tok)

    async def _call_llm_complete(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Complete LLM call (non-streaming)."""
        # Override in subclass or use provided client
        raise NotImplementedError("LLM client not configured")

    async def _call_llm_stream(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        """Streaming LLM call."""
        # Override in subclass
        raise NotImplementedError("LLM streaming not configured")

    def _build_messages(
        self,
        task: str,
        context: AgentContext,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """Build message list for LLM."""
        messages = []

        # System prompt
        sys_prompt = system_prompt or self.config.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        # Short-term memory (recent conversation)
        for mem in context.short_term_memory[-10:]:
            if "role" in mem and "content" in mem:
                messages.append(mem)

        # Working memory context
        if context.working_memory:
            wm_summary = "Working Memory:\n" + "\n".join(
                f"- {k}: {v}" for k, v in context.working_memory.items()
            )
            messages.append({"role": "system", "content": wm_summary})

        # Current task
        messages.append({"role": "user", "content": task})

        return messages

    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------

    def on_state_change(self, callback: Callable[[AgentState], Any]) -> None:
        self._on_state_change = callback

    def on_message(self, callback: Callable[[AgentMessage], Any]) -> None:
        self._on_message = callback

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "state": self.state.value,
            "name": self.config.name,
            "uptime": time.time() - getattr(self, "_start_time", time.time()),
        }

    def get_config(self) -> AgentConfig:
        return self.config

    async def health_check(self) -> dict[str, Any]:
        """Health check for monitoring."""
        return {
            "agent": self.config.name,
            "state": self.state.value,
            "healthy": self.state not in [AgentState.ERROR, AgentState.STOPPED],
            "stats": self.get_stats(),
        }


# =============================================================================
# AGENT FACTORY
# =============================================================================

class AgentFactory:
    """Factory for creating agent instances."""

    _agent_classes: dict[str, type] = {}

    @classmethod
    def register(cls, agent_type: str, agent_class: type) -> None:
        cls._agent_classes[agent_type] = agent_class

    @classmethod
    async def create(
        cls,
        agent_type: str,
        config: AgentConfig,
        event_bus: Any = None,
        memory_manager: Any = None,
        tool_registry: Any = None,
        llm_client: Any = None,
    ) -> BaseAgent:
        if agent_type not in cls._agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_class = cls._agent_classes[agent_type]
        agent = agent_class(
            config=config,
            event_bus=event_bus,
            memory_manager=memory_manager,
            tool_registry=tool_registry,
            llm_client=llm_client,
        )

        await agent.initialize()
        return agent


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "AgentState",
    "AgentMessage",
    "MessageType",
    "ToolCall",
    "ToolResult",
    "AgentContext",
    "AgentFactory",
]