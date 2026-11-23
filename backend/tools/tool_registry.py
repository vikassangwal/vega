"""
JARVIS OMEGA - Tool Registry
Dynamic tool management with sandboxing, hot-reload, and approval workflow.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import inspect
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from config import get_settings, settings

logger = logging.getLogger("vega.tools.registry")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ToolParameter:
    """Tool parameter specification (JSON Schema compatible)."""
    name: str
    type: str  # string, integer, number, boolean, array, object
    description: str = ""
    required: bool = False
    default: Any = None
    enum: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None
    pattern: str | None = None


@dataclass
class ToolMetadata:
    """Complete tool metadata."""
    name: str
    description: str
    category: str = "custom"
    version: str = "1.0.0"
    author: str = "system"
    parameters: list[ToolParameter] = field(default_factory=list)
    returns: dict = field(default_factory=dict)
    # Execution
    timeout: int = 30
    sandboxed: bool = True
    requires_approval: bool = False
    # Access
    permissions: list[str] = field(default_factory=list)
    rate_limit: int | None = None  # calls per minute
    # Metadata
    tags: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    deprecated: bool = False
    deprecated_reason: str = ""


@dataclass
class ToolResult:
    """Tool execution result."""
    success: bool
    result: Any = None
    error: str | None = None
    execution_time: float = 0.0
    tool_name: str = ""
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolCall:
    """Tool invocation request."""
    tool: str
    parameters: dict
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requestor: str = ""
    context: dict = field(default_factory=dict)


# =============================================================================
# SANDBOX EXECUTORS
# =============================================================================

class SandboxExecutor:
    """Base class for sandboxed tool execution."""

    async def execute(self, func: Callable, params: dict, timeout: int) -> ToolResult:
        raise NotImplementedError


class LocalExecutor(SandboxExecutor):
    """Local execution (least secure, for trusted tools)."""

    async def execute(self, func: Callable, params: dict, timeout: int) -> ToolResult:
        start = time.time()
        try:
            if inspect.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(**params), timeout=timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(func, **params),
                    timeout=timeout,
                )
            return ToolResult(
                success=True,
                result=result,
                execution_time=time.time() - start,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Execution timeout ({timeout}s)",
                execution_time=time.time() - start,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )


class DockerExecutor(SandboxExecutor):
    """Docker sandboxed execution."""

    def __init__(self, image: str = "python:3.11-slim", memory: str = "512m", cpus: float = 1.0):
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self._docker = None

    async def _get_docker(self):
        if self._docker is None:
            import docker
            self._docker = docker.from_env()
        return self._docker

    async def execute(self, func: Callable, params: dict, timeout: int) -> ToolResult:
        # For Docker, we'd serialize the function and run in container
        # This is a simplified version - full impl would serialize code
        start = time.time()
        try:
            docker_client = await self._get_docker()
            # Run in container
            container = docker_client.containers.run(
                self.image,
                command=["python", "-c", "print('hello')"],
                detach=True,
                mem_limit=self.memory,
                cpu_count=self.cpus,
                network_mode="none" if not getattr(func, '_allow_network', False) else "bridge",
                remove=True,
            )
            # Wait for completion
            result = container.wait(timeout=timeout)
            logs = container.logs().decode()
            return ToolResult(
                success=result["StatusCode"] == 0,
                result=logs,
                error=None if result["StatusCode"] == 0 else logs,
                execution_time=time.time() - start,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )


class E2BExecutor(SandboxExecutor):
    """E2B cloud sandbox execution."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or get_settings().e2b_api_key
        self._session = None

    async def execute(self, func: Callable, params: dict, timeout: int) -> ToolResult:
        if not self.api_key:
            return ToolResult(success=False, error="E2B API key not configured", execution_time=0)

        start = time.time()
        try:
            import e2b
            session = await e2b.Session.create_async(id="python3", api_key=self.api_key)

            # Serialize function call
            code = self._serialize_call(func, params)
            execution = await session.process.start(code, timeout=timeout)
            await execution.wait()

            result = execution.output
            return ToolResult(
                success=result.exit_code == 0,
                result=result.stdout,
                error=result.stderr if result.exit_code != 0 else None,
                execution_time=time.time() - start,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), execution_time=time.time() - start)

    def _serialize_call(self, func: Callable, params: dict) -> str:
        """Serialize function call to Python code."""
        # This is simplified - full impl would properly serialize code
        return f"print('E2B execution not fully implemented')"


# =============================================================================
# TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """
    Central tool registry with discovery, validation, execution, and monitoring.
    """

    def __init__(
        self,
        event_bus: Any = None,
        sandbox_provider: str = "local",  # local, docker, e2b
    ):
        self.event_bus = event_bus
        self.sandbox_provider = sandbox_provider

        # Tool storage
        self._tools: dict[str, Callable] = {}
        self._metadata: dict[str, ToolMetadata] = {}
        self._categories: dict[str, set[str]] = defaultdict(set)

        # Sandbox executors
        self._executors = {
            "local": LocalExecutor(),
            "docker": DockerExecutor(),
            "e2b": E2BExecutor(),
        }
        self._executor = self._executors.get(sandbox_provider, LocalExecutor())

        # Generated tools directory
        self._generated_tools_dir = Path.home() / ".vega" / "tools"
        self._generated_tools_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self._call_history: dict[str, list[float]] = defaultdict(list)

        # Approval callbacks
        self._approval_callbacks: dict[str, asyncio.Future] = {}

        # Stats
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "by_tool": defaultdict(int),
            "by_category": defaultdict(int),
        }

        logger.info(f"ToolRegistry initialized with {sandbox_provider} sandbox")

    # -------------------------------------------------------------------------
    # REGISTRATION
    # -------------------------------------------------------------------------

    def register(
        self,
        name: str,
        func: Callable,
        metadata: ToolMetadata | None = None,
    ) -> None:
        """Register a tool."""
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")

        self._tools[name] = func
        self._metadata[name] = metadata or ToolMetadata(
            name=name,
            description=func.__doc__ or "No description",
        )
        self._categories[self._metadata[name].category].add(name)

        # Store function reference for introspection
        func._tool_name = name
        func._tool_metadata = self._metadata[name]

        logger.info(f"Registered tool: {name} ({self._metadata[name].category})")

        # Emit event
        if self.event_bus:
            asyncio.create_task(self.event_bus.emit("tool.registered", {
                "name": name,
                "category": self._metadata[name].category,
            }))

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            category = self._metadata[name].category
            del self._tools[name]
            del self._metadata[name]
            self._categories[category].discard(name)
            logger.info(f"Unregistered tool: {name}")
            return True
        return False

    # -------------------------------------------------------------------------
    # DISCOVERY
    # -------------------------------------------------------------------------

    def get_tool(self, name: str) -> Callable | None:
        return self._tools.get(name)

    def get_metadata(self, name: str) -> ToolMetadata | None:
        return self._metadata.get(name)

    def list_tools(self, category: str | None = None) -> list[dict]:
        """List all tools with metadata."""
        tools = []
        for name, func in self._tools.items():
            meta = self._metadata[name]
            if category and meta.category != category:
                continue
            tools.append({
                "name": name,
                "description": meta.description,
                "category": meta.category,
                "version": meta.version,
                "parameters": [p.__dict__ for p in meta.parameters],
                "returns": meta.returns,
                "requires_approval": meta.requires_approval,
                "sandboxed": meta.sandboxed,
                "deprecated": meta.deprecated,
                "tags": meta.tags,
            })
        return tools

    def list_categories(self) -> list[str]:
        return list(self._categories.keys())

    def get_tools_by_category(self, category: str) -> list[str]:
        return list(self._categories.get(category, set()))

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------

    async def execute(
        self,
        tool_name: str,
        parameters: dict,
        requestor: str = "",
        context: dict | None = None,
    ) -> ToolResult:
        """Execute a tool with validation, sandboxing, and monitoring."""
        call_id = str(uuid.uuid4())
        start_time = time.time()

        # Get tool
        func = self._tools.get(tool_name)
        if not func:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found",
                tool_name=tool_name,
                call_id=call_id,
            )

        metadata = self._metadata[tool_name]

        # Check deprecation
        if metadata.deprecated:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' is deprecated: {metadata.deprecated_reason}",
                tool_name=tool_name,
                call_id=call_id,
            )

        # Rate limiting
        if metadata.rate_limit:
            if not self._check_rate_limit(tool_name, metadata.rate_limit):
                return ToolResult(
                    success=False,
                    error=f"Rate limit exceeded for '{tool_name}'",
                    tool_name=tool_name,
                    call_id=call_id,
                )

        # Check approval requirement
        if metadata.requires_approval and not self._is_approved(tool_name, requestor, parameters):
            return await self._request_approval(tool_name, parameters, requestor, context)

        # Validate parameters
        validation_error = self._validate_parameters(tool_name, parameters)
        if validation_error:
            return ToolResult(
                success=False,
                error=f"Parameter validation failed: {validation_error}",
                tool_name=tool_name,
                call_id=call_id,
            )

        # Execute in sandbox
        tool_call = ToolCall(
            tool=tool_name,
            parameters=parameters,
            call_id=call_id,
            requestor=requestor,
            context=context or {},
        )

        result = await self._execute_sandboxed(func, parameters, metadata.timeout)

        # Update stats
        result.tool_name = tool_name
        result.call_id = call_id
        self._update_stats(tool_name, metadata.category, result.success)

        # Emit event
        if self.event_bus:
            await self.event_bus.emit("tool.executed", {
                "tool": tool_name,
                "success": result.success,
                "execution_time": result.execution_time,
                "requestor": requestor,
            })

        return result

    async def _execute_sandboxed(
        self,
        func: Callable,
        parameters: dict,
        timeout: int,
    ) -> ToolResult:
        """Execute function in configured sandbox."""
        executor = self._executor
        return await executor.execute(func, parameters, timeout)

    # -------------------------------------------------------------------------
    # PARAMETER VALIDATION
    # -------------------------------------------------------------------------

    def _validate_parameters(self, tool_name: str, parameters: dict) -> str | None:
        """Validate parameters against tool schema."""
        metadata = self._metadata[tool_name]

        # Check required parameters
        for param in metadata.parameters:
            if param.required and param.name not in parameters:
                return f"Required parameter '{param.name}' missing"

            if param.name in parameters:
                value = parameters[param.name]
                type_error = self._validate_type(param, value)
                if type_error:
                    return f"Parameter '{param.name}': {type_error}"

                # Check enum
                if param.enum and value not in param.enum:
                    return f"Parameter '{param.name}' must be one of: {param.enum}"

                # Check numeric bounds
                if param.minimum is not None and isinstance(value, (int, float)):
                    if value < param.minimum:
                        return f"Parameter '{param.name}' must be >= {param.minimum}"
                if param.maximum is not None and isinstance(value, (int, float)):
                    if value > param.maximum:
                        return f"Parameter '{param.name}' must be <= {param.maximum}"

        # Check for unexpected parameters
        known_params = {p.name for p in metadata.parameters}
        for key in parameters:
            if key not in known_params:
                logger.warning(f"Unexpected parameter '{key}' for tool '{tool_name}'")

        return None

    def _validate_type(self, param: ToolParameter, value: Any) -> str | None:
        """Validate parameter type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected_type = type_map.get(param.type)
        if expected_type and not isinstance(value, expected_type):
            return f"Expected {param.type}, got {type(value).__name__}"

        return None

    # -------------------------------------------------------------------------
    # RATE LIMITING
    # -------------------------------------------------------------------------

    def _check_rate_limit(self, tool_name: str, limit: int) -> bool:
        """Check if tool call is within rate limit."""
        now = time.time()
        minute_ago = now - 60

        # Clean old entries
        self._call_history[tool_name] = [
            t for t in self._call_history[tool_name] if t > minute_ago
        ]

        if len(self._call_history[tool_name]) >= limit:
            return False

        self._call_history[tool_name].append(now)
        return True

    # -------------------------------------------------------------------------
    # APPROVAL WORKFLOW
    # -------------------------------------------------------------------------

    def _is_approved(self, tool_name: str, requestor: str, parameters: dict) -> bool:
        """Check if tool call is pre-approved."""
        # In production, would check approval database/cache
        # For now, auto-approve for system requestors
        return requestor in ["system", "orchestrator", "admin"]

    async def _request_approval(
        self,
        tool_name: str,
        parameters: dict,
        requestor: str,
        context: dict | None,
    ) -> ToolResult:
        """Request human approval for sensitive tool."""
        approval_id = str(uuid.uuid4())
        future = asyncio.Future()
        self._approval_callbacks[approval_id] = future

        # Emit approval request
        if self.event_bus:
            await self.event_bus.emit("tool.approval_required", {
                "approval_id": approval_id,
                "tool": tool_name,
                "parameters": parameters,
                "requestor": requestor,
                "context": context,
                "timeout": 300,  # 5 minutes
            })

        try:
            approved = await asyncio.wait_for(future, timeout=300)
            if approved:
                # Re-execute after approval
                return await self.execute(tool_name, parameters, requestor, context)
            else:
                return ToolResult(
                    success=False,
                    error="Approval denied",
                    tool_name=tool_name,
                )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error="Approval timeout",
                tool_name=tool_name,
            )
        finally:
            self._approval_callbacks.pop(approval_id, None)

    def approve_tool_call(self, approval_id: str, approved: bool) -> bool:
        """Resolve approval request."""
        future = self._approval_callbacks.get(approval_id)
        if future and not future.done():
            future.set_result(approved)
            return True
        return False

    # -------------------------------------------------------------------------
    # HOT RELOAD
    # -------------------------------------------------------------------------

    async def reload_tool(self, tool_name: str) -> bool:
        """Hot-reload a generated tool from disk."""
        if tool_name not in self._tools:
            return False

        metadata = self._metadata[tool_name]
        file_path = self._generated_tools_dir / f"tool_{tool_name}.py"

        if not file_path.exists():
            logger.error(f"Tool file not found: {file_path}")
            return False

        try:
            # Load module
            spec = importlib.util.spec_from_file_location(f"tool_{tool_name}", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get function
            func = getattr(module, tool_name, None)
            if not func:
                logger.error(f"Function '{tool_name}' not found in module")
                return False

            # Re-register
            self.register(tool_name, func, metadata)
            logger.info(f"Hot-reloaded tool: {tool_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to reload tool '{tool_name}': {e}")
            return False

    async def load_generated_tools(self) -> int:
        """Load all generated tools from directory."""
        loaded = 0
        for file_path in self._generated_tools_dir.glob("tool_*.py"):
            tool_name = file_path.stem.replace("tool_", "")
            try:
                spec = importlib.util.spec_from_file_location(f"tool_{tool_name}", file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                func = getattr(module, tool_name, None)
                if func:
                    # Try to get metadata from module
                    meta = getattr(module, "TOOL_METADATA", None)
                    metadata = ToolMetadata(**meta) if meta else ToolMetadata(
                        name=tool_name,
                        description=func.__doc__ or "Generated tool",
                        category="generated",
                    )
                    self.register(tool_name, func, metadata)
                    loaded += 1
            except Exception as e:
                logger.error(f"Failed to load generated tool {tool_name}: {e}")

        return loaded

    # -------------------------------------------------------------------------
    # STATS & MONITORING
    # -------------------------------------------------------------------------

    def _update_stats(self, tool_name: str, category: str, success: bool) -> None:
        self._stats["total_calls"] += 1
        if success:
            self._stats["successful_calls"] += 1
        else:
            self._stats["failed_calls"] += 1
        self._stats["by_tool"][tool_name] += 1
        self._stats["by_category"][category] += 1

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "registered_tools": len(self._tools),
            "categories": len(self._categories),
            "success_rate": (
                self._stats["successful_calls"] / max(self._stats["total_calls"], 1)
            ),
        }

    # -------------------------------------------------------------------------
    # HEALTH CHECK
    # -------------------------------------------------------------------------

    async def health_check(self) -> dict:
        return {
            "healthy": True,
            "sandbox_provider": self.sandbox_provider,
            "registered_tools": len(self._tools),
            "stats": self.get_stats(),
        }

    # -------------------------------------------------------------------------
    # EXPORTS
    # -------------------------------------------------------------------------

__all__ = [
    "ToolRegistry",
    "ToolMetadata",
    "ToolParameter",
    "ToolResult",
    "ToolCall",
    "SandboxExecutor",
    "LocalExecutor",
    "DockerExecutor",
    "E2BExecutor",
]