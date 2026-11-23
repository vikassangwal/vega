"""
JARVIS OMEGA - Executor Agent
Specialist agent for code execution, OS control, browser automation, and system operations.
Powered by open-interpreter patterns and browser-use.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from agents.base_agent import BaseAgent, AgentConfig, AgentContext
from tools.tool_registry import ToolRegistry, ToolMetadata, ToolParameter
from tools.builtin.file_tools import _workspace
from config import get_settings, settings

logger = logging.getLogger("jarvis.agents.executor")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of code/command execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time: float = 0.0
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class SandboxConfig:
    """Sandbox configuration."""
    provider: str = "docker"  # docker, e2b, local
    timeout: int = 300
    memory_limit: str = "2g"
    cpu_limit: float = 2.0
    network_enabled: bool = False
    allow_internet: bool = False
    working_dir: str = "/workspace"


# =============================================================================
# EXECUTOR AGENT
# =============================================================================

class ExecutorAgent(BaseAgent):
    """
    Executor Agent - Handles all code execution and system automation.
    The "hands" of VEGA.
    """

    def __init__(
        self,
        event_bus: Any = None,
        memory_manager: Any = None,
        tool_registry: Any = None,
        llm_client: Any = None,
        sandbox_config: SandboxConfig | None = None,
    ):
        agent_config = AgentConfig(
            name="executor",
            role="executor",
            description="Code execution, OS automation, browser control, file operations, system commands",
            model="executor",
            temperature=0.2,  # Low for precise execution
            max_tokens=8192,
            system_prompt=self._get_system_prompt(),
            tools=[
                "execute_python", "execute_javascript", "execute_shell",
                "file_read", "file_write", "file_list", "file_delete",
                "process_list", "process_kill", "app_open", "app_close",
                "browser_navigate", "browser_click", "browser_type", "browser_screenshot",
                "system_info", "volume_control", "screen_capture",
            ],
            can_delegate=False,
            can_handoff=True,
            stream_response=True,
            timeout=120.0,
        )

        super().__init__(
            config=agent_config,
            event_bus=event_bus,
            memory_manager=memory_manager,
            tool_registry=tool_registry,
            llm_client=llm_client,
        )

        self.sandbox_config = sandbox_config or SandboxConfig()
        self._working_dir = Path(self.sandbox_config.working_dir)
        self._working_dir.mkdir(parents=True, exist_ok=True)

        # Process tracking
        self._active_processes: dict[str, subprocess.Popen] = {}

        # Browser
        self._browser: Any = None
        self._page: Any = None

        # Register tools
        if self.tool_registry:
            self._register_tools()

    def _get_system_prompt(self) -> str:
        return """You are the Executor Agent - the hands of VEGA. You execute code, control the OS, automate browsers, and manage files.

Capabilities:
- CODE EXECUTION: Python, JavaScript/Node.js, Bash/Shell, SQL
- FILE OPERATIONS: Read, write, list, delete, search, organize
- PROCESS CONTROL: List processes, kill, start applications
- BROWSER AUTOMATION: Navigate, click, type, scroll, screenshot, extract data
- SYSTEM CONTROL: Volume, brightness, screen capture, system info
- SANDBOXED EXECUTION: Docker/E2B for safe untrusted code

Guidelines:
1. ALWAYS use sandbox for untrusted/user-provided code
2. Prefer Python for data processing, JS for web tasks, Bash for system ops
3. Return clear results: stdout, stderr, return code, created files
3. Handle errors gracefully - explain what failed and why
4. For long-running tasks, use background processes
5. Ask for clarification if task is ambiguous or dangerous

Security:
- No network access in sandbox by default
- No sudo/root privileges
- File operations limited to workspace
- Process isolation enforced

When user asks you to do something, think: what code/commands achieve this? Then execute step by step.""".

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    async def _agent_initialize(self) -> None:
        """Initialize executor components."""
        # Initialize sandbox
        await self._init_sandbox()

        # Initialize browser
        await self._init_browser()

        logger.info("ExecutorAgent initialized")

    async def _agent_shutdown(self) -> None:
        """Cleanup."""
        # Kill active processes
        for proc in self._active_processes.values():
            try:
                proc.terminate()
                await asyncio.wait_for(
                    asyncio.to_thread(proc.wait),
                    timeout=5,
                )
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        # Close browser
        if self._browser:
            await self._browser.close()

        # Close sandbox
        if self._sandbox:
            await self._close_sandbox()

    async def _init_sandbox(self) -> None:
        """Initialize execution sandbox."""
        provider = self.sandbox_config.provider

        if provider == "docker":
            await self._init_docker_sandbox()
        elif provider == "e2b":
            await self._init_e2b_sandbox()
        else:
            # Local execution (less secure)
            logger.warning("Using local execution - not sandboxed!")

    async def _init_docker_sandbox(self) -> None:
        """Initialize Docker sandbox."""
        try:
            import docker
            self._docker_client = docker.from_env()
            # Test connection
            self._docker_client.ping()
            logger.info("Docker sandbox ready")
        except Exception as e:
            logger.warning(f"Docker not available: {e}")

    async def _init_e2b_sandbox(self) -> None:
        """Initialize E2B sandbox."""
        try:
            import e2b
            api_key = get_settings().e2b_api_key
            if api_key:
                self._e2b_session = await e2b.Session.create_async(
                    id="python3",
                    api_key=api_key,
                )
                logger.info("E2B sandbox ready")
        except Exception as e:
            logger.warning(f"E2B not available: {e}")

    async def _close_sandbox(self) -> None:
        if hasattr(self, '_e2b_session') and self._e2b_session:
            await self._e2b_session.close()

    async def _init_browser(self) -> None:
        """Initialize browser automation."""
        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            self._page = await self._browser.new_page()
            logger.info("Browser automation ready")
        except Exception as e:
            logger.warning(f"Browser not available: {e}")

    # -------------------------------------------------------------------------
    # TOOL REGISTRATION
    # -------------------------------------------------------------------------

    def _register_tools(self) -> None:
        """Register all executor tools."""
        tools = {
            # Code execution
            "execute_python": self._tool_execute_python,
            "execute_javascript": self._tool_execute_javascript,
            "execute_shell": self._tool_execute_shell,
            "execute_sql": self._tool_execute_sql,
            # File operations
            "file_read": self._tool_file_read,
            "file_write": self._tool_file_write,
            "file_list": self._tool_file_list,
            "file_delete": self._tool_file_delete,
            "file_search": self._tool_file_search,
            # Process control
            "process_list": self._tool_process_list,
            "process_kill": self._tool_process_kill,
            "app_open": self._tool_app_open,
            "app_close": self._tool_app_close,
            # Browser
            "browser_navigate": self._tool_browser_navigate,
            "browser_click": self._tool_browser_click,
            "browser_type": self._tool_browser_type,
            "browser_screenshot": self._tool_browser_screenshot,
            "browser_extract": self._tool_browser_extract,
            # System
            "system_info": self._tool_system_info,
            "volume_control": self._tool_volume_control,
            "screen_capture": self._tool_screen_capture,
        }

        for name, func in tools.items():
            self.tool_registry.register(name, func)

    # -------------------------------------------------------------------------
    # TASK PROCESSING
    # -------------------------------------------------------------------------

    async def process_task(
        self,
        task: str,
        context: AgentContext,
    ) -> Any:
        """Process execution task - analyze and execute."""
        # For executor, we often directly execute tools based on task
        # The LLM can also be used to generate and execute code for complex tasks

        # Check if task is a direct tool request
        if self._is_direct_tool_request(task):
            return await self._execute_direct_tool(task, context)

        # Otherwise, use LLM to generate and execute plan
        return await self._llm_execute(task, context)

    async def process_task_stream(
        self,
        task: str,
        context: AgentContext,
    ) -> AsyncGenerator[str, None]:
        """Stream execution progress."""
        yield f"⚡ Executor analyzing: {task[:100]}...\n\n"

        if self._is_direct_tool_request(task):
            async for chunk in self._execute_direct_tool_stream(task, context):
                yield chunk
        else:
            async for chunk in self._llm_execute_stream(task, context):
                yield chunk

    def _is_direct_tool_request(self, task: str) -> bool:
        """Check if task maps directly to a tool."""
        tool_keywords = [
            "run python", "execute python", "python script",
            "run shell", "execute command", "terminal",
            "read file", "write file", "list files",
            "open browser", "go to", "navigate to",
            "screenshot", "volume", "process",
        ]
        task_lower = task.lower()
        return any(kw in task_lower for kw in tool_keywords)

    # -------------------------------------------------------------------------
    # LLM-BASED EXECUTION
    # -------------------------------------------------------------------------

    async def _llm_execute(self, task: str, context: AgentContext) -> Any:
        """Use LLM to generate and execute code."""
        if not self.llm_client:
            return {"error": "LLM client not available"}

        prompt = f"""Generate executable code to accomplish this task:

Task: {task}

Available tools: Python, JavaScript, Shell, Browser, File ops, Process control
Working directory: {self._working_dir}

Return ONLY a JSON object with:
{{
  "language": "python|javascript|shell",
  "code": "executable code",
  "explanation": "what this does"
}}"""

        try:
            response = await self._call_llm([
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ])

            # Parse response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                return await self._execute_code(plan["language"], plan["code"], context)

            # Fallback: execute as shell
            return await self._execute_shell(task, context)

        except Exception as e:
            logger.error(f"LLM execution error: {e}")
            return await self._execute_shell(task, context)

    async def _llm_execute_stream(
        self,
        task: str,
        context: AgentContext,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM-based execution."""
        yield "🧠 Generating execution plan...\n\n"

        if not self.llm_client:
            yield "❌ LLM not available\n"
            return

        # Stream LLM response
        full_response = ""
        async for chunk in self._call_llm_stream([
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": f"Generate code for: {task}"},
        ]):
            full_response += chunk
            yield chunk

        # Parse and execute
        yield "\n\n⚡ Executing...\n\n"
        try:
            import re
            json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                async for chunk in self._execute_code_stream(plan["language"], plan["code"], context):
                    yield chunk
            else:
                async for chunk in self._execute_shell_stream(task, context):
                    yield chunk
        except Exception as e:
            yield f"❌ Execution error: {e}\n"

    # -------------------------------------------------------------------------
    # CODE EXECUTION
    # -------------------------------------------------------------------------

    async def _execute_code(
        self,
        language: str,
        code: str,
        context: AgentContext,
    ) -> ExecutionResult:
        """Execute code in appropriate environment."""
        if language == "python":
            return await self._execute_python(code, context)
        elif language in ["javascript", "js", "node"]:
            return await self._execute_javascript(code, context)
        elif language in ["shell", "bash", "sh"]:
            return await self._execute_shell(code, context)
        else:
            return ExecutionResult(
                success=False,
                error=f"Unsupported language: {language}",
            )

    async def _execute_code_stream(
        self,
        language: str,
        code: str,
        context: AgentContext,
    ) -> AsyncGenerator[str, None]:
        """Stream code execution."""
        yield f"```{language}\n{code}\n```\n\n"

        if language == "python":
            async for chunk in self._execute_python_stream(code, context):
                yield chunk
        elif language in ["javascript", "js", "node"]:
            async for chunk in self._execute_javascript_stream(code, context):
                yield chunk
        elif language in ["shell", "bash", "sh"]:
            async for chunk in self._execute_shell_stream(code, context):
                yield chunk

    # Python execution
    async def _execute_python(self, code: str, context: AgentContext) -> ExecutionResult:
        start = time.time()

        # Use sandbox if available
        if hasattr(self, '_e2b_session') and self._e2b_session:
            return await self._execute_python_e2b(code)
        elif hasattr(self, '_docker_client') and self._docker_client:
            return await self._execute_python_docker(code)
        else:
            return await self._execute_python_local(code)

    async def _execute_python_local(self, code: str) -> ExecutionResult:
        start = time.time()

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=self._working_dir) as f:
            f.write(code)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.sandbox_config.timeout,
            )

            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                return_code=proc.returncode,
                execution_time=time.time() - start,
            )

        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                error=f"Timeout after {self.sandbox_config.timeout}s",
                execution_time=time.time() - start,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )
        finally:
            Path(script_path).unlink(missing_ok=True)

    async def _execute_python_stream(self, code: str, context: AgentContext) -> AsyncGenerator[str, None]:
        """Stream Python execution."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=self._working_dir) as f:
            f.write(code)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-u", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
            )

            # Stream stdout
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                yield line.decode()

            # Stream stderr
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                yield f"[stderr] {line.decode()}"

            await proc.wait()

        except Exception as e:
            yield f"❌ Error: {e}\n"
        finally:
            Path(script_path).unlink(missing_ok=True)

    # JavaScript execution
    async def _execute_javascript(self, code: str, context: AgentContext) -> ExecutionResult:
        start = time.time()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, dir=self._working_dir) as f:
            f.write(code)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "node", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.sandbox_config.timeout,
            )

            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                return_code=proc.returncode,
                execution_time=time.time() - start,
            )

        except Exception as e:
            return ExecutionResult(success=False, error=str(e), execution_time=time.time() - start)
        finally:
            Path(script_path).unlink(missing_ok=True)

    async def _execute_javascript_stream(self, code: str, context: AgentContext) -> AsyncGenerator[str, None]:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, dir=self._working_dir) as f:
            f.write(code)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "node", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
            )

            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                yield line.decode()

            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                yield f"[stderr] {line.decode()}"

            await proc.wait()

        finally:
            Path(script_path).unlink(missing_ok=True)

    # Shell execution
    async def _execute_shell(self, command: str, context: AgentContext) -> ExecutionResult:
        start = time.time()

        try:
            # Parse command safely
            if isinstance(command, str):
                args = shlex.split(command)
            else:
                args = command

            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.sandbox_config.timeout,
            )

            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                return_code=proc.returncode,
                execution_time=time.time() - start,
            )

        except asyncio.TimeoutError:
            return ExecutionResult(success=False, error="Timeout", execution_time=time.time() - start)
        except Exception as e:
            return ExecutionResult(success=False, error=str(e), execution_time=time.time() - start)

    async def _execute_shell_stream(self, command: str, context: AgentContext) -> AsyncGenerator[str, None]:
        try:
            args = shlex.split(command) if isinstance(command, str) else command

            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
            )

            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                yield line.decode()

            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                yield f"[stderr] {line.decode()}"

            await proc.wait()

        except Exception as e:
            yield f"❌ Error: {e}\n"

    # -------------------------------------------------------------------------
    # TOOL IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    # Code execution tools
    async def _tool_execute_python(self, params: dict) -> dict:
        code = params.get("code", "")
        result = await self._execute_python(code, AgentContext(conversation_id="tool"))
        return result.__dict__

    async def _tool_execute_javascript(self, params: dict) -> dict:
        code = params.get("code", "")
        result = await self._execute_javascript(code, AgentContext(conversation_id="tool"))
        return result.__dict__

    async def _tool_execute_shell(self, params: dict) -> dict:
        command = params.get("command", "")
        result = await self._execute_shell(command, AgentContext(conversation_id="tool"))
        return result.__dict__

    async def _tool_execute_sql(self, params: dict) -> dict:
        query = params.get("query", "")
        db_path = params.get("database", str(self._working_dir / "vega.db"))

        import sqlite3
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            if query.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                cols = [d[0] for d in cursor.description] if cursor.description else []
                return {"success": True, "columns": cols, "rows": rows}
            else:
                conn.commit()
                return {"success": True, "rows_affected": cursor.rowcount}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    # File operations
    async def _tool_file_read(self, params: dict) -> dict:
        path = params.get("path", "")
        encoding = params.get("encoding", "utf-8")
        max_size = params.get("max_size", 10 * 1024 * 1024)

        try:
            full_path = _workspace.resolve_path(path)
            if not full_path.exists():
                return {"success": False, "error": "File not found"}

            if full_path.stat().st_size > max_size:
                return {"success": False, "error": f"File too large (max {max_size} bytes)"}

            if encoding == "binary":
                content = full_path.read_bytes()
            else:
                content = full_path.read_text(encoding=encoding)

            return {
                "success": True,
                "content": content,
                "size": full_path.stat().st_size,
                "encoding": encoding,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_file_write(self, params: dict) -> dict:
        path = params.get("path", "")
        content = params.get("content", "")
        encoding = params.get("encoding", "utf-8")
        mode = params.get("mode", "w")

        try:
            full_path = _workspace.resolve_path(path)
            full_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(content, str):
                if encoding == "binary":
                    import base64
                    data = base64.b64decode(content)
                    full_path.write_bytes(data)
                else:
                    full_path.write_text(content, encoding=encoding)
            else:
                return {"success": False, "error": "Content must be string"}

            return {"success": True, "path": str(full_path), "size": full_path.stat().st_size}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_file_list(self, params: dict) -> dict:
        path = params.get("path", ".")
        pattern = params.get("pattern", "*")
        recursive = params.get("recursive", False)
        include_hidden = params.get("include_hidden", False)

        try:
            full_path = _workspace.resolve_path(path)
            if not full_path.exists() or not full_path.is_dir():
                return {"success": False, "error": "Directory not found"}

            if recursive:
                files = full_path.rglob(pattern)
            else:
                files = full_path.glob(pattern)

            result = []
            for f in files:
                if not include_hidden and f.name.startswith("."):
                    continue

                stat = f.stat()
                result.append({
                    "name": f.name,
                    "path": str(f.relative_to(_workspace.workspace_root)),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "is_dir": f.is_dir(),
                    "extension": f.suffix,
                })

            return {"success": True, "files": result, "count": len(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_file_delete(self, params: dict) -> dict:
        path = params.get("path", "")
        recursive = params.get("recursive", False)

        try:
            full_path = _workspace.resolve_path(params["path"])
            if full_path.exists():
                if full_path.is_dir():
                    if recursive:
                        import shutil
                        shutil.rmtree(full_path)
                    else:
                        full_path.rmdir()
                else:
                    full_path.unlink()
                return {"success": True, "path": str(full_path)}
            return {"success": False, "error": "Not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_file_search(self, params: dict) -> dict:
        pattern = params.get("pattern", "*")
        content = params.get("content", "")
        path = params.get("path", ".")
        file_types = params.get("file_types")
        max_results = params.get("max_results", 50)

        try:
            full_path = _workspace.resolve_path(path)
            matches = []

            for f in full_path.rglob("*"):
                if not f.is_file():
                    continue
                if file_types and f.suffix not in file_types:
                    continue
                if content:
                    try:
                        if content not in f.read_text(encoding="utf-8", errors="ignore"):
                            continue
                    except Exception:
                        continue
                matches.append(str(f.relative_to(_workspace.workspace_root)))
                if len(matches) >= max_results:
                    break

            return {"success": True, "matches": matches, "count": len(matches)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_file_copy(self, params: dict) -> dict:
        src = _workspace.resolve_path(params["src"])
        dst = _workspace.resolve_path(params["dst"])
        overwrite = params.get("overwrite", False)

        if not src.exists():
            return {"success": False, "error": "Source not found"}

        if dst.exists() and not overwrite:
            return {"success": False, "error": "Destination exists (use overwrite=true)"}

        _workspace.ensure_parent(dst)

        try:
            if src.is_dir():
                import shutil
                shutil.copytree(src, dst, dirs_exist_ok=overwrite)
            else:
                import shutil
                shutil.copy2(src, dst)
            return {"success": True, "src": str(src), "dst": str(dst)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_file_move(self, params: dict) -> dict:
        src = _workspace.resolve_path(params["src"])
        dst = _workspace.resolve_path(params["dst"])
        overwrite = params.get("overwrite", False)

        if not src.exists():
            return {"success": False, "error": "Source not found"}

        if dst.exists() and not overwrite:
            return {"success": False, "error": "Destination exists (use overwrite=true)"}

        _workspace.ensure_parent(dst)

        try:
            import shutil
            shutil.move(str(src), str(dst))
            return {"success": True, "src": str(src), "dst": str(dst)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_file_info(self, params: dict) -> dict:
        path = params.get("path", "")

        try:
            full_path = _workspace.resolve_path(path)
            if not full_path.exists():
                return {"success": False, "error": "Path not found"}

            stat = full_path.stat()
            return {
                "success": True,
                "path": str(full_path.relative_to(_workspace.workspace_root)),
                "name": full_path.name,
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "is_dir": full_path.is_dir(),
                "is_file": full_path.is_file(),
                "extension": full_path.suffix,
                "permissions": oct(stat.st_mode)[-3:],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_file_archive(self, params: dict) -> dict:
        paths = [_workspace.resolve_path(p) for p in params["paths"]]
        output = _workspace.resolve_path(params["output"])
        format = params.get("format", "zip")

        _workspace.ensure_parent(output)

        try:
            if format == "zip":
                import zipfile
                with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in paths:
                        if p.is_dir():
                            for f in p.rglob("*"):
                                if f.is_file():
                                    zf.write(f, f.relative_to(_workspace.workspace_root))
                        else:
                            zf.write(p, p.relative_to(_workspace.workspace_root))
            elif format in ["tar.gz", "tgz"]:
                import tarfile
                with tarfile.open(output, "w:gz") as tf:
                    for p in paths:
                        tf.add(p, arcname=str(p.relative_to(_workspace.workspace_root)))
            else:
                return {"success": False, "error": f"Unsupported format: {format}"}

            return {"success": True, "output": str(output), "size": output.stat().st_size}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_file_extract(self, params: dict) -> dict:
        archive = _workspace.resolve_path(params["archive"])
        output = _workspace.resolve_path(params["output"])
        format = params.get("format")

        if not archive.exists():
            return {"success": False, "error": "Archive not found"}

        _workspace.ensure_parent(output)

        try:
            if format == "zip" or (not format and archive.suffix == ".zip"):
                import zipfile
                with zipfile.ZipFile(archive, "r") as zf:
                    zf.extractall(output)
            elif format in ["tar.gz", "tgz"] or (not format and archive.suffix in [".tar.gz", ".tgz"]):
                import tarfile
                with tarfile.open(archive, "r:gz") as tf:
                    tf.extractall(output)
            else:
                return {"success": False, "error": f"Unsupported format: {format}"}

            return {"success": True, "output": str(output)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_file_temp(self, params: dict) -> dict:
        prefix = params.get("prefix", "vega_")
        suffix = params.get("suffix", "")
        directory = params.get("directory", False)

        try:
            if directory:
                path = Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix))
            else:
                fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
                os.close(fd)
                path = Path(tmp_path)

            return {"success": True, "path": str(path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Process control
    async def _tool_process_list(self, params: dict) -> dict:
        import psutil

        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'status', 'create_time', 'cmdline']):
                try:
                    info = proc.info
                    if params.get("filter", "").lower() not in (info['name'] or '').lower():
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

            sort_by = params.get("sort_by", "cpu")
            limit = params.get("limit", 50)
            sort_key = {"cpu": "cpu_percent", "memory": "memory_percent", "name": "name", "pid": "pid"}.get(sort_by, "cpu_percent")
            processes.sort(key=lambda p: p.get(sort_key, 0), reverse=True)

            return {"success": True, "processes": processes[:limit], "total": len(processes)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_process_kill(self, params: dict) -> dict:
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
            proc.wait(timeout=5)
            return {"success": True, "message": "Process terminated"}
        except psutil.NoSuchProcess:
            return {"success": False, "error": "Process not found"}
        except psutil.AccessDenied:
            return {"success": False, "error": "Permission denied"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_app_open(self, params: dict) -> dict:
        app = params.get("app", "")
        args = params.get("args", [])

        if not app:
            return {"success": False, "error": "Application name required"}

        try:
            import platform
            system = platform.system().lower()

            if system == "darwin":
                cmd = ["open", "-a", app] + args
            elif system == "windows":
                cmd = ["start", "", app] + args
            else:
                cmd = [app] + args

            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            return {"success": True, "pid": proc.pid, "message": f"Started {app}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_app_close(self, params: dict) -> dict:
        app = params.get("app", "")

        if not app:
            return {"success": False, "error": "Application name required"}

        try:
            import psutil
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

    # Browser automation
    async def _tool_browser_navigate(self, params: dict) -> dict:
        url = params.get("url", "")
        page_id = params.get("page_id", "default")
        wait_until = params.get("wait_until", "networkidle")
        timeout = params.get("timeout", 30000)

        if not url:
            return {"success": False, "error": "URL required"}

        if not self._page:
            return {"success": False, "error": "Browser not initialized"}

        try:
            response = await self._page.goto(url, wait_until=wait_until, timeout=timeout)
            return {
                "success": True,
                "url": self._page.url,
                "title": await self._page.title(),
                "status": response.status if response else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_browser_click(self, params: dict) -> dict:
        selector = params.get("selector", "")
        page_id = params.get("page_id", "default")
        button = params.get("button", "left")
        click_count = params.get("click_count", 1)
        delay = params.get("delay", 0)
        force = params.get("force", False)
        timeout = params.get("timeout", 30000)

        if not selector:
            return {"success": False, "error": "Selector required"}

        page = self._browser_manager.get_page(page_id) if hasattr(self, '_browser_manager') else None
        if not page:
            return {"success": False, "error": "Page not found"}

        try:
            await page.click(selector, button=button, click_count=click_count,
                           delay=delay, force=force, timeout=timeout)
            return {"success": True, "message": f"Clicked {selector}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_browser_type(self, params: dict) -> dict:
        selector = params.get("selector", "")
        text = params.get("text", "")
        page_id = params.get("page_id", "default")
        delay = params.get("delay", 50)
        clear = params.get("clear", True)
        press_enter = params.get("press_enter", False)
        timeout = params.get("timeout", 30000)

        if not selector:
            return {"success": False, "error": "Selector required"}

        page = self._browser_manager.get_page(page_id) if hasattr(self, '_browser_manager') else None
        if not page:
            return {"success": False, "error": "Page not found"}

        try:
            if clear:
                await page.fill(selector, "", timeout=timeout)

            await page.type(selector, text, delay=delay, timeout=timeout)

            if press_enter:
                await page.press(selector, "Enter", timeout=timeout)

            return {"success": True, "message": f"Typed into {selector}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_browser_screenshot(self, params: dict) -> dict:
        output = params.get("output", "screenshot.png")
        page_id = params.get("page_id", "default")
        full_page = params.get("full_page", True)
        selector = params.get("selector")
        img_type = params.get("type", "png")
        quality = params.get("quality", 90)

        page = self._browser_manager.get_page(page_id) if hasattr(self, '_browser_manager') else None
        if not page:
            return {"success": False, "error": "Page not found"}

        try:
            options = {"path": output, "full_page": full_page, "type": img_type}
            if img_type == "jpeg":
                options["quality"] = quality

            if selector:
                element = await page.query_selector(selector)
                if not element:
                    return {"success": False, "error": "Element not found"}
                await element.screenshot(**options)
            else:
                await page.screenshot(**options)

            return {"success": True, "path": output}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_browser_extract(self, params: dict) -> dict:
        selector = params.get("selector", "")
        attribute = params.get("attribute", "text")
        multiple = params.get("multiple", False)
        page_id = params.get("page_id", "default")
        timeout = params.get("timeout", 30000)

        if not selector:
            return {"success": False, "error": "Selector required"}

        page = self._browser_manager.get_page(page_id) if hasattr(self, '_browser_manager') else None
        if not page:
            return {"success": False, "error": "Page not found"}

        try:
            if multiple:
                elements = await page.query_selector_all(selector)
                results = []
                for el in elements:
                    if attribute == "text":
                        val = await el.text_content()
                    elif attribute == "html":
                        val = await el.inner_html()
                    else:
                        val = await el.get_attribute(attribute)
                    results.append(val)
                return {"success": True, "data": results, "count": len(results)}
            else:
                element = await page.query_selector(selector)
                if not element:
                    return {"success": False, "error": "Element not found"}

                if attribute == "text":
                    val = await element.text_content()
                elif attribute == "html":
                    val = await element.inner_html()
                else:
                    val = await element.get_attribute(attribute)

                return {"success": True, "data": val}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_browser_wait(self, params: dict) -> dict:
        condition = params.get("condition", "load")
        selector = params.get("selector")
        timeout = params.get("timeout", 30000)
        page_id = params.get("page_id", "default")

        page = self._browser_manager.get_page(page_id) if hasattr(self, '_browser_manager') else None
        if not page:
            return {"success": False, "error": "Page not found"}

        try:
            if condition == "selector_visible":
                await page.wait_for_selector(selector, state="visible", timeout=timeout)
            elif condition == "selector_hidden":
                await page.wait_for_selector(selector, state="hidden", timeout=timeout)
            elif condition == "selector_attached":
                await page.wait_for_selector(selector, state="attached", timeout=timeout)
            elif condition == "navigation":
                await page.wait_for_load_state("networkidle", timeout=timeout)
            elif condition == "load":
                await page.wait_for_load_state("load", timeout=timeout)
            elif condition == "domcontentloaded":
                await page.wait_for_load_state("domcontentloaded", timeout=timeout)
            elif condition == "networkidle":
                await page.wait_for_load_state("networkidle", timeout=timeout)
            else:
                return {"success": False, "error": f"Unknown condition: {condition}"}

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_browser_evaluate(self, params: dict) -> dict:
        script = params.get("script", "")
        args = params.get("args", [])
        page_id = params.get("page_id", "default")

        if not script:
            return {"success": False, "error": "Script required"}

        page = self._browser_manager.get_page(page_id) if hasattr(self, '_browser_manager') else None
        if not page:
            return {"success": False, "error": "Page not found"}

        try:
            result = await page.evaluate(script, *args)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_browser_new_tab(self, params: dict) -> dict:
        context_id = params.get("context_id", "default")
        page_id = params.get("page_id")
        url = params.get("url")

        try:
            pid = await self._browser_manager.new_page(context_id, page_id)
            page = self._browser_manager.get_page(pid)

            if url:
                await page.goto(url, wait_until="networkidle")

            return {"success": True, "page_id": pid, "url": page.url}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_browser_close_tab(self, params: dict) -> dict:
        page_id = params.get("page_id", "default")

        success = await self._browser_manager.close_page(page_id)
        return {"success": success, "message": "Page closed"}

    async def _tool_browser_list_tabs(self, params: dict) -> dict:
        pages = []
        for pid, page in self._browser_manager._pages.items():
            try:
                pages.append({
                    "page_id": pid,
                    "url": page.url,
                    "title": await page.title(),
                })
            except Exception:
                pages.append({"page_id": pid, "url": "unknown", "title": "unknown"})

        return {"success": True, "pages": pages, "count": len(pages)}

    async def _tool_browser_download(self, params: dict) -> dict:
        selector = params.get("selector", "")
        page_id = params.get("page_id", "default")
        save_as = params.get("save_as")
        timeout = params.get("timeout", 30000)

        page = self._browser_manager.get_page(page_id) if hasattr(self, '_browser_manager') else None
        if not page:
            return {"success": False, "error": "Page not found"}

        try:
            async with page.expect_download(timeout=timeout) as download_info:
                await page.click(selector)

            download = await download_info.value

            if save_as:
                await download.save_as(save_as)
                path = save_as
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=download.suggested_filename) as f:
                    path = f.name
                await download.save_as(path)

            return {"success": True, "path": path, "filename": download.suggested_filename}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # System tools
    async def _tool_system_info(self, params: dict) -> dict:
        import psutil
        import platform

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

            if params.get("detailed", True):
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
                    {
                        "device": p.device,
                        "mountpoint": p.mountpoint,
                        "fstype": p.fstype,
                        "opts": p.opts,
                    }
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

    async def _tool_volume_control(self, params: dict) -> dict:
        action = params.get("action", "get")
        value = params.get("value", 50)
        step = params.get("step", 10)

        try:
            import platform
            system = platform.system().lower()

            if system == "darwin":
                return await self._volume_macos(action, value, step)
            elif system == "windows":
                return await self._volume_windows(action, value, step)
            elif system == "linux":
                return await self._volume_linux(action, value, step)
            else:
                return {"success": False, "error": f"Unsupported platform: {system}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _volume_macos(self, action: str, value: int, step: int) -> dict:
        import subprocess

        if action == "get":
            result = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True, text=True
            )
            return {"success": True, "volume": int(result.stdout.strip())}

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

        return {"success": False, "error": f"Unknown action: {action}"}

    async def _volume_windows(self, action: str, value: int, step: int) -> dict:
        return {"success": False, "error": "Windows volume control requires pycaw/nircmd"}

    async def _volume_linux(self, action: str, value: int, step: int) -> dict:
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

    async def _tool_screen_capture(self, params: dict) -> dict:
        output = params.get("output", "screenshot.png")
        region = params.get("region")
        format = params.get("format", "png")

        output_path = _workspace.resolve_path(output)
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

    # -------------------------------------------------------------------------
    # EXPORTS
    # -------------------------------------------------------------------------

__all__ = [
    "ExecutorAgent",
    "ExecutionResult",
    "SandboxConfig",
]