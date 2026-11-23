"""
================================================================================
VEGA TOOLS - Complete Tool Ecosystem
================================================================================
Unified tool system with:
- ToolRegistry: Central registration, discovery, execution
- Built-in Tools: File, Web, System, Browser, Media, Communication
- Generated Tools: Toolsmith-created tools (auto-loaded, hot-reloadable)
- Plugins: Spotify, Smart Home, Email, Calendar, Notion, Obsidian
- Sandboxing: Local, Docker, E2B execution environments
- Approval Workflow: Human-in-the-loop for sensitive operations

References:
- open-interpreter/open-interpreter: Tool calling patterns
- langchain-ai/langgraph: Tool calling with validation
- mcp (Model Context Protocol): Standardized tool interface
- agentskills.io: Open standard for agent skills
================================================================================
"""

from .tool_registry import (
    ToolRegistry,
    ToolMetadata,
    ToolParameter,
    ToolResult,
    ToolCall,
    SandboxExecutor,
    LocalExecutor,
    DockerExecutor,
    E2BExecutor,
)
from .builtin import (
    register_all_builtin_tools,
    register_categories,
    BUILTIN_CATEGORIES,
)

__all__ = [
    # Core
    "ToolRegistry",
    "ToolMetadata",
    "ToolParameter",
    "ToolResult",
    "ToolCall",
    "SandboxExecutor",
    "LocalExecutor",
    "DockerExecutor",
    "E2BExecutor",
    # Built-in
    "register_all_builtin_tools",
    "register_categories",
    "BUILTIN_CATEGORIES",
]