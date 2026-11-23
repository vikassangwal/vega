"""
================================================================================
VEGA BUILT-IN TOOLS - Tool Package
================================================================================
All built-in tool categories:
- file_tools: File system operations
- web_tools: Web search, fetch, download
- system_tools: System control & monitoring
- browser_tools: Browser automation (Playwright)
- media_tools: Media playback control
- communication_tools: Email, notifications, calendar
================================================================================
"""

from .file_tools import register_file_tools
from .web_tools import register_web_tools
from .system_tools import register_system_tools
from .browser_tools import register_browser_tools
from .media_tools import register_media_tools
from .communication_tools import register_communication_tools

# Category registry
BUILTIN_CATEGORIES = {
    "file": register_file_tools,
    "web": register_web_tools,
    "system": register_system_tools,
    "browser": register_browser_tools,
    "media": register_media_tools,
    "communication": register_communication_tools,
}


def register_all_builtin_tools(registry) -> int:
    """Register all built-in tool categories."""
    total = 0
    for category, register_func in BUILTIN_CATEGORIES.items():
        try:
            register_func(registry)
            total += 1
            print(f"✓ Registered {category} tools")
        except Exception as e:
            print(f"✗ Failed to register {category} tools: {e}")
    return total


def register_categories(registry, categories: list[str]) -> int:
    """Register specific categories."""
    total = 0
    for category in categories:
        if category in BUILTIN_CATEGORIES:
            try:
                BUILTIN_CATEGORIES[category](registry)
                total += 1
            except Exception as e:
                print(f"✗ Failed to register {category} tools: {e}")
        else:
            print(f"✗ Unknown category: {category}")
    return total


__all__ = [
    "register_all_builtin_tools",
    "register_categories",
    "BUILTIN_CATEGORIES",
    "register_file_tools",
    "register_web_tools",
    "register_system_tools",
    "register_browser_tools",
    "register_media_tools",
    "register_communication_tools",
]