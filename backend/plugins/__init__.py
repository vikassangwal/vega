"""
================================================================================
VEGA PLUGINS - Extensible Plugin System
================================================================================
Plugin architecture for external integrations:
- Plugin loader with hot-reload
- Standard plugin interface
- Built-in plugins: Spotify, Smart Home, Email, Calendar, Notion, Obsidian
- Plugin marketplace support
- Dependency management
- Configuration per plugin

References:
- OpenJarvis/OpenJarvis: Plugin presets
- Home Assistant: Integration patterns
- VS Code: Extension architecture
================================================================================
"""

from .plugin_loader import PluginLoader, PluginMetadata, PluginBase
from .spotify import SpotifyPlugin
from .smart_home import SmartHomePlugin
from .email_plugin import EmailPlugin
from .calendar_plugin import CalendarPlugin
from .notion_plugin import NotionPlugin
from .obsidian_plugin import ObsidianPlugin

__all__ = [
    "PluginLoader",
    "PluginMetadata",
    "PluginBase",
    "SpotifyPlugin",
    "SmartHomePlugin",
    "EmailPlugin",
    "CalendarPlugin",
    "NotionPlugin",
    "ObsidianPlugin",
]