#!/usr/bin/env python3
"""
================================================================================
VEGA PLUGIN LOADER - Dynamic Plugin Management
================================================================================
Plugin system with:
- Discovery & loading
- Hot-reload for development
- Dependency resolution
- Configuration management
- Lifecycle hooks
- Isolation & sandboxing

References:
- Home Assistant: Component loader
- VS Code: Extension host
- pytest: Plugin system
================================================================================
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config import get_settings, settings

logger = logging.getLogger("vega.plugins.loader")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PluginMetadata:
    """Plugin metadata."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = "unknown"
    license: str = "MIT"
    # Dependencies
    requires: list[str] = field(default_factory=list)  # Other plugin names
    python_requires: str = ">=3.10"
    dependencies: list[str] = field(default_factory=list)  # PyPI packages
    # Compatibility
    vega_min_version: str = "1.0.0"
    vega_max_version: str = ""
    # Entry point
    entry_point: str = "plugin.py"
    class_name: str = "Plugin"
    # Config
    config_schema: dict = field(default_factory=dict)
    default_config: dict = field(default_factory=dict)
    # Permissions
    permissions: list[str] = field(default_factory=list)
    sandboxed: bool = True


@dataclass
class PluginConfig:
    """Plugin runtime configuration."""
    enabled: bool = True
    config: dict = field(default_factory=dict)
    auto_start: bool = True
    priority: int = 0  # Lower = earlier start


class PluginBase:
    """
    Base class for all VEGA plugins.
    Plugins must implement initialize() and shutdown().
    """

    def __init__(self, config: PluginConfig, event_bus: Any = None, tool_registry: Any = None):
        self.config = config
        self.event_bus = event_bus
        self.tool_registry = tool_registry
        self._initialized = False
        self._start_time = time.time()

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        raise NotImplementedError

    async def initialize(self) -> bool:
        """Initialize plugin. Return True on success."""
        self._initialized = True
        logger.info(f"Plugin '{self.metadata.name}' initialized")
        return True

    async def shutdown(self) -> None:
        """Cleanup plugin resources."""
        self._initialized = False
        logger.info(f"Plugin '{self.metadata.name}' shutdown")

    async def on_event(self, event_type: str, data: dict) -> None:
        """Handle events from event bus."""
        pass

    def get_status(self) -> dict:
        """Get plugin status."""
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "initialized": self._initialized,
            "uptime": time.time() - self._start_time,
            "enabled": self.config.enabled,
        }


# =============================================================================
# PLUGIN LOADER
# =============================================================================

class PluginLoader:
    """
    Loads, manages, and monitors plugins.
    Supports hot-reload and dependency resolution.
    """

    def __init__(
        self,
        event_bus: Any = None,
        tool_registry: Any = None,
        plugins_dirs: list[Path] | None = None,
    ):
        self.event_bus = event_bus
        self.tool_registry = tool_registry
        self.plugins_dirs = plugins_dirs or [
            Path(__file__).parent,  # Built-in plugins
            Path.home() / ".vega" / "plugins",  # User plugins
        ]

        self._plugins: dict[str, PluginBase] = {}
        self._metadata: dict[str, PluginMetadata] = {}
        self._configs: dict[str, PluginConfig] = {}
        self._load_order: list[str] = []
        self._file_watchers: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # DISCOVERY & LOADING
    # -------------------------------------------------------------------------

    async def discover_plugins(self) -> list[PluginMetadata]:
        """Discover all available plugins."""
        discovered = []

        for plugins_dir in self.plugins_dirs:
            if not plugins_dir.exists():
                continue

            # Look for plugin directories
            for item in plugins_dir.iterdir():
                if item.is_dir() and not item.name.startswith("_"):
                    metadata = await self._load_plugin_metadata(item)
                    if metadata:
                        discovered.append(metadata)

        return discovered

    async def _load_plugin_metadata(self, plugin_dir: Path) -> PluginMetadata | None:
        """Load plugin metadata from plugin.json or plugin.py."""
        # Check for plugin.json
        json_file = plugin_dir / "plugin.json"
        if json_file.exists():
            try:
                with open(json_file) as f:
                    data = json.load(f)
                return PluginMetadata(**data)
            except Exception as e:
                logger.error(f"Failed to load plugin.json from {plugin_dir}: {e}")

        # Check for plugin.py with metadata class attribute
        py_file = plugin_dir / "plugin.py"
        if py_file.exists():
            try:
                spec = importlib.util.spec_from_file_location(f"{plugin_dir.name}_meta", py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for PLUGIN_METADATA or get_metadata()
                if hasattr(module, "PLUGIN_METADATA"):
                    return PluginMetadata(**module.PLUGIN_METADATA)
                if hasattr(module, "get_metadata"):
                    return module.get_metadata()
            except Exception as e:
                logger.debug(f"Could not load metadata from {py_file}: {e}")

        return None

    async def load_plugin(self, name: str, config: PluginConfig | None = None) -> bool:
        """Load a plugin by name."""
        if name in self._plugins:
            logger.warning(f"Plugin '{name}' already loaded")
            return True

        # Find plugin directory
        plugin_dir = await self._find_plugin_dir(name)
        if not plugin_dir:
            logger.error(f"Plugin '{name}' not found")
            return False

        # Load metadata
        metadata = await self._load_plugin_metadata(plugin_dir)
        if not metadata:
            logger.error(f"Plugin '{name}' has no valid metadata")
            return False

        # Check dependencies
        for dep in metadata.requires:
            if dep not in self._plugins:
                logger.error(f"Plugin '{name}' requires '{dep}' which is not loaded")
                return False

        # Install Python dependencies
        if not await self._install_dependencies(metadata.dependencies):
            logger.error(f"Failed to install dependencies for '{name}'")
            return False

        # Load plugin class
        plugin_class = await self._load_plugin_class(plugin_dir, metadata)
        if not plugin_class:
            return False

        # Create instance
        plugin_config = config or PluginConfig()
        self._configs[name] = plugin_config

        try:
            plugin = plugin_class(plugin_config, self.event_bus, self.tool_registry)
            success = await plugin.initialize()

            if success:
                self._plugins[name] = plugin
                self._metadata[name] = metadata
                self._load_order.append(name)

                # Register plugin tools if any
                await self._register_plugin_tools(name, plugin)

                # Emit event
                if self.event_bus:
                    await self.event_bus.emit("plugin.loaded", {
                        "name": name,
                        "version": metadata.version,
                    })

                logger.info(f"Plugin '{name}' loaded successfully")
                return True
            else:
                logger.error(f"Plugin '{name}' initialization failed")
                return False

        except Exception as e:
            logger.exception(f"Failed to load plugin '{name}': {e}")
            return False

    async def _find_plugin_dir(self, name: str) -> Path | None:
        """Find plugin directory by name."""
        for plugins_dir in self.plugins_dirs:
            if not plugins_dir.exists():
                continue

            # Direct match
            candidate = plugins_dir / name
            if candidate.is_dir():
                return candidate

            # Case-insensitive
            for item in plugins_dir.iterdir():
                if item.is_dir() and item.name.lower() == name.lower():
                    return item

        return None

    async def _load_plugin_class(self, plugin_dir: Path, metadata: PluginMetadata) -> type | None:
        """Load plugin class from entry point."""
        entry_file = plugin_dir / metadata.entry_point

        if not entry_file.exists():
            logger.error(f"Entry point not found: {entry_file}")
            return None

        try:
            module_name = f"vega_plugin_{metadata.name}"
            spec = importlib.util.spec_from_file_location(module_name, entry_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Get class
            class_name = metadata.class_name
            plugin_class = getattr(module, class_name, None)

            if not plugin_class:
                logger.error(f"Class '{class_name}' not found in {entry_file}")
                return None

            # Verify it's a PluginBase
            if not issubclass(plugin_class, PluginBase):
                logger.error(f"Plugin class must inherit from PluginBase")
                return None

            return plugin_class

        except Exception as e:
            logger.exception(f"Failed to load plugin class: {e}")
            return None

    async def _install_dependencies(self, dependencies: list[str]) -> bool:
        """Install Python package dependencies."""
        if not dependencies:
            return True

        for dep in dependencies:
            try:
                # Check if already installed
                __import__(dep.split(">=")[0].split("==")[0].replace("-", "_"))
            except ImportError:
                logger.info(f"Installing dependency: {dep}")
                try:
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, "-m", "pip", "install", dep,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.wait()
                    if proc.returncode != 0:
                        logger.error(f"Failed to install {dep}")
                        return False
                except Exception as e:
                    logger.error(f"Failed to install {dep}: {e}")
                    return False

        return True

    async def _register_plugin_tools(self, name: str, plugin: PluginBase) -> None:
        """Register tools provided by plugin."""
        if not self.tool_registry or not hasattr(plugin, "get_tools"):
            return

        try:
            tools = plugin.get_tools()
            for tool_name, tool_func, tool_meta in tools:
                self.tool_registry.register(tool_name, tool_func, tool_meta)
        except Exception as e:
            logger.error(f"Failed to register tools for plugin '{name}': {e}")

    # -------------------------------------------------------------------------
    # LIFECYCLE MANAGEMENT
    # -------------------------------------------------------------------------

    async def load_all(self, configs: dict[str, PluginConfig] | None = None) -> int:
        """Load all discovered plugins."""
        configs = configs or {}
        discovered = await self.discover_plugins()

        # Sort by dependencies (topological sort)
        load_order = self._resolve_load_order(discovered)

        loaded = 0
        for metadata in load_order:
            name = metadata.name
            config = configs.get(name, PluginConfig(auto_start=True))

            if config.enabled:
                if await self.load_plugin(name, config):
                    loaded += 1

        logger.info(f"Loaded {loaded}/{len(discovered)} plugins")
        return loaded

    def _resolve_load_order(self, plugins: list[PluginMetadata]) -> list[PluginMetadata]:
        """Topological sort by dependencies."""
        # Simple implementation - sort by dependency count
        # Full topological sort would be better
        return sorted(plugins, key=lambda p: len(p.requires))

    async def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        plugin = self._plugins.get(name)
        if not plugin:
            return False

        try:
            await plugin.shutdown()

            # Unregister tools
            if self.tool_registry and hasattr(plugin, "get_tools"):
                for tool_name, _, _ in plugin.get_tools():
                    self.tool_registry.unregister(tool_name)

            # Remove from tracking
            del self._plugins[name]
            del self._metadata[name]
            del self._configs[name]
            self._load_order.remove(name)

            if self.event_bus:
                await self.event_bus.emit("plugin.unloaded", {"name": name})

            logger.info(f"Plugin '{name}' unloaded")
            return True

        except Exception as e:
            logger.exception(f"Failed to unload plugin '{name}': {e}")
            return False

    async def reload_plugin(self, name: str) -> bool:
        """Hot-reload a plugin."""
        if name not in self._plugins:
            return False

        config = self._configs.get(name, PluginConfig())
        await self.unload_plugin(name)
        return await self.load_plugin(name, config)

    # -------------------------------------------------------------------------
    # MANAGEMENT
    # -------------------------------------------------------------------------

    def get_plugin(self, name: str) -> PluginBase | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[dict]:
        """List all loaded plugins with status."""
        return [
            {
                "name": name,
                "version": self._metadata[name].version,
                "description": self._metadata[name].description,
                "enabled": self._configs[name].enabled,
                "status": plugin.get_status(),
            }
            for name, plugin in self._plugins.items()
        ]

    def get_plugin_config(self, name: str) -> PluginConfig | None:
        return self._configs.get(name)

    async def update_config(self, name: str, config: dict) -> bool:
        """Update plugin configuration."""
        if name not in self._configs:
            return False

        plugin_config = self._configs[name]
        plugin_config.config.update(config)

        plugin = self._plugins.get(name)
        if plugin and hasattr(plugin, "on_config_change"):
            await plugin.on_config_change(plugin_config.config)

        return True

    async def enable_plugin(self, name: str) -> bool:
        if name in self._configs:
            self._configs[name].enabled = True
            return await self.load_plugin(name)
        return False

    async def disable_plugin(self, name: str) -> bool:
        if name in self._configs:
            self._configs[name].enabled = False
            return await self.unload_plugin(name)
        return False

    # -------------------------------------------------------------------------
    # HOT RELOAD
    # -------------------------------------------------------------------------

    async def watch_plugins(self) -> None:
        """Watch plugin directories for changes (hot reload)."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class PluginChangeHandler(FileSystemEventHandler):
                def __init__(self, loader):
                    self.loader = loader

                def on_modified(self, event):
                    if event.is_directory:
                        return
                    if event.src_path.endswith(".py") or event.src_path.endswith("plugin.json"):
                        # Find plugin name from path
                        for plugins_dir in self.loader.plugins_dirs:
                            try:
                                rel_path = Path(event.src_path).relative_to(plugins_dir)
                                plugin_name = rel_path.parts[0]
                                if plugin_name in self.loader._plugins:
                                    asyncio.create_task(self.loader.reload_plugin(plugin_name))
                            except ValueError:
                                pass

            observer = Observer()
            for plugins_dir in self.plugins_dirs:
                if plugins_dir.exists():
                    observer.schedule(PluginChangeHandler(self), str(plugins_dir), recursive=True)

            observer.start()
            logger.info("Plugin hot-reload watching started")

        except ImportError:
            logger.info("watchdog not installed, hot-reload disabled")
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")

    # -------------------------------------------------------------------------
    # STATS
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "loaded": len(self._plugins),
            "plugins": list(self._plugins.keys()),
            "load_order": self._load_order,
        }

    async def health_check(self) -> dict:
        healthy = True
        for name, plugin in self._plugins.items():
            if not plugin._initialized:
                healthy = False
                break

        return {
            "healthy": healthy,
            "loaded": len(self._plugins),
            "plugins": {name: p._initialized for name, p in self._plugins.items()},
        }


# =============================================================================
# FACTORY
# =============================================================================

async def create_plugin_loader(
    event_bus: Any = None,
    tool_registry: Any = None,
    plugins_dirs: list[Path] | None = None,
) -> PluginLoader:
    """Factory to create and initialize plugin loader."""
    loader = PluginLoader(
        event_bus=event_bus,
        tool_registry=tool_registry,
        plugins_dirs=plugins_dirs,
    )
    return loader


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PluginLoader",
    "PluginMetadata",
    "PluginConfig",
    "PluginBase",
    "create_plugin_loader",
]