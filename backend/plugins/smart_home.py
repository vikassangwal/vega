#!/usr/bin/env python3
"""
================================================================================
VEGA SMART HOME PLUGIN - Home Assistant Integration
================================================================================
Smart home control via Home Assistant:
- Light control (on/off, brightness, color)
- Climate control (thermostat)
- Switch/outlet control
- Cover/blind control
- Sensor reading
- Automation triggering
- Scene activation

References:
- home-assistant/core: Home Assistant API
- OpenJarvis/OpenJarvis: Smart home plugin pattern
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from plugins.plugin_loader import PluginBase, PluginConfig, PluginMetadata
from tools.tool_registry import ToolMetadata, ToolParameter

logger = logging.getLogger("vega.plugins.smart_home")


# =============================================================================
# PLUGIN METADATA
# =============================================================================

PLUGIN_METADATA = {
    "name": "smart_home",
    "version": "1.0.0",
    "description": "Home Assistant smart home integration",
    "author": "VEGA Team",
    "license": "MIT",
    "requires": [],
    "dependencies": ["aiohttp"],
    "entry_point": "smart_home.py",
    "class_name": "SmartHomePlugin",
    "config_schema": {
        "home_assistant_url": {"type": "string", "description": "Home Assistant URL", "default": "http://homeassistant.local:8123"},
        "access_token": {"type": "string", "description": "Long-lived access token"},
        "verify_ssl": {"type": "boolean", "default": True},
    },
    "default_config": {
        "enabled": False,
    },
    "permissions": ["home_assistant.control", "home_assistant.read"],
    "sandboxed": True,
}


# =============================================================================
# SMART HOME PLUGIN
# =============================================================================

class SmartHomePlugin(PluginBase):
    """Home Assistant integration plugin."""

    def __init__(self, config: PluginConfig, event_bus: Any = None, tool_registry: Any = None):
        super().__init__(config, event_bus, tool_registry)
        self._ha_url = config.config.get("home_assistant_url", "http://homeassistant.local:8123").rstrip("/")
        self._token = config.config.get("access_token")
        self._verify_ssl = config.config.get("verify_ssl", True)
        self._session: aiohttp.ClientSession | None = None

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(**PLUGIN_METADATA)

    async def initialize(self) -> bool:
        if not self._token:
            logger.warning("Home Assistant token not configured")
            return True

        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
            )

            # Test connection
            async with self._session.get(f"{self._ha_url}/api/") as resp:
                if resp.status != 200:
                    logger.error("Home Assistant connection failed")
                    return False

            logger.info("SmartHomePlugin initialized")
            return True

        except Exception as e:
            logger.error(f"SmartHomePlugin init error: {e}")
            return False

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()
        await super().shutdown()

    async def get_tools(self) -> list:
        return [
            ("ha_call_service", self._tool_call_service, ToolMetadata(
                name="ha_call_service",
                description="Call Home Assistant service",
                category="smart_home",
                parameters=[
                    ToolParameter("domain", "string", "Service domain (light, switch, climate, etc.)", required=True),
                    ToolParameter("service", "string", "Service name (turn_on, turn_off, set_temperature, etc.)", required=True),
                    ToolParameter("entity_id", "string", "Entity ID"),
                    ToolParameter("data", "object", "Service data"),
                ],
                requires_approval=True,
            )),
            ("ha_get_state", self._tool_get_state, ToolMetadata(
                name="ha_get_state",
                description="Get entity state",
                category="smart_home",
                parameters=[
                    ToolParameter("entity_id", "string", "Entity ID", required=True),
                ],
                requires_approval=False,
            )),
            ("ha_light_control", self._tool_light_control, ToolMetadata(
                name="ha_light_control",
                description="Control lights",
                category="smart_home",
                parameters=[
                    ToolParameter("entity_id", "string", "Light entity ID", required=True),
                    ToolParameter("action", "string", "Action: on, off, toggle, brightness, color", required=True),
                    ToolParameter("brightness", "integer", "Brightness 0-255"),
                    ToolParameter("color_temp", "integer", "Color temperature (mireds)"),
                    ToolParameter("rgb_color", "array", "RGB color [r, g, b]"),
                ],
                requires_approval=False,
            )),
            ("ha_climate_control", self._tool_climate_control, ToolMetadata(
                name="ha_climate_control",
                description="Control climate/thermostat",
                category="smart_home",
                parameters=[
                    ToolParameter("entity_id", "string", "Climate entity ID", required=True),
                    ToolParameter("action", "string", "Action: set_temperature, set_hvac_mode, set_fan_mode", required=True),
                    ToolParameter("temperature", "number", "Target temperature"),
                    ToolParameter("hvac_mode", "string", "Mode: heat, cool, auto, off"),
                    ToolParameter("fan_mode", "string", "Fan mode: auto, low, medium, high"),
                ],
                requires_approval=True,
            )),
            ("ha_switch_control", self._tool_switch_control, ToolMetadata(
                name="ha_switch_control",
                description="Control switches/outlets",
                category="smart_home",
                parameters=[
                    ToolParameter("entity_id", "string", "Switch entity ID", required=True),
                    ToolParameter("action", "string", "Action: on, off, toggle", required=True),
                ],
                requires_approval=False,
            )),
            ("ha_cover_control", self._tool_cover_control, ToolMetadata(
                name="ha_cover_control",
                description="Control covers/blinds",
                category="smart_home",
                parameters=[
                    ToolParameter("entity_id", "string", "Cover entity ID", required=True),
                    ToolParameter("action", "string", "Action: open, close, stop, set_position", required=True),
                    ToolParameter("position", "integer", "Position 0-100"),
                ],
                requires_approval=False,
            )),
            ("ha_scene_activate", self._tool_scene_activate, ToolMetadata(
                name="ha_scene_activate",
                description="Activate scene",
                category="smart_home",
                parameters=[
                    ToolParameter("entity_id", "string", "Scene entity ID", required=True),
                ],
                requires_approval=False,
            )),
            ("ha_get_entities", self._tool_get_entities, ToolMetadata(
                name="ha_get_entities",
                description="List entities by domain",
                category="smart_home",
                parameters=[
                    ToolParameter("domain", "string", "Domain filter (light, switch, sensor, etc.)"),
                ],
                requires_approval=False,
            )),
        ]

    # -------------------------------------------------------------------------
    # TOOL IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    async def _tool_call_service(self, params: dict) -> dict:
        domain = params.get("domain")
        service = params.get("service")
        entity_id = params.get("entity_id")
        data = params.get("data", {})

        if not domain or not service:
            return {"success": False, "error": "domain and service required"}

        if entity_id:
            data["entity_id"] = entity_id

        try:
            async with self._session.post(
                f"{self._ha_url}/api/services/{domain}/{service}",
                json=data,
            ) as resp:
                return {"success": resp.status == 200, "status": resp.status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_get_state(self, params: dict) -> dict:
        entity_id = params.get("entity_id")
        if not entity_id:
            return {"success": False, "error": "entity_id required"}

        try:
            async with self._session.get(
                f"{self._ha_url}/api/states/{entity_id}",
            ) as resp:
                if resp.status == 404:
                    return {"success": False, "error": "Entity not found"}
                data = await resp.json()
                return {
                    "success": True,
                    "entity_id": data["entity_id"],
                    "state": data["state"],
                    "attributes": data["attributes"],
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_light_control(self, params: dict) -> dict:
        entity_id = params.get("entity_id")
        action = params.get("action")
        brightness = params.get("brightness")
        color_temp = params.get("color_temp")
        rgb_color = params.get("rgb_color")

        if not entity_id or not action:
            return {"success": False, "error": "entity_id and action required"}

        data = {"entity_id": entity_id}

        if action == "on":
            service = "turn_on"
        elif action == "off":
            service = "turn_off"
        elif action == "toggle":
            service = "toggle"
        elif action == "brightness":
            service = "turn_on"
            if brightness is not None:
                data["brightness"] = brightness
        elif action == "color":
            service = "turn_on"
            if color_temp:
                data["color_temp"] = color_temp
            if rgb_color:
                data["rgb_color"] = rgb_color
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

        try:
            async with self._session.post(
                f"{self._ha_url}/api/services/light/{service}",
                json=data,
            ) as resp:
                return {"success": resp.status == 200, "status": resp.status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_climate_control(self, params: dict) -> dict:
        entity_id = params.get("entity_id")
        action = params.get("action")
        temperature = params.get("temperature")
        hvac_mode = params.get("hvac_mode")
        fan_mode = params.get("fan_mode")

        if not entity_id or not action:
            return {"success": False, "error": "entity_id and action required"}

        data = {"entity_id": entity_id}

        if action == "set_temperature":
            service = "set_temperature"
            if temperature is not None:
                data["temperature"] = temperature
        elif action == "set_hvac_mode":
            service = "set_hvac_mode"
            if hvac_mode:
                data["hvac_mode"] = hvac_mode
        elif action == "set_fan_mode":
            service = "set_fan_mode"
            if fan_mode:
                data["fan_mode"] = fan_mode
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

        try:
            async with self._session.post(
                f"{self._ha_url}/api/services/climate/{service}",
                json=data,
            ) as resp:
                return {"success": resp.status == 200, "status": resp.status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_switch_control(self, params: dict) -> dict:
        entity_id = params.get("entity_id")
        action = params.get("action")

        if not entity_id or not action:
            return {"success": False, "error": "entity_id and action required"}

        data = {"entity_id": entity_id}

        if action == "on":
            service = "turn_on"
        elif action == "off":
            service = "turn_off"
        elif action == "toggle":
            service = "toggle"
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

        try:
            async with self._session.post(
                f"{self._ha_url}/api/services/switch/{service}",
                json=data,
            ) as resp:
                return {"success": resp.status == 200, "status": resp.status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_cover_control(self, params: dict) -> dict:
        entity_id = params.get("entity_id")
        action = params.get("action")
        position = params.get("position")

        if not entity_id or not action:
            return {"success": False, "error": "entity_id and action required"}

        data = {"entity_id": entity_id}

        if action == "open":
            service = "open_cover"
        elif action == "close":
            service = "close_cover"
        elif action == "stop":
            service = "stop_cover"
        elif action == "set_position":
            service = "set_cover_position"
            if position is not None:
                data["position"] = position
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

        try:
            async with self._session.post(
                f"{self._ha_url}/api/services/cover/{service}",
                json=data,
            ) as resp:
                return {"success": resp.status == 200, "status": resp.status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_scene_activate(self, params: dict) -> dict:
        entity_id = params.get("entity_id")
        if not entity_id:
            return {"success": False, "error": "entity_id required"}

        try:
            async with self._session.post(
                f"{self._ha_url}/api/services/scene/turn_on",
                json={"entity_id": entity_id},
            ) as resp:
                return {"success": resp.status == 200, "status": resp.status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_get_entities(self, params: dict) -> dict:
        domain = params.get("domain")

        try:
            async with self._session.get(f"{self._ha_url}/api/states") as resp:
                states = await resp.json()

            entities = []
            for state in states:
                if not domain or state["entity_id"].startswith(f"{domain}."):
                    entities.append({
                        "entity_id": state["entity_id"],
                        "state": state["state"],
                        "attributes": state["attributes"],
                    })

            return {"success": True, "entities": entities, "count": len(entities)}
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SmartHomePlugin",
    "PLUGIN_METADATA",
]