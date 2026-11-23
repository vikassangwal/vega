#!/usr/bin/env python3
"""
================================================================================
VEGA NOTION PLUGIN - Notion Integration
================================================================================
Notion operations via API:
- Database queries
- Page creation/reading/updating
- Block manipulation
- Search
- Webhook handling

References:
- Notion API
- OpenJarvis/OpenJarvis: Notion plugin pattern
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from plugins.plugin_loader import PluginBase, PluginConfig, PluginMetadata
from tools.tool_registry import ToolMetadata, ToolParameter

logger = logging.getLogger("vega.plugins.notion")


# =============================================================================
# PLUGIN METADATA
# =============================================================================

PLUGIN_METADATA = {
    "name": "notion",
    "version": "1.0.0",
    "description": "Notion integration for notes, databases, and knowledge management",
    "author": "VEGA Team",
    "license": "MIT",
    "requires": [],
    "dependencies": ["aiohttp"],
    "entry_point": "notion_plugin.py",
    "class_name": "NotionPlugin",
    "config_schema": {
        "api_token": {"type": "string", "description": "Notion integration token"},
        "workspace_id": {"type": "string", "description": "Workspace ID (optional)"},
    },
    "default_config": {
        "enabled": False,
    },
    "permissions": ["notion.read", "notion.write"],
    "sandboxed": True,
}


# =============================================================================
# NOTION PLUGIN
# =============================================================================

class NotionPlugin(PluginBase):
    """Notion integration plugin."""

    def __init__(self, config: PluginConfig, event_bus: Any = None, tool_registry: Any = None):
        super().__init__(config, event_bus, tool_registry)
        self._api_token = config.config.get("api_token")
        self._session: aiohttp.ClientSession | None = None
        self._base_url = "https://api.notion.com/v1"
        self._headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(**PLUGIN_METADATA)

    async def initialize(self) -> bool:
        if not self._api_token:
            logger.warning("Notion API token not configured")
            return True

        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=self._headers,
            )

            # Test connection
            async with self._session.get(f"{self._base_url}/users/me") as resp:
                if resp.status != 200:
                    logger.error("Notion connection failed")
                    return False

            logger.info("NotionPlugin initialized")
            return True

        except Exception as e:
            logger.error(f"NotionPlugin init error: {e}")
            return False

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()
        await super().shutdown()

    async def get_tools(self) -> list:
        return [
            ("notion_search", self._tool_search, ToolMetadata(
                name="notion_search",
                description="Search Notion pages and databases",
                category="notion",
                parameters=[
                    ToolParameter("query", "string", "Search query", required=True),
                    ToolParameter("filter", "object", "Filter object"),
                    ToolParameter("page_size", "integer", "Results per page", default=20),
                ],
                requires_approval=False,
            )),
            ("notion_get_page", self._tool_get_page, ToolMetadata(
                name="notion_get_page",
                description="Get page content",
                category="notion",
                parameters=[
                    ToolParameter("page_id", "string", "Page ID", required=True),
                ],
                requires_approval=False,
            )),
            ("notion_create_page", self._tool_create_page, ToolMetadata(
                name="notion_create_page",
                description="Create new page",
                category="notion",
                parameters=[
                    ToolParameter("parent_id", "string", "Parent page/database ID", required=True),
                    ToolParameter("title", "string", "Page title", required=True),
                    ToolParameter("properties", "object", "Page properties"),
                    ToolParameter("children", "array", "Block children"),
                ],
                requires_approval=True,
            )),
            ("notion_update_page", self._tool_update_page, ToolMetadata(
                name="notion_update_page",
                description="Update page properties",
                category="notion",
                parameters=[
                    ToolParameter("page_id", "string", "Page ID", required=True),
                    ToolParameter("properties", "object", "Properties to update"),
                    ToolParameter("archived", "boolean", "Archive page"),
                ],
                requires_approval=True,
            )),
            ("notion_query_database", self._tool_query_database, ToolMetadata(
                name="notion_query_database",
                description="Query database with filters",
                category="notion",
                parameters=[
                    ToolParameter("database_id", "string", "Database ID", required=True),
                    ToolParameter("filter", "object", "Filter object"),
                    ToolParameter("sorts", "array", "Sort objects"),
                    ToolParameter("page_size", "integer", "Page size", default=100),
                ],
                requires_approval=False,
            )),
            ("notion_append_blocks", self._tool_append_blocks, ToolMetadata(
                name="notion_append_blocks",
                description="Append blocks to page",
                category="notion",
                parameters=[
                    ToolParameter("block_id", "string", "Block ID to append after", required=True),
                    ToolParameter("children", "array", "Block children to append", required=True),
                ],
                requires_approval=True,
            )),
        ]

    # -------------------------------------------------------------------------
    # TOOL IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    async def _tool_search(self, params: dict) -> dict:
        query = params.get("query", "")
        filter_obj = params.get("filter")
        page_size = params.get("page_size", 20)

        if not query:
            return {"success": False, "error": "Query required"}

        try:
            payload = {"query": query, "page_size": page_size}
            if filter_obj:
                payload["filter"] = filter_obj

            async with self._session.post(
                f"{self._base_url}/search",
                json=payload,
            ) as resp:
                data = await resp.json()
                return {"success": True, "results": data.get("results", [])}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_get_page(self, params: dict) -> dict:
        page_id = params.get("page_id")
        if not page_id:
            return {"success": False, "error": "page_id required"}

        try:
            async with self._session.get(f"{self._base_url}/pages/{page_id}") as resp:
                data = await resp.json()
                return {"success": True, "page": data}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_create_page(self, params: dict) -> dict:
        parent_id = params.get("parent_id")
        title = params.get("title")
        properties = params.get("properties", {})
        children = params.get("children", [])

        if not parent_id or not title:
            return {"success": False, "error": "parent_id and title required"}

        try:
            payload = {
                "parent": {"page_id": parent_id} if not parent_id.startswith("data:") else {"database_id": parent_id},
                "properties": {
                    "title": {"title": [{"text": {"content": title}}]},
                    **properties,
                },
            }
            if children:
                payload["children"] = children

            async with self._session.post(f"{self._base_url}/pages", json=payload) as resp:
                data = await resp.json()
                return {"success": True, "page": data}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_update_page(self, params: dict) -> dict:
        page_id = params.get("page_id")
        properties = params.get("properties")
        archived = params.get("archived")

        if not page_id:
            return {"success": False, "error": "page_id required"}

        try:
            payload = {}
            if properties:
                payload["properties"] = properties
            if archived is not None:
                payload["archived"] = archived

            async with self._session.patch(f"{self._base_url}/pages/{page_id}", json=payload) as resp:
                data = await resp.json()
                return {"success": True, "page": data}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_query_database(self, params: dict) -> dict:
        database_id = params.get("database_id")
        filter_obj = params.get("filter")
        sorts = params.get("sorts", [])
        page_size = params.get("page_size", 100)

        if not database_id:
            return {"success": False, "error": "database_id required"}

        try:
            payload = {"page_size": page_size}
            if filter_obj:
                payload["filter"] = filter_obj
            if sorts:
                payload["sorts"] = sorts

            async with self._session.post(
                f"{self._base_url}/databases/{database_id}/query",
                json=payload,
            ) as resp:
                data = await resp.json()
                return {"success": True, "results": data.get("results", []), "has_more": data.get("has_more")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_append_blocks(self, params: dict) -> dict:
        block_id = params.get("block_id")
        children = params.get("children")

        if not block_id or not children:
            return {"success": False, "error": "block_id and children required"}

        try:
            async with self._session.patch(
                f"{self._base_url}/blocks/{block_id}/children",
                json={"children": children},
            ) as resp:
                data = await resp.json()
                return {"success": True, "results": data.get("results", [])}

        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "NotionPlugin",
    "PLUGIN_METADATA",
]