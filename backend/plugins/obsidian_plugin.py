#!/usr/bin/env python3
"""
================================================================================
VEGA OBSIDIAN PLUGIN - Obsidian Integration
================================================================================
Obsidian vault operations via local file system:
- Note creation/reading/updating
- Search (full-text, tags, links)
- Graph analysis
- Daily notes
- Template rendering
- Link management

References:
- Obsidian Local REST API plugin
- OpenJarvis/OpenJarvis: Obsidian plugin pattern
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from plugins.plugin_loader import PluginBase, PluginConfig, PluginMetadata
from tools.tool_registry import ToolMetadata, ToolParameter
from tools.builtin.file_tools import _workspace

logger = logging.getLogger("vega.plugins.obsidian")


# =============================================================================
# PLUGIN METADATA
# =============================================================================

PLUGIN_METADATA = {
    "name": "obsidian",
    "version": "1.0.0",
    "description": "Obsidian vault integration for local knowledge management",
    "author": "VEGA Team",
    "license": "MIT",
    "requires": [],
    "dependencies": [],
    "entry_point": "obsidian_plugin.py",
    "class_name": "ObsidianPlugin",
    "config_schema": {
        "vault_path": {"type": "string", "description": "Path to Obsidian vault"},
        "use_local_rest_api": {"type": "boolean", "description": "Use Local REST API plugin if available", "default": False},
        "local_rest_api_port": {"type": "integer", "description": "Local REST API port", "default": 27123},
        "local_rest_api_key": {"type": "string", "description": "Local REST API key"},
    },
    "default_config": {
        "enabled": False,
    },
    "permissions": ["obsidian.read", "obsidian.write"],
    "sandboxed": True,
}


# =============================================================================
# OBSIDIAN PLUGIN
# =============================================================================

class ObsidianPlugin(PluginBase):
    """Obsidian vault integration plugin."""

    def __init__(self, config: PluginConfig, event_bus: Any = None, tool_registry: Any = None):
        super().__init__(config, event_bus, tool_registry)
        self._vault_path = Path(config.config.get("vault_path", ""))
        self._use_local_api = config.config.get("use_local_rest_api", False)
        self._api_port = config.config.get("local_rest_api_port", 27123)
        self._api_key = config.config.get("local_rest_api_key")
        self._session: Any = None

        # Expand vault path
        if self._vault_path:
            self._vault_path = self._vault_path.expanduser().resolve()

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(**PLUGIN_METADATA)

    async def initialize(self) -> bool:
        if not self._vault_path or not self._vault_path.exists():
            logger.warning("Obsidian vault path not configured or not found")
            return True

        if self._use_local_api:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30),
                    headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else {},
                )

                # Test connection
                async with self._session.get(f"http://localhost:{self._api_port}/") as resp:
                    if resp.status != 200:
                        logger.warning("Obsidian Local REST API not accessible, falling back to file system")
                        self._use_local_api = False
            except Exception:
                logger.warning("Local REST API not available, using file system")
                self._use_local_api = False

        logger.info("ObsidianPlugin initialized")
        return True

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()
        await super().shutdown()

    async def get_tools(self) -> list:
        return [
            ("obsidian_read", self._tool_read, ToolMetadata(
                name="obsidian_read",
                description="Read note from vault",
                category="obsidian",
                parameters=[
                    ToolParameter("path", "string", "Note path relative to vault", required=True),
                ],
                requires_approval=False,
            )),
            ("obsidian_write", self._tool_write, ToolMetadata(
                name="obsidian_write",
                description="Create or update note",
                category="obsidian",
                parameters=[
                    ToolParameter("path", "string", "Note path relative to vault", required=True),
                    ToolParameter("content", "string", "Note content", required=True),
                    ToolParameter("frontmatter", "object", "YAML frontmatter"),
                ],
                requires_approval=True,
            )),
            ("obsidian_append", self._tool_append, ToolMetadata(
                name="obsidian_append",
                description="Append to note",
                category="obsidian",
                parameters=[
                    ToolParameter("path", "string", "Note path", required=True),
                    ToolParameter("content", "string", "Content to append", required=True),
                ],
                requires_approval=True,
            )),
            ("obsidian_search", self._tool_search, ToolMetadata(
                name="obsidian_search",
                description="Search notes in vault",
                category="obsidian",
                parameters=[
                    ToolParameter("query", "string", "Search query", required=True),
                    ToolParameter("type", "string", "Search type: content, tags, filename", default="content"),
                    ToolParameter("limit", "integer", "Max results", default=50),
                ],
                requires_approval=False,
            )),
            ("obsidian_daily_note", self._tool_daily_note, ToolMetadata(
                name="obsidian_daily_note",
                description="Get or create daily note",
                category="obsidian",
                parameters=[
                    ToolParameter("date", "string", "Date (YYYY-MM-DD), default: today"),
                    ToolParameter("template", "string", "Template path"),
                ],
                requires_approval=False,
            )),
            ("obsidian_tags", self._tool_tags, ToolMetadata(
                name="obsidian_tags",
                description="List all tags in vault",
                category="obsidian",
                parameters=[],
                requires_approval=False,
            )),
            ("obsidian_backlinks", self._tool_backlinks, ToolMetadata(
                name="obsidian_backlinks",
                description="Get backlinks for a note",
                category="obsidian",
                parameters=[
                    ToolParameter("path", "string", "Note path", required=True),
                ],
                requires_approval=False,
            )),
            ("obsidian_graph", self._tool_graph, ToolMetadata(
                name="obsidian_graph",
                description="Get graph data (nodes/edges)",
                category="obsidian",
                parameters=[],
                requires_approval=False,
            )),
        ]

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to vault."""
        note_path = (self._vault_path / path).resolve()
        try:
            note_path.relative_to(self._vault_path)
        except ValueError:
            raise PermissionError("Path outside vault")
        return note_path

    def _read_note(self, note_path: Path) -> dict:
        """Read note and parse frontmatter."""
        content = note_path.read_text(encoding="utf-8")

        # Parse frontmatter
        frontmatter = {}
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                import yaml
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                except Exception:
                    pass
                body = parts[2].lstrip("\n")

        return {
            "path": str(note_path.relative_to(self._vault_path)),
            "frontmatter": frontmatter,
            "content": body,
            "full_content": content,
        }

    def _write_note(self, note_path: Path, content: str, frontmatter: dict = None) -> None:
        """Write note with optional frontmatter."""
        note_path.parent.mkdir(parents=True, exist_ok=True)

        if frontmatter:
            import yaml
            fm = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
            full_content = f"---\n{fm}---\n\n{content}"
        else:
            full_content = content

        note_path.write_text(full_content, encoding="utf-8")

    # -------------------------------------------------------------------------
    # TOOL IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    async def _tool_read(self, params: dict) -> dict:
        path = params.get("path", "")
        if not path:
            return {"success": False, "error": "Path required"}

        try:
            note_path = self._resolve_path(path)
            if not note_path.exists():
                return {"success": False, "error": "Note not found"}

            return {"success": True, **self._read_note(note_path)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_write(self, params: dict) -> dict:
        path = params.get("path", "")
        content = params.get("content", "")
        frontmatter = params.get("frontmatter")

        if not path or not content:
            return {"success": False, "error": "Path and content required"}

        try:
            note_path = self._resolve_path(path)
            self._write_note(note_path, content, frontmatter)
            return {"success": True, "path": str(note_path.relative_to(self._vault_path))}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_append(self, params: dict) -> dict:
        path = params.get("path", "")
        content = params.get("content", "")

        if not path or not content:
            return {"success": False, "error": "Path and content required"}

        try:
            note_path = self._resolve_path(path)
            existing = ""
            if note_path.exists():
                existing = note_path.read_text(encoding="utf-8")

            new_content = existing.rstrip() + "\n\n" + content
            self._write_note(note_path, new_content)
            return {"success": True}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_search(self, params: dict) -> dict:
        query = params.get("query", "")
        search_type = params.get("type", "content")
        limit = params.get("limit", 50)

        if not query:
            return {"success": False, "error": "Query required"}

        try:
            results = []
            query_lower = query.lower()

            for note_path in self._vault_path.rglob("*.md"):
                note = self._read_note(note_path)
                match = False

                if search_type == "content":
                    match = query_lower in note["content"].lower()
                elif search_type == "filename":
                    match = query_lower in note_path.name.lower()
                elif search_type == "tags":
                    # Check frontmatter tags
                    tags = note.get("frontmatter", {}).get("tags", [])
                    if isinstance(tags, str):
                        tags = [tags]
                    match = any(query_lower in tag.lower() for tag in tags)

                if match:
                    results.append({
                        "path": note["path"],
                        "title": note["frontmatter"].get("title", note_path.stem),
                        "snippet": note["content"][:200],
                    })

                    if len(results) >= limit:
                        break

            return {"success": True, "results": results, "count": len(results)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_daily_note(self, params: dict) -> dict:
        date_str = params.get("date")
        template = params.get("template")

        if date_str:
            try:
                date = datetime.fromisoformat(date_str)
            except Exception:
                date = datetime.now()
        else:
            date = datetime.now()

        # Daily note path (configurable)
        daily_path = self._vault_path / "Daily" / f"{date.strftime('%Y-%m-%d')}.md"
        daily_path.parent.mkdir(parents=True, exist_ok=True)

        if not daily_path.exists():
            # Create from template or default
            if template:
                template_path = self._resolve_path(template)
                if template_path.exists():
                    content = template_path.read_text(encoding="utf-8")
                else:
                    content = f"# {date.strftime('%Y-%m-%d')}\n\n"
            else:
                content = f"# {date.strftime('%Y-%m-%d')}\n\n"

            self._write_note(daily_path, content, {
                "date": date.isoformat(),
                "type": "daily",
            })

        return {"success": True, **self._read_note(daily_path)}

    async def _tool_tags(self, params: dict) -> dict:
        try:
            tag_counts = {}

            for note_path in self._vault_path.rglob("*.md"):
                note = self._read_note(note_path)
                tags = note["frontmatter"].get("tags", [])
                if isinstance(tags, str):
                    tags = [tags]
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Sort by count
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            return {"success": True, "tags": [{"tag": t, "count": c} for t, c in sorted_tags]}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_backlinks(self, params: dict) -> dict:
        path = params.get("path", "")
        if not path:
            return {"success": False, "error": "Path required"}

        try:
            target_path = self._resolve_path(path)
            target_name = target_path.stem
            backlinks = []

            for note_path in self._vault_path.rglob("*.md"):
                if note_path == target_path:
                    continue

                note = self._read_note(note_path)
                content = note["full_content"]

                # Look for wiki-links [[target_name]] or markdown links
                if f"[[{target_name}]]" in content or f"]({target_name}" in content:
                    backlinks.append({
                        "path": note["path"],
                        "title": note["frontmatter"].get("title", note_path.stem),
                    })

            return {"success": True, "backlinks": backlinks, "count": len(backlinks)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_graph(self, params: dict) -> dict:
        try:
            nodes = []
            edges = []
            path_to_id = {}

            for i, note_path in enumerate(self._vault_path.rglob("*.md")):
                note = self._read_note(note_path)
                node_id = f"node_{i}"
                path_to_id[note["path"]] = node_id

                nodes.append({
                    "id": node_id,
                    "label": note["frontmatter"].get("title", note_path.stem),
                    "path": note["path"],
                    "tags": note["frontmatter"].get("tags", []),
                })

            # Build edges from wiki-links
            for note_path in self._vault_path.rglob("*.md"):
                note = self._read_note(note_path)
                source_id = path_to_id.get(note["path"])
                if not source_id:
                    continue

                content = note["full_content"]
                # Find wiki-links
                links = re.findall(r'\[\[([^\]]+)\]\]', content)
                for link in links:
                    target_path = self._vault_path / f"{link}.md"
                    if target_path.exists():
                        target_id = path_to_id.get(str(target_path.relative_to(self._vault_path)))
                        if target_id:
                            edges.append({"from": source_id, "to": target_id})

            return {
                "success": True,
                "nodes": nodes,
                "edges": edges,
                "stats": {"nodes": len(nodes), "edges": len(edges)},
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ObsidianPlugin",
    "PLUGIN_METADATA",
]