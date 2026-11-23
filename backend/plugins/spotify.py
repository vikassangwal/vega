#!/usr/bin/env python3
"""
================================================================================
VEGA SPOTIFY PLUGIN - Spotify Integration
================================================================================
Spotify control and playback:
- Play/pause/next/previous
- Search tracks/albums/artists
- Playlist management
- Current playback info
- Volume control
- Device selection

References:
- OpenJarvis/OpenJarvis: Spotify plugin pattern
- Spotify Web API
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp

from plugins.plugin_loader import PluginBase, PluginConfig, PluginMetadata
from tools.tool_registry import ToolMetadata, ToolParameter

logger = logging.getLogger("vega.plugins.spotify")


# =============================================================================
# PLUGIN METADATA
# =============================================================================

PLUGIN_METADATA = {
    "name": "spotify",
    "version": "1.0.0",
    "description": "Spotify playback control and music discovery",
    "author": "VEGA Team",
    "license": "MIT",
    "requires": [],
    "dependencies": ["aiohttp"],
    "entry_point": "spotify.py",
    "class_name": "SpotifyPlugin",
    "config_schema": {
        "client_id": {"type": "string", "description": "Spotify Client ID"},
        "client_secret": {"type": "string", "description": "Spotify Client Secret"},
        "redirect_uri": {"type": "string", "description": "OAuth Redirect URI", "default": "http://localhost:8888/callback"},
        "auto_refresh_token": {"type": "boolean", "default": True},
    },
    "default_config": {
        "enabled": False,
    },
    "permissions": ["media.playback", "media.library", "user.profile"],
    "sandboxed": True,
}


# =============================================================================
# SPOTIFY PLUGIN
# =============================================================================

class SpotifyPlugin(PluginBase):
    """Spotify integration plugin."""

    def __init__(self, config: PluginConfig, event_bus: Any = None, tool_registry: Any = None):
        super().__init__(config, event_bus, tool_registry)
        self._client_id = config.config.get("client_id")
        self._client_secret = config.config.get("client_secret")
        self._redirect_uri = config.config.get("redirect_uri", "http://localhost:8888/callback")
        self._auto_refresh = config.config.get("auto_refresh_token", True)

        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._token_expires: float = 0
        self._session: aiohttp.ClientSession | None = None
        self._user_id: str | None = None

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(**PLUGIN_METADATA)

    async def initialize(self) -> bool:
        if not self._client_id or not self._client_secret:
            logger.warning("Spotify credentials not configured")
            return True  # Allow loading without credentials

        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "VEGA/1.0"},
            )

            # Try to load saved tokens
            await self._load_tokens()

            if self._refresh_token and self._auto_refresh:
                await self._refresh_access_token()

            logger.info("SpotifyPlugin initialized")
            return True

        except Exception as e:
            logger.error(f"SpotifyPlugin init error: {e}")
            return False

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()
        await super().shutdown()

    async def get_tools(self) -> list:
        """Return tools provided by this plugin."""
        return [
            ("spotify_play", self._tool_play, ToolMetadata(
                name="spotify_play",
                description="Play track/album/playlist",
                category="media",
                parameters=[
                    ToolParameter("uri", "string", "Spotify URI (track, album, playlist)", required=False),
                    ToolParameter("context_uri", "string", "Context URI (album, playlist)"),
                    ToolParameter("offset", "integer", "Offset position"),
                    ToolParameter("position_ms", "integer", "Start position in ms"),
                ],
                requires_approval=False,
            )),
            ("spotify_pause", self._tool_pause, ToolMetadata(
                name="spotify_pause",
                description="Pause playback",
                category="media",
                parameters=[],
                requires_approval=False,
            )),
            ("spotify_next", self._tool_next, ToolMetadata(
                name="spotify_next",
                description="Next track",
                category="media",
                parameters=[],
                requires_approval=False,
            )),
            ("spotify_previous", self._tool_previous, ToolMetadata(
                name="spotify_previous",
                description="Previous track",
                category="media",
                parameters=[],
                requires_approval=False,
            )),
            ("spotify_search", self._tool_search, ToolMetadata(
                name="spotify_search",
                description="Search tracks/artists/albums/playlists",
                category="media",
                parameters=[
                    ToolParameter("query", "string", "Search query", required=True),
                    ToolParameter("type", "string", "Type: track, artist, album, playlist", default="track"),
                    ToolParameter("limit", "integer", "Max results", default=20),
                ],
                requires_approval=False,
            )),
            ("spotify_current", self._tool_current, ToolMetadata(
                name="spotify_current",
                description="Get current playback info",
                category="media",
                parameters=[],
                requires_approval=False,
            )),
            ("spotify_devices", self._tool_devices, ToolMetadata(
                name="spotify_devices",
                description="List available devices",
                category="media",
                parameters=[],
                requires_approval=False,
            )),
            ("spotify_volume", self._tool_volume, ToolMetadata(
                name="spotify_volume",
                description="Set playback volume",
                category="media",
                parameters=[
                    ToolParameter("volume_percent", "integer", "Volume 0-100", required=True),
                ],
                requires_approval=False,
            )),
            ("spotify_playlist", self._tool_playlist, ToolMetadata(
                name="spotify_playlist",
                description="Playlist operations",
                category="media",
                parameters=[
                    ToolParameter("action", "string", "Action: list, create, get, add_tracks", default="list"),
                    ToolParameter("playlist_id", "string", "Playlist ID"),
                    ToolParameter("name", "string", "Playlist name"),
                    ToolParameter("tracks", "array", "Track URIs to add"),
                ],
                requires_approval=False,
            )),
        ]

    # -------------------------------------------------------------------------
    # AUTHENTICATION
    # -------------------------------------------------------------------------

    async def _load_tokens(self) -> None:
        """Load saved tokens from config."""
        self._access_token = self.config.config.get("access_token")
        self._refresh_token = self.config.config.get("refresh_token")
        self._token_expires = self.config.config.get("token_expires", 0)

    async def _save_tokens(self) -> None:
        """Save tokens to config."""
        self.config.config["access_token"] = self._access_token
        self.config.config["refresh_token"] = self._refresh_token
        self.config.config["token_expires"] = self._token_expires

    async def _refresh_access_token(self) -> bool:
        """Refresh access token using refresh token."""
        if not self._refresh_token:
            return False

        try:
            auth = aiohttp.BasicAuth(self._client_id, self._client_secret)
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
            }

            async with self._session.post(
                "https://accounts.spotify.com/api/token",
                auth=auth,
                data=data,
            ) as resp:
                if resp.status != 200:
                    return False

                token_data = await resp.json()
                self._access_token = token_data["access_token"]
                self._token_expires = time.time() + token_data["expires_in"]

                if "refresh_token" in token_data:
                    self._refresh_token = token_data["refresh_token"]

                await self._save_tokens()
                return True

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False

    async def _ensure_valid_token(self) -> bool:
        """Ensure access token is valid, refresh if needed."""
        if not self._access_token:
            return await self._refresh_access_token()

        if time.time() >= self._token_expires - 60:  # 1 min buffer
            return await self._refresh_access_token()

        return True

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._access_token}"}

    # -------------------------------------------------------------------------
    # TOOL IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    async def _tool_play(self, params: dict) -> dict:
        if not await self._ensure_valid_token():
            return {"success": False, "error": "Not authenticated"}

        uri = params.get("uri")
        context_uri = params.get("context_uri")
        offset = params.get("offset")
        position_ms = params.get("position_ms")

        payload = {}
        if uri:
            payload["uris"] = [uri]
        if context_uri:
            payload["context_uri"] = context_uri
        if offset is not None:
            payload["offset"] = {"position": offset}
        if position_ms is not None:
            payload["position_ms"] = position_ms

        try:
            async with self._session.put(
                "https://api.spotify.com/v1/me/player/play",
                headers=self._auth_headers(),
                json=payload,
            ) as resp:
                return {"success": resp.status in [200, 204], "status": resp.status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_pause(self, params: dict) -> dict:
        if not await self._ensure_valid_token():
            return {"success": False, "error": "Not authenticated"}

        try:
            async with self._session.put(
                "https://api.spotify.com/v1/me/player/pause",
                headers=self._auth_headers(),
            ) as resp:
                return {"success": resp.status in [200, 204], "status": resp.status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_next(self, params: dict) -> dict:
        if not await self._ensure_valid_token():
            return {"success": False, "error": "Not authenticated"}

        try:
            async with self._session.post(
                "https://api.spotify.com/v1/me/player/next",
                headers=self._auth_headers(),
            ) as resp:
                return {"success": resp.status in [200, 204], "status": resp.status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_previous(self, params: dict) -> dict:
        if not await self._ensure_valid_token():
            return {"success": False, "error": "Not authenticated"}

        try:
            async with self._session.post(
                "https://api.spotify.com/v1/me/player/previous",
                headers=self._auth_headers(),
            ) as resp:
                return {"success": resp.status in [200, 204], "status": resp.status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_search(self, params: dict) -> dict:
        if not await self._ensure_valid_token():
            return {"success": False, "error": "Not authenticated"}

        query = params.get("query", "")
        type_ = params.get("type", "track")
        limit = params.get("limit", 20)

        if not query:
            return {"success": False, "error": "Query required"}

        try:
            params_req = {"q": query, "type": type_, "limit": limit}
            async with self._session.get(
                "https://api.spotify.com/v1/search",
                headers=self._auth_headers(),
                params=params_req,
            ) as resp:
                data = await resp.json()
                return {"success": True, "results": data}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_current(self, params: dict) -> dict:
        if not await self._ensure_valid_token():
            return {"success": False, "error": "Not authenticated"}

        try:
            async with self._session.get(
                "https://api.spotify.com/v1/me/player/currently-playing",
                headers=self._auth_headers(),
            ) as resp:
                if resp.status == 204:
                    return {"success": True, "playing": False}
                data = await resp.json()
                return {"success": True, "playback": data}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_devices(self, params: dict) -> dict:
        if not await self._ensure_valid_token():
            return {"success": False, "error": "Not authenticated"}

        try:
            async with self._session.get(
                "https://api.spotify.com/v1/me/player/devices",
                headers=self._auth_headers(),
            ) as resp:
                data = await resp.json()
                return {"success": True, "devices": data.get("devices", [])}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_volume(self, params: dict) -> dict:
        if not await self._ensure_valid_token():
            return {"success": False, "error": "Not authenticated"}

        volume = params.get("volume_percent", 50)

        try:
            async with self._session.put(
                f"https://api.spotify.com/v1/me/player/volume?volume_percent={volume}",
                headers=self._auth_headers(),
            ) as resp:
                return {"success": resp.status in [200, 204], "status": resp.status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_playlist(self, params: dict) -> dict:
        if not await self._ensure_valid_token():
            return {"success": False, "error": "Not authenticated"}

        action = params.get("action", "list")
        playlist_id = params.get("playlist_id")
        name = params.get("name")
        tracks = params.get("tracks", [])

        try:
            if action == "list":
                async with self._session.get(
                    "https://api.spotify.com/v1/me/playlists",
                    headers=self._auth_headers(),
                ) as resp:
                    data = await resp.json()
                    return {"success": True, "playlists": data.get("items", [])}

            elif action == "get" and playlist_id:
                async with self._session.get(
                    f"https://api.spotify.com/v1/playlists/{playlist_id}",
                    headers=self._auth_headers(),
                ) as resp:
                    data = await resp.json()
                    return {"success": True, "playlist": data}

            elif action == "create" and name:
                payload = {"name": name, "public": False}
                async with self._session.post(
                    f"https://api.spotify.com/v1/users/{self._user_id}/playlists",
                    headers=self._auth_headers(),
                    json=payload,
                ) as resp:
                    data = await resp.json()
                    return {"success": True, "playlist": data}

            elif action == "add_tracks" and playlist_id and tracks:
                payload = {"uris": tracks}
                async with self._session.post(
                    f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks",
                    headers=self._auth_headers(),
                    json=payload,
                ) as resp:
                    return {"success": resp.status in [200, 201], "status": resp.status}

            return {"success": False, "error": "Invalid action or missing parameters"}

        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SpotifyPlugin",
    "PLUGIN_METADATA",
]