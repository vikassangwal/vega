"""
================================================================================
VEGA EPISODIC MEMORY - Long-term User Memory (Mem0 Integration)
================================================================================
Persistent memory across sessions using Mem0:
- User preference learning
- Cross-session context retention
- Automatic memory extraction from conversations
- Memory updates and conflict resolution
- Multi-user support with isolation

References:
- mem0ai/mem0: Persistent memory layer for AI agents
- OpenJarvis/OpenJarvis: Mem0 integration for personal memory
================================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from config import get_settings, settings

logger = logging.getLogger("vega.memory.episodic")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class EpisodicConfig:
    api_key: str | None = None
    local_mode: bool = True
    vector_store: str = "chromadb"
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-4o-mini"
    user_id: str = "default"


# =============================================================================
# EPISODIC MEMORY (Mem0 Wrapper)
# =============================================================================

class EpisodicMemory:
    """
    Wrapper around Mem0 for persistent episodic memory.
    Falls back to local implementation if Mem0 unavailable.
    """

    def __init__(self, config: EpisodicConfig | None = None):
        self.config = config or EpisodicConfig()
        self._mem0_client = None
        self._local_store: dict[str, list] = {}  # user_id -> memories
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize Mem0 client or local fallback."""
        try:
            # Try to import and initialize Mem0
            from mem0 import Memory as Mem0Memory

            mem0_config = {
                "vector_store": {
                    "provider": self.config.vector_store,
                    "config": {
                        "collection_name": "vega_episodic",
                        "path": "./data/chromadb",
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {"model": self.config.embedder},
                },
                "llm": {
                    "provider": "openai" if self.config.api_key else "ollama",
                    "config": {"model": self.config.llm_model},
                },
            }

            if self.config.api_key:
                mem0_config["llm"]["config"]["api_key"] = self.config.api_key

            self._mem0_client = Mem0Memory.from_config(mem0_config)
            logger.info("EpisodicMemory initialized with Mem0")
            self._initialized = True
            return True

        except ImportError:
            logger.info("Mem0 not available, using local episodic memory")
            return await self._init_local()
        except Exception as e:
            logger.warning(f"Mem0 initialization failed, using local: {e}")
            return await self._init_local()

    async def _init_local(self) -> bool:
        """Initialize local fallback storage."""
        self._local_store = {}
        self._initialized = True
        return True

    async def shutdown(self) -> None:
        if self._mem0_client:
            # Mem0 doesn't have explicit shutdown
            pass
        logger.info("EpisodicMemory shutdown")

    # -------------------------------------------------------------------------
    # CORE OPERATIONS
    # -------------------------------------------------------------------------

    async def add(self, entry: Any) -> str:
        """Add memory entry."""
        if not self._initialized:
            await self.initialize()

        memory_data = {
            "content": entry.content,
            "metadata": {
                **entry.metadata,
                "entry_id": entry.entry_id,
                "tier": entry.tier.value if hasattr(entry.tier, 'value') else str(entry.tier),
                "conversation_id": entry.conversation_id,
                "agent_id": entry.agent_id,
                "task_id": entry.task_id,
                "importance": entry.importance,
                "tags": entry.tags,
            },
        }

        user_id = entry.user_id or self.config.user_id

        if self._mem0_client:
            try:
                # Mem0 expects messages format
                messages = [
                    {"role": "user", "content": entry.content}
                ]
                result = self._mem0_client.add(
                    messages=messages,
                    user_id=user_id,
                    metadata=memory_data["metadata"],
                )
                return result.get("id", entry.entry_id)
            except Exception as e:
                logger.error(f"Mem0 add error: {e}")
                # Fallback to local
        return await self._add_local(user_id, memory_data)

    async def _add_local(self, user_id: str, memory_data: dict) -> str:
        """Add to local store."""
        if user_id not in self._local_store:
            self._local_store[user_id] = []

        memory_id = str(uuid.uuid4())
        memory_data["memory_id"] = memory_id
        memory_data["created_at"] = time.time()
        self._local_store[user_id].append(memory_data)
        return memory_id

    async def search(
        self,
        query: str,
        limit: int = 10,
        user_id: str | None = None,
    ) -> list:
        """Search episodic memories."""
        if not self._initialized:
            await self.initialize()

        user_id = user_id or self.config.user_id

        if self._mem0_client:
            try:
                results = self._mem0_client.search(
                    query=query,
                    user_id=user_id,
                    limit=limit,
                )
                return self._format_mem0_results(results)
            except Exception as e:
                logger.error(f"Mem0 search error: {e}")

        return await self._search_local(query, limit, user_id)

    def _format_mem0_results(self, results: list) -> list:
        """Convert Mem0 results to SearchResult format."""
        from memory.memory_manager import SearchResult, MemoryTier, MemoryEntry

        formatted = []
        for r in results:
            metadata = r.get("metadata", {})
            entry = MemoryEntry(
                tier=MemoryTier.EPISODIC,
                content=r.get("memory", ""),
                metadata=metadata,
                created_at=metadata.get("created_at", time.time()),
            )
            entry.entry_id = metadata.get("entry_id", str(uuid.uuid4()))

            formatted.append(SearchResult(
                entry=entry,
                score=r.get("score", 0.8),
                tier=MemoryTier.EPISODIC,
                snippet=r.get("memory", "")[:200],
            ))

        return formatted

    async def _search_local(self, query: str, limit: int, user_id: str) -> list:
        """Search local store with simple text matching."""
        from memory.memory_manager import SearchResult, MemoryTier, MemoryEntry

        results = []
        query_lower = query.lower()

        memories = self._local_store.get(user_id, [])

        for mem in memories:
            content = mem.get("content", "")
            if query_lower in content.lower():
                entry = MemoryEntry(
                    tier=MemoryTier.EPISODIC,
                    content=content,
                    metadata=mem.get("metadata", {}),
                    created_at=mem.get("created_at", time.time()),
                )
                entry.entry_id = mem.get("memory_id", str(uuid.uuid4()))

                results.append(SearchResult(
                    entry=entry,
                    score=0.7,
                    tier=MemoryTier.EPISODIC,
                    snippet=content[:200],
                ))

                if len(results) >= limit:
                    break

        return results

    async def get_all(self, user_id: str | None = None, limit: int = 100) -> list:
        """Get all memories for user."""
        user_id = user_id or self.config.user_id

        if self._mem0_client:
            try:
                return self._mem0_client.get_all(user_id=user_id, limit=limit)
            except Exception:
                pass

        memories = self._local_store.get(user_id, [])
        return memories[:limit]

    async def update(self, memory_id: str, content: str, user_id: str | None = None) -> bool:
        """Update existing memory."""
        user_id = user_id or self.config.user_id

        if self._mem0_client:
            try:
                self._mem0_client.update(memory_id=memory_id, data=content)
                return True
            except Exception:
                pass

        # Local update
        memories = self._local_store.get(user_id, [])
        for mem in memories:
            if mem.get("memory_id") == memory_id:
                mem["content"] = content
                return True

        return False

    async def delete(self, memory_id: str, user_id: str | None = None) -> bool:
        """Delete memory."""
        user_id = user_id or self.config.user_id

        if self._mem0_client:
            try:
                self._mem0_client.delete(memory_id=memory_id)
                return True
            except Exception:
                pass

        # Local delete
        memories = self._local_store.get(user_id, [])
        self._local_store[user_id] = [m for m in memories if m.get("memory_id") != memory_id]
        return True

    # -------------------------------------------------------------------------
    # STATS
    # -------------------------------------------------------------------------

    async def get_stats(self) -> dict:
        user_id = self.config.user_id
        count = len(self._local_store.get(user_id, []))

        return {
            "backend": "mem0" if self._mem0_client else "local",
            "total_memories": count,
            "user_id": user_id,
        }

    async def health_check(self) -> dict:
        return {
            "healthy": self._initialized,
            "backend": "mem0" if self._mem0_client else "local",
            "initialized": self._initialized,
        }

    # -------------------------------------------------------------------------
    # EXPORTS
    # -------------------------------------------------------------------------

__all__ = [
    "EpisodicMemory",
    "EpisodicConfig",
]