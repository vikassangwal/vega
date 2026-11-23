"""
================================================================================
VEGA MEMORY MANAGER - Unified 7-Tier Memory Interface
================================================================================
Orchestrates all memory tiers with intelligent routing:
- Automatic tier selection based on query type
- Cross-tier search and synthesis
- Memory consolidation (short-term → long-term)
- Forgetting curves and relevance decay
- Privacy-compliant data handling

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY MANAGER                            │
├─────────────────────────────────────────────────────────────┤
│  ingest() → auto-routes → search() → merges cross-tier      │
│  consolidate() → promotes memories up the hierarchy         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────┬───────────┼───────────┬─────────┐
        ▼         ▼           ▼           ▼         ▼
   ┌────────┐┌────────┐┌──────────┐┌────────┐┌────────┐
   │Short   ││Working ││Episodic  ││Semantic││Graph   │
   │Term    ││Memory  ││(Mem0)    ││(Chroma)││(Neo4j) │
   └────────┘└────────┘└──────────┘└────────┘└────────┘
        │         │           │           │         │
        └─────────┴───────────┼───────────┴─────────┘
                              ▼
                      ┌──────────────┐
                      │ Procedural   │  ← Learned routines
                      │ Reflective   │  ← Self-evaluation logs
                      └──────────────┘
================================================================================
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from config import get_settings, settings

logger = logging.getLogger("vega.memory.manager")


# =============================================================================
# DATA MODELS
# =============================================================================

class MemoryTier(str, Enum):
    SHORT_TERM = "short_term"      # Conversation buffer
    WORKING = "working"            # Active task context
    EPISODIC = "episodic"          # Timestamped events (Mem0)
    SEMANTIC = "semantic"          # Vector search (ChromaDB)
    GRAPH = "graph"                # Knowledge Graph (Neo4j)
    PROCEDURAL = "procedural"      # Learned routines
    REFLECTIVE = "reflective"      # Self-evaluation logs


@dataclass
class MemoryEntry:
    """Unified memory entry across all tiers."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tier: MemoryTier = MemoryTier.SHORT_TERM
    content: str = ""
    # Metadata
    metadata: dict = field(default_factory=dict)
    # Relations
    conversation_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    # Embedding (for semantic tier)
    embedding: list[float] | None = None
    # Graph (for graph tier)
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    # Timing
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    # Relevance
    importance: float = 1.0  # 0-1
    decay_rate: float = 0.01  # Per day
    # Tags for retrieval
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "tier": self.tier.value,
            "content": self.content,
            "metadata": self.metadata,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "entities": self.entities,
            "relationships": self.relationships,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "importance": self.importance,
            "decay_rate": self.decay_rate,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        entry = cls()
        for key, value in data.items():
            if hasattr(entry, key):
                if key == "tier" and isinstance(value, str):
                    setattr(entry, key, MemoryTier(value))
                else:
                    setattr(entry, key, value)
        return entry


@dataclass
class SearchResult:
    """Search result with tier info."""
    entry: MemoryEntry
    score: float
    tier: MemoryTier
    snippet: str = ""


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    # Short-term
    short_term_max_turns: int = 20
    short_term_max_tokens: int = 8000

    # Working
    working_max_items: int = 50

    # Episodic (Mem0)
    mem0_enabled: bool = True
    mem0_api_key: str | None = None
    mem0_local_mode: bool = True
    mem0_vector_store: str = "chromadb"

    # Semantic (ChromaDB)
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000
    chromadb_collection: str = "vega_memory"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Graph (Neo4j + GraphRAG)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "vega_graph_password"
    graphrag_enabled: bool = True
    graphrag_community_level: int = 2
    graphrag_response_type: str = "multiple paragraphs"
    graphrag_max_input_tokens: int = 8000

    # Procedural
    procedural_enabled: bool = True
    procedural_min_occurrences: int = 3
    procedural_similarity_threshold: float = 0.85

    # Reflective
    reflective_enabled: bool = True
    reflective_log_interval_minutes: int = 60
    reflective_retention_days: int = 90

    # General
    auto_consolidate: bool = True
    consolidate_interval_minutes: int = 30
    max_search_results: int = 10
    cross_tier_search: bool = True


# =============================================================================
# MEMORY MANAGER
# =============================================================================

class MemoryManager:
    """
    Unified interface for all memory tiers.
    Handles routing, consolidation, and cross-tier operations.
    """

    def __init__(
        self,
        event_bus: Any = None,
        config: MemoryConfig | None = None,
    ):
        self.event_bus = event_bus
        self.config = config or MemoryConfig()

        # Tier instances (lazy loaded)
        self._short_term: Any = None
        self._working: Any = None
        self._episodic: Any = None
        self._semantic: Any = None
        self._graph: Any = None
        self._procedural: Any = None
        self._reflective: Any = None

        # Consolidation task
        self._consolidation_task: asyncio.Task | None = None
        self._running = False

        # Stats
        self._stats = {
            "total_entries": 0,
            "entries_by_tier": {t.value: 0 for t in MemoryTier},
            "searches": 0,
            "consolidations": 0,
        }

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    async def initialize(self) -> bool:
        """Initialize all memory tiers."""
        try:
            # Initialize tiers
            await self._init_short_term()
            await self._init_working()
            await self._init_episodic()
            await self._init_semantic()
            await self._init_graph()
            await self._init_procedural()
            await self._init_reflective()

            # Start consolidation loop
            if self.config.auto_consolidate:
                self._running = True
                self._consolidation_task = asyncio.create_task(self._consolidation_loop())

            logger.info("MemoryManager initialized with all tiers")
            return True

        except Exception as e:
            logger.exception(f"MemoryManager initialization failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown memory manager."""
        self._running = False

        if self._consolidation_task:
            self._consolidation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consolidation_task

        # Shutdown tiers
        for tier_name in ["_short_term", "_working", "_episodic", "_semantic", "_graph", "_procedural", "_reflective"]:
            tier = getattr(self, tier_name, None)
            if tier and hasattr(tier, "shutdown"):
                await tier.shutdown()

        logger.info("MemoryManager shutdown")

    # -------------------------------------------------------------------------
    # TIER INITIALIZATION
    # -------------------------------------------------------------------------

    async def _init_short_term(self) -> None:
        from .short_term import ShortTermMemory
        self._short_term = ShortTermMemory(
            max_turns=self.config.short_term_max_turns,
            max_tokens=self.config.short_term_max_tokens,
        )
        await self._short_term.initialize()

    async def _init_working(self) -> None:
        # Working memory is simple in-memory dict
        self._working = {
            "contexts": {},
            "max_items": self.config.working_max_items,
        }

    async def _init_episodic(self) -> None:
        if not self.config.mem0_enabled:
            return

        try:
            from .episodic import EpisodicMemory
            self._episodic = EpisodicMemory(
                api_key=self.config.mem0_api_key,
                local_mode=self.config.mem0_local_mode,
                vector_store=self.config.mem0_vector_store,
            )
            await self._episodic.initialize()
        except Exception as e:
            logger.warning(f"Episodic memory (Mem0) not available: {e}")

    async def _init_semantic(self) -> None:
        try:
            from .semantic import SemanticMemory
            self._semantic = SemanticMemory(
                host=self.config.chromadb_host,
                port=self.config.chromadb_port,
                collection_name=self.config.chromadb_collection,
                embedding_model=self.config.embedding_model,
            )
            await self._semantic.initialize()
        except Exception as e:
            logger.warning(f"Semantic memory (ChromaDB) not available: {e}")

    async def _init_graph(self) -> None:
        try:
            from .graph import GraphMemory
            self._graph = GraphMemory(
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                graphrag_enabled=self.config.graphrag_enabled,
            )
            await self._graph.initialize()
        except Exception as e:
            logger.warning(f"Graph memory (Neo4j) not available: {e}")

    async def _init_procedural(self) -> None:
        if not self.config.procedural_enabled:
            return
        # Procedural memory uses graph + episodic
        self._procedural = {
            "patterns": {},
            "min_occurrences": self.config.procedural_min_occurrences,
            "similarity_threshold": self.config.procedural_similarity_threshold,
        }

    async def _init_reflective(self) -> None:
        if not self.config.reflective_enabled:
            return
        self._reflective = {
            "logs": deque(maxlen=10000),
            "interval": self.config.reflective_log_interval_minutes * 60,
            "retention_days": self.config.reflective_retention_days,
        }

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    async def ingest(
        self,
        content: str,
        metadata: dict | None = None,
        tier: MemoryTier = MemoryTier.EPISODIC,
        conversation_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        task_id: str | None = None,
        importance: float = 1.0,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """
        Store content in memory.
        Automatically routes to appropriate tier(s).
        """
        entry = MemoryEntry(
            tier=tier,
            content=content,
            metadata=metadata or {},
            conversation_id=conversation_id,
            user_id=user_id,
            agent_id=agent_id,
            task_id=task_id,
            importance=importance,
            tags=tags or [],
        )

        # Route to tier
        await self._store_in_tier(entry)

        # Update stats
        self._stats["total_entries"] += 1
        self._stats["entries_by_tier"][tier.value] += 1

        # Emit event
        if self.event_bus:
            await self.event_bus.emit("memory.ingested", {
                "entry_id": entry.entry_id,
                "tier": tier.value,
                "content_preview": content[:100],
            })

        return entry

    async def _store_in_tier(self, entry: MemoryEntry) -> None:
        """Route entry to appropriate tier implementation."""
        tier = entry.tier

        if tier == MemoryTier.SHORT_TERM:
            await self._short_term.add(entry)
        elif tier == MemoryTier.WORKING:
            await self._store_working(entry)
        elif tier == MemoryTier.EPISODIC and self._episodic:
            await self._episodic.add(entry)
        elif tier == MemoryTier.SEMANTIC and self._semantic:
            await self._semantic.add(entry)
        elif tier == MemoryTier.GRAPH and self._graph:
            await self._graph.add(entry)
        elif tier == MemoryTier.PROCEDURAL:
            await self._store_procedural(entry)
        elif tier == MemoryTier.REFLECTIVE:
            await self._store_reflective(entry)

    async def _store_working(self, entry: MemoryEntry) -> None:
        """Store in working memory."""
        key = entry.conversation_id or entry.task_id or "default"
        if key not in self._working["contexts"]:
            self._working["contexts"][key] = []

        contexts = self._working["contexts"][key]
        contexts.append(entry.to_dict())

        # Trim
        if len(contexts) > self._working["max_items"]:
            self._working["contexts"][key] = contexts[-self._working["max_items"]:]

    async def _store_procedural(self, entry: MemoryEntry) -> None:
        """Store procedural pattern."""
        # Extract pattern from content
        pattern_key = hashlib.md5(entry.content.encode()).hexdigest()[:16]
        if pattern_key not in self._procedural["patterns"]:
            self._procedural["patterns"][pattern_key] = {
                "count": 0,
                "entries": [],
                "first_seen": entry.created_at,
            }

        pattern = self._procedural["patterns"][pattern_key]
        pattern["count"] += 1
        pattern["entries"].append(entry.entry_id)
        pattern["last_seen"] = entry.created_at

    async def _store_reflective(self, entry: MemoryEntry) -> None:
        """Store reflective log."""
        self._reflective["logs"].append(entry.to_dict())

    # -------------------------------------------------------------------------
    # SEARCH
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
        tier: MemoryTier | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """
        Search across memory tiers.
        Returns merged, ranked results.
        """
        self._stats["searches"] += 1

        all_results = []

        # Determine which tiers to search
        tiers_to_search = [tier] if tier else [
            MemoryTier.SHORT_TERM,
            MemoryTier.WORKING,
            MemoryTier.EPISODIC,
            MemoryTier.SEMANTIC,
            MemoryTier.GRAPH,
        ]

        # Search each tier
        search_tasks = []

        if MemoryTier.SHORT_TERM in tiers_to_search:
            search_tasks.append(self._search_short_term(query, limit, conversation_id))

        if MemoryTier.WORKING in tiers_to_search:
            search_tasks.append(self._search_working(query, limit, conversation_id))

        if MemoryTier.EPISODIC in tiers_to_search and self._episodic:
            search_tasks.append(self._episodic.search(query, limit))

        if MemoryTier.SEMANTIC in tiers_to_search and self._semantic:
            search_tasks.append(self._semantic.search(query, limit))

        if MemoryTier.GRAPH in tiers_to_search and self._graph:
            search_tasks.append(self._graph.search(query, limit))

        # Execute searches in parallel
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Merge results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Tier search error: {result}")
                continue
            if isinstance(result, list):
                all_results.extend(result)

        # Deduplicate by entry_id
        seen = set()
        unique_results = []
        for r in all_results:
            if r.entry.entry_id not in seen:
                seen.add(r.entry.entry_id)
                unique_results.append(r)

        # Sort by score
        unique_results.sort(key=lambda x: x.score, reverse=True)

        # Filter by min_score
        unique_results = [r for r in unique_results if r.score >= min_score]

        # Limit
        unique_results = unique_results[:limit]

        # Update access stats
        for r in unique_results:
            r.entry.accessed_at = time.time()
            r.entry.access_count += 1

        return {
            "results": [
                {
                    "entry_id": r.entry.entry_id,
                    "tier": r.tier.value,
                    "content": r.entry.content,
                    "metadata": r.entry.metadata,
                    "score": r.score,
                    "snippet": r.snippet,
                    "created_at": r.entry.created_at,
                }
                for r in unique_results
            ],
            "query": query,
            "total_found": len(unique_results),
            "tiers_searched": [t.value for t in tiers_to_search],
        }

    async def _search_short_term(
        self,
        query: str,
        limit: int,
        conversation_id: str | None,
    ) -> list[SearchResult]:
        if not self._short_term:
            return []
        return await self._short_term.search(query, limit, conversation_id)

    async def _search_working(
        self,
        query: str,
        limit: int,
        conversation_id: str | None,
    ) -> list[SearchResult]:
        results = []
        key = conversation_id or "default"
        contexts = self._working["contexts"].get(key, [])

        query_lower = query.lower()
        for ctx in contexts:
            if query_lower in ctx.get("content", "").lower():
                entry = MemoryEntry.from_dict(ctx)
                results.append(SearchResult(
                    entry=entry,
                    score=0.8,
                    tier=MemoryTier.WORKING,
                    snippet=entry.content[:200],
                ))

                if len(results) >= limit:
                    break

        return results

    # -------------------------------------------------------------------------
    # WORKING MEMORY HELPERS
    # -------------------------------------------------------------------------

    async def get_working_context(
        self,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> dict:
        """Get current working context."""
        key = conversation_id or task_id or "default"
        return {
            "context": self._working["contexts"].get(key, []),
            "key": key,
        }

    async def update_working_context(
        self,
        key: str,
        data: dict,
    ) -> None:
        """Update working memory context."""
        if key not in self._working["contexts"]:
            self._working["contexts"][key] = []

        # Merge with existing
        existing = self._working["contexts"][key]
        if existing and isinstance(existing[-1], dict):
            existing[-1].update(data)
        else:
            entry = MemoryEntry(
                tier=MemoryTier.WORKING,
                content=json.dumps(data),
                metadata=data,
            )
            existing.append(entry.to_dict())

    # -------------------------------------------------------------------------
    # CONSOLIDATION
    # -------------------------------------------------------------------------

    async def _consolidation_loop(self) -> None:
        """Periodic memory consolidation."""
        interval = self.config.consolidate_interval_minutes * 60

        while self._running:
            try:
                await asyncio.sleep(interval)
                await self.consolidate()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation error: {e}")

    async def consolidate(self) -> dict:
        """
        Consolidate memories across tiers:
        - Short-term → Episodic (important conversations)
        - Working → Procedural (repeated patterns)
        - Episodic → Graph (entity extraction)
        - Reflective log generation
        """
        self._stats["consolidations"] += 1
        results = {"promoted": 0, "patterns_found": 0, "entities_extracted": 0}

        # 1. Short-term → Episodic (high importance)
        if self._short_term and self._episodic:
            promoted = await self._consolidate_short_term_to_episodic()
            results["promoted"] += promoted

        # 2. Working → Procedural (repeated patterns)
        patterns = await self._consolidate_working_to_procedural()
        results["patterns_found"] += patterns

        # 3. Episodic → Graph (entity extraction)
        if self._episodic and self._graph:
            entities = await self._consolidate_episodic_to_graph()
            results["entities_extracted"] += entities

        # 4. Generate reflective log
        await self._generate_reflective_log()

        logger.info(f"Consolidation complete: {results}")
        return results

    async def _consolidate_short_term_to_episodic(self) -> int:
        """Promote important short-term memories to episodic."""
        promoted = 0

        # Get recent short-term entries
        entries = await self._short_term.get_recent(limit=50)

        for entry in entries:
            if entry.importance > 0.7:  # High importance threshold
                # Store in episodic
                await self._episodic.add(entry)
                promoted += 1

        return promoted

    async def _consolidate_working_to_procedural(self) -> int:
        """Extract repeated patterns from working memory."""
        patterns_found = 0

        for pattern_key, pattern in self._procedural["patterns"].items():
            if pattern["count"] >= self._procedural["min_occurrences"]:
                # This is a learned pattern
                patterns_found += 1

        return patterns_found

    async def _consolidate_episodic_to_graph(self) -> int:
        """Extract entities and relationships from episodic memories."""
        if not self._graph:
            return 0

        # This would use GraphRAG to extract entities
        # Simplified - return count
        return 0

    async def _generate_reflective_log(self) -> None:
        """Generate self-reflection log entry."""
        if not self._reflective:
            return

        log_entry = MemoryEntry(
            tier=MemoryTier.REFLECTIVE,
            content=json.dumps({
                "timestamp": time.time(),
                "stats": self._stats,
                "memory_health": self._assess_memory_health(),
            }),
            metadata={"type": "reflection"},
            importance=0.5,
        )

        self._reflective["logs"].append(log_entry.to_dict())

    def _assess_memory_health(self) -> dict:
        """Assess overall memory system health."""
        return {
            "total_entries": self._stats["total_entries"],
            "tier_distribution": self._stats["entries_by_tier"],
            "search_efficiency": self._stats["searches"] / max(self._stats["total_entries"], 1),
            "consolidation_rate": self._stats["consolidations"],
        }

    # -------------------------------------------------------------------------
    # FORGETTING / DECAY
    # -------------------------------------------------------------------------

    async def apply_decay(self) -> int:
        """Apply forgetting curve to old, low-importance memories."""
        removed = 0
        now = time.time()

        # This would iterate through tiers and remove decayed entries
        # Simplified implementation
        return removed

    # -------------------------------------------------------------------------
    # STATS & HEALTH
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "tiers": {
                "short_term": "active" if self._short_term else "inactive",
                "working": "active",
                "episodic": "active" if self._episodic else "inactive",
                "semantic": "active" if self._semantic else "inactive",
                "graph": "active" if self._graph else "inactive",
                "procedural": "active" if self._procedural else "inactive",
                "reflective": "active" if self._reflective else "inactive",
            },
        }

    async def health_check(self) -> dict:
        health = {"healthy": True, "tiers": {}}

        for tier_name in ["_short_term", "_episodic", "_semantic", "_graph"]:
            tier = getattr(self, tier_name, None)
            if tier and hasattr(tier, "health_check"):
                health["tiers"][tier_name] = await tier.health_check()
            elif tier:
                health["tiers"][tier_name] = "active"
            else:
                health["tiers"][tier_name] = "inactive"

        return health


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MemoryManager",
    "MemoryTier",
    "MemoryEntry",
    "MemoryConfig",
    "SearchResult",
]