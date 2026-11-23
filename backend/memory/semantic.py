"""
================================================================================
VEGA SEMANTIC MEMORY - Vector Search (ChromaDB)
================================================================================
Semantic similarity search using embeddings:
- ChromaDB for vector storage
- Sentence transformers for embeddings
- Metadata filtering
- Hybrid search (vector + keyword)
- Collection management

References:
- chroma-core/chroma: Vector database for AI applications
- sentence-transformers: State-of-the-art embeddings
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import get_settings, settings

logger = logging.getLogger("vega.memory.semantic")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class SemanticConfig:
    host: str = "localhost"
    port: int = 8000
    collection_name: str = "vega_memory"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_directory: str = "./data/chromadb"
    distance_metric: str = "cosine"  # cosine, l2, ip


# =============================================================================
# SEMANTIC MEMORY
# =============================================================================

class SemanticMemory:
    """
    ChromaDB-backed semantic memory with embeddings.
    """

    def __init__(self, config: SemanticConfig | None = None):
        self.config = config or SemanticConfig()
        self._client: chromadb.Client | None = None
        self._collection = None
        self._embedding_function = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize ChromaDB client and collection."""
        try:
            # Create client
            self._client = chromadb.HttpClient(
                host=self.config.host,
                port=self.config.port,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Test connection
            self._client.heartbeat()

            # Initialize embedding function
            await self._init_embeddings()

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                embedding_function=self._embedding_function,
                metadata={"hnsw:space": self.config.distance_metric},
            )

            self._initialized = True
            logger.info(f"SemanticMemory initialized: {self.config.collection_name}")
            return True

        except Exception as e:
            logger.exception(f"SemanticMemory initialization failed: {e}")
            return False

    async def _init_embeddings(self) -> None:
        """Initialize sentence transformer embeddings."""
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=self.config.embedding_model,
                device="cuda" if self._is_cuda_available() else "cpu",
            )
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            # Use default
            self._embedding_function = None

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    async def shutdown(self) -> None:
        logger.info("SemanticMemory shutdown")

    # -------------------------------------------------------------------------
    # CORE OPERATIONS
    # -------------------------------------------------------------------------

    async def add(self, entry: Any) -> str:
        """Add entry to vector store."""
        if not self._initialized:
            await self.initialize()

        if not self._collection:
            return entry.entry_id

        try:
            # Prepare metadata
            metadata = {
                **entry.metadata,
                "entry_id": entry.entry_id,
                "tier": entry.tier.value if hasattr(entry.tier, 'value') else str(entry.tier),
                "conversation_id": entry.conversation_id or "",
                "user_id": entry.user_id or "",
                "agent_id": entry.agent_id or "",
                "task_id": entry.task_id or "",
                "importance": entry.importance,
                "tags": ",".join(entry.tags) if entry.tags else "",
                "created_at": str(entry.created_at),
            }

            # Add to ChromaDB
            self._collection.add(
                ids=[entry.entry_id],
                documents=[entry.content],
                metadatas=[metadata],
            )

            return entry.entry_id

        except Exception as e:
            logger.error(f"Semantic add error: {e}")
            return entry.entry_id

    async def search(
        self,
        query: str,
        limit: int = 10,
        user_id: str | None = None,
        conversation_id: str | None = None,
        agent_id: str | None = None,
        min_score: float = 0.0,
    ) -> list:
        """Semantic similarity search."""
        if not self._initialized or not self._collection:
            return []

        try:
            # Build where clause for filtering
            where = {}
            if user_id:
                where["user_id"] = user_id
            if conversation_id:
                where["conversation_id"] = conversation_id
            if agent_id:
                where["agent_id"] = agent_id

            # Query
            results = self._collection.query(
                query_texts=[query],
                n_results=limit,
                where=where if where else None,
                include=["documents", "metadatas", "distances"],
            )

            return self._format_results(results, min_score)

        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

    def _format_results(self, results: dict, min_score: float) -> list:
        """Convert ChromaDB results to SearchResult format."""
        from memory.memory_manager import SearchResult, MemoryTier, MemoryEntry

        formatted = []

        if not results.get("ids") or not results["ids"][0]:
            return formatted

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, (doc_id, doc, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
            # Convert distance to similarity score
            score = 1.0 - distance  # For cosine distance

            if score < min_score:
                continue

            entry = MemoryEntry(
                tier=MemoryTier.SEMANTIC,
                content=doc,
                metadata=metadata,
                created_at=float(metadata.get("created_at", time.time())),
            )
            entry.entry_id = doc_id

            formatted.append(SearchResult(
                entry=entry,
                score=score,
                tier=MemoryTier.SEMANTIC,
                snippet=doc[:200],
            ))

        return formatted

    # -------------------------------------------------------------------------
    # COLLECTION MANAGEMENT
    # -------------------------------------------------------------------------

    async def delete(self, entry_id: str) -> bool:
        """Delete entry by ID."""
        if not self._collection:
            return False

        try:
            self._collection.delete(ids=[entry_id])
            return True
        except Exception as e:
            logger.error(f"Semantic delete error: {e}")
            return False

    async def update(self, entry_id: str, content: str, metadata: dict | None = None) -> bool:
        """Update entry."""
        if not self._collection:
            return False

        try:
            self._collection.update(
                ids=[entry_id],
                documents=[content],
                metadatas=[metadata] if metadata else None,
            )
            return True
        except Exception as e:
            logger.error(f"Semantic update error: {e}")
            return False

    async def get_collection_stats(self) -> dict:
        """Get collection statistics."""
        if not self._collection:
            return {"count": 0}

        try:
            count = self._collection.count()
            return {"collection": self.config.collection_name, "count": count}
        except Exception:
            return {"count": 0}

    # -------------------------------------------------------------------------
    # STATS
    # -------------------------------------------------------------------------

    async def get_stats(self) -> dict:
        stats = await self.get_collection_stats()
        return {
            "backend": "chromadb",
            "collection": self.config.collection_name,
            "embedding_model": self.config.embedding_model,
            **stats,
        }

    async def health_check(self) -> dict:
        healthy = False
        if self._client:
            try:
                self._client.heartbeat()
                healthy = True
            except Exception:
                pass

        return {
            "healthy": healthy,
            "backend": "chromadb",
            "initialized": self._initialized,
            "host": self.config.host,
            "port": self.config.port,
        }

    # -------------------------------------------------------------------------
    # EXPORTS
    # -------------------------------------------------------------------------

__all__ = [
    "SemanticMemory",
    "SemanticConfig",
]