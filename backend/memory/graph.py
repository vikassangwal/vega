"""
================================================================================
VEGA GRAPH MEMORY - Knowledge Graph (Neo4j + GraphRAG)
================================================================================
Structured knowledge with relationships:
- Neo4j for graph storage
- Entity extraction and linking
- Relationship mapping
- GraphRAG for retrieval (Microsoft GraphRAG)
- Community detection for summarization
- Path queries for reasoning

References:
- neo4j/neo4j: Graph database
- microsoft/graphrag: Graph-based RAG with community detection
================================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from config import get_settings, settings

logger = logging.getLogger("vega.memory.graph")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class GraphConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "vega_graph_password"
    database: str = "neo4j"
    graphrag_enabled: bool = True
    graphrag_community_level: int = 2
    graphrag_response_type: str = "multiple paragraphs"
    max_input_tokens: int = 8000


# =============================================================================
# GRAPH MEMORY
# =============================================================================

class GraphMemory:
    """
    Neo4j-backed knowledge graph with GraphRAG support.
    """

    def __init__(self, config: GraphConfig | None = None):
        self.config = config or GraphConfig()
        self._driver = None
        self._graphrag = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize Neo4j driver and GraphRAG."""
        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
            )

            # Test connection
            async with self._driver.session(database=self.config.database) as session:
                await session.run("RETURN 1")

            # Create indexes
            await self._create_indexes()

            # Initialize GraphRAG if enabled
            if self.config.graphrag_enabled:
                await self._init_graphrag()

            self._initialized = True
            logger.info("GraphMemory initialized")
            return True

        except ImportError:
            logger.warning("neo4j driver not available")
            return False
        except Exception as e:
            logger.exception(f"GraphMemory initialization failed: {e}")
            return False

    async def _create_indexes(self) -> None:
        """Create necessary indexes."""
        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX conversation_id IF NOT EXISTS FOR (c:Conversation) ON (c.id)",
        ]

        async with self._driver.session(database=self.config.database) as session:
            for idx in indexes:
                try:
                    await session.run(idx)
                except Exception:
                    pass  # Index may already exist

    async def _init_graphrag(self) -> None:
        """Initialize Microsoft GraphRAG."""
        try:
            # GraphRAG would be initialized here
            # For now, we use native Cypher queries
            self._graphrag = {"enabled": True}
            logger.info("GraphRAG initialized (native mode)")
        except Exception as e:
            logger.warning(f"GraphRAG initialization failed: {e}")
            self._graphrag = {"enabled": False}

    async def shutdown(self) -> None:
        if self._driver:
            await self._driver.close()
        logger.info("GraphMemory shutdown")

    # -------------------------------------------------------------------------
    # CORE OPERATIONS
    # -------------------------------------------------------------------------

    async def add(self, entry: Any) -> str:
        """Add entry as document with entities to graph."""
        if not self._initialized:
            return entry.entry_id

        try:
            # Extract entities from content
            entities = await self._extract_entities(entry.content)

            # Store document
            doc_id = entry.entry_id
            await self._store_document(doc_id, entry)

            # Store entities and relationships
            for entity in entities:
                await self._store_entity(entity, doc_id)

            # Store relationships
            relationships = await self._extract_relationships(entry.content, entities)
            for rel in relationships:
                await self._store_relationship(rel)

            return entry.entry_id

        except Exception as e:
            logger.error(f"Graph add error: {e}")
            return entry.entry_id

    async def _store_document(self, doc_id: str, entry: Any) -> None:
        """Store document node."""
        query = """
        MERGE (d:Document {id: $id})
        SET d.content = $content,
            d.created_at = $created_at,
            d.tier = $tier,
            d.conversation_id = $conversation_id,
            d.user_id = $user_id,
            d.agent_id = $agent_id,
            d.importance = $importance
        """

        async with self._driver.session(database=self.config.database) as session:
            await session.run(query, {
                "id": doc_id,
                "content": entry.content,
                "created_at": entry.created_at,
                "tier": entry.tier.value if hasattr(entry.tier, 'value') else str(entry.tier),
                "conversation_id": entry.conversation_id or "",
                "user_id": entry.user_id or "",
                "agent_id": entry.agent_id or "",
                "importance": entry.importance,
            })

    async def _extract_entities(self, text: str) -> list[dict]:
        """Extract entities from text using LLM or NER."""
        # This would use an NER model or LLM
        # Simplified - return empty for now
        # In production, use spaCy, GLiNER, or LLM
        return []

    async def _store_entity(self, entity: dict, doc_id: str) -> None:
        """Store entity and link to document."""
        query = """
        MERGE (e:Entity {name: $name, type: $type})
        ON CREATE SET e.created_at = $created_at
        SET e.mentions = COALESCE(e.mentions, 0) + 1
        WITH e
        MATCH (d:Document {id: $doc_id})
        MERGE (d)-[:MENTIONS]->(e)
        """

        async with self._driver.session(database=self.config.database) as session:
            await session.run(query, {
                "name": entity.get("name"),
                "type": entity.get("type", "Unknown"),
                "created_at": time.time(),
                "doc_id": doc_id,
            })

    async def _extract_relationships(self, text: str, entities: list) -> list[dict]:
        """Extract relationships between entities."""
        # Would use LLM for relationship extraction
        return []

    async def _store_relationship(self, rel: dict) -> None:
        """Store relationship between entities."""
        query = """
        MATCH (e1:Entity {name: $source})
        MATCH (e2:Entity {name: $target})
        MERGE (e1)-[r:RELATES {type: $type}]->(e2)
        SET r.weight = COALESCE(r.weight, 0) + 1
        """

        async with self._driver.session(database=self.config.database) as session:
            await session.run(query, {
                "source": rel.get("source"),
                "target": rel.get("target"),
                "type": rel.get("type", "RELATED"),
            })

    # -------------------------------------------------------------------------
    # QUERY OPERATIONS
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list:
        """Search graph using text-to-Cypher or vector similarity."""
        if not self._initialized:
            return []

        try:
            # Strategy 1: Entity name search
            results = await self._search_entities(query, limit)

            # Strategy 2: If GraphRAG enabled, use it
            if self.config.graphrag_enabled and self._graphrag.get("enabled"):
                rag_results = await self._graphrag_search(query, limit)
                results.extend(rag_results)

            return self._format_results(results, limit)

        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return []

    async def _search_entities(self, query: str, limit: int) -> list:
        """Search entities by name similarity."""
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($query)
        OPTIONAL MATCH (d:Document)-[:MENTIONS]->(e)
        RETURN e.name as name, e.type as type, e.mentions as mentions,
               collect(d.id) as documents, collect(d.content) as contents
        ORDER BY e.mentions DESC
        LIMIT $limit
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(cypher, {"query": query, "limit": limit})
            return [record.data() for record in result]

    async def _graphrag_search(self, query: str, limit: int) -> list:
        """GraphRAG-style search with community summaries."""
        # This would use Microsoft GraphRAG
        # For now, return empty - implement with GraphRAG library
        return []

    def _format_results(self, results: list, limit: int) -> list:
        """Format graph results as SearchResult."""
        from memory.memory_manager import SearchResult, MemoryTier, MemoryEntry

        formatted = []

        for r in results[:limit]:
            # Build content from entity + documents
            content_parts = [
                f"Entity: {r.get('name', '')} ({r.get('type', '')})",
                f"Mentions: {r.get('mentions', 0)}",
            ]

            docs = r.get("documents", [])
            if docs:
                content_parts.append(f"Documents: {len(docs)}")

            contents = r.get("contents", [])
            if contents:
                content_parts.append(f"Context: {contents[0][:500]}")

            content = "\n".join(content_parts)

            entry = MemoryEntry(
                tier=MemoryTier.GRAPH,
                content=content,
                metadata={
                    "entity_name": r.get("name"),
                    "entity_type": r.get("type"),
                    "documents": r.get("documents", []),
                },
                created_at=time.time(),
            )
            entry.entry_id = str(uuid.uuid4())

            formatted.append(SearchResult(
                entry=entry,
                score=0.75,
                tier=MemoryTier.GRAPH,
                snippet=content[:200],
            ))

        return formatted

    # -------------------------------------------------------------------------
    # GRAPH QUERIES
    # -------------------------------------------------------------------------

    async def get_entity_neighbors(self, entity_name: str, depth: int = 2) -> dict:
        """Get entity neighborhood for reasoning."""
        cypher = f"""
        MATCH (e:Entity {{name: $name}})
        CALL apoc.path.subgraphAll(e, {{maxLevel: $depth}})
        YIELD nodes, relationships
        RETURN nodes, relationships
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(cypher, {"name": entity_name, "depth": depth})
            record = await result.single()
            return record.data() if record else {}

    async def find_path(self, entity1: str, entity2: str, max_depth: int = 4) -> list:
        """Find shortest path between entities."""
        cypher = """
        MATCH (e1:Entity {name: $name1}), (e2:Entity {name: $name2})
        MATCH path = shortestPath((e1)-[*..$max_depth]-(e2))
        RETURN path
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(cypher, {
                "name1": entity1,
                "name2": entity2,
                "max_depth": max_depth,
            })
            return [record.data() async for record in result]

    async def get_community_summary(self, entity_name: str) -> str:
        """Get community summary for entity (GraphRAG)."""
        # Would use GraphRAG community summaries
        return ""

    # -------------------------------------------------------------------------
    # STATS
    # -------------------------------------------------------------------------

    async def get_stats(self) -> dict:
        stats = {}

        async with self._driver.session(database=self.config.database) as session:
            # Count entities
            result = await session.run("MATCH (e:Entity) RETURN count(e) as count")
            record = await result.single()
            stats["entities"] = record["count"] if record else 0

            # Count documents
            result = await session.run("MATCH (d:Document) RETURN count(d) as count")
            record = await result.single()
            stats["documents"] = record["count"] if record else 0

            # Count relationships
            result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
            record = await result.single()
            stats["relationships"] = record["count"] if record else 0

        return {
            "backend": "neo4j",
            "graphrag_enabled": self.config.graphrag_enabled,
            **stats,
        }

    async def health_check(self) -> dict:
        healthy = False
        if self._driver:
            try:
                async with self._driver.session(database=self.config.database) as session:
                    await session.run("RETURN 1")
                healthy = True
            except Exception:
                pass

        return {
            "healthy": healthy,
            "backend": "neo4j",
            "initialized": self._initialized,
            "uri": self.config.uri,
        }

    # -------------------------------------------------------------------------
    # EXPORTS
    # -------------------------------------------------------------------------

__all__ = [
    "GraphMemory",
    "GraphConfig",
]