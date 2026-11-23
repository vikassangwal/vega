"""
================================================================================
VEGA MEMORY - 7-Tier Persistence Package
================================================================================
Multi-tier memory architecture:
1. SHORT-TERM: Conversation buffer (last N turns)
2. WORKING: Active task context
3. EPISODIC: Timestamped event logs (Mem0)
4. SEMANTIC: Vector search (ChromaDB)
5. GRAPH: Knowledge Graph (Neo4j + GraphRAG)
6. PROCEDURAL: Learned routines & automations
7. REFLECTIVE: Self-evaluation & improvement logs

References:
- mem0ai/mem0: Persistent user memory, preference learning
- chroma-core/chroma: Vector embeddings, semantic search
- neo4j/neo4j: Graph database, relationship mapping
- microsoft/graphrag: Graph-based RAG, community detection
- OpenJarvis/OpenJarvis: Memory presets, daily briefing context
================================================================================
"""

from .memory_manager import MemoryManager, MemoryTier, MemoryEntry, MemoryConfig
from .short_term import ShortTermMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .graph import GraphMemory

__all__ = [
    "MemoryManager",
    "MemoryTier",
    "MemoryEntry",
    "MemoryConfig",
    "ShortTermMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "GraphMemory",
]