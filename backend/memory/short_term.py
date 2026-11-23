"""
================================================================================
VEGA SHORT-TERM MEMORY - Conversation Buffer
================================================================================
Maintains recent conversation history with token-aware management.
- Rolling buffer of last N turns
- Token counting for LLM context window management
- Conversation-scoped isolation
- Fast retrieval for context injection

References:
- OpenJarvis/OpenJarvis: Conversation buffer patterns
- langchain-ai/langgraph: State management
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from config import get_settings, settings

logger = logging.getLogger("vega.memory.short_term")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ConversationTurn:
    """Single conversation turn."""
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "user"  # user, assistant, system
    content: str = ""
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class ShortTermConfig:
    max_turns: int = 20
    max_tokens: int = 8000
    tokenizer_model: str = "gpt-4"  # For token estimation


# =============================================================================
# SHORT-TERM MEMORY
# =============================================================================

class ShortTermMemory:
    """
    Rolling conversation buffer with token awareness.
    Maintains separate buffers per conversation.
    """

    def __init__(self, config: ShortTermConfig | None = None):
        self.config = config or ShortTermConfig()

        # Conversation buffers: conversation_id -> deque of turns
        self._buffers: dict[str, deque] = {}

        # Token estimator
        self._tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        """Initialize tokenizer for accurate counting."""
        try:
            import tiktoken
            self._tokenizer = tiktoken.encoding_for_model(self.config.tokenizer_model)
        except Exception:
            logger.warning("tiktoken not available, using rough estimation")
            self._tokenizer = None

    async def initialize(self) -> bool:
        logger.info("ShortTermMemory initialized")
        return True

    async def shutdown(self) -> None:
        self._buffers.clear()
        logger.info("ShortTermMemory shutdown")

    # -------------------------------------------------------------------------
    # CORE OPERATIONS
    # -------------------------------------------------------------------------

    def _get_buffer(self, conversation_id: str) -> deque:
        """Get or create buffer for conversation."""
        if conversation_id not in self._buffers:
            self._buffers[conversation_id] = deque(maxlen=self.config.max_turns)
        return self._buffers[conversation_id]

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Rough estimation: ~4 chars per token
        return len(text) // 4

    async def add(self, entry: Any) -> None:
        """Add entry to conversation buffer."""
        # Extract conversation_id
        conversation_id = getattr(entry, "conversation_id", None) or "default"

        # Create turn
        turn = ConversationTurn(
            role=entry.metadata.get("role", "user"),
            content=entry.content,
            tokens=self._count_tokens(entry.content),
            metadata=entry.metadata,
        )

        buffer = self._get_buffer(conversation_id)
        buffer.append(turn)

        # Enforce token limit
        await self._enforce_token_limit(conversation_id)

    async def _enforce_token_limit(self, conversation_id: str) -> None:
        """Remove oldest turns until under token limit."""
        buffer = self._buffers.get(conversation_id)
        if not buffer:
            return

        total_tokens = sum(t.tokens for t in buffer)

        while total_tokens > self.config.max_tokens and len(buffer) > 1:
            removed = buffer.popleft()
            total_tokens -= removed.tokens

    # -------------------------------------------------------------------------
    # RETRIEVAL
    # -------------------------------------------------------------------------

    async def get_recent(
        self,
        conversation_id: str = "default",
        limit: int | None = None,
    ) -> list[Any]:
        """Get recent conversation turns."""
        buffer = self._buffers.get(conversation_id, deque())

        turns = list(buffer)
        if limit:
            turns = turns[-limit:]

        # Convert to MemoryEntry-like objects
        from memory.memory_manager import MemoryEntry, MemoryTier
        entries = []
        for turn in turns:
            entry = MemoryEntry(
                tier=MemoryTier.SHORT_TERM,
                content=turn.content,
                metadata={"role": turn.role, **turn.metadata},
                conversation_id=getattr(turn, "conversation_id", "default"),
                created_at=turn.timestamp,
            )
            entry.entry_id = turn.turn_id
            entries.append(entry)

        return entries

    async def get_conversation_context(
        self,
        conversation_id: str = "default",
        max_tokens: int | None = None,
    ) -> list[dict]:
        """Get formatted conversation context for LLM."""
        buffer = self._buffers.get(conversation_id, deque())
        turns = list(buffer)

        # Build context from recent turns
        context = []
        token_budget = max_tokens or self.config.max_tokens
        used_tokens = 0

        for turn in reversed(turns):
            turn_tokens = turn.tokens
            if used_tokens + turn_tokens > token_budget:
                break
            context.insert(0, {"role": turn.role, "content": turn.content})
            used_tokens += turn_tokens

        return context

    async def search(
        self,
        query: str,
        limit: int = 10,
        conversation_id: str | None = None,
    ) -> list[Any]:
        """Search conversation history."""
        from memory.memory_manager import SearchResult, MemoryTier

        results = []
        query_lower = query.lower()

        conv_ids = [conversation_id] if conversation_id else list(self._buffers.keys())

        for conv_id in conv_ids:
            buffer = self._buffers.get(conv_id, deque())

            for turn in buffer:
                if query_lower in turn.content.lower():
                    from memory.memory_manager import MemoryEntry, MemoryTier
                    entry = MemoryEntry(
                        tier=MemoryTier.SHORT_TERM,
                        content=turn.content,
                        metadata={"role": turn.role, **turn.metadata},
                        conversation_id=conv_id,
                        created_at=turn.timestamp,
                    )
                    entry.entry_id = turn.turn_id

                    results.append(SearchResult(
                        entry=entry,
                        score=0.9,
                        tier=MemoryTier.SHORT_TERM,
                        snippet=turn.content[:200],
                    ))

                    if len(results) >= limit:
                        break

        return results

    # -------------------------------------------------------------------------
    # MANAGEMENT
    # -------------------------------------------------------------------------

    async def clear(self, conversation_id: str | None = None) -> None:
        """Clear conversation buffer(s)."""
        if conversation_id:
            self._buffers.pop(conversation_id, None)
        else:
            self._buffers.clear()

    async def get_stats(self) -> dict:
        """Get buffer statistics."""
        total_turns = sum(len(b) for b in self._buffers.values())
        total_tokens = sum(
            sum(t.tokens for t in b) for b in self._buffers.values()
        )

        return {
            "conversations": len(self._buffers),
            "total_turns": total_turns,
            "total_tokens": total_tokens,
            "max_turns_per_conv": self.config.max_turns,
            "max_tokens_per_conv": self.config.max_tokens,
        }

    async def health_check(self) -> dict:
        return {
            "healthy": True,
            "stats": await self.get_stats(),
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ShortTermMemory",
    "ShortTermConfig",
    "ConversationTurn",
]