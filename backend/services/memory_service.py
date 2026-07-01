from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from typing import Any

import structlog

from backend.config import get_settings
from backend.models.schemas import ConversationTurn

logger = structlog.get_logger(__name__)
_settings = get_settings()

# In-memory store keyed by session_id.
# In production, back this with Redis or PostgreSQL.
_store: dict[str, deque[ConversationTurn]] = {}


class MemoryService:
    """Session-scoped conversation memory."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.max_turns = _settings.max_history_turns
        if session_id not in _store:
            _store[session_id] = deque(maxlen=self.max_turns)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_turn(
        self,
        user_query: str,
        final_response: str,
        agent_outputs: dict[str, Any] | None = None,
        intents: list[str] | None = None,
    ) -> ConversationTurn:
        turn = ConversationTurn(
            user_query=user_query,
            final_response=final_response,
            agent_outputs=agent_outputs or {},
            intents=intents or [],
        )
        _store[self.session_id].append(turn)
        logger.info("memory.turn_added", session=self.session_id, turn=turn.turn_id)
        return turn

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_history(self, n: int | None = None) -> list[ConversationTurn]:
        turns = list(_store[self.session_id])
        if n:
            turns = turns[-n:]
        return turns

    def get_history_as_text(self, n: int = 5) -> str:
        turns = self.get_history(n)
        if not turns:
            return "No previous conversation."
        lines: list[str] = []
        for t in turns:
            lines.append(f"User: {t.user_query}")
            lines.append(f"Assistant: {t.final_response}")
        return "\n".join(lines)

    def get_context_summary(self) -> dict[str, Any]:
        turns = self.get_history()
        return {
            "session_id": self.session_id,
            "total_turns": len(turns),
            "recent_intents": [i for t in turns[-3:] for i in t.intents],
            "last_query": turns[-1].user_query if turns else None,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self) -> None:
        _store[self.session_id].clear()

    @staticmethod
    def delete_session(session_id: str) -> None:
        _store.pop(session_id, None)
