"""
memory/session.py

SessionMemory: multi-turn conversation history backed by SQLite (stdlib —
no external DB server needed). Each turn (one user message + one assistant
response) is stored as two rows so role/content/intent/timestamp are all
queryable per-message, not just per-turn.

Input:  session_id, role ("user"/"assistant"), content, intent (optional)
Output: ordered conversation history, formatted as a string ready to inject
        into the generator's prompt (see agent/nodes/generator.py's
        `conversation_history` field, populated by the API layer before
        invoking the graph — see api/routes/chat.py)

Schema:
    sessions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        turn_number INTEGER NOT NULL,
        role TEXT NOT NULL,            -- 'user' or 'assistant'
        content TEXT NOT NULL,
        intent TEXT,                   -- only set on 'user' rows, nullable
        timestamp TEXT NOT NULL        -- ISO 8601
    )
"""

import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
from contextlib import contextmanager

from config.settings import settings

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    intent TEXT,
    timestamp TEXT NOT NULL
);
"""
CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
"""


class SessionMemory:
    """
    Thread-safe-enough for a single-process FastAPI app: opens a fresh
    connection per operation rather than holding one open across requests,
    which sidesteps SQLite's threading restrictions without needing a
    connection pool for a project at this scale.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.session_db_path
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)
            conn.execute(CREATE_INDEX_SQL)
        logger.info("SessionMemory initialized at '%s'", self.db_path)

    def _next_turn_number(self, session_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT COALESCE(MAX(turn_number), 0) FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            max_turn = cursor.fetchone()[0]
        return max_turn + 1

    def add_turn(
        self, session_id: str, role: str, content: str, intent: Optional[str] = None
    ) -> None:
        """
        Adds a single message row. Call this twice per conversational turn:
        once for the user's message (role='user', intent=classified intent)
        and once for the assistant's reply (role='assistant', intent=None).
        Both rows share the same turn_number.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{role}'")

        turn_number = self._next_turn_number(session_id) if role == "user" else self._current_turn_number(session_id)
        timestamp = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, turn_number, role, content, intent, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, turn_number, role, content, intent, timestamp),
            )
        logger.debug("Added %s turn %d for session '%s'", role, turn_number, session_id)

    def _current_turn_number(self, session_id: str) -> int:
        """Used when adding the assistant's reply — reuses the latest turn number."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT COALESCE(MAX(turn_number), 1) FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            return cursor.fetchone()[0]

    def get_history(self, session_id: str, last_n: int = 6) -> List[Dict]:
        """
        Returns the last `last_n` TURNS (not rows) as a list of
        {"role", "content", "intent", "timestamp"} dicts, oldest first.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT turn_number, role, content, intent, timestamp FROM sessions "
                "WHERE session_id = ? ORDER BY turn_number DESC, id DESC LIMIT ?",
                (session_id, last_n * 2),  # *2 because each turn has 2 rows (user+assistant)
            )
            rows = cursor.fetchall()

        rows.reverse()  # back to chronological order
        return [
            {"turn_number": r[0], "role": r[1], "content": r[2], "intent": r[3], "timestamp": r[4]}
            for r in rows
        ]

    def get_history_as_text(self, session_id: str, last_n: int = 6) -> str:
        """
        Formats history as a simple "Role: content" block, ready to inject
        directly into the generator's {history} prompt variable.
        """
        history = self.get_history(session_id, last_n)
        if not history:
            return "(no prior turns)"
        lines = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in history]
        return "\n".join(lines)

    def clear_session(self, session_id: str) -> int:
        """Deletes all rows for a session. Returns number of rows deleted."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            deleted = cursor.rowcount
        logger.info("Cleared session '%s' (%d rows deleted)", session_id, deleted)
        return deleted

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1", (session_id,)
            )
            return cursor.fetchone() is not None
