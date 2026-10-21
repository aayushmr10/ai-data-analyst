"""
In-memory dataset session store.

Each uploaded dataset gets a UUID. We keep the raw dataframe and the
current (possibly cleaned) dataframe in memory, keyed by that id, so the
frontend can call /profile, /stats, /charts, /ml etc. without re-uploading.

This is intentionally simple (no DB, no persistence across restarts) --
good enough for a single-user local / demo deployment. Swap for Redis or
Postgres-backed storage if this needs to survive restarts or scale to
multiple workers.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class DatasetSession:
    id: str
    filename: str
    raw_df: pd.DataFrame
    df: pd.DataFrame  # working copy, mutated by cleaning operations
    created_at: datetime = field(default_factory=datetime.utcnow)
    cleaning_log: list[str] = field(default_factory=list)
    last_access: float = field(default_factory=time.time)


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, default)))
    except ValueError:
        return default


class DatasetStore:
    """
    In-memory sessions. On a public host this MUST be bounded, so sessions expire after
    SESSION_TTL_MIN idle minutes (0 = never) and at most MAX_SESSIONS are kept
    (oldest-idle evicted first). Clients treat an evicted session as "upload again".
    """

    def __init__(self) -> None:
        self._sessions: dict[str, DatasetSession] = {}
        self._lock = threading.Lock()

    def _evict(self, now: float) -> None:
        ttl = _env_int("SESSION_TTL_MIN", 60) * 60
        if ttl:
            for sid in [k for k, v in self._sessions.items() if now - v.last_access > ttl]:
                del self._sessions[sid]
        cap = _env_int("MAX_SESSIONS", 25)
        while cap and len(self._sessions) >= cap:
            oldest = min(self._sessions, key=lambda k: self._sessions[k].last_access)
            del self._sessions[oldest]

    def create(self, filename: str, df: pd.DataFrame) -> DatasetSession:
        session = DatasetSession(id=str(uuid.uuid4()), filename=filename, raw_df=df.copy(), df=df.copy())
        with self._lock:
            self._evict(time.time())
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[DatasetSession]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            now = time.time()
            ttl = _env_int("SESSION_TTL_MIN", 60) * 60
            if ttl and now - session.last_access > ttl:
                del self._sessions[session_id]
                return None
            session.last_access = now
            return session

    def require(self, session_id: str) -> DatasetSession:
        session = self.get(session_id)
        if session is None:
            raise KeyError(f"No dataset session found for id={session_id}")
        return session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())


# Single process-wide store instance
store = DatasetStore()
