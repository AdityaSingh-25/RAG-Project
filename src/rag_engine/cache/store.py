"""Single-file SQLite key/value cache with TTL.

Why SQLite and not Redis: the project ships as Docker Compose for local
development and the surrounding infrastructure already includes Qdrant and
Ollama. Adding a Redis dependency for caches that are inherently throwaway
isn't worth the operational cost. SQLite gives us atomic writes, a single
file we can ``rm`` to reset, and zero new processes.

The store is thread-safe via short-lived connections per call. Each row is
keyed by ``(namespace, key)`` so multiple cache wrappers can share one file
without name collisions.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value BLOB NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (namespace, key)
);
"""


class CacheStore:
    """Thread-safe SQLite-backed cache. One instance per process is enough."""

    def __init__(self, path: Path, default_ttl_seconds: int) -> None:
        self.path = path
        self.default_ttl_seconds = default_ttl_seconds
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get(self, namespace: str, key: str) -> Any | None:
        """Return the cached value or ``None`` if the row is missing or expired."""
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
        if row is None:
            return None
        value_blob, expires_at = row
        if expires_at <= now:
            self.delete(namespace, key)
            return None
        return json.loads(value_blob)

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        expires_at = time.time() + ttl
        blob = json.dumps(value, default=_json_default)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (namespace, key, value, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (namespace, key, blob, expires_at),
            )
            conn.commit()

    def delete(self, namespace: str, key: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM cache WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
            conn.commit()

    def clear(self, namespace: str | None = None) -> int:
        with self._lock, self._connect() as conn:
            if namespace is None:
                cursor = conn.execute("DELETE FROM cache")
            else:
                cursor = conn.execute("DELETE FROM cache WHERE namespace = ?", (namespace,))
            conn.commit()
            return cursor.rowcount

    def purge_expired(self) -> int:
        now = time.time()
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))
            conn.commit()
            return cursor.rowcount


def _json_default(obj: Any) -> Any:
    """Allow common non-JSON types (e.g., tuples) to round-trip."""
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Unserializable value of type {type(obj)!r}")
