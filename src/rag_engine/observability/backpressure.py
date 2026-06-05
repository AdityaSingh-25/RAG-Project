"""Concurrency limiters used to apply backpressure on heavy API endpoints.

A limiter tracks how many requests are currently inside its critical section
and refuses new entrants past a configured ceiling. Refusal is intentionally
*fail-fast* (no queueing): callers receive a 429 immediately rather than
piling up on the server, which keeps p95 latency bounded under overload
instead of letting the queue drift toward timeout.

The limiter does not own the HTTP layer — endpoint code catches
`BackpressureError` and translates it into an HTTPException with a
`Retry-After` header. That keeps this module HTTP-agnostic and testable in
isolation.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class BackpressureError(Exception):
    """Raised when a limiter is at capacity. Carries enough context for the
    HTTP layer to build a meaningful 429 body."""

    def __init__(self, name: str, in_flight: int, limit: int) -> None:
        super().__init__(
            f"{name} limiter at capacity: {in_flight}/{limit} in flight"
        )
        self.name = name
        self.in_flight = in_flight
        self.limit = limit


class ConcurrencyLimiter:
    """Counts active operations and rejects past ``limit``.

    Uses an ``asyncio.Lock`` to make the read-check-increment sequence
    atomic across coroutines on the same event loop. The lock is created
    lazily so the limiter can be instantiated at import time (before any
    loop is running).
    """

    def __init__(self, name: str, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        self.name = name
        self.limit = limit
        self._in_flight = 0
        self._rejected_total = 0
        self._accepted_total = 0
        self._lock: asyncio.Lock | None = None

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire_slot(self) -> None:
        """Reserve a slot or raise ``BackpressureError``.

        Caller is responsible for ``release_slot()`` — use the
        :meth:`acquire` context manager unless you need manual control
        (e.g., to span the slot across a function boundary like an SSE
        StreamingResponse, where acquire and release live in different
        coroutines).
        """
        lock = self._ensure_lock()
        async with lock:
            if self._in_flight >= self.limit:
                self._rejected_total += 1
                raise BackpressureError(self.name, self._in_flight, self.limit)
            self._in_flight += 1
            self._accepted_total += 1

    async def release_slot(self) -> None:
        lock = self._ensure_lock()
        async with lock:
            self._in_flight -= 1

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Reserve a slot for the lifetime of a single ``async with`` block.

        Decrement is guaranteed even on exception, so a crashed handler
        can't leak slots.
        """
        await self.acquire_slot()
        try:
            yield
        finally:
            await self.release_slot()

    def snapshot(self) -> dict[str, int]:
        """Cheap, lock-free read for /metrics. Counts are monotonic ints,
        so a torn read at worst reports a stale value — never corrupt."""
        return {
            "limit": self.limit,
            "in_flight": self._in_flight,
            "accepted_total": self._accepted_total,
            "rejected_total": self._rejected_total,
        }
