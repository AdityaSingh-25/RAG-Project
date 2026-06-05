"""API-key auth + per-key token-bucket rate limiting.

A request hits two gates in order:

1. ``ApiKeyRegistry.authenticate(key)`` — compares the presented key's
   SHA-256 against the configured set. Empty configured set means auth is
   disabled (dev mode), and every request passes through with a
   sentinel "anonymous" identity that still gets rate-limited.
2. ``TokenBucket.try_consume()`` — refills lazily based on wall-clock,
   then atomically debits one token. Failure returns the seconds until
   the next token is available so the HTTP layer can emit a meaningful
   ``Retry-After``.

The rate limiter lives separately from the Phase 22 ``ConcurrencyLimiter``:
backpressure caps *concurrent* in-flight work; rate limiting caps
*per-period* request count. A client may be limited by either, neither,
or both — and the 429 responses are tagged with ``error: "rate_limited"``
vs ``error: "backpressure"`` so callers can tell them apart.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# Sentinel returned by ``ApiKeyRegistry.authenticate`` when auth is
# disabled. Distinct from any real key hash so it can't accidentally
# collide and so logs can show a stable identity.
ANONYMOUS_IDENTITY = "anonymous"


@dataclass
class TokenBucket:
    """Lazy-refill token bucket. One bucket per identity.

    ``rate_per_minute`` is steady-state throughput; ``burst`` is the
    maximum that can be drained instantaneously. A burst of N with a
    rate of R/minute means an idle client can spend N tokens at once
    and then settle into R/minute.
    """

    rate_per_minute: int
    burst: int
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rate_per_minute < 1:
            raise ValueError("rate_per_minute must be >= 1")
        if self.burst < 1:
            raise ValueError("burst must be >= 1")
        # Start full so a brand-new client gets to use its burst budget
        # immediately rather than waiting for the bucket to fill.
        self.tokens = float(self.burst)
        self.last_refill = time.monotonic()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.last_refill)
        if elapsed <= 0:
            return
        gained = elapsed * (self.rate_per_minute / 60.0)
        self.tokens = min(float(self.burst), self.tokens + gained)
        self.last_refill = now

    def try_consume(self, n: int = 1) -> tuple[bool, float]:
        """Atomically take ``n`` tokens.

        Returns ``(ok, retry_after_seconds)`` — on success the second
        element is 0.0; on rejection it's how long the caller should
        wait before the next attempt has a chance.
        """
        now = time.monotonic()
        self._refill(now)
        if self.tokens >= n:
            self.tokens -= n
            return True, 0.0
        deficit = n - self.tokens
        # Refill rate is tokens per second; deficit / rate_per_sec gives wait.
        wait = deficit * (60.0 / self.rate_per_minute)
        return False, wait


class RateLimitExceeded(Exception):
    """Raised by the registry when a caller's bucket is empty."""

    def __init__(self, identity: str, wait_seconds: float, limit_per_minute: int) -> None:
        super().__init__(
            f"rate limit exceeded for {identity}: retry in {wait_seconds:.1f}s"
        )
        self.identity = identity
        self.wait_seconds = wait_seconds
        self.limit_per_minute = limit_per_minute


class ApiKeyRegistry:
    """Holds the set of authorised key hashes and per-identity buckets.

    Keys are stored as SHA-256 hex digests so a memory dump never reveals
    the plaintext. Comparison uses ``secrets.compare_digest`` against
    each configured hash to keep the timing constant.
    """

    def __init__(
        self,
        keys: list[str],
        rate_per_minute: int,
        burst: int,
    ) -> None:
        self._lock = threading.Lock()
        self._rate_per_minute = rate_per_minute
        self._burst = burst
        # Map sha256 → "" (we never need the plaintext back, just membership).
        # Stored as a tuple of hashes for constant-time iteration.
        self._authorised_hashes: tuple[str, ...] = tuple(
            hashlib.sha256(k.encode("utf-8")).hexdigest()
            for k in keys
            if k.strip()
        )
        self._buckets: dict[str, TokenBucket] = {}
        self._accepted_total: dict[str, int] = {}
        self._rejected_total: dict[str, int] = {}
        self._unauthenticated_total: int = 0

    @property
    def enabled(self) -> bool:
        """True when at least one key is configured. False = dev mode."""
        return bool(self._authorised_hashes)

    @property
    def rate_per_minute(self) -> int:
        return self._rate_per_minute

    @property
    def burst(self) -> int:
        return self._burst

    def authenticate(self, presented_key: str | None) -> str:
        """Return a stable identity string or raise on failure.

        When auth is disabled returns ``ANONYMOUS_IDENTITY`` regardless of
        what was presented — but rate limiting still applies per the
        anonymous bucket, which keeps dev runs bounded too.
        """
        if not self.enabled:
            return ANONYMOUS_IDENTITY
        if not presented_key:
            with self._lock:
                self._unauthenticated_total += 1
            raise PermissionError("missing API key")
        # Hash the presented key once, then compare against the configured
        # set with constant-time comparison to avoid leaking which prefix
        # matched via timing.
        presented_hash = hashlib.sha256(presented_key.encode("utf-8")).hexdigest()
        for authorised in self._authorised_hashes:
            if secrets.compare_digest(presented_hash, authorised):
                # Identity is the *hash* of the presented key — never the
                # plaintext — so logs are safe to keep.
                return presented_hash
        with self._lock:
            self._unauthenticated_total += 1
        raise PermissionError("invalid API key")

    def consume(self, identity: str) -> None:
        """Charge one token to ``identity`` or raise ``RateLimitExceeded``."""
        with self._lock:
            bucket = self._buckets.get(identity)
            if bucket is None:
                bucket = TokenBucket(self._rate_per_minute, self._burst)
                self._buckets[identity] = bucket
            ok, wait = bucket.try_consume(1)
            if ok:
                self._accepted_total[identity] = (
                    self._accepted_total.get(identity, 0) + 1
                )
                return
            self._rejected_total[identity] = (
                self._rejected_total.get(identity, 0) + 1
            )
        raise RateLimitExceeded(identity, wait, self._rate_per_minute)

    def snapshot(self) -> dict[str, Any]:
        """Cheap aggregated view for /metrics — never returns plaintext keys."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "keys_configured": len(self._authorised_hashes),
                "rate_per_minute": self._rate_per_minute,
                "burst": self._burst,
                "accepted_total": sum(self._accepted_total.values()),
                "rejected_total": sum(self._rejected_total.values()),
                "unauthenticated_total": self._unauthenticated_total,
                "distinct_identities": len(self._buckets),
            }


def parse_keys_csv(raw: str) -> list[str]:
    """Split a comma-separated key list, ignoring whitespace and empties."""
    return [k.strip() for k in (raw or "").split(",") if k.strip()]
