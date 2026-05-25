"""Thread-safe in-memory counters exposed via /metrics.

Three primitives are enough for the graph today:

- ``increment`` for simple totals (queries, cache hits/misses, fallback events).
- ``observe`` for samples we want to summarise (latency, iteration counts).
- ``snapshot`` to serialise the current state for the /metrics endpoint.

This is deliberately not Prometheus. When the workload outgrows a single
process, swap this module for a real metrics client; the call sites stay
the same.
"""

from __future__ import annotations

import statistics
import threading
from typing import Any


class Counters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, int] = {}
        self._samples: dict[str, list[float]] = {}

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._totals[name] = self._totals.get(name, 0) + amount

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._samples.setdefault(name, []).append(float(value))

    def reset(self) -> None:
        with self._lock:
            self._totals.clear()
            self._samples.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            totals = dict(self._totals)
            sample_summary: dict[str, dict[str, float]] = {}
            for name, values in self._samples.items():
                if not values:
                    continue
                sample_summary[name] = {
                    "count": float(len(values)),
                    "mean": round(statistics.fmean(values), 4),
                    "p50": round(statistics.median(values), 4),
                    "p95": round(_percentile(values, 0.95), 4),
                    "max": round(max(values), 4),
                }
        return {"totals": totals, "samples": sample_summary}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(percentile * (len(ordered) - 1)))))
    return ordered[idx]


_counters = Counters()


def counters() -> Counters:
    """Process-wide singleton. The graph and cache wrappers both target this."""
    return _counters
