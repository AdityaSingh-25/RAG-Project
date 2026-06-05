import { useEffect, useRef, useState } from "react";

import type { BackpressureKind, MetricsSnapshot } from "./types";

export interface HistoryPoint {
  at: number;
  snapshot: MetricsSnapshot;
}

// 150 points × 2s refresh = 5 minutes of history. Large enough to spot a
// shift, small enough that the SVG paths stay cheap.
const DEFAULT_CAP = 150;

/** Rolling client-side buffer of `/metrics` snapshots, append-on-change. */
export function useMetricsHistory(
  snapshot: MetricsSnapshot | null,
  cap: number = DEFAULT_CAP,
): HistoryPoint[] {
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const lastRef = useRef<MetricsSnapshot | null>(null);

  useEffect(() => {
    if (!snapshot || snapshot === lastRef.current) return;
    lastRef.current = snapshot;
    setHistory((prev) => {
      const next = [...prev, { at: Date.now(), snapshot }];
      return next.length > cap ? next.slice(next.length - cap) : next;
    });
  }, [snapshot, cap]);

  return history;
}

/** Pull one statistic over time for a single sample key. Missing → 0. */
export function pluckSampleSeries(
  history: HistoryPoint[],
  key: string,
  stat: "mean" | "p50" | "p95" | "max",
): number[] {
  return history.map((h) => h.snapshot.samples[key]?.[stat] ?? 0);
}

/** Counter values are cumulative; convert to per-interval delta. */
export function pluckCounterRate(
  history: HistoryPoint[],
  key: string,
): number[] {
  if (history.length < 2) return [];
  const out: number[] = [];
  for (let i = 1; i < history.length; i++) {
    const a = history[i - 1].snapshot.totals[key] ?? 0;
    const b = history[i].snapshot.totals[key] ?? 0;
    out.push(Math.max(0, b - a));
  }
  return out;
}

/** Cache hit rate over time for one namespace. */
export function pluckCacheRateSeries(
  history: HistoryPoint[],
  namespace: string,
): number[] {
  return history.map((h) => {
    const hits = h.snapshot.totals[`cache.${namespace}.hit`] ?? 0;
    const misses = h.snapshot.totals[`cache.${namespace}.miss`] ?? 0;
    const total = hits + misses;
    return total === 0 ? 0 : hits / total;
  });
}

/** In-flight gauge over time for one limiter. Missing → 0 so older
 *  snapshots taken before the backpressure field existed render flat. */
export function pluckInFlightSeries(
  history: HistoryPoint[],
  kind: BackpressureKind,
): number[] {
  return history.map((h) => h.snapshot.backpressure?.[kind]?.in_flight ?? 0);
}

/** Per-interval count of rejected requests for one limiter — derived
 *  from the cumulative `rejected_total` counter so the sparkline shows
 *  *new* rejections rather than the lifetime sum. */
export function pluckRejectionRate(
  history: HistoryPoint[],
  kind: BackpressureKind,
): number[] {
  if (history.length < 2) return [];
  const out: number[] = [];
  for (let i = 1; i < history.length; i++) {
    const a = history[i - 1].snapshot.backpressure?.[kind]?.rejected_total ?? 0;
    const b = history[i].snapshot.backpressure?.[kind]?.rejected_total ?? 0;
    out.push(Math.max(0, b - a));
  }
  return out;
}
