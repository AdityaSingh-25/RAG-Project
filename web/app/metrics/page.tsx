"use client";

import {
  AlertCircle,
  AlertTriangle,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, getMetrics } from "@/lib/api";
import type { MetricsSnapshot } from "@/lib/types";
import { cn, formatMs } from "@/lib/utils";

// Auto-refresh interval. 2s is fast enough to feel live but slow enough
// not to be silly — the underlying counters are in-memory and cheap.
const REFRESH_INTERVAL_MS = 2000;

interface Group {
  title: string;
  rows: Array<{ key: string; label: string; value: number }>;
}

/** Partition `totals` by key prefix into the UI sections we actually want. */
function partitionTotals(totals: Record<string, number>): Group[] {
  const groups: Record<string, Group> = {
    api: { title: "API queries", rows: [] },
    cache: { title: "Cache", rows: [] },
    graph: { title: "Pipeline", rows: [] },
  };
  for (const [key, value] of Object.entries(totals).sort()) {
    const top = key.split(".")[0];
    const target = groups[top] ?? null;
    if (!target) continue;
    target.rows.push({ key, label: prettify(key), value });
  }
  return Object.values(groups).filter((g) => g.rows.length > 0);
}

function prettify(key: string): string {
  // "api.query.status.ok" -> "query · status · ok"
  return key
    .split(".")
    .slice(1)
    .map((p) => p.replace(/_/g, " "))
    .join(" · ");
}

function formatTotal(n: number): string {
  return n.toLocaleString();
}

/** Pretty cache hit rate per namespace, computed from the totals. */
function cacheHitRates(totals: Record<string, number>): Array<{
  namespace: string;
  hits: number;
  misses: number;
  rate: number;
}> {
  const namespaces = new Set<string>();
  for (const key of Object.keys(totals)) {
    const m = key.match(/^cache\.([^.]+)\.(hit|miss)$/);
    if (m) namespaces.add(m[1]);
  }
  return Array.from(namespaces).map((ns) => {
    const hits = totals[`cache.${ns}.hit`] ?? 0;
    const misses = totals[`cache.${ns}.miss`] ?? 0;
    const total = hits + misses;
    return {
      namespace: ns,
      hits,
      misses,
      rate: total === 0 ? 0 : hits / total,
    };
  });
}

export default function MetricsPage() {
  const [snapshot, setSnapshot] = useState<MetricsSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const abortRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getMetrics(signal);
      setSnapshot(data);
      setLastUpdated(new Date());
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    abortRef.current = controller;
    refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    if (!autoRefresh) return;
    const id = window.setInterval(() => refresh(), REFRESH_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [autoRefresh, refresh]);

  const groups = useMemo(
    () => (snapshot ? partitionTotals(snapshot.totals) : []),
    [snapshot],
  );
  const cacheStats = useMemo(
    () => (snapshot ? cacheHitRates(snapshot.totals) : []),
    [snapshot],
  );
  const samples = useMemo(
    () =>
      snapshot
        ? Object.entries(snapshot.samples).sort(([a], [b]) => a.localeCompare(b))
        : [],
    [snapshot],
  );

  return (
    <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-normal sm:text-2xl">
            Engine metrics
          </h1>
          <p className="text-muted mt-1 text-sm">
            Live counters from the FastAPI <code className="font-mono">/metrics</code> endpoint.
            Refreshes every {(REFRESH_INTERVAL_MS / 1000).toFixed(0)}s.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-muted inline-flex cursor-pointer items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="h-3.5 w-3.5"
              style={{ accentColor: "var(--color-accent)" }}
            />
            Auto-refresh
          </label>
          <button
            type="button"
            onClick={() => refresh()}
            disabled={loading}
            className="bg-elev hover:border-strong inline-flex h-9 items-center gap-2 rounded-md border px-3 text-sm font-medium transition-colors disabled:opacity-60"
          >
            <RefreshCw size={14} className={cn(loading && "animate-spin")} />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div
          className="mb-6 flex items-start gap-3 rounded-lg border px-4 py-3 text-sm"
          style={{
            borderColor: "color-mix(in oklch, var(--color-danger), transparent 70%)",
            background: "color-mix(in oklch, var(--color-danger), transparent 92%)",
            color: "var(--color-danger)",
          }}
        >
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <div>
            <div className="font-medium">Cannot reach the engine</div>
            <div className="mt-0.5 font-mono text-[12px] opacity-90">{error}</div>
          </div>
        </div>
      )}

      {!snapshot && !error && (
        <div className="bg-sunken text-subtle rounded-lg border border-dashed px-6 py-10 text-center text-sm">
          <Loader2 size={18} className="mx-auto animate-spin" />
          <p className="mt-3">Reading counters…</p>
        </div>
      )}

      {snapshot && (
        <div className="space-y-8">
          {cacheStats.length > 0 && (
            <CacheSection stats={cacheStats} />
          )}

          {groups.map((group) => (
            <CounterGroupSection key={group.title} group={group} />
          ))}

          {samples.length > 0 && <LatencySection samples={samples} />}
        </div>
      )}

      {lastUpdated && (
        <p className="text-subtle mt-8 text-right font-mono text-[11px]">
          updated · {lastUpdated.toLocaleTimeString()}
        </p>
      )}
    </main>
  );
}

function CacheSection({
  stats,
}: {
  stats: Array<{ namespace: string; hits: number; misses: number; rate: number }>;
}) {
  return (
    <section>
      <SectionTitle>Cache hit rates</SectionTitle>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {stats.map((s) => (
          <div key={s.namespace} className="bg-elev rounded-lg border p-4">
            <div className="text-muted text-xs uppercase tracking-wider">
              {s.namespace.replace(/_/g, " ")}
            </div>
            <div className="mt-1 flex items-baseline gap-2">
              <div
                className="text-2xl font-semibold"
                style={{ color: "var(--color-accent)" }}
              >
                {(s.rate * 100).toFixed(1)}%
              </div>
              <div className="text-subtle font-mono text-[11px]">
                {formatTotal(s.hits)} hits · {formatTotal(s.misses)} misses
              </div>
            </div>
            <div className="bg-sunken mt-3 h-1.5 overflow-hidden rounded-full">
              <div
                className="h-full"
                style={{
                  width: `${Math.max(2, s.rate * 100)}%`,
                  background: "var(--color-accent)",
                  transition: "width 220ms ease",
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function CounterGroupSection({ group }: { group: Group }) {
  const isFallback = group.title === "Pipeline";
  return (
    <section>
      <SectionTitle>{group.title}</SectionTitle>
      <div className="bg-elev mt-3 overflow-hidden rounded-lg border">
        <table className="w-full text-sm">
          <tbody>
            {group.rows.map((row) => {
              const isFallbackReason = isFallback && row.key.includes(".fallback.reason.");
              return (
                <tr key={row.key} className="border-b last:border-b-0">
                  <td className="px-4 py-2.5">
                    <span className="font-mono text-[12.5px]">{row.label}</span>
                    {isFallbackReason && (
                      <AlertTriangle
                        size={11}
                        className="ml-1.5 inline-block align-middle"
                        style={{ color: "var(--color-warning)" }}
                      />
                    )}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-[13px] tabular-nums">
                    {formatTotal(row.value)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function LatencySection({
  samples,
}: {
  samples: Array<
    [string, { count: number; mean: number; p50: number; p95: number; max: number }]
  >;
}) {
  return (
    <section>
      <SectionTitle>Latency samples</SectionTitle>
      <div className="bg-elev mt-3 overflow-x-auto rounded-lg border">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-subtle border-b font-mono text-[10px] uppercase tracking-widest">
              <th className="px-4 py-2 text-left font-normal">sample</th>
              <th className="px-3 py-2 text-right font-normal">n</th>
              <th className="px-3 py-2 text-right font-normal">mean</th>
              <th className="px-3 py-2 text-right font-normal">p50</th>
              <th className="px-3 py-2 text-right font-normal">p95</th>
              <th className="px-3 py-2 text-right font-normal">max</th>
            </tr>
          </thead>
          <tbody>
            {samples.map(([key, s]) => {
              const isLatency = key.endsWith("_ms");
              const fmt = (v: number) =>
                isLatency ? formatMs(v) : v.toFixed(2);
              return (
                <tr key={key} className="border-b last:border-b-0">
                  <td className="px-4 py-2.5">
                    <span className="font-mono text-[12.5px]">{key}</span>
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono tabular-nums">
                    {formatTotal(s.count)}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono tabular-nums">
                    {fmt(s.mean)}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono tabular-nums">
                    {fmt(s.p50)}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono tabular-nums">
                    {fmt(s.p95)}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono tabular-nums">
                    {fmt(s.max)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-sm font-semibold">{children}</h2>
  );
}
