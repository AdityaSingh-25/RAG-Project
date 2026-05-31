"use client";

import {
  AlertCircle,
  Database,
  FileText,
  Hash,
  Library,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, getCorpusSources, getCorpusStats } from "@/lib/api";
import type { CorpusSource, CorpusStats } from "@/lib/types";
import { cn } from "@/lib/utils";

export default function CorpusPage() {
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [sources, setSources] = useState<CorpusSource[]>([]);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      // Run in parallel; both endpoints hit the same Qdrant scroll path on
      // the backend, but each computes independently and the round-trip
      // savings are worth the two requests.
      const [s, srcs] = await Promise.all([
        getCorpusStats(signal),
        getCorpusSources(signal),
      ]);
      setStats(s);
      setSources(srcs.sources);
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

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return sources;
    return sources.filter((s) => s.source.toLowerCase().includes(q));
  }, [filter, sources]);

  const totalChunks = stats?.chunks ?? 0;
  const totalSources = stats?.sources ?? 0;
  const empty = !loading && !error && totalChunks === 0;

  return (
    <main className="mx-auto max-w-5xl px-4 py-6 sm:px-6 lg:px-8">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-normal sm:text-2xl">Corpus</h1>
          <p className="text-muted mt-1 text-sm">
            What is currently indexed in Qdrant. Counted by walking every chunk and
            grouping on <code className="font-mono">metadata.source</code>.
          </p>
        </div>
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

      <section className="mb-6 grid gap-3 sm:grid-cols-3">
        <StatTile
          label="Chunks"
          value={totalChunks.toLocaleString()}
          Icon={Hash}
          loading={loading && !stats}
        />
        <StatTile
          label="Sources"
          value={totalSources.toLocaleString()}
          Icon={FileText}
          loading={loading && !stats}
        />
        <StatTile
          label="Collection"
          value={stats?.collection ?? "—"}
          Icon={Database}
          mono
          loading={loading && !stats}
        />
      </section>

      {empty ? (
        <div className="bg-sunken rounded-lg border border-dashed p-10 text-center">
          <Library className="text-subtle mx-auto" size={28} />
          <p className="mt-3 font-medium">Nothing indexed yet</p>
          <p className="text-muted mt-1 text-sm">
            Use the Ingest page to point the engine at a directory or upload files.
          </p>
        </div>
      ) : (
        <section>
          <div className="mb-3 flex items-center justify-between gap-3">
            <h2 className="text-sm font-semibold">Sources</h2>
            <div className="flex items-center gap-2">
              <input
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="filter…"
                className="bg-sunken focus:ring-focus focus:border-strong h-8 rounded-md border px-3 font-mono text-[12px] outline-none"
                disabled={sources.length === 0}
              />
              <span className="text-subtle font-mono text-[11px]">
                {filtered.length}/{sources.length}
              </span>
            </div>
          </div>

          <div className="bg-elev overflow-hidden rounded-lg border">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-subtle border-b font-mono text-[10px] uppercase tracking-widest">
                  <th className="px-4 py-2 text-left font-normal">source</th>
                  <th className="px-3 py-2 text-right font-normal">chunks</th>
                  <th className="px-3 py-2 text-right font-normal">share</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((row) => {
                  const share = totalChunks > 0 ? row.chunks / totalChunks : 0;
                  return (
                    <tr key={row.source} className="border-b last:border-b-0">
                      <td className="break-all px-4 py-2.5 font-mono text-[12.5px]">
                        {row.source}
                      </td>
                      <td className="px-3 py-2.5 text-right font-mono text-[12.5px] tabular-nums">
                        {row.chunks.toLocaleString()}
                      </td>
                      <td className="px-3 py-2.5 text-right">
                        <ShareBar share={share} />
                      </td>
                    </tr>
                  );
                })}
                {filtered.length === 0 && sources.length > 0 && (
                  <tr>
                    <td colSpan={3} className="text-subtle px-4 py-6 text-center text-sm">
                      No source matches <code className="font-mono">{filter}</code>.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {lastUpdated && (
        <p className="text-subtle mt-6 text-right font-mono text-[11px]">
          updated · {lastUpdated.toLocaleTimeString()}
        </p>
      )}
    </main>
  );
}

function StatTile({
  label,
  value,
  Icon,
  mono,
  loading,
}: {
  label: string;
  value: string;
  Icon: typeof Hash;
  mono?: boolean;
  loading?: boolean;
}) {
  return (
    <div className="bg-elev rounded-lg border p-4">
      <div className="text-subtle inline-flex items-center gap-1.5 text-xs uppercase tracking-wider">
        <Icon size={13} />
        {label}
      </div>
      <div
        className={cn(
          "mt-2 text-xl font-semibold",
          mono && "break-all font-mono text-base",
        )}
        style={{ color: "var(--color-accent)" }}
      >
        {loading ? <Loader2 size={18} className="animate-spin" /> : value}
      </div>
    </div>
  );
}

function ShareBar({ share }: { share: number }) {
  return (
    <div className="ml-auto flex w-28 items-center gap-2">
      <div className="bg-sunken h-1.5 flex-1 overflow-hidden rounded-full">
        <div
          className="h-full"
          style={{
            width: `${Math.max(2, share * 100)}%`,
            background: "var(--color-accent)",
            transition: "width 220ms ease",
          }}
        />
      </div>
      <span className="text-subtle w-9 text-right font-mono text-[10.5px] tabular-nums">
        {`${Math.round(share * 100)}%`}
      </span>
    </div>
  );
}
