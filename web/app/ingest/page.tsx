"use client";

import {
  AlertCircle,
  CheckCircle2,
  Copy,
  FolderOpen,
  Hash,
  Loader2,
  Upload,
} from "lucide-react";
import { FormEvent, useState } from "react";

import { ApiError, runIngest } from "@/lib/api";

interface HistoryEntry {
  source: string;
  indexed: number;
  duplicates: number;
  at: Date;
}

export default function IngestPage() {
  const [source, setSource] = useState("data/raw");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (loading || !source.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runIngest({ source_path: source.trim() });
      setHistory((prev) => [
        {
          source: source.trim(),
          indexed: res.ingested_chunks,
          duplicates: res.duplicates_removed,
          at: new Date(),
        },
        ...prev,
      ].slice(0, 8));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const latest = history[0];

  return (
    <main className="mx-auto max-w-3xl px-4 py-6 sm:px-6 lg:px-8">
      <div className="mb-6">
        <h1 className="text-xl font-semibold tracking-normal sm:text-2xl">
          Ingest documents
        </h1>
        <p className="text-muted mt-1 text-sm">
          Point the engine at a directory or file. Chunks are deduplicated by a
          SHA-256 of their normalised text before being indexed in Qdrant and
          the BM25 corpus.
        </p>
      </div>

      <form
        onSubmit={submit}
        className="bg-elev rounded-lg border p-4 shadow-sm"
      >
        <label
          htmlFor="source"
          className="text-muted block text-xs font-medium uppercase tracking-wider"
        >
          Source path
        </label>
        <div className="mt-2 flex gap-2">
          <div className="bg-sunken focus-within:ring-focus focus-within:border-strong relative flex-1 rounded-md border">
            <FolderOpen
              size={14}
              className="text-subtle pointer-events-none absolute top-1/2 left-3 -translate-y-1/2"
            />
            <input
              id="source"
              type="text"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder="data/raw"
              spellCheck={false}
              disabled={loading}
              className="w-full bg-transparent py-2.5 pr-3 pl-9 font-mono text-[13px] outline-none disabled:opacity-60"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !source.trim()}
            className="inline-flex h-10 items-center gap-2 rounded-md px-4 text-sm font-semibold transition-opacity disabled:opacity-50"
            style={{
              background: "var(--color-accent)",
              color: "var(--color-accent-fg)",
            }}
          >
            {loading ? (
              <Loader2 size={15} className="animate-spin" />
            ) : (
              <Upload size={15} />
            )}
            Ingest
          </button>
        </div>
        <p className="text-subtle mt-2 text-[11.5px]">
          Path is resolved against the FastAPI process's working directory. The
          directory is bind-mounted into the API container in the default
          Compose setup.
        </p>
      </form>

      {error && (
        <div
          className="mt-5 flex items-start gap-3 rounded-lg border px-4 py-3 text-sm"
          style={{
            borderColor: "color-mix(in oklch, var(--color-danger), transparent 70%)",
            background: "color-mix(in oklch, var(--color-danger), transparent 92%)",
            color: "var(--color-danger)",
          }}
        >
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <div>
            <div className="font-medium">Ingest failed</div>
            <div className="mt-0.5 font-mono text-[12px] opacity-90">{error}</div>
          </div>
        </div>
      )}

      {latest && (
        <ResultCard latest={latest} />
      )}

      {history.length > 1 && (
        <section className="mt-6">
          <h2 className="text-subtle font-mono text-[10px] uppercase tracking-widest">
            Earlier runs (this session)
          </h2>
          <ul className="mt-2 divide-y rounded-lg border">
            {history.slice(1).map((h, i) => (
              <li
                key={i}
                className="bg-elev flex items-center justify-between gap-3 px-4 py-2.5 text-[13px]"
              >
                <span className="font-mono">{h.source}</span>
                <span className="text-subtle flex items-center gap-3 font-mono text-[11px]">
                  <span>{h.indexed} chunks</span>
                  {h.duplicates > 0 && (
                    <span style={{ color: "var(--color-warning)" }}>
                      −{h.duplicates} dupes
                    </span>
                  )}
                  <span>{h.at.toLocaleTimeString()}</span>
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}

function ResultCard({ latest }: { latest: HistoryEntry }) {
  return (
    <div className="bg-elev mt-5 rounded-lg border p-5">
      <div
        className="inline-flex items-center gap-2 text-sm font-medium"
        style={{ color: "var(--color-success)" }}
      >
        <CheckCircle2 size={15} strokeWidth={2.25} />
        Ingest complete
      </div>
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <Metric
          icon={<Copy size={14} />}
          label="Chunks indexed"
          value={latest.indexed.toString()}
          accent="var(--color-accent)"
        />
        <Metric
          icon={<Hash size={14} />}
          label="Duplicates removed"
          value={latest.duplicates.toString()}
          accent={
            latest.duplicates > 0 ? "var(--color-warning)" : "var(--color-fg-muted)"
          }
        />
      </div>
      <div className="text-subtle mt-4 font-mono text-[11px]">
        source · <span className="text-muted">{latest.source}</span>
        <span className="mx-2">·</span>
        {latest.at.toLocaleString()}
      </div>
    </div>
  );
}

function Metric({
  icon,
  label,
  value,
  accent,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  accent: string;
}) {
  return (
    <div className="bg-sunken rounded-md border p-3">
      <div className="text-subtle inline-flex items-center gap-1.5 text-xs uppercase tracking-wider">
        {icon}
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold" style={{ color: accent }}>
        {value}
      </div>
    </div>
  );
}
