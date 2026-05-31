"use client";

import {
  AlertTriangle,
  Copy,
  FileText,
  Hash,
  Loader2,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";

import { ApiError, getCorpusSource } from "@/lib/api";
import type { CorpusChunk, CorpusSourceDetail } from "@/lib/types";

interface SourceInspectorProps {
  /** When non-null, the modal is open and we fetch this source's chunks. */
  source: string | null;
  onClose: () => void;
}

export function SourceInspector({ source, onClose }: SourceInspectorProps) {
  const [detail, setDetail] = useState<CorpusSourceDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Reset + fetch whenever a new source is opened. Aborting on close keeps
  // the React tree quiet (no setState after unmount warnings).
  useEffect(() => {
    if (!source) {
      setDetail(null);
      setError(null);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setDetail(null);
    getCorpusSource(source, controller.signal)
      .then((d) => setDetail(d))
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError(err instanceof ApiError ? err.message : String(err));
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [source]);

  // Escape closes. Also lock background scroll while the modal's open so
  // mouse-wheel doesn't dump the user out the side of the long chunk list.
  useEffect(() => {
    if (!source) return;
    const original = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = original;
      window.removeEventListener("keydown", onKey);
    };
  }, [source, onClose]);

  if (!source) return null;

  return (
    <>
      <div
        className="fixed inset-0 z-40 bg-black/50"
        onClick={onClose}
        aria-hidden
      />
      <div
        className="bg-elev fixed inset-x-4 top-12 z-50 mx-auto flex max-h-[calc(100vh-6rem)] max-w-4xl flex-col rounded-lg border shadow-2xl"
        role="dialog"
        aria-label={`Chunks from ${source}`}
      >
        <header className="flex items-center justify-between gap-3 border-b px-4 py-3">
          <div className="min-w-0">
            <div className="text-muted text-[10.5px] uppercase tracking-widest">
              Source
            </div>
            <div className="break-all font-mono text-[13px] font-medium">
              {source}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close inspector"
            className="text-muted hover:text-[var(--color-fg)] inline-flex h-8 w-8 items-center justify-center rounded-md transition-colors"
          >
            <X size={16} />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-4">
          {loading && (
            <div className="text-muted grid h-32 place-items-center text-sm">
              <div className="flex items-center gap-2">
                <Loader2 size={16} className="animate-spin" />
                Loading chunks…
              </div>
            </div>
          )}

          {error && (
            <div
              className="flex items-start gap-3 rounded-lg border px-3 py-2 text-sm"
              style={{
                borderColor: "color-mix(in oklch, var(--color-danger), transparent 70%)",
                background: "color-mix(in oklch, var(--color-danger), transparent 92%)",
                color: "var(--color-danger)",
              }}
            >
              <AlertTriangle size={15} className="mt-0.5 shrink-0" />
              <div className="font-mono text-[12px]">{error}</div>
            </div>
          )}

          {detail && !loading && !error && (
            <>
              <div className="mb-3 flex items-center justify-between gap-3 text-[12px]">
                <span className="text-muted">
                  <span className="font-semibold text-[var(--color-fg)]">
                    {detail.total.toLocaleString()}
                  </span>{" "}
                  chunk{detail.total === 1 ? "" : "s"}
                </span>
                {detail.truncated && (
                  <span
                    className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10.5px]"
                    style={{
                      borderColor: "color-mix(in oklch, var(--color-warning), transparent 65%)",
                      color: "var(--color-warning)",
                    }}
                  >
                    <AlertTriangle size={10} />
                    Result truncated
                  </span>
                )}
              </div>
              {detail.chunks.length === 0 ? (
                <div className="bg-sunken text-subtle rounded-lg border border-dashed py-10 text-center text-sm">
                  <FileText className="mx-auto opacity-50" size={22} />
                  <p className="mt-2">No chunks recorded for this source.</p>
                </div>
              ) : (
                <ol className="space-y-3">
                  {detail.chunks.map((chunk, i) => (
                    <ChunkRow key={`${chunk.content_hash ?? i}`} chunk={chunk} index={i} />
                  ))}
                </ol>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}

function ChunkRow({ chunk, index }: { chunk: CorpusChunk; index: number }) {
  const [copied, setCopied] = useState(false);
  async function copyHash() {
    if (!chunk.content_hash) return;
    try {
      await navigator.clipboard.writeText(chunk.content_hash);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      // navigator.clipboard fails in some sandboxes; silently no-op.
    }
  }
  const chunkLabel = chunk.chunk_id !== null ? `chunk ${chunk.chunk_id}` : `#${index}`;
  return (
    <li className="bg-sunken rounded-md border p-3">
      <div className="text-muted mb-2 flex flex-wrap items-center justify-between gap-2 font-mono text-[10.5px]">
        <span className="inline-flex items-center gap-1.5">
          <Hash size={11} />
          {chunkLabel}
          {chunk.page !== null && <span className="text-subtle">· page {chunk.page}</span>}
        </span>
        {chunk.content_hash && (
          <button
            type="button"
            onClick={copyHash}
            className="text-subtle hover:text-[var(--color-fg)] inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10.5px] transition-colors"
            title="Copy SHA-256"
          >
            <Copy size={10} />
            {copied ? "copied" : `${chunk.content_hash.slice(0, 10)}…`}
          </button>
        )}
      </div>
      <pre className="whitespace-pre-wrap break-words font-mono text-[12px] leading-snug">
        {chunk.content}
      </pre>
    </li>
  );
}
