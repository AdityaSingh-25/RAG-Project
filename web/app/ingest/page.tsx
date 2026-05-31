"use client";

import {
  AlertCircle,
  CheckCircle2,
  Copy,
  FolderOpen,
  Hash,
  Loader2,
  Trash2,
  Upload,
  UploadCloud,
} from "lucide-react";
import {
  ChangeEvent,
  DragEvent,
  FormEvent,
  useCallback,
  useMemo,
  useRef,
  useState,
} from "react";

import { ApiError, runIngest, runIngestUpload } from "@/lib/api";
import { cn } from "@/lib/utils";

// Mirrors `_ALLOWED_UPLOAD_SUFFIXES` in src/rag_engine/api/main.py. Keep
// these in sync — the backend will reject anything not in its set with a 415.
const ACCEPTED_EXTENSIONS = [".pdf", ".csv", ".json", ".txt", ".md", ".rst"];
const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.join(",");
const MAX_FILES = 20;

type Mode = "path" | "upload";

interface HistoryEntry {
  source: string;
  indexed: number;
  duplicates: number;
  filesReceived?: number;
  at: Date;
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function hasAcceptedExtension(name: string): boolean {
  const lower = name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

export default function IngestPage() {
  const [mode, setMode] = useState<Mode>("path");
  const [source, setSource] = useState("data/raw");
  const [pending, setPending] = useState<File[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addFiles = useCallback((incoming: FileList | File[]) => {
    const arr = Array.from(incoming);
    const accepted = arr.filter((f) => hasAcceptedExtension(f.name));
    const rejected = arr.length - accepted.length;
    setPending((prev) => {
      // De-dupe by name+size+lastModified so dropping the same files twice
      // doesn't double up the queue.
      const seen = new Set(prev.map(fileKey));
      const next = [...prev];
      for (const f of accepted) {
        const k = fileKey(f);
        if (!seen.has(k)) {
          seen.add(k);
          next.push(f);
        }
      }
      return next.slice(0, MAX_FILES);
    });
    if (rejected > 0) {
      setError(
        `Skipped ${rejected} file${rejected === 1 ? "" : "s"} with unsupported extensions. Allowed: ${ACCEPTED_EXTENSIONS.join(", ")}`,
      );
    } else {
      setError(null);
    }
  }, []);

  function removePending(key: string) {
    setPending((prev) => prev.filter((f) => fileKey(f) !== key));
  }

  function onFileInputChange(e: ChangeEvent<HTMLInputElement>) {
    if (e.target.files) addFiles(e.target.files);
    // Reset the input so re-selecting the same file triggers onChange again.
    e.target.value = "";
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files) addFiles(e.dataTransfer.files);
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (loading) return;
    if (mode === "path" && !source.trim()) return;
    if (mode === "upload" && pending.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      if (mode === "path") {
        const trimmed = source.trim();
        const res = await runIngest({ source_path: trimmed });
        appendHistory({
          source: trimmed,
          indexed: res.ingested_chunks,
          duplicates: res.duplicates_removed,
          at: new Date(),
        });
      } else {
        const files = pending;
        const res = await runIngestUpload(files);
        appendHistory({
          source: summariseUpload(files),
          indexed: res.ingested_chunks,
          duplicates: res.duplicates_removed,
          filesReceived: res.files_received ?? files.length,
          at: new Date(),
        });
        setPending([]);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  function appendHistory(entry: HistoryEntry) {
    setHistory((prev) => [entry, ...prev].slice(0, 8));
  }

  const totalPendingSize = useMemo(
    () => pending.reduce((sum, f) => sum + f.size, 0),
    [pending],
  );
  const latest = history[0];

  return (
    <main className="mx-auto max-w-3xl px-4 py-6 sm:px-6 lg:px-8">
      <div className="mb-6">
        <h1 className="text-xl font-semibold tracking-normal sm:text-2xl">
          Ingest documents
        </h1>
        <p className="text-muted mt-1 text-sm">
          Chunks are deduplicated by a SHA-256 of their normalised text before
          being indexed in Qdrant and the BM25 corpus.
        </p>
      </div>

      <div className="bg-elev rounded-lg border shadow-sm">
        <ModeTabs mode={mode} onChange={setMode} />

        <form onSubmit={submit} className="p-4">
          {mode === "path" ? (
            <PathForm source={source} onChange={setSource} disabled={loading} />
          ) : (
            <UploadForm
              pending={pending}
              dragActive={dragActive}
              onRemove={removePending}
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={onDrop}
              onPickClick={() => fileInputRef.current?.click()}
            />
          )}

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={ACCEPT_ATTR}
            onChange={onFileInputChange}
            className="hidden"
          />

          <div className="mt-4 flex items-center justify-between gap-3">
            <p className="text-subtle text-[11.5px]">
              {mode === "path"
                ? "Path is resolved against the API process's working directory."
                : pending.length > 0
                  ? `${pending.length} file${pending.length === 1 ? "" : "s"} · ${formatBytes(totalPendingSize)}`
                  : `Up to ${MAX_FILES} files. Accepted: ${ACCEPTED_EXTENSIONS.join(", ")}`}
            </p>
            <button
              type="submit"
              disabled={
                loading ||
                (mode === "path" ? !source.trim() : pending.length === 0)
              }
              className="inline-flex h-10 items-center gap-2 rounded-md px-4 text-sm font-semibold transition-opacity disabled:opacity-50"
              style={{
                background: "var(--color-accent)",
                color: "var(--color-accent-fg)",
              }}
            >
              {loading ? (
                <Loader2 size={15} className="animate-spin" />
              ) : mode === "path" ? (
                <Upload size={15} />
              ) : (
                <UploadCloud size={15} />
              )}
              {mode === "path"
                ? "Ingest"
                : loading
                  ? "Uploading…"
                  : `Upload${pending.length ? ` ${pending.length}` : ""}`}
            </button>
          </div>
        </form>
      </div>

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
            <div className="font-medium">
              {mode === "path" ? "Ingest failed" : "Upload failed"}
            </div>
            <div className="mt-0.5 font-mono text-[12px] opacity-90">{error}</div>
          </div>
        </div>
      )}

      {latest && <ResultCard latest={latest} />}

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
                <span className="truncate font-mono">{h.source}</span>
                <span className="text-subtle flex shrink-0 items-center gap-3 font-mono text-[11px]">
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

function ModeTabs({
  mode,
  onChange,
}: {
  mode: Mode;
  onChange: (m: Mode) => void;
}) {
  const tabs: Array<{ id: Mode; label: string; hint: string }> = [
    { id: "path", label: "Server path", hint: "Point at a path on the API host" },
    { id: "upload", label: "Upload", hint: "Drop files from this machine" },
  ];
  return (
    <div className="flex border-b" role="tablist">
      {tabs.map((tab) => {
        const active = tab.id === mode;
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(tab.id)}
            className={cn(
              "relative -mb-px px-4 py-3 text-sm font-medium transition-colors",
              active ? "" : "text-muted hover:text-[var(--color-fg)]",
            )}
            style={
              active
                ? {
                    color: "var(--color-accent)",
                    borderBottom: "2px solid var(--color-accent)",
                  }
                : undefined
            }
          >
            {tab.label}
            <span className="text-subtle ml-1.5 hidden text-[11px] font-normal sm:inline">
              {tab.hint}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function PathForm({
  source,
  onChange,
  disabled,
}: {
  source: string;
  onChange: (v: string) => void;
  disabled: boolean;
}) {
  return (
    <>
      <label
        htmlFor="source"
        className="text-muted block text-xs font-medium uppercase tracking-wider"
      >
        Source path
      </label>
      <div className="bg-sunken focus-within:ring-focus focus-within:border-strong relative mt-2 flex-1 rounded-md border">
        <FolderOpen
          size={14}
          className="text-subtle pointer-events-none absolute top-1/2 left-3 -translate-y-1/2"
        />
        <input
          id="source"
          type="text"
          value={source}
          onChange={(e) => onChange(e.target.value)}
          placeholder="data/raw"
          spellCheck={false}
          disabled={disabled}
          className="w-full bg-transparent py-2.5 pr-3 pl-9 font-mono text-[13px] outline-none disabled:opacity-60"
        />
      </div>
    </>
  );
}

function UploadForm({
  pending,
  dragActive,
  onRemove,
  onDragOver,
  onDragLeave,
  onDrop,
  onPickClick,
}: {
  pending: File[];
  dragActive: boolean;
  onRemove: (key: string) => void;
  onDragOver: (e: DragEvent<HTMLDivElement>) => void;
  onDragLeave: () => void;
  onDrop: (e: DragEvent<HTMLDivElement>) => void;
  onPickClick: () => void;
}) {
  return (
    <>
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={onPickClick}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onPickClick();
          }
        }}
        className={cn(
          "grid cursor-pointer place-items-center rounded-md border-2 border-dashed px-4 py-8 text-center transition-colors",
          dragActive ? "bg-sunken" : "hover:bg-sunken/60",
        )}
        style={
          dragActive
            ? { borderColor: "var(--color-accent)" }
            : undefined
        }
      >
        <UploadCloud
          size={28}
          className={cn(
            "mb-2",
            dragActive ? "" : "text-subtle",
          )}
          style={dragActive ? { color: "var(--color-accent)" } : undefined}
        />
        <p className="text-sm">
          <span className="font-medium">Drag files here</span>{" "}
          <span className="text-muted">or click to browse</span>
        </p>
        <p className="text-subtle mt-1 text-[11px]">
          PDF · CSV · JSON · TXT · MD · RST
        </p>
      </div>

      {pending.length > 0 && (
        <ul className="mt-3 divide-y rounded-md border">
          {pending.map((f) => {
            const k = fileKey(f);
            return (
              <li
                key={k}
                className="bg-sunken flex items-center justify-between gap-3 px-3 py-2 text-[13px]"
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate font-mono text-[12.5px]">{f.name}</div>
                  <div className="text-subtle font-mono text-[10.5px]">
                    {formatBytes(f.size)}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => onRemove(k)}
                  className="text-subtle hover:text-[var(--color-danger)] inline-flex h-7 w-7 items-center justify-center rounded-md transition-colors"
                  aria-label={`Remove ${f.name}`}
                >
                  <Trash2 size={13} />
                </button>
              </li>
            );
          })}
        </ul>
      )}

    </>
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
        {latest.filesReceived !== undefined && (
          <>
            <span className="mx-2">·</span>
            {latest.filesReceived} file
            {latest.filesReceived === 1 ? "" : "s"}
          </>
        )}
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

function fileKey(f: File): string {
  return `${f.name}::${f.size}::${f.lastModified}`;
}

function summariseUpload(files: File[]): string {
  if (files.length === 1) return `upload: ${files[0].name}`;
  if (files.length <= 3) return `upload: ${files.map((f) => f.name).join(", ")}`;
  return `upload: ${files[0].name} +${files.length - 1} more`;
}
