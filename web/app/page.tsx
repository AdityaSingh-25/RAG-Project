"use client";

import {
  AlertTriangle,
  BarChart3,
  Check,
  CheckCircle2,
  Clock3,
  Copy,
  Database,
  Download,
  FileText,
  Loader2,
  Send,
  Sparkles,
  XCircle,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AnswerWithGrounding } from "@/components/AnswerWithGrounding";
import { TraceSidebar } from "@/components/TraceSidebar";
import { ApiError, getMetrics, runQueryStream } from "@/lib/api";
import type {
  Citation,
  ClaimVerifierMode,
  MetricsSnapshot,
  QueryRequest,
  QueryResponse,
} from "@/lib/types";
import {
  buildRunArtifact,
  copyJsonToClipboard,
  defaultExportFilename,
  downloadJson,
} from "@/lib/traceExport";
import { cn, formatMs } from "@/lib/utils";

const exampleQuestions = [
  "What are the main themes in the ingested documents?",
  "Which sources support the answer most strongly?",
  "What should the system refuse to answer?",
];

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value: number | undefined): string {
  if (value === undefined) return "0";
  if (Number.isInteger(value)) return value.toLocaleString();
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function statusCopy(response: QueryResponse | null, streaming: boolean) {
  if (streaming) {
    return {
      label: "Streaming",
      tone: "neutral",
      detail: "Answer is being generated incrementally",
      icon: Loader2,
    };
  }
  if (!response) {
    return {
      label: "Ready",
      tone: "neutral",
      detail: "Waiting for a grounded query",
      icon: Sparkles,
    };
  }
  if (response.status === "insufficient_evidence") {
    return {
      label: "Insufficient Evidence",
      tone: "warning",
      detail: "The critic rejected the draft after retrieval",
      icon: AlertTriangle,
    };
  }
  return {
    label: response.cached ? "Cached Answer" : "Grounded Answer",
    tone: "success",
    detail: response.cached ? "Served from answer cache" : "Passed grounding checks",
    icon: CheckCircle2,
  };
}

function metricValue(metrics: MetricsSnapshot | null, key: string): string {
  return formatNumber(metrics?.totals[key]);
}

// Initial response shape while a stream is in flight. Gets fully replaced
// by the payload from the `done` event.
function emptyStreamingResponse(): QueryResponse {
  return {
    answer: "",
    citations: [],
    grounding_score: 0,
    warnings: [],
    iteration: 0,
    status: "ok",
    grounded_claim_rate: 0,
    claim_grounding: [],
    pipeline_trace: [],
    total_duration_ms: 0,
    trace_id: "",
    cached: false,
  };
}

function sampleMean(metrics: MetricsSnapshot | null, key: string): string {
  const mean = metrics?.samples[key]?.mean;
  return mean === undefined ? "No samples" : formatMs(mean);
}

export default function Home() {
  const [question, setQuestion] = useState(exampleQuestions[0]);
  const [bypassCache, setBypassCache] = useState(false);
  // "default" leaves the field off the request so the API falls back to the
  // deployment-level Settings value. Explicit choices override that.
  const [verifierOverride, setVerifierOverride] =
    useState<"default" | ClaimVerifierMode>("default");
  const [structuredOverride, setStructuredOverride] =
    useState<"default" | "off" | "on">("default");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const status = useMemo(() => statusCopy(response, streaming), [response, streaming]);
  const StatusIcon = status.icon;

  async function refreshMetrics(signal?: AbortSignal) {
    setMetricsLoading(true);
    setMetricsError(null);
    try {
      const snapshot = await getMetrics(signal);
      setMetrics(snapshot);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const message = err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Unable to load metrics";
      setMetricsError(message);
    } finally {
      setMetricsLoading(false);
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    refreshMetrics(controller.signal);
    return () => controller.abort();
  }, []);

  async function submit(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const trimmed = question.trim();
    if (trimmed.length < 3 || loading) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setStreaming(true);
    setError(null);
    setResponse(emptyStreamingResponse());

    const request: QueryRequest = {
      question: trimmed,
      bypass_cache: bypassCache,
    };
    if (verifierOverride !== "default") {
      request.claim_verifier_mode = verifierOverride;
    }
    if (structuredOverride !== "default") {
      request.structured_answers = structuredOverride === "on";
    }

    try {
      await runQueryStream(
        request,
        {
          onTrace: (entry) =>
            setResponse((prev) =>
              prev ? { ...prev, pipeline_trace: [...prev.pipeline_trace, entry] } : prev,
            ),
          onToken: (delta) =>
            setResponse((prev) => (prev ? { ...prev, answer: prev.answer + delta } : prev)),
          onCitations: (citations) =>
            setResponse((prev) => (prev ? { ...prev, citations } : prev)),
          onGrounding: (g) =>
            setResponse((prev) =>
              prev
                ? {
                    ...prev,
                    grounding_score: g.grounding_score,
                    grounded_claim_rate: g.grounded_claim_rate,
                    claim_grounding: g.claim_grounding,
                    warnings: g.warnings,
                  }
                : prev,
            ),
          onDone: (payload) => setResponse(payload),
          onError: (detail) => {
            setError(detail);
            setResponse(null);
          },
        },
        controller.signal,
      );
      refreshMetrics();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const message = err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Query failed";
      setError(message);
      // The streaming placeholder we set at submit-start would otherwise stay
      // on screen as a hollow "Grounded Answer" card alongside the error.
      setResponse(null);
    } finally {
      setLoading(false);
      setStreaming(false);
    }
  }

  function cancel() {
    abortRef.current?.abort();
    setLoading(false);
    setStreaming(false);
  }

  return (
    <main className="min-h-screen">
      <div className="mx-auto max-w-7xl px-4 pt-6 pb-2 sm:px-6 lg:px-8">
        <h1 className="text-xl font-semibold tracking-normal sm:text-2xl">
          Grounded query workspace
        </h1>
        <p className="text-muted mt-1 text-sm">
          Hybrid retrieval, neural reranking, per-claim grounding, and an explicit
          insufficient-evidence exit.
        </p>
      </div>

      <div className="mx-auto grid max-w-7xl gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[minmax(0,1fr)_360px] lg:px-8">
        <section className="space-y-5">
          <form onSubmit={submit} className="rounded-lg border bg-elev p-4 shadow-sm">
            <label htmlFor="question" className="text-sm font-medium">
              Ask a question
            </label>
            <div className="mt-3 grid gap-3">
              <textarea
                id="question"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask about your ingested corpus..."
                className="min-h-28 resize-y rounded-lg border bg-sunken px-3 py-3 text-base leading-6 outline-none transition-shadow placeholder:text-subtle focus:ring-focus"
              />
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <label className="inline-flex items-center gap-2 text-sm text-muted">
                  <input
                    type="checkbox"
                    checked={bypassCache}
                    onChange={(event) => setBypassCache(event.target.checked)}
                    className="h-4 w-4 accent-[var(--color-accent)]"
                  />
                  Bypass answer cache
                </label>
                <div className="flex items-center gap-2">
                  {loading ? (
                    <button
                      type="button"
                      onClick={cancel}
                      className="inline-flex h-10 items-center gap-2 rounded-lg border bg-elev px-4 text-sm font-medium transition-colors hover:border-strong"
                    >
                      <XCircle size={16} />
                      Cancel
                    </button>
                  ) : null}
                  <button
                    type="submit"
                    disabled={loading || question.trim().length < 3}
                    className="inline-flex h-10 items-center gap-2 rounded-lg bg-accent px-4 text-sm font-semibold text-accent-fg transition-opacity disabled:cursor-not-allowed disabled:opacity-55"
                  >
                    {loading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                    Run query
                  </button>
                </div>
              </div>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {exampleQuestions.map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => setQuestion(item)}
                  className="rounded-full border px-3 py-1 text-xs text-muted transition-colors hover:border-strong hover:text-[var(--color-fg)]"
                >
                  {item}
                </button>
              ))}
            </div>
            <AdvancedOptions
              open={advancedOpen}
              onToggle={() => setAdvancedOpen((v) => !v)}
              verifier={verifierOverride}
              onVerifierChange={setVerifierOverride}
              structured={structuredOverride}
              onStructuredChange={setStructuredOverride}
            />
          </form>

          {error ? (
            <div className="rounded-lg border border-danger/40 bg-danger/10 p-4 text-sm">
              <div className="flex items-start gap-2 font-medium text-danger">
                <AlertTriangle size={17} className="mt-0.5 shrink-0" />
                Query failed
              </div>
              <p className="mt-2 break-words text-muted">{error}</p>
            </div>
          ) : null}

          <section className="rounded-lg border bg-elev shadow-sm">
            <div className="flex flex-col gap-3 border-b p-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-start gap-3">
                <div
                  className={cn(
                    "mt-0.5 rounded-lg border p-2",
                    status.tone === "success" && "border-success/30 bg-success/10 text-success",
                    status.tone === "warning" && "border-warning/30 bg-warning/10 text-warning",
                    status.tone === "neutral" && "bg-sunken text-muted",
                  )}
                >
                  <StatusIcon size={18} className={cn(streaming && "animate-spin")} />
                </div>
                <div>
                  <h2 className="font-semibold">{status.label}</h2>
                  <p className="text-sm text-muted">{status.detail}</p>
                </div>
              </div>
              {response && !streaming && response.trace_id ? (
                <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
                  <span className="rounded-full border px-2 py-1">Trace {response.trace_id.slice(0, 10)}</span>
                  <span className="rounded-full border px-2 py-1">{formatMs(response.total_duration_ms)}</span>
                  <ExportButtons
                    question={question}
                    bypassCache={bypassCache}
                    response={response}
                  />
                </div>
              ) : null}
            </div>

            <div className="p-4">
              {streaming && (!response || response.answer.length === 0) ? (
                <div className="grid min-h-56 place-items-center rounded-lg border border-dashed bg-sunken text-sm text-muted">
                  <div className="flex items-center gap-2">
                    <Loader2 size={18} className="animate-spin" />
                    {response && response.pipeline_trace.length > 0
                      ? `Generating answer · ${response.pipeline_trace[response.pipeline_trace.length - 1].node}`
                      : "Running retrieval, answer synthesis, and critique"}
                  </div>
                </div>
              ) : response ? (
                <AnswerPanel response={response} streaming={streaming} />
              ) : (
                <div className="grid min-h-56 place-items-center rounded-lg border border-dashed bg-sunken p-6 text-center">
                  <div>
                    <FileText className="mx-auto text-subtle" size={28} />
                    <p className="mt-3 font-medium">No query run yet</p>
                    <p className="mt-1 text-sm text-muted">Submit a question to inspect the answer, citations, claims, and pipeline trace.</p>
                  </div>
                </div>
              )}
            </div>
          </section>
        </section>

        <aside className="space-y-5">
          <HealthCard response={response} metrics={metrics} metricsError={metricsError} metricsLoading={metricsLoading} />
          <GroundingCard response={response} />
          <TraceSidebar
            trace={response?.pipeline_trace ?? []}
            loading={loading}
            totalDurationMs={response?.total_duration_ms}
          />
        </aside>
      </div>
    </main>
  );
}

function AnswerPanel({ response, streaming }: { response: QueryResponse; streaming: boolean }) {
  const handleCitationClick = useCallback((id: number) => {
    const el = document.getElementById(`citation-${id}`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    // Brief highlight so the eye lands on the right row.
    el.classList.add("citation-flash");
    window.setTimeout(() => el.classList.remove("citation-flash"), 1100);
  }, []);

  return (
    <div className="space-y-5">
      <div className="relative">
        <AnswerWithGrounding
          answer={response.answer}
          claims={response.claim_grounding}
          citations={response.citations}
          onCitationClick={handleCitationClick}
        />
        {streaming && (
          <span
            aria-hidden
            className="ml-0.5 inline-block h-[1.05em] w-[2px] translate-y-[2px] animate-pulse bg-accent align-middle"
          />
        )}
      </div>

      {response.warnings.length ? (
        <div className="rounded-lg border border-warning/30 bg-warning/10 p-3">
          <div className="flex items-center gap-2 text-sm font-medium text-warning">
            <AlertTriangle size={16} />
            Warnings
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {response.warnings.map((warning) => (
              <span key={warning} className="rounded-full border border-warning/30 px-2 py-1 text-xs text-muted">
                {warning}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      <Citations citations={response.citations} />
    </div>
  );
}

function Citations({ citations }: { citations: Citation[] }) {
  return (
    <section>
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold">Citations</h3>
        <span className="text-xs text-muted">{citations.length} sources</span>
      </div>
      {citations.length ? (
        <div className="mt-3 grid gap-2">
          {citations.map((citation) => (
            <div
              key={`${citation.id}-${citation.source}`}
              id={`citation-${citation.id}`}
              className="scroll-mt-24 rounded-lg border bg-sunken p-3 transition-colors"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-sm font-medium">
                    [{citation.id}] {citation.source}
                  </div>
                  <div className="mt-1 text-xs text-muted">
                    Page {citation.page ?? "unknown"} · Score{" "}
                    {citation.score === null ? "n/a" : citation.score.toFixed(3)}
                  </div>
                  {citation.content && (
                    <p className="mt-2 font-mono text-[12px] leading-snug text-muted">
                      {citation.content}
                    </p>
                  )}
                </div>
                <Database size={16} className="shrink-0 text-subtle" />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-2 rounded-lg border border-dashed bg-sunken p-3 text-sm text-muted">No citations were returned.</p>
      )}
    </section>
  );
}

function HealthCard({
  response,
  metrics,
  metricsError,
  metricsLoading,
}: {
  response: QueryResponse | null;
  metrics: MetricsSnapshot | null;
  metricsError: string | null;
  metricsLoading: boolean;
}) {
  const items = [
    { label: "Total queries", value: metricValue(metrics, "api.query.total"), icon: BarChart3 },
    { label: "OK responses", value: metricValue(metrics, "api.query.status.ok"), icon: CheckCircle2 },
    { label: "Refusals", value: metricValue(metrics, "api.query.status.insufficient_evidence"), icon: AlertTriangle },
    { label: "Avg latency", value: sampleMean(metrics, "api.query.latency_ms"), icon: Clock3 },
  ];

  return (
    <section className="rounded-lg border bg-elev p-4 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <h2 className="font-semibold">Runtime</h2>
        {metricsLoading ? <Loader2 size={16} className="animate-spin text-muted" /> : null}
      </div>
      {metricsError ? <p className="mt-2 break-words text-xs text-danger">{metricsError}</p> : null}
      <div className="mt-4 grid grid-cols-2 gap-3">
        {items.map((item) => (
          <div key={item.label} className="rounded-lg border bg-sunken p-3">
            <item.icon size={15} className="text-subtle" />
            <div className="mt-2 text-lg font-semibold">{item.value}</div>
            <div className="text-xs text-muted">{item.label}</div>
          </div>
        ))}
      </div>
      {response ? (
        <div className="mt-4 rounded-lg border bg-sunken p-3 text-sm">
          <div className="flex items-center justify-between gap-3">
            <span className="text-muted">Last run</span>
            <span className="font-medium">{formatMs(response.total_duration_ms)}</span>
          </div>
          <div className="mt-2 flex items-center justify-between gap-3">
            <span className="text-muted">Iterations</span>
            <span className="font-medium">{response.iteration}</span>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function GroundingCard({ response }: { response: QueryResponse | null }) {
  const grounding = response?.grounding_score ?? 0;
  const claimRate = response?.grounded_claim_rate ?? 0;

  return (
    <section className="rounded-lg border bg-elev p-4 shadow-sm">
      <h2 className="font-semibold">Grounding</h2>
      <div className="mt-4 space-y-4">
        <ScoreBar label="Overall score" value={grounding} />
        <ScoreBar label="Grounded claims" value={claimRate} />
      </div>
    </section>
  );
}

/** Collapsible per-request overrides for backend opt-in features. */
function AdvancedOptions({
  open,
  onToggle,
  verifier,
  onVerifierChange,
  structured,
  onStructuredChange,
}: {
  open: boolean;
  onToggle: () => void;
  verifier: "default" | ClaimVerifierMode;
  onVerifierChange: (v: "default" | ClaimVerifierMode) => void;
  structured: "default" | "off" | "on";
  onStructuredChange: (v: "default" | "off" | "on") => void;
}) {
  const dirty =
    verifier !== "default" || structured !== "default";
  return (
    <div className="mt-3 border-t pt-3">
      <button
        type="button"
        onClick={onToggle}
        className="text-muted hover:text-[var(--color-fg)] inline-flex items-center gap-1.5 text-xs font-medium transition-colors"
        aria-expanded={open}
      >
        <span>{open ? "▾" : "▸"}</span>
        Advanced
        {dirty && (
          <span
            className="ml-1 inline-block h-1.5 w-1.5 rounded-full"
            style={{ background: "var(--color-accent)" }}
            aria-label="Per-request overrides active"
          />
        )}
      </button>
      {open && (
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <OverridePicker
            label="Per-claim verifier"
            value={verifier}
            options={[
              { value: "default", label: "Default" },
              { value: "overlap", label: "Overlap" },
              { value: "nli", label: "NLI" },
            ]}
            onChange={(v) => onVerifierChange(v as "default" | ClaimVerifierMode)}
            hint="NLI uses an entailment cross-encoder. ~700MB model; loads once."
          />
          <OverridePicker
            label="Structured answers"
            value={structured}
            options={[
              { value: "default", label: "Default" },
              { value: "off", label: "Off" },
              { value: "on", label: "On" },
            ]}
            onChange={(v) => onStructuredChange(v as "default" | "off" | "on")}
            hint="On = LLM emits per-claim JSON via function calling. Disables token streaming."
          />
        </div>
      )}
    </div>
  );
}

/** Compact segmented control for one tri-state override. */
function OverridePicker({
  label,
  value,
  options,
  onChange,
  hint,
}: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (v: string) => void;
  hint?: string;
}) {
  return (
    <div>
      <div className="text-muted text-xs font-medium">{label}</div>
      <div className="bg-sunken mt-1.5 inline-flex rounded-md border p-0.5">
        {options.map((opt) => {
          const active = opt.value === value;
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => onChange(opt.value)}
              aria-pressed={active}
              className={cn(
                "rounded px-2.5 py-1 text-xs font-medium transition-colors",
                active
                  ? "bg-elev shadow-sm"
                  : "text-muted hover:text-[var(--color-fg)]",
              )}
              style={
                active ? { color: "var(--color-accent)" } : undefined
              }
            >
              {opt.label}
            </button>
          );
        })}
      </div>
      {hint && (
        <p className="text-subtle mt-1.5 text-[11px] leading-snug">{hint}</p>
      )}
    </div>
  );
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className="text-muted">{label}</span>
        <span className="font-semibold">{percent(value)}</span>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-sunken">
        <div className="h-full bg-accent transition-all" style={{ width: `${Math.max(4, Math.min(100, value * 100))}%` }} />
      </div>
    </div>
  );
}

/** Copy-to-clipboard + download-JSON buttons for the current run. Renders
 *  inline alongside the trace-id and duration chips so the export controls
 *  live next to the artefacts they describe. */
function ExportButtons({
  question,
  bypassCache,
  response,
}: {
  question: string;
  bypassCache: boolean;
  response: QueryResponse;
}) {
  const [justCopied, setJustCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);

  async function onCopy() {
    const artifact = buildRunArtifact(question, bypassCache, response);
    const ok = await copyJsonToClipboard(artifact);
    if (ok) {
      setCopyError(false);
      setJustCopied(true);
      window.setTimeout(() => setJustCopied(false), 1400);
    } else {
      setCopyError(true);
      window.setTimeout(() => setCopyError(false), 2000);
    }
  }

  function onDownload() {
    const artifact = buildRunArtifact(question, bypassCache, response);
    downloadJson(defaultExportFilename(response), artifact);
  }

  return (
    <div className="inline-flex overflow-hidden rounded-full border">
      <button
        type="button"
        onClick={onCopy}
        title={
          copyError
            ? "Clipboard unavailable"
            : justCopied
              ? "Copied"
              : "Copy run as JSON"
        }
        aria-label="Copy run as JSON"
        className="text-muted hover:bg-sunken inline-flex h-7 items-center gap-1 px-2 transition-colors"
        style={
          copyError
            ? { color: "var(--color-danger)" }
            : justCopied
              ? { color: "var(--color-success)" }
              : undefined
        }
      >
        {justCopied ? <Check size={12} /> : <Copy size={12} />}
        <span className="text-[11px]">{justCopied ? "Copied" : "Copy"}</span>
      </button>
      <span className="border-l" aria-hidden />
      <button
        type="button"
        onClick={onDownload}
        title="Download run as .json"
        aria-label="Download run as JSON"
        className="text-muted hover:bg-sunken inline-flex h-7 items-center gap-1 px-2 transition-colors"
      >
        <Download size={12} />
        <span className="text-[11px]">JSON</span>
      </button>
    </div>
  );
}

