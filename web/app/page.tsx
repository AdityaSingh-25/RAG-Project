"use client";

import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Clock3,
  Database,
  FileText,
  Loader2,
  RefreshCw,
  Send,
  ShieldCheck,
  Sparkles,
  XCircle,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AnswerWithGrounding } from "@/components/AnswerWithGrounding";
import { ThemeToggle } from "@/components/ThemeToggle";
import { TraceSidebar } from "@/components/TraceSidebar";
import { ApiError, getMetrics, runQuery } from "@/lib/api";
import type { Citation, MetricsSnapshot, QueryResponse } from "@/lib/types";
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

function statusCopy(response: QueryResponse | null) {
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

function sampleMean(metrics: MetricsSnapshot | null, key: string): string {
  const mean = metrics?.samples[key]?.mean;
  return mean === undefined ? "No samples" : formatMs(mean);
}

export default function Home() {
  const [question, setQuestion] = useState(exampleQuestions[0]);
  const [bypassCache, setBypassCache] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const status = useMemo(() => statusCopy(response), [response]);
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
    setError(null);

    try {
      const result = await runQuery({ question: trimmed, bypass_cache: bypassCache }, controller.signal);
      setResponse(result);
      refreshMetrics();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const message = err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Query failed";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  function cancel() {
    abortRef.current?.abort();
    setLoading(false);
  }

  return (
    <main className="min-h-screen">
      <header className="border-b bg-elev/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-sm font-medium text-accent">
              <ShieldCheck size={16} />
              Multi-Agent RAG Intelligence Engine
            </div>
            <h1 className="mt-1 truncate text-xl font-semibold tracking-normal sm:text-2xl">Grounded query workspace</h1>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => refreshMetrics()}
              className="inline-flex h-9 items-center gap-2 rounded-lg border bg-elev px-3 text-sm font-medium transition-colors hover:border-strong disabled:cursor-not-allowed disabled:opacity-60"
              disabled={metricsLoading}
            >
              <RefreshCw size={15} className={cn(metricsLoading && "animate-spin")} />
              Metrics
            </button>
            <ThemeToggle />
          </div>
        </div>
      </header>

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
                  <StatusIcon size={18} />
                </div>
                <div>
                  <h2 className="font-semibold">{status.label}</h2>
                  <p className="text-sm text-muted">{status.detail}</p>
                </div>
              </div>
              {response ? (
                <div className="flex flex-wrap gap-2 text-xs text-muted">
                  <span className="rounded-full border px-2 py-1">Trace {response.trace_id.slice(0, 10)}</span>
                  <span className="rounded-full border px-2 py-1">{formatMs(response.total_duration_ms)}</span>
                </div>
              ) : null}
            </div>

            <div className="p-4">
              {loading && !response ? (
                <div className="grid min-h-56 place-items-center rounded-lg border border-dashed bg-sunken text-sm text-muted">
                  <div className="flex items-center gap-2">
                    <Loader2 size={18} className="animate-spin" />
                    Running retrieval, answer synthesis, and critique
                  </div>
                </div>
              ) : response ? (
                <AnswerPanel response={response} />
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

function AnswerPanel({ response }: { response: QueryResponse }) {
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
      <AnswerWithGrounding
        answer={response.answer}
        claims={response.claim_grounding}
        citations={response.citations}
        onCitationClick={handleCitationClick}
      />

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

