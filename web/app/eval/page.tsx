"use client";

import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ClipboardList,
  Loader2,
  Play,
  XCircle,
} from "lucide-react";
import { FormEvent, useRef, useState } from "react";

import { ApiError, runEval } from "@/lib/api";
import type {
  EvalAggregate,
  EvalCaseResult,
  EvalDataset,
  EvalReport,
} from "@/lib/types";
import { cn, formatMs } from "@/lib/utils";

const DATASETS: Array<{ value: EvalDataset; label: string; hint: string }> = [
  {
    value: "seed",
    label: "Seed",
    hint: "Ok-cases that should be grounded against the corpus.",
  },
  {
    value: "adversarial",
    label: "Adversarial",
    hint: "Cases the engine should refuse with insufficient_evidence.",
  },
];

export default function EvalPage() {
  const [dataset, setDataset] = useState<EvalDataset>("seed");
  const [limit, setLimit] = useState(5);
  const [report, setReport] = useState<EvalReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (running) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setError(null);
    setReport(null);
    try {
      const next = await runEval({ dataset, limit }, controller.signal);
      setReport(next);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  function cancel() {
    abortRef.current?.abort();
    setRunning(false);
  }

  return (
    <main className="mx-auto max-w-5xl px-4 py-6 sm:px-6 lg:px-8">
      <div className="mb-6">
        <h1 className="text-xl font-semibold tracking-normal sm:text-2xl">Evaluation</h1>
        <p className="text-muted mt-1 text-sm">
          Replay the seed or adversarial dataset against the live graph and score
          grounding, citation hits, term recall, and status-match rate. Long-running
          — each case re-invokes the full pipeline.
        </p>
      </div>

      <form onSubmit={submit} className="bg-elev mb-6 rounded-lg border p-4 shadow-sm">
        <div className="grid gap-4 sm:grid-cols-[1fr_auto_auto] sm:items-end">
          <DatasetPicker value={dataset} onChange={setDataset} disabled={running} />
          <LimitInput value={limit} onChange={setLimit} disabled={running} />
          <div className="flex items-center gap-2">
            {running ? (
              <button
                type="button"
                onClick={cancel}
                className="bg-elev hover:border-strong inline-flex h-10 items-center gap-2 rounded-md border px-4 text-sm font-medium transition-colors"
              >
                <XCircle size={15} />
                Cancel
              </button>
            ) : null}
            <button
              type="submit"
              disabled={running}
              className="inline-flex h-10 items-center gap-2 rounded-md px-4 text-sm font-semibold transition-opacity disabled:opacity-50"
              style={{
                background: "var(--color-accent)",
                color: "var(--color-accent-fg)",
              }}
            >
              {running ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
              {running ? "Running…" : "Run"}
            </button>
          </div>
        </div>
        <p className="text-subtle mt-3 text-[11.5px]">
          Each case re-invokes the graph, so expect roughly{" "}
          <span className="font-mono">limit × per-query latency</span> in wall-clock time.
        </p>
      </form>

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
            <div className="font-medium">Eval failed</div>
            <div className="mt-0.5 font-mono text-[12px] opacity-90">{error}</div>
          </div>
        </div>
      )}

      {running && !report && <RunningPlaceholder limit={limit} />}

      {report && (
        <>
          <AggregateCard aggregate={report.aggregate} dataset={report.dataset} />
          <ResultsTable
            results={report.results}
            limit={report.limit}
            totalAvailable={report.total_cases_available}
          />
        </>
      )}
    </main>
  );
}

function DatasetPicker({
  value,
  onChange,
  disabled,
}: {
  value: EvalDataset;
  onChange: (v: EvalDataset) => void;
  disabled: boolean;
}) {
  return (
    <div>
      <div className="text-muted mb-1.5 text-xs font-medium">Dataset</div>
      <div className="bg-sunken inline-flex rounded-md border p-0.5">
        {DATASETS.map((d) => {
          const active = d.value === value;
          return (
            <button
              key={d.value}
              type="button"
              onClick={() => onChange(d.value)}
              disabled={disabled}
              title={d.hint}
              className={cn(
                "rounded px-3 py-1 text-xs font-medium transition-colors disabled:opacity-50",
                active ? "bg-elev shadow-sm" : "text-muted hover:text-[var(--color-fg)]",
              )}
              style={active ? { color: "var(--color-accent)" } : undefined}
            >
              {d.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function LimitInput({
  value,
  onChange,
  disabled,
}: {
  value: number;
  onChange: (n: number) => void;
  disabled: boolean;
}) {
  return (
    <div>
      <div className="text-muted mb-1.5 text-xs font-medium">Limit</div>
      <input
        type="number"
        min={1}
        max={100}
        value={value}
        onChange={(e) => {
          const n = parseInt(e.target.value, 10);
          onChange(Number.isFinite(n) ? Math.max(1, Math.min(100, n)) : 1);
        }}
        disabled={disabled}
        className="bg-sunken focus:ring-focus focus:border-strong h-10 w-24 rounded-md border px-3 text-center font-mono text-[13px] outline-none disabled:opacity-60"
      />
    </div>
  );
}

function RunningPlaceholder({ limit }: { limit: number }) {
  return (
    <div className="bg-elev rounded-lg border px-6 py-10 text-center">
      <Loader2 className="text-accent mx-auto animate-spin" size={28} />
      <p className="mt-3 font-medium">Running up to {limit} cases…</p>
      <p className="text-muted mt-1 text-sm">
        Each case re-runs the full pipeline (retrieve → answer → critique).
      </p>
    </div>
  );
}

function AggregateCard({
  aggregate,
  dataset,
}: {
  aggregate: EvalAggregate;
  dataset: EvalDataset;
}) {
  const statusMatchTone =
    aggregate.status_match_rate >= 0.9
      ? "var(--color-success)"
      : aggregate.status_match_rate >= 0.7
        ? "var(--color-warning)"
        : "var(--color-danger)";
  return (
    <section className="bg-elev mb-6 rounded-lg border p-5 shadow-sm">
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-semibold">Aggregate · {dataset}</h2>
        <span className="text-subtle font-mono text-[11px]">
          n = {aggregate.n.toLocaleString()}
        </span>
      </div>
      <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-5">
        <Metric
          label="Status match"
          value={`${(aggregate.status_match_rate * 100).toFixed(0)}%`}
          accent={statusMatchTone}
        />
        <Metric
          label="Mean grounding"
          value={aggregate.mean_grounding.toFixed(2)}
        />
        <Metric
          label="Citation hits"
          value={aggregate.mean_citation_hit_rate.toFixed(2)}
        />
        <Metric label="Term recall" value={aggregate.mean_term_recall.toFixed(2)} />
        <Metric label="Mean latency" value={formatMs(aggregate.mean_latency_ms)} />
      </div>
    </section>
  );
}

function Metric({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="bg-sunken rounded-md border p-3">
      <div className="text-subtle text-[10.5px] uppercase tracking-wider">{label}</div>
      <div
        className="mt-1 text-xl font-semibold"
        style={accent ? { color: accent } : undefined}
      >
        {value}
      </div>
    </div>
  );
}

function ResultsTable({
  results,
  limit,
  totalAvailable,
}: {
  results: EvalCaseResult[];
  limit: number | null;
  totalAvailable: number;
}) {
  if (results.length === 0) {
    return (
      <div className="bg-sunken rounded-lg border border-dashed px-6 py-10 text-center">
        <ClipboardList className="text-subtle mx-auto" size={22} />
        <p className="text-muted mt-3 text-sm">No cases were executed.</p>
      </div>
    );
  }
  return (
    <section>
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold">Per-case results</h2>
        <span className="text-subtle font-mono text-[11px]">
          {results.length}
          {limit !== null && limit < totalAvailable
            ? ` of ${totalAvailable} (limited)`
            : ""}
        </span>
      </div>
      <div className="bg-elev overflow-hidden rounded-lg border">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-subtle border-b font-mono text-[10px] uppercase tracking-widest">
              <th className="w-8 px-3 py-2 text-left font-normal"></th>
              <th className="px-3 py-2 text-left font-normal">case</th>
              <th className="px-3 py-2 text-right font-normal">status</th>
              <th className="px-3 py-2 text-right font-normal">grounding</th>
              <th className="px-3 py-2 text-right font-normal">cite-hit</th>
              <th className="px-3 py-2 text-right font-normal">term-recall</th>
              <th className="px-3 py-2 text-right font-normal">latency</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <CaseRow key={r.case_id} result={r} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function CaseRow({ result }: { result: EvalCaseResult }) {
  const [expanded, setExpanded] = useState(false);
  const matches = result.status_matches_expected;
  const statusTone = matches ? "var(--color-success)" : "var(--color-danger)";
  return (
    <>
      <tr
        onClick={() => setExpanded((v) => !v)}
        className="hover:bg-sunken cursor-pointer border-b transition-colors last:border-b-0"
      >
        <td className="px-3 py-2 align-top">
          <span style={{ color: statusTone }}>
            {matches ? (
              <CheckCircle2 size={13} strokeWidth={2.5} />
            ) : (
              <AlertTriangle size={13} strokeWidth={2.5} />
            )}
          </span>
        </td>
        <td className="px-3 py-2 align-top">
          <div className="font-mono text-[12.5px] font-medium">{result.case_id}</div>
          <div className="text-muted mt-0.5 line-clamp-2 text-[12px]">
            {result.question}
          </div>
        </td>
        <td className="px-3 py-2 text-right align-top">
          <div className="font-mono text-[12px]">
            {result.status}
            {!matches && (
              <>
                <ChevronDown size={10} className="mx-0.5 inline opacity-50" />
                <span className="text-subtle">{result.expected_status}</span>
              </>
            )}
          </div>
        </td>
        <td className="px-3 py-2 text-right align-top font-mono text-[12.5px] tabular-nums">
          {result.grounding_score.toFixed(2)}
        </td>
        <td className="px-3 py-2 text-right align-top font-mono text-[12.5px] tabular-nums">
          {result.citation_hit_rate.toFixed(2)}
        </td>
        <td className="px-3 py-2 text-right align-top font-mono text-[12.5px] tabular-nums">
          {result.term_recall.toFixed(2)}
        </td>
        <td className="px-3 py-2 text-right align-top font-mono text-[12px] tabular-nums">
          {formatMs(result.latency_ms)}
        </td>
      </tr>
      {expanded && (
        <tr className="bg-sunken border-b last:border-b-0">
          <td colSpan={7} className="px-4 py-3">
            <div className="text-muted mb-2 text-[10.5px] uppercase tracking-widest">
              Answer
            </div>
            <p className="whitespace-pre-wrap font-mono text-[12.5px] leading-snug">
              {result.answer || "(empty)"}
            </p>
            {result.warnings.length > 0 && (
              <div className="mt-3">
                <div className="text-muted mb-1 text-[10.5px] uppercase tracking-widest">
                  Warnings
                </div>
                <div className="flex flex-wrap gap-1">
                  {result.warnings.map((w) => (
                    <span
                      key={w}
                      className="rounded-full border px-2 py-0.5 text-[11px]"
                      style={{ color: "var(--color-warning)" }}
                    >
                      {w}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {result.citations.length > 0 && (
              <div className="mt-3">
                <div className="text-muted mb-1 text-[10.5px] uppercase tracking-widest">
                  Citations
                </div>
                <ul className="space-y-1">
                  {result.citations.map((c, i) => (
                    <li key={i} className="font-mono text-[11.5px]">
                      [{String(c.id ?? i + 1)}] {String(c.source ?? "?")}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}
