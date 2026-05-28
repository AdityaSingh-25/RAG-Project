"use client";

import {
  ArrowRight,
  Circle,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import type { PipelineTraceEntry } from "@/lib/types";
import { formatMs } from "@/lib/utils";

interface TraceSidebarProps {
  trace: PipelineTraceEntry[];
  loading: boolean;
  totalDurationMs?: number;
}

const PENDING_STAGES = ["retrieve", "answer", "critique", "finalize"] as const;

interface Row {
  entry: PipelineTraceEntry;
  /** Cumulative start time in ms, derived by summing prior durations. */
  startMs: number;
  /** True when this row's iteration is the first of a new pass (>0). */
  startsNewPass: boolean;
}

/** Annotate each entry with its cumulative start time. Stages are sequential
 *  in the LangGraph state machine, so start = sum of preceding durations.    */
function buildRows(trace: PipelineTraceEntry[]): Row[] {
  const rows: Row[] = [];
  let cursor = 0;
  let lastIteration = -1;
  for (const entry of trace) {
    const startsNewPass = entry.iteration !== lastIteration && entry.iteration > 0;
    rows.push({ entry, startMs: cursor, startsNewPass });
    cursor += Number(entry.duration_ms) || 0;
    lastIteration = entry.iteration;
  }
  return rows;
}

function nodeColor(node: PipelineTraceEntry["node"]): string {
  if (node === "fallback") return "var(--color-warning)";
  if (node === "rewrite") return "var(--color-fg-muted)";
  return "var(--color-accent)";
}

export function TraceSidebar({
  trace,
  loading,
  totalDurationMs,
}: TraceSidebarProps) {
  const rows = useMemo(() => buildRows(trace), [trace]);
  const measuredTotal = rows.length
    ? rows[rows.length - 1].startMs + (Number(rows[rows.length - 1].entry.duration_ms) || 0)
    : 0;
  // Prefer the server-reported total when present; otherwise the sum of what
  // we've seen so far. During streaming totalDurationMs is 0 until `done`.
  const scaleMs = Math.max(1, totalDurationMs || measuredTotal);

  // Staggered reveal so streamed entries fade in one at a time.
  const [revealed, setRevealed] = useState(0);
  useEffect(() => {
    setRevealed(0);
  }, [trace]);
  useEffect(() => {
    if (revealed >= trace.length) return;
    const t = window.setTimeout(() => setRevealed((n) => n + 1), 80);
    return () => window.clearTimeout(t);
  }, [revealed, trace.length]);

  return (
    <section className="bg-elev rounded-lg border p-4 shadow-sm">
      <header className="flex items-center justify-between gap-3">
        <h2 className="font-semibold">Pipeline Trace</h2>
        {trace.length > 0 ? (
          <span className="text-subtle font-mono text-[11px]">
            {trace.length} nodes{measuredTotal ? ` · ${formatMs(measuredTotal)}` : ""}
          </span>
        ) : loading ? (
          <Loader2 size={14} className="text-subtle animate-spin" />
        ) : null}
      </header>

      <div className="mt-4">
        {rows.length === 0 ? (
          <PreviewStages loading={loading} />
        ) : (
          <Waterfall rows={rows} scaleMs={scaleMs} revealed={revealed} />
        )}
      </div>

      {rows.length > 0 && (
        <div className="text-subtle mt-2 flex justify-between font-mono text-[10px]">
          <span>0 ms</span>
          <span>{formatMs(scaleMs)}</span>
        </div>
      )}
    </section>
  );
}

/** Placeholder pipeline shown before any query has run, and while loading. */
function PreviewStages({ loading }: { loading: boolean }) {
  return (
    <ol className="space-y-1.5">
      {PENDING_STAGES.map((stage, i) => (
        <li
          key={stage}
          className="bg-sunken flex items-center gap-2.5 rounded-md border p-2.5"
          style={{ opacity: loading ? 1 : 0.55 }}
        >
          {loading && i === 0 ? (
            <Loader2
              size={13}
              className="shrink-0 animate-spin"
              style={{ color: "var(--color-accent)" }}
            />
          ) : (
            <Circle size={11} className="text-subtle shrink-0" />
          )}
          <span className="text-muted text-[13px] font-medium capitalize">
            {stage}
          </span>
          {loading && i === 0 && (
            <span className="text-subtle ml-auto font-mono text-[10px]">running…</span>
          )}
        </li>
      ))}
    </ol>
  );
}

function Waterfall({
  rows,
  scaleMs,
  revealed,
}: {
  rows: Row[];
  scaleMs: number;
  revealed: number;
}) {
  return (
    <ol className="space-y-1.5">
      {rows.map((row, i) => {
        const visible = i < revealed;
        return (
          <li
            key={`${row.entry.node}-${i}`}
            style={{
              opacity: visible ? 1 : 0,
              transform: visible ? "translateY(0)" : "translateY(3px)",
              transition: "opacity 220ms ease, transform 220ms ease",
            }}
          >
            {row.startsNewPass && (
              <div className="text-subtle mb-1 mt-2 flex items-center gap-1.5 font-mono text-[9px] uppercase tracking-widest">
                <ArrowRight size={9} className="opacity-60" />
                <RefreshCw size={9} className="opacity-60" />
                <span>pass {row.entry.iteration + 1}</span>
              </div>
            )}
            <WaterfallRow row={row} scaleMs={scaleMs} />
          </li>
        );
      })}
    </ol>
  );
}

function WaterfallRow({ row, scaleMs }: { row: Row; scaleMs: number }) {
  const { entry, startMs } = row;
  const durationMs = Number(entry.duration_ms) || 0;
  const leftPct = (startMs / scaleMs) * 100;
  const widthPct = Math.max(1.5, (durationMs / scaleMs) * 100);
  const color = nodeColor(entry.node);
  const title = buildHoverTitle(entry);

  return (
    <div
      className="grid items-center gap-2 rounded-md border px-2 py-1.5"
      style={{
        background: "var(--color-bg-sunken)",
        gridTemplateColumns: "64px 1fr 44px",
      }}
      title={title}
    >
      <span className="text-[12px] font-medium capitalize">{entry.node}</span>
      <div className="relative h-2.5 overflow-hidden rounded-full">
        <div
          className="absolute inset-0"
          style={{
            background: "color-mix(in oklch, var(--color-border), transparent 40%)",
          }}
        />
        <div
          className="absolute top-0 bottom-0 rounded-full"
          style={{
            left: `${leftPct}%`,
            width: `${widthPct}%`,
            background: color,
            transition: "left 200ms ease, width 200ms ease",
          }}
        />
      </div>
      <span className="text-subtle text-right font-mono text-[10.5px] tabular-nums">
        {formatMs(durationMs)}
      </span>
    </div>
  );
}

const EXTRA_KEYS_BY_NODE: Record<string, readonly string[]> = {
  retrieve: ["n_candidates", "n_reranked"],
  answer: ["model", "answer_chars"],
  critique: ["grounding_score", "grounded_claim_rate", "n_claims"],
  rewrite: ["rewritten_chars"],
  fallback: ["reason"],
  finalize: [],
};

/** Compose a native-tooltip string with the per-node diagnostic fields.
 *  Sidebar width is too tight to render these inline as before, so they
 *  live in `title` — keyboard / screen-reader friendly, no JS popover.   */
function buildHoverTitle(entry: PipelineTraceEntry): string {
  const lines = [`${entry.node} · ${formatMs(Number(entry.duration_ms) || 0)}`];
  if (entry.iteration > 0) {
    lines.push(`iteration ${entry.iteration}`);
  }
  const wantedExtras = EXTRA_KEYS_BY_NODE[entry.node] ?? [];
  for (const k of wantedExtras) {
    const v = entry[k];
    if (v === undefined) continue;
    lines.push(`${k.replace(/_/g, " ")}: ${formatExtra(k, v)}`);
  }
  return lines.join("\n");
}

function formatExtra(key: string, value: unknown): string {
  if (typeof value === "number") {
    if (key === "grounding_score" || key === "grounded_claim_rate") {
      return value.toFixed(2);
    }
    return Number.isInteger(value) ? value.toString() : value.toFixed(2);
  }
  return String(value);
}
