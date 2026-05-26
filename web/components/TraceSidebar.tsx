"use client";

import {
  ArrowRight,
  CheckCircle2,
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

interface IterationBlock {
  iteration: number;
  entries: PipelineTraceEntry[];
}

const PENDING_STAGES = ["retrieve", "answer", "critique", "finalize"] as const;

function groupByIteration(trace: PipelineTraceEntry[]): IterationBlock[] {
  if (trace.length === 0) return [];
  const blocks: IterationBlock[] = [];
  let current: IterationBlock | null = null;
  for (const entry of trace) {
    if (!current || entry.iteration !== current.iteration) {
      current = { iteration: entry.iteration, entries: [] };
      blocks.push(current);
    }
    current.entries.push(entry);
  }
  return blocks;
}

export function TraceSidebar({
  trace,
  loading,
  totalDurationMs,
}: TraceSidebarProps) {
  const blocks = useMemo(() => groupByIteration(trace), [trace]);

  // Staggered reveal: when ``trace`` changes, fade entries in one at a time.
  // Cheap way to make a one-shot response feel like watching the engine
  // execute, without needing real SSE from the backend.
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
          <span className="text-subtle text-xs">
            {trace.length} nodes
            {totalDurationMs ? ` · ${formatMs(totalDurationMs)}` : ""}
          </span>
        ) : loading ? (
          <Loader2 size={14} className="text-subtle animate-spin" />
        ) : null}
      </header>

      <div className="mt-4">
        {trace.length === 0 ? (
          <PreviewStages loading={loading} />
        ) : (
          blocks.map((block, blockIdx) => (
            <IterationGroup
              key={block.iteration}
              block={block}
              isRewrite={blockIdx > 0}
              revealOffset={blocks
                .slice(0, blockIdx)
                .reduce((sum, b) => sum + b.entries.length, 0)}
              revealed={revealed}
            />
          ))
        )}
      </div>
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
          style={{
            opacity: loading ? 1 : 0.55,
          }}
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

function IterationGroup({
  block,
  isRewrite,
  revealOffset,
  revealed,
}: {
  block: IterationBlock;
  isRewrite: boolean;
  revealOffset: number;
  revealed: number;
}) {
  return (
    <div className="mb-4 last:mb-0">
      <div className="text-subtle mb-1.5 flex items-center gap-2 font-mono text-[10px] uppercase tracking-widest">
        <span>Pass {block.iteration + 1}</span>
        {isRewrite && (
          <>
            <ArrowRight size={9} className="opacity-60" />
            <RefreshCw size={9} className="opacity-60" />
            <span className="opacity-80">after rewrite</span>
          </>
        )}
      </div>
      <ol className="space-y-1.5 border-l-2 pl-3" style={{ borderColor: "var(--color-border)" }}>
        {block.entries.map((entry, i) => {
          const globalIndex = revealOffset + i;
          const isVisible = globalIndex < revealed;
          return (
            <TraceEntryRow
              key={`${entry.node}-${globalIndex}`}
              entry={entry}
              visible={isVisible}
            />
          );
        })}
      </ol>
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

function formatExtra(key: string, value: unknown): string {
  if (typeof value === "number") {
    if (key === "grounding_score" || key === "grounded_claim_rate") {
      return value.toFixed(2);
    }
    return Number.isInteger(value) ? value.toString() : value.toFixed(2);
  }
  return String(value);
}

function TraceEntryRow({
  entry,
  visible,
}: {
  entry: PipelineTraceEntry;
  visible: boolean;
}) {
  const accent =
    entry.node === "fallback"
      ? "var(--color-warning)"
      : entry.node === "rewrite"
        ? "var(--color-fg-muted)"
        : "var(--color-accent)";
  const wantedExtras = EXTRA_KEYS_BY_NODE[entry.node] ?? [];
  const extras = wantedExtras
    .map((k) => [k, entry[k]] as const)
    .filter(([, v]) => v !== undefined);

  return (
    <li
      className="bg-sunken rounded-md border p-2.5"
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(4px)",
        transition: "opacity 220ms ease, transform 220ms ease",
      }}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <CheckCircle2 size={12} className="shrink-0" style={{ color: accent }} />
          <span className="text-[13px] font-medium capitalize">{entry.node}</span>
        </div>
        <span className="text-subtle shrink-0 font-mono text-[11px]">
          {formatMs(entry.duration_ms)}
        </span>
      </div>
      {extras.length > 0 && (
        <div className="text-muted mt-1.5 ml-[20px] flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[10.5px]">
          {extras.map(([k, v]) => (
            <span key={k}>
              <span className="text-subtle">{k.replace(/_/g, " ")}</span> {formatExtra(k, v)}
            </span>
          ))}
        </div>
      )}
    </li>
  );
}
