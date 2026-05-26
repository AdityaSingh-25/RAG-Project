"use client";

import { AlertTriangle, CheckCircle2 } from "lucide-react";
import { Fragment, useMemo, useState } from "react";

import type { Citation, ClaimGrounding } from "@/lib/types";

interface AnswerWithGroundingProps {
  answer: string;
  claims: ClaimGrounding[];
  citations: Citation[];
  /** Called when a citation marker is clicked. Used to scroll the page to the
   *  citation card and momentarily highlight it. */
  onCitationClick?: (citationId: number) => void;
}

interface Segment {
  text: string;
  claim: ClaimGrounding | null;
}

/** Walk through ``answer`` and slice it into segments aligned to claim
 *  sentences. The sentence strings in ``claims`` come from
 *  ``split_into_sentences`` on the backend, so each one is a substring of
 *  ``answer``. We find them in order; any text in between (whitespace,
 *  trailing fragments) becomes an un-claimed segment. */
function alignAnswerToClaims(answer: string, claims: ClaimGrounding[]): Segment[] {
  if (claims.length === 0) {
    return [{ text: answer, claim: null }];
  }
  const segments: Segment[] = [];
  let cursor = 0;
  for (const claim of claims) {
    const idx = answer.indexOf(claim.sentence, cursor);
    if (idx === -1) {
      // Sentence not found — bail to a single un-claimed segment so we never
      // mangle the user-visible text.
      return [{ text: answer, claim: null }];
    }
    if (idx > cursor) {
      segments.push({ text: answer.slice(cursor, idx), claim: null });
    }
    segments.push({ text: claim.sentence, claim });
    cursor = idx + claim.sentence.length;
  }
  if (cursor < answer.length) {
    segments.push({ text: answer.slice(cursor), claim: null });
  }
  return segments;
}

export function AnswerWithGrounding({
  answer,
  claims,
  citations,
  onCitationClick,
}: AnswerWithGroundingProps) {
  const segments = useMemo(() => alignAnswerToClaims(answer, claims), [answer, claims]);
  const citationsById = useMemo(() => {
    const map = new Map<number, Citation>();
    for (const c of citations) map.set(c.id, c);
    return map;
  }, [citations]);

  return (
    <p className="text-[15px] leading-7">
      {segments.map((seg, i) =>
        seg.claim ? (
          <ClaimSpan
            key={i}
            claim={seg.claim}
            citationsById={citationsById}
            onCitationClick={onCitationClick}
          />
        ) : (
          <Fragment key={i}>{seg.text}</Fragment>
        ),
      )}
    </p>
  );
}

/** A single grounded/ungrounded sentence with inline interactive [n] markers. */
function ClaimSpan({
  claim,
  citationsById,
  onCitationClick,
}: {
  claim: ClaimGrounding;
  citationsById: Map<number, Citation>;
  onCitationClick?: (id: number) => void;
}) {
  const grounded = claim.is_grounded;
  const Icon = grounded ? CheckCircle2 : AlertTriangle;
  const tone = grounded ? "var(--color-success)" : "var(--color-warning)";

  // Tooltip with the claim's own diagnostic, shown on hover. Pure CSS would
  // be enough, but a state-driven popover gives us better keyboard a11y.
  const [hover, setHover] = useState(false);

  return (
    <span
      className="relative cursor-help rounded-[3px] px-[2px] underline decoration-dotted decoration-2 underline-offset-4"
      style={{
        // Set only the longhand color so Tailwind's `decoration-dotted` and
        // `decoration-2` utility classes survive — `textDecoration` shorthand
        // would reset the style back to solid.
        textDecorationColor: `color-mix(in oklch, ${tone}, transparent 55%)`,
        background: hover
          ? `color-mix(in oklch, ${tone}, transparent 92%)`
          : "transparent",
        transition: "background-color 120ms ease",
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onFocus={() => setHover(true)}
      onBlur={() => setHover(false)}
      tabIndex={0}
      aria-label={
        grounded
          ? `Grounded claim, support ${(claim.support_score * 100).toFixed(0)}%`
          : `Unsupported claim, support ${(claim.support_score * 100).toFixed(0)}%`
      }
    >
      {renderSentenceWithCitations(claim.sentence, citationsById, onCitationClick)}
      {hover && (
        <ClaimTooltip claim={claim} Icon={Icon} tone={tone} />
      )}
    </span>
  );
}

/** Replace each ``[n]`` in ``text`` with a clickable citation chip. */
function renderSentenceWithCitations(
  text: string,
  citationsById: Map<number, Citation>,
  onClick?: (id: number) => void,
) {
  const parts: React.ReactNode[] = [];
  const re = /\[(\d+)\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let nodeKey = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) {
      parts.push(<Fragment key={nodeKey++}>{text.slice(last, m.index)}</Fragment>);
    }
    const id = parseInt(m[1], 10);
    const citation = citationsById.get(id);
    parts.push(
      <CitationChip
        key={nodeKey++}
        id={id}
        citation={citation}
        onClick={onClick}
      />,
    );
    last = m.index + m[0].length;
  }
  if (last < text.length) {
    parts.push(<Fragment key={nodeKey++}>{text.slice(last)}</Fragment>);
  }
  return parts;
}

function CitationChip({
  id,
  citation,
  onClick,
}: {
  id: number;
  citation: Citation | undefined;
  onClick?: (id: number) => void;
}) {
  const [hover, setHover] = useState(false);
  const valid = citation !== undefined;

  return (
    <span
      className="relative mx-0.5 inline-flex"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <button
        type="button"
        onClick={() => onClick?.(id)}
        onFocus={() => setHover(true)}
        onBlur={() => setHover(false)}
        aria-label={
          valid ? `Jump to citation ${id}: ${citation.source}` : `Unknown citation ${id}`
        }
        className="inline-flex h-[18px] min-w-[18px] items-center justify-center rounded font-mono text-[11px] transition-opacity hover:opacity-90"
        style={{
          background: valid
            ? "color-mix(in oklch, var(--color-accent), transparent 86%)"
            : "color-mix(in oklch, var(--color-danger), transparent 86%)",
          color: valid ? "var(--color-accent)" : "var(--color-danger)",
          paddingInline: "4px",
        }}
      >
        {id}
      </button>
      {hover && citation && <CitationPopover citation={citation} />}
    </span>
  );
}

function CitationPopover({ citation }: { citation: Citation }) {
  return (
    <span
      role="tooltip"
      className="bg-elev pointer-events-none absolute bottom-full left-1/2 z-30 mb-2 w-[280px] -translate-x-1/2 rounded-md border p-3 text-left shadow-lg"
      style={{ animation: "fadeIn 80ms ease" }}
    >
      <span className="text-subtle font-mono text-[10px] uppercase tracking-widest">
        [{citation.id}] · score {citation.score?.toFixed(3) ?? "n/a"}
      </span>
      <span className="mt-1 block truncate font-mono text-[12px]">{citation.source}</span>
      {citation.content && (
        <span className="text-muted mt-2 block text-[12.5px] leading-snug">
          {citation.content}
        </span>
      )}
    </span>
  );
}

function ClaimTooltip({
  claim,
  Icon,
  tone,
}: {
  claim: ClaimGrounding;
  Icon: typeof CheckCircle2;
  tone: string;
}) {
  return (
    <span
      role="tooltip"
      className="bg-elev pointer-events-none absolute bottom-full left-0 z-20 mb-2 w-[260px] rounded-md border p-3 text-left shadow-lg"
    >
      <span className="flex items-center gap-1.5 text-[12px] font-medium" style={{ color: tone }}>
        <Icon size={12} strokeWidth={2.5} />
        {claim.is_grounded ? "Grounded claim" : "Not supported by cited chunks"}
      </span>
      <span className="text-muted mt-2 block font-mono text-[11px]">
        support {Math.round(claim.support_score * 100)}% · cites{" "}
        {claim.cited_indices.length === 0
          ? "none"
          : claim.cited_indices.map((i) => `[${i}]`).join("")}
        {claim.cited_indices.length > 0 &&
          claim.valid_indices.length !== claim.cited_indices.length && (
            <span style={{ color: "var(--color-danger)" }}>
              {" "}
              · {claim.cited_indices.length - claim.valid_indices.length} out of range
            </span>
          )}
      </span>
    </span>
  );
}
