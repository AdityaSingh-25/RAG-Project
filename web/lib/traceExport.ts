"use client";

import type { QueryResponse } from "./types";

/** Self-contained record of one query run. Designed for paste-into-issue and
 *  offline diff: include the question, the toggles that were on, an
 *  ISO-format export timestamp, and the full response (which already carries
 *  the pipeline trace, citations, grounding, and trace_id). */
export interface RunArtifact {
  exported_at: string;
  question: string;
  bypass_cache: boolean;
  response: QueryResponse;
}

export function buildRunArtifact(
  question: string,
  bypassCache: boolean,
  response: QueryResponse,
): RunArtifact {
  return {
    exported_at: new Date().toISOString(),
    question,
    bypass_cache: bypassCache,
    response,
  };
}

/** Returns true on success. Some embedded browsers / iframes / non-HTTPS
 *  pages reject `navigator.clipboard`; in that case we report failure so
 *  the UI can fall back to the download path. */
export async function copyJsonToClipboard(data: unknown): Promise<boolean> {
  const text = JSON.stringify(data, null, 2);
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through
  }
  return false;
}

export function downloadJson(filename: string, data: unknown): void {
  if (typeof window === "undefined") return;
  const text = JSON.stringify(data, null, 2);
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  // Some browsers require the anchor to be in the DOM before .click() fires.
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Free the blob after a tick so Safari's download stream has actually
  // started — revoking too eagerly can cancel the download.
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Sanitised filename built from the trace id. Falls back to "trace" when
 *  the response was streamed without a trace id yet. */
export function defaultExportFilename(response: QueryResponse): string {
  const traceFragment = response.trace_id
    ? response.trace_id.slice(0, 10)
    : "trace";
  return `rag-${traceFragment}.json`;
}
