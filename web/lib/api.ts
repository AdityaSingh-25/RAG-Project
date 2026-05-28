// Client-side fetch helpers. The Next.js route handlers under app/api proxy
// to FastAPI so the browser never speaks to the backend directly — this keeps
// the API URL out of the bundle and makes CORS a non-issue.

import { consumeSse } from "./sse";
import type {
  Citation,
  ClaimGrounding,
  IngestRequest,
  IngestResponse,
  MetricsSnapshot,
  PipelineTraceEntry,
  QueryRequest,
  QueryResponse,
} from "./types";

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.text();
    throw new ApiError(detail || res.statusText, res.status);
  }
  return res.json();
}

export async function runQuery(req: QueryRequest, signal?: AbortSignal): Promise<QueryResponse> {
  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  return jsonOrThrow<QueryResponse>(res);
}

export async function getMetrics(signal?: AbortSignal): Promise<MetricsSnapshot> {
  const res = await fetch("/api/metrics", { signal, cache: "no-store" });
  return jsonOrThrow<MetricsSnapshot>(res);
}

export interface GroundingEvent {
  grounding_score: number;
  grounded_claim_rate: number;
  claim_grounding: ClaimGrounding[];
  warnings: string[];
}

export interface QueryStreamHandlers {
  onTrace?: (entry: PipelineTraceEntry) => void;
  onToken?: (delta: string) => void;
  onCitations?: (citations: Citation[]) => void;
  onGrounding?: (g: GroundingEvent) => void;
  onDone: (payload: QueryResponse) => void;
  onError?: (detail: string) => void;
}

export async function runQueryStream(
  req: QueryRequest,
  handlers: QueryStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new ApiError(detail || `Stream failed (${res.status})`, res.status);
  }
  await consumeSse(
    res,
    {
      trace: (d) => handlers.onTrace?.(d as PipelineTraceEntry),
      token: (d) => handlers.onToken?.((d as { delta: string }).delta),
      citations: (d) =>
        handlers.onCitations?.((d as { citations: Citation[] }).citations),
      grounding: (d) => handlers.onGrounding?.(d as GroundingEvent),
      done: (d) => handlers.onDone(d as QueryResponse),
      error: (d) => handlers.onError?.((d as { detail: string }).detail),
    },
    signal,
  );
}

export async function runIngest(
  req: IngestRequest,
  signal?: AbortSignal,
): Promise<IngestResponse> {
  const res = await fetch("/api/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  return jsonOrThrow<IngestResponse>(res);
}

/** Upload files directly. The browser sets the multipart boundary header. */
export async function runIngestUpload(
  files: File[],
  signal?: AbortSignal,
): Promise<IngestResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file, file.name);
  }
  const res = await fetch("/api/ingest/upload", {
    method: "POST",
    body: form,
    signal,
  });
  return jsonOrThrow<IngestResponse>(res);
}
