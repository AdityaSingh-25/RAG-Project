// Client-side fetch helpers. The Next.js route handlers under app/api proxy
// to FastAPI so the browser never speaks to the backend directly — this keeps
// the API URL out of the bundle and makes CORS a non-issue.

import type { MetricsSnapshot, QueryRequest, QueryResponse } from "./types";

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
