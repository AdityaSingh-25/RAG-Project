// Mirror of the FastAPI /query response shape.
// Keep in sync with src/rag_engine/api/main.py.

export type QueryStatus = "ok" | "insufficient_evidence";

export interface Citation {
  id: number;
  source: string;
  page: number | null;
  score: number | null;
}

export interface ClaimGrounding {
  sentence: string;
  cited_indices: number[];
  valid_indices: number[];
  support_score: number;
  is_grounded: boolean;
}

export interface PipelineTraceEntry {
  node: "retrieve" | "answer" | "critique" | "rewrite" | "fallback" | "finalize";
  duration_ms: number;
  iteration: number;
  [key: string]: unknown;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  grounding_score: number;
  warnings: string[];
  iteration: number;
  status: QueryStatus;
  grounded_claim_rate: number;
  claim_grounding: ClaimGrounding[];
  pipeline_trace: PipelineTraceEntry[];
  total_duration_ms: number;
  trace_id: string;
  cached: boolean;
}

export interface QueryRequest {
  question: string;
  bypass_cache?: boolean;
}

export interface MetricsSnapshot {
  totals: Record<string, number>;
  samples: Record<
    string,
    {
      count: number;
      mean: number;
      p50: number;
      p95: number;
      max: number;
    }
  >;
}
