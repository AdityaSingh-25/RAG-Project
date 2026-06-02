// Mirror of the FastAPI /query response shape.
// Keep in sync with src/rag_engine/api/main.py.

export type QueryStatus = "ok" | "insufficient_evidence";

export interface Citation {
  id: number;
  source: string;
  page: number | null;
  score: number | null;
  /** Truncated chunk content for UI hover previews. ~240 chars max. */
  content?: string;
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

export type ClaimVerifierMode = "overlap" | "nli";

export interface QueryRequest {
  question: string;
  bypass_cache?: boolean;
  /** Override the deployment-level claim verifier per request.
   *  Omit (or send `null`) to use the API default. */
  claim_verifier_mode?: ClaimVerifierMode | null;
  /** Override the deployment-level structured-answers flag per request. */
  structured_answers?: boolean | null;
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

export interface IngestRequest {
  source_path: string;
}

export interface IngestResponse {
  ingested_chunks: number;
  duplicates_removed: number;
  /** Only present on the multipart /ingest/upload endpoint. */
  files_received?: number;
}

export interface CorpusStats {
  collection: string;
  chunks: number;
  sources: number;
}

export interface CorpusSource {
  source: string;
  chunks: number;
}

export interface CorpusSourcesResponse {
  sources: CorpusSource[];
}

export interface CorpusChunk {
  chunk_id: number | null;
  content_hash: string | null;
  page: number | null;
  content: string;
}

export interface CorpusSourceDetail {
  source: string;
  chunks: CorpusChunk[];
  total: number;
  truncated: boolean;
}

export type EvalDataset = "seed" | "adversarial";

export interface EvalRequest {
  dataset: EvalDataset;
  limit?: number | null;
}

export interface EvalCaseResult {
  case_id: string;
  question: string;
  answer: string;
  grounding_score: number;
  citation_hit_rate: number;
  term_recall: number;
  iteration: number;
  warnings: string[];
  latency_ms: number;
  grounded_claim_rate: number;
  status: string;
  expected_status: string;
  status_matches_expected: boolean;
  citations: Array<Record<string, unknown>>;
}

export interface EvalAggregate {
  n: number;
  mean_grounding: number;
  mean_citation_hit_rate: number;
  mean_term_recall: number;
  mean_latency_ms: number;
  mean_iteration: number;
  insufficient_evidence_rate: number;
  mean_grounded_claim_rate: number;
  status_match_rate: number;
}

export interface EvalReport {
  dataset: EvalDataset;
  limit: number | null;
  total_cases_available: number;
  aggregate: EvalAggregate;
  results: EvalCaseResult[];
}
