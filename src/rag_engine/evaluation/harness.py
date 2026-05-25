"""Evaluation harness for end-to-end RAG quality.

Given a JSONL of cases and a callable that runs the graph for a question,
this produces grounding, citation-hit-rate, term-recall, and latency
metrics per case plus aggregate summaries.

The harness deliberately keeps the runner injectable so it can be tested
without Qdrant or Ollama, and so callers can swap between in-process
graph invocation and HTTP calls to a deployed API.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

EvalRunner = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    must_cite: tuple[str, ...] = ()
    expected_terms: tuple[str, ...] = ()


@dataclass
class EvalResult:
    case_id: str
    question: str
    answer: str
    grounding_score: float
    citation_hit_rate: float
    term_recall: float
    iteration: int
    warnings: list[str]
    latency_ms: float
    grounded_claim_rate: float = 1.0
    status: str = "ok"
    citations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalReport:
    results: list[EvalResult]
    aggregate: dict[str, float]


def load_cases(path: Path) -> list[EvalCase]:
    """Load EvalCases from a JSONL file."""
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            cases.append(
                EvalCase(
                    id=str(obj["id"]),
                    question=str(obj["question"]),
                    must_cite=tuple(obj.get("must_cite", [])),
                    expected_terms=tuple(obj.get("expected_terms", [])),
                )
            )
    return cases


def citation_hit_rate(citations: list[dict[str, Any]], must_cite: Iterable[str]) -> float:
    """Fraction of expected sources that appear in the answer's citations."""
    expected = [s for s in must_cite if s]
    if not expected:
        return 1.0
    cited_sources = {str(c.get("source", "")) for c in citations}
    hits = sum(1 for source in expected if any(source in cited for cited in cited_sources))
    return hits / len(expected)


def term_recall(answer: str, expected_terms: Iterable[str]) -> float:
    """Fraction of expected terms that appear in the answer (case-insensitive)."""
    expected = [t for t in expected_terms if t]
    if not expected:
        return 1.0
    lowered = answer.lower()
    hits = sum(1 for term in expected if re.search(rf"\b{re.escape(term.lower())}\b", lowered))
    return hits / len(expected)


def evaluate_case(case: EvalCase, runner: EvalRunner) -> EvalResult:
    """Run a single case through the supplied runner and score the result."""
    start = time.perf_counter()
    output = runner(case.question)
    latency_ms = (time.perf_counter() - start) * 1000

    answer = str(output.get("answer", ""))
    citations = list(output.get("citations", []))
    return EvalResult(
        case_id=case.id,
        question=case.question,
        answer=answer,
        grounding_score=float(output.get("grounding_score", 0.0)),
        citation_hit_rate=citation_hit_rate(citations, case.must_cite),
        term_recall=term_recall(answer, case.expected_terms),
        iteration=int(output.get("iteration", 0)),
        warnings=list(output.get("warnings", [])),
        latency_ms=round(latency_ms, 2),
        grounded_claim_rate=float(output.get("grounded_claim_rate", 1.0)),
        status=str(output.get("status") or "ok"),
        citations=citations,
    )


def evaluate_dataset(cases: list[EvalCase], runner: EvalRunner) -> EvalReport:
    """Run every case and compute aggregate metrics."""
    results = [evaluate_case(case, runner) for case in cases]
    if not results:
        aggregate = {
            "n": 0.0,
            "mean_grounding": 0.0,
            "mean_citation_hit_rate": 0.0,
            "mean_term_recall": 0.0,
            "mean_latency_ms": 0.0,
            "mean_iteration": 0.0,
            "insufficient_evidence_rate": 0.0,
            "mean_grounded_claim_rate": 0.0,
        }
    else:
        n = len(results)
        insufficient = sum(1 for r in results if r.status == "insufficient_evidence")
        aggregate = {
            "n": float(n),
            "mean_grounding": round(sum(r.grounding_score for r in results) / n, 3),
            "mean_citation_hit_rate": round(sum(r.citation_hit_rate for r in results) / n, 3),
            "mean_term_recall": round(sum(r.term_recall for r in results) / n, 3),
            "mean_latency_ms": round(sum(r.latency_ms for r in results) / n, 2),
            "mean_iteration": round(sum(r.iteration for r in results) / n, 2),
            "insufficient_evidence_rate": round(insufficient / n, 3),
            "mean_grounded_claim_rate": round(
                sum(r.grounded_claim_rate for r in results) / n, 3
            ),
        }
    return EvalReport(results=results, aggregate=aggregate)


def format_json(report: EvalReport) -> str:
    return json.dumps(
        {
            "aggregate": report.aggregate,
            "results": [asdict(r) for r in report.results],
        },
        indent=2,
    )


def format_markdown(report: EvalReport) -> str:
    agg = report.aggregate
    lines = [
        "# RAG Evaluation Report",
        "",
        "## Aggregate",
        "",
        f"- cases: {int(agg['n'])}",
        f"- mean grounding: {agg['mean_grounding']:.3f}",
        f"- mean citation hit rate: {agg['mean_citation_hit_rate']:.3f}",
        f"- mean term recall: {agg['mean_term_recall']:.3f}",
        f"- mean iterations: {agg['mean_iteration']:.2f}",
        f"- mean latency: {agg['mean_latency_ms']:.1f} ms",
        f"- insufficient-evidence rate: {agg['insufficient_evidence_rate']:.3f}",
        f"- mean grounded-claim rate: {agg['mean_grounded_claim_rate']:.3f}",
        "",
        "## Per Case",
        "",
        "| id | status | grounding | claims | cite hit | term recall | iters | latency (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.results:
        lines.append(
            f"| {r.case_id} | {r.status} | {r.grounding_score:.3f} | "
            f"{r.grounded_claim_rate:.3f} | {r.citation_hit_rate:.3f} | "
            f"{r.term_recall:.3f} | {r.iteration} | {r.latency_ms:.1f} |"
        )
    return "\n".join(lines) + "\n"
