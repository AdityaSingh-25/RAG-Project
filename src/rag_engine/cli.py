import argparse
from pathlib import Path
from typing import Any

from rag_engine.config.settings import get_settings
from rag_engine.evaluation.harness import (
    evaluate_dataset,
    format_json,
    format_markdown,
    load_cases,
)
from rag_engine.ingestion.pipeline import ingest_path


def ingest_command() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector database.")
    parser.add_argument("--source", default="data/raw", help="File or directory to ingest.")
    args = parser.parse_args()

    settings = get_settings()
    report = ingest_path(Path(args.source), settings)
    print(
        f"Ingested {report.indexed} chunks from {args.source} "
        f"(deduplicated {report.duplicates_removed})"
    )


def eval_command() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG graph against a JSONL of cases.",
    )
    parser.add_argument(
        "--cases",
        default="data/eval/seed_cases.jsonl",
        help="Path to a JSONL file of eval cases.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the report. Defaults to stdout.",
    )
    args = parser.parse_args()

    settings = get_settings()
    cases = load_cases(Path(args.cases))

    from rag_engine.agents.graph import build_graph

    graph = build_graph(settings)

    def runner(question: str) -> dict[str, Any]:
        return graph.invoke(
            {
                "question": question,
                "original_question": question,
                "filters": {},
                "documents": [],
                "answer": "",
                "citations": [],
                "grounding_score": 0.0,
                "warnings": [],
                "iteration": 0,
            }
        )

    report = evaluate_dataset(cases, runner)
    rendered = format_json(report) if args.format == "json" else format_markdown(report)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)

