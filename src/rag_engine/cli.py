import argparse
from pathlib import Path

from rag_engine.config.settings import get_settings
from rag_engine.ingestion.pipeline import ingest_path


def ingest_command() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector database.")
    parser.add_argument("--source", default="data/raw", help="File or directory to ingest.")
    args = parser.parse_args()

    settings = get_settings()
    count = ingest_path(Path(args.source), settings)
    print(f"Ingested {count} chunks from {args.source}")

