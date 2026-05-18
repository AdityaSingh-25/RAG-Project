from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document


def load_documents(source: Path) -> list[Document]:
    files = [source] if source.is_file() else [path for path in source.rglob("*") if path.is_file()]
    documents: list[Document] = []
    for file_path in files:
        if file_path.name.startswith("."):
            continue
        documents.extend(_load_file(file_path))
    return documents


def _load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix == ".csv":
        return CSVLoader(str(path)).load()
    if suffix == ".json":
        return JSONLoader(str(path), jq_schema=".", text_content=False).load()
    if suffix in {".txt", ".md", ".rst"}:
        return TextLoader(str(path), encoding="utf-8").load()
    return []

