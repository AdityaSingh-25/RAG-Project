from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from rag_engine.config.settings import Settings
from rag_engine.embeddings.factory import build_embeddings


@lru_cache(maxsize=4)
def build_vectorstore(qdrant_url: str, collection_name: str, embedding_model: str) -> QdrantVectorStore:
    embeddings = build_embeddings(embedding_model)
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=qdrant_url,
    )


def index_documents(documents: list[Document], settings: Settings) -> QdrantVectorStore:
    embeddings = build_embeddings(settings.embedding_model)
    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
    )
