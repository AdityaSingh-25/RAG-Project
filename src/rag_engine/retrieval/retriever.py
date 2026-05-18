from langchain_core.vectorstores import VectorStoreRetriever

from rag_engine.config.settings import Settings
from rag_engine.vectorstore.qdrant_store import build_vectorstore


def build_retriever(settings: Settings) -> VectorStoreRetriever:
    vectorstore = build_vectorstore(
        qdrant_url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
    )
    return vectorstore.as_retriever(search_kwargs={"k": settings.top_k})

