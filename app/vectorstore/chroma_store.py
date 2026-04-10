"""Chroma vector store creation and loading helpers."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config.settings import settings


def create_vectorstore(
    documents: list[Document],
    embedding_model: Embeddings,
) -> Chroma:
    """Create and persist a Chroma vector store from documents.

    Args:
        documents: Chunked documents to index.
        embedding_model: Embedding model used for vector generation.

    Returns:
        Initialized Chroma vector store.
    """
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=settings.chroma_path,
    )

    return db


def load_vectorstore(embedding_model: Embeddings) -> Chroma:
    """Load an existing Chroma vector store.

    Args:
        embedding_model: Embedding function used at query time.

    Returns:
        Loaded Chroma vector store.
    """
    return Chroma(
        persist_directory=settings.chroma_path,
        embedding_function=embedding_model,
    )
