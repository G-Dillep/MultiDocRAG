"""Application entry point for the RAG pipeline."""

import json

from app.config.settings import settings
from app.embeddings.embedder import EmbeddingModelFactory
from app.ingestion.loader_router import DocumentLoaderRouter
from app.orchestration.langgraph_pipeline import LangGraphRAGPipeline
from app.processing.chunker import DocumentChunker
from app.processing.normalizer import DocumentNormalizer
from app.utils.logger import LoggerFactory
from app.vectorstore.chroma_store import create_vectorstore

logger = LoggerFactory.get_logger()


def main() -> None:
    """Run the end-to-end RAG pipeline.

    The workflow includes ingestion, normalization, chunking, embedding,
    vector store persistence, and retrieval.
    """
    try:
        settings.validate_for_langgraph()
        logger.info("[Main] Pipeline started")

        # Ingestion
        router = DocumentLoaderRouter()
        docs = router.load_documents(settings.data_path)

        # Normalization
        normalizer = DocumentNormalizer()
        docs = normalizer.normalize(docs)

        # Chunking
        chunker = DocumentChunker()
        chunks = chunker.chunk(docs)
        # Embeddings + DB

        embedding_model = EmbeddingModelFactory().get_model()
        db = create_vectorstore(chunks, embedding_model)

        # LangGraph Retrieval + Generation
        pipeline = LangGraphRAGPipeline()
        response = pipeline.invoke(db=db, query=settings.default_query)

        logger.info("[Main] Final response payload")
        logger.info(json.dumps(response, ensure_ascii=True, indent=2))

    except Exception:
        logger.exception("[Main] Pipeline failed")


if __name__ == "__main__":
    main()
