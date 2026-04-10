"""Cross-encoder reranking component for retrieval results."""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config.settings import settings
from app.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger()


class BAAIReranker:
    """Cross-encoder reranker using a configurable BAAI model."""

    def __init__(self) -> None:
        """Load and initialize the cross-encoder reranker model."""
        logger.info(f"[Reranker] Loading model: {settings.reranker_model}")
        self.model = CrossEncoder(settings.reranker_model)

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """Rerank documents by query relevance.

        Args:
            query: User query text.
            docs: Candidate documents from vector retrieval.

        Returns:
            Top documents sorted by reranker score in descending order.
            Returns an empty list if ``docs`` is empty.
        """
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]

        scores = self.model.predict(pairs)

        # attach scores
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = float(score)

        # sort by score (descending)
        docs_sorted = sorted(
            docs,
            key=lambda x: x.metadata["rerank_score"],
            reverse=True,
        )

        return docs_sorted[: settings.reranker_top_k]
