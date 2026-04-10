"""Document retrieval pipeline with optional reranking."""

from typing import Protocol

from langchain_core.documents import Document

from app.config.settings import settings
from app.retrieval.reranker import BAAIReranker
from app.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger()


class VectorStoreLike(Protocol):
    """Structural type for vector stores used by the retriever."""

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        """Search by query and return documents with similarity scores."""


class DocumentRetriever:
    """Retrieve relevant documents from a vector store.

    This retriever performs similarity search, deduplication, and optional
    cross-encoder reranking, then returns the top-k ranked documents.
    """

    def __init__(self, top_k: int | None = None) -> None:
        """Initialize the retriever.

        Args:
            top_k: Number of results to return. When ``None``, defaults to the
                configured ``settings.top_k`` value.
        """
        self.top_k = top_k or settings.top_k
        self.fetch_k = settings.reranker_fetch_k

        # Initialize reranker only if enabled
        self.reranker = BAAIReranker() if settings.reranker_enabled else None

    def retrieve(self, db: VectorStoreLike, query: str) -> list[Document]:
        """Retrieve top-k relevant documents.

        Processing stages:
            1. Vector search using ``fetch_k``.
            2. Deduplicate by normalized page content.
            3. Optionally rerank with cross-encoder.
            4. Trim to ``top_k``.

        Args:
            db: Vector store that supports similarity search with scores.
            query: Input query text.

        Returns:
            Final ranked documents. Returns an empty list on failure.
        """
        logger.info(f"[Retriever] Query: {query}")

        try:
            # -------------------------
            # Step 1: Vector Search
            # -------------------------
            results: list[tuple[Document, float]] = db.similarity_search_with_score(
                query,
                k=self.fetch_k,
            )
            retrieved_count = len(results)

            logger.info(f"[Retriever] Raw results: {retrieved_count}")

            # -------------------------
            # Step 2: Deduplication
            # -------------------------
            seen = set()
            docs: list[Document] = []

            for doc, score in results:
                key = doc.page_content.strip()

                if key not in seen:
                    seen.add(key)

                    # attach similarity score
                    doc.metadata["score"] = float(score)
                    docs.append(doc)

            deduplicated_count = len(docs)

            logger.info(f"[Retriever] After dedup: {deduplicated_count}")

            # -------------------------
            # Step 3: Reranking
            # -------------------------
            rerank_input_count = 0
            rerank_output_count = 0

            if self.reranker and docs:
                rerank_input_count = len(docs)
                docs = self.reranker.rerank(query, docs)
                rerank_output_count = len(docs)
                logger.info(
                    "[Retriever] Reranking applied "
                    f"(input={rerank_input_count}, output={rerank_output_count})"
                )
            else:
                docs = docs[: self.top_k]
                logger.info("[Retriever] Reranking skipped")

            # -------------------------
            # Step 4: Final Trim
            # -------------------------
            final_docs = docs[: self.top_k]

            logger.info(f"[Retriever] Final returned: {len(final_docs)}")
            logger.info(
                "[Retriever][Metrics] "
                f"query={query!r}, "
                f"retrieved={retrieved_count}, "
                f"deduplicated={deduplicated_count}, "
                f"rerank_input={rerank_input_count}, "
                f"rerank_output={rerank_output_count}, "
                f"returned={len(final_docs)}"
            )

            return final_docs

        except Exception:
            logger.exception("[Retriever] Failed")
            return []
