"""Chunking strategies for ingested documents."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import settings
from app.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger()


class DocumentChunker:
    """Apply hybrid chunking rules based on source file type."""

    def __init__(self) -> None:
        """Initialize text splitter."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Chunk documents based on source type.

        Args:
            documents: Input documents from ingestion and normalization.

        Returns:
            Chunked documents ready for embedding.
        """
        final_chunks: list[Document] = []

        logger.info(f"[Chunker] Processing {len(documents)} documents")

        for doc in documents:
            source = doc.metadata.get("source", "unknown").lower()
            chunks: list[Document] = []

            try:
                before = len(final_chunks)

                # -------------------------
                # PDF (IMPORTANT FIX)
                # -------------------------
                if source.endswith(".pdf"):
                    chunks = self.splitter.split_documents([doc])

                    for c in chunks:
                        c.metadata.update(doc.metadata)

                    final_chunks.extend(chunks)

                # -------------------------
                # CSV / Excel (skip)
                # -------------------------
                elif source.endswith(".csv") or source.endswith(".xlsx"):
                    final_chunks.append(doc)

                # -------------------------
                # Others (DOCX etc.)
                # -------------------------
                else:
                    chunks = self.splitter.split_documents([doc])
                    final_chunks.extend(chunks)

                after = len(final_chunks)

                logger.info(
                    f"[Chunker] File: {source} | Chunks added: {after - before}"
                )

                # Optional deep debug (first 2 chunks only)
                if source.endswith(".pdf") and chunks:
                    for i, c in enumerate(chunks[:2]):
                        logger.debug(
                            f"[Chunk Sample] {source} | chunk {i} | "
                            f"{len(c.page_content)} chars"
                        )

            except Exception:
                logger.exception(f"[Chunker] Failed for {source}")

        logger.info(f"[Chunker] Total chunks created: {len(final_chunks)}")

        return final_chunks
