"""Text normalization utilities for document processing."""

import re

from langchain_core.documents import Document

from app.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger()


class DocumentNormalizer:
    """Normalize document text content for consistent downstream processing."""

    def __init__(self) -> None:
        """Initialize normalizer."""
        self.whitespace_pattern = re.compile(r"\s+")
        self.non_printable_pattern = re.compile(r"[^\x20-\x7E]")

    def normalize(self, documents: list[Document]) -> list[Document]:
        """Normalize a list of documents.

        Args:
            documents: Input documents.

        Returns:
            Normalized documents.
        """
        normalized_docs: list[Document] = []

        logger.info(f"[Normalizer] Processing {len(documents)} documents")

        for doc in documents:
            try:
                cleaned_text = self._clean_text(doc.page_content)

                # Create new Document (avoid mutating original)
                normalized_doc = Document(
                    page_content=cleaned_text,
                    metadata=doc.metadata.copy(),
                )

                normalized_docs.append(normalized_doc)

            except Exception:
                logger.exception(f"[Normalizer] Failed for doc: {doc.metadata}")

        logger.info(f"[Normalizer] Completed. Output documents: {len(normalized_docs)}")

        return normalized_docs

    def _clean_text(self, text: str) -> str:
        """Apply normalization rules to text.

        Args:
            text: Raw text.

        Returns:
            Cleaned text.
        """
        text = self.whitespace_pattern.sub(" ", text)
        text = self.non_printable_pattern.sub(" ", text)
        return text.strip()
