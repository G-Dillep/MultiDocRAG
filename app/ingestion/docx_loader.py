"""DOCX ingestion loader implementation."""

from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document

from app.ingestion.base_loader import BaseDocumentLoader
from app.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger()


class DocxLoader(BaseDocumentLoader):
    """Load DOCX files with the Unstructured word loader."""

    def load(self, file_path: str) -> list[Document]:
        """Load a DOCX document.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            Extracted documents. Returns an empty list on failure.
        """
        logger.info(f"[DOCX] Processing: {file_path}")

        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file_path

            logger.info(f"[DOCX] Documents created: {len(docs)}")

            return docs

        except Exception:
            logger.exception(f"[DOCX] Failed: {file_path}")
            return []
