"""Excel ingestion loader implementation."""

from langchain_community.document_loaders import (
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document

from app.ingestion.base_loader import BaseDocumentLoader
from app.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger()


class ExcelLoader(BaseDocumentLoader):
    """Load Excel files with the Unstructured Excel loader."""

    def load(self, file_path: str) -> list[Document]:
        """Load an Excel file.

        Args:
            file_path: Path to the Excel file.

        Returns:
            Extracted documents. Returns an empty list on failure.
        """
        logger.info(f"[Excel] Processing: {file_path}")

        try:
            loader = UnstructuredExcelLoader(file_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file_path

            logger.info(f"[Excel] Documents created: {len(docs)}")

            return docs

        except Exception:
            logger.exception(f"[Excel] Failed: {file_path}")
            return []
