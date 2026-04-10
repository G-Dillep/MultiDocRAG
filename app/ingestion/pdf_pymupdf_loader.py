"""PyMuPDF-based PDF ingestion loader."""

from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

from app.config.settings import settings
from app.ingestion.base_loader import BaseDocumentLoader
from app.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger()


class PyMuPDFLoader(BaseDocumentLoader):
    """PDF loader using PyMuPDF4LLMLoader.

    Features:
    - Fast PDF parsing (no heavy model download)
    - Page-level extraction (each page = one Document)
    - Metadata preservation
    """

    def __init__(self) -> None:
        """Initialize loader configuration."""
        self.mode = getattr(settings, "PDF_MODE", "page")

    def load(self, file_path: str) -> list[Document]:
        """Load PDF and return documents.

        Args:
            file_path (str): Path to PDF file

        Returns:
            List[Document]: Extracted documents
        """
        logger.info(f"[PDF-PyMuPDF] Processing: {file_path}")

        try:
            loader = PyMuPDF4LLMLoader(
                file_path,
                mode=self.mode,  # "page" or "single"
            )

            docs: list[Document] = loader.load()

            # Ensure metadata consistency
            for doc in docs:
                doc.metadata["source"] = file_path

            logger.info(f"[PDF-PyMuPDF] Documents created: {len(docs)}")

            return docs

        except Exception:
            logger.exception(f"[PDF-PyMuPDF] Failed: {file_path}")
            return []
