"""Marker-based PDF ingestion loader."""

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from app.config.settings import settings
from app.ingestion.base_loader import BaseDocumentLoader
from app.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger()


class MarkerPDFLoader(BaseDocumentLoader):
    """Load PDF files using Marker for structured extraction."""

    def __init__(self) -> None:
        """Initialize Marker converter."""
        self.converter = PdfConverter(artifact_dict=create_model_dict())

        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=settings.pdf_headers
        )

    def load(self, file_path: str) -> list[Document]:
        """Load and split a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted and header-split documents. Returns an empty list when
            extraction fails or produces empty text.
        """
        logger.info(f"[PDF-Marker] Processing: {file_path}")

        try:
            text, _, _ = text_from_rendered(self.converter(file_path))

            if not text.strip():
                logger.warning(f"[PDF-Marker] Empty content: {file_path}")
                return []

            docs = self.splitter.split_text(text)

            for doc in docs:
                doc.metadata["source"] = file_path

            logger.info(f"[PDF-Marker] Chunks created: {len(docs)}")

            return docs

        except Exception:
            logger.exception(f"[PDF-Marker] Failed: {file_path}")
            return []
