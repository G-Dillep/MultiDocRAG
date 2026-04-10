"""Loader routing for supported document file types."""

import os

from langchain_core.documents import Document

from app.config.settings import settings
from app.ingestion.base_loader import BaseDocumentLoader
from app.ingestion.csv_loader import CSVFileLoader
from app.ingestion.docx_loader import DocxLoader
from app.ingestion.excel_loader import ExcelLoader
from app.ingestion.pdf_loader import MarkerPDFLoader
from app.ingestion.pdf_pymupdf_loader import PyMuPDFLoader
from app.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger()


class DocumentLoaderRouter:
    """Route files to appropriate loaders based on extension."""

    def __init__(self) -> None:
        """Initialize loader router with configurable PDF backend."""
        pdf_loader = self._get_pdf_loader()

        self.supported_loaders: dict[str, BaseDocumentLoader] = {
            ".pdf": pdf_loader,
            ".docx": DocxLoader(),
            ".csv": CSVFileLoader(),
            ".xlsx": ExcelLoader(),
        }

    def _get_pdf_loader(self) -> BaseDocumentLoader:
        """Select the PDF loader based on runtime settings.

        Returns:
            Configured PDF loader implementation.

        Raises:
            ValueError: If an unsupported PDF backend is configured.
        """
        if settings.pdf_backend == "marker":
            logger.info("[Router] Using Marker PDF loader")
            return MarkerPDFLoader()

        elif settings.pdf_backend == "pymupdf":
            logger.info("[Router] Using PyMuPDF PDF loader")
            return PyMuPDFLoader()

        else:
            raise ValueError(f"Invalid PDF_BACKEND: {settings.pdf_backend}")

    def load_documents(self, folder_path: str) -> list[Document]:
        """Load all supported documents from a folder.

        Args:
            folder_path: Folder path that contains source files.

        Returns:
            Flattened list of loaded documents.
        """
        documents: list[Document] = []

        logger.info(f"[Router] Scanning folder: {folder_path}")

        if not os.path.exists(folder_path):
            logger.error(f"[Router] Folder does not exist: {folder_path}")
            return []

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if not os.path.isfile(file_path):
                continue

            extension = os.path.splitext(file_name)[1].lower()
            loader = self.supported_loaders.get(extension)

            if loader:
                logger.info(f"[Router] Processing file: {file_name}")

                try:
                    docs = loader.load(file_path)
                    documents.extend(docs)

                except Exception:
                    logger.exception(f"[Router] Failed loading {file_name}")
            else:
                logger.warning(f"[Router] Unsupported file skipped: {file_name}")

        logger.info(f"[Router] Total documents loaded: {len(documents)}")

        return documents
