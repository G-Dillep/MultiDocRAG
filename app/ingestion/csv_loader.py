"""CSV ingestion loader implementation."""

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

from app.ingestion.base_loader import BaseDocumentLoader
from app.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger()


class CSVFileLoader(BaseDocumentLoader):
    """Load CSV files using LangChain's CSVLoader."""

    def load(self, file_path: str) -> list[Document]:
        """Load a CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Extracted documents. Returns an empty list on failure.
        """
        logger.info(f"[CSV] Processing: {file_path}")

        try:
            loader = CSVLoader(file_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file_path

            logger.info(f"[CSV] Rows loaded: {len(docs)}")

            return docs

        except Exception:
            logger.exception(f"[CSV] Failed: {file_path}")
            return []
