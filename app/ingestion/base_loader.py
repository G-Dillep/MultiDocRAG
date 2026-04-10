"""Base contracts for ingestion loaders."""

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseDocumentLoader(ABC):
    """Abstract contract for all document loaders.

    Implementations must return a list of LangChain ``Document`` objects and
    avoid raising recoverable parsing errors to the caller.
    """

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        """Load a file and return extracted documents.

        Args:
            file_path: Absolute or relative path to the source file.

        Returns:
            A list of extracted documents. Returns an empty list when parsing
            is unsuccessful.
        """
