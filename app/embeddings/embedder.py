"""Embedding model factory utilities."""

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from app.config.settings import settings
from app.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger()


class EmbeddingModelFactory:
    """Factory class for creating embedding models."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the embedding factory.

        Args:
            model_name: Optional model name override.
        """
        self.model_name = model_name or settings.embedding_model

    def get_model(self) -> HuggingFaceEmbeddings:
        """Load and return the embedding model.

        Returns:
            Initialized HuggingFace embedding model.
        """
        logger.info(f"[Embedding] Loading model: {self.model_name}")

        try:
            model = HuggingFaceEmbeddings(model_name=self.model_name)

            logger.info("[Embedding] Model loaded successfully")
            return model

        except Exception:
            logger.exception("[Embedding] Failed to load model")
            raise
