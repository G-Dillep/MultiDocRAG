"""Application settings for the RAG pipeline."""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Centralized configuration for the RAG pipeline."""

    def __init__(self) -> None:
        """Initialize default settings values."""
        self.data_path: str = os.getenv("DATA_PATH", "data")
        self.chroma_path: str = os.getenv("CHROMA_PATH", "chroma_db")

        self.chunk_size: int = self._get_env_int("CHUNK_SIZE", 600)
        self.chunk_overlap: int = self._get_env_int("CHUNK_OVERLAP", 100)

        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        self.top_k: int = self._get_env_int("TOP_K", 3)

        self.marker_enabled: bool = self._get_env_bool("MARKER_ENABLED", True)
        self.pdf_headers: list[tuple[str, str]] = [
            ("#", "document"),
            ("##", "statement"),
            ("###", "section"),
            ("####", "subsection"),
        ]

        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_file_level: str = os.getenv("LOG_FILE_LEVEL", "DEBUG")
        self.log_file_path: str = os.getenv("LOG_FILE_PATH", "logs/pipeline.log")
        self.log_rotation: str = os.getenv("LOG_ROTATION", "5 MB")

        self.pdf_mode: str = os.getenv("PDF_MODE", "page")
        self.pdf_backend: str = os.getenv("PDF_BACKEND", "pymupdf")

        self.reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.reranker_enabled: bool = self._get_env_bool("RERANKER_ENABLED", True)
        self.reranker_top_k: int = self._get_env_int("RERANKER_TOP_K", self.top_k)
        self.reranker_fetch_k: int = self._get_env_int(
            "RERANKER_FETCH_K",
            self.top_k * 5,
        )

        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "")
        self.groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.llm_temperature: float = self._get_env_float("LLM_TEMPERATURE", 0.1)
        self.llm_max_tokens: int = self._get_env_int("LLM_MAX_TOKENS", 1024)
        self.llm_request_timeout: float = self._get_env_float(
            "LLM_REQUEST_TIMEOUT",
            30.0,
        )
        self.langgraph_enabled: bool = self._get_env_bool("LANGGRAPH_ENABLED", True)
        self.default_query: str = os.getenv(
            "DEFAULT_QUERY",
            "Types of Machine Learning Systems",
        )

    @staticmethod
    def _get_env_bool(key: str, default: bool) -> bool:
        value = os.getenv(key)
        if value is None:
            return default

        normalized = value.strip().lower()

        if normalized in {"1", "true", "yes", "y", "on"}:
            return True

        if normalized in {"0", "false", "no", "n", "off"}:
            return False

        raise ValueError(f"Invalid boolean value for {key}: {value}")

    @staticmethod
    def _get_env_int(key: str, default: int) -> int:
        value = os.getenv(key)
        if value is None:
            return default

        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"Invalid integer value for {key}: {value}") from error

    @staticmethod
    def _get_env_float(key: str, default: float) -> float:
        value = os.getenv(key)
        if value is None:
            return default

        try:
            return float(value)
        except ValueError as error:
            raise ValueError(f"Invalid float value for {key}: {value}") from error

    def validate_for_langgraph(self) -> None:
        """Validate required configuration for LangGraph Groq execution."""
        if not self.groq_api_key.strip():
            raise ValueError("GROQ_API_KEY is required for LangGraph execution")

        if self.llm_max_tokens <= 0:
            raise ValueError("LLM_MAX_TOKENS must be greater than 0")

        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError("LLM_TEMPERATURE must be between 0.0 and 2.0")


settings = Settings()
