"""Configuration utilities for loading and validating runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(slots=True)
class AppConfig:
    """Strongly typed runtime configuration loaded from environment variables."""

    groq_api_key: str
    langfuse_secret_key: str | None
    langfuse_public_key: str | None
    langfuse_base_url: str | None
    model_name: str = "qwen/qwen3-32b"
    temperature: float = 0.0
    max_retries: int = 2

    @classmethod
    def from_environment(cls) -> "AppConfig":
        """Build and validate configuration from environment variables.

        Returns:
            AppConfig: Fully initialized application configuration.

        Raises:
            ValueError: If required environment variables are missing.
        """
        load_dotenv()

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your environment or .env file."
            )

        return cls(
            groq_api_key=groq_api_key,
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_base_url=os.getenv("LANGFUSE_BASE_URL"),
        )

    def apply_to_environment(self) -> None:
        """Expose configuration through process environment for SDK compatibility."""
        os.environ["GROQ_API_KEY"] = self.groq_api_key

        if self.langfuse_secret_key:
            os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_secret_key
        if self.langfuse_public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_public_key
        if self.langfuse_base_url:
            os.environ["LANGFUSE_BASE_URL"] = self.langfuse_base_url
