"""Centralized logging configuration for the RAG pipeline."""

import sys
from typing import Any

from loguru import logger

from app.config.settings import settings


class LoggerFactory:
    """Configure and provide a singleton Loguru logger."""

    _configured = False

    @classmethod
    def get_logger(cls) -> Any:
        """Get the configured logger instance.

        Returns:
            Configured Loguru logger.
        """
        if not cls._configured:
            cls._configure_logger()
            cls._configured = True

        return logger

    @staticmethod
    def _configure_logger() -> None:
        """Configure console and file logger sinks."""
        logger.remove()

        # Console logging
        logger.add(
            sys.stdout,
            level=settings.log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level}</level> | "
                "<cyan>{name}:{function}:{line}</cyan> - "
                "<level>{message}</level>"
            ),
        )

        # File logging
        logger.add(
            settings.log_file_path,
            rotation=settings.log_rotation,
            level=settings.log_file_level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level} | "
                "{name}:{function}:{line} - "
                "{message}"
            ),
        )
