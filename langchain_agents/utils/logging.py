"""Logging configuration helpers built on top of Loguru."""

from __future__ import annotations

import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Configure Loguru sinks and format for console-first deployments.

    Args:
        level: Minimum log level to emit.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
