"""Application entrypoint for the LangChain arithmetic agent."""

from __future__ import annotations

import argparse

from loguru import logger

from langchain_agents.agents import ArithmeticAgentService
from langchain_agents.config import AppConfig
from langchain_agents.utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Create command-line parser for runtime options."""
    parser = argparse.ArgumentParser(description="Run arithmetic LangChain agent queries")
    parser.add_argument(
        "--query",
        default="What is 5+3-4*2?",
        help="Question to ask the arithmetic agent.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level for Loguru output (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser


def main() -> None:
    """Run the application with validated configuration and structured logging."""
    args = build_parser().parse_args()
    configure_logging(level=args.log_level)

    config = AppConfig.from_environment()
    config.apply_to_environment()

    service = ArithmeticAgentService(config=config)
    answer = service.ask(args.query)

    logger.info("Final response: {}", answer)
    print(answer)


if __name__ == "__main__":
    main()


