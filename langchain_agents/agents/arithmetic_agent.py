"""Service layer for creating and invoking the arithmetic LangChain agent."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from loguru import logger

from langchain_agents.config import AppConfig
from langchain_agents.tools import ArithmeticTools


class ArithmeticAgentService:
    """Orchestrates LLM, tools, callbacks, and query execution."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize service dependencies.

        Args:
            config: Application configuration object.
        """
        self._config = config
        self._llm = self._build_llm()
        self._agent = self._build_agent()
        self._langfuse_handler = self._build_langfuse_handler()

    def _build_llm(self) -> ChatGroq:
        """Create the configured Groq chat model client."""
        logger.info("Initializing ChatGroq model: {}", self._config.model_name)
        return ChatGroq(
            model=self._config.model_name,
            temperature=self._config.temperature,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=self._config.max_retries,
        )

    def _build_agent(self):
        """Create the LangChain tool-enabled arithmetic agent."""
        logger.info(
            "Creating arithmetic agent with {} tools",
            len(ArithmeticTools.all_tools()),
        )
        return create_agent(
            self._llm,
            ArithmeticTools.all_tools(),
            system_prompt=(
                "You are a helpful assistant that can perform basic arithmetic operations. "
                "When given a question, determine which tool to use and call it with "
                "the appropriate arguments. Only use the tools provided, and do not "
                "attempt to perform calculations yourself."
            ),
        )

    @staticmethod
    def _build_langfuse_handler() -> CallbackHandler:
        """Create a Langfuse callback handler for tracing."""
        get_client()
        return CallbackHandler()

    def ask(self, user_prompt: str) -> str:
        """Execute a single user query through the agent.

        Args:
            user_prompt: Natural language question to process.

        Returns:
            str: Final assistant response message.
        """
        logger.info("Running agent query")
        result: dict[str, Any] = self._agent.invoke(
            {"messages": [{"role": "user", "content": user_prompt}]},
            config={"callbacks": [self._langfuse_handler]},
        )
        return result["messages"][-1].content
