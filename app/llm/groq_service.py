"""Groq LLM service with strict JSON contract validation."""

from __future__ import annotations

import json
from importlib import import_module
from typing import Any, TypedDict

from langchain_core.documents import Document

from app.config.settings import settings
from app.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger()


class SourceItem(TypedDict):
    """Source metadata returned with a generated answer."""

    source: str
    score: float | None
    rerank_score: float | None
    page: int | None


class RetrievalMetadata(TypedDict):
    """Metadata describing retrieval output used for generation."""

    document_count: int


class ModelMetadata(TypedDict):
    """Metadata describing model configuration used for generation."""

    provider: str
    model_name: str


class RAGResponse(TypedDict):
    """Structured response contract returned by the generation layer."""

    answer: str
    sources: list[SourceItem]
    retrieval: RetrievalMetadata
    model: ModelMetadata


class GroqLLMService:
    """Generate structured answers using a Groq-hosted chat model."""

    def __init__(self) -> None:
        """Initialize the Groq chat model client."""
        settings.validate_for_langgraph()

        try:
            module = import_module("langchain_groq")
            chat_groq_cls = getattr(module, "ChatGroq")
        except (ImportError, AttributeError) as error:
            raise ImportError(
                "langchain-groq is required. Install dependencies before running.",
            ) from error

        self._model_name = settings.groq_model
        self._client = chat_groq_cls(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_request_timeout,
        )

    def generate(self, query: str, documents: list[Document]) -> RAGResponse:
        """Generate a structured answer and citation payload.

        Args:
            query: User query text.
            documents: Retrieved documents for context grounding.

        Returns:
            Strictly validated JSON-compatible response payload.

        Raises:
            ValueError: If model output is missing required fields.
            json.JSONDecodeError: If model output is not valid JSON.
        """
        if not query.strip():
            raise ValueError("Query must not be empty")

        prompt = self._build_prompt(query, documents)

        logger.info("[LLMService] Invoking Groq model")
        response = self._client.invoke(prompt)
        raw_output = self._extract_content(response)

        payload = self._parse_response_json(raw_output)
        validated = self._validate_response(payload)

        validated["retrieval"] = {"document_count": len(documents)}
        validated["model"] = {
            "provider": "groq",
            "model_name": self._model_name,
        }

        logger.info("[LLMService] Groq response validated")
        return validated

    def _build_prompt(self, query: str, documents: list[Document]) -> str:
        context_blocks: list[str] = []

        for index, doc in enumerate(documents, start=1):
            metadata = {
                "source": str(doc.metadata.get("source", "unknown")),
                "page": self._to_optional_int(doc.metadata.get("page")),
                "score": self._to_optional_float(doc.metadata.get("score")),
                "rerank_score": self._to_optional_float(
                    doc.metadata.get("rerank_score"),
                ),
            }
            context_blocks.append(
                "\n".join(
                    [
                        f"[Document {index}]",
                        f"metadata={json.dumps(metadata, ensure_ascii=True)}",
                        "content:",
                        doc.page_content,
                    ],
                ),
            )

        context_text = "\n\n".join(context_blocks)

        return "\n".join(
            [
                "You are a retrieval-augmented assistant.",
                "Use only the provided context.",
                "Return valid JSON only without markdown fences.",
                "JSON schema:",
                (
                    '{"answer": "string", "sources": '
                    '[{"source": "string", "score": number|null, '
                    '"rerank_score": number|null, "page": number|null}]}'
                ),
                "Rules:",
                "1. Include only cited sources used in the answer.",
                (
                    "2. If context is insufficient, answer exactly: "
                    '"Insufficient context."'
                ),
                "3. Do not add keys other than answer and sources.",
                f"Query: {query}",
                "Context:",
                context_text,
            ],
        )

    @staticmethod
    def _extract_content(response: Any) -> str:
        content = getattr(response, "content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            joined = "".join(str(part) for part in content)
            return joined.strip()

        return str(content).strip()

    def _parse_response_json(self, raw_output: str) -> dict[str, Any]:
        normalized = raw_output.strip()

        if normalized.startswith("```"):
            normalized = normalized.removeprefix("```").strip()
            if normalized.lower().startswith("json"):
                normalized = normalized[4:].strip()
            if normalized.endswith("```"):
                normalized = normalized[:-3].strip()

        parsed = json.loads(normalized)

        if not isinstance(parsed, dict):
            raise ValueError("Groq response must be a JSON object")

        return parsed

    def _validate_response(self, payload: dict[str, Any]) -> RAGResponse:
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Response field 'answer' must be a non-empty string")

        sources_raw = payload.get("sources")
        if not isinstance(sources_raw, list):
            raise ValueError("Response field 'sources' must be a list")

        sources: list[SourceItem] = []

        for item in sources_raw:
            if not isinstance(item, dict):
                raise ValueError("Each source item must be an object")

            source_name = item.get("source")
            if not isinstance(source_name, str) or not source_name.strip():
                raise ValueError("Each source item must contain a non-empty 'source'")

            sources.append(
                {
                    "source": source_name,
                    "score": self._to_optional_float(item.get("score")),
                    "rerank_score": self._to_optional_float(item.get("rerank_score")),
                    "page": self._to_optional_int(item.get("page")),
                },
            )

        return {
            "answer": answer.strip(),
            "sources": sources,
            "retrieval": {"document_count": 0},
            "model": {"provider": "", "model_name": ""},
        }

    @staticmethod
    def _to_optional_float(value: Any) -> float | None:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        return None

    @staticmethod
    def _to_optional_int(value: Any) -> int | None:
        if value is None:
            return None

        if isinstance(value, int):
            return value

        if isinstance(value, float) and value.is_integer():
            return int(value)

        return None
