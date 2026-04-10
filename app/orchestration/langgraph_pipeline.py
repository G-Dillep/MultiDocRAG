"""LangGraph orchestration with LLM-based query routing."""

from __future__ import annotations

from typing import Any, TypedDict, cast

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.llm.groq_service import GroqLLMService, QueryRoute, RAGResponse
from app.retrieval.retriever import DocumentRetriever, VectorStoreLike
from app.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger()


class PipelineState(TypedDict):
    """State used by the classification, retrieval, and generation graph."""

    query: str
    db: VectorStoreLike
    route: QueryRoute
    documents: list[Document]
    response: RAGResponse


class LangGraphRAGPipeline:
    """Run query classification and route to RAG or general generation."""

    def __init__(
        self,
        retriever: DocumentRetriever | None = None,
        llm_service: GroqLLMService | None = None,
    ) -> None:
        """Initialize graph dependencies and compile execution graph."""
        self._retriever = retriever or DocumentRetriever()
        self._llm_service = llm_service or GroqLLMService()
        self._graph = self._build_graph()

    def invoke(self, db: VectorStoreLike, query: str) -> RAGResponse:
        """Execute the LangGraph pipeline for a single query.

        Args:
            db: Vector store used for retrieval.
            query: User query text.

        Returns:
            Structured JSON-compatible response payload.

        Raises:
            ValueError: If query is empty.
        """
        if not query.strip():
            raise ValueError("Query must not be empty")

        logger.info("[LangGraph] Pipeline execution started")

        initial_state: PipelineState = {
            "query": query,
            "db": db,
            "route": "rag",
            "documents": [],
            "response": {
                "answer": "",
                "sources": [],
                "retrieval": {"document_count": 0},
                "model": {"provider": "", "model_name": ""},
            },
        }

        final_state = self._graph.invoke(initial_state)

        if not isinstance(final_state, dict):
            raise ValueError("LangGraph returned an invalid state type")

        response = final_state.get("response")
        if not isinstance(response, dict):
            raise ValueError("LangGraph final state is missing response payload")

        logger.info("[LangGraph] Pipeline execution completed")
        return cast(RAGResponse, response)

    def _build_graph(self) -> Any:
        graph = StateGraph(PipelineState)

        graph.add_node("classify", self._classify_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate_rag", self._generate_rag_node)
        graph.add_node("generate_general", self._generate_general_node)

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            self._route_after_classification,
            {
                "rag": "retrieve",
                "general": "generate_general",
            },
        )
        graph.add_edge("retrieve", "generate_rag")
        graph.add_edge("generate_rag", END)
        graph.add_edge("generate_general", END)

        return graph.compile()

    def _classify_node(self, state: PipelineState) -> dict[str, Any]:
        logger.info("[LangGraph] Classification node started")
        route = self._llm_service.classify_query_route(state["query"])
        logger.info(f"[LangGraph] Classification node completed with route={route}")
        return {"route": route}

    @staticmethod
    def _route_after_classification(state: PipelineState) -> QueryRoute:
        return state["route"]

    def _retrieve_node(self, state: PipelineState) -> dict[str, Any]:
        logger.info("[LangGraph] Retrieval node started")
        documents = self._retriever.retrieve(state["db"], state["query"])
        logger.info(
            f"[LangGraph] Retrieval node completed with {len(documents)} documents"
        )
        return {"documents": documents}

    def _generate_rag_node(self, state: PipelineState) -> dict[str, Any]:
        logger.info("[LangGraph] RAG generation node started")
        response = self._llm_service.generate(state["query"], state["documents"])
        logger.info("[LangGraph] RAG generation node completed")
        return {"response": response}

    def _generate_general_node(self, state: PipelineState) -> dict[str, Any]:
        logger.info("[LangGraph] General generation node started")
        response = self._llm_service.generate_general(state["query"])
        logger.info("[LangGraph] General generation node completed")
        return {"response": response}
