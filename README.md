# RAG Pipeline

## Runtime

The runtime path executes retrieval plus generation through LangGraph and Groq.

## Required Environment Variables

- `GROQ_API_KEY`: Groq API key.

## Optional Environment Variables

- `GROQ_MODEL`: Defaults to `llama-3.3-70b-versatile`.
- `LLM_TEMPERATURE`: Defaults to `0.1`.
- `LLM_MAX_TOKENS`: Defaults to `1024`.
- `LLM_REQUEST_TIMEOUT`: Defaults to `30.0`.
- `DEFAULT_QUERY`: Defaults to `Types of Machine Learning Systems`.
- `DATA_PATH`: Defaults to `data`.
- `CHROMA_PATH`: Defaults to `chroma_db`.
- `TOP_K`: Defaults to `3`.
- `RERANKER_ENABLED`: Defaults to `true`.
- `RERANKER_TOP_K`: Defaults to `TOP_K`.
- `RERANKER_FETCH_K`: Defaults to `TOP_K * 5`.
- `PDF_BACKEND`: Defaults to `pymupdf`.
- `PDF_MODE`: Defaults to `page`.

`.env` files are supported through `python-dotenv`.

## Structured JSON Response Contract

The generation layer returns a strict JSON object:

```json
{
	"answer": "string",
	"sources": [
		{
			"source": "string",
			"score": 0.0,
			"rerank_score": 0.0,
			"page": 1
		}
	],
	"retrieval": {
		"document_count": 3
	},
	"model": {
		"provider": "groq",
		"model_name": "llama-3.3-70b-versatile"
	}
}
```
