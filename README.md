# LangChain Arithmetic Agent (Production-Style)

Modular LangChain project with:

- OOP service and tool classes
- PEP 8-friendly structure and naming
- Full docstrings and type hints
- Structured logging with Loguru
- `.env`-driven runtime configuration

## Project Structure

```text
.
|-- langchain_agents/
|   |-- __init__.py
|   |-- agents/
|   |   |-- __init__.py
|   |   `-- arithmetic_agent.py
|   |-- config/
|   |   |-- __init__.py
|   |   `-- settings.py
|   |-- tools/
|   |   |-- __init__.py
|   |   `-- arithmetic.py
|   `-- utils/
|       |-- __init__.py
|       `-- logging.py
|-- initial.ipynb
|-- main.py
|-- pyproject.toml
`-- README.md
```

## Prerequisites

- Python 3.11+
- Groq API key
- Optional Langfuse keys for tracing

## Environment Variables

Create `.env` in project root:

```env
GROQ_API_KEY=your_groq_key
LANGFUSE_SECRET_KEY=your_langfuse_secret
LANGFUSE_PUBLIC_KEY=your_langfuse_public
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Only `GROQ_API_KEY` is required.

## Installation

```bash
pip install -e .
```

## Run

```bash
python main.py --query "What is 10 multiplied by 4 and then divided by 2?"
```

Optional logging level:

```bash
python main.py --log-level DEBUG
```

## Design Notes

- `config/settings.py` contains `AppConfig` for env loading and validation.
- `tools/arithmetic.py` contains reusable arithmetic tool callables.
- `agents/arithmetic_agent.py` encapsulates model, callbacks, and invocation.
- `utils/logging.py` centralizes production-ready Loguru formatting.

## Notebook

`initial.ipynb` remains available for experimentation; `main.py` is the script entrypoint for production-style execution.
