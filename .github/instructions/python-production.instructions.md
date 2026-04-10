---
description: "Use when writing or modifying Python code. Enforces strict PEP 8, production-grade quality, OOP where appropriate, stable library usage, deterministic behavior, and structured logging best practices."
name: "Python Production Standards"
applyTo: "**/*.py"
---
# Python Production Standards

- Follow PEP 8 strictly for formatting, naming, spacing, imports, and line length.
- Write clean, readable, maintainable, scalable, production-grade code.
- Apply object-oriented design where it materially improves structure or extensibility.
- Do not produce demo, toy, or illustrative code unless explicitly requested.
- Do not add unnecessary comments, inline explanations, or print statements.
- Use structured, configurable logging with appropriate levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.
- Keep logs concise and useful; never log secrets or sensitive data.
- Use only officially supported, stable libraries and APIs.
- Avoid deprecated libraries, APIs, functions, and patterns.
- When uncertain about an API, verify behavior against official documentation or authoritative sources before implementation.
- Favor deterministic, testable code suitable for production deployment.
- Avoid global mutable state and hard-coded values; prefer configuration-driven behavior.
- Raise explicit, meaningful exceptions; never silently suppress errors.
- Return only valid code or explicitly requested non-code content.
