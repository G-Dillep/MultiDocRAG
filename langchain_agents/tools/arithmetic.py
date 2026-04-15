"""Tool definitions used by the arithmetic agent."""

from __future__ import annotations

from langchain.tools import tool


class ArithmeticTools:
    """Factory class that provides arithmetic tools for LangChain agents."""

    @staticmethod
    @tool("add_numbers")
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together.

        Args:
            a: First integer.
            b: Second integer.

        Returns:
            int: Sum of both integers.
        """
        return a + b

    @staticmethod
    @tool("subtract_numbers")
    def subtract_numbers(a: int, b: int) -> int:
        """Subtract the second number from the first.

        Args:
            a: Minuend.
            b: Subtrahend.

        Returns:
            int: Difference of the two numbers.
        """
        return a - b

    @staticmethod
    @tool("multiply_numbers")
    def multiply_numbers(a: int, b: int) -> int:
        """Multiply two numbers.

        Args:
            a: First integer.
            b: Second integer.

        Returns:
            int: Product of the two numbers.
        """
        return a * b

    @staticmethod
    @tool("divide_numbers")
    def divide_numbers(a: int, b: int) -> float:
        """Divide the first number by the second.

        Args:
            a: Numerator.
            b: Denominator.

        Returns:
            float: Division result.

        Raises:
            ValueError: If denominator is zero.
        """
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    @classmethod
    def all_tools(cls) -> list:
        """Return all arithmetic tool callables in a stable order."""
        return [
            cls.add_numbers,
            cls.subtract_numbers,
            cls.multiply_numbers,
            cls.divide_numbers,
        ]
