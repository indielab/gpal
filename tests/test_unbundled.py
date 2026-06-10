import os
import pytest

from gpal.server import (
    _gemini_search,
    _gemini_code_exec,
)


def test_gemini_search():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    result = _gemini_search("What is the capital of France?")
    print(f"\nSearch Result:\n{result}")
    assert "Paris" in result
    assert "Error" not in result


def test_gemini_code_exec():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    code = "print(123 + 456)"
    result = _gemini_code_exec(code)
    print(f"\nCode Execution Result:\n{result}")
    assert "579" in result
    assert "Error" not in result
