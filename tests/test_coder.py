from multi_agent_analyst.agents.coder import _extract_python_code


def test_extract_python_code_from_fenced_block() -> None:
    output = "```python\nprint('hello')\n```"

    assert _extract_python_code(output) == "print('hello')"


def test_extract_python_code_falls_back_to_plain_text() -> None:
    assert _extract_python_code("print('hello')") == "print('hello')"
