"""
Implements the Coder agent logic.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.state import AgentState


SYSTEM_PROMPT = """
你是一位资深 Python 数据工程师，负责根据 Data Planner 提供的数据分析计划编写可执行的 Python 数据分析代码。

你必须严格遵守以下要求：

1. 只输出 Python 代码，并且必须把代码包裹在 ```python 和 ``` 之间。
2. 不要输出任何解释、寒暄、Markdown 标题或代码之外的文字。
3. 代码应当结构清晰、健壮，包含必要的异常处理。
4. 数据清洗逻辑必须严格依据 Data Planner 的计划实现。
5. 如果存在上一次执行错误，必须根据错误栈修复代码，不要重复同样的问题。
6. 如需使用 matplotlib、seaborn 或其他可视化库，必须使用莫兰迪高级色系（Morandi palette）。
7. 可视化图表的整体排版必须符合顶级学术期刊的精简风格：低饱和配色、简洁坐标轴、清晰标题、适度留白、避免多余装饰。
8. 不要假设不存在的字段；所有字段使用必须来自 Data Planner 的分析计划或用户上下文。
"""


def _build_user_prompt(state: AgentState) -> str:
    """
    Build the prompt payload sent to the Coder LLM.
    """

    plan = state.get("plan", "")
    execution_result = state.get("execution_result", "")

    prompt_parts = [
        "请根据以下 Data Planner 生成的分析计划，编写完整、可执行的 Python 数据分析代码。",
        "",
        "Data Planner 分析计划：",
        str(plan),
    ]

    if execution_result:
        prompt_parts.extend(
            [
                "",
                "这是上次执行的错误栈，请修复代码：",
                str(execution_result),
            ]
        )

    return "\n".join(prompt_parts)


def _extract_python_code(llm_output: str) -> str:
    """
    Extract raw Python code from a Markdown fenced code block.
    """

    if not isinstance(llm_output, str):
        raise TypeError("LLM output must be a string.")

    match = re.search(
        r"```(?:python|py)?\s*(.*?)```",
        llm_output,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    # Defensive fallback:
    # The system prompt strongly requires fenced Python code, but this keeps the
    # workflow usable if the model returns plain code without fences.
    return llm_output.strip()


def _get_response_text(response: Any) -> str:
    """
    Normalize LangChain model responses into plain text.
    """

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "\n".join(part for part in text_parts if part)

    return str(content)


def run_coder(state: AgentState) -> dict:
    """
    Generate or repair Python analysis code based on the current AgentState.
    """

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.strip()),
            HumanMessage(content=_build_user_prompt(state)),
        ]

        response = llm.invoke(messages)
        raw_output = _get_response_text(response)
        clean_code = _extract_python_code(raw_output)

        return {"generated_code": clean_code}

    except Exception as exc:
        # In production, replace this with structured logging and error tracing.
        # Re-raising keeps LangGraph from silently continuing with invalid code.
        raise RuntimeError(f"Coder agent failed to generate Python code: {exc}") from exc
