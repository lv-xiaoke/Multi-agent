"""
Implements the Data Planner agent logic.
"""

from __future__ import annotations

import json
import os
from typing import Any

from multi_agent_analyst.core.state import AgentState

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
except ImportError:  # pragma: no cover
    BaseMessage = Any  # type: ignore[misc, assignment]

    class HumanMessage:  # type: ignore[no-redef]
        def __init__(self, content: str):
            self.content = content

    class AIMessage:  # type: ignore[no-redef]
        def __init__(self, content: str):
            self.content = content

    class SystemMessage:  # type: ignore[no-redef]
        def __init__(self, content: str):
            self.content = content

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment]


def _extract_user_requirement(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    if messages:
        return str(messages[-1].content)
    return ""


def _fallback_plan(user_requirement: str, data_profile: dict[str, Any]) -> str:
    columns = ", ".join(column["name"] for column in data_profile.get("columns", []))
    return f"""
Objective: {user_requirement}

Data profile:
- Rows: {data_profile.get("shape", {}).get("rows", 0)}
- Columns: {columns}

Executable analysis plan:
1. Load the CSV from DATA_SOURCE and validate that the file can be parsed.
2. Standardize column names, inspect missing values, and keep cleaning decisions explicit.
3. Produce descriptive statistics for numeric fields and frequency summaries for categorical fields.
4. Generate publication-style visualizations for informative numeric and categorical fields.
5. Write all findings into analysis_results.json with summary, insights, and chart metadata.
6. Save chart files into OUTPUT_DIR so the report renderer can include them.
""".strip()


def run_planner(state: AgentState) -> dict[str, Any]:
    """
    Build an implementation-oriented analysis plan.
    """

    messages = state.get("messages", [])
    user_requirement = state.get("user_request") or _extract_user_requirement(messages)
    data_profile = state.get("data_profile", {})

    if ChatOpenAI is None or not os.getenv("OPENAI_API_KEY"):
        llm_output = _fallback_plan(user_requirement, data_profile)
        return {"plan": llm_output, "messages": [AIMessage(content=llm_output)]}

    system_prompt = """
You are the Data Planner agent in a LangGraph-based multi-agent data analysis system.
Transform the user's request and data profile into a rigorous, executable analysis plan.
Do not invent fields. Include data validation, cleaning, statistical methods, charts,
and exact deliverables expected from the Coder agent. Do not write executable code.
""".strip()

    response = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0).invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"User request:\n{user_requirement}\n\n"
                    f"Data profile JSON:\n{json.dumps(data_profile, ensure_ascii=False, indent=2)}"
                )
            ),
        ]
    )
    llm_output = str(response.content)
    return {"plan": llm_output, "messages": [AIMessage(content=llm_output)]}
