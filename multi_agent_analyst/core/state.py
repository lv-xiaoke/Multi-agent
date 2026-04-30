"""
Defines the shared state passed between LangGraph nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

try:
    from langchain_core.messages import BaseMessage
except ImportError:  # pragma: no cover
    BaseMessage = Any  # type: ignore[misc, assignment]


class AgentState(TypedDict, total=False):
    """
    Global state shared across the multi-agent analysis workflow.
    """

    messages: Annotated[list[BaseMessage], operator.add]
    user_request: str
    data_source: str
    output_dir: str
    run_id: str
    data_profile: dict[str, Any]
    plan: str
    generated_code: str
    execution_result: dict[str, Any]
    charts: list[dict[str, str]]
    report_path: str
    review_feedback: str
    iterations: int
    max_iterations: int
    memory_db_path: str
