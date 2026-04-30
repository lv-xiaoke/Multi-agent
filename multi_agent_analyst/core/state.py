"""
Defines the global AgentState used by the LangGraph workflow.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Global state shared across the LangGraph multi-agent workflow.
    """

    messages: Annotated[list[BaseMessage], operator.add]
    plan: str
    generated_code: str
    execution_result: str
    iterations: int
