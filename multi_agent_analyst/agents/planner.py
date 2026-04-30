"""
Implements the Data Planner agent logic.
"""

from __future__ import annotations

from core.state import AgentState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def _extract_user_requirement(messages: list[BaseMessage]) -> str:
    """
    Extract the latest human request from the conversation history.
    """

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)

    if messages:
        return str(messages[-1].content)

    return ""


def run_planner(state: AgentState) -> dict:
    """
    Run the Data Planner agent and update the global state with an analysis plan.

    The planner receives the user's analysis request from state["messages"],
    asks the LLM to produce a rigorous data analysis plan, and returns only the
    state fields that should be updated by LangGraph.
    """

    messages = state.get("messages", [])
    user_requirement = _extract_user_requirement(messages)

    system_prompt = """
You are the Data Planner agent in a LangGraph-based multi-agent data analysis system.

Your role is to act as a senior data scientist and mathematician. You must transform
the user's data analysis request and any provided schema information into a rigorous,
step-by-step analysis plan for the downstream Coder agent.

Planning requirements:
1. Clarify the analytical objective, target variables, feature variables, and expected outputs.
2. Design a statistically rigorous workflow, including data validation, missing-value handling,
   outlier detection, distribution checks, and assumptions that must be verified before modeling.
3. Explicitly specify appropriate statistical methods or machine learning models when relevant,
   such as descriptive statistics, normality tests, correlation analysis, hypothesis testing,
   regression models, PCA, multidimensional subspace analysis, clustering, K-Means,
   classification models, or time-series methods.
4. Explain why each method is mathematically or statistically suitable for the task.
5. Give concrete instructions for the Coder agent about how to implement data cleaning,
   feature engineering, transformations, encoding, scaling, train/test splitting,
   model evaluation, and result reporting.
6. Do not invent fields that are not present in the user's request or schema. If information is
   missing, state the uncertainty and provide a conservative executable plan.
7. Output a clear, implementation-oriented plan. Do not write executable Python code.
""".strip()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_requirement),
        ]
    )

    llm_output = str(response.content)

    return {
        "plan": llm_output,
        "messages": [AIMessage(content=llm_output)],
    }
