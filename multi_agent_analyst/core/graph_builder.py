"""
Builds and compiles the LangGraph StateGraph workflow.
"""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_analyst.core.state import AgentState

try:
    from multi_agent_analyst.agents.planner import run_planner
    from multi_agent_analyst.agents.coder import run_coder
    from multi_agent_analyst.agents.reviewer import route_after_execution
    from multi_agent_analyst.tools.executor import run_executor
except ImportError:
    # Temporary mocks for the scaffold stage. Replace them by implementing the
    # real functions in agents/ and tools/.
    def _not_implemented_node(state: AgentState) -> dict[str, Any]:
        raise NotImplementedError("Agent node function has not been implemented.")

    def run_planner(state: AgentState) -> dict[str, Any]:
        return _not_implemented_node(state)

    def run_coder(state: AgentState) -> dict[str, Any]:
        return _not_implemented_node(state)

    def run_executor(state: AgentState) -> dict[str, Any]:
        return _not_implemented_node(state)

    def route_after_execution(state: AgentState) -> str:
        raise NotImplementedError("Conditional route function has not been implemented.")


def build_and_compile_graph() -> Callable[..., Any]:
    """
    Build and compile the LangGraph workflow.

    Workflow:
        planner -> coder -> executor -> conditional route

    Conditional route:
        - "Coder" or "coder": retry by returning to the coder node.
        - "FinalReport", "END", or "__end__": finish the graph.
        - "HumanFallback": finish the graph for now. A dedicated human
          fallback node can be added later if needed.
    """

    graph = StateGraph(AgentState)

    graph.add_node("planner", run_planner)
    graph.add_node("coder", run_coder)
    graph.add_node("executor", run_executor)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "executor")

    graph.add_conditional_edges(
        "executor",
        route_after_execution,
        {
            "Coder": "coder",
            "coder": "coder",
            "FinalReport": END,
            "END": END,
            "__end__": END,
            "HumanFallback": END,
        },
    )

    return graph.compile()
