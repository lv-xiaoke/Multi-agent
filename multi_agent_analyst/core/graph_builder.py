"""
Builds and compiles the LangGraph StateGraph workflow.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from multi_agent_analyst.agents.coder import run_coder
from multi_agent_analyst.agents.planner import run_planner
from multi_agent_analyst.agents.reviewer import route_after_review, run_chart_renderer, run_reviewer
from multi_agent_analyst.core.state import AgentState
from multi_agent_analyst.tools.data_io import run_data_profiler
from multi_agent_analyst.tools.executor import run_executor
from multi_agent_analyst.tools.memory import run_memory_writer
from multi_agent_analyst.tools.reporting import run_reporter

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    END = "__end__"
    StateGraph = None  # type: ignore[assignment]


class _FallbackGraph:
    """
    Sequential local runner used when LangGraph is not installed.
    """

    def invoke(self, initial_state: AgentState) -> AgentState:
        state = dict(initial_state)
        for node in (run_data_profiler, run_planner):
            state.update(node(state))

        while True:
            state.update(run_coder(state))
            state.update(run_executor(state))
            state.update(run_chart_renderer(state))
            state.update(run_reviewer(state))
            if route_after_review(state) == "reporter":
                break

        state.update(run_reporter(state))
        state.update(run_memory_writer(state))
        return state


def build_and_compile_graph() -> Callable[..., Any]:
    """
    Build and compile the multi-agent data analysis workflow.
    """

    if StateGraph is None:
        return _FallbackGraph()

    graph = StateGraph(AgentState)

    graph.add_node("data_profiler", run_data_profiler)
    graph.add_node("planner", run_planner)
    graph.add_node("coder", run_coder)
    graph.add_node("executor", run_executor)
    graph.add_node("chart_renderer", run_chart_renderer)
    graph.add_node("reviewer", run_reviewer)
    graph.add_node("reporter", run_reporter)
    graph.add_node("memory", run_memory_writer)

    graph.set_entry_point("data_profiler")
    graph.add_edge("data_profiler", "planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "executor")
    graph.add_edge("executor", "chart_renderer")
    graph.add_edge("chart_renderer", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "coder": "coder",
            "reporter": "reporter",
        },
    )
    graph.add_edge("reporter", "memory")
    graph.add_edge("memory", END)

    return graph.compile()
