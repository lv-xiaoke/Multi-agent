"""
CLI entry point for the multi-agent analyst application.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any

from multi_agent_analyst.core.graph_builder import build_and_compile_graph
from multi_agent_analyst.core.state import AgentState

try:
    from langchain_core.messages import HumanMessage
except ImportError:  # pragma: no cover
    class HumanMessage:  # type: ignore[no-redef]
        def __init__(self, content: str):
            self.content = content


def run_analysis(
    user_request: str,
    data_source: str,
    output_dir: str,
    max_iterations: int = 3,
) -> AgentState:
    """
    Run the full multi-agent workflow and return the final state.
    """

    run_id = uuid.uuid4().hex[:12]
    root_output = Path(output_dir).resolve()
    run_output = root_output / run_id
    run_output.mkdir(parents=True, exist_ok=True)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_request)],
        "user_request": user_request,
        "data_source": str(Path(data_source).resolve()),
        "output_dir": str(run_output),
        "run_id": run_id,
        "iterations": 0,
        "max_iterations": max_iterations,
        "memory_db_path": str(root_output / "memory.sqlite3"),
    }

    graph = build_and_compile_graph()
    return graph.invoke(initial_state)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a LangGraph multi-agent CSV analysis workflow.")
    parser.add_argument("--data", required=True, help="Path to a local CSV file.")
    parser.add_argument("--task", required=True, help="Natural language analysis request.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum code repair attempts.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for run artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    final_state: dict[str, Any] = run_analysis(
        user_request=args.task,
        data_source=args.data,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
    )

    print(f"Run ID: {final_state.get('run_id')}")
    print(f"Report: {final_state.get('report_path')}")
    print(f"Review: {final_state.get('review_feedback')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
