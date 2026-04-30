"""
Implements artifact normalization, review, and graph routing logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from multi_agent_analyst.core.state import AgentState


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp"}


def _charts_from_results(output_dir: Path) -> list[dict[str, str]]:
    result_path = output_dir / "analysis_results.json"
    if not result_path.exists():
        return []

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    charts = []
    for index, chart in enumerate(payload.get("charts", []), start=1):
        path = Path(chart.get("path", ""))
        if not path.is_absolute():
            path = output_dir / path
        if path.exists():
            charts.append(
                {
                    "title": str(chart.get("title") or f"Chart {index}"),
                    "path": str(path.resolve()),
                }
            )
    return charts


def _charts_from_artifacts(artifacts: list[str]) -> list[dict[str, str]]:
    charts = []
    for path_text in artifacts:
        path = Path(path_text)
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.exists():
            title = path.stem.replace("_", " ").title()
            charts.append({"title": title, "path": str(path.resolve())})
    return charts


def run_chart_renderer(state: AgentState) -> dict[str, Any]:
    """
    Normalize chart metadata from generated artifacts.
    """

    output_dir = Path(state["output_dir"]).resolve()
    execution_result = state.get("execution_result", {})
    charts = _charts_from_results(output_dir)
    if not charts:
        charts = _charts_from_artifacts(execution_result.get("artifacts", []))
    return {"charts": charts}


def run_reviewer(state: AgentState) -> dict[str, Any]:
    """
    Review execution outputs and decide whether another code pass is needed.
    """

    execution_result = state.get("execution_result", {})
    charts = state.get("charts", [])
    output_dir = Path(state["output_dir"])
    result_path = output_dir / "analysis_results.json"

    issues: list[str] = []
    if execution_result.get("status") != "success":
        issues.append(f"Execution status is {execution_result.get('status')}.")
    if not result_path.exists():
        issues.append("analysis_results.json was not generated.")
    if not charts:
        issues.append("No chart artifacts were generated.")

    if issues:
        feedback = "Review failed: " + " ".join(issues)
    else:
        feedback = "Review passed: execution succeeded, structured results exist, and chart artifacts are available."

    return {"review_feedback": feedback}


def route_after_review(state: AgentState) -> str:
    """
    Route to coder for repair or reporter for final output.
    """

    execution_result = state.get("execution_result", {})
    max_iterations = int(state.get("max_iterations", 3))
    iterations = int(state.get("iterations", 0))
    review_failed = state.get("review_feedback", "").startswith("Review failed")

    if (execution_result.get("status") != "success" or review_failed) and iterations < max_iterations:
        return "coder"
    return "reporter"


def route_after_execution(state: AgentState) -> str:
    """
    Backward-compatible route used by the original scaffold tests.
    """

    execution_result = state.get("execution_result", {})
    max_iterations = int(state.get("max_iterations", 3))
    iterations = int(state.get("iterations", 0))
    if execution_result.get("status") != "success" and iterations < max_iterations:
        return "Coder"
    return "FinalReport"
