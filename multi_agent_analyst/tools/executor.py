"""
Provides local code execution utilities for generated analysis scripts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from multi_agent_analyst.core.state import AgentState


ARTIFACT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".json", ".txt", ".md", ".html"}


def _scan_artifacts(output_dir: Path) -> list[str]:
    artifacts: list[str] = []
    for path in output_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in ARTIFACT_EXTENSIONS:
            artifacts.append(str(path.resolve()))
    return sorted(artifacts)


def execute_python_code(
    code: str,
    data_source: str,
    output_dir: str,
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    """
    Write generated code to the run directory and execute it as a subprocess.
    """

    run_dir = Path(output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    script_path = run_dir / "generated_analysis.py"
    script_path.write_text(code, encoding="utf-8")

    env = os.environ.copy()
    env["DATA_SOURCE"] = str(Path(data_source).resolve())
    env["OUTPUT_DIR"] = str(run_dir)

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=run_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        status = "success" if completed.returncode == 0 else "failed"
        return {
            "status": status,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "error": completed.stderr if completed.returncode != 0 else "",
            "script_path": str(script_path),
            "artifacts": _scan_artifacts(run_dir),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": f"Generated analysis timed out after {timeout_seconds} seconds.",
            "script_path": str(script_path),
            "artifacts": _scan_artifacts(run_dir),
        }


def run_executor(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node that executes the latest generated Python code.
    """

    result = execute_python_code(
        code=state.get("generated_code", ""),
        data_source=state["data_source"],
        output_dir=state["output_dir"],
    )
    iterations = int(state.get("iterations", 0)) + 1
    return {"execution_result": result, "iterations": iterations}


def load_execution_results(output_dir: str) -> dict[str, Any]:
    path = Path(output_dir) / "analysis_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
