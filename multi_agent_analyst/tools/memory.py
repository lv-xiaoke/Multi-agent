"""
Long-term memory persistence backed by SQLite.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from multi_agent_analyst.core.state import AgentState


def save_run_memory(state: AgentState) -> None:
    db_path = Path(state.get("memory_db_path") or Path(state["output_dir"]).parent / "memory.sqlite3")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    profile = state.get("data_profile", {})
    payload = {
        "run_id": state.get("run_id", ""),
        "user_request": state.get("user_request", ""),
        "data_source": state.get("data_source", ""),
        "data_profile": {
            "file_name": profile.get("file_name"),
            "shape": profile.get("shape"),
            "columns": profile.get("columns", []),
        },
        "report_path": state.get("report_path", ""),
        "review_feedback": state.get("review_feedback", ""),
    }

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id TEXT PRIMARY KEY,
                user_request TEXT NOT NULL,
                data_source TEXT NOT NULL,
                report_path TEXT,
                payload_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO analysis_runs
            (run_id, user_request, data_source, report_path, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                payload["run_id"],
                payload["user_request"],
                payload["data_source"],
                payload["report_path"],
                json.dumps(payload, ensure_ascii=False),
            ),
        )


def run_memory_writer(state: AgentState) -> dict[str, Any]:
    save_run_memory(state)
    return {}
