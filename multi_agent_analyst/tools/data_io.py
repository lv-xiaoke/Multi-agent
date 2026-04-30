"""
Data loading and profiling utilities for local CSV inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from multi_agent_analyst.core.state import AgentState


def _require_csv_path(data_source: str) -> Path:
    path = Path(data_source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV data source does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("v1 only supports local CSV files.")
    return path


def profile_csv(data_source: str, sample_rows: int = 5) -> dict[str, Any]:
    """
    Read a CSV and return a compact, structured profile for downstream agents.
    """

    path = _require_csv_path(data_source)
    frame = pd.read_csv(path)

    missing = frame.isna().sum()
    profile: dict[str, Any] = {
        "source_path": str(path),
        "file_name": path.name,
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "columns": [],
        "sample_rows": frame.head(sample_rows).where(pd.notna(frame), None).to_dict("records"),
    }

    numeric_frame = frame.select_dtypes(include="number")
    if not numeric_frame.empty:
        profile["numeric_summary"] = (
            numeric_frame.describe().round(4).where(pd.notna(numeric_frame.describe()), None).to_dict()
        )
    else:
        profile["numeric_summary"] = {}

    for column in frame.columns:
        series = frame[column]
        profile["columns"].append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "missing_count": int(missing[column]),
                "missing_rate": round(float(missing[column] / len(frame)), 4) if len(frame) else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": [
                    None if pd.isna(value) else value
                    for value in series.drop_duplicates().head(sample_rows).tolist()
                ],
            }
        )

    return profile


def run_data_profiler(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node that profiles the user-provided CSV.
    """

    profile = profile_csv(state["data_source"])
    return {"data_profile": profile}
