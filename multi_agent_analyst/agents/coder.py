"""
Implements the Coder agent logic.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from multi_agent_analyst.core.state import AgentState

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover
    class HumanMessage:  # type: ignore[no-redef]
        def __init__(self, content: str):
            self.content = content

    class SystemMessage:  # type: ignore[no-redef]
        def __init__(self, content: str):
            self.content = content

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment]


SYSTEM_PROMPT = """
You are a senior Python data engineer. Generate executable data analysis code.
Rules:
1. Output only Python code wrapped in ```python and ```.
2. Read the CSV from environment variable DATA_SOURCE.
3. Write all artifacts into environment variable OUTPUT_DIR.
4. Always create analysis_results.json with summary, insights, cleaning_notes, and charts.
5. Save report-ready charts with a clean low-saturation style.
6. If previous execution failed, repair the code based on the error.
7. Do not invent columns outside the provided data profile.
""".strip()


def _build_user_prompt(state: AgentState) -> str:
    plan = state.get("plan", "")
    execution_result = state.get("execution_result", {})
    data_profile = state.get("data_profile", {})

    prompt_parts = [
        "Build complete executable Python analysis code from this plan and data profile.",
        "",
        "Plan:",
        str(plan),
        "",
        "Data profile JSON:",
        json.dumps(data_profile, ensure_ascii=False, indent=2),
    ]

    if execution_result and execution_result.get("status") != "success":
        prompt_parts.extend(
            [
                "",
                "Previous execution failed. Fix the code using this result:",
                json.dumps(execution_result, ensure_ascii=False, indent=2),
            ]
        )

    return "\n".join(prompt_parts)


def _extract_python_code(llm_output: str) -> str:
    """
    Extract raw Python code from a Markdown fenced code block.
    """

    if not isinstance(llm_output, str):
        raise TypeError("LLM output must be a string.")

    match = re.search(
        r"```(?:python|py)?\s*(.*?)```",
        llm_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return llm_output.strip()


def _get_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "\n".join(part for part in text_parts if part)
    return str(content)


def _fallback_code() -> str:
    """
    Deterministic script used when no LLM credentials are configured.
    """

    return r'''
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_SOURCE = Path(os.environ["DATA_SOURCE"])
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette=["#7A8F87", "#C3A995", "#8EA4B8", "#B48E92", "#A7A37E"])

df = pd.read_csv(DATA_SOURCE)
original_shape = df.shape
df.columns = [str(col).strip().replace(" ", "_") for col in df.columns]

missing_summary = df.isna().sum().sort_values(ascending=False)
numeric_columns = df.select_dtypes(include="number").columns.tolist()
categorical_columns = [col for col in df.columns if col not in numeric_columns]
charts = []

if numeric_columns:
    plt.figure(figsize=(9, 5))
    corr = df[numeric_columns].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, linewidths=0.5)
    plt.title("Numeric Correlation Heatmap")
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "numeric_correlation_heatmap.png"
    plt.savefig(chart_path, dpi=160)
    plt.close()
    charts.append({"title": "Numeric Correlation Heatmap", "path": str(chart_path)})

    first_numeric = numeric_columns[0]
    plt.figure(figsize=(8, 5))
    sns.histplot(df[first_numeric].dropna(), kde=True, color="#7A8F87")
    plt.title(f"Distribution of {first_numeric}")
    plt.xlabel(first_numeric)
    plt.ylabel("Count")
    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"distribution_{first_numeric}.png"
    plt.savefig(chart_path, dpi=160)
    plt.close()
    charts.append({"title": f"Distribution of {first_numeric}", "path": str(chart_path)})

if categorical_columns:
    first_category = categorical_columns[0]
    counts = df[first_category].astype(str).value_counts().head(10)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=counts.values, y=counts.index, color="#8EA4B8")
    plt.title(f"Top Categories of {first_category}")
    plt.xlabel("Count")
    plt.ylabel(first_category)
    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"top_categories_{first_category}.png"
    plt.savefig(chart_path, dpi=160)
    plt.close()
    charts.append({"title": f"Top Categories of {first_category}", "path": str(chart_path)})

insights = [
    f"Loaded {original_shape[0]} rows and {original_shape[1]} columns from {DATA_SOURCE.name}.",
    f"Detected {len(numeric_columns)} numeric columns and {len(categorical_columns)} non-numeric columns.",
]
if int(missing_summary.sum()) > 0:
    top_missing = missing_summary[missing_summary > 0].head(3)
    insights.append("Top missing fields: " + ", ".join(f"{col}={int(val)}" for col, val in top_missing.items()))
else:
    insights.append("No missing values were detected in the loaded CSV.")

results = {
    "summary": "The workflow profiled the dataset, generated descriptive statistics, and rendered report-ready charts.",
    "insights": insights,
    "cleaning_notes": [
        "Column names were stripped and spaces were replaced with underscores for code safety.",
        "Rows were not dropped automatically; missingness is reported for review."
    ],
    "numeric_summary": df[numeric_columns].describe().round(4).to_dict() if numeric_columns else {},
    "charts": charts,
}

(OUTPUT_DIR / "analysis_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": "ok", "charts": len(charts)}, ensure_ascii=False))
'''.strip()


def run_coder(state: AgentState) -> dict[str, Any]:
    """
    Generate or repair Python analysis code based on the current AgentState.
    """

    if ChatOpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return {"generated_code": _fallback_code()}

    try:
        response = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0).invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=_build_user_prompt(state)),
            ]
        )
        return {"generated_code": _extract_python_code(_get_response_text(response))}
    except Exception as exc:
        raise RuntimeError(f"Coder agent failed to generate Python code: {exc}") from exc
