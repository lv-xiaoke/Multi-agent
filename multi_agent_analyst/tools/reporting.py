"""
HTML report rendering for completed analysis runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Template

from multi_agent_analyst.core.state import AgentState


REPORT_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi-Agent Analysis Report</title>
  <style>
    :root { color-scheme: light; --ink: #22252a; --muted: #6a717c; --line: #d8dde3; --accent: #4e7a78; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: #f7f8f7; }
    main { max-width: 1080px; margin: 0 auto; padding: 40px 24px 56px; }
    header { border-bottom: 1px solid var(--line); padding-bottom: 20px; margin-bottom: 28px; }
    h1 { margin: 0 0 10px; font-size: 32px; line-height: 1.2; }
    h2 { margin: 30px 0 12px; font-size: 20px; }
    p, li { line-height: 1.7; }
    .meta { color: var(--muted); font-size: 14px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .card { background: white; border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
    .metric { font-size: 26px; font-weight: 700; color: var(--accent); }
    table { border-collapse: collapse; width: 100%; background: white; border: 1px solid var(--line); }
    th, td { border-bottom: 1px solid var(--line); padding: 10px 12px; text-align: left; vertical-align: top; }
    th { background: #eef2f1; }
    pre { white-space: pre-wrap; background: #202327; color: #f5f7f9; padding: 16px; border-radius: 8px; overflow-x: auto; }
    img { max-width: 100%; border: 1px solid var(--line); border-radius: 8px; background: white; }
    .chart { margin: 18px 0 26px; }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>多智能体数据分析报告</h1>
      <div class="meta">Run ID: {{ run_id }} | Data: {{ data_profile.file_name }}</div>
    </header>

    <h2>任务目标</h2>
    <p>{{ user_request }}</p>

    <section class="grid">
      <div class="card"><div class="metric">{{ data_profile.shape.rows }}</div><div>Rows</div></div>
      <div class="card"><div class="metric">{{ data_profile.shape.columns }}</div><div>Columns</div></div>
      <div class="card"><div class="metric">{{ charts|length }}</div><div>Charts</div></div>
    </section>

    <h2>数据字段画像</h2>
    <table>
      <thead><tr><th>Field</th><th>Type</th><th>Missing</th><th>Unique</th><th>Sample</th></tr></thead>
      <tbody>
      {% for column in data_profile.columns %}
        <tr>
          <td>{{ column.name }}</td>
          <td>{{ column.dtype }}</td>
          <td>{{ column.missing_count }} ({{ "%.1f"|format(column.missing_rate * 100) }}%)</td>
          <td>{{ column.unique_count }}</td>
          <td>{{ column.sample_values|join(", ") }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>

    <h2>分析计划</h2>
    <pre>{{ plan }}</pre>

    <h2>执行摘要</h2>
    <p>{{ result_summary }}</p>

    {% if result_insights %}
    <h2>关键洞察</h2>
    <ul>
      {% for item in result_insights %}
      <li>{{ item }}</li>
      {% endfor %}
    </ul>
    {% endif %}

    {% if charts %}
    <h2>可视化图表</h2>
    {% for chart in charts %}
    <div class="chart">
      <h3>{{ chart.title }}</h3>
      <img src="{{ chart.relative_path }}" alt="{{ chart.title }}">
    </div>
    {% endfor %}
    {% endif %}

    <h2>评审结果</h2>
    <p>{{ review_feedback }}</p>
  </main>
</body>
</html>
""".strip()


def _load_analysis_results(output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "analysis_results.json"
    if not result_path.exists():
        return {"summary": "分析脚本未生成结构化结果文件。", "insights": []}
    return json.loads(result_path.read_text(encoding="utf-8"))


def render_report(state: AgentState) -> str:
    output_dir = Path(state["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_results = _load_analysis_results(output_dir)

    charts = []
    for index, chart in enumerate(state.get("charts", []), start=1):
        path = Path(chart["path"]).resolve()
        try:
            relative_path = path.relative_to(output_dir).as_posix()
        except ValueError:
            relative_path = path.as_uri()
        charts.append(
            {
                "title": chart.get("title") or f"Chart {index}",
                "relative_path": relative_path,
            }
        )

    html = Template(REPORT_TEMPLATE).render(
        run_id=state.get("run_id", ""),
        user_request=state.get("user_request", ""),
        data_profile=state.get("data_profile", {}),
        plan=state.get("plan", ""),
        charts=charts,
        result_summary=analysis_results.get("summary", ""),
        result_insights=analysis_results.get("insights", []),
        review_feedback=state.get("review_feedback", ""),
    )

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return str(report_path)


def run_reporter(state: AgentState) -> dict[str, Any]:
    return {"report_path": render_report(state)}
