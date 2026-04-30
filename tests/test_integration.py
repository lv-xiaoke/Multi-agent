from pathlib import Path

from multi_agent_analyst.main import run_analysis


def test_full_workflow_generates_report_and_chart(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "region,revenue,cost\nEast,100,40\nWest,130,60\nEast,90,30\n",
        encoding="utf-8",
    )

    final_state = run_analysis(
        user_request="Analyze revenue and cost by region.",
        data_source=str(csv_path),
        output_dir=str(tmp_path / "outputs"),
        max_iterations=2,
    )

    assert Path(final_state["report_path"]).exists()
    assert final_state["charts"]
    assert final_state["execution_result"]["status"] == "success"
