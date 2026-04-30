from pathlib import Path

from multi_agent_analyst.tools.data_io import profile_csv


def test_profile_csv_reports_shape_columns_and_missing_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("region,revenue,cost\nEast,10,4\nWest,,5\n", encoding="utf-8")

    profile = profile_csv(str(csv_path))

    assert profile["shape"] == {"rows": 2, "columns": 3}
    assert [column["name"] for column in profile["columns"]] == ["region", "revenue", "cost"]
    revenue = next(column for column in profile["columns"] if column["name"] == "revenue")
    assert revenue["missing_count"] == 1
    assert revenue["missing_rate"] == 0.5
