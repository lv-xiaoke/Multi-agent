from pathlib import Path

from multi_agent_analyst.tools.executor import execute_python_code


def test_execute_python_code_success(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x\n1\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    result = execute_python_code(
        "from pathlib import Path\nimport os\nPath(os.environ['OUTPUT_DIR'], 'done.txt').write_text('ok')",
        str(csv_path),
        str(output_dir),
    )

    assert result["status"] == "success"
    assert str((output_dir / "done.txt").resolve()) in result["artifacts"]


def test_execute_python_code_failure(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x\n1\n", encoding="utf-8")

    result = execute_python_code("raise RuntimeError('boom')", str(csv_path), str(tmp_path / "out"))

    assert result["status"] == "failed"
    assert "boom" in result["stderr"]


def test_execute_python_code_timeout(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x\n1\n", encoding="utf-8")

    result = execute_python_code("import time\ntime.sleep(2)", str(csv_path), str(tmp_path / "out"), timeout_seconds=1)

    assert result["status"] == "timeout"
