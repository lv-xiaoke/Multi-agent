"""
setup_project.py

Run this script to scaffold a LangGraph-based multi-agent data analyst project.
"""

from pathlib import Path


PROJECT_NAME = "multi_agent_analyst"


REQUIREMENTS = """langgraph
langchain
langchain-core
pydantic
openai
python-dotenv
pandas
numpy
scikit-learn
"""


PYTHON_FILES = {
    "main.py": "Main entry point for the multi-agent analyst application.",
    "core/__init__.py": "Core package initialization.",
    "core/state.py": "Defines the global AgentState used by the LangGraph workflow.",
    "core/graph_builder.py": "Builds and compiles the LangGraph StateGraph workflow.",
    "agents/__init__.py": "Agents package initialization.",
    "agents/planner.py": "Implements the Data Planner agent logic.",
    "agents/coder.py": "Implements the Coder agent logic.",
    "agents/reviewer.py": "Implements routing and review logic after code execution.",
    "tools/__init__.py": "Tools package initialization.",
    "tools/executor.py": "Provides sandboxed code execution utilities.",
    "config/__init__.py": "Config package initialization.",
    "config/prompts.py": "Stores system prompts and prompt templates for agents.",
}


def write_file_if_missing(file_path: Path, content: str) -> None:
    """
    Create a file with content if it does not already exist.
    """

    if file_path.exists():
        print(f"[SKIP] File already exists: {file_path}")
        return

    file_path.write_text(content, encoding="utf-8")
    print(f"[CREATE] File created: {file_path}")


def create_project_structure() -> None:
    """
    Create the multi_agent_analyst project directory tree.
    """

    root_dir = Path.cwd() / PROJECT_NAME

    try:
        if root_dir.exists():
            print(f"[INFO] Project directory already exists: {root_dir}")
        else:
            root_dir.mkdir(parents=True, exist_ok=True)
            print(f"[CREATE] Project directory created: {root_dir}")

        for relative_file, description in PYTHON_FILES.items():
            file_path = root_dir / relative_file
            file_path.parent.mkdir(parents=True, exist_ok=True)

            content = f'"""\n{description}\n"""\n'
            write_file_if_missing(file_path, content)

        requirements_path = root_dir / "requirements.txt"
        write_file_if_missing(requirements_path, REQUIREMENTS)

        print("\n[DONE] Project scaffold generated successfully.")
        print(f"[PATH] {root_dir}")

    except PermissionError as exc:
        print(f"[ERROR] Permission denied while creating project files: {exc}")
    except OSError as exc:
        print(f"[ERROR] OS error while creating project files: {exc}")
    except Exception as exc:
        print(f"[ERROR] Unexpected error: {exc}")


if __name__ == "__main__":
    create_project_structure()
