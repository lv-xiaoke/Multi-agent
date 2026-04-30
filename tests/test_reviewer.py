from multi_agent_analyst.agents.reviewer import route_after_execution, route_after_review


def test_route_after_execution_retries_failed_run_under_limit() -> None:
    state = {"execution_result": {"status": "failed"}, "iterations": 1, "max_iterations": 3}

    assert route_after_execution(state) == "Coder"


def test_route_after_execution_finishes_at_limit() -> None:
    state = {"execution_result": {"status": "failed"}, "iterations": 3, "max_iterations": 3}

    assert route_after_execution(state) == "FinalReport"


def test_route_after_review_sends_success_to_reporter() -> None:
    state = {
        "execution_result": {"status": "success"},
        "review_feedback": "Review passed: ok",
        "iterations": 1,
        "max_iterations": 3,
    }

    assert route_after_review(state) == "reporter"
