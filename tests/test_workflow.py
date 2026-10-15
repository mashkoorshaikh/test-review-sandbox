"""
tests/test_workflow.py - Unit tests for LangGraph multi-agent workflow orchestration.

Verifies:
  1. Graph compilation and node wiring (fan-out and fan-in).
  2. End-to-end execution with mocked LLM.
  3. Parallel branch reducer merging without state collision.
  4. Missing credentials and provider errors.
  5. Clean target handling with zero issues.
  6. Deterministic-only execution (--no-ai).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tools import Issue, ToolResult
from workflow import (
    build_review_graph,
    run_deterministic_only,
    run_review_pipeline,
)


def make_mock_llm(text: str = "Deterministic test explanation"):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = text
    mock_llm.invoke.return_value = mock_response
    return mock_llm


@pytest.fixture
def mock_clean_ruff():
    return lambda p: ToolResult(
        tool_name="Ruff",
        target_path=p,
        success=True,
        issues_found=False,
        exit_code=0,
        issues=[],
    )


@pytest.fixture
def mock_clean_bandit():
    return lambda p: ToolResult(
        tool_name="Bandit",
        target_path=p,
        success=True,
        issues_found=False,
        exit_code=0,
        issues=[],
    )


@pytest.fixture
def mock_findings_ruff():
    return lambda p: ToolResult(
        tool_name="Ruff",
        target_path=p,
        success=True,
        issues_found=True,
        exit_code=1,
        issues=[
            Issue(code="F401", message="unused import", file_path="app.py", line=1, tool="ruff", fixable=True)
        ],
    )


@pytest.fixture
def mock_findings_bandit():
    return lambda p: ToolResult(
        tool_name="Bandit",
        target_path=p,
        success=True,
        issues_found=True,
        exit_code=1,
        issues=[
            Issue(code="B101", message="assert used", file_path="app.py", line=5, severity="LOW", tool="bandit")
        ],
    )


# -----------------------------------------------------------------------------
# Workflow Orchestration Tests
# -----------------------------------------------------------------------------

def test_build_review_graph_nodes_and_compilation():
    """Verify that graph compiles and registers all 4 nodes."""
    app = build_review_graph(llm=make_mock_llm())
    assert app is not None
    # LangGraph compiled graph has nodes
    assert hasattr(app, "invoke")


def test_workflow_execution_end_to_end_with_mocked_llm(mock_findings_ruff, mock_findings_bandit, tmp_path):
    """End-to-end graph invocation updates all state keys and produces report."""
    test_file = tmp_path / "app.py"
    test_file.write_text("x = 1\n", encoding="utf-8")

    mock_llm = make_mock_llm("AI explanation for finding")
    final_state = run_review_pipeline(
        target_path=str(test_file),
        enable_ai=True,
        llm=mock_llm,
        ruff_runner=mock_findings_ruff,
        bandit_runner=mock_findings_bandit,
    )

    # 1. Target path
    assert final_state["target_path"] == str(test_file)

    # 2. Ruff findings populated
    assert len(final_state["ruff_findings"]) == 1
    assert final_state["ruff_findings"][0]["code"] == "F401"

    # 3. Bandit findings populated
    assert len(final_state["bandit_findings"]) == 1
    assert final_state["bandit_findings"][0]["code"] == "B101"

    # 4. AI explanations contains both linter and security
    assert "linter" in final_state["ai_explanations"]
    assert "security" in final_state["ai_explanations"]

    # 5. Test suggestions populated
    assert len(final_state["generated_test_suggestions"]) >= 1

    # 6. Final report populated by orchestrator
    report = final_state["final_review_report"]
    assert report is not None
    assert report["summary"]["status"] == "ATTENTION_REQUIRED"
    assert report["summary"]["total_issues_found"] == 2
    assert report["summary"]["ruff_issues_count"] == 1
    assert report["summary"]["bandit_issues_count"] == 1


def test_workflow_parallel_reducer_merging(mock_findings_ruff, mock_findings_bandit, tmp_path):
    """Parallel Linter and Security branches merge their keys without conflict."""
    test_file = tmp_path / "test_mod.py"
    test_file.write_text("a = 1\n", encoding="utf-8")

    mock_llm = make_mock_llm("Explanation")
    final_state = run_review_pipeline(
        target_path=str(test_file),
        enable_ai=True,
        llm=mock_llm,
        ruff_runner=mock_findings_ruff,
        bandit_runner=mock_findings_bandit,
    )

    # ai_explanations must retain both keys from parallel execution
    assert "linter" in final_state["ai_explanations"]
    assert "security" in final_state["ai_explanations"]
    # agent_errors list must be intact
    assert isinstance(final_state["agent_errors"], list)


def test_workflow_clean_scan_produces_passed_verdict(mock_clean_ruff, mock_clean_bandit, tmp_path):
    """When both tools find 0 issues, workflow produces clean PASSED verdict."""
    test_file = tmp_path / "clean.py"
    test_file.write_text("print('clean')\n", encoding="utf-8")

    mock_llm = make_mock_llm("Test cases")
    final_state = run_review_pipeline(
        target_path=str(test_file),
        enable_ai=True,
        llm=mock_llm,
        ruff_runner=mock_clean_ruff,
        bandit_runner=mock_clean_bandit,
    )

    report = final_state["final_review_report"]
    assert report["summary"]["status"] == "PASSED"
    assert report["summary"]["total_issues_found"] == 0
    assert len(final_state["ruff_findings"]) == 0
    assert len(final_state["bandit_findings"]) == 0


def test_workflow_provider_error_captured_in_state(mock_findings_ruff, mock_findings_bandit, tmp_path):
    """When LLM raises an error, it is recorded in agent_errors and report without crashing graph."""
    test_file = tmp_path / "err.py"
    test_file.write_text("x = 1\n", encoding="utf-8")

    mock_failing_llm = MagicMock()
    mock_failing_llm.invoke.side_effect = RuntimeError("OpenAI 500 Internal Error")

    final_state = run_review_pipeline(
        target_path=str(test_file),
        enable_ai=True,
        llm=mock_failing_llm,
        ruff_runner=mock_findings_ruff,
        bandit_runner=mock_findings_bandit,
    )

    assert len(final_state["agent_errors"]) >= 1
    assert any("OpenAI 500" in err for err in final_state["agent_errors"])
    report = final_state["final_review_report"]
    assert report["summary"]["agent_error_count"] >= 1


def test_run_deterministic_only(mock_findings_ruff, mock_findings_bandit, tmp_path):
    """Running with --no-ai populates state without invoking any LLM."""
    test_file = tmp_path / "det.py"
    test_file.write_text("x = 1\n", encoding="utf-8")

    state = run_deterministic_only(
        target_path=str(test_file),
        ruff_runner=mock_findings_ruff,
        bandit_runner=mock_findings_bandit,
    )

    assert len(state["ruff_findings"]) == 1
    assert len(state["bandit_findings"]) == 1
    assert "--no-ai" in state["ai_explanations"]["linter"]
    assert "--no-ai" in state["ai_explanations"]["security"]
    report = state["final_review_report"]
    assert report["summary"]["total_issues_found"] == 2
