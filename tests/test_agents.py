"""
tests/test_agents.py - Unit tests for Linter, Security, and Test-Writer agents.

Uses deterministic mock LLM responses and mock static analysis runners.
Ensures no real API calls are made during tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents import (
    LINTER_DISCLAIMER,
    SECURITY_DISCLAIMER,
    TEST_WRITER_DISCLAIMER,
    LinterAgent,
    SecurityAgent,
    TestWriterAgent,
)
from llm_config import MissingApiKeyError
from tools import Issue, ToolResult

# -----------------------------------------------------------------------------
# Reusable Helpers & Fixtures
# -----------------------------------------------------------------------------

def make_mock_llm(response_text: str = "Mocked LLM diagnostic analysis"):
    """Create a mock BaseChatModel returning a deterministic AIMessage response."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm.invoke.return_value = mock_response
    return mock_llm


@pytest.fixture
def sample_python_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    p = tmp_path / "calculator.py"
    p.write_text(
        "import os\n"
        "def add(a: int, b: int) -> int:\n"
        "    assert a >= 0\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    return str(p)


# -----------------------------------------------------------------------------
# 1. Linter Agent Tests
# -----------------------------------------------------------------------------

class TestLinterAgent:
    """Test suite for LinterAgent."""

    def test_linter_agent_success_with_findings(self, sample_python_file):
        """Linter agent calls LLM with snippets and includes advisory disclaimer."""
        mock_llm = make_mock_llm("F401 means the import was not used. You can safely remove it.")
        mock_runner = MagicMock(return_value=ToolResult(
            tool_name="Ruff",
            target_path=sample_python_file,
            success=True,
            issues_found=True,
            exit_code=1,
            issues=[
                Issue(
                    code="F401",
                    message="`os` imported but unused",
                    file_path=sample_python_file,
                    line=1,
                    column=8,
                    severity="ERROR",
                    tool="ruff",
                    fixable=True,
                )
            ],
        ))

        agent = LinterAgent(llm=mock_llm, ruff_runner=mock_runner)
        result = agent.run(sample_python_file)

        assert len(result["ruff_findings"]) == 1
        assert result["ruff_findings"][0]["code"] == "F401"
        assert result["agent_errors"] == []
        explanation = result["ai_explanations"]["linter"]
        assert "F401 means the import was not used" in explanation
        assert LINTER_DISCLAIMER in explanation

        # Verify LLM was invoked with safe prompt
        mock_llm.invoke.assert_called_once()
        call_messages = mock_llm.invoke.call_args[0][0]
        # Check that prompt contains untrusted boundary marker
        assert any("UNTRUSTED DATA" in m.content for m in call_messages)

    def test_linter_agent_clean_no_findings(self, sample_python_file):
        """When Ruff finds 0 issues, agent reports clean result with disclaimer without calling LLM."""
        mock_llm = make_mock_llm()
        mock_runner = MagicMock(return_value=ToolResult(
            tool_name="Ruff",
            target_path=sample_python_file,
            success=True,
            issues_found=False,
            exit_code=0,
            issues=[],
        ))

        agent = LinterAgent(llm=mock_llm, ruff_runner=mock_runner)
        result = agent.run(sample_python_file)

        assert len(result["ruff_findings"]) == 0
        assert result["agent_errors"] == []
        assert "zero code style or linting issues" in result["ai_explanations"]["linter"]
        assert LINTER_DISCLAIMER in result["ai_explanations"]["linter"]
        mock_llm.invoke.assert_not_called()

    def test_linter_agent_tool_failure(self, sample_python_file):
        """When Ruff fails to execute, agent records tool error."""
        mock_runner = MagicMock(return_value=ToolResult(
            tool_name="Ruff",
            target_path=sample_python_file,
            success=False,
            issues_found=False,
            exit_code=2,
            error_message="Ruff binary crashed",
        ))

        agent = LinterAgent(llm=make_mock_llm(), ruff_runner=mock_runner)
        result = agent.run(sample_python_file)

        assert len(result["agent_errors"]) == 1
        assert "Ruff binary crashed" in result["agent_errors"][0]
        assert "unsuccessful" in result["ai_explanations"]["linter"]

    def test_linter_agent_missing_api_key(self, sample_python_file):
        """Missing API credentials must be handled gracefully without crashing."""
        mock_runner = MagicMock(return_value=ToolResult(
            tool_name="Ruff",
            target_path=sample_python_file,
            success=True,
            issues_found=True,
            exit_code=1,
            issues=[
                Issue(code="F401", message="unused import", file_path=sample_python_file, line=1, tool="ruff")
            ],
        ))

        # Agent with no LLM provided and get_llm raising MissingApiKeyError
        agent = LinterAgent(llm=None, ruff_runner=mock_runner)
        with patch("agents.get_llm", side_effect=MissingApiKeyError("API key not set")):
            result = agent.run(sample_python_file)

        assert len(result["ruff_findings"]) == 1
        assert len(result["agent_errors"]) == 1
        assert "API key not set" in result["agent_errors"][0]
        assert "Unavailable" in result["ai_explanations"]["linter"]
        assert LINTER_DISCLAIMER in result["ai_explanations"]["linter"]

    def test_linter_agent_provider_error(self, sample_python_file):
        """Provider runtime exceptions (e.g. rate limit, connection failure) are caught."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("503 Service Unavailable")
        mock_runner = MagicMock(return_value=ToolResult(
            tool_name="Ruff",
            target_path=sample_python_file,
            success=True,
            issues_found=True,
            exit_code=1,
            issues=[
                Issue(code="F401", message="unused import", file_path=sample_python_file, line=1, tool="ruff")
            ],
        ))

        agent = LinterAgent(llm=mock_llm, ruff_runner=mock_runner)
        result = agent.run(sample_python_file)

        assert len(result["agent_errors"]) == 1
        assert "503 Service Unavailable" in result["agent_errors"][0]
        assert "Failed" in result["ai_explanations"]["linter"]

    def test_linter_agent_empty_model_output(self, sample_python_file):
        """Empty or whitespace responses from LLM are handled safely."""
        mock_llm = make_mock_llm("")
        mock_runner = MagicMock(return_value=ToolResult(
            tool_name="Ruff",
            target_path=sample_python_file,
            success=True,
            issues_found=True,
            exit_code=1,
            issues=[
                Issue(code="F401", message="unused import", file_path=sample_python_file, line=1, tool="ruff")
            ],
        ))

        agent = LinterAgent(llm=mock_llm, ruff_runner=mock_runner)
        result = agent.run(sample_python_file)

        assert "empty explanation" in result["ai_explanations"]["linter"]


# -----------------------------------------------------------------------------
# 2. Security Agent Tests
# -----------------------------------------------------------------------------

class TestSecurityAgent:
    """Test suite for SecurityAgent."""

    def test_security_agent_success_with_findings(self, sample_python_file):
        """Security agent distinguishes tool findings from hypotheses and attaches disclaimer."""
        mock_llm = make_mock_llm(
            "[CONFIRMED TOOL FINDING: Bandit] B101 assert used\n"
            "[AI SECURITY HYPOTHESIS & REMEDIATION] Asserts are stripped in optimized Python (-O)."
        )
        mock_runner = MagicMock(return_value=ToolResult(
            tool_name="Bandit",
            target_path=sample_python_file,
            success=True,
            issues_found=True,
            exit_code=1,
            issues=[
                Issue(
                    code="B101",
                    message="Use of assert detected",
                    file_path=sample_python_file,
                    line=3,
                    severity="LOW",
                    tool="bandit",
                )
            ],
        ))

        agent = SecurityAgent(llm=mock_llm, bandit_runner=mock_runner)
        result = agent.run(sample_python_file)

        assert len(result["bandit_findings"]) == 1
        assert result["bandit_findings"][0]["code"] == "B101"
        assert result["agent_errors"] == []
        explanation = result["ai_explanations"]["security"]
        assert "B101 assert used" in explanation
        assert SECURITY_DISCLAIMER in explanation

    def test_security_agent_clean_no_findings_does_not_claim_absolute_security(self, sample_python_file):
        """Clean Bandit result must explicitly state that absence of findings != secure."""
        mock_llm = make_mock_llm()
        mock_runner = MagicMock(return_value=ToolResult(
            tool_name="Bandit",
            target_path=sample_python_file,
            success=True,
            issues_found=False,
            exit_code=0,
            issues=[],
        ))

        agent = SecurityAgent(llm=mock_llm, bandit_runner=mock_runner)
        result = agent.run(sample_python_file)

        assert len(result["bandit_findings"]) == 0
        explanation = result["ai_explanations"]["security"]
        assert "zero AST-level security violations" in explanation
        # MUST state absence does NOT guarantee security
        assert "NOT guarantee that the application is secure" in explanation
        assert SECURITY_DISCLAIMER in explanation
        mock_llm.invoke.assert_not_called()

    def test_security_agent_tool_failure(self, sample_python_file):
        """Tool failure records error in state."""
        mock_runner = MagicMock(return_value=ToolResult(
            tool_name="Bandit",
            target_path=sample_python_file,
            success=False,
            issues_found=False,
            exit_code=2,
            error_message="Bandit crashed on syntax error",
        ))

        agent = SecurityAgent(llm=make_mock_llm(), bandit_runner=mock_runner)
        result = agent.run(sample_python_file)

        assert len(result["agent_errors"]) == 1
        assert "Bandit crashed" in result["agent_errors"][0]
        assert "failed" in result["ai_explanations"]["security"]

    def test_security_agent_missing_api_key(self, sample_python_file):
        """Missing API key handled gracefully with clear notice."""
        mock_runner = MagicMock(return_value=ToolResult(
            tool_name="Bandit",
            target_path=sample_python_file,
            success=True,
            issues_found=True,
            exit_code=1,
            issues=[
                Issue(code="B101", message="assert used", file_path=sample_python_file, line=3, tool="bandit")
            ],
        ))

        agent = SecurityAgent(llm=None, bandit_runner=mock_runner)
        with patch("agents.get_llm", side_effect=MissingApiKeyError("Missing credentials")):
            result = agent.run(sample_python_file)

        assert len(result["bandit_findings"]) == 1
        assert len(result["agent_errors"]) == 1
        assert "Unavailable" in result["ai_explanations"]["security"]
        assert SECURITY_DISCLAIMER in result["ai_explanations"]["security"]

    def test_security_agent_provider_error(self, sample_python_file):
        """Provider exception handled without raising."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = TimeoutError("Request timed out")
        mock_runner = MagicMock(return_value=ToolResult(
            tool_name="Bandit",
            target_path=sample_python_file,
            success=True,
            issues_found=True,
            exit_code=1,
            issues=[
                Issue(code="B101", message="assert used", file_path=sample_python_file, line=3, tool="bandit")
            ],
        ))

        agent = SecurityAgent(llm=mock_llm, bandit_runner=mock_runner)
        result = agent.run(sample_python_file)

        assert len(result["agent_errors"]) == 1
        assert "TimeoutError" in result["agent_errors"][0]
        assert "Failed" in result["ai_explanations"]["security"]


# -----------------------------------------------------------------------------
# 3. Test-Writer Agent Tests
# -----------------------------------------------------------------------------

class TestTestWriterAgent:
    """Test suite for TestWriterAgent."""

    def test_test_writer_agent_suggests_tests_and_stores_them(self, sample_python_file, tmp_path):
        """Generated tests include disclaimer and are stored in temp workspace."""
        suggested_code = (
            "def test_add_positive():\n"
            "    assert add(2, 3) == 5\n\n"
            "def test_add_edge_zero():\n"
            "    assert add(0, 0) == 0\n"
        )
        mock_llm = make_mock_llm(suggested_code)
        workspace = str(tmp_path / "test_ws")

        agent = TestWriterAgent(llm=mock_llm, test_workspace_dir=workspace, store_tests=True)
        result = agent.run(sample_python_file)

        assert len(result["generated_test_suggestions"]) >= 1
        suggestion = result["generated_test_suggestions"][0]
        assert "test_add_positive" in suggestion["suggestions"]
        assert "test_add_edge_zero" in suggestion["suggestions"]
        assert TEST_WRITER_DISCLAIMER in suggestion["suggestions"]
        # M3: stored_test_files key must exist
        assert "stored_test_files" in result
        # At least one file should be stored
        assert len(result["stored_test_files"]) >= 1
        # No errors
        assert result["agent_errors"] == []

    def test_test_writer_agent_store_tests_disabled(self, sample_python_file, tmp_path):
        """When store_tests=False, no files are stored."""
        suggested_code = (
            "def test_add_positive():\n"
            "    assert add(2, 3) == 5\n"
        )
        mock_llm = make_mock_llm(suggested_code)

        agent = TestWriterAgent(llm=mock_llm, store_tests=False)
        result = agent.run(sample_python_file)

        assert "stored_test_files" in result
        assert result["stored_test_files"] == []

    def test_test_writer_agent_no_files_found(self, tmp_path):
        """Handles directory with no Python source files."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()

        agent = TestWriterAgent(llm=make_mock_llm())
        result = agent.run(str(empty_dir))

        assert len(result["generated_test_suggestions"]) == 1
        assert "No reviewable Python source files found" in result["generated_test_suggestions"][0]["suggestions"]
        assert "stored_test_files" in result

    def test_test_writer_agent_missing_api_key(self, sample_python_file):
        """Gracefully degrades when API key is missing."""
        agent = TestWriterAgent(llm=None, store_tests=False)
        with patch("agents.get_llm", side_effect=MissingApiKeyError("No API key")):
            result = agent.run(sample_python_file)

        assert len(result["agent_errors"]) == 1
        assert "Unavailable" in result["generated_test_suggestions"][0]["suggestions"]
        assert TEST_WRITER_DISCLAIMER in result["generated_test_suggestions"][0]["suggestions"]

    def test_test_writer_agent_provider_error(self, sample_python_file):
        """Catches provider timeout or connection failure."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API rate limit exceeded")

        agent = TestWriterAgent(llm=mock_llm, store_tests=False)
        result = agent.run(sample_python_file)

        assert len(result["agent_errors"]) == 1
        assert "Failed" in result["generated_test_suggestions"][0]["suggestions"]

    def test_test_writer_agent_validation_failure_recorded_in_errors(self, sample_python_file, tmp_path, monkeypatch):
        """When test validation fails, error is recorded but agent does not crash."""
        monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
        # LLM returns code that would fail safety validation (contains eval)
        unsafe_code = "def test_x():\n    eval('rm -rf /')\n"
        mock_llm = make_mock_llm(unsafe_code)
        workspace = str(tmp_path / "test_ws")

        agent = TestWriterAgent(llm=mock_llm, test_workspace_dir=workspace, store_tests=True)
        result = agent.run(sample_python_file)

        # Agent should not crash; validation error captured in agent_errors
        assert isinstance(result.get("agent_errors"), list)
        # stored_test_files should be empty since validation rejected it
        assert result["stored_test_files"] == []
