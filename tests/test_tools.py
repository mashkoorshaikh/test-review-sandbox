"""
Unit tests for tools.py and pipeline execution.

Uses mocked subprocess calls to test:
  1. Successful execution without findings (exit code 0).
  2. Successful execution with detected findings (exit code 1).
  3. Tool execution errors (exit code >= 2, tool crash).
  4. Missing tool / FileNotFoundError.
  5. Subprocess timeout scenarios.
  6. Argument generation (shell=False, safe argument lists).
  7. Pipeline orchestration and exit codes in main.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import run_pipeline
from tools import (
    Issue,
    ToolResult,
    parse_bandit_json,
    parse_ruff_json,
    resolve_tool_command,
    run_bandit,
    run_ruff,
)

# -----------------------------------------------------------------------------
# Fixtures & Sample Mock Data
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_file_path(tmp_path):
    """Create a temporary real file for path validation tests."""
    test_file = tmp_path / "dummy.py"
    test_file.write_text("x = 1\n", encoding="utf-8")
    return str(test_file)


@pytest.fixture
def sample_dir_path(tmp_path):
    """Create a temporary real directory for path validation tests."""
    sub_dir = tmp_path / "subpkg"
    sub_dir.mkdir()
    (sub_dir / "dummy.py").write_text("x = 1\n", encoding="utf-8")
    return str(sub_dir)


RUFF_CLEAN_JSON = "[]"

RUFF_FINDINGS_JSON = json.dumps([
    {
        "code": "F401",
        "message": "`os` imported but unused",
        "filename": "sample_project/app.py",
        "location": {"row": 10, "column": 8},
        "end_location": {"row": 10, "column": 10},
        "severity": "error",
        "fix": {"message": "Remove unused import: `os`"},
    },
    {
        "code": "F841",
        "message": "Local variable `unused_var` is assigned to but never used",
        "filename": "sample_project/app.py",
        "location": {"row": 18, "column": 5},
        "end_location": {"row": 18, "column": 15},
        "severity": "error",
        "fix": None,
    },
])

BANDIT_CLEAN_JSON = json.dumps({
    "errors": [],
    "results": [],
})

BANDIT_FINDINGS_JSON = json.dumps({
    "errors": [],
    "results": [
        {
            "test_id": "B101",
            "test_name": "assert_used",
            "issue_text": "Use of assert detected.",
            "issue_severity": "LOW",
            "filename": "sample_project/app.py",
            "line_number": 22,
            "col_offset": 4,
        },
        {
            "test_id": "B105",
            "test_name": "hardcoded_password_string",
            "issue_text": "Possible hardcoded password.",
            "issue_severity": "MEDIUM",
            "filename": "sample_project/app.py",
            "line_number": 37,
            "col_offset": 18,
        },
    ],
})


# -----------------------------------------------------------------------------
# Tests for Ruff Execution
# -----------------------------------------------------------------------------

class TestRunRuff:
    """Test suite for run_ruff."""

    @patch("subprocess.run")
    def test_run_ruff_success_no_issues(self, mock_subprocess, sample_file_path):
        """Exit code 0 with empty JSON indicates successful run with no issues."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout=RUFF_CLEAN_JSON,
            stderr="",
        )

        result = run_ruff(sample_file_path)

        assert result.success is True
        assert result.issues_found is False
        assert result.exit_code == 0
        assert len(result.issues) == 0
        assert result.error_message is None

        # Verify subprocess was called safely with shell=False
        mock_subprocess.assert_called_once()
        _, kwargs = mock_subprocess.call_args
        assert kwargs["shell"] is False

    @patch("subprocess.run")
    def test_run_ruff_detected_findings(self, mock_subprocess, sample_file_path):
        """Exit code 1 with JSON array indicates code violations found (not tool failure)."""
        mock_subprocess.return_value = MagicMock(
            returncode=1,
            stdout=RUFF_FINDINGS_JSON,
            stderr="",
        )

        result = run_ruff(sample_file_path)

        assert result.success is True
        assert result.issues_found is True
        assert result.exit_code == 1
        assert len(result.issues) == 2

        # Check first parsed issue
        first_issue = result.issues[0]
        assert first_issue.code == "F401"
        assert first_issue.line == 10
        assert first_issue.column == 8
        assert first_issue.tool == "ruff"
        assert first_issue.fixable is True

        # Check second parsed issue
        second_issue = result.issues[1]
        assert second_issue.code == "F841"
        assert second_issue.line == 18
        assert second_issue.fixable is False

    @patch("subprocess.run")
    def test_run_ruff_execution_error_exit_code_2(self, mock_subprocess, sample_file_path):
        """Exit code 2 indicates configuration or syntax crash of Ruff itself."""
        mock_subprocess.return_value = MagicMock(
            returncode=2,
            stdout="",
            stderr="error: Failed to parse configuration file",
        )

        result = run_ruff(sample_file_path)

        assert result.success is False
        assert result.issues_found is False
        assert result.exit_code == 2
        assert "Failed to parse configuration file" in (result.error_message or "")

    @patch("subprocess.run")
    def test_run_ruff_file_not_found(self, mock_subprocess, sample_file_path):
        """When the ruff binary is missing, FileNotFoundError must be handled cleanly."""
        mock_subprocess.side_effect = FileNotFoundError("No such file or directory: 'ruff'")

        result = run_ruff(sample_file_path)

        assert result.success is False
        assert result.issues_found is False
        assert result.exit_code is None
        assert "Ruff executable not found" in (result.error_message or "")

    @patch("subprocess.run")
    def test_run_ruff_timeout(self, mock_subprocess, sample_file_path):
        """When subprocess times out, mark as execution failure."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired(cmd=["ruff"], timeout=10)

        result = run_ruff(sample_file_path, timeout=10)

        assert result.success is False
        assert result.exit_code is None
        assert "timed out" in (result.error_message or "")

    def test_run_ruff_nonexistent_target_path(self):
        """Nonexistent target path returns error without calling subprocess."""
        result = run_ruff("nonexistent/path/that/does/not/exist.py")

        assert result.success is False
        assert result.issues_found is False
        assert "Target path does not exist" in (result.error_message or "")

    @patch("subprocess.run")
    def test_run_ruff_malformed_json_fallback(self, mock_subprocess, sample_file_path):
        """If exit code is 1 but stdout is malformed, handle gracefully."""
        mock_subprocess.return_value = MagicMock(
            returncode=1,
            stdout="Invalid non-json output",
            stderr="",
        )

        result = run_ruff(sample_file_path)

        assert result.success is False
        assert "Failed to parse Ruff output" in (result.error_message or "")


# -----------------------------------------------------------------------------
# Tests for Bandit Execution
# -----------------------------------------------------------------------------

class TestRunBandit:
    """Test suite for run_bandit."""

    @patch("subprocess.run")
    def test_run_bandit_success_no_issues(self, mock_subprocess, sample_file_path):
        """Exit code 0 indicates no security vulnerabilities found."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout=BANDIT_CLEAN_JSON,
            stderr="",
        )

        result = run_bandit(sample_file_path)

        assert result.success is True
        assert result.issues_found is False
        assert result.exit_code == 0
        assert len(result.issues) == 0

        # Verify shell=False
        mock_subprocess.assert_called_once()
        _, kwargs = mock_subprocess.call_args
        assert kwargs["shell"] is False

    @patch("subprocess.run")
    def test_run_bandit_detected_findings(self, mock_subprocess, sample_file_path):
        """Exit code 1 indicates security issues found."""
        mock_subprocess.return_value = MagicMock(
            returncode=1,
            stdout=BANDIT_FINDINGS_JSON,
            stderr="",
        )

        result = run_bandit(sample_file_path)

        assert result.success is True
        assert result.issues_found is True
        assert result.exit_code == 1
        assert len(result.issues) == 2

        assert result.issues[0].code == "B101"
        assert result.issues[0].severity == "LOW"
        assert result.issues[0].line == 22
        assert result.issues[0].tool == "bandit"

        assert result.issues[1].code == "B105"
        assert result.issues[1].severity == "MEDIUM"

    @patch("subprocess.run")
    def test_run_bandit_execution_error_exit_code_2(self, mock_subprocess, sample_file_path):
        """Exit code 2 indicates Bandit runtime or syntax failure."""
        mock_subprocess.return_value = MagicMock(
            returncode=2,
            stdout="",
            stderr="Bandit encountered a syntax error in target file",
        )

        result = run_bandit(sample_file_path)

        assert result.success is False
        assert result.issues_found is False
        assert result.exit_code == 2
        assert "syntax error" in (result.error_message or "")

    @patch("subprocess.run")
    def test_run_bandit_file_not_found(self, mock_subprocess, sample_file_path):
        """Missing bandit binary handled cleanly."""
        mock_subprocess.side_effect = FileNotFoundError("bandit not found")

        result = run_bandit(sample_file_path)

        assert result.success is False
        assert result.issues_found is False
        assert result.exit_code is None
        assert "Bandit executable not found" in (result.error_message or "")

    @patch("subprocess.run")
    def test_run_bandit_timeout(self, mock_subprocess, sample_file_path):
        """Timeout handled cleanly."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired(cmd=["bandit"], timeout=5)

        result = run_bandit(sample_file_path, timeout=5)

        assert result.success is False
        assert "timed out" in (result.error_message or "")

    def test_run_bandit_nonexistent_target_path(self):
        """Nonexistent target returns error."""
        result = run_bandit("nonexistent/folder")

        assert result.success is False
        assert "Target path does not exist" in (result.error_message or "")

    @patch("subprocess.run")
    def test_run_bandit_recursive_flag_for_directory(self, mock_subprocess, sample_dir_path):
        """Directory targets must include recursive flag '-r'."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout=BANDIT_CLEAN_JSON,
            stderr="",
        )

        run_bandit(sample_dir_path)

        cmd_called = mock_subprocess.call_args[0][0]
        assert "-r" in cmd_called
        assert sample_dir_path in cmd_called

    @patch("subprocess.run")
    def test_run_bandit_no_recursive_flag_for_file(self, mock_subprocess, sample_file_path):
        """Single file targets should not use recursive flag '-r'."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout=BANDIT_CLEAN_JSON,
            stderr="",
        )

        run_bandit(sample_file_path)

        cmd_called = mock_subprocess.call_args[0][0]
        assert "-r" not in cmd_called
        assert sample_file_path in cmd_called


# -----------------------------------------------------------------------------
# Tests for Parser Helpers & Command Resolution
# -----------------------------------------------------------------------------

def test_parse_ruff_json_handles_empty_or_whitespace():
    assert parse_ruff_json("") == []
    assert parse_ruff_json("   \n") == []


def test_parse_bandit_json_with_preceding_logs():
    """Bandit can prefix logs like [main] INFO ... before the JSON payload."""
    payload = f"[main] INFO logging\n{BANDIT_FINDINGS_JSON}"
    issues = parse_bandit_json(payload)
    assert len(issues) == 2
    assert issues[0].code == "B101"


def test_resolve_tool_command_returns_non_empty_list():
    cmd = resolve_tool_command("ruff")
    assert isinstance(cmd, list)
    assert len(cmd) >= 1


# -----------------------------------------------------------------------------
# Tests for main.py run_pipeline Orchestration
# -----------------------------------------------------------------------------

class TestPipelineOrchestration:
    """Test suite for main.py run_pipeline."""

    @patch("main.run_bandit")
    @patch("main.run_ruff")
    def test_pipeline_clean_returns_zero(self, mock_ruff, mock_bandit, sample_file_path):
        mock_ruff.return_value = ToolResult(
            tool_name="Ruff",
            target_path=sample_file_path,
            success=True,
            issues_found=False,
            exit_code=0,
            issues=[],
        )
        mock_bandit.return_value = ToolResult(
            tool_name="Bandit",
            target_path=sample_file_path,
            success=True,
            issues_found=False,
            exit_code=0,
            issues=[],
        )

        exit_code = run_pipeline(sample_file_path)
        assert exit_code == 0

    @patch("main.run_bandit")
    @patch("main.run_ruff")
    def test_pipeline_with_code_issues_returns_one(self, mock_ruff, mock_bandit, sample_file_path):
        mock_ruff.return_value = ToolResult(
            tool_name="Ruff",
            target_path=sample_file_path,
            success=True,
            issues_found=True,
            exit_code=1,
            issues=[
                Issue(
                    code="F401",
                    message="unused import",
                    file_path="app.py",
                    line=1,
                    tool="ruff",
                )
            ],
        )
        mock_bandit.return_value = ToolResult(
            tool_name="Bandit",
            target_path=sample_file_path,
            success=True,
            issues_found=False,
            exit_code=0,
            issues=[],
        )

        exit_code = run_pipeline(sample_file_path)
        assert exit_code == 1

    @patch("main.run_bandit")
    @patch("main.run_ruff")
    def test_pipeline_tool_error_returns_two(self, mock_ruff, mock_bandit, sample_file_path):
        mock_ruff.return_value = ToolResult(
            tool_name="Ruff",
            target_path=sample_file_path,
            success=False,
            issues_found=False,
            exit_code=2,
            error_message="Configuration error",
        )
        mock_bandit.return_value = ToolResult(
            tool_name="Bandit",
            target_path=sample_file_path,
            success=True,
            issues_found=False,
            exit_code=0,
            issues=[],
        )

        exit_code = run_pipeline(sample_file_path)
        assert exit_code == 2

    def test_pipeline_nonexistent_target_returns_two(self):
        exit_code = run_pipeline("this/does/not/exist_at_all.py")
        assert exit_code == 2
