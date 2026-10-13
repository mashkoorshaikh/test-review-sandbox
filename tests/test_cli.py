"""
tests/test_cli.py - Unit tests for CLI entry point and user interface.

Verifies:
  1. CLI respects --no-ai flag.
  2. Missing API credentials print clear error without pipeline crash.
  3. All separate sections (findings, AI explanations, test suggestions, errors) render.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from main import main, run_pipeline
from tools import Issue, ToolResult


@pytest.fixture
def sample_file(tmp_path):
    p = tmp_path / "app.py"
    p.write_text("x = 1\n", encoding="utf-8")
    return str(p)


def test_cli_no_ai_flag_disables_ai(sample_file, capsys):
    """Running with --no-ai prints the disabled section and skips LLM."""
    with patch("main.run_ruff") as mock_ruff, patch("main.run_bandit") as mock_bandit:
        mock_ruff.return_value = ToolResult(
            tool_name="Ruff", target_path=sample_file, success=True, issues_found=False, exit_code=0
        )
        mock_bandit.return_value = ToolResult(
            tool_name="Bandit", target_path=sample_file, success=True, issues_found=False, exit_code=0
        )

        exit_code = run_pipeline(sample_file, enable_ai=False)

        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "AI MULTI-AGENT REVIEW: [DISABLED]" in captured
        assert "--no-ai specified" in captured


def test_cli_missing_api_key_shows_guidance_and_continues(sample_file, capsys):
    """When credentials are not set, CLI shows guidance and completes deterministic scan."""
    with patch("main.run_ruff") as mock_ruff, patch("main.run_bandit") as mock_bandit, \
         patch("main.is_api_key_configured", return_value=False):
        mock_ruff.return_value = ToolResult(
            tool_name="Ruff", target_path=sample_file, success=True, issues_found=False, exit_code=0
        )
        mock_bandit.return_value = ToolResult(
            tool_name="Bandit", target_path=sample_file, success=True, issues_found=False, exit_code=0
        )

        exit_code = run_pipeline(sample_file, enable_ai=True)

        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "CREDENTIALS MISSING" in captured
        assert "OPENAI_API_KEY" in captured


def test_cli_renders_all_sections_when_ai_enabled(sample_file, capsys):
    """When AI is enabled, all designated sections are rendered."""
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = "Explanation for test"
    mock_llm.invoke.return_value = mock_resp

    with patch("main.run_ruff") as mock_ruff, patch("main.run_bandit") as mock_bandit:
        mock_ruff.return_value = ToolResult(
            tool_name="Ruff", target_path=sample_file, success=True, issues_found=True, exit_code=1,
            issues=[Issue(code="F401", message="unused import", file_path=sample_file, line=1, tool="ruff")]
        )
        mock_bandit.return_value = ToolResult(
            tool_name="Bandit", target_path=sample_file, success=True, issues_found=False, exit_code=0
        )

        exit_code = run_pipeline(sample_file, enable_ai=True, llm=mock_llm)

        assert exit_code == 1
        captured = capsys.readouterr().out

        # Section 1: Deterministic overview & findings
        assert "OVERVIEW SUMMARY:" in captured
        assert "DETAILED DETERMINISTIC FINDINGS:" in captured

        # Section 2: AI Explanations
        assert "SECTION: AI AGENT EXPLANATIONS & THREAT MODELING" in captured
        assert "Linter Agent" in captured
        assert "Security Agent" in captured

        # Section 3: Test Suggestions
        assert "SECTION: AI TEST-WRITER AGENT SUGGESTIONS (FOR HUMAN REVIEW ONLY)" in captured

        # Section 4: Pipeline Verdict
        assert "PIPELINE RESULT:" in captured
