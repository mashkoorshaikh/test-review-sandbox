"""
tests/test_cli.py - Unit tests for CLI entry point and user interface.

Verifies:
  1. CLI respects --no-ai flag.
  2. Missing API credentials print clear error without pipeline crash.
  3. All separate sections (findings, AI explanations, test suggestions, errors) render.
  4. Token CLI argument emits deprecation warning without leaking secrets.
"""

from __future__ import annotations

import sys
import warnings
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


class TestCLITokenHandlingAndDeprecation:
    """Tests for CLI token handling, deprecation warnings, and credential protection."""

    def test_token_cli_arg_emits_deprecation_warning(self, sample_file):
        test_args = ["main.py", "--token", "ghp_insecure_cli_token_9999", sample_file]
        with (
            patch.object(sys, "argv", test_args),
            patch("main.run_pipeline", return_value=0) as mock_pipeline,
            pytest.warns(UserWarning, match="Passing GitHub token via '--token' CLI option is deprecated"),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0
        mock_pipeline.assert_called_once()

    def test_token_cli_arg_warning_never_exposes_token(self, sample_file, capsys):
        secret = "ghp_super_secret_never_print_me_123456"
        test_args = ["main.py", "--token", secret, sample_file]
        with (
            patch.object(sys, "argv", test_args),
            patch("main.run_pipeline", return_value=0),
            pytest.warns(UserWarning),
            pytest.raises(SystemExit),
        ):
            main()
        captured = capsys.readouterr()
        # Neither stdout nor stderr must contain the token
        assert secret not in captured.out
        assert secret not in captured.err
        assert "[Security Warning]" in captured.err

    def test_cli_without_token_arg_does_not_warn(self, sample_file, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_env_var_token_valid_12345")
        test_args = ["main.py", sample_file]
        with (
            patch.object(sys, "argv", test_args),
            patch("main.run_pipeline", return_value=0),
            warnings.catch_warnings(record=True) as recorded,
        ):
            warnings.simplefilter("always")
            with pytest.raises(SystemExit):
                main()
            # Must not emit any warning about --token
            assert not any("CLI option is deprecated" in str(w.message) for w in recorded)

    def test_token_parser_help_indicates_deprecation(self, capsys):
        with (
            patch.object(sys, "argv", ["main.py", "--help"]),
            pytest.raises(SystemExit),
        ):
            main()
        captured = capsys.readouterr()
        assert "DEPRECATED" in captured.out
