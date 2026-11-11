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
    """Tests for CLI token handling, rejection, and credential protection."""

    def test_token_cli_arg_is_rejected_for_security(self, sample_file):
        test_args = ["main.py", "--token", "ghp_insecure_cli_token_9999", sample_file]
        with (
            patch.object(sys, "argv", test_args),
            patch("main.run_pipeline", return_value=0) as mock_pipeline,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 2
        mock_pipeline.assert_not_called()

    def test_token_cli_arg_rejection_never_exposes_token(self, sample_file, capsys):
        secret = "ghp_super_secret_never_print_me_123456"
        test_args = ["main.py", "--token", secret, sample_file]
        with (
            patch.object(sys, "argv", test_args),
            patch("main.run_pipeline", return_value=0),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        # Neither stdout nor stderr must contain the token
        assert secret not in captured.out
        assert secret not in captured.err
        assert "[Security Error]" in captured.err

    def test_cli_without_token_arg_uses_environment(self, sample_file, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_env_var_token_valid_12345")
        test_args = ["main.py", sample_file]
        with (
            patch.object(sys, "argv", test_args),
            patch("main.run_pipeline", return_value=0) as mock_pipeline,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0
        mock_pipeline.assert_called_once()

    def test_token_parser_help_indicates_disabled(self, capsys):
        with (
            patch.object(sys, "argv", ["main.py", "--help"]),
            pytest.raises(SystemExit),
        ):
            main()
        captured = capsys.readouterr()
        assert "DISABLED" in captured.out

    def test_cli_auth_with_gh_token_env_var(self, sample_file, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "ghp_gh_token_env_var_99999")
        test_args = ["main.py", sample_file]
        with (
            patch.object(sys, "argv", test_args),
            patch("main.run_pipeline", return_value=0) as mock_pipeline,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0
        mock_pipeline.assert_called_once()

    def test_cli_auth_with_github_token_file(self, sample_file, tmp_path, monkeypatch):
        token_file = tmp_path / "token.secret"
        token_file.write_text("ghp_token_from_secure_file_7777\n", encoding="utf-8")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN_FILE", str(token_file))
        test_args = ["main.py", sample_file]
        with (
            patch.object(sys, "argv", test_args),
            patch("main.run_pipeline", return_value=0) as mock_pipeline,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0
        mock_pipeline.assert_called_once()

    def test_pr_mode_cli_invokes_pipeline_with_none_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_env_token_for_pr_mode")
        test_args = ["main.py", "--pr", "42", "--repo", "owner/repo"]
        with (
            patch.object(sys, "argv", test_args),
            patch("main.run_pr_pipeline", return_value=0) as mock_pr_pipeline,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0
        mock_pr_pipeline.assert_called_once()
        # Verify token parameter passed to run_pr_pipeline is strictly None
        assert mock_pr_pipeline.call_args.kwargs["token"] is None
