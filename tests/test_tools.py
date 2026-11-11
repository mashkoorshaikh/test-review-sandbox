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
import tempfile
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
    is_trusted_config_path,
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

    @patch("subprocess.run")
    def test_run_ruff_guards_against_flag_like_path(self, mock_subprocess, tmp_path):
        """Paths starting with a dash or flag name must be preceded by '--' and sanitized."""
        flag_file = tmp_path / "-v.py"
        flag_file.write_text("x = 1\n", encoding="utf-8")
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="[]", stderr="")

        result = run_ruff(str(flag_file))
        assert result.success is True

        cmd_called = mock_subprocess.call_args[0][0]
        assert "--" in cmd_called
        dash_idx = cmd_called.index("--")
        path_arg = cmd_called[dash_idx + 1]
        assert path_arg != "-v.py"
        assert not path_arg.startswith("-")

    def test_run_ruff_rejects_null_bytes(self):
        result = run_ruff("app\x00.py")
        assert result.success is False
        assert result.exit_code is None



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

    @patch("subprocess.run")
    def test_run_bandit_guards_against_flag_like_path(self, mock_subprocess, tmp_path):
        """Paths starting with a dash or flag name must be preceded by '--' and sanitized."""
        flag_file = tmp_path / "-v.py"
        flag_file.write_text("x = 1\n", encoding="utf-8")
        mock_subprocess.return_value = MagicMock(returncode=0, stdout=BANDIT_CLEAN_JSON, stderr="")

        result = run_bandit(str(flag_file))
        assert result.success is True

        cmd_called = mock_subprocess.call_args[0][0]
        assert "--" in cmd_called
        dash_idx = cmd_called.index("--")
        path_arg = cmd_called[dash_idx + 1]
        assert path_arg != "-v.py"
        assert not path_arg.startswith("-")

    def test_run_bandit_rejects_null_bytes(self):
        result = run_bandit("app\x00.py")
        assert result.success is False
        assert result.exit_code is None

    def test_run_bandit_uses_config_to_skip_b101_in_test_files(self, tmp_path):
        """Test files using pytest assertions do not generate B101 findings when bandit.yaml is present."""
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_foo():\n    assert 1 == 1\n", encoding="utf-8")
        result = run_bandit(str(test_file))
        assert result.success is True
        assert not any(i.code == "B101" for i in result.issues)

    def test_run_bandit_detects_b101_in_production_files(self):
        """Production files using assertions are strictly flagged with B101."""
        with tempfile.TemporaryDirectory(prefix="service_") as td:
            prod_file = Path(td) / "production_service.py"
            prod_file.write_text("def auth(user):\n    assert user.is_admin\n", encoding="utf-8")
            result = run_bandit(str(prod_file))
            assert result.success is True
            assert result.issues_found is True
            b101_issues = [i for i in result.issues if i.code == "B101"]
            assert len(b101_issues) >= 1
            assert b101_issues[0].file_path == str(prod_file)

    def test_run_bandit_detects_real_vulnerabilities_in_test_files(self, tmp_path):
        """Non-assertion security vulnerabilities in test files (e.g. eval, B307) remain detected."""
        test_file = tmp_path / "test_vulnerable.py"
        test_file.write_text("def test_vulnerable():\n    assert True\n    eval('2 + 2')\n", encoding="utf-8")
        result = run_bandit(str(test_file))
        assert result.success is True
        assert result.issues_found is True
        codes = [i.code for i in result.issues]
        assert "B307" in codes  # eval
        assert "B101" not in codes  # assert is skipped

    def test_run_bandit_explicit_config_path_respected(self, tmp_path):
        """Passing explicit config_path passes '-c' with that config."""
        custom_cfg = tmp_path / "custom_bandit.yaml"
        custom_cfg.write_text("assert_used:\n  skips: ['*']\n", encoding="utf-8")
        target_file = tmp_path / "app.py"
        target_file.write_text("assert True\n", encoding="utf-8")
        result = run_bandit(str(target_file), config_path=str(custom_cfg))
        assert result.success is True
        assert not any(i.code == "B101" for i in result.issues)



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


class TestScannerConfigurationHardening:
    """Tests verifying that untrusted repo files cannot hijack scanner configurations (SEC-01 & SEC-02)."""

    def test_is_trusted_config_path_validation(self, tmp_path):
        """Validate boundary checks for configuration file paths."""
        trusted_root = tmp_path / "maintainer_pipeline"
        trusted_root.mkdir()
        trusted_file = trusted_root / "bandit.yaml"
        trusted_file.write_text("skips: []\n", encoding="utf-8")

        untrusted_root = tmp_path / "pr_checkout"
        untrusted_root.mkdir()
        untrusted_file = untrusted_root / "bandit.yaml"
        untrusted_file.write_text("skips: ['B101']\n", encoding="utf-8")

        # None and empty candidates
        assert is_trusted_config_path(None, [str(trusted_root)]) is False
        assert is_trusted_config_path("", [str(trusted_root)]) is False

        # Non-existent file
        assert is_trusted_config_path(str(trusted_root / "missing.yaml"), [str(trusted_root)]) is False

        # Untrusted location outside root
        assert is_trusted_config_path(str(untrusted_file), [str(trusted_root)]) is False

        # Valid trusted location
        assert is_trusted_config_path(str(trusted_file), [str(trusted_root)]) is True

    def test_run_bandit_ignores_malicious_config_in_target_dir(self, tmp_path):
        """Untrusted bandit.yaml in target_path must NOT be discovered or suppress findings."""
        target_dir = tmp_path / "untrusted_repo"
        target_dir.mkdir()
        # Malicious config trying to skip B307 (eval_used)
        malicious_cfg = target_dir / "bandit.yaml"
        malicious_cfg.write_text("skips: ['B307']\n", encoding="utf-8")
        malicious_bandit = target_dir / ".bandit"
        malicious_bandit.write_text("skips: ['B307']\n", encoding="utf-8")

        # Vulnerable production file
        vuln_file = target_dir / "eval_service.py"
        vuln_file.write_text("def run_code(code):\n    return eval(code)\n", encoding="utf-8")

        result = run_bandit(str(target_dir))
        assert result.success is True
        assert result.issues_found is True
        # B307 must still be reported despite malicious bandit.yaml in target directory
        b307_issues = [i for i in result.issues if i.code == "B307"]
        assert len(b307_issues) >= 1

    def test_run_bandit_ignores_malicious_config_in_cwd(self, tmp_path, monkeypatch):
        """Untrusted bandit.yaml in current working directory must NOT suppress findings."""
        untrusted_cwd = tmp_path / "untrusted_cwd"
        untrusted_cwd.mkdir()
        malicious_cfg = untrusted_cwd / "bandit.yaml"
        malicious_cfg.write_text("skips: ['B307']\n", encoding="utf-8")

        target_dir = tmp_path / "clean_target"
        target_dir.mkdir()
        vuln_file = target_dir / "service.py"
        vuln_file.write_text("def run(s):\n    return eval(s)\n", encoding="utf-8")

        # Switch cwd to untrusted directory during scan
        monkeypatch.chdir(untrusted_cwd)

        result = run_bandit(str(target_dir))
        assert result.success is True
        assert result.issues_found is True
        b307_issues = [i for i in result.issues if i.code == "B307"]
        assert len(b307_issues) >= 1

    def test_run_bandit_rejects_explicit_untrusted_config_path(self, tmp_path):
        """Explicit config_path pointing outside trusted directories must be rejected and not used."""
        untrusted_dir = tmp_path / "attacker_controlled"
        untrusted_dir.mkdir()
        malicious_cfg = untrusted_dir / "malicious_bandit.yaml"
        malicious_cfg.write_text("skips: ['B307']\n", encoding="utf-8")

        target_dir = tmp_path / "target_src"
        target_dir.mkdir()
        vuln_file = target_dir / "app.py"
        vuln_file.write_text("def test(x):\n    return eval(x)\n", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout='{"results": []}', stderr="")
            # Pass untrusted config_path
            run_bandit(str(target_dir), config_path=str(malicious_cfg))

            mock_run.assert_called_once()
            cmd_args = mock_run.call_args[0][0]
            # Ensure the untrusted malicious_bandit.yaml was NOT passed via -c
            assert str(malicious_cfg) not in cmd_args

    def test_run_bandit_accepts_valid_trusted_config_path(self, tmp_path):
        """Explicit config_path within trusted boundary must be accepted."""
        trusted_dir = tmp_path / "pipeline_trusted"
        trusted_dir.mkdir()
        trusted_cfg = trusted_dir / "trusted_bandit.yaml"
        trusted_cfg.write_text("skips: []\n", encoding="utf-8")

        target_dir = tmp_path / "target_src"
        target_dir.mkdir()
        vuln_file = target_dir / "app.py"
        vuln_file.write_text("x = 1\n", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='{"results": []}', stderr="")
            run_bandit(str(target_dir), config_path=str(trusted_cfg), trusted_dirs=[str(trusted_dir)])

            mock_run.assert_called_once()
            cmd_args = mock_run.call_args[0][0]
            assert "-c" in cmd_args
            c_idx = cmd_args.index("-c")
            assert cmd_args[c_idx + 1] == str(trusted_cfg.resolve())

    def test_run_bandit_normal_findings_clean_and_vulnerable(self, tmp_path):
        """Verify normal Bandit findings for both clean and vulnerable Python files."""
        # Clean file
        clean_file = tmp_path / "clean.py"
        clean_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
        clean_res = run_bandit(str(clean_file))
        assert clean_res.success is True
        assert clean_res.issues_found is False
        assert len(clean_res.issues) == 0

        # Vulnerable file (eval)
        vuln_file = tmp_path / "vuln.py"
        vuln_file.write_text("def run(cmd: str):\n    return eval(cmd)\n", encoding="utf-8")
        vuln_res = run_bandit(str(vuln_file))
        assert vuln_res.success is True
        assert vuln_res.issues_found is True
        assert any(i.code == "B307" for i in vuln_res.issues)

    def test_run_ruff_ignores_malicious_config_with_isolated(self, tmp_path):
        """Untrusted pyproject.toml, ruff.toml, and .ruff.toml in target must NOT suppress rules."""
        target_dir = tmp_path / "untrusted_repo"
        target_dir.mkdir()
        # Malicious config attempting to ignore all linter checks
        pyproject = target_dir / "pyproject.toml"
        pyproject.write_text('[tool.ruff.lint]\nignore = ["ALL"]\nselect = []\n', encoding="utf-8")
        ruff_toml = target_dir / "ruff.toml"
        ruff_toml.write_text('lint.ignore = ["ALL"]\nlint.select = []\n', encoding="utf-8")
        dot_ruff = target_dir / ".ruff.toml"
        dot_ruff.write_text('lint.ignore = ["ALL"]\nlint.select = []\n', encoding="utf-8")

        # Source file with unused import (F401) and undefined variable (F821)
        test_file = target_dir / "service.py"
        test_file.write_text("import sys\nimport os\n\ndef run():\n    return undefined_var\n", encoding="utf-8")

        result = run_ruff(str(test_file))
        assert result.success is True
        assert result.issues_found is True
        codes = {i.code for i in result.issues}
        assert "F401" in codes
        assert "F821" in codes

    def test_run_ruff_preserves_rules_with_explicit_select(self, tmp_path):
        """Explicit select_rules CLI argument must be forwarded to ensure required rules are enforced."""
        target_file = tmp_path / "test_module.py"
        target_file.write_text("import sys\n", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="[]", stderr="")
            run_ruff(str(target_file), select_rules=["E", "F", "B"])

            mock_run.assert_called_once()
            cmd_args = mock_run.call_args[0][0]
            assert "--isolated" in cmd_args
            assert "--select=E,F,B" in cmd_args
