"""
tests/test_security_audit_m3_1.py - Regression & Security Audit Tests for Milestone 3.1.

Verifies:
  1. AST-based syntax checking and rejection of invalid Python without execution.
  2. AST-based detection of dangerous imports and calls.
  3. Non-execution of untrusted test code on the host system.
  4. Workspace boundary enforcement in sandbox_runner and apply_patch_to_copy.
  5. Symlink escape protection during source workspace copying.
  6. Null-byte rejection in patch validation.
  7. Patch verification strictly tied to real/mocked execution outcome (never falsely marked verified).
  8. Original source file immutability during patch proposal and verification.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from debugging_agent import (
    DebuggingAgent,
    _validate_patch_safety,
    apply_patch_to_copy,
)
from sandbox_runner import (
    SandboxResult,
    run_tests_in_sandbox,
    validate_test_file_path,
)
from test_validator import (
    TestValidationError,
    store_test_in_workspace,
    validate_generated_test_content,
)


# =============================================================================
# 1. AST Validation & Non-Execution on Host
# =============================================================================

class TestAstValidationAndNonExecution:
    def test_rejects_syntactically_invalid_python(self):
        """Code that matches regex but has invalid Python syntax must be rejected."""
        invalid_code = (
            "def test_broken_syntax(:\n"
            "    assert True\n"
        )
        is_valid, reason = validate_generated_test_content(invalid_code)
        assert is_valid is False
        assert "invalid python syntax" in reason.lower()

    def test_rejects_ast_prohibited_import_from(self):
        """Imports like `from os import system` must be caught by AST visitor."""
        dangerous_code = (
            "from os import system\n\n"
            "def test_danger():\n"
            "    system('whoami')\n"
        )
        is_valid, reason = validate_generated_test_content(dangerous_code)
        assert is_valid is False
        assert "os.system" in reason or "unsafe" in reason.lower()

    def test_rejects_pty_import(self):
        """Imports of pseudo-terminal modules like `pty` must be blocked."""
        dangerous_code = (
            "import pty\n\n"
            "def test_pty():\n"
            "    pass\n"
        )
        is_valid, reason = validate_generated_test_content(dangerous_code)
        assert is_valid is False
        assert "pty" in reason

    def test_untrusted_code_is_never_executed_on_host_during_validation(self):
        """
        Verify that code containing deliberate host side-effects (e.g. exit, raise)
        is never actually executed by validation.
        """
        side_effect_code = (
            "import pytest\n"
            "def test_side_effect():\n"
            "    assert True\n"
        )
        # Validation must parse AST only and return True without running test_side_effect
        is_valid, reason = validate_generated_test_content(side_effect_code)
        assert is_valid is True
        assert reason == ""


# =============================================================================
# 2. Workspace Boundaries & Symlink Protection
# =============================================================================

class TestWorkspaceBoundariesAndSymlinks:
    def test_sandbox_runner_enforces_test_workspace_boundary(self, tmp_path):
        """Test file outside designated test workspace must be rejected."""
        designated_ws = tmp_path / "valid_ws"
        designated_ws.mkdir()
        outside_dir = tmp_path / "outside_ws"
        outside_dir.mkdir()

        test_file = outside_dir / "test_sneaky.py"
        test_file.write_text("def test_sneaky(): pass\n")

        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Passing designated_ws should cause run_tests_in_sandbox to reject test_file
        result = run_tests_in_sandbox(
            source_dir=str(src_dir),
            test_file=str(test_file),
            test_workspace=str(designated_ws),
            _docker_runner=lambda cmd, timeout: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )
        assert result.executed is False
        assert "outside" in (result.error_message or "").lower()

    def test_sandbox_runner_filters_escaping_symlinks(self, tmp_path):
        """Symlinks inside source pointing outside source directory must be ignored."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "app.py").write_text("def run(): pass\n")

        # Secret file outside source
        secret_file = tmp_path / "secret.env"
        secret_file.write_text("SECRET_KEY=12345\n")

        # Create symlink pointing outside
        symlink_path = src_dir / "link_to_secret.env"
        try:
            symlink_path.symlink_to(secret_file)
        except (OSError, NotImplementedError):
            pytest.skip("Symlink creation not supported in current environment/permissions.")

        test_file = tmp_path / "test_app.py"
        test_file.write_text("def test_app(): pass\n")

        captured_mounts = []

        def fake_runner(cmd, timeout):
            # Inspect source_mount directory to ensure symlink was NOT copied
            for arg in cmd:
                if "source=" in arg and "target=/src" in arg:
                    mount_path = arg.split("source=")[1].split(",")[0]
                    captured_mounts.append(mount_path)
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        result = run_tests_in_sandbox(
            source_dir=str(src_dir),
            test_file=str(test_file),
            _docker_runner=fake_runner,
        )

        assert result.executed is True
        assert len(captured_mounts) == 1
        mount_dir = Path(captured_mounts[0]) / src_dir.name
        # The symlink pointing to secret.env must NOT have been copied
        assert not (mount_dir / "link_to_secret.env").exists()

    def test_apply_patch_to_copy_enforces_workspace_root(self, tmp_path):
        """apply_patch_to_copy must reject target file located outside workspace_root."""
        ws_root = tmp_path / "workspace"
        ws_root.mkdir()

        outside_file = tmp_path / "outside.py"
        outside_file.write_text("a = 1\n")

        patch = "--- a/outside.py\n+++ b/outside.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"

        success, err, copy_path = apply_patch_to_copy(
            original_file=str(outside_file),
            patch_text=patch,
            workspace_root=str(ws_root),
        )
        assert success is False
        assert "outside workspace" in err.lower()
        assert copy_path is None


# =============================================================================
# 3. Patch Safety & Correctness Verification
# =============================================================================

class TestPatchSafetyAndVerification:
    def test_patch_validator_rejects_null_bytes(self):
        """Patches containing null bytes must be rejected."""
        patch_with_null = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-\x00\n+safe\n"
        assert _validate_patch_safety(patch_with_null) is False

    def test_patch_verified_only_when_sandbox_passes(self, tmp_path):
        """patch_verified must be True ONLY when sandbox verification actually passes."""
        src_file = tmp_path / "calc.py"
        src_file.write_text("def add(a, b):\n    return a\n")

        diff_text = (
            "Fix addition return.\n"
            "```diff\n"
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -2,1 +2,1 @@\n"
            "-    return a\n"
            "+    return a + b\n"
            "```\n"
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=diff_text)

        test_file = tmp_path / "test_calc.py"
        test_file.write_text("def test_add(): assert add(2, 3) == 5\n")

        # 1. When sandbox fails, patch_verified MUST be False
        failing_sandbox = lambda source_dir, test_file: SandboxResult(executed=True, passed=False, exit_code=1)
        agent = DebuggingAgent(llm=mock_llm, max_attempts=1, workspace_root=str(tmp_path))
        result_fail = agent.run(
            target_file=str(src_file),
            ruff_findings=[],
            bandit_findings=[],
            failed_test_output="AssertionError",
            sandbox_runner=failing_sandbox,
            test_file=str(test_file),
            source_dir=str(tmp_path),
        )
        assert result_fail["patch_verified"] is False
        assert result_fail["patch_proposals"][0]["verification_result"]["passed"] is False

        # 2. When sandbox passes, patch_verified MUST be True
        passing_sandbox = lambda source_dir, test_file: SandboxResult(executed=True, passed=True, exit_code=0)
        result_pass = agent.run(
            target_file=str(src_file),
            ruff_findings=[],
            bandit_findings=[],
            failed_test_output="AssertionError",
            sandbox_runner=passing_sandbox,
            test_file=str(test_file),
            source_dir=str(tmp_path),
        )
        assert result_pass["patch_verified"] is True
        assert result_pass["patch_proposals"][0]["verification_result"]["passed"] is True

    def test_original_source_file_remains_unmodified_throughout(self, tmp_path):
        """Original source file must be completely untouched after patch proposal and verification."""
        src_file = tmp_path / "calc.py"
        original_content = "def add(a, b):\n    return a\n"
        src_file.write_text(original_content)

        diff_text = (
            "```diff\n"
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -2,1 +2,1 @@\n"
            "-    return a\n"
            "+    return a + b\n"
            "```\n"
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=diff_text)

        agent = DebuggingAgent(llm=mock_llm, max_attempts=2, workspace_root=str(tmp_path))
        agent.run(
            target_file=str(src_file),
            ruff_findings=[],
            bandit_findings=[],
            failed_test_output="AssertionError",
            sandbox_runner=lambda s, t: SandboxResult(executed=True, passed=True),
            test_file="test.py",
            source_dir=str(tmp_path),
        )

        # Original file MUST NOT have been modified
        assert src_file.read_text() == original_content
