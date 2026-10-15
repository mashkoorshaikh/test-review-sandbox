"""
tests/test_sandbox_runner.py - Unit tests for sandbox_runner.py.

All tests use mocks and a fake subprocess runner.
No Docker calls are made; no network access required.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from sandbox_runner import (
    SANDBOX_IMAGE,
    _build_docker_run_command,
    is_docker_available,
    run_tests_in_sandbox,
    validate_test_file_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proc(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


# ---------------------------------------------------------------------------
# is_docker_available
# ---------------------------------------------------------------------------

class TestIsDockerAvailable:
    def test_returns_false_when_binary_not_on_path(self):
        with patch("sandbox_runner.shutil.which", return_value=None):
            assert is_docker_available() is False

    def test_returns_false_when_docker_info_fails(self):
        with (
            patch("sandbox_runner.shutil.which", return_value="/usr/bin/docker"),
            patch("sandbox_runner.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1)
            assert is_docker_available() is False

    def test_returns_true_when_docker_info_succeeds(self):
        with (
            patch("sandbox_runner.shutil.which", return_value="/usr/bin/docker"),
            patch("sandbox_runner.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            assert is_docker_available() is True

    def test_returns_false_on_file_not_found_exception(self):
        with (
            patch("sandbox_runner.shutil.which", return_value="/usr/bin/docker"),
            patch("sandbox_runner.subprocess.run", side_effect=FileNotFoundError),
        ):
            assert is_docker_available() is False



# ---------------------------------------------------------------------------
# _build_docker_run_command
# ---------------------------------------------------------------------------

class TestBuildDockerRunCommand:
    def test_command_contains_required_security_flags(self):
        cmd = _build_docker_run_command(
            source_mount="/tmp/src",
            test_mount="/tmp/tests",
            test_filename="test_foo.py",
            timeout=60,
        )
        cmd_str = " ".join(cmd)
        assert "--network=none" in cmd_str
        assert "--read-only" in cmd_str
        assert "--cap-drop=ALL" in cmd_str
        assert "--security-opt=no-new-privileges" in cmd_str
        assert "--user=65534:65534" in cmd_str
        assert "--memory=256m" in cmd_str
        assert "--cpus=1.0" in cmd_str
        assert "--pids-limit=64" in cmd_str

    def test_command_includes_correct_image(self):
        cmd = _build_docker_run_command("/src", "/tests", "test_foo.py", 60)
        assert SANDBOX_IMAGE in cmd

    def test_command_includes_source_mount_readonly(self):
        cmd = _build_docker_run_command("/tmp/src", "/tmp/tests", "test_foo.py", 60)
        cmd_str = " ".join(cmd)
        assert "/tmp/src" in cmd_str
        assert "readonly" in cmd_str

    def test_command_includes_test_filename_in_pytest_invocation(self):
        cmd = _build_docker_run_command("/src", "/tests", "test_calc.py", 60)
        # The sh -c argument should include the test filename
        sh_args = " ".join(cmd)
        assert "test_calc.py" in sh_args

    def test_command_starts_with_docker_run(self):
        cmd = _build_docker_run_command("/src", "/tests", "test_foo.py", 60)
        assert cmd[0] == "docker"
        assert cmd[1] == "run"


# ---------------------------------------------------------------------------
# validate_test_file_path
# ---------------------------------------------------------------------------

class TestValidateTestFilePath:
    def test_raises_permission_error_for_path_outside_workspace(self, tmp_path):
        outside = "/etc/passwd"
        workspace = str(tmp_path)
        with pytest.raises(PermissionError, match="outside"):
            validate_test_file_path(outside, workspace)

    def test_raises_file_not_found_for_missing_file(self, tmp_path):
        missing = str(tmp_path / "nonexistent.py")
        with pytest.raises(FileNotFoundError):
            validate_test_file_path(missing, str(tmp_path))

    def test_returns_resolved_path_for_valid_file(self, tmp_path):
        test_file = tmp_path / "test_valid.py"
        test_file.write_text("def test_x(): pass\n")
        result = validate_test_file_path(str(test_file), str(tmp_path))
        assert result.is_file()


# ---------------------------------------------------------------------------
# run_tests_in_sandbox
# ---------------------------------------------------------------------------

class TestRunTestsInSandbox:
    """All tests inject a fake Docker runner - no real Docker calls."""

    def test_returns_docker_unavailable_when_not_found(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        tf = tmp_path / "test_foo.py"
        tf.write_text("def test_x(): pass\n")

        with patch("sandbox_runner.is_docker_available", return_value=False):
            result = run_tests_in_sandbox(
                source_dir=str(src),
                test_file=str(tf),
            )

        assert result.docker_unavailable is True
        assert result.executed is False
        assert result.passed is False
        assert "Docker" in result.error_message

    def test_returns_error_when_source_dir_missing(self, tmp_path):
        tf = tmp_path / "test_foo.py"
        tf.write_text("def test_x(): pass\n")

        def fake_runner(cmd, timeout):
            return _make_proc(0, b"passed", b"")

        result = run_tests_in_sandbox(
            source_dir=str(tmp_path / "nonexistent"),
            test_file=str(tf),
            _docker_runner=fake_runner,
        )

        assert result.executed is False
        assert "does not exist" in result.error_message

    def test_returns_error_when_test_file_missing(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()

        def fake_runner(cmd, timeout):
            return _make_proc(0, b"passed", b"")

        result = run_tests_in_sandbox(
            source_dir=str(src),
            test_file=str(tmp_path / "nonexistent_test.py"),
            _docker_runner=fake_runner,
        )

        assert result.executed is False
        assert "does not exist" in result.error_message

    def test_passed_result_when_exit_code_zero(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "calc.py").write_text("def add(a, b): return a + b\n")
        tf = tmp_path / "test_calc.py"
        tf.write_text("def test_add(): assert True\n")

        captured_cmd = []

        def fake_runner(cmd, timeout):
            captured_cmd.extend(cmd)
            return _make_proc(0, b"1 passed", b"")

        result = run_tests_in_sandbox(
            source_dir=str(src),
            test_file=str(tf),
            _docker_runner=fake_runner,
        )

        assert result.executed is True
        assert result.passed is True
        assert result.exit_code == 0
        assert "1 passed" in result.stdout
        assert result.timed_out is False

    def test_failed_result_when_exit_code_nonzero(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "calc.py").write_text("def add(a, b): return a + b\n")
        tf = tmp_path / "test_calc.py"
        tf.write_text("def test_fail(): assert False\n")

        def fake_runner(cmd, timeout):
            return _make_proc(1, b"1 failed", b"AssertionError")

        result = run_tests_in_sandbox(
            source_dir=str(src),
            test_file=str(tf),
            _docker_runner=fake_runner,
        )

        assert result.executed is True
        assert result.passed is False
        assert result.exit_code == 1
        assert "1 failed" in result.stdout

    def test_timeout_sets_timed_out_flag(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        tf = tmp_path / "test_x.py"
        tf.write_text("def test_x(): pass\n")

        def fake_runner(cmd, timeout):
            raise subprocess.TimeoutExpired(cmd, timeout)

        result = run_tests_in_sandbox(
            source_dir=str(src),
            test_file=str(tf),
            _docker_runner=fake_runner,
        )

        assert result.timed_out is True
        assert result.passed is False
        assert result.executed is True

    def test_exception_in_runner_returns_error_result(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        tf = tmp_path / "test_x.py"
        tf.write_text("def test_x(): pass\n")

        def fake_runner(cmd, timeout):
            raise RuntimeError("Docker daemon not running")

        result = run_tests_in_sandbox(
            source_dir=str(src),
            test_file=str(tf),
            _docker_runner=fake_runner,
        )

        assert result.executed is False
        assert "Docker daemon not running" in result.error_message

    def test_stdout_truncated_at_max_bytes(self, tmp_path):
        from sandbox_runner import MAX_OUTPUT_BYTES

        src = tmp_path / "src"
        src.mkdir()
        tf = tmp_path / "test_big.py"
        tf.write_text("def test_x(): pass\n")

        huge_output = b"x" * (MAX_OUTPUT_BYTES + 1000)

        def fake_runner(cmd, timeout):
            return _make_proc(0, huge_output, b"")

        result = run_tests_in_sandbox(
            source_dir=str(src),
            test_file=str(tf),
            _docker_runner=fake_runner,
        )

        assert len(result.stdout) <= MAX_OUTPUT_BYTES
