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
    DEFAULT_CLEANUP_TIMEOUT,
    MAX_OUTPUT_BYTES,
    SANDBOX_IMAGE,
    _build_docker_run_command,
    _cleanup_container,
    generate_sandbox_container_name,
    is_docker_available,
    run_tests_in_sandbox,
    validate_test_file_path,
    validate_test_filename,
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
# generate_sandbox_container_name
# ---------------------------------------------------------------------------

class TestGenerateSandboxContainerName:
    def test_generates_unique_unpredictable_names(self):
        n1 = generate_sandbox_container_name()
        n2 = generate_sandbox_container_name()
        assert n1.startswith("agy_sbx_")
        assert n2.startswith("agy_sbx_")
        assert n1 != n2
        assert len(n1) == len("agy_sbx_") + 16


# ---------------------------------------------------------------------------
# validate_test_filename
# ---------------------------------------------------------------------------

class TestValidateTestFilename:
    @pytest.mark.parametrize(
        "valid_name",
        [
            "test_calc.py",
            "test_foo_bar.py",
            "test_123.py",
            "test-unit.py",
            "test_a-b_c.py",
            "calc_test.py",
        ],
    )
    def test_accepts_valid_test_filenames(self, valid_name):
        assert validate_test_filename(valid_name) == valid_name

    @pytest.mark.parametrize(
        "invalid_name",
        [
            "",
            "   ",
            "test calc.py",
            "test_calc.sh",
            "test_calc.txt",
            "test_calc",
            ".test_calc.py",
            "-test_calc.py",
            "../test_calc.py",
            "foo/test_calc.py",
            "foo\\test_calc.py",
            "/etc/passwd",
            "C:\\test.py",
            "test;rm.py",
            "test&whoami.py",
            "test|whoami.py",
            "test$(id).py",
            "test`id`.py",
            'test"quote.py',
            "test'quote.py",
            "test\x00null.py",
            "test\ninjection.py",
            "test..name.py",
        ],
    )
    def test_rejects_malicious_and_invalid_filenames(self, invalid_name):
        with pytest.raises(ValueError, match=r"(?i)(test filename|forbidden character|path separators)"):
            validate_test_filename(invalid_name)


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
        sh_args = " ".join(cmd)
        assert "test_calc.py" in sh_args

    def test_command_starts_with_docker_run(self):
        cmd = _build_docker_run_command("/src", "/tests", "test_foo.py", 60)
        assert cmd[0] == "docker"
        assert cmd[1] == "run"

    def test_command_contains_no_shell_invocation(self):
        cmd = _build_docker_run_command("/src", "/tests", "test_calc.py", 60)
        assert "sh" not in cmd
        assert "-c" not in cmd

    def test_pytest_invoked_as_separate_direct_arguments(self):
        cmd = _build_docker_run_command("/src", "/tests", "test_calc.py", 60)
        image_idx = cmd.index(SANDBOX_IMAGE)
        pytest_args = cmd[image_idx + 1 :]
        expected_args = [
            "python",
            "-m",
            "pytest",
            "/tests/test_calc.py",
            "-v",
            "--tb=short",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ]
        assert pytest_args == expected_args

    def test_command_includes_unique_container_name(self):
        cmd1 = _build_docker_run_command("/src", "/tests", "test_foo.py", 60)
        cmd2 = _build_docker_run_command("/src", "/tests", "test_foo.py", 60)
        name1 = next(arg for arg in cmd1 if arg.startswith("--name="))
        name2 = next(arg for arg in cmd2 if arg.startswith("--name="))
        assert name1.startswith("--name=agy_sbx_")
        assert name2.startswith("--name=agy_sbx_")
        assert name1 != name2

    def test_command_uses_custom_container_name_when_provided(self):
        custom_name = "agy_sbx_custom123"
        cmd = _build_docker_run_command(
            "/src", "/tests", "test_foo.py", 60, container_name=custom_name
        )
        assert f"--name={custom_name}" in cmd

    def test_build_command_rejects_malicious_test_filenames(self):
        with pytest.raises(ValueError, match=r"(?i)(test filename|forbidden character)"):
            _build_docker_run_command(
                "/src", "/tests", "test_calc.py; rm -rf /", 60
            )


# ---------------------------------------------------------------------------
# _cleanup_container
# ---------------------------------------------------------------------------

class TestCleanupContainer:
    def test_cleanup_container_invokes_docker_rm_f(self):
        with patch("sandbox_runner.subprocess.run") as mock_run:
            _cleanup_container("agy_sbx_test123")
            mock_run.assert_called_once_with(
                ["docker", "rm", "-f", "agy_sbx_test123"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=DEFAULT_CLEANUP_TIMEOUT,
                check=False,
            )

    def test_cleanup_container_empty_name_is_noop(self):
        with patch("sandbox_runner.subprocess.run") as mock_run:
            _cleanup_container("")
            mock_run.assert_not_called()

    def test_cleanup_container_handles_subprocess_timeout(self):
        with patch(
            "sandbox_runner.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["docker"], 10.0),
        ):
            # Must not raise an exception
            _cleanup_container("agy_sbx_test123")

    def test_cleanup_container_handles_os_error(self):
        with patch(
            "sandbox_runner.subprocess.run",
            side_effect=FileNotFoundError("docker binary not found"),
        ):
            # Must not raise an exception
            _cleanup_container("agy_sbx_test123")


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
        assert result.container_name is not None
        assert result.container_name.startswith("agy_sbx_")

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

    def test_timeout_sets_timed_out_flag_and_triggers_cleanup(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        tf = tmp_path / "test_x.py"
        tf.write_text("def test_x(): pass\n")

        cleanup_mock = MagicMock()

        def fake_runner(cmd, timeout):
            raise subprocess.TimeoutExpired(cmd, timeout)

        result = run_tests_in_sandbox(
            source_dir=str(src),
            test_file=str(tf),
            _docker_runner=fake_runner,
            _cleanup_fn=cleanup_mock,
        )

        assert result.timed_out is True
        assert result.passed is False
        assert result.executed is True
        assert result.container_name is not None
        cleanup_mock.assert_called_once_with(result.container_name)

    def test_timeout_cleanup_failure_handled_safely(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        tf = tmp_path / "test_x.py"
        tf.write_text("def test_x(): pass\n")

        def failing_cleanup(name):
            raise RuntimeError("Docker socket connection dropped")

        def fake_runner(cmd, timeout):
            raise subprocess.TimeoutExpired(cmd, timeout)

        result = run_tests_in_sandbox(
            source_dir=str(src),
            test_file=str(tf),
            _docker_runner=fake_runner,
            _cleanup_fn=failing_cleanup,
        )

        # Must still return the timeout result cleanly without raising
        assert result.timed_out is True
        assert result.passed is False
        assert result.executed is True
        assert "timed out" in (result.error_message or "").lower()

    def test_exception_in_runner_triggers_cleanup(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        tf = tmp_path / "test_x.py"
        tf.write_text("def test_x(): pass\n")

        cleanup_mock = MagicMock()

        def fake_runner(cmd, timeout):
            raise RuntimeError("Docker daemon not running")

        result = run_tests_in_sandbox(
            source_dir=str(src),
            test_file=str(tf),
            _docker_runner=fake_runner,
            _cleanup_fn=cleanup_mock,
        )

        assert result.executed is False
        assert "Docker daemon not running" in result.error_message
        assert result.container_name is not None
        cleanup_mock.assert_called_once_with(result.container_name)

    def test_stdout_truncated_at_max_bytes(self, tmp_path):
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

    def test_rejects_malicious_test_filename_without_execution(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        # Create a file whose name contains prohibited characters (e.g. spaces)
        tf = tmp_path / "test calc.py"
        tf.write_text("def test_x(): pass\n")

        runner_mock = MagicMock()

        result = run_tests_in_sandbox(
            source_dir=str(src),
            test_file=str(tf),
            _docker_runner=runner_mock,
        )

        assert result.executed is False
        assert "Invalid test file" in (result.error_message or "")
        runner_mock.assert_not_called()
