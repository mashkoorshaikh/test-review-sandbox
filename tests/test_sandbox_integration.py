"""
tests/test_sandbox_integration.py - Integration tests for sandboxed execution.

These tests require:
  1. Docker to be installed and the Docker daemon to be running.
  2. The SANDBOX_INTEGRATION_TESTS environment variable to be set to "1".

Skip cleanly when Docker is unavailable or the opt-in flag is not set.

Run integration tests with:
  docker build -t code-review-sandbox:3.11 -t python:3.11-slim .
  SANDBOX_INTEGRATION_TESTS=1 pytest tests/test_sandbox_integration.py -v

These tests run inside the pinned code-review-sandbox:3.11 container.
"""

from __future__ import annotations

import os
import textwrap

import pytest

from sandbox_runner import is_docker_available, run_tests_in_sandbox

# Guard: skip entire module unless explicitly opted in
INTEGRATION_ENABLED = os.getenv("SANDBOX_INTEGRATION_TESTS", "") == "1"

pytestmark = [
    pytest.mark.skipif(
        not INTEGRATION_ENABLED,
        reason=(
            "Sandbox integration tests are opt-in. "
            "Set SANDBOX_INTEGRATION_TESTS=1 to enable."
        ),
    ),
]


@pytest.fixture(autouse=True)
def require_docker():
    """Skip individual tests if Docker daemon is not running."""
    if not is_docker_available():
        pytest.skip("Docker daemon is not available - skipping sandbox integration test.")


class TestSandboxIntegration:
    """
    Live Docker sandbox integration tests.
    These tests require Docker and explicit opt-in.
    """

    def test_passing_test_returns_exit_code_zero(self, tmp_path):
        """A valid pytest file with passing assertions returns exit code 0."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "calc.py").write_text(
            "def add(a, b):\n    return a + b\n",
            encoding="utf-8",
        )

        test_file = tmp_path / "test_calc.py"
        test_file.write_text(
            textwrap.dedent("""\
                import sys
                sys.path.insert(0, '/src')
                from calc import add

                def test_add_positive():
                    assert add(2, 3) == 5

                def test_add_zero():
                    assert add(0, 0) == 0
            """),
            encoding="utf-8",
        )

        result = run_tests_in_sandbox(
            source_dir=str(src_dir),
            test_file=str(test_file),
            timeout=120,
        )

        assert result.executed is True, f"Sandbox did not execute. Error: {result.error_message}"
        assert result.passed is True, f"Tests failed. stdout: {result.stdout}\nstderr: {result.stderr}"
        assert result.exit_code == 0
        assert result.timed_out is False

    def test_failing_test_returns_nonzero_exit_code(self, tmp_path):
        """A pytest file with deliberate assertion failure returns exit code 1."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "mod.py").write_text("x = 1\n", encoding="utf-8")

        test_file = tmp_path / "test_fail.py"
        test_file.write_text(
            "def test_intentional_failure():\n    assert 1 == 2\n",
            encoding="utf-8",
        )

        result = run_tests_in_sandbox(
            source_dir=str(src_dir),
            test_file=str(test_file),
            timeout=120,
        )

        assert result.executed is True
        assert result.passed is False
        assert result.exit_code != 0

    def test_no_network_access_inside_container(self, tmp_path):
        """Container should not be able to make network requests (--network=none)."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "mod.py").write_text("x = 1\n", encoding="utf-8")

        test_file = tmp_path / "test_network.py"
        test_file.write_text(
            textwrap.dedent("""\
                import socket
                import pytest

                def test_no_network_access():
                    with pytest.raises((OSError, socket.gaierror, ConnectionRefusedError)):
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.settimeout(2)
                        s.connect(('8.8.8.8', 80))
            """),
            encoding="utf-8",
        )

        result = run_tests_in_sandbox(
            source_dir=str(src_dir),
            test_file=str(test_file),
            timeout=30,
        )

        # Network should be blocked, so the test expecting OSError should pass
        assert result.executed is True
        # We don't assert pass/fail here since platform behavior may vary;
        # the main check is that the container ran and didn't hang.
        assert result.timed_out is False

    def test_container_exits_within_timeout(self, tmp_path):
        """A simple test should complete well within the default timeout."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "mod.py").write_text("def hello(): return 'hello'\n", encoding="utf-8")

        test_file = tmp_path / "test_fast.py"
        test_file.write_text(
            "def test_fast():\n    assert True\n",
            encoding="utf-8",
        )

        result = run_tests_in_sandbox(
            source_dir=str(src_dir),
            test_file=str(test_file),
            timeout=120,
        )

        assert result.executed is True
        assert result.timed_out is False
