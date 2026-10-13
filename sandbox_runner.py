"""
sandbox_runner.py - Docker Sandboxed Test Execution for Milestone 3.

Provides safe, isolated test execution using Docker containers:
  - Checks Docker availability before attempting execution.
  - If Docker is unavailable, reports skip; never falls back to direct host execution.
  - Uses a pinned, minimal Python image with no network access by default.
  - Mounts only the minimum disposable workspace required.
  - Enforces resource limits, timeouts, and capability drops.
  - Never mounts Docker socket, home directory, or credentials.
  - Treats all source code and generated tests as untrusted.

Security constraints enforced per container:
  --network=none          No network access
  --read-only             Read-only root filesystem (with explicit writable /tmp)
  --cap-drop=ALL          Drop all Linux capabilities
  --security-opt=no-new-privileges Prevent privilege escalation
  --user=65534:65534      Run as non-root (nobody:nogroup)
  --memory=256m           Memory limit
  --cpus=1.0              CPU limit
  --pids-limit=64         Process count limit
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Pinned, minimal Python sandbox image with pre-installed pytest.
# Can be overridden via the SANDBOX_IMAGE environment variable.
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "code-review-sandbox:3.11")

# Default timeout for sandboxed execution (seconds)
DEFAULT_SANDBOX_TIMEOUT = 60

# Maximum size of stdout/stderr captured from container (bytes)
MAX_OUTPUT_BYTES = 64 * 1024  # 64 KB


@dataclass
class SandboxResult:
    """
    Captures the outcome of a sandboxed test execution attempt.

    Attributes:
        executed: True if a container was actually started and ran.
        exit_code: Container exit code (0=pass, 1=test failures, None=did not run).
        passed: True only if exit_code == 0 AND executed is True.
        stdout: Captured standard output from pytest inside the container.
        stderr: Captured standard error from the container.
        timed_out: True if the container was killed due to timeout.
        docker_unavailable: True if Docker was not found on the host.
        error_message: Human-readable error if execution failed or was skipped.
        source_workspace: Path to the disposable source copy used.
        test_workspace: Path to the disposable test workspace used.
    """
    executed: bool = False
    exit_code: Optional[int] = None
    passed: bool = False
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    docker_unavailable: bool = False
    error_message: Optional[str] = None
    source_workspace: Optional[str] = None
    test_workspace: Optional[str] = None


def is_docker_available() -> bool:
    """
    Check whether the Docker CLI binary is reachable and the daemon is responsive.

    Returns False (not True) if:
      - The `docker` binary is not on PATH.
      - `docker info` fails (daemon not running, permission denied, etc.).
    """
    docker_path = shutil.which("docker")
    if docker_path is None:
        logger.info("Docker binary not found on PATH.")
        return False

    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def validate_test_file_path(test_file: str, test_workspace: str) -> Path:
    """
    Validate that a test file path resolves within the dedicated test workspace.

    Raises:
        PermissionError: If the path escapes the test workspace.
        FileNotFoundError: If the file does not exist.
    """
    workspace = Path(test_workspace).resolve()
    resolved = Path(test_file).resolve()

    if not resolved.is_relative_to(workspace):
        raise PermissionError(
            f"Security Violation: Test file '{test_file}' is outside "
            f"the designated test workspace '{test_workspace}'."
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Test file not found: '{test_file}'")

    return resolved


def _build_docker_run_command(
    source_mount: str,
    test_mount: str,
    test_filename: str,
    timeout: int,
) -> List[str]:
    """
    Build the complete `docker run` command with security constraints.

    Mount layout inside container:
      /src    - read-only disposable copy of source under review
      /tests  - read-only disposable test workspace
      /tmp    - writable temporary directory for pytest cache

    Security flags applied:
      --network=none, --read-only, --cap-drop=ALL,
      --security-opt=no-new-privileges, --user=65534:65534,
      --memory=256m, --cpus=1.0, --pids-limit=64,
      --rm (auto-remove container on exit)
    """
    return [
        "docker", "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65534:65534",
        "--memory=256m",
        "--cpus=1.0",
        "--pids-limit=64",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=32m",
        f"--mount=type=bind,source={source_mount},target=/src,readonly",
        f"--mount=type=bind,source={test_mount},target=/tests,readonly",
        "--workdir=/src",
        SANDBOX_IMAGE,
        "sh", "-c",
        (
            f"python -m pytest /tests/{test_filename} -v --tb=short "
            "--no-header -p no:cacheprovider"
        ),
    ]


def run_tests_in_sandbox(
    source_dir: str,
    test_file: str,
    timeout: int = DEFAULT_SANDBOX_TIMEOUT,
    _docker_runner: Optional[object] = None,  # injection point for unit tests
    test_workspace: Optional[str] = None,
) -> SandboxResult:
    """
    Execute a test file against a source directory inside a disposable Docker container.

    Args:
        source_dir: Path to the source directory to mount read-only inside container.
        test_file: Absolute path to the generated test file to execute.
        timeout: Execution timeout in seconds before the container is killed.
        _docker_runner: Optional injectable callable for unit test mocking.
                        Signature: (cmd: List[str], timeout: int) -> subprocess.CompletedProcess
        test_workspace: Optional designated workspace directory. If provided,
                        validates that test_file does not escape it.

    Returns:
        SandboxResult describing execution outcome.

    Safety guarantees:
      - If Docker is not available, returns SandboxResult(docker_unavailable=True).
      - Source and test files are mounted read-only; container runs as non-root.
      - No network access, no privilege escalation, limited resources.
      - Symlinks pointing outside the source tree are ignored.
      - Never executes code directly on the host.
      - All temp directories are cleaned up regardless of outcome.
    """
    # 1. Gate: check Docker availability
    if _docker_runner is None and not is_docker_available():
        return SandboxResult(
            docker_unavailable=True,
            error_message=(
                "Docker is not available on this host. "
                "Sandbox execution was skipped. "
                "Install Docker Desktop or Docker Engine to enable sandboxed test execution."
            ),
        )

    # Validate test file path against test workspace if provided
    if test_workspace is not None:
        try:
            test_path = validate_test_file_path(test_file, test_workspace)
        except (PermissionError, FileNotFoundError) as err:
            return SandboxResult(error_message=str(err))
    else:
        test_path = Path(test_file).resolve()
        if not test_path.is_file():
            return SandboxResult(
                error_message=f"Test file does not exist: '{test_file}'"
            )

    source_path = Path(source_dir).resolve()
    if not source_path.exists():
        return SandboxResult(
            error_message=f"Source directory does not exist: '{source_dir}'"
        )

    test_filename = test_path.name

    disposable_src: Optional[str] = None
    disposable_tests: Optional[str] = None

    try:
        # 2. Create disposable copies in isolated temp directories
        disposable_src = tempfile.mkdtemp(prefix="sandbox_src_")
        disposable_tests = tempfile.mkdtemp(prefix="sandbox_tests_")

        # Symlink guard: ignore any symlinks pointing outside the source tree
        def _filter_escaping_symlinks(dir_path: str, names: List[str]) -> List[str]:
            ignored = []
            dir_p = Path(dir_path).resolve()
            for name in names:
                item_p = dir_p / name
                if item_p.is_symlink():
                    try:
                        target = item_p.resolve()
                        if not target.is_relative_to(source_path):
                            logger.warning(
                                "Ignoring symlink '%s' pointing outside source tree: '%s'",
                                item_p, target
                            )
                            ignored.append(name)
                    except (ValueError, RuntimeError):
                        ignored.append(name)
            return ignored

        # Copy source to disposable workspace
        src_dest = Path(disposable_src) / source_path.name
        if source_path.is_dir():
            shutil.copytree(str(source_path), str(disposable_src), dirs_exist_ok=True, ignore=_filter_escaping_symlinks)
            shutil.copytree(str(source_path), str(src_dest), dirs_exist_ok=True, ignore=_filter_escaping_symlinks)
        else:
            shutil.copy2(str(source_path), str(src_dest))
            shutil.copy2(str(source_path), str(Path(disposable_src) / source_path.name))

        # Copy only the single test file
        shutil.copy2(str(test_path), os.path.join(disposable_tests, test_filename))

        # 3. Build docker run command
        cmd = _build_docker_run_command(
            source_mount=disposable_src,
            test_mount=disposable_tests,
            test_filename=test_filename,
            timeout=timeout,
        )

        logger.info("Launching sandbox container: %s", " ".join(shlex.quote(c) for c in cmd))

        # 4. Execute: use injected runner (for tests) or real subprocess
        if _docker_runner is not None:
            proc = _docker_runner(cmd, timeout)
        else:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )

        stdout_raw = proc.stdout if isinstance(proc.stdout, str) else (proc.stdout or b"").decode("utf-8", errors="replace")
        stderr_raw = proc.stderr if isinstance(proc.stderr, str) else (proc.stderr or b"").decode("utf-8", errors="replace")

        # Truncate to prevent runaway output
        stdout_text = stdout_raw[:MAX_OUTPUT_BYTES]
        stderr_text = stderr_raw[:MAX_OUTPUT_BYTES]

        passed = proc.returncode == 0

        return SandboxResult(
            executed=True,
            exit_code=proc.returncode,
            passed=passed,
            stdout=stdout_text,
            stderr=stderr_text,
            source_workspace=disposable_src,
            test_workspace=disposable_tests,
        )

    except subprocess.TimeoutExpired:
        logger.warning("Sandbox execution timed out after %d seconds.", timeout)
        return SandboxResult(
            executed=True,
            exit_code=None,
            passed=False,
            timed_out=True,
            error_message=f"Sandbox execution timed out after {timeout} seconds.",
            source_workspace=disposable_src,
            test_workspace=disposable_tests,
        )
    except Exception as exc:
        logger.error("Sandbox execution failed: %s: %s", type(exc).__name__, exc)
        return SandboxResult(
            error_message=f"Sandbox execution error: {type(exc).__name__}: {exc}",
            source_workspace=disposable_src,
            test_workspace=disposable_tests,
        )
    finally:
        # 5. Cleanup disposable workspaces unconditionally
        for temp_dir in [disposable_src, disposable_tests]:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass
