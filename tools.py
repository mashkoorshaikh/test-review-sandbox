"""
tools.py - Static Analysis Execution Module for Code Review.

This module provides wrapper functions to safely execute static analysis tools
(Ruff and Bandit) via Python's subprocess module. It parses their output and
distinguishes between:
  1. Successful execution with no issues detected.
  2. Successful execution with code issues (lint/security findings) detected.
  3. Tool execution failures (e.g., tool crash, syntax errors, missing binary).

Safety guarantees:
  - shell=False is strictly enforced to prevent command injection.
  - Safe argument lists are passed directly to subprocess.run.
  - Target code is never executed (static analysis only).
  - Reviewed source code is never modified (read-only inspection).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Issue:
    """
    Represents an individual lint or security issue identified by a scanner.

    Attributes:
        code: Identifier of the rule or violation (e.g. 'F401', 'B101').
        message: Human-readable description of what was detected.
        file_path: Relative or absolute path to the file containing the issue.
        line: 1-indexed line number where the issue occurred, if available.
        column: 1-indexed column number where the issue occurred, if available.
        severity: Severity level (e.g. 'LOW', 'MEDIUM', 'HIGH', 'ERROR', 'WARNING').
        tool: Name of the tool that detected the issue ('ruff' or 'bandit').
        fixable: Boolean flag indicating if an automated fix is known to exist.
    """
    code: str
    message: str
    file_path: str
    line: Optional[int] = None
    column: Optional[int] = None
    severity: str = "WARNING"
    tool: str = "unknown"
    fixable: bool = False


@dataclass
class ToolResult:
    """
    Captures the complete outcome of a tool execution.

    Distinguishes clearly between:
      - `success`: Whether the tool executed successfully (did not crash).
      - `issues_found`: Whether the code under review contains violations.

    Attributes:
        tool_name: Friendly name of the tool (e.g., 'Ruff', 'Bandit').
        target_path: Path that was analyzed.
        success: True if the tool ran to completion without crashing.
        issues_found: True if one or more code issues were detected.
        exit_code: Process exit code returned by the subprocess.
        issues: Parsed list of Issue instances.
        raw_stdout: Standard output string from the process.
        raw_stderr: Standard error string from the process.
        error_message: Detailed message if tool execution failed (success=False).
        execution_time: Elapsed time for tool execution in seconds.
    """
    tool_name: str
    target_path: str
    success: bool
    issues_found: bool
    exit_code: Optional[int]
    issues: List[Issue] = field(default_factory=list)
    raw_stdout: str = ""
    raw_stderr: str = ""
    error_message: Optional[str] = None
    execution_time: float = 0.0


def resolve_tool_command(tool_name: str) -> List[str]:
    """
    Resolve the command arguments to invoke a tool safely.

    Resolution order:
      1. Virtualenv or current Python environment Scripts/bin directory.
         This ensures we use the tool installed in the active environment.
      2. System PATH via shutil.which().
      3. Fallback to `python -m <tool_name>`.

    Args:
        tool_name: Binary or module name ('ruff' or 'bandit').

    Returns:
        List of strings representing the executable invocation prefix.
    """
    # 1. Check inside current Python's script directory (e.g., .venv/Scripts/ruff.exe)
    py_dir = os.path.dirname(sys.executable)
    exe_suffix = ".exe" if sys.platform == "win32" else ""
    local_binary = os.path.join(py_dir, f"{tool_name}{exe_suffix}")

    if os.path.isfile(local_binary):
        return [local_binary]

    # 2. Check system PATH
    which_path = shutil.which(tool_name)
    if which_path:
        return [which_path]

    # 3. Fallback to running through current Python interpreter as module
    return [sys.executable, "-m", tool_name]


def _extract_json_block(text: str) -> str:
    """
    Helper to extract JSON content even if the tool prefixed or suffixed log lines.

    Tries direct parse first, then attempts extracting valid outer JSON object ({...})
    or outer JSON array ([...]).
    """
    stripped = text.strip()
    if not stripped:
        return ""

    # 1. Attempt direct parsing first
    try:
        json.loads(stripped)
        return stripped
    except Exception:
        pass

    # 2. Look for outer JSON object: {...}
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = stripped[first_brace:last_brace + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    # 3. Look for outer JSON array: [...]
    first_bracket = stripped.find("[")
    last_bracket = stripped.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        candidate = stripped[first_bracket:last_bracket + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    return stripped


def parse_ruff_json(stdout_text: str) -> List[Issue]:
    """
    Parse the JSON output generated by Ruff.

    Args:
        stdout_text: Raw stdout from `ruff check --output-format=json`.

    Returns:
        List of parsed Issue objects.
    """
    json_str = _extract_json_block(stdout_text)
    if not json_str:
        return []

    data = json.loads(json_str)
    if not isinstance(data, list):
        return []

    issues: List[Issue] = []
    for item in data:
        loc = item.get("location") or {}
        fix = item.get("fix")
        issues.append(
            Issue(
                code=item.get("code", "UNKNOWN"),
                message=item.get("message", "No description provided"),
                file_path=item.get("filename", ""),
                line=loc.get("row"),
                column=loc.get("column"),
                severity=str(item.get("severity", "error")).upper(),
                tool="ruff",
                fixable=bool(fix),
            )
        )
    return issues


def parse_bandit_json(stdout_text: str) -> List[Issue]:
    """
    Parse the JSON output generated by Bandit.

    Args:
        stdout_text: Raw stdout from `bandit -f json`.

    Returns:
        List of parsed Issue objects.
    """
    json_str = _extract_json_block(stdout_text)
    if not json_str:
        return []

    data = json.loads(json_str)
    if not isinstance(data, dict):
        return []

    results = data.get("results", [])
    issues: List[Issue] = []
    for item in results:
        issues.append(
            Issue(
                code=item.get("test_id", "SECURITY_ISSUE"),
                message=item.get("issue_text", "No description provided"),
                file_path=item.get("filename", ""),
                line=item.get("line_number"),
                column=item.get("col_offset"),
                severity=str(item.get("issue_severity", "MEDIUM")).upper(),
                tool="bandit",
                fixable=False,
            )
        )
    return issues


def run_ruff(
    target_path: str,
    custom_cmd: Optional[List[str]] = None,
    timeout: int = 60,
) -> ToolResult:
    """
    Execute Ruff linter safely on the target path using Python subprocess.

    Key behaviors:
      - Uses shell=False with argument list.
      - Ruff exit code 0: Clean code, no lint issues. (success=True, issues_found=False)
      - Ruff exit code 1: Lint issues found. (success=True, issues_found=True)
      - Ruff exit code >= 2: Tool runtime error or invalid config. (success=False)
      - Exceptions (FileNotFoundError, Timeout, etc.) mark success=False with message.

    Args:
        target_path: File or directory path to inspect.
        custom_cmd: Optional override for the base command executable.
        timeout: Maximum seconds to wait before terminating the subprocess.

    Returns:
        ToolResult instance with status, issues, and raw outputs.
    """
    start_time = time.time()

    # Pre-validation: ensure target path exists
    if not os.path.exists(target_path):
        return ToolResult(
            tool_name="Ruff",
            target_path=target_path,
            success=False,
            issues_found=False,
            exit_code=None,
            error_message=f"Target path does not exist: '{target_path}'",
            execution_time=0.0,
        )

    # Build safe argument list: shell=False
    base_cmd = custom_cmd if custom_cmd is not None else resolve_tool_command("ruff")
    # Note: --fix is deliberately NOT passed to protect source code from unwanted modifications
    cmd = list(base_cmd) + ["check", target_path, "--output-format=json"]

    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=timeout,
            check=False,
        )
        elapsed = time.time() - start_time
        exit_code = process.returncode
        stdout = process.stdout
        stderr = process.stderr

        # Ruff Exit Codes:
        # 0 = No violations detected
        # 1 = Violations detected
        # 2 = Error occurred (syntax error in target, invalid flag, etc.)
        if exit_code == 0:
            return ToolResult(
                tool_name="Ruff",
                target_path=target_path,
                success=True,
                issues_found=False,
                exit_code=0,
                issues=[],
                raw_stdout=stdout,
                raw_stderr=stderr,
                execution_time=elapsed,
            )
        elif exit_code == 1:
            # Code issues detected: tool execution itself was successful!
            try:
                issues = parse_ruff_json(stdout)
            except Exception as parse_err:
                # Fallback if JSON parsing fails unexpectedly
                return ToolResult(
                    tool_name="Ruff",
                    target_path=target_path,
                    success=False,
                    issues_found=True,
                    exit_code=1,
                    raw_stdout=stdout,
                    raw_stderr=stderr,
                    error_message=f"Failed to parse Ruff output as JSON: {parse_err}",
                    execution_time=elapsed,
                )

            return ToolResult(
                tool_name="Ruff",
                target_path=target_path,
                success=True,
                issues_found=len(issues) > 0,
                exit_code=1,
                issues=issues,
                raw_stdout=stdout,
                raw_stderr=stderr,
                execution_time=elapsed,
            )
        else:
            # Exit code 2 or other indicates tool execution failure
            err_msg = stderr.strip() or f"Ruff failed with process exit code {exit_code}"
            return ToolResult(
                tool_name="Ruff",
                target_path=target_path,
                success=False,
                issues_found=False,
                exit_code=exit_code,
                raw_stdout=stdout,
                raw_stderr=stderr,
                error_message=err_msg,
                execution_time=elapsed,
            )

    except FileNotFoundError as fnf_err:
        return ToolResult(
            tool_name="Ruff",
            target_path=target_path,
            success=False,
            issues_found=False,
            exit_code=None,
            error_message=(
                f"Ruff executable not found ({fnf_err}). "
                "Ensure ruff is installed in the current virtual environment."
            ),
            execution_time=time.time() - start_time,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool_name="Ruff",
            target_path=target_path,
            success=False,
            issues_found=False,
            exit_code=None,
            error_message=f"Ruff execution timed out after {timeout} seconds.",
            execution_time=time.time() - start_time,
        )
    except Exception as exc:
        return ToolResult(
            tool_name="Ruff",
            target_path=target_path,
            success=False,
            issues_found=False,
            exit_code=None,
            error_message=f"Unexpected error running Ruff: {type(exc).__name__}: {exc}",
            execution_time=time.time() - start_time,
        )


def run_bandit(
    target_path: str,
    custom_cmd: Optional[List[str]] = None,
    timeout: int = 120,
) -> ToolResult:
    """
    Execute Bandit security scanner safely on the target path using Python subprocess.

    Key behaviors:
      - Uses shell=False with argument list.
      - Bandit exit code 0: Clean code, no security findings. (success=True, issues_found=False)
      - Bandit exit code 1: Security findings detected. (success=True, issues_found=True)
      - Bandit exit code >= 2: Tool runtime error or fatal syntax problem. (success=False)
      - Untrusted code is never executed; Bandit only parses AST.

    Args:
        target_path: File or directory path to inspect.
        custom_cmd: Optional override for the base command executable.
        timeout: Maximum seconds to wait before terminating the subprocess.

    Returns:
        ToolResult instance with status, issues, and raw outputs.
    """
    start_time = time.time()

    # Pre-validation: ensure target path exists
    if not os.path.exists(target_path):
        return ToolResult(
            tool_name="Bandit",
            target_path=target_path,
            success=False,
            issues_found=False,
            exit_code=None,
            error_message=f"Target path does not exist: '{target_path}'",
            execution_time=0.0,
        )

    # Build safe argument list: shell=False
    base_cmd = custom_cmd if custom_cmd is not None else resolve_tool_command("bandit")

    # If the target is a directory, use recursive flag '-r'
    cmd = list(base_cmd) + ["-q"]
    if os.path.isdir(target_path):
        cmd.extend(["-r", target_path])
    else:
        cmd.append(target_path)
    cmd.extend(["-f", "json"])

    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=timeout,
            check=False,
        )
        elapsed = time.time() - start_time
        exit_code = process.returncode
        stdout = process.stdout
        stderr = process.stderr

        # Bandit Exit Codes:
        # 0 = No security issues detected
        # 1 = One or more security issues detected
        # 2 = Fatal error occurred during analysis
        if exit_code == 0:
            return ToolResult(
                tool_name="Bandit",
                target_path=target_path,
                success=True,
                issues_found=False,
                exit_code=0,
                issues=[],
                raw_stdout=stdout,
                raw_stderr=stderr,
                execution_time=elapsed,
            )
        elif exit_code == 1:
            # Security findings detected: tool execution itself was successful!
            try:
                issues = parse_bandit_json(stdout)
            except Exception as parse_err:
                return ToolResult(
                    tool_name="Bandit",
                    target_path=target_path,
                    success=False,
                    issues_found=True,
                    exit_code=1,
                    raw_stdout=stdout,
                    raw_stderr=stderr,
                    error_message=f"Failed to parse Bandit output as JSON: {parse_err}",
                    execution_time=elapsed,
                )

            return ToolResult(
                tool_name="Bandit",
                target_path=target_path,
                success=True,
                issues_found=len(issues) > 0,
                exit_code=1,
                issues=issues,
                raw_stdout=stdout,
                raw_stderr=stderr,
                execution_time=elapsed,
            )
        else:
            err_msg = stderr.strip() or f"Bandit failed with process exit code {exit_code}"
            return ToolResult(
                tool_name="Bandit",
                target_path=target_path,
                success=False,
                issues_found=False,
                exit_code=exit_code,
                raw_stdout=stdout,
                raw_stderr=stderr,
                error_message=err_msg,
                execution_time=elapsed,
            )

    except FileNotFoundError as fnf_err:
        return ToolResult(
            tool_name="Bandit",
            target_path=target_path,
            success=False,
            issues_found=False,
            exit_code=None,
            error_message=(
                f"Bandit executable not found ({fnf_err}). "
                "Ensure bandit is installed in the current virtual environment."
            ),
            execution_time=time.time() - start_time,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool_name="Bandit",
            target_path=target_path,
            success=False,
            issues_found=False,
            exit_code=None,
            error_message=f"Bandit execution timed out after {timeout} seconds.",
            execution_time=time.time() - start_time,
        )
    except Exception as exc:
        return ToolResult(
            tool_name="Bandit",
            target_path=target_path,
            success=False,
            issues_found=False,
            exit_code=None,
            error_message=f"Unexpected error running Bandit: {type(exc).__name__}: {exc}",
            execution_time=time.time() - start_time,
        )
