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
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SeverityLevel(str, Enum):
    """Structured severity levels for code review findings."""

    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

    @classmethod
    def from_str(cls, val: Any) -> SeverityLevel:
        """Parse arbitrary severity string or enum into canonical SeverityLevel."""
        if isinstance(val, SeverityLevel):
            return val
        s = str(val or "").strip().upper()
        if s in ("CRITICAL", "FATAL", "BLOCKER"):
            return cls.CRITICAL
        if s in ("HIGH", "ERROR"):
            return cls.HIGH
        if s in ("MEDIUM", "WARN", "WARNING", "MODERATE"):
            return cls.MEDIUM
        if s in ("LOW", "INFO", "NOTE", "INFORMATIONAL"):
            return cls.LOW
        return cls.MEDIUM


def normalize_severity(val: Any) -> str:
    """Normalize arbitrary severity strings or enums to canonical: Critical, High, Medium, Low."""
    return SeverityLevel.from_str(val).value


@dataclass
class Issue:
    """
    Represents an individual lint, security, or semantic issue identified by a scanner.

    Attributes:
        code: Identifier of the rule or violation (e.g. 'F401', 'B101', 'SEM001').
        message: Human-readable description of what was detected.
        file_path: Relative or absolute path to the file containing the issue.
        line: 1-indexed line number where the issue occurred, if available.
        column: 1-indexed column number where the issue occurred, if available.
        severity: Severity level (e.g. 'LOW', 'MEDIUM', 'HIGH', 'ERROR', 'WARNING', 'CRITICAL').
        tool: Name of the tool that detected the issue ('ruff', 'bandit', 'semantic_analyzer').
        fixable: Boolean flag indicating if an automated fix is known to exist.
        confidence: Categorical confidence level ('HIGH', 'MEDIUM', 'LOW', 'CRITICAL').
        explanation: Additional narrative explanation of the finding.
        suggested_fix: Actionable, practical guidance or code to remediate the issue.
        confidence_score: Numeric confidence score between 0.0 and 1.0.
        what: Precise description of what the issue is.
        why: Explanation of why the issue matters for security/reliability/maintainability.
        evidence: Code context, line, or snippet demonstrating the issue.
    """
    code: str
    message: str
    file_path: str
    line: int | None = None
    column: int | None = None
    severity: str = "WARNING"
    tool: str = "unknown"
    fixable: bool = False
    confidence: str = "HIGH"
    explanation: str = ""
    suggested_fix: str = ""
    confidence_score: float | None = None
    what: str = ""
    why: str = ""
    evidence: str = ""

    def __post_init__(self) -> None:
        # 1. Synchronize confidence and confidence_score (0.0 to 1.0)
        if self.confidence_score is None:
            c = str(self.confidence or "").strip().upper()
            if c in ("CRITICAL", "VERY_HIGH"):
                self.confidence_score = 1.0
            elif c == "HIGH":
                self.confidence_score = 0.9
            elif c == "MEDIUM":
                self.confidence_score = 0.6
            elif c == "LOW":
                self.confidence_score = 0.3
            else:
                self.confidence_score = 0.5
        else:
            self.confidence_score = max(0.0, min(1.0, float(self.confidence_score)))
            # If confidence is default "HIGH" but score was explicitly low, adjust category
            if self.confidence == "HIGH" and self.confidence_score < 0.5:
                self.confidence = "LOW"
            elif self.confidence == "HIGH" and self.confidence_score < 0.8:
                self.confidence = "MEDIUM"

        # 2. Populate 4-part structured explanation fields if omitted
        if not self.what:
            self.what = self.message or "Issue detected."
        if not self.why:
            self.why = self.explanation
        if not self.evidence:
            loc = f"{self.file_path}:{self.line}" if self.line else self.file_path
            if self.column:
                loc = f"{loc}:{self.column}"
            self.evidence = loc

    @property
    def severity_level(self) -> SeverityLevel:
        """Return canonical structured SeverityLevel enum (Critical, High, Medium, Low)."""
        return SeverityLevel.from_str(self.severity)

    def format_explanation(self) -> str:
        """Format 4-part structured finding explanation: what, why, evidence, suggested fix."""
        parts = [f"**What:** {self.what}"]
        if self.why:
            parts.append(f"**Why it matters:** {self.why}")
        if self.evidence:
            parts.append(f"**Evidence:** {self.evidence}")
        if self.suggested_fix:
            parts.append(f"**Suggested Fix:** {self.suggested_fix}")
        return "\n\n".join(parts)


def format_finding_explanation(issue: Issue | dict[str, Any]) -> str:
    """Format finding explanation with what, why, evidence, and practical suggested fix."""
    if isinstance(issue, Issue):
        return issue.format_explanation()
    what = issue.get("what") or issue.get("message", "Issue detected.")
    why = issue.get("why") or issue.get("explanation", "")
    evidence = issue.get("evidence")
    if not evidence:
        fp = issue.get("file_path", "")
        ln = issue.get("line")
        col = issue.get("column")
        loc = f"{fp}:{ln}" if ln else fp
        if col:
            loc = f"{loc}:{col}"
        evidence = loc
    fix = issue.get("suggested_fix", "")

    parts = [f"**What:** {what}"]
    if why:
        parts.append(f"**Why it matters:** {why}")
    if evidence:
        parts.append(f"**Evidence:** {evidence}")
    if fix:
        parts.append(f"**Suggested Fix:** {fix}")
    return "\n\n".join(parts)


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
    exit_code: int | None
    issues: list[Issue] = field(default_factory=list)
    raw_stdout: str = ""
    raw_stderr: str = ""
    error_message: str | None = None
    execution_time: float = 0.0


def resolve_tool_command(tool_name: str) -> list[str]:
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
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Look for outer JSON object: {...}
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = stripped[first_brace:last_brace + 1]
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Look for outer JSON array: [...]
    first_bracket = stripped.find("[")
    last_bracket = stripped.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        candidate = stripped[first_bracket:last_bracket + 1]
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    return stripped



def parse_ruff_json(stdout_text: str) -> list[Issue]:
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

    issues: list[Issue] = []
    for item in data:
        loc = item.get("location") or {}
        fix = item.get("fix")
        code = item.get("code", "UNKNOWN")
        msg = item.get("message", "No description provided")
        filename = item.get("filename", "")
        row = loc.get("row")
        col = loc.get("column")
        loc_str = f"{filename}:{row}:{col}" if row else filename
        fix_suggestion = "Apply automated Ruff fix (--fix)." if fix else "Refactor code to satisfy linter rule."
        issues.append(
            Issue(
                code=code,
                message=msg,
                file_path=filename,
                line=row,
                column=col,
                severity=str(item.get("severity", "error")).upper(),
                tool="ruff",
                fixable=bool(fix),
                confidence="HIGH",
                confidence_score=0.95,
                what=msg,
                why=f"Ruff linter violation for rule {code}.",
                evidence=loc_str,
                suggested_fix=fix_suggestion,
            )
        )
    return issues


def parse_bandit_json(stdout_text: str) -> list[Issue]:
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
    issues: list[Issue] = []
    conf_map = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
    for item in results:
        raw_sev = str(item.get("issue_severity", "MEDIUM")).upper()
        raw_conf = str(item.get("issue_confidence", "HIGH")).upper()
        score = conf_map.get(raw_conf, 0.7)
        code_snippet = item.get("code", "").strip()
        filename = item.get("filename", "")
        line_num = item.get("line_number")
        col_offset = item.get("col_offset")
        test_id = item.get("test_id", "SECURITY_ISSUE")
        msg = item.get("issue_text", "No description provided")
        evidence_str = f"{filename}:{line_num}" if line_num else filename
        if code_snippet:
            evidence_str = f"{evidence_str}\n```python\n{code_snippet}\n```"
        more_info = item.get("more_info", "")
        fix_str = f"Remediate security vulnerability per Bandit advisory: {more_info}" if more_info else "Apply defensive validation, parameterized inputs, or safe APIs."
        issues.append(
            Issue(
                code=test_id,
                message=msg,
                file_path=filename,
                line=line_num,
                column=col_offset,
                severity=raw_sev,
                tool="bandit",
                fixable=False,
                confidence=raw_conf,
                confidence_score=score,
                what=msg,
                why=f"Bandit security finding: potential {test_id} vulnerability with {raw_sev} severity.",
                evidence=evidence_str,
                suggested_fix=fix_str,
            )
        )
    return issues


def run_ruff(
    target_path: str,
    custom_cmd: list[str] | None = None,
    timeout: int = 60,
    select_rules: list[str] | tuple[str, ...] | None = None,
) -> ToolResult:
    """
    Execute Ruff linter safely on the target path using Python subprocess.

    Key behaviors:
      - Uses shell=False with argument list.
      - Enforces '--isolated' to prevent untrusted repo configs (pyproject.toml/ruff.toml) from overriding rules.
      - Optionally accepts trusted rule sets via select_rules (e.g. ['E', 'F', 'B']).
      - Ruff exit code 0: Clean code, no lint issues. (success=True, issues_found=False)
      - Ruff exit code 1: Lint issues found. (success=True, issues_found=True)
      - Ruff exit code >= 2: Tool runtime error or invalid config. (success=False)
      - Exceptions (FileNotFoundError, Timeout, etc.) mark success=False with message.

    Args:
        target_path: File or directory path to inspect.
        custom_cmd: Optional override for the base command executable.
        timeout: Maximum seconds to wait before terminating the subprocess.
        select_rules: Optional list/tuple of rule codes to enforce (passed via trusted --select CLI argument).

    Returns:
        ToolResult instance with status, issues, and raw outputs.
    """
    start_time = time.time()

    # Pre-validation: ensure target path is non-empty, contains no null bytes, and exists
    if not target_path or "\x00" in target_path or not os.path.exists(target_path):
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
    safe_path = target_path
    if safe_path.startswith("-"):
        safe_path = f".{os.sep}{safe_path}"
    # Use '--isolated' to prevent untrusted repo configs (pyproject.toml/ruff.toml) from overriding rules
    # Use '--' to prevent paths starting with a dash or flag-like names from being parsed as options
    cmd = list(base_cmd) + ["check", "--output-format=json", "--isolated"]
    if select_rules:
        cmd.append(f"--select={','.join(select_rules)}")
    cmd.extend(["--", safe_path])

    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
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
            except (json.JSONDecodeError, ValueError, KeyError) as parse_err:
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
    except Exception as exc:  # noqa: BLE001 - tool runner exception containment
        return ToolResult(
            tool_name="Ruff",
            target_path=target_path,
            success=False,
            issues_found=False,
            exit_code=None,
            error_message=f"Unexpected error running Ruff: {type(exc).__name__}: {exc}",
            execution_time=time.time() - start_time,
        )



def is_trusted_config_path(
    candidate: str | None,
    trusted_roots: list[str] | tuple[str, ...] | None = None,
) -> bool:
    """
    Verify that a configuration path resolves strictly within maintainer-controlled trusted directories.

    Prevents untrusted PR repositories or arbitrary working directories from supplying malicious scanner configs
    (e.g., bandit.yaml or .bandit) to suppress security rules.

    Args:
        candidate: File path to validate.
        trusted_roots: Directory paths considered trusted (defaults to the pipeline package directory).

    Returns:
        True if the candidate is an existing regular file within a trusted root; False otherwise.
    """
    if not candidate or not isinstance(candidate, str):
        return False

    project_dir = os.path.dirname(os.path.abspath(__file__))
    roots = list(trusted_roots) if trusted_roots is not None else [project_dir]

    try:
        abs_candidate = os.path.abspath(os.path.realpath(candidate))
        if not os.path.isfile(abs_candidate):
            return False

        for root in roots:
            abs_root = os.path.abspath(os.path.realpath(root))
            try:
                common = os.path.commonpath([abs_candidate, abs_root])
                if common == abs_root:
                    return True
            except ValueError:
                # Handles path comparison across different drive letters on Windows (e.g. C: vs E:)
                continue
        return False
    except (OSError, ValueError):
        return False


def run_bandit(
    target_path: str,
    custom_cmd: list[str] | None = None,
    timeout: int = 120,
    config_path: str | None = None,
    trusted_dirs: list[str] | tuple[str, ...] | None = None,
) -> ToolResult:
    """
    Execute Bandit security scanner safely on the target path using Python subprocess.

    Key behaviors:
      - Uses shell=False with argument list.
      - Enforces strict configuration trust boundaries (SEC-01):
        * Never discovers configuration from the untrusted PR target_path.
        * Never implicitly trusts os.getcwd().
        * Discovers configuration solely from verified maintainer directory (project_dir).
        * Validates any explicit config_path against trusted_dirs (project_dir by default).
      - Bandit exit code 0: Clean code, no security findings. (success=True, issues_found=False)
      - Bandit exit code 1: Security findings detected. (success=True, issues_found=True)
      - Bandit exit code >= 2: Tool runtime error or fatal syntax problem. (success=False)
      - Untrusted code is never executed; Bandit only parses AST.

    Args:
        target_path: File or directory path to inspect.
        custom_cmd: Optional override for the base command executable.
        timeout: Maximum seconds to wait before terminating the subprocess.
        config_path: Optional explicit path to Bandit YAML configuration file.
        trusted_dirs: Optional trusted root directories for config validation (defaults to pipeline project_dir).

    Returns:
        ToolResult instance with status, issues, and raw outputs.
    """
    start_time = time.time()

    # Pre-validation: ensure target path is non-empty, contains no null bytes, and exists
    if not target_path or "\x00" in target_path or not os.path.exists(target_path):
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

    safe_path = target_path
    if safe_path.startswith("-"):
        safe_path = f".{os.sep}{safe_path}"

    project_dir = os.path.dirname(os.path.abspath(__file__))
    effective_trusted_roots = list(trusted_dirs) if trusted_dirs is not None else [project_dir]

    # Resolve Bandit configuration file (only trusted maintainer locations, never untrusted target_path or os.getcwd())
    resolved_config: str | None = None
    if config_path:
        if is_trusted_config_path(config_path, trusted_roots=effective_trusted_roots):
            resolved_config = os.path.abspath(os.path.realpath(config_path))
        else:
            logger.warning(
                "Untrusted Bandit configuration path rejected: '%s' is outside trusted boundary %s. "
                "Falling back to maintainer pipeline configuration.",
                config_path,
                effective_trusted_roots,
            )
            resolved_config = None

    if not resolved_config:
        # Only search verified maintainer-controlled locations (project_dir)
        # NEVER search untrusted target_path and NEVER implicitly trust os.getcwd()
        candidates = [
            os.path.join(project_dir, "bandit.yaml"),
            os.path.join(project_dir, ".bandit"),
        ]
        for cand in candidates:
            if is_trusted_config_path(cand, trusted_roots=effective_trusted_roots):
                resolved_config = cand
                break

    # Use '--' to prevent paths from being parsed as CLI options
    cmd = list(base_cmd)
    if resolved_config and os.path.isfile(resolved_config):
        cmd.extend(["-c", resolved_config])
    cmd.extend(["-q", "-f", "json"])
    if os.path.isdir(target_path):
        cmd.extend(["-r", "--", safe_path])
    else:
        cmd.extend(["--", safe_path])

    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
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
            except (json.JSONDecodeError, ValueError, KeyError) as parse_err:
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
    except Exception as exc:  # noqa: BLE001 - tool runner exception containment
        return ToolResult(
            tool_name="Bandit",
            target_path=target_path,
            success=False,
            issues_found=False,
            exit_code=None,
            error_message=f"Unexpected error running Bandit: {type(exc).__name__}: {exc}",
            execution_time=time.time() - start_time,
        )
