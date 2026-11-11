"""
main.py - Entry Point for the Multi-Agent Code Review & Debugging Pipeline (Milestone 3).

Integrates:
  1. Deterministic Static Scanners: Ruff (linting) and Bandit (security).
  2. LangGraph AI Multi-Agent Workflow:
     - Linter Agent: Beginner-friendly explanations with advisory caveats.
     - Security Agent: Risk modeling and remediation distinguishing confirmed findings
       from AI hypotheses, emphasizing that clean scans do not imply total security.
     - Test-Writer Agent: Validated pytest suggestions stored in isolated temp workspace.
     - Sandbox Runner: Docker-based isolated test execution (skipped if Docker unavailable).
     - Debugging Agent: Minimal patch proposals for failed tests (applied to disposable copy).
  3. Graceful Degradation: If API credentials are not configured or if --no-ai is supplied,
     deterministic analysis executes cleanly without interruption.
  4. Docker Unavailability: Sandbox is silently skipped; all other agents still run.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from llm_config import is_api_key_configured
from sandbox_runner import is_docker_available
from semantic_analyzer import run_semantic_analysis
from tools import ToolResult, normalize_severity, run_bandit, run_ruff
from workflow import run_review_pipeline

logger = logging.getLogger(__name__)


def format_separator(char: str = "=", length: int = 78) -> str:
    """Return a horizontal separator line."""
    return char * length


def print_banner() -> None:
    """Print the application welcome banner."""
    print(format_separator("="))
    print("  MULTI-AGENT CODE REVIEW & DEBUGGING PIPELINE")
    print("  Milestone 3: Test Generation, Sandboxed Execution & Verified Fix Proposals")
    print(format_separator("="))
    print()


def display_tool_summary(results: dict[str, ToolResult]) -> None:
    """Print an executive summary table of all tool execution results."""
    print("OVERVIEW SUMMARY:")
    print("-" * 78)
    print(f"{'Tool':<12} | {'Execution':<15} | {'Issues Found':<14} | {'Duration':<10} | {'Status'}")
    print("-" * 78)

    for tool_name, result in results.items():
        exec_status = "SUCCESS" if result.success else "FAILED"
        issue_count = str(len(result.issues)) if result.success else "N/A"
        duration_str = f"{result.execution_time:.2f}s"

        if not result.success:
            verdict = "TOOL_ERROR"
        elif result.issues_found:
            verdict = "WARNINGS"
        else:
            verdict = "PASSED"

        print(
            f"{tool_name:<12} | {exec_status:<15} | {issue_count:<14} | {duration_str:<10} | {verdict}"
        )
    print("-" * 78)
    print()


def display_tool_findings(result: ToolResult) -> None:
    """Print the detailed findings for a single tool."""
    print(format_separator("-"))
    print(f"Tool Report: {result.tool_name} (Exit Code: {result.exit_code})")
    print(format_separator("-"))

    # Case 1: The tool failed to execute properly
    if not result.success:
        print(f"  [!] EXECUTION ERROR: {result.tool_name} could not complete its analysis.")
        print(f"      Details: {result.error_message}")
        print()
        return

    # Case 2: The tool executed and found zero issues
    if not result.issues_found:
        print(f"  [OK] CLEAN: No code or security violations detected by {result.tool_name}.")
        print()
        return

    # Case 3: The tool executed and identified code issues
    print(f"  Found {len(result.issues)} issue(s) in '{result.target_path}':\n")
    for idx, issue in enumerate(result.issues, start=1):
        # Format location (file:line:col)
        loc = issue.file_path
        if issue.line is not None:
            loc += f":{issue.line}"
            if issue.column is not None:
                loc += f":{issue.column}"

        # Severity tag
        sev_tag = f"[{issue.severity}]"

        print(f"  {idx}. {sev_tag:<10} Rule: {issue.code}")
        print(f"     Location : {loc}")
        print(f"     Message  : {issue.message}")
        if issue.fixable:
            print("     Note     : Safe auto-fix is available (deferred to future milestone).")
        print()


def display_ai_explanations(ai_explanations: dict[str, str]) -> None:
    """Print AI agent diagnostic explanations."""
    print(format_separator("="))
    print("SECTION: AI AGENT EXPLANATIONS & THREAT MODELING")
    print(format_separator("="))
    print()

    # Linter Agent
    print("[1/2] Linter Agent (Beginner-Friendly Code Analysis):")
    print("-" * 78)
    linter_text = ai_explanations.get("linter", "No explanation available.")
    print(linter_text)
    print()

    # Security Agent
    print("[2/2] Security Agent (Vulnerability Analysis & Hypotheses):")
    print("-" * 78)
    security_text = ai_explanations.get("security", "No explanation available.")
    print(security_text)
    print()


def display_test_suggestions(suggestions: list[dict[str, Any]]) -> None:
    """Print test-writer agent suggestions."""
    print(format_separator("="))
    print("SECTION: AI TEST-WRITER AGENT SUGGESTIONS (FOR HUMAN REVIEW ONLY)")
    print(format_separator("="))
    print()

    if not suggestions:
        print("  No test suggestions generated.")
        print()
        return

    for idx, item in enumerate(suggestions, start=1):
        target = item.get("target_file", "unknown")
        content = item.get("suggestions", "")
        stored_path = item.get("stored_test_path")
        val_err = item.get("validation_error")

        print(f"Target File #{idx}: {target}")
        print("-" * 78)
        print(content)
        if stored_path:
            print(f"\n  [+] Validated test stored at: {stored_path}")
        if val_err:
            print(f"\n  [!] Validation Warning: {val_err}")
        print()


def display_sandbox_results(
    sandbox_results: list[dict[str, Any]],
    sandbox_skipped: bool,
) -> None:
    """Print sandbox execution results."""
    print(format_separator("="))
    print("SECTION: SANDBOXED TEST EXECUTION")
    print(format_separator("="))
    print()

    if sandbox_skipped or not sandbox_results:
        # Check if Docker was the cause
        docker_unavail = any(r.get("docker_unavailable", False) for r in sandbox_results)
        if docker_unavail:
            print("  [SKIPPED] Docker is not available on this host.")
            print("  To enable sandboxed test execution, install Docker Desktop or Docker Engine.")
        elif not sandbox_results:
            print("  [SKIPPED] No stored test files were available for sandbox execution.")
        else:
            print("  [SKIPPED] Sandbox execution was bypassed.")
        print()
        return

    for idx, result in enumerate(sandbox_results, start=1):
        executed = result.get("executed", False)
        passed = result.get("passed", False)
        timed_out = result.get("timed_out", False)
        exit_code = result.get("exit_code")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        err_msg = result.get("error_message")

        print(f"Sandbox Run #{idx}:")
        if not executed:
            print("  Status   : NOT EXECUTED")
            if err_msg:
                print(f"  Reason   : {err_msg}")
        elif timed_out:
            print("  Status   : TIMED OUT")
            print(f"  Reason   : {err_msg or 'Execution exceeded time limit.'}")
        elif passed:
            print("  Status   : [PASSED] All tests passed (exit code 0)")
        else:
            print(f"  Status   : [FAILED] Tests failed (exit code {exit_code})")

        if stdout:
            print("\n  --- pytest stdout ---")
            # Show first 2000 chars
            print(stdout[:2000].strip())
            if len(stdout) > 2000:
                print(f"  ... [{len(stdout) - 2000} bytes truncated]")
        if stderr:
            print("\n  --- stderr ---")
            print(stderr[:500].strip())
        print()


def display_patch_proposals(
    proposals: list[dict[str, Any]],
    patch_verified: bool,
) -> None:
    """Print debugging agent patch proposals."""
    print(format_separator("="))
    print("SECTION: DEBUGGING AGENT — PATCH PROPOSALS")
    print(format_separator("="))
    print()

    if not proposals:
        print("  No patch proposals generated.")
        print()
        return

    if patch_verified:
        print("  [✓] At least one proposed patch passed sandbox verification.\n")

    for proposal in proposals:
        attempt = proposal.get("attempt_number", "?")
        target = proposal.get("target_file", "unknown")
        diff = proposal.get("unified_diff", "")
        patch_ok = proposal.get("patch_succeeded", False)
        copy_path = proposal.get("patch_applied_to_copy")
        err_msg = proposal.get("error_message")
        disclaimer = proposal.get("disclaimer", "")
        verification = proposal.get("verification_result")

        print(f"Patch Attempt #{attempt} — Target: {target}")
        print("-" * 78)

        if err_msg:
            print(f"  [!] Error: {err_msg}")

        if diff:
            print("\n  Proposed Patch (unified diff):\n")
            print(diff[:3000])
            if len(diff) > 3000:
                print(f"  ... [{len(diff) - 3000} chars truncated]")

        if patch_ok and copy_path:
            print(f"\n  [+] Patch applied to disposable copy at: {copy_path}")
        elif not patch_ok and not err_msg:
            print("\n  [-] Patch could not be applied cleanly to disposable copy.")

        if verification:
            v_passed = verification.get("passed", False)
            v_timed_out = verification.get("timed_out", False)
            v_docker = verification.get("docker_unavailable", False)
            if v_docker:
                print("  Verification: [SKIPPED] Docker unavailable")
            elif v_timed_out:
                print("  Verification: [TIMED OUT]")
            elif v_passed:
                print("  Verification: [PASSED] Tests pass against patched copy")
            else:
                print("  Verification: [FAILED] Tests still fail after patch")

        print(f"\n  {disclaimer}")
        print()


def display_agent_errors(errors: list[str]) -> None:
    """Print any provider or agent execution errors."""
    if not errors:
        return

    print(format_separator("!"))
    print("SECTION: AGENT & PROVIDER WARNINGS / ERRORS")
    print(format_separator("!"))
    for err in errors:
        print(f"  [!] {redact_token(err)}")
    print()


def run_pipeline(
    target_path: str,
    enable_ai: bool = True,
    enable_sandbox: bool = True,
    enable_debugging: bool = True,
    llm: BaseChatModel | None = None,
) -> int:
    """
    Run the multi-agent code review pipeline on target_path.

    Workflow:
      1. Validates target path existence.
      2. Runs deterministic static analysis (Ruff & Bandit).
      3. Displays tool overview summary and detailed findings.
      4. If enable_ai=True and credentials available, runs the LangGraph workflow:
         a. Linter Agent & Security Agent (parallel)
         b. Test-Writer Agent (with validated test storage)
         c. Sandbox Runner (Docker, if available)
         d. Debugging Agent (if tests failed, bounded retry)
      5. Displays all sections.
      6. Returns exit status:
           0 = Clean code, no issues found.
           1 = Code violations or security findings detected.
           2 = Runtime/execution failure of tools.
    """
    print_banner()

    # Normalize target path
    normalized_path = os.path.normpath(target_path)
    print(f"Target Under Review: {os.path.abspath(normalized_path)}")
    print()

    if not os.path.exists(normalized_path):
        print(f"ERROR: Target path '{normalized_path}' does not exist.")
        return 2

    # Announce Docker status
    docker_available = is_docker_available()
    if enable_ai and enable_sandbox:
        docker_status = "AVAILABLE" if docker_available else "NOT AVAILABLE (sandbox will be skipped)"
        print(f"Docker: {docker_status}")
        print()

    # Step 1: Run deterministic Ruff linter
    print("[1/3] Running Ruff Linter...")
    ruff_result = run_ruff(normalized_path)

    # Step 2: Run deterministic Bandit security scanner
    print("[2/3] Running Bandit Security Scanner...")
    bandit_result = run_bandit(normalized_path)

    # Step 3: Run deterministic Semantic Bug Analyzer
    print("[3/3] Running Semantic Bug Analyzer...")
    semantic_result = run_semantic_analysis(normalized_path)
    print()

    results: dict[str, ToolResult] = {
        "Ruff": ruff_result,
        "Bandit": bandit_result,
        "Semantic": semantic_result,
    }

    # Display Deterministic Summary & Detailed Findings
    display_tool_summary(results)

    print("DETAILED DETERMINISTIC FINDINGS:")
    for result in results.values():
        display_tool_findings(result)

    # Step 3: AI Multi-Agent Workflow
    has_credentials = is_api_key_configured() or llm is not None

    if not enable_ai:
        print(format_separator("="))
        print("AI MULTI-AGENT REVIEW: [DISABLED]")
        print("Deterministic scanners executed only (--no-ai specified).")
        print(format_separator("="))
        print()
    elif not has_credentials:
        print(format_separator("="))
        print("AI MULTI-AGENT REVIEW: [UNAVAILABLE - CREDENTIALS MISSING]")
        print("Deterministic analysis completed successfully above.")
        print("To enable AI explanations, security threat modeling, and test suggestions:")
        print("  1. Copy '.env.example' to '.env'")
        print("  2. Set your OPENAI_API_KEY")
        print("  3. Run the pipeline again")
        print(format_separator("="))
        print()
    else:
        # Run LangGraph workflow with deterministic runners injected
        print("Running AI Multi-Agent Workflow (LangGraph)...")
        workflow_state = run_review_pipeline(
            target_path=normalized_path,
            enable_ai=True,
            llm=llm,
            ruff_runner=lambda p: ruff_result,
            bandit_runner=lambda p: bandit_result,
            enable_sandbox=enable_sandbox,
            enable_debugging=enable_debugging,
        )

        # Display AI explanations
        display_ai_explanations(workflow_state.get("ai_explanations", {}))

        # Display Test suggestions
        display_test_suggestions(workflow_state.get("generated_test_suggestions", []))

        # Display Sandbox results
        display_sandbox_results(
            sandbox_results=workflow_state.get("sandbox_results", []),
            sandbox_skipped=workflow_state.get("sandbox_skipped", False),
        )

        # Display Patch proposals
        display_patch_proposals(
            proposals=workflow_state.get("patch_proposals", []),
            patch_verified=workflow_state.get("patch_verified", False),
        )

        # Display Errors if any
        display_agent_errors(workflow_state.get("agent_errors", []))

    # Overall Verdict Calculation
    has_execution_error = any(not r.success for r in results.values())
    has_code_issues = any(r.issues_found for r in results.values())

    print(format_separator("="))
    if has_execution_error:
        print("PIPELINE RESULT: [FAILED] One or more tools encountered execution errors.")
        print("Please check the details above to fix tool installation or environment issues.")
        print(format_separator("="))
        return 2
    elif has_code_issues:
        total_issues = sum(len(r.issues) for r in results.values())
        print(f"PIPELINE RESULT: [ATTENTION REQUIRED] {total_issues} issue(s) detected.")
        print("Static analysis completed successfully.")
        print(format_separator("="))
        return 1
    else:
        print("PIPELINE RESULT: [PASSED] All checks passed with zero issues.")
        print(format_separator("="))
        return 0


from approval_gate import (
    ApprovalGateError,
    ApprovalRejectedError,
    NonInteractivePublishingDisabledError,
)
from github_client import (
    GitHubClient,
    GitHubClientError,
    GitHubUncertainWriteError,
    GitHubValidationError,
    redact_token,
    register_sensitive_token,
)
from github_review_workflow import (
    PrePublicationValidationError,
    PRReviewWorkflowError,
    run_pr_publishing_workflow,
    run_pr_review_workflow,
)


def _escape_markdown_table_cell(text: str) -> str:
    """
    Escape PR-derived text placed inside a GitHub Flavored Markdown table cell.

    Guarantees:
      - Flattens newlines, carriage returns, and control characters to spaces.
      - Escapes table column delimiters ('|' -> '\\|') to prevent column splitting.
      - Neutralizes raw HTML tags ('<' -> '&lt;', '>' -> '&gt;').
      - Neutralizes markdown image embeds (![alt](url)) and links ([text](url)) by escaping brackets
        ('\\[', '\\]') to prevent external asset requests, tracking pixels, or phishing links in CI.
      - Replaces backticks with apostrophes to prevent breaking out of code spans.
    """
    if not text:
        return ""

    # 1. Flatten newlines and control characters to prevent multi-line table breakdown
    cleaned = "".join(c if (ord(c) >= 32 or c in "\t") else " " for c in text)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # 2. Escape backslashes first
    cleaned = cleaned.replace("\\", "\\\\")

    # 3. Escape pipes to prevent table column injection
    cleaned = cleaned.replace("|", "\\|")

    # 4. Neutralize HTML tags
    cleaned = cleaned.replace("<", "&lt;").replace(">", "&gt;")

    # 5. Neutralize markdown image and link brackets
    cleaned = cleaned.replace("[", "\\[").replace("]", "\\]")

    # 6. Neutralize backticks to prevent breaking inline code formatting
    cleaned = cleaned.replace("`", "'")

    return cleaned.strip()


def _sanitize_markdown_text(text: str) -> str:
    """
    Sanitize general PR-derived markdown text (e.g. summaries).

    Disarms raw HTML tags, tracking pixels, and external image embeds.
    """
    if not text:
        return ""
    # Neutralize HTML script, iframe, object, and img tags
    sanitized = re.sub(
        r"<\s*(script|iframe|object|embed|img|link|style)\b[^>]*>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Neutralize markdown image embeds (![...](...)) -> [Image Disabled: ...]
    sanitized = re.sub(
        r"!\[(.*?)\]\([^\)]+\)",
        r"[Image Disabled: \1]",
        sanitized,
    )
    return sanitized


def _extract_comment_severity(comment: Any) -> str:
    """Extract or infer normalized severity (Critical, High, Medium, Low) for an inline comment."""
    sev = getattr(comment, "severity", None)
    if sev:
        return normalize_severity(sev)
    body = getattr(comment, "body", "") or ""
    m = re.search(r"\((CRITICAL|HIGH|MEDIUM|LOW|ERROR|WARNING|FATAL)\b", body, re.IGNORECASE)
    if m:
        return normalize_severity(m.group(1))
    return "Medium"


def format_github_step_summary(result: Any) -> str:
    """
    Format a GitHub Actions step summary markdown document from a PR review result.

    Includes:
      - Repository, PR number, target commit SHA, and review verdict.
      - Executive summary of findings (from generate_review_summary).
      - Table of findings grouped by severity (Critical, High, Medium, Low).
      - Table of proposed inline comments grouped by severity tier.
      - Required advisory notice.
    """
    owner = getattr(result, "owner", "")
    repo = getattr(result, "repo", "")
    pr_number = getattr(result, "pr_number", "")
    head_sha = getattr(result, "head_sha", "")
    action_plan = getattr(result, "action_plan", None)
    verdict = action_plan.verdict if action_plan else "COMMENT"
    summary_text = getattr(result, "summary", "") or (action_plan.summary if action_plan else "")
    comments = getattr(result, "review_comments", None) or (action_plan.comments if action_plan else ())

    # Clean publication marker if present in summary_text, and sanitize external embeds
    clean_summary = re.sub(r"<!-- agy-review-marker: [^>]+ -->", "", summary_text).strip()
    clean_summary = _sanitize_markdown_text(clean_summary)

    safe_owner = _escape_markdown_table_cell(str(owner))
    safe_repo = _escape_markdown_table_cell(str(repo))
    safe_pr_number = _escape_markdown_table_cell(str(pr_number))
    safe_head_sha = _escape_markdown_table_cell(str(head_sha))
    safe_verdict = _escape_markdown_table_cell(str(verdict))

    target_link = (
        f"[#{safe_pr_number}](https://github.com/{safe_owner}/{safe_repo}/pull/{safe_pr_number})"
        if owner and repo and pr_number
        else f"#{safe_pr_number}"
    )
    sha_display = f"`{safe_head_sha[:10]}` (`{safe_head_sha}`)" if safe_head_sha else "N/A"

    lines: list[str] = [
        f"## 🤖 Multi-Agent PR Review: #{safe_pr_number}",
        "",
        "| Repository | Pull Request | Target Head SHA | Review Verdict |",
        "|:---|:---|:---|:---|",
        f"| `{safe_owner}/{safe_repo}` | {target_link} | {sha_display} | **{safe_verdict}** |",
        "",
    ]

    if clean_summary:
        lines.extend([
            clean_summary,
            "",
        ])

    if comments:
        # Group comments by structured severity tier
        tier_badges = [
            ("Critical", "🔴 Critical"),
            ("High", "🟠 High"),
            ("Medium", "🟡 Medium"),
            ("Low", "🔵 Low"),
        ]
        comments_by_sev: dict[str, list[Any]] = {
            "Critical": [],
            "High": [],
            "Medium": [],
            "Low": [],
        }
        for c in comments:
            tier = _extract_comment_severity(c)
            comments_by_sev.setdefault(tier, []).append(c)

        lines.extend([
            "### 📊 Findings by Severity",
            "",
            "| Severity Tier | Findings Count | Status |",
            "|:---|:---:|:---|",
            f"| 🔴 **Critical** | {len(comments_by_sev.get('Critical', []))} | {'Attention Required' if comments_by_sev.get('Critical') else 'Clean'} |",
            f"| 🟠 **High** | {len(comments_by_sev.get('High', []))} | {'Attention Required' if comments_by_sev.get('High') else 'Clean'} |",
            f"| 🟡 **Medium** | {len(comments_by_sev.get('Medium', []))} | {'Advisory' if comments_by_sev.get('Medium') else 'Clean'} |",
            f"| 🔵 **Low** | {len(comments_by_sev.get('Low', []))} | {'Informational' if comments_by_sev.get('Low') else 'Clean'} |",
            "",
            f"### Proposed Inline Comments ({len(comments)})",
            "",
        ])

        for tier_key, tier_label in tier_badges:
            tier_comments = comments_by_sev.get(tier_key, [])
            if not tier_comments:
                continue
            lines.extend([
                f"#### {tier_label} Severity ({len(tier_comments)})",
                "",
                "| File | Line | Side | Comment Preview |",
                "|:---|:---:|:---:|:---|",
            ])
            for c in tier_comments:
                raw_path = getattr(c, "path", "")
                c_path = _escape_markdown_table_cell(str(raw_path))
                c_line = _escape_markdown_table_cell(str(getattr(c, "line", None) or "file-level"))
                c_side = _escape_markdown_table_cell(str(getattr(c, "side", "RIGHT")))
                raw_body = getattr(c, "body", "").strip()
                clean_body = _escape_markdown_table_cell(raw_body)
                c_body = (clean_body[:97] + "...") if len(clean_body) > 100 else clean_body
                lines.append(f"| `{c_path}` | `{c_line}` | `{c_side}` | {c_body} |")
            lines.append("")

    # Ensure advisory notice is explicitly included if not already in summary
    if "Advisory Notice:" not in clean_summary:
        lines.extend([
            "> [!NOTE]",
            (
                "> **Advisory Notice:** Automated analysis findings, test generation, and AI explanations represent advisory hypotheses. "
                "Static analysis and sandboxed tests do not guarantee total runtime security. "
                "Human maintainer review is required before merging."
            ),
            "",
        ])

    return "\n".join(lines).strip() + "\n"


def write_github_step_summary(
    result: Any,
    token: str | None = None,
    step_summary_path: str | Path | None = None,
) -> bool:
    """
    Safely append a sanitized, token-redacted review summary to $GITHUB_STEP_SUMMARY.

    Guarantees:
      - If GITHUB_STEP_SUMMARY is not set or empty, silently returns False without error.
      - If GITHUB_STEP_SUMMARY cannot be written or opened (e.g. invalid path/permissions),
        catches the exception and returns False without crashing the CLI.
      - Automatically applies token redaction to all content before writing.
      - Appends to the summary file using UTF-8 encoding.

    Returns:
      True if the step summary was successfully written, False otherwise.
    """
    target_path = step_summary_path or os.getenv("GITHUB_STEP_SUMMARY")
    if not target_path or not str(target_path).strip():
        return False

    summary_file = Path(str(target_path).strip())

    try:
        summary_md = format_github_step_summary(result)
        # Apply token redaction for absolute credential safety
        safe_summary = redact_token(summary_md, token)

        # Open in append mode as expected by GitHub Actions Step Summaries
        with summary_file.open("a", encoding="utf-8") as f:
            f.write(f"\n{safe_summary}\n")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not write GITHUB_STEP_SUMMARY to '%s': %s", summary_file, exc)
        return False


def run_pr_pipeline(
    pr_target: str,
    repo: str | None = None,
    token: str | None = None,
    enable_ai: bool = True,
    enable_sandbox: bool = True,
    enable_debugging: bool = True,
    llm: BaseChatModel | None = None,
    publish: bool = False,
    dry_run: bool = False,
    interactive: bool | None = None,
    input_func: Callable[[str], str] | None = None,
    output_func: Callable[[str], None] | None = None,
    client: GitHubClient | None = None,
    clock: Callable[[], float] | None = None,
    utc_now: Callable[[], datetime] | None = None,
    publication_marker: str | None = None,
    ruff_runner: Callable[..., ToolResult] | None = None,
    bandit_runner: Callable[..., ToolResult] | None = None,
    semantic_runner: Callable[..., ToolResult] | None = None,
    sandbox_runner_fn: Callable[..., Any] | None = None,
    step_summary_path: str | Path | None = None,
) -> int:
    """
    Execute the PR review workflow or controlled publishing workflow for a GitHub PR.
    """
    out = output_func or print
    if token:
        register_sensitive_token(token)
    has_credentials = is_api_key_configured() or llm is not None
    effective_enable_ai = enable_ai and has_credentials
    if enable_ai and not has_credentials:
        out("[Notice] LLM credentials not configured. Running PR review in deterministic mode.")
        out("")

    if publish:
        out(format_separator("="))
        out("  MULTI-AGENT CODE REVIEW & DEBUGGING PIPELINE")
        if dry_run:
            out("  Milestone 4 — Stage 5: PR Review Publishing (Dry-Run Mode)")
        else:
            out("  Milestone 4 — Stage 5: Controlled Review Publication")
        out(format_separator("="))
        out("")
        out(f"Target PR: {pr_target} (Publishing {'Dry-Run' if dry_run else 'Controlled Mode'})")
        out(format_separator("-"))

        try:
            workflow_res = run_pr_publishing_workflow(
                repo_or_url=repo if repo else pr_target,
                pr_number=pr_target if repo else None,
                dry_run=dry_run,
                client=client,
                token=token,
                enable_ai=effective_enable_ai,
                llm=llm,
                enable_sandbox=enable_sandbox,
                enable_debugging=enable_debugging,
                interactive=interactive,
                input_func=input_func,
                output_func=out,
                clock=clock,
                utc_now=utc_now,
                publication_marker=publication_marker,
                ruff_runner=ruff_runner,
                bandit_runner=bandit_runner,
                semantic_runner=semantic_runner,
                sandbox_runner_fn=sandbox_runner_fn,
            )
        except NonInteractivePublishingDisabledError as e:
            safe_msg = redact_token(str(e), token)
            out(format_separator("="))
            out(f"PUBLISHING REJECTED (NON-INTERACTIVE): {safe_msg}")
            out(format_separator("="))
            return 2
        except ApprovalRejectedError as e:
            safe_msg = redact_token(str(e), token)
            out(format_separator("="))
            out(f"PUBLISHING CANCELLED: {safe_msg}")
            out(format_separator("="))
            return 1
        except GitHubValidationError as e:
            safe_msg = redact_token(str(e), token)
            out(format_separator("="))
            out(f"PUBLISHING FAILED (HTTP 422 VALIDATION ERROR): {safe_msg}")
            out(format_separator("="))
            return 2
        except GitHubUncertainWriteError as e:
            safe_msg = redact_token(str(e), token)
            out(format_separator("="))
            out(f"PUBLISHING UNCERTAIN (TIMEOUT/CONNECTION DROP): {safe_msg}")
            out("No retry performed. Reconcile PR reviews before proceeding.")
            out(format_separator("="))
            return 2
        except (PrePublicationValidationError, ApprovalGateError, GitHubClientError, PRReviewWorkflowError) as e:
            safe_msg = redact_token(str(e), token)
            out(format_separator("="))
            out(f"PUBLISHING FAILED: {safe_msg}")
            out(format_separator("="))
            return 2
        except Exception as e:  # noqa: BLE001
            safe_msg = redact_token(str(e), token)
            out(format_separator("="))
            out(f"PUBLISHING WORKFLOW ERROR: {safe_msg}")
            out(format_separator("="))
            return 2

        # Write GitHub Actions Step Summary if running in CI
        write_github_step_summary(
            workflow_res, token=token, step_summary_path=step_summary_path
        )

        if not workflow_res.success:
            return 2
        if workflow_res.action_plan.verdict == "REQUEST_CHANGES":
            return 1
        if workflow_res.action_plan.comments:
            return 1
        return 0

    # Default read-only PR preview mode
    out(format_separator("="))
    out("  MULTI-AGENT CODE REVIEW & DEBUGGING PIPELINE")
    out("  Milestone 4: Read-Only Pull Request Review & Approval Preview")
    out(format_separator("="))
    out("")
    out(f"Target PR: {pr_target} (Read-Only Preview Mode)")
    out(format_separator("-"))

    try:
        result = run_pr_review_workflow(
            repo_or_url=repo if repo else pr_target,
            pr_number=pr_target if repo else None,
            client=client,
            token=token,
            enable_ai=effective_enable_ai,
            llm=llm,
            enable_sandbox=enable_sandbox,
            enable_debugging=enable_debugging,
            output_func=out,
            display_preview=True,
            clock=clock,
            ruff_runner=ruff_runner,
            bandit_runner=bandit_runner,
            semantic_runner=semantic_runner,
            sandbox_runner_fn=sandbox_runner_fn,
        )
    except Exception as e:  # noqa: BLE001
        safe_msg = redact_token(str(e), token)
        out(format_separator("="))
        out(f"PR WORKFLOW FAILED: {safe_msg}")
        out(format_separator("="))
        return 2

    # Display errors / warnings in PR preview mode if any
    wf_state = getattr(result, "workflow_state", None) or {}
    agent_errors = wf_state.get("agent_errors", []) if isinstance(wf_state, dict) else []
    if agent_errors:
        display_agent_errors(agent_errors)

    # Write GitHub Actions Step Summary if running in CI
    write_github_step_summary(
        result, token=token, step_summary_path=step_summary_path
    )
    # Exit code:
    # A completed read-only review should not fail CI merely because issues were found.
    # Return 2 only when the review execution or workflow itself fails.
    raw_tool_status = getattr(result, "tool_status", None)
    if not isinstance(raw_tool_status, dict):
        raw_tool_status = wf_state.get("tool_status", {}) if isinstance(wf_state, dict) else {}
    tool_status = raw_tool_status if isinstance(raw_tool_status, dict) else {}

    has_tool_failure = (
        any(tool_status.get(t) is False for t in ("ruff", "bandit", "semantic"))
        if tool_status
        else False
    )
    if not result.success or has_tool_failure:
        return 2
    return 0

def main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Code Review & Debugging Pipeline - Milestone 5"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="sample_project",
        help="Path to Python file or directory to review (default: 'sample_project')",
    )
    parser.add_argument(
        "--pr",
        dest="pr",
        default=None,
        help="GitHub PR to review in read-only preview mode (e.g., 'owner/repo#123' or PR URL)",
    )
    parser.add_argument(
        "--repo",
        dest="repo",
        default=None,
        help="GitHub repository name if --pr is only the PR number (e.g. 'owner/repo')",
    )
    parser.add_argument(
        "--token",
        dest="token",
        default=None,
        help="DEPRECATED & DISABLED: Passing tokens via CLI is disabled for security. Use GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN_FILE.",
    )
    parser.add_argument(
        "--publish",
        dest="publish",
        action="store_true",
        default=False,
        help="Publish the review to GitHub PR after interactive human approval",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Simulate review publishing without performing any write requests",
    )
    parser.add_argument(
        "--no-ai",
        dest="enable_ai",
        action="store_false",
        default=True,
        help="Disable AI analysis and use only deterministic Ruff and Bandit scanners",
    )
    parser.add_argument(
        "--no-sandbox",
        dest="enable_sandbox",
        action="store_false",
        default=True,
        help="Disable Docker sandbox execution even when Docker is available",
    )
    parser.add_argument(
        "--no-debug",
        dest="enable_debugging",
        action="store_false",
        default=True,
        help="Disable the Debugging Agent (no patch proposals)",
    )
    args = parser.parse_args()

    if args.token is not None:
        register_sensitive_token(args.token)
        sys.stderr.write(
            "[Security Error] Passing GitHub token via '--token' CLI option is disabled for security reasons: "
            "command-line arguments can be exposed in process tables and shell history.\n"
            "Please set the GITHUB_TOKEN or GH_TOKEN environment variable (or GITHUB_TOKEN_FILE) instead.\n"
        )
        sys.exit(2)

    if args.publish and not args.pr:
        print("ERROR: --publish requires a GitHub PR target specified via --pr.")
        sys.exit(2)

    if args.pr:
        exit_code = run_pr_pipeline(
            pr_target=args.pr,
            repo=args.repo,
            token=None,
            enable_ai=args.enable_ai,
            enable_sandbox=args.enable_sandbox,
            enable_debugging=args.enable_debugging,
            publish=args.publish,
            dry_run=args.dry_run,
        )
    else:
        exit_code = run_pipeline(
            args.target,
            enable_ai=args.enable_ai,
            enable_sandbox=args.enable_sandbox,
            enable_debugging=args.enable_debugging,
        )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
