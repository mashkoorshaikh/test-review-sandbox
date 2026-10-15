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
import os
import sys
from typing import Any

from langchain_core.language_models import BaseChatModel

from llm_config import is_api_key_configured
from sandbox_runner import is_docker_available
from tools import ToolResult, run_bandit, run_ruff
from workflow import run_review_pipeline


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
        print(f"  [!] {err}")
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
    print("[1/2] Running Ruff Linter...")
    ruff_result = run_ruff(normalized_path)

    # Step 2: Run deterministic Bandit security scanner
    print("[2/2] Running Bandit Security Scanner...")
    bandit_result = run_bandit(normalized_path)
    print()

    results: dict[str, ToolResult] = {
        "Ruff": ruff_result,
        "Bandit": bandit_result,
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


from github_client import redact_token
from github_review_workflow import (
    run_pr_review_workflow,
)


def run_pr_pipeline(
    pr_target: str,
    repo: str | None = None,
    token: str | None = None,
    enable_ai: bool = True,
    enable_sandbox: bool = True,
    enable_debugging: bool = True,
    llm: BaseChatModel | None = None,
) -> int:
    """
    Execute the read-only PR review workflow for a GitHub repository and PR.
    """
    print(format_separator("="))
    print("  MULTI-AGENT CODE REVIEW & DEBUGGING PIPELINE")
    print("  Milestone 4: Read-Only Pull Request Review & Approval Preview")
    print(format_separator("="))
    print()
    print(f"Target PR: {pr_target} (Read-Only Preview Mode)")
    print(format_separator("-"))

    has_credentials = is_api_key_configured() or llm is not None
    effective_enable_ai = enable_ai and has_credentials
    if enable_ai and not has_credentials:
        print("[Notice] LLM credentials not configured. Running PR review in deterministic mode.")
        print()

    try:
        result = run_pr_review_workflow(
            repo_or_url=repo if repo else pr_target,
            pr_number=pr_target if repo else None,
            token=token,
            enable_ai=effective_enable_ai,
            llm=llm,
            enable_sandbox=enable_sandbox,
            enable_debugging=enable_debugging,
            display_preview=True,
        )
    except Exception as e:  # noqa: BLE001
        safe_msg = redact_token(str(e), token)
        print(format_separator("="))
        print(f"PR WORKFLOW FAILED: {safe_msg}")
        print(format_separator("="))
        return 2


    # Exit code based on verdict and issues
    if not result.success:
        return 2
    if result.action_plan.verdict == "REQUEST_CHANGES":
        return 1
    if result.review_comments:
        return 1
    return 0


def main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Code Review & Debugging Pipeline - Milestone 4"
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
        help="GitHub token (defaults to GITHUB_TOKEN or GH_TOKEN env var)",
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

    if args.pr:
        exit_code = run_pr_pipeline(
            pr_target=args.pr,
            repo=args.repo,
            token=args.token,
            enable_ai=args.enable_ai,
            enable_sandbox=args.enable_sandbox,
            enable_debugging=args.enable_debugging,
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
