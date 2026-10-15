"""
github_review_workflow.py - Read-Only GitHub PR Review Workflow Integration.

Milestone 4 — Stage 4, Phase 1:
  1. Accepts a GitHub repository and pull-request number (or PR URL / target string).
  2. Fetches PR metadata and unified diff via GitHubClient.
  3. Pins analysis to the exact PR head SHA.
  4. Prepares an isolated source workspace for that exact commit using secure archive extraction.
  5. Executes deterministic analysis (Ruff & Bandit) and multi-agent review pipeline.
  6. Maps issues to diff hunks/lines to formulate structured inline ReviewComment proposals.
  7. Formulates a validated ActionPlan in permanent dry-run mode (read-only).
  8. Displays an approval preview without publishing or triggering write requests.
  9. Strictly read-only: No review creation, comments, approvals, merges, or write API calls.
 10. Fail-closed: Aborts on auth errors, invalid data, archive violations, analysis crashes,
     and head-SHA mismatches (TOCTOU race defense).
 11. Credential defense: Automatic redaction of tokens from logs, errors, and reports.
"""

from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from approval_gate import (
    ActionPlan,
    InvalidActionPlanError,
    ReviewComment,
    format_approval_preview,
    verify_head_sha_unchanged,
)
from debugging_agent import MAX_DEBUG_ATTEMPTS
from github_client import (
    DiffFile,
    GitHubClient,
    PRMetadata,
    parse_diff_hunks,
    redact_token,
)
from pr_workspace import (
    CommitBindingError,
    WorkspaceLimits,
    prepare_pr_workspace,
)
from state import ReviewState
from tools import ToolResult
from workflow import run_review_pipeline

logger = logging.getLogger(__name__)

# Patterns for repository and PR URL parsing
_PR_URL_PATTERN = re.compile(
    r"^https?://github\.com/([a-zA-Z0-9_\-\.]+)/([a-zA-Z0-9_\-\.]+)/pull/(\d+)(?:[/?#].*)?$"
)
_REPO_PR_PATTERN = re.compile(
    r"^([a-zA-Z0-9_\-\.]+)/([a-zA-Z0-9_\-\.]+)#(\d+)$"
)
_REPO_PAIR_PATTERN = re.compile(
    r"^([a-zA-Z0-9_\-\.]+)/([a-zA-Z0-9_\-\.]+)$"
)


# ============================================================================
# Exceptions
# ============================================================================


class PRReviewWorkflowError(Exception):
    """Base exception for PR review workflow errors."""

    def __init__(self, message: str, token: str | None = None) -> None:
        super().__init__(redact_token(message, token))


class PRReviewInputError(PRReviewWorkflowError):
    """Raised on invalid repository or PR input arguments."""


# ============================================================================
# Data Models
# ============================================================================


@dataclass(frozen=True)
class PRReviewResult:
    """Structured result of the read-only PR review workflow execution."""

    owner: str
    repo: str
    pr_number: int
    head_sha: str
    action_plan: ActionPlan
    preview_text: str
    report: dict[str, Any]
    review_comments: tuple[ReviewComment, ...]
    changed_files: tuple[str, ...]
    workflow_state: dict[str, Any]
    success: bool
    summary: str


# ============================================================================
# Input Helpers
# ============================================================================


def parse_repo_and_pr(
    repo_or_url: str,
    pr_number: int | str | None = None,
) -> tuple[str, str, int]:
    """
    Parse repository owner, repository name, and PR number from varied inputs.

    Supported formats:
      - Full URL: 'https://github.com/owner/repo/pull/123'
      - Hash syntax: 'owner/repo#123'
      - Separate parameters: repo_or_url='owner/repo', pr_number=123
      - Separate parameters: repo_or_url='owner', pr_number='repo/123' (not recommended)

    Raises:
        PRReviewInputError: If input cannot be parsed or parameters are invalid.
    """
    if not repo_or_url or not isinstance(repo_or_url, str):
        raise PRReviewInputError("Repository or PR URL must be a non-empty string.")

    cleaned = repo_or_url.strip()

    # 1. Full GitHub PR URL
    url_match = _PR_URL_PATTERN.match(cleaned)
    if url_match:
        return url_match.group(1), url_match.group(2), int(url_match.group(3))

    # 2. owner/repo#pr_number format
    hash_match = _REPO_PR_PATTERN.match(cleaned)
    if hash_match:
        return hash_match.group(1), hash_match.group(2), int(hash_match.group(3))

    # 3. owner/repo with separate pr_number
    pair_match = _REPO_PAIR_PATTERN.match(cleaned)
    if pair_match:
        if pr_number is None:
            raise PRReviewInputError(
                f"PR number is required when specifying repository as '{cleaned}'. "
                f"Example: '{cleaned}#12' or pass pr_number=12."
            )
        try:
            num = int(pr_number)
            if num <= 0:
                raise ValueError
        except (ValueError, TypeError):
            raise PRReviewInputError(
                f"PR number must be a positive integer, got {pr_number!r}."
            )
        return pair_match.group(1), pair_match.group(2), num

    raise PRReviewInputError(
        f"Unable to parse repository and pull-request number from '{repo_or_url}'. "
        "Expected 'owner/repo#<pr>', 'https://github.com/owner/repo/pull/<pr>', or 'owner/repo' with pr_number."
    )


# ============================================================================
# Finding Path Normalization & Comment Construction
# ============================================================================


def normalize_finding_rel_path(raw_path: str, workspace_root: Path) -> str:
    """
    Normalize a tool finding file path into a clean, forward-slash relative path
    rooted at the workspace directory.
    """
    if not raw_path:
        return ""
    p = Path(raw_path)
    try:
        resolved_ws = workspace_root.resolve()
        if p.is_absolute():
            resolved_p = p.resolve()
            if resolved_p.is_relative_to(resolved_ws):
                rel = resolved_p.relative_to(resolved_ws)
                return rel.as_posix()
            return p.name
        else:
            # Already relative
            posix_path = p.as_posix().lstrip("./")
            return posix_path
    except (ValueError, RuntimeError):
        # Fallback to sanitized posix path
        return p.as_posix().lstrip("./")



def build_inline_comments(
    issues: list[dict[str, Any]],
    diff_map: dict[str, DiffFile],
    workspace_root: Path,
    ai_explanations: dict[str, str] | None = None,
) -> tuple[list[ReviewComment], list[dict[str, Any]]]:
    """
    Map static analysis issues to diff hunks/lines to formulate inline ReviewComment proposals.

    Returns:
        A tuple of (inline_comments, unmapped_issues).
        Inline comments are guaranteed to anchor to valid new-side lines in the PR diff.
        Unmapped issues are issues outside the diff hunks or lacking line numbers.
    """
    explanations = ai_explanations or {}
    comments: list[ReviewComment] = []
    unmapped: list[dict[str, Any]] = []

    for issue in issues:
        raw_path = issue.get("file_path", "")
        rel_path = normalize_finding_rel_path(raw_path, workspace_root)
        line = issue.get("line")

        # Check if the file is part of the PR diff
        diff_file = diff_map.get(rel_path)
        if not diff_file:
            # Also check casefolded or stripped matches
            for df_path, df in diff_map.items():
                if df_path.casefold() == rel_path.casefold():
                    diff_file = df
                    rel_path = df_path
                    break

        if not diff_file or line is None or not isinstance(line, int):
            unmapped.append(issue)
            continue

        # Check if the line is part of the PR diff hunks
        if not diff_file.is_line_in_diff(line):
            unmapped.append(issue)
            continue

        # Construct structured inline review comment
        code = issue.get("code", "UNKNOWN")
        tool = str(issue.get("tool", "scanner")).capitalize()
        severity = issue.get("severity", "WARNING")
        message = issue.get("message", "Issue detected.")

        body_parts = [
            f"**[{tool} - {code}]** ({severity})",
            message,
        ]

        if tool.lower() == "ruff" and "linter" in explanations:
            body_parts.append(f"\n**Linter Context:**\n{explanations['linter']}")
        elif tool.lower() == "bandit" and "security" in explanations:
            body_parts.append(f"\n**Security Threat Assessment:**\n{explanations['security']}")

        body_parts.append(
            "\n> *Advisory Notice:* Automated analysis findings and AI explanations represent hypotheses. "
            "Static scans do not guarantee runtime security."
        )

        comment_body = "\n".join(body_parts)

        try:
            rc = ReviewComment(
                path=rel_path,
                line=line,
                body=comment_body,
                side="RIGHT",
            )
            rc.validate()
            comments.append(rc)
        except InvalidActionPlanError as e:
            logger.warning("Skipping invalid comment proposal for %s:%s: %s", rel_path, line, e)
            unmapped.append(issue)

    return comments, unmapped


def determine_review_verdict(
    issues: list[dict[str, Any]],
    agent_errors: list[str],
    inline_comments: list[ReviewComment],
) -> str:
    """
    Determine the proposed GitHub review verdict based on findings and errors.

    Verdicts:
      - 'REQUEST_CHANGES': High/critical security findings or blocking analysis errors.
      - 'COMMENT': Medium/low warnings, lint findings, or informational suggestions.
      - 'APPROVE': Zero issues detected across all analyzed changed files.
    """
    # 1. Critical errors or high-severity security issues require changes
    has_high_security = any(
        i.get("tool") == "bandit" and str(i.get("severity", "")).upper() in ("HIGH", "CRITICAL")
        for i in issues
    )
    has_blocking_errors = any(
        "error" in err.lower() or "failed" in err.lower() for err in agent_errors
    )

    if has_high_security or has_blocking_errors:
        return "REQUEST_CHANGES"

    # 2. If there are any issues or inline comments, submit as COMMENT
    if issues or inline_comments:
        return "COMMENT"

    # 3. Clean review with no findings
    return "APPROVE"


def generate_review_summary(
    meta: PRMetadata,
    verdict: str,
    total_issues: int,
    inline_comments_count: int,
    unmapped_issues_count: int,
    workflow_state: ReviewState,
    changed_files: list[str],
) -> str:
    """
    Generate an executive markdown summary of the PR review findings.
    """
    rep = workflow_state.get("final_review_report", {})
    summary_rep = rep.get("summary", {})
    sandbox_executed = summary_rep.get("sandbox_executed", False)
    sandbox_passed = summary_rep.get("sandbox_passed", False)
    sandbox_skipped = summary_rep.get("sandbox_skipped", False)
    patch_verified = summary_rep.get("patch_verified", False)

    lines = [
        f"### Automated Code Review & Security Report: PR #{meta.pr_number}",
        f"**Repository:** `{meta.owner}/{meta.repo}`  ",
        f"**Target Head Commit:** `{meta.head_sha[:10]}` (`{meta.head_sha}`)  ",
        f"**PR Author:** @{meta.author} | **Base Branch:** `{meta.base_branch}`  ",
        f"**Overall Review Verdict:** **{verdict}**  ",
        "",
        "#### Summary of Findings",
        f"- **Changed files analyzed:** {len(changed_files)}",
        f"- **Total code issues found:** {total_issues}",
        f"- **Proposed inline comments (anchored in diff):** {inline_comments_count}",
        f"- **File-level / unmapped findings:** {unmapped_issues_count}",
    ]

    # Sandbox execution details
    if sandbox_executed:
        status_str = "Passed" if sandbox_passed else "Failed"
        lines.append(f"- **Docker Sandbox Test Execution:** {status_str}")
        if patch_verified:
            lines.append("- **Automated Patch Verification:** Minimal patch generated and verified!")
    elif sandbox_skipped:
        lines.append("- **Docker Sandbox Test Execution:** Skipped (Docker unavailable or disabled)")

    # Advisory notice
    lines.extend([
        "",
        "> [!NOTE]",
        (
            "> **Advisory Notice:** Automated analysis findings, test generation, and AI explanations "
            "represent advisory hypotheses. Static analysis and sandboxed tests do not guarantee total runtime security. "
            "Human maintainer review is required before merging."
        ),
    ])


    return "\n".join(lines)


# ============================================================================
# Core Read-Only Workflow Orchestration
# ============================================================================


def run_pr_review_workflow(
    repo_or_url: str,
    pr_number: int | str | None = None,
    *,
    expected_head_sha: str | None = None,
    client: GitHubClient | None = None,
    token: str | None = None,
    enable_ai: bool = True,
    llm: BaseChatModel | None = None,
    enable_sandbox: bool = True,
    enable_debugging: bool = True,
    max_debug_attempts: int = MAX_DEBUG_ATTEMPTS,
    limits: WorkspaceLimits | None = None,
    output_func: Callable[[str], None] | None = None,
    display_preview: bool = True,
    ruff_runner: Callable[..., ToolResult] | None = None,
    bandit_runner: Callable[..., ToolResult] | None = None,
    sandbox_runner_fn: Callable[..., Any] | None = None,
) -> PRReviewResult:
    """
    Execute an end-to-end, strictly read-only PR review workflow.

    Workflow Steps:
      1. Parse repository and PR identifiers.
      2. Fetch PR metadata and unified diff from GitHub REST API.
      3. Pin analysis strictly to the exact PR head SHA (fail closed on mismatch).
      4. Prepare an isolated, commit-bound workspace via secure archive extraction.
      5. Run deterministic static scanners (Ruff, Bandit) and review pipeline.
      6. Parse unified diff and map findings to valid diff lines to formulate ReviewComments.
      7. Verify PR head SHA remains unchanged (anti-TOCTOU race defense).
      8. Formulate a validated ActionPlan in dry-run mode (read-only).
      9. Generate and display the approval preview without executing any write API calls.

    Guarantees:
      - Strictly read-only: Never makes GitHub write requests (POST/PUT/PATCH/DELETE).
      - No code execution on the host: Untrusted PR code is never run on the host;
        all test execution is confined to Docker sandboxes.
      - Fail-closed: Aborts with clear exceptions on auth, network, validation, or SHA errors.
      - Credential protection: All tokens are automatically scrubbed from logs and errors.

    Raises:
        PRReviewInputError: If repo or PR parameters are invalid.
        GitHubAuthError: If authentication fails (HTTP 401).
        GitHubNotFoundError: If repo or PR is not found (HTTP 404).
        CommitBindingError: If head SHA is missing or mismatches expected SHA.
        HeadShaMismatchError: If PR head commit changes during review analysis.
        ArchiveSecurityError: If PR archive violates path traversal or security boundaries.
        PRReviewWorkflowError: On other unexpected workflow failures.
    """
    out = output_func or print

    # 1. Parse repository and PR number
    owner, repo, pr_num = parse_repo_and_pr(repo_or_url, pr_number)

    # 2. Instantiate GitHub client
    gh_client = client or GitHubClient(token=token)

    # 3. Fetch PR metadata
    meta = gh_client.get_pr_metadata(owner, repo, pr_num)
    live_sha = (meta.head_sha or "").strip()

    if not live_sha or len(live_sha) < 7:
        raise CommitBindingError(
            f"Could not establish valid head SHA from PR metadata for {owner}/{repo} #{pr_num}."
        )

    # Validate against expected_head_sha if provided
    if expected_head_sha:
        clean_expected = expected_head_sha.strip().lower()
        if clean_expected != live_sha.lower():
            raise CommitBindingError(
                f"PR head SHA mismatch for {owner}/{repo} #{pr_num}: expected commit "
                f"'{expected_head_sha}', but PR metadata reports '{live_sha}'. Aborting."
            )

    # 4. Fetch unified diff and parse changed files & hunks
    diff_text = gh_client.get_pr_diff(owner, repo, pr_num)
    diff_files = parse_diff_hunks(diff_text, strict=False)

    diff_map: dict[str, DiffFile] = {}
    changed_files: list[str] = []
    for df in diff_files:
        p = df.path.replace("\\", "/").lstrip("/")
        if p:
            diff_map[p] = df
            changed_files.append(p)

    # 5. Prepare isolated, commit-bound source workspace
    # PRWorkspace context manager guarantees atomic disposable cleanup on exit
    with prepare_pr_workspace(
        owner=owner,
        repo=repo,
        pr_number=pr_num,
        expected_head_sha=live_sha,
        client=gh_client,
        limits=limits,
    ) as ws:
        workspace_dir = ws.workspace_path

        # 6. Execute review pipeline within the isolated workspace
        workflow_state = run_review_pipeline(
            target_path=str(workspace_dir),
            enable_ai=enable_ai,
            llm=llm,
            ruff_runner=ruff_runner,
            bandit_runner=bandit_runner,
            workspace_root=str(workspace_dir),
            enable_sandbox=enable_sandbox,
            enable_debugging=enable_debugging,
            max_debug_attempts=max_debug_attempts,
            sandbox_runner_fn=sandbox_runner_fn,
        )

        # 7. Collect all deterministic findings from pipeline state
        ruff_findings = workflow_state.get("ruff_findings", [])
        bandit_findings = workflow_state.get("bandit_findings", [])
        all_issues = ruff_findings + bandit_findings
        ai_explanations = workflow_state.get("ai_explanations", {})
        agent_errors = workflow_state.get("agent_errors", [])

        # 8. Map findings to diff hunks to build inline ReviewComments
        inline_comments, unmapped_issues = build_inline_comments(
            issues=all_issues,
            diff_map=diff_map,
            workspace_root=workspace_dir,
            ai_explanations=ai_explanations,
        )

        # 9. Verify PR head SHA remains unchanged (TOCTOU race check)
        fresh_meta = gh_client.get_pr_metadata(owner, repo, pr_num)
        verify_head_sha_unchanged(expected_sha=live_sha, live_sha=fresh_meta.head_sha)

        # 10. Determine review verdict and formulate executive summary
        verdict = determine_review_verdict(
            issues=all_issues,
            agent_errors=agent_errors,
            inline_comments=inline_comments,
        )

        summary_text = generate_review_summary(
            meta=meta,
            verdict=verdict,
            total_issues=len(all_issues),
            inline_comments_count=len(inline_comments),
            unmapped_issues_count=len(unmapped_issues),
            workflow_state=workflow_state,
            changed_files=changed_files,
        )

        # 11. Formulate ActionPlan strictly in dry_run=True mode
        action_plan = ActionPlan(
            owner=owner,
            repo=repo,
            pr_number=pr_num,
            head_sha=live_sha,
            verdict=verdict,
            summary=summary_text,
            comments=tuple(inline_comments),
            dry_run=True,  # STRICTLY True for Phase 1 read-only integration
            metadata={
                "workflow_phase": "milestone_4_stage_4_phase_1",
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "total_issues": len(all_issues),
                "inline_comments_count": len(inline_comments),
                "unmapped_issues_count": len(unmapped_issues),
                "changed_files_count": len(changed_files),
            },
        )
        action_plan.validate()

        # 12. Format approval preview
        preview_text = format_approval_preview(action_plan)
        if display_preview:
            out(preview_text)

        report = workflow_state.get("final_review_report", {})
        has_execution_error = any("error" in e.lower() for e in agent_errors)

        return PRReviewResult(
            owner=owner,
            repo=repo,
            pr_number=pr_num,
            head_sha=live_sha,
            action_plan=action_plan,
            preview_text=preview_text,
            report=report,
            review_comments=tuple(inline_comments),
            changed_files=tuple(changed_files),
            workflow_state=workflow_state,
            success=not has_execution_error,
            summary=summary_text,
        )
