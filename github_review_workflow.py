"""
github_review_workflow.py - GitHub PR Review Workflow and Publication Integration.

Milestone 4 — Stage 4:
  1. Accepts a GitHub repository and pull-request number (or PR URL / target string).
  2. Fetches PR metadata and unified diff via GitHubClient.
  3. Pins analysis to the exact PR head SHA.
  4. Prepares an isolated source workspace for that exact commit using secure archive extraction.
  5. Executes deterministic analysis (Ruff & Bandit) and multi-agent review pipeline.
  6. Maps issues to diff hunks/lines to formulate structured inline ReviewComment proposals.
  7. Formulates a validated ActionPlan in dry-run mode (read-only default) or write mode.
  8. Displays an approval preview with strict human authorization gating.
  9. Pre-publication revalidation (revalidate_review_for_publication):
     - Enforces TTL window (600s) and plan freshness.
     - Strictly checks for TOCTOU commit SHA changes and repository binding.
     - Verifies inline comment anchoring against the current diff.
     - Verifies cryptographic plan fingerprint against tampering.
 10. Publication (publish_approved_review): Posts approved review to GitHub REST API.
 11. Fail-closed: Aborts on auth errors, invalid data, archive violations, analysis crashes,
     and head-SHA mismatches.
 12. Credential defense: Automatic redaction of tokens from logs, errors, and reports.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from langchain_core.language_models import BaseChatModel

from approval_gate import (
    DEFAULT_APPROVAL_TTL_SECONDS,
    DEFAULT_MAX_PLAN_AGE_SECONDS,
    ActionPlan,
    ApprovalExpiredError,
    ApprovalGateError,
    ApprovalResult,
    InvalidActionPlanError,
    ReviewComment,
    format_approval_preview,
    request_approval,
    verify_approval_for_write,
    verify_head_sha_unchanged,
)
from debugging_agent import MAX_DEBUG_ATTEMPTS
from github_client import (
    DiffFile,
    GitHubClient,
    GitHubUncertainWriteError,
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
from workflow import run_review_pipeline, split_agent_errors

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


class PrePublicationValidationError(PRReviewWorkflowError, ApprovalGateError):
    """Raised when pre-publication revalidation checks fail."""


class CommentAnchoringError(PrePublicationValidationError):
    """Raised when an inline review comment cannot anchor to valid live diff hunks."""


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
        Unmapped issues are issues outside the diff hunks, in binary or deleted files,
        cross-hunk multi-line spans, or lacking line numbers.
    """
    explanations = ai_explanations or {}
    comments: list[ReviewComment] = []
    unmapped: list[dict[str, Any]] = []

    for issue in issues:
        raw_path = issue.get("file_path", "")
        rel_path = normalize_finding_rel_path(raw_path, workspace_root)
        line = issue.get("line")
        start_line = issue.get("start_line")

        # Check if the file is part of the PR diff
        diff_file = diff_map.get(rel_path)
        if not diff_file:
            # Also check casefolded or stripped matches, or renamed old_path matches
            for df_path, df in diff_map.items():
                if df_path.casefold() == rel_path.casefold():
                    diff_file = df
                    rel_path = df_path
                    break
                if df.old_path and df.old_path.replace("\\", "/").lstrip("/").casefold() == rel_path.casefold():
                    diff_file = df
                    rel_path = df.new_path or df_path
                    break

        # Reject if not in diff, or if deleted, or if binary, or if line is missing/invalid
        if (
            not diff_file
            or diff_file.is_binary
            or diff_file.status in ("deleted", "binary")
            or diff_file.new_path is None
            or line is None
            or not isinstance(line, int)
            or line <= 0
        ):
            unmapped.append(issue)
            continue

        # Single-line or multi-line hunk anchoring
        effective_start_line: int | None = None
        if start_line is not None:
            if not isinstance(start_line, int) or start_line <= 0 or start_line > line:
                unmapped.append(issue)
                continue
            if start_line < line:
                # Multi-line comment: must be contained entirely in the same diff hunk
                hunk = diff_file.find_hunk_for_span(start_line, line)
                if hunk is None:
                    unmapped.append(issue)
                    continue
                effective_start_line = start_line
            else:
                # start_line == line: treat as single-line
                if not diff_file.is_line_in_diff(line):
                    unmapped.append(issue)
                    continue
        else:
            # Single-line comment
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

        # Use new_path for renamed/modified files
        target_path = diff_file.new_path or rel_path

        try:
            rc = ReviewComment(
                path=target_path,
                line=line,
                body=comment_body,
                side="RIGHT",
                start_line=effective_start_line,
            )
            rc.validate()
            comments.append(rc)
        except InvalidActionPlanError as e:
            logger.warning("Skipping invalid comment proposal for %s:%s: %s", target_path, line, e)
            unmapped.append(issue)

    return comments, unmapped


def is_high_confidence_critical_issue(issue: dict[str, Any]) -> bool:
    """
    Check if an issue represents a high-confidence critical security finding
    or a fatal analysis error.

    Rules:
      - Fatal analysis errors on issues with severity 'FATAL'.
      - High-confidence critical security findings:
        Severity 'CRITICAL' or (tool is 'bandit' and severity is 'HIGH')
        AND confidence is explicitly 'HIGH' or 'CRITICAL'.
      - Missing, unknown, LOW, or MEDIUM confidence never qualifies as high confidence;
        a finding with missing or unconfirmed confidence maps to COMMENT.
    """
    sev = str(issue.get("severity", "")).strip().upper()
    if sev == "FATAL":
        return True

    tool = str(issue.get("tool", "")).strip().lower()
    conf = str(issue.get("confidence") or issue.get("issue_confidence", "")).strip().upper()

    is_qualifying_severity = (sev == "CRITICAL") or (tool == "bandit" and sev == "HIGH")
    is_high_conf = conf in ("HIGH", "CRITICAL")

    return bool(is_qualifying_severity and is_high_conf)


def has_fatal_analysis_error(agent_errors: list[str]) -> bool:
    """
    Check if agent errors contain genuine fatal analysis failures or primary tool crashes.
    Auxiliary advisory warnings (such as Test-Writer provider errors or test validation failures)
    are excluded from fatal error determination and do not trigger REQUEST_CHANGES.
    """
    fatal_errors, _ = split_agent_errors(agent_errors)
    return len(fatal_errors) > 0


def determine_review_verdict(
    issues: list[dict[str, Any]],
    agent_errors: list[str],
    inline_comments: list[ReviewComment] | None = None,
) -> str:
    """
    Determine the deterministic GitHub review verdict for an automated analysis plan.

    Approved Policy:
      - 'REQUEST_CHANGES': High-confidence critical security findings or fatal analysis errors.
      - 'COMMENT': Ordinary findings (warnings, medium/low security, lint), lower-confidence concerns,
                   advisory recommendations, or clean PR scans with zero findings.
      - 'APPROVE': NEVER chosen automatically. Clean reviews default to 'COMMENT'. The automated pipeline
                   never presumes authority to grant merge approval without explicit human discretion.
    """
    # 1. Fatal analysis errors require changes (conservative fail-safe)
    if has_fatal_analysis_error(agent_errors):
        return "REQUEST_CHANGES"

    # 2. High-confidence critical security findings require changes
    if any(is_high_confidence_critical_issue(issue) for issue in issues):
        return "REQUEST_CHANGES"

    # 3. All other cases (ordinary findings, lower-confidence concerns, clean scans with no issues):
    # Never automatically select APPROVE; always submit as COMMENT.
    return "COMMENT"


def generate_review_summary(
    meta: PRMetadata,
    verdict: str,
    total_issues: int,
    inline_comments_count: int,
    unmapped_issues_count: int,
    workflow_state: ReviewState,
    changed_files: list[str],
    unmapped_issues: list[dict[str, Any]] | None = None,
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

    agent_errors = workflow_state.get("agent_errors", [])
    fatal_errors, advisory_warnings = split_agent_errors(agent_errors)

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
        f"- **Primary Static Analysis Status:** {'FAILED (' + str(len(fatal_errors)) + ' tool error(s))' if fatal_errors else 'PASSED (Ruff & Bandit completed)'}",
    ]
    if advisory_warnings:
        lines.append(f"- **Auxiliary Agent Status:** {len(advisory_warnings)} non-blocking advisory warning(s)")

    # File-level & unmapped findings detail
    if unmapped_issues:
        lines.extend([
            "",
            "#### File-Level & Unmapped Findings (Outside Diff Hunks)",
            (
                "The following findings were detected by static scanners but are not anchored as inline "
                "comments (e.g. unmodified lines, binary files, deleted files, or file-level warnings):"
            ),
        ])
        for u in unmapped_issues:
            f_path = u.get("file_path", "unknown")
            f_line = u.get("line")
            loc = f"`{f_path}:{f_line}`" if f_line else f"`{f_path}`"
            u_code = u.get("code", "UNKNOWN")
            u_tool = str(u.get("tool", "scanner")).capitalize()
            u_sev = u.get("severity", "WARNING")
            u_msg = u.get("message", "Issue detected.")
            lines.append(f"- {loc} **[{u_tool} - {u_code}]** ({u_sev}): {u_msg}")

    # Sandbox execution details
    if sandbox_executed:
        status_str = "Passed" if sandbox_passed else "Failed"
        lines.append(f"- **Docker Sandbox Test Execution:** {status_str}")
        if patch_verified:
            lines.append("- **Automated Patch Verification:** Minimal patch generated and verified!")
    elif sandbox_skipped:
        lines.append("- **Docker Sandbox Test Execution:** Skipped (Docker unavailable or disabled)")

    # Required primary tool failures detail
    if fatal_errors:
        lines.extend([
            "",
            "#### Required Primary Tool Failures (Analysis Incomplete)",
            (
                "> [!WARNING]\n"
                "> One or more required primary analysis tools failed or timed out. "
                "The review verdict is set to REQUEST_CHANGES to prevent merging unanalyzed code:"
            ),
        ])
        for fe in fatal_errors:
            safe_fe = redact_token(fe)
            lines.append(f"- **[Primary Tool Failure]** {safe_fe}")

    # Auxiliary agent warnings detail
    if advisory_warnings:
        lines.extend([
            "",
            "#### Auxiliary Agent Warnings (Non-Blocking)",
            (
                "The following non-blocking warnings occurred during auxiliary agent analysis "
                "(such as AI test generation or test validation) and did not block the review:"
            ),
        ])
        for aw in advisory_warnings:
            safe_aw = redact_token(aw)
            lines.append(f"- **[Advisory Warning]** {safe_aw}")

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
# Pre-Publication Revalidation Helpers
# ============================================================================


@dataclass(frozen=True)
class PrePublicationRevalidationResult:
    """Outcome of pre-publication revalidation."""

    is_valid: bool
    live_metadata: PRMetadata
    serialized_payload: dict[str, Any]
    plan_fingerprint: str
    revalidated_at_mono: float
    revalidated_at_utc: str


def revalidate_review_for_publication(
    approval_result: ApprovalResult,
    current_plan: ActionPlan,
    client: GitHubClient,
    clock: Callable[[], float] | None = None,
    max_plan_age_seconds: float = DEFAULT_MAX_PLAN_AGE_SECONDS,
    live_diff_text: str | None = None,
    publication_marker: str | None = None,
) -> PrePublicationRevalidationResult:
    """
    Revalidate pull-request state, approval status, plan integrity, and diff anchoring
    immediately prior to review publication.

    Enforces:
      1. Gate was approved, not dry_run, and approval is within the 600-second TTL.
      2. Plan is not stale under max_plan_age_seconds.
      3. Live pull-request repository owner, name, and PR number match the approved plan.
      4. Live PR head commit SHA exactly matches the approved plan's head SHA (no TOCTOU).
      5. Approved plan fingerprint matches the exact current plan.
      6. Review payload serializes and passes validation.
      7. Every inline comment still anchors to a valid editable line within the current PR diff.

    Fails closed with typed exceptions upon ANY discrepancy. Never silently updates
    or refreshes the approved plan or commit SHA.

    Raises:
        ApprovalRejectedError: If approval was not granted.
        DryRunBlockedError: If approved in dry-run mode.
        ApprovalExpiredError: If approval TTL or plan max age has expired.
        PlanFingerprintMismatchError: If plan or current_plan fingerprint has changed.
        HeadShaMismatchError: If the live PR head SHA differs from the approved SHA.
        PrePublicationValidationError: If repo/PR mismatch, closed PR, or malformed payload.
        CommentAnchoringError: If inline comment coordinates cannot anchor to current diff.
    """
    _clock = clock or time.monotonic
    current_mono = _clock()
    current_utc = datetime.now(UTC).isoformat()

    # 1. Verify approval validity, non-dry-run, TTL expiry, and fingerprint match
    verify_approval_for_write(
        approval_result, current_plan=current_plan, current_mono=current_mono
    )

    # 2. Verify plan staleness
    if current_plan.created_at_mono is not None:
        plan_age = current_mono - current_plan.created_at_mono
        if plan_age > max_plan_age_seconds:
            raise ApprovalExpiredError(
                f"Action plan is stale (age {plan_age:.1f}s exceeds maximum allowed "
                f"{max_plan_age_seconds:.1f}s). Re-run review analysis and obtain human authorization."
            )

    # 3. Strictly validate plan contents and serialize payload
    current_plan.validate()
    payload = current_plan.to_github_payload(publication_marker=publication_marker)

    # 4. Re-fetch live PR metadata and verify exact target binding
    live_meta = client.get_pr_metadata(
        current_plan.owner, current_plan.repo, current_plan.pr_number
    )

    if live_meta.owner.casefold() != current_plan.owner.casefold():
        raise PrePublicationValidationError(
            f"Repository owner mismatch: approved plan was for {current_plan.owner!r}, "
            f"but live PR reports {live_meta.owner!r}."
        )

    if live_meta.repo.casefold() != current_plan.repo.casefold():
        raise PrePublicationValidationError(
            f"Repository name mismatch: approved plan was for {current_plan.repo!r}, "
            f"but live PR reports {live_meta.repo!r}."
        )

    if live_meta.pr_number != current_plan.pr_number:
        raise PrePublicationValidationError(
            f"PR number mismatch: approved plan was for #{current_plan.pr_number}, "
            f"but live PR reports #{live_meta.pr_number}."
        )

    if live_meta.state.lower() != "open":
        raise PrePublicationValidationError(
            f"Pull request #{current_plan.pr_number} is not open (state: {live_meta.state!r}). "
            "Cannot publish review on closed or merged pull request."
        )

    # 5. Strict head SHA match (never silently refresh SHA)
    verify_head_sha_unchanged(expected_sha=current_plan.head_sha, live_sha=live_meta.head_sha)

    # 6. Revalidate inline comment diff-hunk anchoring against current diff
    if current_plan.comments:
        diff_text = live_diff_text
        if diff_text is None:
            diff_text = client.get_pr_diff(
                current_plan.owner, current_plan.repo, current_plan.pr_number
            )

        diff_files = parse_diff_hunks(diff_text, strict=False)
        diff_map: dict[str, DiffFile] = {}
        for df in diff_files:
            p = df.path.replace("\\", "/").lstrip("/")
            if p:
                diff_map[p] = df
            if df.old_path:
                old_p = df.old_path.replace("\\", "/").lstrip("/")
                if old_p and old_p not in diff_map:
                    diff_map[old_p] = df

        for comment in current_plan.comments:
            c_path = comment.path.replace("\\", "/").lstrip("/")
            diff_file = diff_map.get(c_path)
            if not diff_file:
                raise CommentAnchoringError(
                    f"Inline comment targets path '{comment.path}' which does not exist in the PR diff."
                )

            if diff_file.is_binary or diff_file.status in ("deleted", "binary"):
                raise CommentAnchoringError(
                    f"Inline comment targets binary or deleted file '{comment.path}' in PR diff."
                )

            if comment.start_line is not None:
                # Multi-line comment
                hunk = diff_file.find_hunk_for_span(comment.start_line, comment.line)
                if hunk is None:
                    raise CommentAnchoringError(
                        f"Multi-line comment span [{comment.start_line}, {comment.line}] in '{comment.path}' "
                        "does not reside within a single diff hunk on the new side."
                    )
            else:
                # Single-line comment
                if not diff_file.is_line_in_diff(comment.line):
                    raise CommentAnchoringError(
                        f"Inline comment line {comment.line} in '{comment.path}' is not a valid editable "
                        "line within any PR diff hunk."
                    )

    return PrePublicationRevalidationResult(
        is_valid=True,
        live_metadata=live_meta,
        serialized_payload=payload,
        plan_fingerprint=current_plan.fingerprint,
        revalidated_at_mono=current_mono,
        revalidated_at_utc=current_utc,
    )


@dataclass(frozen=True)
class PRPublicationResult:
    """Outcome of publishing an approved pull-request review to GitHub."""

    success: bool
    owner: str
    repo: str
    pr_number: int
    head_sha: str
    review_id: int | None
    html_url: str | None
    verdict: str
    comments_count: int
    published_at_utc: str
    plan_fingerprint: str
    response_data: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    revalidation_result: PrePublicationRevalidationResult | None = None


def generate_publication_marker(
    owner: str,
    repo: str,
    pr_number: int,
    head_sha: str,
    verdict: str,
    comments: tuple[ReviewComment, ...] | list[ReviewComment] | None = None,
) -> str:
    """
    Generate an approved deterministic publication marker binding target and proposed review.
    """
    raw_key = f"{owner.lower()}/{repo.lower()}#{pr_number}@{head_sha.lower()}:{verdict}"
    if comments:
        sorted_comments = sorted(
            comments,
            key=lambda c: (c.path, c.line or 0, c.start_line or 0, c.body),
        )
        for c in sorted_comments:
            raw_key += f":{c.path}:{c.line or 0}:{c.body}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    clean_owner = re.sub(r"[^a-zA-Z0-9_\-\.]", "", owner)
    clean_repo = re.sub(r"[^a-zA-Z0-9_\-\.]", "", repo)
    clean_sha = head_sha.strip().lower()[:8]
    return f"agy-{clean_owner}-{clean_repo}-{pr_number}-{clean_sha}-{digest}"


def reconcile_published_review(
    client: GitHubClient,
    owner: str,
    repo: str,
    pr_number: int,
    publication_marker: str,
    *,
    expected_head_sha: str | None = None,
    expected_author: str | None = None,
) -> dict[str, Any] | None:
    """
    Safe reconciliation mechanism: queries GitHub PR reviews to check if a review with the
    given publication marker was posted prior to a connection drop or timeout.
    Performs only read requests (GET) and never writes.

    Strictly validates:
      1. Review body contains the exact approved publication marker.
      2. Review commit_id exactly matches the approved head SHA (if expected_head_sha is provided).
      3. Review author login exactly matches the authenticated bot user (if expected_author is provided).
    """
    clean_marker = publication_marker.strip()
    marker_str = f"<!-- agy-review-marker: {clean_marker} -->"

    auth_bot_login = expected_author
    if not auth_bot_login and hasattr(client, "get_authenticated_user"):
        try:
            user_data = client.get_authenticated_user()
            if isinstance(user_data, dict):
                auth_bot_login = user_data.get("login")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not query authenticated user for review reconciliation: %s", exc)

    try:
        reviews = client.get_pr_reviews(owner, repo, pr_number)
        for rev in reviews:
            if not isinstance(rev, dict):
                continue
            body = rev.get("body", "")
            if marker_str not in body:
                continue

            # Verify exact commit SHA
            if expected_head_sha:
                rev_commit = (rev.get("commit_id") or "").strip().lower()
                if rev_commit != expected_head_sha.strip().lower():
                    logger.warning(
                        "Reconciliation review rejected: commit SHA %s != expected %s",
                        rev_commit,
                        expected_head_sha,
                    )
                    continue

            # Verify authenticated bot login
            if auth_bot_login:
                rev_user = (rev.get("user") or {}).get("login", "")
                if rev_user.strip().casefold() != auth_bot_login.strip().casefold():
                    logger.warning(
                        "Reconciliation review rejected: author %s != authenticated bot %s",
                        rev_user,
                        auth_bot_login,
                    )
                    continue

            # All required identity checks satisfied
            return rev
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Safe review reconciliation query failed for %s/%s #%s: %s",
            owner,
            repo,
            pr_number,
            exc,
        )
        return None
    return None


def publish_approved_review(
    approval_result: ApprovalResult,
    current_plan: ActionPlan,
    client: GitHubClient,
    *,
    clock: Callable[[], float] | None = None,
    max_plan_age_seconds: float = DEFAULT_MAX_PLAN_AGE_SECONDS,
    live_diff_text: str | None = None,
    publication_marker: str | None = None,
) -> PRPublicationResult:
    """
    Publish an approved pull-request review to GitHub after comprehensive pre-publication revalidation.

    Enforces:
      1. Pre-publication revalidation via revalidate_review_for_publication:
         - Gate was approved, not dry-run, within TTL window (default: 600s).
         - Plan is not stale under max_plan_age_seconds.
         - Exact repository, PR number, and head commit SHA match (anti-TOCTOU).
         - Cryptographic plan fingerprint match ensures zero tampering.
         - All inline comments anchor cleanly within active diff hunks on the new side.
      2. Invokes GitHubClient.create_pull_request_review with validated payload (at most once).
      3. Never automatically retries an uncertain POST.
      4. If POST times out or connection drops, reports uncertain result and uses safe reconciliation.
      5. Automatic credential protection: Tokens are scrubbed from logs and exceptions.
      6. Zero partial writes: Fails closed on any revalidation discrepancy.

    Raises:
        ApprovalRejectedError: If human approval was not granted.
        DryRunBlockedError: If approved in dry-run mode.
        ApprovalExpiredError: If approval TTL or plan max age has expired.
        PlanFingerprintMismatchError: If plan has been modified since approval.
        HeadShaMismatchError: If the live PR head SHA differs from approved plan SHA.
        PrePublicationValidationError: If repo/PR mismatch, closed PR, or malformed payload.
        CommentAnchoringError: If inline comment coordinates cannot anchor to current diff.
        GitHubAuthError: If authentication token lacks permissions.
        GitHubRateLimitError: If API rate limit exceeded.
        GitHubValidationError: If payload rejected by GitHub API.
        GitHubUncertainWriteError: If write timed out or connection dropped mid-request and reconciliation was negative.
        GitHubClientError: On other API errors.
    """
    _clock = clock or time.monotonic

    # 1. Full Pre-Publication Revalidation (strictly fails closed on any discrepancy)
    reval = revalidate_review_for_publication(
        approval_result=approval_result,
        current_plan=current_plan,
        client=client,
        clock=_clock,
        max_plan_age_seconds=max_plan_age_seconds,
        live_diff_text=live_diff_text,
        publication_marker=publication_marker,
    )

    payload = reval.serialized_payload

    # 2. Execute GitHub Review POST API Call (exactly one attempt, no automated retry)
    try:
        resp_data = client.create_pull_request_review(
            owner=current_plan.owner,
            repo=current_plan.repo,
            pr_number=current_plan.pr_number,
            commit_sha=payload["commit_id"],
            body=payload["body"],
            event=payload["event"],
            comments=payload.get("comments"),
        )
    except GitHubUncertainWriteError:
        effective_marker = (
            publication_marker
            or (
                current_plan.metadata.get("publication_marker")
                if isinstance(current_plan.metadata, Mapping)
                else None
            )
        )
        if effective_marker:
            reconciled = reconcile_published_review(
                client=client,
                owner=current_plan.owner,
                repo=current_plan.repo,
                pr_number=current_plan.pr_number,
                publication_marker=str(effective_marker),
                expected_head_sha=current_plan.head_sha,
            )
            if reconciled and isinstance(reconciled, dict):
                logger.info(
                    "Reconciliation confirmed review creation (review id %s) for %s/%s #%s",
                    reconciled.get("id"),
                    current_plan.owner,
                    current_plan.repo,
                    current_plan.pr_number,
                )
                review_id = (
                    reconciled.get("id")
                    if isinstance(reconciled.get("id"), int)
                    else None
                )
                html_url = (
                    reconciled.get("html_url")
                    if isinstance(reconciled.get("html_url"), str)
                    else None
                )
                return PRPublicationResult(
                    success=True,
                    owner=current_plan.owner,
                    repo=current_plan.repo,
                    pr_number=current_plan.pr_number,
                    head_sha=current_plan.head_sha,
                    review_id=review_id,
                    html_url=html_url,
                    verdict=current_plan.verdict,
                    comments_count=len(payload.get("comments", [])),
                    published_at_utc=datetime.now(UTC).isoformat(),
                    plan_fingerprint=reval.plan_fingerprint,
                    response_data=MappingProxyType(reconciled),
                    revalidation_result=reval,
                )
        raise

    review_id = resp_data.get("id") if isinstance(resp_data, dict) else None
    html_url = resp_data.get("html_url") if isinstance(resp_data, dict) else None

    frozen_response: MappingProxyType[str, Any]
    if isinstance(resp_data, dict):
        frozen_response = MappingProxyType(resp_data)
    else:
        frozen_response = MappingProxyType({})

    return PRPublicationResult(
        success=True,
        owner=current_plan.owner,
        repo=current_plan.repo,
        pr_number=current_plan.pr_number,
        head_sha=current_plan.head_sha,
        review_id=review_id if isinstance(review_id, int) else None,
        html_url=html_url if isinstance(html_url, str) else None,
        verdict=current_plan.verdict,
        comments_count=len(payload.get("comments", [])),
        published_at_utc=datetime.now(UTC).isoformat(),
        plan_fingerprint=reval.plan_fingerprint,
        response_data=frozen_response,
        revalidation_result=reval,
    )


@dataclass(frozen=True)
class PRPublicationWorkflowResult:
    """Outcome of running the controlled PR publication workflow."""

    success: bool
    owner: str
    repo: str
    pr_number: int
    head_sha: str
    action_plan: ActionPlan
    preview_text: str
    published: bool
    dry_run: bool
    publication_result: PRPublicationResult | None = None
    revalidation_result: PrePublicationRevalidationResult | None = None
    approval_result: ApprovalResult | None = None
    error_message: str | None = None


def run_pr_publishing_workflow(
    repo_or_url: str,
    pr_number: int | str | None = None,
    *,
    dry_run: bool = False,
    client: GitHubClient | None = None,
    token: str | None = None,
    enable_ai: bool = True,
    llm: BaseChatModel | None = None,
    enable_sandbox: bool = True,
    enable_debugging: bool = True,
    max_debug_attempts: int = MAX_DEBUG_ATTEMPTS,
    limits: WorkspaceLimits | None = None,
    interactive: bool | None = None,
    input_func: Callable[[str], str] | None = None,
    output_func: Callable[[str], None] | None = None,
    clock: Callable[[], float] | None = None,
    utc_now: Callable[[], datetime] | None = None,
    max_plan_age_seconds: float = DEFAULT_MAX_PLAN_AGE_SECONDS,
    approval_ttl_seconds: float = DEFAULT_APPROVAL_TTL_SECONDS,
    publication_marker: str | None = None,
    ruff_runner: Callable[..., ToolResult] | None = None,
    bandit_runner: Callable[..., ToolResult] | None = None,
    sandbox_runner_fn: Callable[..., Any] | None = None,
) -> PRPublicationWorkflowResult:
    """
    Execute the controlled PR review publishing workflow.

    Enforces the strict publication sequence:
      1. Fetch and analyze the PR using the existing safe workflow.
      2. Construct the final immutable action plan, including repository identity, PR number,
         head SHA, verdict, review body, inline comments, and publication marker.
      3. Display the exact review that would be submitted (and proposed payload if dry_run).
      4. Obtain interactive human approval bound to that exact plan (fails closed if non-interactive).
      5. Immediately before publishing, run Stage 4 pre-publication revalidation against fresh PR metadata.
      6. Verify the approval, fingerprint, plan age, approval TTL, and comment anchoring again.
      7. Make at most one review POST attempt.
      8. Never automatically retry an uncertain POST.
      9. If a POST times out or connection drops, report an uncertain result and use safe reconciliation.
      10. Report success only when a valid response confirms the review was created.
    """
    out = output_func or print
    _clock = clock or time.monotonic
    _utc_now = utc_now or (lambda: datetime.now(UTC))

    owner, repo, pr_num = parse_repo_and_pr(repo_or_url, pr_number)
    canonical_target = f"{owner}/{repo}"
    gh_client = client or GitHubClient(token=token)

    # 1. Fetch and analyze PR using existing safe workflow
    review_result = run_pr_review_workflow(
        repo_or_url=canonical_target,
        pr_number=pr_num,
        client=gh_client,
        token=token,
        enable_ai=enable_ai,
        llm=llm,
        enable_sandbox=enable_sandbox,
        enable_debugging=enable_debugging,
        max_debug_attempts=max_debug_attempts,
        limits=limits,
        output_func=out,
        display_preview=False,
        dry_run=dry_run,
        clock=_clock,
        ruff_runner=ruff_runner,
        bandit_runner=bandit_runner,
        sandbox_runner_fn=sandbox_runner_fn,
    )

    # 2. Construct final immutable action plan including publication marker
    marker = publication_marker or generate_publication_marker(
        owner=review_result.owner,
        repo=review_result.repo,
        pr_number=review_result.pr_number,
        head_sha=review_result.head_sha,
        verdict=review_result.action_plan.verdict,
        comments=review_result.action_plan.comments,
    )
    marker_tag = f"<!-- agy-review-marker: {marker} -->"
    base_summary = review_result.summary
    summary_with_marker = (
        base_summary
        if marker_tag in base_summary
        else f"{base_summary.rstrip()}\n\n{marker_tag}"
    )

    plan_created_mono = _clock()
    final_plan = ActionPlan(
        owner=review_result.owner,
        repo=review_result.repo,
        pr_number=review_result.pr_number,
        head_sha=review_result.head_sha,
        verdict=review_result.action_plan.verdict,
        summary=summary_with_marker,
        comments=review_result.action_plan.comments,
        dry_run=dry_run,
        created_at_mono=plan_created_mono,
        metadata={
            "workflow_phase": "milestone_4_stage_5",
            "publication_marker": marker,
            "timestamp": _utc_now().isoformat(),
            "total_issues": len(review_result.review_comments),
            "changed_files_count": len(review_result.changed_files),
        },
    )
    final_plan.validate()

    # 3. Display exact review that would be submitted
    preview_text = format_approval_preview(final_plan)
    out(preview_text)

    # Dry-run mode: clearly display proposed review payload and perform ZERO write requests
    if dry_run:
        payload = final_plan.to_github_payload(publication_marker=marker)
        out("=" * 78)
        out("        PROPOSED GITHUB REVIEW PAYLOAD (DRY-RUN — NO WRITES)")
        out("=" * 78)
        out(json.dumps(payload, indent=2))
        out("=" * 78)
        out(
            "[DRY RUN] Zero write requests performed. Review publication simulated successfully.\n"
        )
        return PRPublicationWorkflowResult(
            success=review_result.success,
            owner=final_plan.owner,
            repo=final_plan.repo,
            pr_number=final_plan.pr_number,
            head_sha=final_plan.head_sha,
            action_plan=final_plan,
            preview_text=preview_text,
            published=False,
            dry_run=True,
        )

    # 4. Obtain interactive human approval bound to that exact plan
    approval_result = request_approval(
        plan=final_plan,
        interactive=interactive,
        input_func=input_func,
        output_func=out,
        clock=_clock,
        utc_now=_utc_now,
        max_plan_age_seconds=max_plan_age_seconds,
        approval_ttl_seconds=approval_ttl_seconds,
    )

    # 5 & 6 & 7 & 8 & 9 & 10. Pre-publication revalidation and single POST attempt
    pub_result = publish_approved_review(
        approval_result=approval_result,
        current_plan=final_plan,
        client=gh_client,
        clock=_clock,
        max_plan_age_seconds=max_plan_age_seconds,
        publication_marker=marker,
    )

    out(
        f"[SUCCESS] Review #{pub_result.review_id or 'unknown'} successfully published to "
        f"{final_plan.owner}/{final_plan.repo} PR #{final_plan.pr_number}."
    )
    if pub_result.html_url:
        out(f"Review URL: {pub_result.html_url}")

    return PRPublicationWorkflowResult(
        success=True,
        owner=final_plan.owner,
        repo=final_plan.repo,
        pr_number=final_plan.pr_number,
        head_sha=final_plan.head_sha,
        action_plan=final_plan,
        preview_text=preview_text,
        published=True,
        dry_run=False,
        publication_result=pub_result,
        revalidation_result=pub_result.revalidation_result,
        approval_result=approval_result,
    )


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
    dry_run: bool = True,
    clock: Callable[[], float] | None = None,
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
            unmapped_issues=unmapped_issues,
        )

        # 11. Formulate ActionPlan
        _clock = clock or time.monotonic
        plan_created_mono = _clock()
        action_plan = ActionPlan(
            owner=owner,
            repo=repo,
            pr_number=pr_num,
            head_sha=live_sha,
            verdict=verdict,
            summary=summary_text,
            comments=tuple(inline_comments),
            dry_run=dry_run,
            created_at_mono=plan_created_mono,
            metadata={
                "workflow_phase": "milestone_4_stage_4_phase_2" if not dry_run else "milestone_4_stage_4_phase_1",
                "timestamp": datetime.now(UTC).isoformat(),
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
        has_execution_error = has_fatal_analysis_error(agent_errors)

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
