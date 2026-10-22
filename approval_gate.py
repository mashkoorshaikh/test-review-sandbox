"""
approval_gate.py - Human Approval Gate for GitHub Write Operations.

Milestone 4 — Stage 2:
  1. Data structures: ActionPlan and ReviewComment representing proposed GitHub operations.
  2. Strict validation: Rejects incomplete plans (missing repo, invalid SHA, invalid paths).
  3. Formatted preview: Shows repo, PR number, head SHA, verdict, summary, comments, and dry-run state.
  4. Explicit human authorization: Interactive [y/N] prompt defaulting to rejection.
  5. Dry-run safety: --dry-run strictly prevents writes regardless of approval input.
  6. Non-interactive fail-closed: Rejects by default unless authorized via deliberate secret token.
  7. Anti-tampering: LLM outputs cannot grant approval or bypass the gate.
  8. Head SHA verification: Guards against publishing reviews on modified PR heads.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

# SHA verification pattern (Git full 40-char SHA or at least 7-char short SHA)
_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_REPO_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

DEFAULT_APPROVAL_TTL_SECONDS: float = 600.0  # 10-minute approval window
DEFAULT_MAX_PLAN_AGE_SECONDS: float = 600.0  # 10-minute maximum age before approval


# ============================================================================
# Exceptions
# ============================================================================


class ApprovalGateError(Exception):
    """Base exception for all approval gate errors."""


class InvalidActionPlanError(ApprovalGateError):
    """Raised when an ActionPlan fails validation."""


class ApprovalRejectedError(ApprovalGateError):
    """Raised when human approval is denied or aborted."""


class ApprovalExpiredError(ApprovalGateError):
    """Raised when approval has expired or the plan is stale."""


class NonInteractivePublishingDisabledError(ApprovalRejectedError):
    """Raised when review publishing is attempted in a non-interactive environment."""


class PlanFingerprintMismatchError(ApprovalGateError):
    """Raised when plan fingerprint verification fails due to tampering or mismatch."""


class DryRunBlockedError(ApprovalGateError):
    """Raised when a write operation is attempted in dry-run mode."""


class HeadShaMismatchError(ApprovalGateError):
    """Raised when the PR head commit has changed between review and publication."""


# ============================================================================
# Data Models
# ============================================================================


@dataclass(frozen=True)
class ReviewComment:
    """A proposed inline comment for a GitHub pull request review."""

    path: str
    line: int | None
    body: str
    side: str = "RIGHT"
    start_line: int | None = None

    def validate(self) -> None:
        """Validate comment attributes."""
        if not self.path or not isinstance(self.path, str) or not self.path.strip():
            raise InvalidActionPlanError(
                "ReviewComment path must be a non-empty string."
            )
        # Prevent directory traversal in comment path
        if ".." in self.path or self.path.startswith("/") or self.path.startswith("\\"):
            raise InvalidActionPlanError(
                f"ReviewComment path '{self.path}' is invalid."
            )
        if self.line is not None and (not isinstance(self.line, int) or self.line <= 0):
            raise InvalidActionPlanError(
                f"ReviewComment line must be a positive integer, got {self.line!r}."
            )
        if self.start_line is not None and (
            not isinstance(self.start_line, int) or self.start_line <= 0
        ):
            raise InvalidActionPlanError(
                f"ReviewComment start_line must be a positive integer, got {self.start_line!r}."
            )
        if (
            self.start_line is not None
            and self.line is not None
            and self.start_line > self.line
        ):
            raise InvalidActionPlanError(
                f"ReviewComment start_line ({self.start_line}) cannot be greater than line ({self.line})."
            )
        if not self.body or not isinstance(self.body, str) or not self.body.strip():
            raise InvalidActionPlanError(
                "ReviewComment body must be a non-empty string."
            )
        if self.side not in ("LEFT", "RIGHT"):
            raise InvalidActionPlanError(
                f"ReviewComment side must be 'LEFT' or 'RIGHT', got {self.side!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "body": self.body,
            "side": self.side,
            "start_line": self.start_line,
        }

    def to_github_payload(self) -> dict[str, Any]:
        """
        Deterministic serialization of this ReviewComment into GitHub REST API format.

        Single-line format:
            {"path": path, "line": line, "side": side, "body": body}
        Multi-line format:
            {"path": path, "line": line, "side": side, "start_line": start_line, "start_side": side, "body": body}
        """
        self.validate()
        payload: dict[str, Any] = {
            "path": self.path,
            "body": self.body,
            "side": self.side,
        }
        if self.line is not None:
            payload["line"] = self.line

        if self.start_line is not None:
            payload["start_line"] = self.start_line
            payload["start_side"] = self.side

        return payload


def _freeze_metadata(obj: Any) -> Any:
    """Recursively freeze mappings and sequences into immutable representations."""
    if isinstance(obj, Mapping):
        return MappingProxyType({k: _freeze_metadata(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze_metadata(v) for v in obj)
    return obj


def _validate_metadata_value(val: Any, path: str = "metadata") -> None:
    """
    Validate that val is a supported JSON-compatible type.

    Allowed:
      - None
      - str
      - bool
      - int (excluding bool)
      - float (finite only; NaN and Infinities are rejected)
      - list / tuple composed only of supported values
      - Mapping (keys must be str; values must be supported values)

    Rejected:
      - custom objects, sets, frozensets, bytes, bytearrays, non-finite floats,
        non-string mapping keys, and any other unsupported types.
    """
    if val is None or isinstance(val, (str, bool)):
        return
    if isinstance(val, int):
        return
    if isinstance(val, float):
        if not math.isfinite(val):
            raise InvalidActionPlanError(
                f"Non-finite float {val!r} at {path} is not supported in metadata."
            )
        return
    if isinstance(val, (bytes, bytearray)):
        raise InvalidActionPlanError(
            f"Bytes value at {path} is not supported in metadata; use strings."
        )
    if isinstance(val, (set, frozenset)):
        raise InvalidActionPlanError(
            f"Set collection at {path} is not supported in metadata; use lists or tuples."
        )
    if isinstance(val, (list, tuple)):
        for idx, item in enumerate(val):
            _validate_metadata_value(item, f"{path}[{idx}]")
        return
    if isinstance(val, Mapping):
        for k, v in val.items():
            if not isinstance(k, str):
                raise InvalidActionPlanError(
                    f"Metadata mapping key must be a string at {path}, got {k!r} ({type(k).__name__})."
                )
            _validate_metadata_value(v, f"{path}.{k}")
        return

    raise InvalidActionPlanError(
        f"Unsupported metadata value type at {path}: {type(val).__name__} ({val!r}). "
        "Metadata only supports strings, integers, finite floats, booleans, None, "
        "and nested mappings/sequences."
    )


def _canonicalize_value(val: Any, path: str = "metadata") -> Any:
    """
    Recursively canonicalize supported JSON-compatible values for fingerprinting.
    Fails closed by raising InvalidActionPlanError on any unsupported type.
    """
    if val is None or isinstance(val, (str, bool)):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if not math.isfinite(val):
            raise InvalidActionPlanError(
                f"Non-finite float {val!r} at {path} is not supported in metadata."
            )
        return val
    if isinstance(val, (bytes, bytearray, set, frozenset)):
        raise InvalidActionPlanError(
            f"Unsupported type {type(val).__name__} at {path} cannot be serialized."
        )
    if isinstance(val, (list, tuple)):
        return [_canonicalize_value(v, f"{path}[{idx}]") for idx, v in enumerate(val)]
    if isinstance(val, Mapping):
        canonical_map: dict[str, Any] = {}
        for k in sorted(val.keys()):
            if not isinstance(k, str):
                raise InvalidActionPlanError(
                    f"Metadata mapping key must be a string at {path}, got {k!r} ({type(k).__name__})."
                )
            canonical_map[k] = _canonicalize_value(val[k], f"{path}.{k}")
        return canonical_map

    raise InvalidActionPlanError(
        f"Unsupported metadata value type at {path}: {type(val).__name__} ({val!r})."
    )


def calculate_plan_fingerprint(plan: ActionPlan) -> str:
    """
    Compute an unambiguous, deterministic SHA-256 fingerprint of an ActionPlan.

    Binds every security-relevant field:
      - repository owner and repo name
      - pr_number
      - head_sha (normalized stripped lowercase hex)
      - review verdict and summary
      - dry_run state
      - review comments (path, line, start_line, side, body)
      - metadata (recursively sorted and canonicalized key-value pairs)
    """
    plan.validate()

    canonical_comments = [
        {
            "body": c.body,
            "line": c.line,
            "path": c.path,
            "side": c.side,
            "start_line": c.start_line,
        }
        for c in plan.comments
    ]

    canonical_metadata = _canonicalize_value(plan.metadata, "metadata")

    payload = {
        "owner": plan.owner,
        "repo": plan.repo,
        "pr_number": plan.pr_number,
        "head_sha": plan.head_sha.strip().lower(),
        "verdict": plan.verdict,
        "summary": plan.summary,
        "dry_run": bool(plan.dry_run),
        "comments": canonical_comments,
        "metadata": canonical_metadata,
    }

    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActionPlan:
    """Structured plan of proposed GitHub write actions requiring human approval."""

    owner: str
    repo: str
    pr_number: int
    head_sha: str
    verdict: str
    summary: str
    comments: tuple[ReviewComment, ...] = field(default_factory=tuple)
    dry_run: bool = False
    metadata: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    created_at_mono: float | None = None
    created_at_utc: str = ""

    def __post_init__(self) -> None:
        raw_comments = self.comments or ()
        object.__setattr__(self, "comments", tuple(raw_comments))
        if isinstance(self.metadata, Mapping):
            raw_metadata = dict(self.metadata)
            object.__setattr__(self, "metadata", _freeze_metadata(raw_metadata))
        elif self.metadata is None:
            object.__setattr__(self, "metadata", MappingProxyType({}))
        else:
            # Non-mapping metadata is left as-is so validate() will catch and reject it
            pass

        if self.created_at_mono is None:
            object.__setattr__(self, "created_at_mono", time.monotonic())
        if not self.created_at_utc:
            object.__setattr__(
                self, "created_at_utc", datetime.now(UTC).isoformat()
            )

    @property
    def fingerprint(self) -> str:
        """Deterministic cryptographic SHA-256 fingerprint of all plan contents."""
        return calculate_plan_fingerprint(self)

    def validate(self) -> None:
        """Ensure the ActionPlan is complete, well-formed, and secure."""
        if (
            not self.owner
            or not isinstance(self.owner, str)
            or not _REPO_NAME_RE.match(self.owner)
        ):
            raise InvalidActionPlanError(f"Invalid repository owner: {self.owner!r}")
        if (
            not self.repo
            or not isinstance(self.repo, str)
            or not _REPO_NAME_RE.match(self.repo)
        ):
            raise InvalidActionPlanError(f"Invalid repository name: {self.repo!r}")
        if not isinstance(self.pr_number, int) or self.pr_number <= 0:
            raise InvalidActionPlanError(
                f"PR number must be a positive integer, got {self.pr_number!r}."
            )
        if (
            not self.head_sha
            or not isinstance(self.head_sha, str)
            or not _GIT_SHA_RE.match(self.head_sha)
        ):
            raise InvalidActionPlanError(
                f"Head SHA must be a valid 7-40 character hexadecimal git commit SHA, got {self.head_sha!r}."
            )
        if (
            not self.verdict
            or not isinstance(self.verdict, str)
            or not self.verdict.strip()
        ):
            raise InvalidActionPlanError("Review verdict must be a non-empty string.")
        if (
            not self.summary
            or not isinstance(self.summary, str)
            or not self.summary.strip()
        ):
            raise InvalidActionPlanError("Review summary must be a non-empty string.")

        for idx, comment in enumerate(self.comments):
            if not isinstance(comment, ReviewComment):
                raise InvalidActionPlanError(
                    f"Comment at index {idx} is not a ReviewComment instance."
                )
            comment.validate()

        if not isinstance(self.metadata, Mapping):
            raise InvalidActionPlanError(
                f"ActionPlan metadata must be a mapping, got {type(self.metadata).__name__}."
            )
        for k, v in self.metadata.items():
            if not isinstance(k, str):
                raise InvalidActionPlanError(
                    f"Metadata key must be a string, got {k!r} ({type(k).__name__})."
                )
            _validate_metadata_value(v, f"metadata.{k}")

    def to_github_payload(self, publication_marker: str | None = None) -> dict[str, Any]:
        """
        Deterministic serialization of the ActionPlan into GitHub PR Review POST payload format.

        - Comments are sorted deterministically by (path, line or 0, start_line or 0, body).
        - If publication_marker is provided, appends '<!-- agy-review-marker: <marker> -->' to the body.
        """
        self.validate()
        final_body = self.summary
        if publication_marker:
            clean_marker = publication_marker.strip()
            if clean_marker:
                marker_str = f"<!-- agy-review-marker: {clean_marker} -->"
                if marker_str not in final_body:
                    final_body = f"{final_body.rstrip()}\n\n{marker_str}"

        sorted_comments = sorted(
            self.comments,
            key=lambda c: (c.path, c.line or 0, c.start_line or 0, c.body),
        )
        comments_payload = [c.to_github_payload() for c in sorted_comments]

        payload: dict[str, Any] = {
            "commit_id": self.head_sha.strip().lower(),
            "event": self.verdict,
            "body": final_body,
        }
        if comments_payload:
            payload["comments"] = comments_payload

        return payload


@dataclass(frozen=True)
class ApprovalResult:
    """Outcome of an approval gate evaluation."""

    approved: bool
    dry_run: bool
    plan: ActionPlan
    source: str  # "interactive", "non_interactive", "rejected", "dry_run"
    reason: str = ""
    plan_fingerprint: str = ""
    created_at_mono: float = 0.0
    approved_at_mono: float = 0.0
    approved_at_utc: str = ""
    expires_at_mono: float = 0.0
    approval_ttl: float = DEFAULT_APPROVAL_TTL_SECONDS
    clock: Callable[[], float] = field(default=time.monotonic, compare=False)

    def __post_init__(self) -> None:
        if not self.plan_fingerprint and self.plan is not None:
            object.__setattr__(self, "plan_fingerprint", self.plan.fingerprint)
        if self.approved and self.expires_at_mono <= 0.0 and self.approved_at_mono > 0.0:
            object.__setattr__(
                self, "expires_at_mono", self.approved_at_mono + self.approval_ttl
            )

    def is_expired(self, current_mono: float | None = None) -> bool:
        """
        Check if the approval window has expired.
        Measured strictly against expires_at_mono (at or beyond boundary is expired).
        """
        if not self.approved or self.expires_at_mono <= 0.0:
            return True
        now = current_mono if current_mono is not None else self.clock()
        return now >= self.expires_at_mono

    @property
    def can_write(self) -> bool:
        """
        Returns True ONLY if approved AND NOT dry_run AND unexpired AND plan fingerprint matches.
        Guarantees that --dry-run always blocks write actions, expired approvals cannot write,
        and plan has not been tampered with.
        """
        if not self.approved or self.dry_run:
            return False
        if self.is_expired():
            return False
        return self.plan.fingerprint == self.plan_fingerprint


def check_can_write(
    result: ApprovalResult,
    current_plan: ActionPlan | None = None,
    current_mono: float | None = None,
) -> bool:
    """
    Verify whether write operations can proceed.

    Ensures:
      1. Gate was approved.
      2. Operation is not dry_run.
      3. Approval has not expired (strictly within TTL measured from approval event).
      4. The result's plan fingerprint matches the recorded approved fingerprint.
      5. If current_plan is supplied, its fingerprint also matches the approved fingerprint.
    """
    if not result.approved or result.dry_run:
        return False
    if result.is_expired(current_mono=current_mono):
        return False
    if result.plan.fingerprint != result.plan_fingerprint:
        return False
    if current_plan is not None:
        return current_plan.fingerprint == result.plan_fingerprint
    return True


def verify_approval_for_write(
    result: ApprovalResult,
    current_plan: ActionPlan | None = None,
    current_mono: float | None = None,
) -> None:
    """
    Strictly verify that an approval result is valid and unexpired before publishing.
    Fails closed by raising specific typed errors.

    Raises:
        ApprovalRejectedError: If gate was not approved.
        DryRunBlockedError: If operation is in dry-run mode.
        ApprovalExpiredError: If approval window has elapsed.
        PlanFingerprintMismatchError: If plan or current_plan fingerprint has changed.
    """
    if not result.approved:
        raise ApprovalRejectedError(
            f"Write operation rejected: approval was not granted ({result.reason})."
        )
    if result.dry_run:
        raise DryRunBlockedError("Write operation blocked: running in dry-run mode.")
    if result.is_expired(current_mono=current_mono):
        now = current_mono if current_mono is not None else result.clock()
        elapsed = now - result.approved_at_mono
        raise ApprovalExpiredError(
            f"Approval has expired (elapsed {elapsed:.1f}s at or beyond {result.approval_ttl:.1f}s TTL). "
            "A fresh review and human authorization are required."
        )
    if result.plan.fingerprint != result.plan_fingerprint:
        raise PlanFingerprintMismatchError(
            "Plan tampering detected! The approved plan's fingerprint does not match the recorded authorization fingerprint."
        )
    if current_plan is not None and current_plan.fingerprint != result.plan_fingerprint:
        raise PlanFingerprintMismatchError(
            "Current plan does not match the approved plan fingerprint. Re-authorization required."
        )


# ============================================================================
# Preview Formatting
# ============================================================================


def format_approval_preview(plan: ActionPlan) -> str:
    """
    Format a clear, comprehensive human-readable preview of proposed GitHub actions.

    Includes:
      - Repository & PR number
      - Analyzed PR head SHA
      - Review verdict
      - Summary
      - Proposed review comments and their file/line locations
      - Whether the operation is a dry run
    """
    plan.validate()

    mode_banner = " [DRY RUN — NO WRITES WILL BE COMMITTED]" if plan.dry_run else ""
    lines: list[str] = [
        "=" * 78,
        f"           PROPOSED GITHUB REVIEW ACTION PLAN{mode_banner}",
        "=" * 78,
        f"Repository:       {plan.owner}/{plan.repo}",
        f"Pull Request:     #{plan.pr_number}",
        f"Target Head SHA:  {plan.head_sha}",
        f"Review Verdict:   {plan.verdict}",
        f"Plan Fingerprint: {plan.fingerprint}",
        f"Dry Run Mode:     {'YES (Writes blocked)' if plan.dry_run else 'NO (Live write if approved)'}",
        "-" * 78,
        "Summary:",
        f"  {plan.summary.strip()}",
        "-" * 78,
    ]

    if plan.comments:
        lines.append(f"Proposed Review Comments ({len(plan.comments)}):")
        for i, c in enumerate(plan.comments, 1):
            line_str = f"L{c.line}" if c.line is not None else "file-level"
            lines.append(f"  {i}. [{c.path}:{line_str}] (side={c.side})")
            # Indent comment snippet
            body_snippet = c.body.strip()
            if len(body_snippet) > 120:
                body_snippet = body_snippet[:117] + "..."
            lines.append(f'     "{body_snippet}"')
    else:
        lines.append("Proposed Review Comments: None (Top-level summary only)")

    lines.append("=" * 78)
    return "\n".join(lines)


# ============================================================================
# Approval Gate Core
# ============================================================================


def request_approval(
    plan: ActionPlan,
    interactive: bool | None = None,
    input_func: Callable[[str], str] | None = None,
    non_interactive_token: str | None = None,
    expected_non_interactive_token: str | None = None,
    output_func: Callable[[str], None] | None = None,
    clock: Callable[[], float] | None = None,
    utc_now: Callable[[], datetime] | None = None,
    max_plan_age_seconds: float = DEFAULT_MAX_PLAN_AGE_SECONDS,
    approval_ttl_seconds: float = DEFAULT_APPROVAL_TTL_SECONDS,
) -> ApprovalResult:
    """
    Evaluate human authorization for the proposed ActionPlan.

    Guarantees:
      1. Plan is strictly validated before any approval is requested.
      2. Plan staleness is checked: plans older than max_plan_age_seconds are rejected immediately.
      3. Shows complete formatted preview.
      4. Dry-run mode is explicitly displayed and permanently prevents writes.
      5. Interactive approval defaults to rejection (unambiguous '[y/N]' prompt).
      6. Non-interactive publishing is disabled and fails closed with NonInteractivePublishingDisabledError.
      7. LLM content cannot grant approval.
      8. Approval records monotonic approval time and enforces a 600-second approval window.

    Args:
        plan: The validated ActionPlan.
        interactive: Override for TTY check. If None, checks sys.stdin.isatty().
        input_func: Callable for reading user input (defaults to built-in input).
        non_interactive_token: Deprecated/disabled. Non-interactive publishing is prohibited.
        expected_non_interactive_token: Deprecated/disabled.
        output_func: Callable for printing preview (defaults to print).
        clock: Monotonic clock callable (defaults to time.monotonic).
        utc_now: UTC datetime callable (defaults to datetime.now(timezone.utc)).
        max_plan_age_seconds: Maximum allowed plan age before approval (default: 600.0s).
        approval_ttl_seconds: Approval validity window from approval event (default: 600.0s).

    Returns:
        ApprovalResult with approved=True and valid monotonic timestamps.

    Raises:
        InvalidActionPlanError: If plan data is invalid or incomplete.
        ApprovalExpiredError: If the plan is stale before approval.
        ApprovalRejectedError: If human rejects or aborts approval.
        NonInteractivePublishingDisabledError: If execution is non-interactive.
    """
    # 1. Strictly validate plan
    plan.validate()

    # 2. Clock setup and Staleness Check (Before Approval)
    _clock = clock or time.monotonic
    _utc_now = utc_now or (lambda: datetime.now(UTC))
    current_mono = _clock()

    if plan.created_at_mono is not None:
        plan_age = current_mono - plan.created_at_mono
        if plan_age > max_plan_age_seconds:
            raise ApprovalExpiredError(
                f"Action plan is stale (age {plan_age:.1f}s exceeds maximum allowed "
                f"{max_plan_age_seconds:.1f}s). A fresh review analysis is required."
            )

    # 3. Render preview
    out = output_func or print
    preview = format_approval_preview(plan)
    out(preview)

    # 4. Determine interactivity
    is_interactive = sys.stdin.isatty() if interactive is None else bool(interactive)
    _input = input_func or input

    # 5. Non-Interactive Mode: Publishing is strictly disabled
    if not is_interactive:
        raise NonInteractivePublishingDisabledError(
            "Non-interactive review publishing is disabled. "
            "Human approval must be granted interactively in an interactive terminal."
        )

    # 6. Interactive Mode
    prompt = (
        "Authorize publishing this review to GitHub? [y/N]: "
        if not plan.dry_run
        else "Acknowledge dry-run preview? [y/N]: "
    )
    try:
        user_choice = _input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise ApprovalRejectedError(
            "Approval aborted by user (EOF/Interrupt)."
        ) from None

    if user_choice in ("y", "yes"):
        approval_time_mono = _clock()
        approval_time_utc = _utc_now().isoformat()
        return ApprovalResult(
            approved=True,
            dry_run=plan.dry_run,
            plan=plan,
            source="interactive",
            reason="Explicit human approval granted via interactive prompt.",
            plan_fingerprint=plan.fingerprint,
            created_at_mono=plan.created_at_mono or approval_time_mono,
            approved_at_mono=approval_time_mono,
            approved_at_utc=approval_time_utc,
            expires_at_mono=approval_time_mono + approval_ttl_seconds,
            approval_ttl=approval_ttl_seconds,
            clock=_clock,
        )
    else:
        raise ApprovalRejectedError(
            f"Action plan rejected by user (response: {user_choice!r})."
        )


def verify_head_sha_unchanged(expected_sha: str, live_sha: str) -> bool:
    """
    Verify that the PR head SHA has not changed between analysis and publication.

    Prevents race conditions where new commits are pushed after approval.

    Raises:
        HeadShaMismatchError: If the live SHA differs from the approved SHA.
    """
    if not expected_sha or not live_sha:
        raise HeadShaMismatchError("Both expected and live head SHAs must be provided.")
    if expected_sha.strip().lower() != live_sha.strip().lower():
        raise HeadShaMismatchError(
            f"PR Head SHA mismatch! Approved analysis was for commit '{expected_sha}', "
            f"but current PR head is '{live_sha}'. Re-run review analysis on the latest commit."
        )
    return True
