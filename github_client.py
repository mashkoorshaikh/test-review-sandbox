"""
github_client.py - Minimal GitHub REST API client and Unified Diff Parser.

Stage 1 of Milestone 4:
  1. GitHubClient: Minimal REST API client for fetching PR metadata and PR diffs.
  2. PRMetadata: Dataclass for PR metadata (title, body, SHAs, author, branch, etc.).
  3. parse_diff_hunks: Unified diff parser extracting changed files, hunks, and
     valid new-side line numbers for inline review comments.
  4. DiffFile, DiffHunk, DiffLine: Structured representations of parsed diffs.
  5. Exception hierarchy: GitHubClientError, GitHubAuthError, GitHubRateLimitError,
     GitHubNotFoundError, GitHubNetworkError, DiffParsingError.
  6. Credential protection: Automatic redaction of auth tokens from all logs,
     representations, and exception messages.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# Maximum allowed size for raw pull request diff downloads (20 MB)
DEFAULT_MAX_DIFF_BYTES: int = 20 * 1024 * 1024

# Patterns used for credential redaction
# Registered active tokens across client instances and secure file sources
_REGISTERED_TOKENS: set[str] = set()


def register_sensitive_token(token: str | None) -> None:
    """
    Register a token to be globally scrubbed by redact_token.
    Prevents leaking credentials passed explicitly to client instances
    or loaded from secure credential files.
    """
    if token and isinstance(token, str):
        cleaned = token.strip()
        if len(cleaned) >= 4:
            _REGISTERED_TOKENS.add(cleaned)


def clear_sensitive_tokens() -> None:
    """Clear registered sensitive tokens (primarily used for test cleanup)."""
    _REGISTERED_TOKENS.clear()


def resolve_github_token(token: str | None = None) -> str | None:
    """
    Resolve GitHub credentials following strict precedence:
      1. Explicit function argument / CLI input (highest precedence).
      2. GITHUB_TOKEN environment variable.
      3. GH_TOKEN environment variable.
      4. GITHUB_TOKEN_FILE environment variable (path to token file).

    Registers any resolved token for automated log/exception redaction.
    """
    resolved, _ = resolve_github_token_with_source(token)
    return resolved


def resolve_github_token_with_source(
    token: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Resolve GitHub credentials and identify the source.

    Precedence:
      1. Explicit argument (or CLI --token)
      2. GITHUB_TOKEN env var
      3. GH_TOKEN env var
      4. GITHUB_TOKEN_FILE env var

    Returns:
        tuple[token, source] where source is one of:
        'explicit', 'GITHUB_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN_FILE', or None.
    """
    # 1. Explicit token argument
    if token is not None:
        cleaned = token.strip()
        if cleaned:
            register_sensitive_token(cleaned)
            return cleaned, "explicit"
        return "", "explicit"

    # 2. GITHUB_TOKEN environment variable
    env_gh = os.getenv("GITHUB_TOKEN")
    if env_gh is not None and env_gh.strip():
        cleaned = env_gh.strip()
        register_sensitive_token(cleaned)
        return cleaned, "GITHUB_TOKEN"

    # 3. GH_TOKEN environment variable
    env_gh_short = os.getenv("GH_TOKEN")
    if env_gh_short is not None and env_gh_short.strip():
        cleaned = env_gh_short.strip()
        register_sensitive_token(cleaned)
        return cleaned, "GH_TOKEN"

    # 4. GITHUB_TOKEN_FILE environment variable
    token_file = os.getenv("GITHUB_TOKEN_FILE")
    if token_file is not None and token_file.strip():
        file_path = Path(token_file.strip()).resolve()
        if not file_path.is_file():
            raise GitHubAuthError(
                f"Configured GITHUB_TOKEN_FILE does not exist or is not a file: {file_path}"
            )
        try:
            file_content = file_path.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise GitHubAuthError(
                f"Failed to read token from GITHUB_TOKEN_FILE '{file_path}': {e}"
            ) from None

        if not file_content:
            raise GitHubAuthError(
                f"Configured GITHUB_TOKEN_FILE '{file_path}' is empty."
            )
        register_sensitive_token(file_content)
        return file_content, "GITHUB_TOKEN_FILE"

    return None, None


_TOKEN_PATTERNS = [
    re.compile(r"(?i)(authorization[:=]\s*(?:Bearer|token)\s+)[^\s,'\"\\]+"),
    re.compile(r"(?i)\b(bearer\s+)[a-zA-Z0-9_\-\.]+"),
    re.compile(r"(?i)([?&](?:access_token|token)=)[^&\s'\"]+"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{16,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{16,}\b"),
    re.compile(r"\bsk-[a-zA-Z0-9_\-]{20,}\b"),
    re.compile(r"\bsk-proj-[a-zA-Z0-9_\-]{20,}\b"),
]

_REPO_NAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")
_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


def redact_token(text: str, token: str | None = None) -> str:
    """
    Scrub GitHub tokens, PATs, and Authorization credentials from strings.
    Never exposes raw secrets in logs, LLM prompts, or exception messages.
    """
    if not text:
        return ""
    sanitized = str(text)

    # 1. Redact explicit token argument if provided
    if token and isinstance(token, str):
        cleaned_token = token.strip()
        if len(cleaned_token) >= 3:
            sanitized = sanitized.replace(cleaned_token, "[REDACTED]")

    # 2. Redact all registered tokens (from GitHubClient instances or secure file inputs)
    for reg_token in _REGISTERED_TOKENS:
        if reg_token and len(reg_token) >= 4:
            sanitized = sanitized.replace(reg_token, "[REDACTED]")

    # 3. Redact environment variables
    for _env in ("GITHUB_TOKEN", "GH_TOKEN", "OPENAI_API_KEY"):
        _v = os.getenv(_env, "").strip()
        if _v and len(_v) >= 4:
            sanitized = sanitized.replace(_v, "[REDACTED]")

    # 4. Regex pattern scrubbing for standard token formats and URL parameters
    for pattern in _TOKEN_PATTERNS:
        if pattern.groups == 1:
            sanitized = pattern.sub(r"\1[REDACTED]", sanitized)
        else:
            if "sk-" in pattern.pattern:
                sanitized = pattern.sub("[REDACTED_API_KEY]", sanitized)
            else:
                sanitized = pattern.sub("[REDACTED_GH_TOKEN]", sanitized)
    return sanitized


class TokenRedactingFilter(logging.Filter):
    """Logging filter to scrub credentials from log records automatically."""

    def __init__(self, token: str | None = None) -> None:
        super().__init__()
        self.token = token

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_token(record.msg, self.token)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: redact_token(str(v), self.token) for k, v in record.args.items()
                }
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(
                    redact_token(str(v), self.token) for v in record.args
                )
        return True


# Apply token redacting filter to module logger
logger.addFilter(TokenRedactingFilter())


# ============================================================================
# Exception Hierarchy
# ============================================================================


class GitHubClientError(Exception):
    """Base exception for all GitHub client errors."""

    def __init__(
        self,
        message: str,
        token: str | None = None,
        response_message: str | None = None,
        sso_hint: str | None = None,
    ) -> None:
        super().__init__(redact_token(message, token))
        self.response_message = (
            redact_token(response_message, token) if response_message else None
        )
        self.sso_hint = redact_token(sso_hint, token) if sso_hint else None


class GitHubAuthError(GitHubClientError):
    """Raised on missing token, invalid token, or HTTP 401 Unauthorized."""


class GitHubRateLimitError(GitHubClientError):
    """Raised when GitHub API rate limits are exceeded (HTTP 403 or 429)."""

    def __init__(
        self,
        message: str,
        retry_after: int | None = None,
        reset_at: int | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(message, token=token)
        self.retry_after = retry_after
        self.reset_at = reset_at


class GitHubNotFoundError(GitHubClientError):
    """Raised when requested repository or PR does not exist (HTTP 404)."""


class GitHubNetworkError(GitHubClientError):
    """Raised on connection timeout or network connectivity failure."""


class GitHubUncertainWriteError(GitHubNetworkError):
    """Raised when a write operation encounters a timeout or network drop and its outcome is uncertain."""


class GitHubValidationError(GitHubClientError):
    """Raised when GitHub API rejects a payload as invalid or unprocessable (HTTP 422)."""

    def __init__(
        self,
        message: str,
        errors: list[dict[str, Any]] | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(message, token=token)
        self.errors = errors or []


class DiffParsingError(GitHubClientError):
    """Raised when diff format is invalid or cannot be parsed."""


# ============================================================================
# Data Models
# ============================================================================


@dataclass(frozen=True)
class PRMetadata:
    """Structured pull request metadata fetched from GitHub REST API."""

    owner: str
    repo: str
    pr_number: int
    title: str
    body: str
    state: str
    head_sha: str
    base_sha: str
    head_branch: str
    base_branch: str
    author: str
    html_url: str
    raw_data: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def number(self) -> int:
        return self.pr_number

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        return self.raw_data[item]

    def get(self, item: str, default: Any = None) -> Any:
        if hasattr(self, item):
            val = getattr(self, item)
            return val if val is not None else default
        return self.raw_data.get(item, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "repo": self.repo,
            "pr_number": self.pr_number,
            "title": self.title,
            "body": self.body,
            "state": self.state,
            "head_sha": self.head_sha,
            "base_sha": self.base_sha,
            "head_branch": self.head_branch,
            "base_branch": self.base_branch,
            "author": self.author,
            "html_url": self.html_url,
        }


@dataclass
class DiffLine:
    """Individual line inside a diff hunk."""

    line_type: str  # "context", "add", "delete", "marker"
    content: str
    old_line_number: int | None = None
    new_line_number: int | None = None


@dataclass
class DiffHunk:
    """A single change hunk within a diff file."""

    header: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section_heading: str
    lines: list[DiffLine] = field(default_factory=list)

    @property
    def valid_new_lines(self) -> set[int]:
        """Set of new-side line numbers (context + added) valid for inline comments."""
        return {
            line.new_line_number
            for line in self.lines
            if line.new_line_number is not None
        }

    @property
    def added_lines(self) -> set[int]:
        """Set of line numbers newly added in this hunk."""
        return {
            line.new_line_number
            for line in self.lines
            if line.line_type == "add" and line.new_line_number is not None
        }


@dataclass
class DiffFile:
    """Structured representation of changes for a single file in a diff."""

    old_path: str | None
    new_path: str | None
    status: str  # "modified", "added", "deleted", "renamed", "copied", "binary"
    is_binary: bool = False
    hunks: list[DiffHunk] = field(default_factory=list)
    similarity_index: int | None = None

    @property
    def path(self) -> str:
        """The primary file path for this change (new_path unless deleted)."""
        return self.new_path or self.old_path or ""

    @property
    def valid_new_lines(self) -> set[int]:
        """All line numbers on the new side that can receive inline comments."""
        lines: set[int] = set()
        for hunk in self.hunks:
            lines.update(hunk.valid_new_lines)
        return lines

    @property
    def added_lines(self) -> set[int]:
        """All line numbers added on the new side."""
        lines: set[int] = set()
        for hunk in self.hunks:
            lines.update(hunk.added_lines)
        return lines

    def is_line_in_diff(self, line_number: int) -> bool:
        """Check if a new-side line number is valid for inline review comments."""
        return line_number in self.valid_new_lines

    def get_diff_line(self, new_line_number: int) -> DiffLine | None:
        """Return the DiffLine for a given new-side line number, if present."""
        for hunk in self.hunks:
            for line in hunk.lines:
                if line.new_line_number == new_line_number:
                    return line
        return None

    def find_hunk_for_line(self, new_line_number: int) -> DiffHunk | None:
        """Return the DiffHunk containing a given new-side line number, if present."""
        for hunk in self.hunks:
            if new_line_number in hunk.valid_new_lines:
                return hunk
        return None

    def find_hunk_for_span(self, start_line: int, end_line: int) -> DiffHunk | None:
        """
        Return the DiffHunk containing the entire [start_line, end_line] span on the new side.
        Returns None if start_line > end_line or if the span crosses multiple hunks or falls outside diff hunks.
        """
        if start_line > end_line or start_line <= 0:
            return None
        for hunk in self.hunks:
            hunk_end = (
                hunk.new_start + hunk.new_count - 1
                if hunk.new_count > 0
                else hunk.new_start
            )
            if (
                hunk.new_start <= start_line
                and end_line <= hunk_end
                and start_line in hunk.valid_new_lines
                and end_line in hunk.valid_new_lines
            ):
                return hunk
        return None


# ============================================================================
# GitHub Client
# ============================================================================


def _get_response_header(response: requests.Response, name: str) -> str | None:
    """Safely extract a header value from a response with case-insensitive fallback."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    if hasattr(headers, "get"):
        val = headers.get(name)
        if val is not None:
            return str(val)
        target = name.casefold()
        if hasattr(headers, "items"):
            for k, v in headers.items():
                if isinstance(k, str) and k.casefold() == target:
                    return str(v)
    return None


class GitHubClient:
    """
    Minimal GitHub REST API client for code review workflow.
    """

    DEFAULT_BASE_URL = "https://api.github.com"
    API_VERSION = "2022-11-28"

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        session: requests.Session | None = None,
        require_token: bool = False,
    ) -> None:
        raw_token, token_source = resolve_github_token_with_source(token)
        if raw_token is not None:
            raw_token = raw_token.strip()
            if not raw_token:
                raise GitHubAuthError(
                    "Invalid GitHub token: token cannot be empty or whitespace."
                )
            if any(c in raw_token for c in "\r\n\x00"):
                raise GitHubAuthError(
                    "Invalid GitHub token: contains forbidden control characters."
                )
            register_sensitive_token(raw_token)
        elif require_token:
            raise GitHubAuthError(
                "Missing GitHub token. Provide a token, set GITHUB_TOKEN or GH_TOKEN, or specify GITHUB_TOKEN_FILE."
            )

        self._token: str | None = raw_token
        self._token_source: str | None = token_source
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout)
        self._session = session or requests.Session()

    def __repr__(self) -> str:
        return f"<GitHubClient base_url={self.base_url!r} authenticated={bool(self._token)}>"

    def __str__(self) -> str:
        return self.__repr__()

    def _sanitize(self, text: str) -> str:
        return redact_token(text, self._token)

    def _get_headers(
        self, accept: str = "application/vnd.github.v3+json"
    ) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "X-GitHub-Api-Version": self.API_VERSION,
            "User-Agent": "code-review-agent/1.0",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _validate_repo_params(self, owner: str, repo: str, pr_number: int) -> int:
        if not owner or not isinstance(owner, str) or not owner.strip():
            raise ValueError("Repository owner must be a non-empty string.")
        if not repo or not isinstance(repo, str) or not repo.strip():
            raise ValueError("Repository name must be a non-empty string.")
        if "/" in owner or "\\" in owner or ".." in owner:
            raise ValueError(f"Invalid repository owner: {owner!r}")
        if "/" in repo or "\\" in repo or ".." in repo:
            raise ValueError(f"Invalid repository name: {repo!r}")
        try:
            num = int(pr_number)
            if num <= 0:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(
                f"PR number must be a positive integer, got {pr_number!r}."
            )
        return num

    def _handle_response_error(
        self, response: requests.Response, owner: str, repo: str, pr_number: int
    ) -> None:
        status = response.status_code
        if status == 401:
            snippet = f": {response.text[:200]}" if response.text else "."
            raise GitHubAuthError(
                f"GitHub authentication failed (HTTP 401 Unauthorized) for {owner}/{repo}{snippet}",
                token=self._token,
            )
        if status == 404:
            raise GitHubNotFoundError(
                f"GitHub resource not found: {owner}/{repo} PR #{pr_number} (HTTP 404).",
                token=self._token,
            )
        if status in (403, 429):
            remaining = _get_response_header(response, "X-RateLimit-Remaining")
            retry_after_str = _get_response_header(response, "Retry-After")
            reset_str = _get_response_header(response, "X-RateLimit-Reset")
            retry_after = (
                int(retry_after_str)
                if retry_after_str and retry_after_str.isdigit()
                else None
            )
            reset_at = int(reset_str) if reset_str and reset_str.isdigit() else None

            raw_text = (
                response.text
                if hasattr(response, "text") and isinstance(response.text, str)
                else ""
            )

            # Attempt to extract GitHub's response JSON message
            github_msg = ""
            if hasattr(response, "json") and callable(response.json):
                try:
                    err_data = response.json()
                    if isinstance(err_data, dict):
                        raw_msg = err_data.get("message")
                        if isinstance(raw_msg, str) and raw_msg.strip():
                            github_msg = raw_msg.strip()
                except (
                    ValueError,
                    requests.exceptions.JSONDecodeError,
                    AttributeError,
                    TypeError,
                ):
                    pass

            is_rate_limit = (
                status == 429
                or remaining == "0"
                or "rate limit" in raw_text.lower()
                or "rate limit" in github_msg.lower()
            )
            if is_rate_limit:
                msg = f"GitHub API rate limit exceeded for {owner}/{repo}."
                if retry_after:
                    msg += f" Retry after {retry_after}s."
                elif reset_at:
                    msg += f" Rate limit resets at Unix timestamp {reset_at}."
                raise GitHubRateLimitError(
                    msg,
                    retry_after=retry_after,
                    reset_at=reset_at,
                    token=self._token,
                )

            # Preserve GitHub's response JSON message and diagnostic headers for non-rate-limit 403
            if not github_msg and raw_text.strip():
                github_msg = raw_text.strip()[:200]

            details: list[str] = []
            if github_msg:
                clean_msg = " ".join(github_msg.split())
                if len(clean_msg) > 250:
                    clean_msg = clean_msg[:247] + "..."
                details.append(clean_msg)

            sso_hint: str | None = None
            sso_header = _get_response_header(response, "X-GitHub-SSO")
            if sso_header and sso_header.strip():
                clean_sso = " ".join(sso_header.strip().split())
                if len(clean_sso) > 250:
                    clean_sso = clean_sso[:247] + "..."
                sso_hint = f"SSO authorization required: {clean_sso}"
                details.append(sso_hint)

            accepted_scopes = _get_response_header(response, "X-Accepted-OAuth-Scopes")
            oauth_scopes = _get_response_header(response, "X-OAuth-Scopes")
            scope_hints: list[str] = []
            if accepted_scopes and accepted_scopes.strip():
                scope_hints.append(f"required: {accepted_scopes.strip()}")
            if oauth_scopes and oauth_scopes.strip():
                scope_hints.append(f"token: {oauth_scopes.strip()}")
            if scope_hints:
                details.append(f"OAuth scopes ({', '.join(scope_hints)})")

            if details:
                diagnostic_str = "; ".join(details)
                safe_details = self._sanitize(diagnostic_str)
                snippet = f": {safe_details.rstrip('.')}."
            else:
                snippet = "."

            raise GitHubClientError(
                f"GitHub API access forbidden: HTTP 403 for {owner}/{repo} PR #{pr_number}{snippet}",
                token=self._token,
                response_message=github_msg or None,
                sso_hint=sso_hint,
            )
        if status == 422:
            error_details = ""
            errors_list: list[dict[str, Any]] = []
            try:
                err_data = response.json()
                msg = err_data.get("message", "")
                raw_errors = err_data.get("errors", [])
                if isinstance(raw_errors, list) and raw_errors:
                    errors_list = raw_errors
                    formatted = []
                    for e in raw_errors:
                        if isinstance(e, dict):
                            field_name = e.get("field", "")
                            code = e.get("code", "")
                            err_msg = e.get("message", "")
                            detail = f"{field_name}: {err_msg or code}".strip(": ")
                            formatted.append(detail)
                        else:
                            formatted.append(str(e))
                    if formatted:
                        error_details = f" ({'; '.join(formatted)})"
                elif msg:
                    error_details = f" ({msg})"
            except (ValueError, requests.exceptions.JSONDecodeError, AttributeError, KeyError):
                error_details = f": {response.text[:200]}" if response.text else ""
            raise GitHubValidationError(
                f"GitHub API validation failed (HTTP 422 Unprocessable Entity) for {owner}/{repo} PR #{pr_number}{error_details}.",
                errors=errors_list,
                token=self._token,
            )
        if 300 <= status < 400:
            location = response.headers.get("Location", "")
            loc_msg = f" to {location}" if location else "."
            raise GitHubClientError(
                f"Unexpected redirect response (HTTP {status}) from GitHub API for {owner}/{repo} PR #{pr_number}{loc_msg}",
                token=self._token,
            )
        if status >= 500:
            raise GitHubClientError(
                f"GitHub server error: HTTP {status} for {owner}/{repo} PR #{pr_number}.",
                token=self._token,
            )

        snippet = response.text[:200] if response.text else ""
        raise GitHubClientError(
            f"GitHub API request failed with HTTP {status}: {snippet}",
            token=self._token,
        )

    def get_authenticated_user(self) -> dict[str, Any]:
        """
        Fetch authenticated user / bot metadata from GitHub REST API (GET /user).

        Raises:
            GitHubAuthError: If authentication fails (401).
            GitHubRateLimitError: If rate limit exceeded (403/429).
            GitHubNetworkError: On timeout or connection failure.
            GitHubClientError: On other HTTP or JSON parsing errors.
        """
        url = f"{self.base_url}/user"
        headers = self._get_headers(accept="application/vnd.github+json")
        try:
            resp = self._session.get(
                url, headers=headers, timeout=self.timeout, allow_redirects=False
            )
        except requests.exceptions.Timeout as e:
            raise GitHubNetworkError(
                f"GitHub API user request timed out after {self.timeout}s: {e}",
                token=self._token,
            ) from None
        except requests.exceptions.ConnectionError as e:
            raise GitHubNetworkError(
                f"GitHub API user connection failed: {e}",
                token=self._token,
            ) from None
        except requests.exceptions.RequestException as e:
            raise GitHubClientError(
                f"GitHub API user request error: {e}",
                token=self._token,
            ) from None

        if resp.status_code != 200:
            snippet = f": {resp.text[:200]}" if resp.text else "."
            if resp.status_code == 401:
                raise GitHubAuthError(
                    f"GitHub authentication failed (HTTP 401 Unauthorized) for /user{snippet}",
                    token=self._token,
                )
            if resp.status_code in (403, 429):
                remaining = _get_response_header(resp, "X-RateLimit-Remaining")
                raw_u_text = (
                    resp.text
                    if hasattr(resp, "text") and isinstance(resp.text, str)
                    else ""
                )
                is_rate_limit = (
                    resp.status_code == 429
                    or remaining == "0"
                    or "rate limit" in raw_u_text.lower()
                )
                if is_rate_limit:
                    raise GitHubRateLimitError(
                        "GitHub API rate limit exceeded for /user.",
                        token=self._token,
                    )
                raise GitHubClientError(
                    f"GitHub user request failed with HTTP 403{snippet}",
                    token=self._token,
                )
            raise GitHubClientError(
                f"GitHub user request failed with HTTP {resp.status_code}{snippet}",
                token=self._token,
            )

        try:
            data = resp.json()
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            raise GitHubClientError(
                f"Failed to parse JSON response from GitHub user API: {e}",
                token=self._token,
            ) from None

        if not isinstance(data, dict):
            raise GitHubClientError(
                f"Unexpected response type from GitHub user API: expected dict, got {type(data).__name__}.",
                token=self._token,
            )

        return data

    def get_pr_metadata(self, owner: str, repo: str, pr_number: int) -> PRMetadata:
        """
        Fetch pull request metadata from GitHub REST API.

        Raises:
            GitHubAuthError: If authentication fails (401) or credentials invalid.
            GitHubNotFoundError: If repository or PR does not exist (404).
            GitHubRateLimitError: If rate limit is exceeded (403/429).
            GitHubNetworkError: If connection or timeout occurs.
            GitHubClientError: On other HTTP errors.
        """
        num = self._validate_repo_params(owner, repo, pr_number)
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{num}"
        headers = self._get_headers(accept="application/vnd.github.v3+json")

        try:
            resp = self._session.get(
                url, headers=headers, timeout=self.timeout, allow_redirects=False
            )
        except requests.exceptions.Timeout as e:
            raise GitHubNetworkError(
                f"GitHub API request timed out after {self.timeout}s: {e}",
                token=self._token,
            ) from None
        except requests.exceptions.ConnectionError as e:
            raise GitHubNetworkError(
                f"GitHub API connection failed: {e}",
                token=self._token,
            ) from None
        except requests.exceptions.RequestException as e:
            raise GitHubClientError(
                f"GitHub API request error: {e}",
                token=self._token,
            ) from None

        if resp.status_code != 200:
            self._handle_response_error(resp, owner, repo, num)

        try:
            data = resp.json()
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            raise GitHubClientError(
                f"Failed to parse JSON response from GitHub API: {e}",
                token=self._token,
            ) from None

        return PRMetadata(
            owner=owner,
            repo=repo,
            pr_number=num,
            title=data.get("title", ""),
            body=data.get("body") or "",
            state=data.get("state", "open"),
            head_sha=data.get("head", {}).get("sha", ""),
            base_sha=data.get("base", {}).get("sha", ""),
            head_branch=data.get("head", {}).get("ref", ""),
            base_branch=data.get("base", {}).get("ref", ""),
            author=data.get("user", {}).get("login", ""),
            html_url=data.get("html_url", ""),
            raw_data=data,
        )

    def get_pr_diff(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        max_bytes: int = DEFAULT_MAX_DIFF_BYTES,
    ) -> str:
        """
        Fetch raw unified diff text for a pull request from GitHub REST API.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.
            max_bytes: Maximum allowed diff size in bytes (default: 20 MB).

        Raises:
            GitHubAuthError: If authentication fails (401) or credentials invalid.
            GitHubNotFoundError: If repository or PR does not exist (404).
            GitHubRateLimitError: If rate limit is exceeded (403/429).
            GitHubNetworkError: If connection or timeout occurs.
            GitHubClientError: On other HTTP errors or if diff exceeds max_bytes.
        """
        num = self._validate_repo_params(owner, repo, pr_number)
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{num}"
        headers = self._get_headers(accept="application/vnd.github.v3.diff")

        resp = None
        try:
            resp = self._session.get(
                url, headers=headers, stream=True, timeout=self.timeout, allow_redirects=False
            )
            if resp.status_code != 200:
                self._handle_response_error(resp, owner, repo, num)

            # Stream response content while enforcing max_bytes limit
            chunks: list[bytes] = []
            total_bytes = 0
            if hasattr(resp, "iter_content") and callable(resp.iter_content):
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        total_bytes += len(chunk)
                        if total_bytes > max_bytes:
                            raise GitHubClientError(
                                f"PR diff exceeded maximum allowed size of {max_bytes} bytes "
                                f"for {owner}/{repo} PR #{num}.",
                                token=self._token,
                            )
                        chunks.append(chunk)

            # Safely bounded fallback only for mock responses or non-streaming responses where iter_content was not configured
            if not chunks and hasattr(resp, "text") and isinstance(resp.text, str) and resp.text:
                raw_text = resp.text
                text_bytes = raw_text.encode("utf-8", errors="replace")
                if len(text_bytes) > max_bytes:
                    raise GitHubClientError(
                        f"PR diff exceeded maximum allowed size of {max_bytes} bytes "
                        f"for {owner}/{repo} PR #{num}.",
                        token=self._token,
                    )
                return raw_text

            diff_bytes = b"".join(chunks)
            encoding = getattr(resp, "encoding", None) or "utf-8"
            try:
                return diff_bytes.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                return diff_bytes.decode("utf-8", errors="replace")

        except (
            GitHubClientError,
            GitHubAuthError,
            GitHubNotFoundError,
            GitHubRateLimitError,
            GitHubNetworkError,
        ):
            raise
        except requests.exceptions.Timeout as e:
            raise GitHubNetworkError(
                f"GitHub API diff request timed out after {self.timeout}s: {e}",
                token=self._token,
            ) from None
        except requests.exceptions.ConnectionError as e:
            raise GitHubNetworkError(
                f"GitHub API diff connection failed: {e}",
                token=self._token,
            ) from None
        except requests.exceptions.RequestException as e:
            raise GitHubClientError(
                f"GitHub API diff request error: {e}",
                token=self._token,
            ) from None
        finally:
            if resp is not None and hasattr(resp, "close") and callable(resp.close):
                resp.close()

    def download_commit_archive(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
        dest_path: Path | None = None,
        max_bytes: int = 50 * 1024 * 1024,
    ) -> Path:
        """
        Download repository source zipball for an exact commit SHA.

        Raises:
            GitHubAuthError: If unauthorized.
            GitHubNotFoundError: If repository or commit SHA not found.
            GitHubRateLimitError: If rate limit exceeded.
            GitHubNetworkError: On timeout or connection error.
            GitHubClientError: On HTTP error or archive exceeding max_bytes.
        """
        if not owner or not _REPO_NAME_RE.match(owner):
            raise ValueError(f"Invalid repository owner: {owner!r}")
        if not repo or not _REPO_NAME_RE.match(repo):
            raise ValueError(f"Invalid repository name: {repo!r}")
        if not commit_sha or not _GIT_SHA_RE.match(commit_sha):
            raise ValueError(f"Invalid commit SHA: {commit_sha!r}")

        url = f"{self.base_url}/repos/{owner}/{repo}/zipball/{commit_sha}"
        headers = self._get_headers(accept="application/vnd.github+json")

        if dest_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_f:
                out_path = Path(tmp_f.name)
        else:
            out_path = Path(dest_path)

        try:
            with self._session.get(
                url, headers=headers, stream=True, timeout=self.timeout
            ) as resp:
                if resp.status_code != 200:
                    self._handle_response_error(resp, owner, repo, 0)

                # Validate redirect origin if 3xx redirect was followed
                if hasattr(resp, "history") and resp.history:
                    for prev_resp in resp.history:
                        loc = prev_resp.headers.get("Location", "")
                        if loc:
                            parsed_loc = urlparse(loc)
                            loc_host = parsed_loc.netloc.lower()
                            if loc_host != "github.com" and not loc_host.endswith(".github.com"):
                                raise GitHubClientError(
                                    f"Untrusted redirect origin detected during archive download: {loc_host!r}",
                                    token=self._token,
                                )

                total_downloaded = 0
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            total_downloaded += len(chunk)
                            if total_downloaded > max_bytes:
                                raise GitHubClientError(
                                    f"Archive download exceeded maximum allowed size of {max_bytes} bytes.",
                                    token=self._token,
                                )
                            f.write(chunk)
            return out_path
        except requests.exceptions.Timeout as e:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            raise GitHubNetworkError(
                f"GitHub API archive download timed out after {self.timeout}s: {e}",
                token=self._token,
            ) from None
        except requests.exceptions.ConnectionError as e:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            raise GitHubNetworkError(
                f"GitHub API archive download connection failed: {e}",
                token=self._token,
            ) from None
        except Exception:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            raise

    def create_pull_request_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        body: str,
        event: str = "COMMENT",
        comments: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    ) -> dict[str, Any]:
        """
        Submit a pull request review with top-level summary and inline comments (POST).

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.
            commit_sha: Hexadecimal git commit SHA to anchor the review.
            body: Top-level review markdown summary.
            event: Review verdict. Must be one of 'COMMENT', 'REQUEST_CHANGES', or 'APPROVE'.
            comments: Optional list of inline comment dictionaries.

        Returns:
            dict containing the created GitHub review object.

        Raises:
            ValueError: If input arguments or comment coordinates fail validation.
            GitHubAuthError: If unauthorized (HTTP 401).
            GitHubNotFoundError: If repository or PR not found (HTTP 404).
            GitHubRateLimitError: If rate limit exceeded (HTTP 403/429).
            GitHubValidationError: If payload rejected by GitHub (HTTP 422).
            GitHubUncertainWriteError: If POST request timed out or connection dropped.
            GitHubClientError: On other HTTP errors.
        """
        num = self._validate_repo_params(owner, repo, pr_number)
        if not _REPO_NAME_RE.match(owner):
            raise ValueError(f"Invalid repository owner: {owner!r}")
        if not _REPO_NAME_RE.match(repo):
            raise ValueError(f"Invalid repository name: {repo!r}")

        if not commit_sha or not isinstance(commit_sha, str) or not _GIT_SHA_RE.match(commit_sha.strip()):
            raise ValueError(f"Invalid commit SHA: {commit_sha!r}. Expected 7-40 hex git commit SHA.")
        clean_sha = commit_sha.strip().lower()

        if not isinstance(body, str) or not body.strip():
            raise ValueError("Review body must be a non-empty string.")
        clean_body = body.strip()

        clean_event = event.strip().upper() if isinstance(event, str) else ""
        if clean_event not in ("COMMENT", "REQUEST_CHANGES", "APPROVE"):
            raise ValueError(
                f"Invalid review event: {event!r}. Must be 'COMMENT', 'REQUEST_CHANGES', or 'APPROVE'."
            )

        validated_comments: list[dict[str, Any]] = []
        if comments:
            for idx, c in enumerate(comments):
                if not isinstance(c, dict):
                    raise TypeError(f"Comment at index {idx} must be a dictionary, got {type(c).__name__}.")

                path = c.get("path")
                if not path or not isinstance(path, str) or not path.strip():
                    raise ValueError(f"Comment at index {idx} has missing or empty 'path'.")
                clean_path = path.strip().replace("\\", "/")
                if ".." in clean_path or clean_path.startswith("/"):
                    raise ValueError(f"Comment at index {idx} has invalid path '{path}'.")

                c_body = c.get("body")
                if not c_body or not isinstance(c_body, str) or not c_body.strip():
                    raise ValueError(f"Comment at index {idx} has missing or empty 'body'.")

                line = c.get("line")
                if line is not None and (not isinstance(line, int) or line <= 0):
                    raise ValueError(f"Comment at index {idx} 'line' must be a positive integer, got {line!r}.")

                side = c.get("side", "RIGHT")
                if side not in ("LEFT", "RIGHT"):
                    raise ValueError(f"Comment at index {idx} 'side' must be 'LEFT' or 'RIGHT', got {side!r}.")

                comment_obj: dict[str, Any] = {
                    "path": clean_path,
                    "body": c_body.strip(),
                    "side": side,
                }
                if line is not None:
                    comment_obj["line"] = line

                start_line = c.get("start_line")
                if start_line is not None:
                    if not isinstance(start_line, int) or start_line <= 0:
                        raise ValueError(
                            f"Comment at index {idx} 'start_line' must be a positive integer, got {start_line!r}."
                        )
                    if line is not None and start_line > line:
                        raise ValueError(
                            f"Comment at index {idx} 'start_line' ({start_line}) cannot be greater than 'line' ({line})."
                        )
                    comment_obj["start_line"] = start_line
                    start_side = c.get("start_side", side)
                    if start_side not in ("LEFT", "RIGHT"):
                        raise ValueError(
                            f"Comment at index {idx} 'start_side' must be 'LEFT' or 'RIGHT', got {start_side!r}."
                        )
                    comment_obj["start_side"] = start_side

                validated_comments.append(comment_obj)

        payload: dict[str, Any] = {
            "commit_id": clean_sha,
            "body": clean_body,
            "event": clean_event,
        }
        if validated_comments:
            payload["comments"] = validated_comments

        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{num}/reviews"
        headers = self._get_headers(accept="application/vnd.github+json")

        try:
            resp = self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                allow_redirects=False,
            )
        except requests.exceptions.Timeout as e:
            raise GitHubUncertainWriteError(
                f"GitHub review creation request timed out after {self.timeout}s for {owner}/{repo} PR #{num}: {e}. "
                "Outcome is uncertain; check PR review state before retrying.",
                token=self._token,
            ) from None
        except requests.exceptions.ConnectionError as e:
            raise GitHubUncertainWriteError(
                f"GitHub review creation connection failed for {owner}/{repo} PR #{num}: {e}",
                token=self._token,
            ) from None
        except requests.exceptions.RequestException as e:
            raise GitHubClientError(
                f"GitHub review creation request error for {owner}/{repo} PR #{num}: {e}",
                token=self._token,
            ) from None

        if resp.status_code not in (200, 201):
            self._handle_response_error(resp, owner, repo, num)

        try:
            data = resp.json()
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            raise GitHubClientError(
                f"Failed to parse JSON response from GitHub review creation API: {e}",
                token=self._token,
            ) from None

        if not isinstance(data, dict):
            raise GitHubClientError(
                f"Unexpected response type from GitHub review creation API: expected dict, got {type(data).__name__}.",
                token=self._token,
            )

        return data

    def get_pr_reviews(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        per_page: int = 100,
        max_pages: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Fetch all reviews for a pull request, following GitHub Link pagination headers safely.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.
            per_page: Items per page (default 100, max 100).
            max_pages: Maximum pages to retrieve to prevent unbounded loops.

        Returns:
            List of review dictionaries from GitHub REST API.

        Raises:
            GitHubAuthError: If authentication fails (401).
            GitHubNotFoundError: If repository or PR not found (404).
            GitHubRateLimitError: If rate limit exceeded.
            GitHubNetworkError: On timeout or connection failure.
            GitHubClientError: On invalid response or HTTP errors.
        """
        num = self._validate_repo_params(owner, repo, pr_number)
        safe_per_page = max(1, min(int(per_page), 100))
        safe_max_pages = max(1, int(max_pages))

        next_url: str | None = f"{self.base_url}/repos/{owner}/{repo}/pulls/{num}/reviews?per_page={safe_per_page}"
        headers = self._get_headers(accept="application/vnd.github+json")

        all_reviews: list[dict[str, Any]] = []
        pages_fetched = 0
        seen_urls: set[str] = set()

        base_parsed = urlparse(self.base_url)
        expected_path = f"{base_parsed.path.rstrip('/')}/repos/{owner}/{repo}/pulls/{num}/reviews"

        while next_url and pages_fetched < safe_max_pages:
            pages_fetched += 1
            seen_urls.add(next_url)

            # SSRF and path tampering defense: ensure next_url does not escape expected API base scheme, host, and path
            next_parsed = urlparse(next_url)
            if (
                next_parsed.scheme != base_parsed.scheme
                or next_parsed.netloc.lower() != base_parsed.netloc.lower()
                or next_parsed.path.rstrip("/") != expected_path
            ):
                raise GitHubClientError(
                    f"Untrusted pagination URL detected in Link header: {next_url!r}",
                    token=self._token,
                )

            try:
                resp = self._session.get(
                    next_url,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=False,
                )
            except requests.exceptions.Timeout as e:
                raise GitHubNetworkError(
                    f"GitHub API reviews request timed out after {self.timeout}s: {e}",
                    token=self._token,
                ) from None
            except requests.exceptions.ConnectionError as e:
                raise GitHubNetworkError(
                    f"GitHub API reviews connection failed: {e}",
                    token=self._token,
                ) from None
            except requests.exceptions.RequestException as e:
                raise GitHubClientError(
                    f"GitHub API reviews request error: {e}",
                    token=self._token,
                ) from None

            if resp.status_code != 200:
                self._handle_response_error(resp, owner, repo, num)

            try:
                data = resp.json()
            except (ValueError, requests.exceptions.JSONDecodeError) as e:
                raise GitHubClientError(
                    f"Failed to parse JSON response from GitHub reviews API: {e}",
                    token=self._token,
                ) from None

            if not isinstance(data, list):
                raise GitHubClientError(
                    f"Unexpected response type from GitHub reviews API: expected list, got {type(data).__name__}.",
                    token=self._token,
                )

            for item in data:
                if isinstance(item, dict):
                    all_reviews.append(item)

            # Follow Link header rel="next" safely
            candidate_next = None
            if hasattr(resp, "links") and isinstance(resp.links, dict):
                next_info = resp.links.get("next")
                if next_info and isinstance(next_info, dict) and "url" in next_info:
                    candidate_next = next_info["url"]

            if not candidate_next and resp.headers.get("Link"):
                match = _LINK_NEXT_RE.search(resp.headers["Link"])
                if match:
                    candidate_next = match.group(1)

            if candidate_next and candidate_next not in seen_urls:
                next_url = candidate_next
            else:
                next_url = None

        return all_reviews

    def parse_diff_hunks(self, diff_text: str, strict: bool = True) -> list[DiffFile]:
        """Convenience wrapper for parse_diff_hunks."""
        return parse_diff_hunks(diff_text, strict=strict)


# ============================================================================
# Unified Diff Parser
# ============================================================================

_DIFF_GIT_RE = re.compile(
    r"^diff --git (?:\"a/(.+?)\"|a/(.+?)) (?:\"b/(.+?)\"|b/(.+?))$"
)
_HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:[ \t]+(.*))?$"
)
_SIMILARITY_RE = re.compile(r"^similarity index (\d+)%$")
_BINARY_FILES_RE = re.compile(
    r"^Binary files (?:\"?a/.*\"?|/dev/null) and (?:\"?b/.*\"?|/dev/null) differ$"
)


def _clean_diff_path(path_str: str, prefix: str) -> str | None:
    cleaned = path_str.strip().strip('"')
    if cleaned in ("/dev/null", "dev/null"):
        return None
    if cleaned.startswith(prefix):
        return cleaned[len(prefix) :]
    return cleaned


def parse_diff_hunks(diff_text: str, strict: bool = True) -> list[DiffFile]:
    """
    Parse a unified diff string into structured DiffFile, DiffHunk, and DiffLine objects.

    Extracts:
      - Changed files and modification statuses (modified, added, deleted, renamed, binary).
      - Hunk coordinates (old/new start lines and line counts).
      - Valid new-side line numbers for anchoring GitHub inline review comments.
      - Line additions and deletions.

    Args:
        diff_text: Raw unified diff string from Git or GitHub REST API.
        strict: If True, raises DiffParsingError on malformed hunk headers or lines.
                If False, logs warnings and recovers where possible.

    Returns:
        List of parsed DiffFile objects.

    Raises:
        DiffParsingError: If diff syntax is corrupted, malformed, or invalid in strict mode.
    """
    if not diff_text or not diff_text.strip():
        return []

    lines = diff_text.splitlines()
    files: list[DiffFile] = []

    current_file: DiffFile | None = None
    current_hunk: DiffHunk | None = None
    current_old_line = 0
    current_new_line = 0

    has_diff_headers = False

    def finalize_hunk() -> None:
        nonlocal current_hunk
        if current_file is not None and current_hunk is not None:
            current_file.hunks.append(current_hunk)
            current_hunk = None

    def finalize_file() -> None:
        nonlocal current_file
        finalize_hunk()
        if current_file is not None:
            # Adjust status based on paths and binary flag
            if current_file.old_path is None and current_file.new_path is not None:
                current_file.status = "added"
            elif current_file.new_path is None and current_file.old_path is not None:
                current_file.status = "deleted"
            elif current_file.is_binary and current_file.status == "modified":
                current_file.status = "binary"
            files.append(current_file)
            current_file = None

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # Check for diff --git file header
        diff_git_match = _DIFF_GIT_RE.match(line)
        if diff_git_match:
            has_diff_headers = True
            finalize_file()
            old_p = diff_git_match.group(1) or diff_git_match.group(2)
            new_p = diff_git_match.group(3) or diff_git_match.group(4)
            current_file = DiffFile(
                old_path=old_p,
                new_path=new_p,
                status="modified",
                is_binary=False,
                hunks=[],
            )
            i += 1
            continue

        # Check for standalone --- a/... +++ b/... without diff --git
        if line.startswith("--- ") and current_file is None:
            has_diff_headers = True
            old_p = _clean_diff_path(line[4:], "a/")
            next_line = lines[i + 1] if i + 1 < n else ""
            if next_line.startswith("+++ "):
                new_p = _clean_diff_path(next_line[4:], "b/")
                current_file = DiffFile(
                    old_path=old_p,
                    new_path=new_p,
                    status="modified",
                    is_binary=False,
                    hunks=[],
                )
                i += 2
                continue

        # Process metadata headers inside a file before hunks
        if current_file is not None and current_hunk is None:
            if line.startswith("new file mode"):
                current_file.status = "added"
                current_file.old_path = None
                i += 1
                continue
            if line.startswith("deleted file mode"):
                current_file.status = "deleted"
                current_file.new_path = None
                i += 1
                continue
            sim_match = _SIMILARITY_RE.match(line)
            if sim_match:
                current_file.similarity_index = int(sim_match.group(1))
                i += 1
                continue
            if line.startswith("rename from "):
                current_file.old_path = line[12:].strip().strip('"')
                current_file.status = "renamed"
                i += 1
                continue
            if line.startswith("rename to "):
                current_file.new_path = line[10:].strip().strip('"')
                current_file.status = "renamed"
                i += 1
                continue
            if line.startswith("copy from "):
                current_file.old_path = line[10:].strip().strip('"')
                current_file.status = "copied"
                i += 1
                continue
            if line.startswith("copy to "):
                current_file.new_path = line[8:].strip().strip('"')
                current_file.status = "copied"
                i += 1
                continue
            if _BINARY_FILES_RE.match(line) or line.startswith("Binary files "):
                current_file.is_binary = True
                if current_file.status == "modified":
                    current_file.status = "binary"
                i += 1
                continue
            if line.startswith("GIT binary patch"):
                current_file.is_binary = True
                if current_file.status == "modified":
                    current_file.status = "binary"
                i += 1
                continue
            if line.startswith("--- "):
                old_p = _clean_diff_path(line[4:], "a/")
                if old_p is None:
                    current_file.old_path = None
                    if current_file.status == "modified":
                        current_file.status = "added"
                else:
                    current_file.old_path = old_p
                i += 1
                continue
            if line.startswith("+++ "):
                new_p = _clean_diff_path(line[4:], "b/")
                if new_p is None:
                    current_file.new_path = None
                    if current_file.status == "modified":
                        current_file.status = "deleted"
                else:
                    current_file.new_path = new_p
                i += 1
                continue

        # Check for hunk header @@ ... @@
        if line.startswith("@@"):
            has_diff_headers = True
            hunk_match = _HUNK_HEADER_RE.match(line)
            if not hunk_match:
                if strict:
                    raise DiffParsingError(f"Malformed diff hunk header: {line!r}")
                i += 1
                continue

            finalize_hunk()

            old_start = int(hunk_match.group(1))
            old_count = (
                int(hunk_match.group(2)) if hunk_match.group(2) is not None else 1
            )
            new_start = int(hunk_match.group(3))
            new_count = (
                int(hunk_match.group(4)) if hunk_match.group(4) is not None else 1
            )
            heading = (hunk_match.group(5) or "").strip()

            current_old_line = old_start
            current_new_line = new_start

            current_hunk = DiffHunk(
                header=line,
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                section_heading=heading,
                lines=[],
            )
            i += 1
            continue

        # Process lines within a hunk
        if current_hunk is not None:
            if line.startswith("+"):
                current_hunk.lines.append(
                    DiffLine(
                        line_type="add",
                        content=line[1:],
                        old_line_number=None,
                        new_line_number=current_new_line,
                    )
                )
                current_new_line += 1
                i += 1
                continue
            elif line.startswith("-"):
                current_hunk.lines.append(
                    DiffLine(
                        line_type="delete",
                        content=line[1:],
                        old_line_number=current_old_line,
                        new_line_number=None,
                    )
                )
                current_old_line += 1
                i += 1
                continue
            elif line.startswith(" ") or line == "":
                content = line[1:] if line.startswith(" ") else ""
                current_hunk.lines.append(
                    DiffLine(
                        line_type="context",
                        content=content,
                        old_line_number=current_old_line,
                        new_line_number=current_new_line,
                    )
                )
                current_old_line += 1
                current_new_line += 1
                i += 1
                continue
            elif line.startswith("\\"):
                # "\ No newline at end of file"
                current_hunk.lines.append(
                    DiffLine(
                        line_type="marker",
                        content=line,
                        old_line_number=None,
                        new_line_number=None,
                    )
                )
                i += 1
                continue
            else:
                # Unexpected line inside a hunk
                if strict:
                    file_desc = current_file.path if current_file else "unknown"
                    raise DiffParsingError(
                        f"Malformed diff line in hunk for '{file_desc}': {line!r}"
                    )
                # If non-strict, ignore/skip line
                i += 1
                continue

        # Lines outside files / hunks (e.g. index headers, commit messages before diff)
        i += 1

    finalize_file()

    if strict and not has_diff_headers and diff_text.strip():
        raise DiffParsingError(
            "Malformed diff: input contains no valid diff or hunk headers."
        )

    return files


# ============================================================================
# Module-level Convenience Functions
# ============================================================================


def get_pr_metadata(
    owner: str,
    repo: str,
    pr_number: int,
    token: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
    **kwargs: Any,
) -> PRMetadata:
    """Convenience helper to fetch PR metadata."""
    client = GitHubClient(token=token, base_url=base_url, timeout=timeout, **kwargs)
    return client.get_pr_metadata(owner, repo, pr_number)


def get_pr_diff(
    owner: str,
    repo: str,
    pr_number: int,
    token: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
    max_bytes: int | None = None,
    **kwargs: Any,
) -> str:
    """Convenience helper to fetch PR diff."""
    client = GitHubClient(token=token, base_url=base_url, timeout=timeout, **kwargs)
    if max_bytes is not None:
        return client.get_pr_diff(owner, repo, pr_number, max_bytes=max_bytes)
    return client.get_pr_diff(owner, repo, pr_number)


def create_pull_request_review(
    owner: str,
    repo: str,
    pr_number: int,
    commit_sha: str,
    body: str,
    event: str = "COMMENT",
    comments: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    token: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience helper to create a pull request review."""
    client = GitHubClient(token=token, base_url=base_url, timeout=timeout, **kwargs)
    return client.create_pull_request_review(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        commit_sha=commit_sha,
        body=body,
        event=event,
        comments=comments,
    )


def get_pr_reviews(
    owner: str,
    repo: str,
    pr_number: int,
    token: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
    per_page: int = 100,
    max_pages: int = 10,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Convenience helper to fetch all reviews for a pull request."""
    client = GitHubClient(token=token, base_url=base_url, timeout=timeout, **kwargs)
    return client.get_pr_reviews(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        per_page=per_page,
        max_pages=max_pages,
    )
