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

import requests

logger = logging.getLogger(__name__)

# Patterns used for credential redaction
_TOKEN_PATTERNS = [
    re.compile(r"(?i)(authorization[:=]\s*(?:Bearer|token)\s+)[^\s,'\"\\]+"),
    re.compile(r"(?i)\b(bearer\s+)[a-zA-Z0-9_\-\.]+"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{16,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{16,}\b"),
]

_REPO_NAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")
_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def redact_token(text: str, token: str | None = None) -> str:
    """
    Scrub GitHub tokens, PATs, and Authorization credentials from strings.
    Never exposes raw secrets in logs, LLM prompts, or exception messages.
    """
    if not text:
        return ""
    sanitized = str(text)
    if token and len(token) >= 3:
        sanitized = sanitized.replace(token, "[REDACTED]")
    for pattern in _TOKEN_PATTERNS:
        if pattern.groups == 1:
            sanitized = pattern.sub(r"\1[REDACTED]", sanitized)
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

    def __init__(self, message: str, token: str | None = None) -> None:
        super().__init__(redact_token(message, token))


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


# ============================================================================
# GitHub Client
# ============================================================================


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
        raw_token = (
            token
            if token is not None
            else (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"))
        )
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
        elif require_token:
            raise GitHubAuthError(
                "Missing GitHub token. Provide a token or set GITHUB_TOKEN."
            )

        self._token: str | None = raw_token
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
            raise GitHubAuthError(
                f"GitHub authentication failed (HTTP 401 Unauthorized) for {owner}/{repo}.",
                token=self._token,
            )
        if status == 404:
            raise GitHubNotFoundError(
                f"GitHub resource not found: {owner}/{repo} PR #{pr_number} (HTTP 404).",
                token=self._token,
            )
        if status in (403, 429):
            remaining = response.headers.get("X-RateLimit-Remaining")
            retry_after_str = response.headers.get("Retry-After")
            reset_str = response.headers.get("X-RateLimit-Reset")
            retry_after = (
                int(retry_after_str)
                if retry_after_str and retry_after_str.isdigit()
                else None
            )
            reset_at = int(reset_str) if reset_str and reset_str.isdigit() else None

            is_rate_limit = (
                status == 429
                or remaining == "0"
                or "rate limit" in response.text.lower()
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
            raise GitHubClientError(
                f"GitHub API access forbidden: HTTP 403 for {owner}/{repo} PR #{pr_number}.",
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
            resp = self._session.get(url, headers=headers, timeout=self.timeout)
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

    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """
        Fetch raw unified diff text for a pull request from GitHub REST API.

        Raises:
            GitHubAuthError: If authentication fails (401) or credentials invalid.
            GitHubNotFoundError: If repository or PR does not exist (404).
            GitHubRateLimitError: If rate limit is exceeded (403/429).
            GitHubNetworkError: If connection or timeout occurs.
            GitHubClientError: On other HTTP errors.
        """
        num = self._validate_repo_params(owner, repo, pr_number)
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{num}"
        headers = self._get_headers(accept="application/vnd.github.v3.diff")

        try:
            resp = self._session.get(url, headers=headers, timeout=self.timeout)
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

        if resp.status_code != 200:
            self._handle_response_error(resp, owner, repo, num)

        return resp.text

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
    **kwargs: Any,
) -> str:
    """Convenience helper to fetch PR diff."""
    client = GitHubClient(token=token, base_url=base_url, timeout=timeout, **kwargs)
    return client.get_pr_diff(owner, repo, pr_number)
