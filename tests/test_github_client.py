"""
tests/test_github_client.py - Unit and integration tests for GitHub REST client and Diff Parser.

All HTTP requests are mocked so tests require no network access or credentials.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest
import requests

from github_client import (
    DiffParsingError,
    GitHubAuthError,
    GitHubClient,
    GitHubClientError,
    GitHubNetworkError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    GitHubUncertainWriteError,
    GitHubValidationError,
    PRMetadata,
    TokenRedactingFilter,
    clear_sensitive_tokens,
    create_pull_request_review,
    get_pr_diff,
    get_pr_metadata,
    get_pr_reviews,
    parse_diff_hunks,
    redact_token,
    register_sensitive_token,
    resolve_github_token,
    resolve_github_token_with_source,
)

# ============================================================================
# Client Initialization, Authentication, and Secret Protection Tests
# ============================================================================


class TestGitHubClientAuthAndInit:
    """Test client initialization, token discovery, and credential safety."""

    def test_explicit_token_sets_authorization_header(self):
        client = GitHubClient(token="ghp_explicit_secret_token_1234567890")
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer ghp_explicit_secret_token_1234567890"
        assert headers["X-GitHub-Api-Version"] == "2022-11-28"

    def test_env_var_github_token_is_discovered(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_env_secret_token_1234567890")
        monkeypatch.delenv("GH_TOKEN", raising=False)
        client = GitHubClient()
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer ghp_env_secret_token_1234567890"

    def test_env_var_gh_token_fallback(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "ghp_gh_token_secret_1234567890")
        client = GitHubClient()
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer ghp_gh_token_secret_1234567890"

    def test_unauthenticated_client_has_no_auth_header(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        client = GitHubClient()
        headers = client._get_headers()
        assert "Authorization" not in headers

    def test_require_token_raises_when_no_token_present(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        with pytest.raises(GitHubAuthError, match="Missing GitHub token"):
            GitHubClient(require_token=True)

    def test_empty_or_whitespace_token_raises_auth_error(self):
        with pytest.raises(GitHubAuthError, match="token cannot be empty"):
            GitHubClient(token="   ")

    def test_token_with_control_chars_raises_auth_error(self):
        with pytest.raises(GitHubAuthError, match="forbidden control characters"):
            GitHubClient(token="ghp_secret\nnewline")

    def test_client_repr_and_str_never_contain_token(self):
        secret = "ghp_super_secret_pat_987654321012345"
        client = GitHubClient(token=secret)
        repr_str = repr(client)
        str_str = str(client)
        assert secret not in repr_str
        assert secret not in str_str
        assert "authenticated=True" in repr_str


# ============================================================================
# PR Metadata Retrieval Tests
# ============================================================================


class TestGetPRMetadata:
    """Test fetching PR metadata via GitHub REST API with mocked responses."""

    def test_get_pr_metadata_success(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "number": 42,
            "title": "Add feature X",
            "body": "Detailed PR description here",
            "state": "open",
            "head": {"sha": "aabbcc112233", "ref": "feature-x"},
            "base": {"sha": "ddeeff445566", "ref": "main"},
            "user": {"login": "octocat"},
            "html_url": "https://github.com/octocat/repo/pull/42",
        }

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(token="ghp_test_token_1234567890", session=mock_session)
        meta = client.get_pr_metadata("octocat", "repo", 42)

        assert isinstance(meta, PRMetadata)
        assert meta.owner == "octocat"
        assert meta.repo == "repo"
        assert meta.pr_number == 42
        assert meta.number == 42
        assert meta.title == "Add feature X"
        assert meta.body == "Detailed PR description here"
        assert meta.state == "open"
        assert meta.head_sha == "aabbcc112233"
        assert meta.base_sha == "ddeeff445566"
        assert meta.head_branch == "feature-x"
        assert meta.base_branch == "main"
        assert meta.author == "octocat"
        assert meta.html_url == "https://github.com/octocat/repo/pull/42"

        # Verify dict-like access and to_dict()
        assert meta["title"] == "Add feature X"
        assert meta.get("head_branch") == "feature-x"
        as_dict = meta.to_dict()
        assert as_dict["pr_number"] == 42
        assert as_dict["author"] == "octocat"

    def test_get_pr_metadata_handles_none_body(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "number": 1,
            "title": "Empty body PR",
            "body": None,
            "state": "open",
            "head": {"sha": "111", "ref": "patch"},
            "base": {"sha": "222", "ref": "main"},
            "user": {"login": "dev"},
            "html_url": "https://github.com/org/repo/pull/1",
        }
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        meta = client.get_pr_metadata("org", "repo", 1)
        assert meta.body == ""

    def test_get_pr_metadata_http_401_raises_auth_error(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 401
        mock_response.text = "Bad credentials"
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(token="ghp_bad_token_1234567890", session=mock_session)
        with pytest.raises(GitHubAuthError, match="HTTP 401 Unauthorized"):
            client.get_pr_metadata("owner", "repo", 10)

    def test_get_pr_metadata_http_404_raises_not_found_error(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubNotFoundError, match="HTTP 404"):
            client.get_pr_metadata("owner", "nonexistent-repo", 999)

    def test_get_pr_metadata_http_403_rate_limit(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 403
        mock_response.headers = {
            "X-RateLimit-Remaining": "0",
            "Retry-After": "120",
            "X-RateLimit-Reset": "1700000000",
        }
        mock_response.text = "API rate limit exceeded for user ID 123"
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubRateLimitError) as exc_info:
            client.get_pr_metadata("owner", "repo", 5)

        assert exc_info.value.retry_after == 120
        assert exc_info.value.reset_at == 1700000000
        assert "rate limit exceeded" in str(exc_info.value).lower()

    def test_get_pr_metadata_http_429_rate_limit(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "45"}
        mock_response.text = "Too Many Requests"
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubRateLimitError) as exc_info:
            client.get_pr_metadata("owner", "repo", 5)

        assert exc_info.value.retry_after == 45

    def test_get_pr_metadata_http_403_forbidden_non_rate_limit(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 403
        mock_response.headers = {"X-RateLimit-Remaining": "50"}
        mock_response.text = "Resource not accessible by integration"
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="HTTP 403"):
            client.get_pr_metadata("owner", "repo", 5)

    def test_get_pr_metadata_http_500_server_error(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="HTTP 500"):
            client.get_pr_metadata("owner", "repo", 5)

    def test_get_pr_metadata_timeout(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.side_effect = requests.exceptions.Timeout("Request timed out")

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubNetworkError, match="timed out"):
            client.get_pr_metadata("owner", "repo", 1)

    def test_get_pr_metadata_connection_error(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.side_effect = requests.exceptions.ConnectionError(
            "Connection refused"
        )

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubNetworkError, match="connection failed"):
            client.get_pr_metadata("owner", "repo", 1)

    def test_get_pr_metadata_invalid_params_raise_value_error(self):
        client = GitHubClient()
        with pytest.raises(ValueError, match="owner"):
            client.get_pr_metadata("", "repo", 1)
        with pytest.raises(ValueError, match="name"):
            client.get_pr_metadata("owner", "", 1)
        with pytest.raises(ValueError, match="positive integer"):
            client.get_pr_metadata("owner", "repo", 0)
        with pytest.raises(ValueError, match="positive integer"):
            client.get_pr_metadata("owner", "repo", -1)
        with pytest.raises(ValueError, match="Invalid repository owner"):
            client.get_pr_metadata("owner/escape", "repo", 1)


# ============================================================================
# PR Diff Retrieval Tests
# ============================================================================


class TestGetPRDiff:
    """Test fetching PR diff via GitHub REST API."""

    SAMPLE_DIFF = (
        "diff --git a/src/main.py b/src/main.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1,3 +1,4 @@\n"
        " import os\n"
        "+import sys\n"
        " def main():\n"
        "     pass\n"
    )

    def test_get_pr_diff_success(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.text = self.SAMPLE_DIFF

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        diff = client.get_pr_diff("owner", "repo", 7)

        assert diff == self.SAMPLE_DIFF
        # Verify requested Accept header was diff format
        call_kwargs = mock_session.get.call_args[1]
        assert call_kwargs["headers"]["Accept"] == "application/vnd.github.v3.diff"

    def test_get_pr_diff_401(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 401
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(token="ghp_invalid_1234567890", session=mock_session)
        with pytest.raises(GitHubAuthError):
            client.get_pr_diff("owner", "repo", 7)

    def test_get_pr_diff_404(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 404
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubNotFoundError):
            client.get_pr_diff("owner", "repo", 7)

    def test_get_pr_diff_timeout(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.side_effect = requests.exceptions.Timeout("Read timeout")

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubNetworkError, match="timed out"):
            client.get_pr_diff("owner", "repo", 7)

    def test_get_pr_diff_oversized_rejected_while_streaming(self):
        """Oversized diff must raise GitHubClientError and close response socket."""
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        # Yield two chunks of 60 bytes each = 120 bytes
        mock_response.iter_content.return_value = [b"A" * 60, b"B" * 60]

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="exceeded maximum allowed size of 100 bytes"):
            client.get_pr_diff("owner", "repo", 7, max_bytes=100)

        # Verify response stream was closed
        assert mock_response.close.called

    def test_get_pr_diff_streaming_normal_sized_success(self):
        """Streaming normal-sized diff succeeds and closes response socket."""
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.encoding = "utf-8"
        mock_response.iter_content.return_value = [
            b"diff --git a/foo.py b/foo.py\n",
            b"--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-a\n+b\n",
        ]

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        diff = client.get_pr_diff("owner", "repo", 7, max_bytes=1024)

        assert diff == "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-a\n+b\n"
        assert mock_response.close.called

    def test_get_pr_diff_resources_closed_on_error(self):
        """Response resources must be closed even when streaming encounters a network error."""
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.iter_content.side_effect = requests.exceptions.ConnectionError("Connection dropped mid-stream")

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubNetworkError, match="connection failed"):
            client.get_pr_diff("owner", "repo", 7)

        assert mock_response.close.called

    def test_get_pr_diff_empty_response(self):
        """Empty diff response returns an empty string and closes resources."""
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.iter_content.return_value = []

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        diff = client.get_pr_diff("owner", "repo", 7)
        assert diff == ""
        assert mock_response.close.called

    def test_get_pr_diff_malformed_encoding_fallback(self):
        """Malformed bytes decode safely using replacement characters and close response."""
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.encoding = "utf-8"
        # Invalid UTF-8 byte sequence
        mock_response.iter_content.return_value = [b"diff --git a/file.py \xff\xfe\n"]

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        diff = client.get_pr_diff("owner", "repo", 7)
        assert "diff --git a/file.py" in diff
        assert mock_response.close.called

    def test_get_pr_diff_mock_text_fallback_bounded_oversized(self):
        """In-memory mock text fallback is safely bounded and rejects oversized text."""
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.iter_content.return_value = []
        mock_response.text = "X" * 500

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="exceeded maximum allowed size of 100 bytes"):
            client.get_pr_diff("owner", "repo", 7, max_bytes=100)
        assert mock_response.close.called


# ============================================================================
# Unified Diff Parsing Tests
# ============================================================================


class TestParseDiffHunks:
    """Comprehensive tests for diff parsing covering all required file types and states."""

    def test_parse_empty_and_whitespace_diff(self):
        assert parse_diff_hunks("") == []
        assert parse_diff_hunks("   \n\n  \t  ") == []

    def test_parse_modified_file_single_hunk(self):
        diff = (
            "diff --git a/calc.py b/calc.py\n"
            "index 1234567..89abcdef 100644\n"
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -10,4 +10,5 @@ def calculate(a, b):\n"
            "     res = a + b\n"
            "-    res = res - 1\n"
            "+    res = res * 2\n"
            "+    res += 1\n"
            "     return res\n"
        )
        files = parse_diff_hunks(diff)
        assert len(files) == 1
        f = files[0]
        assert f.old_path == "calc.py"
        assert f.new_path == "calc.py"
        assert f.path == "calc.py"
        assert f.status == "modified"
        assert not f.is_binary
        assert len(f.hunks) == 1

        hunk = f.hunks[0]
        assert hunk.old_start == 10
        assert hunk.old_count == 4
        assert hunk.new_start == 10
        assert hunk.new_count == 5
        assert hunk.section_heading == "def calculate(a, b):"

        # Check valid lines for comments on new side
        # Lines on new side:
        # line 10: context ("     res = a + b") -> new_line_number=10
        # line 11: deleted ("-    res = res - 1") -> old_line_number=11, new=None
        # line 11: added ("+    res = res * 2") -> new_line_number=11
        # line 12: added ("+    res += 1") -> new_line_number=12
        # line 13: context ("     return res") -> new_line_number=13
        assert f.valid_new_lines == {10, 11, 12, 13}
        assert f.added_lines == {11, 12}
        assert f.is_line_in_diff(10)
        assert f.is_line_in_diff(11)
        assert f.is_line_in_diff(12)
        assert f.is_line_in_diff(13)
        assert not f.is_line_in_diff(9)
        assert not f.is_line_in_diff(14)

    def test_parse_added_file(self):
        diff = (
            "diff --git a/new_module.py b/new_module.py\n"
            "new file mode 100644\n"
            "index 0000000..1234567\n"
            "--- /dev/null\n"
            "+++ b/new_module.py\n"
            "@@ -0,0 +1,3 @@\n"
            "+def greet():\n"
            "+    return 'hello'\n"
            "+\n"
        )
        files = parse_diff_hunks(diff)
        assert len(files) == 1
        f = files[0]
        assert f.status == "added"
        assert f.old_path is None
        assert f.new_path == "new_module.py"
        assert f.path == "new_module.py"
        assert f.valid_new_lines == {1, 2, 3}
        assert f.added_lines == {1, 2, 3}

    def test_parse_deleted_file(self):
        diff = (
            "diff --git a/legacy.py b/legacy.py\n"
            "deleted file mode 100644\n"
            "index 1234567..0000000\n"
            "--- a/legacy.py\n"
            "+++ /dev/null\n"
            "@@ -1,2 +0,0 @@\n"
            "-def old():\n"
            "-    pass\n"
        )
        files = parse_diff_hunks(diff)
        assert len(files) == 1
        f = files[0]
        assert f.status == "deleted"
        assert f.old_path == "legacy.py"
        assert f.new_path is None
        assert f.path == "legacy.py"
        assert f.valid_new_lines == set()
        assert not f.is_line_in_diff(1)

    def test_parse_renamed_file_with_modifications(self):
        diff = (
            "diff --git a/old_name.py b/new_name.py\n"
            "similarity index 85%\n"
            "rename from old_name.py\n"
            "rename to new_name.py\n"
            "index 1111111..2222222 100644\n"
            "--- a/old_name.py\n"
            "+++ b/new_name.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def foo():\n"
            "-    return 'old'\n"
            "+    return 'new'\n"
            "     pass\n"
        )
        files = parse_diff_hunks(diff)
        assert len(files) == 1
        f = files[0]
        assert f.status == "renamed"
        assert f.old_path == "old_name.py"
        assert f.new_path == "new_name.py"
        assert f.similarity_index == 85
        assert f.valid_new_lines == {1, 2, 3}
        assert f.added_lines == {2}

    def test_parse_pure_rename_without_hunks(self):
        diff = (
            "diff --git a/original.txt b/moved.txt\n"
            "similarity index 100%\n"
            "rename from original.txt\n"
            "rename to moved.txt\n"
        )
        files = parse_diff_hunks(diff)
        assert len(files) == 1
        f = files[0]
        assert f.status == "renamed"
        assert f.old_path == "original.txt"
        assert f.new_path == "moved.txt"
        assert f.similarity_index == 100
        assert f.hunks == []
        assert f.valid_new_lines == set()

    def test_parse_binary_files(self):
        # 1. New binary file
        new_binary_diff = (
            "diff --git a/logo.png b/logo.png\n"
            "new file mode 100644\n"
            "index 0000000..abcdef1\n"
            "Binary files /dev/null and b/logo.png differ\n"
        )
        files = parse_diff_hunks(new_binary_diff)
        assert len(files) == 1
        assert files[0].is_binary
        assert files[0].status == "added"
        assert files[0].new_path == "logo.png"

        # 2. Modified binary file
        mod_binary_diff = (
            "diff --git a/icon.ico b/icon.ico\n"
            "index 1111111..2222222 100644\n"
            "Binary files a/icon.ico and b/icon.ico differ\n"
        )
        files = parse_diff_hunks(mod_binary_diff)
        assert len(files) == 1
        assert files[0].is_binary
        assert files[0].status == "binary"

        # 3. GIT binary patch
        git_bin_patch = (
            "diff --git a/data.bin b/data.bin\n"
            "index 1111111..2222222 100644\n"
            "GIT binary patch\n"
            "literal 12\n"
            "zc$@)z\n\n"
        )
        files = parse_diff_hunks(git_bin_patch)
        assert len(files) == 1
        assert files[0].is_binary

    def test_parse_multiple_hunks_in_single_file(self):
        diff = (
            "diff --git a/service.py b/service.py\n"
            "--- a/service.py\n"
            "+++ b/service.py\n"
            "@@ -5,3 +5,4 @@ class Service:\n"
            "     def init(self):\n"
            "+        self.ready = True\n"
            "         pass\n"
            "@@ -20,3 +21,4 @@ class Service:\n"
            "     def run(self):\n"
            "+        print('running')\n"
            "         return 0\n"
        )
        files = parse_diff_hunks(diff)
        assert len(files) == 1
        f = files[0]
        assert len(f.hunks) == 2

        hunk1, hunk2 = f.hunks
        assert hunk1.new_start == 5
        assert hunk1.valid_new_lines == {5, 6, 7}
        assert hunk1.added_lines == {6}

        assert hunk2.new_start == 21
        assert hunk2.valid_new_lines == {21, 22, 23}
        assert hunk2.added_lines == {22}

        # Combined line mapping
        assert f.valid_new_lines == {5, 6, 7, 21, 22, 23}
        assert f.is_line_in_diff(6)
        assert f.is_line_in_diff(22)
        assert not f.is_line_in_diff(10)  # Gap between hunks

    def test_parse_multiple_files_in_single_diff(self):
        diff = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,2 +1,3 @@\n"
            " x = 1\n"
            "+y = 2\n"
            " z = 3\n"
            "diff --git a/b.py b/b.py\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ -10,2 +10,3 @@\n"
            " a = 10\n"
            "+b = 20\n"
            " c = 30\n"
        )
        files = parse_diff_hunks(diff)
        assert len(files) == 2
        assert files[0].path == "a.py"
        assert files[1].path == "b.py"
        assert files[0].valid_new_lines == {1, 2, 3}
        assert files[1].valid_new_lines == {10, 11, 12}

    def test_parse_diff_with_quoted_paths_with_spaces(self):
        diff = (
            'diff --git "a/path with spaces/file.py" "b/path with spaces/file.py"\n'
            '--- "a/path with spaces/file.py"\n'
            '+++ "b/path with spaces/file.py"\n'
            "@@ -1,2 +1,3 @@\n"
            " # comment\n"
            "+x = 10\n"
            " y = 20\n"
        )
        files = parse_diff_hunks(diff)
        assert len(files) == 1
        assert files[0].path == "path with spaces/file.py"
        assert files[0].valid_new_lines == {1, 2, 3}

    def test_parse_diff_with_no_newline_marker(self):
        diff = (
            "diff --git a/test.txt b/test.txt\n"
            "--- a/test.txt\n"
            "+++ b/test.txt\n"
            "@@ -1 +1,2 @@\n"
            " line 1\n"
            "+line 2\n"
            "\\ No newline at end of file\n"
        )
        files = parse_diff_hunks(diff)
        assert len(files) == 1
        hunk = files[0].hunks[0]
        markers = [l for l in hunk.lines if l.line_type == "marker"]
        assert len(markers) == 1
        assert "No newline" in markers[0].content

    def test_parse_standalone_diff_without_diff_git_header(self):
        diff = (
            "--- a/standalone.py\n"
            "+++ b/standalone.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-foo\n"
            "+bar\n"
            " baz\n"
        )
        files = parse_diff_hunks(diff)
        assert len(files) == 1
        assert files[0].path == "standalone.py"
        assert files[0].valid_new_lines == {1, 2}

    def test_malformed_input_strict_mode_raises(self):
        # 1. Total garbage with no diff headers
        with pytest.raises(DiffParsingError, match="no valid diff or hunk headers"):
            parse_diff_hunks("just some random text\nwith no diff headers", strict=True)

        # 2. Malformed hunk header
        bad_hunk_diff = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ bad hunk header @@\n"
            "+x = 1\n"
        )
        with pytest.raises(DiffParsingError, match="Malformed diff hunk header"):
            parse_diff_hunks(bad_hunk_diff, strict=True)

        # 3. Malformed line inside a hunk
        bad_line_diff = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,2 +1,2 @@\n"
            " context line\n"
            "? unexpected line symbol\n"
        )
        with pytest.raises(DiffParsingError, match="Malformed diff line"):
            parse_diff_hunks(bad_line_diff, strict=True)

    def test_malformed_input_non_strict_mode_recovers(self):
        # Total garbage returns empty list in non-strict mode
        assert parse_diff_hunks("random non diff text", strict=False) == []

        # Malformed line inside hunk skipped gracefully
        diff = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,2 +1,2 @@\n"
            " context\n"
            "? unexpected line\n"
            "+added\n"
        )
        files = parse_diff_hunks(diff, strict=False)
        assert len(files) == 1
        assert 1 in files[0].valid_new_lines


# ============================================================================
# Inline Comment Line Mapping Helper Tests
# ============================================================================


class TestInlineCommentLineMapping:
    """Test precise line lookup, diff line extraction, and hunk association."""

    def setup_method(self):
        self.diff = (
            "diff --git a/example.py b/example.py\n"
            "--- a/example.py\n"
            "+++ b/example.py\n"
            "@@ -10,3 +10,4 @@\n"
            " line 10\n"
            "-line 11 old\n"
            "+line 11 new\n"
            "+line 12 added\n"
            " line 13\n"
            "@@ -50,2 +51,3 @@\n"
            " line 51\n"
            "+line 52 added\n"
            " line 53\n"
        )
        self.files = parse_diff_hunks(self.diff)
        self.f = self.files[0]

    def test_get_diff_line(self):
        # Context line 10
        line10 = self.f.get_diff_line(10)
        assert line10 is not None
        assert line10.line_type == "context"
        assert line10.content == "line 10"

        # Added line 11
        line11 = self.f.get_diff_line(11)
        assert line11 is not None
        assert line11.line_type == "add"
        assert line11.content == "line 11 new"

        # Out of range line
        assert self.f.get_diff_line(30) is None

    def test_find_hunk_for_line(self):
        hunk1 = self.f.find_hunk_for_line(12)
        assert hunk1 is not None
        assert hunk1.new_start == 10

        hunk2 = self.f.find_hunk_for_line(52)
        assert hunk2 is not None
        assert hunk2.new_start == 51

        # Unmodified line outside any hunk
        assert self.f.find_hunk_for_line(35) is None


# ============================================================================
# Credential Redaction & Security Tests
# ============================================================================


class TestCredentialRedaction:
    """Test that auth tokens are never leaked into errors, logs, or outputs."""

    def test_redact_token_string_variants(self):
        secret = "ghp_VerySecretTokenValue123456789"
        # Exact token match
        assert (
            redact_token(f"Failed with {secret}", token=secret)
            == "Failed with [REDACTED]"
        )

        # Pattern match for Bearer
        assert (
            redact_token("Authorization: Bearer my_random_secret_string")
            == "Authorization: Bearer [REDACTED]"
        )

        # Pattern match for ghp_
        assert (
            redact_token("Token is ghp_11223344556677889900aabbccdd")
            == "Token is [REDACTED_GH_TOKEN]"
        )

        # Pattern match for github_pat_
        assert (
            redact_token("github_pat_11223344556677889900112233")
            == "[REDACTED_GH_TOKEN]"
        )

    def test_github_client_error_redacts_token(self):
        secret = "ghp_do_not_leak_me_in_exceptions"
        err = GitHubClientError(f"Request failed with {secret}", token=secret)
        assert secret not in str(err)
        assert "[REDACTED]" in str(err)

    def test_logger_filter_redacts_tokens(self, caplog):
        caplog.set_level(logging.INFO)
        test_logger = logging.getLogger("test_redact")
        token_filter = TokenRedactingFilter(token="super_secret_key_12345")
        test_logger.addFilter(token_filter)

        test_logger.info("Connecting using token super_secret_key_12345 to host")
        assert "super_secret_key_12345" not in caplog.text
        assert "[REDACTED]" in caplog.text


# ============================================================================
# Module-level Convenience Functions Tests
# ============================================================================


class TestModuleLevelFunctions:
    """Test get_pr_metadata and get_pr_diff module helpers."""

    @patch("github_client.GitHubClient.get_pr_metadata")
    def test_get_pr_metadata_helper(self, mock_get_meta):
        mock_meta = MagicMock(spec=PRMetadata)
        mock_get_meta.return_value = mock_meta

        res = get_pr_metadata("owner", "repo", 12, token="dummy_token")
        assert res == mock_meta
        mock_get_meta.assert_called_once_with("owner", "repo", 12)

    @patch("github_client.GitHubClient.get_pr_diff")
    def test_get_pr_diff_helper(self, mock_get_diff):
        mock_get_diff.return_value = "diff --git a/x b/x\n"

        res = get_pr_diff("owner", "repo", 12, token="dummy_token")
        assert res == "diff --git a/x b/x\n"
        mock_get_diff.assert_called_once_with("owner", "repo", 12)

    @patch("github_client.GitHubClient.create_pull_request_review")
    def test_create_pull_request_review_helper(self, mock_create):
        mock_create.return_value = {"id": 123, "state": "COMMENTED"}
        res = create_pull_request_review(
            "owner",
            "repo",
            10,
            "1234567890abcdef1234567890abcdef12345678",
            "LGTM",
            event="COMMENT",
            token="dummy_token",
        )
        assert res == {"id": 123, "state": "COMMENTED"}
        mock_create.assert_called_once_with(
            owner="owner",
            repo="repo",
            pr_number=10,
            commit_sha="1234567890abcdef1234567890abcdef12345678",
            body="LGTM",
            event="COMMENT",
            comments=None,
        )

    @patch("github_client.GitHubClient.get_pr_reviews")
    def test_get_pr_reviews_helper(self, mock_get_reviews):
        mock_get_reviews.return_value = [{"id": 1}, {"id": 2}]
        res = get_pr_reviews("owner", "repo", 10, token="dummy_token", per_page=50, max_pages=3)
        assert res == [{"id": 1}, {"id": 2}]
        mock_get_reviews.assert_called_once_with(
            owner="owner",
            repo="repo",
            pr_number=10,
            per_page=50,
            max_pages=3,
        )


# ============================================================================
# Create PR Review (POST) Tests
# ============================================================================


class TestCreatePullRequestReview:
    """Test create_pull_request_review with mocked responses."""

    def test_create_review_success_no_comments(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": 999,
            "user": {"login": "bot-reviewer"},
            "body": "Executive Summary",
            "state": "COMMENTED",
            "html_url": "https://github.com/octocat/repo/pull/10#pullrequestreview-999",
            "commit_id": "1234567890abcdef1234567890abcdef12345678",
        }

        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(token="ghp_test_token_1234567890", session=mock_session)
        review = client.create_pull_request_review(
            owner="octocat",
            repo="repo",
            pr_number=10,
            commit_sha="1234567890abcdef1234567890abcdef12345678",
            body="Executive Summary",
            event="COMMENT",
        )

        assert review["id"] == 999
        assert review["state"] == "COMMENTED"

        mock_session.post.assert_called_once()
        args, kwargs = mock_session.post.call_args
        assert args[0] == "https://api.github.com/repos/octocat/repo/pulls/10/reviews"
        assert kwargs["headers"]["Authorization"] == "Bearer ghp_test_token_1234567890"
        assert kwargs["headers"]["Accept"] == "application/vnd.github+json"
        assert kwargs["json"] == {
            "commit_id": "1234567890abcdef1234567890abcdef12345678",
            "body": "Executive Summary",
            "event": "COMMENT",
        }

    def test_create_review_success_with_single_and_multiline_comments(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1001, "state": "CHANGES_REQUESTED"}

        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(token="ghp_test_token_1234567890", session=mock_session)
        comments = [
            {
                "path": "src/app.py",
                "line": 42,
                "body": "Fix this bug.",
                "side": "RIGHT",
            },
            {
                "path": "src/utils.py",
                "line": 80,
                "start_line": 75,
                "body": "Refactor multi-line block.",
                "side": "RIGHT",
                "start_side": "RIGHT",
            },
        ]

        review = client.create_pull_request_review(
            owner="octocat",
            repo="repo",
            pr_number=42,
            commit_sha="abcdef1234567890abcdef1234567890abcdef12",
            body="Review with inline comments",
            event="REQUEST_CHANGES",
            comments=comments,
        )

        assert review["id"] == 1001
        _, kwargs = mock_session.post.call_args
        payload = kwargs["json"]
        assert payload["event"] == "REQUEST_CHANGES"
        assert len(payload["comments"]) == 2
        assert payload["comments"][0]["path"] == "src/app.py"
        assert payload["comments"][0]["line"] == 42
        assert payload["comments"][0]["side"] == "RIGHT"
        assert payload["comments"][1]["start_line"] == 75
        assert payload["comments"][1]["line"] == 80
        assert payload["comments"][1]["start_side"] == "RIGHT"

    def test_create_review_verdicts(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 1}

        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(session=mock_session)

        # APPROVE
        client.create_pull_request_review(
            "owner", "repo", 1, "1234567", "Clean code", event="APPROVE"
        )
        assert mock_session.post.call_args[1]["json"]["event"] == "APPROVE"

        # COMMENT
        client.create_pull_request_review(
            "owner", "repo", 1, "1234567", "Suggestions", event="comment"
        )
        assert mock_session.post.call_args[1]["json"]["event"] == "COMMENT"

        # REQUEST_CHANGES
        client.create_pull_request_review(
            "owner", "repo", 1, "1234567", "Issues found", event="request_changes"
        )
        assert mock_session.post.call_args[1]["json"]["event"] == "REQUEST_CHANGES"

    def test_create_review_validation_errors(self):
        client = GitHubClient()

        # Invalid owner/repo
        with pytest.raises(ValueError, match="Repository owner"):
            client.create_pull_request_review("", "repo", 1, "1234567", "body")
        with pytest.raises(ValueError, match="Invalid repository owner"):
            client.create_pull_request_review("owner/slash", "repo", 1, "1234567", "body")
        with pytest.raises(ValueError, match="Invalid repository name"):
            client.create_pull_request_review("owner", "repo/slash", 1, "1234567", "body")
        with pytest.raises(ValueError, match="PR number"):
            client.create_pull_request_review("owner", "repo", 0, "1234567", "body")

        # Invalid commit SHA
        with pytest.raises(ValueError, match="Invalid commit SHA"):
            client.create_pull_request_review("owner", "repo", 1, "", "body")
        with pytest.raises(ValueError, match="Invalid commit SHA"):
            client.create_pull_request_review("owner", "repo", 1, "short", "body")  # < 7 chars
        with pytest.raises(ValueError, match="Invalid commit SHA"):
            client.create_pull_request_review("owner", "repo", 1, "not_hex_commit_sha!", "body")

        # Invalid body
        with pytest.raises(ValueError, match="Review body must be a non-empty string"):
            client.create_pull_request_review("owner", "repo", 1, "1234567", "   ")

        # Invalid event
        with pytest.raises(ValueError, match="Invalid review event"):
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body", event="MERGE")

        # Non-dict comment
        with pytest.raises(TypeError, match="must be a dictionary"):
            client.create_pull_request_review(
                "owner", "repo", 1, "1234567", "body", comments=["not a dict"]  # type: ignore[list-item]
            )

        # Comment missing path
        with pytest.raises(ValueError, match="missing or empty 'path'"):
            client.create_pull_request_review(
                "owner", "repo", 1, "1234567", "body", comments=[{"path": "", "body": "x"}]
            )

        # Comment path traversal
        with pytest.raises(ValueError, match="invalid path"):
            client.create_pull_request_review(
                "owner", "repo", 1, "1234567", "body", comments=[{"path": "../escape.py", "body": "x"}]
            )
        with pytest.raises(ValueError, match="invalid path"):
            client.create_pull_request_review(
                "owner", "repo", 1, "1234567", "body", comments=[{"path": "/abs.py", "body": "x"}]
            )

        # Comment missing body
        with pytest.raises(ValueError, match="missing or empty 'body'"):
            client.create_pull_request_review(
                "owner", "repo", 1, "1234567", "body", comments=[{"path": "a.py", "body": ""}]
            )

        # Invalid line numbers
        with pytest.raises(ValueError, match="'line' must be a positive integer"):
            client.create_pull_request_review(
                "owner", "repo", 1, "1234567", "body", comments=[{"path": "a.py", "body": "x", "line": 0}]
            )
        with pytest.raises(ValueError, match="'line' must be a positive integer"):
            client.create_pull_request_review(
                "owner", "repo", 1, "1234567", "body", comments=[{"path": "a.py", "body": "x", "line": -5}]
            )

        # Invalid side
        with pytest.raises(ValueError, match="'side' must be 'LEFT' or 'RIGHT'"):
            client.create_pull_request_review(
                "owner",
                "repo",
                1,
                "1234567",
                "body",
                comments=[{"path": "a.py", "body": "x", "line": 10, "side": "TOP"}],
            )

        # Invalid start_line > line
        with pytest.raises(ValueError, match="cannot be greater than 'line'"):
            client.create_pull_request_review(
                "owner",
                "repo",
                1,
                "1234567",
                "body",
                comments=[{"path": "a.py", "body": "x", "line": 10, "start_line": 20}],
            )

    def test_create_review_http_401_auth_error(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 401
        mock_response.text = "Bad credentials"
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(token="ghp_bad_secret_token", session=mock_session)
        with pytest.raises(GitHubAuthError, match="HTTP 401 Unauthorized"):
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

    def test_create_review_http_403_forbidden(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 403
        mock_response.headers = {"X-RateLimit-Remaining": "50"}
        mock_response.text = "Resource not accessible by integration"
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="HTTP 403"):
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

    def test_create_review_http_403_resource_not_accessible_pat(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 403
        mock_response.headers = {"X-RateLimit-Remaining": "50"}
        mock_response.json.return_value = {
            "message": "Resource not accessible by personal access token",
            "documentation_url": "https://docs.github.com/rest/pulls/reviews",
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(token="ghp_test_secret_token", session=mock_session)
        with pytest.raises(GitHubClientError) as exc_info:
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

        err = str(exc_info.value)
        assert "HTTP 403" in err
        assert "Resource not accessible by personal access token" in err
        assert "ghp_test_secret_token" not in err
        assert (
            exc_info.value.response_message
            == "Resource not accessible by personal access token"
        )

    def test_create_review_http_403_must_have_push_access(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 403
        mock_response.headers = {
            "X-RateLimit-Remaining": "50",
            "X-Accepted-OAuth-Scopes": "repo",
            "X-OAuth-Scopes": "read:user",
        }
        mock_response.json.return_value = {
            "message": "Must have push access to repository",
            "documentation_url": "https://docs.github.com/rest/pulls/reviews",
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(token="ghp_test_secret_token", session=mock_session)
        with pytest.raises(GitHubClientError) as exc_info:
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

        err = str(exc_info.value)
        assert "HTTP 403" in err
        assert "Must have push access to repository" in err
        assert "OAuth scopes (required: repo, token: read:user)" in err
        assert "ghp_test_secret_token" not in err
        assert exc_info.value.response_message == "Must have push access to repository"

    def test_create_review_http_403_sso_authorization_required(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 403
        mock_response.headers = {
            "X-RateLimit-Remaining": "50",
            "X-GitHub-SSO": "required; url=https://github.com/orgs/my-org/sso?authorization_request=abc123",
        }
        mock_response.json.return_value = {
            "message": "Resource protected by organization SAML enforcement.",
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(token="ghp_test_secret_token", session=mock_session)
        with pytest.raises(GitHubClientError) as exc_info:
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

        err = str(exc_info.value)
        assert "HTTP 403" in err
        assert "SSO authorization required" in err
        assert "https://github.com/orgs/my-org/sso?authorization_request=abc123" in err
        assert "ghp_test_secret_token" not in err
        assert exc_info.value.sso_hint is not None
        assert "SSO authorization required" in exc_info.value.sso_hint

    def test_create_review_http_403_rate_limit(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 403
        mock_response.headers = {
            "X-RateLimit-Remaining": "0",
            "Retry-After": "90",
            "X-RateLimit-Reset": "1740000000",
        }
        mock_response.json.return_value = {
            "message": "API rate limit exceeded for user ID 123",
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(token="ghp_test_secret_token", session=mock_session)
        with pytest.raises(GitHubRateLimitError) as exc_info:
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

        assert exc_info.value.retry_after == 90
        assert exc_info.value.reset_at == 1740000000
        assert "rate limit exceeded" in str(exc_info.value).lower()
        assert "ghp_test_secret_token" not in str(exc_info.value)

    def test_create_review_http_403_token_redaction(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 403
        mock_response.headers = {
            "X-RateLimit-Remaining": "50",
            "X-GitHub-SSO": "required; url=https://github.com/sso?token=ghp_leaked_secret_token",
        }
        mock_response.json.return_value = {
            "message": "Token ghp_leaked_secret_token lacks permissions.",
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(token="ghp_leaked_secret_token", session=mock_session)
        with pytest.raises(GitHubClientError) as exc_info:
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

        err = str(exc_info.value)
        assert "ghp_leaked_secret_token" not in err
        assert "[REDACTED]" in err
        if exc_info.value.response_message:
            assert "ghp_leaked_secret_token" not in exc_info.value.response_message
        if exc_info.value.sso_hint:
            assert "ghp_leaked_secret_token" not in exc_info.value.sso_hint

    def test_create_review_http_404_not_found(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubNotFoundError, match="HTTP 404"):
            client.create_pull_request_review("owner", "repo", 999, "1234567", "body")

    def test_create_review_http_422_validation_error(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 422
        mock_response.json.return_value = {
            "message": "Validation Failed",
            "errors": [
                {"field": "line", "code": "invalid", "message": "Line must be part of the diff"}
            ],
        }
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubValidationError) as exc_info:
            client.create_pull_request_review(
                "owner",
                "repo",
                1,
                "1234567",
                "body",
                comments=[{"path": "a.py", "body": "x", "line": 999}],
            )

        err = exc_info.value
        assert "HTTP 422" in str(err)
        assert len(err.errors) == 1
        assert err.errors[0]["field"] == "line"

    def test_create_review_http_429_rate_limit(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_response.text = "Too Many Requests"
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubRateLimitError) as exc_info:
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")
        assert exc_info.value.retry_after == 60

    def test_create_review_http_500_server_error(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="HTTP 500"):
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

    def test_create_review_timeout_raises_uncertain_write_error(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.side_effect = requests.exceptions.Timeout("Read timeout")

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubUncertainWriteError) as exc_info:
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

        assert issubclass(GitHubUncertainWriteError, GitHubNetworkError)
        assert "timed out" in str(exc_info.value)
        assert "Outcome is uncertain" in str(exc_info.value)

    def test_create_review_connection_error_raises_uncertain_write_error(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.side_effect = requests.exceptions.ConnectionError("Connection reset by peer")

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubUncertainWriteError) as exc_info:
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

        assert "connection failed" in str(exc_info.value)

    def test_create_review_request_exception(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.side_effect = requests.exceptions.RequestException("Generic failure")

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="request error"):
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

    def test_create_review_malformed_json_response(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 201
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="Failed to parse JSON response"):
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

    def test_create_review_unexpected_json_type(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = ["unexpected", "list"]
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="Unexpected response type"):
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")


# ============================================================================
# Get PR Reviews (Paginated GET) Tests
# ============================================================================


class TestGetPRReviews:
    """Test get_pr_reviews pagination and error handling."""

    def test_get_pr_reviews_single_page(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 1, "state": "COMMENTED", "commit_id": "1111111"},
            {"id": 2, "state": "APPROVED", "commit_id": "1111111"},
        ]
        mock_response.headers = {}
        mock_response.links = {}

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        reviews = client.get_pr_reviews("owner", "repo", 10, per_page=50)

        assert len(reviews) == 2
        assert reviews[0]["id"] == 1
        assert reviews[1]["id"] == 2
        mock_session.get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/pulls/10/reviews?per_page=50",
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "code-review-agent/1.0",
            },
            timeout=30.0,
            allow_redirects=False,
        )

    def test_get_pr_reviews_pagination_two_pages(self):
        # Page 1
        resp_p1 = MagicMock(spec=requests.Response)
        resp_p1.status_code = 200
        resp_p1.json.return_value = [{"id": 10}, {"id": 11}]
        resp_p1.headers = {
            "Link": '<https://api.github.com/repos/owner/repo/pulls/5/reviews?per_page=2&page=2>; rel="next"'
        }
        resp_p1.links = {
            "next": {
                "url": "https://api.github.com/repos/owner/repo/pulls/5/reviews?per_page=2&page=2"
            }
        }

        # Page 2
        resp_p2 = MagicMock(spec=requests.Response)
        resp_p2.status_code = 200
        resp_p2.json.return_value = [{"id": 12}]
        resp_p2.headers = {}
        resp_p2.links = {}

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.side_effect = [resp_p1, resp_p2]

        client = GitHubClient(session=mock_session)
        reviews = client.get_pr_reviews("owner", "repo", 5, per_page=2)

        assert len(reviews) == 3
        assert [r["id"] for r in reviews] == [10, 11, 12]
        assert mock_session.get.call_count == 2

    def test_get_pr_reviews_pagination_fallback_header_regex(self):
        # resp.links is empty dict, but Link header present
        resp_p1 = MagicMock(spec=requests.Response)
        resp_p1.status_code = 200
        resp_p1.json.return_value = [{"id": 1}]
        resp_p1.headers = {
            "Link": '<https://api.github.com/repos/owner/repo/pulls/1/reviews?per_page=1&page=2>; rel="next"'
        }
        resp_p1.links = {}

        resp_p2 = MagicMock(spec=requests.Response)
        resp_p2.status_code = 200
        resp_p2.json.return_value = [{"id": 2}]
        resp_p2.headers = {}
        resp_p2.links = {}

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.side_effect = [resp_p1, resp_p2]

        client = GitHubClient(session=mock_session)
        reviews = client.get_pr_reviews("owner", "repo", 1, per_page=1)

        assert len(reviews) == 2
        assert mock_session.get.call_count == 2

    def test_get_pr_reviews_respects_max_pages(self):
        # Continuous rel="next" links
        def create_page(page_num):
            r = MagicMock(spec=requests.Response)
            r.status_code = 200
            r.json.return_value = [{"id": page_num}]
            next_url = f"https://api.github.com/repos/owner/repo/pulls/1/reviews?page={page_num + 1}"
            r.headers = {"Link": f'<{next_url}>; rel="next"'}
            r.links = {"next": {"url": next_url}}
            return r

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.side_effect = [create_page(1), create_page(2), create_page(3)]

        client = GitHubClient(session=mock_session)
        reviews = client.get_pr_reviews("owner", "repo", 1, max_pages=2)

        assert len(reviews) == 2
        assert mock_session.get.call_count == 2

    def test_get_pr_reviews_rejects_untrusted_pagination_url(self):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.json.return_value = [{"id": 1}]
        resp.headers = {"Link": '<https://attacker.evil.com/leak>; rel="next"'}
        resp.links = {"next": {"url": "https://attacker.evil.com/leak"}}

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = resp

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="Untrusted pagination URL"):
            client.get_pr_reviews("owner", "repo", 1)

    def test_get_pr_reviews_http_errors(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_resp_401 = MagicMock(status_code=401)
        mock_resp_404 = MagicMock(status_code=404)

        mock_session.get.side_effect = [mock_resp_401, mock_resp_404]
        client = GitHubClient(session=mock_session)

        with pytest.raises(GitHubAuthError):
            client.get_pr_reviews("owner", "repo", 1)
        with pytest.raises(GitHubNotFoundError):
            client.get_pr_reviews("owner", "repo", 1)

    def test_get_pr_reviews_network_errors(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.side_effect = requests.exceptions.Timeout("Timeout")
        client = GitHubClient(session=mock_session)

        with pytest.raises(GitHubNetworkError, match="timed out"):
            client.get_pr_reviews("owner", "repo", 1)

        mock_session.get.side_effect = requests.exceptions.ConnectionError("Refused")
        with pytest.raises(GitHubNetworkError, match="connection failed"):
            client.get_pr_reviews("owner", "repo", 1)

    def test_get_pr_reviews_malformed_json_response(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Bad JSON")

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="Failed to parse JSON response"):
            client.get_pr_reviews("owner", "repo", 1)

    def test_get_pr_reviews_unexpected_json_type(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "not a list"}

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_response

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="Unexpected response type"):
            client.get_pr_reviews("owner", "repo", 1)


# ============================================================================
# Token Redaction in Stage 1 Methods Tests
# ============================================================================


class TestTokenRedactionInReviewMethods:
    """Test token scrub guarantees on write/review operations."""

    def test_create_review_token_redacted_in_auth_error(self):
        secret = "ghp_stage1_super_secret_pat_987654321"
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 401
        mock_response.text = f"Bad credentials for bearer {secret}"

        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(token=secret, session=mock_session)
        with pytest.raises(GitHubAuthError) as exc_info:
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

        err_str = str(exc_info.value)
        assert secret not in err_str
        assert "[REDACTED]" in err_str

    def test_create_review_token_redacted_in_422_error(self):
        secret = "ghp_stage1_super_secret_pat_987654321"
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 422
        mock_response.json.return_value = {
            "message": f"Token {secret} cannot comment on this line",
            "errors": [],
        }

        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_response

        client = GitHubClient(token=secret, session=mock_session)
        with pytest.raises(GitHubValidationError) as exc_info:
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

        err_str = str(exc_info.value)
        assert secret not in err_str
        assert "[REDACTED]" in err_str

    def test_create_review_token_redacted_in_timeout_error(self):
        secret = "ghp_stage1_super_secret_pat_987654321"
        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.side_effect = requests.exceptions.Timeout(
            f"Timeout while communicating with auth header Bearer {secret}"
        )

        client = GitHubClient(token=secret, session=mock_session)
        with pytest.raises(GitHubUncertainWriteError) as exc_info:
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

        err_str = str(exc_info.value)
        assert secret not in err_str

    def test_get_pr_reviews_token_redacted_in_errors(self):
        secret = "ghp_stage1_super_secret_pat_987654321"
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.side_effect = requests.exceptions.ConnectionError(
            f"Failed to connect using token {secret}"
        )

        client = GitHubClient(token=secret, session=mock_session)
        with pytest.raises(GitHubNetworkError) as exc_info:
            client.get_pr_reviews("owner", "repo", 1)

        err_str = str(exc_info.value)
        assert secret not in err_str


# ============================================================================
# Stage 2: Security Checks & Hunk Span Tests
# ============================================================================


class TestStage1SecurityFixesAndHunkSpan:
    """Tests for pagination path validation, redirect blocking, and hunk span resolution."""

    def test_pagination_url_path_tampering_rejected(self):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.json.return_value = [{"id": 1}]
        # Same host api.github.com, but different repo path
        tampered_url = "https://api.github.com/repos/attacker/malicious-repo/pulls/1/reviews"
        resp.headers = {"Link": f'<{tampered_url}>; rel="next"'}
        resp.links = {"next": {"url": tampered_url}}

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = resp

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="Untrusted pagination URL"):
            client.get_pr_reviews("owner", "repo", 1)

    def test_create_review_rejects_3xx_redirect(self):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 302
        resp.headers = {"Location": "https://api.github.com/redirected"}
        resp.text = "Found"

        mock_session = MagicMock(spec=requests.Session)
        mock_session.post.return_value = resp

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="Unexpected redirect response"):
            client.create_pull_request_review("owner", "repo", 1, "1234567", "body")

        # Verify allow_redirects=False was passed
        _, kwargs = mock_session.post.call_args
        assert kwargs["allow_redirects"] is False

    def test_get_pr_reviews_rejects_3xx_redirect(self):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 301
        resp.headers = {"Location": "https://api.github.com/redirected"}
        resp.text = "Moved Permanently"

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = resp

        client = GitHubClient(session=mock_session)
        with pytest.raises(GitHubClientError, match="Unexpected redirect response"):
            client.get_pr_reviews("owner", "repo", 1)

        _, kwargs = mock_session.get.call_args
        assert kwargs["allow_redirects"] is False

    def test_diff_file_find_hunk_for_span_success(self):
        diff_text = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -10,6 +10,8 @@
 context line 10
+added line 11
+added line 12
 context line 13
 context line 14
"""
        client = GitHubClient()
        files = client.parse_diff_hunks(diff_text)
        assert len(files) == 1
        df = files[0]

        # Valid span within single hunk (10 to 12)
        hunk = df.find_hunk_for_span(10, 12)
        assert hunk is not None
        assert hunk.new_start == 10

        # Exact boundary span
        hunk_exact = df.find_hunk_for_span(10, 14)
        assert hunk_exact is not None

    def test_diff_file_find_hunk_for_span_cross_hunk_returns_none(self):
        diff_text = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -10,3 +10,3 @@
 line 10
 line 11
 line 12
@@ -50,3 +50,3 @@
 line 50
 line 51
 line 52
"""
        client = GitHubClient()
        files = client.parse_diff_hunks(diff_text)
        df = files[0]
        assert len(df.hunks) == 2

        # Span crossing hunk 1 (10..12) and hunk 2 (50..52)
        assert df.find_hunk_for_span(11, 51) is None
        assert df.find_hunk_for_span(12, 50) is None

    def test_diff_file_find_hunk_for_span_out_of_bounds_and_inverted(self):
        diff_text = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -10,3 +10,3 @@
 line 10
 line 11
 line 12
"""
        client = GitHubClient()
        files = client.parse_diff_hunks(diff_text)
        df = files[0]

        # Out of bounds
        assert df.find_hunk_for_span(5, 10) is None
        assert df.find_hunk_for_span(10, 20) is None
        assert df.find_hunk_for_span(100, 105) is None

        # Inverted coordinates
        assert df.find_hunk_for_span(12, 10) is None

        # Zero or negative
        assert df.find_hunk_for_span(0, 10) is None
        assert df.find_hunk_for_span(-5, -1) is None



# ============================================================================
# Token Precedence, Resolution, and Instance Redaction Tests
# ============================================================================


class TestTokenPrecedenceAndResolution:
    """Tests for token source precedence, fallback, and file-based token input."""

    def test_explicit_token_has_highest_precedence(self, monkeypatch, tmp_path):
        token_file = tmp_path / "token.txt"
        token_file.write_text("file_token_4444", encoding="utf-8")
        monkeypatch.setenv("GITHUB_TOKEN", "env_token_2222")
        monkeypatch.setenv("GH_TOKEN", "gh_token_3333")
        monkeypatch.setenv("GITHUB_TOKEN_FILE", str(token_file))

        resolved, source = resolve_github_token_with_source("explicit_token_1111")
        assert resolved == "explicit_token_1111"
        assert source == "explicit"

    def test_github_token_precedes_gh_token_and_file(self, monkeypatch, tmp_path):
        token_file = tmp_path / "token.txt"
        token_file.write_text("file_token_4444", encoding="utf-8")
        monkeypatch.setenv("GITHUB_TOKEN", "env_token_2222")
        monkeypatch.setenv("GH_TOKEN", "gh_token_3333")
        monkeypatch.setenv("GITHUB_TOKEN_FILE", str(token_file))

        resolved, source = resolve_github_token_with_source(None)
        assert resolved == "env_token_2222"
        assert source == "GITHUB_TOKEN"

    def test_gh_token_precedes_file(self, monkeypatch, tmp_path):
        token_file = tmp_path / "token.txt"
        token_file.write_text("file_token_4444", encoding="utf-8")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "gh_token_3333")
        monkeypatch.setenv("GITHUB_TOKEN_FILE", str(token_file))

        resolved, source = resolve_github_token_with_source(None)
        assert resolved == "gh_token_3333"
        assert source == "GH_TOKEN"

    def test_github_token_file_resolved_when_env_vars_absent(self, monkeypatch, tmp_path):
        token_file = tmp_path / "token.txt"
        token_file.write_text("file_token_4444\n", encoding="utf-8")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN_FILE", str(token_file))

        resolved, source = resolve_github_token_with_source(None)
        assert resolved == "file_token_4444"
        assert source == "GITHUB_TOKEN_FILE"

    def test_github_token_file_missing_raises_auth_error(self, monkeypatch, tmp_path):
        missing_path = tmp_path / "nonexistent_token.txt"
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN_FILE", str(missing_path))

        with pytest.raises(GitHubAuthError, match="Configured GITHUB_TOKEN_FILE does not exist"):
            resolve_github_token_with_source(None)

    def test_github_token_file_empty_raises_auth_error(self, monkeypatch, tmp_path):
        empty_file = tmp_path / "empty_token.txt"
        empty_file.write_text("   \n", encoding="utf-8")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN_FILE", str(empty_file))

        with pytest.raises(GitHubAuthError, match="Configured GITHUB_TOKEN_FILE .* is empty"):
            resolve_github_token_with_source(None)

    def test_missing_credentials_raises_when_required(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN_FILE", raising=False)

        with pytest.raises(GitHubAuthError, match="Missing GitHub token"):
            GitHubClient(require_token=True)


class TestInstanceTokenRedaction:
    """Tests verifying that tokens passed to GitHubClient instances are globally scrubbed."""

    def test_client_instance_token_is_automatically_registered_and_redacted(self):
        clear_sensitive_tokens()
        secret = "custom_instance_pat_9876543210123"
        client = GitHubClient(token=secret)
        assert client._token == secret

        # Calling redact_token without passing token explicitly must scrub the secret
        raw_msg = f"API error occurred while using token {secret} to connect"
        scrubbed = redact_token(raw_msg)
        assert secret not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_logger_filter_redacts_instance_token_without_token_arg(self, caplog):
        clear_sensitive_tokens()
        secret = "instance_key_for_logger_test_5555"
        GitHubClient(token=secret)

        test_logger = logging.getLogger("test_instance_redact")
        test_logger.addFilter(TokenRedactingFilter())

        with caplog.at_level(logging.INFO):
            test_logger.info("Connecting using %s to API endpoint", secret)
        assert secret not in caplog.text
        assert "[REDACTED]" in caplog.text

    def test_github_client_error_redacts_registered_token_without_param(self):
        clear_sensitive_tokens()
        secret = "secret_inside_exception_msg_7777"
        GitHubClient(token=secret)

        err = GitHubClientError(f"HTTP 500 server error with {secret}")
        assert secret not in str(err)
        assert "[REDACTED]" in str(err)

    def test_multiple_client_instances_register_and_redact_all_tokens(self):
        clear_sensitive_tokens()
        secret1 = "client_one_secret_token_1111"
        secret2 = "client_two_secret_token_2222"
        GitHubClient(token=secret1)
        GitHubClient(token=secret2)

        combined_msg = f"Credentials {secret1} and {secret2} were active"
        scrubbed = redact_token(combined_msg)
        assert secret1 not in scrubbed
        assert secret2 not in scrubbed
        assert scrubbed.count("[REDACTED]") == 2

    def test_url_query_parameters_with_token_are_redacted(self):
        url_with_param = "https://api.github.com/repos/owner/repo/pulls?access_token=secret_in_url_8888&page=2"
        scrubbed = redact_token(url_with_param)
        assert "secret_in_url_8888" not in scrubbed
        assert "?access_token=[REDACTED]&page=2" in scrubbed

    def test_resolve_github_token_convenience_helper(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "helper_test_token_9999")
        tok = resolve_github_token()
        assert tok == "helper_test_token_9999"

    def test_register_sensitive_token_manually(self):
        clear_sensitive_tokens()
        register_sensitive_token("manual_secret_token_8888")
        assert "manual_secret_token_8888" not in redact_token("leaked manual_secret_token_8888 here")
        assert "[REDACTED]" in redact_token("leaked manual_secret_token_8888 here")
