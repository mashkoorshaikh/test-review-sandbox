"""
tests/test_github_client.py - Unit and integration tests for GitHub REST client and Diff Parser.

All HTTP requests are mocked so tests require no network access or credentials.
"""

from __future__ import annotations

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
    PRMetadata,
    TokenRedactingFilter,
    get_pr_diff,
    get_pr_metadata,
    parse_diff_hunks,
    redact_token,
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
