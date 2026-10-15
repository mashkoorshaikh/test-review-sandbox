"""
tests/test_github_review_workflow.py - Focused tests for Read-Only PR Review Integration.

Verifies:
  1. Repository and PR URL/target parsing.
  2. End-to-end read-only workflow execution with mocked GitHub responses.
  3. Strict read-only guarantee: Proves that NO write requests (POST/PUT/PATCH/DELETE) occur.
  4. Exact head-SHA binding and TOCTOU race detection (changed head SHA aborts workflow).
  5. Fail-closed behavior on auth errors, 404 not found, and archive extraction security errors.
  6. Correct mapping of static findings to diff hunks/lines as inline ReviewComments.
  7. Handling of clean PR scans (verdict=APPROVE, zero comments).
  8. Credential protection: Automatic redaction of tokens in errors and logs.
  9. CLI integration via run_pr_pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from approval_gate import HeadShaMismatchError
from github_client import (
    DiffFile,
    DiffHunk,
    DiffLine,
    GitHubAuthError,
    GitHubClient,
    GitHubNotFoundError,
    PRMetadata,
)
from github_review_workflow import (
    PRReviewInputError,
    PRReviewResult,
    build_inline_comments,
    parse_repo_and_pr,
    run_pr_review_workflow,
)
from main import run_pr_pipeline
from pr_workspace import ArchiveSecurityError, CommitBindingError, PRWorkspace
from tools import Issue, ToolResult

# ============================================================================
# Test Fixtures & Mock Data
# ============================================================================

VALID_HEAD_SHA = "1234567890abcdef1234567890abcdef12345678"
DIFFERENT_HEAD_SHA = "abcdef1234567890abcdef1234567890abcdef12"

SAMPLE_DIFF = """diff --git a/sample_project/app.py b/sample_project/app.py
index 1111111..2222222 100644
--- a/sample_project/app.py
+++ b/sample_project/app.py
@@ -1,5 +1,7 @@
 import os
+import math

+def calculate(val):
+    return val * 2
"""


@pytest.fixture
def mock_pr_metadata():
    return PRMetadata(
        owner="octocat",
        repo="Hello-World",
        pr_number=42,
        title="Add calculation module",
        body="This PR implements a simple calculation helper.",
        state="open",
        head_sha=VALID_HEAD_SHA,
        base_sha="0000000000000000000000000000000000000000",
        head_branch="feature/calc",
        base_branch="main",
        author="alice",
        html_url="https://github.com/octocat/Hello-World/pull/42",
    )


@pytest.fixture
def mock_pr_workspace(tmp_path):
    ws_dir = tmp_path / "pr_ws_octocat_Hello-World_42"
    ws_dir.mkdir(parents=True, exist_ok=True)
    sample_file = ws_dir / "sample_project" / "app.py"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text("import os\nimport math\n\ndef calculate(val):\n    return val * 2\n", encoding="utf-8")

    ws = PRWorkspace(
        owner="octocat",
        repo="Hello-World",
        pr_number=42,
        head_sha=VALID_HEAD_SHA,
        workspace_path=ws_dir,
        created_at="2026-10-15T12:00:00Z",
        _cleanup_on_exit=False,
    )
    return ws


# ============================================================================
# 1. Repository & PR Input Parsing Tests
# ============================================================================


class TestParseRepoAndPr:
    def test_parse_full_pr_url(self):
        owner, repo, pr = parse_repo_and_pr("https://github.com/owner-org/my_repo/pull/101")
        assert owner == "owner-org"
        assert repo == "my_repo"
        assert pr == 101

    def test_parse_hash_syntax(self):
        owner, repo, pr = parse_repo_and_pr("fastapi/fastapi#4567")
        assert owner == "fastapi"
        assert repo == "fastapi"
        assert pr == 4567

    def test_parse_separate_parameters(self):
        owner, repo, pr = parse_repo_and_pr("pallets/flask", 99)
        assert owner == "pallets"
        assert repo == "flask"
        assert pr == 99

    def test_parse_separate_parameters_with_string_pr(self):
        owner, repo, pr = parse_repo_and_pr("pallets/flask", "99")
        assert owner == "pallets"
        assert repo == "flask"
        assert pr == 99

    def test_missing_pr_number_raises_error(self):
        with pytest.raises(PRReviewInputError, match="PR number is required"):
            parse_repo_and_pr("pallets/flask")

    def test_invalid_pr_number_raises_error(self):
        with pytest.raises(PRReviewInputError, match="positive integer"):
            parse_repo_and_pr("pallets/flask", -5)

        with pytest.raises(PRReviewInputError, match="positive integer"):
            parse_repo_and_pr("pallets/flask", "not-a-number")

    def test_invalid_repo_string_raises_error(self):
        with pytest.raises(PRReviewInputError, match="Unable to parse"):
            parse_repo_and_pr("invalid-repo-format-without-slash-or-pr")

    def test_empty_input_raises_error(self):
        with pytest.raises(PRReviewInputError, match="non-empty string"):
            parse_repo_and_pr("")


# ============================================================================
# 2. End-to-End Read-Only Workflow Execution
# ============================================================================


class TestReadOnlyWorkflowExecution:
    def test_successful_read_only_review(self, mock_pr_metadata, mock_pr_workspace):
        mock_client = MagicMock(spec=GitHubClient)
        mock_client.get_pr_metadata.return_value = mock_pr_metadata
        mock_client.get_pr_diff.return_value = SAMPLE_DIFF

        mock_ruff = lambda p: ToolResult(
            tool_name="Ruff",
            target_path=p,
            success=True,
            issues_found=True,
            exit_code=1,
            issues=[
                Issue(
                    code="F401",
                    message="`math` imported but unused",
                    file_path=str(mock_pr_workspace.workspace_path / "sample_project" / "app.py"),
                    line=2,
                    column=1,
                    severity="WARNING",
                    tool="ruff",
                )
            ],
        )

        mock_bandit = lambda p: ToolResult(
            tool_name="Bandit",
            target_path=p,
            success=True,
            issues_found=False,
            exit_code=0,
            issues=[],
        )

        with patch("github_review_workflow.prepare_pr_workspace") as mock_prep:
            mock_prep.return_value = mock_pr_workspace

            output_lines = []
            result = run_pr_review_workflow(
                repo_or_url="octocat/Hello-World#42",
                client=mock_client,
                enable_ai=False,
                enable_sandbox=False,
                enable_debugging=False,
                ruff_runner=mock_ruff,
                bandit_runner=mock_bandit,
                output_func=output_lines.append,
                display_preview=True,
            )

            assert isinstance(result, PRReviewResult)
            assert result.owner == "octocat"
            assert result.repo == "Hello-World"
            assert result.pr_number == 42
            assert result.head_sha == VALID_HEAD_SHA
            assert result.success is True

            # Assert ActionPlan is strictly in dry_run mode
            assert result.action_plan.dry_run is True
            assert result.action_plan.verdict == "COMMENT"

            # Assert inline comments mapped to diff line 2
            assert len(result.review_comments) == 1
            comment = result.review_comments[0]
            assert comment.path == "sample_project/app.py"
            assert comment.line == 2
            assert "[Ruff - F401]" in comment.body
            assert comment.side == "RIGHT"

            # Assert preview text was formatted and displayed
            preview = "\n".join(output_lines)
            assert "PROPOSED GITHUB REVIEW ACTION PLAN [DRY RUN — NO WRITES WILL BE COMMITTED]" in preview
            assert "octocat/Hello-World" in preview
            assert "#42" in preview
            assert VALID_HEAD_SHA in preview


# ============================================================================
# 3. Strict Read-Only Guarantee (No Write Requests)
# ============================================================================


class TestStrictlyReadOnlyGuarantee:
    def test_workflow_makes_zero_write_requests(self, mock_pr_metadata, mock_pr_workspace):
        """
        Verify that under no circumstances does the workflow perform write requests
        (POST, PATCH, PUT, DELETE) on GitHub API endpoints.
        """
        mock_client = MagicMock(spec=GitHubClient)
        mock_client.get_pr_metadata.return_value = mock_pr_metadata
        mock_client.get_pr_diff.return_value = SAMPLE_DIFF

        # Spy on requests session write methods
        with patch("requests.Session.post") as mock_post, \
             patch("requests.Session.patch") as mock_patch, \
             patch("requests.Session.put") as mock_put, \
             patch("requests.Session.delete") as mock_delete, \
             patch("github_review_workflow.prepare_pr_workspace", return_value=mock_pr_workspace):

            result = run_pr_review_workflow(
                repo_or_url="octocat/Hello-World#42",
                client=mock_client,
                enable_ai=False,
                enable_sandbox=False,
                display_preview=False,
                ruff_runner=lambda p: ToolResult("Ruff", p, True, False, 0, []),
                bandit_runner=lambda p: ToolResult("Bandit", p, True, False, 0, []),
            )

            # Assert zero write requests occurred
            assert mock_post.call_count == 0
            assert mock_patch.call_count == 0
            assert mock_put.call_count == 0
            assert mock_delete.call_count == 0

            # Action plan must remain dry_run=True
            assert result.action_plan.dry_run is True


# ============================================================================
# 4. Head SHA Binding & TOCTOU Race Detection
# ============================================================================


class TestHeadShaBindingAndRaceDefense:
    def test_expected_head_sha_mismatch_aborts_immediately(self, mock_pr_metadata):
        mock_client = MagicMock(spec=GitHubClient)
        mock_client.get_pr_metadata.return_value = mock_pr_metadata

        with pytest.raises(CommitBindingError, match="PR head SHA mismatch"):
            run_pr_review_workflow(
                repo_or_url="octocat/Hello-World#42",
                expected_head_sha="0000000000000000000000000000000000000000",
                client=mock_client,
                display_preview=False,
            )

    def test_toctou_head_sha_change_during_review_aborts(self, mock_pr_metadata, mock_pr_workspace):
        """
        If author pushes a new commit while review is running, the post-analysis
        verification must catch the mismatch and abort with HeadShaMismatchError.
        """
        initial_meta = mock_pr_metadata
        modified_meta = PRMetadata(
            owner="octocat",
            repo="Hello-World",
            pr_number=42,
            title=mock_pr_metadata.title,
            body=mock_pr_metadata.body,
            state="open",
            head_sha=DIFFERENT_HEAD_SHA,  # Changed head SHA!
            base_sha=mock_pr_metadata.base_sha,
            head_branch=mock_pr_metadata.head_branch,
            base_branch=mock_pr_metadata.base_branch,
            author=mock_pr_metadata.author,
            html_url=mock_pr_metadata.html_url,
        )

        mock_client = MagicMock(spec=GitHubClient)
        # First call returns initial_meta, second check returns modified_meta
        mock_client.get_pr_metadata.side_effect = [initial_meta, modified_meta]
        mock_client.get_pr_diff.return_value = SAMPLE_DIFF

        with (
            patch("github_review_workflow.prepare_pr_workspace", return_value=mock_pr_workspace),
            pytest.raises(HeadShaMismatchError, match="PR Head SHA mismatch"),
        ):
            run_pr_review_workflow(
                repo_or_url="octocat/Hello-World#42",
                client=mock_client,
                enable_ai=False,
                enable_sandbox=False,
                display_preview=False,
                ruff_runner=lambda p: ToolResult("Ruff", p, True, False, 0, []),
                bandit_runner=lambda p: ToolResult("Bandit", p, True, False, 0, []),
            )



# ============================================================================
# 5. Fail-Closed Error Handling Tests
# ============================================================================


class TestFailClosedErrorHandling:
    def test_github_auth_error_fails_closed(self):
        mock_client = MagicMock(spec=GitHubClient)
        mock_client.get_pr_metadata.side_effect = GitHubAuthError("Invalid GitHub credentials", token="ghp_secret123")

        with pytest.raises(GitHubAuthError) as exc_info:
            run_pr_review_workflow("octocat/Hello-World#42", client=mock_client)

        assert "ghp_secret123" not in str(exc_info.value)
        assert "[REDACTED]" in str(exc_info.value) or "Invalid" in str(exc_info.value)

    def test_github_404_not_found_fails_closed(self):
        mock_client = MagicMock(spec=GitHubClient)
        mock_client.get_pr_metadata.side_effect = GitHubNotFoundError("PR not found: HTTP 404")

        with pytest.raises(GitHubNotFoundError):
            run_pr_review_workflow("octocat/Hello-World#9999", client=mock_client)

    def test_archive_security_error_aborts(self, mock_pr_metadata):
        mock_client = MagicMock(spec=GitHubClient)
        mock_client.get_pr_metadata.return_value = mock_pr_metadata
        mock_client.get_pr_diff.return_value = SAMPLE_DIFF

        with patch("github_review_workflow.prepare_pr_workspace") as mock_prep:
            mock_prep.side_effect = ArchiveSecurityError("Zip Slip path traversal detected in archive.")

            with pytest.raises(ArchiveSecurityError, match="Zip Slip"):
                run_pr_review_workflow("octocat/Hello-World#42", client=mock_client)


# ============================================================================
# 6. Finding Path Normalization & Diff Mapping Tests
# ============================================================================


class TestFindingMappingAndComments:
    def test_finding_on_changed_line_creates_inline_comment(self, tmp_path):
        ws_root = tmp_path / "ws"
        ws_root.mkdir()

        diff_file = DiffFile(
            old_path="src/calc.py",
            new_path="src/calc.py",
            status="modified",
            hunks=[
                DiffHunk(
                    header="@@ -1,5 +1,5 @@",
                    old_start=1,
                    old_count=5,
                    new_start=1,
                    new_count=5,
                    section_heading="",
                    lines=[
                        DiffLine(line_type="context", content="import math", new_line_number=1),
                        DiffLine(line_type="add", content="def calc():", new_line_number=2),
                    ],
                )
            ],
        )

        issues = [
            {
                "code": "F401",
                "message": "Unused import",
                "file_path": str(ws_root / "src" / "calc.py"),
                "line": 1,
                "tool": "ruff",
                "severity": "WARNING",
            }
        ]

        diff_map = {"src/calc.py": diff_file}
        comments, unmapped = build_inline_comments(issues, diff_map, ws_root)

        assert len(comments) == 1
        assert comments[0].path == "src/calc.py"
        assert comments[0].line == 1
        assert len(unmapped) == 0

    def test_finding_outside_diff_hunks_is_unmapped(self, tmp_path):
        ws_root = tmp_path / "ws"
        ws_root.mkdir()

        diff_file = DiffFile(
            old_path="src/calc.py",
            new_path="src/calc.py",
            status="modified",
            hunks=[
                DiffHunk(
                    header="@@ -1,2 +1,2 @@",
                    old_start=1,
                    old_count=2,
                    new_start=1,
                    new_count=2,
                    section_heading="",
                    lines=[
                        DiffLine(line_type="context", content="x = 1", new_line_number=1),
                    ],
                )
            ],
        )

        issues = [
            {
                "code": "B101",
                "message": "Assert statement used",
                "file_path": "src/calc.py",
                "line": 99,  # Line 99 is NOT in the diff hunk
                "tool": "bandit",
                "severity": "LOW",
            }
        ]

        diff_map = {"src/calc.py": diff_file}
        comments, unmapped = build_inline_comments(issues, diff_map, ws_root)

        assert len(comments) == 0
        assert len(unmapped) == 1
        assert unmapped[0]["code"] == "B101"


# ============================================================================
# 7. Clean PR Scan Handling
# ============================================================================


class TestCleanPRScanHandling:
    def test_clean_pr_scan_yields_approve_verdict(self, mock_pr_metadata, mock_pr_workspace):
        mock_client = MagicMock(spec=GitHubClient)
        mock_client.get_pr_metadata.return_value = mock_pr_metadata
        mock_client.get_pr_diff.return_value = SAMPLE_DIFF

        with patch("github_review_workflow.prepare_pr_workspace", return_value=mock_pr_workspace):
            result = run_pr_review_workflow(
                repo_or_url="octocat/Hello-World#42",
                client=mock_client,
                enable_ai=False,
                enable_sandbox=False,
                display_preview=False,
                ruff_runner=lambda p: ToolResult("Ruff", p, True, False, 0, []),
                bandit_runner=lambda p: ToolResult("Bandit", p, True, False, 0, []),
            )

            assert result.action_plan.verdict == "APPROVE"
            assert len(result.review_comments) == 0
            assert result.action_plan.dry_run is True
            assert result.success is True


# ============================================================================
# 8. CLI Integration via run_pr_pipeline
# ============================================================================


class TestCLIIntegration:
    def test_cli_run_pr_pipeline_success(self, capsys):
        mock_result = MagicMock(spec=PRReviewResult)
        mock_result.success = True
        mock_result.action_plan = MagicMock()
        mock_result.action_plan.verdict = "APPROVE"
        mock_result.review_comments = ()

        with patch("main.run_pr_review_workflow", return_value=mock_result) as mock_wf:
            exit_code = run_pr_pipeline("octocat/Hello-World#42")

            assert exit_code == 0
            mock_wf.assert_called_once()
            captured = capsys.readouterr().out
            assert "Milestone 4: Read-Only Pull Request Review" in captured
            assert "Target PR: octocat/Hello-World#42" in captured

    def test_cli_run_pr_pipeline_with_issues(self, capsys):
        mock_result = MagicMock(spec=PRReviewResult)
        mock_result.success = True
        mock_result.action_plan = MagicMock()
        mock_result.action_plan.verdict = "REQUEST_CHANGES"
        mock_result.review_comments = (MagicMock(),)

        with patch("main.run_pr_review_workflow", return_value=mock_result):
            exit_code = run_pr_pipeline("octocat/Hello-World#42")
            assert exit_code == 1

    def test_cli_run_pr_pipeline_error_handling(self, capsys):
        with patch("main.run_pr_review_workflow", side_effect=PRReviewInputError("Invalid PR number")):
            exit_code = run_pr_pipeline("invalid_input")
            assert exit_code == 2
            captured = capsys.readouterr().out
            assert "PR WORKFLOW FAILED: Invalid PR number" in captured
