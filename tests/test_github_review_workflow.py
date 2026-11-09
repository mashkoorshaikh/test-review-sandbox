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

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from approval_gate import (
    ActionPlan,
    ApprovalExpiredError,
    ApprovalRejectedError,
    ApprovalResult,
    DryRunBlockedError,
    HeadShaMismatchError,
    PlanFingerprintMismatchError,
    ReviewComment,
)
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
    CommentAnchoringError,
    PrePublicationRevalidationResult,
    PrePublicationValidationError,
    PRReviewInputError,
    PRReviewResult,
    build_inline_comments,
    determine_review_verdict,
    generate_review_summary,
    has_fatal_analysis_error,
    is_high_confidence_critical_issue,
    parse_repo_and_pr,
    revalidate_review_for_publication,
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
    @pytest.fixture
    def sample_hunk(self):
        return DiffHunk(
            header="@@ -1,5 +1,5 @@",
            old_start=1,
            old_count=5,
            new_start=1,
            new_count=5,
            section_heading="def calc():",
            lines=[
                DiffLine(line_type="context", content="import math", old_line_number=1, new_line_number=1),
                DiffLine(line_type="add", content="def calc(val):", old_line_number=None, new_line_number=2),
                DiffLine(line_type="add", content="    res = val * 2", old_line_number=None, new_line_number=3),
                DiffLine(line_type="add", content="    return res", old_line_number=None, new_line_number=4),
                DiffLine(line_type="context", content="# end of calc", old_line_number=2, new_line_number=5),
            ],
        )

    def test_finding_on_added_line_creates_inline_comment(self, tmp_path, sample_hunk):
        ws_root = tmp_path / "ws"
        ws_root.mkdir()

        diff_file = DiffFile(
            old_path="src/calc.py",
            new_path="src/calc.py",
            status="modified",
            hunks=[sample_hunk],
        )

        issues = [
            {
                "code": "F841",
                "message": "Local variable 'res' is assigned to but never used",
                "file_path": str(ws_root / "src" / "calc.py"),
                "line": 3,
                "tool": "ruff",
                "severity": "WARNING",
            }
        ]

        diff_map = {"src/calc.py": diff_file}
        comments, unmapped = build_inline_comments(issues, diff_map, ws_root)

        assert len(comments) == 1
        assert comments[0].path == "src/calc.py"
        assert comments[0].line == 3
        assert comments[0].start_line is None
        assert comments[0].side == "RIGHT"
        assert "F841" in comments[0].body
        assert len(unmapped) == 0

    def test_finding_on_context_line_creates_inline_comment(self, tmp_path, sample_hunk):
        ws_root = tmp_path / "ws"
        ws_root.mkdir()

        diff_file = DiffFile(
            old_path="src/calc.py",
            new_path="src/calc.py",
            status="modified",
            hunks=[sample_hunk],
        )

        issues = [
            {
                "code": "F401",
                "message": "Unused import math",
                "file_path": "src/calc.py",
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
        assert comments[0].side == "RIGHT"
        assert len(unmapped) == 0

    def test_finding_on_deleted_line_routed_to_unmapped(self, tmp_path):
        ws_root = tmp_path / "ws"
        ws_root.mkdir()

        # A hunk where line 3 on old side was removed and not present on new side
        diff_file = DiffFile(
            old_path="src/calc.py",
            new_path="src/calc.py",
            status="modified",
            hunks=[
                DiffHunk(
                    header="@@ -1,3 +1,2 @@",
                    old_start=1,
                    old_count=3,
                    new_start=1,
                    new_count=2,
                    section_heading="",
                    lines=[
                        DiffLine(line_type="context", content="import math", old_line_number=1, new_line_number=1),
                        DiffLine(line_type="delete", content="import old_mod", old_line_number=2, new_line_number=None),
                        DiffLine(line_type="context", content="x = 1", old_line_number=3, new_line_number=2),
                    ],
                )
            ],
        )

        # Line 99 or a non-existent new line
        issues = [
            {
                "code": "F401",
                "message": "Unused deleted import",
                "file_path": "src/calc.py",
                "line": 99,
                "tool": "ruff",
                "severity": "WARNING",
            }
        ]

        diff_map = {"src/calc.py": diff_file}
        comments, unmapped = build_inline_comments(issues, diff_map, ws_root)

        assert len(comments) == 0
        assert len(unmapped) == 1
        assert unmapped[0]["code"] == "F401"

    def test_finding_on_deleted_file_routed_to_unmapped(self, tmp_path):
        ws_root = tmp_path / "ws"
        ws_root.mkdir()

        diff_file = DiffFile(
            old_path="src/deprecated.py",
            new_path=None,
            status="deleted",
            hunks=[],
        )

        issues = [
            {
                "code": "E999",
                "message": "Syntax error in deleted file",
                "file_path": "src/deprecated.py",
                "line": 10,
                "tool": "ruff",
                "severity": "ERROR",
            }
        ]

        diff_map = {"src/deprecated.py": diff_file}
        comments, unmapped = build_inline_comments(issues, diff_map, ws_root)

        assert len(comments) == 0
        assert len(unmapped) == 1
        assert unmapped[0]["code"] == "E999"

    def test_finding_on_binary_file_routed_to_unmapped(self, tmp_path):
        ws_root = tmp_path / "ws"
        ws_root.mkdir()

        diff_file = DiffFile(
            old_path="assets/logo.png",
            new_path="assets/logo.png",
            status="modified",
            is_binary=True,
            hunks=[],
        )

        issues = [
            {
                "code": "BIN001",
                "message": "Binary asset too large",
                "file_path": "assets/logo.png",
                "line": 1,
                "tool": "scanner",
                "severity": "LOW",
            }
        ]

        diff_map = {"assets/logo.png": diff_file}
        comments, unmapped = build_inline_comments(issues, diff_map, ws_root)

        assert len(comments) == 0
        assert len(unmapped) == 1
        assert unmapped[0]["code"] == "BIN001"

    def test_finding_on_renamed_file_anchors_to_new_path(self, tmp_path, sample_hunk):
        ws_root = tmp_path / "ws"
        ws_root.mkdir()

        diff_file = DiffFile(
            old_path="src/old_calc.py",
            new_path="src/new_calc.py",
            status="renamed",
            hunks=[sample_hunk],
        )

        # 1. Finding references new_path
        issues_new = [
            {
                "code": "F841",
                "message": "Unused variable",
                "file_path": "src/new_calc.py",
                "line": 2,
                "tool": "ruff",
                "severity": "WARNING",
            }
        ]
        diff_map = {"src/new_calc.py": diff_file}
        comments, unmapped = build_inline_comments(issues_new, diff_map, ws_root)

        assert len(comments) == 1
        assert comments[0].path == "src/new_calc.py"
        assert comments[0].line == 2
        assert len(unmapped) == 0

        # 2. Finding references old_path (should still resolve and anchor to new_path)
        issues_old = [
            {
                "code": "F841",
                "message": "Unused variable",
                "file_path": "src/old_calc.py",
                "line": 2,
                "tool": "ruff",
                "severity": "WARNING",
            }
        ]
        comments_old, unmapped_old = build_inline_comments(issues_old, diff_map, ws_root)

        assert len(comments_old) == 1
        assert comments_old[0].path == "src/new_calc.py"
        assert comments_old[0].line == 2
        assert len(unmapped_old) == 0

    def test_finding_with_valid_multiline_span_anchors_inline(self, tmp_path, sample_hunk):
        ws_root = tmp_path / "ws"
        ws_root.mkdir()

        diff_file = DiffFile(
            old_path="src/calc.py",
            new_path="src/calc.py",
            status="modified",
            hunks=[sample_hunk],
        )

        # Lines 2 to 4 are entirely within sample_hunk (lines 1..5)
        issues = [
            {
                "code": "B102",
                "message": "Multi-line code block issue",
                "file_path": "src/calc.py",
                "line": 4,
                "start_line": 2,
                "tool": "bandit",
                "severity": "MEDIUM",
            }
        ]

        diff_map = {"src/calc.py": diff_file}
        comments, unmapped = build_inline_comments(issues, diff_map, ws_root)

        assert len(comments) == 1
        assert comments[0].path == "src/calc.py"
        assert comments[0].line == 4
        assert comments[0].start_line == 2
        assert comments[0].side == "RIGHT"
        assert len(unmapped) == 0

    def test_finding_with_cross_hunk_span_routed_to_unmapped(self, tmp_path):
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
                        DiffLine(line_type="context", content="a = 1", new_line_number=1),
                        DiffLine(line_type="add", content="b = 2", new_line_number=2),
                    ],
                ),
                DiffHunk(
                    header="@@ -20,5 +20,5 @@",
                    old_start=20,
                    old_count=5,
                    new_start=20,
                    new_count=5,
                    section_heading="",
                    lines=[
                        DiffLine(line_type="context", content="y = 20", new_line_number=20),
                        DiffLine(line_type="add", content="z = 21", new_line_number=21),
                    ],
                ),
            ],
        )

        # Span starts in hunk 1 (line 2) and ends in hunk 2 (line 21)
        issues = [
            {
                "code": "B102",
                "message": "Cross-hunk multi-line vulnerability",
                "file_path": "src/calc.py",
                "line": 21,
                "start_line": 2,
                "tool": "bandit",
                "severity": "HIGH",
            }
        ]

        diff_map = {"src/calc.py": diff_file}
        comments, unmapped = build_inline_comments(issues, diff_map, ws_root)

        assert len(comments) == 0
        assert len(unmapped) == 1
        assert unmapped[0]["code"] == "B102"

    def test_finding_with_inverted_or_invalid_span_routed_to_unmapped(self, tmp_path, sample_hunk):
        ws_root = tmp_path / "ws"
        ws_root.mkdir()

        diff_file = DiffFile(
            old_path="src/calc.py",
            new_path="src/calc.py",
            status="modified",
            hunks=[sample_hunk],
        )

        issues = [
            # Inverted span (start > line)
            {
                "code": "INV01",
                "message": "Inverted span",
                "file_path": "src/calc.py",
                "line": 2,
                "start_line": 4,
                "tool": "ruff",
                "severity": "LOW",
            },
            # Negative start_line
            {
                "code": "INV02",
                "message": "Negative start line",
                "file_path": "src/calc.py",
                "line": 3,
                "start_line": -1,
                "tool": "ruff",
                "severity": "LOW",
            },
            # Non-integer start_line
            {
                "code": "INV03",
                "message": "Non-integer start line",
                "file_path": "src/calc.py",
                "line": 3,
                "start_line": "invalid",
                "tool": "ruff",
                "severity": "LOW",
            },
        ]

        diff_map = {"src/calc.py": diff_file}
        comments, unmapped = build_inline_comments(issues, diff_map, ws_root)

        assert len(comments) == 0
        assert len(unmapped) == 3

    def test_finding_with_out_of_hunk_coordinates_routed_to_unmapped(self, tmp_path, sample_hunk):
        ws_root = tmp_path / "ws"
        ws_root.mkdir()

        diff_file = DiffFile(
            old_path="src/calc.py",
            new_path="src/calc.py",
            status="modified",
            hunks=[sample_hunk],
        )

        issues = [
            # Line far outside hunk
            {
                "code": "OUT01",
                "message": "Out of hunk line",
                "file_path": "src/calc.py",
                "line": 999,
                "tool": "ruff",
                "severity": "LOW",
            },
            # Line is zero or negative
            {
                "code": "OUT02",
                "message": "Zero line",
                "file_path": "src/calc.py",
                "line": 0,
                "tool": "ruff",
                "severity": "LOW",
            },
            # Line is None
            {
                "code": "OUT03",
                "message": "Missing line",
                "file_path": "src/calc.py",
                "line": None,
                "tool": "ruff",
                "severity": "LOW",
            },
        ]

        diff_map = {"src/calc.py": diff_file}
        comments, unmapped = build_inline_comments(issues, diff_map, ws_root)

        assert len(comments) == 0
        assert len(unmapped) == 3

    def test_finding_with_malformed_path_routed_to_unmapped(self, tmp_path, sample_hunk):
        ws_root = tmp_path / "ws"
        ws_root.mkdir()

        diff_file = DiffFile(
            old_path="src/calc.py",
            new_path="src/calc.py",
            status="modified",
            hunks=[sample_hunk],
        )

        issues = [
            {"code": "MAL01", "message": "Empty path", "file_path": "", "line": 2, "tool": "ruff", "severity": "LOW"},
            {"code": "MAL02", "message": "Unknown path", "file_path": "nonexistent.py", "line": 2, "tool": "ruff", "severity": "LOW"},
        ]

        diff_map = {"src/calc.py": diff_file}
        comments, unmapped = build_inline_comments(issues, diff_map, ws_root)

        assert len(comments) == 0
        assert len(unmapped) == 2


class TestReviewSummaryWithUnmappedFindings:
    def test_review_summary_renders_unmapped_findings_section(self, mock_pr_metadata):
        unmapped = [
            {
                "file_path": "src/old.py",
                "line": 50,
                "code": "E501",
                "tool": "ruff",
                "severity": "WARNING",
                "message": "Line too long in unmodified file",
            },
            {
                "file_path": "assets/logo.png",
                "line": None,
                "code": "BIN01",
                "tool": "scanner",
                "severity": "LOW",
                "message": "Binary asset uncompressed",
            },
        ]

        summary = generate_review_summary(
            meta=mock_pr_metadata,
            verdict="COMMENT",
            total_issues=2,
            inline_comments_count=0,
            unmapped_issues_count=2,
            workflow_state={},
            changed_files=["src/calc.py"],
            unmapped_issues=unmapped,
        )

        assert "#### File-Level & Unmapped Findings (Outside Diff Hunks)" in summary
        assert "`src/old.py:50`" in summary
        assert "**[Ruff - E501]** (WARNING): Line too long in unmodified file" in summary
        assert "`assets/logo.png`" in summary
        assert "**[Scanner - BIN01]** (LOW): Binary asset uncompressed" in summary



# ============================================================================
# 7. Clean PR Scan Handling (Stage 4 Refined: Verdict is COMMENT, never APPROVE)
# ============================================================================


class TestCleanPRScanHandling:
    def test_clean_pr_scan_yields_comment_verdict(self, mock_pr_metadata, mock_pr_workspace):
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

            # Never automatically choose APPROVE; clean PR scans yield COMMENT
            assert result.action_plan.verdict == "COMMENT"
            assert result.action_plan.verdict != "APPROVE"
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
            # In read-only PR preview mode, successful review completion returns 0
            # even when issues are found (findings are rendered in summary/output).
            assert exit_code == 0

    def test_cli_run_pr_pipeline_error_handling(self, capsys):
        with patch("main.run_pr_review_workflow", side_effect=PRReviewInputError("Invalid PR number")):
            exit_code = run_pr_pipeline("invalid_input")
            assert exit_code == 2
            captured = capsys.readouterr().out
            assert "PR WORKFLOW FAILED: Invalid PR number" in captured


# ============================================================================
# 9. Stage 4: Deterministic Verdict Mapping Tests
# ============================================================================


class TestVerdictMapping:
    """Verifies deterministic verdict mapping according to approved policy."""

    def test_qualifying_critical_findings_with_explicit_high_confidence_map_to_request_changes(self):
        # 1. Critical severity with explicit HIGH confidence
        crit_issue = {"tool": "bandit", "severity": "CRITICAL", "confidence": "HIGH", "code": "B101"}
        assert determine_review_verdict([crit_issue], []) == "REQUEST_CHANGES"

        # 2. Critical severity with explicit CRITICAL confidence
        crit_issue_crit = {"severity": "CRITICAL", "confidence": "CRITICAL", "code": "C001"}
        assert determine_review_verdict([crit_issue_crit], []) == "REQUEST_CHANGES"

        # 3. Bandit HIGH severity + HIGH confidence
        bandit_high_high = {"tool": "bandit", "severity": "HIGH", "confidence": "HIGH", "code": "B608"}
        assert determine_review_verdict([bandit_high_high], []) == "REQUEST_CHANGES"

        # 4. Bandit HIGH severity + CRITICAL confidence
        bandit_high_crit = {"tool": "bandit", "severity": "HIGH", "confidence": "CRITICAL", "code": "B608"}
        assert determine_review_verdict([bandit_high_crit], []) == "REQUEST_CHANGES"

        # 5. Bandit HIGH severity with issue_confidence field as HIGH
        bandit_alt_conf = {"tool": "bandit", "severity": "HIGH", "issue_confidence": "HIGH", "code": "B608"}
        assert determine_review_verdict([bandit_alt_conf], []) == "REQUEST_CHANGES"

    def test_high_severity_with_missing_or_unknown_confidence_maps_to_comment(self):
        # Missing confidence field -> COMMENT
        bandit_high_noconf = {"tool": "bandit", "severity": "HIGH", "code": "B608"}
        assert determine_review_verdict([bandit_high_noconf], []) == "COMMENT"

        # Empty string confidence -> COMMENT
        bandit_high_emptyconf = {"tool": "bandit", "severity": "HIGH", "confidence": "", "code": "B608"}
        assert determine_review_verdict([bandit_high_emptyconf], []) == "COMMENT"

        # Unknown confidence -> COMMENT
        bandit_high_unknownconf = {"tool": "bandit", "severity": "HIGH", "confidence": "UNKNOWN", "code": "B608"}
        assert determine_review_verdict([bandit_high_unknownconf], []) == "COMMENT"

        # Critical severity with missing confidence -> COMMENT
        crit_noconf = {"severity": "CRITICAL", "code": "C001"}
        assert determine_review_verdict([crit_noconf], []) == "COMMENT"

    def test_high_severity_with_low_or_medium_confidence_maps_to_comment(self):
        # Bandit HIGH severity but LOW confidence -> maps to COMMENT
        bandit_high_low = {"tool": "bandit", "severity": "HIGH", "confidence": "LOW", "code": "B608"}
        assert determine_review_verdict([bandit_high_low], []) == "COMMENT"

        # Bandit HIGH severity but MEDIUM confidence -> maps to COMMENT
        bandit_high_med = {"tool": "bandit", "severity": "HIGH", "confidence": "MEDIUM", "code": "B608"}
        assert determine_review_verdict([bandit_high_med], []) == "COMMENT"

        # Critical severity with LOW confidence -> maps to COMMENT
        crit_low = {"severity": "CRITICAL", "confidence": "LOW", "code": "C001"}
        assert determine_review_verdict([crit_low], []) == "COMMENT"

    def test_fatal_analysis_errors_map_to_request_changes(self):
        # 1. Fatal severity issue (e.g. fatal tool failure issue)
        fatal_issue = {"tool": "scanner", "severity": "FATAL", "code": "F001"}
        assert determine_review_verdict([fatal_issue], []) == "REQUEST_CHANGES"

        # 2. Tool execution crashes / exceptions in agent_errors
        assert determine_review_verdict([], ["Fatal error: Ruff crashed"]) == "REQUEST_CHANGES"
        assert determine_review_verdict([], ["Tool execution failed: timeout after 60s"]) == "REQUEST_CHANGES"
        assert determine_review_verdict([], ["Process exception: memory error"]) == "REQUEST_CHANGES"

    def test_clean_scans_and_ordinary_findings_map_to_comment(self):
        # Clean scan with no findings and no errors
        assert determine_review_verdict([], []) == "COMMENT"

        # Ordinary findings
        ordinary_issues = [
            {"tool": "ruff", "severity": "WARNING", "code": "W291", "message": "Trailing whitespace"},
            {"tool": "ruff", "severity": "ERROR", "code": "E501", "message": "Line too long"},
            {"tool": "bandit", "severity": "MEDIUM", "confidence": "HIGH", "code": "B105"},
            {"tool": "bandit", "severity": "LOW", "confidence": "LOW", "code": "B104"},
        ]
        assert determine_review_verdict(ordinary_issues, []) == "COMMENT"

    def test_no_verdict_path_automatically_returns_approve(self):
        # Even with 0 issues, verdict is COMMENT, never APPROVE
        verdict = determine_review_verdict([], [])
        assert verdict == "COMMENT"
        assert verdict != "APPROVE"

        # With ordinary issues
        verdict_ord = determine_review_verdict([{"tool": "ruff", "severity": "WARNING"}], [])
        assert verdict_ord == "COMMENT"
        assert verdict_ord != "APPROVE"

    def test_is_high_confidence_critical_issue_helper_boundaries(self):
        # Explicit high/critical confidence on qualifying severity
        assert is_high_confidence_critical_issue({"severity": "CRITICAL", "confidence": "HIGH"}) is True
        assert is_high_confidence_critical_issue({"severity": "critical", "confidence": "critical"}) is True
        assert is_high_confidence_critical_issue({"tool": "BANDIT", "severity": "high", "confidence": "high"}) is True
        assert is_high_confidence_critical_issue({"tool": "BANDIT", "severity": "high", "issue_confidence": "HIGH"}) is True

        # Fatal severity is always fatal analysis error
        assert is_high_confidence_critical_issue({"severity": "FATAL"}) is True

        # Missing or non-high confidence must return False
        assert is_high_confidence_critical_issue({"severity": "critical"}) is False
        assert is_high_confidence_critical_issue({"tool": "bandit", "severity": "high"}) is False
        assert is_high_confidence_critical_issue({"tool": "bandit", "severity": "high", "confidence": "low"}) is False
        assert is_high_confidence_critical_issue({"tool": "bandit", "severity": "high", "confidence": "medium"}) is False
        assert is_high_confidence_critical_issue({"tool": "bandit", "severity": "medium", "confidence": "high"}) is False
        assert is_high_confidence_critical_issue({}) is False

    def test_has_fatal_analysis_error_helper_boundaries(self):
        assert has_fatal_analysis_error(["Analysis failed with segfault"]) is True
        assert has_fatal_analysis_error([""]) is False
        assert has_fatal_analysis_error(["   "]) is False
        assert has_fatal_analysis_error([]) is False
        assert has_fatal_analysis_error(["Warning: linter found 2 notices"]) is False
        # Structured tool_status takes precedence over empty or non-keyword error strings
        assert has_fatal_analysis_error([], tool_status={"semantic": False}) is True
        assert has_fatal_analysis_error([], tool_status={"semantic": True, "ruff": True, "bandit": True}) is False
        assert has_fatal_analysis_error(["Target path does not exist"], tool_status={"semantic": False}) is True

    def test_failed_semantic_analyzer_without_error_keywords_maps_to_request_changes(self):
        """Failed semantic analyzer maps to REQUEST_CHANGES even without error/failed keywords."""
        assert (
            determine_review_verdict([], ["SemanticAnalyzer: Target path does not exist"])
            == "REQUEST_CHANGES"
        )
        assert (
            determine_review_verdict([], [], tool_status={"semantic": False})
            == "REQUEST_CHANGES"
        )

    def test_test_writer_http_401_auth_error_maps_to_comment(self):
        """HTTP 401 / invalid API key in TestWriterAgent is an advisory warning, not fatal."""
        auth_err = (
            "Test-Writer Agent Provider Error: OpenAIAuthenticationError, "
            "HTTP 401, invalid_api_key."
        )
        assert determine_review_verdict([], [auth_err]) == "COMMENT"

    def test_test_validation_and_storage_failures_map_to_comment(self):
        """Test validation or workspace storage errors are advisory and map to COMMENT."""
        val_err = "Test validation failed for 'calc.py': Content is too short (< 20 chars) to be valid test code."
        assert determine_review_verdict([], [val_err]) == "COMMENT"

        store_err = "Test storage error for 'calc.py': disk full"
        assert determine_review_verdict([], [store_err]) == "COMMENT"

    def test_genuine_fatal_primary_tool_failure_maps_to_request_changes(self):
        """Primary static analysis tool crashes or timeouts map to REQUEST_CHANGES."""
        assert determine_review_verdict([], ["Fatal error: Ruff crashed with SIGSEGV"]) == "REQUEST_CHANGES"
        assert determine_review_verdict([], ["Linter tool error: Failed to parse configuration"]) == "REQUEST_CHANGES"
        assert determine_review_verdict([], ["Security tool error: Bandit process timed out after 30s"]) == "REQUEST_CHANGES"
        assert determine_review_verdict([], ["Tool execution failed: non-zero exit"]) == "REQUEST_CHANGES"

    def test_fatal_primary_failure_overrides_auxiliary_advisory_warning(self):
        """When both a primary tool crash and an auxiliary warning exist, fatal failure takes precedence."""
        assert determine_review_verdict(
            [],
            [
                "Fatal error: Ruff crashed",
                "Test-Writer Agent Provider Error: OpenAIAuthenticationError, HTTP 401, invalid_api_key.",
            ],
        ) == "REQUEST_CHANGES"

    def test_generate_review_summary_surfaces_advisory_warnings_and_redacts_credentials(self):
        """Executive summary surfaces advisory warnings and redacts secrets."""
        from github_client import PRMetadata
        meta = PRMetadata(
            owner="octocat",
            repo="Hello-World",
            pr_number=42,
            title="Test PR",
            body="Desc",
            state="open",
            head_sha="0123456789abcdef0123456789abcdef01234567",
            base_sha="base_sha",
            head_branch="feature",
            base_branch="main",
            author="alice",
            html_url="https://github.com/octocat/Hello-World/pull/42",
        )
        workflow_state = {
            "agent_errors": [
                "Test-Writer Agent Provider Error: OpenAIAuthenticationError sk-1234567890abcdef1234567890 HTTP 401"
            ],
            "final_review_report": {},
        }
        summary = generate_review_summary(
            meta=meta,
            verdict="COMMENT",
            total_issues=0,
            inline_comments_count=0,
            unmapped_issues_count=0,
            workflow_state=workflow_state,
            changed_files=["calc.py"],
        )
        assert "Auxiliary Agent Warnings" in summary
        assert "sk-1234567890abcdef1234567890" not in summary
        assert "[REDACTED" in summary
        assert "Primary Static Analysis Status:" in summary
        assert "PASSED (Ruff & Bandit completed)" in summary


# ============================================================================
# 10. Stage 4: Pre-Publication Revalidation Tests
# ============================================================================


class TestPrePublicationRevalidation:
    """Verifies all fail-closed pre-publication revalidation checks."""

    @pytest.fixture
    def valid_plan(self):
        return ActionPlan(
            owner="octocat",
            repo="Hello-World",
            pr_number=42,
            head_sha=VALID_HEAD_SHA,
            verdict="COMMENT",
            summary="Review summary body.",
            comments=(
                ReviewComment(
                    path="sample_project/app.py",
                    line=2,
                    body="Consider checking return value.",
                ),
            ),
            dry_run=False,
            created_at_mono=100.0,
            created_at_utc="2026-10-15T12:00:00Z",
        )

    @pytest.fixture
    def valid_approval(self, valid_plan):
        return ApprovalResult(
            approved=True,
            dry_run=False,
            plan=valid_plan,
            source="interactive",
            plan_fingerprint=valid_plan.fingerprint,
            created_at_mono=100.0,
            approved_at_mono=110.0,
            approved_at_utc="2026-10-15T12:01:50Z",
            expires_at_mono=710.0,
            approval_ttl=600.0,
        )

    @pytest.fixture
    def mock_revalidation_client(self, mock_pr_metadata):
        client = MagicMock(spec=GitHubClient)
        client.get_pr_metadata.return_value = mock_pr_metadata
        client.get_pr_diff.return_value = SAMPLE_DIFF
        return client

    def test_successful_pre_publication_revalidation(
        self, valid_plan, valid_approval, mock_revalidation_client, mock_pr_metadata
    ):
        # Clock at 200.0 (approved at 110.0, expires at 710.0, plan created at 100.0)
        clock = lambda: 200.0
        result = revalidate_review_for_publication(
            approval_result=valid_approval,
            current_plan=valid_plan,
            client=mock_revalidation_client,
            clock=clock,
            publication_marker="marker-12345",
        )

        assert isinstance(result, PrePublicationRevalidationResult)
        assert result.is_valid is True
        assert result.live_metadata == mock_pr_metadata
        assert result.plan_fingerprint == valid_plan.fingerprint
        assert result.revalidated_at_mono == 200.0
        assert "<!-- agy-review-marker: marker-12345 -->" in result.serialized_payload["body"]
        assert result.serialized_payload["commit_id"] == VALID_HEAD_SHA.lower()
        assert result.serialized_payload["event"] == "COMMENT"
        assert len(result.serialized_payload["comments"]) == 1

    def test_rejection_when_live_head_sha_changes(
        self, valid_plan, valid_approval, mock_revalidation_client, mock_pr_metadata
    ):
        mock_revalidation_client.get_pr_metadata.return_value = replace(
            mock_pr_metadata, head_sha=DIFFERENT_HEAD_SHA
        )
        with pytest.raises(HeadShaMismatchError, match="Head SHA mismatch"):
            revalidate_review_for_publication(
                approval_result=valid_approval,
                current_plan=valid_plan,
                client=mock_revalidation_client,
                clock=lambda: 200.0,
            )

    def test_rejection_when_repository_owner_differs(
        self, valid_plan, valid_approval, mock_revalidation_client, mock_pr_metadata
    ):
        mock_revalidation_client.get_pr_metadata.return_value = replace(
            mock_pr_metadata, owner="attacker"
        )
        with pytest.raises(PrePublicationValidationError, match="Repository owner mismatch"):
            revalidate_review_for_publication(
                approval_result=valid_approval,
                current_plan=valid_plan,
                client=mock_revalidation_client,
                clock=lambda: 200.0,
            )

    def test_rejection_when_repository_name_differs(
        self, valid_plan, valid_approval, mock_revalidation_client, mock_pr_metadata
    ):
        mock_revalidation_client.get_pr_metadata.return_value = replace(
            mock_pr_metadata, repo="different-repo"
        )
        with pytest.raises(PrePublicationValidationError, match="Repository name mismatch"):
            revalidate_review_for_publication(
                approval_result=valid_approval,
                current_plan=valid_plan,
                client=mock_revalidation_client,
                clock=lambda: 200.0,
            )

    def test_rejection_when_pr_number_differs(
        self, valid_plan, valid_approval, mock_revalidation_client, mock_pr_metadata
    ):
        mock_revalidation_client.get_pr_metadata.return_value = replace(
            mock_pr_metadata, pr_number=999
        )
        with pytest.raises(PrePublicationValidationError, match="PR number mismatch"):
            revalidate_review_for_publication(
                approval_result=valid_approval,
                current_plan=valid_plan,
                client=mock_revalidation_client,
                clock=lambda: 200.0,
            )

    def test_rejection_when_pr_is_closed_or_merged(
        self, valid_plan, valid_approval, mock_revalidation_client, mock_pr_metadata
    ):
        mock_revalidation_client.get_pr_metadata.return_value = replace(
            mock_pr_metadata, state="closed"
        )
        with pytest.raises(PrePublicationValidationError, match="Cannot publish review on closed"):
            revalidate_review_for_publication(
                approval_result=valid_approval,
                current_plan=valid_plan,
                client=mock_revalidation_client,
                clock=lambda: 200.0,
            )

    def test_rejection_when_approval_is_missing_or_rejected(
        self, valid_plan, mock_revalidation_client
    ):
        rejected_approval = ApprovalResult(
            approved=False,
            dry_run=False,
            plan=valid_plan,
            source="rejected",
            reason="User typed n",
        )
        with pytest.raises(ApprovalRejectedError):
            revalidate_review_for_publication(
                approval_result=rejected_approval,
                current_plan=valid_plan,
                client=mock_revalidation_client,
                clock=lambda: 200.0,
            )

    def test_rejection_when_approval_was_in_dry_run_mode(
        self, valid_plan, mock_revalidation_client
    ):
        dry_run_approval = ApprovalResult(
            approved=True,
            dry_run=True,
            plan=valid_plan,
            source="dry_run",
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )
        with pytest.raises(DryRunBlockedError):
            revalidate_review_for_publication(
                approval_result=dry_run_approval,
                current_plan=valid_plan,
                client=mock_revalidation_client,
                clock=lambda: 200.0,
            )

    def test_rejection_when_approval_ttl_has_expired_exact_boundary(
        self, valid_plan, valid_approval, mock_revalidation_client
    ):
        # Exact boundary: expires_at_mono = 710.0
        # Exactly at 710.0 -> expired (now_mono >= expires_at_mono)
        with pytest.raises(ApprovalExpiredError, match="has expired"):
            revalidate_review_for_publication(
                approval_result=valid_approval,
                current_plan=valid_plan,
                client=mock_revalidation_client,
                clock=lambda: 710.0,
            )

        # Beyond boundary: 710.1 -> expired
        with pytest.raises(ApprovalExpiredError, match="has expired"):
            revalidate_review_for_publication(
                approval_result=valid_approval,
                current_plan=valid_plan,
                client=mock_revalidation_client,
                clock=lambda: 710.1,
            )

    def test_rejection_when_plan_is_stale(
        self, valid_plan, valid_approval, mock_revalidation_client
    ):
        # Plan created at 100.0, approved at 650.0 (valid TTL until 1250.0).
        # But at clock 700.1, plan age is 600.1s which exceeds max_plan_age_seconds=600.0
        approval_late = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=valid_plan,
            source="interactive",
            plan_fingerprint=valid_plan.fingerprint,
            created_at_mono=100.0,
            approved_at_mono=650.0,
            expires_at_mono=1250.0,
        )
        with pytest.raises(ApprovalExpiredError, match="Action plan is stale"):
            revalidate_review_for_publication(
                approval_result=approval_late,
                current_plan=valid_plan,
                client=mock_revalidation_client,
                clock=lambda: 700.1,
                max_plan_age_seconds=600.0,
            )

    def test_rejection_when_plan_fingerprint_changes_after_approval(
        self, valid_plan, valid_approval, mock_revalidation_client
    ):
        # Tampered / modified plan (verdict modified)
        modified_plan = ActionPlan(
            owner=valid_plan.owner,
            repo=valid_plan.repo,
            pr_number=valid_plan.pr_number,
            head_sha=valid_plan.head_sha,
            verdict="REQUEST_CHANGES",  # Changed from COMMENT
            summary=valid_plan.summary,
            comments=valid_plan.comments,
            dry_run=valid_plan.dry_run,
            created_at_mono=valid_plan.created_at_mono,
            created_at_utc=valid_plan.created_at_utc,
        )
        with pytest.raises(PlanFingerprintMismatchError):
            revalidate_review_for_publication(
                approval_result=valid_approval,
                current_plan=modified_plan,
                client=mock_revalidation_client,
                clock=lambda: 200.0,
            )

    def test_rejection_when_comments_modified_after_approval(
        self, valid_plan, valid_approval, mock_revalidation_client
    ):
        # Adding an extra comment alters the fingerprint
        tampered_comments_plan = ActionPlan(
            owner=valid_plan.owner,
            repo=valid_plan.repo,
            pr_number=valid_plan.pr_number,
            head_sha=valid_plan.head_sha,
            verdict=valid_plan.verdict,
            summary=valid_plan.summary,
            comments=valid_plan.comments + (
                ReviewComment(path="sample_project/app.py", line=7, body="Sneaky comment"),
            ),
            dry_run=valid_plan.dry_run,
            created_at_mono=valid_plan.created_at_mono,
            created_at_utc=valid_plan.created_at_utc,
        )
        with pytest.raises(PlanFingerprintMismatchError):
            revalidate_review_for_publication(
                approval_result=valid_approval,
                current_plan=tampered_comments_plan,
                client=mock_revalidation_client,
                clock=lambda: 200.0,
            )

    def test_rejection_when_inline_comment_targets_unmatched_file(
        self, valid_plan, mock_revalidation_client
    ):
        plan_missing_file = ActionPlan(
            owner="octocat",
            repo="Hello-World",
            pr_number=42,
            head_sha=VALID_HEAD_SHA,
            verdict="COMMENT",
            summary="Summary",
            comments=(ReviewComment(path="nonexistent.py", line=1, body="Msg"),),
            created_at_mono=100.0,
        )
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=plan_missing_file,
            source="interactive",
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )
        with pytest.raises(CommentAnchoringError, match="does not exist in the PR diff"):
            revalidate_review_for_publication(
                approval_result=approval,
                current_plan=plan_missing_file,
                client=mock_revalidation_client,
                clock=lambda: 200.0,
            )

    def test_rejection_when_inline_comment_targets_binary_or_deleted_file(
        self, mock_revalidation_client
    ):
        diff_deleted = """diff --git a/old.py b/old.py
deleted file mode 100644
--- a/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-line1
-line2
"""
        mock_revalidation_client.get_pr_diff.return_value = diff_deleted
        plan_deleted = ActionPlan(
            owner="octocat",
            repo="Hello-World",
            pr_number=42,
            head_sha=VALID_HEAD_SHA,
            verdict="COMMENT",
            summary="Summary",
            comments=(ReviewComment(path="old.py", line=1, body="Msg"),),
            created_at_mono=100.0,
        )
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=plan_deleted,
            source="interactive",
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )
        with pytest.raises(CommentAnchoringError, match="targets binary or deleted file"):
            revalidate_review_for_publication(
                approval_result=approval,
                current_plan=plan_deleted,
                client=mock_revalidation_client,
                clock=lambda: 200.0,
            )

    def test_rejection_when_inline_comment_line_outside_diff_hunks(
        self, mock_revalidation_client
    ):
        plan_out_of_bounds = ActionPlan(
            owner="octocat",
            repo="Hello-World",
            pr_number=42,
            head_sha=VALID_HEAD_SHA,
            verdict="COMMENT",
            summary="Summary",
            comments=(ReviewComment(path="sample_project/app.py", line=999, body="Out of bounds"),),
            created_at_mono=100.0,
        )
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=plan_out_of_bounds,
            source="interactive",
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )
        with pytest.raises(CommentAnchoringError, match="is not a valid editable line"):
            revalidate_review_for_publication(
                approval_result=approval,
                current_plan=plan_out_of_bounds,
                client=mock_revalidation_client,
                clock=lambda: 200.0,
            )

    def test_rejection_when_multiline_comment_crosses_diff_hunks(
        self, mock_revalidation_client
    ):
        diff_two_hunks = """diff --git a/sample_project/app.py b/sample_project/app.py
--- a/sample_project/app.py
+++ b/sample_project/app.py
@@ -1,3 +1,3 @@
 line1
 line2
 line3
@@ -20,3 +20,3 @@
 line20
 line21
 line22
"""
        mock_revalidation_client.get_pr_diff.return_value = diff_two_hunks
        plan_cross_hunk = ActionPlan(
            owner="octocat",
            repo="Hello-World",
            pr_number=42,
            head_sha=VALID_HEAD_SHA,
            verdict="COMMENT",
            summary="Summary",
            comments=(ReviewComment(path="sample_project/app.py", line=21, start_line=2, body="Cross hunk"),),
            created_at_mono=100.0,
        )
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=plan_cross_hunk,
            source="interactive",
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )
        with pytest.raises(CommentAnchoringError, match="does not reside within a single diff hunk"):
            revalidate_review_for_publication(
                approval_result=approval,
                current_plan=plan_cross_hunk,
                client=mock_revalidation_client,
                clock=lambda: 200.0,
            )


# ============================================================================
# 11. Zero Write Requests Guarantee on Revalidation Failures
# ============================================================================


class TestZeroWriteRequestsPrePublication:
    """Proves that under NO circumstance does pre-publication revalidation trigger write requests."""

    def test_all_failure_paths_produce_zero_writes(self, mock_pr_metadata):
        mock_client = MagicMock(spec=GitHubClient)
        mock_client.get_pr_metadata.return_value = mock_pr_metadata
        mock_client.get_pr_diff.return_value = SAMPLE_DIFF

        # Methods that would execute writes
        mock_client.create_pull_request_review = MagicMock()
        mock_client.post = MagicMock()
        mock_client.put = MagicMock()
        mock_client.patch = MagicMock()
        mock_client.delete = MagicMock()

        plan = ActionPlan(
            owner="octocat",
            repo="Hello-World",
            pr_number=42,
            head_sha=VALID_HEAD_SHA,
            verdict="COMMENT",
            summary="Summary",
            created_at_mono=100.0,
        )
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=plan,
            source="interactive",
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )

        # 1. Failure via SHA mismatch
        mock_client.get_pr_metadata.return_value = replace(
            mock_pr_metadata, head_sha=DIFFERENT_HEAD_SHA
        )
        with pytest.raises(HeadShaMismatchError):
            revalidate_review_for_publication(approval, plan, mock_client, clock=lambda: 200.0)

        # 2. Failure via expired TTL
        mock_client.get_pr_metadata.return_value = mock_pr_metadata
        with pytest.raises(ApprovalExpiredError):
            revalidate_review_for_publication(approval, plan, mock_client, clock=lambda: 800.0)

        # 3. Failure via stale plan
        with pytest.raises(ApprovalExpiredError):
            revalidate_review_for_publication(
                approval, plan, mock_client, clock=lambda: 200.0, max_plan_age_seconds=50.0
            )

        # Assert ZERO calls to any write method
        mock_client.create_pull_request_review.assert_not_called()
        mock_client.post.assert_not_called()
        mock_client.put.assert_not_called()
        mock_client.patch.assert_not_called()
        mock_client.delete.assert_not_called()
