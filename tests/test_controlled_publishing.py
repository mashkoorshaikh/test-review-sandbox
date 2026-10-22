"""
tests/test_controlled_publishing.py - Stage 5 Controlled Publishing Integration Tests.

Validates:
  1. Default CLI execution performs no writes.
  2. --publish --dry-run performs no writes and clearly displays the proposed review payload.
  3. Publishing without an interactive TTY is rejected (fail-closed).
  4. User rejection ('n'), empty input (''), EOF, or keyboard interruption performs no writes.
  5. Expired approval and stale plan perform no writes.
  6. Tampered plan or fingerprint mismatch performs no writes.
  7. Head SHA, repository identity, or PR number mismatch performs no writes.
  8. Invalid comment anchoring performs no writes.
  9. A valid approval and successful revalidation result in exactly one POST.
  10. HTTP validation failure (422) is reported correctly.
  11. POST timeout or connection loss is treated as uncertain, with no automatic POST retry.
      Safe reconciliation mechanism is verified.
  12. A successful POST reports success only after a valid server response.
  13. Existing local and read-only PR modes continue to work.
  14. No code path automatically returns APPROVE.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from approval_gate import (
    ActionPlan,
    ApprovalExpiredError,
    ApprovalResult,
    PlanFingerprintMismatchError,
    ReviewComment,
)
from github_client import (
    GitHubClient,
    GitHubUncertainWriteError,
    GitHubValidationError,
    PRMetadata,
)
from github_review_workflow import (
    CommentAnchoringError,
    PrePublicationValidationError,
    PRReviewResult,
    determine_review_verdict,
    publish_approved_review,
    run_pr_publishing_workflow,
)
from main import main, run_pipeline, run_pr_pipeline
from tools import ToolResult

# ---------------------------------------------------------------------------
# Test Fixtures & Deterministic Constants
# ---------------------------------------------------------------------------

VALID_OWNER = "octocat"
VALID_REPO = "Hello-World"
VALID_PR_NUM = 42
VALID_HEAD_SHA = "0123456789abcdef0123456789abcdef01234567"
DIFFERENT_HEAD_SHA = "fedcba9876543210fedcba9876543210fedcba98"

SAMPLE_DIFF = """\
diff --git a/sample_project/app.py b/sample_project/app.py
index 0000001..0000002 100644
--- a/sample_project/app.py
+++ b/sample_project/app.py
@@ -10,3 +10,3 @@ def calculate():
-    return 1
+    return 2
"""


@pytest.fixture
def mock_pr_meta() -> PRMetadata:
    return PRMetadata(
        owner=VALID_OWNER,
        repo=VALID_REPO,
        pr_number=VALID_PR_NUM,
        title="Test PR",
        body="PR description",
        state="open",
        head_sha=VALID_HEAD_SHA,
        base_sha="base_sha_1234567890",
        head_branch="feature",
        base_branch="main",
        author="alice",
        html_url=f"https://github.com/{VALID_OWNER}/{VALID_REPO}/pull/{VALID_PR_NUM}",
    )


@pytest.fixture
def mock_gh_client(mock_pr_meta) -> MagicMock:
    client = MagicMock(spec=GitHubClient)
    client.get_pr_metadata.return_value = mock_pr_meta
    client.get_pr_diff.return_value = SAMPLE_DIFF
    client.get_pr_reviews.return_value = []
    client.get_authenticated_user.return_value = {"login": "bot-user", "id": 12345}
    client.create_pull_request_review.return_value = {
        "id": 101,
        "html_url": f"https://github.com/{VALID_OWNER}/{VALID_REPO}/pull/{VALID_PR_NUM}#pullrequestreview-101",
        "state": "COMMENTED",
        "commit_id": VALID_HEAD_SHA,
        "body": "Automated Review",
    }
    return client


@pytest.fixture
def valid_review_comment() -> ReviewComment:
    return ReviewComment(
        path="sample_project/app.py",
        line=10,
        body="Consider performance optimization here.",
        side="RIGHT",
    )


@pytest.fixture
def base_action_plan(valid_review_comment) -> ActionPlan:
    return ActionPlan(
        owner=VALID_OWNER,
        repo=VALID_REPO,
        pr_number=VALID_PR_NUM,
        head_sha=VALID_HEAD_SHA,
        verdict="COMMENT",
        summary="Automated Review Summary",
        comments=(valid_review_comment,),
        dry_run=False,
        created_at_mono=100.0,
    )


@pytest.fixture
def mock_review_result(base_action_plan, valid_review_comment) -> PRReviewResult:
    return PRReviewResult(
        owner=VALID_OWNER,
        repo=VALID_REPO,
        pr_number=VALID_PR_NUM,
        head_sha=VALID_HEAD_SHA,
        action_plan=base_action_plan,
        preview_text="PROPOSED GITHUB REVIEW ACTION PLAN",
        report={"summary": {"total_issues": 1}},
        review_comments=(valid_review_comment,),
        changed_files=("sample_project/app.py",),
        workflow_state={},
        success=True,
        summary="Automated Review Summary",
    )


# ---------------------------------------------------------------------------
# 1. Default CLI execution performs no writes
# ---------------------------------------------------------------------------


class TestDefaultCLIPerformsNoWrites:
    def test_default_cli_read_only(self, mock_gh_client, capsys):
        """Default CLI run without --publish performs zero write requests."""
        with patch("main.run_pr_review_workflow") as mock_review_wf:
            mock_res = MagicMock(spec=PRReviewResult)
            mock_res.success = True
            mock_res.action_plan = MagicMock(verdict="COMMENT")
            mock_res.review_comments = ()
            mock_review_wf.return_value = mock_res

            test_args = ["main.py", "--pr", f"{VALID_OWNER}/{VALID_REPO}#{VALID_PR_NUM}"]
            with patch.object(sys, "argv", test_args):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0

            # Must NOT call create_pull_request_review
            mock_gh_client.create_pull_request_review.assert_not_called()
            captured = capsys.readouterr().out
            assert "Read-Only Preview Mode" in captured


# ---------------------------------------------------------------------------
# 2. --publish --dry-run performs no writes and clearly displays payload
# ---------------------------------------------------------------------------


class TestPublishDryRunPerformsNoWrites:
    def test_publish_dry_run_displays_payload_zero_writes(
        self, mock_gh_client, mock_review_result, capsys
    ):
        """--publish --dry-run displays payload and performs zero write calls."""
        with patch("github_review_workflow.run_pr_review_workflow", return_value=mock_review_result):
            exit_code = run_pr_pipeline(
                pr_target=f"{VALID_OWNER}/{VALID_REPO}#{VALID_PR_NUM}",
                publish=True,
                dry_run=True,
                client=mock_gh_client,
            )
            assert exit_code == 1  # 1 because there is a review comment
            mock_gh_client.create_pull_request_review.assert_not_called()

            captured = capsys.readouterr().out
            assert "PROPOSED GITHUB REVIEW PAYLOAD (DRY-RUN — NO WRITES)" in captured
            assert "commit_id" in captured
            assert VALID_HEAD_SHA.lower() in captured
            assert "[DRY RUN] Zero write requests performed" in captured


# ---------------------------------------------------------------------------
# 3. Publishing without an interactive TTY is rejected
# ---------------------------------------------------------------------------


class TestNonInteractiveRejection:
    def test_non_interactive_fails_closed(self, mock_gh_client, mock_review_result, capsys):
        """When not interactive, publishing fails closed without writes."""
        with patch("github_review_workflow.run_pr_review_workflow", return_value=mock_review_result):
            exit_code = run_pr_pipeline(
                pr_target=f"{VALID_OWNER}/{VALID_REPO}#{VALID_PR_NUM}",
                publish=True,
                dry_run=False,
                interactive=False,
                client=mock_gh_client,
            )
            assert exit_code == 2
            mock_gh_client.create_pull_request_review.assert_not_called()

            captured = capsys.readouterr().out
            assert "PUBLISHING REJECTED (NON-INTERACTIVE)" in captured


# ---------------------------------------------------------------------------
# 4. User rejection, empty input, EOF, or keyboard interruption
# ---------------------------------------------------------------------------


class TestUserRejectionAndAbortion:
    @pytest.mark.parametrize(
        "user_input",
        ["n", "no", "N", "NO", "", "   ", "skip", "random"],
    )
    def test_rejection_or_empty_input_performs_no_writes(
        self, user_input, mock_gh_client, mock_review_result
    ):
        """Any answer other than 'y'/'yes', including empty input, cancels write."""
        with patch("github_review_workflow.run_pr_review_workflow", return_value=mock_review_result):
            exit_code = run_pr_pipeline(
                pr_target=f"{VALID_OWNER}/{VALID_REPO}#{VALID_PR_NUM}",
                publish=True,
                dry_run=False,
                interactive=True,
                input_func=lambda _: user_input,
                client=mock_gh_client,
            )
            assert exit_code == 1
            mock_gh_client.create_pull_request_review.assert_not_called()

    def test_eof_performs_no_writes(self, mock_gh_client, mock_review_result):
        """EOF during prompt cancels publishing with zero writes."""
        def raise_eof(_):
            raise EOFError

        with patch("github_review_workflow.run_pr_review_workflow", return_value=mock_review_result):
            exit_code = run_pr_pipeline(
                pr_target=f"{VALID_OWNER}/{VALID_REPO}#{VALID_PR_NUM}",
                publish=True,
                dry_run=False,
                interactive=True,
                input_func=raise_eof,
                client=mock_gh_client,
            )
            assert exit_code == 1
            mock_gh_client.create_pull_request_review.assert_not_called()

    def test_keyboard_interrupt_performs_no_writes(self, mock_gh_client, mock_review_result):
        """KeyboardInterrupt during prompt cancels publishing with zero writes."""
        def raise_ki(_):
            raise KeyboardInterrupt

        with patch("github_review_workflow.run_pr_review_workflow", return_value=mock_review_result):
            exit_code = run_pr_pipeline(
                pr_target=f"{VALID_OWNER}/{VALID_REPO}#{VALID_PR_NUM}",
                publish=True,
                dry_run=False,
                interactive=True,
                input_func=raise_ki,
                client=mock_gh_client,
            )
            assert exit_code == 1
            mock_gh_client.create_pull_request_review.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Expired approval and stale plan perform no writes
# ---------------------------------------------------------------------------


class TestStalenessAndExpiryDefenses:
    def test_expired_approval_performs_no_writes(
        self, base_action_plan, mock_gh_client
    ):
        """If approval window expires (> 600s), revalidation rejects with zero writes."""
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=base_action_plan,
            source="interactive",
            created_at_mono=100.0,
            approved_at_mono=110.0,
            expires_at_mono=710.0,
            approval_ttl=600.0,
            clock=lambda: 800.0,
        )
        with pytest.raises(ApprovalExpiredError, match="Approval has expired"):
            publish_approved_review(
                approval_result=approval,
                current_plan=base_action_plan,
                client=mock_gh_client,
                clock=lambda: 800.0,
            )
        mock_gh_client.create_pull_request_review.assert_not_called()

    def test_stale_plan_before_revalidation_performs_no_writes(
        self, base_action_plan, mock_gh_client
    ):
        """If plan age exceeds max_plan_age_seconds, revalidation rejects with zero writes."""
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=base_action_plan,
            source="interactive",
            created_at_mono=100.0,
            approved_at_mono=110.0,
            expires_at_mono=710.0,
            approval_ttl=600.0,
            clock=lambda: 200.0,
        )
        with pytest.raises(ApprovalExpiredError, match="Action plan is stale"):
            publish_approved_review(
                approval_result=approval,
                current_plan=base_action_plan,
                client=mock_gh_client,
                clock=lambda: 200.0,
                max_plan_age_seconds=50.0,  # plan age is 100s > 50s
            )
        mock_gh_client.create_pull_request_review.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Tampered plan or fingerprint mismatch performs no writes
# ---------------------------------------------------------------------------


class TestAntiTamperingFingerprintDefenses:
    def test_modified_summary_fingerprint_mismatch(
        self, base_action_plan, mock_gh_client
    ):
        """Modifying plan summary after approval fails closed with zero writes."""
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=base_action_plan,
            source="interactive",
            created_at_mono=100.0,
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )
        tampered_plan = ActionPlan(
            owner=base_action_plan.owner,
            repo=base_action_plan.repo,
            pr_number=base_action_plan.pr_number,
            head_sha=base_action_plan.head_sha,
            verdict=base_action_plan.verdict,
            summary="Tampered Summary!",
            comments=base_action_plan.comments,
            created_at_mono=100.0,
        )
        with pytest.raises(PlanFingerprintMismatchError):
            publish_approved_review(
                approval_result=approval,
                current_plan=tampered_plan,
                client=mock_gh_client,
                clock=lambda: 200.0,
            )
        mock_gh_client.create_pull_request_review.assert_not_called()

    def test_modified_verdict_fingerprint_mismatch(
        self, base_action_plan, mock_gh_client
    ):
        """Modifying plan verdict after approval fails closed with zero writes."""
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=base_action_plan,
            source="interactive",
            created_at_mono=100.0,
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )
        tampered_plan = ActionPlan(
            owner=base_action_plan.owner,
            repo=base_action_plan.repo,
            pr_number=base_action_plan.pr_number,
            head_sha=base_action_plan.head_sha,
            verdict="APPROVE",
            summary=base_action_plan.summary,
            comments=base_action_plan.comments,
            created_at_mono=100.0,
        )
        with pytest.raises(PlanFingerprintMismatchError):
            publish_approved_review(
                approval_result=approval,
                current_plan=tampered_plan,
                client=mock_gh_client,
                clock=lambda: 200.0,
            )
        mock_gh_client.create_pull_request_review.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Head SHA, repository identity, or PR number mismatch performs no writes
# ---------------------------------------------------------------------------


class TestTargetIdentityAndShaMismatchDefenses:
    def test_head_sha_change_performs_no_writes(
        self, base_action_plan, mock_gh_client, mock_pr_meta
    ):
        """If PR author pushes a new commit before publication, reject with zero writes."""
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=base_action_plan,
            source="interactive",
            created_at_mono=100.0,
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )
        # Live PR metadata reports a different SHA
        from dataclasses import replace
        mock_gh_client.get_pr_metadata.return_value = replace(
            mock_pr_meta, head_sha=DIFFERENT_HEAD_SHA
        )
        from approval_gate import HeadShaMismatchError

        with pytest.raises(HeadShaMismatchError, match="Head SHA mismatch"):
            publish_approved_review(
                approval_result=approval,
                current_plan=base_action_plan,
                client=mock_gh_client,
                clock=lambda: 200.0,
            )
        mock_gh_client.create_pull_request_review.assert_not_called()

    def test_repo_owner_mismatch_performs_no_writes(
        self, base_action_plan, mock_gh_client, mock_pr_meta
    ):
        """Owner mismatch fails closed with zero writes."""
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=base_action_plan,
            source="interactive",
            created_at_mono=100.0,
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )
        from dataclasses import replace
        mock_gh_client.get_pr_metadata.return_value = replace(
            mock_pr_meta, owner="attacker"
        )
        with pytest.raises(PrePublicationValidationError, match="Repository owner mismatch"):
            publish_approved_review(
                approval_result=approval,
                current_plan=base_action_plan,
                client=mock_gh_client,
                clock=lambda: 200.0,
            )
        mock_gh_client.create_pull_request_review.assert_not_called()

    def test_repo_name_mismatch_performs_no_writes(
        self, base_action_plan, mock_gh_client, mock_pr_meta
    ):
        """Repository name mismatch fails closed with zero writes."""
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=base_action_plan,
            source="interactive",
            created_at_mono=100.0,
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )
        from dataclasses import replace
        mock_gh_client.get_pr_metadata.return_value = replace(
            mock_pr_meta, repo="other-repo"
        )
        with pytest.raises(PrePublicationValidationError, match="Repository name mismatch"):
            publish_approved_review(
                approval_result=approval,
                current_plan=base_action_plan,
                client=mock_gh_client,
                clock=lambda: 200.0,
            )
        mock_gh_client.create_pull_request_review.assert_not_called()

    def test_pr_number_mismatch_performs_no_writes(
        self, base_action_plan, mock_gh_client, mock_pr_meta
    ):
        """PR number mismatch fails closed with zero writes."""
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=base_action_plan,
            source="interactive",
            created_at_mono=100.0,
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )
        from dataclasses import replace
        mock_gh_client.get_pr_metadata.return_value = replace(
            mock_pr_meta, pr_number=999
        )
        with pytest.raises(PrePublicationValidationError, match="PR number mismatch"):
            publish_approved_review(
                approval_result=approval,
                current_plan=base_action_plan,
                client=mock_gh_client,
                clock=lambda: 200.0,
            )
        mock_gh_client.create_pull_request_review.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Invalid comment anchoring performs no writes
# ---------------------------------------------------------------------------


class TestCommentAnchoringFailureZeroWrites:
    def test_comment_on_unmodified_line_rejected(self, mock_gh_client):
        """Comments outside diff hunks are rejected during pre-publication revalidation."""
        bad_comment_plan = ActionPlan(
            owner=VALID_OWNER,
            repo=VALID_REPO,
            pr_number=VALID_PR_NUM,
            head_sha=VALID_HEAD_SHA,
            verdict="COMMENT",
            summary="Review Summary",
            comments=(ReviewComment(path="sample_project/app.py", line=999, body="Invalid line"),),
            created_at_mono=100.0,
        )
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=bad_comment_plan,
            source="interactive",
            created_at_mono=100.0,
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )
        with pytest.raises(CommentAnchoringError, match="not a valid editable line"):
            publish_approved_review(
                approval_result=approval,
                current_plan=bad_comment_plan,
                client=mock_gh_client,
                clock=lambda: 200.0,
            )
        mock_gh_client.create_pull_request_review.assert_not_called()


# ---------------------------------------------------------------------------
# 9. A valid approval and successful revalidation result in exactly one POST
# ---------------------------------------------------------------------------


class TestSinglePostOnValidApproval:
    def test_valid_approval_makes_exactly_one_post(
        self, base_action_plan, mock_gh_client
    ):
        """Valid approval and revalidation issue exactly one review POST."""
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=base_action_plan,
            source="interactive",
            created_at_mono=100.0,
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )
        res = publish_approved_review(
            approval_result=approval,
            current_plan=base_action_plan,
            client=mock_gh_client,
            clock=lambda: 200.0,
            publication_marker="test-marker-123",
        )
        assert res.success is True
        assert res.review_id == 101
        assert mock_gh_client.create_pull_request_review.call_count == 1

        call_args = mock_gh_client.create_pull_request_review.call_args[1]
        assert call_args["owner"] == VALID_OWNER
        assert call_args["repo"] == VALID_REPO
        assert call_args["pr_number"] == VALID_PR_NUM
        assert call_args["commit_sha"] == VALID_HEAD_SHA.lower()
        assert call_args["event"] == "COMMENT"
        assert "<!-- agy-review-marker: test-marker-123 -->" in call_args["body"]


# ---------------------------------------------------------------------------
# 10. HTTP validation failure is reported correctly
# ---------------------------------------------------------------------------


class TestHTTPValidationFailureHandling:
    def test_http_422_validation_error_reported(
        self, base_action_plan, mock_gh_client, mock_review_result, capsys
    ):
        """HTTP 422 Unprocessable Entity is caught, reported, and exits with code 2."""
        mock_gh_client.create_pull_request_review.side_effect = GitHubValidationError(
            "Review comments are invalid.",
            errors=[{"resource": "PullRequestReview", "code": "invalid"}],
        )

        with patch("github_review_workflow.run_pr_review_workflow", return_value=mock_review_result):
            exit_code = run_pr_pipeline(
                pr_target=f"{VALID_OWNER}/{VALID_REPO}#{VALID_PR_NUM}",
                publish=True,
                dry_run=False,
                interactive=True,
                input_func=lambda _: "y",
                client=mock_gh_client,
            )
            assert exit_code == 2
            assert mock_gh_client.create_pull_request_review.call_count == 1

            captured = capsys.readouterr().out
            assert "HTTP 422 VALIDATION ERROR" in captured


# ---------------------------------------------------------------------------
# 11. POST timeout or connection loss is treated as uncertain, with no POST retry
# ---------------------------------------------------------------------------


class TestUncertainPostHandlingAndReconciliation:
    def test_post_timeout_uncertain_and_no_retry(
        self, base_action_plan, mock_gh_client, mock_review_result, capsys
    ):
        """Timeout during POST raises GitHubUncertainWriteError; POST is NEVER retried."""
        mock_gh_client.create_pull_request_review.side_effect = GitHubUncertainWriteError(
            "Connection timed out waiting for server response."
        )
        mock_gh_client.get_pr_reviews.return_value = []  # Reconciliation finds nothing

        with patch("github_review_workflow.run_pr_review_workflow", return_value=mock_review_result):
            exit_code = run_pr_pipeline(
                pr_target=f"{VALID_OWNER}/{VALID_REPO}#{VALID_PR_NUM}",
                publish=True,
                dry_run=False,
                interactive=True,
                input_func=lambda _: "y",
                client=mock_gh_client,
            )
            assert exit_code == 2
            # Exactly one POST attempt was made — ZERO retries!
            assert mock_gh_client.create_pull_request_review.call_count == 1

            captured = capsys.readouterr().out
            assert "PUBLISHING UNCERTAIN (TIMEOUT/CONNECTION DROP)" in captured
            assert "No retry performed" in captured

    def test_reconciliation_succeeds_when_review_found_on_github(
        self, base_action_plan, mock_gh_client
    ):
        """When POST times out but review exists on GitHub, reconciliation confirms success."""
        marker = "reconcile-test-marker"
        mock_gh_client.create_pull_request_review.side_effect = GitHubUncertainWriteError(
            "Timeout during POST"
        )
        # Mock get_pr_reviews returning the published review with the marker
        mock_gh_client.get_pr_reviews.return_value = [
            {
                "id": 999,
                "html_url": "https://github.com/octocat/Hello-World/pull/42#pullrequestreview-999",
                "body": f"Review body\n\n<!-- agy-review-marker: {marker} -->",
                "commit_id": VALID_HEAD_SHA,
                "user": {"login": "bot-user"},
            }
        ]
        approval = ApprovalResult(
            approved=True,
            dry_run=False,
            plan=base_action_plan,
            source="interactive",
            created_at_mono=100.0,
            approved_at_mono=110.0,
            expires_at_mono=710.0,
        )

        res = publish_approved_review(
            approval_result=approval,
            current_plan=base_action_plan,
            client=mock_gh_client,
            clock=lambda: 200.0,
            publication_marker=marker,
        )
        assert res.success is True
        assert res.review_id == 999
        assert res.html_url == "https://github.com/octocat/Hello-World/pull/42#pullrequestreview-999"
        # Zero retries: create_pull_request_review called only once
        assert mock_gh_client.create_pull_request_review.call_count == 1

    def test_reconciliation_verifies_bot_author_and_commit_sha(self, mock_gh_client):
        """Reconciliation rejects reviews if commit SHA or bot author does not match."""
        from github_review_workflow import reconcile_published_review

        marker = "unique-test-marker"
        # 1. Candidate with different commit SHA is rejected
        mock_gh_client.get_pr_reviews.return_value = [
            {
                "id": 1,
                "commit_id": "different_commit_sha",
                "user": {"login": "my-bot"},
                "body": f"Review <!-- agy-review-marker: {marker} -->",
            }
        ]
        reconciled = reconcile_published_review(
            client=mock_gh_client,
            owner="octocat",
            repo="Hello-World",
            pr_number=42,
            publication_marker=marker,
            expected_head_sha=VALID_HEAD_SHA,
            expected_author="my-bot",
        )
        assert reconciled is None

        # 2. Candidate with different author is rejected
        mock_gh_client.get_pr_reviews.return_value = [
            {
                "id": 2,
                "commit_id": VALID_HEAD_SHA,
                "user": {"login": "impostor"},
                "body": f"Review <!-- agy-review-marker: {marker} -->",
            }
        ]
        reconciled = reconcile_published_review(
            client=mock_gh_client,
            owner="octocat",
            repo="Hello-World",
            pr_number=42,
            publication_marker=marker,
            expected_head_sha=VALID_HEAD_SHA,
            expected_author="my-bot",
        )
        assert reconciled is None

        # 3. Matching candidate with correct marker, commit SHA, and author is accepted
        mock_gh_client.get_pr_reviews.return_value = [
            {
                "id": 3,
                "commit_id": VALID_HEAD_SHA,
                "user": {"login": "my-bot"},
                "body": f"Review <!-- agy-review-marker: {marker} -->",
            }
        ]
        reconciled = reconcile_published_review(
            client=mock_gh_client,
            owner="octocat",
            repo="Hello-World",
            pr_number=42,
            publication_marker=marker,
            expected_head_sha=VALID_HEAD_SHA,
            expected_author="my-bot",
        )
        assert reconciled is not None
        assert reconciled["id"] == 3


# ---------------------------------------------------------------------------
# 12. Successful POST reports success only after valid server response
# ---------------------------------------------------------------------------


class TestSuccessfulPostReporting:
    def test_successful_post_reports_success_details(
        self, mock_gh_client, mock_review_result, capsys
    ):
        """Successful POST announces review ID and review URL."""
        with patch("github_review_workflow.run_pr_review_workflow", return_value=mock_review_result):
            exit_code = run_pr_pipeline(
                pr_target=f"{VALID_OWNER}/{VALID_REPO}#{VALID_PR_NUM}",
                publish=True,
                dry_run=False,
                interactive=True,
                input_func=lambda _: "y",
                client=mock_gh_client,
            )
            assert exit_code == 1  # 1 because review comment is present
            assert mock_gh_client.create_pull_request_review.call_count == 1

            captured = capsys.readouterr().out
            assert "[SUCCESS] Review #101 successfully published" in captured
            assert "pullrequestreview-101" in captured


# ---------------------------------------------------------------------------
# 13. Existing local and read-only PR modes continue to work
# ---------------------------------------------------------------------------


class TestExistingModesContinueToWork:
    def test_local_analysis_mode_still_works(self, tmp_path, capsys):
        """Local file analysis executes without regression."""
        p = tmp_path / "script.py"
        p.write_text("a = 10\n", encoding="utf-8")

        with patch("main.run_ruff") as mock_ruff, patch("main.run_bandit") as mock_bandit:
            mock_ruff.return_value = ToolResult("Ruff", str(p), success=True, issues_found=False, exit_code=0)
            mock_bandit.return_value = ToolResult("Bandit", str(p), success=True, issues_found=False, exit_code=0)

            exit_code = run_pipeline(str(p), enable_ai=False)
            assert exit_code == 0
            captured = capsys.readouterr().out
            assert "PIPELINE RESULT: [PASSED]" in captured

    def test_read_only_pr_mode_still_works(self, mock_gh_client, capsys):
        """Read-only PR preview mode executes without write calls."""
        with patch("main.run_pr_review_workflow") as mock_rw:
            mock_rw.return_value = MagicMock(
                success=True,
                action_plan=MagicMock(verdict="COMMENT"),
                review_comments=(),
            )
            exit_code = run_pr_pipeline(
                pr_target="octocat/Hello-World#42",
                publish=False,
                client=mock_gh_client,
            )
            assert exit_code == 0
            mock_gh_client.create_pull_request_review.assert_not_called()
            captured = capsys.readouterr().out
            assert "Read-Only Pull Request Review & Approval Preview" in captured


# ---------------------------------------------------------------------------
# 14. No code path automatically returns APPROVE
# ---------------------------------------------------------------------------


class TestNoAutomaticApprove:
    def test_determine_review_verdict_never_approves(self):
        """determine_review_verdict never outputs APPROVE under any input."""
        # 1. Clean PR with 0 issues
        assert determine_review_verdict([], []) == "COMMENT"

        # 2. PR with warnings / medium issues
        ordinary_issues = [
            {"tool": "ruff", "code": "E501", "severity": "WARNING", "confidence": "HIGH"},
            {"tool": "bandit", "code": "B101", "severity": "LOW", "confidence": "HIGH"},
        ]
        assert determine_review_verdict(ordinary_issues, []) == "COMMENT"

        # 3. PR with critical issue
        crit_issues = [
            {"tool": "bandit", "code": "B608", "severity": "HIGH", "confidence": "HIGH"}
        ]
        assert determine_review_verdict(crit_issues, []) == "REQUEST_CHANGES"

        # 4. PR with agent error
        assert determine_review_verdict([], ["Fatal runner error"]) == "REQUEST_CHANGES"

    def test_generated_plan_has_no_approve_verdict(
        self, mock_gh_client, mock_review_result
    ):
        """End-to-end workflow action plan verdict is COMMENT, never APPROVE."""
        with patch("github_review_workflow.run_pr_review_workflow", return_value=mock_review_result):
            res = run_pr_publishing_workflow(
                repo_or_url=f"{VALID_OWNER}/{VALID_REPO}",
                pr_number=VALID_PR_NUM,
                dry_run=True,
                client=mock_gh_client,
            )
            assert res.action_plan.verdict != "APPROVE"
            assert res.action_plan.verdict in ("COMMENT", "REQUEST_CHANGES")


# ---------------------------------------------------------------------------
# 15. CLI Argument Parsing Tests
# ---------------------------------------------------------------------------


class TestCLIArgumentParsing:
    def test_publish_requires_pr_flag(self, capsys):
        """Running with --publish without --pr fails with an error and exit code 2."""
        test_args = ["main.py", "--publish"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2
            captured = capsys.readouterr().out
            assert "--publish requires a GitHub PR target" in captured


# ---------------------------------------------------------------------------
# 16. Redirect and Credential Protection Tests
# ---------------------------------------------------------------------------


class TestRedirectDefense:
    def test_get_pr_metadata_rejects_3xx_redirect(self):
        """get_pr_metadata blocks redirects and passes allow_redirects=False."""
        import requests

        from github_client import GitHubClient, GitHubClientError

        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 302
        mock_resp.headers = {"Location": "https://attacker.com/leak"}
        mock_resp.text = "Found"

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_resp

        client = GitHubClient(token="secret-token-12345", session=mock_session)
        with pytest.raises(GitHubClientError, match="Unexpected redirect response"):
            client.get_pr_metadata("owner", "repo", 1)

        _, kwargs = mock_session.get.call_args
        assert kwargs["allow_redirects"] is False

    def test_get_pr_diff_rejects_3xx_redirect(self):
        """get_pr_diff blocks redirects and passes allow_redirects=False."""
        import requests

        from github_client import GitHubClient, GitHubClientError

        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 301
        mock_resp.headers = {"Location": "https://attacker.com/leak"}
        mock_resp.text = "Moved Permanently"

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value = mock_resp

        client = GitHubClient(token="secret-token-12345", session=mock_session)
        with pytest.raises(GitHubClientError, match="Unexpected redirect response"):
            client.get_pr_diff("owner", "repo", 1)

        _, kwargs = mock_session.get.call_args
        assert kwargs["allow_redirects"] is False

    def test_download_commit_archive_rejects_untrusted_redirect(self, tmp_path):
        """download_commit_archive validates redirect origin and blocks non-github.com."""
        import requests

        from github_client import GitHubClient, GitHubClientError

        mock_prev_resp = MagicMock(spec=requests.Response)
        mock_prev_resp.status_code = 302
        mock_prev_resp.headers = {"Location": "https://attacker.com/malicious.zip"}

        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.history = [mock_prev_resp]
        mock_resp.iter_content.return_value = [b"PK\x03\x04zipdata"]

        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.return_value.__enter__.return_value = mock_resp

        dest = tmp_path / "test.zip"
        client = GitHubClient(token="secret-token-12345", session=mock_session)
        with pytest.raises(GitHubClientError, match="Untrusted redirect origin"):
            client.download_commit_archive("owner", "repo", "0123456789abcdef0123456789abcdef01234567", dest_path=dest)
