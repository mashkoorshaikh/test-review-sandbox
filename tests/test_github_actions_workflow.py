"""
tests/test_github_actions_workflow.py - Tests for GitHub Actions Workflow Configuration and Safety.

Verifies:
  1. Workflow file exists at .github/workflows/pr-review.yml.
  2. Workflow is syntactically valid YAML.
  3. Triggers use strictly 'pull_request' on [opened, synchronize, reopened].
  4. 'pull_request_target' is strictly forbidden (prevents secret exfiltration from forked PRs).
  5. Workflow explicitly defines least-privilege permissions (contents: read, pull-requests: read).
  6. No write permissions exist in the workflow.
  7. Checkout step explicitly specifies 'persist-credentials: false'.
  8. Execution step invokes read-only review without --publish.
  9. Execution step uses --no-sandbox to skip dynamic execution safely on CI host.
  10. Step summary output includes PR number, head SHA, verdict, findings, and advisory notice.
  11. Step summary content is safely written and token-redacted.
  12. Missing or unwritable GITHUB_STEP_SUMMARY does not crash the CLI.
  13. Non-interactive publishing attempts fail closed.
  14. Forked PR workflows work safely when OPENAI_API_KEY is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from approval_gate import ActionPlan, ReviewComment
from github_review_workflow import PRReviewResult
from main import (
    format_github_step_summary,
    run_pr_pipeline,
    write_github_step_summary,
)

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "pr-review.yml"


@pytest.fixture(scope="module")
def workflow_yaml_text() -> str:
    assert WORKFLOW_PATH.is_file(), f"Workflow file not found at {WORKFLOW_PATH}"
    return WORKFLOW_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow_data(workflow_yaml_text: str) -> dict[str, Any]:
    data = yaml.safe_load(workflow_yaml_text)
    assert isinstance(data, dict), f"Expected dict from YAML, got {type(data).__name__}"
    return data


class TestGitHubActionsWorkflowConfiguration:
    """Verifies that the GitHub Actions workflow conforms to strict security and least-privilege standards."""

    def test_workflow_file_exists(self):
        assert WORKFLOW_PATH.is_file(), f"Workflow file must exist at {WORKFLOW_PATH}"

    def test_workflow_valid_yaml(self, workflow_data):
        assert "name" in workflow_data, "Workflow must define a 'name'."
        assert "jobs" in workflow_data, "Workflow must define 'jobs'."
        trigger_section = workflow_data.get("on")
        if trigger_section is None and True in workflow_data:
            trigger_section = workflow_data[True]
        assert isinstance(trigger_section, dict), (
            "Workflow must define a valid mapping for the 'on' trigger specification."
        )

    def test_workflow_trigger_is_pull_request_only(self, workflow_data):
        on_triggers = workflow_data.get("on")
        if on_triggers is None and True in workflow_data:
            on_triggers = workflow_data[True]
        assert on_triggers is not None, "Workflow must define an 'on' trigger section."
        assert isinstance(on_triggers, dict), "Trigger section must be a dictionary."
        assert "pull_request" in on_triggers, "Workflow must be triggered by 'pull_request'."
        assert list(on_triggers.keys()) == ["pull_request"], (
            f"Workflow triggers must strictly be ['pull_request'], got: {list(on_triggers.keys())}"
        )

        # Strict security rule: pull_request_target must NEVER be used for PR reviews of untrusted code
        assert "pull_request_target" not in on_triggers, (
            "SECURITY VIOLATION: 'pull_request_target' must not be used because it grants "
            "privileged write access and secret access when evaluating untrusted fork PRs."
        )

        pr_config = on_triggers["pull_request"]
        assert isinstance(pr_config, dict), "pull_request trigger must be a dictionary with types."
        types = pr_config.get("types", [])
        assert isinstance(types, list)
        for expected_type in ("opened", "synchronize", "reopened"):
            assert expected_type in types, f"pull_request trigger types must include '{expected_type}'."

    def test_workflow_least_privilege_permissions(self, workflow_data):
        """Top-level permissions must be explicitly declared and strictly read-only."""
        permissions = workflow_data.get("permissions")
        assert permissions is not None, "Workflow must explicitly declare top-level permissions."
        assert isinstance(permissions, dict), "Permissions must be a dictionary."

        # Required read permissions
        assert permissions.get("contents") == "read", "Permission 'contents' must be 'read'."
        assert permissions.get("pull-requests") == "read", "Permission 'pull-requests' must be 'read'."

        # Prohibit any write permission across all declared permissions
        for perm_key, perm_val in permissions.items():
            assert perm_val != "write", (
                f"SECURITY VIOLATION: Permission '{perm_key}' has write access ('{perm_val}'). "
                "The workflow must remain strictly read-only."
            )

    def test_workflow_checkout_persists_no_credentials(self, workflow_data):
        """Checkout step must set persist-credentials: false to avoid leaving credentials on disk."""
        jobs = workflow_data.get("jobs", {})
        assert "review" in jobs, "Workflow must define a 'review' job."
        steps = jobs["review"].get("steps", [])

        checkout_steps = [
            s for s in steps
            if isinstance(s, dict) and "checkout" in str(s.get("uses", "")).lower()
        ]
        assert checkout_steps, "Workflow must contain an actions/checkout step."
        for step in checkout_steps:
            step_with = step.get("with", {})
            assert step_with.get("persist-credentials") is False, (
                "actions/checkout must specify 'persist-credentials: false' to prevent leaking "
                "ephemeral tokens into the runner git configuration."
            )

    def test_workflow_checkout_targets_trusted_base_branch(self, workflow_data):
        """
        Checkout step must check out the trusted base branch rather than untrusted PR code.

        This guarantees that workflow scripts (main.py) and dependencies (requirements.txt)
        are trusted code vetted by repository maintainers, preventing untrusted PR code from
        executing with secrets on the runner host.
        """
        jobs = workflow_data.get("jobs", {})
        assert "review" in jobs, "Workflow must define a 'review' job."
        steps = jobs["review"].get("steps", [])

        checkout_steps = [
            s for s in steps
            if isinstance(s, dict) and "checkout" in str(s.get("uses", "")).lower()
        ]
        assert checkout_steps, "Workflow must contain an actions/checkout step."
        for step in checkout_steps:
            step_with = step.get("with", {})
            ref_val = step_with.get("ref", "")
            assert "github.base_ref" in ref_val, (
                f"actions/checkout must specify base ref to prevent executing untrusted PR code, got: {ref_val}"
            )

    def test_workflow_step_invokes_read_only_pipeline(self, workflow_data):
        """Review execution step must run read-only preview without --publish."""
        jobs = workflow_data.get("jobs", {})
        steps = jobs["review"].get("steps", [])

        review_steps = [
            s for s in steps
            if isinstance(s, dict) and "main.py" in str(s.get("run", ""))
        ]
        assert review_steps, "Workflow must contain a step executing 'python main.py'."

        for step in review_steps:
            run_cmd = step.get("run", "")
            # Must reference PR target
            assert "--pr" in run_cmd, "Command must specify --pr target."
            # Must use --no-sandbox to avoid dynamic test execution on CI host
            assert "--no-sandbox" in run_cmd, (
                "Command must specify --no-sandbox in CI to ensure untrusted dynamic code execution "
                "is skipped, while static analysis (Ruff & Bandit) remains active."
            )
            # Must NOT publish in CI
            assert "--publish" not in run_cmd, (
                "SECURITY VIOLATION: Workflow step must not specify '--publish'. "
                "All CI reviews must remain strictly read-only previews."
            )

    def test_workflow_avoids_inline_script_injection_in_run(self, workflow_data):
        """Review execution step must pass PR context via environment variables rather than inline expressions in run."""
        jobs = workflow_data.get("jobs", {})
        steps = jobs["review"].get("steps", [])
        review_steps = [
            s for s in steps
            if isinstance(s, dict) and "main.py" in str(s.get("run", ""))
        ]
        assert review_steps
        for step in review_steps:
            run_cmd = step.get("run", "")
            assert "${{" not in run_cmd, (
                "SECURITY RISK: Do not interpolate GitHub context expressions '${{ ... }}' directly "
                "into the 'run:' shell script block. Pass them via 'env:' to prevent shell injection."
            )
            env = step.get("env", {})
            assert "PR_NUMBER" in env
            assert "REPO_NAME" in env

    def test_workflow_concurrency_group_defined(self, workflow_data):
        """Workflow should define concurrency to cancel outdated in-progress runs."""
        concurrency = workflow_data.get("concurrency")
        assert concurrency is not None, "Workflow should define a concurrency group."
        assert isinstance(concurrency, dict)
        assert concurrency.get("cancel-in-progress") is True, "concurrency must specify cancel-in-progress: true."

    def test_workflow_is_secret_free(self, workflow_data):
        """
        Workflow steps must NOT receive custom repository secrets (e.g. OPENAI_API_KEY).

        Because pull_request workflows run the workflow definition from the PR branch,
        exposing repository secrets would allow untrusted PR authors to modify the workflow
        file and exfiltrate secrets. Only the platform-provided read-only GITHUB_TOKEN is permitted.
        """
        jobs = workflow_data.get("jobs", {})
        review_job = jobs.get("review", {})
        steps = review_job.get("steps", [])

        for step in steps:
            env = step.get("env", {})
            if not isinstance(env, dict):
                continue
            # Ensure OPENAI_API_KEY is NOT passed
            assert "OPENAI_API_KEY" not in env, (
                "SECURITY VIOLATION: OPENAI_API_KEY must not be exposed to pull_request workflows. "
                "CI static reviews must run secret-free in deterministic mode."
            )
            # Ensure no custom repository secrets are passed
            for env_var, env_val in env.items():
                if env_var == "GITHUB_TOKEN":
                    continue
                assert "secrets." not in str(env_val), (
                    f"SECURITY VIOLATION: Step exposed secret '{env_val}' to environment variable '{env_var}'. "
                    "PR review workflow must remain completely secret-free."
                )


class TestGitHubActionsStepSummary:
    """Verifies generation, content, token redaction, and resilience of GITHUB_STEP_SUMMARY."""

    def test_format_github_step_summary_direct(self):
        """Verifies structure and formatting of markdown step summary."""
        comment = ReviewComment(
            path="math_utils.py",
            line=15,
            body="Potential division by zero.",
            side="RIGHT",
        )
        plan = ActionPlan(
            owner="org",
            repo="repo",
            pr_number=42,
            head_sha="11223344556677889900aabbccddeeff11223344",
            verdict="REQUEST_CHANGES",
            summary="Review completed with 1 warning.",
            comments=(comment,),
            dry_run=True,
            created_at_mono=100.0,
        )
        mock_result = PRReviewResult(
            owner="org",
            repo="repo",
            pr_number=42,
            head_sha="11223344556677889900aabbccddeeff11223344",
            action_plan=plan,
            preview_text="preview",
            report={},
            review_comments=(comment,),
            changed_files=("math_utils.py",),
            workflow_state={},
            success=True,
            summary="Review completed with 1 warning.",
        )
        formatted = format_github_step_summary(mock_result)
        assert "## 🤖 Multi-Agent PR Review: #42" in formatted
        assert "math_utils.py" in formatted
        assert "Potential division by zero." in formatted
        assert "Advisory Notice:" in formatted
        assert "1122334455" in formatted

    def test_step_summary_includes_required_elements(self, tmp_path, monkeypatch):
        """Summary output must include PR number, head SHA, verdict, findings, and advisory notice."""
        summary_file = tmp_path / "step_summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

        comment = ReviewComment(
            path="app.py",
            line=42,
            body="Use safe json parsing instead of eval.",
            side="RIGHT",
        )
        plan = ActionPlan(
            owner="test-owner",
            repo="test-repo",
            pr_number=99,
            head_sha="0123456789abcdef0123456789abcdef01234567",
            verdict="REQUEST_CHANGES",
            summary="### Findings Summary\n- Total code issues found: 1\n",
            comments=(comment,),
            dry_run=True,
            created_at_mono=100.0,
        )
        mock_result = PRReviewResult(
            owner="test-owner",
            repo="test-repo",
            pr_number=99,
            head_sha="0123456789abcdef0123456789abcdef01234567",
            action_plan=plan,
            preview_text="PROPOSED REVIEW",
            report={"summary": {"total_issues": 1}},
            review_comments=(comment,),
            changed_files=("app.py",),
            workflow_state={},
            success=True,
            summary="### Findings Summary\n- Total code issues found: 1\n",
        )

        success = write_github_step_summary(mock_result)
        assert success is True
        assert summary_file.is_file()

        content = summary_file.read_text(encoding="utf-8")
        # 1. PR number
        assert "#99" in content
        # 2. Target head SHA
        assert "0123456789abcdef0123456789abcdef01234567" in content
        # 3. Review verdict
        assert "REQUEST_CHANGES" in content
        # 4. Findings & comments
        assert "Total code issues found: 1" in content
        assert "app.py" in content
        assert "Use safe json parsing" in content
        # 5. Advisory notice
        assert "Advisory Notice:" in content

    def test_step_summary_token_redaction(self, tmp_path):
        """Summary content must redact secret tokens before writing."""
        summary_file = tmp_path / "step_summary_redact.md"
        secret_token = "ghp_leaked_ci_secret_token_123456"

        comment = ReviewComment(
            path="secret.py",
            line=10,
            body=f"Found hardcoded token: {secret_token}",
            side="RIGHT",
        )
        plan = ActionPlan(
            owner="test-owner",
            repo="test-repo",
            pr_number=10,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            verdict="COMMENT",
            summary=f"Analysis flagged {secret_token}",
            comments=(comment,),
            dry_run=True,
            created_at_mono=100.0,
        )
        mock_result = PRReviewResult(
            owner="test-owner",
            repo="test-repo",
            pr_number=10,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            action_plan=plan,
            preview_text="preview",
            report={},
            review_comments=(comment,),
            changed_files=("secret.py",),
            workflow_state={},
            success=True,
            summary=f"Analysis flagged {secret_token}",
        )

        success = write_github_step_summary(
            mock_result,
            token=secret_token,
            step_summary_path=summary_file,
        )
        assert success is True
        content = summary_file.read_text(encoding="utf-8")

        # Secret must NEVER appear in the written file
        assert secret_token not in content
        assert "[REDACTED]" in content

    def test_missing_github_step_summary_does_not_crash(self, monkeypatch):
        """Missing or unset GITHUB_STEP_SUMMARY returns False and does not crash."""
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        plan = ActionPlan(
            owner="test-owner",
            repo="test-repo",
            pr_number=1,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            verdict="COMMENT",
            summary="Clean",
            comments=(),
            dry_run=True,
            created_at_mono=100.0,
        )
        mock_result = PRReviewResult(
            owner="test-owner",
            repo="test-repo",
            pr_number=1,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            action_plan=plan,
            preview_text="",
            report={},
            review_comments=(),
            changed_files=(),
            workflow_state={},
            success=True,
            summary="Clean",
        )

        assert write_github_step_summary(mock_result) is False

    def test_unwritable_github_step_summary_does_not_crash(self, tmp_path):
        """Unwritable or invalid path gracefully fails closed without crashing."""
        invalid_path = tmp_path / "non_existent_subdir" / "summary.md"
        plan = ActionPlan(
            owner="test-owner",
            repo="test-repo",
            pr_number=1,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            verdict="COMMENT",
            summary="Clean",
            comments=(),
            dry_run=True,
            created_at_mono=100.0,
        )
        mock_result = PRReviewResult(
            owner="test-owner",
            repo="test-repo",
            pr_number=1,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            action_plan=plan,
            preview_text="",
            report={},
            review_comments=(),
            changed_files=(),
            workflow_state={},
            success=True,
            summary="Clean",
        )

        # Passing a path where parent directory does not exist
        assert write_github_step_summary(mock_result, step_summary_path=invalid_path) is False

    def test_step_summary_markdown_table_escaping(self):
        """
        Verifies that PR-derived content (paths, comments) with pipes, backticks, HTML,
        and markdown links/images is escaped to prevent table or summary injection.
        """
        malicious_comment = ReviewComment(
            path="exploit|file`test.py",
            line=42,
            body="Inject | cell | split <script>alert(1)</script> and ![tracking](https://evil.com/pixel.png) [phish](https://evil.com)",
            side="RIGHT",
        )
        plan = ActionPlan(
            owner="attacker|org",
            repo="victim|repo",
            pr_number=1337,
            head_sha="deadbeef0123456789abcdef0123456789abcdef",
            verdict="REQUEST_CHANGES",
            summary="Found <script>evil</script> and ![embed](https://attacker.com/leak.png)",
            comments=(malicious_comment,),
            dry_run=True,
            created_at_mono=100.0,
        )
        mock_result = PRReviewResult(
            owner="attacker|org",
            repo="victim|repo",
            pr_number=1337,
            head_sha="deadbeef0123456789abcdef0123456789abcdef",
            action_plan=plan,
            preview_text="preview",
            report={},
            review_comments=(malicious_comment,),
            changed_files=("exploit|file`test.py",),
            workflow_state={},
            success=True,
            summary="Found <script>evil</script> and ![embed](https://attacker.com/leak.png)",
        )

        formatted = format_github_step_summary(mock_result)

        # 1. Backticks must be neutralized
        assert "`exploit|file`" not in formatted
        assert "exploit\\|file'test.py" in formatted

        # 2. Table row must have exactly 4 columns despite pipes in path and body
        comment_row = [line for line in formatted.splitlines() if "exploit" in line]
        assert len(comment_row) == 1, "Expected single comment row in markdown table"
        import re
        cells = re.split(r"(?<!\\)\|", comment_row[0].strip().strip("|"))
        assert len(cells) == 4, f"Table row must have exactly 4 columns despite pipes, got {len(cells)}: {cells}"

        # 3. HTML script tags must be neutralized
        assert "<script>" not in formatted

        # 4. Image tags / markdown embeds must not render as live image tags or unescaped embeds
        assert "![tracking]" not in formatted
        assert "![embed]" not in formatted


class TestCIWorkflowSafetyBehavior:
    """Verifies security behavior in CI: non-interactive rejection and forked PR handling."""

    def test_non_interactive_publishing_fails_closed(self, capsys):
        """In CI (non-interactive), --publish strictly fails closed and issues no writes."""
        mock_client = MagicMock()
        plan = ActionPlan(
            owner="test-owner",
            repo="test-repo",
            pr_number=5,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            verdict="COMMENT",
            summary="Review summary",
            comments=(),
            dry_run=False,
            created_at_mono=100.0,
        )
        mock_res = PRReviewResult(
            owner="test-owner",
            repo="test-repo",
            pr_number=5,
            head_sha="abcdef0123456789abcdef0123456789abcdef01",
            action_plan=plan,
            preview_text="preview",
            report={},
            review_comments=(),
            changed_files=(),
            workflow_state={},
            success=True,
            summary="Review summary",
        )
        with patch("github_review_workflow.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline(
                pr_target="test-owner/test-repo#5",
                publish=True,
                dry_run=False,
                interactive=False,  # Emulates non-interactive CI terminal
                client=mock_client,
            )

            assert exit_code == 2
            mock_client.create_pull_request_review.assert_not_called()

            captured = capsys.readouterr().out
            assert "PUBLISHING REJECTED (NON-INTERACTIVE)" in captured

    def test_forked_pr_workflow_without_openai_api_key(self, monkeypatch, capsys, tmp_path):
        """Forked PRs without OPENAI_API_KEY run deterministic scans without crashing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        summary_file = tmp_path / "ci_summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

        mock_client = MagicMock()
        with (
            patch("main.is_api_key_configured", return_value=False),
            patch("main.run_pr_review_workflow") as mock_rwf,
        ):
            plan = ActionPlan(
                owner="fork-owner",
                repo="upstream-repo",
                pr_number=12,
                head_sha="abcdef0123456789abcdef0123456789abcdef01",
                verdict="COMMENT",
                summary="Deterministic scan completed",
                comments=(),
                dry_run=True,
                created_at_mono=100.0,
            )
            mock_rwf.return_value = PRReviewResult(
                owner="fork-owner",
                repo="upstream-repo",
                pr_number=12,
                head_sha="abcdef0123456789abcdef0123456789abcdef01",
                action_plan=plan,
                preview_text="preview",
                report={},
                review_comments=(),
                changed_files=(),
                workflow_state={},
                success=True,
                summary="Deterministic scan completed",
            )

            exit_code = run_pr_pipeline(
                pr_target="fork-owner/upstream-repo#12",
                publish=False,
                client=mock_client,
            )

            assert exit_code == 0
            captured = capsys.readouterr().out
            assert "LLM credentials not configured" in captured
            assert "deterministic mode" in captured

            # Step summary was still generated
            assert summary_file.is_file()
            summary_content = summary_file.read_text(encoding="utf-8")
            assert "#12" in summary_content
            assert "Deterministic scan completed" in summary_content


class TestReadOnlyPRReviewExitSemantics:
    """
    Verifies that read-only PR review in CI returns 0 upon successful analysis,
    regardless of whether findings were detected or changes were requested,
    while actual execution/workflow failures strictly return 2.
    """

    def _make_mock_result(
        self,
        success: bool = True,
        verdict: str = "COMMENT",
        comments: tuple[ReviewComment, ...] = (),
        summary: str = "Analysis summary",
        tool_status: dict[str, bool] | None = None,
    ) -> PRReviewResult:
        plan = ActionPlan(
            owner="test-owner",
            repo="test-repo",
            pr_number=100,
            head_sha="0123456789abcdef0123456789abcdef01234567",
            verdict=verdict,
            summary=summary,
            comments=comments,
            dry_run=True,
            created_at_mono=100.0,
        )
        t_status = (
            tool_status
            if tool_status is not None
            else {"ruff": True, "bandit": True, "semantic": success}
        )
        return PRReviewResult(
            owner="test-owner",
            repo="test-repo",
            pr_number=100,
            head_sha="0123456789abcdef0123456789abcdef01234567",
            action_plan=plan,
            preview_text="preview",
            report={},
            review_comments=comments,
            changed_files=("app.py",),
            workflow_state={"tool_status": t_status},
            success=success,
            summary=summary,
            tool_status=t_status,
        )

    def test_read_only_pr_review_clean_returns_0(self):
        """Clean PR scan with no findings returns 0."""
        mock_res = self._make_mock_result(success=True, verdict="COMMENT", comments=())
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 0

    def test_read_only_pr_review_with_findings_returns_0(self):
        """Successful review reporting findings returns 0 (review task succeeded)."""
        mock_res = self._make_mock_result(
            success=True,
            verdict="COMMENT",
            summary="Found 3 lint warnings",
            comments=(),
        )
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 0

    def test_read_only_pr_review_with_request_changes_verdict_returns_0(self):
        """Review completing with REQUEST_CHANGES returns 0 (findings communicate via step summary)."""
        mock_res = self._make_mock_result(
            success=True,
            verdict="REQUEST_CHANGES",
            summary="Critical security vulnerability detected",
            comments=(),
        )
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 0

    def test_read_only_pr_review_with_comments_returns_0(self):
        """Review proposing inline comments returns 0."""
        comment = ReviewComment(
            path="app.py",
            line=12,
            body="Use safe_load instead of load",
            side="RIGHT",
        )
        mock_res = self._make_mock_result(
            success=True,
            verdict="REQUEST_CHANGES",
            comments=(comment,),
            summary="Proposed 1 comment",
        )
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 0

    def test_read_only_pr_review_execution_failure_returns_2(self):
        """Tool crash or fatal analysis failure (success=False) returns 2."""
        mock_res = self._make_mock_result(
            success=False,
            verdict="REQUEST_CHANGES",
            summary="Fatal tool crash",
        )
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 2

    def test_read_only_pr_review_exception_fails_safely_returns_2(self, capsys):
        """Exceptions in PR workflow return 2 and display safe error."""
        from github_review_workflow import PRReviewWorkflowError

        with patch(
            "main.run_pr_review_workflow",
            side_effect=PRReviewWorkflowError("Network connection reset"),
        ):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 2
            out = capsys.readouterr().out
            assert "PR WORKFLOW FAILED: Network connection reset" in out

    def test_step_summary_written_even_with_findings_and_verdict(self, tmp_path, monkeypatch):
        """Verifies Step Summary contains findings and verdict when exit code is 0."""
        summary_file = tmp_path.joinpath("ci_step_summary.md")
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

        comment = ReviewComment(
            path="math_utils.py",
            line=40,
            body="Division by zero guard required",
            side="RIGHT",
        )
        mock_res = self._make_mock_result(
            success=True,
            verdict="REQUEST_CHANGES",
            comments=(comment,),
            summary="### Findings\n- Division by zero at line 40\n",
        )
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 0
            assert summary_file.is_file()
            content = summary_file.read_text(encoding="utf-8")
            assert "REQUEST_CHANGES" in content
            assert "math_utils.py" in content
            assert "Division by zero guard required" in content

    def test_local_deterministic_pipeline_semantics_retained(self):
        """Local deterministic analysis retains its distinct exit semantics: 0=clean, 1=findings, 2=error."""
        from main import run_pipeline
        from tools import ToolResult

        clean_tool = ToolResult(
            tool_name="Ruff", target_path=".", success=True, issues_found=False, exit_code=0, issues=[]
        )
        error_tool = ToolResult(
            tool_name="Ruff", target_path=".", success=False, issues_found=False, exit_code=2, issues=[], error_message="Crash"
        )

        mock_issue = MagicMock()
        issue_tool = ToolResult(
            tool_name="Ruff", target_path=".", success=True, issues_found=True, exit_code=1, issues=[mock_issue]
        )

        # 1. Clean -> 0
        with (
            patch("main.run_ruff", return_value=clean_tool),
            patch("main.run_bandit", return_value=clean_tool),
            patch("main.run_semantic_analysis", return_value=clean_tool),
        ):
            assert run_pipeline(".", enable_ai=False) == 0

        # 2. Issues found -> 1
        with (
            patch("main.run_ruff", return_value=issue_tool),
            patch("main.run_bandit", return_value=clean_tool),
            patch("main.run_semantic_analysis", return_value=clean_tool),
        ):
            assert run_pipeline(".", enable_ai=False) == 1

        # 3. Execution error -> 2
        with (
            patch("main.run_ruff", return_value=error_tool),
            patch("main.run_bandit", return_value=clean_tool),
            patch("main.run_semantic_analysis", return_value=clean_tool),
        ):
            assert run_pipeline(".", enable_ai=False) == 2

    def test_failed_semantic_analyzer_returns_exit_code_2(self):
        """Regression test: failed semantic analyzer returning 'Target path does not exist' returns exit code 2."""
        mock_res = self._make_mock_result(
            success=False,
            verdict="REQUEST_CHANGES",
            summary="Target path '/nonexistent' does not exist.",
            tool_status={"ruff": True, "bandit": True, "semantic": False},
        )
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 2

    def test_failed_semantic_analyzer_structured_tool_status_returns_2(self):
        """Structured tool_status causes exit code 2 even if result.success flag is True."""
        mock_res = self._make_mock_result(
            success=True,
            verdict="REQUEST_CHANGES",
            summary="Target path 'missing.py' does not exist.",
            tool_status={"ruff": True, "bandit": True, "semantic": False},
        )
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 2

    def test_successful_semantic_analyzer_with_findings_returns_exit_code_0(self):
        """Regression test: successful semantic analyzer with AST findings returns 0 in read-only PR review."""
        comment = ReviewComment(
            path="math_utils.py",
            line=40,
            body="SEM001: Possible division by zero",
            side="RIGHT",
        )
        mock_res = self._make_mock_result(
            success=True,
            verdict="REQUEST_CHANGES",
            comments=(comment,),
            summary="Semantic findings detected",
            tool_status={"ruff": True, "bandit": True, "semantic": True},
        )
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 0

    def test_run_review_pipeline_with_failed_semantic_analyzer(self, tmp_path):
        """Integration test: semantic runner returning 'Target path does not exist' produces FAILED status and tool_status."""
        from tools import ToolResult
        from workflow import run_review_pipeline

        test_file = tmp_path.joinpath("test.py")
        test_file.write_text("x = 1\n", encoding="utf-8")

        mock_semantic = MagicMock(return_value=ToolResult(
            tool_name="SemanticAnalyzer",
            target_path=str(test_file),
            success=False,
            issues_found=False,
            exit_code=2,
            issues=[],
            error_message=f"Target path '{test_file}' does not exist.",
        ))

        state = run_review_pipeline(
            target_path=str(test_file),
            enable_ai=False,
            semantic_runner=mock_semantic,
        )

        assert state["tool_status"]["semantic"] is False
        assert state["final_review_report"]["summary"]["status"] == "FAILED"

    def test_successful_bandit_with_security_findings_returns_exit_code_0(self):
        """Regression test: successful Bandit scan with code findings returns 0 in read-only PR review (findings posted to summary)."""
        comment = ReviewComment(
            path="app.py",
            line=10,
            body="B101: assert used",
            side="RIGHT",
        )
        mock_res = self._make_mock_result(
            success=True,
            verdict="COMMENT",
            comments=(comment,),
            summary="Bandit security findings detected",
            tool_status={"ruff": True, "bandit": True, "semantic": True},
        )
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 0

    def test_failed_bandit_analyzer_returns_exit_code_2(self):
        """Regression test: failed Bandit execution (crash or exit code 2) returns exit code 2."""
        mock_res = self._make_mock_result(
            success=False,
            verdict="REQUEST_CHANGES",
            summary="Bandit failed with exit code 2",
            tool_status={"ruff": True, "bandit": False, "semantic": True},
        )
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 2

    def test_failed_bandit_analyzer_structured_tool_status_returns_2(self):
        """Structured tool_status bandit=False causes exit code 2 even if result.success is True."""
        mock_res = self._make_mock_result(
            success=True,
            verdict="REQUEST_CHANGES",
            summary="Bandit execution failed",
            tool_status={"ruff": True, "bandit": False, "semantic": True},
        )
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 2

    def test_failed_ruff_analyzer_returns_exit_code_2(self):
        """Regression test: failed Ruff execution returns exit code 2."""
        mock_res = self._make_mock_result(
            success=False,
            verdict="REQUEST_CHANGES",
            summary="Ruff failed with fatal error",
            tool_status={"ruff": False, "bandit": True, "semantic": True},
        )
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 2

    def test_failed_ruff_analyzer_structured_tool_status_returns_2(self):
        """Structured tool_status ruff=False causes exit code 2 even if result.success is True."""
        mock_res = self._make_mock_result(
            success=True,
            verdict="REQUEST_CHANGES",
            summary="Ruff syntax crash",
            tool_status={"ruff": False, "bandit": True, "semantic": True},
        )
        with patch("main.run_pr_review_workflow", return_value=mock_res):
            exit_code = run_pr_pipeline("test-owner/test-repo#100", publish=False)
            assert exit_code == 2
