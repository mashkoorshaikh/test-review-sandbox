"""
tests/test_semantic_analyzer.py - Comprehensive Unit & Regression Tests for Semantic Analyzer.

Verifies:
  1. Literal zero divisors ('/', '//', '%') detected with HIGH confidence.
  2. Non-zero constants and safe numeric expressions are not flagged.
  3. Unguarded variable divisors flagged with MEDIUM confidence.
  4. Reliably guarded branches ('if b == 0: raise/return', 'if b != 0: ...',
     'try ... except ZeroDivisionError') are correctly suppressed.
  5. Ineffective guards (e.g. 'if b == 0: print(...)', unrelated exceptions,
     or reassignment) are NOT suppressed.
  6. Resilience on syntax errors, empty files, and unsupported file types.
  7. Structured finding serialization, workflow aggregation, diff anchoring,
     verdict preservation (COMMENT, never REQUEST_CHANGES or APPROVE), and zero write calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from github_client import GitHubClient, PRMetadata
from github_review_workflow import (
    build_inline_comments,
    determine_review_verdict,
    parse_diff_hunks,
    run_pr_review_workflow,
)
from semantic_analyzer import (
    analyze_python_source,
    run_semantic_analysis,
)
from state import dict_to_issue, issue_to_dict
from tools import Issue, ToolResult
from workflow import run_deterministic_only, run_review_pipeline

# ============================================================================
# 1. Literal Zero Divisor Tests ('/', '//', '%')
# ============================================================================


class TestLiteralZeroDivisors:
    def test_division_by_literal_zero(self):
        source = "x = 10 / 0\n"
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].line == 1
        assert issues[0].confidence == "HIGH"
        assert issues[0].severity == "WARNING"
        assert issues[0].tool == "semantic_analyzer"
        assert "ZeroDivisionError will occur" in issues[0].message

    def test_floor_division_by_literal_zero(self):
        source = "result = a // 0\n"
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "HIGH"
        assert "// 0" in issues[0].message

    def test_modulo_by_literal_zero(self):
        source = "rem = n % 0\n"
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "HIGH"
        assert "% 0" in issues[0].message

    def test_float_literal_zero(self):
        source = "val = 42 / 0.0\n"
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].confidence == "HIGH"

    def test_unary_zero(self):
        source = "neg_zero = 5 / -0\npos_zero = 5 / +0\n"
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 2
        assert all(i.confidence == "HIGH" for i in issues)

    def test_boolean_false_divisor(self):
        # In Python, 1 / False raises ZeroDivisionError: division by zero
        source = "v = 10 / False\n"
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].confidence == "HIGH"


# ============================================================================
# 2. Non-Zero Constants & Variable Divisors
# ============================================================================


class TestNonZeroConstantsAndVariables:
    def test_nonzero_integer_constants_not_flagged(self):
        source = "x = 10 / 2\ny = 20 // 3\nz = 30 % 4\n"
        assert analyze_python_source(source, "calc.py") == []

    def test_nonzero_float_constants_not_flagged(self):
        source = "discounted = price * (1 - discount / 100.0)\n"
        assert analyze_python_source(source, "calc.py") == []

    def test_boolean_true_not_flagged(self):
        source = "v = 10 / True\n"
        assert analyze_python_source(source, "calc.py") == []

    def test_unguarded_function_parameter_reported(self):
        source = "def divide(a, b):\n    return a / b\n"
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].line == 2
        assert issues[0].confidence == "MEDIUM"
        assert "divisor 'b' may evaluate to zero" in issues[0].message
        assert "cannot determine runtime values" in issues[0].explanation

    def test_unguarded_floor_division_and_modulo_variables(self):
        source = (
            "def div_floor(a, b):\n"
            "    return a // b\n"
            "def mod_op(a, b):\n"
            "    return a % b\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 2
        assert issues[0].line == 2
        assert issues[1].line == 4
        assert all(i.confidence == "MEDIUM" for i in issues)


# ============================================================================
# 3. Reliably Guarded Branches & Try/Except Suppression
# ============================================================================


class TestReliablyGuardedBranches:
    def test_guarded_by_if_zero_raise(self):
        source = (
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        raise ValueError('divisor cannot be 0')\n"
            "    return a / b\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_guarded_by_if_not_var_return(self):
        source = (
            "def divide(a, b):\n"
            "    if not b:\n"
            "        return 0\n"
            "    return a / b\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_guarded_by_if_nonzero_branch(self):
        source = (
            "def divide(a, b):\n"
            "    if b != 0:\n"
            "        return a / b\n"
            "    return 0\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_guarded_by_if_gt_zero(self):
        source = (
            "def divide(a, b):\n"
            "    if b > 0:\n"
            "        return a / b\n"
            "    return None\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_guarded_in_else_branch(self):
        source = (
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        return 0\n"
            "    else:\n"
            "        return a / b\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_guarded_by_try_except_zerodivisionerror(self):
        source = (
            "def divide(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except ZeroDivisionError:\n"
            "        return 0\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_guarded_by_try_except_tuple_handler(self):
        source = (
            "def divide(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except (ValueError, ZeroDivisionError):\n"
            "        return 0\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_guarded_by_try_except_exception_base(self):
        source = (
            "def divide(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except Exception:\n"
            "        return 0\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_guarded_by_bare_except(self):
        source = (
            "def divide(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except:\n"
            "        return 0\n"
        )
        assert analyze_python_source(source, "calc.py") == []


# ============================================================================
# 4. Ineffective / Incomplete Guards (Must NOT Suppress)
# ============================================================================


class TestIneffectiveGuards:
    def test_guard_without_exit_does_not_suppress(self):
        """When 'if b == 0' merely logs or prints without exiting, flag as hazard."""
        source = (
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        print('Warning: b is zero')\n"
            "    return a / b\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].line == 4

    def test_guard_on_unrelated_variable_does_not_suppress(self):
        source = (
            "def divide(a, b, c):\n"
            "    if c == 0:\n"
            "        raise ValueError('c cannot be zero')\n"
            "    return a / b\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].line == 4
        assert "divisor 'b'" in issues[0].message

    def test_try_except_unrelated_exception_does_not_suppress(self):
        """try/except KeyError does NOT catch ZeroDivisionError."""
        source = (
            "def divide(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except KeyError:\n"
            "        return 0\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].line == 3

    def test_variable_reassignment_invalidates_guard(self):
        source = (
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        return 0\n"
            "    b = 0\n"
            "    return a / b\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].line == 5


# ============================================================================
# 5. File System & Resilience (Syntax Errors, Empty Files, Non-Python Files)
# ============================================================================


class TestFileSystemAndResilience:
    def test_syntax_error_handled_gracefully(self, tmp_path):
        bad_file = tmp_path / "broken.py"
        bad_file.write_text("def broken(:\n    pass\n", encoding="utf-8")
        res = run_semantic_analysis(str(bad_file))
        assert res.success is True
        assert res.issues == []

    def test_empty_file_returns_clean(self, tmp_path):
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("", encoding="utf-8")
        res = run_semantic_analysis(str(empty_file))
        assert res.success is True
        assert res.issues == []

    def test_non_python_file_skipped(self, tmp_path):
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("10 / 0", encoding="utf-8")
        res = run_semantic_analysis(str(txt_file))
        assert res.success is True
        assert res.issues == []

    def test_nonexistent_path_returns_error(self, tmp_path):
        missing = tmp_path / "does_not_exist.py"
        res = run_semantic_analysis(str(missing))
        assert res.success is False
        assert "does not exist" in str(res.error_message)

    def test_directory_scan_aggregates_python_files(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "mod1.py").write_text("def f(a, b): return a / b\n", encoding="utf-8")
        (sub / "mod2.py").write_text("x = 10 / 0\n", encoding="utf-8")
        (sub / "mod3.py").write_text("x = 10 / 2\n", encoding="utf-8")

        res = run_semantic_analysis(str(sub))
        assert res.success is True
        assert len(res.issues) == 2


# ============================================================================
# 6. Structured Finding Serialization & Data Model
# ============================================================================


class TestStructuredFindingSerialization:
    def test_issue_serialization_roundtrip(self):
        issue = Issue(
            code="SEM001",
            message="Possible division by zero",
            file_path="calc.py",
            line=10,
            column=4,
            severity="WARNING",
            tool="semantic_analyzer",
            fixable=False,
            confidence="MEDIUM",
            explanation="Unvalidated divisor",
            suggested_fix="Add if b == 0 check",
        )
        d = issue_to_dict(issue)
        assert d["code"] == "SEM001"
        assert d["confidence"] == "MEDIUM"
        assert d["explanation"] == "Unvalidated divisor"
        assert d["suggested_fix"] == "Add if b == 0 check"

        reconstructed = dict_to_issue(d)
        assert reconstructed.code == "SEM001"
        assert reconstructed.confidence == "MEDIUM"
        assert reconstructed.explanation == "Unvalidated divisor"
        assert reconstructed.suggested_fix == "Add if b == 0 check"


# ============================================================================
# 7. Workflow & Orchestrator Report Integration
# ============================================================================


class TestWorkflowIntegration:
    def test_run_deterministic_only_includes_semantic_findings(self, tmp_path):
        calc_file = tmp_path / "calc.py"
        calc_file.write_text("def divide(a, b):\n    return a / b\n", encoding="utf-8")

        state = run_deterministic_only(str(calc_file))
        assert "semantic_findings" in state
        assert len(state["semantic_findings"]) == 1
        assert state["semantic_findings"][0]["code"] == "SEM001"

        rep = state["final_review_report"]
        assert rep["summary"]["semantic_issues_count"] == 1
        assert len(rep["deterministic_findings"]["semantic"]) == 1

    def test_run_review_pipeline_executes_semantic_agent(self, tmp_path):
        calc_file = tmp_path / "calc.py"
        calc_file.write_text("x = 10 / 0\n", encoding="utf-8")

        state = run_review_pipeline(
            target_path=str(calc_file),
            enable_ai=True,
            enable_sandbox=False,
            enable_debugging=False,
        )
        assert "semantic_findings" in state
        assert len(state["semantic_findings"]) == 1
        assert state["semantic_findings"][0]["code"] == "SEM001"
        assert state["semantic_findings"][0]["confidence"] == "HIGH"


# ============================================================================
# 8. GitHub PR Review Diff Anchoring, Verdict Policy & Zero Writes
# ============================================================================


class TestGitHubReviewWorkflowIntegration:
    def test_sem001_anchors_to_diff_hunk_as_inline_comment(self, tmp_path):
        diff_text = (
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -1,2 +1,4 @@\n"
            " def add(a, b):\n"
            "     return a + b\n"
            "+def divide(a, b):\n"
            "+    return a / b\n"
        )
        diff_files = parse_diff_hunks(diff_text, strict=False)
        diff_map = {df.path: df for df in diff_files}

        issues = [
            {
                "code": "SEM001",
                "message": "Possible division by zero: divisor 'b' may evaluate to zero without explicit ZeroDivisionError handling.",
                "file_path": "calc.py",
                "line": 4,
                "severity": "WARNING",
                "tool": "semantic_analyzer",
                "confidence": "MEDIUM",
                "explanation": "Static AST analysis detected unguarded divisor 'b'.",
                "suggested_fix": "Add a guard such as 'if b == 0: raise ValueError(...)'.",
            }
        ]

        comments, unmapped = build_inline_comments(
            issues=issues,
            diff_map=diff_map,
            workspace_root=tmp_path,
        )

        assert len(comments) == 1
        assert len(unmapped) == 0
        assert comments[0].path == "calc.py"
        assert comments[0].line == 4
        assert "[Semantic_analyzer - SEM001]" in comments[0].body
        assert "Suggested Fix:" in comments[0].body

    def test_sem001_preserves_milestone_4_verdict_policy(self):
        """SEM001 alone has severity WARNING; verdict must be COMMENT, never REQUEST_CHANGES or APPROVE."""
        sem_issue = {
            "code": "SEM001",
            "severity": "WARNING",
            "confidence": "HIGH",
            "tool": "semantic_analyzer",
        }
        verdict = determine_review_verdict([sem_issue], [])
        assert verdict == "COMMENT"
        assert verdict != "REQUEST_CHANGES"
        assert verdict != "APPROVE"

    def test_pr_workflow_detects_divide_by_zero_and_makes_zero_writes(self, tmp_path):
        """End-to-end read-only PR review workflow detects SEM001 with 0 GitHub write calls."""
        calc_content = "def divide(a, b):\n    return a / b\n"
        diff_text = (
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def divide(a, b):\n"
            "+    return a / b\n"
        )
        meta = PRMetadata(
            owner="octocat",
            repo="Hello-World",
            pr_number=42,
            title="Add divide function",
            body="Implements basic division.",
            state="open",
            head_sha="0123456789abcdef0123456789abcdef01234567",
            base_sha="0000000000000000000000000000000000000000",
            head_branch="feature/divide",
            base_branch="main",
            author="alice",
            html_url="https://github.com/octocat/Hello-World/pull/42",
        )

        mock_client = MagicMock(spec=GitHubClient)
        mock_client.get_pr_metadata.return_value = meta
        mock_client.get_pr_diff.return_value = diff_text

        # Prepare workspace mock
        ws_mock = MagicMock()
        ws_mock.__enter__.return_value = ws_mock
        ws_mock.workspace_path = tmp_path
        (tmp_path / "calc.py").write_text(calc_content, encoding="utf-8")

        with patch("requests.Session.post") as mock_post, \
             patch("requests.Session.patch") as mock_patch, \
             patch("requests.Session.put") as mock_put, \
             patch("requests.Session.delete") as mock_delete, \
             patch("github_review_workflow.prepare_pr_workspace", return_value=ws_mock):

            result = run_pr_review_workflow(
                repo_or_url="octocat/Hello-World#42",
                client=mock_client,
                enable_ai=False,
                enable_sandbox=False,
                display_preview=False,
                ruff_runner=lambda p: ToolResult("Ruff", p, True, False, 0, []),
                bandit_runner=lambda p: ToolResult("Bandit", p, True, False, 0, []),
            )

            # 1. Assert SEM001 was found and formulated into an inline comment
            assert any(c.line == 2 and "SEM001" in c.body for c in result.review_comments)

            # 2. Assert verdict is COMMENT
            assert result.action_plan.verdict == "COMMENT"
            assert result.action_plan.verdict != "APPROVE"
            assert result.action_plan.verdict != "REQUEST_CHANGES"

            # 3. Assert zero write calls were made to GitHub API
            assert mock_post.call_count == 0
            assert mock_patch.call_count == 0
            assert mock_put.call_count == 0
            assert mock_delete.call_count == 0

            # 4. Action plan remains dry_run
            assert result.action_plan.dry_run is True
