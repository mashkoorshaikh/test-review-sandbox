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


# ============================================================================
# 9. Literal None Dereference Tests (SEM002 HIGH Confidence)
# ============================================================================


class TestLiteralNoneDereference:
    def test_direct_attribute_on_literal_none(self):
        source = "res = None.strip()\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 1
        assert issues[0].confidence == "HIGH"
        assert issues[0].severity == "WARNING"
        assert "attribute 'strip' access will raise AttributeError" in issues[0].message

    def test_direct_subscript_on_literal_none(self):
        source = "item = None[0]\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 1
        assert issues[0].confidence == "HIGH"
        assert "Direct subscript on literal None will raise TypeError" in issues[0].message

    def test_direct_call_on_literal_none(self):
        source = "val = None()\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 1
        assert issues[0].confidence == "HIGH"
        assert "Calling literal None will raise TypeError" in issues[0].message


# ============================================================================
# 10. Known None In Branch / Inverted Guard Tests (SEM002 HIGH Confidence)
# ============================================================================


class TestBranchKnownNoneDereference:
    def test_dereference_in_if_is_none_branch(self):
        source = (
            "def fn(x):\n"
            "    if x is None:\n"
            "        return x.strip()\n"
            "    return 'ok'\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 3
        assert issues[0].confidence == "HIGH"
        assert "known to be None" in issues[0].message

    def test_dereference_in_else_of_if_is_not_none(self):
        source = (
            "def fn(x):\n"
            "    if x is not None:\n"
            "        pass\n"
            "    else:\n"
            "        return x.lower()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 5
        assert issues[0].confidence == "HIGH"

    def test_safe_in_else_of_if_is_none(self):
        source = (
            "def fn(x):\n"
            "    if x is None:\n"
            "        return 0\n"
            "    else:\n"
            "        return x.strip()\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_safe_in_body_of_if_is_not_none(self):
        source = (
            "def fn(x):\n"
            "    if x is not None:\n"
            "        return x.strip()\n"
            "    return ''\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_safe_in_body_of_truthy_check(self):
        source = (
            "def fn(x):\n"
            "    if x:\n"
            "        return x.strip()\n"
            "    return ''\n"
        )
        assert analyze_python_source(source, "app.py") == []


# ============================================================================
# 11. Local Variables Assigned None (SEM002 HIGH Confidence)
# ============================================================================


class TestLocalVariableAssignedNone:
    def test_local_variable_assigned_none_and_dereferenced(self):
        source = "val = None\nres = val.strip()\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 2
        assert issues[0].confidence == "HIGH"
        assert "val" in issues[0].message

    def test_local_variable_assigned_none_and_subscripted(self):
        source = "val = None\nitem = val['key']\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 2
        assert issues[0].confidence == "HIGH"

    def test_local_variable_assigned_none_and_called(self):
        source = "func = None\nfunc()\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 2
        assert issues[0].confidence == "HIGH"

    def test_local_variable_reassigned_to_non_none_is_safe(self):
        source = "val = None\nval = 'hello'\nres = val.strip()\n"
        assert analyze_python_source(source, "app.py") == []

    def test_reassignment_to_none_invalidates_clean_state(self):
        source = "val = 'hello'\nval = None\nres = val.strip()\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 3
        assert issues[0].confidence == "HIGH"


# ============================================================================
# 12. None-Defaulted Function Parameters (SEM002 MEDIUM Confidence)
# ============================================================================


class TestNoneDefaultedParameters:
    def test_none_defaulted_param_attribute_dereference_medium_confidence(self):
        source = (
            "def fn(data=None):\n"
            "    return data.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 2
        assert issues[0].confidence == "MEDIUM"
        assert "defaults to None" in issues[0].message

    def test_none_defaulted_param_subscript_medium_confidence(self):
        source = (
            "def fn(data=None):\n"
            "    return data[0]\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 2
        assert issues[0].confidence == "MEDIUM"

    def test_none_defaulted_param_call_medium_confidence(self):
        source = (
            "def fn(callback=None):\n"
            "    return callback()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 2
        assert issues[0].confidence == "MEDIUM"

    def test_none_defaulted_keyword_only_param(self):
        source = (
            "def fn(*, config=None):\n"
            "    return config['timeout']\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 2
        assert issues[0].confidence == "MEDIUM"

    def test_non_none_defaults_not_flagged(self):
        source = (
            "def fn(data='default', count=0, items=[]):\n"
            "    return data.strip() + str(count) + str(len(items))\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_parameter_without_default_not_flagged(self):
        source = (
            "def fn(data):\n"
            "    return data.strip()\n"
        )
        assert analyze_python_source(source, "app.py") == []


# ============================================================================
# 13. Guard Handling, Branch Sensitivity & Exits
# ============================================================================


class TestGuardsAndBranchSensitivity:
    def test_guarded_by_is_none_raise(self):
        source = (
            "def fn(data=None):\n"
            "    if data is None:\n"
            "        raise ValueError('data required')\n"
            "    return data['key']\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_guarded_by_is_none_return(self):
        source = (
            "def fn(data=None):\n"
            "    if data is None:\n"
            "        return {}\n"
            "    return data['key']\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_guarded_by_default_assignment(self):
        source = (
            "def fn(data=None):\n"
            "    if data is None:\n"
            "        data = {}\n"
            "    return data['key']\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_guarded_by_falsy_check_exit(self):
        source = (
            "def fn(data=None):\n"
            "    if not data:\n"
            "        return None\n"
            "    return data.strip()\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_ineffective_guard_without_exit_still_flags(self):
        source = (
            "def fn(data=None):\n"
            "    if data is None:\n"
            "        print('Warning: data is None')\n"
            "    return data.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 4
        assert issues[0].confidence == "MEDIUM"

    def test_guard_on_different_variable_does_not_suppress(self):
        source = (
            "def fn(data=None, other=None):\n"
            "    if other is None:\n"
            "        raise ValueError()\n"
            "    return data.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].line == 4
        assert "data" in issues[0].message


# ============================================================================
# 14. Immediate Regex Match Chaining (SEM002 MEDIUM Confidence)
# ============================================================================


class TestRegexMatchChaining:
    def test_regex_search_group_chaining_flagged(self):
        source = (
            "import re\n"
            "def parse_id(s):\n"
            "    return re.search(r'\\d+', s).group(0)\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 3
        assert issues[0].confidence == "MEDIUM"
        assert "re.search()" in issues[0].message

    def test_regex_match_group_chaining_flagged(self):
        source = (
            "import re\n"
            "def parse_name(s):\n"
            "    return re.match(r'\\w+', s).group()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 3
        assert issues[0].confidence == "MEDIUM"
        assert "re.match()" in issues[0].message

    def test_regex_fullmatch_groups_chaining_flagged(self):
        source = (
            "import re\n"
            "def parse_tokens(s):\n"
            "    return re.fullmatch(r'(\\w+)', s).groups()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 3
        assert issues[0].confidence == "MEDIUM"

    def test_regex_stored_and_guarded_not_flagged(self):
        source = (
            "import re\n"
            "def parse_id(s):\n"
            "    m = re.search(r'\\d+', s)\n"
            "    if m:\n"
            "        return m.group(0)\n"
            "    return None\n"
        )
        assert analyze_python_source(source, "app.py") == []


# ============================================================================
# 15. Exception-Handling Scope Tests (Policy & Non-Coverage)
# ============================================================================


class TestExceptionHandlingScope:
    def test_attribute_dereference_suppressed_by_attribute_error(self):
        source = (
            "try:\n"
            "    x = None.strip()\n"
            "except AttributeError:\n"
            "    x = ''\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_subscript_suppressed_by_type_error(self):
        source = (
            "try:\n"
            "    x = None[0]\n"
            "except TypeError:\n"
            "    x = None\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_call_suppressed_by_type_error(self):
        source = (
            "try:\n"
            "    None()\n"
            "except TypeError:\n"
            "    pass\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_suppressed_by_exception_and_bare_except(self):
        source = (
            "try:\n"
            "    x = None.strip()\n"
            "except Exception:\n"
            "    x = ''\n"
            "try:\n"
            "    y = None[0]\n"
            "except:\n"
            "    y = 0\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_attribute_dereference_not_suppressed_by_unrelated_exception(self):
        source = (
            "try:\n"
            "    x = None.strip()\n"
            "except KeyError:\n"
            "    x = ''\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 2

    def test_subscript_not_suppressed_by_key_error(self):
        # In Python, None[0] raises TypeError, not KeyError!
        source = (
            "try:\n"
            "    x = None[0]\n"
            "except KeyError:\n"
            "    x = None\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 2

    def test_dereference_outside_try_not_suppressed(self):
        source = (
            "try:\n"
            "    pass\n"
            "except AttributeError:\n"
            "    pass\n"
            "x = None.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 5


# ============================================================================
# 16. Combined SEM001 and SEM002 Regression Tests
# ============================================================================


class TestCombinedSEM001AndSEM002:
    def test_file_with_both_sem001_and_sem002(self):
        source = (
            "def calculate(a, b=None):\n"
            "    div_result = 10 / 0\n"
            "    return div_result + b.total\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 2
        assert issues[0].code == "SEM001"
        assert issues[0].line == 2
        assert issues[0].confidence == "HIGH"
        assert issues[1].code == "SEM002"
        assert issues[1].line == 3
        assert issues[1].confidence == "MEDIUM"


# ============================================================================
# 17. GitHub PR Review Diff Anchoring & Verdict Preservation for SEM002
# ============================================================================


class TestSEM002WorkflowAndGitHub:
    def test_sem002_anchors_to_diff_hunk_as_inline_comment(self, tmp_path):
        diff_text = (
            "--- a/handler.py\n"
            "+++ b/handler.py\n"
            "@@ -1,2 +1,4 @@\n"
            " def process(user=None):\n"
            "-    return None\n"
            "+    return user.name\n"
        )
        diff_files = parse_diff_hunks(diff_text, strict=False)
        diff_map = {df.path: df for df in diff_files}

        issues = [
            {
                "code": "SEM002",
                "message": "Possible None dereference: parameter 'user' defaults to None and is accessed via attribute 'name'.",
                "file_path": "handler.py",
                "line": 2,
                "severity": "WARNING",
                "tool": "semantic_analyzer",
                "confidence": "MEDIUM",
                "explanation": "Parameter 'user' defaults to None without guard.",
                "suggested_fix": "Add a check 'if user is None: ...'.",
            }
        ]

        comments, unmapped = build_inline_comments(
            issues=issues,
            diff_map=diff_map,
            workspace_root=tmp_path,
        )

        assert len(comments) == 1
        assert len(unmapped) == 0
        assert comments[0].path == "handler.py"
        assert comments[0].line == 2
        assert "[Semantic_analyzer - SEM002]" in comments[0].body
        assert "Suggested Fix:" in comments[0].body

    def test_sem002_alone_preserves_milestone_4_verdict_policy(self):
        """SEM002 alone has severity WARNING; verdict must be COMMENT, never REQUEST_CHANGES or APPROVE."""
        sem002_issue = {
            "code": "SEM002",
            "severity": "WARNING",
            "confidence": "HIGH",
            "tool": "semantic_analyzer",
        }
        verdict = determine_review_verdict([sem002_issue], [])
        assert verdict == "COMMENT"
        assert verdict != "REQUEST_CHANGES"
        assert verdict != "APPROVE"

    def test_pr_workflow_detects_none_dereference_with_zero_writes(self, tmp_path):
        """End-to-end read-only PR review workflow detects SEM002 with 0 GitHub write calls."""
        handler_content = "def process(user=None):\n    return user.name\n"
        diff_text = (
            "--- a/handler.py\n"
            "+++ b/handler.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def process(user=None):\n"
            "+    return user.name\n"
        )
        meta = PRMetadata(
            owner="octocat",
            repo="Hello-World",
            pr_number=50,
            title="Add user process handler",
            body="Implements user processing.",
            state="open",
            head_sha="0123456789abcdef0123456789abcdef01234567",
            base_sha="0000000000000000000000000000000000000000",
            head_branch="feature/handler",
            base_branch="main",
            author="alice",
            html_url="https://github.com/octocat/Hello-World/pull/50",
        )

        mock_client = MagicMock(spec=GitHubClient)
        mock_client.get_pr_metadata.return_value = meta
        mock_client.get_pr_diff.return_value = diff_text

        ws_mock = MagicMock()
        ws_mock.__enter__.return_value = ws_mock
        ws_mock.workspace_path = tmp_path
        (tmp_path / "handler.py").write_text(handler_content, encoding="utf-8")

        with patch("requests.Session.post") as mock_post, \
             patch("requests.Session.patch") as mock_patch, \
             patch("requests.Session.put") as mock_put, \
             patch("requests.Session.delete") as mock_delete, \
             patch("github_review_workflow.prepare_pr_workspace", return_value=ws_mock):

            result = run_pr_review_workflow(
                repo_or_url="octocat/Hello-World#50",
                client=mock_client,
                enable_ai=False,
                enable_sandbox=False,
                display_preview=False,
                ruff_runner=lambda p: ToolResult("Ruff", p, True, False, 0, []),
                bandit_runner=lambda p: ToolResult("Bandit", p, True, False, 0, []),
            )

            # 1. Assert SEM002 was found and formulated into an inline comment
            assert any(c.line == 2 and "SEM002" in c.body for c in result.review_comments)

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


# ============================================================================
# 18. Literal Sequence Index Out-Of-Bounds Tests (SEM003 HIGH Confidence)
# ============================================================================


class TestLiteralSequenceIndexOutOfBounds:
    def test_direct_list_literal_positive_oob(self):
        source = "val = [10, 20, 30][5]\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 1
        assert issues[0].confidence == "HIGH"
        assert issues[0].severity == "WARNING"
        assert "Sequence index out of range" in issues[0].message
        assert "constant index 5 is out of bounds for literal list of length 3" in issues[0].message

    def test_direct_tuple_literal_negative_oob(self):
        source = "val = (10, 20)[-3]\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 1
        assert issues[0].confidence == "HIGH"
        assert "constant index -3 is out of bounds for literal tuple of length 2" in issues[0].message

    def test_direct_empty_list_literal_oob(self):
        source = "val = [][0]\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 1
        assert issues[0].confidence == "HIGH"
        assert "length 0" in issues[0].message

    def test_direct_empty_tuple_literal_oob(self):
        source = "val = ()[-1]\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 1
        assert issues[0].confidence == "HIGH"

    def test_direct_valid_positive_index_clean(self):
        source = "x = [1, 2, 3][0]\ny = (10, 20)[1]\n"
        assert analyze_python_source(source, "app.py") == []

    def test_direct_valid_negative_index_clean(self):
        source = "x = [1, 2, 3][-1]\ny = (10, 20)[-2]\n"
        assert analyze_python_source(source, "app.py") == []

    def test_direct_slice_subscript_clean(self):
        # Slices in Python never raise IndexError
        source = "x = [1, 2, 3][0:10]\ny = [1, 2][5:]\nz = (10, 20)[:]\n"
        assert analyze_python_source(source, "app.py") == []

    def test_starred_unpacking_sequence_clean(self):
        source = "x = [1, *other, 2][5]\n"
        assert analyze_python_source(source, "app.py") == []


# ============================================================================
# 19. Local Variable Sequence Index Out-Of-Bounds Tests
# ============================================================================


class TestLocalVariableSequenceIndexOutOfBounds:
    def test_local_list_variable_oob(self):
        source = "items = ['a', 'b']\nv = items[2]\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 2
        assert issues[0].confidence == "HIGH"
        assert "items" in issues[0].message
        assert "length 2" in issues[0].message

    def test_local_tuple_variable_negative_oob(self):
        source = "t = (1, 2, 3)\nv = t[-4]\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 2
        assert issues[0].confidence == "HIGH"

    def test_local_empty_list_variable_oob(self):
        source = "items = []\nv = items[0]\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 2
        assert issues[0].confidence == "HIGH"

    def test_local_valid_indexes_clean(self):
        source = "items = ['a', 'b']\nx = items[0]\ny = items[1]\nz = items[-1]\n"
        assert analyze_python_source(source, "app.py") == []

    def test_local_slice_access_clean(self):
        source = "items = ['a', 'b']\nx = items[10:20]\n"
        assert analyze_python_source(source, "app.py") == []

    def test_local_dynamic_index_clean(self):
        source = "items = ['a', 'b']\nv = items[idx]\n"
        assert analyze_python_source(source, "app.py") == []

    def test_function_parameter_sequence_clean(self):
        source = "def f(items):\n    return items[5]\n"
        assert analyze_python_source(source, "app.py") == []


# ============================================================================
# 20. Literal Dictionary Missing Key Tests (SEM003 HIGH Confidence)
# ============================================================================


class TestLiteralDictionaryMissingKey:
    def test_direct_dict_literal_missing_key(self):
        source = "v = {'a': 1}['b']\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 1
        assert issues[0].confidence == "HIGH"
        assert issues[0].severity == "WARNING"
        assert "Missing dictionary key: key 'b' is provably absent from literal dictionary" in issues[0].message

    def test_direct_empty_dict_literal_missing_key(self):
        source = "v = {}[0]\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 1
        assert issues[0].confidence == "HIGH"

    def test_direct_dict_literal_valid_key_clean(self):
        source = "v = {'a': 1, 'b': 2}['a']\n"
        assert analyze_python_source(source, "app.py") == []

    def test_local_dict_variable_missing_key(self):
        source = "d = {'host': 'localhost'}\np = d['port']\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 2
        assert issues[0].confidence == "HIGH"
        assert "dictionary 'd'" in issues[0].message

    def test_local_dict_variable_valid_key_clean(self):
        source = "d = {'host': 'localhost'}\nh = d['host']\n"
        assert analyze_python_source(source, "app.py") == []

    def test_local_dict_variable_none_key_supported(self):
        source = (
            "d = {None: 'null_val', 'key': 1}\n"
            "v1 = d[None]\n"
            "v2 = d['missing']\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 3
        assert "'missing'" in issues[0].message

    def test_local_dict_integer_and_negative_keys(self):
        source = (
            "d = {1: 'one', -1: 'neg_one'}\n"
            "v1 = d[1]\n"
            "v2 = d[-1]\n"
            "v3 = d[2]\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 4
        assert "2" in issues[0].message

    def test_dict_unpacking_clean(self):
        source = "d = {'a': 1, **extra}\nv = d['b']\n"
        assert analyze_python_source(source, "app.py") == []

    def test_dynamic_key_clean(self):
        source = "d = {'a': 1, dynamic_var: 2}\nv = d['b']\n"
        assert analyze_python_source(source, "app.py") == []

    def test_dynamic_lookup_key_clean(self):
        source = "d = {'a': 1}\nv = d[user_input]\n"
        assert analyze_python_source(source, "app.py") == []


# ============================================================================
# 21. Membership Guards & Branch State Merging
# ============================================================================


class TestMembershipGuardsAndBranchState:
    def test_dict_key_guarded_by_in_check(self):
        source = (
            "d = {'a': 1}\n"
            "if 'b' in d:\n"
            "    v = d['b']\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_dict_key_unguarded_in_else_branch(self):
        source = (
            "d = {'a': 1}\n"
            "if 'b' in d:\n"
            "    pass\n"
            "else:\n"
            "    v = d['b']\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 5

    def test_dict_key_guarded_after_not_in_early_return(self):
        source = (
            "def fn():\n"
            "    d = {'a': 1}\n"
            "    if 'b' not in d:\n"
            "        return None\n"
            "    return d['b']\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_dict_key_guarded_after_not_in_early_raise(self):
        source = (
            "d = {'a': 1}\n"
            "if 'b' not in d:\n"
            "    raise KeyError('b missing')\n"
            "v = d['b']\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_branch_mutation_invalidates_collection_after_if(self):
        source = (
            "d = {'a': 1}\n"
            "if cond:\n"
            "    d['b'] = 2\n"
            "v = d['b']\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_branch_unmodified_collection_preserved_after_if(self):
        source = (
            "d = {'a': 1}\n"
            "if cond:\n"
            "    x = 10\n"
            "v = d['b']\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 4


# ============================================================================
# 22. Mutation, Aliasing, Call Escapes & Scope Isolation
# ============================================================================


class TestMutationAliasingAndScope:
    def test_list_append_invalidates(self):
        source = "lst = [1, 2]\nlst.append(3)\nv = lst[2]\n"
        assert analyze_python_source(source, "app.py") == []

    def test_list_extend_invalidates(self):
        source = "lst = [1, 2]\nlst.extend([3, 4])\nv = lst[3]\n"
        assert analyze_python_source(source, "app.py") == []

    def test_dict_update_invalidates(self):
        source = "d = {'a': 1}\nd.update({'b': 2})\nv = d['b']\n"
        assert analyze_python_source(source, "app.py") == []

    def test_subscript_write_invalidates_dict(self):
        source = "d = {'a': 1}\nd['b'] = 2\nv = d['b']\n"
        assert analyze_python_source(source, "app.py") == []

    def test_subscript_write_invalidates_list(self):
        source = "lst = [1, 2]\nlst[0] = 99\nv = lst[1]\n"
        assert analyze_python_source(source, "app.py") == []

    def test_delete_subscript_invalidates(self):
        source = "d = {'a': 1, 'b': 2}\ndel d['a']\nv = d['c']\n"
        assert analyze_python_source(source, "app.py") == []

    def test_call_escape_invalidates(self):
        source = "items = [1, 2]\nprocess(items)\nv = items[2]\n"
        assert analyze_python_source(source, "app.py") == []

    def test_alias_escape_invalidates(self):
        source = "items = [1, 2]\nalias = items\nv = items[2]\n"
        assert analyze_python_source(source, "app.py") == []

    def test_reassignment_updates_metadata(self):
        source = (
            "items = [1, 2]\n"
            "items = [1, 2, 3, 4]\n"
            "v = items[3]\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_reassignment_to_smaller_collection_flags(self):
        source = (
            "items = [1, 2, 3, 4]\n"
            "items = [10]\n"
            "v = items[1]\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 3

    def test_reassignment_to_non_literal_clears_tracking(self):
        source = "items = [1, 2]\nitems = get_items()\nv = items[5]\n"
        assert analyze_python_source(source, "app.py") == []

    def test_function_scope_isolation_and_shadowing(self):
        source = (
            "items = [1, 2]\n"
            "def fn():\n"
            "    items = [1, 2, 3, 4]\n"
            "    return items[3]\n"
            "val = items[3]\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 5
        assert "length 2" in issues[0].message


# ============================================================================
# 23. Exception Handling Scope (IndexError & KeyError)
# ============================================================================


class TestSEM003ExceptionHandlingScope:
    def test_index_error_suppressed_by_index_error(self):
        source = (
            "try:\n"
            "    x = [1, 2][5]\n"
            "except IndexError:\n"
            "    x = None\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_index_error_suppressed_by_lookup_error(self):
        source = (
            "try:\n"
            "    x = [1, 2][5]\n"
            "except LookupError:\n"
            "    x = None\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_key_error_suppressed_by_key_error(self):
        source = (
            "try:\n"
            "    x = {'a': 1}['b']\n"
            "except KeyError:\n"
            "    x = None\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_key_error_suppressed_by_lookup_error(self):
        source = (
            "try:\n"
            "    x = {'a': 1}['b']\n"
            "except LookupError:\n"
            "    x = None\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_suppressed_by_exception_and_bare_except(self):
        source = (
            "try:\n"
            "    x = [1, 2][5]\n"
            "except Exception:\n"
            "    x = None\n"
            "try:\n"
            "    y = {'a': 1}['b']\n"
            "except:\n"
            "    y = None\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_not_suppressed_by_unrelated_exception(self):
        source = (
            "try:\n"
            "    x = [1, 2][5]\n"
            "except KeyError:\n"
            "    x = None\n"
            "try:\n"
            "    y = {'a': 1}['b']\n"
            "except IndexError:\n"
            "    y = None\n"
            "try:\n"
            "    z = [1, 2][5]\n"
            "except ValueError:\n"
            "    z = None\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 3
        assert all(i.code == "SEM003" for i in issues)

    def test_nested_try_blocks_outer_catches(self):
        source = (
            "try:\n"
            "    try:\n"
            "        x = [1, 2][5]\n"
            "    except ValueError:\n"
            "        x = 0\n"
            "except IndexError:\n"
            "    x = -1\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_nested_try_unrelated_not_suppressed(self):
        source = (
            "try:\n"
            "    try:\n"
            "        x = [1, 2][5]\n"
            "    except ValueError:\n"
            "        x = 0\n"
            "except KeyError:\n"
            "    x = -1\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].line == 3


# ============================================================================
# 24. Combined SEM001 + SEM002 + SEM003 Regression Tests
# ============================================================================


class TestCombinedSEM001SEM002SEM003:
    def test_all_three_detectors_in_single_analysis(self):
        source = (
            "def compute(user=None):\n"
            "    div_zero = 10 / 0\n"
            "    deref_none = user.name\n"
            "    bad_index = [1, 2][5]\n"
            "    missing_key = {'a': 1}['x']\n"
            "    return div_zero\n"
        )
        issues = analyze_python_source(source, "src/calc.py")
        assert len(issues) == 4
        # Verify ordering strictly by line and column
        assert issues[0].code == "SEM001"
        assert issues[0].line == 2
        assert issues[0].file_path == "src/calc.py"

        assert issues[1].code == "SEM002"
        assert issues[1].line == 3
        assert issues[1].file_path == "src/calc.py"

        assert issues[2].code == "SEM003"
        assert issues[2].line == 4
        assert issues[2].file_path == "src/calc.py"

        assert issues[3].code == "SEM003"
        assert issues[3].line == 5
        assert issues[3].file_path == "src/calc.py"

    def test_subscript_none_does_not_duplicate_sem003(self):
        # In Python, val = None; val[0] raises TypeError, handled by SEM002, not SEM003
        source = "val = None\nx = val[0]\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].line == 2
        assert "TypeError" in issues[0].message


# ============================================================================
# 25. GitHub PR Review Diff Anchoring & Verdict Policy for SEM003
# ============================================================================


class TestSEM003WorkflowAndGitHub:
    def test_sem003_anchors_to_diff_hunk_as_inline_comment(self, tmp_path):
        diff_text = (
            "--- a/config.py\n"
            "+++ b/config.py\n"
            "@@ -1,2 +1,4 @@\n"
            " def get_port():\n"
            "-    return 8080\n"
            "+    d = {'host': 'localhost'}\n"
            "+    return d['port']\n"
        )
        diff_files = parse_diff_hunks(diff_text, strict=False)
        diff_map = {df.path: df for df in diff_files}

        issues = [
            {
                "code": "SEM003",
                "message": "Missing dictionary key: key 'port' is provably absent from dictionary 'd'.",
                "file_path": "config.py",
                "line": 3,
                "severity": "WARNING",
                "tool": "semantic_analyzer",
                "confidence": "HIGH",
                "explanation": "Static AST analysis detected missing key 'port'.",
                "suggested_fix": "Add 'port' to dictionary or use d.get('port').",
            }
        ]

        comments, unmapped = build_inline_comments(
            issues=issues,
            diff_map=diff_map,
            workspace_root=tmp_path,
        )

        assert len(comments) == 1
        assert len(unmapped) == 0
        assert comments[0].path == "config.py"
        assert comments[0].line == 3
        assert "[Semantic_analyzer - SEM003]" in comments[0].body
        assert "Suggested Fix:" in comments[0].body

    def test_sem003_alone_preserves_milestone_4_verdict_policy(self):
        """SEM003 alone has severity WARNING; verdict must be COMMENT, never REQUEST_CHANGES or APPROVE."""
        sem003_issue = {
            "code": "SEM003",
            "severity": "WARNING",
            "confidence": "HIGH",
            "tool": "semantic_analyzer",
        }
        verdict = determine_review_verdict([sem003_issue], [])
        assert verdict == "COMMENT"
        assert verdict != "REQUEST_CHANGES"
        assert verdict != "APPROVE"

    def test_pr_workflow_detects_sem003_with_zero_writes(self, tmp_path):
        """End-to-end read-only PR review workflow detects SEM003 with 0 GitHub write calls."""
        config_content = "def get_port():\n    d = {'host': 'localhost'}\n    return d['port']\n"
        diff_text = (
            "--- a/config.py\n"
            "+++ b/config.py\n"
            "@@ -0,0 +1,3 @@\n"
            "+def get_port():\n"
            "+    d = {'host': 'localhost'}\n"
            "+    return d['port']\n"
        )
        meta = PRMetadata(
            owner="octocat",
            repo="Hello-World",
            pr_number=60,
            title="Add port configuration",
            body="Configures service port.",
            state="open",
            head_sha="0123456789abcdef0123456789abcdef01234567",
            base_sha="0000000000000000000000000000000000000000",
            head_branch="feature/port",
            base_branch="main",
            author="alice",
            html_url="https://github.com/octocat/Hello-World/pull/60",
        )

        mock_client = MagicMock(spec=GitHubClient)
        mock_client.get_pr_metadata.return_value = meta
        mock_client.get_pr_diff.return_value = diff_text

        ws_mock = MagicMock()
        ws_mock.__enter__.return_value = ws_mock
        ws_mock.workspace_path = tmp_path
        (tmp_path / "config.py").write_text(config_content, encoding="utf-8")

        with patch("requests.Session.post") as mock_post, \
             patch("requests.Session.patch") as mock_patch, \
             patch("requests.Session.put") as mock_put, \
             patch("requests.Session.delete") as mock_delete, \
             patch("github_review_workflow.prepare_pr_workspace", return_value=ws_mock):

            result = run_pr_review_workflow(
                repo_or_url="octocat/Hello-World#60",
                client=mock_client,
                enable_ai=False,
                enable_sandbox=False,
                display_preview=False,
                ruff_runner=lambda p: ToolResult("Ruff", p, True, False, 0, []),
                bandit_runner=lambda p: ToolResult("Bandit", p, True, False, 0, []),
            )

            # 1. Assert SEM003 was found and formulated into an inline comment
            assert any(c.line == 3 and "SEM003" in c.body for c in result.review_comments)

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
