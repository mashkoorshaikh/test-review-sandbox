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
# 2b. Pathlib Path-Joining vs Arithmetic Division (SEM001)
# ============================================================================


class TestSEM001PathJoiningHandling:
    """
    Regression tests verifying that pathlib.Path '/' joining operations
    are never misidentified as unhandled ZeroDivisionError (SEM001), while
    genuine arithmetic zero-division risks remain strictly detected.
    """

    def test_path_division_string_does_not_trigger_sem001(self):
        source = "file_path = Path('src') / 'test.py'\n"
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_tmp_path_division_string_does_not_trigger_sem001(self):
        source = "test_file = tmp_path / 'test.py'\n"
        issues = analyze_python_source(source, "test_app.py")
        assert issues == []

    def test_chained_path_division_does_not_trigger_sem001(self):
        source = "workflow = Path('.github') / 'workflows' / 'pr-review.yml'\n"
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_path_variable_division_does_not_trigger_sem001(self):
        source = (
            "p = Path('src')\n"
            "out = p / filename\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_tmp_path_parameter_division_does_not_trigger_sem001(self):
        source = (
            "def test_example(tmp_path):\n"
            "    config = tmp_path / filename\n"
            "    return config\n"
        )
        issues = analyze_python_source(source, "test_app.py")
        assert issues == []

    def test_path_parent_division_does_not_trigger_sem001(self):
        source = "summary = Path(__file__).resolve().parent / 'sub' / 'file.txt'\n"
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_path_fstring_division_does_not_trigger_sem001(self):
        source = "artifact = base_dir / f'artifact_{item_id}.json'\n"
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_path_annotated_parameter_division_does_not_trigger_sem001(self):
        source = (
            "from pathlib import Path\n"
            "def process_workspace(ws: Path, filename: str):\n"
            "    return ws / filename\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_numeric_division_by_zero_still_triggers_sem001(self):
        cases = [
            ("x = 10 / 0\n", "/"),
            ("x = 10 / 0.0\n", "/"),
            ("x = 10 // 0\n", "//"),
            ("x = 10 % 0\n", "%"),
        ]
        for src, op in cases:
            issues = analyze_python_source(src, "calc.py")
            assert len(issues) == 1, f"Expected 1 issue for {src.strip()}"
            assert issues[0].code == "SEM001"
            assert issues[0].confidence == "HIGH"
            assert issues[0].severity == "WARNING"

    def test_numeric_division_by_variable_still_triggers_sem001(self):
        source = (
            "def calculate(numerator, divisor):\n"
            "    return numerator / divisor\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert "divisor 'divisor' may evaluate to zero" in issues[0].message

    def test_path_divided_by_zero_literal_triggers_sem001(self):
        source = "p = Path('src') / 0\n"
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "HIGH"
        assert "Division by zero literal" in issues[0].message



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


# ============================================================================
# 26. Batch 1 Accuracy Regression Tests (Try/Loop Scoping & Literal Zero Handler Suppression)
# ============================================================================


class TestSEM001LiteralZeroTrySuppression:
    def test_literal_zero_suppressed_by_zerodivisionerror(self):
        source = (
            "try:\n"
            "    x = 10 / 0\n"
            "except ZeroDivisionError:\n"
            "    x = 0\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_literal_zero_suppressed_by_arithmeticerror_and_exception(self):
        source = (
            "try:\n"
            "    a = 10 // 0\n"
            "except ArithmeticError:\n"
            "    a = 0\n"
            "try:\n"
            "    b = 10 % 0\n"
            "except Exception:\n"
            "    b = 0\n"
            "try:\n"
            "    c = 10 / 0.0\n"
            "except:\n"
            "    c = 0\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_unhandled_literal_zero_produces_high_confidence_sem001(self):
        source = "x = 10 / 0\n"
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "HIGH"
        assert issues[0].severity == "WARNING"
        assert issues[0].line == 1

    def test_literal_zero_not_suppressed_by_unrelated_exception_handler(self):
        source = (
            "try:\n"
            "    x = 10 / 0\n"
            "except KeyError:\n"
            "    x = 0\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 2

    def test_literal_zero_in_except_handler_not_suppressed(self):
        source = (
            "try:\n"
            "    pass\n"
            "except Exception:\n"
            "    val = 10 / 0\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 4

    def test_literal_zero_nested_try_blocks(self):
        source = (
            "try:\n"
            "    try:\n"
            "        val = 1 / 0\n"
            "    except KeyError:\n"
            "        pass\n"
            "except ZeroDivisionError:\n"
            "    val = 0\n"
        )
        assert analyze_python_source(source, "calc.py") == []


class TestSEM002TryAndLoopAssignmentTracking:
    def test_local_none_assigned_in_try_body_and_dereferenced(self):
        source = (
            "try:\n"
            "    val = None\n"
            "    print(val.strip())\n"
            "except ValueError:\n"
            "    pass\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 3
        assert "known to be None" in issues[0].message

    def test_local_none_assigned_in_try_body_suppressed_when_attribute_error_caught(self):
        source = (
            "try:\n"
            "    val = None\n"
            "    print(val.strip())\n"
            "except AttributeError:\n"
            "    pass\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_local_none_assigned_in_for_loop_and_dereferenced(self):
        source = (
            "for item in [1, 2, 3]:\n"
            "    node = None\n"
            "    print(node.attr)\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 3
        assert "known to be None" in issues[0].message

    def test_local_none_assigned_in_while_loop_and_dereferenced(self):
        source = (
            "while True:\n"
            "    handler = None\n"
            "    handler()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 3
        assert "known to be None" in issues[0].message

    def test_loop_target_variable_not_falsely_flagged_from_outer_scope(self):
        source = (
            "x = None\n"
            "for x in [1, 2]:\n"
            "    print(x.name)\n"
        )
        assert analyze_python_source(source, "app.py") == []


class TestSEM003TryAndLoopCollectionTracking:
    def test_literal_dict_missing_key_in_try_body(self):
        source = (
            "try:\n"
            "    mapping = {'alpha': 1}\n"
            "    print(mapping['beta'])\n"
            "except ValueError:\n"
            "    pass\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 3
        assert "provably absent" in issues[0].message

    def test_literal_dict_missing_key_in_try_body_suppressed_by_keyerror(self):
        source = (
            "try:\n"
            "    mapping = {'alpha': 1}\n"
            "    print(mapping['beta'])\n"
            "except KeyError:\n"
            "    pass\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_local_sequence_index_oob_in_for_loop(self):
        source = (
            "for item in [1, 2]:\n"
            "    items_list = [10, 20]\n"
            "    print(items_list[5])\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 3
        assert "constant index 5 is out of bounds" in issues[0].message

    def test_local_dict_missing_key_in_while_loop(self):
        source = (
            "while True:\n"
            "    d = {'id': 100}\n"
            "    print(d['unknown'])\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 3
        assert "provably absent" in issues[0].message

    def test_no_duplicate_findings_on_complex_nested_blocks(self):
        source = (
            "for i in [1]:\n"
            "    try:\n"
            "        while True:\n"
            "            d = {'k': 1}\n"
            "            print(d['missing'])\n"
            "            break\n"
            "    except KeyError:\n"
            "        pass\n"
        )
        assert analyze_python_source(source, "app.py") == []


# ============================================================================
# 27. Batch 1.1 Edge-Case Regression Tests (Scope-Aware Walk, Finally Invalidation, If-Branch Symmetry, While Guards)
# ============================================================================


class TestScopeAwareAssignmentExtraction:
    def test_outer_none_retained_when_nested_function_assigns_same_name(self):
        source = (
            "def outer():\n"
            "    x = None\n"
            "    for item in [1, 2]:\n"
            "        def helper():\n"
            "            x = 10\n"
            "    print(x.attr)\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 6
        assert "known to be None" in issues[0].message

    def test_outer_guard_not_invalidated_by_nested_function_assignment(self):
        source = (
            "def compute(b: float):\n"
            "    if b != 0:\n"
            "        for item in [1, 2]:\n"
            "            def helper():\n"
            "                b = 0\n"
            "        return 10 / b\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_nested_async_function_and_class_scopes_do_not_leak_assignments(self):
        source = (
            "def outer():\n"
            "    x = None\n"
            "    try:\n"
            "        async def async_helper():\n"
            "            x = 20\n"
            "        class Inner:\n"
            "            x = 30\n"
            "    except Exception:\n"
            "        pass\n"
            "    print(x.attr)\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 10


class TestFinallyStatePropagation:
    def test_dict_reassigned_in_try_does_not_falsely_flag_in_finally(self):
        source = (
            "def f():\n"
            "    d = {'a': 1}\n"
            "    try:\n"
            "        d = {'b': 2}\n"
            "    finally:\n"
            "        print(d['b'])\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_none_reassigned_in_try_does_not_report_high_confidence_in_finally(self):
        source = (
            "def g():\n"
            "    x = None\n"
            "    try:\n"
            "        x = 'valid'\n"
            "    finally:\n"
            "        print(x.upper())\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_dict_reassigned_in_except_handler_invalidated_in_finally(self):
        source = (
            "def h():\n"
            "    d = {'a': 1}\n"
            "    try:\n"
            "        pass\n"
            "    except Exception:\n"
            "        d = {'c': 3}\n"
            "    finally:\n"
            "        print(d['c'])\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_nested_try_finally_invalidates_modified_collections(self):
        source = (
            "def nested():\n"
            "    m = {'k1': 10}\n"
            "    try:\n"
            "        try:\n"
            "            m = {'k2': 20}\n"
            "        finally:\n"
            "            pass\n"
            "    finally:\n"
            "        print(m['k2'])\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_valid_findings_in_finally_preserved(self):
        source = (
            "def genuine():\n"
            "    x = None\n"
            "    try:\n"
            "        pass\n"
            "    finally:\n"
            "        print(x.strip())\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 6

    def test_unmodified_collection_in_try_preserves_missing_key_finding_in_finally(self):
        source = (
            "def genuine_dict():\n"
            "    d = {'a': 1}\n"
            "    try:\n"
            "        pass\n"
            "    finally:\n"
            "        print(d['missing'])\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 6


class TestSymmetricIfBranchInvalidation:
    def test_conditional_reassignment_does_not_report_definite_none_in_sem002(self):
        source = (
            "def f(flag):\n"
            "    x = None\n"
            "    if flag:\n"
            "        x = 'initialized'\n"
            "    print(x.upper())\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_nested_conditional_zero_assignment_invalidates_guard_in_sem001(self):
        source = (
            "def calc(b: float, flag: bool):\n"
            "    if b != 0:\n"
            "        if flag:\n"
            "            b = 0\n"
            "        return 10 / b\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 5

    def test_unconditional_exit_in_if_preserves_guard_and_state(self):
        source = (
            "def early_exit(b: float, x=None):\n"
            "    if b == 0:\n"
            "        return 0\n"
            "    if x is None:\n"
            "        return 1\n"
            "    return (10 / b) + len(x.attr)\n"
        )
        assert analyze_python_source(source, "app.py") == []


class TestWhileConditionGuards:
    def test_while_condition_guards_nonzero_divisor(self):
        source = (
            "def drain(divisor: float):\n"
            "    while divisor != 0:\n"
            "        return 10 / divisor\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_while_condition_guards_greater_than_zero(self):
        source = (
            "def drain_gt(divisor: float):\n"
            "    while divisor > 0:\n"
            "        return 10 / divisor\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_while_condition_guards_not_none(self):
        source = (
            "def iterate(x=None):\n"
            "    while x is not None:\n"
            "        print(x.attr)\n"
            "        break\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_while_condition_truthiness_guards_not_none(self):
        source = (
            "def iterate_truthy(x=None):\n"
            "    while x:\n"
            "        print(x.attr)\n"
            "        break\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_while_condition_guards_membership_in_dict(self):
        source = (
            "def check_dict():\n"
            "    d = {'a': 1}\n"
            "    while 'b' in d:\n"
            "        print(d['b'])\n"
            "        break\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_while_guard_restored_after_loop_exit(self):
        source = (
            "def after_while(x=None):\n"
            "    while x is not None:\n"
            "        break\n"
            "    print(x.attr)\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 4


class TestFinallyElsePathInvalidation:
    def test_try_else_zero_assignment_invalidates_guard_in_finally_sem001(self):
        source = (
            "def compute(b: float):\n"
            "    if b != 0:\n"
            "        try:\n"
            "            pass\n"
            "        except Exception:\n"
            "            pass\n"
            "        else:\n"
            "            b = 0\n"
            "        finally:\n"
            "            return 10 / b\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 10

    def test_try_else_literal_zero_reassignment_in_finally_sem001(self):
        source = (
            "def compute(b: float):\n"
            "    if b != 0:\n"
            "        try:\n"
            "            pass\n"
            "        except Exception:\n"
            "            pass\n"
            "        else:\n"
            "            b = 0\n"
            "        finally:\n"
            "            print(100 // b)\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"

    def test_try_else_none_reassigned_not_flagged_in_finally_sem002(self):
        source = (
            "def process():\n"
            "    x = None\n"
            "    try:\n"
            "        pass\n"
            "    except Exception:\n"
            "        pass\n"
            "    else:\n"
            "        x = 'valid_string'\n"
            "    finally:\n"
            "        print(x.upper())\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_try_else_dict_subscript_mutation_not_flagged_in_finally_sem003(self):
        source = (
            "def lookup():\n"
            "    d = {'a': 1}\n"
            "    try:\n"
            "        pass\n"
            "    except Exception:\n"
            "        pass\n"
            "    else:\n"
            "        d['b'] = 2\n"
            "    finally:\n"
            "        print(d['b'])\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_try_else_dict_method_mutation_not_flagged_in_finally_sem003(self):
        source = (
            "def lookup():\n"
            "    d = {'a': 1}\n"
            "    try:\n"
            "        pass\n"
            "    except Exception:\n"
            "        pass\n"
            "    else:\n"
            "        d.update({'b': 2})\n"
            "    finally:\n"
            "        print(d['b'])\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_finally_positive_control_unmodified_zero_division_sem001(self):
        source = (
            "def genuine_div():\n"
            "    z = 0\n"
            "    try:\n"
            "        pass\n"
            "    except Exception:\n"
            "        pass\n"
            "    else:\n"
            "        other = 1\n"
            "    finally:\n"
            "        print(10 / z)\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 10

    def test_finally_positive_control_unmodified_none_dereference_sem002(self):
        source = (
            "def genuine_none():\n"
            "    x = None\n"
            "    try:\n"
            "        pass\n"
            "    except Exception:\n"
            "        pass\n"
            "    else:\n"
            "        other = 1\n"
            "    finally:\n"
            "        print(x.strip())\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 10

    def test_finally_positive_control_unmodified_missing_key_sem003(self):
        source = (
            "def genuine_key():\n"
            "    d = {'a': 1}\n"
            "    try:\n"
            "        pass\n"
            "    except Exception:\n"
            "        pass\n"
            "    else:\n"
            "        other = 1\n"
            "    finally:\n"
            "        print(d['missing'])\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 10


class TestScopeAwareFunctionClassBindings:
    def test_none_rebound_to_function_def_in_branch_clears_definite_none(self):
        source = (
            "def worker(flag: bool):\n"
            "    handler = None\n"
            "    if flag:\n"
            "        def handler():\n"
            "            return 42\n"
            "    print(handler.attribute)\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_none_rebound_to_async_function_def_in_branch_clears_definite_none(self):
        source = (
            "def worker(flag: bool):\n"
            "    handler = None\n"
            "    if flag:\n"
            "        async def handler():\n"
            "            return 42\n"
            "    print(handler.attribute)\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_none_rebound_to_function_def_in_loop_clears_definite_none(self):
        source = (
            "def worker():\n"
            "    handler = None\n"
            "    for _ in [1, 2]:\n"
            "        def handler():\n"
            "            return 10\n"
            "    print(handler.attribute)\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_none_rebound_to_class_def_in_branch_clears_definite_none(self):
        source = (
            "def worker(flag: bool):\n"
            "    cls = None\n"
            "    if flag:\n"
            "        class cls:\n"
            "            val = 1\n"
            "    print(cls.val)\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_none_rebound_to_class_def_in_sequence_clears_definite_none(self):
        source = (
            "def worker():\n"
            "    obj = None\n"
            "    class obj:\n"
            "        val = 1\n"
            "    print(obj.val)\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_none_rebound_to_function_def_in_sequence_clears_definite_none(self):
        source = (
            "def worker():\n"
            "    func = None\n"
            "    def func():\n"
            "        pass\n"
            "    print(func.attr)\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_lambda_does_not_act_as_statement_binding_or_leak(self):
        source = (
            "def worker(flag: bool):\n"
            "    x = None\n"
            "    if flag:\n"
            "        fn = lambda x=1: x + 10\n"
            "    print(x.attr)\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 5

    def test_nested_class_assignment_does_not_invalidate_outer_guard(self):
        source = (
            "def compute(b: float):\n"
            "    if b != 0:\n"
            "        class Inner:\n"
            "            b = 0\n"
            "        return 10 / b\n"
        )
        assert analyze_python_source(source, "calc.py") == []


# ============================================================================
# 28. Phase 5A Batch 2: Compound Boolean Guards
# ============================================================================


class TestCompoundBooleanGuards:
    def test_compound_and_nonzero_guard(self):
        source = (
            "def calc(divisor: int, value: int):\n"
            "    if divisor != 0 and value > 0:\n"
            "        result = value / divisor\n"
            "        return result\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_compound_and_none_guard(self):
        source = (
            "def process(x):\n"
            "    if x is not None and x.attr:\n"
            "        return x.attr\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_compound_and_dict_guard(self):
        source = (
            "def check_config(d):\n"
            "    if 'k' in d and d['k'] > 0:\n"
            "        return d['k']\n"
        )
        assert analyze_python_source(source, "config.py") == []

    def test_compound_and_definite_none_dereference_detected(self):
        source = (
            "def handler(flag: bool):\n"
            "    x = None\n"
            "    if x is None and flag:\n"
            "        print(x.attr)\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 4
        assert "known to be None" in issues[0].message

    def test_nested_parenthesized_boolean_expressions_safe(self):
        source = (
            "def run(divisor: int, flag: bool, x, d: dict):\n"
            "    if (divisor != 0 and (flag or not flag)) and True:\n"
            "        res = 10 / divisor\n"
            "    if ((x is not None and flag) and True):\n"
            "        val = x.attr\n"
            "    if ('k' in d and (flag or True)):\n"
            "        item = d['k']\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_nested_parenthesized_boolean_expressions_definite_none(self):
        source = (
            "def run(flag: bool):\n"
            "    x = None\n"
            "    if (x is None and (flag and True)):\n"
            "        print(x.attr)\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 4

    def test_multiple_independent_facts_in_single_condition(self):
        source = (
            "def process(divisor: int, x, d: dict):\n"
            "    if divisor != 0 and x is not None and 'k' in d:\n"
            "        return (100 / divisor) + len(x.attr) + d['k']\n"
        )
        assert analyze_python_source(source, "calc.py") == []


# ============================================================================
# 29. Phase 5A Batch 2: Compound Exits and Loop Guards
# ============================================================================


class TestCompoundExitsAndLoopGuards:
    def test_compound_or_exit_nonzero(self):
        source = (
            "def compute(divisor: int, error: bool):\n"
            "    if divisor == 0 or error:\n"
            "        return\n"
            "    result = 10 / divisor\n"
            "    return result\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_compound_or_exit_none(self):
        source = (
            "def process(x, error: bool):\n"
            "    if x is None or error:\n"
            "        return\n"
            "    return x.attr\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_compound_or_exit_dict(self):
        source = (
            "def get_val(d: dict, error: bool):\n"
            "    if 'k' not in d or error:\n"
            "        return\n"
            "    return d['k']\n"
        )
        assert analyze_python_source(source, "dict_app.py") == []

    def test_negative_control_and_exit_does_not_establish_safe_exit_nonzero(self):
        source = (
            "def compute(error: bool):\n"
            "    divisor = 0\n"
            "    if divisor == 0 and error:\n"
            "        return\n"
            "    return 10 / divisor\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 5

    def test_negative_control_and_exit_does_not_establish_safe_exit_none(self):
        source = (
            "def process(error: bool):\n"
            "    x = None\n"
            "    if x is None and error:\n"
            "        return\n"
            "    return x.attr\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 5

    def test_negative_control_and_exit_does_not_establish_safe_exit_dict(self):
        source = (
            "def get_val(error: bool):\n"
            "    d = {'other': 1}\n"
            "    if 'k' not in d and error:\n"
            "        return\n"
            "    return d['k']\n"
        )
        issues = analyze_python_source(source, "dict_app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 5

    def test_loop_guard_clause_continue_nonzero(self):
        source = (
            "def run(items, divisor: int):\n"
            "    for item in items:\n"
            "        if divisor == 0:\n"
            "            continue\n"
            "        result = 10 / divisor\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_loop_guard_clause_break_nonzero(self):
        source = (
            "def run(items, divisor: int):\n"
            "    for item in items:\n"
            "        if divisor == 0:\n"
            "            break\n"
            "        result = 10 / divisor\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_loop_guard_clause_continue_none(self):
        source = (
            "def run(items, x):\n"
            "    for item in items:\n"
            "        if x is None:\n"
            "            continue\n"
            "        print(x.attr)\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_loop_guard_clause_continue_dict(self):
        source = (
            "def run(items, d: dict):\n"
            "    for item in items:\n"
            "        if 'k' not in d:\n"
            "            continue\n"
            "        val = d['k']\n"
        )
        assert analyze_python_source(source, "dict_app.py") == []

    def test_loop_guard_clause_does_not_leak_after_loop_termination_nonzero(self):
        source = (
            "def run(items):\n"
            "    divisor = 0\n"
            "    for item in items:\n"
            "        if divisor == 0:\n"
            "            break\n"
            "        val = 10 / divisor\n"
            "    return 10 / divisor\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 7

    def test_loop_guard_clause_does_not_leak_after_loop_termination_none(self):
        source = (
            "def run(items):\n"
            "    x = None\n"
            "    for item in items:\n"
            "        if x is None:\n"
            "            continue\n"
            "        print(x.attr)\n"
            "    print(x.attr)\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 7


# ============================================================================
# 30. Phase 5A Batch 2: Short-Circuit Expressions
# ============================================================================


class TestShortCircuitExpressions:
    def test_short_circuit_and_nonzero(self):
        source = (
            "def check(divisor: int):\n"
            "    return divisor != 0 and (10 / divisor > 1)\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_short_circuit_or_nonzero(self):
        source = (
            "def check(divisor: int):\n"
            "    return divisor == 0 or (10 / divisor > 1)\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_short_circuit_and_none(self):
        source = (
            "def check(x):\n"
            "    return x is not None and x.attr\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_short_circuit_or_none(self):
        source = (
            "def check(x):\n"
            "    return x is None or x.attr\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_short_circuit_and_dict(self):
        source = (
            "def check(d: dict):\n"
            "    return 'k' in d and d['k']\n"
        )
        assert analyze_python_source(source, "dict_app.py") == []

    def test_short_circuit_or_dict(self):
        source = (
            "def check(d: dict):\n"
            "    return 'k' not in d or d['k']\n"
        )
        assert analyze_python_source(source, "dict_app.py") == []

    def test_short_circuit_negative_and_nonzero(self):
        source = (
            "def run():\n"
            "    b = 0\n"
            "    return b == 0 and (10 / b > 1)\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 3

    def test_short_circuit_negative_or_none(self):
        source = (
            "def run():\n"
            "    x = None\n"
            "    return x is not None or x.attr\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 3

    def test_conditional_ifexp_safe(self):
        source = (
            "def run(d: int, x, dct: dict):\n"
            "    res1 = 10 / d if d != 0 else 0\n"
            "    res2 = x.attr if x is not None else ''\n"
            "    res3 = dct['k'] if 'k' in dct else None\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_conditional_ifexp_unsafe_nonzero(self):
        source = (
            "def run():\n"
            "    d = 0\n"
            "    return 0 if d != 0 else 10 / d\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 3

    def test_conditional_ifexp_unsafe_none(self):
        source = (
            "def run():\n"
            "    x = None\n"
            "    return 'empty' if x is not None else x.attr\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 3

    def test_conditional_ifexp_unsafe_dict(self):
        source = (
            "def run():\n"
            "    d = {'a': 1}\n"
            "    return 'present' if 'missing' in d else d['missing']\n"
        )
        issues = analyze_python_source(source, "dict_app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 3

    def test_temporary_facts_do_not_leak_to_next_statement(self):
        source = (
            "def run():\n"
            "    d = 0\n"
            "    flag = (d != 0 and (10 / d > 1))\n"
            "    return 10 / d\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 4


# ============================================================================
# 31. Phase 5A Batch 2: State Invalidation
# ============================================================================


class TestStateInvalidationCompoundAndShortCircuit:
    def test_tuple_unpacking_reassigns_guarded_divisor(self):
        source = (
            "def run(d: int):\n"
            "    if d != 0:\n"
            "        d, c = 0, 1\n"
            "        return 10 / d\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 4

    def test_walrus_assignment_reassigns_guarded_divisor(self):
        source = (
            "def run(d: int):\n"
            "    if d != 0:\n"
            "        val = (d := 0) + 1\n"
            "        return 10 / d\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 4

    def test_walrus_assignment_reassigns_guarded_none(self):
        source = (
            "def run(x):\n"
            "    if x is not None:\n"
            "        val = (x := None)\n"
            "        print(x.attr)\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 4

    def test_dict_mutation_within_compound_condition(self):
        source = (
            "def run():\n"
            "    d = {'a': 1}\n"
            "    if 'k' in d and (d.clear() or d['a']):\n"
            "        pass\n"
        )
        # d.clear() mutates d, invalidating tracking, so no KeyError is incorrectly flagged
        assert analyze_python_source(source, "app.py") == []


# ============================================================================
# 32. Phase 5A Batch 3: Comprehension Scoping and Filter Guards
# ============================================================================


class TestComprehensionScopingAndFilterGuards:
    def test_listcomp_shadows_outer_zero_divisor(self):
        source = (
            "def run(items):\n"
            "    x = 0\n"
            "    res = [10 / x for x in items if x != 0]\n"
            "    return res\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_listcomp_preserves_outer_zero_after_exit(self):
        source = (
            "def run(items):\n"
            "    x = 0\n"
            "    res = [10 / x for x in items if x != 0]\n"
            "    return 10 / x\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 4

    def test_listcomp_shadowed_target_does_not_inherit_outer_guard(self):
        source = (
            "def run(x, items):\n"
            "    if x != 0:\n"
            "        return [10 / x for x in items]\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 3

    def test_setcomp_and_dictcomp_shadowing_outer_none(self):
        source = (
            "def run():\n"
            "    x = None\n"
            "    s = {x.strip() for x in ['a', 'b']}\n"
            "    d = {x: x.upper() for x in ['a', 'b']}\n"
            "    print(x.strip())\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 5

    def test_dictcomp_shadows_outer_dict(self):
        source = (
            "def run():\n"
            "    d = {'a': 1}\n"
            "    res = {k: d[k] for d in [{'b': 2}] for k in d}\n"
            "    print(d['missing'])\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM003"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 4

    def test_filter_guarded_nonzero_listcomp(self):
        source = (
            "def run(items):\n"
            "    return [10 / x for x in items if x != 0]\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_filter_guarded_gt_zero_generator(self):
        source = (
            "def run(items):\n"
            "    return list(10 / x for x in items if x > 0)\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_filter_guarded_not_none_comp(self):
        source = (
            "def run(items):\n"
            "    return [x.attr for x in items if x is not None]\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_filter_guarded_dict_key_membership(self):
        source = (
            "def run(items):\n"
            "    d = {'a': 1}\n"
            "    return [d['b'] for x in items if 'b' in d]\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_nested_comprehensions_with_shadowing_and_guards(self):
        source = (
            "def run(matrix):\n"
            "    return [[10 / x for x in row if x != 0] for row in matrix]\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_generator_expression_walrus_lazy_boundary(self):
        source = (
            "def run():\n"
            "    gen = ((w := None) for _ in [1])\n"
            "    print(w.strip())\n"
        )
        # Deferred generator does not execute walrus at definition time in enclosing scope
        # w is undefined/dynamic in enclosing scope, not statically confirmed None literal
        assert analyze_python_source(source, "app.py") == []

    def test_genuine_bugs_inside_comprehensions_detected(self):
        source = (
            "def run(items):\n"
            "    a = [10 / 0 for x in items]\n"
            "    b = [None.strip() for x in items]\n"
            "    c = [{'a': 1}['missing'] for x in items]\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 3
        codes = {iss.code for iss in issues}
        assert codes == {"SEM001", "SEM002", "SEM003"}
        for iss in issues:
            assert iss.confidence == "HIGH"
            assert iss.severity == "WARNING"


# ============================================================================
# 33. Phase 5A Batch 3: Early-Exit Guard Propagation with Continuing Else
# ============================================================================


class TestEarlyExitGuardPropagationWithContinuingElse:
    def test_sem001_if_body_exits_else_continues(self):
        source = (
            "def f(d):\n"
            "    if d == 0:\n"
            "        return 0\n"
            "    else:\n"
            "        print('valid divisor')\n"
            "    return 10 / d\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_sem002_if_body_exits_else_continues(self):
        source = (
            "def f(x=None):\n"
            "    if x is None:\n"
            "        raise ValueError('x is required')\n"
            "    else:\n"
            "        print('valid parameter')\n"
            "    return x.strip()\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_sem003_if_body_exits_else_continues(self):
        source = (
            "def f():\n"
            "    d = {'a': 1}\n"
            "    if 'b' not in d:\n"
            "        raise KeyError('missing b')\n"
            "    else:\n"
            "        print('b exists')\n"
            "    return d['b']\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_negative_control_else_reassigns_zero_invalidates_guard(self):
        source = (
            "def f(d):\n"
            "    if d == 0:\n"
            "        return 0\n"
            "    else:\n"
            "        d = 0\n"
            "    return 10 / d\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 6

    def test_negative_control_else_reassigns_none_invalidates_guard(self):
        source = (
            "def f(x=None):\n"
            "    if x is None:\n"
            "        return 0\n"
            "    else:\n"
            "        x = None\n"
            "    return x.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 6


# ============================================================================
# 34. Phase 5A Batch 3: Loop-Else State and Fact Propagation
# ============================================================================


class TestLoopElseStateAndFactPropagation:
    def test_while_else_false_condition_establishes_nonzero_fact(self):
        source = (
            "def f(d):\n"
            "    while d == 0:\n"
            "        d = 5\n"
            "    else:\n"
            "        return 10 / d\n"
        )
        assert analyze_python_source(source, "calc.py") == []

    def test_while_else_false_condition_establishes_none_fact(self):
        source = (
            "def f(x):\n"
            "    while x is not None:\n"
            "        x = x.next\n"
            "    else:\n"
            "        return x.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 5

    def test_while_else_false_condition_establishes_not_none_fact(self):
        source = (
            "def f(x):\n"
            "    while x is None:\n"
            "        x = 'ready'\n"
            "    else:\n"
            "        return x.strip()\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_for_else_loop_body_assignment_invalidates_prior_nonzero_guard(self):
        source = (
            "def f(d: int):\n"
            "    if d != 0:\n"
            "        for i in [1, 2]:\n"
            "            d = 0\n"
            "        else:\n"
            "            return 10 / d\n"
        )
        issues = analyze_python_source(source, "calc.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM001"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 6

    def test_for_else_loop_body_assignment_invalidates_prior_none_guard(self):
        source = (
            "def f(x=None):\n"
            "    if x is not None:\n"
            "        for i in [1, 2]:\n"
            "            x = None\n"
            "        else:\n"
            "            return x.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 6

    def test_while_else_loop_body_assignment_invalidates_prior_guard(self):
        source = (
            "def f(x=None, flag=True):\n"
            "    if x is not None:\n"
            "        while flag:\n"
            "            x = None\n"
            "        else:\n"
            "            return x.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "MEDIUM"
        assert issues[0].line == 6

    def test_zero_iteration_for_else_preserves_unmodified_guard(self):
        source = (
            "def f(d: int):\n"
            "    if d != 0:\n"
            "        for i in []:\n"
            "            pass\n"
            "        else:\n"
            "            return 10 / d\n"
        )
        assert analyze_python_source(source, "calc.py") == []


# ============================================================================
# 35. Phase 5A Batch 3: Severity and Verdict Policy Verification
# ============================================================================


class TestBatch3SeverityAndVerdictPolicy:
    def test_batch3_findings_severity_and_verdict(self):
        source = (
            "def run():\n"
            "    a = [10 / 0 for _ in [1]]\n"
            "    b = [None.strip() for _ in [1]]\n"
            "    c = [{'k': 1}['missing'] for _ in [1]]\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 3
        for issue in issues:
            assert issue.severity == "WARNING"
            dict_rep = issue_to_dict(issue)
            verdict = determine_review_verdict([dict_rep], [])
            assert verdict == "COMMENT"
            assert verdict != "REQUEST_CHANGES"
            assert verdict != "APPROVE"


# ============================================================================
# 36. Phase 6 Step 4: Branch-Convergence Fact Merging Analysis
# ============================================================================


class TestBranchConvergenceNoneAnalysis:
    """Regression tests for branch convergence fact merging in NoneDereferenceDetector."""

    def test_branch_convergence_both_branches_assign_none_detected_len(self):
        source = (
            "def process_record(flag: bool) -> int:\n"
            "    if flag:\n"
            "        val = None\n"
            "    else:\n"
            "        val = None\n"
            "    return len(val)\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 6

    def test_branch_convergence_both_branches_assign_none_detected_attr(self):
        source = (
            "def process_record(flag: bool) -> str:\n"
            "    if flag:\n"
            "        val = None\n"
            "    else:\n"
            "        val = None\n"
            "    return val.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 6

    def test_branch_convergence_one_branch_assigns_none_is_safe(self):
        source = (
            "def process_record(flag: bool) -> str:\n"
            "    if flag:\n"
            "        val = None\n"
            "    else:\n"
            "        val = 'non_none'\n"
            "    return val.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_branch_convergence_both_branches_assign_non_none_is_safe(self):
        source = (
            "def process_record(flag: bool) -> str:\n"
            "    if flag:\n"
            "        val = 'first'\n"
            "    else:\n"
            "        val = 'second'\n"
            "    return val.strip()\n"
        )
        assert analyze_python_source(source, "app.py") == []

    def test_branch_convergence_terminating_branch_preserves_reachable_state(self):
        source = (
            "def process_record(flag: bool) -> str:\n"
            "    if flag:\n"
            "        val = None\n"
            "    else:\n"
            "        raise ValueError('failed')\n"
            "    return val.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 6

    def test_branch_convergence_multi_branch_elif_all_assign_none(self):
        source = (
            "def process_record(kind: int) -> str:\n"
            "    if kind == 1:\n"
            "        val = None\n"
            "    elif kind == 2:\n"
            "        val = None\n"
            "    else:\n"
            "        val = None\n"
            "    return val.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 8

    def test_branch_convergence_try_except_suppresses_sem002(self):
        source = (
            "def process_record(flag: bool) -> int:\n"
            "    if flag:\n"
            "        val = None\n"
            "    else:\n"
            "        val = None\n"
            "    try:\n"
            "        return len(val)\n"
            "    except TypeError:\n"
            "        return 0\n"
        )
        assert analyze_python_source(source, "app.py") == []



# ============================================================================
# 37. Phase 6 Step 6: Interprocedural None-Return Summary Analysis
# ============================================================================


class TestInterproceduralNoneReturnAnalysis:
    """Regression tests for narrowly scoped interprocedural None-return detection."""

    def test_unconditional_return_none_attribute_dereference(self):
        """Helper unconditionally returns None -> caller attribute access flagged."""
        source = (
            "def fetch_user(uid: int):\n"
            "    return None\n"
            "\n"
            "def get_user_name(uid: int) -> str:\n"
            "    user = fetch_user(uid)\n"
            "    return user.name\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 6

    def test_bare_return_attribute_dereference(self):
        """Helper with bare return (implicit None) -> caller attribute access flagged."""
        source = (
            "def get_nothing():\n"
            "    return\n"
            "\n"
            "def process():\n"
            "    result = get_nothing()\n"
            "    return result.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 6

    def test_unconditional_return_none_subscript_dereference(self):
        """Helper returns None -> caller subscripts the result."""
        source = (
            "def fetch_config():\n"
            "    return None\n"
            "\n"
            "def get_setting():\n"
            "    cfg = fetch_config()\n"
            "    return cfg['timeout']\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 6

    def test_unconditional_return_none_len_dereference(self):
        """Helper returns None -> caller passes result to len()."""
        source = (
            "def fetch_items():\n"
            "    return None\n"
            "\n"
            "def count_items():\n"
            "    items = fetch_items()\n"
            "    return len(items)\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 6

    def test_conditional_mixed_returns_no_false_positive(self):
        """Helper sometimes returns non-None -> caller dereference is NOT flagged."""
        source = (
            "def maybe_fetch(uid: int):\n"
            "    if uid > 0:\n"
            "        return 'user_object'\n"
            "    return None\n"
            "\n"
            "def process(uid: int):\n"
            "    user = maybe_fetch(uid)\n"
            "    return user.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_non_none_return_no_false_positive(self):
        """Helper always returns non-None -> caller dereference is NOT flagged."""
        source = (
            "def make_user():\n"
            "    return {'name': 'Alice'}\n"
            "\n"
            "def process():\n"
            "    user = make_user()\n"
            "    return user['name']\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_helper_defined_after_caller(self):
        """Helper defined below the caller -> still detected (definition-order independent)."""
        source = (
            "def get_user_name(uid: int) -> str:\n"
            "    user = fetch_user(uid)\n"
            "    return user.name\n"
            "\n"
            "def fetch_user(uid: int):\n"
            "    return None\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 3

    def test_parameter_shadowing_no_false_positive(self):
        """Caller has parameter with same name as module helper -> summary not applied."""
        source = (
            "def fetch_user():\n"
            "    return None\n"
            "\n"
            "def process(fetch_user=None):\n"
            "    result = fetch_user()\n"
            "    return result.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        # fetch_user is a MAYBE_NONE parameter; calling it is a MEDIUM issue,
        # but the module-level summary should NOT apply.
        sem002_high = [i for i in issues if i.code == "SEM002" and i.confidence == "HIGH"]
        assert sem002_high == []

    def test_local_assignment_shadowing_no_false_positive(self):
        """Callee name is reassigned locally -> summary not applied."""
        source = (
            "def fetch_user():\n"
            "    return None\n"
            "\n"
            "def process():\n"
            "    fetch_user = lambda: {'name': 'Bob'}\n"
            "    result = fetch_user()\n"
            "    return result['name']\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_guard_protection_preserved(self):
        """Caller guards with 'if user is None' -> no finding after guard."""
        source = (
            "def fetch_user(uid: int):\n"
            "    return None\n"
            "\n"
            "def get_user_name(uid: int) -> str:\n"
            "    user = fetch_user(uid)\n"
            "    if user is None:\n"
            "        return 'unknown'\n"
            "    return user.name\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_exception_handler_suppression(self):
        """Caller wraps dereference in try/except AttributeError -> suppressed."""
        source = (
            "def fetch_user(uid: int):\n"
            "    return None\n"
            "\n"
            "def get_user_name(uid: int) -> str:\n"
            "    user = fetch_user(uid)\n"
            "    try:\n"
            "        return user.name\n"
            "    except AttributeError:\n"
            "        return 'unknown'\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_decorated_function_not_summarized(self):
        """Decorated function is UNKNOWN -> caller dereference not flagged."""
        source = (
            "def my_decorator(f):\n"
            "    return f\n"
            "\n"
            "@my_decorator\n"
            "def fetch_user():\n"
            "    return None\n"
            "\n"
            "def process():\n"
            "    user = fetch_user()\n"
            "    return user.name\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_generator_function_not_summarized(self):
        """Generator function is not classified as returns-None."""
        source = (
            "def gen_items():\n"
            "    yield None\n"
            "    return None\n"
            "\n"
            "def process():\n"
            "    result = gen_items()\n"
            "    return result.send(None)\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_no_explicit_return_not_summarized(self):
        """Function with no explicit return is not classified (stub/hook pattern)."""
        source = (
            "def setup():\n"
            "    pass\n"
            "\n"
            "def process():\n"
            "    result = setup()\n"
            "    return result.name\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert issues == []

    def test_multiple_return_none_branches_all_none(self):
        """Function with multiple branches all returning None -> still summarized."""
        source = (
            "def lookup(key: str):\n"
            "    if key == 'a':\n"
            "        return None\n"
            "    elif key == 'b':\n"
            "        return None\n"
            "    else:\n"
            "        return\n"
            "\n"
            "def process():\n"
            "    val = lookup('x')\n"
            "    return val.strip()\n"
        )
        issues = analyze_python_source(source, "app.py")
        assert len(issues) == 1
        assert issues[0].code == "SEM002"
        assert issues[0].confidence == "HIGH"
        assert issues[0].line == 11
