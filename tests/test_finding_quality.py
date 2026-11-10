"""
tests/test_finding_quality.py - Comprehensive Unit & Integration Tests for Milestone 6.

Validates:
  1. Structured severity levels: Critical, High, Medium, Low (SeverityLevel enum & normalize_severity).
  2. Confidence score calculation (0.0 to 1.0) and categorical synchronization.
  3. Structured 4-part finding explanation (what, why, evidence, suggested fix).
  4. Deduplication of overlapping findings across multiple analysis agents without evidence loss.
  5. GitHub Actions Step Summary grouping by severity.
  6. Backward compatibility for legacy Issue, ReviewComment, and ActionPlan constructs.
"""

from __future__ import annotations

import pytest

from approval_gate import ActionPlan, ReviewComment
from github_review_workflow import (
    PRReviewResult,
    are_findings_overlapping,
    build_inline_comments,
    deduplicate_findings,
    merge_two_findings,
    normalize_finding_rel_path,
)
from main import format_github_step_summary
from state import dict_to_issue, issue_to_dict
from tools import (
    Issue,
    SeverityLevel,
    format_finding_explanation,
    normalize_severity,
    parse_bandit_json,
    parse_ruff_json,
)

# ============================================================================
# 1. Structured Severity Levels Tests
# ============================================================================


class TestStructuredSeverityLevels:
    """Verifies canonical severity levels: Critical, High, Medium, Low."""

    def test_canonical_severity_enum_values(self):
        assert SeverityLevel.CRITICAL.value == "Critical"
        assert SeverityLevel.HIGH.value == "High"
        assert SeverityLevel.MEDIUM.value == "Medium"
        assert SeverityLevel.LOW.value == "Low"

    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            ("CRITICAL", "Critical"),
            ("critical", "Critical"),
            ("FATAL", "Critical"),
            ("blocker", "Critical"),
            (SeverityLevel.CRITICAL, "Critical"),
            ("HIGH", "High"),
            ("high", "High"),
            ("ERROR", "High"),
            ("error", "High"),
            (SeverityLevel.HIGH, "High"),
            ("MEDIUM", "Medium"),
            ("medium", "Medium"),
            ("WARNING", "Medium"),
            ("warn", "Medium"),
            ("moderate", "Medium"),
            (SeverityLevel.MEDIUM, "Medium"),
            ("LOW", "Low"),
            ("low", "Low"),
            ("INFO", "Low"),
            ("note", "Low"),
            ("informational", "Low"),
            (SeverityLevel.LOW, "Low"),
            ("UNKNOWN_LABEL", "Medium"),
            ("", "Medium"),
            (None, "Medium"),
        ],
    )
    def test_normalize_severity_mapping(self, input_val, expected):
        assert normalize_severity(input_val) == expected

    def test_issue_severity_level_property(self):
        issue_crit = Issue(code="C1", message="m", file_path="f.py", severity="CRITICAL")
        assert issue_crit.severity_level == SeverityLevel.CRITICAL
        assert issue_crit.severity_level.value == "Critical"

        issue_warn = Issue(code="W1", message="m", file_path="f.py", severity="WARNING")
        assert issue_warn.severity_level == SeverityLevel.MEDIUM

        issue_err = Issue(code="E1", message="m", file_path="f.py", severity="ERROR")
        assert issue_warn.severity_level == SeverityLevel.MEDIUM
        assert issue_err.severity_level == SeverityLevel.HIGH


# ============================================================================
# 2. Confidence Score (0 to 1) Tests
# ============================================================================


class TestConfidenceScore:
    """Verifies numeric confidence score between 0.0 and 1.0."""

    def test_default_confidence_score_for_high(self):
        issue = Issue(code="B101", message="m", file_path="a.py", confidence="HIGH")
        assert issue.confidence_score == 0.9
        assert 0.0 <= issue.confidence_score <= 1.0

    def test_confidence_score_for_medium_and_low(self):
        issue_med = Issue(code="B105", message="m", file_path="a.py", confidence="MEDIUM")
        assert issue_med.confidence_score == 0.6

        issue_low = Issue(code="B104", message="m", file_path="a.py", confidence="LOW")
        assert issue_low.confidence_score == 0.3

    def test_explicit_numeric_confidence_score(self):
        issue = Issue(code="S01", message="m", file_path="a.py", confidence_score=0.82)
        assert issue.confidence_score == 0.82

    def test_confidence_score_clamping(self):
        issue_high = Issue(code="S01", message="m", file_path="a.py", confidence_score=1.5)
        assert issue_high.confidence_score == 1.0

        issue_neg = Issue(code="S01", message="m", file_path="a.py", confidence_score=-0.5)
        assert issue_neg.confidence_score == 0.0

    def test_bandit_parser_populates_confidence_score(self):
        sample_json = """{
            "results": [
                {
                    "test_id": "B101",
                    "issue_text": "assert used",
                    "filename": "app.py",
                    "line_number": 12,
                    "col_offset": 4,
                    "issue_severity": "LOW",
                    "issue_confidence": "HIGH"
                },
                {
                    "test_id": "B105",
                    "issue_text": "hardcoded password",
                    "filename": "app.py",
                    "line_number": 20,
                    "col_offset": 8,
                    "issue_severity": "MEDIUM",
                    "issue_confidence": "LOW"
                }
            ]
        }"""
        issues = parse_bandit_json(sample_json)
        assert len(issues) == 2
        assert issues[0].confidence == "HIGH"
        assert issues[0].confidence_score == 0.9
        assert issues[1].confidence == "LOW"
        assert issues[1].confidence_score == 0.3

    def test_ruff_parser_populates_high_confidence_score(self):
        sample_json = """[
            {
                "code": "F401",
                "message": "`os` imported but unused",
                "filename": "app.py",
                "location": {"row": 1, "column": 1},
                "fix": {"applicability": "safe"}
            }
        ]"""
        issues = parse_ruff_json(sample_json)
        assert len(issues) == 1
        assert issues[0].confidence_score == 0.95
        assert issues[0].fixable is True


# ============================================================================
# 3. 4-Part Finding Explanation Tests
# ============================================================================


class TestFindingExplanation:
    """Verifies: What, Why it matters, Evidence, Practical Suggested Fix."""

    def test_issue_populates_explanation_defaults(self):
        issue = Issue(
            code="B602",
            message="subprocess call with shell=True",
            file_path="service.py",
            line=45,
            column=8,
            severity="HIGH",
            explanation="Invoking shell=True allows arbitrary command execution via injection.",
            suggested_fix="Use a list of arguments and set shell=False.",
        )
        assert issue.what == "subprocess call with shell=True"
        assert "shell=True allows arbitrary command execution" in issue.why
        assert issue.evidence == "service.py:45:8"
        assert issue.suggested_fix == "Use a list of arguments and set shell=False."

        formatted = issue.format_explanation()
        assert "**What:**" in formatted
        assert "**Why it matters:**" in formatted
        assert "**Evidence:**" in formatted
        assert "**Suggested Fix:**" in formatted

    def test_format_finding_explanation_helper_with_dict(self):
        finding_dict = {
            "code": "SEM001",
            "message": "Division by zero",
            "file_path": "math.py",
            "line": 10,
            "what": "Denominator evaluated to constant 0",
            "why": "Unconditionally raises ZeroDivisionError at runtime.",
            "evidence": "math.py:10",
            "suggested_fix": "Add a zero-check guard before dividing.",
        }
        text = format_finding_explanation(finding_dict)
        assert "**What:** Denominator evaluated to constant 0" in text
        assert "**Why it matters:** Unconditionally raises ZeroDivisionError" in text
        assert "**Evidence:** math.py:10" in text
        assert "**Suggested Fix:** Add a zero-check guard" in text


# ============================================================================
# 4. Deduplication of Overlapping Findings Tests
# ============================================================================


class TestFindingDeduplication:
    """Verifies multi-agent deduplication without discarding evidence."""

    def test_are_findings_overlapping_different_files(self):
        a = {"file_path": "src/a.py", "line": 10, "code": "B101"}
        b = {"file_path": "src/b.py", "line": 10, "code": "B101"}
        assert are_findings_overlapping(a, b) is False

    def test_are_findings_overlapping_different_lines(self):
        a = {"file_path": "src/a.py", "line": 10, "code": "B101"}
        b = {"file_path": "src/a.py", "line": 50, "code": "B101"}
        assert are_findings_overlapping(a, b) is False

    def test_are_findings_overlapping_same_line_different_tools(self):
        # Bandit and Ruff reporting on the same line
        bandit_finding = {
            "file_path": "app.py",
            "line": 15,
            "tool": "bandit",
            "code": "B101",
            "message": "assert used",
        }
        ruff_finding = {
            "file_path": "app.py",
            "line": 15,
            "tool": "ruff",
            "code": "S101",
            "message": "Use of assert detected",
        }
        assert are_findings_overlapping(bandit_finding, ruff_finding) is True

    def test_are_findings_overlapping_rule_equivalence(self):
        # B307 (eval) and S307 (eval) across tools
        a = {"file_path": "app.py", "line": 20, "tool": "bandit", "code": "B307"}
        b = {"file_path": "app.py", "line": 20, "tool": "ruff", "code": "S307"}
        assert are_findings_overlapping(a, b) is True

    def test_merge_two_findings_preserves_all_evidence(self):
        f1 = {
            "code": "B101",
            "tool": "bandit",
            "file_path": "auth.py",
            "line": 30,
            "column": 4,
            "severity": "LOW",
            "confidence_score": 0.8,
            "message": "Bandit: assert used in production",
            "what": "Assert statement found in auth check",
            "why": "Assertions are removed in optimized bytecode (-O).",
            "evidence": "auth.py:30:4\n```python\nassert user.is_authenticated\n```",
            "suggested_fix": "Raise an explicit AuthenticationError instead.",
            "fixable": False,
        }
        f2 = {
            "code": "S101",
            "tool": "ruff",
            "file_path": "auth.py",
            "line": 30,
            "column": 4,
            "severity": "MEDIUM",
            "confidence_score": 0.95,
            "message": "Ruff: assert detected (S101)",
            "what": "Assert statement violation",
            "why": "Flake8-bandit forbids assert in release builds.",
            "evidence": "auth.py:30",
            "suggested_fix": "Use proper error handling.",
            "fixable": True,
        }

        merged = merge_two_findings(f1, f2)

        # 1. Severity escalated to highest (MEDIUM > LOW)
        assert merged["severity_level"] == "Medium"

        # 2. Confidence taken as maximum
        assert merged["confidence_score"] == 0.95
        assert merged["confidence"] == "HIGH"

        # 3. Both tools recorded
        assert "bandit" in merged["tool"]
        assert "ruff" in merged["tool"]

        # 4. Both rule codes recorded
        assert "B101" in merged["code"]
        assert "S101" in merged["code"]

        # 5. Fixable preserved as True if either is fixable
        assert merged["fixable"] is True

        # 6. Evidence preserved from both without loss
        assert "auth.py:30:4" in merged["evidence"]
        assert "assert user.is_authenticated" in merged["evidence"]

        # 7. Rationales and suggestions preserved
        assert "Assertions are removed" in merged["why"]
        assert "Flake8-bandit forbids" in merged["why"]
        assert "Raise an explicit AuthenticationError" in merged["suggested_fix"]

    def test_deduplicate_findings_empty_list(self):
        assert deduplicate_findings([]) == []

    def test_deduplicate_findings_combines_overlapping_and_keeps_distinct(self):
        findings = [
            # Finding 1 & 2 overlap at calc.py:10
            {
                "file_path": "calc.py",
                "line": 10,
                "tool": "bandit",
                "code": "B101",
                "severity": "LOW",
                "message": "assert used",
            },
            {
                "file_path": "calc.py",
                "line": 10,
                "tool": "ruff",
                "code": "S101",
                "severity": "MEDIUM",
                "message": "assert statement",
            },
            # Finding 3 is distinct at calc.py:25
            {
                "file_path": "calc.py",
                "line": 25,
                "tool": "semantic_analyzer",
                "code": "SEM001",
                "severity": "HIGH",
                "message": "Division by zero literal",
            },
        ]

        deduped = deduplicate_findings(findings)
        assert len(deduped) == 2

        line_10 = next(d for d in deduped if d["line"] == 10)
        line_25 = next(d for d in deduped if d["line"] == 25)

        assert "bandit" in line_10["tool"] and "ruff" in line_10["tool"]
        assert line_25["code"] == "SEM001"
        assert line_25["tool"] == "semantic_analyzer"

    def test_different_tools_unrelated_issues_same_line_remain_separate(self):
        # 1. Bandit assert (B101) vs Ruff line too long (E501) on the same line
        f_bandit = {
            "file_path": "app.py",
            "line": 15,
            "tool": "bandit",
            "code": "B101",
            "message": "Use of assert detected. The assert statement will be removed in production.",
        }
        f_ruff_line_length = {
            "file_path": "app.py",
            "line": 15,
            "tool": "ruff",
            "code": "E501",
            "message": "Line too long (95 > 88 characters)",
        }
        assert are_findings_overlapping(f_bandit, f_ruff_line_length) is False

        # 2. Bandit shell=True (B602) vs Ruff unused variable (F841) on the same line
        f_shell = {
            "file_path": "worker.py",
            "line": 42,
            "tool": "bandit",
            "code": "B602",
            "message": "subprocess call with shell=True identified, security issue.",
        }
        f_unused = {
            "file_path": "worker.py",
            "line": 42,
            "tool": "ruff",
            "code": "F841",
            "message": "Local variable 'proc' is assigned to but never used",
        }
        assert are_findings_overlapping(f_shell, f_unused) is False

        # 3. Semantic division by zero vs Ruff trailing whitespace on the same line
        f_sem = {
            "file_path": "calc.py",
            "line": 20,
            "tool": "semantic_analyzer",
            "code": "SEM001",
            "message": "Possible division by zero: divisor '0'",
        }
        f_ws = {
            "file_path": "calc.py",
            "line": 20,
            "tool": "ruff",
            "code": "W291",
            "message": "Trailing whitespace",
        }
        assert are_findings_overlapping(f_sem, f_ws) is False

        # 4. Ruff undefined name vs Bandit hardcoded password with same token name
        f_undefined = {
            "file_path": "auth.py",
            "line": 5,
            "tool": "ruff",
            "code": "F821",
            "message": "Undefined name 'secret_token'",
        }
        f_secret = {
            "file_path": "auth.py",
            "line": 5,
            "tool": "bandit",
            "code": "B105",
            "message": "Possible hardcoded password: 'secret_token'",
        }
        assert are_findings_overlapping(f_undefined, f_secret) is False

    def test_deduplicate_findings_keeps_unrelated_same_line_issues_separate(self):
        findings = [
            {
                "file_path": "main.py",
                "line": 10,
                "tool": "bandit",
                "code": "B101",
                "severity": "LOW",
                "message": "assert used in production",
            },
            {
                "file_path": "main.py",
                "line": 10,
                "tool": "ruff",
                "code": "E501",
                "severity": "LOW",
                "message": "Line too long (99 > 88)",
            },
        ]
        deduped = deduplicate_findings(findings)
        assert len(deduped) == 2
        assert {d["code"] for d in deduped} == {"B101", "E501"}

    def test_cross_tool_same_issue_without_mapped_rule_merges(self):
        # Semantic analyzer and another scanner reporting division by zero without equivalent rule codes
        sem_finding = {
            "file_path": "calc.py",
            "line": 33,
            "tool": "semantic_analyzer",
            "code": "SEM001",
            "message": "Possible division by zero: divisor 'den'",
        }
        custom_finding = {
            "file_path": "calc.py",
            "line": 33,
            "tool": "custom_checker",
            "code": "DIV99",
            "message": "Division by zero in calculation",
        }
        assert are_findings_overlapping(sem_finding, custom_finding) is True

        merged = deduplicate_findings([sem_finding, custom_finding])
        assert len(merged) == 1
        assert "semantic_analyzer" in merged[0]["tool"]
        assert "custom_checker" in merged[0]["tool"]

    def test_cross_tool_explicit_issue_identity_matching_and_differing(self):
        matching_a = {
            "file_path": "db.py",
            "line": 50,
            "tool": "scanner_a",
            "code": "RES01",
            "issue_identity": "db_connection_leak",
            "message": "Connection not closed",
        }
        matching_b = {
            "file_path": "db.py",
            "line": 50,
            "tool": "scanner_b",
            "code": "CHK99",
            "issue_identity": "db_connection_leak",
            "message": "Resource leak detected",
        }
        assert are_findings_overlapping(matching_a, matching_b) is True

        differing_b = {
            "file_path": "db.py",
            "line": 50,
            "tool": "scanner_b",
            "code": "CHK99",
            "issue_identity": "sql_injection",
            "message": "Resource leak detected",
        }
        assert are_findings_overlapping(matching_a, differing_b) is False

    def test_file_level_findings_identical_non_empty_rule_codes_merge(self):
        f1 = {"file_path": "lib.py", "line": None, "code": "F401", "message": "unused import"}
        f2 = {"file_path": "lib.py", "line": None, "code": "F401", "message": "unused import"}
        assert are_findings_overlapping(f1, f2) is True
        deduped = deduplicate_findings([f1, f2])
        assert len(deduped) == 1
        assert deduped[0]["code"] == "F401"

    def test_file_level_findings_mapped_equivalent_codes_merge(self):
        f1 = {"file_path": "lib.py", "line": None, "code": "B101", "tool": "bandit", "message": "assert used"}
        f2 = {"file_path": "lib.py", "line": None, "code": "S101", "tool": "ruff", "message": "assert detected"}
        assert are_findings_overlapping(f1, f2) is True
        deduped = deduplicate_findings([f1, f2])
        assert len(deduped) == 1
        assert "bandit" in deduped[0]["tool"] and "ruff" in deduped[0]["tool"]

    def test_file_level_findings_unrelated_codes_and_descriptions_remain_separate(self):
        f1 = {"file_path": "lib.py", "line": None, "code": "B101", "tool": "bandit", "message": "assert used"}
        f2 = {"file_path": "lib.py", "line": None, "code": "E501", "tool": "ruff", "message": "line too long"}
        assert are_findings_overlapping(f1, f2) is False
        deduped = deduplicate_findings([f1, f2])
        assert len(deduped) == 2

    def test_file_level_findings_same_substantive_description_merge(self):
        f1 = {
            "file_path": "lib.py",
            "line": None,
            "code": "SEM001",
            "tool": "semantic",
            "message": "Possible division by zero: divisor 'den'",
        }
        f2 = {
            "file_path": "lib.py",
            "line": None,
            "code": "DIV99",
            "tool": "checker",
            "message": "Division by zero in calculation",
        }
        assert are_findings_overlapping(f1, f2) is True
        deduped = deduplicate_findings([f1, f2])
        assert len(deduped) == 1
        assert "semantic" in deduped[0]["tool"] and "checker" in deduped[0]["tool"]

    def test_file_level_findings_empty_codes_and_empty_descriptions_remain_separate(self):
        f1 = {"file_path": "lib.py", "line": None, "code": "", "message": ""}
        f2 = {"file_path": "lib.py", "line": None, "code": "", "message": ""}
        assert are_findings_overlapping(f1, f2) is False
        deduped = deduplicate_findings([f1, f2])
        assert len(deduped) == 2

    def test_file_level_findings_missing_lines_no_unbound_local_error(self):
        # Explicit test confirming no UnboundLocalError occurs when both line numbers are missing / omitted
        f1 = {"file_path": "calc.py", "code": "B101", "message": "assert used"}
        f2 = {"file_path": "calc.py", "code": "S101", "message": "assert detected"}
        # Both lack 'line' key entirely
        assert are_findings_overlapping(f1, f2) is True

    @pytest.mark.parametrize(
        "bad_score,expected_score",
        [
            (float("nan"), 0.9),  # Falls back to categorical HIGH (0.9)
            (float("inf"), 0.9),  # Falls back to categorical HIGH (0.9)
            (float("-inf"), 0.9),  # Falls back to categorical HIGH (0.9)
            ("nan", 0.9),
            ("inf", 0.9),
            ("not_a_number", 0.9),
            (None, 0.9),
            (1.75, 1.0),  # Clamped to 1.0
            (-0.35, 0.0),  # Clamped to 0.0
            ([], 0.9),
            ({}, 0.9),
        ],
    )
    def test_merge_two_findings_handles_malformed_and_non_finite_confidence(
        self, bad_score, expected_score
    ):
        f1 = {
            "code": "B101",
            "tool": "bandit",
            "file_path": "calc.py",
            "line": 10,
            "confidence": "HIGH",
            "confidence_score": bad_score,
            "message": "assert used",
        }
        f2 = {
            "code": "S101",
            "tool": "ruff",
            "file_path": "calc.py",
            "line": 10,
            "confidence": "MEDIUM",
            "confidence_score": 0.6,
            "message": "assert statement",
        }
        merged = merge_two_findings(f1, f2)
        import math
        assert math.isfinite(merged["confidence_score"])
        assert 0.0 <= merged["confidence_score"] <= 1.0
        assert merged["confidence_score"] == pytest.approx(max(expected_score, 0.6))
        assert merged["confidence"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")


# ============================================================================
# 5. GitHub Actions Step Summary Grouping by Severity Tests
# ============================================================================


class TestStepSummaryGroupingBySeverity:
    """Verifies that the Step Summary renders a breakdown and groups inline comments by severity."""

    def test_step_summary_groups_comments_by_severity(self):
        c_crit = ReviewComment(
            path="auth.py",
            line=12,
            body="**[Bandit - B602]** (CRITICAL)\nSubprocess execution with shell=True",
            severity="CRITICAL",
        )
        c_high = ReviewComment(
            path="math_utils.py",
            line=40,
            body="**[Semantic - SEM001]** (HIGH)\nDivision by zero literal",
            severity="HIGH",
        )
        c_med = ReviewComment(
            path="app.py",
            line=22,
            body="**[Ruff - F821]** (MEDIUM)\nUndefined variable used",
            severity="MEDIUM",
        )
        c_low = ReviewComment(
            path="views.py",
            line=5,
            body="**[Ruff - W291]** (LOW)\nTrailing whitespace",
            severity="LOW",
        )

        plan = ActionPlan(
            owner="org",
            repo="repo",
            pr_number=50,
            head_sha="abcdef1234567890abcdef1234567890abcdef12",
            verdict="REQUEST_CHANGES",
            summary="Review complete.",
            comments=(c_crit, c_high, c_med, c_low),
            dry_run=True,
            created_at_mono=100.0,
        )

        res = PRReviewResult(
            owner="org",
            repo="repo",
            pr_number=50,
            head_sha="abcdef1234567890abcdef1234567890abcdef12",
            action_plan=plan,
            preview_text="preview",
            report={},
            review_comments=(c_crit, c_high, c_med, c_low),
            changed_files=("auth.py", "math_utils.py", "app.py", "views.py"),
            workflow_state={},
            success=True,
            summary="Review complete.",
        )

        md = format_github_step_summary(res)

        # 1. Severity summary table presence
        assert "### 📊 Findings by Severity" in md
        assert "| Severity Tier | Findings Count | Status |" in md
        assert "🔴 **Critical**" in md
        assert "🟠 **High**" in md
        assert "🟡 **Medium**" in md
        assert "🔵 **Low**" in md

        # 2. Comments grouped under tier headers
        assert "#### 🔴 Critical Severity (1)" in md
        assert "#### 🟠 High Severity (1)" in md
        assert "#### 🟡 Medium Severity (1)" in md
        assert "#### 🔵 Low Severity (1)" in md

        # 3. All files present in tables
        assert "`auth.py`" in md
        assert "`math_utils.py`" in md
        assert "`app.py`" in md
        assert "`views.py`" in md


# ============================================================================
# 6. Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """Verifies that legacy Issue, ReviewComment, and state helpers remain fully functional."""

    def test_legacy_issue_construction(self):
        # Positional arguments matching legacy signature
        issue = Issue("F401", "unused import", "calc.py", 1, 1, "WARNING", "ruff", True)
        assert issue.code == "F401"
        assert issue.severity == "WARNING"
        assert issue.severity_level == SeverityLevel.MEDIUM
        assert issue.confidence == "HIGH"
        assert issue.confidence_score == 0.9

    def test_issue_serialization_roundtrip(self):
        original = Issue(
            code="B307",
            message="eval used",
            file_path="exec.py",
            line=5,
            column=2,
            severity="HIGH",
            tool="bandit",
            confidence="HIGH",
            confidence_score=0.92,
            what="Dangerous dynamic evaluation",
            why="Untrusted input passed to eval may lead to remote code execution.",
            evidence="exec.py:5:2",
            suggested_fix="Use ast.literal_eval for safe literal parsing.",
        )
        d = issue_to_dict(original)
        assert d["severity_level"] == "High"
        assert d["confidence_score"] == 0.92

        reconstructed = dict_to_issue(d)
        assert reconstructed.code == original.code
        assert reconstructed.severity == original.severity
        assert reconstructed.confidence_score == 0.92
        assert reconstructed.what == original.what
        assert reconstructed.why == original.why
        assert reconstructed.evidence == original.evidence
        assert reconstructed.suggested_fix == original.suggested_fix

    def test_legacy_review_comment_without_severity_field(self):
        rc = ReviewComment(
            path="main.py",
            line=10,
            body="Review note",
            side="RIGHT",
        )
        assert rc.severity is None
        d = rc.to_dict()
        assert d["path"] == "main.py"
        assert d["line"] == 10
        gh_payload = rc.to_github_payload()
        assert "severity" not in gh_payload
        assert gh_payload["line"] == 10


# ============================================================================
# 7. Path Traversal & Normalization Regression Tests (normalize_finding_rel_path)
# ============================================================================


class TestNormalizeFindingRelPath:
    """Verifies strict path traversal defense, normalization, and resolution."""

    def test_valid_relative_nested_paths(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        assert normalize_finding_rel_path("calc.py", ws) == "calc.py"
        assert normalize_finding_rel_path("src/calc.py", ws) == "src/calc.py"
        assert normalize_finding_rel_path("src/sub/nested/file.py", ws) == "src/sub/nested/file.py"
        assert normalize_finding_rel_path("./src/calc.py", ws) == "src/calc.py"
        assert normalize_finding_rel_path("src/./calc.py", ws) == "src/calc.py"

    def test_windows_style_backslashes(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        assert normalize_finding_rel_path(r"src\calc.py", ws) == "src/calc.py"
        assert normalize_finding_rel_path(r"src\sub\nested\file.py", ws) == "src/sub/nested/file.py"
        assert normalize_finding_rel_path(r"src/sub\nested/file.py", ws) == "src/sub/nested/file.py"

    def test_valid_internal_relative_dotdot(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        # Internal .. that does not escape the workspace
        assert normalize_finding_rel_path("src/sub/../calc.py", ws) == "src/calc.py"

    @pytest.mark.parametrize(
        "bad_path",
        [
            "../outside.py",
            r"..\outside.py",
            "src/../../outside.py",
            r"src\..\..\outside.py",
            "sub/../../../outside.py",
            "./../outside.py",
            "src/../..",
        ],
    )
    def test_rejection_of_path_traversal_attempts(self, tmp_path, bad_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        with pytest.raises(ValueError, match="escapes workspace root"):
            normalize_finding_rel_path(bad_path, ws)

    def test_absolute_path_inside_workspace(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        target = ws / "src" / "calc.py"

        assert normalize_finding_rel_path(str(target), ws) == "src/calc.py"

    def test_rejection_of_absolute_path_outside_workspace(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outside = tmp_path / "outside_dir" / "secret.py"

        with pytest.raises(ValueError, match="escapes workspace root|outside workspace root"):
            normalize_finding_rel_path(str(outside), ws)

    @pytest.mark.parametrize("root_rel", ["/outside.py", r"\outside.py"])
    def test_rejection_of_root_relative_paths(self, tmp_path, root_rel):
        ws = tmp_path / "workspace"
        ws.mkdir()

        with pytest.raises(ValueError, match="escapes workspace root"):
            normalize_finding_rel_path(root_rel, ws)

    @pytest.mark.parametrize("empty_val", ["", "   ", "\t\n"])
    def test_rejection_of_empty_and_whitespace_paths(self, tmp_path, empty_val):
        ws = tmp_path / "workspace"
        ws.mkdir()

        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_finding_rel_path(empty_val, ws)

    @pytest.mark.parametrize("root_val", [".", "./", r".\\"])
    def test_rejection_of_workspace_root_itself(self, tmp_path, root_val):
        ws = tmp_path / "workspace"
        ws.mkdir()

        with pytest.raises(ValueError, match="resolves to workspace root itself"):
            normalize_finding_rel_path(root_val, ws)

    def test_rejection_of_null_byte_paths(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        with pytest.raises(ValueError, match="contains null bytes"):
            normalize_finding_rel_path("calc.py\0.txt", ws)

    def test_symlink_pointing_outside_workspace_rejected(self, tmp_path):
        import os
        ws = tmp_path / "workspace"
        ws.mkdir()
        secret_dir = tmp_path / "secret_dir"
        secret_dir.mkdir()
        secret_file = secret_dir / "passwords.txt"
        secret_file.write_text("secret_tokens")

        symlink_path = ws / "symlink_dir"
        try:
            os.symlink(str(secret_dir), str(symlink_path))
        except (OSError, NotImplementedError):
            pytest.skip("Symlink creation not permitted without elevated permissions")

        with pytest.raises(ValueError, match="escapes workspace root"):
            normalize_finding_rel_path("symlink_dir/passwords.txt", ws)

    def test_build_inline_comments_routes_traversal_to_unmapped(self, tmp_path):
        from github_client import DiffFile, DiffHunk, DiffLine
        ws = tmp_path / "workspace"
        ws.mkdir()

        hunk = DiffHunk(
            header="@@ -1,5 +1,5 @@",
            old_start=1,
            old_count=5,
            new_start=1,
            new_count=5,
            section_heading="def foo():",
            lines=[
                DiffLine("context", "def foo():", 1, 1),
                DiffLine("added", "    return 1", None, 2),
            ],
        )
        diff_file = DiffFile(
            old_path="calc.py",
            new_path="calc.py",
            status="modified",
            hunks=[hunk],
        )

        issues = [
            # Traversal path must never anchor to outside.py!
            {
                "code": "TRV01",
                "message": "Path traversal attempt",
                "file_path": "../outside.py",
                "line": 2,
                "tool": "bandit",
                "severity": "HIGH",
            },
            # Valid path
            {
                "code": "VAL01",
                "message": "Valid finding",
                "file_path": "calc.py",
                "line": 2,
                "tool": "ruff",
                "severity": "LOW",
            },
        ]

        diff_map = {"calc.py": diff_file}
        comments, unmapped = build_inline_comments(issues, diff_map, ws)

        assert len(comments) == 1
        assert comments[0].path == "calc.py"

        assert len(unmapped) == 1
        assert unmapped[0]["code"] == "TRV01"

