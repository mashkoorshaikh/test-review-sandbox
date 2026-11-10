"""
tests/test_evaluation_framework.py - Tests for Milestone 7 Stage 7.1 Evaluation Framework.

Validates:
  1. Evaluation dataset integrity (34 cases in benchmark/eval_manifest.json).
  2. Ground-truth matching with rule code, line tolerance, tool, and file path.
  3. Duplicate extra findings counted as False Positives.
  4. Safe division and edge cases in Precision, Recall, and F1 calculations.
  5. Category-level metrics breakdown (precision, recall, f1_score).
  6. Machine-readable JSON evaluation report schema and case results.
  7. Source file immutability (SHA-256 hash checks before and after evaluation).
  8. Deterministic evaluation execution on error_handling dataset.
  9. Runner mode routing (deterministic vs full multi-agent) and deduplication.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark.metrics import (
    ExpectedFinding,
    MatchResult,
    calculate_f1,
    calculate_precision,
    calculate_recall,
    classify_detector_name,
    compute_benchmark_summary,
    export_summary_json,
    match_findings,
    safe_div,
)
from benchmark.runner import (
    load_manifest,
    run_evaluation,
)


@pytest.fixture
def eval_manifest_path() -> str:
    """Return absolute path to the evaluation manifest."""
    root = Path(__file__).resolve().parent.parent
    return str(root / "benchmark" / "eval_manifest.json")


class TestEvaluationDatasetIntegrity:
    """Validate eval_manifest schema, case counts, and test files on disk."""

    def test_eval_manifest_loads_and_has_all_34_cases(self, eval_manifest_path: str):
        cases = load_manifest(eval_manifest_path)
        assert len(cases) == 34, f"Expected 34 evaluation cases, found {len(cases)}"

        baseline = [c for c in cases if c.get("suite") == "baseline"]
        expanded = [c for c in cases if c.get("suite") == "expanded"]
        evaluation = [c for c in cases if c.get("suite") == "evaluation"]

        assert len(baseline) == 15, f"Expected 15 baseline cases, got {len(baseline)}"
        assert len(expanded) == 15, f"Expected 15 expanded cases, got {len(expanded)}"
        assert len(evaluation) == 4, f"Expected 4 evaluation cases, got {len(evaluation)}"

        valid_categories = {
            "clean",
            "bugs",
            "security",
            "multi",
            "safe_edges",
            "error_handling",
        }
        seen_ids: set[str] = set()

        root = Path(__file__).resolve().parent.parent
        for c in cases:
            cid = c.get("case_id")
            assert cid, "Case missing 'case_id'"
            assert cid not in seen_ids, f"Duplicate case ID '{cid}'"
            seen_ids.add(cid)

            cat = c.get("category")
            assert cat in valid_categories, f"Invalid category '{cat}' for case '{cid}'"

            fpath = c.get("file_path")
            assert fpath, f"Case '{cid}' missing 'file_path'"
            full_path = root / fpath
            assert full_path.is_file(), f"Case file does not exist on disk: {full_path}"

            expected = c.get("expected_findings", [])
            for exp in expected:
                assert "rule" in exp, f"Expected finding missing 'rule' in case '{cid}'"
                assert "line" in exp or exp.get("line") is None

    def test_error_handling_cases_structure(self, eval_manifest_path: str):
        cases = load_manifest(eval_manifest_path)
        err_cases = {c["case_id"]: c for c in cases if c.get("category") == "error_handling"}

        assert len(err_cases) == 4, "Expected exactly 4 error_handling cases"
        assert set(err_cases.keys()) == {"err_01", "err_02", "err_03", "err_04"}

        # err_01: bare except (E722)
        assert len(err_cases["err_01"]["expected_findings"]) == 1
        assert err_cases["err_01"]["expected_findings"][0]["rule"] == "E722"

        # err_02: silent pass (B110, S110, BLE001)
        err_02_rules = {ef["rule"] for ef in err_cases["err_02"]["expected_findings"]}
        assert err_02_rules == {"B110", "S110", "BLE001"}

        # err_03: return in finally (B012, SIM107)
        err_03_rules = {ef["rule"] for ef in err_cases["err_03"]["expected_findings"]}
        assert err_03_rules == {"B012", "SIM107"}

        # err_04: safe exception handling (clean, 0 findings)
        assert len(err_cases["err_04"]["expected_findings"]) == 0


class TestGroundTruthMatchingAndTolerance:
    """Verify rule code, line tolerance, tool, and file path matching."""

    def test_matching_with_file_path_constraint(self):
        expected = [
            ExpectedFinding(rule="SEM001", line=10, file_path="module/math_utils.py"),
        ]
        # Same rule and line, but different file path -> No match
        actual_other_file = [
            {"code": "SEM001", "line": 10, "file_path": "other/different.py", "tool": "semantic_analyzer"}
        ]
        tp, fp, fn, _, _, _ = match_findings(expected, actual_other_file)
        assert tp == 0
        assert fp == 1
        assert fn == 1

        # Same rule, line, and matching file path -> Match!
        actual_matching_file = [
            {"code": "SEM001", "line": 10, "file_path": "module/math_utils.py", "tool": "semantic_analyzer"}
        ]
        tp, fp, fn, matched, _, _ = match_findings(expected, actual_matching_file)
        assert tp == 1
        assert fp == 0
        assert fn == 0
        assert len(matched) == 1

    def test_matching_with_tool_constraint(self):
        expected = [ExpectedFinding(rule="B101", line=15, tool="bandit")]

        # Mismatched tool
        actual_diff_tool = [{"code": "B101", "line": 15, "tool": "other_tool"}]
        tp, fp, fn, _, _, _ = match_findings(expected, actual_diff_tool)
        assert tp == 0 and fp == 1 and fn == 1

        # Matching tool
        actual_same_tool = [{"code": "B101", "line": 15, "tool": "bandit"}]
        tp, fp, fn, _, _, _ = match_findings(expected, actual_same_tool)
        assert tp == 1 and fp == 0 and fn == 0

    def test_matching_case_insensitivity_and_whitespace(self):
        expected = [ExpectedFinding(rule="sem001", line=10)]
        actual = [{"code": "  SEM001  ", "line": 10}]
        tp, fp, fn, _, _, _ = match_findings(expected, actual)
        assert tp == 1 and fp == 0 and fn == 0

    def test_duplicate_findings_counted_as_false_positives(self):
        """When detector emits duplicates for 1 expected finding, extra count as FP."""
        expected = [ExpectedFinding(rule="E722", line=8)]
        actual = [
            {"code": "E722", "tool": "ruff", "line": 8},
            {"code": "E722", "tool": "ruff", "line": 8},
            {"code": "E722", "tool": "ruff", "line": 8},
        ]
        tp, fp, fn, matched, unmatched_act, unmatched_exp = match_findings(expected, actual)
        assert tp == 1, "Expected exactly 1 true positive"
        assert fp == 2, "Expected remaining duplicates to count as false positives"
        assert fn == 0, "No false negatives"
        assert len(matched) == 1
        assert len(unmatched_act) == 2
        assert len(unmatched_exp) == 0

    def test_line_tolerance_exact_and_window(self):
        expected = [ExpectedFinding(rule="B101", line=20, tolerance=2)]

        # Exact line
        tp, fp, fn, _, _, _ = match_findings(expected, [{"code": "B101", "line": 20}])
        assert tp == 1 and fp == 0 and fn == 0

        # Within tolerance window (+2)
        tp, fp, fn, _, _, _ = match_findings(expected, [{"code": "B101", "line": 22}])
        assert tp == 1 and fp == 0 and fn == 0

        # Outside tolerance window (+3)
        tp, fp, fn, _, _, _ = match_findings(expected, [{"code": "B101", "line": 23}])
        assert tp == 0 and fp == 1 and fn == 1


class TestMetricCalculationsAndEdgeCases:
    """Verify safe division, precision, recall, F1, and category breakdowns."""

    def test_safe_div_edge_cases(self):
        assert safe_div(0.0, 0.0, default=0.0) == 0.0
        assert safe_div(0.0, 0.0, default=1.0) == 1.0
        assert safe_div(5.0, 0.0, default=0.0) == 0.0
        assert safe_div(10.0, 2.0) == 5.0

    def test_precision_recall_f1_edge_cases(self):
        # 0 TP, 0 FP -> Precision defaults to 1.0 (clean case convention)
        assert calculate_precision(0, 0) == 1.0
        # 0 TP, 0 FN -> Recall defaults to 1.0
        assert calculate_recall(0, 0) == 1.0
        # Both 1.0 -> F1 is 1.0
        assert calculate_f1(1.0, 1.0) == 1.0

        # 0 TP, with FP > 0 -> Precision is 0.0
        assert calculate_precision(0, 5) == 0.0
        # 0 TP, with FN > 0 -> Recall is 0.0
        assert calculate_recall(0, 3) == 0.0
        # 0.0 precision and recall -> F1 is 0.0
        assert calculate_f1(0.0, 0.0) == 0.0

    def test_summary_category_breakdown_metrics(self):
        case_1 = MatchResult(
            case_id="err_01",
            category="error_handling",
            file_path="err_01.py",
            expected_count=1,
            actual_count=1,
            tp=1,
            fp=0,
            fn=0,
        )
        case_2 = MatchResult(
            case_id="sec_01",
            category="security",
            file_path="sec_01.py",
            expected_count=2,
            actual_count=3,
            tp=2,
            fp=1,
            fn=0,
        )
        summary = compute_benchmark_summary(
            case_results=[case_1, case_2],
            timestamp="2026-11-10T00:00:00Z",
        )
        cats = summary["categories"]
        assert "error_handling" in cats
        assert cats["error_handling"]["precision"] == 1.0000
        assert cats["error_handling"]["recall"] == 1.0000
        assert cats["error_handling"]["f1_score"] == 1.0000

        assert "security" in cats
        assert cats["security"]["precision"] == 0.6667
        assert cats["security"]["recall"] == 1.0000
        assert cats["security"]["f1_score"] == 0.8000

    def test_detector_classification(self):
        # Explicit tool overrides code prefix
        assert classify_detector_name({"tool": "ruff", "code": "B012"}) == "ruff"
        assert classify_detector_name({"tool": "bandit", "code": "B101"}) == "bandit"
        assert classify_detector_name({"tool": "semantic_analyzer", "code": "SEM001"}) == "semantic_analyzer"

        # Code prefix fallback when tool missing
        assert classify_detector_name({"code": "SEM002"}) == "semantic_analyzer"
        assert classify_detector_name({"code": "B110"}) == "bandit"
        assert classify_detector_name({"code": "E722"}) == "ruff"


class TestEvaluationReportSchemaAndExport:
    """Verify machine-readable JSON evaluation report schema and contents."""

    def test_export_summary_json_schema(self):
        case_res = MatchResult(
            case_id="err_01",
            category="error_handling",
            file_path="err_01.py",
            expected_count=1,
            actual_count=1,
            tp=1,
            fp=0,
            fn=0,
            matched_pairs=[{"expected": {"rule": "E722"}, "actual": {"code": "E722"}}],
            runtime_sec=0.25,
        )
        summary = compute_benchmark_summary(
            case_results=[case_res],
            timestamp="2026-11-10T12:00:00Z",
            git_commit="abcdef1",
            total_runtime_seconds=0.25,
            tool_failures=0,
            mode="deterministic",
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        try:
            export_summary_json(summary, temp_path)
            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Top-level schema validation
            expected_keys = {
                "timestamp",
                "git_commit",
                "mode",
                "total_runtime_seconds",
                "tool_failures",
                "defect_metrics",
                "clean_metrics",
                "overall",
                "by_detector",
                "categories",
                "cases",
            }
            assert expected_keys.issubset(data.keys()), f"Missing keys in export: {expected_keys - data.keys()}"

            # Case details schema validation
            assert len(data["cases"]) == 1
            case_entry = data["cases"][0]
            case_keys = {
                "case_id",
                "category",
                "file_path",
                "expected_count",
                "actual_count",
                "tp",
                "fp",
                "fn",
                "status",
                "runtime_sec",
            }
            assert case_keys.issubset(case_entry.keys())
            assert case_entry["case_id"] == "err_01"
            assert case_entry["status"] == "PASSED"
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSourceFileImmutability:
    """Verify evaluation execution does not modify test sample files."""

    def test_error_handling_cases_hashes_unmodified_after_run(self):
        root = Path(__file__).resolve().parent.parent
        err_dir = root / "benchmark" / "cases" / "error_handling"
        files = list(err_dir.glob("*.py"))
        assert len(files) >= 4, f"Expected at least 4 error handling files, found {len(files)}"

        def get_hashes() -> dict[str, str]:
            hashes = {}
            for f in files:
                hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
            return hashes

        hashes_before = get_hashes()

        # Run error_handling evaluation
        _summary, exit_code = run_evaluation(
            category="error_handling",
            output_report=None,
            quiet=True,
        )
        assert exit_code == 0

        hashes_after = get_hashes()
        assert hashes_before == hashes_after, "Sample case files were modified during evaluation run!"


class TestRunnerExecutionAndModes:
    """Verify execution of error_handling evaluation, pipeline routing, and dedup."""

    def test_deterministic_evaluation_on_error_handling(self):
        summary, exit_code = run_evaluation(
            category="error_handling",
            output_report=None,
            quiet=True,
        )
        assert exit_code == 0
        overall = summary["overall"]
        assert overall["cases_total"] == 4
        assert overall["cases_passed"] == 4
        assert overall["cases_failed"] == 0
        assert overall["cases_mismatched"] == 0
        assert overall["expected_findings"] == 6
        assert overall["actual_findings"] == 6
        assert overall["matched_findings"] == 6
        assert overall["unmatched_actual"] == 0
        assert overall["unmatched_expected"] == 0
        assert overall["precision"] == 1.0000
        assert overall["recall"] == 1.0000
        assert overall["f1_score"] == 1.0000

    def test_multi_agent_mode_routes_to_review_pipeline(self):
        with patch("benchmark.runner.run_review_pipeline") as mock_pipeline:
            mock_pipeline.return_value = {
                "ruff_findings": [],
                "bandit_findings": [],
                "semantic_findings": [],
                "tool_status": {"ruff": True, "bandit": True, "semantic": True},
            }
            _summary, exit_code = run_evaluation(
                case_id="err_04",
                mode="full",
                output_report=None,
                quiet=True,
            )
            assert exit_code == 0
            assert mock_pipeline.called
            call_kwargs = mock_pipeline.call_args.kwargs
            assert call_kwargs.get("enable_ai") is True

    def test_deterministic_mode_routes_to_run_deterministic_only(self):
        with patch("benchmark.runner.run_deterministic_only") as mock_det:
            mock_det.return_value = {
                "ruff_findings": [],
                "bandit_findings": [],
                "semantic_findings": [],
                "tool_status": {"ruff": True, "bandit": True, "semantic": True},
            }
            _summary, exit_code = run_evaluation(
                case_id="err_04",
                mode="deterministic",
                output_report=None,
                quiet=True,
            )
            assert exit_code == 0
            assert mock_det.called

    def test_deduplication_flag_in_runner(self):
        fake_state = {
            "ruff_findings": [
                {
                    "code": "E722",
                    "file_path": "benchmark/cases/error_handling/err_01_bare_except.py",
                    "line": 8,
                    "message": "Do not use bare except",
                    "severity": "medium",
                },
                {
                    "code": "E722",
                    "file_path": "benchmark/cases/error_handling/err_01_bare_except.py",
                    "line": 8,
                    "message": "Do not use bare except",
                    "severity": "medium",
                },
            ],
            "bandit_findings": [],
            "semantic_findings": [],
            "tool_status": {"ruff": True, "bandit": True, "semantic": True},
        }

        # With dedup=True, duplicates should be collapsed before matching
        with patch("benchmark.runner.run_deterministic_only", return_value=fake_state):
            summary, exit_code = run_evaluation(
                case_id="err_01",
                dedup=True,
                output_report=None,
                quiet=True,
            )
            assert exit_code == 0
            assert summary["overall"]["actual_findings"] == 1
            assert summary["overall"]["matched_findings"] == 1
            assert summary["overall"]["unmatched_actual"] == 0
