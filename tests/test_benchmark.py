"""
tests/test_benchmark.py - Unit and Integration Tests for Phase 6 Benchmark.

Validates:
  1. Manifest schema integrity and case file existence on disk.
  2. Safe metric calculations and zero-denominator behavior.
  3. One-to-one bipartite matching and duplicate finding handling.
  4. Line tolerance windows and rule code mismatches.
  5. Clean-case false-positive rate calculations.
  6. Category and single-case CLI filtering.
  7. Error handling and containment when tools or cases fail.
  8. JSON and CSV output artifact generation.
  9. Preservation of benchmark source files (no mutation guarantee).
 10. End-to-end execution of the 15-case deterministic benchmark.
"""

from __future__ import annotations

import csv
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
    compute_benchmark_summary,
    match_findings,
    safe_div,
)
from benchmark.runner import (
    load_manifest,
    run_benchmark,
)


@pytest.fixture
def manifest_path() -> str:
    """Return the absolute path to the production benchmark manifest."""
    root = Path(__file__).resolve().parent.parent
    return str(root / "benchmark" / "manifest.json")


class TestManifestIntegrity:
    """Validate manifest schema and ensure all test case files exist."""

    def test_manifest_loads_and_has_required_keys(self, manifest_path: str):
        cases = load_manifest(manifest_path)
        assert len(cases) == 15, f"Expected 15 cases, got {len(cases)}"

        valid_categories = {"clean", "bugs", "security", "multi", "safe_edges"}
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
            assert full_path.is_file(), f"Case file does not exist: {full_path}"

            expected = c.get("expected_findings", [])
            for exp in expected:
                assert "rule" in exp, f"Expected finding missing 'rule' in case '{cid}'"
                assert "line" in exp or exp.get("line") is None

    def test_missing_manifest_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_manifest("non_existent_manifest_xyz.json")


class TestMetricCalculationsAndZeroDivision:
    """Verify safety of division and metric formulas under edge cases."""

    def test_safe_div_zero_denominator(self):
        assert safe_div(10.0, 0.0, default=0.0) == 0.0
        assert safe_div(0.0, 0.0, default=1.0) == 1.0
        assert safe_div(10.0, 2.0) == 5.0

    def test_precision_recall_f1_zero_cases(self):
        # Perfect clean case: 0 TP, 0 FP, 0 FN -> Precision and Recall default to 1.0
        assert calculate_precision(0, 0) == 1.0
        assert calculate_recall(0, 0) == 1.0
        assert calculate_f1(1.0, 1.0) == 1.0

        # Zero TP with FP > 0
        assert calculate_precision(0, 2) == 0.0
        # Zero TP with FN > 0
        assert calculate_recall(0, 3) == 0.0

        # Both zero F1
        assert calculate_f1(0.0, 0.0) == 0.0

    def test_precision_recall_normal_values(self):
        prec = calculate_precision(8, 2)  # 8 / 10 = 0.8
        rec = calculate_recall(8, 0)     # 8 / 8 = 1.0
        f1 = calculate_f1(prec, rec)      # 2 * 0.8 * 1.0 / 1.8 = 0.8888...
        assert round(prec, 4) == 0.8000
        assert round(rec, 4) == 1.0000
        assert round(f1, 4) == 0.8889


class TestFindingMatchingLogic:
    """Verify 1-to-1 bipartite matching, line tolerance, and duplicate handling."""

    def test_one_to_one_matching_prevents_duplicate_inflation(self):
        # 1 expected finding, but detector emitted 3 duplicate warnings for the same line
        expected = [ExpectedFinding(rule="SEM001", line=10, tolerance=1)]
        actual = [
            {"code": "SEM001", "tool": "semantic_analyzer", "line": 10},
            {"code": "SEM001", "tool": "semantic_analyzer", "line": 10},
            {"code": "SEM001", "tool": "semantic_analyzer", "line": 10},
        ]
        tp, fp, fn, matched, un_act, un_exp = match_findings(expected, actual)
        assert tp == 1, "Duplicate findings must not inflate True Positives"
        assert fp == 2, "Extra duplicate findings must count as False Positives"
        assert fn == 0
        assert len(matched) == 1
        assert len(un_act) == 2
        assert len(un_exp) == 0

    def test_one_to_one_matching_multiple_expected_and_duplicates(self):
        # 2 expected findings; 3 actual findings (2 identical for exp1, 1 for exp2)
        expected = [
            ExpectedFinding(rule="SEM001", line=10, tolerance=1),
            ExpectedFinding(rule="B101", line=20, tolerance=1),
        ]
        actual = [
            {"code": "SEM001", "tool": "semantic_analyzer", "line": 10},
            {"code": "SEM001", "tool": "semantic_analyzer", "line": 10},
            {"code": "B101", "tool": "bandit", "line": 20},
        ]
        tp, fp, fn, matched, un_act, un_exp = match_findings(expected, actual)
        assert tp == 2, "TP must equal matched expected findings, exactly 2"
        assert fp == 1, "Duplicate SEM001 must count as FP"
        assert fn == 0, "All expected findings matched"
        assert len(matched) == 2
        assert len(un_act) == 1
        assert len(un_exp) == 0

    def test_line_tolerance_window(self):
        # Expected at line 20 with tolerance 2
        expected = [ExpectedFinding(rule="B101", line=20, tolerance=2)]

        # Actual at line 22 (distance 2 <= tolerance 2) -> Matches!
        actual_within = [{"code": "B101", "tool": "bandit", "line": 22}]
        tp, fp, fn, _, _, _ = match_findings(expected, actual_within)
        assert tp == 1 and fp == 0 and fn == 0

        # Actual at line 23 (distance 3 > tolerance 2) -> Does not match
        actual_outside = [{"code": "B101", "tool": "bandit", "line": 23}]
        tp2, fp2, fn2, _, _, _ = match_findings(expected, actual_outside)
        assert tp2 == 0 and fp2 == 1 and fn2 == 1

    def test_rule_mismatch(self):
        expected = [ExpectedFinding(rule="SEM001", line=10)]
        actual = [{"code": "SEM002", "tool": "semantic_analyzer", "line": 10}]
        tp, fp, fn, _, _, _ = match_findings(expected, actual)
        assert tp == 0
        assert fp == 1
        assert fn == 1


class TestCleanCaseFalsePositiveRate:
    """Verify calculation of clean-case false-positive rate."""

    def test_clean_cases_fpr_calculation(self):
        # 2 clean cases: 1 has 0 actual findings, 1 has 1 unexpected finding
        res1 = MatchResult(
            case_id="clean_1",
            category="clean",
            file_path="clean1.py",
            expected_count=0,
            actual_count=0,
            tp=0,
            fp=0,
            fn=0,
        )
        res2 = MatchResult(
            case_id="clean_2",
            category="clean",
            file_path="clean2.py",
            expected_count=0,
            actual_count=1,
            tp=0,
            fp=1,
            fn=0,
            unmatched_actual=[{"code": "SEM001", "line": 5}],
        )
        summary = compute_benchmark_summary(
            case_results=[res1, res2],
            timestamp="2026-10-27T00:00:00Z",
        )
        assert summary["overall"]["clean_cases_total"] == 2
        assert summary["overall"]["clean_cases_with_fp"] == 1
        assert summary["overall"]["clean_case_fpr"] == 0.5

    def test_defect_metrics_reported_separately_from_clean_fpr(self):
        # 1 defect case (1 expected, 1 matched) + 1 clean case with 1 FP
        defect_case = MatchResult(
            case_id="defect_1",
            category="bugs",
            file_path="bug1.py",
            expected_count=1,
            actual_count=1,
            tp=1,
            fp=0,
            fn=0,
            matched_pairs=[{"expected": {"rule": "SEM001"}, "actual": {"code": "SEM001"}}],
        )
        clean_case_with_fp = MatchResult(
            case_id="clean_1",
            category="clean",
            file_path="clean1.py",
            expected_count=0,
            actual_count=1,
            tp=0,
            fp=1,
            fn=0,
            unmatched_actual=[{"code": "B101", "line": 5}],
        )
        summary = compute_benchmark_summary(
            case_results=[defect_case, clean_case_with_fp],
            timestamp="2026-10-27T00:00:00Z",
        )
        # Defect metrics must remain 1.0 (unaffected by clean-case FP)
        defect = summary["defect_metrics"]
        assert defect["precision"] == 1.0
        assert defect["recall"] == 1.0
        assert defect["f1_score"] == 1.0
        assert defect["expected_findings"] == 1
        assert defect["actual_findings"] == 1
        assert defect["matched_findings"] == 1
        assert defect["unmatched_actual"] == 0
        assert defect["unmatched_expected"] == 0

        # Clean metrics capture the clean-case false positive rate
        clean = summary["clean_metrics"]
        assert clean["cases_total"] == 1
        assert clean["cases_with_fp"] == 1
        assert clean["clean_case_fpr"] == 1.0
        assert clean["unmatched_actual"] == 1

        # Overall metrics reflect both
        overall = summary["overall"]
        assert overall["tp"] == 1
        assert overall["fp"] == 1
        assert overall["precision"] == 0.5


class TestBenchmarkRunnerAndFiltering:
    """Verify runner filtering by category and single case ID."""

    def test_filtering_by_category(self, manifest_path: str):
        summary, exit_code = run_benchmark(
            manifest_path=manifest_path,
            category="bugs",
            quiet=True,
        )
        assert exit_code == 0
        assert summary["overall"]["cases_total"] == 3
        for r in summary["case_results"]:
            assert r["category"] == "bugs"

    def test_filtering_by_case_id(self, manifest_path: str):
        summary, exit_code = run_benchmark(
            manifest_path=manifest_path,
            case_id="sec_01",
            quiet=True,
        )
        assert exit_code == 0
        assert summary["overall"]["cases_total"] == 1
        assert summary["case_results"][0]["case_id"] == "sec_01"

    def test_unmatched_filter_returns_empty_summary(self, manifest_path: str):
        summary, exit_code = run_benchmark(
            manifest_path=manifest_path,
            case_id="non_existent_id",
            quiet=True,
        )
        assert exit_code == 0
        assert summary["overall"]["cases_total"] == 0


class TestErrorHandlingAndContainment:
    """Verify runner handles scanner failures without crashing."""

    def test_runner_records_failure_when_pipeline_raises(self, manifest_path: str):
        with patch("benchmark.runner.run_deterministic_only", side_effect=RuntimeError("Tool crashed")):
            summary, exit_code = run_benchmark(
                manifest_path=manifest_path,
                case_id="bug_01",
                quiet=True,
            )
            assert exit_code == 1, "Exit code must be 1 on tool failure"
            assert summary["overall"]["tool_failures"] == 1
            case_res = summary["case_results"][0]
            assert case_res["success"] is False
            assert "Tool crashed" in (case_res["error_message"] or "")

    def test_scanner_crash_on_clean_case_marked_as_failed_not_passed(self, manifest_path: str):
        # Mock run_deterministic_only returning an agent error on clean_01 (0 expected findings)
        failed_state = {
            "target_path": "clean_01.py",
            "ruff_findings": [],
            "bandit_findings": [],
            "semantic_findings": [],
            "agent_errors": ["Ruff: process crashed with SIGSEGV"],
        }
        with patch("benchmark.runner.run_deterministic_only", return_value=failed_state):
            summary, exit_code = run_benchmark(
                manifest_path=manifest_path,
                case_id="clean_01",
                quiet=True,
            )
            assert exit_code == 1, "Exit code must be 1 on scanner crash"
            assert summary["overall"]["tool_failures"] == 1
            case_res = summary["case_results"][0]
            assert case_res["success"] is False
            assert case_res["status"] == "FAILED", "Case must be marked FAILED, never PASSED"
            assert summary["clean_metrics"]["cases_failed"] == 1
            assert summary["clean_metrics"]["cases_passed"] == 0

    def test_missing_scanner_output_detected_and_fails_case(self, manifest_path: str):
        # State with missing ruff_findings (None instead of list)
        missing_state = {
            "target_path": "bug_01.py",
            "ruff_findings": None,
            "bandit_findings": [],
            "semantic_findings": [],
            "agent_errors": [],
        }
        with patch("benchmark.runner.run_deterministic_only", return_value=missing_state):
            summary, exit_code = run_benchmark(
                manifest_path=manifest_path,
                case_id="bug_01",
                quiet=True,
            )
            assert exit_code == 1
            assert summary["overall"]["tool_failures"] >= 1
            case_res = summary["case_results"][0]
            assert case_res["success"] is False
            assert case_res["status"] == "FAILED"
            assert "Missing required scanner output: Ruff" in (case_res["error_message"] or "")


class TestOutputArtifacts:
    """Verify JSON summary and CSV export structures."""

    def test_json_and_csv_generation(self, manifest_path: str):
        with tempfile.TemporaryDirectory() as td:
            _summary, exit_code = run_benchmark(
                manifest_path=manifest_path,
                case_id="clean_01",
                output_dir=td,
                quiet=True,
            )
            assert exit_code == 0

            json_file = Path(td) / "benchmark_summary.json"
            csv_file = Path(td) / "benchmark_cases.csv"
            assert json_file.is_file(), "benchmark_summary.json not written"
            assert csv_file.is_file(), "benchmark_cases.csv not written"

            # Validate JSON
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            assert "overall" in data
            assert "by_detector" in data
            assert data["overall"]["cases_total"] == 1

            # Validate CSV
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["case_id"] == "clean_01"
            assert rows[0]["category"] == "clean"


class TestNoSourceMutation:
    """Ensure benchmark run never modifies any case or project source files."""

    def test_source_files_remain_unmodified(self, manifest_path: str):
        root = Path(__file__).resolve().parent.parent
        cases = load_manifest(manifest_path)

        def hash_file(p: Path) -> str:
            return hashlib.sha256(p.read_bytes()).hexdigest()

        before_hashes: dict[str, str] = {}
        for c in cases:
            p = root / c["file_path"]
            before_hashes[str(p)] = hash_file(p)

        # Run benchmark
        _summary, exit_code = run_benchmark(manifest_path=manifest_path, quiet=True)
        assert exit_code == 0

        # Verify hashes
        for c in cases:
            p = root / c["file_path"]
            after_hash = hash_file(p)
            assert after_hash == before_hashes[str(p)], f"File was mutated: {p}"


class TestFullDeterministicBenchmarkExecution:
    """Run all 15 cases end-to-end against the deterministic pipeline."""

    def test_full_benchmark_executes_cleanly(self, manifest_path: str):
        summary, exit_code = run_benchmark(manifest_path=manifest_path, quiet=True)
        assert exit_code == 0, "Benchmark run encountered tool failures"

        overall = summary["overall"]
        assert overall["cases_total"] == 15
        assert overall["cases_passed"] == 15
        assert overall["cases_failed"] == 0
        assert overall["cases_mismatched"] == 0
        assert overall["tool_failures"] == 0
        assert overall["expected_findings"] == 13
        assert overall["actual_findings"] == 13
        assert overall["matched_findings"] == 13
        assert overall["unmatched_actual"] == 0
        assert overall["unmatched_expected"] == 0

        # Accuracy checks
        assert overall["precision"] == 1.0, f"Expected precision 1.0, got {overall['precision']}"
        assert overall["recall"] == 1.0, f"Expected recall 1.0, got {overall['recall']}"
        assert overall["f1_score"] == 1.0, f"Expected F1 1.0, got {overall['f1_score']}"
        assert overall["clean_case_fpr"] == 0.0, f"Expected clean FPR 0.0, got {overall['clean_case_fpr']}"

        # Defect metrics check
        defect = summary["defect_metrics"]
        assert defect["cases_total"] == 9
        assert defect["cases_passed"] == 9
        assert defect["cases_failed"] == 0
        assert defect["expected_findings"] == 13
        assert defect["actual_findings"] == 13
        assert defect["matched_findings"] == 13
        assert defect["unmatched_actual"] == 0
        assert defect["unmatched_expected"] == 0
        assert defect["precision"] == 1.0
        assert defect["recall"] == 1.0
        assert defect["f1_score"] == 1.0

        # Clean metrics check
        clean = summary["clean_metrics"]
        assert clean["cases_total"] == 6
        assert clean["cases_passed"] == 6
        assert clean["cases_failed"] == 0
        assert clean["cases_with_fp"] == 0
        assert clean["clean_case_fpr"] == 0.0

        # Per-detector checks
        detectors = summary["by_detector"]
        assert detectors["semantic_analyzer"]["tp"] == 6
        assert detectors["semantic_analyzer"]["expected"] == 6
        assert detectors["semantic_analyzer"]["actual"] == 6
        assert detectors["bandit"]["tp"] == 5
        assert detectors["bandit"]["expected"] == 5
        assert detectors["bandit"]["actual"] == 5
        assert detectors["ruff"]["tp"] == 2
        assert detectors["ruff"]["expected"] == 2
        assert detectors["ruff"]["actual"] == 2
