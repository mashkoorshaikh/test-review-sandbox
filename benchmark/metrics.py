"""
metrics.py - Finding Matching and Evaluation Metrics for Benchmark.

Provides:
  1. Normalized path matching across platforms (Windows / POSIX).
  2. Bipartite 1-to-1 finding matching between expected and actual findings with line tolerance.
  3. Safe Precision, Recall, and F1 calculations avoiding division-by-zero errors.
  4. Per-detector breakdown (Ruff, Bandit, Semantic Analyzer).
  5. Case-level False Positive Rate for clean and safe-edge test cases.
  6. Serialization helpers for JSON summary and CSV exports.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


def normalize_path(path: str) -> str:
    """Normalize file paths to forward-slash relative strings for portable matching."""
    norm = os.path.normpath(path).replace("\\", "/")
    return norm.removeprefix("./")


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Perform division safely, returning default when denominator is zero."""
    if denominator == 0.0:
        return default
    return numerator / denominator


def calculate_precision(tp: int, fp: int, default: float = 1.0) -> float:
    """Calculate precision safely. If tp + fp == 0, returns default (1.0 for clean)."""
    if tp + fp == 0:
        return default
    return tp / (tp + fp)


def calculate_recall(tp: int, fn: int, default: float = 1.0) -> float:
    """Calculate recall safely. If tp + fn == 0, returns default (1.0 for clean)."""
    if tp + fn == 0:
        return default
    return tp / (tp + fn)


def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score safely from precision and recall."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


@dataclass(frozen=True)
class ExpectedFinding:
    """Expected finding in a benchmark ground truth case."""

    rule: str
    tool: str | None = None
    line: int | None = None
    tolerance: int = 1
    file_path: str | None = None


@dataclass
class MatchResult:
    """Result of matching actual findings against expected ground truth for a case."""

    case_id: str
    category: str
    file_path: str
    expected_count: int
    actual_count: int
    tp: int
    fp: int
    fn: int
    matched_pairs: list[dict[str, Any]] = field(default_factory=list)
    unmatched_actual: list[dict[str, Any]] = field(default_factory=list)
    unmatched_expected: list[dict[str, Any]] = field(default_factory=list)
    runtime_sec: float = 0.0
    success: bool = True
    status: str = "PASSED"
    error_message: str | None = None
    raw_findings: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Derive case status from tool execution success and finding matching."""
        if not self.success:
            self.status = "FAILED"
        elif self.fp == 0 and self.fn == 0:
            self.status = "PASSED"
        else:
            self.status = "MISMATCH"

    @property
    def is_clean_or_safe(self) -> bool:
        """True if the case expected zero findings (clean or safe edge)."""
        return self.expected_count == 0


def classify_detector_name(finding: ExpectedFinding | dict[str, Any]) -> str:
    """Classify finding into one of the three primary detector families."""
    if isinstance(finding, ExpectedFinding):
        tool = (finding.tool or "").lower().strip()
        code = (finding.rule or "").upper().strip()
    else:
        tool = str(finding.get("tool", "")).lower().strip()
        code = str(finding.get("code", finding.get("rule", ""))).upper().strip()

    if tool in ("semantic_analyzer", "bandit", "ruff"):
        return tool
    if tool == "semantic" or code.startswith("SEM"):
        return "semantic_analyzer"
    if tool == "bandit" or code.startswith("B"):
        return "bandit"
    if tool == "ruff" or code.startswith(("F", "E", "W", "I", "SIM", "UP", "RUF")):
        return "ruff"
    return tool or "unknown"


def match_findings(
    expected_findings: list[ExpectedFinding | dict[str, Any]],
    actual_findings: list[dict[str, Any]],
    default_tolerance: int = 1,
) -> tuple[int, int, int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Perform 1-to-1 bipartite matching between expected and actual findings.

    Matching criteria:
      1. Rule code matches (case-insensitive).
      2. Line distance is within tolerance (default ±1).
      3. Optional tool name matches if specified in expected.

    Guarantees:
      - Each expected finding matches at most one actual finding.
      - Extra duplicate findings for the same defect count as False Positives.
      - Unmatched expected findings count as False Negatives.

    Returns:
      (tp, fp, fn, matched_pairs, unmatched_actual, unmatched_expected)
    """
    # Normalize expected findings
    norm_expected: list[dict[str, Any]] = []
    for exp in expected_findings:
        if isinstance(exp, ExpectedFinding):
            norm_expected.append({
                "rule": exp.rule.upper(),
                "tool": exp.tool.lower() if exp.tool else None,
                "line": exp.line,
                "tolerance": exp.tolerance,
                "file_path": exp.file_path,
            })
        elif isinstance(exp, dict):
            norm_expected.append({
                "rule": str(exp.get("rule", "")).upper(),
                "tool": str(exp.get("tool", "")).lower() if exp.get("tool") else None,
                "line": exp.get("line"),
                "tolerance": exp.get("tolerance", default_tolerance),
                "file_path": exp.get("file_path"),
            })

    # Prepare actual findings with assigned index
    actual_candidates = list(enumerate(actual_findings))
    matched_actual_indices: set[int] = set()
    matched_expected_indices: set[int] = set()
    matched_pairs: list[dict[str, Any]] = []

    # First pass: look for rule code + line tolerance matches
    for exp_idx, exp in enumerate(norm_expected):
        target_rule = exp["rule"]
        target_line = exp["line"]
        tol = exp.get("tolerance", default_tolerance)
        target_tool = exp.get("tool")
        target_file = exp.get("file_path")

        best_actual_idx: int | None = None
        best_distance: float = float("inf")

        for act_idx, act in actual_candidates:
            if act_idx in matched_actual_indices:
                continue

            act_code = str(act.get("code", "")).upper().strip()
            act_tool = str(act.get("tool", "")).lower().strip()
            act_line = act.get("line")
            act_file = act.get("file_path")

            # Check rule code match
            if act_code != target_rule:
                continue

            # Check tool match if specified
            if target_tool and act_tool and target_tool != act_tool:
                continue

            # Check file_path match if specified in expected
            if target_file and act_file:
                norm_target = normalize_path(target_file).lower()
                norm_act = normalize_path(act_file).lower()
                if not (norm_target == norm_act or norm_act.endswith("/" + norm_target) or norm_target.endswith("/" + norm_act)):
                    continue

            # Check line distance
            if target_line is not None and act_line is not None:
                dist = abs(act_line - target_line)
                if dist <= tol and dist < best_distance:
                    best_distance = dist
                    best_actual_idx = act_idx
            elif target_line is None:
                best_distance = 0
                best_actual_idx = act_idx
                break

        if best_actual_idx is not None:
            matched_actual_indices.add(best_actual_idx)
            matched_expected_indices.add(exp_idx)
            matched_pairs.append({
                "expected": exp,
                "actual": actual_findings[best_actual_idx],
                "line_distance": best_distance,
            })

    tp = len(matched_pairs)
    fp = len(actual_findings) - len(matched_actual_indices)
    fn = len(norm_expected) - len(matched_expected_indices)

    unmatched_actual = [
        act for idx, act in enumerate(actual_findings) if idx not in matched_actual_indices
    ]
    unmatched_expected = [
        exp for idx, exp in enumerate(norm_expected) if idx not in matched_expected_indices
    ]

    return tp, fp, fn, matched_pairs, unmatched_actual, unmatched_expected


def evaluate_case(
    case_id: str,
    category: str,
    file_path: str,
    expected_findings: list[ExpectedFinding | dict[str, Any]],
    actual_findings: list[dict[str, Any]],
    runtime_sec: float = 0.0,
    success: bool = True,
    error_message: str | None = None,
    default_tolerance: int = 1,
) -> MatchResult:
    """Evaluate a single benchmark case against ground truth."""
    tp, fp, fn, matched, un_act, un_exp = match_findings(
        expected_findings=expected_findings,
        actual_findings=actual_findings,
        default_tolerance=default_tolerance,
    )
    return MatchResult(
        case_id=case_id,
        category=category,
        file_path=normalize_path(file_path),
        expected_count=len(expected_findings),
        actual_count=len(actual_findings),
        tp=tp,
        fp=fp,
        fn=fn,
        matched_pairs=matched,
        unmatched_actual=un_act,
        unmatched_expected=un_exp,
        runtime_sec=runtime_sec,
        success=success,
        error_message=error_message,
        raw_findings=actual_findings,
    )


def compute_benchmark_summary(
    case_results: list[MatchResult],
    timestamp: str,
    git_commit: str | None = None,
    total_runtime_seconds: float = 0.0,
    tool_failures: int = 0,
    mode: str = "deterministic",
) -> dict[str, Any]:
    """
    Compute aggregate benchmark summary metrics, per-detector breakdown, and category breakdown.

    Reports labeled defect findings precision, recall, and F1 separately from clean/safe-edge FPR.
    Clearly tracks expected count, actual count, matched findings (TP), unmatched findings (FP/FN),
    and tool failures.
    """
    # Defect cases vs clean/safe cases
    defect_cases = [r for r in case_results if not r.is_clean_or_safe]
    clean_cases = [r for r in case_results if r.is_clean_or_safe]

    # Defect findings metrics (labeled defect cases with expected > 0)
    defect_expected = sum(r.expected_count for r in defect_cases)
    defect_actual = sum(r.actual_count for r in defect_cases)
    defect_tp = sum(r.tp for r in defect_cases)
    defect_fp = sum(r.fp for r in defect_cases)
    defect_fn = sum(r.fn for r in defect_cases)
    defect_precision = calculate_precision(defect_tp, defect_fp)
    defect_recall = calculate_recall(defect_tp, defect_fn)
    defect_f1 = calculate_f1(defect_precision, defect_recall)
    defect_passed = sum(1 for r in defect_cases if r.status == "PASSED")
    defect_failed = sum(1 for r in defect_cases if r.status == "FAILED")
    defect_mismatched = sum(1 for r in defect_cases if r.status == "MISMATCH")

    # Clean & safe-edge subset (false positive control)
    clean_cases_total = len(clean_cases)
    clean_cases_passed = sum(1 for r in clean_cases if r.status == "PASSED")
    clean_cases_failed = sum(1 for r in clean_cases if r.status == "FAILED")
    clean_cases_with_fp = sum(1 for r in clean_cases if r.fp > 0)
    clean_case_fpr = safe_div(clean_cases_with_fp, clean_cases_total, 0.0)
    clean_fp_count = sum(r.fp for r in clean_cases)

    # Overall finding counts
    overall_expected = sum(r.expected_count for r in case_results)
    overall_actual = sum(r.actual_count for r in case_results)
    overall_tp = sum(r.tp for r in case_results)
    overall_fp = sum(r.fp for r in case_results)
    overall_fn = sum(r.fn for r in case_results)

    overall_precision = calculate_precision(overall_tp, overall_fp)
    overall_recall = calculate_recall(overall_tp, overall_fn)
    overall_f1 = calculate_f1(overall_precision, overall_recall)
    overall_passed = sum(1 for r in case_results if r.status == "PASSED")
    overall_failed = sum(1 for r in case_results if r.status == "FAILED")
    overall_mismatched = sum(1 for r in case_results if r.status == "MISMATCH")

    # Per-detector breakdown
    detectors = ("semantic_analyzer", "bandit", "ruff")
    detector_stats: dict[str, dict[str, Any]] = {
        d: {"expected": 0, "actual": 0, "tp": 0, "fp": 0, "fn": 0} for d in detectors
    }

    for r in case_results:
        # Expected by detector
        for exp in r.unmatched_expected:
            det = classify_detector_name(exp)
            if det in detector_stats:
                detector_stats[det]["expected"] += 1
                detector_stats[det]["fn"] += 1
            else:
                detector_stats.setdefault(det, {"expected": 0, "actual": 0, "tp": 0, "fp": 0, "fn": 0})["expected"] += 1
                detector_stats[det]["fn"] += 1

        for pair in r.matched_pairs:
            det_exp = classify_detector_name(pair["expected"])
            det_act = classify_detector_name(pair["actual"])
            target_det = det_act if det_act in detector_stats else det_exp
            if target_det in detector_stats:
                detector_stats[target_det]["expected"] += 1
                detector_stats[target_det]["actual"] += 1
                detector_stats[target_det]["tp"] += 1
            else:
                s = detector_stats.setdefault(target_det, {"expected": 0, "actual": 0, "tp": 0, "fp": 0, "fn": 0})
                s["expected"] += 1
                s["actual"] += 1
                s["tp"] += 1

        # FPs by detector
        for act in r.unmatched_actual:
            det = classify_detector_name(act)
            if det in detector_stats:
                detector_stats[det]["actual"] += 1
                detector_stats[det]["fp"] += 1
            else:
                s = detector_stats.setdefault(det, {"expected": 0, "actual": 0, "tp": 0, "fp": 0, "fn": 0})
                s["actual"] += 1
                s["fp"] += 1

    detector_metrics: dict[str, Any] = {}
    for det, counts in detector_stats.items():
        dtp = counts["tp"]
        dfp = counts["fp"]
        dfn = counts["fn"]
        dexp = counts["expected"]
        dact = counts["actual"]
        dprec = calculate_precision(dtp, dfp)
        drec = calculate_recall(dtp, dfn)
        df1 = calculate_f1(dprec, drec)
        detector_metrics[det] = {
            "expected": dexp,
            "actual": dact,
            "tp": dtp,
            "fp": dfp,
            "fn": dfn,
            "precision": round(dprec, 4),
            "recall": round(drec, 4),
            "f1_score": round(df1, 4),
        }

    # Category breakdown
    categories: dict[str, dict[str, Any]] = {}
    for r in case_results:
        cat = categories.setdefault(
            r.category,
            {
                "cases": 0,
                "passed": 0,
                "failed": 0,
                "mismatched": 0,
                "expected": 0,
                "actual": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
            },
        )
        cat["cases"] += 1
        if r.status == "PASSED":
            cat["passed"] += 1
        elif r.status == "FAILED":
            cat["failed"] += 1
        else:
            cat["mismatched"] += 1
        cat["expected"] += r.expected_count
        cat["actual"] += r.actual_count
        cat["tp"] += r.tp
        cat["fp"] += r.fp
        cat["fn"] += r.fn

    for cat in categories.values():
        c_prec = calculate_precision(cat["tp"], cat["fp"])
        c_rec = calculate_recall(cat["tp"], cat["fn"])
        c_f1 = calculate_f1(c_prec, c_rec)
        cat["precision"] = round(c_prec, 4)
        cat["recall"] = round(c_rec, 4)
        cat["f1_score"] = round(c_f1, 4)

    return {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "mode": mode,
        "defect_metrics": {
            "cases_total": len(defect_cases),
            "cases_passed": defect_passed,
            "cases_failed": defect_failed,
            "cases_mismatched": defect_mismatched,
            "expected_findings": defect_expected,
            "actual_findings": defect_actual,
            "matched_findings": defect_tp,
            "unmatched_actual": defect_fp,
            "unmatched_expected": defect_fn,
            "tp": defect_tp,
            "fp": defect_fp,
            "fn": defect_fn,
            "precision": round(defect_precision, 4),
            "recall": round(defect_recall, 4),
            "f1_score": round(defect_f1, 4),
        },
        "clean_metrics": {
            "cases_total": clean_cases_total,
            "cases_passed": clean_cases_passed,
            "cases_failed": clean_cases_failed,
            "cases_with_fp": clean_cases_with_fp,
            "clean_case_fpr": round(clean_case_fpr, 4),
            "expected_findings": 0,
            "actual_findings": sum(r.actual_count for r in clean_cases),
            "matched_findings": 0,
            "unmatched_actual": clean_fp_count,
            "unmatched_expected": 0,
            "fp": clean_fp_count,
        },
        "overall": {
            "cases_total": len(case_results),
            "cases_passed": overall_passed,
            "cases_failed": overall_failed,
            "cases_mismatched": overall_mismatched,
            "expected_findings": overall_expected,
            "actual_findings": overall_actual,
            "matched_findings": overall_tp,
            "unmatched_actual": overall_fp,
            "unmatched_expected": overall_fn,
            "tp": overall_tp,
            "fp": overall_fp,
            "fn": overall_fn,
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1_score": round(overall_f1, 4),
            "defect_precision": round(defect_precision, 4),
            "defect_recall": round(defect_recall, 4),
            "defect_f1_score": round(defect_f1, 4),
            "clean_cases_total": clean_cases_total,
            "clean_cases_passed": clean_cases_passed,
            "clean_cases_failed": clean_cases_failed,
            "clean_cases_with_fp": clean_cases_with_fp,
            "clean_case_fpr": round(clean_case_fpr, 4),
            "total_runtime_seconds": round(total_runtime_seconds, 4),
            "tool_failures": tool_failures,
        },
        "total_runtime_seconds": round(total_runtime_seconds, 4),
        "tool_failures": tool_failures,
        "by_detector": detector_metrics,
        "by_category": categories,
        "categories": categories,
        "case_results": [asdict(r) for r in case_results],
        "cases": [asdict(r) for r in case_results],
    }


def export_summary_json(summary: dict[str, Any], output_path: str) -> None:
    """Export summary dictionary to formatted JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def export_cases_csv(case_results: list[MatchResult], output_path: str) -> None:
    """Export case results list to CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = [
        "case_id",
        "category",
        "file_path",
        "status",
        "expected_count",
        "actual_count",
        "tp",
        "fp",
        "fn",
        "matched_rules",
        "unmatched_actual_rules",
        "unmatched_expected_rules",
        "runtime_sec",
        "success",
        "error_message",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in case_results:
            matched_str = ";".join(
                f"{p['expected']['rule']}@{p['actual'].get('line')}" for p in r.matched_pairs
            )
            un_act_str = ";".join(
                f"{a.get('code', a.get('rule', ''))}@{a.get('line')}" for a in r.unmatched_actual
            )
            un_exp_str = ";".join(
                f"{e.get('rule')}@{e.get('line')}" for e in r.unmatched_expected
            )
            writer.writerow({
                "case_id": r.case_id,
                "category": r.category,
                "file_path": r.file_path,
                "status": r.status,
                "expected_count": r.expected_count,
                "actual_count": r.actual_count,
                "tp": r.tp,
                "fp": r.fp,
                "fn": r.fn,
                "matched_rules": matched_str,
                "unmatched_actual_rules": un_act_str,
                "unmatched_expected_rules": un_exp_str,
                "runtime_sec": round(r.runtime_sec, 4),
                "success": r.success,
                "error_message": r.error_message or "",
            })
