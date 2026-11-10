"""
runner.py - CLI Runner for Phase 6 Deterministic Benchmark.

Executes the benchmark cases through the existing production deterministic pipeline
(workflow.run_deterministic_only), evaluates findings against ground truth, and outputs
both a readable console summary and optional JSON/CSV artifacts.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any

from benchmark.metrics import (
    ExpectedFinding,
    MatchResult,
    compute_benchmark_summary,
    evaluate_case,
    export_cases_csv,
    export_summary_json,
    normalize_path,
)
from github_review_workflow import deduplicate_findings
from workflow import run_deterministic_only, run_review_pipeline


def get_current_git_commit(cwd: str | None = None) -> str | None:
    """Safely obtain current git commit hash if git is available."""
    try:
        git_bin = shutil.which("git") or "git"
        res = subprocess.run(
            [git_bin, "rev-parse", "HEAD"],
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    return None


def load_manifest(manifest_path: str) -> list[dict[str, Any]]:
    """Load benchmark cases from the manifest JSON file."""
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise TypeError("Invalid manifest format: 'cases' must be a list.")
    return cases


def print_console_summary(summary: dict[str, Any]) -> None:
    """Print an executive summary of the benchmark results to stdout."""
    overall = summary["overall"]
    defect = summary.get("defect_metrics", {})
    clean = summary.get("clean_metrics", {})
    detectors = summary.get("by_detector", {})
    categories = summary.get("by_category", {})

    print("=" * 78)
    print("  PHASE 6: DETERMINISTIC CODE REVIEW BENCHMARK REPORT")
    print(f"  Mode: {summary.get('mode', 'deterministic').upper()} | Timestamp: {summary.get('timestamp')}")
    if summary.get("git_commit"):
        print(f"  Git Commit: {summary['git_commit']}")
    print("=" * 78)
    print()

    # Section 1: Labeled Defect Findings Performance
    print("1. LABELED DEFECT FINDINGS PERFORMANCE (Precision / Recall / F1):")
    print("-" * 78)
    print(
        f"  Defect Cases Evaluated  : {defect.get('cases_total', 0)} "
        f"(Passed: {defect.get('cases_passed', 0)}, "
        f"Failed: {defect.get('cases_failed', 0)}, "
        f"Mismatched: {defect.get('cases_mismatched', 0)})"
    )
    print(f"  Expected Findings Count : {defect.get('expected_findings', 0)}")
    print(f"  Actual Findings Count   : {defect.get('actual_findings', 0)}")
    print(f"  Matched Findings (TP)   : {defect.get('matched_findings', 0)}")
    print(f"  Unmatched Actual (FP)   : {defect.get('unmatched_actual', 0)}")
    print(f"  Unmatched Expected (FN) : {defect.get('unmatched_expected', 0)}")
    print(f"  Defect Precision        : {defect.get('precision', 0.0):.4f}")
    print(f"  Defect Recall           : {defect.get('recall', 0.0):.4f}")
    print(f"  Defect F1 Score         : {defect.get('f1_score', 0.0):.4f}")
    print("-" * 78)
    print()

    # Section 2: Clean & Safe-Edge Subset (False Positive Control)
    print("2. CLEAN & SAFE-EDGE EVALUATION (False Positive Control):")
    print("-" * 78)
    print(
        f"  Clean/Safe Cases Total  : {clean.get('cases_total', 0)} "
        f"(Passed: {clean.get('cases_passed', 0)}, Failed: {clean.get('cases_failed', 0)})"
    )
    print(f"  Cases with False Pos.   : {clean.get('cases_with_fp', 0)}")
    print(f"  Clean Case FPR          : {clean.get('clean_case_fpr', 0.0):.4f}")
    print(f"  Clean FP Findings Count : {clean.get('unmatched_actual', 0)}")
    print("-" * 78)
    print()

    # Section 3: Overall Performance Table
    print("3. OVERALL BENCHMARK PERFORMANCE:")
    print("-" * 78)
    print(
        f"  Total Cases Evaluated   : {overall['cases_total']} "
        f"(Passed: {overall.get('cases_passed', 0)}, "
        f"Failed: {overall.get('cases_failed', 0)}, "
        f"Mismatched: {overall.get('cases_mismatched', 0)})"
    )
    print(f"  Expected Findings Count : {overall.get('expected_findings', overall['tp'] + overall['fn'])}")
    print(f"  Actual Findings Count   : {overall.get('actual_findings', overall['tp'] + overall['fp'])}")
    print(f"  Matched Findings (TP)   : {overall['tp']}")
    print(f"  Unmatched Actual (FP)   : {overall['fp']}")
    print(f"  Unmatched Expected (FN) : {overall['fn']}")
    print(f"  Overall Precision       : {overall['precision']:.4f}")
    print(f"  Overall Recall          : {overall['recall']:.4f}")
    print(f"  Overall F1 Score        : {overall['f1_score']:.4f}")
    print(f"  Total Runtime           : {overall['total_runtime_seconds']:.2f}s")
    print(f"  Tool Failures / Crashes : {overall['tool_failures']}")
    print("-" * 78)
    print()

    # Section 4: Detector Breakdown Table
    print("4. DETECTOR BREAKDOWN:")
    print("-" * 78)
    print(f"{'Detector':<20} | {'Exp':<5} | {'Act':<5} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Precision':<10} | {'Recall':<10} | {'F1 Score'}")
    print("-" * 78)
    for det_name, m in detectors.items():
        print(
            f"{det_name:<20} | {m.get('expected', m['tp'] + m['fn']):<5} | "
            f"{m.get('actual', m['tp'] + m['fp']):<5} | {m['tp']:<5} | {m['fp']:<5} | {m['fn']:<5} | "
            f"{m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:.4f}"
        )
    print("-" * 78)
    print()

    # Section 5: Category Breakdown Table
    print("5. CATEGORY BREAKDOWN:")
    print("-" * 78)
    print(f"{'Category':<15} | {'Cases':<6} | {'Passed':<6} | {'Failed':<6} | {'Exp':<5} | {'Act':<5} | {'TP':<4} | {'FP':<4} | {'FN'}")
    print("-" * 78)
    for cat_name, c in categories.items():
        print(
            f"{cat_name:<15} | {c['cases']:<6} | {c.get('passed', 0):<6} | {c.get('failed', 0):<6} | "
            f"{c['expected']:<5} | {c['actual']:<5} | {c['tp']:<4} | {c['fp']:<4} | {c['fn']}"
        )
    print("-" * 78)
    print()

    # Section 6: Case Level Outcomes
    print("6. CASE LEVEL OUTCOMES:")
    print("-" * 78)
    print(f"{'Case ID':<12} | {'Category':<12} | {'Exp':<4} | {'Act':<4} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Status'}")
    print("-" * 78)
    for r in summary.get("case_results", []):
        status = r.get("status", "PASSED" if r.get("success") and r.get("fp") == 0 and r.get("fn") == 0 else ("FAILED" if not r.get("success") else "MISMATCH"))
        print(
            f"{r['case_id']:<12} | {r['category']:<12} | {r['expected_count']:<4} | "
            f"{r['actual_count']:<4} | {r['tp']:<4} | {r['fp']:<4} | {r['fn']:<4} | {status}"
        )
    print("-" * 78)
    print()


def run_benchmark(
    manifest_path: str = "benchmark/manifest.json",
    category: str | None = None,
    case_id: str | None = None,
    suite: str | None = None,
    output_dir: str | None = None,
    tolerance: int = 1,
    cwd: str | None = None,
    quiet: bool = False,
    mode: str = "deterministic",
    eval_report: str | None = None,
    dedup: bool = False,
) -> tuple[dict[str, Any], int]:
    """
    Execute the benchmark or evaluation pipeline across selected cases.

    Parameters:
        manifest_path: Path to manifest JSON file.
        category: Optional category filter (e.g. 'security', 'bugs', 'error_handling', 'clean').
        case_id: Optional single case ID to run.
        suite: Optional suite filter ('baseline', 'expanded', 'evaluation', or 'all').
        output_dir: Optional directory to export benchmark_summary.json and benchmark_cases.csv.
        tolerance: Line distance matching tolerance window (default: 1 line).
        cwd: Base workspace directory.
        quiet: Suppress stdout printing when True.
        mode: Pipeline mode ('deterministic' or 'full').
        eval_report: Optional file path to export standalone JSON evaluation report.
        dedup: If True, apply finding deduplication before ground-truth matching.

    Returns:
        (summary_dict, exit_code) where exit_code is 0 on success, 1 on tool failure.
    """
    base_dir = cwd or os.getcwd()
    full_manifest_path = (
        manifest_path
        if os.path.isabs(manifest_path)
        else os.path.join(base_dir, manifest_path)
    )

    # If category or suite refers to evaluation error_handling, automatically select eval_manifest if using default
    if (category == "error_handling" or suite == "evaluation") and manifest_path == "benchmark/manifest.json":
        eval_cand = os.path.join(base_dir, "benchmark/eval_manifest.json")
        if os.path.isfile(eval_cand):
            full_manifest_path = eval_cand

    all_cases = load_manifest(full_manifest_path)

    # Apply filters
    filtered_cases: list[dict[str, Any]] = []
    for c in all_cases:
        if category and c.get("category") != category:
            continue
        if case_id and c.get("case_id") != case_id:
            continue
        if suite and suite != "all" and c.get("suite") != suite:
            continue
        filtered_cases.append(c)

    if not filtered_cases:
        if not quiet:
            print(f"Warning: No benchmark cases matched criteria (category={category}, case_id={case_id}, suite={suite}).")
        empty_summary = compute_benchmark_summary(
            case_results=[],
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            git_commit=get_current_git_commit(base_dir),
            total_runtime_seconds=0.0,
            tool_failures=0,
            mode=mode,
        )
        return empty_summary, 0

    case_results: list[MatchResult] = []
    total_start = time.perf_counter()
    tool_failures = 0

    for c in filtered_cases:
        cid = c["case_id"]
        cat = c.get("category", "unknown")
        rel_path = c.get("file_path", "")
        target_path = (
            rel_path if os.path.isabs(rel_path) else os.path.join(base_dir, rel_path)
        )

        # Expected findings
        exp_raw = c.get("expected_findings", [])
        expected = [
            ExpectedFinding(
                rule=e["rule"],
                tool=e.get("tool"),
                line=e.get("line"),
                tolerance=e.get("tolerance", tolerance),
                file_path=e.get("file_path", rel_path),
            )
            for e in exp_raw
        ]

        # Execute designated pipeline on case
        case_start = time.perf_counter()
        actual_findings: list[dict[str, Any]] = []
        case_success = True
        err_msg: str | None = None

        try:
            if mode == "full":
                state = run_review_pipeline(
                    target_path=target_path,
                    enable_ai=True,
                    enable_sandbox=False,
                    enable_debugging=False,
                )
            else:
                state = run_deterministic_only(target_path)

            ruff_findings = state.get("ruff_findings")
            bandit_findings = state.get("bandit_findings")
            semantic_findings = state.get("semantic_findings")

            # Validate that required scanners produced output lists (not None)
            missing_scanners: list[str] = []
            if ruff_findings is None:
                missing_scanners.append("Ruff")
            else:
                actual_findings.extend(ruff_findings)

            if bandit_findings is None:
                missing_scanners.append("Bandit")
            else:
                actual_findings.extend(bandit_findings)

            if semantic_findings is None:
                missing_scanners.append("SemanticAnalyzer")
            else:
                actual_findings.extend(semantic_findings)

            if dedup:
                actual_findings = deduplicate_findings(actual_findings)

            if missing_scanners:
                err_msg = f"Missing required scanner output: {', '.join(missing_scanners)}"
                tool_failures += len(missing_scanners)
                case_success = False

            agent_errors = state.get("agent_errors", [])
            if agent_errors:
                err_str = "; ".join(agent_errors)
                err_msg = f"{err_msg}; {err_str}" if err_msg else err_str
                tool_failures += len(agent_errors)
                case_success = False

        except Exception as exc:  # noqa: BLE001
            case_success = False
            err_msg = f"Benchmark runner error on case {cid}: {exc}"
            tool_failures += 1

        case_duration = time.perf_counter() - case_start

        match_res = evaluate_case(
            case_id=cid,
            category=cat,
            file_path=rel_path,
            expected_findings=expected,
            actual_findings=actual_findings,
            runtime_sec=case_duration,
            success=case_success,
            error_message=err_msg,
            default_tolerance=tolerance,
        )
        case_results.append(match_res)

    total_duration = time.perf_counter() - total_start
    timestamp_str = datetime.datetime.now(datetime.UTC).isoformat()
    commit_sha = get_current_git_commit(base_dir)

    summary = compute_benchmark_summary(
        case_results=case_results,
        timestamp=timestamp_str,
        git_commit=commit_sha,
        total_runtime_seconds=total_duration,
        tool_failures=tool_failures,
        mode=mode,
    )

    if not quiet:
        print_console_summary(summary)

    # Write output artifacts if requested
    if output_dir:
        abs_output_dir = (
            output_dir if os.path.isabs(output_dir) else os.path.join(base_dir, output_dir)
        )
        json_path = os.path.join(abs_output_dir, "benchmark_summary.json")
        csv_path = os.path.join(abs_output_dir, "benchmark_cases.csv")
        export_summary_json(summary, json_path)
        export_cases_csv(case_results, csv_path)
        if not quiet:
            print(f"Saved machine-readable results to:\n  - {normalize_path(json_path)}\n  - {normalize_path(csv_path)}\n")

    if eval_report:
        abs_eval_report = (
            eval_report if os.path.isabs(eval_report) else os.path.join(base_dir, eval_report)
        )
        export_summary_json(summary, abs_eval_report)
        if not quiet:
            print(f"Saved evaluation report JSON to: {normalize_path(abs_eval_report)}")

    exit_code = 1 if tool_failures > 0 else 0
    return summary, exit_code


def run_evaluation(
    manifest_path: str = "benchmark/eval_manifest.json",
    output_report: str | None = "evaluation_report.json",
    category: str | None = None,
    case_id: str | None = None,
    suite: str | None = None,
    mode: str = "deterministic",
    tolerance: int = 1,
    dedup: bool = False,
    quiet: bool = True,
    cwd: str | None = None,
) -> tuple[dict[str, Any], int]:
    """
    Execute the Stage 7.1 Real-World Evaluation Framework against the evaluation dataset.

    Returns:
        (summary_dict, exit_code)
    """
    return run_benchmark(
        manifest_path=manifest_path,
        category=category,
        case_id=case_id,
        suite=suite,
        output_dir=None,
        tolerance=tolerance,
        cwd=cwd,
        quiet=quiet,
        mode=mode,
        eval_report=output_report,
        dedup=dedup,
    )


def main() -> None:
    """CLI entry point for running the benchmark."""
    parser = argparse.ArgumentParser(
        description="Phase 6 Deterministic Benchmark Runner for Code Review Pipeline"
    )
    parser.add_argument(
        "--manifest",
        dest="manifest",
        default="benchmark/manifest.json",
        help="Path to manifest JSON file (default: 'benchmark/manifest.json')",
    )
    parser.add_argument(
        "--category",
        dest="category",
        default=None,
        choices=["clean", "bugs", "security", "multi", "safe_edges", "error_handling"],
        help="Filter benchmark cases by category",
    )
    parser.add_argument(
        "--case",
        dest="case_id",
        default=None,
        help="Run only a single specific benchmark case ID (e.g. 'bug_01')",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Directory to write benchmark_summary.json and benchmark_cases.csv",
    )
    parser.add_argument(
        "--eval-report",
        dest="eval_report",
        default=None,
        help="File path to write standalone machine-readable evaluation report JSON",
    )
    parser.add_argument(
        "--tolerance",
        dest="tolerance",
        type=int,
        default=1,
        help="Line matching tolerance window (default: 1 line)",
    )
    parser.add_argument(
        "--suite",
        dest="suite",
        default="all",
        choices=["all", "baseline", "expanded", "evaluation"],
        help="Filter benchmark cases by suite ('baseline', 'expanded', 'evaluation', or 'all')",
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        choices=["deterministic", "full"],
        default="deterministic",
        help="Pipeline execution mode ('deterministic' or 'full')",
    )
    parser.add_argument(
        "--dedup",
        dest="dedup",
        action="store_true",
        default=False,
        help="Apply finding deduplication before ground-truth matching",
    )
    args = parser.parse_args()

    _, exit_code = run_benchmark(
        manifest_path=args.manifest,
        category=args.category,
        case_id=args.case_id,
        suite=args.suite,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
        mode=args.mode,
        eval_report=args.eval_report,
        dedup=args.dedup,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
