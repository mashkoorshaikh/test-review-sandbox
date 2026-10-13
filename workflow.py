"""
workflow.py - LangGraph Workflow Orchestration for Multi-Agent Review.

Builds and coordinates a graph of specialized review agents:
  1. Linter Agent (Ruff + AI explanation)
  2. Security Agent (Bandit + AI threat & remediation assessment)
  3. Test-Writer Agent (pytest test generation with validated temp-workspace storage)
  4. Sandbox Runner (Docker sandboxed test execution - skipped if Docker unavailable)
  5. Debugging Agent (patch proposal on test failure, with retry limit)
  6. Orchestrator (consolidation into structured final report)

Execution Flow (Milestone 3):
  - Fan-out: START -> linter_agent AND security_agent (parallel)
  - Fan-in:  linter_agent & security_agent -> test_writer_agent
  - Conditional: test_writer_agent -> sandbox_runner (if stored tests available)
                                   -> orchestrator (if no stored tests / no AI)
  - Conditional: sandbox_runner -> orchestrator (if passed or Docker unavailable)
                                -> debugging_agent (if tests failed)
  - Conditional: debugging_agent -> orchestrator (if patch verified or max attempts reached)
  - Finalize: orchestrator -> END
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Callable, Dict, List, Literal, Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agents import LinterAgent, SecurityAgent, TestWriterAgent
from debugging_agent import DebuggingAgent, MAX_DEBUG_ATTEMPTS
from sandbox_runner import SandboxResult, is_docker_available, run_tests_in_sandbox
from state import (
    ReviewState,
    create_initial_state,
    issue_to_dict,
)
from tools import ToolResult, run_bandit, run_ruff

logger = logging.getLogger(__name__)


def _sandbox_result_to_dict(result: SandboxResult) -> Dict[str, Any]:
    """Convert a SandboxResult to a JSON-serializable dict for state storage."""
    return {
        "executed": result.executed,
        "exit_code": result.exit_code,
        "passed": result.passed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": result.timed_out,
        "docker_unavailable": result.docker_unavailable,
        "error_message": result.error_message,
    }


def create_orchestrator_report(state: ReviewState) -> Dict[str, Any]:
    """
    Synthesize all agent outputs from the graph state into a structured report.
    """
    target = state.get("target_path", "")
    ruff_issues = state.get("ruff_findings", [])
    bandit_issues = state.get("bandit_findings", [])
    ai_explanations = state.get("ai_explanations", {})
    test_suggestions = state.get("generated_test_suggestions", [])
    errors = state.get("agent_errors", [])

    # Milestone 3 fields
    stored_test_files = state.get("stored_test_files", [])
    sandbox_results = state.get("sandbox_results", [])
    patch_proposals = state.get("patch_proposals", [])
    patch_verified = state.get("patch_verified", False)
    debug_attempts = state.get("debug_attempts", 0)
    sandbox_skipped = state.get("sandbox_skipped", False)

    total_code_issues = len(ruff_issues) + len(bandit_issues)

    # Determine sandbox/test status
    sandbox_passed = any(r.get("passed", False) for r in sandbox_results)
    sandbox_executed = any(r.get("executed", False) for r in sandbox_results)

    if errors:
        status = (
            "FAILED"
            if (total_code_issues == 0 and any("error" in e.lower() for e in errors))
            else "COMPLETED_WITH_WARNINGS"
        )
    elif total_code_issues > 0:
        status = "ATTENTION_REQUIRED"
    else:
        status = "PASSED"

    return {
        "target_path": target,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "summary": {
            "status": status,
            "total_issues_found": total_code_issues,
            "ruff_issues_count": len(ruff_issues),
            "bandit_issues_count": len(bandit_issues),
            "agent_error_count": len(errors),
            # Milestone 3 summary
            "stored_test_files_count": len(stored_test_files),
            "sandbox_executed": sandbox_executed,
            "sandbox_passed": sandbox_passed,
            "sandbox_skipped": sandbox_skipped,
            "debug_attempts": debug_attempts,
            "patch_verified": patch_verified,
            "patch_proposals_count": len(patch_proposals),
        },
        "deterministic_findings": {
            "ruff": ruff_issues,
            "bandit": bandit_issues,
        },
        "ai_explanations": ai_explanations,
        "test_suggestions": test_suggestions,
        "stored_test_files": stored_test_files,
        "sandbox_results": sandbox_results,
        "patch_proposals": patch_proposals,
        "agent_errors": errors,
    }


def build_review_graph(
    llm: Optional[BaseChatModel] = None,
    ruff_runner: Optional[Callable[..., ToolResult]] = None,
    bandit_runner: Optional[Callable[..., ToolResult]] = None,
    workspace_root: Optional[str] = None,
    enable_sandbox: bool = True,
    enable_debugging: bool = True,
    max_debug_attempts: int = MAX_DEBUG_ATTEMPTS,
    test_workspace_dir: Optional[str] = None,
    sandbox_runner_fn: Optional[Callable] = None,
):
    """
    Construct and compile the LangGraph StateGraph review workflow.

    Args:
        llm: Optional language model instance.
        ruff_runner: Callable for executing Ruff (defaults to tools.run_ruff).
        bandit_runner: Callable for executing Bandit (defaults to tools.run_bandit).
        workspace_root: Optional root directory override for workspace sandboxing.
        enable_sandbox: If False, sandbox execution node is bypassed.
        enable_debugging: If False, debugging agent is never invoked.
        max_debug_attempts: Maximum debugging retry iterations (default 2).
        test_workspace_dir: Directory for generated test files (temp dir if None).
        sandbox_runner_fn: Injectable sandbox runner (for testing).

    Returns:
        Compiled StateGraph runnable.
    """
    linter_agent = LinterAgent(
        llm=llm, ruff_runner=ruff_runner, workspace_root=workspace_root
    )
    security_agent = SecurityAgent(
        llm=llm, bandit_runner=bandit_runner, workspace_root=workspace_root
    )
    test_writer_agent = TestWriterAgent(
        llm=llm,
        workspace_root=workspace_root,
        test_workspace_dir=test_workspace_dir,
        store_tests=True,
    )
    debugging_agent = DebuggingAgent(
        llm=llm,
        max_attempts=max_debug_attempts,
        workspace_root=workspace_root,
    )

    # Use injected sandbox runner (for tests) or real one
    _sandbox_fn = sandbox_runner_fn or run_tests_in_sandbox

    # -------------------------------------------------------------------------
    # Node functions
    # -------------------------------------------------------------------------

    def linter_node(state: ReviewState) -> Dict[str, Any]:
        target = state.get("target_path", "")
        return linter_agent.run(target)

    def security_node(state: ReviewState) -> Dict[str, Any]:
        target = state.get("target_path", "")
        return security_agent.run(target)

    def test_writer_node(state: ReviewState) -> Dict[str, Any]:
        target = state.get("target_path", "")
        result = test_writer_agent.run(target)
        return result

    def sandbox_node(state: ReviewState) -> Dict[str, Any]:
        """Execute stored tests in a Docker sandbox, or skip gracefully."""
        stored_files = state.get("stored_test_files", [])
        target = state.get("target_path", "")

        # Resolve source directory (parent of target if it's a file)
        import os
        from pathlib import Path as _Path

        target_p = _Path(target).resolve()
        source_dir = str(target_p.parent if target_p.is_file() else target_p)

        if not stored_files:
            return {
                "sandbox_results": [{"executed": False, "passed": False, "error_message": "No stored test files to execute."}],
                "sandbox_skipped": True,
                "agent_errors": [],
            }

        # Check Docker availability (or use injected runner)
        if sandbox_runner_fn is None and not is_docker_available():
            logger.info("Docker unavailable, skipping sandbox execution.")
            return {
                "sandbox_results": [_sandbox_result_to_dict(SandboxResult(
                    docker_unavailable=True,
                    error_message=(
                        "Docker is not available. Sandbox test execution was skipped. "
                        "Install Docker to enable sandboxed execution."
                    ),
                ))],
                "sandbox_skipped": True,
                "agent_errors": [],
            }

        sandbox_results: List[Dict[str, Any]] = []
        errors: List[str] = []

        for test_file in stored_files:
            try:
                result = _sandbox_fn(
                    source_dir=source_dir,
                    test_file=test_file,
                )
                sandbox_results.append(_sandbox_result_to_dict(result))
            except Exception as exc:
                err_msg = f"Sandbox runner error for '{test_file}': {type(exc).__name__}: {exc}"
                errors.append(err_msg)
                sandbox_results.append({
                    "executed": False,
                    "passed": False,
                    "error_message": err_msg,
                })

        return {
            "sandbox_results": sandbox_results,
            "sandbox_skipped": False,
            "agent_errors": errors,
        }

    def debugging_node(state: ReviewState) -> Dict[str, Any]:
        """Propose patches for failed sandbox tests (bounded retry)."""
        target = state.get("target_path", "")
        ruff_findings = state.get("ruff_findings", [])
        bandit_findings = state.get("bandit_findings", [])
        sandbox_results = state.get("sandbox_results", [])
        stored_files = state.get("stored_test_files", [])
        current_attempts = state.get("debug_attempts", 0)

        # Collect failed test output
        failed_output_parts: List[str] = []
        for sr in sandbox_results:
            if not sr.get("passed", False):
                stdout = sr.get("stdout", "")
                stderr = sr.get("stderr", "")
                if stdout:
                    failed_output_parts.append(f"STDOUT:\n{stdout}")
                if stderr:
                    failed_output_parts.append(f"STDERR:\n{stderr}")

        failed_output = "\n\n".join(failed_output_parts)[:4000]

        # Determine primary source file to patch (first file in target dir)
        import os
        from pathlib import Path as _Path
        target_p = _Path(target).resolve()
        if target_p.is_file():
            target_file_to_patch = str(target_p)
        else:
            # Find first Python file to patch
            py_files = list(target_p.glob("*.py"))
            target_file_to_patch = str(py_files[0]) if py_files else str(target_p)

        test_file = stored_files[0] if stored_files else None

        result = debugging_agent.run(
            target_file=target_file_to_patch,
            ruff_findings=ruff_findings,
            bandit_findings=bandit_findings,
            failed_test_output=failed_output,
            sandbox_runner=_sandbox_fn if test_file else None,
            test_file=test_file,
            source_dir=str(target_p.parent if target_p.is_file() else target_p),
        )

        return {
            "patch_proposals": result.get("patch_proposals", []),
            "patch_verified": result.get("patch_verified", False),
            "debug_attempts": current_attempts + max_debug_attempts,
            "agent_errors": result.get("agent_errors", []),
        }

    def orchestrator_node(state: ReviewState) -> Dict[str, Any]:
        report = create_orchestrator_report(state)
        return {"final_review_report": report}

    # -------------------------------------------------------------------------
    # Routing functions
    # -------------------------------------------------------------------------

    def route_after_test_writer(state: ReviewState) -> Literal["sandbox_runner", "orchestrator"]:
        """Go to sandbox if sandbox execution is enabled."""
        if not enable_sandbox:
            return "orchestrator"
        return "sandbox_runner"

    def route_after_sandbox(state: ReviewState) -> Literal["debugging_agent", "orchestrator"]:
        """Go to debugging agent if tests failed and debugging is enabled."""
        if not enable_debugging:
            return "orchestrator"
        if state.get("sandbox_skipped", False):
            return "orchestrator"
        sandbox_results = state.get("sandbox_results", [])
        any_failed = any(
            r.get("executed", False) and not r.get("passed", False)
            for r in sandbox_results
        )
        current_attempts = state.get("debug_attempts", 0)
        if any_failed and current_attempts < max_debug_attempts:
            return "debugging_agent"
        return "orchestrator"

    def route_after_debugging(state: ReviewState) -> Literal["orchestrator"]:
        """Always proceed to orchestrator after debugging (bounded by max_attempts in agent)."""
        return "orchestrator"

    # -------------------------------------------------------------------------
    # Build graph
    # -------------------------------------------------------------------------

    builder = StateGraph(ReviewState)

    # Register nodes
    builder.add_node("linter_agent", linter_node)
    builder.add_node("security_agent", security_node)
    builder.add_node("test_writer_agent", test_writer_node)
    builder.add_node("sandbox_runner", sandbox_node)
    builder.add_node("debugging_agent", debugging_node)
    builder.add_node("orchestrator", orchestrator_node)

    # Fan-out: START executes Linter and Security agents in parallel
    builder.add_edge(START, "linter_agent")
    builder.add_edge(START, "security_agent")

    # Fan-in: Wait for both Linter and Security before running Test-Writer
    builder.add_edge("linter_agent", "test_writer_agent")
    builder.add_edge("security_agent", "test_writer_agent")

    # Conditional: test_writer -> sandbox or orchestrator
    builder.add_conditional_edges(
        "test_writer_agent",
        route_after_test_writer,
        {"sandbox_runner": "sandbox_runner", "orchestrator": "orchestrator"},
    )

    # Conditional: sandbox -> debugging or orchestrator
    builder.add_conditional_edges(
        "sandbox_runner",
        route_after_sandbox,
        {"debugging_agent": "debugging_agent", "orchestrator": "orchestrator"},
    )

    # Conditional: debugging -> orchestrator
    builder.add_conditional_edges(
        "debugging_agent",
        route_after_debugging,
        {"orchestrator": "orchestrator"},
    )

    # Finalize
    builder.add_edge("orchestrator", END)

    return builder.compile()


def run_deterministic_only(
    target_path: str,
    ruff_runner: Optional[Callable[..., ToolResult]] = None,
    bandit_runner: Optional[Callable[..., ToolResult]] = None,
) -> ReviewState:
    """
    Run purely deterministic scanners (Ruff and Bandit) without AI agents.
    """
    actual_ruff = ruff_runner or run_ruff
    actual_bandit = bandit_runner or run_bandit

    ruff_res = actual_ruff(target_path)
    bandit_res = actual_bandit(target_path)

    state = create_initial_state(target_path)
    state["ruff_findings"] = [issue_to_dict(i) for i in ruff_res.issues]
    state["bandit_findings"] = [issue_to_dict(i) for i in bandit_res.issues]
    state["ai_explanations"] = {
        "linter": "[AI analysis disabled by user (--no-ai).]",
        "security": "[AI analysis disabled by user (--no-ai).]",
    }
    state["generated_test_suggestions"] = [
        {"target_file": target_path, "suggestions": "[AI test generation disabled by user (--no-ai).]"}
    ]

    errors: List[str] = []
    if not ruff_res.success:
        errors.append(f"Ruff: {ruff_res.error_message}")
    if not bandit_res.success:
        errors.append(f"Bandit: {bandit_res.error_message}")
    state["agent_errors"] = errors
    state["final_review_report"] = create_orchestrator_report(state)

    return state


def run_review_pipeline(
    target_path: str,
    enable_ai: bool = True,
    llm: Optional[BaseChatModel] = None,
    ruff_runner: Optional[Callable[..., ToolResult]] = None,
    bandit_runner: Optional[Callable[..., ToolResult]] = None,
    workspace_root: Optional[str] = None,
    enable_sandbox: bool = True,
    enable_debugging: bool = True,
    max_debug_attempts: int = MAX_DEBUG_ATTEMPTS,
    test_workspace_dir: Optional[str] = None,
    sandbox_runner_fn: Optional[Callable] = None,
) -> ReviewState:
    """
    Execute the code review pipeline on the designated target path.

    If enable_ai is False, runs only the deterministic scanner.
    If enable_ai is True, executes the full LangGraph multi-agent workflow.
    """
    if not enable_ai:
        return run_deterministic_only(
            target_path=target_path,
            ruff_runner=ruff_runner,
            bandit_runner=bandit_runner,
        )

    app = build_review_graph(
        llm=llm,
        ruff_runner=ruff_runner,
        bandit_runner=bandit_runner,
        workspace_root=workspace_root,
        enable_sandbox=enable_sandbox,
        enable_debugging=enable_debugging,
        max_debug_attempts=max_debug_attempts,
        test_workspace_dir=test_workspace_dir,
        sandbox_runner_fn=sandbox_runner_fn,
    )
    initial_state = create_initial_state(target_path)
    final_state = app.invoke(initial_state)
    return final_state
