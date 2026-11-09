"""
state.py - Shared Typed State Schema for LangGraph Review Pipeline.

Defines the centralized state exchanged across the specialized review agents:
  1. target_path: Target file or directory being reviewed.
  2. ruff_findings: List of serialized lint findings from Ruff.
  3. bandit_findings: List of serialized security findings from Bandit.
  4. ai_explanations: Map of agent identifier to AI-generated explanation texts.
  5. generated_test_suggestions: List of test cases suggested by the Test-Writer agent.
  6. agent_errors: Aggregated list of error messages from any agent or provider failure.
  7. final_review_report: Consolidated structured report produced by the orchestrator.

  Milestone 3 additions:
  8. stored_test_files: Paths to validated test files written to isolated temp workspace.
  9. sandbox_results: Outcomes of sandboxed test execution attempts.
  10. patch_proposals: Structured patch suggestions from the Debugging Agent.
  11. patch_verified: Whether any proposed patch passed sandbox verification.
  12. debug_attempts: Count of debugging retry iterations performed.
  13. sandbox_skipped: True if Docker was unavailable and execution was skipped.
"""

from __future__ import annotations

import operator
from dataclasses import asdict
from typing import Annotated, Any

from typing_extensions import TypedDict

from tools import Issue


def merge_dict_reducer(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
    State reducer for merging dictionary updates across parallel graph branches.

    Allows Linter and Security agents to update `ai_explanations` concurrently
    without overwriting each other.
    """
    merged = dict(a or {})
    merged.update(b or {})
    return merged


class ReviewState(TypedDict, total=False):
    """
    Shared typed state for the multi-agent code review LangGraph workflow.
    """
    # Milestone 1 & 2 fields
    target_path: str
    ruff_findings: list[dict[str, Any]]
    bandit_findings: list[dict[str, Any]]
    ai_explanations: Annotated[dict[str, str], merge_dict_reducer]
    generated_test_suggestions: list[dict[str, Any]]
    agent_errors: Annotated[list[str], operator.add]
    final_review_report: dict[str, Any]

    # Milestone 3 fields
    stored_test_files: list[str]           # Paths to validated test files in temp workspace
    sandbox_results: list[dict[str, Any]]  # SandboxResult serialized dicts
    patch_proposals: list[dict[str, Any]]  # PatchProposal serialized dicts
    patch_verified: bool                   # Did any proposed patch pass sandbox tests?
    debug_attempts: int                    # Number of debugging attempts used
    sandbox_skipped: bool                  # Was sandbox skipped (Docker unavailable)?
    semantic_findings: list[dict[str, Any]]  # Milestone 5: AST semantic bug findings
    tool_status: Annotated[dict[str, bool], merge_dict_reducer]  # Structured tool execution success/failure state


def issue_to_dict(issue: Issue) -> dict[str, Any]:
    """Convert an Issue dataclass instance to a serializable dictionary."""
    return asdict(issue)


def dict_to_issue(data: dict[str, Any]) -> Issue:
    """Reconstruct an Issue dataclass instance from a dictionary."""
    return Issue(
        code=data.get("code", "UNKNOWN"),
        message=data.get("message", ""),
        file_path=data.get("file_path", ""),
        line=data.get("line"),
        column=data.get("column"),
        severity=data.get("severity", "WARNING"),
        tool=data.get("tool", "unknown"),
        fixable=data.get("fixable", False),
        confidence=data.get("confidence", "HIGH"),
        explanation=data.get("explanation", ""),
        suggested_fix=data.get("suggested_fix", ""),
    )


def create_initial_state(target_path: str) -> ReviewState:
    """Create an empty, initialized ReviewState instance for a target path."""
    return {
        "target_path": target_path,
        "ruff_findings": [],
        "bandit_findings": [],
        "semantic_findings": [],
        "ai_explanations": {},
        "generated_test_suggestions": [],
        "agent_errors": [],
        "final_review_report": {},
        # Milestone 3
        "stored_test_files": [],
        "sandbox_results": [],
        "patch_proposals": [],
        "patch_verified": False,
        "debug_attempts": 0,
        "sandbox_skipped": False,
        "tool_status": {},
    }
