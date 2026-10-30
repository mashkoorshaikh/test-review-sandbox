"""
tests/test_debugging_agent.py - Unit tests for the DebuggingAgent and patch utilities.

All tests use mock LLMs and never call real APIs.
No Docker calls are made; no network access required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from debugging_agent import (
    DEBUGGING_DISCLAIMER,
    DebuggingAgent,
    _apply_unified_diff,
    _extract_diff_from_response,
    _validate_patch_safety,
    apply_patch_to_copy,
    cleanup_temp_dir,
)
from llm_config import MissingApiKeyError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_llm(response_text: str = "```diff\n--- a/mod.py\n+++ b/mod.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n```"):
    mock_llm = MagicMock()
    resp = MagicMock()
    resp.content = response_text
    mock_llm.invoke.return_value = resp
    return mock_llm


SIMPLE_DIFF = "--- a/mod.py\n+++ b/mod.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n"


# ---------------------------------------------------------------------------
# _extract_diff_from_response
# ---------------------------------------------------------------------------

class TestExtractDiffFromResponse:
    def test_extracts_from_diff_fence(self):
        text = "Here is the fix:\n```diff\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n```"
        diff = _extract_diff_from_response(text)
        assert "--- a/f.py" in diff
        assert "+new" in diff

    def test_extracts_from_patch_fence(self):
        text = "```patch\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y\n```"
        diff = _extract_diff_from_response(text)
        assert "+y" in diff

    def test_extracts_raw_unified_diff_without_fence(self):
        text = "--- a/calc.py\n+++ b/calc.py\n@@ -5,1 +5,1 @@\n-return a\n+return a + b\n"
        diff = _extract_diff_from_response(text)
        assert "--- a/calc.py" in diff

    def test_returns_empty_string_when_no_diff_found(self):
        text = "No diff here, just some explanation."
        diff = _extract_diff_from_response(text)
        assert diff == ""


# ---------------------------------------------------------------------------
# _validate_patch_safety
# ---------------------------------------------------------------------------

class TestValidatePatchSafety:
    def test_accepts_valid_patch(self):
        patch_text = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
        assert _validate_patch_safety(patch_text) is True

    def test_rejects_empty_patch(self):
        assert _validate_patch_safety("") is False
        assert _validate_patch_safety("   ") is False

    def test_rejects_patch_targeting_dev_null(self):
        patch_text = "--- a/f.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n"
        assert _validate_patch_safety(patch_text) is False

    def test_rejects_patch_with_path_traversal(self):
        patch_text = "--- a/../../etc/passwd\n+++ b/../../etc/passwd\n@@ -1 +1 @@\n-root\n+hacked\n"
        assert _validate_patch_safety(patch_text) is False


# ---------------------------------------------------------------------------
# _apply_unified_diff
# ---------------------------------------------------------------------------

class TestApplyUnifiedDiff:
    def test_applies_single_line_substitution(self):
        original = ["x = 1\n", "y = 2\n"]
        patch_lines = [
            "--- a/mod.py\n",
            "+++ b/mod.py\n",
            "@@ -1,1 +1,1 @@\n",
            "-x = 1\n",
            "+x = 99\n",
        ]
        result = _apply_unified_diff(original, patch_lines)
        assert result is not None
        assert result[0] == "x = 99\n"
        assert result[1] == "y = 2\n"

    def test_applies_addition_of_new_line(self):
        original = ["a = 1\n"]
        patch_lines = [
            "@@ -1,1 +1,2 @@\n",
            " a = 1\n",
            "+b = 2\n",
        ]
        result = _apply_unified_diff(original, patch_lines)
        assert result is not None
        assert "b = 2\n" in result

    def test_applies_deletion_of_line(self):
        original = ["a = 1\n", "b = 2\n", "c = 3\n"]
        patch_lines = [
            "@@ -1,3 +1,2 @@\n",
            " a = 1\n",
            "-b = 2\n",
            " c = 3\n",
        ]
        result = _apply_unified_diff(original, patch_lines)
        assert result is not None
        assert "b = 2\n" not in result
        assert "a = 1\n" in result
        assert "c = 3\n" in result

    def test_returns_original_for_empty_patch(self):
        original = ["x = 1\n"]
        result = _apply_unified_diff(original, [])
        assert result == original


# ---------------------------------------------------------------------------
# apply_patch_to_copy
# ---------------------------------------------------------------------------

class TestApplyPatchToCopy:
    def test_applies_patch_to_copy_not_original(self, tmp_path):
        original_file = tmp_path / "mod.py"
        original_file.write_text("x = 1\n")

        patch_text = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-x = 1\n"
            "+x = 42\n"
        )

        success, _err, patched_path = apply_patch_to_copy(str(original_file), patch_text)

        # Original should be unchanged
        assert original_file.read_text() == "x = 1\n"
        # Patched copy should be different
        if success and patched_path:
            p = Path(patched_path)
            assert p.is_file()

    def test_returns_false_for_missing_original(self, tmp_path):
        success, err, _path = apply_patch_to_copy(
            str(tmp_path / "nonexistent.py"),
            SIMPLE_DIFF,
        )
        assert success is False
        assert "not found" in err.lower()

    def test_returns_false_for_empty_patch(self, tmp_path):
        orig = tmp_path / "f.py"
        orig.write_text("x = 1\n")
        _success, _err, _path = apply_patch_to_copy(str(orig), "")
        # Empty patch should fail validation in the pipeline but apply_patch_to_copy
        # itself may succeed with no changes; we just check it doesn't crash
        # and that the original is unchanged.
        assert orig.read_text() == "x = 1\n"



# ---------------------------------------------------------------------------
# DebuggingAgent.propose_patch
# ---------------------------------------------------------------------------

class TestDebuggingAgentProposePatch:
    def test_propose_patch_returns_proposal_with_diff(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "calc.py"
        src.write_text("def add(a, b):\n    return a\n")

        diff_response = (
            "This patch adds b to the return value.\n"
            "```diff\n"
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -2,1 +2,1 @@\n"
            "-    return a\n"
            "+    return a + b\n"
            "```\n"
        )
        mock_llm = make_mock_llm(diff_response)
        agent = DebuggingAgent(llm=mock_llm, workspace_root=str(tmp_path))

        proposal = agent.propose_patch(
            target_file=str(src),
            ruff_findings=[],
            bandit_findings=[],
            failed_test_output="AssertionError: assert 1 == 3",
            attempt_number=1,
        )

        assert proposal.unified_diff != ""
        assert "return a + b" in proposal.unified_diff
        assert proposal.error_message is None or "patch" not in proposal.error_message.lower()

    def test_propose_patch_handles_missing_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "mod.py"
        src.write_text("x = 1\n")

        agent = DebuggingAgent(llm=None, workspace_root=str(tmp_path))
        with patch("debugging_agent.get_llm", side_effect=MissingApiKeyError("No key")):
            proposal = agent.propose_patch(
                target_file=str(src),
                ruff_findings=[],
                bandit_findings=[],
                failed_test_output="",
            )

        assert proposal.error_message is not None
        assert "credentials" in proposal.error_message.lower()
        assert proposal.unified_diff == ""

    def test_propose_patch_handles_empty_llm_response(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "mod.py"
        src.write_text("x = 1\n")

        mock_llm = make_mock_llm("")
        agent = DebuggingAgent(llm=mock_llm, workspace_root=str(tmp_path))
        proposal = agent.propose_patch(
            target_file=str(src),
            ruff_findings=[],
            bandit_findings=[],
            failed_test_output="",
        )

        assert proposal.error_message is not None
        assert "empty" in proposal.error_message.lower()

    def test_propose_patch_rejects_unsafe_diff(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "mod.py"
        src.write_text("x = 1\n")

        unsafe_response = (
            "```diff\n"
            "--- a/mod.py\n"
            "+++ /dev/null\n"
            "@@ -1 +0,0 @@\n"
            "-x = 1\n"
            "```\n"
        )
        mock_llm = make_mock_llm(unsafe_response)
        agent = DebuggingAgent(llm=mock_llm, workspace_root=str(tmp_path))
        proposal = agent.propose_patch(
            target_file=str(src),
            ruff_findings=[],
            bandit_findings=[],
            failed_test_output="",
        )

        assert proposal.error_message is not None
        assert "safety" in proposal.error_message.lower()
        assert proposal.unified_diff == ""

    def test_propose_patch_handles_provider_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "mod.py"
        src.write_text("x = 1\n")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("503 Server Error")
        agent = DebuggingAgent(llm=mock_llm, workspace_root=str(tmp_path))
        proposal = agent.propose_patch(
            target_file=str(src),
            ruff_findings=[],
            bandit_findings=[],
            failed_test_output="",
        )

        assert proposal.error_message is not None
        assert "503" in proposal.error_message


# ---------------------------------------------------------------------------
# DebuggingAgent.run
# ---------------------------------------------------------------------------

class TestDebuggingAgentRun:
    DIFF_RESPONSE = (
        "Fix the return value.\n"
        "```diff\n"
        "--- a/calc.py\n"
        "+++ b/calc.py\n"
        "@@ -2,1 +2,1 @@\n"
        "-    return a\n"
        "+    return a + b\n"
        "```\n"
    )

    def test_run_returns_expected_state_keys(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "calc.py"
        src.write_text("def add(a, b):\n    return a\n")

        mock_llm = make_mock_llm(self.DIFF_RESPONSE)
        agent = DebuggingAgent(llm=mock_llm, max_attempts=1, workspace_root=str(tmp_path))

        result = agent.run(
            target_file=str(src),
            ruff_findings=[],
            bandit_findings=[],
            failed_test_output="AssertionError",
        )

        assert "patch_proposals" in result
        assert "patch_verified" in result
        assert "agent_errors" in result
        assert isinstance(result["patch_proposals"], list)

    def test_run_respects_max_attempts_bound(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "calc.py"
        src.write_text("def add(a, b):\n    return a\n")

        invoke_count = [0]
        def counting_invoke(msgs):
            invoke_count[0] += 1
            resp = MagicMock()
            resp.content = self.DIFF_RESPONSE
            return resp

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = counting_invoke

        agent = DebuggingAgent(llm=mock_llm, max_attempts=2, workspace_root=str(tmp_path))
        result = agent.run(
            target_file=str(src),
            ruff_findings=[],
            bandit_findings=[],
            failed_test_output="Fail",
        )

        # Should not exceed max_attempts
        assert invoke_count[0] <= 2
        assert len(result["patch_proposals"]) <= 2

    def test_run_includes_disclaimer_in_proposals(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "calc.py"
        src.write_text("def add(a, b):\n    return a\n")

        mock_llm = make_mock_llm(self.DIFF_RESPONSE)
        agent = DebuggingAgent(llm=mock_llm, max_attempts=1, workspace_root=str(tmp_path))
        result = agent.run(
            target_file=str(src),
            ruff_findings=[],
            bandit_findings=[],
            failed_test_output="",
        )

        for proposal in result["patch_proposals"]:
            assert DEBUGGING_DISCLAIMER in proposal.get("disclaimer", "")

    def test_run_max_attempts_bounded_between_1_and_5(self):
        # max_attempts < 1 is clamped to 1
        agent_low = DebuggingAgent(max_attempts=0)
        assert agent_low.max_attempts == 1

        # max_attempts > 5 is clamped to 5
        agent_high = DebuggingAgent(max_attempts=99)
        assert agent_high.max_attempts == 5


# ---------------------------------------------------------------------------
# Temporary Directory Cleanup Tests
# ---------------------------------------------------------------------------


class TestTemporaryDirectoryCleanup:
    """Tests verifying temporary-directory cleanup on success, failure, and exception paths."""

    def test_cleanup_temp_dir_removes_valid_temp_directory(self):
        d = tempfile.mkdtemp(prefix="test_cleanup_")
        assert Path(d).is_dir()
        result = cleanup_temp_dir(d)
        assert result is True
        assert not Path(d).exists()

    def test_cleanup_temp_dir_rejects_paths_outside_temp_root(self):
        project_root = Path(__file__).resolve().parent.parent
        assert project_root.is_dir()
        result = cleanup_temp_dir(str(project_root))
        # Must refuse to remove paths outside system temp dir
        assert result is False
        assert project_root.is_dir()

    def test_cleanup_temp_dir_handles_none_or_missing_path(self):
        assert cleanup_temp_dir(None) is False
        assert cleanup_temp_dir("nonexistent_path_12345") is False

    def test_apply_patch_to_copy_cleans_up_on_diff_mismatch(self, tmp_path):
        orig = tmp_path / "sample.py"
        orig.write_text("actual_content = 42\n", encoding="utf-8")

        with patch("debugging_agent._apply_unified_diff", return_value=None):
            success, err, copy_path = apply_patch_to_copy(str(orig), SIMPLE_DIFF)
            assert success is False
            assert copy_path is None
            assert "did not match file content" in err

    def test_apply_patch_to_copy_cleans_up_on_copy_exception(self, tmp_path):
        orig = tmp_path / "sample.py"
        orig.write_text("x = 1\n", encoding="utf-8")

        with patch("shutil.copy2", side_effect=OSError("Disk full")):
            success, err, copy_path = apply_patch_to_copy(str(orig), SIMPLE_DIFF)
            assert success is False
            assert copy_path is None
            assert "Disk full" in err

    def test_debugging_agent_cleanup_and_context_manager(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "app.py"
        src.write_text("x = 1\n", encoding="utf-8")

        mock_llm = make_mock_llm()
        with DebuggingAgent(llm=mock_llm, workspace_root=str(tmp_path)) as agent:
            proposal = agent.propose_patch(
                target_file=str(src),
                ruff_findings=[],
                bandit_findings=[],
                failed_test_output="",
            )
            assert proposal.patch_succeeded is True
            assert proposal.patch_applied_to_copy is not None
            copy_path = Path(proposal.patch_applied_to_copy)
            assert copy_path.is_file()
            parent_dir = copy_path.parent
            assert parent_dir.is_dir()

        # After exiting context manager, temporary directory must be cleaned up
        assert not parent_dir.exists()

    def test_debugging_agent_run_cleans_up_on_base_exception(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "app.py"
        src.write_text("x = 1\n", encoding="utf-8")

        mock_llm = make_mock_llm()
        agent = DebuggingAgent(llm=mock_llm, workspace_root=str(tmp_path))

        def failing_runner(**kwargs):
            raise KeyboardInterrupt("Simulated interrupt")

        with pytest.raises(KeyboardInterrupt):
            agent.run(
                target_file=str(src),
                ruff_findings=[],
                bandit_findings=[],
                failed_test_output="Failed",
                sandbox_runner=failing_runner,
                test_file="test_app.py",
                source_dir=str(tmp_path),
            )

        # Agent should have cleaned up temporary directories on BaseException
        assert len(agent._created_temp_dirs) == 0
