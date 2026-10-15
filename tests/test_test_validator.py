"""
tests/test_test_validator.py - Unit tests for test_validator.py.

All tests are fully deterministic (no LLM, no Docker, no network).
Uses tmp_path for file system isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from test_validator import (
    MAX_TEST_FILE_SIZE_BYTES,
    TestValidationError,
    _extract_python_code_block,
    store_test_in_workspace,
    validate_generated_test_content,
)

# ---------------------------------------------------------------------------
# _extract_python_code_block
# ---------------------------------------------------------------------------

class TestExtractPythonCodeBlock:
    def test_extracts_from_python_fenced_block(self):
        content = "Some text\n```python\ndef test_foo():\n    assert True\n```\nMore text"
        extracted = _extract_python_code_block(content)
        assert "def test_foo():" in extracted
        assert "```" not in extracted

    def test_extracts_from_generic_fenced_block(self):
        content = "```\ndef test_bar():\n    pass\n```"
        extracted = _extract_python_code_block(content)
        assert "def test_bar():" in extracted

    def test_returns_content_as_is_when_no_fences(self):
        raw = "def test_baz():\n    assert 1 == 1\n"
        extracted = _extract_python_code_block(raw)
        assert "def test_baz():" in extracted

    def test_joins_multiple_code_blocks(self):
        content = "```python\ndef test_a(): pass\n```\n\n```python\ndef test_b(): pass\n```"
        extracted = _extract_python_code_block(content)
        assert "test_a" in extracted
        assert "test_b" in extracted


# ---------------------------------------------------------------------------
# validate_generated_test_content
# ---------------------------------------------------------------------------

class TestValidateGeneratedTestContent:
    def test_valid_test_function(self):
        valid = "def test_addition():\n    assert 1 + 1 == 2\n"
        ok, reason = validate_generated_test_content(valid)
        assert ok is True
        assert reason == ""

    def test_valid_test_class(self):
        valid = "class TestCalc:\n    def test_add(self):\n        assert True\n"
        ok, _ = validate_generated_test_content(valid)
        assert ok is True

    def test_rejects_empty_content(self):
        ok, reason = validate_generated_test_content("")
        assert ok is False
        assert "too short" in reason.lower()

    def test_rejects_content_without_test_function(self):
        content = "def helper():\n    return 42\n\nx = helper()\n"
        ok, reason = validate_generated_test_content(content)
        assert ok is False
        assert "def test_" in reason

    def test_rejects_subprocess_import(self):
        content = "import subprocess\ndef test_x():\n    subprocess.run(['ls'])\n"
        ok, reason = validate_generated_test_content(content)
        assert ok is False
        assert "unsafe pattern" in reason.lower()

    def test_rejects_os_system_call(self):
        content = "import os\ndef test_y():\n    os.system('rm -rf /')\n"
        ok, _ = validate_generated_test_content(content)
        assert ok is False

    def test_rejects_eval_call(self):
        content = "def test_z():\n    eval('print(1)')\n"
        ok, _ = validate_generated_test_content(content)
        assert ok is False

    def test_rejects_exec_call(self):
        content = "def test_w():\n    exec('import os')\n"
        ok, _ = validate_generated_test_content(content)
        assert ok is False

    def test_rejects_write_mode_open(self):
        content = "def test_write():\n    open('out.txt', 'w').write('x')\n"
        ok, _ = validate_generated_test_content(content)
        assert ok is False

    def test_rejects_oversized_content(self):
        huge = "def test_x(): pass\n" + "x" * (MAX_TEST_FILE_SIZE_BYTES + 1)
        ok, reason = validate_generated_test_content(huge)
        assert ok is False
        assert "size" in reason.lower()

    def test_accepts_valid_complex_test(self):
        content = (
            "import pytest\n"
            "from mymodule import add\n\n"
            "def test_add_positives():\n"
            "    assert add(2, 3) == 5\n\n"
            "def test_add_negative():\n"
            "    assert add(-1, 1) == 0\n\n"
            "@pytest.mark.parametrize('a,b,expected', [(0,0,0),(1,1,2)])\n"
            "def test_add_parametrized(a, b, expected):\n"
            "    assert add(a, b) == expected\n"
        )
        ok, _ = validate_generated_test_content(content)
        assert ok is True



# ---------------------------------------------------------------------------
# store_test_in_workspace
# ---------------------------------------------------------------------------

class TestStoreTestInWorkspace:
    VALID_TEST = (
        "def test_simple():\n"
        "    assert 1 + 1 == 2\n"
    )

    def test_stores_file_in_isolated_workspace(self, tmp_path):
        workspace = str(tmp_path / "test_ws")
        stored = store_test_in_workspace(
            content=self.VALID_TEST,
            target_source_file="mymodule.py",
            workspace_dir=workspace,
        )

        from pathlib import Path
        p = Path(stored)
        assert p.is_file()
        # Must be inside the workspace
        assert p.is_relative_to(tmp_path / "test_ws")

    def test_filename_contains_safe_identifier(self, tmp_path):
        workspace = str(tmp_path / "test_ws")
        stored = store_test_in_workspace(
            content=self.VALID_TEST,
            target_source_file="calculator.py",
            workspace_dir=workspace,
        )
        assert "test_" in stored
        assert "calculator" in stored

    def test_stored_file_contains_header_comment(self, tmp_path):
        workspace = str(tmp_path / "ws")
        stored = store_test_in_workspace(
            content=self.VALID_TEST,
            target_source_file="mod.py",
            workspace_dir=workspace,
        )
        content = Path(stored).read_text(encoding="utf-8")
        assert "AUTO-GENERATED TEST FILE" in content
        assert "SANDBOX REVIEW ONLY" in content

    def test_stored_file_contains_extracted_test_code(self, tmp_path):
        workspace = str(tmp_path / "ws")
        stored = store_test_in_workspace(
            content=self.VALID_TEST,
            target_source_file="mod.py",
            workspace_dir=workspace,
        )
        content = Path(stored).read_text(encoding="utf-8")
        assert "def test_simple():" in content

    def test_raises_validation_error_for_invalid_content(self, tmp_path):
        workspace = str(tmp_path / "ws")
        with pytest.raises(TestValidationError):
            store_test_in_workspace(
                content="this is not a test file at all",
                target_source_file="mod.py",
                workspace_dir=workspace,
            )

    def test_extracts_code_from_markdown_fences(self, tmp_path):
        workspace = str(tmp_path / "ws")
        md_content = (
            "Here are some tests:\n"
            "```python\n"
            "def test_extracted():\n"
            "    assert True\n"
            "```\n"
            )
        stored = store_test_in_workspace(
            content=md_content,
            target_source_file="app.py",
            workspace_dir=workspace,
        )
        content = Path(stored).read_text(encoding="utf-8")
        assert "def test_extracted():" in content

    def test_creates_workspace_directory_if_missing(self, tmp_path):
        workspace = str(tmp_path / "brand" / "new" / "dir")
        stored = store_test_in_workspace(
            content=self.VALID_TEST,
            target_source_file="mod.py",
            workspace_dir=workspace,
        )
        assert Path(stored).is_file()


    def test_raises_validation_error_for_unsafe_content(self, tmp_path):
        workspace = str(tmp_path / "ws")
        unsafe = "def test_x():\n    eval('rm -rf /')\n"
        with pytest.raises(TestValidationError):
            store_test_in_workspace(
                content=unsafe,
                target_source_file="mod.py",
                workspace_dir=workspace,
            )
