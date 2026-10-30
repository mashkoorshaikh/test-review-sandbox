"""
test_validator.py - Test Content Validation and Safe Workspace Storage.

Validates LLM-generated pytest test code and stores it in a dedicated
temporary workspace that is completely separate from the original source tree.

Safety guarantees:
  - Generated tests are NEVER written into the original source project.
  - Test content is validated for basic structural integrity before storage.
  - All paths are checked for traversal attacks before use.
  - The temp workspace has a predictable, isolated directory structure.
  - Returned paths are fully resolved and verified before use.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum allowable generated test file size (bytes) to prevent bloat
MAX_TEST_FILE_SIZE_BYTES = 128 * 1024  # 128 KB

# Minimum number of characters a test suggestion must contain to be considered valid
MIN_TEST_CONTENT_LENGTH = 20

# Patterns that indicate a valid pytest test file structure
_VALID_TEST_PATTERNS = [
    re.compile(r"\bdef\s+test_\w+\s*\(", re.MULTILINE),  # def test_xxx(
    re.compile(r"\bclass\s+Test\w+\s*[:(]", re.MULTILINE),  # class TestXxx
]

# Prohibited modules and calls for static defense-in-depth screening.
# NOTE: Static AST/regex screening is a secondary defense-in-depth measure;
# Docker container isolation (sandbox_runner.py) remains the primary sandbox boundary.
PROHIBITED_MODULES: frozenset[str] = frozenset({
    "subprocess",
    "pty",
    "posix",
    "ctypes",
    "socket",
    "http.client",
    "urllib.request",
    "importlib",
})

PROHIBITED_OS_FUNCS: frozenset[str] = frozenset({
    "system",
    "popen",
    "spawn",
    "remove",
    "unlink",
    "rmdir",
    "kill",
})

PROHIBITED_BUILTINS: frozenset[str] = frozenset({
    "eval",
    "exec",
    "__import__",
    "compile",
    "breakpoint",
})

# Patterns that indicate suspicious prompt injection attempts in generated code
_SUSPICIOUS_INJECTION_PATTERNS = [
    re.compile(r"\bimport\s+(subprocess|pty|posix|ctypes|socket)", re.IGNORECASE),
    re.compile(r"\bfrom\s+(subprocess|pty|posix|ctypes|socket)\b", re.IGNORECASE),
    re.compile(r"\bos\.(system|popen|spawn|remove|unlink|rmdir|kill)\s*\(", re.IGNORECASE),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
    re.compile(r"\beval\s*\(", re.IGNORECASE),
    re.compile(r"\bcompile\s*\(", re.IGNORECASE),
    re.compile(r"\bbreakpoint\s*\(", re.IGNORECASE),
    re.compile(r"\b__import__\s*\(", re.IGNORECASE),
    re.compile(r"\bimportlib\.(import_module|__import__)\s*\(", re.IGNORECASE),
    re.compile(r"\bopen\s*\(.*['\"]w['\"]", re.IGNORECASE),
    re.compile(r"\bshutil\.(rmtree|move|copy)", re.IGNORECASE),
    re.compile(r"\b__subclasses__\s*\(", re.IGNORECASE),
]


class TestValidationError(ValueError):
    """Raised when generated test content fails safety or structural validation."""
    __test__ = False


def _extract_python_code_block(content: str) -> str:
    """
    Extract Python code from a markdown code block if present.

    LLMs often wrap code in ```python ... ``` fences. This extracts
    just the code portion for storage.
    """
    # Match ```python ... ``` or ``` ... ``` fences
    pattern = re.compile(
        r"```(?:python)?\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(content)
    if matches:
        # Join all found code blocks (in case there are multiple)
        return "\n\n".join(block.strip() for block in matches)

    # No fences found - strip any trailing bracketed advisory disclaimers
    cleaned = re.sub(r"\[(?:GENERATED TESTS|ADVISORY|SECURITY NOTICE).*?\]", "", content, flags=re.DOTALL)
    return cleaned.strip()


def validate_generated_test_content(content: str) -> tuple[bool, str]:
    """
    Validate that LLM-generated content looks like a safe pytest test file.

    Checks performed:
      1. Content is non-empty and meets minimum length.
      2. Contains at least one `def test_` function or `class Test*` class.
      3. Does not contain suspicious code patterns (subprocess, exec, eval, etc.).
      4. Does not exceed size limit.

    Returns:
        Tuple of (is_valid: bool, reason: str).
        reason is an empty string on success or a human-readable rejection reason.
    """
    if not content or len(content.strip()) < MIN_TEST_CONTENT_LENGTH:
        return False, f"Content is too short (< {MIN_TEST_CONTENT_LENGTH} chars) to be valid test code."

    if len(content.encode("utf-8")) > MAX_TEST_FILE_SIZE_BYTES:
        return False, f"Generated test content exceeds maximum allowed size ({MAX_TEST_FILE_SIZE_BYTES} bytes)."

    # Extract from markdown code blocks if present
    extracted = _extract_python_code_block(content)

    # 1. Regex safety filters (fast first line of defense)
    for pattern in _SUSPICIOUS_INJECTION_PATTERNS:
        if pattern.search(extracted):
            matched = pattern.pattern
            return False, (
                f"Generated test content contains a potentially unsafe pattern "
                f"('{matched}'). Content rejected for security reasons."
            )

    # 2. AST parsing to verify valid Python syntax without executing any code
    try:
        tree = ast.parse(extracted)
    except SyntaxError as syn_err:
        return False, f"Generated test content has invalid Python syntax: {syn_err}"

    # 3. Structural AST check: must contain test functions or Test classes
    has_test_function = False
    for node in ast.walk(tree):
        if (
            (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (node.name.startswith("test_") or node.name == "test")
            )
            or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
        ):
            has_test_function = True
            break


    if not has_test_function:
        return False, (
            "Generated content does not contain any `def test_*` functions or "
            "`class Test*` classes. Cannot treat as a pytest test file."
        )

    # 4. AST visitor for dangerous calls/imports (defense-in-depth screening)
    for node in ast.walk(tree):
        # Disallow dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_pkg = alias.name.split(".")[0]
                if alias.name in PROHIBITED_MODULES or base_pkg in PROHIBITED_MODULES:
                    return False, f"Generated test imports prohibited module '{alias.name}'."
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            base_pkg = mod.split(".")[0]
            if mod in PROHIBITED_MODULES or base_pkg in PROHIBITED_MODULES:
                return False, f"Generated test imports from prohibited module '{mod}'."
            if mod == "os":
                for alias in node.names:
                    if alias.name in PROHIBITED_OS_FUNCS:
                        return False, f"Generated test imports prohibited function 'os.{alias.name}'."
        # Disallow dangerous function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in PROHIBITED_BUILTINS:
                return False, f"Generated test calls prohibited built-in '{node.func.id}()'."
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "__subclasses__":
                    return False, "Generated test calls prohibited method '__subclasses__()'."
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr in PROHIBITED_OS_FUNCS
                ):
                    return False, f"Generated test calls prohibited function 'os.{node.func.attr}()'."


    return True, ""


def store_test_in_workspace(
    content: str,
    target_source_file: str,
    workspace_dir: str | None = None,
) -> str:
    """
    Validate and safely store generated test content in an isolated temp workspace.

    The test workspace is COMPLETELY SEPARATE from the original source tree.
    The test file will be named `test_<hash>_<basename>.py` to avoid collisions.

    Args:
        content: Raw LLM-generated test content (may include markdown fences).
        target_source_file: Path of the source file this test targets (used for naming).
        workspace_dir: Optional explicit workspace directory. If None, uses system temp.

    Returns:
        Absolute path to the saved test file.

    Raises:
        TestValidationError: If content fails validation.
        OSError: If the file cannot be written.
    """
    # 1. Extract clean Python code
    extracted = _extract_python_code_block(content)

    # 2. Validate
    is_valid, reason = validate_generated_test_content(extracted)
    if not is_valid:
        raise TestValidationError(
            f"Generated test content rejected: {reason}"
        )

    # 3. Prepare workspace directory (completely outside source tree)
    if workspace_dir is None:
        workspace_dir = tempfile.mkdtemp(prefix="ai_tests_")
    else:
        os.makedirs(workspace_dir, exist_ok=True)

    # 4. Build safe filename: test_<short_hash>_<source_basename>.py
    source_basename = Path(target_source_file).stem[:40]  # cap length
    content_hash = hashlib.sha256(extracted.encode()).hexdigest()[:8]
    safe_basename = re.sub(r"[^\w]", "_", source_basename)
    test_filename = f"test_{content_hash}_{safe_basename}.py"

    # 5. Resolve final path and verify it stays within workspace
    workspace_path = Path(workspace_dir).resolve()
    dest_path = (workspace_path / test_filename).resolve()

    if not dest_path.is_relative_to(workspace_path):
        raise PermissionError(
            f"Security Violation: Computed test path '{dest_path}' "
            f"escapes workspace '{workspace_path}'."
        )

    # 6. Add standard header comment to the generated test file
    header = (
        "# AUTO-GENERATED TEST FILE — FOR SANDBOX REVIEW ONLY\n"
        "# This file was generated by an AI agent. It has NOT been\n"
        "# reviewed by a human engineer. Run only in an isolated sandbox.\n"
        "# DO NOT commit this file or run it directly on production systems.\n\n"
    )

    full_content = header + extracted + "\n"

    # 7. Write to workspace (never inside original source tree)
    dest_path.write_text(full_content, encoding="utf-8")
    logger.info("Stored generated test at: %s", dest_path)

    return str(dest_path)
