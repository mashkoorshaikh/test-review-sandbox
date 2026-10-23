"""
semantic_analyzer.py - Deterministic AST Semantic Bug Analysis Module.

Provides pure AST-based inspection of Python source code to detect runtime hazards
such as division-by-zero (SEM001).

Guarantees:
  - Deterministic: 100% reproducible results with zero random or heuristic variation.
  - Read-Only: Never modifies inspected source code or calls any write/network endpoints.
  - Offline: Operates entirely locally using Python standard library 'ast'. No LLM required.
  - Fail-Safe: Gracefully handles syntax errors, unsupported files, and empty directories.

Supported Rules:
  - SEM001: Potential or definite unhandled ZeroDivisionError in arithmetic division ('/'),
            floor division ('//'), or modulo ('%').

Confidence Classification:
  - HIGH: Direct literal-zero divisors (e.g. 'x / 0', 'x // 0', 'x % 0').
  - MEDIUM: Variable or dynamic expression divisors where the divisor cannot be
            proven non-zero and is not enclosed in a ZeroDivisionError handler.

Unsupported Control-Flow Cases (Analyzer Limitations):
  - Interprocedural invariants (caller guaranteeing parameter values).
  - External validation helpers (e.g. 'if is_valid(b): ...').
  - Dynamic loop modifications (e.g. 'while b == 0: ...').
  - Complex mathematical invariants (e.g. 'n**2 + 1').
  - Aliasing and object attributes modified outside local scope.
"""

from __future__ import annotations

import ast
import os
import time
from pathlib import Path

from tools import Issue, ToolResult


def _is_zero_literal(node: ast.AST) -> bool:
    """Check if an AST node represents a literal zero (int, float, or unary zero)."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value is False  # 1 / False actually raises ZeroDivisionError in Python!
        return node.value in (0, 0.0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_zero_literal(node.operand)
    return False


def _is_nonzero_constant(node: ast.AST) -> bool:
    """Check if an AST node represents a known non-zero literal constant."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value is True  # True is non-zero
        if isinstance(node.value, (int, float, complex)):
            return node.value != 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_nonzero_constant(node.operand)
    return False


def _get_name_id(node: ast.AST) -> str | None:
    """Extract variable identifier name if node is an ast.Name."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def _format_divisor_expr(node: ast.AST) -> str:
    """Produce a concise string representation of the divisor AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return "<expr>"


def _catches_zero_division(handler: ast.ExceptHandler) -> bool:
    """Check if an except handler catches ZeroDivisionError or its base classes."""
    if handler.type is None:
        # Bare 'except:' catches everything
        return True

    def _type_matches(expr: ast.AST) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in ("ZeroDivisionError", "ArithmeticError", "Exception", "BaseException")
        if isinstance(expr, ast.Attribute):
            return expr.attr in ("ZeroDivisionError", "ArithmeticError", "Exception", "BaseException")
        return False

    if isinstance(handler.type, ast.Tuple):
        return any(_type_matches(elt) for elt in handler.type.elts)

    return _type_matches(handler.type)


def _statement_always_exits(stmt: ast.stmt) -> bool:
    """Check if a statement unconditionally exits the current block."""
    if isinstance(stmt, (ast.Raise, ast.Return)):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        func = stmt.value.func
        if isinstance(func, ast.Name) and func.id in ("exit", "quit"):
            return True
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "exit"
            and isinstance(func.value, ast.Name)
            and func.value.id == "sys"
        ):
            return True
    return bool(
        isinstance(stmt, ast.If)
        and stmt.orelse
        and all(_statement_always_exits(s) for s in stmt.body)
        and all(_statement_always_exits(s) for s in stmt.orelse)
    )


def _is_test_checking_zero(test: ast.AST, var_name: str) -> bool:
    """
    Check if a test expression tests if var_name is zero:
      - 'var == 0' or '0 == var'
      - 'not var'
    """
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
        if (
            isinstance(test.left, ast.Name)
            and test.left.id == var_name
            and _is_zero_literal(test.comparators[0])
        ):
            return True
        if (
            _is_zero_literal(test.left)
            and isinstance(test.comparators[0], ast.Name)
            and test.comparators[0].id == var_name
        ):
            return True
    elif (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and isinstance(test.operand, ast.Name)
        and test.operand.id == var_name
    ):
        return True
    return False


def _is_test_checking_nonzero(test: ast.AST, var_name: str) -> bool:
    """
    Check if a test expression tests if var_name is non-zero:
      - 'var != 0' or '0 != var'
      - 'var > 0' or 'var < 0'
      - 'var' (truthy)
    """
    if isinstance(test, ast.Compare):
        if len(test.ops) == 1 and isinstance(test.ops[0], ast.NotEq):
            if (
                isinstance(test.left, ast.Name)
                and test.left.id == var_name
                and _is_zero_literal(test.comparators[0])
            ):
                return True
            if (
                _is_zero_literal(test.left)
                and isinstance(test.comparators[0], ast.Name)
                and test.comparators[0].id == var_name
            ):
                return True
        elif (
            len(test.ops) == 1
            and isinstance(test.ops[0], (ast.Gt, ast.Lt))
            and isinstance(test.left, ast.Name)
            and test.left.id == var_name
            and _is_zero_literal(test.comparators[0])
        ):
            return True
    elif isinstance(test, ast.Name) and test.id == var_name:
        return True
    return False


class DivisionByZeroDetector(ast.NodeVisitor):
    """
    AST visitor that scans for division, floor division, and modulo operations
    and determines whether they present a potential or definite division-by-zero risk.
    """

    def __init__(self, file_path: str = "<string>") -> None:
        self.file_path = file_path
        self.issues: list[Issue] = []
        # Stack of try-blocks currently protecting execution
        self._try_handlers_stack: list[list[ast.ExceptHandler]] = []
        # Stack of sets of variables currently known to be non-zero in the active branch
        self._guarded_vars_stack: list[set[str]] = [set()]

    @property
    def _current_guarded_vars(self) -> set[str]:
        return self._guarded_vars_stack[-1]

    def _is_enclosed_in_zero_division_handler(self) -> bool:
        """Check if current execution is protected by a try block catching ZeroDivisionError."""
        for handlers in self._try_handlers_stack:
            if any(_catches_zero_division(h) for h in handlers):
                return True
        return False

    def visit_Try(self, node: ast.Try) -> None:
        # 1. Visit try body with its handlers on the try stack
        self._try_handlers_stack.append(node.handlers)
        for stmt in node.body:
            self.visit(stmt)
        self._try_handlers_stack.pop()

        # 2. Visit handlers (exceptions inside handlers are not protected by themselves)
        for handler in node.handlers:
            self.visit(handler)

        # 3. Visit orelse (runs if no exception was raised in try)
        for stmt in node.orelse:
            self.visit(stmt)

        # 4. Visit finalbody
        for stmt in node.finalbody:
            self.visit(stmt)

    def _visit_statement_sequence(self, stmts: list[ast.stmt]) -> None:
        """Visit a sequence of statements while tracking variable guards and exits."""
        # Inherit current guarded vars
        current_guards = set(self._current_guarded_vars)

        for stmt in stmts:
            self._guarded_vars_stack.append(set(current_guards))
            self.visit(stmt)
            self._guarded_vars_stack.pop()

            # Check if this statement is an 'if <var> == 0: exit' guard
            if isinstance(stmt, ast.If):
                # Check for zero test on variables
                for var in self._extract_names_from_test(stmt.test):
                    if _is_test_checking_zero(stmt.test, var):
                        # Both all statements in body must exit, and no orelse (or orelse also exits)
                        body_exits = any(_statement_always_exits(s) for s in stmt.body)
                        orelse_exits = (not stmt.orelse) or any(_statement_always_exits(s) for s in stmt.orelse)
                        if body_exits and orelse_exits:
                            current_guards.add(var)

            # Check if variable reassignment invalidates guard
            assigned_names = self._extract_assigned_names(stmt)
            for name in assigned_names:
                current_guards.discard(name)

    def _extract_names_from_test(self, test: ast.AST) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(test):
            if isinstance(node, ast.Name):
                names.add(node.id)
        return names

    def _extract_assigned_names(self, stmt: ast.stmt) -> set[str]:
        names: set[str] = set()
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
        return names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._guarded_vars_stack.append(set())
        self._visit_statement_sequence(node.body)
        self._guarded_vars_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._guarded_vars_stack.append(set())
        self._visit_statement_sequence(node.body)
        self._guarded_vars_stack.pop()

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_statement_sequence(node.body)

    def visit_If(self, node: ast.If) -> None:
        # Visit condition
        self.visit(node.test)

        # 1. Body branch: check if test guarantees non-zero
        body_guards = set(self._current_guarded_vars)
        for var in self._extract_names_from_test(node.test):
            if _is_test_checking_nonzero(node.test, var):
                body_guards.add(var)

        self._guarded_vars_stack.append(body_guards)
        self._visit_statement_sequence(node.body)
        self._guarded_vars_stack.pop()

        # 2. Else branch: check if test was checking zero ('if b == 0: ... else: return a / b')
        else_guards = set(self._current_guarded_vars)
        for var in self._extract_names_from_test(node.test):
            if _is_test_checking_zero(node.test, var):
                else_guards.add(var)

        self._guarded_vars_stack.append(else_guards)
        self._visit_statement_sequence(node.orelse)
        self._guarded_vars_stack.pop()

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # Recurse into children first
        self.generic_visit(node)

        # Check if operation is division, floor division, or modulo
        if not isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            return

        op_name = {
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
        }.get(type(node.op), "/")

        right = node.right

        # Case 1: Direct literal zero -> HIGH confidence finding
        if _is_zero_literal(right):
            self.issues.append(
                Issue(
                    code="SEM001",
                    message=f"Division by zero literal ({op_name} 0): ZeroDivisionError will occur at runtime.",
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    severity="WARNING",
                    tool="semantic_analyzer",
                    fixable=False,
                    confidence="HIGH",
                    explanation=(
                        "Direct division, floor division, or modulo by constant 0 unconditionally "
                        "raises an unhandled ZeroDivisionError exception when reached."
                    ),
                    suggested_fix="Replace the zero divisor with a non-zero value, or remove the invalid arithmetic expression.",
                )
            )
            return

        # Case 2: Known non-zero constant (e.g. 2, 100.0) -> Completely safe
        if _is_nonzero_constant(right):
            return

        # Case 3: Variable or expression divisor
        # Check if enclosed in a try...except ZeroDivisionError handler
        if self._is_enclosed_in_zero_division_handler():
            return

        var_name = _get_name_id(right)
        if var_name and var_name in self._current_guarded_vars:
            # Reliably guarded against zero
            return

        # Not proven non-zero and not handled by try/except -> report as possible risk
        divisor_str = _format_divisor_expr(right)
        self.issues.append(
            Issue(
                code="SEM001",
                message=(
                    f"Possible division by zero: divisor '{divisor_str}' may evaluate to zero "
                    f"without explicit ZeroDivisionError handling."
                ),
                file_path=self.file_path,
                line=node.lineno,
                column=node.col_offset,
                severity="WARNING",
                tool="semantic_analyzer",
                fixable=False,
                confidence="MEDIUM",
                explanation=(
                    f"Static AST analysis detected an arithmetic division ('{op_name}') where divisor "
                    f"'{divisor_str}' is not proven non-zero and has no enclosing ZeroDivisionError handler. "
                    f"Note: Static analysis cannot determine runtime values; divisor '{divisor_str}' "
                    f"might never be zero in practice."
                ),
                suggested_fix=(
                    f"Ensure divisor '{divisor_str}' is non-zero, validate with "
                    f"'if {divisor_str} == 0: raise ValueError(...)', or wrap in a "
                    f"'try ... except ZeroDivisionError:' block."
                ),
            )
        )


def analyze_python_source(source_code: str, file_path: str = "<string>") -> list[Issue]:
    """
    Analyze Python source code string for semantic bug hazards.

    Args:
        source_code: Python code content.
        file_path: Relative or absolute path for location reporting.

    Returns:
        List of parsed Issue objects.
    """
    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError:
        # Gracefully handle syntax errors without crashing
        return []

    detector = DivisionByZeroDetector(file_path=file_path)
    detector.visit(tree)
    return detector.issues


def run_semantic_analysis(target_path: str) -> ToolResult:
    """
    Execute semantic AST bug analysis on a target file or directory.

    Guarantees:
      - Strictly read-only: no files are modified.
      - Offline: zero network calls, zero live LLM dependencies.
      - Safe: does not execute untrusted target code.

    Args:
        target_path: File or directory path to inspect.

    Returns:
        ToolResult summarizing execution outcome and detected issues.
    """
    start_time = time.monotonic()
    p = Path(target_path)

    if not p.exists():
        return ToolResult(
            tool_name="SemanticAnalyzer",
            target_path=target_path,
            success=False,
            issues_found=False,
            exit_code=2,
            issues=[],
            error_message=f"Target path '{target_path}' does not exist.",
            execution_time=time.monotonic() - start_time,
        )

    all_issues: list[Issue] = []
    py_files: list[Path] = []

    if p.is_file():
        if p.suffix.lower() == ".py":
            py_files.append(p)
    elif p.is_dir():
        # Walk directory, excluding common non-source directories
        ignored_dirs = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "build", "dist"}
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]
            for f in files:
                if f.endswith(".py"):
                    py_files.append(Path(root) / f)

    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
            file_issues = analyze_python_source(content, str(py_file))
            all_issues.extend(file_issues)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue

    elapsed = time.monotonic() - start_time
    return ToolResult(
        tool_name="SemanticAnalyzer",
        target_path=target_path,
        success=True,
        issues_found=len(all_issues) > 0,
        exit_code=1 if all_issues else 0,
        issues=all_issues,
        raw_stdout=f"Analyzed {len(py_files)} Python file(s). Found {len(all_issues)} issue(s).",
        raw_stderr="",
        error_message=None,
        execution_time=elapsed,
    )
