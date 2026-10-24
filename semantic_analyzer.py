"""
semantic_analyzer.py - Deterministic AST Semantic Bug Analysis Module.

Provides pure AST-based inspection of Python source code to detect runtime hazards:
  - SEM001: Potential or definite unhandled ZeroDivisionError in arithmetic division ('/'),
            floor division ('//'), or modulo ('%').
  - SEM002: Potential or definite unhandled None dereference (AttributeError / TypeError)
            on literal None, variables known to be None, None-defaulted parameters,
            or unvalidated regex match chaining.
  - SEM003: Provably out-of-bounds constant integer sequence indexing (IndexError)
            on locally known literal lists and tuples, or provably missing constant-key
            access (KeyError) on locally known literal dictionaries.

Guarantees:
  - Deterministic: 100% reproducible results with zero random or heuristic variation.
  - Read-Only: Never modifies inspected source code or calls any write/network endpoints.
  - Offline: Operates entirely locally using Python standard library 'ast'. No LLM required.
  - Fail-Safe: Gracefully handles syntax errors, unsupported files, and empty directories.

Confidence Classification:
  - HIGH:
      * SEM001: Direct literal-zero divisors (e.g. 'x / 0', 'x // 0', 'x % 0').
      * SEM002: Direct dereference of literal None ('None.attr', 'None[k]', 'None()'),
                variables in branches where explicitly known to be None ('if x is None: x.attr'),
                or local variables assigned None without reassignment.
      * SEM003: Direct literal sequence out-of-bounds indexing ('[1, 2][5]', '()[0]'),
                direct literal dictionary missing key ('{"a": 1}["b"]'),
                or local variables assigned literal collections accessed out of bounds /
                with missing key without intervening mutation or reassignment.
  - MEDIUM:
      * SEM001: Variable or dynamic expression divisors without non-zero guards.
      * SEM002: None-defaulted function parameters accessed without an effective guard,
                or immediate '.group()' chaining on re.match() / re.search() results.

Exception-Handling Policy:
  - Findings are suppressed ONLY if the enclosing try-block catches the specific runtime exception:
      * SEM001: ZeroDivisionError, ArithmeticError, Exception, BaseException, or bare except.
      * SEM002: AttributeError (for attribute access), TypeError (for subscript/call),
                Exception, BaseException, or bare except.
      * SEM003: IndexError (for sequence indexing), KeyError (for dictionary key lookup),
                LookupError, Exception, BaseException, or bare except.
  - Unrelated handlers (e.g. KeyError for SEM002 subscript, ValueError for SEM003) do NOT suppress findings.
  - Operations outside the try body are NOT suppressed.

Unsupported Control-Flow Cases (Analyzer Limitations):
  - Interprocedural invariants (caller guaranteeing parameter values or collection bounds).
  - External validation helpers (e.g. 'if is_valid(b): ...').
  - Dynamic loop modifications (e.g. 'while b is None: ...').
  - Aliasing and object attributes modified outside local scope.
  - Dynamic indices, non-constant keys, or collections with dynamic unpacking.
"""

from __future__ import annotations

import ast
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools import Issue, ToolResult


def _is_none_literal(node: ast.AST) -> bool:
    """Check if node is literal None (ast.Constant(value=None))."""
    return isinstance(node, ast.Constant) and node.value is None


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


def _catches_attribute_error(handler: ast.ExceptHandler) -> bool:
    """Check if an except handler catches AttributeError or its base classes."""
    if handler.type is None:
        # Bare 'except:' catches everything
        return True

    def _type_matches(expr: ast.AST) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in ("AttributeError", "Exception", "BaseException")
        if isinstance(expr, ast.Attribute):
            return expr.attr in ("AttributeError", "Exception", "BaseException")
        return False

    if isinstance(handler.type, ast.Tuple):
        return any(_type_matches(elt) for elt in handler.type.elts)

    return _type_matches(handler.type)


def _catches_type_error(handler: ast.ExceptHandler) -> bool:
    """Check if an except handler catches TypeError or its base classes."""
    if handler.type is None:
        # Bare 'except:' catches everything
        return True

    def _type_matches(expr: ast.AST) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in ("TypeError", "Exception", "BaseException")
        if isinstance(expr, ast.Attribute):
            return expr.attr in ("TypeError", "Exception", "BaseException")
        return False

    if isinstance(handler.type, ast.Tuple):
        return any(_type_matches(elt) for elt in handler.type.elts)

    return _type_matches(handler.type)


def _catches_index_error(handler: ast.ExceptHandler) -> bool:
    """Check if an except handler catches IndexError or its base classes."""
    if handler.type is None:
        return True

    def _type_matches(expr: ast.AST) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in ("IndexError", "LookupError", "Exception", "BaseException")
        if isinstance(expr, ast.Attribute):
            return expr.attr in ("IndexError", "LookupError", "Exception", "BaseException")
        return False

    if isinstance(handler.type, ast.Tuple):
        return any(_type_matches(elt) for elt in handler.type.elts)

    return _type_matches(handler.type)


def _catches_key_error(handler: ast.ExceptHandler) -> bool:
    """Check if an except handler catches KeyError or its base classes."""
    if handler.type is None:
        return True

    def _type_matches(expr: ast.AST) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in ("KeyError", "LookupError", "Exception", "BaseException")
        if isinstance(expr, ast.Attribute):
            return expr.attr in ("KeyError", "LookupError", "Exception", "BaseException")
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


def _is_test_checking_is_none(test: ast.AST, var_name: str) -> bool:
    """Check if test checks if var_name is None (is None / == None)."""
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op = test.ops[0]
        if isinstance(op, (ast.Is, ast.Eq)):
            if (
                isinstance(test.left, ast.Name)
                and test.left.id == var_name
                and _is_none_literal(test.comparators[0])
            ):
                return True
            if (
                _is_none_literal(test.left)
                and isinstance(test.comparators[0], ast.Name)
                and test.comparators[0].id == var_name
            ):
                return True
    return False


def _is_test_checking_is_not_none(test: ast.AST, var_name: str) -> bool:
    """Check if test checks if var_name is not None (is not None / != None)."""
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op = test.ops[0]
        if isinstance(op, (ast.IsNot, ast.NotEq)):
            if (
                isinstance(test.left, ast.Name)
                and test.left.id == var_name
                and _is_none_literal(test.comparators[0])
            ):
                return True
            if (
                _is_none_literal(test.left)
                and isinstance(test.comparators[0], ast.Name)
                and test.comparators[0].id == var_name
            ):
                return True
    return False


def _is_test_checking_truthy(test: ast.AST, var_name: str) -> bool:
    """Check if test directly tests truthiness of var_name ('if x:')."""
    return isinstance(test, ast.Name) and test.id == var_name


def _is_test_checking_falsy(test: ast.AST, var_name: str) -> bool:
    """Check if test directly tests falsiness of var_name ('if not x:')."""
    return (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and isinstance(test.operand, ast.Name)
        and test.operand.id == var_name
    )


def _body_always_assigns_non_none(body: list[ast.stmt], var_name: str) -> bool:
    """Check if statement sequence unconditionally assigns var_name a non-None value."""
    assigned_non_none = False
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id == var_name:
                    assigned_non_none = not _is_none_literal(stmt.value)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == var_name:
            if stmt.value and not _is_none_literal(stmt.value):
                assigned_non_none = True
            else:
                assigned_non_none = False
    return assigned_non_none


def _is_regex_match_call(node: ast.AST) -> tuple[bool, str]:
    """Check if node is a call to re.match, re.search, or re.fullmatch."""
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("match", "search", "fullmatch")
            and isinstance(func.value, ast.Name)
            and func.value.id == "re"
        ):
            return True, f"re.{func.attr}()"
    return False, ""


# Dedicated sentinel object representing an unknown/unsupported literal value
_UNKNOWN_KEY = object()


def _extract_constant_value(node: ast.AST) -> Any:
    """Extract constant value from AST node, or _UNKNOWN_KEY if dynamic/unsupported."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _extract_constant_value(node.operand)
        if isinstance(val, (int, float, complex)) and not isinstance(val, bool):
            return -val
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        val = _extract_constant_value(node.operand)
        if isinstance(val, (int, float, complex)) and not isinstance(val, bool):
            return +val
    if isinstance(node, ast.Tuple):
        elts: list[Any] = []
        for elt in node.elts:
            v = _extract_constant_value(elt)
            if v is _UNKNOWN_KEY:
                return _UNKNOWN_KEY
            elts.append(v)
        try:
            return tuple(elts)
        except Exception:  # noqa: BLE001
            return _UNKNOWN_KEY
    return _UNKNOWN_KEY


def _extract_sequence_length(node: ast.List | ast.Tuple) -> int | None:
    """Extract length of literal list or tuple, or None if any element is starred unpacking."""
    for elt in node.elts:
        if isinstance(elt, ast.Starred):
            return None
    return len(node.elts)


def _extract_dict_literal_keys(node: ast.Dict) -> set[Any] | None:
    """Extract set of constant keys from an ast.Dict, or None if any key/unpacking is dynamic."""
    keys: set[Any] = set()
    for k in node.keys:
        if k is None:
            # **other unpacking
            return None
        val = _extract_constant_value(k)
        if val is _UNKNOWN_KEY:
            return None
        try:
            keys.add(val)
        except TypeError:
            # unhashable key
            return None
    return keys


def _extract_membership_guard(test: ast.AST) -> tuple[str, Any, bool] | None:
    """
    Check if test expression is 'key in dict_var' or 'key not in dict_var'.
    Returns (var_name, key_value, is_in_operator) or None.
    """
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op = test.ops[0]
        if isinstance(op, ast.In):
            key_val = _extract_constant_value(test.left)
            if key_val is not _UNKNOWN_KEY and isinstance(test.comparators[0], ast.Name):
                return test.comparators[0].id, key_val, True
        elif isinstance(op, ast.NotIn):
            key_val = _extract_constant_value(test.left)
            if key_val is not _UNKNOWN_KEY and isinstance(test.comparators[0], ast.Name):
                return test.comparators[0].id, key_val, False
    return None


def _extract_all_modified_names(node: ast.AST) -> set[str]:
    """Extract all variable names that are modified, assigned, deleted, or called in an AST subtree."""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for t in child.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                    names.add(t.value.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    for elt in ast.walk(t):
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(child.target, ast.Name):
                names.add(child.target.id)
            elif isinstance(child.target, ast.Subscript) and isinstance(child.target.value, ast.Name):
                names.add(child.target.value.id)
        elif isinstance(child, ast.Delete):
            for t in child.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                    names.add(t.value.id)
        elif isinstance(child, ast.Call):
            if isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name):
                names.add(child.func.value.id)
            for arg in child.args:
                if isinstance(arg, ast.Name):
                    names.add(arg.id)
            for kw in child.keywords:
                if isinstance(kw.value, ast.Name):
                    names.add(kw.value.id)
    return names


@dataclass
class CollectionLiteralState:
    kind: str  # "LIST", "TUPLE", "DICT"
    length: int | None = None
    known_keys: set[Any] | None = None
    guarded_keys: set[Any] = field(default_factory=set)

    def copy(self) -> CollectionLiteralState:
        return CollectionLiteralState(
            kind=self.kind,
            length=self.length,
            known_keys=set(self.known_keys) if self.known_keys is not None else None,
            guarded_keys=set(self.guarded_keys),
        )


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


class NoneDereferenceDetector(ast.NodeVisitor):
    """
    AST visitor that scans for possible or definite None dereferences (SEM002):
      - Direct dereference of literal None (AttributeError / TypeError) [HIGH confidence]
      - Dereference of variable in branches where known to be None [HIGH confidence]
      - Local variables assigned None without reassignment or guard [HIGH confidence]
      - None-defaulted parameters dereferenced without effective guard [MEDIUM confidence]
      - Immediate .group() chaining on re.match() / re.search() [MEDIUM confidence]
    """

    def __init__(self, file_path: str = "<string>") -> None:
        self.file_path = file_path
        self.issues: list[Issue] = []
        self._try_handlers_stack: list[list[ast.ExceptHandler]] = []
        # Stack of variable state dicts: var_name -> "DEFINITELY_NONE" | "MAYBE_NONE" | "KNOWN_NOT_NONE"
        self._var_state_stack: list[dict[str, str]] = [{}]

    @property
    def _current_var_state(self) -> dict[str, str]:
        return self._var_state_stack[-1]

    def _is_enclosed_in_attribute_error_handler(self) -> bool:
        for handlers in self._try_handlers_stack:
            if any(_catches_attribute_error(h) for h in handlers):
                return True
        return False

    def _is_enclosed_in_type_error_handler(self) -> bool:
        for handlers in self._try_handlers_stack:
            if any(_catches_type_error(h) for h in handlers):
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

    def _extract_names_from_test(self, test: ast.AST) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(test):
            if isinstance(node, ast.Name):
                names.add(node.id)
        return names

    def _visit_statement_sequence(self, stmts: list[ast.stmt]) -> None:
        """Visit a sequence of statements while tracking variable None states and guards."""
        current_vars = dict(self._current_var_state)

        for stmt in stmts:
            self._var_state_stack.append(dict(current_vars))
            self.visit(stmt)
            self._var_state_stack.pop()

            # 1. Assignment handling
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if _is_none_literal(stmt.value):
                            current_vars[target.id] = "DEFINITELY_NONE"
                        else:
                            current_vars[target.id] = "KNOWN_NOT_NONE"
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.value and _is_none_literal(stmt.value):
                    current_vars[stmt.target.id] = "DEFINITELY_NONE"
                elif stmt.value:
                    current_vars[stmt.target.id] = "KNOWN_NOT_NONE"
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                current_vars[stmt.target.id] = "KNOWN_NOT_NONE"

            # 2. If guards with exit or assignment
            if isinstance(stmt, ast.If):
                body_exits = any(_statement_always_exits(s) for s in stmt.body)
                orelse_exits = bool(stmt.orelse) and any(_statement_always_exits(s) for s in stmt.orelse)

                for var in self._extract_names_from_test(stmt.test):
                    if _is_test_checking_is_none(stmt.test, var) or _is_test_checking_falsy(stmt.test, var):
                        if (body_exits and (not stmt.orelse or orelse_exits)) or (
                            not stmt.orelse and _body_always_assigns_non_none(stmt.body, var)
                        ):
                            current_vars[var] = "KNOWN_NOT_NONE"
                    elif (
                        _is_test_checking_is_not_none(stmt.test, var) or _is_test_checking_truthy(stmt.test, var)
                    ) and orelse_exits:
                        current_vars[var] = "KNOWN_NOT_NONE"

    def _extract_function_params(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
        """Extract function parameters and determine if any default to None."""
        var_state: dict[str, str] = {}
        # Positional arguments
        args = node.args.args
        defaults = node.args.defaults
        num_no_defaults = len(args) - len(defaults)
        for i, default in enumerate(defaults):
            arg = args[num_no_defaults + i]
            if _is_none_literal(default):
                var_state[arg.arg] = "MAYBE_NONE"

        # Keyword-only arguments
        for kwarg, kw_default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            if kw_default is not None and _is_none_literal(kw_default):
                var_state[kwarg.arg] = "MAYBE_NONE"

        return var_state

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        params_state = self._extract_function_params(node)
        self._var_state_stack.append(params_state)
        self._visit_statement_sequence(node.body)
        self._var_state_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        params_state = self._extract_function_params(node)
        self._var_state_stack.append(params_state)
        self._visit_statement_sequence(node.body)
        self._var_state_stack.pop()

    def visit_Module(self, node: ast.Module) -> None:
        self._var_state_stack.append({})
        self._visit_statement_sequence(node.body)
        self._var_state_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        # Visit condition expression first
        self.visit(node.test)

        # 1. Body branch
        body_vars = dict(self._current_var_state)
        for var in self._extract_names_from_test(node.test):
            if _is_test_checking_is_none(node.test, var):
                body_vars[var] = "DEFINITELY_NONE"
            elif _is_test_checking_is_not_none(node.test, var) or _is_test_checking_truthy(node.test, var):
                body_vars[var] = "KNOWN_NOT_NONE"

        self._var_state_stack.append(body_vars)
        self._visit_statement_sequence(node.body)
        self._var_state_stack.pop()

        # 2. Else branch
        else_vars = dict(self._current_var_state)
        for var in self._extract_names_from_test(node.test):
            if _is_test_checking_is_none(node.test, var):
                else_vars[var] = "KNOWN_NOT_NONE"
            elif _is_test_checking_is_not_none(node.test, var):
                else_vars[var] = "DEFINITELY_NONE"

        self._var_state_stack.append(else_vars)
        self._visit_statement_sequence(node.orelse)
        self._var_state_stack.pop()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Recurse into children first
        self.generic_visit(node)

        # Case 1: Direct literal None attribute dereference (e.g. 'None.attr')
        if _is_none_literal(node.value):
            if self._is_enclosed_in_attribute_error_handler():
                return
            self.issues.append(
                Issue(
                    code="SEM002",
                    message=f"Direct dereference of literal None: attribute '{node.attr}' access will raise AttributeError.",
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    severity="WARNING",
                    tool="semantic_analyzer",
                    fixable=False,
                    confidence="HIGH",
                    explanation=(
                        f"Accessing attribute '{node.attr}' directly on literal None unconditionally "
                        f"raises an unhandled AttributeError exception when reached."
                    ),
                    suggested_fix="Do not access attributes on None; ensure the object is initialized to a valid instance.",
                )
            )
            return

        # Case 2: Immediate regex match chaining (e.g. 're.match(...).group()')
        if node.attr in ("group", "groups", "groupdict"):
            is_regex, regex_call_str = _is_regex_match_call(node.value)
            if is_regex:
                if self._is_enclosed_in_attribute_error_handler():
                    return
                self.issues.append(
                    Issue(
                        code="SEM002",
                        message=(
                            f"Immediate '{node.attr}()' call on '{regex_call_str}' may dereference "
                            f"None if no match is found."
                        ),
                        file_path=self.file_path,
                        line=node.lineno,
                        column=node.col_offset,
                        severity="WARNING",
                        tool="semantic_analyzer",
                        fixable=False,
                        confidence="MEDIUM",
                        explanation=(
                            f"Static AST analysis detected immediate '.{node.attr}()' chaining on '{regex_call_str}'. "
                            f"When no regex match is found, '{regex_call_str}' returns None, causing an unhandled "
                            f"AttributeError at runtime."
                        ),
                        suggested_fix=(
                            f"Store the regex match in a variable and verify it is not None before calling '.{node.attr}()', "
                            f"or handle AttributeError."
                        ),
                    )
                )
                return

        # Case 3: Variable attribute dereference ('x.attr')
        var_name = _get_name_id(node.value)
        if not var_name:
            return

        var_state = self._current_var_state.get(var_name)
        if not var_state or var_state == "KNOWN_NOT_NONE":
            return

        if self._is_enclosed_in_attribute_error_handler():
            return

        if var_state == "DEFINITELY_NONE":
            self.issues.append(
                Issue(
                    code="SEM002",
                    message=(
                        f"Dereference of variable '{var_name}' which is known to be None: "
                        f"attribute '{node.attr}' access will raise AttributeError."
                    ),
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    severity="WARNING",
                    tool="semantic_analyzer",
                    fixable=False,
                    confidence="HIGH",
                    explanation=(
                        f"Variable '{var_name}' is proven to evaluate to None in this execution path. "
                        f"Accessing attribute '{node.attr}' on None unconditionally raises AttributeError."
                    ),
                    suggested_fix=(
                        f"Verify that '{var_name}' is not None prior to accessing '{node.attr}', "
                        f"or provide a valid non-None assignment."
                    ),
                )
            )
        elif var_state == "MAYBE_NONE":
            self.issues.append(
                Issue(
                    code="SEM002",
                    message=(
                        f"Possible None dereference: parameter '{var_name}' defaults to None "
                        f"and is accessed via attribute '{node.attr}' without an effective guard."
                    ),
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    severity="WARNING",
                    tool="semantic_analyzer",
                    fixable=False,
                    confidence="MEDIUM",
                    explanation=(
                        f"Parameter '{var_name}' has a default value of None and is dereferenced "
                        f"via '.{node.attr}' without a preceding None-check or early return guard. "
                        f"If the caller relies on the default or passes None, AttributeError will occur."
                    ),
                    suggested_fix=(
                        f"Add a guard such as 'if {var_name} is None: raise ValueError(...)' or "
                        f"'if {var_name} is not None: ...' before accessing '.{node.attr}'."
                    ),
                )
            )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Recurse into children first
        self.generic_visit(node)

        # Case 1: Direct literal None subscript ('None[k]')
        if _is_none_literal(node.value):
            if self._is_enclosed_in_type_error_handler():
                return
            self.issues.append(
                Issue(
                    code="SEM002",
                    message="Direct subscript on literal None will raise TypeError.",
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    severity="WARNING",
                    tool="semantic_analyzer",
                    fixable=False,
                    confidence="HIGH",
                    explanation=(
                        "Subscripting literal None ('None[...]') unconditionally raises TypeError: "
                        "'NoneType' object is not subscriptable."
                    ),
                    suggested_fix="Do not subscript None; ensure the object is initialized to a valid container (dict, list, etc.).",
                )
            )
            return

        # Case 2: Variable subscript ('x[k]')
        var_name = _get_name_id(node.value)
        if not var_name:
            return

        var_state = self._current_var_state.get(var_name)
        if not var_state or var_state == "KNOWN_NOT_NONE":
            return

        if self._is_enclosed_in_type_error_handler():
            return

        if var_state == "DEFINITELY_NONE":
            self.issues.append(
                Issue(
                    code="SEM002",
                    message=(
                        f"Subscript on variable '{var_name}' which is known to be None: "
                        f"subscript access will raise TypeError."
                    ),
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    severity="WARNING",
                    tool="semantic_analyzer",
                    fixable=False,
                    confidence="HIGH",
                    explanation=(
                        f"Variable '{var_name}' is proven to evaluate to None in this execution path. "
                        f"Subscripting None raises TypeError: 'NoneType' object is not subscriptable."
                    ),
                    suggested_fix=(
                        f"Verify that '{var_name}' is not None prior to indexing, "
                        f"or assign a valid container default."
                    ),
                )
            )
        elif var_state == "MAYBE_NONE":
            self.issues.append(
                Issue(
                    code="SEM002",
                    message=(
                        f"Possible None dereference: parameter '{var_name}' defaults to None "
                        f"and is subscripted without an effective guard."
                    ),
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    severity="WARNING",
                    tool="semantic_analyzer",
                    fixable=False,
                    confidence="MEDIUM",
                    explanation=(
                        f"Parameter '{var_name}' has a default value of None and is subscripted "
                        f"without a preceding None-check or early return guard. "
                        f"If the caller relies on the default or passes None, TypeError will occur."
                    ),
                    suggested_fix=(
                        f"Add a guard such as 'if {var_name} is None: raise ValueError(...)' or "
                        f"'if {var_name} is None: {var_name} = {{}}' before subscripting."
                    ),
                )
            )

    def visit_Call(self, node: ast.Call) -> None:
        # Recurse into children first
        self.generic_visit(node)

        # Case 1: Direct literal None call ('None()')
        if _is_none_literal(node.func):
            if self._is_enclosed_in_type_error_handler():
                return
            self.issues.append(
                Issue(
                    code="SEM002",
                    message="Calling literal None will raise TypeError.",
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    severity="WARNING",
                    tool="semantic_analyzer",
                    fixable=False,
                    confidence="HIGH",
                    explanation="Calling literal None unconditionally raises TypeError: 'NoneType' object is not callable.",
                    suggested_fix="Do not invoke None; ensure a valid callable is provided.",
                )
            )
            return

        # Case 2: Variable call ('x()')
        var_name = _get_name_id(node.func)
        if not var_name:
            return

        var_state = self._current_var_state.get(var_name)
        if not var_state or var_state == "KNOWN_NOT_NONE":
            return

        if self._is_enclosed_in_type_error_handler():
            return

        if var_state == "DEFINITELY_NONE":
            self.issues.append(
                Issue(
                    code="SEM002",
                    message=(
                        f"Calling variable '{var_name}' which is known to be None: "
                        f"invocation will raise TypeError."
                    ),
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    severity="WARNING",
                    tool="semantic_analyzer",
                    fixable=False,
                    confidence="HIGH",
                    explanation=(
                        f"Variable '{var_name}' is proven to evaluate to None in this execution path. "
                        f"Calling None raises TypeError: 'NoneType' object is not callable."
                    ),
                    suggested_fix=(
                        f"Verify that '{var_name}' is callable before invoking, "
                        f"e.g. 'if callable({var_name}): {var_name}()'."
                    ),
                )
            )
        elif var_state == "MAYBE_NONE":
            self.issues.append(
                Issue(
                    code="SEM002",
                    message=(
                        f"Possible None dereference: parameter '{var_name}' defaults to None "
                        f"and is called without an effective guard."
                    ),
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    severity="WARNING",
                    tool="semantic_analyzer",
                    fixable=False,
                    confidence="MEDIUM",
                    explanation=(
                        f"Parameter '{var_name}' has a default value of None and is invoked directly "
                        f"without a preceding None or callable check. "
                        f"If the caller relies on the default or passes None, TypeError will occur."
                    ),
                    suggested_fix=(
                        f"Add a guard such as 'if {var_name} is not None: {var_name}()' or "
                        f"'if callable({var_name}): {var_name}()'."
                    ),
                )
            )


class IndexAndKeyAccessDetector(ast.NodeVisitor):
    """
    AST visitor that scans for provably out-of-bounds constant integer sequence
    indexing or provably missing constant dictionary keys (SEM003):
      - Direct out-of-bounds index on literal list or tuple [HIGH confidence]
      - Direct missing key on literal dictionary [HIGH confidence]
      - Local variable assigned literal list/tuple accessed out-of-bounds without mutation [HIGH confidence]
      - Local variable assigned literal dictionary accessed with absent key without mutation [HIGH confidence]
    """

    def __init__(self, file_path: str = "<string>") -> None:
        self.file_path = file_path
        self.issues: list[Issue] = []
        self._try_handlers_stack: list[list[ast.ExceptHandler]] = []
        self._collection_stack: list[dict[str, CollectionLiteralState]] = [{}]

    @property
    def _current_collections(self) -> dict[str, CollectionLiteralState]:
        return self._collection_stack[-1]

    def _is_enclosed_in_index_error_handler(self) -> bool:
        for handlers in self._try_handlers_stack:
            if any(_catches_index_error(h) for h in handlers):
                return True
        return False

    def _is_enclosed_in_key_error_handler(self) -> bool:
        for handlers in self._try_handlers_stack:
            if any(_catches_key_error(h) for h in handlers):
                return True
        return False

    def visit_Try(self, node: ast.Try) -> None:
        self._try_handlers_stack.append(node.handlers)
        for stmt in node.body:
            self.visit(stmt)
        self._try_handlers_stack.pop()

        for handler in node.handlers:
            self.visit(handler)

        for stmt in node.orelse:
            self.visit(stmt)

        for stmt in node.finalbody:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._collection_stack.append({})
        self._visit_statement_sequence(node.body)
        self._collection_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._collection_stack.append({})
        self._visit_statement_sequence(node.body)
        self._collection_stack.pop()

    def visit_Module(self, node: ast.Module) -> None:
        self._collection_stack.append({})
        self._visit_statement_sequence(node.body)
        self._collection_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)

        guard = _extract_membership_guard(node.test)

        # 1. Body branch
        body_collections = {k: v.copy() for k, v in self._current_collections.items()}
        if guard and guard[2] is True:
            var_name, key_val, _ = guard
            if var_name in body_collections:
                body_collections[var_name].guarded_keys.add(key_val)

        self._collection_stack.append(body_collections)
        self._visit_statement_sequence(node.body)
        self._collection_stack.pop()

        # 2. Else branch
        else_collections = {k: v.copy() for k, v in self._current_collections.items()}
        if guard and guard[2] is False:
            var_name, key_val, _ = guard
            if var_name in else_collections:
                else_collections[var_name].guarded_keys.add(key_val)

        self._collection_stack.append(else_collections)
        self._visit_statement_sequence(node.orelse)
        self._collection_stack.pop()

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.generic_visit(node)

        # Slices are always safe in Python (never raise IndexError)
        if isinstance(node.slice, ast.Slice):
            return

        # Do not duplicate SEM002 findings for subscripting literal None
        if _is_none_literal(node.value):
            return

        # Case 1: Direct literal sequence ([1, 2][5] or (1, 2)[-3])
        if isinstance(node.value, (ast.List, ast.Tuple)):
            length = _extract_sequence_length(node.value)
            if length is not None:
                idx = _extract_constant_value(node.slice)
                if isinstance(idx, int) and not isinstance(idx, bool):
                    is_oob = (length == 0) or (idx >= length) or (idx < -length)
                    if is_oob:
                        if self._is_enclosed_in_index_error_handler():
                            return
                        kind = "list" if isinstance(node.value, ast.List) else "tuple"
                        self.issues.append(
                            Issue(
                                code="SEM003",
                                message=(
                                    f"Sequence index out of range: constant index {idx} is out of bounds for "
                                    f"literal {kind} of length {length}."
                                ),
                                file_path=self.file_path,
                                line=node.lineno,
                                column=node.col_offset,
                                severity="WARNING",
                                tool="semantic_analyzer",
                                fixable=False,
                                confidence="HIGH",
                                explanation=(
                                    f"Static AST analysis detected constant index {idx} accessing a literal {kind} "
                                    f"containing {length} element(s). Valid indices are between "
                                    f"{-length if length > 0 else 0} and {length - 1 if length > 0 else -1}. "
                                    f"This subscript access unconditionally raises an unhandled IndexError at runtime."
                                ),
                                suggested_fix=(
                                    f"Use an index within bounds (between {-length if length > 0 else 0} and "
                                    f"{length - 1 if length > 0 else -1}), or guard with 'if 0 <= idx < len(...):'."
                                ),
                            )
                        )
                        return

        # Case 2: Direct literal dictionary ({"a": 1}["b"])
        if isinstance(node.value, ast.Dict):
            if not isinstance(node.ctx, ast.Load):
                return
            keys = _extract_dict_literal_keys(node.value)
            if keys is not None:
                key = _extract_constant_value(node.slice)
                if key is not _UNKNOWN_KEY and key not in keys:
                    if self._is_enclosed_in_key_error_handler():
                        return
                    key_repr = repr(key)
                    keys_repr = "{" + ", ".join(repr(k) for k in sorted(keys, key=lambda x: str(x))) + "}"
                    self.issues.append(
                        Issue(
                            code="SEM003",
                            message=(
                                f"Missing dictionary key: key {key_repr} is provably absent from "
                                f"literal dictionary."
                            ),
                            file_path=self.file_path,
                            line=node.lineno,
                            column=node.col_offset,
                            severity="WARNING",
                            tool="semantic_analyzer",
                            fixable=False,
                            confidence="HIGH",
                            explanation=(
                                f"Static AST analysis detected subscript access with key {key_repr} on a literal "
                                f"dictionary with defined keys {keys_repr}. "
                                f"This subscript access unconditionally raises an unhandled KeyError at runtime."
                            ),
                            suggested_fix=(
                                f"Verify that key {key_repr} exists in the dictionary, use "
                                f"'.get({key_repr}, default)' instead of subscripting, or handle KeyError."
                            ),
                        )
                    )
                    return

        # Case 3: Local variable sequence or dictionary
        var_name = _get_name_id(node.value)
        if not var_name:
            return

        state = self._current_collections.get(var_name)
        if not state:
            return

        if state.kind in ("LIST", "TUPLE"):
            length = state.length
            if length is not None:
                idx = _extract_constant_value(node.slice)
                if isinstance(idx, int) and not isinstance(idx, bool):
                    is_oob = (length == 0) or (idx >= length) or (idx < -length)
                    if is_oob:
                        if self._is_enclosed_in_index_error_handler():
                            return
                        kind = state.kind.lower()
                        self.issues.append(
                            Issue(
                                code="SEM003",
                                message=(
                                    f"Sequence index out of range: constant index {idx} is out of bounds for "
                                    f"{kind} '{var_name}' of length {length}."
                                ),
                                file_path=self.file_path,
                                line=node.lineno,
                                column=node.col_offset,
                                severity="WARNING",
                                tool="semantic_analyzer",
                                fixable=False,
                                confidence="HIGH",
                                explanation=(
                                    f"Static AST analysis detected constant index {idx} on variable '{var_name}' "
                                    f"known to be a {kind} of length {length}. Valid indices are between "
                                    f"{-length if length > 0 else 0} and {length - 1 if length > 0 else -1}. "
                                    f"This access unconditionally raises an unhandled IndexError at runtime."
                                ),
                                suggested_fix=(
                                    f"Use an index within bounds (between {-length if length > 0 else 0} and "
                                    f"{length - 1 if length > 0 else -1}), or guard with 'if 0 <= idx < len({var_name}):'."
                                ),
                            )
                        )
                        return

        elif state.kind == "DICT":
            if not isinstance(node.ctx, ast.Load):
                return
            keys = state.known_keys
            if keys is not None:
                key = _extract_constant_value(node.slice)
                if key is not _UNKNOWN_KEY:
                    if key in state.guarded_keys:
                        return
                    if key not in keys:
                        if self._is_enclosed_in_key_error_handler():
                            return
                        key_repr = repr(key)
                        keys_repr = "{" + ", ".join(repr(k) for k in sorted(keys, key=lambda x: str(x))) + "}"
                        self.issues.append(
                            Issue(
                                code="SEM003",
                                message=(
                                    f"Missing dictionary key: key {key_repr} is provably absent from "
                                    f"dictionary '{var_name}'."
                                ),
                                file_path=self.file_path,
                                line=node.lineno,
                                column=node.col_offset,
                                severity="WARNING",
                                tool="semantic_analyzer",
                                fixable=False,
                                confidence="HIGH",
                                explanation=(
                                    f"Static AST analysis detected subscript access for key {key_repr} on variable "
                                    f"'{var_name}' known to be a dictionary with keys {keys_repr}. "
                                    f"This access unconditionally raises an unhandled KeyError at runtime."
                                ),
                                suggested_fix=(
                                    f"Verify that key {key_repr} exists before accessing, use 'if {key_repr} in {var_name}:', "
                                    f"or use '{var_name}.get({key_repr}, default)' instead of subscripting."
                                ),
                            )
                        )
                        return

    def _visit_statement_sequence(self, stmts: list[ast.stmt]) -> None:
        current_collections = {k: v.copy() for k, v in self._current_collections.items()}

        for stmt in stmts:
            self._collection_stack.append({k: v.copy() for k, v in current_collections.items()})
            self.visit(stmt)
            self._collection_stack.pop()

            # 1. Check for subscript writes (e.g. d['k'] = v, lst[0] = v)
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        current_collections.pop(target.value.id, None)

                # Check for alias creation (e.g. y = x, where x is in current_collections)
                if isinstance(stmt.value, ast.Name) and stmt.value.id in current_collections:
                    current_collections.pop(stmt.value.id, None)

                # Clear previous tracking for any target variable being assigned to
                for target in stmt.targets:
                    for child_target in ast.walk(target):
                        if isinstance(child_target, ast.Name):
                            current_collections.pop(child_target.id, None)

                # Process direct collection assignments to single names
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        if isinstance(stmt.value, (ast.List, ast.Tuple)):
                            length = _extract_sequence_length(stmt.value)
                            if length is not None:
                                kind = "LIST" if isinstance(stmt.value, ast.List) else "TUPLE"
                                current_collections[var_name] = CollectionLiteralState(kind=kind, length=length)
                        elif isinstance(stmt.value, ast.Dict):
                            keys = _extract_dict_literal_keys(stmt.value)
                            if keys is not None:
                                current_collections[var_name] = CollectionLiteralState(kind="DICT", known_keys=keys)

            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Subscript) and isinstance(stmt.target.value, ast.Name):
                    current_collections.pop(stmt.target.value.id, None)

                if isinstance(stmt.target, ast.Name):
                    var_name = stmt.target.id
                    current_collections.pop(var_name, None)

                    if stmt.value and isinstance(stmt.value, ast.Name) and stmt.value.id in current_collections:
                        current_collections.pop(stmt.value.id, None)

                    if stmt.value and isinstance(stmt.value, (ast.List, ast.Tuple)):
                        length = _extract_sequence_length(stmt.value)
                        if length is not None:
                            kind = "LIST" if isinstance(stmt.value, ast.List) else "TUPLE"
                            current_collections[var_name] = CollectionLiteralState(kind=kind, length=length)
                    elif stmt.value and isinstance(stmt.value, ast.Dict):
                        keys = _extract_dict_literal_keys(stmt.value)
                        if keys is not None:
                            current_collections[var_name] = CollectionLiteralState(kind="DICT", known_keys=keys)

            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Name):
                    current_collections.pop(stmt.target.id, None)
                elif isinstance(stmt.target, ast.Subscript) and isinstance(stmt.target.value, ast.Name):
                    current_collections.pop(stmt.target.value.id, None)

            elif isinstance(stmt, ast.Delete):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        current_collections.pop(target.id, None)
                    elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        current_collections.pop(target.value.id, None)

            # 2. Invalidate on method calls and call escapes
            for child in ast.walk(stmt):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name):
                        current_collections.pop(child.func.value.id, None)
                    for arg in child.args:
                        if isinstance(arg, ast.Name):
                            current_collections.pop(arg.id, None)
                    for kw in child.keywords:
                        if isinstance(kw.value, ast.Name):
                            current_collections.pop(kw.value.id, None)

            # 3. Handle If statements and branch merging
            if isinstance(stmt, ast.If):
                guard = _extract_membership_guard(stmt.test)
                body_exits = any(_statement_always_exits(s) for s in stmt.body)

                if guard and not guard[2] and body_exits:
                    var_name, key_val, _ = guard
                    if var_name in current_collections:
                        current_collections[var_name].guarded_keys.add(key_val)

                if body_exits and not stmt.orelse:
                    pass
                else:
                    modified = _extract_all_modified_names(stmt)
                    for m in modified:
                        current_collections.pop(m, None)


def analyze_python_source(source_code: str, file_path: str = "<string>") -> list[Issue]:
    """
    Analyze Python source code string for semantic bug hazards (SEM001, SEM002, SEM003).

    Args:
        source_code: Python code content.
        file_path: Relative or absolute path for location reporting.

    Returns:
        List of parsed Issue objects, sorted by line and column.
    """
    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError:
        # Gracefully handle syntax errors without crashing
        return []

    div_detector = DivisionByZeroDetector(file_path=file_path)
    div_detector.visit(tree)

    none_detector = NoneDereferenceDetector(file_path=file_path)
    none_detector.visit(tree)

    index_key_detector = IndexAndKeyAccessDetector(file_path=file_path)
    index_key_detector.visit(tree)

    all_issues = div_detector.issues + none_detector.issues + index_key_detector.issues
    all_issues.sort(key=lambda i: (i.line or 0, i.column or 0))
    return all_issues


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
