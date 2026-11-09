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


def _block_always_exits(stmts: list[ast.stmt], in_loop: bool = False) -> bool:
    """Check if any statement in the block unconditionally exits."""
    return any(_statement_always_exits(s, in_loop=in_loop) for s in stmts)


def _statement_always_exits(stmt: ast.stmt, in_loop: bool = False) -> bool:
    """Check if a statement unconditionally exits the current block."""
    if isinstance(stmt, (ast.Raise, ast.Return)):
        return True
    if in_loop and isinstance(stmt, (ast.Break, ast.Continue)):
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
        and _block_always_exits(stmt.body, in_loop=in_loop)
        and _block_always_exits(stmt.orelse, in_loop=in_loop)
    )


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


@dataclass(frozen=True)
class BranchFacts:
    """Facts established along a single execution path."""

    nonzero_vars: frozenset[str] = frozenset()
    zero_vars: frozenset[str] = frozenset()
    known_not_none_vars: frozenset[str] = frozenset()
    definitely_none_vars: frozenset[str] = frozenset()
    guarded_keys: frozenset[tuple[str, Any]] = frozenset()
    lacking_keys: frozenset[tuple[str, Any]] = frozenset()

    def merge_conjunction(self, other: BranchFacts) -> BranchFacts:
        """Combine facts when both conditions must hold simultaneously."""
        new_nonzero = self.nonzero_vars | other.nonzero_vars
        new_zero = (self.zero_vars | other.zero_vars) - new_nonzero

        new_not_none = self.known_not_none_vars | other.known_not_none_vars
        new_def_none = (self.definitely_none_vars | other.definitely_none_vars) - new_not_none

        return BranchFacts(
            nonzero_vars=new_nonzero,
            zero_vars=new_zero,
            known_not_none_vars=new_not_none,
            definitely_none_vars=new_def_none,
            guarded_keys=self.guarded_keys | other.guarded_keys,
            lacking_keys=self.lacking_keys | other.lacking_keys,
        )


def _extract_branch_facts(test: ast.AST) -> tuple[BranchFacts, BranchFacts]:
    """
    Extract facts that hold when `test` evaluates to True vs False.
    Returns (true_facts, false_facts).
    """
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        tf, ff = _extract_branch_facts(test.operand)
        return ff, tf

    if isinstance(test, ast.BoolOp):
        if isinstance(test.op, ast.And):
            tf = BranchFacts()
            for v in test.values:
                v_tf, _ = _extract_branch_facts(v)
                tf = tf.merge_conjunction(v_tf)
            return tf, BranchFacts()
        if isinstance(test.op, ast.Or):
            ff = BranchFacts()
            for v in test.values:
                _, v_ff = _extract_branch_facts(v)
                ff = ff.merge_conjunction(v_ff)
            return BranchFacts(), ff

    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op = test.ops[0]
        left = test.left
        right = test.comparators[0]

        left_name = left.target.id if isinstance(left, ast.NamedExpr) and isinstance(left.target, ast.Name) else _get_name_id(left)
        right_name = right.target.id if isinstance(right, ast.NamedExpr) and isinstance(right.target, ast.Name) else _get_name_id(right)

        if isinstance(op, ast.Eq):
            if left_name and _is_zero_literal(right):
                return BranchFacts(zero_vars=frozenset({left_name})), BranchFacts(nonzero_vars=frozenset({left_name}))
            if right_name and _is_zero_literal(left):
                return BranchFacts(zero_vars=frozenset({right_name})), BranchFacts(nonzero_vars=frozenset({right_name}))
            if left_name and _is_none_literal(right):
                return BranchFacts(definitely_none_vars=frozenset({left_name})), BranchFacts(known_not_none_vars=frozenset({left_name}))
            if right_name and _is_none_literal(left):
                return BranchFacts(definitely_none_vars=frozenset({right_name})), BranchFacts(known_not_none_vars=frozenset({right_name}))

        elif isinstance(op, ast.NotEq):
            if left_name and _is_zero_literal(right):
                return BranchFacts(nonzero_vars=frozenset({left_name})), BranchFacts(zero_vars=frozenset({left_name}))
            if right_name and _is_zero_literal(left):
                return BranchFacts(nonzero_vars=frozenset({right_name})), BranchFacts(zero_vars=frozenset({right_name}))
            if left_name and _is_none_literal(right):
                return BranchFacts(known_not_none_vars=frozenset({left_name})), BranchFacts(definitely_none_vars=frozenset({left_name}))
            if right_name and _is_none_literal(left):
                return BranchFacts(known_not_none_vars=frozenset({right_name})), BranchFacts(definitely_none_vars=frozenset({right_name}))

        elif isinstance(op, ast.Is):
            if left_name and _is_none_literal(right):
                return BranchFacts(definitely_none_vars=frozenset({left_name})), BranchFacts(known_not_none_vars=frozenset({left_name}))
            if right_name and _is_none_literal(left):
                return BranchFacts(definitely_none_vars=frozenset({right_name})), BranchFacts(known_not_none_vars=frozenset({right_name}))

        elif isinstance(op, ast.IsNot):
            if left_name and _is_none_literal(right):
                return BranchFacts(known_not_none_vars=frozenset({left_name})), BranchFacts(definitely_none_vars=frozenset({left_name}))
            if right_name and _is_none_literal(left):
                return BranchFacts(known_not_none_vars=frozenset({right_name})), BranchFacts(definitely_none_vars=frozenset({right_name}))

        elif isinstance(op, (ast.Gt, ast.Lt)):
            if left_name and _is_zero_literal(right):
                return BranchFacts(nonzero_vars=frozenset({left_name})), BranchFacts()
            if right_name and _is_zero_literal(left):
                return BranchFacts(nonzero_vars=frozenset({right_name})), BranchFacts()

        elif isinstance(op, ast.In):
            key_val = _extract_constant_value(left)
            if key_val is not _UNKNOWN_KEY and right_name:
                return BranchFacts(guarded_keys=frozenset({(right_name, key_val)})), BranchFacts(lacking_keys=frozenset({(right_name, key_val)}))

        elif isinstance(op, ast.NotIn):
            key_val = _extract_constant_value(left)
            if key_val is not _UNKNOWN_KEY and right_name:
                return BranchFacts(lacking_keys=frozenset({(right_name, key_val)})), BranchFacts(guarded_keys=frozenset({(right_name, key_val)}))

    if isinstance(test, ast.Name):
        var = test.id
        return BranchFacts(nonzero_vars=frozenset({var}), known_not_none_vars=frozenset({var})), BranchFacts(zero_vars=frozenset({var}))

    if isinstance(test, ast.NamedExpr) and isinstance(test.target, ast.Name):
        var = test.target.id
        val_tf, val_ff = _extract_branch_facts(test.value)
        tf = val_tf.merge_conjunction(BranchFacts(nonzero_vars=frozenset({var}), known_not_none_vars=frozenset({var})))
        ff = val_ff.merge_conjunction(BranchFacts(zero_vars=frozenset({var})))
        return tf, ff

    return BranchFacts(), BranchFacts()


def _is_test_checking_zero(test: ast.AST, var_name: str) -> bool:
    """Check if test implies var_name is zero when true."""
    tf, _ = _extract_branch_facts(test)
    return var_name in tf.zero_vars


def _is_test_checking_nonzero(test: ast.AST, var_name: str) -> bool:
    """Check if test implies var_name is non-zero when true."""
    tf, _ = _extract_branch_facts(test)
    return var_name in tf.nonzero_vars


def _is_test_checking_is_none(test: ast.AST, var_name: str) -> bool:
    """Check if test implies var_name is None when true."""
    tf, _ = _extract_branch_facts(test)
    return var_name in tf.definitely_none_vars


def _is_test_checking_is_not_none(test: ast.AST, var_name: str) -> bool:
    """Check if test implies var_name is not None when true."""
    tf, _ = _extract_branch_facts(test)
    return var_name in tf.known_not_none_vars


def _is_test_checking_truthy(test: ast.AST, var_name: str) -> bool:
    """Check if test directly tests truthiness of var_name."""
    tf, _ = _extract_branch_facts(test)
    return var_name in tf.known_not_none_vars and var_name in tf.nonzero_vars


def _is_test_checking_falsy(test: ast.AST, var_name: str) -> bool:
    """Check if test implies falsiness of var_name when true."""
    tf, _ = _extract_branch_facts(test)
    return var_name in tf.zero_vars or (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and isinstance(test.operand, ast.Name) and test.operand.id == var_name)


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
        elif (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.NamedExpr)
            and isinstance(stmt.value.target, ast.Name)
            and stmt.value.target.id == var_name
        ):
            assigned_non_none = not _is_none_literal(stmt.value.value)
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


def _walk_enclosing_scope(node: ast.AST):
    """
    Traverse AST nodes in the enclosing scope without descending into
    nested functions, async functions, lambdas, or class definitions.
    Nested function, class definition, and lambda nodes are yielded so
    enclosing scope bindings can be inspected, but their inner bodies
    and local scopes are not traversed.
    """
    queue = [node]
    while queue:
        current = queue.pop(0)
        yield current
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.GeneratorExp)):
            continue
        queue.extend(ast.iter_child_nodes(current))


def _classify_function_returns_none(node: ast.FunctionDef) -> bool:
    """Determine if a synchronous, undecorated function unconditionally returns None.

    Conservative classification rules:
      - Decorated functions are UNKNOWN (decorators can wrap return values).
      - Functions containing yield/yield-from are generators, not None-returners.
      - Functions with no explicit return statements are UNKNOWN (could be stubs/hooks).
      - All reachable return statements must be ``return None`` or bare ``return``.
      - If any return has a non-None value expression, the function is UNKNOWN.

    This intentionally does NOT descend into nested functions, lambdas, or classes.
    """
    if node.decorator_list:
        return False
    has_return = False
    # Walk the function body statements (not the FunctionDef node itself,
    # because _walk_enclosing_scope skips children of FunctionDef nodes).
    for body_stmt in node.body:
        for child in _walk_enclosing_scope(body_stmt):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return False
            if isinstance(child, ast.Return):
                has_return = True
                if child.value is not None and not _is_none_literal(child.value):
                    return False
    return has_return


def _extract_all_modified_names(node: ast.AST) -> set[str]:
    """Extract all variable names that are modified, assigned, deleted, or called in an AST subtree without descending into nested scopes."""
    names: set[str] = set()
    for child in _walk_enclosing_scope(node):
        if isinstance(child, ast.Assign):
            for t in child.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                elif isinstance(t, ast.Subscript):
                    root: ast.AST = t.value
                    while isinstance(root, ast.Subscript):
                        root = root.value
                    if isinstance(root, ast.Name):
                        names.add(root.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    for elt in ast.walk(t):
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(child.target, ast.Name):
                names.add(child.target.id)
            elif isinstance(child.target, ast.Subscript):
                root = child.target.value
                while isinstance(root, ast.Subscript):
                    root = root.value
                if isinstance(root, ast.Name):
                    names.add(root.id)
        elif isinstance(child, ast.Delete):
            for t in child.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                elif isinstance(t, ast.Subscript):
                    root = t.value
                    while isinstance(root, ast.Subscript):
                        root = root.value
                    if isinstance(root, ast.Name):
                        names.add(root.id)
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            for n in ast.walk(child.target):
                if isinstance(n, ast.Name):
                    names.add(n.id)
        elif isinstance(child, (ast.With, ast.AsyncWith)):
            for item in child.items:
                if item.optional_vars:
                    for n in ast.walk(item.optional_vars):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
        elif isinstance(child, ast.Call):
            if isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name):
                names.add(child.func.value.id)
            for arg in child.args:
                if isinstance(arg, ast.Name):
                    names.add(arg.id)
            for kw in child.keywords:
                if isinstance(kw.value, ast.Name):
                    names.add(kw.value.id)
        elif isinstance(child, ast.NamedExpr):
            if isinstance(child.target, ast.Name):
                names.add(child.target.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or (
            isinstance(child, ast.ExceptHandler) and child.name
        ):
            names.add(child.name)
    return names


def _extract_assigned_names_in_subtree(node: ast.AST) -> set[str]:
    """Extract all variable names that are targets of assignments, deletions, or definitions in an AST subtree without descending into nested scopes."""
    names: set[str] = set()
    for child in _walk_enclosing_scope(node):
        if isinstance(child, ast.Assign):
            for t in child.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(child, (ast.AugAssign, ast.AnnAssign)):
            for n in ast.walk(child.target):
                if isinstance(n, ast.Name):
                    names.add(n.id)
        elif isinstance(child, ast.NamedExpr):
            if isinstance(child.target, ast.Name):
                names.add(child.target.id)
        elif isinstance(child, ast.Delete):
            for t in child.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            for n in ast.walk(child.target):
                if isinstance(n, ast.Name):
                    names.add(n.id)
        elif isinstance(child, (ast.With, ast.AsyncWith)):
            for item in child.items:
                if item.optional_vars:
                    for n in ast.walk(item.optional_vars):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or (
            isinstance(child, ast.ExceptHandler) and child.name
        ):
            names.add(child.name)
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


# ============================================================================
# Path-Joining & Filesystem Path Heuristics
# ============================================================================

_PATH_CLASS_NAMES = {
    "Path",
    "PurePath",
    "PosixPath",
    "WindowsPath",
    "PurePosixPath",
    "PureWindowsPath",
}

_PATH_METHOD_NAMES = {
    "cwd",
    "home",
    "resolve",
    "absolute",
    "joinpath",
    "expanduser",
    "with_name",
    "with_suffix",
    "with_stem",
}

_PATH_VAR_NAMES = {
    "tmp_path",
    "tmp_path_factory",
    "tmpdir",
    "tmpdir_factory",
    "path",
    "filepath",
    "file_path",
    "dirpath",
    "dir_path",
    "directory",
    "folder",
    "base_dir",
    "root_dir",
    "out_dir",
    "output_dir",
    "src_dir",
    "dst_dir",
    "target_dir",
    "dest_dir",
    "workdir",
    "workspace",
}

_PATH_NAME_SUFFIXES = (
    "_path",
    "_dir",
    "_file",
    "_folder",
    "Path",
    "Dir",
    "File",
    "Folder",
)

_PATH_NAME_PREFIXES = (
    "path_",
    "dir_",
    "file_",
    "folder_",
)


def _is_path_name(name: str) -> bool:
    """Check if variable name suggests a filesystem path."""
    name_lower = name.lower()
    if name_lower in _PATH_VAR_NAMES:
        return True
    if any(name.endswith(suffix) for suffix in _PATH_NAME_SUFFIXES):
        return True
    return any(name.startswith(prefix) for prefix in _PATH_NAME_PREFIXES)


def _is_path_type_annotation(node: ast.AST | None) -> bool:
    """Check if a type annotation indicates a Path or PathLike object."""
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id in _PATH_CLASS_NAMES or node.id in ("PathLike", "os.PathLike")
    if isinstance(node, ast.Attribute):
        return node.attr in _PATH_CLASS_NAMES or node.attr in ("PathLike",)
    if isinstance(node, ast.Subscript):
        return _is_path_type_annotation(node.slice)
    if isinstance(node, ast.Tuple):
        return any(_is_path_type_annotation(elt) for elt in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_path_type_annotation(node.left) or _is_path_type_annotation(node.right)
    return False


def _is_path_call(node: ast.AST) -> bool:
    """Check if an AST node is a call returning a path object."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _PATH_CLASS_NAMES or _is_path_name(func.id)
    if isinstance(func, ast.Attribute):
        if func.attr in _PATH_CLASS_NAMES or func.attr in _PATH_METHOD_NAMES:
            return True
        return isinstance(func.value, ast.Name) and func.value.id == "pathlib"
    return False


def _is_string_or_bytes_like(node: ast.AST) -> bool:
    """Check if an AST node represents a string, bytes, or formatted string."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, bytes))
    return isinstance(node, ast.JoinedStr)


def _is_path_like_expression(node: ast.AST, path_vars: set[str] | None = None) -> bool:
    """Check if an AST node represents a path or path-like object."""
    if isinstance(node, ast.Name):
        if path_vars and node.id in path_vars:
            return True
        return _is_path_name(node.id)

    if _is_path_call(node):
        return True

    if isinstance(node, ast.Attribute):
        if node.attr in ("parent", "parents", "stem", "suffix", "suffixes"):
            return True
        if _is_path_like_expression(node.value, path_vars):
            return True
        return _is_path_name(node.attr)

    if isinstance(node, ast.Subscript) and _is_path_like_expression(node.value, path_vars):
        return True

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _is_path_joining_operation(node, path_vars)

    return False


def _is_path_joining_operation(node: ast.BinOp, path_vars: set[str] | None = None) -> bool:
    """
    Determine if a BinOp node is a pathlib.Path joining operation rather than arithmetic division.

    Pathlib only overloads '/' (ast.Div), never '//' or '%'.
    """
    if not isinstance(node.op, ast.Div):
        return False

    # Explicit zero literals are never treated as path joining
    if _is_zero_literal(node.right):
        return False

    # In Python, standard '/' with a string/bytes divisor cannot raise ZeroDivisionError:
    # it is either path joining (Path / str) or a TypeError at runtime.
    if _is_string_or_bytes_like(node.right):
        return True

    # Left or right operand is path-like
    return bool(_is_path_like_expression(node.left, path_vars) or _is_path_like_expression(node.right, path_vars))


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
        # Stack of sets of variables known to hold filesystem Path objects
        self._path_vars_stack: list[set[str]] = [set()]
        # Loop context stack tracking whether execution is inside a loop iteration
        self._in_loop_stack: list[bool] = [False]

    @property
    def _current_guarded_vars(self) -> set[str]:
        return self._guarded_vars_stack[-1]

    @property
    def _current_path_vars(self) -> set[str]:
        return self._path_vars_stack[-1]

    @property
    def _in_loop(self) -> bool:
        return self._in_loop_stack[-1]

    def _is_enclosed_in_zero_division_handler(self) -> bool:
        """Check if current execution is protected by a try block catching ZeroDivisionError."""
        for handlers in self._try_handlers_stack:
            if any(_catches_zero_division(h) for h in handlers):
                return True
        return False

    def visit_Try(self, node: ast.Try) -> None:
        # 1. Visit try body with its handlers on the try stack
        self._try_handlers_stack.append(node.handlers)
        self._guarded_vars_stack.append(set(self._current_guarded_vars))
        self._path_vars_stack.append(set(self._current_path_vars))
        self._visit_statement_sequence(node.body)
        self._path_vars_stack.pop()
        self._guarded_vars_stack.pop()
        self._try_handlers_stack.pop()

        # 2. Visit handlers (exceptions inside handlers are not protected by themselves)
        for handler in node.handlers:
            self.visit(handler)

        # 3. Visit orelse (runs if no exception was raised in try)
        self._guarded_vars_stack.append(set(self._current_guarded_vars))
        self._path_vars_stack.append(set(self._current_path_vars))
        self._visit_statement_sequence(node.orelse)
        self._path_vars_stack.pop()
        self._guarded_vars_stack.pop()

        # 4. Visit finalbody
        finally_guards = set(self._current_guarded_vars)
        for stmt in node.body + [s for h in node.handlers for s in h.body] + node.orelse:
            for name in _extract_assigned_names_in_subtree(stmt):
                finally_guards.discard(name)
        for handler in node.handlers:
            if handler.name:
                finally_guards.discard(handler.name)

        self._guarded_vars_stack.append(finally_guards)
        self._path_vars_stack.append(set(self._current_path_vars))
        self._visit_statement_sequence(node.finalbody)
        self._path_vars_stack.pop()
        self._guarded_vars_stack.pop()

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type:
            self.visit(node.type)
        self._guarded_vars_stack.append(set(self._current_guarded_vars))
        self._path_vars_stack.append(set(self._current_path_vars))
        self._visit_statement_sequence(node.body)
        self._path_vars_stack.pop()
        self._guarded_vars_stack.pop()

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.target)
        self.visit(node.iter)
        loop_guards = set(self._current_guarded_vars)
        for t in ast.walk(node.target):
            if isinstance(t, ast.Name):
                loop_guards.discard(t.id)
        self._guarded_vars_stack.append(loop_guards)
        self._in_loop_stack.append(True)
        self._visit_statement_sequence(node.body)
        self._in_loop_stack.pop()
        self._guarded_vars_stack.pop()

        else_guards = set(self._current_guarded_vars)
        for s in node.body:
            for name in _extract_assigned_names_in_subtree(s):
                else_guards.discard(name)
        self._guarded_vars_stack.append(else_guards)
        self._visit_statement_sequence(node.orelse)
        self._guarded_vars_stack.pop()

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit(node.target)
        self.visit(node.iter)
        loop_guards = set(self._current_guarded_vars)
        for t in ast.walk(node.target):
            if isinstance(t, ast.Name):
                loop_guards.discard(t.id)
        self._guarded_vars_stack.append(loop_guards)
        self._in_loop_stack.append(True)
        self._visit_statement_sequence(node.body)
        self._in_loop_stack.pop()
        self._guarded_vars_stack.pop()

        else_guards = set(self._current_guarded_vars)
        for s in node.body:
            for name in _extract_assigned_names_in_subtree(s):
                else_guards.discard(name)
        self._guarded_vars_stack.append(else_guards)
        self._visit_statement_sequence(node.orelse)
        self._guarded_vars_stack.pop()

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        tf, ff = _extract_branch_facts(node.test)
        loop_guards = set(self._current_guarded_vars)
        loop_guards.update(tf.nonzero_vars)

        self._guarded_vars_stack.append(loop_guards)
        self._in_loop_stack.append(True)
        self._visit_statement_sequence(node.body)
        self._in_loop_stack.pop()
        self._guarded_vars_stack.pop()

        else_guards = set(self._current_guarded_vars)
        for s in node.body:
            for name in _extract_assigned_names_in_subtree(s):
                else_guards.discard(name)
        else_guards.update(ff.nonzero_vars)
        self._guarded_vars_stack.append(else_guards)
        self._visit_statement_sequence(node.orelse)
        self._guarded_vars_stack.pop()

    if hasattr(ast, "TryStar"):
        visit_TryStar = visit_Try

    def _visit_statement_sequence(self, stmts: list[ast.stmt]) -> None:
        """Visit a sequence of statements while tracking variable guards, exits, and path variables."""
        current_guards = set(self._current_guarded_vars)
        current_paths = set(self._current_path_vars)

        for stmt in stmts:
            self._guarded_vars_stack.append(set(current_guards))
            self._path_vars_stack.append(set(current_paths))
            self.visit(stmt)
            self._path_vars_stack.pop()
            self._guarded_vars_stack.pop()

            # Track path variables assigned in this statement
            if isinstance(stmt, ast.Assign):
                if _is_path_like_expression(stmt.value, current_paths):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            current_paths.add(target.id)
                        elif isinstance(target, (ast.Tuple, ast.List)):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name) and _is_path_name(elt.id):
                                    current_paths.add(elt.id)
                else:
                    for target in stmt.targets:
                        for name in _extract_assigned_names_in_subtree(target):
                            if not _is_path_name(name):
                                current_paths.discard(name)
            elif isinstance(stmt, ast.AnnAssign):
                if (stmt.value and _is_path_like_expression(stmt.value, current_paths)) or _is_path_type_annotation(stmt.annotation):
                    if isinstance(stmt.target, ast.Name):
                        current_paths.add(stmt.target.id)
                elif isinstance(stmt.target, ast.Name) and not _is_path_name(stmt.target.id):
                    current_paths.discard(stmt.target.id)
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                for name in _extract_assigned_names_in_subtree(stmt.target):
                    if _is_path_name(name):
                        current_paths.add(name)

            # Check if this statement is an exit guard
            if isinstance(stmt, ast.If):
                body_exits = _block_always_exits(stmt.body, in_loop=self._in_loop)
                orelse_exits = bool(stmt.orelse) and _block_always_exits(stmt.orelse, in_loop=self._in_loop)
                promoted: set[str] = set()

                tf, ff = _extract_branch_facts(stmt.test)

                # If body exits, subsequent statements continue only if condition was FALSE
                if body_exits:
                    for var in ff.nonzero_vars:
                        current_guards.add(var)
                        promoted.add(var)
                    for s in stmt.orelse:
                        for name in _extract_assigned_names_in_subtree(s):
                            current_guards.discard(name)

                # If orelse exits, subsequent statements continue only if condition was TRUE
                elif orelse_exits:
                    for var in tf.nonzero_vars:
                        current_guards.add(var)
                        promoted.add(var)
                    for s in stmt.body:
                        for name in _extract_assigned_names_in_subtree(s):
                            current_guards.discard(name)

                else:
                    for name in _extract_assigned_names_in_subtree(stmt):
                        if name not in promoted:
                            current_guards.discard(name)

            # Check if variable reassignment invalidates guard
            assigned_names = self._extract_assigned_names(stmt)
            for name in assigned_names:
                current_guards.discard(name)
            if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.Try)):
                for name in _extract_assigned_names_in_subtree(stmt):
                    current_guards.discard(name)

    def _extract_names_from_test(self, test: ast.AST) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(test):
            if isinstance(node, ast.Name):
                names.add(node.id)
        return names

    def _extract_assigned_names(self, stmt: ast.stmt) -> set[str]:
        names: set[str] = set()
        for child in _walk_enclosing_scope(stmt):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    for n in ast.walk(target):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
            elif isinstance(child, (ast.AugAssign, ast.AnnAssign)):
                for n in ast.walk(child.target):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
            elif isinstance(child, ast.NamedExpr):
                if isinstance(child.target, ast.Name):
                    names.add(child.target.id)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
        return names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._guarded_vars_stack.append(set())
        self._in_loop_stack.append(False)
        fn_paths = set()
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            if _is_path_name(arg.arg) or _is_path_type_annotation(arg.annotation):
                fn_paths.add(arg.arg)
        if node.args.vararg and _is_path_name(node.args.vararg.arg):
            fn_paths.add(node.args.vararg.arg)
        if node.args.kwarg and _is_path_name(node.args.kwarg.arg):
            fn_paths.add(node.args.kwarg.arg)
        self._path_vars_stack.append(fn_paths)
        self._visit_statement_sequence(node.body)
        self._path_vars_stack.pop()
        self._in_loop_stack.pop()
        self._guarded_vars_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._guarded_vars_stack.append(set())
        self._in_loop_stack.append(False)
        fn_paths = set()
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            if _is_path_name(arg.arg) or _is_path_type_annotation(arg.annotation):
                fn_paths.add(arg.arg)
        if node.args.vararg and _is_path_name(node.args.vararg.arg):
            fn_paths.add(node.args.vararg.arg)
        if node.args.kwarg and _is_path_name(node.args.kwarg.arg):
            fn_paths.add(node.args.kwarg.arg)
        self._path_vars_stack.append(fn_paths)
        self._visit_statement_sequence(node.body)
        self._path_vars_stack.pop()
        self._in_loop_stack.pop()
        self._guarded_vars_stack.pop()

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_statement_sequence(node.body)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._current_guarded_vars.discard(node.target.id)
        self.visit(node.target)

    def _visit_comprehension(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> None:
        outer_guards = set(self._current_guarded_vars)
        comp_guards = set(outer_guards)
        self._guarded_vars_stack.append(comp_guards)
        self._path_vars_stack.append(set(self._current_path_vars))

        for gen in node.generators:
            self.visit(gen.iter)
            for t in ast.walk(gen.target):
                if isinstance(t, ast.Name):
                    comp_guards.discard(t.id)
            for if_expr in gen.ifs:
                self.visit(if_expr)
                tf, _ = _extract_branch_facts(if_expr)
                comp_guards.update(tf.nonzero_vars)

        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)

        self._path_vars_stack.pop()
        self._guarded_vars_stack.pop()

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if isinstance(node.op, ast.And):
            accumulated_tf = BranchFacts()
            orig_guards = set(self._current_guarded_vars)
            for val in node.values:
                step_guards = set(orig_guards)
                step_guards.update(accumulated_tf.nonzero_vars)
                self._guarded_vars_stack.append(step_guards)
                self.visit(val)
                self._guarded_vars_stack.pop()

                for name in _extract_all_modified_names(val):
                    orig_guards.discard(name)

                val_tf, _ = _extract_branch_facts(val)
                accumulated_tf = accumulated_tf.merge_conjunction(val_tf)

        elif isinstance(node.op, ast.Or):
            accumulated_ff = BranchFacts()
            orig_guards = set(self._current_guarded_vars)
            for val in node.values:
                step_guards = set(orig_guards)
                step_guards.update(accumulated_ff.nonzero_vars)
                self._guarded_vars_stack.append(step_guards)
                self.visit(val)
                self._guarded_vars_stack.pop()

                for name in _extract_all_modified_names(val):
                    orig_guards.discard(name)

                _, val_ff = _extract_branch_facts(val)
                accumulated_ff = accumulated_ff.merge_conjunction(val_ff)
        else:
            self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        tf, ff = _extract_branch_facts(node.test)

        body_guards = set(self._current_guarded_vars)
        body_guards.update(tf.nonzero_vars)
        self._guarded_vars_stack.append(body_guards)
        self.visit(node.body)
        self._guarded_vars_stack.pop()

        else_guards = set(self._current_guarded_vars)
        else_guards.update(ff.nonzero_vars)
        self._guarded_vars_stack.append(else_guards)
        self.visit(node.orelse)
        self._guarded_vars_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        # Visit condition
        self.visit(node.test)
        tf, ff = _extract_branch_facts(node.test)

        # 1. Body branch: check if test guarantees non-zero
        body_guards = set(self._current_guarded_vars)
        body_guards.update(tf.nonzero_vars)

        self._guarded_vars_stack.append(body_guards)
        self._visit_statement_sequence(node.body)
        self._guarded_vars_stack.pop()

        # 2. Else branch: check if test being false guarantees non-zero
        else_guards = set(self._current_guarded_vars)
        else_guards.update(ff.nonzero_vars)

        self._guarded_vars_stack.append(else_guards)
        self._visit_statement_sequence(node.orelse)
        self._guarded_vars_stack.pop()

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # Recurse into children first
        self.generic_visit(node)

        # Check if operation is division, floor division, or modulo
        if not isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            return

        # Check for pathlib.Path joining operation (e.g. tmp_path / "test.py", p / filename)
        if _is_path_joining_operation(node, self._current_path_vars):
            return

        op_name = {
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
        }.get(type(node.op), "/")

        right = node.right

        # Case 1: Direct literal zero -> HIGH confidence finding
        if _is_zero_literal(right):
            if self._is_enclosed_in_zero_division_handler():
                return
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
        self._var_state_stack: list[dict[str, str]] = [{}]
        self._in_loop_stack: list[bool] = [False]
        self._function_params_stack: list[dict[str, str]] = []
        self._last_if_branches: tuple[dict[str, str], dict[str, str]] | None = None
        self._function_return_summaries: dict[str, str] = {}

    def _invalidate_var(self, vars_dict: dict[str, str], name: str) -> None:
        if self._function_params_stack and name in self._function_params_stack[-1]:
            vars_dict[name] = self._function_params_stack[-1][name]
        else:
            vars_dict.pop(name, None)

    def _call_returns_none(self, value: ast.expr, current_vars: dict[str, str] | None = None) -> bool:
        """Check if a call expression invokes a module-level function summarized as always returning None.

        Returns False (conservative) when:
          - The value is not a simple ``name(...)`` call.
          - The callee name is not in the module-level return summaries.
          - The callee name is shadowed by a local variable or function parameter.
        """
        if not isinstance(value, ast.Call):
            return False
        if not isinstance(value.func, ast.Name):
            return False
        callee = value.func.id
        if self._function_return_summaries.get(callee) != "DEFINITELY_NONE":
            return False
        # Guard against local/parameter shadowing of the module-level name.
        if self._function_params_stack:
            for params in self._function_params_stack:
                if callee in params:
                    return False
            vars_to_check = current_vars if current_vars is not None else self._current_var_state
            if callee in vars_to_check:
                return False
        return True

    @property
    def _current_var_state(self) -> dict[str, str]:
        return self._var_state_stack[-1]

    @property
    def _in_loop(self) -> bool:
        return self._in_loop_stack[-1]

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
        self._var_state_stack.append(dict(self._current_var_state))
        self._visit_statement_sequence(node.body)
        self._var_state_stack.pop()
        self._try_handlers_stack.pop()

        # 2. Visit handlers (exceptions inside handlers are not protected by themselves)
        for handler in node.handlers:
            self.visit(handler)

        # 3. Visit orelse (runs if no exception was raised in try)
        self._var_state_stack.append(dict(self._current_var_state))
        self._visit_statement_sequence(node.orelse)
        self._var_state_stack.pop()

        # 4. Visit finalbody
        finally_vars = dict(self._current_var_state)
        for stmt in node.body + [s for h in node.handlers for s in h.body] + node.orelse:
            for name in _extract_assigned_names_in_subtree(stmt):
                finally_vars.pop(name, None)
        for handler in node.handlers:
            if handler.name:
                finally_vars.pop(handler.name, None)

        self._var_state_stack.append(finally_vars)
        self._visit_statement_sequence(node.finalbody)
        self._var_state_stack.pop()

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type:
            self.visit(node.type)
        handler_vars = dict(self._current_var_state)
        if node.name:
            handler_vars[node.name] = "KNOWN_NOT_NONE"
        self._var_state_stack.append(handler_vars)
        self._visit_statement_sequence(node.body)
        self._var_state_stack.pop()

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.target)
        self.visit(node.iter)
        loop_vars = dict(self._current_var_state)
        for t in ast.walk(node.target):
            if isinstance(t, ast.Name):
                loop_vars[t.id] = "KNOWN_NOT_NONE"
        self._var_state_stack.append(loop_vars)
        self._in_loop_stack.append(True)
        self._visit_statement_sequence(node.body)
        self._in_loop_stack.pop()
        self._var_state_stack.pop()

        else_vars = dict(self._current_var_state)
        for s in node.body:
            for name in _extract_assigned_names_in_subtree(s):
                self._invalidate_var(else_vars, name)
        self._var_state_stack.append(else_vars)
        self._visit_statement_sequence(node.orelse)
        self._var_state_stack.pop()

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit(node.target)
        self.visit(node.iter)
        loop_vars = dict(self._current_var_state)
        for t in ast.walk(node.target):
            if isinstance(t, ast.Name):
                loop_vars[t.id] = "KNOWN_NOT_NONE"
        self._var_state_stack.append(loop_vars)
        self._in_loop_stack.append(True)
        self._visit_statement_sequence(node.body)
        self._in_loop_stack.pop()
        self._var_state_stack.pop()

        else_vars = dict(self._current_var_state)
        for s in node.body:
            for name in _extract_assigned_names_in_subtree(s):
                self._invalidate_var(else_vars, name)
        self._var_state_stack.append(else_vars)
        self._visit_statement_sequence(node.orelse)
        self._var_state_stack.pop()

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        tf, ff = _extract_branch_facts(node.test)
        loop_vars = dict(self._current_var_state)
        for var in tf.known_not_none_vars:
            loop_vars[var] = "KNOWN_NOT_NONE"
        for var in tf.definitely_none_vars:
            loop_vars[var] = "DEFINITELY_NONE"

        self._var_state_stack.append(loop_vars)
        self._in_loop_stack.append(True)
        self._visit_statement_sequence(node.body)
        self._in_loop_stack.pop()
        self._var_state_stack.pop()

        else_vars = dict(self._current_var_state)
        for s in node.body:
            for name in _extract_assigned_names_in_subtree(s):
                self._invalidate_var(else_vars, name)
        for var in ff.known_not_none_vars:
            else_vars[var] = "KNOWN_NOT_NONE"
        for var in ff.definitely_none_vars:
            else_vars[var] = "DEFINITELY_NONE"
        self._var_state_stack.append(else_vars)
        self._visit_statement_sequence(node.orelse)
        self._var_state_stack.pop()

    if hasattr(ast, "TryStar"):
        visit_TryStar = visit_Try

    def _extract_names_from_test(self, test: ast.AST) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(test):
            if isinstance(node, ast.Name):
                names.add(node.id)
        return names

    def _visit_statement_sequence(self, stmts: list[ast.stmt]) -> dict[str, str]:
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
                        if _is_none_literal(stmt.value) or self._call_returns_none(stmt.value, current_vars):
                            current_vars[target.id] = "DEFINITELY_NONE"
                        else:
                            current_vars[target.id] = "KNOWN_NOT_NONE"
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for elt in ast.walk(target):
                            if isinstance(elt, ast.Name):
                                current_vars.pop(elt.id, None)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.value and (_is_none_literal(stmt.value) or self._call_returns_none(stmt.value, current_vars)):
                    current_vars[stmt.target.id] = "DEFINITELY_NONE"
                elif stmt.value:
                    current_vars[stmt.target.id] = "KNOWN_NOT_NONE"
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                current_vars[stmt.target.id] = "KNOWN_NOT_NONE"
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                current_vars[stmt.name] = "KNOWN_NOT_NONE"

            # 1b. NamedExpr assignment tracking
            if not isinstance(stmt, (ast.If, ast.Try, ast.For, ast.AsyncFor, ast.While)):
                for child in _walk_enclosing_scope(stmt):
                    if isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
                        if _is_none_literal(child.value):
                            current_vars[child.target.id] = "DEFINITELY_NONE"
                        else:
                            current_vars[child.target.id] = "KNOWN_NOT_NONE"

            # 2. If guards with exit or assignment
            if isinstance(stmt, ast.If):
                body_exits = _block_always_exits(stmt.body, in_loop=self._in_loop)
                orelse_exits = bool(stmt.orelse) and _block_always_exits(stmt.orelse, in_loop=self._in_loop)
                branch_states = self._last_if_branches
                final_body, final_else = branch_states if branch_states else ({}, {})

                if body_exits and orelse_exits:
                    current_vars.clear()
                elif body_exits:
                    current_vars = dict(final_else)
                elif orelse_exits:
                    current_vars = dict(final_body)
                elif not stmt.orelse:
                    promoted_guards: set[str] = set()
                    for var in self._extract_names_from_test(stmt.test):
                        if (_is_test_checking_is_none(stmt.test, var) or _is_test_checking_falsy(stmt.test, var)) and _body_always_assigns_non_none(stmt.body, var):
                            current_vars[var] = "KNOWN_NOT_NONE"
                            promoted_guards.add(var)
                    for name in _extract_assigned_names_in_subtree(stmt):
                        if name not in promoted_guards:
                            self._invalidate_var(current_vars, name)
                else:
                    # Both branches reach convergence: merge facts that hold on every reachable path
                    merged_vars: dict[str, str] = {}
                    all_candidates = set(current_vars.keys()) | set(final_body.keys()) | set(final_else.keys())
                    for var in all_candidates:
                        state_b = final_body.get(var)
                        state_e = final_else.get(var)
                        if state_b == state_e and state_b is not None:
                            merged_vars[var] = state_b
                        else:
                            self._invalidate_var(merged_vars, var)
                    current_vars = merged_vars
            elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.Try)):
                for name in _extract_assigned_names_in_subtree(stmt):
                    self._invalidate_var(current_vars, name)

        return current_vars

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
        self._function_params_stack.append(params_state)
        self._var_state_stack.append(params_state)
        self._in_loop_stack.append(False)
        self._visit_statement_sequence(node.body)
        self._in_loop_stack.pop()
        self._var_state_stack.pop()
        self._function_params_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        params_state = self._extract_function_params(node)
        self._function_params_stack.append(params_state)
        self._var_state_stack.append(params_state)
        self._in_loop_stack.append(False)
        self._visit_statement_sequence(node.body)
        self._in_loop_stack.pop()
        self._var_state_stack.pop()
        self._function_params_stack.pop()

    def visit_Module(self, node: ast.Module) -> None:
        # Pre-pass: build conservative return summaries for top-level functions.
        # Definition-order independent: scan all top-level FunctionDefs before
        # visiting any statements so callers above or below the definition benefit.
        self._function_return_summaries = {
            stmt.name: "DEFINITELY_NONE"
            for stmt in node.body
            if isinstance(stmt, ast.FunctionDef)
            and _classify_function_returns_none(stmt)
        }
        self._var_state_stack.append({})
        self._visit_statement_sequence(node.body)
        self._var_state_stack.pop()

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            if _is_none_literal(node.value):
                self._current_var_state[node.target.id] = "DEFINITELY_NONE"
            else:
                self._current_var_state[node.target.id] = "KNOWN_NOT_NONE"
        self.visit(node.target)

    def _visit_comprehension(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> None:
        outer_vars = dict(self._current_var_state)
        comp_vars = dict(outer_vars)
        self._var_state_stack.append(comp_vars)

        for gen in node.generators:
            self.visit(gen.iter)
            for t in ast.walk(gen.target):
                if isinstance(t, ast.Name):
                    comp_vars.pop(t.id, None)
            for if_expr in gen.ifs:
                self.visit(if_expr)
                tf, _ = _extract_branch_facts(if_expr)
                for v in tf.known_not_none_vars:
                    comp_vars[v] = "KNOWN_NOT_NONE"
                for v in tf.definitely_none_vars:
                    comp_vars[v] = "DEFINITELY_NONE"

        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)

        self._var_state_stack.pop()

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if isinstance(node.op, ast.And):
            accumulated_tf = BranchFacts()
            orig_vars = dict(self._current_var_state)
            for val in node.values:
                step_vars = dict(orig_vars)
                for v in accumulated_tf.known_not_none_vars:
                    step_vars[v] = "KNOWN_NOT_NONE"
                for v in accumulated_tf.definitely_none_vars:
                    step_vars[v] = "DEFINITELY_NONE"
                self._var_state_stack.append(step_vars)
                self.visit(val)
                self._var_state_stack.pop()

                for name in _extract_all_modified_names(val):
                    orig_vars.pop(name, None)

                val_tf, _ = _extract_branch_facts(val)
                accumulated_tf = accumulated_tf.merge_conjunction(val_tf)

        elif isinstance(node.op, ast.Or):
            accumulated_ff = BranchFacts()
            orig_vars = dict(self._current_var_state)
            for val in node.values:
                step_vars = dict(orig_vars)
                for v in accumulated_ff.known_not_none_vars:
                    step_vars[v] = "KNOWN_NOT_NONE"
                for v in accumulated_ff.definitely_none_vars:
                    step_vars[v] = "DEFINITELY_NONE"
                self._var_state_stack.append(step_vars)
                self.visit(val)
                self._var_state_stack.pop()

                for name in _extract_all_modified_names(val):
                    orig_vars.pop(name, None)

                _, val_ff = _extract_branch_facts(val)
                accumulated_ff = accumulated_ff.merge_conjunction(val_ff)
        else:
            self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        tf, ff = _extract_branch_facts(node.test)

        body_vars = dict(self._current_var_state)
        for v in tf.known_not_none_vars:
            body_vars[v] = "KNOWN_NOT_NONE"
        for v in tf.definitely_none_vars:
            body_vars[v] = "DEFINITELY_NONE"
        self._var_state_stack.append(body_vars)
        self.visit(node.body)
        self._var_state_stack.pop()

        else_vars = dict(self._current_var_state)
        for v in ff.known_not_none_vars:
            else_vars[v] = "KNOWN_NOT_NONE"
        for v in ff.definitely_none_vars:
            else_vars[v] = "DEFINITELY_NONE"
        self._var_state_stack.append(else_vars)
        self.visit(node.orelse)
        self._var_state_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        # Visit condition expression first
        self.visit(node.test)
        tf, ff = _extract_branch_facts(node.test)

        # 1. Body branch
        body_vars = dict(self._current_var_state)
        for var in tf.known_not_none_vars:
            body_vars[var] = "KNOWN_NOT_NONE"
        for var in tf.definitely_none_vars:
            body_vars[var] = "DEFINITELY_NONE"

        self._var_state_stack.append(body_vars)
        final_body = self._visit_statement_sequence(node.body)
        self._var_state_stack.pop()

        # 2. Else branch
        else_vars = dict(self._current_var_state)
        for var in ff.known_not_none_vars:
            else_vars[var] = "KNOWN_NOT_NONE"
        for var in ff.definitely_none_vars:
            else_vars[var] = "DEFINITELY_NONE"

        self._var_state_stack.append(else_vars)
        final_else = self._visit_statement_sequence(node.orelse)
        self._var_state_stack.pop()

        self._last_if_branches = (final_body, final_else)

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

        # Case 1b: Built-in len() called on literal None or variable known to be None
        if isinstance(node.func, ast.Name) and node.func.id == "len" and node.args:
            first_arg = node.args[0]
            if _is_none_literal(first_arg):
                if not self._is_enclosed_in_type_error_handler():
                    self.issues.append(
                        Issue(
                            code="SEM002",
                            message="Calling len() on literal None will raise TypeError.",
                            file_path=self.file_path,
                            line=node.lineno,
                            column=node.col_offset,
                            severity="WARNING",
                            tool="semantic_analyzer",
                            fixable=False,
                            confidence="HIGH",
                            explanation=(
                                "Calling len(None) unconditionally raises TypeError: "
                                "object of type 'NoneType' has no len()."
                            ),
                            suggested_fix="Ensure the argument is a valid sequence or mapping before calling len().",
                        )
                    )
                return

            arg_name = _get_name_id(first_arg)
            if arg_name:
                arg_state = self._current_var_state.get(arg_name)
                if arg_state == "DEFINITELY_NONE":
                    if not self._is_enclosed_in_type_error_handler():
                        self.issues.append(
                            Issue(
                                code="SEM002",
                                message=(
                                    f"Calling len() on variable '{arg_name}' which is known to be None: "
                                    f"call will raise TypeError."
                                ),
                                file_path=self.file_path,
                                line=node.lineno,
                                column=node.col_offset,
                                severity="WARNING",
                                tool="semantic_analyzer",
                                fixable=False,
                                confidence="HIGH",
                                explanation=(
                                    f"Variable '{arg_name}' is proven to evaluate to None in this execution path. "
                                    f"Calling len({arg_name}) raises TypeError: object of type 'NoneType' has no len()."
                                ),
                                suggested_fix=f"Verify that '{arg_name}' is not None prior to calling len({arg_name}).",
                            )
                        )
                    return
                if arg_state == "MAYBE_NONE":
                    if not self._is_enclosed_in_type_error_handler():
                        self.issues.append(
                            Issue(
                                code="SEM002",
                                message=(
                                    f"Possible None dereference: parameter '{arg_name}' defaults to None "
                                    f"and is passed to len() without an effective guard."
                                ),
                                file_path=self.file_path,
                                line=node.lineno,
                                column=node.col_offset,
                                severity="WARNING",
                                tool="semantic_analyzer",
                                fixable=False,
                                confidence="MEDIUM",
                                explanation=(
                                    f"Parameter '{arg_name}' has a default value of None and is passed to len() "
                                    f"without an effective guard."
                                ),
                                suggested_fix=f"Verify that '{arg_name}' is not None prior to calling len({arg_name}).",
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
        self._in_loop_stack: list[bool] = [False]

    @property
    def _in_loop(self) -> bool:
        return self._in_loop_stack[-1]

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
        self._collection_stack.append({k: v.copy() for k, v in self._current_collections.items()})
        self._visit_statement_sequence(node.body)
        self._collection_stack.pop()
        self._try_handlers_stack.pop()

        for handler in node.handlers:
            self.visit(handler)

        self._collection_stack.append({k: v.copy() for k, v in self._current_collections.items()})
        self._visit_statement_sequence(node.orelse)
        self._collection_stack.pop()

        # 4. Visit finalbody
        finally_collections = {k: v.copy() for k, v in self._current_collections.items()}
        for stmt in node.body + [s for h in node.handlers for s in h.body] + node.orelse:
            for name in _extract_all_modified_names(stmt):
                finally_collections.pop(name, None)
        for handler in node.handlers:
            if handler.name:
                finally_collections.pop(handler.name, None)

        self._collection_stack.append(finally_collections)
        self._visit_statement_sequence(node.finalbody)
        self._collection_stack.pop()

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type:
            self.visit(node.type)
        handler_collections = {k: v.copy() for k, v in self._current_collections.items()}
        if node.name:
            handler_collections.pop(node.name, None)
        self._collection_stack.append(handler_collections)
        self._visit_statement_sequence(node.body)
        self._collection_stack.pop()

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.target)
        self.visit(node.iter)
        loop_collections = {k: v.copy() for k, v in self._current_collections.items()}
        for t in ast.walk(node.target):
            if isinstance(t, ast.Name):
                loop_collections.pop(t.id, None)
        self._in_loop_stack.append(True)
        self._collection_stack.append(loop_collections)
        self._visit_statement_sequence(node.body)
        self._collection_stack.pop()
        self._in_loop_stack.pop()

        else_collections = {k: v.copy() for k, v in self._current_collections.items()}
        for s in node.body:
            for name in _extract_all_modified_names(s):
                else_collections.pop(name, None)
        self._collection_stack.append(else_collections)
        self._visit_statement_sequence(node.orelse)
        self._collection_stack.pop()

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit(node.target)
        self.visit(node.iter)
        loop_collections = {k: v.copy() for k, v in self._current_collections.items()}
        for t in ast.walk(node.target):
            if isinstance(t, ast.Name):
                loop_collections.pop(t.id, None)
        self._in_loop_stack.append(True)
        self._collection_stack.append(loop_collections)
        self._visit_statement_sequence(node.body)
        self._collection_stack.pop()
        self._in_loop_stack.pop()

        else_collections = {k: v.copy() for k, v in self._current_collections.items()}
        for s in node.body:
            for name in _extract_all_modified_names(s):
                else_collections.pop(name, None)
        self._collection_stack.append(else_collections)
        self._visit_statement_sequence(node.orelse)
        self._collection_stack.pop()

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        tf, ff = _extract_branch_facts(node.test)
        loop_collections = {k: v.copy() for k, v in self._current_collections.items()}
        for var_name, key_val in tf.guarded_keys:
            if var_name in loop_collections:
                loop_collections[var_name].guarded_keys.add(key_val)

        self._in_loop_stack.append(True)
        self._collection_stack.append(loop_collections)
        self._visit_statement_sequence(node.body)
        self._collection_stack.pop()
        self._in_loop_stack.pop()

        else_collections = {k: v.copy() for k, v in self._current_collections.items()}
        for s in node.body:
            for name in _extract_all_modified_names(s):
                else_collections.pop(name, None)
        for var_name, key_val in ff.guarded_keys:
            if var_name in else_collections:
                else_collections[var_name].guarded_keys.add(key_val)
        self._collection_stack.append(else_collections)
        self._visit_statement_sequence(node.orelse)
        self._collection_stack.pop()

    if hasattr(ast, "TryStar"):
        visit_TryStar = visit_Try

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._in_loop_stack.append(False)
        self._collection_stack.append({})
        self._visit_statement_sequence(node.body)
        self._collection_stack.pop()
        self._in_loop_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._in_loop_stack.append(False)
        self._collection_stack.append({})
        self._visit_statement_sequence(node.body)
        self._collection_stack.pop()
        self._in_loop_stack.pop()

    def visit_Module(self, node: ast.Module) -> None:
        self._collection_stack.append({})
        self._visit_statement_sequence(node.body)
        self._collection_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)

        tf, ff = _extract_branch_facts(node.test)

        # 1. Body branch
        body_collections = {k: v.copy() for k, v in self._current_collections.items()}
        for var_name, key_val in tf.guarded_keys:
            if var_name in body_collections:
                body_collections[var_name].guarded_keys.add(key_val)

        self._collection_stack.append(body_collections)
        self._visit_statement_sequence(node.body)
        self._collection_stack.pop()

        # 2. Else branch
        else_collections = {k: v.copy() for k, v in self._current_collections.items()}
        for var_name, key_val in ff.guarded_keys:
            if var_name in else_collections:
                else_collections[var_name].guarded_keys.add(key_val)

        self._collection_stack.append(else_collections)
        self._visit_statement_sequence(node.orelse)
        self._collection_stack.pop()

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if isinstance(node.op, ast.And):
            accumulated_tf = BranchFacts()
            for value in node.values:
                saved_guarded: dict[str, set[Any]] = {
                    k: set(v.guarded_keys) for k, v in self._current_collections.items()
                }
                for var, key in accumulated_tf.guarded_keys:
                    if var in self._current_collections:
                        self._current_collections[var].guarded_keys.add(key)
                self.visit(value)
                for var, keys in saved_guarded.items():
                    if var in self._current_collections:
                        self._current_collections[var].guarded_keys = set(keys)

                val_tf, _ = _extract_branch_facts(value)
                accumulated_tf = accumulated_tf.merge_conjunction(val_tf)

        elif isinstance(node.op, ast.Or):
            accumulated_ff = BranchFacts()
            for value in node.values:
                saved_guarded = {
                    k: set(v.guarded_keys) for k, v in self._current_collections.items()
                }
                for var, key in accumulated_ff.guarded_keys:
                    if var in self._current_collections:
                        self._current_collections[var].guarded_keys.add(key)
                self.visit(value)
                for var, keys in saved_guarded.items():
                    if var in self._current_collections:
                        self._current_collections[var].guarded_keys = set(keys)

                _, val_ff = _extract_branch_facts(value)
                accumulated_ff = accumulated_ff.merge_conjunction(val_ff)
        else:
            self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        tf, ff = _extract_branch_facts(node.test)

        # Body under True facts
        saved_guarded = {
            k: set(v.guarded_keys) for k, v in self._current_collections.items()
        }
        for var, key in tf.guarded_keys:
            if var in self._current_collections:
                self._current_collections[var].guarded_keys.add(key)
        self.visit(node.body)
        for var, keys in saved_guarded.items():
            if var in self._current_collections:
                self._current_collections[var].guarded_keys = set(keys)

        # Orelse under False facts
        saved_guarded = {
            k: set(v.guarded_keys) for k, v in self._current_collections.items()
        }
        for var, key in ff.guarded_keys:
            if var in self._current_collections:
                self._current_collections[var].guarded_keys.add(key)
        self.visit(node.orelse)
        for var, keys in saved_guarded.items():
            if var in self._current_collections:
                self._current_collections[var].guarded_keys = set(keys)

        for m in _extract_assigned_names_in_subtree(node.body) | _extract_assigned_names_in_subtree(node.orelse):
            self._current_collections.pop(m, None)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        target_name = _get_name_id(node.target)
        if target_name:
            self._current_collections.pop(target_name, None)
            if isinstance(node.value, (ast.List, ast.Tuple)):
                length = _extract_sequence_length(node.value)
                if length is not None:
                    kind = "LIST" if isinstance(node.value, ast.List) else "TUPLE"
                    self._current_collections[target_name] = CollectionLiteralState(kind=kind, length=length)
            elif isinstance(node.value, ast.Dict):
                keys = _extract_dict_literal_keys(node.value)
                if keys is not None:
                    self._current_collections[target_name] = CollectionLiteralState(kind="DICT", known_keys=keys)
        self.visit(node.target)

    def _visit_comprehension(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> None:
        outer_collections = {k: v.copy() for k, v in self._current_collections.items()}
        comp_collections = {k: v.copy() for k, v in outer_collections.items()}
        self._collection_stack.append(comp_collections)

        for gen in node.generators:
            self.visit(gen.iter)
            for t in ast.walk(gen.target):
                if isinstance(t, ast.Name):
                    comp_collections.pop(t.id, None)
            for if_expr in gen.ifs:
                self.visit(if_expr)
                tf, _ = _extract_branch_facts(if_expr)
                for var_name, key_val in tf.guarded_keys:
                    if var_name in comp_collections:
                        comp_collections[var_name].guarded_keys.add(key_val)

        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)

        self._collection_stack.pop()

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            self._current_collections.pop(node.func.value.id, None)
        for arg in node.args:
            if isinstance(arg, ast.Name):
                self._current_collections.pop(arg.id, None)
        for kw in node.keywords:
            if isinstance(kw.value, ast.Name):
                self._current_collections.pop(kw.value.id, None)
        self.generic_visit(node)

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
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                current_collections.pop(stmt.name, None)

            # 2. Invalidate on method calls and call escapes
            call_scope_nodes: list[ast.AST] = []
            if isinstance(stmt, ast.If):
                body_exits = any(_statement_always_exits(s, in_loop=self._in_loop) for s in stmt.body)
                orelse_exits = bool(stmt.orelse) and any(_statement_always_exits(s, in_loop=self._in_loop) for s in stmt.orelse)
                call_scope_nodes.append(stmt.test)
                if not body_exits:
                    call_scope_nodes.extend(stmt.body)
                if not orelse_exits:
                    call_scope_nodes.extend(stmt.orelse)
            else:
                call_scope_nodes.append(stmt)

            for root_node in call_scope_nodes:
                for child in _walk_enclosing_scope(root_node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name):
                            current_collections.pop(child.func.value.id, None)
                        for arg in child.args:
                            if isinstance(arg, ast.Name):
                                current_collections.pop(arg.id, None)
                        for kw in child.keywords:
                            if isinstance(kw.value, ast.Name):
                                current_collections.pop(kw.value.id, None)

            # 2b. Invalidate/update from NamedExpr within stmt
            if not isinstance(stmt, (ast.If, ast.Try, ast.For, ast.AsyncFor, ast.While)):
                for child in _walk_enclosing_scope(stmt):
                    if isinstance(child, ast.NamedExpr):
                        for target_name in _extract_all_modified_names(child):
                            current_collections.pop(target_name, None)
                        if isinstance(child.target, ast.Name):
                            var_name = child.target.id
                            if isinstance(child.value, (ast.List, ast.Tuple)):
                                length = _extract_sequence_length(child.value)
                                if length is not None:
                                    kind = "LIST" if isinstance(child.value, ast.List) else "TUPLE"
                                    current_collections[var_name] = CollectionLiteralState(kind=kind, length=length)
                            elif isinstance(child.value, ast.Dict):
                                keys = _extract_dict_literal_keys(child.value)
                                if keys is not None:
                                    current_collections[var_name] = CollectionLiteralState(kind="DICT", known_keys=keys)

            # 3. Handle If statements and branch merging
            if isinstance(stmt, ast.If):
                tf, ff = _extract_branch_facts(stmt.test)
                body_exits = any(_statement_always_exits(s, in_loop=self._in_loop) for s in stmt.body)
                orelse_exits = bool(stmt.orelse) and any(_statement_always_exits(s, in_loop=self._in_loop) for s in stmt.orelse)

                if body_exits:
                    for var_name, key_val in ff.guarded_keys:
                        if var_name in current_collections:
                            current_collections[var_name].guarded_keys.add(key_val)
                    for s in stmt.orelse:
                        for m in _extract_all_modified_names(s):
                            current_collections.pop(m, None)
                elif orelse_exits:
                    for var_name, key_val in tf.guarded_keys:
                        if var_name in current_collections:
                            current_collections[var_name].guarded_keys.add(key_val)
                    for s in stmt.body:
                        for m in _extract_all_modified_names(s):
                            current_collections.pop(m, None)
                else:
                    modified = _extract_all_modified_names(stmt)
                    for m in modified:
                        current_collections.pop(m, None)
            elif isinstance(stmt, (ast.Try, ast.For, ast.AsyncFor, ast.While)):
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
