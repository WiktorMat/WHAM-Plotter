"""Cut/weight expression parser.

One grammar, two backends:

    parsed = parse("abs(eta_1) < 2.4 & (1 < idTau < 5)")
    parsed.columns          -> frozenset of column names used
    parsed.arrow()          -> pyarrow.dataset Expression (filter pushdown)
    parsed.evaluate(cols)   -> numpy array (bool mask or float values)

Supported grammar:
    comparisons   < <= > >= == !=      (chained "a < b < c" is normalized)
    logic         &  |  ~   (or the spellings and/or/not; both get sane
                             precedence: "pt > 26 & iso < 0.15" works)
    arithmetic    +  -  *  /
    functions     abs(...)
    operands      column names, int/float/bool literals, parentheses
"""

from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


class ExprError(ValueError):
    """Invalid or unsupported expression, with the offending source segment."""


_ALLOWED_CMP = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
_ALLOWED_BIN = (ast.BitAnd, ast.BitOr, ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARY = (ast.Invert, ast.USub)


def _err(source: str, node: ast.AST, message: str) -> ExprError:
    segment = ast.get_source_segment(source, node) or "<expr>"
    col = getattr(node, "col_offset", -1)
    return ExprError(f"{message}: '{segment}' (column {col}) in: {source!r}")


_TOKEN_ALIASES = {"&": "and", "|": "or", "~": "not"}


def _preprocess(source: str) -> str:
    """Rewrite & | ~ into and/or/not so comparisons bind tighter.

    Python gives & higher precedence than <, so "a < 1 & b > 2" would parse
    as "a < (1 & b) > 2". The keyword forms have the precedence users expect;
    _BoolRewriter converts them back to bitwise ops for numpy/arrow.
    """
    source = " ".join(source.split())  # single line, normalized whitespace
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError as e:
        raise ExprError(f"syntax error in {source!r}: {e}") from None

    out = source
    for tok in reversed(tokens):
        if tok.type == tokenize.OP and tok.string in _TOKEN_ALIASES:
            col = tok.start[1]
            out = f"{out[:col]} {_TOKEN_ALIASES[tok.string]} {out[col + 1:]}"
    return out.strip()


class _BoolRewriter(ast.NodeTransformer):
    """Turn BoolOp(and/or) and Not back into & | ~ semantics for arrays."""

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        op: ast.operator = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        out: ast.expr = node.values[0]
        for value in node.values[1:]:
            out = ast.BinOp(left=out, op=op, right=value)
        return ast.copy_location(out, node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.copy_location(
                ast.UnaryOp(op=ast.Invert(), operand=node.operand), node
            )
        return node


class _ChainSplitter(ast.NodeTransformer):
    """Rewrite 'a < b < c' into '(a < b) & (b < c)'.

    Required because chained comparisons don't work on numpy arrays and
    have no pyarrow equivalent.
    """

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) == 1:
            return node
        operands = [node.left, *node.comparators]
        parts = [
            ast.Compare(left=operands[i], ops=[node.ops[i]], comparators=[operands[i + 1]])
            for i in range(len(node.ops))
        ]
        out: ast.expr = parts[0]
        for part in parts[1:]:
            out = ast.BinOp(left=out, op=ast.BitAnd(), right=part)
        return ast.copy_location(out, node)


def _validate(source: str, node: ast.AST, columns: set[str]) -> None:
    if isinstance(node, ast.Expression):
        _validate(source, node.body, columns)
    elif isinstance(node, ast.Compare):
        # single op guaranteed after _ChainSplitter
        if not isinstance(node.ops[0], _ALLOWED_CMP):
            raise _err(source, node, "unsupported comparison operator")
        _validate(source, node.left, columns)
        _validate(source, node.comparators[0], columns)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BIN):
            raise _err(source, node, "unsupported operator")
        _validate(source, node.left, columns)
        _validate(source, node.right, columns)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY):
            raise _err(source, node, "unsupported unary operator")
        _validate(source, node.operand, columns)
    elif isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id == "abs"):
            raise _err(source, node, "only abs() is allowed")
        if len(node.args) != 1 or node.keywords:
            raise _err(source, node, "abs() takes exactly one argument")
        _validate(source, node.args[0], columns)
    elif isinstance(node, ast.Name):
        columns.add(node.id)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, bool)):
            raise _err(source, node, "only numeric literals are allowed")
    else:
        raise _err(source, node, f"unsupported syntax ({type(node).__name__})")


def _comparison_operand(node: ast.AST, schema):
    """Use numeric 0/1 for boolean comparison operands, as NumPy does."""
    import pyarrow as pa

    value = _to_arrow(node, schema)
    if schema is None:
        return value
    is_bool = isinstance(node, ast.Constant) and isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        index = schema.get_field_index(node.id)
        is_bool = index >= 0 and pa.types.is_boolean(schema.field(index).type)
    if is_bool:
        return value.cast(pa.int8())
    return value


def _to_arrow(node: ast.AST, schema=None):
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    if isinstance(node, ast.Compare):
        left = _comparison_operand(node.left, schema)
        right = _comparison_operand(node.comparators[0], schema)
        op = node.ops[0]
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.Eq):
            return left == right
        return left != right
    if isinstance(node, ast.BinOp):
        left = _to_arrow(node.left, schema)
        right = _to_arrow(node.right, schema)
        op = node.op
        if isinstance(op, ast.BitAnd):
            return left & right
        if isinstance(op, ast.BitOr):
            return left | right
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        return left / right
    if isinstance(node, ast.UnaryOp):
        operand = _to_arrow(node.operand, schema)
        if isinstance(node.op, ast.Invert):
            return ~operand
        return ds.scalar(0) - operand  # USub; Expression has no __neg__
    if isinstance(node, ast.Call):  # validated: abs(x)
        return pc.abs(_to_arrow(node.args[0], schema))
    if isinstance(node, ast.Name):
        return ds.field(node.id)
    if isinstance(node, ast.Constant):
        return ds.scalar(node.value)
    raise AssertionError(f"unvalidated node reached arrow backend: {node!r}")


@dataclass(frozen=True)
class Expr:
    source: str
    tree: ast.Expression
    columns: frozenset[str]

    def arrow(self, schema=None):
        """Dataset filter; pass the input schema for boolean/numeric comparisons."""
        return _to_arrow(self.tree.body, schema)

    def evaluate(self, cols: Mapping[str, np.ndarray]) -> np.ndarray:
        """Evaluate against numpy columns; bool mask for cuts, floats for weights."""
        missing = self.columns - set(cols)
        if missing:
            raise ExprError(f"missing columns {sorted(missing)} for {self.source!r}")
        code = compile(self.tree, "<expr>", "eval")
        # Safe: the AST was whitelisted in parse(); no attribute access,
        # no calls except abs, no names resolved outside the column dict.
        return eval(code, {"__builtins__": {}, "abs": np.abs}, dict(cols))


def conjuncts(expr: Expr) -> list[ast.expr]:
    """The top-level AND atoms of an expression (an OR-group is one atom)."""
    out: list[ast.expr] = []

    def walk(node: ast.expr) -> None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            walk(node.left)
            walk(node.right)
        else:
            out.append(node)

    walk(expr.tree.body)
    return out


def _atom_columns(atom: ast.expr) -> set[str]:
    cols: set[str] = set()
    _validate("<atom>", atom, cols)
    return cols


def _const_value(node: ast.expr) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))):
        return -float(node.operand.value)
    return None


def _scaled_operand(node: ast.expr) -> str | None:
    """Column of a scale-covariant comparison side: a bare column or abs(col)
    (abs(s*x) == s*abs(x) for the positive factors used here)."""
    if isinstance(node, ast.Name):
        return node.id
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "abs" and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)):
        return node.args[0].id
    return None


_MIRROR_CMP = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE}


def _widen_atom(atom: ast.expr, bounds: Mapping[str, tuple[float, float]]) -> ast.expr | None:
    """A relaxed copy of `column <op> constant` such that the cut passes for
    every scaling of the column within its bounds, or None if the atom's
    shape can't be relaxed (it must then be re-applied in memory)."""
    if not isinstance(atom, ast.Compare):
        return None
    left, op, right = atom.left, atom.ops[0], atom.comparators[0]
    operand, col, const = left, _scaled_operand(left), _const_value(right)
    if col is None or const is None:  # try the mirrored orientation
        operand, col, const = right, _scaled_operand(right), _const_value(left)
        mirrored = _MIRROR_CMP.get(type(op))
        if col is None or const is None or mirrored is None:
            return None
        op = mirrored()
    smin, smax = bounds[col]
    if smin <= 0:
        return None
    if isinstance(op, (ast.Gt, ast.GtE)):  # s*x > c <=> x > c/s; loosest bound
        relaxed = const / (smax if const >= 0 else smin)
    elif isinstance(op, (ast.Lt, ast.LtE)):
        relaxed = const / (smin if const >= 0 else smax)
    else:  # ==/!= have no useful relaxation
        return None
    # reuse the operand node so an abs() wrapper survives the rewrite
    return ast.Compare(left=operand, ops=[op],
                       comparators=[ast.Constant(value=relaxed)])


def column_bounds(expr: Expr, column: str) -> tuple[float | None, float | None]:
    """(lo, hi) window a conjunction of simple comparisons puts on a column
    (e.g. the pt window of a category cut); None for an unbounded side."""
    lo: float | None = None
    hi: float | None = None
    for atom in conjuncts(expr):
        if not isinstance(atom, ast.Compare):
            continue
        left, op, right = atom.left, atom.ops[0], atom.comparators[0]
        col, const = _scaled_operand(left), _const_value(right)
        if col is None or const is None:
            col, const = _scaled_operand(right), _const_value(left)
            mirrored = _MIRROR_CMP.get(type(op))
            if col is None or const is None or mirrored is None:
                continue
            op = mirrored()
        if col != column:
            continue
        if isinstance(op, (ast.Gt, ast.GtE)):
            lo = const if lo is None else max(lo, const)
        elif isinstance(op, (ast.Lt, ast.LtE)):
            hi = const if hi is None else min(hi, const)
    return lo, hi


def widened_arrow(expr: Expr, bounds: Mapping[str, tuple[float, float]], schema=None):
    """A read-filter superset of the expression under column scaling.

    `bounds` maps column -> (smin, smax), the extreme constant factors the
    column may be multiplied by before the cut is evaluated (include 1.0 for
    the nominal evaluation). Atoms not touching a scaled column pass through
    exactly; simple comparisons on a scaled column are relaxed so every
    scaled evaluation still reads its events; anything else on a scaled
    column is dropped. The exact cut MUST be re-applied in memory. Returns a
    pyarrow dataset Expression, or None when every atom was dropped."""
    kept = []
    for atom in conjuncts(expr):
        if not (_atom_columns(atom) & set(bounds)):
            kept.append(atom)
            continue
        widened = _widen_atom(atom, bounds)
        if widened is not None:
            kept.append(widened)
    if not kept:
        return None
    out = _to_arrow(kept[0], schema)
    for atom in kept[1:]:
        out = out & _to_arrow(atom, schema)
    return out


@lru_cache(maxsize=256)
def parse(source: str) -> Expr:
    source = source.strip()
    if not source:
        raise ExprError("empty expression")
    rewritten = _preprocess(source)
    try:
        tree = ast.parse(rewritten, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"syntax error in {source!r}: {e.msg} (column {e.offset})") from None

    tree = _BoolRewriter().visit(tree)
    tree = ast.fix_missing_locations(_ChainSplitter().visit(tree))
    columns: set[str] = set()
    _validate(rewritten, tree, columns)
    return Expr(source=source, tree=tree, columns=frozenset(columns))
