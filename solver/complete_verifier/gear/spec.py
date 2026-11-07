"""Sparse output spec and the strict parser for the converter's vnnlib subset (G5 WP3, D2 rule L3).

The legacy artifacts carry their property in a ``.vnnlib`` file written by the converter. Only these
statements occur, one per line (comments start with ``;``):

    (declare-const X_i Real)         (declare-const Y_i Real)
    (assert (<= X_i c))              (assert (>= X_i c))
    (assert (<= Y_i c))              (assert (>= Y_i c))

``c`` is a finite decimal literal. Anything else (disjunctions, Y-to-Y rows, ``<``, multi-line
statements, a gap in the declared indices, a missing or empty input bound, ...) raises
:class:`LegacyArtifactError`: this parser never guesses and never falls back to the general reader.

The values equal what ``read_vnnlib.read_vnnlib`` produces for the same file, bit for bit, including
its mean/half-width box rewrite and the ``-1 * c`` right-hand side of a ``>=`` row. Unlike that reader it
holds the rows sparse (one entry per row) and never writes a ``.compiled`` cache.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .errors import LegacyArtifactError  # (re-exported: gear.spec.LegacyArtifactError is gear.errors's)


@dataclass(frozen=True)
class RowSpec:
    """Output rows ``C y <= rhs`` in CSR form: ``C`` is ``[nrows, num_outputs]``."""

    num_outputs: int
    indptr: np.ndarray   # int64 [nrows + 1]
    indices: np.ndarray  # int64 [nnz]
    data: np.ndarray     # float64 [nnz]
    rhs: np.ndarray      # float64 [nrows]

    @property
    def nrows(self) -> int:
        return int(self.rhs.shape[0])

    @property
    def nnz(self) -> int:
        return int(self.data.shape[0])

    def to_dense(self) -> np.ndarray:
        """The dense float64 ``C``; allocated once, this is the boundary where sparsity ends."""
        dense = np.zeros((self.nrows, self.num_outputs), dtype=np.float64)
        rows = np.repeat(np.arange(self.nrows, dtype=np.int64), np.diff(self.indptr))
        dense[rows, self.indices] = self.data
        return dense


_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_DECL = re.compile(r"\(declare-const ([XY])_(0|[1-9]\d*) Real\)")
_ASSERT = re.compile(r"\(assert \((<=|>=) ([XY])_(0|[1-9]\d*) (" + _NUM + r")\)\)")


def _finite(token: str, where: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise LegacyArtifactError(f"{where}: non-finite constant {token!r}")
    return value


def parse_vnnlib_subset(text: str, name: str = "<vnnlib>") -> tuple[np.ndarray, np.ndarray, RowSpec]:
    """Return ``(lower, upper, rowspec)`` for the converter subset; raise on anything else.

    ``lower``/``upper`` are float64 ``[n_inputs]`` and already carry the rewrite that ``read_vnnlib``
    applies (``mean -/+ half_width * 1.0``), which is not always the bound written in the file.
    """
    n_in_decl: set[int] = set()
    n_out_decl: set[int] = set()
    asserts: list[tuple[int, str, str, int, str]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        m = _DECL.fullmatch(line)
        if m:
            index = int(m.group(2))
            seen = n_in_decl if m.group(1) == "X" else n_out_decl
            if index in seen:
                raise LegacyArtifactError(f"{name}:{lineno}: duplicate declaration {m.group(1)}_{index}")
            seen.add(index)
            continue
        m = _ASSERT.fullmatch(line)
        if m:
            asserts.append((lineno, m.group(1), m.group(2), int(m.group(3)), m.group(4)))
            continue
        raise LegacyArtifactError(f"{name}:{lineno}: statement outside the converter subset: {line[:80]!r}")

    n_in, n_out = len(n_in_decl), len(n_out_decl)
    if n_in == 0:
        raise LegacyArtifactError(f"{name}: no input variable declared")
    if n_in_decl != set(range(n_in)) or n_out_decl != set(range(n_out)):
        raise LegacyArtifactError(f"{name}: declared indices are not contiguous from 0")

    lo = np.full(n_in, -np.inf)
    hi = np.full(n_in, np.inf)
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    for lineno, op, kind, index, token in asserts:
        where = f"{name}:{lineno}"
        value = _finite(token, where)
        if kind == "X":
            if index >= n_in:
                raise LegacyArtifactError(f"{where}: X_{index} is not declared")
            if op == "<=":
                hi[index] = min(value, hi[index])
            else:
                lo[index] = max(value, lo[index])
        else:
            if index >= n_out:
                raise LegacyArtifactError(f"{where}: Y_{index} is not declared")
            cols.append(index)
            if op == "<=":
                data.append(1.0)
                rhs.append(value)
            else:  # (>= Y c) is read as (<= c Y): -Y <= -c, with read_vnnlib's `-1 * c`
                data.append(-1.0)
                rhs.append(-1 * value)
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise LegacyArtifactError(f"{name}: an input variable lacks a bound")
    if not np.all(lo <= hi):
        raise LegacyArtifactError(f"{name}: an input range is empty")
    mean = (lo + hi) / 2
    half = (hi - lo) / 2 * 1.0
    m = len(cols)
    rowspec = RowSpec(
        num_outputs=n_out,
        indptr=np.arange(m + 1, dtype=np.int64),
        indices=np.asarray(cols, dtype=np.int64),
        data=np.asarray(data, dtype=np.float64),
        rhs=np.asarray(rhs, dtype=np.float64),
    )
    return mean - half, mean + half, rowspec
