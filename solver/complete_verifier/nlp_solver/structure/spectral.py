"""Normalized symmetric eigenblock certificate: exact sparse detection, verified Gram proof.

Detection is one O(nnz) pass over the exact RowPool rows. The dense numerical setup (order <= 512) runs once on the
model device. No numerical output is trusted: for ANY finite L, L L^T is PSD, so with T = A - sI and an outward bound
rho >= ||T - L L^T||_inf, symmetry gives A - (s - rho) I >= L L^T >= 0. The residual bound also covers the input
conversion, the shift subtraction and the FP64 dot products, independently of the Cholesky accuracy.
theory/tgt_misc/tgt_misc.md item 5.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math

import numpy as np
import torch

from .cert_kernels import down, up, tree
from .certificates import Certificate, NoPattern, require, scaled_row
from .rowpool import round_down, round_up

# A work limit, not a numerical tolerance: larger blocks decline until a sparse verified factorization exists.
MAX_DENSE_ORDER = 512


@dataclass
class EigenBlock:
    variable: int
    coordinates: tuple
    scales: tuple
    entries: dict                 # exact rational symmetric sparse A in y = S q
    norm_lower: F
    norm_upper: F
    residual_squared: F           # sum of squared component band radii
    objective_scale: F
    objective_offset: F


def rational_sqrt(value):
    value = F(value)
    if value <= 0:
        return None
    n, d = math.isqrt(value.numerator), math.isqrt(value.denominator)
    return F(n, d) if n * n == value.numerator and d * d == value.denominator else None


def detect_eigenblock(g):
    """Recognize the objective a*lambda + b (a > 0), a norm row sum (s_i q_i)^2 in [l, u] with l > 0, and one
    homogeneous row lambda*q_i - sum_j A_ij q_j per coordinate with two finite sides. Rational coordinate scales and
    per-row scales and bands are allowed; symmetry is an EXACT rational equality."""
    terms = [(m, F(c)) for m, c in (g.obj or {}).items() if m]
    require(len(terms) == 1 and len(terms[0][0]) == 1, 'objective is not a positive affine eigenvalue variable')
    ((lam, exponent),), coefficient = terms[0]
    require(lam < g.n and exponent == 1 and coefficient > 0, 'objective is not a positive affine eigenvalue variable')
    equations, normals = {}, []
    for row in g.pool.rows:
        g.check()
        bilinear, linear, squares, valid = [], {}, {}, True
        for m, c in row.terms:
            if len(m) == 1 and m[0][1] == 1 and m[0][0] < g.n:
                linear[m[0][0]] = F(c)
            elif len(m) == 1 and m[0][1] == 2 and m[0][0] < g.n:
                squares[m[0][0]] = F(c)
            elif len(m) == 2 and all(e == 1 and v < g.n for v, e in m) and any(v == lam for v, _ in m):
                bilinear.append((next(v for v, _ in m if v != lam), F(c)))
            else:
                valid = False
        if not valid:
            continue
        if squares and not linear and not bilinear and lam not in squares:
            normals.append((row, squares))
        elif len(bilinear) == 1 and not squares and lam not in linear:
            i, c = bilinear[0]
            if i not in equations and row.ex_lo is not None and row.ex_hi is not None:
                equations[i] = (row, c, linear)
    require(equations, 'no homogeneous eigen rows with two finite sides')
    chosen = None
    for row, squares in normals:
        if squares.keys() != equations.keys():
            continue
        scale = next(iter(squares.values()))
        _, lo, hi = scaled_row(row, scale)
        if lo is None or hi is None or lo <= 0:
            continue
        scales = {v: rational_sqrt(c / scale) for v, c in squares.items()}
        if all(s is not None for s in scales.values()):
            chosen = (lo, hi, scales)
            break
    require(chosen is not None, 'missing positive normalization with rational q scaling')
    lo, hi, scales = chosen
    coordinates = tuple(scales)
    index = {v: i for i, v in enumerate(coordinates)}
    entries, error2 = {}, F(0)
    for v, (row, c, linear) in equations.items():
        g.check()
        _, rl, ru = scaled_row(row, c)
        radius = max(abs(rl), abs(ru)) * scales[v]
        error2 += radius * radius
        for w, a in linear.items():
            require(w in index, 'eigen row has linear terms outside the normalized block')
            entries[index[v], index[w]] = -a / c * scales[v] / scales[w]
    require(all(entries.get((j, i), F(0)) == a for (i, j), a in entries.items()),
            'eigen matrix is asymmetric (exact coefficient check)')
    return EigenBlock(lam, coordinates, tuple(scales.values()), entries, lo, hi,
                      error2, coefficient, F(g.obj.get((), 0)))


def gram_error(lower, upper, shift, factor):
    """Outward infinity-norm bound on [lower, upper] - shift*I - factor factor^T (the last two dimensions batch).

    IEEE FP64 dot bound gamma_(2n) = 2nu/(1-2nu), u = 2^-53: the nonnegative computed product of |L||L|^T encloses the
    exact one by (P + eta)/(1 - gamma), where eta = 2n*tiny/(1-2nu) covers underflow in all 2n dot operations
    (flush-to-zero GEMM included). Every later basic operation and pairwise row sum rounds outward."""
    n = lower.shape[-1]
    gamma = round_up(F(2 * n, 2 ** 53 - 2 * n))
    underflow = round_up(F(2 * n) * F(torch.finfo(torch.float64).tiny) / (1 - F(2 * n, 2 ** 53)))
    product = factor @ factor.transpose(-1, -2)
    absolute = factor.abs() @ factor.abs().transpose(-1, -2)
    magnitude = up(up(absolute + underflow) / down(lower.new_tensor(1.) - gamma))
    error = up(up(gamma * magnitude) + underflow)
    identity = torch.eye(n, dtype=torch.float64, device=lower.device)
    a = down(lower - shift[..., None, None] * identity)
    b = up(upper - shift[..., None, None] * identity)
    residual = torch.maximum(down(a - product).abs(), up(b - product).abs())
    return tree(up(residual + error), up).amax(-1)


def verified_shift(block, device, check=lambda: None):
    """(mu, estimate, rho, eigenvector): a dense eigensystem proposal and a Cholesky factor of A - sI on `device`,
    certified by the Gram residual: lambda_min(A) >= mu = s - rho. O(n^3) work, once. A failed proposal never
    publishes a bound; the margin and its retries are multiples of the verification dot-error bound, not fitted."""
    n = len(block.coordinates)
    require(n <= MAX_DENSE_ORDER, f'eigen block exceeds dense setup work limit {MAX_DENSE_ORDER}')
    check()
    lower = torch.zeros(n, n, dtype=torch.float64, device=device)
    upper = torch.zeros_like(lower)
    ij = list(block.entries)
    if ij:
        ids = torch.tensor(ij, dtype=torch.long, device=device).T
        lower[ids[0], ids[1]] = torch.tensor([round_down(block.entries[p]) for p in ij], dtype=torch.float64,
                                             device=device)
        upper[ids[0], ids[1]] = torch.tensor([round_up(block.entries[p]) for p in ij], dtype=torch.float64,
                                             device=device)
    require(bool(torch.isfinite(lower).all() & torch.isfinite(upper).all()), 'nonfinite converted eigen matrix')
    try:
        values, vectors = torch.linalg.eigh(lower)      # either symmetric endpoint is a legitimate seed
    except torch.linalg.LinAlgError as exc:
        raise NoPattern('eigensystem proposal failed') from exc
    check()
    estimate = values[0]
    gamma = round_up(F(2 * n, 2 ** 53 - 2 * n))
    magnitude = tree(up(torch.maximum(lower.abs(), upper.abs())), up).amax()
    margin = up(up(gamma * magnitude) + n * torch.finfo(torch.float64).tiny)
    for _ in range(8):
        check()
        shift = down(estimate - margin)
        trial = lower.clone()
        trial.diagonal().sub_(shift)
        factor, info = torch.linalg.cholesky_ex(trial, check_errors=False)
        if int(info) == 0:
            break
        margin = up(2 * margin)
    require(int(info) == 0, 'Cholesky proposal failed within bounded setup work')
    rho = gram_error(lower, upper, shift, factor)
    mu = float(down(shift - rho))
    require(math.isfinite(mu), 'nonfinite spectral residual proof')
    return mu, float(estimate), float(rho), vectors[:, 0].detach().cpu().double().numpy()


def spectral(g):
    """D = b + a*(mu - sqrt(sum tau_i^2 / norm_lower)): every feasible point has lambda q = A q + r with
    |r_i| <= tau_i, so lambda >= q'Aq/|q|^2 - |r|/|q| >= mu - |r|/|q|. Candidate: the proposed eigenvector scaled to
    the middle of the norm band, with lambda at the estimate."""
    block = detect_eigenblock(g)
    mu, estimate, rho, vector = verified_shift(block, g.device, g.check)
    correction = g.iv.sqrt(g.num(block.residual_squared) / g.num(block.norm_lower))
    D = g.down(g.num(block.objective_offset) + g.num(block.objective_scale) * (g.num(mu) - correction))
    candidate = (g.lo + g.hi) / 2
    radius = math.sqrt(float((block.norm_lower + block.norm_upper) / 2))
    vector = vector * radius / np.linalg.norm(vector)
    candidate[list(block.coordinates)] = vector / np.asarray(block.scales, dtype=float)
    candidate[block.variable] = estimate
    return Certificate('normalized-eigenblock', D,
                       dict(order=len(block.coordinates), nnz=len(block.entries), mu=mu, estimate=estimate,
                            residual_bound=rho, correction=g.up(correction)), candidate=candidate)
