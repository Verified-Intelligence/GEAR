"""Outward rounding margins for floating-point affine concretization."""
import torch

from .patches import Patches
from .utils import eyeC


def finite_abs(value):
    """Magnitude of finite terms only; infinities already give vacuous bounds."""
    return torch.where(torch.isfinite(value), value.abs(), 0)


# Largest |A| temporary, in elements, that abs_matvec materializes at once (1 GiB in
# float64); a dense A with more rows than fit is multiplied in blocks of rows.
DENSE_BLOCK_ELEMS = 1 << 27


def abs_matvec(A, vector, finite=True):
    """|A| @ vector for a dense A (..., N, X) and a vector (..., X, 1), N rows at a time.

    |A| is a full copy of A, and finite_abs adds two more (mask and values): on
    a (2, 22780, 6613) constraint matrix those were 8.5 GiB at once. A matrix within
    DENSE_BLOCK_ELEMS takes the single product; the rows of larger ones are independent
    dot products, so blocks give the same result. finite=False is a plain abs."""
    absolute = finite_abs if finite else torch.abs
    rows = A.shape[-2]
    block = max(1, DENSE_BLOCK_ELEMS * rows // max(A.numel(), 1))
    if rows <= block:
        return absolute(A).matmul(vector)
    return torch.cat([absolute(A.narrow(-2, start, min(block, rows - start))).matmul(vector)
                      for start in range(0, rows, block)], dim=-2)


def rounding_error(magnitude, n, bias=0):
    r"""Enclose rounding in a dot product and its bias accumulation.

    fl(sum a_i x_i + b) differs from the exact value by at most
    gamma_n * (sum |a_i x_i| + |b|), where gamma_n = n u / (1 - n u).
    We use (n + 16) * torch.finfo(dtype).eps * 4: the factor 4 and +16
    leave a safety margin for the reductions' association order and bias
    accumulation. The error stays in the bound's dtype (2**-52 for float64,
    2**-23 for float32). Non-finite operands are excluded from the magnitude.
    """
    if isinstance(bias, torch.Tensor):
        magnitude = magnitude + finite_abs(bias)
    else:
        magnitude = magnitude + abs(bias)
    return magnitude * ((n + 16) * torch.finfo(magnitude.dtype).eps * 4)


def widen_bound(bound, error, sign):
    """Widen endpoints; an infinite rounding error gives a vacuous bound."""
    widened = torch.where(torch.isfinite(bound), bound + sign * error, bound)
    # Finite operands can overflow both the dot product and its magnitude.
    # An infinite error must give a vacuous endpoint, including for +inf
    # lower bounds or -inf upper bounds produced by that overflow.
    return torch.where(torch.isposinf(error), sign * torch.inf, widened)


def box_rounding_error(A, x_L, x_U, bias=0, input_shape=None, abs_A=None):
    """One absolute dot product over the largest box endpoint magnitudes.

    abs_A: finite_abs(A) when the caller already has it (dense A only)."""
    max_abs = finite_abs(torch.maximum(x_L.abs(), x_U.abs()))
    if isinstance(A, eyeC):
        magnitude = max_abs.flatten(1)  # flatten, not reshape(-1): a batch may be empty
        n = 1
    elif isinstance(A, Patches):
        if A.identity:
            magnitude, n = max_abs, 1
        else:
            patches = A.create_similar(patches=finite_abs(A.patches))
            magnitude = patches.matmul(max_abs, input_shape=input_shape)
            n = A.patches.shape[-3] * A.patches.shape[-2] * A.patches.shape[-1]
    else:
        max_abs = max_abs.flatten(1).unsqueeze(-1)
        if abs_A is None:
            magnitude = abs_matvec(A, max_abs).squeeze(-1)
        else:
            magnitude = abs_A.matmul(max_abs).squeeze(-1)
        n = A.shape[-1]
    return rounding_error(magnitude, n, bias)


def coefficient_terms(A):
    """Number of input elements reduced by one coefficient row."""
    if isinstance(A, eyeC) or (isinstance(A, Patches) and A.identity):
        return 1
    if isinstance(A, Patches):
        return A.patches.shape[-3] * A.patches.shape[-2] * A.patches.shape[-1]
    return A[0, 0].numel()


def add_bound(first, second, sign):
    """Enclose one rounded addition of enclosed bounds: its error is at most
    u * |result|, and an exact cancellation is exact."""
    result = first + second
    return widen_bound(result, rounding_error(finite_abs(result), 1), sign)
