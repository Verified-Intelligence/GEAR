"""Numerical reference checks limited to the affine rounding enclosure."""

import torch

from auto_LiRPA.fp_enclosure import box_rounding_error, finite_abs, rounding_error
from auto_LiRPA.operators import BoundParams
from auto_LiRPA.perturbations import PerturbationLinear, PerturbationLpNorm


def assert_enclosed(actual, exact, margin, sign):
    """Require the endpoint to move outward, by at most the given margin.

    Equal infinities are checked exactly, so finite references cannot silently
    become vacuous. No relative tolerance or unrelated absolute floor is used.
    """
    exact = torch.as_tensor(exact, dtype=actual.dtype, device=actual.device)
    if exact.ndim:
        assert actual.shape == exact.shape
    exact = exact.expand_as(actual)
    assert not torch.isnan(actual).any()
    finite = torch.isfinite(exact)
    assert torch.equal(actual[~finite], exact[~finite])
    difference = sign * (actual[finite] - exact[finite])
    assert (difference >= 0).all(), (actual, exact, difference)
    assert (difference <= margin.expand_as(actual)[finite]).all(), (
        actual, exact, difference, margin)


def entry_margins(entry, lower, upper):
    """Allow at most 10x the margin of each returned affine form."""
    return tuple(10 * box_rounding_error(
        entry[side + 'A'].flatten(2), lower, upper, entry[side + 'bias'])
        for side in ('l', 'u'))


def backward_margins(model, bounds):
    """Use the actual root coefficients, boxes, and accumulated affine bias."""
    margins = []
    for side, bound, sign in zip(('l', 'u'), bounds, (-1, 1)):
        error = torch.zeros_like(bound)
        contribution = torch.zeros_like(bound)
        n = 0
        for root in model.roots():
            A = getattr(root, side + 'A', None)
            if A is None:
                continue
            A = A.reshape(bound.shape[1], bound.shape[0], -1).transpose(0, 1)
            center = root.center
            if isinstance(root, BoundParams) or root.batch_dim == -1:
                center = center.reshape(1, -1)
            perturbation = getattr(root, 'perturbation', None)
            if isinstance(perturbation, PerturbationLpNorm):
                lower, upper, _, _ = perturbation.get_input_bounds(center, A)
                error = error + box_rounding_error(A, lower, upper)
            elif isinstance(perturbation, PerturbationLinear):
                pos, neg = A.clamp(min=0), A.clamp(max=0)
                lo_A, hi_A = perturbation.lower_A, perturbation.upper_A
                lo_b, hi_b = perturbation.lower_b, perturbation.upper_b
                composite_A = pos @ (lo_A if sign == -1 else hi_A)
                composite_A = composite_A + neg @ (hi_A if sign == -1 else lo_A)
                composite_b = pos @ (lo_b if sign == -1 else hi_b)
                composite_b = composite_b + neg @ (hi_b if sign == -1 else lo_b)
                error = error + box_rounding_error(
                    composite_A, perturbation.input_lb.squeeze(-1),
                    perturbation.input_ub.squeeze(-1), composite_b.squeeze(-1))
            else:
                error = error + box_rounding_error(A, center, center)
            n += A.shape[-1]
            if perturbation is None:
                term = A.matmul(center.reshape(center.shape[0], -1, 1)).squeeze(-1)
            else:
                term = perturbation.concretize(center, A, sign=sign)
            contribution = contribution + term
        bias = bound - contribution
        margins.append(10 * (error + rounding_error(finite_abs(bias), n)))
    return tuple(margins)


def linear_ibp_margins(model):
    """Margin of the final linear operator over its actual input interval."""
    node = model.final_node()
    weight = node.inputs[1].forward_value
    if not node.transB:
        weight = weight.transpose(-1, -2)
    if weight.ndim == 2:
        weight = weight.unsqueeze(0)
    bias = node.inputs[2].forward_value if len(node.inputs) == 3 else 0
    box = model.IBP_general(node=node.inputs[0])
    margin = 10 * box_rounding_error(weight, box.lower, box.upper, bias)
    return margin, margin
