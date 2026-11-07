"""Merging intermediate bounds with reference bounds must not create NaN at +-inf."""
import torch

from auto_LiRPA.bound_general import _tighten_with_reference


def test_infinite_computed_bound_takes_the_reference_value():
    # hs102: CROWN's NaN lower bound had been sanitized to -inf; the straight-through
    # expression max(ref, lo).detach() - lo.detach() + lo is then inf - inf = NaN.
    inf = float('inf')
    lo = torch.tensor([-inf, -2.0, 1.0], requires_grad=True)
    hi = torch.tensor([inf, 5.0, 3.0], requires_grad=True)
    ref_lo = torch.tensor([-1.0, -1.0, 0.0])
    ref_hi = torch.tensor([4.0, 4.0, 9.0])
    new_lo, new_hi = _tighten_with_reference(ref_lo, ref_hi, lo, hi)
    assert torch.equal(new_lo.detach(), torch.tensor([-1.0, -1.0, 1.0]))
    assert torch.equal(new_hi.detach(), torch.tensor([4.0, 4.0, 3.0]))
    (new_lo.sum() + new_hi.sum()).backward()
    # Gradient still reaches the computed bounds where they are finite.
    assert torch.equal(lo.grad, torch.tensor([0.0, 1.0, 1.0]))
    assert torch.equal(hi.grad, torch.tensor([0.0, 1.0, 1.0]))
