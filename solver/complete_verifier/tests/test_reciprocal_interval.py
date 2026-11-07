import torch

from auto_LiRPA.operators.base import Interval
from auto_LiRPA.operators.convex_concave import BoundReciprocal


def test_interval_propagate_zero_lower_returns_infinite_upper():
    node = BoundReciprocal.__new__(BoundReciprocal)

    result = node.interval_propagate(Interval(
        torch.tensor([0.0, 0.5]),
        torch.tensor([2.0, 4.0]),
    ))
    lower, upper = result.lower, result.upper

    torch.testing.assert_close(lower, torch.tensor([0.5, 0.25]))
    assert torch.isinf(upper[0])
    assert upper[0] > 0
    torch.testing.assert_close(upper[1], torch.tensor(2.0))


def test_interval_propagate_negative_and_crossing_zero_is_sound():
    node = BoundReciprocal.__new__(BoundReciprocal)

    result = node.interval_propagate(Interval(
        torch.tensor([-2.0, -1.0, 0.0]),
        torch.tensor([-1.0, 2.0, 0.0]),
    ))
    lower, upper = result.lower, result.upper

    torch.testing.assert_close(lower[0], torch.tensor(-1.0))
    torch.testing.assert_close(upper[0], torch.tensor(-0.5))
    assert torch.isneginf(lower[1:]).all()
    assert torch.isposinf(upper[1:]).all()
