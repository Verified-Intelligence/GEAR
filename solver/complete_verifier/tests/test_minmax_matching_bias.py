import pytest
import torch

from auto_LiRPA.operators.minmax import _assert_matching_biases


def test_matching_biases_accept_identical_nan_positions():
    first = torch.tensor([[1.0, torch.nan], [torch.inf, -torch.inf]])
    second = first.clone()

    _assert_matching_biases(first, second)


def test_matching_biases_reject_different_finite_values():
    first = torch.tensor([1.0, torch.nan])
    second = torch.tensor([2.0, torch.nan])

    with pytest.raises(AssertionError):
        _assert_matching_biases(first, second)


def test_matching_biases_reject_different_nan_positions():
    first = torch.tensor([1.0, torch.nan])
    second = torch.tensor([torch.nan, 1.0])

    with pytest.raises(AssertionError):
        _assert_matching_biases(first, second)
