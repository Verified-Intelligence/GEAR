import torch

from auto_LiRPA.operators.clampmult import (
    ClampedMultiplication,
    multiply_by_A_signs,
)


def test_multiply_by_a_signs_treats_zero_times_infinity_as_zero():
    A = torch.tensor([[[0.0, 1.0], [0.0, -2.0]]])
    d_pos = torch.tensor([[[float("inf"), 3.0], [float("inf"), 4.0]]])
    d_neg = torch.tensor([[[5.0, 6.0], [7.0, float("inf")]]])
    b_pos = torch.tensor([[[float("inf"), 2.0], [float("inf"), 3.0]]])
    b_neg = torch.tensor([[[1.0, 2.0], [3.0, float("inf")]]])

    new_A, bias = multiply_by_A_signs(A, d_pos, d_neg, b_pos, b_neg)

    assert not torch.isnan(new_A).any()
    assert not torch.isnan(bias).any()
    assert new_A[0, 0, 0] == 0
    assert bias[0, 0] == 2


def test_multiply_by_a_signs_1d_fast_path_treats_zero_times_infinity_as_zero():
    A = torch.tensor([[0.0, 2.0]])
    d_pos = torch.tensor([float("inf"), 3.0])
    d_neg = torch.tensor([5.0, 6.0])
    b_pos = torch.tensor([float("inf"), 2.0])
    b_neg = torch.tensor([7.0, 8.0])

    new_A, bias = multiply_by_A_signs(A, d_pos, d_neg, b_pos, b_neg)

    assert not torch.isnan(new_A).any()
    assert not torch.isnan(bias).any()
    assert torch.equal(new_A, torch.tensor([[0.0, 6.0]]))
    assert torch.equal(bias, torch.tensor([[0.0, 4.0]]))


def test_clamped_multiplication_patches_reduction_treats_zero_times_infinity_as_zero():
    A = torch.zeros(2, 1, 3, 2, 2)
    d_pos = torch.full_like(A, float("inf"))
    d_neg = torch.ones_like(A)
    b_pos = torch.full_like(A, float("inf"))
    b_neg = torch.ones_like(A)

    new_A, bias = ClampedMultiplication.clamp_mutiply_forward(
        A, d_pos, d_neg, b_pos, b_neg, True, True, False)

    assert not torch.isnan(new_A).any()
    assert not torch.isnan(bias).any()
    assert torch.equal(new_A, torch.zeros_like(new_A))
    assert torch.equal(bias, torch.zeros_like(bias))
