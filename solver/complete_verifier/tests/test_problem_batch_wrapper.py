"""The per-sample batch wrapper of NNProblemSingleDomain must match the loop exactly."""

import torch

from nlp_solver.problem import NNProblemSingleDomain

DT = torch.float64


class _ConcatBatchOneConstant(torch.nn.Module):
    """Concatenates a batch-1 constant, so a batch of 2 raises (like st_e06)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("const", torch.tensor([[2.0]], dtype=DT))

    def forward(self, x):
        prod = (x[:, 0] * x[:, 1]).reshape(-1, 1)
        return torch.cat([self.const, prod, x[:, :1].square()], dim=1)


def _problem(model):
    return NNProblemSingleDomain(
        model=model,
        dm_l=torch.zeros(2, dtype=DT),
        dm_u=torch.ones(2, dtype=DT),
        C_obj=torch.tensor([[0.0, 1.0, 0.0]], dtype=DT),
        rhs_obj=torch.zeros(1, dtype=DT),
        C_constr=torch.tensor([[1.0, 0.0, 1.0]], dtype=DT),
        rhs_constr=torch.tensor([3.0], dtype=DT),
        device=torch.device("cpu"),
        dtype=DT,
    )


def test_wrapper_matches_per_sample_loop_values_and_gradients():
    base = _ConcatBatchOneConstant()
    problem = _problem(base)
    wrapped = problem.model
    assert wrapped is not base  # the probe detected the unsafe batch axis
    x = torch.rand(7, 2, dtype=DT, requires_grad=True)
    out = wrapped(x)
    ref = torch.cat([base(x[i:i + 1]) for i in range(7)], dim=0)
    assert torch.equal(out, ref)
    (grad,) = torch.autograd.grad(out.square().sum(), x)
    (grad_ref,) = torch.autograd.grad(ref.square().sum(), x)
    assert torch.allclose(grad, grad_ref)


def test_wrapper_falls_back_to_the_loop_when_vmap_fails():
    class _NeedsItem(_ConcatBatchOneConstant):
        def forward(self, x):
            if float(x[0, 0].item()) > 2.0:  # data-dependent control flow
                return x
            return super().forward(x)

    base = _NeedsItem()
    wrapped = _problem(base).model
    x = torch.rand(5, 2, dtype=DT)
    out = wrapped(x)
    assert wrapped.use_vmap is False
    ref = torch.cat([base(x[i:i + 1]) for i in range(5)], dim=0)
    assert torch.equal(out, ref)
