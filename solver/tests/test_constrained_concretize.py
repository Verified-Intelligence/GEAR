"""Test optimized bounds in simple_verification."""
import pytest
import torch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.concretize_func import (
    constraints_solving,
    sort_out_constr_batches,
)
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import eyeC
from auto_LiRPA.operators import BoundRelu
from auto_LiRPA.fp_enclosure import box_rounding_error
from enclosure_test_utils import assert_enclosed

from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE

class ConstrainedConcretizeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.tensor([[1., -1.], [2., -1.]])
        self.w2 = torch.tensor([[1., -1.]])

    def forward(self, x):
        z1 = x.matmul(self.w1.t())
        hz1 = torch.nn.functional.relu(z1)
        z2 = hz1.matmul(self.w2.t())
        return z2

class TestConstrainedConcretize(TestCase):
    def __init__(self, methodName='runTest', generate=False, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName, 1, "test_constrained_concretize", generate, device=device, dtype=dtype)

    def test(self):
        model = ConstrainedConcretizeModel().to(self.default_device).to(self.default_dtype)
        # Input x.
        x = torch.tensor([[1., 1.]], dtype=self.default_dtype, device=self.default_device)
        # Lower and upper bounds of x.
        lower = torch.tensor([[-1., -2.]], dtype=self.default_dtype, device=self.default_device)
        upper = torch.tensor([[2., 1.]], dtype=self.default_dtype, device=self.default_device)

        # Wrap model with auto_LiRPA for bound computation.
        # The second parameter is for constructing the trace of the computational graph,
        # and its content is not important.

        lirpa_model = BoundedModule(model, torch.empty_like(x))
        pred = lirpa_model(x)
        print(f'Model prediction: {pred.item()}')

        # Compute bounds using LiRPA using the given lower and upper bounds.
        norm = float("inf")
        ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper)
        bounded_x = BoundedTensor(x, ptb)

        # Compute bounds.
        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
        print(f'CROWN bounds: lower={lb.item()}, upper={ub.item()}')

        # Add a new constraint of :
        #    1*x_0 + 1*x_1 + 2 <= 0
        constraint_a = torch.tensor([[[1.0, 1.0]]], dtype=self.default_dtype, device=self.default_device)
        constraint_b = torch.tensor([[2.0]], dtype=self.default_dtype, device=self.default_device)
        constraints = (constraint_a, constraint_b)

        norm = float("inf")
        ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper, constraints=constraints)
        bounded_x = BoundedTensor(x, ptb)
        # Compute bounds.
        constrained_lb, constrained_ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
        print(f'CROWN bounds (with constraints): lower={constrained_lb.item()}, upper={constrained_ub.item()}')

        self.result = (lb, ub, constrained_lb, constrained_ub)
        if self.generate:
            self.check()
        else:
            # The output dot product reduces the two ReLU activations. Its
            # actual intermediate box also covers changes to their planes.
            relu = next(n for n in lirpa_model.nodes() if isinstance(n, BoundRelu))
            box = lirpa_model.IBP_general(node=relu)
            margin = 10 * box_rounding_error(
                model.w2.unsqueeze(0), box.lower, box.upper)
            for actual, exact, sign in zip(self.result, self.reference, (-1, 1, -1, 1)):
                assert_enclosed(actual, exact.to(actual), margin, sign)


def test_float64_constrained_crown_matches_exact_box_extrema():
    center = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    constraints = (
        torch.tensor([[[1.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[2.0]], dtype=torch.float64),
    )
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(
            norm=float('inf'),
            x_L=torch.tensor([[-1.0, -2.0]], dtype=torch.float64),
            x_U=torch.tensor([[2.0, 1.0]], dtype=torch.float64),
            constraints=constraints,
        ),
    )
    bounded_model = BoundedModule(torch.nn.Identity().double(), center)

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_input,), method='CROWN')

    torch.testing.assert_close(
        lower, torch.tensor([[-1.0, -2.0]], dtype=torch.float64))
    torch.testing.assert_close(
        upper, torch.tensor([[0.0, -1.0]], dtype=torch.float64))


def test_mixed_input_and_constraint_dtypes_promote_constrained_crown():
    class ToDouble(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.float64)

    center = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    constraints = (
        torch.tensor([[[1.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[2.0]], dtype=torch.float64),
    )
    bounded_input = BoundedTensor(
        center,
        PerturbationLpNorm(
            norm=float('inf'),
            x_L=torch.tensor([[-1.0, -2.0]], dtype=torch.float32),
            x_U=torch.tensor([[2.0, 1.0]], dtype=torch.float32),
            constraints=constraints,
        ),
    )
    bounded_model = BoundedModule(ToDouble(), center)

    lower, upper = bounded_model.compute_bounds(
        x=(bounded_input,), method='CROWN')

    torch.testing.assert_close(
        lower, torch.tensor([[-1.0, -2.0]], dtype=torch.float64))
    torch.testing.assert_close(
        upper, torch.tensor([[0.0, -1.0]], dtype=torch.float64))


def test_cpu_eye_c_constraint_solving_promotes_operands():
    x_lower = torch.tensor(
        [[-1.0, -2.0], [-2.0, -1.0]], dtype=torch.float32)
    x_upper = torch.tensor(
        [[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    raw_constraints = (
        torch.tensor(
            [[[1.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[2.0], [2.0]], dtype=torch.float64),
    )
    constraints, sorted_batches = sort_out_constr_batches(
        x_lower, x_upper, raw_constraints)

    lower = constraints_solving(
        x_lower,
        x_upper,
        eyeC([2, 2, 2], x_lower.device),
        constraints,
        sign=-1.0,
        sorted_out_batches=sorted_batches,
        max_chunk_size=1,
    ).squeeze(-1)
    upper = constraints_solving(
        x_lower,
        x_upper,
        eyeC([2, 2, 2], x_lower.device),
        constraints,
        sign=1.0,
        sorted_out_batches=sorted_batches,
        max_chunk_size=1,
    ).squeeze(-1)

    torch.testing.assert_close(
        lower,
        torch.tensor([[-1.0, -2.0], [-2.0, -1.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        upper,
        torch.tensor([[0.0, -1.0], [-1.0, 0.0]], dtype=torch.float64),
    )


def test_sparse_linf_constraint_mapping_promotes_constraints_only():
    center = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    lower = center.clone()
    upper = center.clone()
    upper[0, 0] += 0.1
    constraints = (
        torch.tensor([[[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[10.0]], dtype=torch.float64),
    )
    perturbation = PerturbationLpNorm(
        norm=float('inf'), x_L=lower, x_U=upper, constraints=constraints)

    linear, _, _ = perturbation.init(center, forward=True)

    assert perturbation.constraints_sparse[0].dtype == torch.float64
    assert perturbation.constraints_sparse[1].dtype == torch.float64
    assert linear.lw.dtype == torch.float32

if __name__ == '__main__':
    testcase = TestConstrainedConcretize(generate=True)
    testcase.setUp()
    testcase.test()


def test_chunking_counts_device_memory_not_only_the_allocator_cache(monkeypatch):
    """After empty_cache the allocator holds almost nothing free; the device does."""
    from types import SimpleNamespace
    from auto_LiRPA import concretize_func

    monkeypatch.setattr(concretize_func.torch.cuda, "get_device_properties",
                        lambda d: SimpleNamespace(total_memory=32 * 2**30))
    monkeypatch.setattr(concretize_func.torch.cuda, "memory_allocated", lambda d: 10 * 2**30)
    monkeypatch.setattr(concretize_func.torch.cuda, "memory_reserved",
                        lambda d: 10 * 2**30 + 9 * 2**20)
    free = concretize_func._chunking_free_memory("cuda")
    assert free == int(0.9 * 32 * 2**30) - 10 * 2**30
    # A large cached pool still counts when it is the larger estimate.
    monkeypatch.setattr(concretize_func.torch.cuda, "memory_reserved", lambda d: 31 * 2**30)
    assert concretize_func._chunking_free_memory("cuda") == 21 * 2**30


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('aux', [False, True])
def test_supplied_active_center_keeps_its_own_objective_base(sign, aux):
    lower = -torch.ones(3, 2, dtype=torch.float64)
    upper = -lower
    objective = torch.tensor([[[1., 2.], [-2., 1.]]], dtype=lower.dtype)
    active = torch.tensor([0, 2])
    constraints = (torch.zeros(2, 1, 2, dtype=lower.dtype),
                   -torch.ones(2, 1, dtype=lower.dtype))
    naive = constraints_solving(lower, upper, objective, None, sign)
    result = constraints_solving(
        lower, upper, objective, constraints, sign,
        sorted_out_batches={'active_indices': active},
        objective_indices=torch.tensor([[0], [1], [0]]),
        act_x0=torch.full((2, 2, 1), 0.3, dtype=lower.dtype),
        act_eps=torch.full((2, 2, 1), 0.25, dtype=lower.dtype),
        aux_bounds=naive if aux else None, max_chunk_size=1, use_grad=False)
    torch.testing.assert_close(result[active, 0, 0],
                               torch.full((2,), 0.9 + sign * 0.75, dtype=lower.dtype))
    assert torch.equal(result[1], naive[1])
    assert torch.equal(result[:, 1], naive[:, 1])


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('broadcast', [False, True])
def test_cpu_fused_failure_keeps_selected_objective_gradients(monkeypatch, sign, broadcast):
    import auto_LiRPA.concretize_func as cf
    import auto_LiRPA.concretize_triton as ct

    lower = torch.zeros(3, 2, dtype=torch.float64)
    upper = torch.ones_like(lower)
    cons, batches = sort_out_constr_batches(
        lower, upper, (torch.ones(3, 1, 2, dtype=lower.dtype),
                       -torch.ones(3, 1, dtype=lower.dtype)))
    objective = torch.tensor([[[1., 2.], [-2., 1.]]], dtype=lower.dtype)
    if not broadcast:
        objective = objective.expand(3, -1, -1).clone()
    indices = torch.tensor([[0], [1], [0]])

    def solve():
        a = objective.clone().requires_grad_()
        result = cf.constraints_solving(
            lower, upper, a, cons, sign, sorted_out_batches=batches,
            objective_indices=indices, max_chunk_size=1, use_grad=True)
        return result, torch.autograd.grad(result.sum(), a)[0]

    monkeypatch.setattr(cf, 'FUSED_SWEEP', False)
    reference, grad_reference = solve()
    monkeypatch.setattr(cf, 'FUSED_SWEEP', True)
    monkeypatch.setattr(ct, 'fused_sweep_supported', lambda *args: True)

    def broken(*args, **kwargs):
        raise RuntimeError('test kernel failure')

    monkeypatch.setattr(ct, 'fused_dual_sweep', broken)
    with pytest.warns(UserWarning, match='Fused constrained-concretization kernel failed'):
        result, gradient = solve()
    assert torch.equal(result, reference)
    assert torch.equal(gradient, grad_reference)
    assert cf.FUSED_SWEEP is False
