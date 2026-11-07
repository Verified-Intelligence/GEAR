"""CPU root objective decomposition and hybrid bound regressions."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from input_split.hybrid_bound import root_hybrid_objective_lb


class Griewank(torch.nn.Module):
    def forward(self, x):
        cos = torch.cos(x)
        # ReduceProd is not a supported bound operator; use the product chain.
        product = cos[:, 0] * cos[:, 1] * cos[:, 2]
        return 1 + 0.00025 * (x * x).sum(dim=1, keepdim=True) - product.unsqueeze(1)


def bound_graph(model, dimensions, radius):
    center = torch.zeros(1, dimensions, dtype=torch.float64)
    lower, upper = center - radius, center + radius
    x = BoundedTensor(center, PerturbationLpNorm(x_L=lower, x_U=upper))
    net = BoundedModule(model.double(), center, device='cpu', bound_opts={
        'compare_crown_with_ibp_intermediate': True,
    })
    return net, x, lower, upper


def test_griewank_hybrid_closes_root_and_preserves_graph():
    net, x, lower, upper = bound_graph(Griewank(), 3, 500)
    with torch.no_grad():
        crown_lower, _ = net.compute_bounds(x=(x,), method='CROWN')
    assert crown_lower.item() < -1
    original_target = net.final_node_name
    original_lower = net.final_node().lower
    diagnostics = {}
    hybrid = root_hybrid_objective_lb(net, lower, upper, diagnostics=diagnostics)
    assert hybrid is not None and hybrid >= -1e-9
    assert hybrid <= 0  # f(0) = 0 is a feasible witness.
    assert diagnostics['terms'] == 4
    assert net.final_node_name == original_target
    assert net.final_node().lower is original_lower


class ManyTerms(torch.nn.Module):
    def forward(self, x):
        return (x * x).sum(dim=1, keepdim=True)


@pytest.mark.parametrize('dimensions,max_terms', [(65, 64), (4, 3)])
def test_term_limit_skips_bound_queries(dimensions, max_terms):
    net, _, lower, upper = bound_graph(ManyTerms(), dimensions, 1)
    with patch.object(BoundedModule, 'compute_bounds', side_effect=AssertionError) as compute:
        assert root_hybrid_objective_lb(net, lower, upper, max_terms=max_terms) is None
    compute.assert_not_called()


class AffineTerms(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weights', torch.tensor([[2.], [-3.], [4.]]))

    def forward(self, x):
        squares = x * x
        joined = torch.cat([squares[:, :2], squares[:, 2:]], dim=1)
        objective = -(joined.reshape(-1, 3) @ self.weights) + 7
        return torch.cat([objective, squares.sum(dim=1, keepdim=True)], dim=1)


def test_affine_walk_output_zero_and_negative_coefficients():
    net, _, lower, upper = bound_graph(AffineTerms(), 3, 2)
    diagnostics = {}
    assert root_hybrid_objective_lb(net, lower, upper, diagnostics=diagnostics) == pytest.approx(-17)
    assert diagnostics['terms'] == 3


@pytest.mark.parametrize('method', ['IBP', 'CROWN'])
def test_bound_failure_returns_none(method):
    net, _, lower, upper = bound_graph(Griewank(), 3, 500)
    compute_bounds = BoundedModule.compute_bounds

    def fail(self, *args, **kwargs):
        if kwargs.get('method') == method:
            raise RuntimeError('Failed term query')
        return compute_bounds(self, *args, **kwargs)

    with patch.object(BoundedModule, 'compute_bounds', fail):
        assert root_hybrid_objective_lb(net, lower, upper) is None


@pytest.mark.parametrize('enabled', [False, True])
def test_pipeline_uses_constant_plane_for_stronger_root_bound(monkeypatch, enabled):
    import arguments
    from input_split.batch_branch_and_bound import input_bab_parallel
    from input_split.clip import deconstruct_bias

    config = arguments.ConfigHandler()
    config.construct_config_dict(config.default_args)
    config['solving']['solving_mode'] = True
    config['solver']['nlp']['root_hybrid_term_bound'] = enabled
    monkeypatch.setattr(arguments, 'Config', config)
    net, x, lower, upper = bound_graph(ManyTerms(), 3, 2)
    wrapper = SimpleNamespace(net=net)
    reference = {
        'global_lb': torch.tensor([[-10., -2.]], dtype=torch.float64),
        'lA': {net.input_name[0]: torch.ones(1, 2, 3, dtype=torch.float64)},
        'alphas': {},
    }

    def inspect_root(_net, ret, dm_l, dm_u, c, rhs, global_lb, **kwargs):
        assert global_lb[0, 0] == (0 if enabled else -10)
        assert global_lb[0, 1] == -2
        root_A = ret['lA'][net.input_name[0]]
        if enabled:
            assert torch.count_nonzero(root_A[:, 0]) == 0
            bias = deconstruct_bias(dm_l, dm_u, root_A, global_lb)
            # The reconstructed objective plane is valid at the minimizer x=0.
            assert bias[0, 0] <= net(torch.zeros_like(lower))[0, 0]
        assert torch.equal(root_A[:, 1], reference['lA'][net.input_name[0]][:, 1])
        raise StopIteration

    c = torch.tensor([[[1., 0.], [0., 1.]]], dtype=torch.float64)
    rhs = torch.full((1, 2), float('inf'), dtype=torch.float64)
    with patch('input_split.nlp_in_bab.compute_root_dual_warmstart', inspect_root), \
            patch('input_split.hybrid_bound.root_hybrid_objective_lb',
                  wraps=root_hybrid_objective_lb) as hybrid_call:
        with pytest.raises(StopIteration):
            input_bab_parallel(wrapper, x, c, rhs, torch.tensor([2]), reference_dict=reference)
    assert hybrid_call.call_count == int(enabled)
    assert reference['global_lb'][0, 0] == -10
    assert torch.all(reference['lA'][net.input_name[0]] == 1)
