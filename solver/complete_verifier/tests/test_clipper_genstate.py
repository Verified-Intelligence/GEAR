"""Sanity tests for the alpha-zono substitution BaB (clipper_*).

Everything is pinned to the worked example in zonotope.md section 2 (a 2-2-2-1
ReLU net), whose numbers are computed by hand in the doc:
  * root zonotope bound = -1/2 (s2.4);
  * splitting neuron (2,1)=e3 gives both children's exact LP bound 0 (s2.5);
  * the property f >= 0 is proved after one split.

Covered:
  * clipper_gen_state.build_gen_state on a REAL tiny BoundedModule reproduces the
    doc's generator state eq 14 / eq 15 (autograd backward vs hand derivation);
  * the factored dual solver is sound (bound <= exact LP, via a scipy oracle);
  * the factored single-constraint infeasibility prune is SOUND (a feasible node is
    never falsely certified) -- this test catches the sign-bug fixed on 2026-06-28.

Run: python -m pytest tests/test_clipper_genstate.py -v
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
import pytest

from clipper_dual_kernel import build_problem

_DT = torch.float64


def _doc_state(device):
    """The zonotope.md section-2 worked example as an alpha-zono gen-state.

    generators: [e1^0, e2^0, e1, e2, e3, e4] (n_gens=6, n_input=2).
    output zonotope (eq 14): f = .5 e1^0 + .5 e2^0 + .5 e1 + 0 e2 + .5 e3 + .5 e4 + 2.
    """
    output_gen_coeffs = torch.tensor([[0.5, 0.5, 0.5, 0.0, 0.5, 0.5]], dtype=_DT, device=device)
    output_center = torch.tensor([2.0], dtype=_DT, device=device)

    def neuron(row, relu_error_col_idx):
        return dict(layer_idx=0, neuron_idx=relu_error_col_idx, preact_center=0.0, lo=-2.0, hi=2.0,
                    relu_slope=0.5, band_half_width=0.5, relu_error_col=relu_error_col_idx,
                    row_indices=np.array([i for i, v in enumerate(row) if v != 0],
                                         dtype=np.int64),
                    row_values=np.array([v for v in row if v != 0], dtype=np.float64))

    ul = [
        neuron([1, 1, 0, 0, 0, 0], 2),      # L1n1 (e1): x = e1^0 + e2^0
        neuron([1, -1, 0, 0, 0, 0], 3),     # L1n2 (e2): x = e1^0 - e2^0
        neuron([1, 0, 0.5, 0.5, 0, 0], 4),  # L2n1 (e3): x = e1^0 + .5e1 + .5e2
        neuron([0, 1, 0.5, -0.5, 0, 0], 5),  # L2n2 (e4): x = e2^0 + .5e1 - .5e2
    ]
    return dict(n_gens=6, n_input=2, n_out=1, output_gen_coeffs=output_gen_coeffs, output_center=output_center,
                e_lb=torch.full((6,), -1.0, dtype=_DT, device=device),
                e_hi=torch.full((6,), 1.0, dtype=_DT, device=device),
                unstable_list=ul)


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_clipper_has_three_runtime_modules():
    """The clipper-only implementation should live in bab, dual_kernel, and gen_state."""
    root = Path(__file__).resolve().parents[1]
    assert not (root / 'clipper_lp_kernel.py').exists()
    assert not (root / 'clipper_subst_problem.py').exists()
    assert (root / 'clipper_bab.py').exists()
    assert (root / 'clipper_dual_kernel.py').exists()
    assert (root / 'clipper_gen_state.py').exists()


def test_root_bound_matches_doc():
    """Root zonotope bound = -1/2 (zonotope.md section 2.4)."""
    dev = _device()
    state = _doc_state(dev)
    problem = build_problem(state, np.array([1.0]), 0.0, list(range(4)), dev, _DT)
    # root LP min_e d.e + spec_center over the box [-1,1] = spec_center - sum|d|  (eq 13)
    root = float(problem['spec_center']) - float(problem['spec_gen_coeffs'].abs().sum())
    assert abs(root - (-0.5)) < 1e-9, root


def test_root_decomposition_exposes_center_coefficients_and_error_terms():
    """Root diagnostics should isolate center, generator coefficients, and ReLU error terms."""
    from clipper_gen_state import root_decomposition
    dev = _device()
    state = _doc_state(dev)
    diag = root_decomposition(state, np.array([1.0]), 0.0)
    assert torch.allclose(diag['spec_center'], torch.tensor([2.0], dtype=_DT, device=dev))
    assert torch.allclose(
        diag['spec_gen_coeffs'][0],
        torch.tensor([0.5, 0.5, 0.5, 0.0, 0.5, 0.5], dtype=_DT, device=dev))
    assert torch.allclose(diag['input_abs_contrib'], torch.tensor([1.0], dtype=_DT, device=dev))
    assert torch.allclose(diag['relu_error_abs_contrib'], torch.tensor([1.5], dtype=_DT, device=dev))
    assert torch.allclose(diag['lower_bound'], torch.tensor([-0.5], dtype=_DT, device=dev))
    assert diag['unstable_error_cols'] == [2, 3, 4, 5]
    assert torch.allclose(
        diag['unstable_error_coeffs'][0],
        torch.tensor([0.5, 0.0, 0.5, 0.5], dtype=_DT, device=dev))


def test_intermediate_bounds_table_exposes_per_unstable_neuron_rows():
    """Bounds diagnostics should give one alignment row per unstable neuron."""
    from clipper_gen_state import intermediate_bounds_table
    dev = _device()
    rows = intermediate_bounds_table(_doc_state(dev))
    assert len(rows) == 4
    assert rows[0]['layer_idx'] == 0
    assert rows[0]['neuron_idx'] == 2
    assert rows[0]['lo'] == -2.0
    assert rows[0]['hi'] == 2.0
    assert rows[0]['preact_center'] == 0.0
    assert rows[0]['relu_slope'] == 0.5
    assert rows[0]['band_half_width'] == 0.5
    assert rows[0]['relu_error_col'] == 2


# --- factored dual kernel (clipper_dual_kernel; derived from zonotope.md) ---

def test_factored_dual_certifies_doc_example():
    """The factored dual kernel certifies the worked example."""
    from clipper_dual_kernel import verify_spec as dverify
    dev = _device()
    state = _doc_state(dev)
    verdict, info = dverify(state, np.array([1.0]), 0.0, dev, time_limit=10,
                            K=1, bab_dtype=_DT, verbose=False)
    assert verdict == 'safe', (verdict, info)


# =============================================================================
# build_gen_state on a REAL BoundedModule (autograd backward vs zonotope.md eq14)
# =============================================================================

def _doc_net_gen_state(device, opt_slope_qw=None, opt_iters=0):
    """Build the zonotope.md s2 net (2-2-2-1) as an auto_LiRPA BoundedModule and
    run build_gen_state with the doc's assumed pre-act bounds [-2,2].  Returns the
    gen-state.  The default slope on [-2,2] is hi/(hi-lo)=1/2, exactly the doc's
    c=1/2, so build_gen_state must reproduce eq 14 / eq 15 exactly."""
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from clipper_gen_state import build_gen_state

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 2), torch.nn.ReLU(),
        torch.nn.Linear(2, 2), torch.nn.ReLU(),
        torch.nn.Linear(2, 1)).to(device).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1., 1.], [1., -1.]])); net[0].bias.zero_()
        net[2].weight.copy_(torch.tensor([[1., 1.], [1., -1.]]))
        net[2].bias.copy_(torch.tensor([-1., 0.]))
        net[4].weight.copy_(torch.tensor([[1., 1.]])); net[4].bias.copy_(torch.tensor([1.]))
    box_minimizer = torch.zeros(1, 2, device=device, dtype=_DT)
    bm = BoundedModule(net, box_minimizer, device=device)
    ptb = PerturbationLpNorm(eps=1.0,
                             x_L=torch.full((1, 2), -1., device=device, dtype=_DT),
                             x_U=torch.full((1, 2), 1., device=device, dtype=_DT))
    bx = BoundedTensor(box_minimizer, ptb)
    bm.compute_bounds(x=(bx,), method='backward')      # populate graph + names
    lower_bounds, upper_bounds = {}, {}
    for r in bm.relus:                                  # force the doc's [-2,2]
        pre = r.inputs[0].name
        sz = 1
        for d in r.inputs[0].output_shape[1:]:
            sz *= d
        lower_bounds[pre] = torch.full((1, sz), -2., device=device, dtype=_DT)
        upper_bounds[pre] = torch.full((1, sz), 2., device=device, dtype=_DT)
    ret = {'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds}
    wrap = type('N', (), {})()
    wrap.net = bm; wrap.final_name = bm.final_name; wrap.device = device
    return build_gen_state(wrap, ret, bx, device, opt_slope_qw=opt_slope_qw,
                           opt_iters=opt_iters, verbose=False)


def test_build_gen_state_reproduces_doc_eq14():
    """The autograd-backward generator state matches the doc's hand derivation:
    output_center=2, output_gen_coeffs=[.5,.5,.5,0,.5,.5] (eq 14), and each unstable neuron's row
    matches its input-side zonotope (eq 15 for e3: e1^0 + .5 e1 + .5 e2)."""
    dev = _device()
    st = _doc_net_gen_state(dev)
    assert st['n_gens'] == 6 and st['n_input'] == 2
    assert abs(float(st['output_center'][0]) - 2.0) < 1e-9
    assert np.allclose(st['output_gen_coeffs'][0].cpu().numpy(),
                       [0.5, 0.5, 0.5, 0.0, 0.5, 0.5], atol=1e-9)
    # every unstable neuron: relu_slope=band_half_width=1/2, preact_center=0 (doc step 2); rows match eq 14/15.
    rows = {}
    for u in st['unstable_list']:
        assert abs(u['relu_slope'] - 0.5) < 1e-9 and abs(u['band_half_width'] - 0.5) < 1e-9
        assert abs(u['preact_center']) < 1e-9
        rows[u['relu_error_col']] = dict(zip(u['row_indices'].tolist(),
                                        np.round(u['row_values'], 6).tolist()))
    assert rows[2] == {0: 1.0, 1: 1.0}          # L1n1: e1^0 + e2^0
    assert rows[3] == {0: 1.0, 1: -1.0}         # L1n2: e1^0 - e2^0
    assert rows[4] == {0: 1.0, 2: 0.5, 3: 0.5}  # L2n1 = e3: e1^0 + .5e1 + .5e2 (eq 15)
    assert rows[5] == {1: 1.0, 2: 0.5, 3: -0.5}  # L2n2 = e4: e2^0 + .5e1 - .5e2


# ---------------------------------------------------------------------------
# build_gen_state on a multi-layer net (beyond the 2-2-2-1 doc net): the
# auto_LiRPA-backward extraction + Adam slope-opt give a SOUND, tightening root.
# ---------------------------------------------------------------------------
def test_build_gen_state_adam_sound_and_tightening_on_mlp():
    """On a random 5-8-8-3 MLP: build_gen_state's root bound is a valid lower bound
    of the true network output over the box (soundness), and Adam slope-opt only
    improves it vs the default slope init (tightening)."""
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from clipper_gen_state import build_gen_state, root_bound
    dev = _device(); torch.manual_seed(3)
    net = torch.nn.Sequential(
        torch.nn.Linear(5, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 3)).to(dev).double()
    x0 = torch.zeros(1, 5, dtype=_DT, device=dev)
    bm = BoundedModule(net, x0, device=dev)
    eps = 0.5
    ptb = PerturbationLpNorm(eps=eps, x_L=x0 - eps, x_U=x0 + eps)
    bx = BoundedTensor(x0, ptb)
    bm.compute_bounds(x=(bx,), method='backward')
    lower, upper = {}, {}
    for r in bm.relus:
        pre = r.inputs[0].name
        lower[pre] = r.inputs[0].lower.detach(); upper[pre] = r.inputs[0].upper.detach()
    ret = {'lower_bounds': lower, 'upper_bounds': upper}
    wrap = type('N', (), {})(); wrap.net = bm; wrap.final_name = bm.final_name; wrap.device = dev
    spec = np.array([1.0, -1.0, 0.0])                      # a linear spec over 3 outputs
    base_bound = float(root_bound(
        build_gen_state(wrap, ret, bx, dev, dtype=_DT, verbose=False), spec, 0.0))
    adam = float(root_bound(
        build_gen_state(wrap, ret, bx, dev, dtype=_DT,
                        opt_slope_qw=spec[None], opt_iters=40, verbose=False), spec, 0.0))
    assert adam >= base_bound - 1e-6, (base_bound, adam)  # Adam tightens (or ties)
    # soundness: root <= true min of (spec . net(x)) over the box (sampled)
    xs = x0 + eps * (2 * torch.rand(4000, 5, dtype=_DT, device=dev) - 1)
    fx = net(xs) @ torch.tensor(spec, dtype=_DT, device=dev)
    assert adam <= float(fx.min()) + 1e-6, (adam, float(fx.min()))


def _mlp_wrap(dev):
    """A 5-8-8-3 MLP with its pre-activation bounds frozen, ready for the builder."""
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    torch.manual_seed(3)
    net = torch.nn.Sequential(
        torch.nn.Linear(5, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 3)).to(dev).double()
    x0 = torch.zeros(1, 5, dtype=_DT, device=dev)
    eps = 0.5
    bm = BoundedModule(net, x0, device=dev)
    bx = BoundedTensor(x0, PerturbationLpNorm(eps=eps, x_L=x0 - eps, x_U=x0 + eps))
    bm.compute_bounds(x=(bx,), method='backward')
    lower, upper = {}, {}
    for r in bm.relus:
        pre = r.inputs[0].name
        lower[pre] = r.inputs[0].lower.detach()
        upper[pre] = r.inputs[0].upper.detach()
    wrap = type('N', (), {})()
    wrap.net = bm
    wrap.final_name = bm.final_name
    wrap.device = dev
    return wrap, bm, bx, {'lower_bounds': lower, 'upper_bounds': upper}


def test_build_gen_state_gives_back_the_alpha_it_borrowed():
    """The lines go in through alpha, which belongs to whoever called this.

    The clipper runs after alpha-CROWN, on alpha-CROWN's module, and falls
    through to beta-CROWN whenever it does not close the property -- so alphas
    it silently replaced with its own single '_forward' entry would be gone
    from under the solver that comes next.
    """
    from clipper_gen_state import build_gen_state
    dev = _device()
    wrap, bm, bx, ret = _mlp_wrap(dev)
    C = torch.tensor([[[1., -1., 0.], [0., 1., -1.]]], dtype=_DT, device=dev)
    bm.init_alpha(x=(bx,), method='backward', c=C)
    before = {r.name: (dict(r.alpha), r.opt_stage) for r in bm.relus}
    assert all(alpha for alpha, _ in before.values()), before

    build_gen_state(wrap, ret, bx, dev, opt_slope_qw=np.array([[1., -1., 0.]]),
                    opt_iters=5, verbose=False)

    for r in bm.relus:
        alpha, opt_stage = before[r.name]
        assert r.opt_stage == opt_stage
        assert set(r.alpha) == set(alpha), (r.name, list(r.alpha), list(alpha))
        for key in alpha:
            assert r.alpha[key] is alpha[key], (r.name, key)


def test_build_gen_state_ignores_a_backward_alpha():
    """opt_stage is not a line. A module in alpha-CROWN's optimization stage
    carries alphas keyed by start node, which say nothing about this pass, so
    the builder relaxes at the default triangle line rather than reading a key
    that is not there."""
    from clipper_gen_state import build_gen_state
    dev = _device()
    wrap, bm, bx, ret = _mlp_wrap(dev)
    C = torch.tensor([[[1., -1., 0.], [0., 1., -1.]]], dtype=_DT, device=dev)
    bm.init_alpha(x=(bx,), method='backward', c=C)
    for r in bm.relus:
        r.opt_start()

    st = build_gen_state(wrap, ret, bx, dev, verbose=False)
    for u in st['unstable_list']:
        lo, hi = u['lo'], u['hi']
        assert abs(u['relu_slope'] - hi / (hi - lo)) < 1e-12, u


def test_gen_state_root_is_sound_lower_bound():
    """The gen-state root bound must be a valid lower bound of the true network
    minimum.  For the doc net the true min over [-1,1]^2 is >= 1 (f(0,-1)=1; the
    doc remark); the zonotope root is -1/2 <= 1.  Also sample-check no input beats
    the certified-after-split value 0."""
    from clipper_gen_state import root_bound
    dev = _device()
    st = _doc_net_gen_state(dev)
    rb = float(root_bound(st, np.array([1.0]), 0.0))
    assert abs(rb - (-0.5)) < 1e-6, rb                 # matches doc s2.4
    # brute-force the true min of f on a grid -> must be >= the sound bound
    g = torch.linspace(-1, 1, 41, dtype=_DT)
    xx, yy = torch.meshgrid(g, g, indexing='ij')
    pts = torch.stack([xx.flatten(), yy.flatten()], 1)
    h1 = torch.relu(pts @ torch.tensor([[1., 1.], [1., -1.]], dtype=_DT).T)
    h2 = torch.relu(h1 @ torch.tensor([[1., 1.], [1., -1.]], dtype=_DT).T
                    + torch.tensor([-1., 0.], dtype=_DT))
    f = h2 @ torch.tensor([1., 1.], dtype=_DT) + 1.0
    assert rb <= float(f.min()) + 1e-9                 # sound: root <= true min


def test_a_k_excludes_own_e_new():
    """z_k does not depend on its own post-activation error, so a_k[relu_error_col]=0."""
    dev = _device()
    st = _doc_net_gen_state(dev)
    for u in st['unstable_list']:
        assert u['relu_error_col'] not in set(u['row_indices'].tolist())


def test_slope_opt_does_not_lower_root():
    """Per-neuron slope optimization can only improve (or keep) the root bound vs
    the default slope init (it maximizes the same bound, restoring the best relu_slope seen)."""
    from clipper_gen_state import root_bound
    dev = _device()
    spec_output_coeffs = np.array([1.0])
    base = float(root_bound(_doc_net_gen_state(dev), spec_output_coeffs, 0.0))
    opt = float(root_bound(_doc_net_gen_state(dev, opt_slope_qw=spec_output_coeffs, opt_iters=30), spec_output_coeffs, 0.0))
    assert opt >= base - 1e-6, (base, opt)


# =============================================================================
# Dual solver soundness
# =============================================================================

def _exact_node_lp(problem, split_sides):
    """Ground-truth node LP min via scipy: min_e path_center + path_gen_coeffs.e s.t. split
    halfspaces, e in [-1,1].  Returns +inf if infeasible.  (scipy = oracle.)"""
    from scipy.optimize import linprog
    preact_gen_coeffs = problem['preact_gen_coeffs'].cpu().numpy(); preact_center_value = problem['preact_center'].cpu().numpy()
    inactive_lower_shift_value = problem['inactive_lower_shift'].cpu().numpy(); active_upper_shift_value = problem['active_upper_shift'].cpu().numpy()
    subst_gen_scale = problem['subst_gen_scale'].cpu().numpy(); subst_const_shift = problem['subst_const_shift'].cpu().numpy()
    spec_gen_coeffs = problem['spec_gen_coeffs'].cpu().numpy(); spec_center = problem['spec_center']; relu_error_col_idx = problem['relu_error_col'].cpu().numpy()
    num_gens = problem['n_gens']; depth = len(split_sides)
    path_gen_coeffs = spec_gen_coeffs.copy()
    path_gen_coeffs[relu_error_col_idx[:depth]] = 0.0
    path_center = spec_center
    constraint_coeffs, constraint_bias = [], []
    for split_index, split_side in enumerate(split_sides):
        phase_sign = 1.0 if split_side == 0 else -1.0
        path_gen_coeffs = (
            path_gen_coeffs
            + subst_gen_scale[split_index, split_side] * preact_gen_coeffs[split_index])
        path_center += subst_const_shift[split_index, split_side]
        constraint_coeffs.append(phase_sign * preact_gen_coeffs[split_index])
        constraint_bias.append(
            preact_center_value[split_index] if split_side == 0 else -preact_center_value[split_index])
        constraint_coeffs.append(-phase_sign * preact_gen_coeffs[split_index])
        constraint_bias.append(
            -inactive_lower_shift_value[split_index]
            if split_side == 0 else -active_upper_shift_value[split_index])
    r = linprog(path_gen_coeffs, A_ub=np.array(constraint_coeffs), b_ub=-np.array(constraint_bias),
                bounds=[(-1, 1)] * num_gens, method='highs')
    return (float(r.fun) + path_center) if r.success else float('inf')


def test_warm_logbucket_node_bounds_sound_vs_exact_lp():
    """The warm logbucket kernel is sound against the scipy LP oracle."""
    pytest.importorskip("scipy")
    from clipper_dual_kernel import (
        build_problem as dual_build,
        _fast_problem_at_depth,
        node_bound_warm_fast,
        verify_spec as dverify,
    )
    dev = _device()
    state = _doc_state(dev)
    problem = dual_build(state, np.array([1.0]), 0.0, list(range(4)), dev, _DT)
    for depth in (1, 2, 3, 4):
        F = _fast_problem_at_depth(problem, depth)
        B = 1 << depth
        for bb in range(B):
            split_sides = [(bb >> j) & 1 for j in range(depth)]
            ex = _exact_node_lp(problem, split_sides)
            st = torch.tensor([split_sides], device=dev, dtype=torch.long)
            z = torch.zeros(1, depth, device=dev, dtype=_DT)
            dual = float(node_bound_warm_fast(F, st, z, z, K=3, Kb=64)[0])
            assert dual <= ex + 1e-6, ('warm dual>exact', split_sides, dual, ex)
            if np.isfinite(ex):
                assert np.isfinite(dual), ('warm dual false +inf', split_sides, ex)
    verdict, info = dverify(state, np.array([1.0]), 0.0, dev, time_limit=10,
                            K=1, bab_dtype=_DT, verbose=False, fast_Kb=64)
    assert verdict == 'safe', (verdict, info)


def test_infeasibility_prune_is_sound():
    """The factored single-constraint infeasibility prune must NOT certify a
    feasible node (regression for the 2026-06-28 sign-bug).  Construct a 1-neuron
    problem: OFF constraint e0 + preact_center <= 0 with |preact_gen_coeffs|_1 = 1.
      preact_center=0.5  -> feasible (e0 <= -0.5); true OFF bound = -1, NOT certified.
      preact_center=2.0  -> infeasible (e0 <= -2 impossible); certified (+inf)."""
    from clipper_dual_kernel import build_problem as dual_build
    from clipper_dual_kernel import _fast_problem_at_depth, node_bound_warm_fast
    dev = _device()

    def one_neuron(preact_center_value):
        state = dict(
            n_gens=4, n_input=3, n_out=1,
            output_gen_coeffs=torch.tensor([[1., 0., 0., 0.]], dtype=_DT, device=dev),
            output_center=torch.tensor([0.0], dtype=_DT, device=dev),
            e_lb=torch.full((4,), -1., dtype=_DT, device=dev),
            e_hi=torch.full((4,), 1., dtype=_DT, device=dev),
            unstable_list=[dict(layer_idx=0, neuron_idx=3, preact_center=preact_center_value, lo=-2., hi=2.,
                                relu_slope=0.5, band_half_width=0.5, relu_error_col=3,
                                row_indices=np.array([0], np.int64),
                                row_values=np.array([1.0]))])
        return dual_build(state, np.array([1.0]), 0.0, [0], dev, _DT)

    split_sides = torch.tensor([[0]], device=dev, dtype=torch.int8)   # OFF
    z = torch.zeros(1, 1, device=dev, dtype=_DT)
    feas, _, _ = node_bound_warm_fast(
        _fast_problem_at_depth(one_neuron(0.5), 1), split_sides.to(torch.long), z, z, K=3, Kb=64)
    assert torch.isfinite(feas).all(), feas
    infe, _, _ = node_bound_warm_fast(
        _fast_problem_at_depth(one_neuron(2.0), 1), split_sides.to(torch.long), z.clone(), z.clone(), K=3, Kb=64)
    assert torch.isinf(infe[0]) and infe[0] > 0, infe


# =============================================================================
# the gen state is read off a dynamic-forward pass: it covers a DAG, and where
# it has no construction the caller is told so rather than handed a wrong state.
# =============================================================================

def _tiny_gen_state(model, n_in, device, eps=0.5, **kwargs):
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from clipper_gen_state import build_gen_state

    # deliberately NOT centred on zero: the input generators carry a centre of
    # their own, and a symmetric box hides anyone who forgets it
    x0 = torch.linspace(-1., 1., n_in, dtype=_DT, device=device).reshape(1, n_in)
    bm = BoundedModule(model, x0, device=device)
    bx = BoundedTensor(x0, PerturbationLpNorm(eps=eps, x_L=x0 - eps, x_U=x0 + eps))
    bm.compute_bounds(x=(bx,), method='backward')
    ret = {'lower_bounds': {r.inputs[0].name: r.inputs[0].lower.detach()
                            for r in bm.relus},
           'upper_bounds': {r.inputs[0].name: r.inputs[0].upper.detach()
                            for r in bm.relus}}
    wrap = type('N', (), {})()
    wrap.net = bm
    wrap.final_name = bm.final_name
    wrap.device = device
    state = build_gen_state(wrap, ret, bx, device, dtype=_DT, verbose=False,
                            **kwargs)
    interm = {k: (v, ret['upper_bounds'][k]) for k, v in ret['lower_bounds'].items()}
    return state, bm, bx, interm


class _Residual(torch.nn.Module):
    """Two branches merging, so the two of them have to share a column space."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = torch.nn.Linear(4, 5)
        self.h = torch.nn.Linear(5, 5)
        self.skip = torch.nn.Linear(4, 5)
        self.out = torch.nn.Linear(5, 2)

    def forward(self, x):
        return self.out(torch.relu(self.h(torch.relu(self.a(x))) + self.skip(x)))


def test_build_gen_state_on_a_residual_net_is_sound():
    """Both branches open error columns, and adding them is only legal because
    a column index means the same variable on both sides."""
    from clipper_gen_state import root_bound
    dev = _device()
    model = _Residual().to(dev).double()
    st, _, bx, _ = _tiny_gen_state(model, 4, dev)
    spec = np.array([1.0, -1.0])
    root = float(root_bound(st, spec, 0.0))
    x_L, x_U = bx.ptb.x_L, bx.ptb.x_U
    xs = x_L + (x_U - x_L) * torch.rand(20000, 4, dtype=_DT, device=dev)
    fx = model(xs) @ torch.tensor(spec, dtype=_DT, device=dev)
    assert root <= float(fx.min()) + 1e-9, (root, float(fx.min()))
    assert st['n_gens'] > st['n_input'], st


def test_build_gen_state_refuses_a_graph_dynamic_forward_cannot_propagate():
    class Cat(torch.nn.Module):
        """Concat has no dynamic-forward construction."""

        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.a = torch.nn.Linear(4, 5)
            self.out = torch.nn.Linear(9, 2)

        def forward(self, x):
            return self.out(torch.cat([torch.relu(self.a(x)), x], dim=1))

    dev = _device()
    with pytest.raises(NotImplementedError):
        _tiny_gen_state(Cat().to(dev).double(), 4, dev)


def test_the_slope_search_and_the_pass_agree_on_the_root_bound():
    """The optimizer's objective and the extracted rows are the same number.

    The search reads the bound off a reverse pass over the relaxed network at a
    point -- one gradient, no generator matrix. The gen state reads it off the
    rows the dynamic-forward pass leaves behind, which is that matrix. Same
    relaxation, same affine function of the generators, so the two have to
    land on the same value; if they did not, the search would be tightening
    something the state does not report.
    """
    from clipper_gen_state import root_bound
    dev = _device()
    model = _Residual().to(dev).double()
    spec = np.array([1.0, -1.0])
    st, bm, bx, interm = _tiny_gen_state(
        model, 4, dev, opt_slope_qw=spec[None], opt_iters=15)
    from_rows = float(root_bound(st, spec, 0.0))
    # the same bound, but concretized by the pass instead of from the rows
    C = torch.tensor(spec, dtype=_DT, device=dev).reshape(1, 1, 2)
    from_pass = float(bm.compute_bounds(
        x=(bx,), method='dynamic-forward', C=C, interm_bounds=interm,
        bound_upper=False)[0].reshape(-1)[0])
    assert abs(from_rows - from_pass) < 1e-9, (from_rows, from_pass)


# =============================================================================
# the builder this replaced, kept as a ruler (tests/reference_gen_state.py)
# =============================================================================

def _both_gen_states(model, in_shape, device, eps=0.5):
    """The same query built each way, each on its own module.

    Separate modules because building a state leaves relaxation state on the
    operators, and the point is for the two runs to share nothing but the
    network, the box and the pre-activation bounds.
    """
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from clipper_gen_state import build_gen_state

    # by path: it sits next to this file rather than on the import path, which
    # is part of it not being importable from production code
    spec = importlib.util.spec_from_file_location(
        'reference_gen_state', Path(__file__).with_name('reference_gen_state.py'))
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    reference_build = reference.build_gen_state

    states = []
    for build in (build_gen_state, reference_build):
        # off-centre, so the input generators' own centre is exercised
        x0 = torch.linspace(-1., 1., int(np.prod(in_shape)), dtype=_DT,
                            device=device).reshape(1, *in_shape)
        bm = BoundedModule(model, x0, device=device)
        bx = BoundedTensor(x0, PerturbationLpNorm(
            eps=eps, x_L=x0 - eps, x_U=x0 + eps))
        bm.compute_bounds(x=(bx,), method='backward')
        ret = {'lower_bounds': {r.inputs[0].name: r.inputs[0].lower.detach()
                                for r in bm.relus},
               'upper_bounds': {r.inputs[0].name: r.inputs[0].upper.detach()
                                for r in bm.relus}}
        wrap = type('N', (), {})()
        wrap.net = bm
        wrap.final_name = bm.final_name
        wrap.device = device
        states.append(build(wrap, ret, bx, device, dtype=_DT, verbose=False))
    return states


def _manual_concretized_bounds(state, specs):
    """Concretize affine output rows over the state generator box.

    Keep this separate from ``root_bound``: the comparison below is meant to
    exercise the canonical G/center representation and both sides of the
    generator interval directly, rather than call a builder-specific helper.
    """
    specs = torch.as_tensor(specs, dtype=state['output_gen_coeffs'].dtype,
                            device=state['output_gen_coeffs'].device)
    coeffs = specs @ state['output_gen_coeffs']
    centers = specs @ state['output_center']
    e_lo, e_hi = state['e_lb'], state['e_hi']
    lower = centers + torch.minimum(coeffs * e_lo, coeffs * e_hi).sum(dim=-1)
    upper = centers + torch.maximum(coeffs * e_lo, coeffs * e_hi).sum(dim=-1)
    return lower, upper


class _ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.c = torch.nn.Conv2d(1, 3, 3, stride=2, padding=1)
        self.l = torch.nn.Linear(3 * 4 * 4, 6)
        self.o = torch.nn.Linear(6, 2)

    def forward(self, x):
        h = torch.relu(self.c(x)).flatten(1)
        return self.o(torch.relu(self.l(h)))


def _chain_net():
    torch.manual_seed(3)
    return torch.nn.Sequential(
        torch.nn.Linear(5, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 3))


@pytest.mark.parametrize('name', ['chain', 'residual', 'conv'])
def test_forward_gen_state_matches_the_backward_reference(name):
    """One zonotope, two implementations that share no code.

    The reference reconstructs each row from a CROWN traversal per
    pre-activation; this builds them all in one forward pass. Given the same
    relaxation -- both start at the band-minimizing u/(u-l), and neither
    optimizes here -- the rows, the centres and the column each error term
    lands in all have to agree.
    """
    model, in_shape = {
        'chain': (_chain_net(), (5,)),
        'residual': (_Residual(), (4,)),
        'conv': (_ConvNet(), (1, 7, 7)),
    }[name]
    dev = _device()
    got, want = _both_gen_states(model.to(dev).double(), in_shape, dev)

    assert (got['n_gens'], got['n_input'], got['n_out']) == (
        want['n_gens'], want['n_input'], want['n_out'])
    assert got['relu_error_col'] == want['relu_error_col'], 'columns disagree'
    for key in ('output_gen_coeffs', 'output_center'):
        torch.testing.assert_close(got[key], want[key], rtol=0, atol=1e-9)

    assert len(got['unstable_list']) == len(want['unstable_list'])
    for a, b in zip(got['unstable_list'], want['unstable_list']):
        assert (a['layer_idx'], a['neuron_idx']) == (b['layer_idx'], b['neuron_idx'])
        assert a['relu_error_col'] == b['relu_error_col']
        for key in ('lo', 'hi', 'relu_slope', 'band_half_width', 'preact_center'):
            assert abs(a[key] - b[key]) < 1e-9, (name, key, a, b)
        row_a = np.zeros(got['n_gens'])
        row_a[a['row_indices']] = a['row_values']
        row_b = np.zeros(want['n_gens'])
        row_b[b['row_indices']] = b['row_values']
        assert np.abs(row_a - row_b).max() < 1e-9, (name, a['neuron_idx'])

    # Explicitly concretize several fixed specs on each state.  Include both
    # signs so this checks upper as well as lower bounds, and use the state
    # intervals rather than assuming they are [-1, 1].
    identity = np.eye(got['n_out'])
    mixed = identity - np.roll(identity, 1, axis=1)
    specs = np.concatenate((identity, -identity, mixed, -mixed), axis=0)
    for state in (got, want):
        torch.testing.assert_close(
            state['e_lb'], torch.full_like(state['e_lb'], -1.0), rtol=0, atol=0)
        torch.testing.assert_close(
            state['e_hi'], torch.full_like(state['e_hi'], 1.0), rtol=0, atol=0)
    want_lower, want_upper = _manual_concretized_bounds(want, specs)
    from clipper_gen_state import root_bound
    got_lower = root_bound(got, specs, 0.0)
    got_upper = -root_bound(got, -specs, 0.0)
    torch.testing.assert_close(got_lower, want_lower, rtol=0, atol=1e-9)
    torch.testing.assert_close(got_upper, want_upper, rtol=0, atol=1e-9)


def test_build_gen_state_refuses_an_unrepresented_relaxation_upstream_of_a_relu():
    """A pre-activation row is an equality only if every gap on its path owns a
    column. ReLU is the only inexact operator dynamic forward has a
    construction for, so that holds today -- but if one is ever added for
    another, its gap would sit in the bias with nothing to hold it, and a split
    would read the row as an equality and cut real points. Refuse instead.
    """
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.operators.relu import BoundRelu
    from auto_LiRPA.operators.s_shaped import BoundSigmoid
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from clipper_gen_state import build_gen_state

    dev = _device()
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 6), torch.nn.Sigmoid(),
        torch.nn.Linear(6, 6), torch.nn.ReLU(),
        torch.nn.Linear(6, 2)).to(dev)
    # float32: the guard is dtype-agnostic, and a curved activation's relaxation
    # does not survive float64 on this base
    x0 = torch.zeros(1, 4, device=dev)
    bm = BoundedModule(model, x0, device=dev)
    bx = BoundedTensor(x0, PerturbationLpNorm(eps=0.5, x_L=x0 - 0.5, x_U=x0 + 0.5))
    bm.compute_bounds(x=(bx,), method='backward')
    ret = {'lower_bounds': {r.inputs[0].name: r.inputs[0].lower.detach()
                            for r in bm.relus},
           'upper_bounds': {r.inputs[0].name: r.inputs[0].upper.detach()
                            for r in bm.relus}}
    wrap = type('N', (), {})()
    wrap.net = bm
    wrap.final_name = bm.final_name
    wrap.device = dev
    # stand in for the day someone gives the sigmoid a construction: without it
    # the pass refuses first and the guard is never reached
    BoundSigmoid.bound_dynamic_forward = BoundRelu.bound_dynamic_forward
    try:
        with pytest.raises(NotImplementedError,
                           match='without a generator column'):
            build_gen_state(wrap, ret, bx, dev, verbose=False)
    finally:
        del BoundSigmoid.bound_dynamic_forward


def test_verify_spec_does_not_certify_without_a_split():
    """With nothing to split the root bound is the whole answer. A property
    false at a sampled input must come back 'unknown', not fall through the
    empty BaB loop to 'safe'. An all-stable ReLU net reaches S = 0."""
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from clipper_gen_state import build_gen_state
    from clipper_dual_kernel import verify_spec
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.ReLU(),
                                torch.nn.Linear(6, 2)).double()
    with torch.no_grad():
        model[0].bias.fill_(10.0)           # every ReLU active over the box
    x0 = torch.zeros(1, 3, dtype=_DT)
    bm = BoundedModule(model, x0, device='cpu')
    bx = BoundedTensor(x0, PerturbationLpNorm(
        norm=float('inf'), x_L=x0 - 1.5, x_U=x0 + 1.5))
    bm.compute_bounds(x=(bx,), method='backward')
    ret = {'lower_bounds': {r.inputs[0].name: r.inputs[0].lower.detach()
                            for r in bm.relus},
           'upper_bounds': {r.inputs[0].name: r.inputs[0].upper.detach()
                            for r in bm.relus}}
    wrap = type('N', (), {})()
    wrap.net, wrap.final_name = bm, bm.final_name
    state = build_gen_state(wrap, ret, bx, 'cpu', dtype=_DT, verbose=False)
    assert len(state['unstable_list']) == 0
    pts = x0 + (torch.rand(50000, 3, dtype=_DT,
                           generator=torch.Generator().manual_seed(0)) * 2 - 1) * 1.5
    with torch.no_grad():
        smin = float((model(pts) @ torch.tensor([1., -1.], dtype=_DT)).min())
    verdict, info = verify_spec(state, np.array([1., -1.]), -smin - 0.05, 'cpu',
                                time_limit=5, K=1, bab_dtype=_DT, verbose=False)
    assert verdict == 'unknown', info
