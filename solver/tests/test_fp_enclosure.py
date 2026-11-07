"""Concretized bounds enclose exact rational values, including cancellation."""
from fractions import Fraction

import pytest
import torch
from torch import nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.concretize_func import constraints_solving, sort_out_constr_batches
from auto_LiRPA.fp_enclosure import box_rounding_error
from auto_LiRPA.patches import Patches
from auto_LiRPA.operators.linear import BoundLinear
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import eyeC


def _exact_dot(a, x, bias=0):
    return sum((Fraction(float(ai)) * Fraction(float(xi))
                for ai, xi in zip(a.flatten(), x.flatten())), Fraction(bias))


def _encloses(lower, upper, exact):
    assert Fraction(float(lower)) <= exact <= Fraction(float(upper))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('norm', [torch.inf, 2])
def test_matrix_cancellation_encloses_exact_value(dtype, norm):
    large = 2 ** (26 if dtype == torch.float32 else 56)
    x = torch.tensor([[large, 1, -large]], dtype=dtype)
    A = torch.ones(1, 1, 3, dtype=dtype)
    ptb = PerturbationLpNorm(norm=norm, eps=0, x_L=x, x_U=x)
    lb = ptb.concretize(x, A, -1)
    ub = ptb.concretize(x, A, 1)
    exact = _exact_dot(A, x)
    assert exact == 1
    _encloses(lb[0, 0], ub[0, 0], exact)
    error = (A.abs() @ x.abs().unsqueeze(-1)).squeeze(-1)
    error *= (3 + 16) * torch.finfo(dtype).eps * 4
    center = (A @ x.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(lb, center - error, rtol=0, atol=0)
    torch.testing.assert_close(ub, center + error, rtol=0, atol=0)
    assert lb.dtype == dtype and ub.dtype == dtype


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('norm', [torch.inf, 2])
def test_patch_cancellation_encloses_exact_value(dtype, norm):
    large = 2 ** (26 if dtype == torch.float32 else 56)
    x = torch.tensor([[[[large, 1, -large]]]], dtype=dtype)
    patches = torch.ones(1, 1, 1, 1, 1, 1, 3, dtype=dtype)
    A = Patches(patches, stride=1, padding=0, shape=patches.shape,
                output_shape=(1, 1, 1, 1))
    ptb = PerturbationLpNorm(norm=norm, eps=0, x_L=x, x_U=x)
    lb = ptb.concretize(x, A, -1)
    ub = ptb.concretize(x, A, 1)
    _encloses(lb.flatten()[0], ub.flatten()[0], _exact_dot(patches, x))
    assert lb.dtype == dtype and ub.dtype == dtype


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_identity_and_nonfinite_terms_do_not_create_nan(dtype):
    x = torch.tensor([[1., -2.]], dtype=dtype)
    A = eyeC((1, 2, 2), x.device)
    ptb = PerturbationLpNorm(x_L=x, x_U=x)
    lb, ub = ptb.concretize(x, A, -1), ptb.concretize(x, A, 1)
    assert (lb < x).all() and (ub > x).all()
    lower = torch.tensor([[-torch.inf, 3]], dtype=dtype)
    upper = torch.tensor([[torch.inf, 4]], dtype=dtype)
    error = box_rounding_error(torch.tensor([[[0., 1.]]], dtype=dtype), lower, upper)
    assert torch.isfinite(error).all() and (error > 0).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('method', ['IBP', 'CROWN', 'forward'])
def test_affine_bias_cancellation_encloses_exact_value(dtype, method):
    large = 2 ** (26 if dtype == torch.float32 else 56)
    model = nn.Linear(2, 1).to(dtype=dtype)
    with torch.no_grad():
        model.weight.fill_(1)
        model.bias.fill_(-large)
    x = torch.tensor([[large, 1]], dtype=dtype)
    net = BoundedModule(model, x, device='cpu')
    bx = BoundedTensor(x, PerturbationLpNorm(x_L=x, x_U=x))
    lb, ub = net.compute_bounds(x=(bx,), method=method)
    _encloses(lb[0, 0], ub[0, 0], _exact_dot(model.weight, x, -large))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_constraint_cancellation_does_not_discard_a_feasible_box(dtype):
    large = 2 ** (26 if dtype == torch.float32 else 56)
    x = torch.tensor([[large, 1, large]], dtype=dtype)
    A = torch.tensor([[[1., -1., -1.]]], dtype=dtype)
    b = torch.tensor([[0.5]], dtype=dtype)
    assert _exact_dot(A, x, Fraction(1, 2)) < 0
    constraints, batches = sort_out_constr_batches(x, x, (A, b), no_return_inf=True)
    assert batches['infeasible_batches'] is None
    assert batches['active_indices'].tolist() == [0]
    objective = torch.ones(1, 1, 3, dtype=dtype)
    for sign in (-1, 1):
        bound, infeasible = constraints_solving(
            x, x, objective, constraints, sign, sorted_out_batches=batches,
            no_return_inf=True)
        assert infeasible is None
        exact = _exact_dot(objective, x)
        assert sign * Fraction(float(bound[0, 0, 0])) >= sign * exact


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_constrained_bounds_enclose_feasible_vertices(dtype):
    lower = torch.zeros(1, 2, dtype=dtype)
    upper = torch.ones_like(lower)
    A = torch.tensor([[[1., 1.]]], dtype=dtype)
    b = torch.tensor([[-1.]], dtype=dtype)
    objective = torch.tensor([[[1e5, -1e5], [1., 2.]]], dtype=dtype)
    constraints, batches = sort_out_constr_batches(lower, upper, (A, b))
    lb = constraints_solving(lower, upper, objective, constraints, -1,
                             sorted_out_batches=batches)
    ub = constraints_solving(lower, upper, objective, constraints, 1,
                             sorted_out_batches=batches)
    for vertex in ([0, 0], [1, 0], [0, 1]):
        for j, row in enumerate(objective[0]):
            _encloses(lb[0, j, 0], ub[0, j, 0],
                      _exact_dot(row, torch.tensor(vertex, dtype=dtype)))


@pytest.mark.parametrize('sign', [-1, 1])
def test_constrained_enclosure_retains_finite_gradients(sign):
    lower = torch.zeros(1, 2, dtype=torch.float64)
    upper = torch.ones_like(lower)
    A = torch.tensor([[[1., 1.]]], dtype=lower.dtype)
    b = torch.tensor([[-1.]], dtype=lower.dtype)
    constraints, batches = sort_out_constr_batches(lower, upper, (A, b))
    objective = torch.tensor([[[1., 2.]]], dtype=lower.dtype, requires_grad=True)
    bound = constraints_solving(lower, upper, objective, constraints, sign,
                                sorted_out_batches=batches, use_grad=True)
    bound.sum().backward()
    assert objective.grad is not None and torch.isfinite(objective.grad).all()
    assert torch.isfinite(bound).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_structured_lp_ibp_encloses_cancellation(dtype):
    large = 2 ** (26 if dtype == torch.float32 else 56)
    x = torch.tensor([[large, 1, -large]], dtype=dtype)
    weight = torch.ones(1, 3, dtype=dtype)
    bounds = BoundLinear._bound_input_perturbation(
        x, PerturbationLpNorm(norm=2, eps=0), weight)
    _encloses(bounds.lower[0, 0], bounds.upper[0, 0], _exact_dot(weight, x))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_overflowing_finite_dot_product_has_vacuous_endpoints(dtype):
    large = 2 ** (100 if dtype == torch.float32 else 600)
    x = torch.tensor([[large]], dtype=dtype)
    A = x.unsqueeze(1)
    ptb = PerturbationLpNorm(x_L=x, x_U=x)
    assert torch.isneginf(ptb.concretize(x, A, -1)).all()
    assert torch.isposinf(ptb.concretize(x, A, 1)).all()


def test_box_rounding_error_handles_an_empty_batch():
    # kall_circles_c7a: constrained concretization with zero active domains reshaped an
    # empty tensor to (0, -1, 1), which is ambiguous, and crashed the run.
    import torch
    from auto_LiRPA.fp_enclosure import box_rounding_error
    A = torch.zeros(0, 4, 3, dtype=torch.float64)
    x_L = torch.zeros(0, 3, 1, dtype=torch.float64)
    x_U = torch.ones(0, 3, 1, dtype=torch.float64)
    err = box_rounding_error(A, x_L, x_U)
    assert err.shape == (0, 4)


def test_ibp_sum_of_squares_keeps_its_exact_zero_lower_bound():
    # dea: 1 / (sum of squares) after operand-magnitude widening of each add saw an
    # interval [-2e-9, 5e4] around its pole and the objective bound became -inf.
    class Model(nn.Module):
        def forward(self, x):
            s = x[:, 0:1] * x[:, 0:1] + x[:, 1:2] * x[:, 1:2] + x[:, 2:3] * x[:, 2:3]
            return s + 1.0
    lower = torch.tensor([[0., 0., 0.]], dtype=torch.float64)
    upper = torch.tensor([[100., 100., 100.]], dtype=torch.float64)
    net = BoundedModule(Model(), lower, device='cpu')
    bx = BoundedTensor(lower, PerturbationLpNorm(x_L=lower, x_U=upper))
    net.compute_bounds(x=(bx,), method='IBP')
    sums = [n for n in net.nodes() if type(n).__name__ == 'BoundAdd']
    assert sums and float(sums[-2].lower.min()) == 0.0
    lb, _ = net.compute_bounds(x=(bx,), method='IBP')
    assert 1.0 - 1e-12 < float(lb.min()) <= 1.0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_ibp_add_sub_chain_encloses_cancellation(dtype):
    large = 2 ** (26 if dtype == torch.float32 else 56)

    class Model(nn.Module):
        def forward(self, x):
            return (x[:, 0:1] + x[:, 1:2]) - x[:, 0:1].detach() * 0 - x[:, 2:3]
    x = torch.tensor([[large, 1., large]], dtype=dtype)
    net = BoundedModule(Model(), x, device='cpu')
    bx = BoundedTensor(x, PerturbationLpNorm(x_L=x, x_U=x))
    lb, ub = net.compute_bounds(x=(bx,), method='IBP')
    _encloses(lb[0, 0], ub[0, 0], Fraction(1))


AFFINE_BACKWARD = {'fp_enclosure': {'affine_backward': True}}


class _Fold(nn.Module):
    """f = sum((1e16, -1, -1e16) x) = -x: the A.W product of the second Linear's backward cancels (A67)."""

    def __init__(self):
        super().__init__()
        self.register_buffer('w1', torch.tensor([[1e16], [-1.0], [-1e16]], dtype=torch.float64))
        self.register_buffer('w2', torch.ones(1, 3, dtype=torch.float64))

    def forward(self, x):
        return (x @ self.w1.t()) @ self.w2.t()


@pytest.mark.parametrize('method', ['CROWN', 'CROWN-Optimized'])
@pytest.mark.parametrize('box', [(1.0, 1.0), (0.5, 1.5)])
def test_affine_backward_encloses_the_linear_fold(method, box):
    lo, hi = box
    x_L, x_U = torch.tensor([[lo]], dtype=torch.float64), torch.tensor([[hi]], dtype=torch.float64)
    x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U))
    default = BoundedModule(_Fold(), x_L, device='cpu').compute_bounds(x=(x,), method=method)
    lb, ub = BoundedModule(_Fold(), x_L, device='cpu', bound_opts=AFFINE_BACKWARD).compute_bounds(
        x=(x,), method=method)
    assert not (Fraction(float(default[0])) <= -Fraction(hi) and Fraction(float(default[1])) >= -Fraction(lo))  # (off by default)
    assert Fraction(float(lb)) <= -Fraction(hi) and Fraction(float(ub)) >= -Fraction(lo)
    off = BoundedModule(_Fold(), x_L, device='cpu', bound_opts={'fp_enclosure': {'affine_backward': False}}
                        ).compute_bounds(x=(x,), method=method)
    assert all(torch.equal(a, b) for a, b in zip(default, off))
    assert not (Fraction(float(off[0])) <= -Fraction(hi) and Fraction(float(off[1])) >= -Fraction(lo))


def _cancelling(rng, n):
    big = rng.choice((1, -1)) * 2.0 ** rng.randint(20, 53) * (1 + rng.random())
    v = [big if i % 3 == 0 else -big if i % 3 == 2 else rng.uniform(-3, 3) for i in range(n)]
    rng.shuffle(v)
    return v


class _Affine(nn.Module):
    """A random affine program over x (values v[0] = x, v[1], ...) with cancelling constants."""

    def __init__(self, rng, d):
        super().__init__()
        self.steps, widths = [], [d]
        for i in range(rng.randint(4, 8)):
            s = rng.randrange(len(widths))
            n = widths[s]
            op = rng.choice(('linear', 'matmul', 'gather', 'mulc', 'add', 'index_add', 'mean', 'sum'))
            if op in ('linear', 'matmul'):
                m = rng.randint(2, 4)
                # Rows that cancel (the composition) or columns that cancel (the backward's A.W sums).
                W = [_cancelling(rng, n) for _ in range(m)]
                if rng.random() < 0.5:
                    W = [list(r) for r in zip(*[_cancelling(rng, m) for _ in range(n)])]
                self.register_buffer(f'W{i}', torch.tensor(W, dtype=torch.float64))
                self.register_buffer(f'b{i}', torch.tensor(_cancelling(rng, m), dtype=torch.float64))
                args = {'W': W, 'b': self.get_buffer(f'b{i}').tolist() if op == 'linear' else [0.0] * m}
            elif op == 'gather' or op == 'index_add':
                m = rng.randint(2, 4)
                idx = [rng.randrange(n) for _ in range(m)]
                idx[-1] = idx[0]
                self.register_buffer(f'i{i}', torch.tensor(idx))
                args = {'idx': idx}
                if op == 'index_add':
                    q = [j for j, w in enumerate(widths) if w == m]
                    if not q:
                        continue
                    args.update(q=rng.choice(q), alpha=2.5)
                    m = n
            elif op == 'mulc':
                m, c = n, _cancelling(rng, n)
                self.register_buffer(f'c{i}', torch.tensor(c, dtype=torch.float64))
                args = {'c': c}
            elif op == 'add':
                q = [j for j, w in enumerate(widths) if w == n]
                m, args = n, {'q': rng.choice(q)}
            else:
                m, args = 1, {}
            self.steps.append((i, op, s, args))
            widths.append(m)
        self.width = widths[-1]

    def forward(self, x):
        v = [x]
        for i, op, s, a in self.steps:
            if op == 'linear':
                y = nn.functional.linear(v[s], self.get_buffer(f'W{i}'), self.get_buffer(f'b{i}'))
            elif op == 'matmul':
                y = v[s].matmul(self.get_buffer(f'W{i}').t())
            elif op == 'gather':
                y = torch.index_select(v[s], 1, self.get_buffer(f'i{i}'))
            elif op == 'index_add':
                y = v[s].index_add(1, self.get_buffer(f'i{i}'), v[a['q']], alpha=a['alpha'])
            elif op == 'mulc':
                y = v[s] * self.get_buffer(f'c{i}')
            elif op == 'add':
                y = v[s] + v[a['q']]
            elif op == 'mean':
                y = v[s].mean(dim=1, keepdim=True)
            else:
                y = v[s].sum(dim=1, keepdim=True)
            v.append(y)
        return v[-1]

    def exact(self, d):
        """Every output as (coefficients over x, constant) in Fractions."""
        F = Fraction
        v = [[([F(int(i == j)) for j in range(d)], F(0)) for i in range(d)]]
        scale = lambda c, f: ([c * t for t in f[0]], c * f[1])
        add = lambda f, g: ([s + t for s, t in zip(f[0], g[0])], f[1] + g[1])
        for i, op, s, a in self.steps:
            src = v[s]
            if op in ('linear', 'matmul'):
                out = []
                for row, b in zip(a['W'], a['b']):
                    f = ([F(0)] * d, F(b))
                    for w, t in zip(row, src):
                        f = add(f, scale(F(w), t))
                    out.append(f)
            elif op == 'gather':
                out = [src[j] for j in a['idx']]
            elif op == 'index_add':
                out = list(src)
                for k, j in enumerate(a['idx']):
                    out[j] = add(out[j], scale(F(a['alpha']), v[a['q']][k]))
            elif op == 'mulc':
                out = [scale(F(c), t) for c, t in zip(a['c'], src)]
            elif op == 'add':
                out = [add(f, g) for f, g in zip(src, v[a['q']])]
            else:
                f = ([F(0)] * d, F(0))
                for t in src:
                    f = add(f, t)
                out = [scale(F(1, len(src)) if op == 'mean' else F(1), f)]
            v.append(out)
        return v[-1]


@pytest.mark.parametrize('seed', range(200))
def test_affine_backward_encloses_random_cancellations(seed):
    import random
    rng = random.Random(seed)
    d = rng.randint(1, 3)
    model = _Affine(rng, d)
    c = [rng.uniform(-4, 4) for _ in range(d)]
    boxes = [(c, c), ([t - 1 for t in c], [t + 1 for t in c])]
    x_L = torch.tensor([b[0] for b in boxes], dtype=torch.float64)
    x_U = torch.tensor([b[1] for b in boxes], dtype=torch.float64)
    forms = model.exact(d)
    x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U))
    plain = BoundedModule(model, x_L, device='cpu')
    net = BoundedModule(model, x_L, device='cpu', bound_opts=AFFINE_BACKWARD)
    for method in ('CROWN', 'CROWN-Optimized'):
        lb0, ub0 = plain.compute_bounds(x=(x,), method=method)
        lb, ub = net.compute_bounds(x=(x,), method=method)
        for b, (lo, hi) in enumerate(boxes):
            for o, (coef, const) in enumerate(forms):
                mn = const + sum(min(a * Fraction(l), a * Fraction(h)) for a, l, h in zip(coef, lo, hi))
                mx = const + sum(max(a * Fraction(l), a * Fraction(h)) for a, l, h in zip(coef, lo, hi))
                assert Fraction(float(lb[b, o])) <= mn and Fraction(float(ub[b, o])) >= mx
                # The enclosure adds rounding-sized slack, not infinities.
                assert torch.isfinite(lb[b, o]) == torch.isfinite(lb0[b, o])
                assert torch.isfinite(ub[b, o]) == torch.isfinite(ub0[b, o])


def test_affine_backward_magnitude_follows_in_place_bound_updates():
    # _magnitude keeps a node's magnitude for the next passes with the same bound tensors; an in-place update of
    # those bounds must not reuse it.
    from auto_LiRPA.backward_bound import _magnitude

    class Model(nn.Module):
        def forward(self, x):
            return (x * 3.0) ** 2

    x_L = torch.tensor([[-1.0, 2.0]], dtype=torch.float64)
    x_U = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
    net = BoundedModule(Model(), x_L, device='cpu', bound_opts=AFFINE_BACKWARD)
    net.compute_bounds(x=(BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)),), method='CROWN')
    node = next(n for n in net.nodes() if type(n).__name__ == 'BoundMul')
    before = _magnitude(net, node, 1, {})[0].clone()
    node.upper.mul_(2)
    after = _magnitude(net, node, 1, {})[0]
    assert torch.equal(after, torch.maximum(-node.lower, node.upper)) and not torch.equal(before, after)
