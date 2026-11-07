"""Reading a zonotope off a dynamic-forward pass.

Dynamic forward already propagates one: an affine form in the input plus a
column per relaxation error, each column on its own interval. So there is
nothing to reconstruct and nothing to keep in step with a second propagation --
`zonotope_rows` just puts the columns on [-1, 1]. These tests pin the two things
that makes it: the rows have to describe the same set the pass concretized, and
that set has to contain the network.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

_DT = torch.float64


def _chain(dtype=_DT):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 6, dtype=dtype), nn.ReLU(),
                         nn.Linear(6, 6, dtype=dtype), nn.ReLU(),
                         nn.Linear(6, 2, dtype=dtype))


def _conv(dtype=_DT):
    torch.manual_seed(3)
    return nn.Sequential(nn.Conv2d(1, 3, 3, stride=2, padding=1, dtype=dtype),
                         nn.ReLU(), nn.Flatten(),
                         nn.Linear(3 * 4 * 4, 6, dtype=dtype), nn.ReLU(),
                         nn.Linear(6, 2, dtype=dtype))


def _run(model, shape, max_dim=int(1e9), dtype=_DT, eps=1.0):  # noqa: E302
    x_L = torch.full(shape, -eps, dtype=dtype)
    x_U = torch.full(shape, eps, dtype=dtype)
    centre = (x_L + x_U) / 2
    bm = BoundedModule(model, centre, device='cpu',
                       bound_opts={'forward_max_dim': max_dim})
    bx = BoundedTensor(centre, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))
    lower, upper = bm.compute_bounds(x=(bx,), method='dynamic-forward')
    return bm, lower, upper


def _nodes(bm):
    """Every pre-activation, then the output -- what a verifier asks for."""
    return [r.inputs[0] for r in bm.relus] + [bm.final_node()]


@pytest.mark.parametrize('name', ['chain', 'conv'])
def test_the_rows_describe_what_the_pass_concretized(name):
    """The rows are the pass's own bound, so concretizing them repeats it."""
    model, shape = ((_chain(), (1, 3)) if name == 'chain'
                    else (_conv(), (1, 1, 7, 7)))
    bm, lower, upper = _run(model, shape)
    rows, centre = bm.zonotope_rows(bm.final_node())
    reach = rows.abs().sum(1)
    torch.testing.assert_close(centre - reach, lower.reshape(-1),
                               rtol=0, atol=1e-12)
    torch.testing.assert_close(centre + reach, upper.reshape(-1),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize('name', ['chain', 'conv'])
def test_the_rows_contain_the_network(name):
    """Whatever they describe still has to be an over-approximation."""
    model, shape = ((_chain(), (1, 3)) if name == 'chain'
                    else (_conv(), (1, 1, 7, 7)))
    bm, _, _ = _run(model, shape)
    rows, centre = bm.zonotope_rows(bm.final_node())
    lower = centre - rows.abs().sum(1)
    upper = centre + rows.abs().sum(1)
    pts = torch.rand(2000, *shape[1:], dtype=_DT) * 2 - 1
    with torch.no_grad():
        v = model(pts).reshape(2000, -1)
    assert (lower <= v.min(0).values + 1e-9).all()
    assert (upper >= v.max(0).values - 1e-9).all()


def test_a_point_in_the_zonotope_is_reached_by_the_network():
    """The generators are the real degrees of freedom, not a bounding box.

    Reading the input columns as a point in the box and every error column at
    zero has to land on the network's own value there -- so the rows really are
    the affine form the relaxation left behind.
    """
    model = _chain()
    bm, _, _ = _run(model, (1, 3))
    rows, centre = bm.zonotope_rows(bm.final_node())
    n_input = 3
    for _ in range(20):
        e = torch.rand(n_input, dtype=_DT) * 2 - 1
        # every ReLU is exact at a point where none of them is unstable is not
        # something we can arrange, so compare against the relaxation's own
        # value: the error columns at zero give the midpoint line.
        z = centre + rows[:, :n_input].matmul(e)
        with torch.no_grad():
            v = model(e.reshape(1, 3)).reshape(-1)
        band = rows[:, n_input:].abs().sum(1)
        assert ((z - band <= v + 1e-9) & (v <= z + band + 1e-9)).all()


@pytest.mark.parametrize('name', ['chain', 'conv'])
@pytest.mark.parametrize('window', ['one', 'half', 'all-but-one'])
def test_the_rows_survive_windowing(name, window):
    """`forward_max_dim` splits the columns into windows; a row is still whole.

    A window carries at most `max_dim` columns and an operator emits only the
    new ones that fall inside it, leaving the rest to the window that owns them.
    A caller handed one window would get a zonotope missing generators, which is
    narrower than the real one -- unsound. However the columns are split, the
    rows have to come out the same.
    """
    model, shape = ((_chain(), (1, 3)) if name == 'chain'
                    else (_conv(), (1, 1, 7, 7)))
    bm, _, _ = _run(model, shape)
    want = {n.name: bm.zonotope_rows(n) for n in _nodes(bm)}
    n_gens = max(r.shape[1] for r, _ in want.values())
    max_dim = {'one': 1, 'half': max(1, n_gens // 2),
               'all-but-one': n_gens - 1}[window]
    assert max_dim < n_gens, 'the window has to actually cut something'

    bm_w, _, _ = _run(model, shape, max_dim=max_dim)
    for node in _nodes(bm_w):
        rows, centre = bm_w.zonotope_rows(node)
        rows_want, centre_want = want[node.name]
        assert rows.shape == rows_want.shape, (node.name, rows.shape)
        torch.testing.assert_close(rows, rows_want, rtol=0, atol=1e-12)
        torch.testing.assert_close(centre, centre_want, rtol=0, atol=1e-12)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_it_keeps_the_model_dtype(dtype):
    """The columns a ReLU opens join the ones already there, dtype included."""
    bm, _, _ = _run(_chain(dtype), (1, 3), dtype=dtype)
    rows, centre = bm.zonotope_rows(bm.final_node())
    assert rows.dtype is dtype and centre.dtype is dtype


def test_a_list_of_nodes_is_read_from_one_pass():
    """The batch form returns {name: (rows, centre)} for several nodes at once
    and leaves nothing cached -- the same rows the one-at-a-time form gives.
    """
    model = _chain()

    def nodes_of(bm):
        return [bm.final_node()] + [r.inputs[0] for r in bm.relus]

    bm, _, _ = _run(model, (1, 3))
    want = {n.name: bm.zonotope_rows(n) for n in nodes_of(bm)}

    bm, _, _ = _run(model, (1, 3))
    got = bm.zonotope_rows(nodes_of(bm))
    assert set(got) == set(want)
    for name in want:
        for a, b in zip(got[name], want[name]):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
    assert not any(hasattr(n, 'linear') for n in bm.nodes()), 'left a bound cached'


def test_every_pre_activation_has_one_too():
    """A verifier wants the rows of the neurons it may split, not just the
    output -- and asking in graph order re-uses what the last ask left behind.
    """
    model = _chain()
    bm, _, _ = _run(model, (1, 3))
    n_gens = None
    for relu in bm.relus:
        node = relu.inputs[0]
        rows, centre = bm.zonotope_rows(node)
        numel = int(np.prod([int(d) for d in node.output_shape[1:]]))
        assert rows.shape[0] == numel and centre.shape[0] == numel
        # one column space, so a later node reaches at least as many columns
        assert n_gens is None or rows.shape[1] >= n_gens
        n_gens = rows.shape[1]
        reach = rows.abs().sum(1)
        torch.testing.assert_close(centre - reach, node.lower.reshape(-1),
                                   rtol=0, atol=1e-12)
        torch.testing.assert_close(centre + reach, node.upper.reshape(-1),
                                   rtol=0, atol=1e-12)


def test_only_a_dynamic_pass_is_read_as_a_zonotope():
    """An ordinary forward leaves a LinearBound too, and it is not a zonotope.

    It is a pair of one-sided lines, so reading it as `c + G e` takes the lower
    one for both and describes a set that need not contain the network -- and
    nothing downstream can tell it from one that does. So the readers ask the
    perturbed root for the propagation's generator book first: compute_bounds
    builds the roots' bounds fresh every call and only the dynamic branch
    seeds one, so carrying a book is exactly "this is a dynamic pass".
    """
    model = _chain()
    bm, _, _ = _run(model, (1, 3))
    bm.zonotope_rows(bm.final_node())              # a dynamic pass reads fine

    x_L, x_U = (torch.full((1, 3), -1., dtype=_DT),
                torch.full((1, 3), 1., dtype=_DT))
    bx = BoundedTensor((x_L + x_U) / 2, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))
    bm.compute_bounds(x=(bx,), method='forward')
    for read in (lambda: bm.zonotope_rows(bm.final_node()),
                 lambda: bm.zonotope_bound(
                     bm.final_node(), torch.eye(2, dtype=_DT))):
        with pytest.raises(RuntimeError, match='no zonotope to read'):
            read()


def test_a_batch_of_boxes_is_refused_by_both_readers():
    """A zonotope describes one input box, so a batch of them is not one read.

    zonotope_bound flattens the batch into the output dimension, so a batched
    C reshapes into rows that span two boxes at once and the number it hands
    back is one box's objective added to another's -- a bound on neither, and
    indistinguishable from a real one. zonotope_rows already refuses; this has
    to refuse the same way rather than answer.
    """
    model = _chain()
    x = torch.tensor([[0.1, 0.2, 0.3], [-0.3, 0.0, 0.4]], dtype=_DT)
    eps = 0.25
    bm = BoundedModule(model, x, device='cpu')
    bx = BoundedTensor(x, PerturbationLpNorm(
        norm=float('inf'), x_L=x - eps, x_U=x + eps))
    C = torch.eye(2, dtype=_DT).expand(x.shape[0], 2, 2)
    bm.compute_bounds(x=(bx,), method='dynamic-forward', C=C)
    for read in (lambda: bm.zonotope_rows(bm.final_node()),
                 lambda: bm.zonotope_bound(bm.final_node(), C)):
        with pytest.raises(ValueError, match='one input box'):
            read()

    # one box at a time is the read that was being asked for, and it is right
    for i in range(x.shape[0]):
        bm_one = BoundedModule(model, x[i:i + 1], device='cpu')
        bx_one = BoundedTensor(x[i:i + 1], PerturbationLpNorm(
            norm=float('inf'), x_L=x[i:i + 1] - eps, x_U=x[i:i + 1] + eps))
        lower, _ = bm_one.compute_bounds(
            x=(bx_one,), method='dynamic-forward',
            C=torch.eye(2, dtype=_DT).unsqueeze(0))
        bound = bm_one.zonotope_bound(
            bm_one.final_node(), torch.eye(2, dtype=_DT))
        torch.testing.assert_close(bound, lower.reshape(-1),
                                   rtol=0, atol=1e-12)


def test_a_backward_alpha_is_not_read_as_an_imposed_line():
    """A line is imposed by installing one, not by being in an opt stage.

    CROWN puts its activations in one to optimize its own alphas, which are
    keyed by start node. A pass that took the stage for the line would read a
    key that is not there and raise, on a module in the state alpha-CROWN
    leaves -- which is the state a caller reaching for this is most likely in.
    """
    model = _chain()
    bm, want_l, want_u = _run(model, (1, 3))

    x_L, x_U = (torch.full((1, 3), -1., dtype=_DT),
                torch.full((1, 3), 1., dtype=_DT))
    bx = BoundedTensor((x_L + x_U) / 2, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))
    C = torch.eye(2, dtype=_DT).unsqueeze(0)
    bm.init_alpha(x=(bx,), method='backward', c=C)
    assert all('_forward' not in relu.alpha for relu in bm.relus)
    for relu in bm.relus:
        relu.opt_start()

    lower, upper = bm.compute_bounds(x=(bx,), method='dynamic-forward')
    torch.testing.assert_close(lower, want_l, rtol=0, atol=1e-12)
    torch.testing.assert_close(upper, want_u, rtol=0, atol=1e-12)


def _impose(bm, slope_of):
    """Relax every ReLU with a chosen line instead of u/(u-l).

    Through alpha, which is where the operator reads its line from -- the same
    parameter the two-line forward relaxation optimizes.
    """
    for relu in bm.relus:
        shape = [int(d) for d in relu.output_shape[1:]]
        line = slope_of(relu).reshape(1, *shape)
        relu.alpha = {'_forward': line.unsqueeze(0).expand(
            relu.alpha_size, 1, *line.shape)}
        relu.opt_start()


def _interm(bm):
    return {r.inputs[0].name: (r.inputs[0].lower.detach(),
                               r.inputs[0].upper.detach()) for r in bm.relus}


@pytest.mark.parametrize('slope', [0.0, 0.5, 1.0])
def test_an_imposed_slope_still_bounds_the_network(slope):
    """Any line in [0, 1] is sound -- the band covers the gap it leaves.

    An imposed line costs tightness; it must never cost soundness, including at
    the two ends, where the relaxation is a single edge of the triangle.
    """
    model = _chain()
    bm, _, _ = _run(model, (1, 3))
    interm = _interm(bm)
    _impose(bm, lambda r: torch.full(
        (int(np.prod([int(d) for d in r.output_shape[1:]])),), slope, dtype=_DT))
    bm.compute_bounds(x=(BoundedTensor(
        torch.zeros(1, 3, dtype=_DT), PerturbationLpNorm(
            norm=float('inf'), x_L=torch.full((1, 3), -1.0, dtype=_DT),
            x_U=torch.full((1, 3), 1.0, dtype=_DT))),),
        method='dynamic-forward', interm_bounds=interm)
    rows, centre = bm.zonotope_rows(bm.final_node())
    lower = centre - rows.abs().sum(1)
    upper = centre + rows.abs().sum(1)
    pts = torch.rand(2000, 3, dtype=_DT) * 2 - 1
    with torch.no_grad():
        v = model(pts)
    assert (lower <= v.min(0).values + 1e-9).all(), (lower, v.min(0).values)
    assert (upper >= v.max(0).values - 1e-9).all(), (upper, v.max(0).values)


class _ReluPlusInput(nn.Module):
    """relu(x) + x, where the line's own term cancels against the skip, so what
    reaches the output is the band alone."""

    def forward(self, x):
        return torch.relu(x) + x


def _relu_plus_input_bound(bm, interm, line):
    """`_ReluPlusInput` over x in [-1, 1], relaxed with one imposed line."""
    _impose(bm, lambda r: torch.full((1,), line, dtype=_DT))
    bx = BoundedTensor(torch.zeros(1, 1, dtype=_DT), PerturbationLpNorm(
        norm=float('inf'), x_L=torch.full((1, 1), -1.0, dtype=_DT),
        x_U=torch.full((1, 1), 1.0, dtype=_DT)))
    return bm.compute_bounds(x=(bx,), method='dynamic-forward',
                             interm_bounds=interm)


@pytest.mark.parametrize('line', [-1.0, 2.0])
def test_a_line_outside_the_unit_range_is_projected_onto_it(line):
    """The band starts at 0, which only covers the gap for a line in [0, 1].

    At slope -1 over [-1, 1] the gap is [-1, 2], and a band anchored at 0
    leaves out every point below it: the pass used to claim [0, 2] for
    relu(x) + x, which does not contain f(-1) = -1. The bound has to be the
    projection's -- which on [0, 1] is the line's own, so nothing a caller can
    legitimately ask for moves.
    """
    model = _ReluPlusInput()
    bm, _, _ = _run(model, (1, 1))
    interm = _interm(bm)
    lower, upper = _relu_plus_input_bound(bm, interm, line)

    pts = torch.linspace(-1., 1., 101, dtype=_DT).reshape(-1, 1)
    with torch.no_grad():
        v = model(pts)
    lo, hi = float(lower), float(upper)
    assert lo <= float(v.min()) + 1e-12, (lo, float(v.min()))
    assert hi >= float(v.max()) - 1e-12, (hi, float(v.max()))

    want_l, want_u = _relu_plus_input_bound(bm, interm, min(max(line, 0.), 1.))
    torch.testing.assert_close(lower, want_l, rtol=0, atol=1e-12)
    torch.testing.assert_close(upper, want_u, rtol=0, atol=1e-12)


def test_a_line_that_is_not_a_number_does_not_become_one():
    """A NaN line has no projection, and clamping does not invent one: it
    reaches the framework's own guard rather than being read as 0 or 1."""
    model = _ReluPlusInput()
    bm, _, _ = _run(model, (1, 1))
    with pytest.raises(RuntimeError, match='NaN detected'):
        _relu_plus_input_bound(bm, _interm(bm), float('nan'))


def test_the_imposed_slope_is_optimizable():
    """The bound is differentiable in the slope, which is what lets a caller
    search for the one that suits its query -- something a per-start-node alpha
    cannot express, since it would give one neuron a different slope in
    different rows of the same zonotope.
    """
    model = _chain()
    bm, _, _ = _run(model, (1, 3))
    interm = _interm(bm)
    bx = BoundedTensor(torch.zeros(1, 3, dtype=_DT), PerturbationLpNorm(
        norm=float('inf'), x_L=torch.full((1, 3), -1.0, dtype=_DT),
        x_U=torch.full((1, 3), 1.0, dtype=_DT)))
    spec = torch.tensor([[[1.0, -1.0]]], dtype=_DT)
    leaves = {r.name: (r.inputs[0].upper / (r.inputs[0].upper - r.inputs[0].lower)
                       ).clamp(1e-6, 1 - 1e-6).reshape(-1).detach().clone(
                       ).requires_grad_(True) for r in bm.relus}

    def bound():
        _impose(bm, lambda r: leaves[r.name])
        return bm.compute_bounds(x=(bx,), method='dynamic-forward', C=spec,
                                 interm_bounds=interm, bound_upper=False)[0].sum()

    start = bound()
    start.backward()
    grads = [leaves[r.name].grad for r in bm.relus]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().max()) > 0 for g in grads), 'the slope does nothing'

    opt = torch.optim.Adam(list(leaves.values()), lr=0.05)
    for _ in range(30):
        opt.zero_grad()
        (-bound()).backward()
        opt.step()
        with torch.no_grad():
            for leaf in leaves.values():
                leaf.clamp_(1e-6, 1 - 1e-6)
    assert float(bound().detach()) > float(start.detach()) + 1e-9, (
        "optimizing did not tighten")


class _Diamond(nn.Module):
    """Both branches relax, so both open error columns."""

    def __init__(self, sub=False):
        super().__init__()
        torch.manual_seed(28)
        self.l = nn.Linear(4, 2, dtype=_DT)
        self.r = nn.Linear(4, 2, dtype=_DT)
        self.lo = nn.Linear(2, 2, dtype=_DT)
        self.ro = nn.Linear(2, 2, dtype=_DT)
        self.sub = sub

    def forward(self, x):
        a = self.lo(torch.relu(self.l(x)))
        b = self.ro(torch.relu(self.r(x)))
        return a - b if self.sub else a + b


class _Residual(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(12)
        self.a = nn.Linear(4, 5, dtype=_DT)
        self.h = nn.Linear(5, 5, dtype=_DT)
        self.skip = nn.Linear(4, 5, dtype=_DT)
        self.out = nn.Linear(5, 2, dtype=_DT)

    def forward(self, x):
        return self.out(torch.relu(self.h(torch.relu(self.a(x))) + self.skip(x)))


@pytest.mark.parametrize('model', [_Diamond(), _Diamond(sub=True), _Residual()],
                         ids=['diamond-add', 'diamond-sub', 'residual'])
def test_a_dag_gets_one_column_space(model):
    """Two branches only add up if equal column positions are equal variables.

    Numbering a ReLU's columns from its own input's width -- per path -- gives
    the two branches of a diamond the same indices for different error terms,
    and adding them cancels errors that are independent. That is not loose, it
    is unsound: it cuts off points the network really attains.
    """
    bm, lower, upper = _run(model, (1, 4))
    rows, centre = bm.zonotope_rows(bm.final_node())
    reach = rows.abs().sum(1)
    torch.testing.assert_close(centre - reach, lower.reshape(-1),
                               rtol=0, atol=1e-12)
    pts = torch.rand(20000, 4, dtype=_DT) * 2 - 1
    with torch.no_grad():
        v = model(pts)
    assert (centre - reach <= v.min(0).values + 1e-9).all()
    assert (centre + reach >= v.max(0).values - 1e-9).all()
    # every column is somebody's, and nobody's twice
    blocks = sorted(bm.roots()[0].linear.dynamic_variable_ranges.values())
    assert rows.shape[1] == sum(count for _, count in blocks), (rows.shape, blocks)
    assert [s for s, _ in blocks] == list(
        np.cumsum([0] + [c for _, c in blocks])[:-1])


@pytest.mark.parametrize('max_dim', [1, 3, 5])
def test_a_dag_survives_windowing(max_dim):
    """A window is a slice of the one column space, not of each branch's own."""
    model = _Residual()
    bm, _, _ = _run(model, (1, 4))
    want = bm.zonotope_rows(bm.final_node())
    bm_w, _, _ = _run(model, (1, 4), max_dim=max_dim)
    rows, centre = bm_w.zonotope_rows(bm_w.final_node())
    torch.testing.assert_close(rows, want[0], rtol=0, atol=1e-12)
    torch.testing.assert_close(centre, want[1], rtol=0, atol=1e-12)


def test_a_dag_is_traversed_once_per_node():
    """A shared ancestor is visited once, not once per path that reaches it.

    Every residual block gives its successor two paths back to it, so without
    remembering, the block before it is recomputed twice, the one before that
    four times, and the first ReLU of an N-block net 2^(N-1) times. It only
    shows up when the caller supplies the pre-activation bounds -- otherwise
    computing them caches each layer on the way and hides it -- which is
    exactly what a verifier reading a zonotope does.
    """
    from auto_LiRPA.operators.relu import BoundRelu

    class Deep(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            torch.manual_seed(1)
            self.blocks = nn.ModuleList(
                [nn.Linear(4, 4, dtype=_DT) for _ in range(blocks)])
            self.skips = nn.ModuleList(
                [nn.Linear(4, 4, dtype=_DT) for _ in range(blocks)])
            self.out = nn.Linear(4, 2, dtype=_DT)

        def forward(self, x):
            for block, skip in zip(self.blocks, self.skips):
                x = torch.relu(block(x)) + skip(x)
            return self.out(x)

    model = Deep(5)
    bm, _, _ = _run(model, (1, 4))
    interm = _interm(bm)
    calls = {}
    real = BoundRelu.bound_dynamic_forward

    def counted(self, x, max_dim=None, offset=0):
        calls[self.name] = calls.get(self.name, 0) + 1
        return real(self, x, max_dim=max_dim, offset=offset)

    x_L = torch.full((1, 4), -1.0, dtype=_DT)
    x_U = torch.full((1, 4), 1.0, dtype=_DT)
    bx = BoundedTensor((x_L + x_U) / 2, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))
    BoundRelu.bound_dynamic_forward = counted
    try:
        bm.compute_bounds(x=(bx,), method='dynamic-forward',
                          interm_bounds=interm)
    finally:
        BoundRelu.bound_dynamic_forward = real
    assert calls and max(calls.values()) == 1, calls


@pytest.mark.parametrize('relu_layers', [1, 3])
def test_it_agrees_with_same_slope_crown(relu_layers):
    """The two directions expose the same relaxed affine object.

    CROWN's same-slope option forces the upper line parallel to the lower and
    puts its intercept through whichever vertex is worse, `max(-c*l, (1-c)*u)`
    -- which is exactly twice the zonotope's band, since the zonotope centres
    the same pair on their midpoint. So the relaxed network is one affine
    object, and propagating it forwards as a zonotope or backwards as CROWN has
    to expose the same input coefficients and to concretize to the same lower
    and upper numbers.

    The representations are not compared column-for-column: dynamic forward
    keeps one free variable for every input coordinate and every ReLU band,
    while backward folds the latter into its lower/upper bias. If ``G`` and
    ``d`` are the dynamic rows and centre, ``q = C @ G`` and ``c = C @ d``;
    the input block of ``q`` maps to CROWN's input-space A by the input radii,
    and ``c +/- |q_error|.sum(1)`` maps to the two CROWN affine forms at the
    input centre. This checks the coefficients as well as the concretization.

    The lines have to be the same lines, which means handing the backward side
    the slopes in the model's dtype: init_opt_parameters allocates alpha as
    float32 whatever the model is, and truncating into it moves the relaxation
    by about 1e-8.
    """
    torch.manual_seed(0)
    layers = [nn.Linear(4, 6, dtype=_DT), nn.ReLU()]
    for _ in range(relu_layers - 1):
        layers += [nn.Linear(6, 6, dtype=_DT), nn.ReLU()]
    layers += [nn.Linear(6, 3, dtype=_DT)]
    model = nn.Sequential(*layers)

    eps = 0.4
    x0 = torch.linspace(-1., 1., 4, dtype=_DT).reshape(1, 4)
    bm = BoundedModule(model, x0, device='cpu', bound_opts={
        'activation_bound_option': 'same-slope',
        'sparse_features_alpha': False, 'sparse_spec_alpha': False})
    bx = BoundedTensor(x0, PerturbationLpNorm(
        norm=float('inf'), x_L=x0 - eps, x_U=x0 + eps))
    bm.compute_bounds(x=(bx,), method='backward')
    interm = _interm(bm)
    assert any(int(((lo < 0) & (hi > 0)).sum()) for lo, hi in interm.values()), (
        'with nothing unstable there is no relaxation to agree about')
    C = torch.cat([
        torch.eye(3, dtype=_DT).unsqueeze(0),
        torch.tensor([[[1., -1., 0.], [0., 1., -1.]]], dtype=_DT),
    ], dim=1)

    torch.manual_seed(3)
    lines = {r.name: torch.rand(interm[r.inputs[0].name][0].shape, dtype=_DT)
             * 0.8 + 0.1 for r in bm.relus}
    for r in bm.relus:
        lo, hi = interm[r.inputs[0].name]
        # Both ends of the range a line is defined on, on neurons that are
        # actually relaxed: projecting onto that range has to be the identity
        # there, or the two directions stop agreeing at the boundary.
        unstable = ((lo < 0) & (hi > 0)).reshape(-1).nonzero().reshape(-1)
        flat = lines[r.name].reshape(-1)
        flat[unstable[:1]] = 0.
        flat[unstable[-1:]] = 1.

    for relu in bm.relus:
        line = lines[relu.name]
        relu.alpha = {'_forward': line.unsqueeze(0).expand(
            relu.alpha_size, 1, *line.shape)}
        relu.opt_start()
    # Keep the full generator matrix before applying C. The input columns are
    # scaled input coordinates; all later columns are independent ReLU errors.
    bm.compute_bounds(x=(bx,), method='dynamic-forward',
                      interm_bounds=interm)
    rows, centre = bm.zonotope_rows(bm.final_node())
    rows, centre = rows.detach(), centre.detach()

    # This is the production dynamic-forward concretization of the same C.
    forward_lower, forward_upper = bm.compute_bounds(
        x=(bx,), method='dynamic-forward', C=C, interm_bounds=interm)

    for relu in bm.relus:
        relu.opt_end()
        relu.alpha = {}
    bm.init_alpha(x=(bx,), method='backward', c=C, interm_bounds=interm)
    for relu in bm.relus:
        line = lines[relu.name]
        shape = relu.alpha[bm.final_name].shape
        relu.alpha[bm.final_name] = line.reshape(1, 1, *line.shape).expand(
            shape).clone().requires_grad_(True)
        relu.opt_start()
    backward_lower, backward_upper, A = bm.compute_bounds(
        x=(bx,), method='backward', C=C, interm_bounds=interm,
        return_A=True)

    # Canonicalize both directions to C's rows over the dynamic variables.
    # Dynamic input generators represent x = x_center + radii * e_input;
    # CROWN's A is instead expressed directly over x.
    q = C[0].matmul(rows)
    c = C[0].matmul(centre)
    n_input = x0.numel()
    input_radii = (bx.ptb.x_U - bx.ptb.x_L).reshape(-1) / 2
    q_input, q_error = q[:, :n_input], q[:, n_input:]
    canonical_lower = c - q.abs().sum(dim=1)
    canonical_upper = c + q.abs().sum(dim=1)

    input_name = next(root.name for root in bm.roots()
                      if root.perturbation is not None)
    entry = A[bm.final_name][input_name]
    lower_A = entry['lA'].reshape(q.shape[0], n_input)
    upper_A = entry['uA'].reshape(q.shape[0], n_input)
    lower_bias = entry['lbias'].reshape(-1)
    upper_bias = entry['ubias'].reshape(-1)
    x_center = x0.reshape(-1)

    # The root/input coefficient block is the part that return_A exposes.
    torch.testing.assert_close(q_input, lower_A * input_radii,
                               rtol=0, atol=1e-12)
    torch.testing.assert_close(q_input, upper_A * input_radii,
                               rtol=0, atol=1e-12)
    torch.testing.assert_close(lower_A, upper_A, rtol=0, atol=1e-12)

    # The error columns have no backward A counterpart. Their signed L1
    # reach is exactly the amount folded into each same-slope bias.
    error_lower_at_center = c - q_error.abs().sum(dim=1)
    error_upper_at_center = c + q_error.abs().sum(dim=1)
    torch.testing.assert_close(
        lower_A.mul(x_center).sum(1) + lower_bias,
        error_lower_at_center, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        upper_A.mul(x_center).sum(1) + upper_bias,
        error_upper_at_center, rtol=0, atol=1e-12)

    # Concretize the full dynamic rows and the returned input-space A forms;
    # both directions must agree on both endpoints under the same box.
    torch.testing.assert_close(forward_lower.reshape(-1), canonical_lower,
                               rtol=0, atol=1e-12)
    torch.testing.assert_close(forward_upper.reshape(-1), canonical_upper,
                               rtol=0, atol=1e-12)
    backward_from_A_lower = (
        lower_A.mul(x_center).sum(1) + lower_bias
        - lower_A.abs().mul(input_radii).sum(1))
    backward_from_A_upper = (
        upper_A.mul(x_center).sum(1) + upper_bias
        + upper_A.abs().mul(input_radii).sum(1))
    torch.testing.assert_close(backward_from_A_lower, canonical_lower,
                               rtol=0, atol=1e-12)
    torch.testing.assert_close(backward_from_A_upper, canonical_upper,
                               rtol=0, atol=1e-12)
    torch.testing.assert_close(backward_lower.reshape(-1), canonical_lower,
                               rtol=0, atol=1e-12)
    torch.testing.assert_close(backward_upper.reshape(-1), canonical_upper,
                               rtol=0, atol=1e-12)


def test_reusing_the_input_rebuilds_the_generator_book():
    """The book is per propagation; compute_bounds with x=None skips set_input.

    So the last propagation's bounds can survive into this one with columns
    numbered under the book it is about to replace, and the zonotope read off
    them would be indexed in a column space that no longer exists.
    """
    model = _Residual()
    bm, _, _ = _run(model, (1, 4))
    bx = BoundedTensor(torch.linspace(-1., 1., 4, dtype=_DT).reshape(1, 4),
                       PerturbationLpNorm(
                           norm=float('inf'),
                           x_L=torch.full((1, 4), -1.0, dtype=_DT),
                           x_U=torch.full((1, 4), 1.0, dtype=_DT)))
    bm.compute_bounds(x=(bx,), method='dynamic-forward',
                      final_node_name=bm.relus[0].name)
    reused = bm.compute_bounds(x=None, method='dynamic-forward')[0]

    fresh_bm, fresh, _ = _run(model, (1, 4))
    torch.testing.assert_close(reused, fresh, rtol=0, atol=1e-12)
    rows, centre = bm.zonotope_rows(bm.final_node())
    want_rows, want_centre = fresh_bm.zonotope_rows(fresh_bm.final_node())
    torch.testing.assert_close(rows, want_rows, rtol=0, atol=1e-12)
    torch.testing.assert_close(centre, want_centre, rtol=0, atol=1e-12)


@pytest.mark.parametrize('subtract', [False, True])
def test_a_broadcasting_constant_widens_the_coefficients(subtract):
    """A coefficient-free operand still sets the result's shape."""

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                'constant', torch.tensor([1., 2., 3.], dtype=_DT))
            self.out = nn.Linear(3, 1, dtype=_DT)
            with torch.no_grad():
                self.out.weight.fill_(1.)
                self.out.bias.zero_()

        def forward(self, x):
            merged = self.constant - x if subtract else x + self.constant
            return self.out(merged)

    x_L = torch.tensor([[-1.]], dtype=_DT)
    x_U = torch.tensor([[1.]], dtype=_DT)
    bm = BoundedModule(Net(), (x_L + x_U) / 2, device='cpu')
    bx = BoundedTensor((x_L + x_U) / 2, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))
    lower, upper = bm.compute_bounds(
        x=(bx,), method='dynamic-forward', C=torch.ones(1, 1, 1, dtype=_DT))
    torch.testing.assert_close(lower, torch.tensor([[3.]], dtype=_DT))
    torch.testing.assert_close(upper, torch.tensor([[9.]], dtype=_DT))


class _UnperturbedBranch(nn.Module):
    """One branch reaches the perturbed input, the other reaches only the
    unperturbed one, so it arrives at the merge carrying no columns at all."""

    def __init__(self, n_in, sub=False):
        super().__init__()
        torch.manual_seed(3)
        self.sub = sub
        self.perturbed = nn.Linear(n_in, 3, dtype=_DT)
        self.fixed = nn.Linear(3, 3, dtype=_DT)
        self.out = nn.Linear(3, 2, dtype=_DT)

    def forward(self, x, c):
        a, b = self.perturbed(x), self.fixed(c)
        return self.out(b - a if self.sub else a + b)


@pytest.mark.parametrize('subtract', [False, True])
@pytest.mark.parametrize('n_in', [1, 3])
def test_a_coefficient_free_branch_is_widened_over_the_batch(n_in, subtract):
    """A branch reaching no generator still carries a batch dimension.

    Its bias is laid out as (batch, *shape) where a coefficient tensor is
    (batch, generator, *shape), so lining the two up without putting the
    generator axis back reads the batch as a generator: it refuses to
    broadcast when the two disagree, and when the node happens to carry one
    column it copies that column batch-many times, concretizing the same
    generator over and over. Both need batch > 1 to show, so a batch of boxes
    is checked against the same boxes run one at a time.
    """
    model = _UnperturbedBranch(n_in, sub=subtract)
    eps = 0.5
    # Four boxes, so the batch matches neither column count and the mismatch
    # is a broadcast error at n_in=3 and a duplicated column at n_in=1.
    x = torch.arange(4 * n_in, dtype=_DT).reshape(4, n_in) / 10
    c = torch.arange(12, dtype=_DT).reshape(4, 3)

    def run(x, c):
        bm = BoundedModule(model, (x, c), device='cpu')
        bx = BoundedTensor(x, PerturbationLpNorm(
            norm=float('inf'), x_L=x - eps, x_U=x + eps))
        return bm.compute_bounds(
            x=(bx, c), method='dynamic-forward',
            C=torch.eye(2, dtype=_DT).expand(x.shape[0], 2, 2))

    lower, upper = run(x, c)
    for i in range(x.shape[0]):
        want_l, want_u = run(x[i:i + 1], c[i:i + 1])
        torch.testing.assert_close(lower[i:i + 1], want_l, rtol=0, atol=1e-12)
        torch.testing.assert_close(upper[i:i + 1], want_u, rtol=0, atol=1e-12)


class _ConvResidual(nn.Module):
    """A residual net over 4-D tensors, which is the shape this is for."""

    def __init__(self, blocks=2):
        super().__init__()
        torch.manual_seed(5)
        self.b = nn.ModuleList([nn.Conv2d(3, 3, 3, padding=1, dtype=_DT)
                                for _ in range(blocks)])
        self.s = nn.ModuleList([nn.Conv2d(3, 3, 1, dtype=_DT)
                                for _ in range(blocks)])
        self.o = nn.Linear(3 * 5 * 5, 2, dtype=_DT)

    def forward(self, x):
        for block, skip in zip(self.b, self.s):
            x = torch.relu(block(x)) + skip(x)
        return self.o(x.flatten(1))


@pytest.mark.parametrize('max_dim', [3, 20, int(1e9)])
def test_a_conv_dag_reads_the_same_zonotope_at_any_window(max_dim):
    """The zero padding a branch needs and the widening a merge needs both have
    to hold for 4-D coefficients, at any window size."""
    model = _ConvResidual()
    eps = 0.15
    x0 = torch.zeros(1, 3, 5, 5, dtype=_DT)

    def read(width):
        bm = BoundedModule(model, x0, device='cpu',
                           bound_opts={'forward_max_dim': width})
        bx = BoundedTensor(x0, PerturbationLpNorm(
            norm=float('inf'), x_L=x0 - eps, x_U=x0 + eps))
        bm.compute_bounds(x=(bx,), method='dynamic-forward')
        return bm, bm.zonotope_rows(bm.final_node())

    bm, (rows, centre) = read(max_dim)
    _, (want_rows, want_centre) = read(int(1e9))
    torch.testing.assert_close(rows, want_rows, rtol=0, atol=1e-12)
    torch.testing.assert_close(centre, want_centre, rtol=0, atol=1e-12)

    reach = rows.abs().sum(1)
    pts = (torch.rand(20000, 3, 5, 5, dtype=_DT) * 2 - 1) * eps
    with torch.no_grad():
        value = model(pts)
    assert (centre - reach <= value.min(0).values + 1e-9).all()
    assert (centre + reach >= value.max(0).values - 1e-9).all()
