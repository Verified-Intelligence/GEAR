"""CPU soundness and scope tests for the LP-to-CROWN interface."""
from types import SimpleNamespace
from fractions import Fraction
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import linprog
import torch
from nlp_solver.domain_certificate import CertificateRegistry, exact_residual


def lift():
    return SimpleNamespace(n=2, nb=2, N=3, a=torch.tensor([0]), b=torch.tensor([1]),
                           levels=[torch.tensor([0])], atoms=None)


def problem():
    return (np.array([-1., -1., 1.]), 0.,
            sp.csr_matrix([[0., 0., -1.], [1., 1., -1.], [-1., 0., 1.], [0., -1., 1.]]),
            np.array([0., 1., 0., 0.]), np.zeros(3), np.ones(3))


def test_random_descendants_and_brute_force():
    data = problem()
    reg = CertificateRegistry(lift())
    c, c0, G, h, lo, hi = data
    sol = linprog(c, A_ub=G, b_ub=h, bounds=list(zip(lo, hi)), method='highs')
    cert = reg.add(lo[:2], hi[:2], data, -sol.ineqlin.marginals)
    rng = np.random.default_rng(19)
    for _ in range(64):
        l, u = np.sort(rng.uniform(0, 1, (2, 2)), axis=0)
        L, U = torch.tensor(l)[None], torch.tensor(u)[None]
        bound, refs = reg.bound(L, U, torch.tensor([float('inf')]))
        zl, zu = cert.columns.intervals(L, U, cert.aux_lower, cert.aux_upper)
        child = linprog(c, A_ub=G, b_ub=h, bounds=list(zip(zl[0], zu[0])), method='highs')
        exact = min(x*y-x-y for x in (l[0], u[0]) for y in (l[1], u[1]))
        assert bound[0] <= child.fun
        assert bound[0] <= exact
        assert bound[0] >= cert.scalar_bound
        assert refs[0, 0] == cert.scope_id


def test_cutoff_scope_and_reordering():
    reg = CertificateRegistry(lift())
    cert = reg.add([0., 0.], [1., 1.], problem(), np.ones(4), cutoff=2.)
    L = torch.tensor([[.2, .4], [-.1, .3], [.1, .2]], dtype=torch.float64)
    U = torch.tensor([[.3, .8], [.2, .5], [.9, .9]], dtype=torch.float64)
    P = torch.tensor([1., 1., 3.])
    assert cert.applies(L, U, P).tolist() == [True, False, False]
    order = torch.tensor([2, 0, 1, 0])
    assert cert.applies(L[order], U[order], P[order]).tolist() == [False, True, False, True]
    assert torch.isneginf(reg.bound(L, U, P)[0][1:]).all()


def test_exact_compression_enclosures():
    rng = np.random.default_rng(4)
    data = list(problem())
    data[0] = rng.normal(size=3)*1e-7
    data[2] = sp.csr_matrix(rng.normal(size=(70, 3))*1e5)
    data[3] = rng.normal(size=70)
    y = rng.uniform(0, 10, 70)
    reg = CertificateRegistry(lift())
    cert = reg.add([0., 0.], [1., 1.], data, y)
    r, k = exact_residual(data, y)
    assert abs(k-Fraction(cert.kappa)) <= Fraction(cert.kappa_error)
    for v, w, e in zip(r, cert.residual_r, cert.residual_error):
        assert abs(v-Fraction(float(w))) <= Fraction(float(e))
    with pytest.raises(ValueError, match='nonfinite'):
        reg.add([0., 0.], [1., 1.], data, np.full(70, np.nan))


def test_lp_producer_exports_without_resolve():
    from nlp_solver.lifted_lp import LiftedLP
    records = []
    result = LiftedLP.solve_node(*problem(), certificate_sink=lambda d, y: records.append((d, y)))
    assert np.isfinite(result[0]) and len(records) == 1
    reg = CertificateRegistry(lift())
    cert = reg.add([0., 0.], [1., 1.], *records[0])
    assert cert.scalar_bound == result[0]


def test_graph_reference_map_and_scope():
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    class Model(torch.nn.Module):
        def forward(self, x):
            product = x[:, :1] * x[:, 1:]
            return product * product
    x = torch.tensor([[.5, .5]], dtype=torch.float64)
    bm = BoundedModule(Model(), x)
    bm.compute_bounds(x=(BoundedTensor(x, PerturbationLpNorm(x_L=x*0, x_U=x*0+1)),), method='IBP')
    reg = CertificateRegistry(lift())
    reg.add_auxiliary([0., 0.], [1., 1.], [0., 0., .2], [1., 1., .3], cutoff=.5,
                      proof='test supplied certified interval')
    L = torch.tensor([[.1, .1], [-1., 0.], [.2, .2]], dtype=torch.float64)
    U = torch.tensor([[.8, .8], [.5, .5], [.9, .9]], dtype=torch.float64)
    refs, cap = reg.graph_references(bm, x, L, U, torch.tensor([.4, .4, .6]))
    product = [n.name for n in bm.nodes() if type(n).__name__ == 'BoundMul'][0]
    assert refs[product][0][:, 0].tolist() == [.2, -float('inf'), -float('inf')]
    assert refs[product][1][:, 0].tolist() == [.3, float('inf'), float('inf')]
    assert cap.tolist() == [.5, float('inf'), float('inf')]


def test_aux_obbt_export():
    from complete_verifier.tests.test_lifted_lp import _setup
    from nlp_solver.lifted_lp import build_lifted_lp
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    lp.certificate_registry = CertificateRegistry(lp)
    lp.transfer_aux = True
    lp.obbt(l, u, rhs[:, 1:]+1e-4, rounds=1, time_budget=2.)
    assert lp.certificate_registry.auxiliary_scopes
    aux = lp.certificate_registry.auxiliary_scopes[-1]
    assert len(aux.aux_lower) == lp.N
    assert (aux.aux_lower <= aux.aux_upper).all()


def test_projected_cut_signs_and_tolerance(monkeypatch):
    from nlp_solver.domain_certificate import project_cutoff
    from nlp_solver.structure.rowpool import RowPool
    import arguments
    arguments.Config.parse_config(['--device', 'cpu'], verbose=False)
    reg = CertificateRegistry(lift(), input_cutoff_cuts=True)
    # Nonzero residuals on input and product, with both product signs.
    rng = np.random.default_rng(8)
    for sign in (1., -1.):
        data = list(problem())
        data[0] = np.array([.2, -.3, sign])
        P = .1
        cert = reg.add([0., 0.], [1., 1.], data, np.zeros(4), cutoff=P)
        A, b = cert.input_cut
        for x in rng.uniform(0, 1, (1000, 2)):
            f = data[0] @ np.r_[x, x.prod()]
            if f <= P:
                assert A @ x <= b
    pool = RowPool(torch.tensor([[P, 0.]], dtype=torch.float64))
    pool.derived_cut_registry = reg
    original = pool.stats()
    L = torch.tensor([[0., 0.], [-1., 0.], [0., 0.]], dtype=torch.float64)
    U = torch.ones_like(L)
    A, b = pool.derived_cut_view(L, U, torch.tensor([P, P, P+1], dtype=torch.float64))
    assert (A[1:] == 0).all() and (b[1:] == 0).all()
    assert b[0, 0] == reg.scopes[-1].input_cut[1]
    assert pool.stats() == original
    # The global spec pool still applies exactly its one original-row tolerance.
    assert pool.thresholds(torch.zeros(1, 2, dtype=torch.float64))[0, 1] == pool.tau[1]


def test_derived_cut_reaches_clipping(monkeypatch):
    import arguments
    from input_split.clip import clip_domains
    from nlp_solver.structure.rowpool import RowPool
    monkeypatch.setitem(arguments.Config['solver']['nlp'], 'input_cutoff_cuts', True)
    reg = CertificateRegistry(lift(), input_cutoff_cuts=True)
    data = list(problem())
    data[0] = np.array([1., 0., 0.])
    reg.add([0., 0.], [1., 1.], data, np.zeros(4), cutoff=.25)
    pool = RowPool(torch.tensor([[.25, 0.]], dtype=torch.float64))
    pool.derived_cut_registry = reg
    l, u = torch.zeros(2, 2, dtype=torch.float64), torch.ones(2, 2, dtype=torch.float64)
    # Second domain's cutoff has increased: it must not receive the cut.
    thresholds = torch.tensor([[.25, 0.], [.5, 0.]], dtype=torch.float64)
    _, clipped = clip_domains(l, u, thresholds, torch.zeros(2, 2, 2, dtype=torch.float64),
                             torch.zeros(2, 2, dtype=torch.float64), rowpool=pool)
    assert clipped[0, 0] <= .25+1e-12
    assert clipped[1, 0] == 1.


def test_queue_scope_refs_survive_sort_and_pick(monkeypatch):
    import arguments
    from history import SplitHistory
    from input_split.branching_domains import UnsortedInputDomainList
    monkeypatch.setitem(arguments.Config['bab'], 'tree_traversal', 'depth_first')
    reg = CertificateRegistry(lift())
    data = problem()
    reg.add([0., 0.], [.4, 1.], data, np.zeros(4))
    reg.add([.6, 0.], [1., 1.], data, np.zeros(4))
    h = SplitHistory();h.certificate_registry = reg
    domains = UnsortedInputDomainList(1, 'cpu', history=h, use_split_idx=False)
    l = torch.tensor([[.1, .2], [.7, .2]], dtype=torch.float64)
    u = torch.tensor([[.3, .8], [.9, .8]], dtype=torch.float64)
    domains.add(torch.tensor([[-2.], [-1.]]), l, u, torch.ones(2, 1, 1),
                torch.ones(2, 1), check_dm_lbs=False)
    assert sorted(domains.certificate_scope_refs._storage[:2, 0].tolist()) == [0, 1]
    domains.sort()
    picked = domains.pick_out_batch(2)
    refs = h.certificate_scope_refs.flatten()
    # Each ID still names the scope of its picked box after sorting.
    assert sorted(refs.tolist()) == [0, 1]
    assert domains.certificate_scope_refs.num_used == 0


def test_ibp_reference_mask_matches_unverified_domains(monkeypatch):
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    from input_split.bounding import get_lower_bound_with_ibp_enhancement
    class Model(torch.nn.Module):
        def forward(self, x):
            return (x[:, :1]*x[:, 1:]).square()
    x = torch.full((2, 2), .5, dtype=torch.float64)
    bm = BoundedModule(Model(), x[:1])
    new_x = BoundedTensor(x, PerturbationLpNorm(x_L=x*0, x_U=x*0+1))
    bm.compute_bounds(x=(new_x,), method='IBP')
    product = [n.name for n in bm.nodes() if type(n).__name__ == 'BoundMul'][0]
    refs = {product: (torch.tensor([[.2], [.3]], dtype=torch.float64),
                      torch.tensor([[.4], [.5]], dtype=torch.float64))}
    calls = []
    original = bm.compute_bounds
    def record(*args, **kwargs):
        if kwargs.get('method', '').lower() == 'crown':
            calls.append(kwargs['reference_bounds'])
        return original(*args, **kwargs)
    monkeypatch.setattr(bm, 'compute_bounds', record)
    get_lower_bound_with_ibp_enhancement(SimpleNamespace(net=bm), new_x,
        torch.ones(2, 1, 1, dtype=torch.float64), 'crown',
        torch.tensor([[-1.], [1.]], dtype=torch.float64), external_reference_bounds=refs)
    assert len(calls) == 1
    assert calls[0][product][0].shape == (1, 1)
    assert calls[0][product][0][0, 0] == .3
    assert calls[0][product][1][0, 0] == .5


@pytest.mark.parametrize('workers,gain', [(1, False), (1, True), (2, True)])
def test_lazy_frontier_matches_eager_leaf_exports(monkeypatch, workers, gain):
    from nlp_solver.domain_certificate import DomainCertificate, LeafCertificates
    from tests.test_lifted_lp_rlt import make_lp
    lp, rh = make_lp([1, 1, 1], 1, [(0, 1), (1, 2)], [1, 1])
    registry = lp.certificate_registry = CertificateRegistry(lp, input_cutoff_cuts=True)
    original = LeafCertificates.capture

    def capture(self, lower, upper, data, y, cutoff):
        # The old implementation compressed here, before any subsequent OBBT
        # mutation. Keep that as an independent oracle for the live leaves.
        oracle = CertificateRegistry(lp, input_cutoff_cuts=True)
        eager = oracle.add(lower, upper, data, y, cutoff)
        if not hasattr(self, 'expected'):
            self.expected = []
        self.expected.append(eager)
        original(self, lower, upper, data, y, cutoff)
    monkeypatch.setattr(LeafCertificates, 'capture', capture)
    lo, hi = torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, 3, dtype=torch.float64)
    result = lp.branch_and_bound(lo, hi, rh, cutoff=10., max_nodes=12,
                                 time_budget=100., return_leaves=True, workers=workers,
                                 node_obbt_gain=gain)
    assert not registry._scopes  # no compression/publication in the tree loop
    expected = [c for leaf in sorted(registry._frontier, key=lambda s: s.order)
                for c in getattr(leaf, 'expected', [])]
    assert expected
    actual = registry.scopes
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        for name in DomainCertificate.__dataclass_fields__:
            if name in ('scope_id', 'ancestor_id', 'columns'):
                continue
            x, y = getattr(a, name), getattr(b, name)
            if torch.is_tensor(x):
                assert torch.equal(x, y), name
            elif isinstance(x, np.ndarray):
                np.testing.assert_array_equal(x, y)
            elif name == 'input_cut' and x is not None:
                np.testing.assert_array_equal(x[0], y[0]); assert x[1] == y[1]
            else:
                assert x == y, name
        L, U = torch.tensor(result[3][0]), torch.tensor(result[3][1])
        assert torch.equal(a.evaluate(L, U), b.evaluate(L, U))
    # Resume carries the published ancestor ID, without a containment scan.
    published = len(actual)
    lp.branch_and_bound(lo, hi, rh, cutoff=10., max_nodes=1, time_budget=100.,
                        return_leaves=True, resume=result[3], node_obbt_gain=gain)
    new = registry.scopes[published:]
    assert all(c.ancestor_id is None or c.ancestor_id < c.scope_id for c in new)


def test_retained_tree_exports_only_when_consumed(monkeypatch):
    from tests.test_lifted_lp_rlt import make_lp
    lp, rh = make_lp([1, 1, 1], 1, [(0, 1), (1, 2)], [1, 1])
    registry = lp.certificate_registry = CertificateRegistry(lp)
    lo, hi = torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, 3, dtype=torch.float64)
    iterator = lp.branch_and_bound(lo, hi, rh, cutoff=10., max_nodes=12,
                                   sliced=True, deadline=float('inf'), _iterate=True)
    for _ in range(4):
        next(iterator)
        assert not registry._scopes
    scopes = list(registry.scopes)
    assert scopes
    next(iterator)
    iterator.close()
    assert isinstance(registry._frontier, list)
    assert len(registry.scopes) >= len(scopes)


def test_scope_index_matches_exhaustive_overlaps_and_skips_siblings(monkeypatch):
    from nlp_solver.domain_certificate import DomainCertificate
    reg = CertificateRegistry(lift())
    for i in range(128):
        reg.add([i/128, 0.], [(i+1)/128, 1.], problem(), np.zeros(4), cutoff=2.)
    # Overlap, repeated boxes and a root scope exercise all applicable scopes,
    # rather than choosing only one leaf at shared boundaries.
    for lo, hi in (([0., 0.], [1., 1.]), ([.1, .2], [.7, .8]), ([0., 0.], [1/128, 1.])):
        reg.add(lo, hi, problem(), np.zeros(4), cutoff=1.)
    rng = np.random.default_rng(9)
    L = torch.tensor(rng.uniform(0, 1, (50, 2)))
    U = L + (1-L)*.001
    P = torch.tensor(rng.uniform(.5, 2.5, 50))
    expected = [(c.scope_id, c.applies(L, U, P)) for c in reg.scopes]
    expected = [(i, m) for i, m in expected if m.any()]
    refs = reg.references(L, U, P)
    assert [c.scope_id for c, _ in refs] == [i for i, _ in expected]
    assert all(torch.equal(a, b) for (_, a), (_, b) in zip(refs, expected))
    calls = []
    original = DomainCertificate.applies
    def applies(self, *args):
        calls.append(self.scope_id)
        return original(self, *args)
    monkeypatch.setattr(DomainCertificate, 'applies', applies)
    reg.bound(L[:1], U[:1], P[:1])
    assert len(calls) <= 4


def test_lazy_timeout_keeps_original_zero_dual_without_resolve(monkeypatch):
    from nlp_solver.domain_certificate import LeafCertificates, DomainCertificate
    from tests.test_lifted_lp_rlt import make_lp
    lp, rh = make_lp([1, 1, 1], 1, [(0, 1), (1, 2)], [1, 1])
    registry = lp.certificate_registry = CertificateRegistry(lp)
    proposals = []
    def timeout(*data, **kwargs):
        proposals.append(data)
        return -np.inf, None
    monkeypatch.setattr(lp, 'solve_node', timeout)
    pending = LeafCertificates()
    lp.solve_bab_node(np.zeros(3), np.ones(3), rh[0].numpy(), None, None, 10.,
                       certificates=pending)
    assert len(proposals) == len(pending.records) == 1
    lower, upper, data, y, cutoff = pending.records[0]
    assert not y.any()
    expected = DomainCertificate.export(0, registry.columns, lower, upper, data, y, cutoff)
    registry.defer([pending])
    assert registry.scopes[0].row_version == expected.row_version
    assert registry.scopes[0].scalar_bound == expected.scalar_bound
    assert len(proposals) == 1


def test_lazy_publication_preserves_latest_input_cut_order():
    from nlp_solver.structure.rowpool import RowPool
    registry = CertificateRegistry(lift(), input_cutoff_cuts=True)
    first, last = registry.new_leaf(), registry.new_leaf()
    for leaf, slope in ((first, 1.), (last, 2.)):
        data = list(problem())
        data[0] = np.array([slope, 0., 0.])
        leaf.capture(np.zeros(2), np.ones(2), data, np.zeros(4), .25)
    # Heap order is deliberately the reverse of solve order; scopes overlap.
    registry.defer([last, first])
    pool = RowPool(torch.tensor([[.25, 0.]], dtype=torch.float64))
    pool.derived_cut_registry = registry
    A, b = pool.derived_cut_view(torch.zeros(1, 2, dtype=torch.float64),
                                torch.ones(1, 2, dtype=torch.float64), .25)
    assert A[0, 0, 0] == 2.
    assert b[0, 0] == registry.scopes[-1].input_cut[1]


def test_packed_graph_references_match_scope_loop_and_growing_banks():
    """Overlaps, cutoffs, queue ordering and new scopes preserve all extrema."""
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    from nlp_solver.rigorous_dual import round_down
    class Model(torch.nn.Module):
        def forward(self, x):
            return x[:, :1] * x[:, 1:]
    x = torch.tensor([[.5, .5]], dtype=torch.float64)
    bm = BoundedModule(Model(), x)
    bm.compute_bounds(x=(BoundedTensor(x, PerturbationLpNorm(x_L=x*0, x_U=x*0+1)),), method='IBP')
    reg = CertificateRegistry(lift())
    rng = np.random.default_rng(43)
    for _ in range(2):
        for i in range(18):
            a = i / 32
            reg.add_auxiliary([a, 0.], [1., 1.], [a, 0., a*.1], [1., 1., 1.-a*.1],
                              cutoff=.5 if i % 2 else None)
            reg.add([a, 0.], [1., 1.], problem(), np.zeros(4), cutoff=.5 if i % 2 else None)
        L = torch.tensor(rng.uniform(0, .8, (37, 2)), dtype=torch.float32)
        U = L + .1
        P = torch.tensor(rng.uniform(.1, .9, 37))
        mapped, cap = reg.graph_references(bm, x, L, U, P)
        expected, expected_cap = {}, L.new_full((len(L),), torch.inf)
        expected_ids = torch.full((len(L), 1), -1, dtype=torch.long)
        for bank in (reg.scopes, reg.auxiliary_scopes):
            for scope in bank:
                mask = scope.applies(L, U, P)
                expected_cap[mask] = torch.minimum(expected_cap[mask], L.new_tensor(scope.P0))
                if bank is reg.scopes:
                    expected_ids[mask] = scope.scope_id
                graph = reg._graph_maps[(id(bm), scope.columns.column_version)]
                for name, (lo, hi) in graph.bounds(scope.aux_lower, scope.aux_upper).items():
                    lo = round_down(lo, L.dtype)
                    hi = -round_down(-hi, U.dtype)
                    shape = (len(L), *lo.shape[1:])
                    if name not in expected:
                        expected[name] = (L.new_full(shape, -torch.inf), U.new_full(shape, torch.inf))
                    expected[name][0][mask] = torch.maximum(expected[name][0][mask], lo.expand(shape)[mask])
                    expected[name][1][mask] = torch.minimum(expected[name][1][mask], hi.expand(shape)[mask])
        assert torch.equal(cap, expected_cap)
        assert mapped.keys() == expected.keys()
        assert all(torch.equal(a, b) for k in mapped for a, b in zip(mapped[k], expected[k]))
        assert torch.equal(reg.select_scopes(L, U, P), expected_ids)
        order = torch.tensor([5, 2, 5, 0, 36])
        reordered, reordered_cap = reg.graph_references(bm, x, L[order], U[order], P[order])
        assert torch.equal(reordered_cap, cap[order])
        assert all(torch.equal(a, b[order]) for k in mapped for a, b in zip(reordered[k], mapped[k]))


def test_batched_graph_images_equal_fraction_oracle():
    from nlp_solver.domain_certificate import GraphReferenceMap
    from nlp_solver.structure.rowpool import round_down, round_up
    graph = object.__new__(GraphReferenceMap)
    rng = np.random.default_rng(71)
    terms = [[(j, Fraction(float(rng.normal()))) for j in range(4)] for _ in range(6)]
    terms += [[], [(None, Fraction(1, 3))], [(1, Fraction(-1))], terms[0],
              [(0, Fraction(2)**900)], [(0, Fraction(2)**-900)],
              [(1, Fraction(1, 7)), (None, Fraction(-1, 13))]]
    graph.nodes = {'node': ((1, len(terms)), terms)}
    lo = torch.tensor(rng.normal(size=(21, 4)), dtype=torch.float64)
    hi = lo + torch.tensor(rng.uniform(0, 1, (21, 4)))
    lo[0, 0], hi[0, 0] = -np.finfo(np.float64).max, np.finfo(np.float64).max
    lo[1, 0], hi[1, 0] = -float(np.nextafter(0., 1.)), float(np.nextafter(0., 1.))
    actual_lo, actual_hi = graph.bounds_batch(lo, hi)
    for i in range(len(lo)):
        for k, expression in enumerate(terms):
            low = high = Fraction(0)
            for j, c in expression:
                a, b = (c, c) if j is None else (c*Fraction(float(lo[i,j])), c*Fraction(float(hi[i,j])))
                low += min(a,b)
                high += max(a,b)
            assert actual_lo[i,k] == round_down(low)
            assert actual_hi[i,k] == round_up(high)


def test_scope_index_preserves_python_float_cutoff_precision():
    reg = CertificateRegistry(lift())
    reg.add_auxiliary([0., 0.], [1., 1.], np.zeros(3), np.ones(3), cutoff=.5)
    L, U = torch.zeros(1, 2), torch.ones(1, 2)
    P = .5 + 2.**-30  # This rounds to .5 in float32, but exceeds the scope cutoff.
    assert not reg.auxiliary_scopes[0].applies(L, U, P).any()
    assert not reg._references(reg.auxiliary_scopes, L, U, P, 'auxiliary')


@pytest.mark.parametrize('cutoff', [float('inf'), -float('inf'), float('nan')])
def test_scope_index_keeps_raw_nonfinite_caps(cutoff):
    from nlp_solver.domain_certificate import ScopeIndex
    reg = CertificateRegistry(lift())
    scope = reg.add_auxiliary([0., 0.], [1., 1.], np.zeros(3), np.ones(3), cutoff=cutoff)
    L, U = torch.zeros(1, 2), torch.ones(1, 2)
    index = ScopeIndex([scope])
    rows, ids = index.matches(L, U, 1.)
    assert rows.tolist() == ids.tolist() == [0]
    assert index.cutoff[0] == scope.P0 or np.isnan(index.cutoff[0]) and np.isnan(scope.P0)
