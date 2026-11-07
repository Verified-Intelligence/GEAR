"""The compiler's branching space (nlp_solver.structure.branching): product graph, colouring into independent blocks,
the exact set S (a vertex cover plus the variables that occur nonlinearly alone), the basis B and the map T."""
import itertools
import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nlp_solver.structure.branching import (basis_of, branching_space, colouring, koenig_cover,  # noqa: E402
                                            product_structure)

BENCH = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "benchmarks")


def mono(*vars_):
    return tuple(sorted((v, 1) for v in vars_))


def test_products_squares_powers_and_atoms():
    polys = [{mono(0, 1): 2, ((2, 2),): 1, mono(3): 1},          # x0 x1, x2^2, x3
             {mono(1, 4, 5): 1, ((6, 1), (7, 3)): 1},             # x1 x4 x5 (a clique), x6 x7^3
             {((9, 1),): 1, mono(8, 3): -1}]                      # an atom (id 9) over x8; x8 x3
    edges, fixed = product_structure(polys, 9, {9: frozenset({8})})
    assert fixed.tolist() == [False, False, True, False, False, False, False, True, True]
    assert sorted(map(tuple, edges.tolist())) == [(0, 1), (1, 4), (1, 5), (4, 5)]  # x6/x8 pairs drop: x7, x8 fixed


def test_colouring_is_proper_and_bipartite_when_it_can_be():
    path = np.array([(0, 1), (1, 2), (2, 3)])
    col, bip = colouring(path, 4)
    assert bip and col.tolist() == [0, 1, 0, 1]
    tri = np.array([(0, 1), (1, 2), (0, 2), (2, 3)])
    col, bip = colouring(tri, 5)
    assert not bip and col[4] == -1
    assert all(col[i] != col[j] for i, j in tri.tolist())


def brute_cover(edges, n):
    for k in range(n + 1):
        for S in itertools.combinations(range(n), k):
            if all(i in S or j in S for i, j in edges):
                return k
    return n


@pytest.mark.parametrize('seed', range(12))
def test_koenig_cover_is_a_minimum_cover(seed):
    rng = np.random.default_rng(seed)
    L, R = rng.integers(2, 6), rng.integers(2, 6)
    edges = sorted({(int(i), int(L + j)) for i, j in zip(rng.integers(0, L, 9), rng.integers(0, R, 9))})
    edges = np.array(edges)
    col, bip = colouring(edges, L + R)
    assert bip
    cover = set(koenig_cover(edges, col).tolist())
    assert all(i in cover or j in cover for i, j in edges.tolist())
    assert len(cover) == brute_cover(edges.tolist(), L + R)


def rank(M):
    if M.shape[0] == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > max(M.shape) * np.finfo(float).eps * s[0]))


@pytest.mark.parametrize('seed', range(10))
def test_basis_size_and_the_affine_map(seed):
    """|B| = rank([A_eq; E_S]) - rank(A_eq), and x_{S \\ B} = c + T x_B at every point of A_eq x = b_eq"""
    rng = np.random.default_rng(seed)
    n, m = 12, int(rng.integers(1, 8))
    A = rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.4)
    A[0] = A[1] if m > 2 else A[0]                         # a repeated row: rank deficient
    b = A @ rng.standard_normal(n)
    S = np.sort(rng.choice(n, int(rng.integers(2, 8)), replace=False))
    B, D, T, c = basis_of(S, sp.csr_matrix(A), b, n)
    E = np.zeros((len(S), n))
    E[np.arange(len(S)), S] = 1.0
    assert len(B) == rank(np.vstack([A, E])) - rank(A)
    assert sorted(np.r_[B, D].tolist()) == S.tolist()
    null = np.linalg.svd(A)[2][rank(A):].T
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(3):
        x = x0 + null @ rng.standard_normal(null.shape[1])
        assert np.allclose(x[D], c + T @ x[B], atol=1e-9)


def test_branching_space_invariants():
    # x0 x1, x1 x2, x2 x3 (a path), x4^2, and the equality x0 + x3 = 1
    polys = [{mono(0, 1): 1, mono(1, 2): 1}, {mono(2, 3): 1, ((4, 2),): 1}, {mono(0): 1, mono(3): 1}]
    space = branching_space(polys, 6, None, sp.csr_matrix(([1.0, 1.0], ([0, 0], [0, 3])), shape=(1, 6)),
                            np.array([1.0]))
    S = set(space.cover.tolist())
    assert space.bipartite and 4 in S and space.fixed[4]
    assert all(i in S or j in S for i, j in space.edges.tolist())
    assert set(space.basis.tolist()) | set(space.dependent.tolist()) == S
    assert len(space.basis) == len(S) - (1 if {0, 3} <= S else 0)
    for k in range(space.colours):                         # every block is an independent set of the product graph
        free = space.block(k)
        assert not any(free[i] and free[j] for i, j in space.edges.tolist())
        assert not free[4]
    assert space.block(0)[5] and space.block(1)[5]           # a product-free variable is free in every block


@pytest.mark.skipif(not os.path.isdir(BENCH), reason='eqexact benchmarks not available')
@pytest.mark.parametrize('name,B', [('wastewater05m2', 9), ('wastewater13m2', 30), ('waterund11', 15)])
def test_census_values_on_benchmark_models(name, B):
    """the arch2 spec's branching-space census (data/branching_space.json)"""
    from types import SimpleNamespace
    import torch
    import structure_instances as si
    from nlp_solver import lp_alternation
    for m in [m for m in sys.modules if m.split('.')[0] in ('modules', 'mod_files_gams', 'py_minlplib_linear')]:
        del sys.modules[m]
    d = si.load('minlplib_linear', name, {}, root=BENCH)
    K = d['C'].shape[1]
    state = SimpleNamespace(net=SimpleNamespace(model_ori=d['model']), dm_l=d['L'][None], dm_u=d['U'][None], c=d['C'],
                            rhs=torch.zeros(1, K, dtype=torch.float64), or_spec_size=None,
                            base_input_linear_constraints=None, nlp_config={'lp_alternation': True},
                            primal_solution=None, best_primal_x=None)
    space = lp_alternation._setup(state)[2].space
    assert space.bipartite and space.colours == 2 and len(space.basis) == B
