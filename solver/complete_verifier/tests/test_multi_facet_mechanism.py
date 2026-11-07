"""Mechanism guard for multi-facet relaxation (see .planning/2026-08-05-multifacet-relaxation).

Proves, through the REAL solver `auto_LiRPA.concretize_func.constraints_solving`, that running it
over a LIFTED variable vector (node values as free variables) with BOTH McCormick facets per bilinear
is never looser than with a single facet, and is strictly tighter for a substantial fraction of
objectives. This is the load-bearing premise of the whole feature: appending a facet to the
input-space constraint set is vacuous; the lift is what makes the second facet bite.
"""
import numpy as np
import torch
from auto_LiRPA.concretize_func import constraints_solving, sort_out_constr_batches


def _mccormick(ia, ib, iz, al, au, bl, bu, dim):
    """4 McCormick planes for z=a*b as rows (A, c) meaning A@u + c <= 0."""
    def row(coefs, c):
        A = np.zeros(dim)
        for i, v in coefs:
            A[i] += v
        return (A, c)
    lower = [row([(ia, bl), (ib, al), (iz, -1)], -al * bl),   # z >= bl*a + al*b - al*bl
             row([(ia, bu), (ib, au), (iz, -1)], -au * bu)]
    upper = [row([(ia, -bu), (ib, -al), (iz, 1)], bu * al),   # z <= bu*a + al*b - bu*al
             row([(ia, -bl), (ib, -au), (iz, 1)], bl * au)]
    return lower, upper


def _solve(dim, LB, UB, facets, cobj):
    """min cobj . u  s.t. facets, box  -- via the real constraints_solving path."""
    A = torch.tensor(np.array([r[0] for r in facets]), dtype=torch.float64).unsqueeze(0)
    b = torch.tensor(np.array([r[1] for r in facets]), dtype=torch.float64).unsqueeze(0)
    xL = torch.tensor(LB, dtype=torch.float64).unsqueeze(0)
    xU = torch.tensor(UB, dtype=torch.float64).unsqueeze(0)
    obj = torch.tensor(cobj, dtype=torch.float64).reshape(1, 1, dim)
    cons, sob = sort_out_constr_batches(xL, xU, (A, b), rearrange_constraints=False)
    if cons is None:  # fully covered => naive bound
        return constraints_solving(xL, xU, obj, None, sign=-1.0).flatten().item()
    return constraints_solving(xL, xU, obj, cons, sign=-1.0, sorted_out_batches=sob).flatten().item()


def _build():
    """Lifted vars u = [x, y, z, w1=xy, w2=yz, w3=zx] over [-1,1]^3; return (dim, LB, UB, Fhull, Fsingle)."""
    r = 1.0
    dim = 6
    LB = np.array([-r, -r, -r, -r * r, -r * r, -r * r])
    UB = np.array([r, r, r, r * r, r * r, r * r])
    Fh, Fs = [], []
    for iz, ia, ib in [(3, 0, 1), (4, 1, 2), (5, 2, 0)]:
        L, U = _mccormick(ia, ib, iz, -r, r, -r, r, dim)
        Fh += L + U            # both facets per bilinear (hull)
        Fs += [L[0], U[0]]     # one fixed facet per bilinear (single-plane)
    return dim, LB, UB, Fh, Fs


def test_hull_never_looser_than_single():
    dim, LB, UB, Fh, Fs = _build()
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = rng.standard_normal(dim)
        assert _solve(dim, LB, UB, Fh, c) >= _solve(dim, LB, UB, Fs, c) - 1e-9


def test_second_facet_strictly_tightens_many_objectives():
    dim, LB, UB, Fh, Fs = _build()
    rng = np.random.default_rng(0)
    improved = 0
    total = 400
    for _ in range(total):
        c = rng.standard_normal(dim)
        if _solve(dim, LB, UB, Fh, c) > _solve(dim, LB, UB, Fs, c) + 1e-6:
            improved += 1
    # empirical sweep gave ~57%; assert a safe lower bound so the test is not flaky.
    assert improved / total > 0.20, f"only {improved}/{total} objectives improved"


def test_constraints_solving_matches_scipy_hull_on_one_case():
    scipy = __import__("scipy.optimize", fromlist=["linprog"])
    dim, LB, UB, Fh, _ = _build()
    c = np.array([0.3, -0.7, 0.5, 1.0, -0.4, 0.2])
    got = _solve(dim, LB, UB, Fh, c)
    A_ub = np.array([r[0] for r in Fh]); b_ub = np.array([-r[1] for r in Fh])
    ref = scipy.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(LB, UB)), method="highs").fun
    assert abs(got - ref) < 1e-6, f"solver {got} vs scipy {ref}"
