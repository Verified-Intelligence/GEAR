"""A feasible tight-cutoff McCormick LP must survive false presolve infeasibility."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from nlp_solver import lifted_lp as ll


def test_narrow_paviani_root_recovers_and_certifies(monkeypatch):
    # Row-deleted snapshot of the production Paviani warm-up root LP. All
    # original column intervals and stored row coefficients are preserved.
    with np.load(Path(__file__).parent / 'data' / 'paviani_narrow_lp.npz') as d:
        c, G, h, lo, hi = (d[k] for k in ('c', 'G', 'h', 'lo', 'hi'))
    G = sp.csr_matrix(G)
    original = ll._highs_proposal
    seen = []

    def proposal(c, G, h, lo, hi, options):
        seen.append(options['presolve'])
        if options['presolve']:
            # Reproduce the observed false status regardless of HiGHS version.
            return SimpleNamespace(status=2)
        return original(c, G, h, lo, hi, options)

    monkeypatch.setattr(ll, '_highs_proposal', proposal)
    certificates = []
    bound, point = ll.LiftedLP.solve_node(
        c, 0., G, h, lo, hi, certificate_sink=lambda data, y: certificates.append((data, y)))
    assert seen == [True, False]
    assert np.isfinite(bound) and point is not None
    assert len(certificates) == 1
    data, y = certificates[0]
    assert data[2] is G  # certification evaluates the original rows and box
    assert bound == ll._dual_value_sparse(*data, y)
    assert bound <= float(c @ point) + 1e-6


def test_exhausted_budget_does_not_start_retry(monkeypatch):
    monkeypatch.setattr(ll, '_epigraph_proposal', lambda *a: (SimpleNamespace(status=2), a[1], a[2]))
    calls = []
    monkeypatch.setattr(ll, '_highs_proposal', lambda *a: calls.append(a))
    result = ll.LiftedLP.solve_node(np.ones(1), 0., sp.csr_matrix([[1.]]),
                                  np.ones(1), np.zeros(1), np.ones(1), time_limit=0.)
    assert result[0] == -np.inf and not calls
