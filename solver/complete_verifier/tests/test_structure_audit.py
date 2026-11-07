import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from nlp_solver.structure.facts import RowFacts
from structure_audit import violations


def rows_at(x):
    return np.stack([x, x + 5, x ** 2], 1)


X = np.array([-1., 1., -2., 2.])           # pairs (-1, 1) and (-2, 2); the model is x, x + 5, x ** 2


def audit(rows):
    Y = rows_at(X)
    return violations(SimpleNamespace(rows=rows), Y, rows_at((X[::2] + X[1::2]) / 2), np.stack([Y[::2], Y[1::2]]),
                      ((X[::2] - X[1::2]) ** 2)[:, None], np.zeros(len(rows)))


def test_true_facts_pass_and_refuted_facts_are_reported():
    good = [RowFacts(), RowFacts(box=(3., 7.), duplicate_of=(0, 1, 5), infeasible=True),
            RowFacts(box=(0., 4.), lambda_min=2., curvature='convex', support=frozenset({0}))]       # x ** 2 has Hessian 2
    assert audit(good) == []
    refuted = ((1, dict(box=(3.5, 7.))), (1, dict(redundant=True)), (1, dict(duplicate_of=(0, 1, 4))),
               (1, dict(opposite_of=(0, -1, 5))), (1, dict(alias=(1, 4))),
               (2, dict(lambda_min=3.)), (2, dict(curvature='concave', lambda_min=-1.)))
    for row, facts in refuted:
        rows = [RowFacts(), RowFacts(), RowFacts()]
        rows[row] = RowFacts(support=frozenset({0}), **facts)
        assert audit(rows), facts
