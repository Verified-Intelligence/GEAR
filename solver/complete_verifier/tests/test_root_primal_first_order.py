"""root_primal_first_order: the root LP start comes from PDHG, never from a CPU LP, and lp_alternation is off."""
import os
import sys
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arguments  # noqa: E402
from input_split import nlp_in_bab  # noqa: E402


def _problem():
    class Net:
        class net:
            input_name = ['x']
    lA = torch.tensor([[[1., 1.], [-1., -1.]]], dtype=torch.float64)        # min x0+x1 s.t. x0+x1 >= 1
    l, u = torch.zeros(1, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.float64)
    glb = torch.tensor([[0., -2.]], dtype=torch.float64)                       # box minima of the two rows
    rhs = torch.tensor([[0., -1.]], dtype=torch.float64)
    return Net, {'lA': {'x': lA}}, l, u, rhs, glb


def _call():
    net, ret, l, u, rhs, glb = _problem()
    return net, ret, l, u, rhs, glb


def _nlp(**kw):
    nlp = arguments.Config['solver']['nlp']
    old = {k: nlp.get(k) for k in kw}
    nlp.update(kw)
    return nlp, old


def test_first_order_start_has_no_cpu_lp():
    nlp, old = _nlp(root_lp_local_start=True, root_primal_first_order=True)
    try:
        with mock.patch('scipy.optimize.linprog', side_effect=AssertionError('CPU LP')):
            x = nlp_in_bab.compute_root_lp_point(*_call())
    finally:
        nlp.update(old)
    assert x is not None and x.shape == (1, 2)
    assert 1.0 - 1e-2 <= x.sum().item() <= 1.0 + 1e-1       # near the LP optimum x0+x1 = 1


def test_flag_off_still_uses_linprog():
    nlp, old = _nlp(root_lp_local_start=True, root_primal_first_order=False)
    try:
        with mock.patch('scipy.optimize.linprog', side_effect=AssertionError('CPU LP')) as lp:
            nlp_in_bab.compute_root_lp_point(*_call())
    finally:
        nlp.update(old)
    # the failure is swallowed into "skipped"; the point is that linprog was reached
    assert lp.called


def test_memory_check_skips_the_start():
    nlp, old = _nlp(root_lp_local_start=True, root_primal_first_order=True)
    try:
        with mock.patch('psutil.virtual_memory', return_value=mock.Mock(available=1)):
            assert nlp_in_bab.compute_root_lp_point(*_call()) is None
    finally:
        nlp.update(old)
