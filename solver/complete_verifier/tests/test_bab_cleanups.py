import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DECLARED = {
    'qtb': None, 'convex_row_cuts': None, 'root_width': None,
    'qtb_mean_y': None, 'lp_mean_y': None, 'lifted': None, 'lifted_mean_y': None,
    'lifted_box': None, 'lifted_root_box': None, 'lifted_obbt_rows': (None, None, None),
    'lifted_cutoff': None, 'lifted_branch_score': None, 'lifted_split_point': None,
    'lifted_bab_cutoff': None, 'lifted_bab_leaves': None, 'lifted_bab_open': None,
    'lifted_bab_new_incumbent': None,
    'lifted_bab_spent': 0.0, 'lifted_reobbt_time': 0.0,
}


def test_split_history_declares_mechanism_fields_with_getattr_defaults():
    from history import SplitHistory
    history = SplitHistory()
    for name, default in DECLARED.items():
        assert name in vars(history), name
        assert getattr(history, name) == default, name


class _FakeCertificate:
    def __init__(self, dual):
        self.dual = dual
        self.calls = []

    def solve(self, x, budget, primal, tol):
        self.calls.append((budget, primal, tol))
        return self.dual, 3


def test_root_certificate_closes_within_tolerance_and_rounds_down():
    from input_split.batch_branch_and_bound import _root_certificate_dual
    primal = torch.tensor([10.0], dtype=torch.float64)
    cert = _FakeCertificate(9.9995)
    dual = _root_certificate_dual('GP', cert, 'gp_certificate_time', {}, None, primal,
                                  timeout=100.0, start=time.time())
    assert dual is not None and dual <= 9.9995
    budget, p, tol = cert.calls[0]
    assert p == 10.0 and math.isclose(tol, 1e-3) and budget == 20.0


def test_root_certificate_inconclusive_returns_none():
    from input_split.batch_branch_and_bound import _root_certificate_dual
    primal = torch.tensor([10.0], dtype=torch.float64)
    assert _root_certificate_dual('exp', _FakeCertificate(9.0), 'exp_certificate_time', {},
                                  None, primal, timeout=100.0, start=time.time()) is None
    assert _root_certificate_dual('exp', _FakeCertificate(float('nan')), 'exp_certificate_time',
                                  {}, None, primal, timeout=100.0, start=time.time()) is None


def test_instance_state_is_reset_at_the_start_of_each_instance():
    import inspect
    import arguments
    from input_split import batch_branch_and_bound as bab
    arguments.Globals["discarded_threshold_min"] = 123.0
    bab._reset_instance_state()
    assert arguments.Globals["discarded_threshold_min"] is None
    source = inspect.getsource(bab.input_bab_parallel)
    assert source.index("_reset_instance_state()") < source.index("RootState(")


def test_qtb_covers_linear_rows_predicate():
    import types
    from input_split.lifted_bound import _qtb_covers_linear_rows
    linear = types.SimpleNamespace(rows=(None, None, torch.zeros(2, 3)), cleared_rhs=None, atoms=None)
    quadratic = types.SimpleNamespace(rows=(None, None, torch.ones(2, 3)), cleared_rhs=None, atoms=None)
    convex = types.SimpleNamespace(qtb=types.SimpleNamespace(lam_lb=0.0))
    concave = types.SimpleNamespace(qtb=types.SimpleNamespace(lam_lb=-1.0))
    assert _qtb_covers_linear_rows(convex, linear)
    assert not _qtb_covers_linear_rows(convex, quadratic)
    assert not _qtb_covers_linear_rows(concave, linear)
    assert not _qtb_covers_linear_rows(types.SimpleNamespace(qtb=None), linear)


def test_root_structure_summary_and_skip():
    import torch
    from torch import nn
    from auto_LiRPA import BoundedModule
    from input_split.batch_branch_and_bound import _root_structure
    from nlp_solver.structure import RowPool

    class Model(nn.Module):
        def forward(self, x):
            return torch.cat([x[:, :1] ** 2, x[:, 1:2] - 1], 1)

    x0 = torch.zeros(1, 2, dtype=torch.float64)
    net = type('Net', (), {})()
    net.net = BoundedModule(Model(), x0)
    ptb = type('Ptb', (), {'x_L': -torch.ones(1, 2, dtype=torch.float64),
                           'x_U': torch.ones(1, 2, dtype=torch.float64)})()
    x = type('X', (), {'ptb': ptb})()
    C = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    rhs = torch.zeros(1, 2, dtype=torch.float64)
    pool = RowPool(rhs)
    report = _root_structure(net, x, C, rhs, {'structure_report': True}, pool)
    assert report is not None and report.objective.cls == 'quad' and report.pool is pool
    assert _root_structure(net, x, C, rhs, {'structure_report': False}, pool) is None


def test_root_structure_fails_closed():
    import torch
    from unittest.mock import MagicMock
    from input_split.batch_branch_and_bound import _root_structure
    from nlp_solver.structure import RowPool

    net = type('Net', (), {})()
    net.net = MagicMock()
    ptb = type('Ptb', (), {'x_L': -torch.ones(1, 2), 'x_U': torch.ones(1, 2)})()
    x = type('X', (), {'ptb': ptb})()
    C = torch.eye(2).unsqueeze(0)
    assert _root_structure(net, x, C, torch.zeros(1, 2), {'structure_report': True}, RowPool(torch.zeros(1, 2))) is None
