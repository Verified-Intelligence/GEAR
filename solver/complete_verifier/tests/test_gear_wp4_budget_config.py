"""model.op_budget.* config schema (G5 section 4.1): defaults, dump omission, LiRPANet wiring. CPU only."""
import pytest
import torch

import arguments
from arguments import ConfigHandler


def parse(argv):
    h = ConfigHandler()
    h.parse_config(argv, verbose=False)
    return h


def test_defaults_are_todays_values():
    h = parse([])
    assert h['model']['op_budget'] == {
        'hidden_squares': 1 << 20, 'complete_squares': 1 << 24, 'prove_rows': 1 << 23, 'commit': 'partial'}


def test_dump_omits_op_budget_at_defaults_only():
    h = parse([])
    assert 'op_budget' not in h.dump_config(h.all_args)
    h = parse(['--op_budget_prove_rows', '5'])
    dump = h.dump_config(h.all_args)
    assert 'op_budget:' in dump and 'prove_rows: 5' in dump
    h = parse(['--op_budget_commit', 'whole'])
    assert 'commit: whole' in h.dump_config(h.all_args)


def test_yaml_key_is_accepted(tmp_path):
    f = tmp_path / 'c.yaml'
    f.write_text('model:\n  op_budget:\n    hidden_squares: 7\n')
    h = parse(['--config', str(f)])
    assert h['model']['op_budget']['hidden_squares'] == 7
    assert h['model']['op_budget']['prove_rows'] == 1 << 23


def test_lirpanet_passes_budget_and_refuses_bound_opts_literal():
    from beta_CROWN_solver import LiRPANet
    arguments.Config.parse_config([], verbose=False)
    arguments.Config['model']['op_budget']['hidden_squares'] = 11
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    net = LiRPANet(model, in_size=(1, 2), device='cpu')
    assert net.net.bound_opts['op_budget'] == {
        'hidden_squares': 11, 'complete_squares': 1 << 24, 'prove_rows': 1 << 23, 'commit': 'partial'}
    arguments.Config['model']['bound_opts'] = "{'op_budget': {'hidden_squares': 3}}"
    with pytest.raises(ValueError, match='op_budget'):
        LiRPANet(model, in_size=(1, 2), device='cpu')
