"""The incumbent's coordinates must leave the solver so a runner can replay them.

Every reduced-grid runner sets ABCROWN_BEST_PRIMAL_PATH and replays the saved point
against the exact constraints at zero tolerance; nothing in the verifier ever wrote
that file, so every "solved" claim came back as `solver_claim_replay_infeasible`.
"""
import os
from types import SimpleNamespace

import torch

from input_split.batch_branch_and_bound import export_best_primal


def _run(tmp_path, monkeypatch, history_x, root_x, found=True):
    path = tmp_path / "best_primal.pt"
    monkeypatch.setenv("ABCROWN_BEST_PRIMAL_PATH", str(path))
    net = SimpleNamespace()
    domains = SimpleNamespace(history=SimpleNamespace(alm_best_primal_x=history_x))
    x = export_best_primal(net, domains, root_x, torch.tensor([0.5]), found)
    return path, net, x


def test_in_bab_incumbent_wins_over_root_point(tmp_path, monkeypatch):
    path, net, x = _run(tmp_path, monkeypatch,
                        history_x=torch.tensor([0.9, 0.04], dtype=torch.float64),
                        root_x=torch.tensor([[0.1, 0.2]], dtype=torch.float64))
    saved = torch.load(path)
    assert torch.equal(saved, torch.tensor([0.9, 0.04], dtype=torch.float64))
    assert torch.equal(net._best_primal_x, saved) and torch.equal(x, saved)


def test_root_point_is_used_when_bab_found_nothing_better(tmp_path, monkeypatch):
    path, _, _ = _run(tmp_path, monkeypatch, history_x=None,
                      root_x=torch.tensor([[0.1, 0.2]], dtype=torch.float64))
    assert torch.equal(torch.load(path), torch.tensor([0.1, 0.2], dtype=torch.float64))


def test_nothing_is_written_without_an_incumbent(tmp_path, monkeypatch):
    path, net, x = _run(tmp_path, monkeypatch, history_x=None, root_x=None, found=False)
    assert not path.exists() and x is None and net._best_primal_x is None


def test_no_env_var_still_records_on_net(tmp_path, monkeypatch):
    monkeypatch.delenv("ABCROWN_BEST_PRIMAL_PATH", raising=False)
    net = SimpleNamespace()
    domains = SimpleNamespace(history=SimpleNamespace(alm_best_primal_x=torch.tensor([1.0, 2.0])))
    x = export_best_primal(net, domains, None, torch.tensor([0.5]), True)
    assert torch.equal(net._best_primal_x, torch.tensor([1.0, 2.0])) and torch.equal(x, net._best_primal_x)
    assert not os.path.exists("best_primal.pt")
