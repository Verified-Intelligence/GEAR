"""Unit tests for input_split/strong_branching.py."""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _config():
    import arguments
    saved = arguments.Config
    cfg = arguments.ConfigHandler()
    cfg.construct_config_dict(cfg.default_args)
    cfg.file = None
    arguments.Config = cfg
    yield
    arguments.Config = saved


def test_candidate_children_layout():
    from input_split.strong_branching import candidate_children
    x_L = torch.tensor([[0., -2., 1.]])
    x_U = torch.tensor([[4., 2., 3.]])
    cand = torch.tensor([[1, 0]])
    lo, hi = candidate_children(x_L, x_U, cand)
    assert lo.shape == hi.shape == (4, 3)
    # [k=0 (dim 1)][lower, upper], then [k=1 (dim 0)][lower, upper]
    assert torch.equal(lo[0], x_L[0]) and torch.equal(hi[0], torch.tensor([4., 0., 3.]))
    assert torch.equal(lo[1], torch.tensor([0., 0., 1.])) and torch.equal(hi[1], x_U[0])
    assert torch.equal(lo[2], x_L[0]) and torch.equal(hi[2], torch.tensor([2., 2., 3.]))
    assert torch.equal(lo[3], torch.tensor([2., -2., 1.])) and torch.equal(hi[3], x_U[0])
    # The two children of a candidate cover the parent exactly.
    for k in range(2):
        a, b = 2 * k, 2 * k + 1
        assert torch.equal(torch.minimum(lo[a], lo[b]), x_L[0])
        assert torch.equal(torch.maximum(hi[a], hi[b]), x_U[0])
        assert torch.equal(hi[a][cand[0, k]], lo[b][cand[0, k]])


def test_child_objective_gains_pruning_rules():
    from input_split.strong_branching import child_objective_gains
    parent = torch.tensor([1.0])
    thr = torch.tensor([[3.0, 1e-4]])                    # incumbent 3, one constraint row (relaxed by 1e-4 like the pool)
    child = torch.tensor([[[[1.5, -1.0], [0.5, -1.0]],   # gain 0.5; below parent -> 0
                           [[3.5, -1.0], [2.0, 0.5]],    # objective-pruned; constraint-pruned
                           [[float("inf"), -1.0], [2.0, 5e-5]]]])  # +inf; within tolerance
    g = child_objective_gains(parent, child, thr)
    assert g.shape == (1, 3, 2)
    assert torch.allclose(g[0, 0], torch.tensor([0.5, 0.0]))
    assert torch.allclose(g[0, 1], torch.tensor([2.5, 2.0]))    # pruned -> gain >= gap 2
    assert torch.allclose(g[0, 2], torch.tensor([2.0, 1.0]))


def test_child_objective_gains_without_incumbent():
    from input_split.strong_branching import child_objective_gains
    g = child_objective_gains(torch.tensor([0.0]),
                              torch.tensor([[[[float("inf")], [1.0]]]]),
                              torch.tensor([[float("inf")]]))
    assert torch.isposinf(g[0, 0, 0]) and g[0, 0, 1] == 1.0


def test_product_score_prefers_balanced_children():
    from input_split.strong_branching import product_score
    g = torch.tensor([[[1.0, 1.0], [2.0, 0.0], [float("inf"), 0.5]]])
    s = product_score(g)
    assert s[0, 0] > s[0, 1]
    assert torch.isfinite(s).all() and s[0, 2] > s[0, 0]


class _QuadNet:
    """f(x) = sum_i c_i x_i^2 bounded by interval arithmetic: the lower bound of a box
    is sum_i c_i min_{x_i in box} x_i^2, so splitting the widest-curvature input of a
    box straddling 0 does nothing for the child containing 0 while splitting an input
    with a positive box gains in both children."""

    def __init__(self, c):
        self.c = c
        self.calls = 0

    def get_lower_bound_naive(self, dm_lb, dm_l, dm_u, history, bounding_method, C,
                              stop_criterion, thresholds, return_A, return_b, constraints,
                              base_input_linear_constraints, stats):
        self.calls += 1
        stats.visited += len(dm_l)
        m = torch.where((dm_l <= 0) & (dm_u >= 0), torch.zeros_like(dm_l),
                        torch.minimum(dm_l.abs(), dm_u.abs())) ** 2
        lb = (m * self.c).sum(1, keepdim=True)
        history.lp_mean_y = "clobbered"
        return torch.max(lb, dm_lb), None, None, None


def test_strong_branching_picks_the_input_that_moves_the_weaker_child():
    from input_split.strong_branching import strong_branching_decision
    net = _QuadNet(torch.tensor([10.0, 1.0]))
    x_L = torch.tensor([[-1.0, 1.0]])
    x_U = torch.tensor([[1.0, 3.0]])
    dm_lb = net.get_lower_bound_naive(torch.full((1, 1), -1e9), x_L, x_U, SimpleNamespace(),
                                      None, None, None, None, None, None, None, None,
                                      SimpleNamespace(visited=0))[0]
    history = SimpleNamespace(lp_mean_y="keep")
    stats = SimpleNamespace(visited=7)
    base = torch.tensor([[10.0, 1.0]])       # a first-order score would pick input 0
    chosen = strong_branching_decision(
        net, history, dm_lb, x_L, x_U, torch.ones(1, 1, 1), torch.tensor([[100.0]]), None, None,
        base, torch.tensor([[0]]), -1, "crown", None, stats, max_children=2)
    # Input 0 straddles 0: its lower child keeps bound 1 (x1 >= 1). Input 1 raises both.
    assert chosen.tolist() == [[1]]
    assert history.lp_mean_y == "keep"       # warm starts restored
    assert stats.visited == 7                # candidate children are not tree nodes
    assert net.calls == 1 + 1                # chunks hold whole domains (4 children)


def test_strong_branching_falls_back_when_nothing_improves():
    from input_split.strong_branching import strong_branching_decision
    net = _QuadNet(torch.tensor([1.0, 1.0]))
    x_L = torch.tensor([[-1.0, -1.0]])
    x_U = torch.tensor([[1.0, 1.0]])
    chosen = strong_branching_decision(
        net, SimpleNamespace(), torch.zeros(1, 1), x_L, x_U, torch.ones(1, 1, 1), torch.tensor([[5.0]]),
        None, None, torch.tensor([[0.0, 1.0]]), torch.tensor([[0]]), -1, "crown", None,
        SimpleNamespace(visited=0), max_children=64)
    assert chosen.tolist() == [[0]]


def test_clipped_away_child_counts_as_pruned():
    from input_split.strong_branching import evaluate_candidates
    net = _QuadNet(torch.tensor([1.0, 1.0]))
    x_L = torch.tensor([[-1.0, -1.0]])
    x_U = torch.tensor([[1.0, 1.0]])
    seen = []

    def clip(cl, cu, s, e, reps):
        seen.append((s, e, reps, cl.shape[0]))
        cu = cu.clone()
        cu[0, 0] = cl[0, 0] - 1.0       # the lower child of input 0 clips to an empty box
        return cl, cu

    g = evaluate_candidates(net, SimpleNamespace(), torch.zeros(1, 1), x_L, x_U,
                            torch.ones(1, 1, 1), torch.tensor([[5.0]]), None, None,
                            torch.tensor([[0, 1]]), "crown", None,
                            SimpleNamespace(visited=0), 64, clip=clip)
    assert seen == [(0, 1, 4, 4)]
    assert g[0, 0, 0] == 5.0             # discarded: closes the whole gap to the incumbent
    assert g[0, 0, 1] == 0.0 and (g[0, 1] == 0).all()


class _WidthNet(_QuadNet):
    """lb = -sum_j c_j w_j: halving input j lifts BOTH children by c_j w_j / 2."""

    def get_lower_bound_naive(self, dm_lb, dm_l, dm_u, history, bounding_method, C,
                              stop_criterion, thresholds, return_A, return_b, constraints,
                              base_input_linear_constraints, stats):
        lb = -((dm_u - dm_l) * self.c).sum(1, keepdim=True)
        return torch.max(lb, dm_lb), None, None, None


def test_strong_branching_split_only_touches_the_lowest_domains():
    import arguments
    from input_split.batch_branch_and_bound import _strong_branching_split
    arguments.Config["bab"]["clip_n_verify"]["clip_input_domain"]["enabled"] = False
    net = _WidthNet(torch.tensor([1.0, 3.0]))           # strong branching picks input 1
    B = 4
    x_L, x_U = torch.zeros(B, 2), torch.ones(B, 2)
    dm_lb = torch.tensor([[-4.0], [-9.0], [-5.0], [-8.0]])
    lA = torch.tensor([[[5.0, 0.0]]]).expand(B, 1, 2)   # sb picks input 0
    split_idx = torch.zeros(B, 1, dtype=torch.long)
    args = (net, SimpleNamespace(root_width=None), dm_lb, x_L, x_U,
            torch.ones(1, 1, 1).expand(B, 1, 1), torch.full((B, 1), 96.0), None, None, lA,
            None, split_idx, -1, "crown", None, SimpleNamespace(visited=0), 64)
    out = _strong_branching_split(*args, max_domains=2)
    assert out[:, 0].tolist() == [0, 1, 0, 1]            # the two lowest bounds: rows 1, 3
    assert split_idx[:, 0].tolist() == [0, 0, 0, 0]      # input not modified
    assert _strong_branching_split(*args, max_domains=1)[:, 0].tolist() == [0, 1, 0, 0]
    assert _strong_branching_split(*args, max_domains=0)[:, 0].tolist() == [0, 0, 0, 0]


def test_cheap_lookahead_disables_dual_ascent_only_inside_the_call():
    import arguments
    from input_split.strong_branching import evaluate_candidates
    nlp = arguments.Config["solver"]["nlp"]
    nlp["dual_ascent_enabled"] = True
    seen = []

    class _Net(_WidthNet):
        def get_lower_bound_naive(self, *a, **k):
            seen.append(nlp["dual_ascent_enabled"])
            return super().get_lower_bound_naive(*a, **k)

    evaluate_candidates(_Net(torch.tensor([1.0, 1.0])), SimpleNamespace(), torch.zeros(1, 1),
                        torch.zeros(1, 2), torch.ones(1, 2), torch.ones(1, 1, 1),
                        torch.tensor([[5.0]]), None, None, torch.tensor([[0, 1]]), "crown",
                        None, SimpleNamespace(visited=0), 64)
    assert seen == [False] and nlp["dual_ascent_enabled"] is True


def test_objective_only_lookahead_takes_intermediate_bounds_from_ibp():
    import arguments
    from input_split.strong_branching import evaluate_candidates
    seen = []

    class _Net(_WidthNet):
        def get_lower_bound_naive(self, dm_lb, dm_l, dm_u, history, bounding_method, *a, **k):
            seen.append(bounding_method)
            return super().get_lower_bound_naive(dm_lb, dm_l, dm_u, history, bounding_method, *a, **k)

    def run(n_rows, base):
        seen.clear()
        evaluate_candidates(_Net(torch.tensor([1.0, 1.0])), SimpleNamespace(), torch.zeros(1, n_rows),
                            torch.zeros(1, 2), torch.ones(1, 2), torch.ones(1, n_rows, 1),
                            torch.full((1, n_rows), 5.0), None, base, torch.tensor([[0, 1]]),
                            "crown", None, SimpleNamespace(visited=0), 64)
        return seen[0]

    assert run(1, None) == "crown-ibp"
    assert run(2, None) == "crown"                             # constraint rows tighten intermediates
    assert run(1, (torch.zeros(1, 1, 2), torch.zeros(1, 1))) == "crown"


def test_strong_candidates_per_problem_class():
    from input_split.batch_branch_and_bound import _strong_candidates
    cfg = {"sb_strong_candidates": 3, "sb_strong_candidates_unconstrained": -1,
           "sb_strong_max_inputs_unconstrained": 6}
    assert _strong_candidates(cfg, 5, False, 20) == 3       # constraint rows
    assert _strong_candidates(cfg, 1, True, 4) == 3         # input linear constraints
    assert _strong_candidates(cfg, 1, False, 6) == -1       # objective only, few inputs
    assert _strong_candidates(cfg, 1, False, 10) == 0       # objective only, too many
    assert _strong_candidates({}, 1, False, 2) == 0         # defaults: off


def test_lookahead_chunk_shrinks_on_oom_and_remembers():
    import input_split.strong_branching as sbm
    from input_split.strong_branching import evaluate_candidates

    class _OOMNet(_WidthNet):
        def get_lower_bound_naive(self, *a, **k):
            if len(k["dm_l"]) > 4:
                raise torch.OutOfMemoryError("fake")
            return super().get_lower_bound_naive(*a, **k)

    sbm._CHILD_CAP[0] = None
    net = _OOMNet(torch.tensor([1.0, 1.0]))
    B = 3
    g = evaluate_candidates(net, SimpleNamespace(), torch.full((B, 1), -2.0), torch.zeros(B, 2),
                            torch.ones(B, 2), torch.ones(1, 1, 1).expand(B, 1, 1),
                            torch.full((B, 1), 5.0), None, None,
                            torch.tensor([[0, 1]] * B), "crown", None,
                            SimpleNamespace(visited=0), 64)
    assert g.shape == (B, 2, 2) and torch.allclose(g, torch.full_like(g, 0.5))
    assert sbm._CHILD_CAP[0] == 4                 # one domain (4 children) per call fits
    sbm._CHILD_CAP[0] = None


def test_strong_domain_count():
    from input_split.batch_branch_and_bound import _strong_domain_count
    cfg = {"sb_strong_domain_frac": 0.05, "sb_strong_min_domains": 256,
           "sb_strong_max_domains": 10000}
    assert _strong_domain_count(cfg, 262144, False) == 10000   # capped
    assert _strong_domain_count(cfg, 8192, False) == 410       # 5%
    assert _strong_domain_count(cfg, 1000, False) == 256       # at least 256
    assert _strong_domain_count(cfg, 100, False) == 100        # never more than the batch
    assert _strong_domain_count(cfg, 8192, True) == 8192       # objective only: frac 1
    cfg["sb_strong_domain_frac_unconstrained"] = 0.5
    assert _strong_domain_count(cfg, 8192, True) == 4096
    assert _strong_domain_count({}, 77, False) == 77           # defaults: every domain


