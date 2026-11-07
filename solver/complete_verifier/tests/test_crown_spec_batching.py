"""Tests for final-spec-row CROWN batching."""
import os
import sys

import torch
from torch import nn

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arguments
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_batch_any
from crown_spec_batching import compute_bounds_in_spec_batches
from input_split.bounding import _get_crown_spec_batch_size


def _assert_nested_tensors_allclose(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_tensors_allclose(actual[key], expected[key])
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_tensors_allclose(actual_item, expected_item)
    else:
        assert actual == expected


def test_get_crown_spec_batch_size_uses_config_handler_defaults():
    """The helper must work with the real arguments.ConfigHandler."""
    original_config = arguments.Config
    arguments.Config = arguments.ConfigHandler()
    arguments.Config.parse_config(args=[], verbose=False)
    try:
        # Default is a no-op sentinel (1e9): spec batching is opt-in per-instance
        # via solver.crown.spec_batch_size, so it never changes existing behavior.
        assert _get_crown_spec_batch_size() == int(1e9)
    finally:
        arguments.Config = original_config


def test_compute_bounds_in_spec_batches_chunks_and_concatenates_return_A():
    """Large C[:, spec, :] calls are split and reassembled in row order."""
    calls = []

    class MockNet:
        input_name = ["input"]
        output_name = ["output"]

        def compute_bounds(self, **kwargs):
            C = kwargs["C"]
            calls.append(C.clone())
            row_ids = C[:, :, 0].to(torch.float32)
            lA = row_ids.unsqueeze(-1).repeat(1, 1, 3)
            lbias = row_ids + 100.0
            return (
                row_ids,
                None,
                {
                    self.output_name[0]: {
                        self.input_name[0]: {
                            "lA": lA,
                            "lbias": lbias,
                            "unstable_idx": torch.tensor([1, 2]),
                        }
                    }
                },
            )

    C = torch.zeros(1, 5, 4)
    C[0, :, 0] = torch.arange(5)

    lb, ub, A = compute_bounds_in_spec_batches(
        MockNet(),
        spec_batch_size=2,
        x=("x",),
        C=C,
        method="backward",
        bound_upper=False,
        return_A=True,
        needed_A_dict={"output": {"input"}},
    )

    assert ub is None
    assert len(calls) == 3
    assert [call.shape[1] for call in calls] == [2, 2, 1]
    assert torch.equal(lb, torch.arange(5, dtype=torch.float32).view(1, 5))
    assert torch.equal(
        A["output"]["input"]["lA"][:, :, 0],
        torch.arange(5, dtype=torch.float32).view(1, 5),
    )
    assert torch.equal(
        A["output"]["input"]["lbias"],
        torch.arange(100, 105, dtype=torch.float32).view(1, 5),
    )
    assert torch.equal(A["output"]["input"]["unstable_idx"], torch.tensor([1, 2]))


def test_compute_bounds_in_spec_batches_restores_net_lA_state():
    """Callers that read lA from net state after compute_bounds need merged lA."""

    class InputNode:
        lA = None

    class MockNet:
        input_name = ["input"]
        output_name = ["output"]

        def __init__(self):
            self.input_node = InputNode()

        def __getitem__(self, name):
            if name == self.input_name[0]:
                return self.input_node
            raise KeyError(name)

        def compute_bounds(self, **kwargs):
            C = kwargs["C"]
            row_ids = C[:, :, 0].to(torch.float32)
            lA = row_ids.unsqueeze(-1).repeat(1, 1, 3)
            lbias = row_ids + 100.0
            self.input_node.lA = lA.transpose(0, 1)
            return (
                row_ids,
                None,
                {
                    self.output_name[0]: {
                        self.input_name[0]: {
                            "lA": lA,
                            "lbias": lbias,
                            "unstable_idx": None,
                        }
                    }
                },
            )

    C = torch.zeros(1, 5, 4)
    C[0, :, 0] = torch.arange(5)
    net = MockNet()

    _, _, A = compute_bounds_in_spec_batches(
        net,
        spec_batch_size=2,
        x=("x",),
        C=C,
        method="backward",
        bound_upper=False,
        return_A=True,
        needed_A_dict={"output": {"input"}},
    )

    expected_lA = A["output"]["input"]["lA"].transpose(0, 1)
    assert torch.equal(net["input"].lA, expected_lA)


def test_compute_bounds_in_spec_batches_keeps_lA_aligned_with_pruned_specs():
    """Spec-batched build returns should match the post-filtered lb shape."""

    class InputNode:
        lA = None

    class MockNet:
        input_name = ["input"]
        output_name = ["output"]

        def __init__(self, rhs):
            self.input_node = InputNode()
            self.bound_opts = {
                "optimize_bound_args": {
                    "stop_criterion_func": stop_criterion_batch_any(rhs),
                },
            }

        def __getitem__(self, name):
            if name == self.input_name[0]:
                return self.input_node
            raise KeyError(name)

        def compute_bounds(self, **kwargs):
            C = kwargs["C"]
            row_ids = C[:, :, 0].to(torch.float32)
            # The first 2048 specs are already verified; the last 1265 remain.
            lb = torch.where(row_ids < 2048, torch.ones_like(row_ids), -torch.ones_like(row_ids))
            lA = row_ids.unsqueeze(-1).repeat(1, 1, 2)
            lbias = row_ids + 100.0
            self.input_node.lA = lA.transpose(0, 1)
            return (
                lb,
                None,
                {
                    self.output_name[0]: {
                        self.input_name[0]: {
                            "lA": lA,
                            "lbias": lbias,
                            "unstable_idx": None,
                        }
                    }
                },
            )

    num_specs = 3313
    num_unverified = 1265
    C = torch.zeros(1, num_specs, 3)
    C[0, :, 0] = torch.arange(num_specs)
    rhs = torch.zeros(1, num_specs)
    net = MockNet(rhs)

    lb, _, A = compute_bounds_in_spec_batches(
        net,
        spec_batch_size=2048,
        x=("x",),
        C=C,
        method="backward",
        bound_upper=False,
        return_A=True,
        needed_A_dict={"output": {"input"}},
    )

    lA = A["output"]["input"]["lA"]
    lbias = A["output"]["input"]["lbias"]
    assert lb.shape == (1, num_unverified)
    assert lA.shape[:2] == lb.shape
    assert lbias.shape == lb.shape
    assert net["input"].lA.transpose(0, 1).shape[:2] == lb.shape
    assert torch.equal(lA[0, :, 0], torch.arange(2048, num_specs, dtype=torch.float32))


def test_compute_bounds_in_spec_batches_matches_full_compute_bounds_return_A():
    """Spec-batched compute_bounds must match a single full-C call exactly."""
    torch.manual_seed(1234)

    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    x = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32)
    ptb = PerturbationLpNorm(x_L=x - 0.05, x_U=x + 0.05)
    bounded_x = BoundedTensor(x, ptb)
    full_model = BoundedModule(model, bounded_x, device="cpu")
    batched_model = BoundedModule(model, bounded_x, device="cpu")

    C = torch.tensor([[
        [1.0, -1.0],
        [-1.0, 1.0],
        [0.5, 0.25],
        [-0.75, 0.1],
        [0.3, -0.6],
    ]])
    needed_A_dict = {
        full_model.output_name[0]: {full_model.input_name[0]},
    }

    full_ret = full_model.compute_bounds(
        x=(bounded_x,),
        C=C,
        method="CROWN",
        bound_upper=False,
        return_A=True,
        needed_A_dict=needed_A_dict,
    )
    batched_ret = compute_bounds_in_spec_batches(
        batched_model,
        spec_batch_size=2,
        x=(bounded_x,),
        C=C,
        method="CROWN",
        bound_upper=False,
        return_A=True,
        needed_A_dict=needed_A_dict,
    )

    _assert_nested_tensors_allclose(batched_ret, full_ret)


def test_compute_bounds_in_spec_batches_retries_smaller_chunks_on_oom():
    """With oom_retry, a call that runs out of GPU memory is repeated with half as
    many rows per chunk; the rows' bounds are the same as one full call."""
    calls = []

    class MockNet:
        input_name = ["input"]
        output_name = ["output"]

        def compute_bounds(self, **kwargs):
            C = kwargs["C"]
            calls.append(C.shape[1])
            if C.shape[1] > 2:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return (C[:, :, 0].to(torch.float32), None)

    C = torch.zeros(1, 5, 4)
    C[0, :, 0] = torch.arange(5)
    lb, ub = compute_bounds_in_spec_batches(
        MockNet(), spec_batch_size=int(1e9), oom_retry=True,
        x=("x",), C=C, method="backward", bound_upper=False)
    assert calls == [5, 2, 2, 1]
    assert torch.equal(lb, torch.arange(5, dtype=torch.float32).view(1, 5))

    # Without the flag the error propagates unchanged.
    try:
        compute_bounds_in_spec_batches(
            MockNet(), spec_batch_size=int(1e9), x=("x",), C=C,
            method="backward", bound_upper=False)
    except torch.OutOfMemoryError:
        pass
    else:
        raise AssertionError("expected OutOfMemoryError")


def test_retained_crown_keeps_every_spec_and_matches_an_uninterrupted_bound():
    from nlp_solver.continuation import Continuation
    torch.manual_seed(891)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.tensor([[.2, -.1, .3]], dtype=torch.float32)
    bounded_x = BoundedTensor(x, PerturbationLpNorm(x_L=x-.05, x_U=x+.05))
    direct, sliced = BoundedModule(model, bounded_x), BoundedModule(model, bounded_x)
    C = torch.tensor([[[1., -1.], [-1., 1.], [.5, .25], [-.75, .1], [.3, -.6]]])
    kwargs = dict(x=(bounded_x,), C=C, method='CROWN', bound_upper=False, return_A=True,
                  needed_A_dict={direct.output_name[0]: {direct.input_name[0]}})
    expected = direct.compute_bounds(**kwargs)
    state = Continuation(compute_bounds_in_spec_batches(sliced, spec_batch_size=2, _sliced=True,
                                                        _work_cap=lambda: 0., _iterate=True, **kwargs))
    chunks = 0
    while not state.done:
        chunks += state.step() is not None
        torch.rand(5)
    assert chunks == C.shape[1]  # a zero quote requests the smallest abortable row
    _assert_nested_tensors_allclose(state.result, expected)
    assert sliced[sliced.input_name[0]].lA.shape[0] == C.shape[1]


def test_retained_ibp_enhancement_yields_and_preserves_all_returned_planes():
    from types import SimpleNamespace
    from input_split.bounding import get_lower_bound_with_ibp_enhancement
    from nlp_solver.continuation import Continuation
    import arguments
    torch.manual_seed(892)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.tensor([[.2, -.1, .3]], dtype=torch.float32)
    bounded_x = BoundedTensor(x, PerturbationLpNorm(x_L=x-.05, x_U=x+.05))
    C = torch.tensor([[[1., -1.], [-1., 1.], [.5, .25], [-.75, .1], [.3, -.6]]])
    direct, sliced = BoundedModule(model, bounded_x), BoundedModule(model, bounded_x)
    thresholds = torch.full((1, C.shape[1]), 100.)
    def run(net, **extra):
        return get_lower_bound_with_ibp_enhancement(
            SimpleNamespace(net=net), bounded_x, C, 'crown', thresholds, return_A=True,
            needed_A_dict={net.output_name[0]: {net.input_name[0]}}, **extra)
    previous = arguments.Config['bab']['clip_n_verify']['clip_input_domain']['enabled']
    arguments.Config['bab']['clip_n_verify']['clip_input_domain']['enabled'] = False
    try:
        expected = run(direct)
        state = Continuation(run(sliced, _sliced=True, _iterate=True))
        chunks = 0
        while not state.done:
            chunks += state.step() is not None
        assert chunks >= 1
        _assert_nested_tensors_allclose(state.result, expected)
        assert state.result[1].shape[1] == C.shape[1]
    finally:
        arguments.Config['bab']['clip_n_verify']['clip_input_domain']['enabled'] = previous
