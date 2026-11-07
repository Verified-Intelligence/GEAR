import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arguments
arguments.Config.parse_config(args=[], verbose=False)

from beta_CROWN_solver import _collect_ibp_reference_bounds


class _Node:
    def __init__(self, name, perturbed, lower=None, upper=None):
        self.name = name
        self.perturbed = perturbed
        self.lower = lower
        self.upper = upper


class _Net:
    def __init__(self, exc=None):
        self.exc = exc
        self.calls = []
        self.lower = torch.tensor([0.0])
        self.upper = torch.tensor([1.0])
        self._nodes = [
            _Node("kept", True, self.lower, self.upper),
            _Node("unperturbed", False, self.lower, self.upper),
            _Node("missing_bounds", True, None, self.upper),
        ]

    def compute_bounds(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return (torch.tensor([[0.0]]),)

    def nodes(self):
        return list(self._nodes)


def test_collect_ibp_reference_bounds_skips_failed_ibp():
    net = _Net(exc=ValueError("unsupported interval"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bounds = _collect_ibp_reference_bounds(net, torch.zeros(1, 1), torch.ones(1, 1, 1))

    assert bounds == {}
    assert len(net.calls) == 1
    assert net.calls[0]["method"] == "ibp"
    assert any("IBP enhancement failed" in str(item.message) for item in caught)


def test_collect_ibp_reference_bounds_keeps_perturbed_tensor_bounds():
    net = _Net()

    bounds = _collect_ibp_reference_bounds(net, torch.zeros(1, 1), torch.ones(1, 1, 1))

    assert list(bounds) == ["kept"]
    lower, upper = bounds["kept"]
    torch.testing.assert_close(lower, net.lower)
    torch.testing.assert_close(upper, net.upper)
