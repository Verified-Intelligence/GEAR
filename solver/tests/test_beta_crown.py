import sys
import pytest
import torch.nn as nn
import numpy as np
import random
import torch
from auto_LiRPA.utils import stop_criterion_batch_any

sys.path.insert(0, '../complete_verifier')

import arguments
from beta_CROWN_solver import LiRPANet
from activation_split.bab_bootstrap import general_bab

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.opt_pruner import OptPruner
from testcase import DEFAULT_DEVICE, DEFAULT_DTYPE, set_default_dtype_device


def test_new_sparse_beta_uses_node_dtype():
    center = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    bounded_model = BoundedModule(nn.Identity().double(), center)
    node = bounded_model.final_node()

    bounded_model.reset_beta(
        node, shape=(1, 2), betas=[None], bias=True)

    sparse_beta = node.sparse_betas[0]
    assert sparse_beta.val.dtype == torch.float64
    assert sparse_beta.sign.dtype == torch.float64
    assert sparse_beta.bias.dtype == torch.float64
    assert sparse_beta.loc.dtype == torch.int64


def test_splittable_activations_include_first_bound_target():
    """Initial split discovery uses supplied bounds on the target's inputs."""
    model = nn.ReLU()
    center = torch.zeros(1, 2)
    bounded_model = BoundedModule(model, center, device='cpu')
    bounded_x = BoundedTensor(
        center,
        PerturbationLpNorm(
            norm=np.inf, x_L=center - 1, x_U=center + 1))

    input_node = bounded_model[bounded_model.input_name[0]]
    # Verifier callers may supply intermediate bounds before their first solve.
    bounded_model.set_input(
        bounded_x, interm_bounds={input_node.name: (center - 1, center + 1)})
    split_nodes, split_activations = bounded_model.get_split_nodes()

    assert split_nodes == [input_node]
    assert split_activations == {
        input_node.name: [(bounded_model.final_node(), 0)]}


def test_bilinear_split_candidates_follow_input_perturbations():
    """Split candidates track bilinear and affine multiplication."""
    class Product(nn.Module):
        def forward(self, x, y):
            return x * y

    center = torch.ones(1, 2)
    model = BoundedModule(Product(), (center, center), device='cpu')
    bounded = BoundedTensor(
        center, PerturbationLpNorm(norm=np.inf, eps=0.5))
    # Reuse one graph across several input-perturbation configurations.
    for x, y, expected_count in (
            (bounded, bounded, 2), (bounded, center, 0),
            (center, bounded, 0), (bounded, bounded, 2)):
        model.compute_bounds(x=(x, y), method='IBP')
        split_nodes, split_activations = model.get_split_nodes()
        assert len(split_nodes) == expected_count
        if expected_count:
            assert {node.name for node in split_nodes} == set(model.input_name)
            assert {index for entries in split_activations.values()
                    for _, index in entries} == {0, 1}


def cifar_model_wide():
    # cifar wide
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def bab(model_ori, data, target, norm, eps, data_max=None, data_min=None,
        device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):

    if norm == np.inf:
        if data_max is None:
            data_ub = data + eps
            data_lb = data - eps
        else:
            data_ub = torch.min(data + eps, data_max)
            data_lb = torch.max(data - eps, data_min)
    else:
        data_ub = data_lb = data
        
    data = data.to(device=device, dtype=dtype)
    data_lb = data_lb.to(device=device, dtype=dtype)
    data_ub = data_ub.to(device=device, dtype=dtype)
    
    pred = torch.argmax(model_ori(data), dim=1)

    c = torch.zeros((1, 1, 10), dtype=dtype, device=device)  # we only support c with shape of (1, 1, n)
    c[0, 0, pred] = 1
    c[0, 0, target] = -1
    rhs = torch.tensor(arguments.Config["bab"]["decision_thresh"], dtype=dtype, device=device).view(c.shape[:2])

    # LiRPA wrapper
    model = LiRPANet(model_ori, device=device, in_size=(1, 3, 32, 32))

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb)

    # Standalone BaB consumes the same alpha-CROWN bounds as incomplete
    # verification. Keep full alpha metadata for build_with_refined_bounds().
    _, reference_dict = model.build(
        x, c, rhs, stop_criterion_batch_any,
        bounding_method="alpha-crown", full_alpha_info=True)
    min_lb = general_bab(model, x, c, rhs, reference_dict=reference_dict)[0]

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()

    min_lb += arguments.Config["bab"]["decision_thresh"]
    print(min_lb)
    # PGD bound = 0.62972, best certified bound so far is 0.6034 (using CPU)
    # The actual result may be unstable and have some randomness.
    valid_lb = 0.485
    valid_ub = 0.62972
    assert valid_lb < min_lb < valid_ub

# This test takes long time so it is set as the last test case.
@pytest.mark.order(-1)
def test_fsb():
    device = DEFAULT_DEVICE
    dtype = DEFAULT_DTYPE
    set_default_dtype_device(dtype, device)
    model_ori = cifar_model_wide()
    data = torch.load('data/beta_crown_test_data')
    model_ori.load_state_dict(data['state_dict'])
    model_ori = model_ori.to(device=device, dtype=dtype)
    x = data['x']
    pidx = data['pidx']
    eps_temp = data['eps_temp']
    data_max = data['data_max']
    data_min = data['data_min']

    arguments.Config.parse_config(args={})

    arguments.Config['general']['device'] = 'cpu'
    arguments.Config["solver"]["batch_size"] = 200
    arguments.Config["bab"]["decision_thresh"] = np.float64(10)  # naive float obj has no max() function, np.inf will lead infeasible domain
    arguments.Config["solver"]["beta-crown"]["iteration"] = 20
    arguments.Config["bab"]["timeout"] = 60
    arguments.Config["solver"]["alpha-crown"]["lr_alpha"] = 0.1
    arguments.Config["solver"]["beta-crown"]["lr_beta"] = 0.1
    arguments.Config["bab"]["sort_domain_interval"] = 1
    arguments.Config["bab"]["branching"]["method"] = 'fsb'
    arguments.Config["bab"]["branching"]["candidates"] = 2
    arguments.Config["data"]["dataset"] = 'cifar'

    bab(model_ori, x, pidx, float('inf'), eps_temp,
        data_max=data_max, data_min=data_min, device=device, dtype=dtype)

@pytest.mark.order(-2)
def test_pruning_with_intermediate_layer_optimization(mocker):
    device = DEFAULT_DEVICE
    dtype = DEFAULT_DTYPE
    set_default_dtype_device(dtype, device)
    model_ori = cifar_model_wide()
    data = torch.load('data/beta_crown_test_data')
    model_ori.load_state_dict(data['state_dict'])
    model_ori = model_ori.to(device=device, dtype=dtype)
    x = data['x']
    pidx = data['pidx']
    eps_temp = data['eps_temp']
    data_max = data['data_max']
    data_min = data['data_min']

    arguments.Config.parse_config(args={})

    arguments.Config['general']['device'] = 'cpu'
    arguments.Config["solver"]["batch_size"] = 200
    # Lower than the value in test_fsb to allow some positive domains during splitting
    # As alpha-beta-CROWN and auto_LiRPA improve, this value might have to be tweaked
    # It should be high enough to make several BaB rounds necessary, but low enough
    # to allow some pruning within the iterations.
    arguments.Config["bab"]["decision_thresh"] = np.float64(0.8)
    arguments.Config["solver"]["beta-crown"]["iteration"] = 20
    arguments.Config["bab"]["timeout"] = 60
    arguments.Config["solver"]["alpha-crown"]["lr_alpha"] = 0.1
    arguments.Config["solver"]["beta-crown"]["lr_beta"] = 0.1
    arguments.Config["bab"]["sort_domain_interval"] = 1
    arguments.Config["bab"]["branching"]["method"] = 'fsb'
    arguments.Config["bab"]["branching"]["candidates"] = 2
    arguments.Config["data"]["dataset"] = 'cifar'
    arguments.Config["bab"]["optimized_interm"] = "/input.12"  # name of linear layer before last ReLU
    arguments.Config["bab"]["pruning_in_iteration_ratio"] = 0.00001

    mocked_method = mocker.spy(OptPruner, '_prune_bounds_by_mask')
    bab(model_ori, x, pidx, float('inf'), eps_temp,
        data_max=data_max, data_min=data_min, device=device, dtype=dtype)
    mocked_method.assert_called()


if __name__ == "__main__":
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    test_fsb()
    test_pruning_with_intermediate_layer_optimization()
