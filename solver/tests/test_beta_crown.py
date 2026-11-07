import sys
import torch.nn as nn
import numpy as np
import torch

sys.path.insert(0, '../complete_verifier')


from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *


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
