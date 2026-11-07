"""NNProblemSingleDomain must wrap models that cannot take a batch of 2."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nlp_solver.problem import NNProblemSingleDomain  # noqa: E402


class BatchOneOnly(torch.nn.Module):
    """Concatenates a batch-1 constant, like the generated st_e06 model."""

    def forward(self, x):
        const = torch.zeros(1, 1, dtype=x.dtype)
        return torch.cat([x.sum(-1, keepdim=True), const, x[:, :1] ** 2], dim=1)


def test_model_that_raises_on_batch_two_is_wrapped():
    dm_l = -torch.ones(1, 2, dtype=torch.float64)
    dm_u = torch.ones(1, 2, dtype=torch.float64)
    C = torch.eye(3, dtype=torch.float64)
    prob = NNProblemSingleDomain(BatchOneOnly(), dm_l, dm_u, C[:1], torch.zeros(1),
                                 C[1:], torch.zeros(2), device='cpu', dtype=torch.float64)
    out = prob.model(torch.zeros(4, 2, dtype=torch.float64))
    assert out.shape == (4, 3)
