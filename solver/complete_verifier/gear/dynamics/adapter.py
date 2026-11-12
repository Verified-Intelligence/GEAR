"""Adapter for the paper's trajectory Case protocol; no experiment imports."""
import torch
from torch import nn

from ..errors import ModelError
from .problem import DynamicsProblem


class _Initial(nn.Module):
    def __init__(self, case):
        super().__init__()
        self.register_buffer('matrix', case.init_M.detach().cpu().double().clone())
        self.register_buffer('constant', case.init_c.detach().cpu().double().clone())

    def forward(self, z):
        return z @ self.matrix.T + self.constant


def from_case(case, *, original, **kwargs):
    required = ('lower', 'upper', 'simulate', 'init_M', 'init_c', 'residual', 'obj', 'groups')
    missing = [name for name in required if not hasattr(case, name)]
    if missing:
        raise ModelError('Case adapter is missing: ' + ', '.join(missing))

    def outputs(trajectory, z):
        body = case.obj
        f = body.fn(trajectory) if hasattr(body, 'fn') else trajectory.flatten(1) @ body.A.T.to(trajectory) + body.b.to(trajectory)
        if getattr(case, 'obj_z', None) is not None:
            f = f + case.obj_z(z)
        return torch.cat((f, *(group.values(z, trajectory) for group in case.groups)), 1)

    return DynamicsProblem(original=original, box=(case.lower, case.upper), simulate=case.simulate,
        initial_state=_Initial(case), outputs=outputs, residual=case.residual,
        exists=getattr(case, 'traj_exists', None), relevant_nodes=getattr(case, 'relevant_nodes', None), **kwargs)
