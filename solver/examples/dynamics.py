"""A discrete controlled system through the native L1 -> L2 API.

PYTHONPATH=solver:solver/complete_verifier python solver/examples/dynamics.py --device cpu
Add --force-surrogate to give L1 a zero solve budget and exercise L2 on this small example.
"""
import argparse
import json

import torch
import gear as gr


class Original(torch.nn.Module):
    def forward(self, z):
        return torch.cat((-2 * z[:, :1], 2 * z[:, :1] - .8), 1)


class Initial(torch.nn.Module):
    def forward(self, z):
        return z[:, :1] * 0


def simulate(z):
    return torch.stack((z * 0, z, 2 * z), 1)


def outputs(trajectory, z):
    final = trajectory[:, -1]
    return torch.cat((-final, final - .8), 1)


def residual(trajectory, z):
    return trajectory[:, 1:] - trajectory[:, :-1] - z[:, None]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--total-time-limit', type=float, default=10000)
    ap.add_argument('--force-surrogate', action='store_true')
    a = ap.parse_args()
    box = ([0.0], [1.0])
    problem = gr.DynamicsProblem(
        original=lambda options: gr.compile(Original(), box=box, objective=0, le=[1], options=options),
        box=box, simulate=simulate, initial_state=Initial(), outputs=outputs, residual=residual)
    options = gr.DynamicsOptions(
        total_time_limit=a.total_time_limit,
        original=gr.Options(device=a.device, time_limit=0 if a.force_surrogate else 20),
        surrogate=gr.Options(device=a.device, time_limit=20), max_refinements=1,
        training=gr.VAPINNOptions(train_size=64, validation_size=32, test_size=64,
                                 steps=30, refinement_steps=10, width=8, depth=2,
                                 ridge_every=5, evaluate_every=5, batch_size=32))
    result = gr.solve_dynamics(problem, options)
    print('DYNAMICS_RESULT ' + json.dumps(result.to_json(), allow_nan=False))


if __name__ == '__main__':
    main()
