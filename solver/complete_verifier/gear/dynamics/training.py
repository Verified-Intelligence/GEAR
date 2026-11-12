"""Portable verification-aware PINN training; no case files or solver-log parsing."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from ..errors import ModelError


class Normalization:
    """Fixed from the initial training set; never refitted on either test set."""
    def __init__(self, data):
        future = data.trajectory[:, 1:].flatten(1)
        self.state_mean = future.mean(0)
        sd = future.std(0)
        self.free = sd > 1e-9 * float(sd.max())
        floor = max(1e-12, float(sd[self.free].median()) * 1e-3) if self.free.any() else 1e-12
        self.state_sigma = sd.clamp_min(floor)
        self.state_scale = torch.where(self.free, self.state_sigma, 0.0)
        self.output_mean = data.values.mean(0)
        sy = data.values.std(0)
        self.output_sigma = sy.clamp_min(max(1e-12, float(sy.median()) * 1e-3))
        self.sigma_f = max(1e-12, float(sy[0]))


def _linear_interval(lo, hi, layer):
    centre = (lo + hi) @ layer.weight.T / 2 + layer.bias
    radius = (hi - lo) @ layer.weight.abs().T / 2
    return centre - radius, centre + radius


class TrajectoryNetwork(nn.Module):
    """Affine + quadratic + tanh features, shared by state and functional heads.

    The optimizer reads only the functional head. The consistency loss ties it
    to F/G of the predicted trajectory; reference validation is still required.
    """
    def __init__(self, problem, data, norm, options):
        super().__init__()
        lo, hi = problem.box
        self.T, self.S = data.trajectory.shape[1] - 1, data.trajectory.shape[2]
        d = lo.numel()
        for key, value in dict(lo=lo, hi=hi, centre=(lo + hi) / 2,
                               radius=torch.where(hi > lo, (hi - lo) / 2, 1.0),
                               inverse_radius=torch.where(hi > lo, (hi - lo) / 2, 1.0).reciprocal(),
                               state_mean=norm.state_mean, state_scale=norm.state_scale,
                               output_mean=norm.output_mean, output_sigma=norm.output_sigma).items():
            self.register_buffer(key, value.clone())
        self.initial_state = copy.deepcopy(problem.initial_state)
        if isinstance(self.initial_state, nn.Module):
            for p in self.initial_state.parameters():
                p.requires_grad_(False)
        self.layers = nn.ModuleList([nn.Linear(d, options.width, dtype=torch.float64)] +
                                    [nn.Linear(options.width, options.width, dtype=torch.float64)
                                     for _ in range(options.depth - 1)])
        if d <= 64:
            rows = [torch.eye(d, dtype=torch.float64)]
            i, j = torch.triu_indices(d, d, 1)
            pair = torch.zeros(len(i), d, dtype=torch.float64)
            pair[torch.arange(len(i)), i] = pair[torch.arange(len(i)), j] = 2 ** -.5
            rows.append(pair)
            self.squares = nn.Linear(d, d + len(i), bias=False, dtype=torch.float64)
            self.squares.weight = nn.Parameter(torch.cat(rows), requires_grad=False)
        else:
            self.squares = nn.Linear(d, min(2 * d, 16), bias=False, dtype=torch.float64)
        width = d + self.squares.out_features + options.width
        self.trajectory_head = nn.Linear(width, self.T * self.S, dtype=torch.float64)
        self.functional_head = nn.Linear(width, data.values.shape[1], dtype=torch.float64)

    def features(self, z):
        x = (z - self.centre) * self.inverse_radius
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return torch.cat((x, self.squares(x) ** 2, h), 1)

    def evaluate(self, z):
        features = self.features(z)
        future = self.state_mean + self.state_scale * self.trajectory_head(features)
        traj = torch.cat((self.initial_state(z)[:, None], future.reshape(-1, self.T, self.S)), 1)
        return traj, self.output_mean + self.output_sigma * self.functional_head(features)

    def trajectory(self, z):
        return self.evaluate(z)[0]

    def forward(self, z):
        return self.output_mean + self.output_sigma * self.functional_head(self.features(z))

    def interval(self, lo, hi):
        lo, hi = (lo - self.centre) / self.radius, (hi - self.centre) / self.radius
        centre = (lo + hi) @ self.squares.weight.T / 2
        radius = (hi - lo) @ self.squares.weight.abs().T / 2
        a, b = centre - radius, centre + radius
        qlo = torch.where(a * b <= 0, 0.0, torch.minimum(a.square(), b.square()))
        qhi = torch.maximum(a.square(), b.square())
        l, u = lo, hi
        for layer in self.layers:
            l, u = _linear_interval(l, u, layer)
            l, u = l.tanh(), u.tanh()
        l, u = _linear_interval(torch.cat((lo, qlo, l), 1), torch.cat((hi, qhi, u), 1), self.functional_head)
        return self.output_mean + self.output_sigma * l, self.output_mean + self.output_sigma * u

    def relaxation_penalty(self):
        lo, hi = (self.lo - self.centre) / self.radius, (self.hi - self.centre) / self.radius
        gaps, slopes = [], []
        for layer in self.layers:
            a, b = _linear_interval(lo, hi, layer)
            width = b - a
            error = (4 / (3 * math.sqrt(3)) / 8) * width.square()
            gaps.append(error / (1 + error))
            lo, hi = a.tanh(), b.tanh()
            slopes.append(torch.where(width > 1e-4, (hi - lo) / width.clamp_min(1e-4),
                                      1 - ((a + b) / 2).tanh().square()))
        c = self.functional_head.weight[:, -self.layers[-1].out_features:]
        total = c.new_zeros(len(c))
        for k in reversed(range(len(self.layers))):
            total = total + c.abs() @ gaps[k]
            if k:
                c = (c * slopes[k]) @ self.layers[k].weight
        return total.mean()


def trajectory_errors(problem, predicted, data, norm):
    t, s = data.trajectory.shape[1] - 1, data.trajectory.shape[2]
    prefix = torch.full((len(data),), t, dtype=torch.long)
    if problem.relevant_nodes is not None:
        prefix = problem.relevant_nodes(data.z, data.trajectory).cpu().long()
        if prefix.shape != (len(data),) or (prefix < 0).any() or (prefix > t).any():
            raise ModelError("relevant_nodes must return a prefix length in [0,T] per sample")
    else:
        prefix = torch.where(data.exists, prefix, 0)
    if not norm.free.any():
        return torch.zeros(len(data), dtype=torch.float64)
    error = ((predicted[:, 1:].flatten(1)[:, norm.free] -
              data.trajectory[:, 1:].flatten(1)[:, norm.free]) / norm.state_sigma[norm.free])
    node = torch.arange(t * s)[norm.free] // s + 1
    mask = node[None] <= prefix[:, None]
    return ((error.square() * mask).sum(1) / mask.sum(1).clamp_min(1)).sqrt()


def _score(model, problem, data, norm, tau):
    with torch.no_grad():
        predicted, values = model.evaluate(data.z)
        error = (values[:, 0] - data.values[:, 0]).abs()
        state_score = float(torch.quantile(trajectory_errors(problem, predicted, data, norm), .5)) / .01
        return max(float(torch.quantile(error, .5)) / tau, float(torch.quantile(error, .95)) / (5 * tau), state_score)


class VAPINNTrainer:
    """A fit receives only training/calibration data, never the independent tests.

    An alternative trainer may implement the same fit(...) method. Models must
    provide forward(z)->[Fhat,Ghat] and trajectory(z)->[B,T+1,S].
    """
    def __init__(self, options):
        self.options = options

    def fit(self, problem, data, validation, norm, initial, round_index, tau, device):
        o = self.options
        device = o.device or device
        cuda = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
        with torch.random.fork_rng(devices=cuda):
            torch.manual_seed(o.seed + 1000 * round_index)
            return self._fit(problem, data, validation, norm, initial, round_index, tau, device)

    def _fit(self, problem, data, validation, norm, initial, round_index, tau, device):
        o = self.options
        model = copy.deepcopy(initial) if initial is not None else TrajectoryNetwork(problem, data, norm, o)
        model = model.to(device=device, dtype=torch.float64)
        # Reference callbacks with buffers are copied; caller-owned modules stay on their device.
        outputs, residual = (copy.deepcopy(fn) for fn in (problem.outputs, problem.residual))
        if isinstance(outputs, nn.Module):
            outputs.to(device=device, dtype=torch.float64)
            outputs.requires_grad_(False)
        if isinstance(residual, nn.Module):
            residual.to(device=device, dtype=torch.float64)
            residual.requires_grad_(False)
        z, true_traj, true_y = (x.to(device) for x in (data.z, data.trajectory, data.values))
        exists = data.exists.to(device)
        ss, sy = norm.state_sigma.to(device), norm.output_sigma.to(device)
        free = norm.free.to(device)
        steps = o.steps if initial is None else o.refinement_steps
        lr = o.learning_rate if initial is None else o.refinement_rate
        model.trajectory_head.requires_grad_(False)
        model.functional_head.requires_grad_(False)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        best_score, best = math.inf, None

        def ridge_fit():
            with torch.no_grad():
                phi = model.features(z)
                phi = torch.cat((phi, phi.new_ones(len(phi), 1)), 1)
                targets = torch.cat(((true_traj[:, 1:].flatten(1) - model.state_mean) / ss,
                                     (true_y - model.output_mean) / sy), 1)
                gram = phi.T @ phi
                reg = o.ridge * len(phi) * torch.eye(phi.shape[1], device=device, dtype=phi.dtype)
                reg[-1, -1] = 0
                weights = torch.linalg.solve(gram + reg, phi.T @ targets).T
                for head, w in zip((model.trajectory_head, model.functional_head),
                                   weights.split((model.T * model.S, len(sy)))):
                    head.weight.copy_(w[:, :-1]); head.bias.copy_(w[:, -1])

        ridge_fit()
        for step in range(steps + 1):
            if step and step % o.ridge_every == 0:
                ridge_fit()
            if step % o.evaluate_every == 0 or step == steps:
                candidate = copy.deepcopy(model).cpu().eval()
                score = _score(candidate, problem, validation, norm, tau)
                if score < best_score:
                    best_score, best = score, candidate
            if step == steps:
                break
            indices = torch.randint(len(z), (min(o.batch_size, len(z)),), device=device)
            zb, tb, yb, ex = z[indices], true_traj[indices], true_y[indices], exists[indices]
            predicted, yh = model.evaluate(zb)
            state = ((predicted[:, 1:].flatten(1)[:, free] - tb[:, 1:].flatten(1)[:, free]) / ss[free]).square().mean() if free.any() else yh.new_zeros(())
            r = residual(predicted, zb)
            if r.shape != predicted[:, 1:].shape:
                raise ModelError("residual must return [batch,T,S] in state units")
            r2 = (r.flatten(1) / ss).square().mean(1)
            physics = (r2 * ex).mean()
            error = (yh - yb) / sy
            fidelity = error.square().mean() + torch.relu(-error[:, 1:]).square().mean() if yh.shape[1] > 1 else error.square().mean()
            consistent = ((yh - outputs(predicted, zb)) / sy).square().mean()
            loss = state + physics + fidelity + consistent
            if o.relaxation_weight:
                loss = loss + o.relaxation_weight * model.relaxation_penalty()
            if o.bound_weight:
                centres = model.lo + (model.hi - model.lo) * torch.rand(o.bound_boxes, len(model.lo), device=device, dtype=zb.dtype)
                radius = (model.hi - model.lo) * torch.rand(o.bound_boxes, 1, device=device, dtype=zb.dtype) / 2
                lo, hi = torch.maximum(centres - radius, model.lo), torch.minimum(centres + radius, model.hi)
                lb, ub = model.interval(lo, hi)
                points = lo[:, None] + (hi - lo)[:, None] * torch.rand(o.bound_boxes, o.bound_points, len(model.lo), device=device, dtype=zb.dtype)
                yp = model(points.reshape(-1, len(model.lo))).reshape(o.bound_boxes, o.bound_points, -1)
                slack = ((yp[:, :, 0].amin(1) - lb[:, 0]).clamp_min(0) / sy[0]).mean() if int(problem.sense) == 1 else ((ub[:, 0] - yp[:, :, 0].amax(1)).clamp_min(0) / sy[0]).mean()
                if len(sy) > 1:
                    slack = slack + ((ub[:, 1:] - yp[:, :, 1:].amax(1)).clamp_min(0) / sy[1:]).mean()
                loss = loss + o.bound_weight * slack
            if not torch.isfinite(loss):
                raise ModelError("nonfinite VA-PINN training loss")
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10)
            optimizer.step()
        if best is None:
            raise ModelError("no finite validation checkpoint from VA-PINN training")
        return best
