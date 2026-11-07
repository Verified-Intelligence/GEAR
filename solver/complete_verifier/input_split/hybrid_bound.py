"""Unconstrained per-term IBP/CROWN objective bounds on the root box."""

import copy
import math
import types

import torch

from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


_AFFINE_OPS = {
    'BoundAdd', 'BoundSub', 'BoundNeg', 'BoundReduceSum', 'BoundATenSum',
    'BoundReduceMean', 'BoundGather', 'BoundGatherElements', 'BoundSlice',
    'BoundConcat', 'BoundReshape', 'BoundUnsqueeze', 'BoundSqueeze',
    'BoundTranspose', 'BoundIdentity', 'BoundFlatten', 'BoundExpand',
}
_FIXED_STRUCTURE_OPS = {
    'BoundGather', 'BoundGatherElements', 'BoundSlice', 'BoundReshape',
    'BoundExpand', 'BoundUnsqueeze', 'BoundSqueeze', 'BoundTranspose',
    'BoundReduceSum', 'BoundATenSum', 'BoundReduceMean',
}


def _copy_bound_graph(net):
    # Cached tensors can be non-leaves or BoundedTensor subclasses, for which
    # PyTorch's default deepcopy fails. Copy them explicitly, preserving aliases.
    memo, seen, pending = {}, set(), [net]
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            tensor = value.as_subclass(torch.Tensor).detach().clone()
            if isinstance(value, torch.nn.Parameter):
                tensor = torch.nn.Parameter(tensor, requires_grad=value.requires_grad)
            memo[id(value)] = tensor
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple, set)):
            pending.extend(value)
        elif hasattr(value, '__dict__') and not isinstance(value, (type, types.ModuleType)):
            pending.append(vars(value))
    return copy.deepcopy(net, memo)


def _is_affine(node):
    kind = type(node).__name__
    if kind in _AFFINE_OPS:
        return kind not in _FIXED_STRUCTURE_OPS or not any(
            inp.perturbed for inp in node.inputs[1:])
    if kind in ('BoundMul', 'BoundMatMul', 'BoundLinear'):
        return sum(inp.perturbed for inp in node.inputs[:2]) <= 1
    if kind == 'BoundCast':
        return (node.inputs[0].forward_value.dtype.is_floating_point
                and node.forward_value.dtype.is_floating_point)
    return False


def _decompose(net, max_terms):
    final = net.final_node()
    weights = torch.zeros_like(net.get_forward_value(final))
    weights.flatten()[0] = 1
    ordered, seen = [], set()

    def visit(node):
        if node.name in seen:
            return
        seen.add(node.name)
        for inp in node.inputs:
            visit(inp)
        ordered.append(node)

    visit(final)
    pending, leaves, constants = {final.name: weights}, {}, []
    num_terms = 0
    for node in reversed(ordered):
        coeff = pending.pop(node.name, None)
        if coeff is None or not (coeff != 0).any():
            continue
        if not torch.isfinite(coeff).all():
            raise ValueError('Non-finite objective coefficients')
        if not node.perturbed:
            constants.append(float((net.get_forward_value(node) * coeff).sum()))
        elif _is_affine(node):
            args, active = [], []
            for inp in node.inputs:
                value = net.get_forward_value(inp)
                if inp.perturbed:
                    value = torch.zeros_like(value).requires_grad_(True)
                    active.append((inp, value))
                elif torch.is_tensor(value):
                    value = value.detach()
                args.append(value)
            with torch.enable_grad():
                value = node.forward(*args)
                constants.append(float((value.detach() * coeff).sum()))
                gradients = torch.autograd.grad(
                    value, [arg for _, arg in active], grad_outputs=coeff,
                    allow_unused=True)
            for (inp, _), grad in zip(active, gradients):
                if grad is not None:
                    pending[inp.name] = pending.get(inp.name, 0) + grad.detach()
        else:
            leaves[node.name] = coeff.flatten()
            num_terms += int(torch.count_nonzero(coeff))
            if num_terms > max_terms:
                raise ValueError('Root hybrid term limit exceeded')
    return math.fsum(constants), leaves, num_terms


@torch.no_grad()
def root_hybrid_objective_lb(net, x_L, x_U, max_terms=64, *, diagnostics=None):
    """Return a certified lower bound on output 0, or None on failure.

    ``net`` is the already-converted BoundedModule. Work on a copy so root
    planes, intermediate bounds and optimization state remain intact. Queries
    use only the box, with no input/output constraints, betas or cuts.

    For f = const + sum(c_j * t_j), choose max(IBP_lb, CROWN_lb) when
    c_j >= 0 and min(IBP_ub, CROWN_ub) when c_j < 0. Each contribution
    lower-bounds c_j * t_j throughout the root box, so their sum bounds f
    there and on every descendant box, regardless of dependencies among terms.
    ``diagnostics`` optionally receives the successful decomposition's term count.
    """
    try:
        if (max_terms < 0 or x_L.shape != x_U.shape or x_L.shape[0] != 1
                or not torch.isfinite(x_L).all() or not torch.isfinite(x_U).all()
                or (x_L > x_U).any()):
            return None
        # Count the terms on the pipeline's graph first (read-only: forward values and
        # node.forward only), so large models never pay for the graph copy.
        _decompose(net, max_terms)
        net = _copy_bound_graph(net)
        opts = net.bound_opts['optimize_bound_args']
        opts.update(enable_beta_crown=False, apply_output_constraints_to=[],
                    tighten_input_bounds=False)
        net.cut_used = False
        for node in net.nodes():
            if hasattr(node, 'opt_stage'):
                node.opt_stage = None
            if hasattr(node, 'cut_used'):
                node.cut_used = False
        x = BoundedTensor((x_L + x_U) / 2,
                          PerturbationLpNorm(x_L=x_L, x_U=x_U))
        net(x)
        constant, leaves, num_terms = _decompose(net, max_terms)
        # Finish pure IBP queries before CROWN changes any intermediate caches.
        ibp = {}
        for name in leaves:
            lower, upper = net.compute_bounds(
                x=(x,), method='IBP', final_node_name=name)
            ibp[name] = (lower.flatten().clone(), upper.flatten().clone())
        contributions = [constant]
        for name, coeff in leaves.items():
            indices = (coeff != 0).nonzero().flatten()
            value = net.get_forward_value(net[name])
            spec = value.new_zeros(1, len(indices), value[0].numel())
            spec[0, torch.arange(len(indices), device=value.device), indices] = 1
            lower, upper = net.compute_bounds(
                x=(x,), C=spec, method='CROWN', final_node_name=name)
            ibp_lower, ibp_upper = ibp[name]
            best_lower = torch.maximum(ibp_lower[indices], lower.flatten())
            best_upper = torch.minimum(ibp_upper[indices], upper.flatten())
            chosen = torch.where(coeff[indices] >= 0, best_lower, best_upper)
            contributions.extend((coeff[indices] * chosen).tolist())
        if not all(math.isfinite(value) for value in contributions):
            return None
        lower_bound = math.fsum(contributions)
        if not math.isfinite(lower_bound):
            return None
        if diagnostics is not None:
            diagnostics['terms'] = num_terms
        return lower_bound
    except Exception:  # This optional tightening must never prevent normal BaB.
        return None
