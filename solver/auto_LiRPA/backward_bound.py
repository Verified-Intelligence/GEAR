#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import math
import os
import torch
from torch import Tensor
from collections import deque
from tqdm import tqdm
from .patches import Patches
from .utils import *
from .bound_ops import *
from .interval_bound import _unbounded_interval, _interval_propagate, _affine_output_error
from .fp_enclosure import (add_bound as add_enclosed_bound, finite_abs,
                           rounding_error, widen_bound, coefficient_terms)
import warnings

from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from .bound_general import BoundedModule



# A67's affine-backward rounding enclosure when bound_opts do not set it (off by default in the shared auto_LiRPA;
# the GEAR configs turn it on: bound_opts {'fp_enclosure': {'affine_backward': True}})
FP_AFFINE_DEFAULT = False

def batched_backward(self: 'BoundedModule', node, C, unstable_idx, batch_size,
                     bound_lower=True, bound_upper=True, return_A=None):
    if return_A is None: return_A = self.return_A
    output_shape = node.output_shape[1:]
    dim = int(prod(output_shape))
    # Batched dense specifications must follow the bounded node's dtype. The
    # cached dtype avoids rebuilding concrete values during bound iterations.
    specification_dtype = self.get_output_dtype(node)
    if unstable_idx is None:
        unstable_idx = torch.arange(dim, device=self.device)
        dense = True
    else:
        dense = False
    unstable_size = get_unstable_size(unstable_idx)
    print(f'Batched CROWN: node {node}, unstable size {unstable_size}')
    crown_batch_size = self.bound_opts['crown_batch_size']
    auto_batch_size = AutoBatchSize(self.bound_opts['crown_batch_size'], self.device, vram_ratio=self.bound_opts['batched_crown_max_vram_ratio'])

    ret = []
    ret_A = {} # if return_A, we will store A here
    i = 0
    torch.cuda.empty_cache()
    with tqdm(total=unstable_size) as pbar:
        while i < unstable_size:
            crown_batch_size = auto_batch_size.batch_size
            if isinstance(unstable_idx, tuple):
                unstable_idx_batch = tuple(
                    u[i : i + crown_batch_size]
                    for u in unstable_idx
                )
                unstable_size_batch = len(unstable_idx_batch[0])
            else:
                unstable_idx_batch = unstable_idx[i : i + crown_batch_size]
                unstable_size_batch = len(unstable_idx_batch)
            auto_batch_size.record_actual_batch_size(unstable_size_batch)

            if node.patches_start and node.mode == "patches":
                assert C is None or C.type == 'Patches'
                C_batch = Patches(shape=[
                    unstable_size_batch, batch_size, *node.output_shape[1:-2], 1, 1],
                    identity=1, unstable_idx=unstable_idx_batch,
                    output_shape=[batch_size, *node.output_shape[1:]])
            elif C.type == 'OneHot':
                assert isinstance(node, (BoundLinear, BoundMatMul))
                C_batch = OneHotC(
                    [batch_size, unstable_size_batch, *node.output_shape[1:]],
                    self.device, unstable_idx_batch, None)
            else:
                assert C is None or C.type == 'eye'
                C_batch = torch.zeros(
                    [1, unstable_size_batch, dim],
                    dtype=specification_dtype,
                    device=self.device,
                )
                C_batch[0, torch.arange(unstable_size_batch), unstable_idx_batch] = 1.0
                C_batch = C_batch.expand(batch_size, -1, -1).view(
                    batch_size, unstable_size_batch, *output_shape)
            # overwrite return_A options to run backward general
            ori_return_A_option = self.return_A
            self.return_A = return_A

            batch_ret = self.backward_general(
                node, C_batch,
                bound_lower=bound_lower, bound_upper=bound_upper,
                average_A=False, need_A_only=False, unstable_idx=unstable_idx_batch)
            ret.append(batch_ret[:2])

            if len(batch_ret) > 2:
                # A found, we merge A
                batch_A = batch_ret[2]
                ret_A = merge_A(node, batch_A, ret_A)

            # restore return_A options
            self.return_A = ori_return_A_option

            pbar.update(unstable_size_batch)
            i += unstable_size_batch
            auto_batch_size.update()

    if bound_lower:
        lb = torch.cat([item[0].view(batch_size, -1) for item in ret], dim=1)
        if dense:
            # In this case, restore_sparse_bounds will not be called.
            # And thus we restore the shape here.
            lb = lb.reshape(batch_size, *output_shape)
    else:
        lb = None
    if bound_upper:
        ub = torch.cat([item[1].view(batch_size, -1) for item in ret], dim=1)
        if dense:
            # In this case, restore_sparse_bounds will not be called.
            # And thus we restore the shape here.
            ub = ub.reshape(batch_size, *output_shape)
    else:
        ub = None

    if return_A:
        return lb, ub, ret_A
    else:
        return lb, ub


def _undefined_rows(A, no_pos, no_neg):
    """(spec, batch) rows of one bound pass that use an undefined relaxation side.

    A backward pass takes each element's relaxation side from the sign of its
    coefficient: `no_pos` masks the elements whose side for a positive
    coefficient is undefined, `no_neg` those for a negative one (input shape
    (batch, *elem)). Rows are exact for a dense A and an identity C; otherwise a
    batch element with any undefined element loses every row (shape (1, batch)).
    """
    if A is None:
        return None
    batch = no_pos.shape[0]
    if isinstance(A, eyeC) and A.shape[1] == batch:
        flat = no_pos.reshape(batch, -1)
        if flat.shape[1] == A.shape[0]:
            # Row s has coefficient +1 on element s and 0 elsewhere.
            return flat.transpose(0, 1)
    elif (isinstance(A, Tensor) and A.ndim >= 2 and A.shape[1] == batch
            and no_pos.ndim == A.ndim - 1):
        elem = tuple(A.shape[2:])
        try:
            pos = torch.broadcast_to(no_pos, (batch, *elem)).unsqueeze(0)
            neg = torch.broadcast_to(no_neg, (batch, *elem)).unsqueeze(0)
        except RuntimeError:
            pos = neg = None
        if pos is not None:
            hit = ((A > 0) & pos) | ((A < 0) & neg)
            return hit.reshape(A.shape[0], batch, -1).any(-1)
    return (no_pos | no_neg).reshape(batch, -1).any(-1).unsqueeze(0)


def _all_finite(A, lower_b, upper_b):
    """One host synchronization for all propagated coefficients and biases."""
    tensors = [t for pair in A for t in pair if isinstance(t, Tensor)]
    tensors += [t for t in (lower_b, upper_b) if isinstance(t, Tensor)]
    if not tensors:
        return True
    return bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all())


def _drop_nonfinite_rows(A, side, bias, last_A):
    """Rows of one side whose propagated coefficients or bias are not finite.

    Returns (A, bias, rows) with those rows zeroed (they are discarded at the end
    of the pass, so their content is irrelevant); rows is None when all are finite.
    """
    if last_A is None or isinstance(last_A, Patches):
        return A, bias, None
    spec, batch = last_A.shape[:2]
    tensors = [a[side] for a in A if isinstance(a[side], Tensor)]
    if isinstance(bias, Tensor):
        tensors.append(bias)
    if not tensors or bool(torch.stack(
            [torch.isfinite(t).all() for t in tensors]).all()):
        return A, bias, None
    rows = torch.zeros(spec, batch, dtype=torch.bool, device=tensors[0].device)
    for t in tensors:
        if t.ndim >= 2 and tuple(t.shape[:2]) == (spec, batch):
            rows |= ~torch.isfinite(t).reshape(spec, batch, -1).all(-1)
        elif not bool(torch.isfinite(t).all()):
            rows[:] = True
    A = [tuple(_zero_rows(a[k], rows, spec, batch) if k == side else a[k]
               for k in range(2)) for a in A]
    bias = _zero_rows(bias, rows, spec, batch)
    return A, bias, rows


def _zero_rows(t, rows, spec, batch):
    if not isinstance(t, Tensor):
        return t
    if t.ndim >= 2 and tuple(t.shape[:2]) == (spec, batch):
        mask = rows.reshape(spec, batch, *([1] * (t.ndim - 2)))
        return torch.where(mask, torch.zeros_like(t), t)
    # Not laid out by rows: only a tensor of an all-discarded pass may be zeroed
    # (a non-finite one makes every row discarded above).
    return torch.zeros_like(t) if bool(rows.all()) else t


def _node_interval(self, l, mask):
    """l's own interval: its stored bounds when current, else interval arithmetic from
    its inputs' bounds (a node feeding only linear consumers never gets stored bounds;
    a square of an unbounded residual is still [0, inf))."""
    if (l.is_lower_bound_current() and l.is_upper_bound_current()
            and isinstance(l.lower, Tensor) and isinstance(l.upper, Tensor)
            and l.lower.shape == mask.shape):
        box = Interval(l.lower, l.upper)
    else:
        box = None
    try:
        if box is None:
            box = _interval_from_inputs(self, l, mask.shape[0])
        if isinstance(l, BoundSqr):
            # Intermediate/reference dictionaries store only hulls. Reconstruct
            # the gap chain even if a cached residual hull is (-inf, inf).
            gap_box = _interval_from_inputs(self, l, mask.shape[0], recompute_gap=True)
            if gap_box is not None:
                box = (gap_box if box is None else Interval(
                    torch.maximum(box.lower, gap_box.lower),
                    torch.minimum(box.upper, gap_box.upper)))
    except Exception:
        return None
    if box is None:
        return None
    return box.lower, box.upper


def _stop_undefined_interval(l, lA, uA, no_lower, no_upper, interval=None):
    """Replace coefficients selecting the undefined side by interval contributions.

    `interval` is l's own interval (_node_interval: stored bounds, else interval
    arithmetic from its inputs). Only elementwise activations with an identity
    input/output element map support this stop. Fixed
    parameters (such as a unary power's exponent) do not participate in the map.
    Matrix products and pooling can mix elements even when their shapes match.
    Unsupported coefficient layouts keep the undefined-row fallback.
    """
    mask = no_lower | no_upper
    if (not isinstance(l, BoundActivation)
            or isinstance(l, (BoundLinear, BoundMaxPool))
            or len(l.requires_input_bounds) not in (1, 2)
            or tuple(l.output_shape[1:]) != tuple(mask.shape[1:])
            or any(tuple(inp.output_shape[1:]) != tuple(l.output_shape[1:])
                   or not isinstance(inp.lower, Tensor) or inp.lower.shape != mask.shape
                   for inp in (l.inputs[i] for i in l.requires_input_bounds))
            or interval is None
            or any(A is not None and (not isinstance(A, Tensor)
                   or tuple(A.shape[1:]) != tuple(mask.shape))
                   for A in (lA, uA))):
        return None
    lower, upper = interval
    if (not isinstance(lower, Tensor) or not isinstance(upper, Tensor)
            or lower.shape != mask.shape or upper.shape != mask.shape
            or bool((torch.isnan(lower) | torch.isnan(upper) | (lower > upper)).any())):
        return None

    def stop(A, no_pos, no_neg, first, second):
        if A is None:
            return None, 0
        hit = ((A > 0) & no_pos.unsqueeze(0)) | ((A < 0) & no_neg.unsqueeze(0))
        selected = torch.where(hit, A, torch.zeros_like(A))
        # Select endpoints before multiplication: a zero coefficient contributes
        # zero even at an infinite endpoint, including during differentiation.
        endpoints = torch.where(selected > 0, first.unsqueeze(0),
                                torch.where(selected < 0, second.unsqueeze(0), 0))
        bias = (selected * endpoints).reshape(A.shape[0], A.shape[1], -1).sum(-1)
        return torch.where(hit, torch.zeros_like(A), A), bias

    lA, lower_b = stop(lA, no_lower, no_upper, lower, upper)
    uA, upper_b = stop(uA, no_upper, no_lower, upper, lower)
    return lA, uA, lower_b, upper_b


def _bound_backward_tolerant(self: 'BoundedModule', l, lA, uA, **kwargs):
    """l.bound_backward(), tolerating relaxations undefined on some elements.

    With the bound option `undefined_relaxation_rows`, an operator error that
    carries `.undefined` (an UndefinedRelaxation: a pole inside an input
    interval, an infinite endpoint) does not abort the pass. The rows whose
    coefficients select an undefined side are recorded -- backward_general sets
    their bounds to -inf / +inf, which is sound whatever else the row holds --
    and the operator runs again with those elements' input interval replaced;
    the replacement keeps every side that is still used valid. Input bounds are
    restored afterwards. Returns (A, lbias, ubias, rows_l, rows_u).

    `undefined_relaxation_interval` requires `undefined_relaxation_rows`. When
    enabled, an identity element map and an output interval allow
    stopping only coefficients selecting the undefined side: they are zeroed
    and their sign-selected interval contributions are added to the bias. Only
    non-finite contributions discard rows; the remaining coefficients retain their
    valid relaxation. For every x in the domain, z_e(x) lies in [z_l(e), z_u(e)],
    so lA[r,e] * z_e(x) is at least its constant contribution (and uA[r,e] *
    z_e(x) is at most its upper contribution). Adding the valid relaxations of
    the other coefficients therefore gives a sound bound. The operator then runs
    on the substituted input interval, whose defined side remains valid on the
    original domain.
    """
    if not self.bound_opts.get('undefined_relaxation_rows', False):
        A, lower_b, upper_b = l.bound_backward(lA, uA, *l.inputs, **kwargs)
        return A, lower_b, upper_b, None, None
    rows_l = rows_u = None
    stop_lower = stop_upper = None
    original_interval = None
    interval_checked = False
    saved = []
    try:
        for _ in range(2 * len(l.inputs) + 2):
            try:
                A, lower_b, upper_b = l.bound_backward(
                    lA, uA, *l.inputs, **kwargs)
                if stop_lower is not None:
                    lower_b, upper_b = lower_b + stop_lower, upper_b + stop_upper
                r_l = r_u = None
                if l.requires_input_bounds and not _all_finite(
                        A, lower_b, upper_b):
                    # A relaxation that overflowed (an inf slope, 0 * inf) leaves
                    # non-finite coefficients: those rows are unknown as well, and
                    # are zeroed so that no NaN reaches the concretization. Only
                    # relaxing nodes (they read input bounds) can create them.
                    A, lower_b, r_l = _drop_nonfinite_rows(A, 0, lower_b, lA)
                    A, upper_b, r_u = _drop_nonfinite_rows(A, 1, upper_b, uA)
                if r_l is not None:
                    rows_l = r_l if rows_l is None else rows_l | r_l
                if r_u is not None:
                    rows_u = r_u if rows_u is None else rows_u | r_u
                return A, lower_b, upper_b, rows_l, rows_u
            except (OperatorDomainError, OperatorCapabilityError) as error:
                info = getattr(error, 'undefined', None)
                if info is None:
                    # A capability/domain error without a per-element descriptor
                    # has no usable affine relaxation. Only rows using this
                    # output lose their bound; unrelated rows keep propagating.
                    mask = torch.ones_like(
                        _unbounded_interval(self, l, self.batch_size).lower,
                        dtype=torch.bool)
                    r_l = _undefined_rows(lA, mask, mask)
                    r_u = _undefined_rows(uA, mask, mask)
                    if r_l is not None:
                        rows_l = r_l if rows_l is None else rows_l | r_l
                    if r_u is not None:
                        rows_u = r_u if rows_u is None else rows_u | r_u
                    return ([(None, None) for _ in l.inputs],
                            0 if stop_lower is None else stop_lower,
                            0 if stop_upper is None else stop_upper, rows_l, rows_u)
                inp = l.inputs[info.input_index]
                if (not isinstance(inp.lower, Tensor)
                        or info.mask.shape != inp.lower.shape
                        or not bool(info.mask.any())):
                    raise
                # A lower bound uses the lower relaxation where the coefficient
                # is positive and the upper one where it is negative; an upper
                # bound the other way round.
                if (self.bound_opts.get('undefined_relaxation_interval', False)
                        and not interval_checked):
                    # A binary operator can fail once per input. Compute its
                    # interval before the FIRST substitution, and reuse it:
                    # later retries must not derive an enclosure from dummy inputs.
                    original_interval = _node_interval(self, l, info.mask)
                    interval_checked = True
                if isinstance(l, BoundSqr) and original_interval is not None:
                    # A square on an unbounded hull still has a defined tangent
                    # (usually zero). A range gap can give a stronger constant
                    # lower plane. Stop those elements at the square as well,
                    # before a zero tangent loses the gap at the reciprocal.
                    hull_lower = torch.where(inp.upper < 0, inp.upper ** 2,
                                             torch.where(inp.lower > 0, inp.lower ** 2, 0))
                    info.no_lower = info.no_lower | (
                        info.mask & (original_interval[0] > hull_lower))
                stopped = (_stop_undefined_interval(
                               l, lA, uA, info.no_lower, info.no_upper, original_interval)
                           if self.bound_opts.get('undefined_relaxation_interval', False)
                           else None)
                if stopped is not None:
                    lA, uA, lower_b, upper_b = stopped
                    if stop_lower is None:
                        stop_lower, stop_upper = lower_b, upper_b
                    else:
                        stop_lower, stop_upper = stop_lower + lower_b, stop_upper + upper_b
                    r_l = r_u = None
                else:
                    r_l = _undefined_rows(lA, info.no_lower, info.no_upper)
                    r_u = _undefined_rows(uA, info.no_upper, info.no_lower)
                if r_l is not None:
                    rows_l = r_l if rows_l is None else rows_l | r_l
                if r_u is not None:
                    rows_u = r_u if rows_u is None else rows_u | r_u
                saved.append((inp, inp._lower, inp._upper,
                              inp._is_lower_bound_current,
                              inp._is_upper_bound_current))
                inp.lower, inp.upper = info.substitute(inp.lower, inp.upper)
        raise RuntimeError(
            f'{l}: relaxation still undefined after replacing input intervals')
    finally:
        for inp, lower, upper, lower_current, upper_current in reversed(saved):
            inp._lower, inp._upper = lower, upper
            inp._is_lower_bound_current = lower_current
            inp._is_upper_bound_current = upper_current


def _rows_like(rows, bound):
    """A (spec, batch) or (1, batch) row mask shaped like a (batch, spec...) bound.

    A mask whose row count does not match falls back to whole batch elements, so a
    discarded row is never kept.
    """
    flat = bound.reshape(bound.shape[0], -1)
    mask = rows.transpose(0, 1)
    if mask.shape[1] not in (1, flat.shape[1]):
        mask = mask.any(1, keepdim=True)
    return mask.expand_as(flat).reshape(bound.shape)


def _discard_undefined_rows(lb, ub, undefined_l, undefined_u):
    """Bounds of rows with an undefined relaxation are unknown: -inf / +inf."""
    if undefined_l is not None and isinstance(lb, Tensor) and lb.ndim >= 1:
        lb = torch.where(_rows_like(undefined_l, lb),
                         torch.full_like(lb, -torch.inf), lb)
    if undefined_u is not None and isinstance(ub, Tensor) and ub.ndim >= 1:
        ub = torch.where(_rows_like(undefined_u, ub),
                         torch.full_like(ub, torch.inf), ub)
    return lb, ub


def _zero_undefined_root_A(A_record, roots, undefined_l, undefined_u):
    """The saved plane of a discarded row is vacuous: A = 0 with bias -inf / +inf."""
    for root in roots:
        entry = A_record.get(root.name)
        if entry is None:
            continue
        entry['lbias'], entry['ubias'] = _discard_undefined_rows(
            entry.get('lbias'), entry.get('ubias'), undefined_l, undefined_u)
        for key, rows in (('lA', undefined_l), ('uA', undefined_u)):
            A = entry.get(key)
            if rows is None or not isinstance(A, Tensor) or A.ndim < 2:
                continue
            mask = rows.transpose(0, 1)
            if mask.shape[1] not in (1, A.shape[1]):
                mask = mask.any(1, keepdim=True)
            mask = mask.expand(A.shape[0], A.shape[1])
            entry[key] = torch.where(
                mask.reshape(*mask.shape, *([1] * (A.ndim - 2))),
                torch.zeros_like(A), A)


def _interval_from_inputs(self: 'BoundedModule', node, batch, cache=None,
                          recompute_gap=False):
    """Interval of `node` by interval arithmetic from its inputs, without storing.

    Current bounds of the right batch size are used where a node has them (they
    are sound); other perturbed ancestors are evaluated here, in topological
    order, and only kept locally. None if an input has no usable interval.
    """
    def stored(n):
        if not n.perturbed:
            value = self.get_forward_value(n)
            return Interval(value, value)
        if recompute_gap and isinstance(n, (
                BoundSqr, BoundReciprocal, BoundDiv, BoundMul, BoundAdd, BoundSub, BoundNeg)):
            return None
        if (n is not node and n.is_lower_bound_current()
                and n.is_upper_bound_current()
                and isinstance(n.lower, Tensor) and isinstance(n.upper, Tensor)
                and n.lower.ndim > 0 and n.lower.shape[0] == batch):
            lower, upper = n.lower, n.upper
            inverted = lower > upper
            if inverted.any():
                # An inverted interval means "infeasible" only if a sound tightening
                # made it; interval arithmetic on it is otherwise meaningless (hs025:
                # squares of [+inf, -c] gave c^2 > 0). Use the whole real line there.
                lower = torch.where(inverted, -torch.inf, lower)
                upper = torch.where(inverted, torch.inf, upper)
            return Interval(lower, upper, getattr(n, '_range_gap', None))
        return None

    known = {} if cache is None else cache
    order, stack, seen = [], [(node, False)], set()
    while stack:
        n, expanded = stack.pop()
        if expanded:
            order.append(n)
            continue
        if n.name in seen:
            continue
        seen.add(n.name)
        if n.name in known:
            continue
        box = stored(n)
        if box is not None:
            known[n.name] = box
            continue
        if n.name in self.root_names:
            return None
        stack.append((n, True))
        for inp in n.inputs:
            if inp.name not in seen:
                stack.append((inp, False))
    for n in order:
        try:
            box = _interval_propagate(n, [known[inp.name] for inp in n.inputs], [])
        except (OperatorDomainError, OperatorCapabilityError):
            box = _unbounded_interval(self, n, batch)
        lower = torch.where(torch.isnan(box.lower), -torch.inf, box.lower)
        upper = torch.where(torch.isnan(box.upper), torch.inf, box.upper)
        known[n.name] = Interval(lower, upper, box.range_gap)
    return known.get(node.name)


def _abs_dot(A, magnitude, unbounded=None):
    """<|A|, magnitude> per (spec, batch) row, one pass over |A|; magnitude (finite) and `unbounded` (an indicator,
    > 0 where a row meets an unbounded value) broadcast to A's per-row shape; a row with |A| > 0 on an indicated
    entry gets +inf."""
    spec, batch = A.shape[0], A.shape[1]
    abs_A = A.abs().reshape(spec, batch, -1)
    vectors = [magnitude] if unbounded is None else [magnitude, unbounded]
    if all(v.ndim < A.ndim - 1 for v in vectors):     # no batch dimension: one vector for every row
        V = torch.stack([torch.broadcast_to(v, A.shape[2:]).reshape(-1) for v in vectors], -1)
        out = abs_A.matmul(V.to(A.dtype)).unbind(-1)
    else:
        try:
            V = torch.stack([torch.broadcast_to(v, A.shape[1:]).reshape(batch, -1) for v in vectors], -1)
        except RuntimeError:    # a magnitude with a larger batch than A: sum over it (conservative)
            out = [(A.abs() * v).reshape(spec, batch, -1).sum(-1) for v in vectors]
        else:
            out = torch.einsum('sbn,bnk->sbk', abs_A, V.to(A.dtype)).unbind(-1)
    total = torch.nan_to_num(out[0], nan=torch.inf)
    return total if unbounded is None else torch.where(out[1] > 0, torch.inf, total)


def _higham(magnitude, k):
    """gamma_k times a computed magnitude, rigorously: gamma_k = k u / (1 - k u) (Higham Lemma 3.1, u = eps / 2) on the
    exact magnitude, which the computed one (sums and products of nonnegative terms) underestimates by at most
    gamma_n relatively; (k + 1) u (1 + 1e-6) covers both for k and n up to 1e9 terms. Unlike rounding_error's
    (k + 16) * 4 eps, no further safety factor."""
    return magnitude * ((k + 1) * torch.finfo(magnitude.dtype).eps / 2 * (1 + 1e-6))


def _exact_charge(error, unbounded):
    """An upper bound of <|error|, inf where unbounded> for exactly known errors: +inf on rows where a nonzero error
    meets an unbounded value, else 0."""
    return _abs_dot(error, torch.zeros((), dtype=error.dtype, device=error.device), unbounded)


def _sum_error(a, b, total):
    """(a + b) - total for total = fl(a + b), exactly (Knuth's TwoSum); +inf where total is not finite."""
    b_virtual = total - a
    error = (a - (total - b_virtual)) + (b - b_virtual)
    return torch.where(torch.isfinite(total), error, torch.inf)


def _product_error(a, b):
    """a b - fl(a b), exactly (Dekker's TwoProduct, no FMA needed; eager, uncontracted arithmetic); +inf where the
    product is not finite or a split could overflow. Like rounding_error and _higham, it ignores underflow."""
    product = a * b
    info = torch.finfo(product.dtype)
    half = (round(-math.log2(info.eps)) + 2) // 2   # ceil(precision / 2): 27 for float64
    factor = 2.0 ** half + 1

    def split(x):
        t = factor * x
        high = t - (t - x)
        return high, x - high
    a_high, a_low = split(a)
    b_high, b_low = split(b)
    error = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low
    limit = info.max / 2.0 ** (half + 1)
    ok = torch.isfinite(product) & (a.abs() < limit) & (b.abs() < limit)
    return torch.where(ok, error, torch.inf)


def _abs_matmul(magnitude, constant, magnitude_first):
    """magnitude @ |constant| (or |constant| @ magnitude) for a nonnegative magnitude that may be +inf: +inf where an
    unbounded entry meets a nonzero coefficient, an exact zero term adds 0."""
    unbounded = torch.isinf(magnitude)
    both = torch.stack([torch.where(unbounded, 0, magnitude), unbounded.to(magnitude.dtype)])
    product = both.matmul(constant.abs()) if magnitude_first else constant.abs().matmul(both)
    return torch.where(product[1] > 0, torch.inf, product[0])


def _gather_repeats(node):
    """How often the most repeated index of a 1-D Gather occurs (its backward scatter-adds that many terms), cached
    for the node's index tensor."""
    cached = getattr(node, '_fp_gather_repeats', None)
    if cached is None or cached[0] is not node.indices:
        indices = node.indices.reshape(-1)
        repeats = int(torch.bincount(indices).max()) if indices.numel() else 0
        cached = node._fp_gather_repeats = (node.indices, repeats)
    return cached[1]


def _affine_backward_terms(node, last_A, A):
    """The rounding of an affine node's backward step from last_A to its inputs' A.

    Returns [(i, k, magnitude, product)]: the step's coefficients for input i round at most k times each (Higham's
    k, any summation order, FMA or not), so their error times the input's values is at most
    gamma_k <|last_A|, magnitude(m)>, magnitude(m) the node's output magnitude from input i alone on input
    magnitudes m, with |constants| (<|last_A|, |T_i|(m)> = <|T_i^*|(|last_A|), m>); i = 'bias' is the dot product of
    last_A with a Linear's folded bias (magnitude(None)). Where m is unbounded the term is vacuous, except for a
    product by constants (product = the constant): its exact error is used there instead. Exact steps (copies,
    slices, the expansion of ReduceSum, the gather of index_add with alpha 1) give no term, and neither do
    relaxations (left to their operators; None for a Linear/MatMul with a perturbed weight)."""
    if isinstance(node, BoundLinear):   # and BoundMatMul
        if isinstance(node, BoundMatMul):
            relaxed = node.is_input_perturbed(0) and node.is_input_perturbed(1)
        else:
            relaxed = node.is_input_perturbed(1)    # a perturbed weight: bound_backward_with_weight
        if relaxed:
            return None
        if isinstance(last_A, eyeC) or isinstance(last_A, OneHotC) and last_A.coeffs is None:
            return []   # rows of the weight and the bias, copied
        if (not isinstance(last_A, Tensor) or node.batched_weight_and_bias
                or node.use_seperate_weights_for_lower_and_upper_bounds):
            raise NotImplementedError(f'fp_enclosure affine_backward: {type(last_A).__name__} A at {node}')
        # Every entry of the input's A sums at most one product per output element.
        k = int(prod(node.output_shape[1:]))
        if isinstance(node, BoundMatMul):
            p = 1 if node.swap_x_and_weight else 0
            other = node.inputs[1 - p].forward_value
            return [(p, k, lambda m: _abs_matmul(m, other, p == 0), None)]
        weight = node.inputs[1].forward_value    # forward: x @ weight.t() (transB) or x @ weight
        right = weight.transpose(-2, -1) if node.transB else weight
        terms = [(0, k, lambda m: _abs_matmul(m, right, True), None)]
        # A perturbed bias takes last_A itself (exact); a fixed one is folded into the bias.
        if len(node.inputs) > 2 and not node.is_input_perturbed(2):
            bias = node.inputs[2].forward_value.abs() * abs(node.beta_linear)
            terms.append(('bias', k + (node.beta_linear != 1.0), lambda m: bias, None))
        return terms
    if isinstance(node, BoundGather):
        if node.indices.ndim == 0 or _gather_repeats(node) < 2:
            return []
        # The scatter-add into zeros: its first term is exact.
        return [(0, _gather_repeats(node) - 1,
                 lambda m: torch.index_select(m, dim=node.axis, index=node.indices), None)]
    if isinstance(node, BoundATenIndex_add):
        if node.alpha == 1.0:
            return []
        # src's A is the gather of last_A times alpha: one rounding, on the output positions src is added to.
        def scattered(m):
            shape = list(last_A.shape[1:])
            shape[node.axis] = m.shape[node.axis]
            return torch.zeros(last_A.shape[1:], dtype=m.dtype, device=m.device).index_add(
                node.axis, node.index, abs(node.alpha) * m.expand(shape))
        return [(3, 1, scattered, None)]
    if isinstance(node, BoundReduceMean):
        # The expansion of last_A (exact) divided by the reduced size: one rounding.
        return [(0, 1, lambda m: node.forward(m, *[inp.forward_value for inp in node.inputs[1:]]), None)]
    if isinstance(node, BoundMul) and node.is_linear_op:
        value = node.inputs[1 if node.inputs[0].perturbed else 0].forward_value
        cached = getattr(node, '_fp_factor', None)
        if (cached is None or cached[0] is not value or cached[1].dtype != last_A.dtype
                or cached[1].device != last_A.device):
            factor = value.to(last_A)
            # Products by 0 or a power of two are exact.
            inexact = torch.where((factor == 0) | (torch.frexp(factor).mantissa.abs() == 0.5), 0, factor.abs())
            cached = node._fp_factor = (value, factor, inexact)
        factor, inexact = cached[1], cached[2]
    elif isinstance(node, (BoundAdd, BoundSub, BoundWhere, BoundExpand)):
        factor = None
    else:
        return []
    # Elementwise with broadcasting: the product by a constant (Mul), then the sum over the broadcast elements.
    expand = lambda m: torch.broadcast_to(m, last_A.shape[1:])
    terms = []
    for i, (lA_i, uA_i) in enumerate(A):
        A_i = lA_i if lA_i is not None else uA_i
        if not isinstance(A_i, Tensor):
            continue
        if factor is not None:
            terms.append((i, 1, lambda m: expand(inexact * m), factor))
        r = last_A[0, 0].numel() // max(A_i[0, 0].numel(), 1)
        if r > 1:
            terms.append((i, r - 1, lambda m: expand(m if factor is None else factor.abs() * m), None))
    return terms


def _magnitude(self: 'BoundedModule', node, batch, cache):
    """(m, unbounded): an elementwise bound m >= |h| of node's values (its infinite entries zeroed) and the 0/1
    indicator of the unbounded entries (None if there are none), from the node's own current bounds (an inverted
    interval counts as unbounded, as in _interval_from_inputs), a root's box, or interval arithmetic from its inputs;
    None if no interval is known. Cached per backward pass."""
    key = ('magnitude', node.name)
    if key not in cache:
        box = None
        if not node.perturbed:
            lower = upper = self.get_forward_value(node)
        elif (node.is_lower_bound_current() and node.is_upper_bound_current() and isinstance(node.lower, Tensor)
                and isinstance(node.upper, Tensor) and node.lower.ndim > 0 and node.lower.shape[0] == batch):
            lower, upper = node.lower, node.upper
        else:
            lower = upper = None
        # From the node's own values or bounds it is also kept on the node, for the next passes with the same ones
        # (the same tensors, not modified in place since).
        kept = getattr(node, '_fp_magnitude', None)
        if (lower is not None and kept is not None and kept[0] is lower and kept[1] is upper
                and kept[2] == (lower._version, upper._version)):
            cache[key] = kept[3]
            return kept[3]
        if lower is None:
            box = _interval_from_inputs(self, node, batch, cache=cache)
            if (box is None and node.name in self.root_names and isinstance(node.lower, Tensor)
                    and isinstance(node.upper, Tensor)):
                box = Interval(node.lower, node.upper)  # a perturbed root's own box (compute_bounds sets it)
            lower, upper = (None, None) if box is None else (box.lower, box.upper)
        if lower is None:
            cache[key] = None
        else:
            # max(|l|, |u|) = max(-l, u) for l <= u
            m = torch.where(lower > upper, torch.inf, torch.maximum(-lower, upper))
            unbounded = torch.isinf(m)
            cache[key] = (torch.where(unbounded, 0, m), unbounded.to(m.dtype) if bool(unbounded.any()) else None)
            if box is None:
                node._fp_magnitude = (lower, upper, (lower._version, upper._version), cache[key])
    return cache[key]


def _merge_charge(old, new, m, unbounded, now):
    """add_bound's old + new errs by at most u/(1 - u) |fl(old + new)| per entry: charged now as gamma_1 on the
    finite magnitudes (`now`), or later by the node itself on its final A (_enclose_affine_backward's `deferred`).
    Where the values are unbounded, the exact error (TwoSum) is charged now, so an exact merge costs nothing."""
    if not now and unbounded is None:
        return 0
    total = old + new   # what add_bound's addA computes
    charge = _higham(_abs_dot(total, m), 1) if now else 0
    if unbounded is not None:
        charge = charge + _exact_charge(_sum_error(old, new, total), unbounded)
    return charge


def _enclose_affine_backward(self: 'BoundedModule', node, lA, uA, A, batch_size, cache, remaining, deferred):
    """Rounding of one backward step through `node`, as (lower, upper) bias corrections
    (bound_opts fp_enclosure affine_backward).

    Every step is charged against the computed last_A it starts from, so the charges of all steps add up: after each
    step, C f(x) >= sum_n A_n h_n(x) + lb - e for the computed A_n. The affine step's own rounding
    (_affine_backward_terms) is concretized on its inputs' magnitudes, all terms in one pass over |last_A|. So is
    add_bound's merge of each new A into the A accumulated at an input: the last merge into a node that is processed
    later (`remaining` contributions none, recorded in `deferred`) is charged there, as gamma_1 <|its A|, its
    magnitude>, in the same pass as its own step; other merges right away (_merge_charge). Relaxations are left to
    their operators."""
    terms = _affine_backward_terms(node, lA if lA is not None else uA, A) or []
    need = {i for i, _, _, _ in terms if i != 'bias'}
    for i, inp in enumerate(node.inputs):
        for side in (0, 1):
            if A[i][side] is not None and ((inp.lA, inp.uA)[side] is not None or node.inputs.count(inp) > 1):
                need.add(i)
    own = [side for side in (0, 1) if node.name in deferred.get(side, ())]
    if not terms and not need and not own:
        return 0, 0
    magnitudes = {}
    for i in need:
        magnitudes[i] = _magnitude(self, node.inputs[i], batch_size, cache)
        if magnitudes[i] is None:
            return torch.inf, torch.inf

    corrections = []
    for side, last_A in enumerate((lA, uA)):
        error = 0
        if last_A is None:
            corrections.append(error)
            continue
        vector, unbounded = 0, 0
        for i, k, magnitude, product in terms:
            if i == 'bias':
                vector = vector + _higham(magnitude(None), k)
                continue
            m, m_unbounded = magnitudes[i]
            vector = vector + _higham(magnitude(m), k)
            if m_unbounded is None:
                continue
            if product is None:
                unbounded = unbounded + magnitude(m_unbounded)
            else:   # the exact error of the products on the unbounded entries
                hit = magnitude(m_unbounded)
                if bool((hit > 0).any()):
                    error = error + _exact_charge(_product_error(last_A, product), hit)
        if side in own:
            m, m_unbounded = _magnitude(self, node, batch_size, cache)
            if (node.lA, node.uA)[side] is last_A:
                vector = vector + _higham(m, 1)
            else:   # last_A is not the accumulated A (beta-CROWN replaced it)
                error = error + _higham(_abs_dot((node.lA, node.uA)[side], m), 1)
            deferred[side].discard(node.name)
        if torch.is_tensor(vector):
            error = error + _abs_dot(last_A, vector, unbounded if torch.is_tensor(unbounded) else None)
        accumulated = {}
        for i, inp in enumerate(node.inputs):
            new = A[i][side]
            old = accumulated.get(inp.name, (inp.lA, inp.uA)[side])
            later = inp in node.inputs[i + 1:]
            if new is not None and old is not None:
                if not (isinstance(old, Tensor) and isinstance(new, Tensor)):
                    raise NotImplementedError(f'fp_enclosure affine_backward: merging {type(old).__name__} and '
                                              f'{type(new).__name__} A at {inp}')
                last = (not later and remaining.get(inp.name, 0) == 0 and inp.perturbed
                        and inp.name not in self.root_names)
                m, m_unbounded = magnitudes[i]
                error = error + _merge_charge(old, new, m, m_unbounded, now=not last)
                if last:
                    deferred.setdefault(side, set()).add(inp.name)
            if later:
                accumulated[inp.name] = new if old is None else old if new is None else old + new
        corrections.append(error)
    return corrections[0], corrections[1]


def _interval_fallback_rows(self: 'BoundedModule', node, C, lb, ub,
                            undefined_l, undefined_u):
    """Discarded rows of a dense or identity specification C get interval bounds.

    The node's interval follows from its inputs' current bounds by interval
    arithmetic, which needs no relaxation; C maps it to row bounds with the
    matching endpoint per coefficient sign. Sound whenever the input bounds are.
    """
    if isinstance(C, eyeC):
        # Identity rows: row s is element s of the node.
        C_dense = None
    elif isinstance(C, Tensor) and C.ndim >= 2:
        C_dense = C
    else:
        return lb, ub
    spec, batch = C.shape[:2]
    try:
        box = _interval_from_inputs(self, node, batch)
    except (OperatorDomainError, OperatorCapabilityError, NotImplementedError,
            TypeError, RuntimeError):
        return lb, ub
    if box is None:
        return lb, ub
    lower, upper = box.lower, box.upper
    if (not isinstance(lower, Tensor) or not isinstance(upper, Tensor)
            or lower.ndim == 0 or lower.shape[0] != batch):
        return lb, ub
    lower = torch.where(torch.isnan(lower), -torch.inf, lower)
    upper = torch.where(torch.isnan(upper), torch.inf, upper)
    if C_dense is None:
        if lower[0].numel() != spec:
            return lb, ub

        def _row(first, second):
            return first.reshape(batch, spec)
    else:
        lower, upper = lower.unsqueeze(0), upper.unsqueeze(0)
        zero = torch.zeros((), dtype=C.dtype, device=C.device)

        def _row(first, second):
            # sum_j C_j * (first_j if C_j > 0 else second_j), skipping C_j == 0.
            value = (torch.where(C > 0, C * first, zero)
                     + torch.where(C < 0, C * second, zero))
            return value.reshape(spec, batch, -1).sum(-1).transpose(0, 1)

    if undefined_l is not None and isinstance(lb, Tensor) and lb.shape == (batch, spec):
        lb = torch.where(_rows_like(undefined_l, lb),
                         torch.fmax(lb, _row(lower, upper).to(lb)), lb)
    if undefined_u is not None and isinstance(ub, Tensor) and ub.shape == (batch, spec):
        ub = torch.where(_rows_like(undefined_u, ub),
                         torch.fmin(ub, _row(upper, lower).to(ub)), ub)
    return lb, ub


def backward_general(
    self: 'BoundedModule',
    bound_node,
    C,
    start_backpropagation_at_node = None,
    bound_lower=True,
    bound_upper=True,
    average_A=False,
    need_A_only=False,
    unstable_idx=None,
    update_mask=None,
    apply_output_constraints_to: Optional[List[str]] = None,
    initial_As: Optional[dict] = None,
    initial_lb: Optional[torch.tensor] = None,
    initial_ub: Optional[torch.tensor] = None,
):
    use_beta_crown = self.bound_opts['optimize_bound_args']['enable_beta_crown']
    tighten_input_bounds = (
        self.bound_opts['optimize_bound_args']['tighten_input_bounds']
    )

    if self.invprop_enabled():
        self.invprop_init_infeasible_bounds(bound_node, C)
    if bound_node.are_output_constraints_activated_for_layer(apply_output_constraints_to):
        return self.backward_general_invprop(
            initial_As=initial_As, initial_lb=initial_lb, initial_ub=initial_ub,
            bound_node=bound_node, C=C,
            start_backpropagation_at_node=start_backpropagation_at_node,
            bound_lower=bound_lower, bound_upper=bound_upper,
            average_A=average_A, need_A_only=need_A_only,
            unstable_idx=unstable_idx, update_mask=update_mask
        )

    roots = self.roots()

    if start_backpropagation_at_node is None:
        # When output constraints are used, backward_general_with_output_constraint()
        # adds additional layers at the end, performs the backpropagation through these,
        # and then calls backward_general() on the output layer.
        # In this case, the layer we start from (start_backpropagation_at_node) differs
        # from the layer that should be bounded (bound_node)

        # When output constraints are not used, the bounded node is the one where
        # backpropagation starts.
        start_backpropagation_at_node = bound_node

    if self.verbose:
        logger.debug(f'Bound backward from {start_backpropagation_at_node.__class__.__name__}({start_backpropagation_at_node.name}) '
                     f'to bound {bound_node.__class__.__name__}({bound_node.name})')
        if isinstance(C, BatchedCrownC):
            logger.debug(f'  C: {C}')
        elif C is not None:
            logger.debug(f'  C: shape {C.shape}, type {type(C)}')
    _print_time = bool(os.environ.get('AUTOLIRPA_PRINT_TIME', 0))

    if isinstance(C, BatchedCrownC):
        # If C is a str, use batched CROWN. If batched CROWN is not intended to
        # be enabled, C must be a explicitly provided non-str object for this function.
        if need_A_only or average_A:
            raise ValueError(
                'Batched CROWN is not compatible with '
                f'need_A_only={need_A_only}, average_A={average_A}')
        ret = self.batched_backward(
            bound_node, C, unstable_idx,
            batch_size=self.get_forward_value(roots[0]).shape[0],
            bound_lower=bound_lower, bound_upper=bound_upper,
        )
        bound_node.lower, bound_node.upper = ret[:2]
        return ret

    # Runs for every bounded node, over all nodes. lA/uA are plain attributes: going through
    # nn.Module.__setattr__ (parameter/buffer/submodule checks) cost 8.9 ms per reset on
    # cute_spanhyd's 2305 nodes, most of a CROWN call there; a dict update costs 0.4 ms.
    for n in self.nodes():
        vars(n).update(lA=None, uA=None)

    degree_out = get_degrees(start_backpropagation_at_node)
    # Enclose the rounding of the affine backward steps and of the A merges (_enclose_affine_backward) when bound_opts
    # {'fp_enclosure': {'affine_backward': True}} (A67: without it 51 of 960 random cancelling graphs gave bounds above
    # the exact ones); off by default.
    fp_affine = self.bound_opts.get('fp_enclosure', {}).get('affine_backward', FP_AFFINE_DEFAULT)
    deferred_merges = {}
    C, batch_size, output_dim, output_shape = self._preprocess_C(C, bound_node)

    if initial_As is None:
        start_backpropagation_at_node.lA = C if bound_lower else None
        start_backpropagation_at_node.uA = C if bound_upper else None
    else:
        for layer_name, (lA, uA) in initial_As.items():
            self[layer_name].lA = lA
            self[layer_name].uA = uA
        assert start_backpropagation_at_node.lA is not None or start_backpropagation_at_node.uA is not None
    specification_dtype = self.get_output_dtype(bound_node)
    if initial_lb is None:
        lb = torch.tensor(0., dtype=specification_dtype, device=self.device)
    else:
        lb = initial_lb
    if initial_ub is None:
        ub = torch.tensor(0., dtype=specification_dtype, device=self.device)
    else:
        ub = initial_ub

    # Save intermediate layer A matrices when required.
    A_record = {}
    # (spec, batch) rows whose relaxation is undefined somewhere (see
    # _bound_backward_tolerant); their bounds become -inf / +inf.
    undefined_l = undefined_u = None
    rounding_intervals = {}

    queue = deque([start_backpropagation_at_node])
    while len(queue) > 0:
        l = queue.popleft()  # backward from l

        if l.name in self.root_names:
            continue

        # if all the succeeds are done, then we can turn to this node in the
        # next iteration.
        for l_pre in l.inputs:
            degree_out[l_pre.name] -= 1
            if degree_out[l_pre.name] == 0:
                queue.append(l_pre)

        # Initially, l.lA or l.uA will be set to C for this node.
        if l.lA is not None or l.uA is not None:
            if self.verbose:
                logger.debug(f'  Bound backward to {l} (out shape {l.output_shape})')
                if l.lA is not None:
                    logger.debug('    lA type %s shape %s',
                                 type(l.lA), list(l.lA.shape))
                if l.uA is not None:
                    logger.debug('    uA type %s shape %s',
                                 type(l.uA), list(l.uA.shape))

            if _print_time:
                start_time = time.time()

            self.backward_from[l.name].append(bound_node)

            if not l.perturbed:
                if l.forward_value is None:
                    self.get_forward_value(l)
                lb, ub = add_constant_node(lb, ub, l)
                continue

            if l.zero_uA_mtx and l.zero_lA_mtx:
                # A matrices are all zero, no need to propagate.
                continue

            lA, uA = l.lA, l.uA
            if (use_beta_crown and getattr(l, 'sparse_betas', None)):
                lA, uA, lbias, ubias = self.beta_crown_backward_bound(
                    l, lA, uA, start_node=start_backpropagation_at_node)
                lb = add_enclosed_bound(lb, lbias, -1)
                ub = add_enclosed_bound(ub, ubias, 1)

            if isinstance(l, BoundOptimizableActivation):
                # For other optimizable activation functions (TODO: unify with ReLU).
                if bound_node.name != self.final_node_name:
                    start_shape = bound_node.output_shape[1:]
                else:
                    start_shape = C.shape[0]
                l.preserve_mask = update_mask
                # Pass iteration info for caching relaxation parameters
                l._opt_iter = getattr(self, '_opt_iter', None)
                l._fix_interm_bounds = getattr(self, '_fix_interm_bounds', False)
            else:
                start_shape = None
            # Prepare concrete values for fixed operands used by bound_backward().
            for input_node in l.inputs:
                if (not input_node.perturbed
                        and input_node.forward_value is None):
                    self.get_forward_value(input_node)
            A, lower_b, upper_b, rows_l, rows_u = _bound_backward_tolerant(
                self, l, lA, uA,
                start_node=bound_node, unstable_idx=unstable_idx,
                start_shape=start_shape)
            if isinstance(l, (BoundAdd, BoundSub)):
                intervals = [_interval_from_inputs(self, inp, batch_size,
                              cache=rounding_intervals) for inp in l.inputs]
                if all(box is not None for box in intervals):
                    error = _affine_output_error(l, intervals)
                    for last_A, sign in ((lA, -1), (uA, 1)):
                        if isinstance(last_A, Tensor):
                            contribution = (finite_abs(last_A) * error.unsqueeze(0))
                            contribution = contribution.reshape(
                                last_A.shape[0], batch_size, -1).sum(-1)
                            if sign == -1:
                                lower_b = lower_b - contribution
                            else:
                                upper_b = upper_b + contribution
            if fp_affine:
                error_l, error_u = _enclose_affine_backward(
                    self, l, lA, uA, A, batch_size, rounding_intervals, degree_out, deferred_merges)
                lower_b = lower_b - error_l
                upper_b = upper_b + error_u
            if rows_l is not None:
                undefined_l = rows_l if undefined_l is None else undefined_l | rows_l
            if rows_u is not None:
                undefined_u = rows_u if undefined_u is None else undefined_u | rows_u

            # After propagation through this node, we delete its lA, uA variables.
            if bound_node.name != self.final_name:
                del l.lA, l.uA
            if _print_time:
                torch.cuda.synchronize()
                time_elapsed = time.time() - start_time
                if time_elapsed > 5e-3:
                    print(l, time_elapsed)
            if lb.ndim > 0 and type(lower_b) == Tensor and self.conv_mode == 'patches':
                lb, ub, lower_b, upper_b = check_patch_biases(lb, ub, lower_b, upper_b)
            lb = add_enclosed_bound(lb, lower_b, -1)
            ub = add_enclosed_bound(ub, upper_b, 1)
            if self.return_A and self.needed_A_dict and bound_node.name in self.needed_A_dict:
                # FIXME remove [0][0] and [0][1]?
                if len(self.needed_A_dict[bound_node.name]) == 0 or l.name in self.needed_A_dict[bound_node.name]:
                    # A could be either patches (in this case we cannot transpose so directly return)
                    # or matrix (in this case we transpose)
                    A_record.update({
                        l.name: {
                            "lA": (
                                A[0][0].detach() if isinstance(A[0][0], Patches)
                                else A[0][0].transpose(0, 1).detach()
                            ) if A[0][0] is not None else None,
                            "uA": (
                                A[0][1].detach() if isinstance(A[0][1], Patches)
                                else A[0][1].transpose(0, 1).detach()
                            ) if A[0][1] is not None else None,
                            # When not used, lb or ub is tensor(0).
                            "lbias": lb.transpose(0, 1).detach() if lb.ndim > 1 else None,
                            "ubias": ub.transpose(0, 1).detach() if ub.ndim > 1 else None,
                            "unstable_idx": unstable_idx
                    }})
                # FIXME: solve conflict with the following case
                self.A_dict.update({bound_node.name: A_record})
                if need_A_only and set(self.needed_A_dict[bound_node.name]) == set(A_record.keys()):
                    # We have collected all A matrices we need. We can return now!
                    _zero_undefined_root_A(
                        A_record, self.nodes(), undefined_l, undefined_u)
                    self.A_dict.update({bound_node.name: A_record})
                    # Do not concretize to save time. We just need the A matrices.
                    # return A matrix as a dict: {node_start.name: [A_lower, A_upper]}
                    return None, None, self.A_dict

            for i, l_pre in enumerate(l.inputs):
                add_bound(l, l_pre, lA=A[i][0], uA=A[i][1])

    if lb.ndim >= 2:
        lb = lb.transpose(0, 1)
    if ub.ndim >= 2:
        ub = ub.transpose(0, 1)
    # A descriptorless failure may leave only the initial scalar bias: there
    # are still (batch, spec) rows to discard, even when no root A survives.
    if lb.ndim == 0:
        lb = lb.expand(batch_size, output_dim)
    if ub.ndim == 0:
        ub = ub.expand(batch_size, output_dim)

    # TODO merge into `concretize`
    if (self.cut_used and getattr(self, 'cut_module', None) is not None
            and self.cut_module.x_coeffs is not None):
        # propagate input neuron in cut constraints
        roots[0].lA, roots[0].uA = self.cut_module.input_cut(
            bound_node, roots[0].lA, roots[0].uA, roots[0].lower.size()[1:], unstable_idx,
            batch_mask=update_mask)

    lb, ub = self.concretize_bounds(
        node=bound_node,
        lower=lb,
        upper=ub,
        concretize_mode='backward',
        batch_size=batch_size,
        output_dim=output_dim,
        average_A=average_A,
        clip_neuron_selection_value=self.clip_neuron_selection_value,
        clip_neuron_selection_type=self.clip_neuron_selection_type
    )
    if undefined_l is not None or undefined_u is not None:
        lb, ub = _discard_undefined_rows(lb, ub, undefined_l, undefined_u)
    if self.return_A and self.needed_A_dict and bound_node.name in self.needed_A_dict:
        save_root_A(
            bound_node, A_record, self.A_dict, roots,
            self.needed_A_dict[bound_node.name],
            lb=lb, ub=ub, unstable_idx=unstable_idx)
        if undefined_l is not None or undefined_u is not None:
            _zero_undefined_root_A(
                self.A_dict[bound_node.name], self.nodes(), undefined_l, undefined_u)
    if undefined_l is not None or undefined_u is not None:
        lb, ub = _interval_fallback_rows(
            self, bound_node, C, lb, ub, undefined_l, undefined_u)
    for root in self.roots():
        # These are saved for `save_root_A`. We do not need them afterwards.
        root.lb = root.ub = None

    if tighten_input_bounds and isinstance(bound_node, BoundInput):
        shape = bound_node.perturbation.x_L.shape
        lb_reshaped = lb.reshape(shape)
        bound_node.perturbation.x_L = lb_reshaped - lb_reshaped.detach() + torch.max(bound_node.perturbation.x_L.detach(), lb_reshaped.detach())
        ub_reshaped = ub.reshape(shape)
        bound_node.perturbation.x_U = ub_reshaped - ub_reshaped.detach() + torch.min(bound_node.perturbation.x_U.detach(), ub_reshaped.detach())

    lb = lb.view(batch_size, *output_shape) if bound_lower else None
    ub = ub.view(batch_size, *output_shape) if bound_upper else None

    # TODO merge into `concretize`
    if (self.cut_used and getattr(self, "cut_module", None) is not None
            and self.cut_module.cut_bias is not None):
        # propagate cut bias in cut constraints
        lb, ub = self.cut_module.bias_cut(bound_node, lb, ub, unstable_idx, batch_mask=update_mask)
        if lb is not None and ub is not None and ((lb-ub)>0).sum().item() > 0:
            # make sure there is no bug for cut constraints propagation
            print(f"Warning: lb is larger than ub with diff: {(lb-ub)[(lb-ub)>0].max().item()}")

    if self.verbose:
        logger.debug('')

    if self.invprop_enabled():
        lb, ub = self.invprop_check_infeasible_bounds(lb, ub)

    if self.return_A:
        if self.bound_opts['clip_in_alpha_crown'] and self.final_name in self.A_dict.keys():
            for v in self.A_dict[self.final_name].values():
                if v["lA"] is not None:
                    self.constraints_optimized = (v["lA"], v["lbias"])
        return lb, ub, self.A_dict
    else:
        return lb, ub


def get_unstable_size(unstable_idx):
    if isinstance(unstable_idx, tuple):
        return unstable_idx[0].numel()
    else:
        return unstable_idx.numel()


def check_optimized_variable_sparsity(self: 'BoundedModule', node):
    alpha_sparsity = None  # unknown, optimizable variables are not created for this node.
    for relu in self.relus:
        # FIXME: this hardcoded for ReLUs. Need to support other optimized nonlinear functions.
        # alpha_lookup_idx is only created for sparse-spec alphas.
        if relu.alpha_lookup_idx is not None and node.name in relu.alpha_lookup_idx:
            if relu.alpha_lookup_idx[node.name] is not None:
                # This node was created with sparse alpha
                alpha_sparsity = True
            elif self.bound_opts['optimize_bound_args']['use_shared_alpha']:
                # Shared alpha, the spec dimension is 1, and sparsity can be supported.
                alpha_sparsity = True
            else:
                alpha_sparsity = False
            break
    return alpha_sparsity


def get_sparse_C(self: 'BoundedModule', node, ref_intermediate):
    (sparse_intermediate_bounds,
     ref_intermediate_lb, ref_intermediate_ub) = ref_intermediate
    sparse_conv_intermediate_bounds = self.bound_opts.get('sparse_conv_intermediate_bounds', False)
    minimum_sparsity = self.bound_opts.get('minimum_sparsity', 0.9)
    crown_batch_size = self.bound_opts.get('crown_batch_size', 1e9)
    # Get refresh interval for unstable indices cache
    refresh_interval = self.bound_opts.get('unstable_idx_refresh_interval')
    if refresh_interval is None:
        refresh_interval = int(os.environ.get('AUTOLIRPA_UNSTABLE_REFRESH_INTERVAL', '1'))
    dim = int(prod(node.output_shape[1:]))
    batch_size = self.batch_size
    specification_dtype = self.get_output_dtype(node)

    reduced_dim = False  # Only partial neurons (unstable neurons) are bounded.
    unstable_idx = None
    unstable_size = np.inf
    newC = None

    alpha_is_sparse = self.check_optimized_variable_sparsity(node)

    # New: Check if we have cached sparse C and unstable indices.
    current_iter = getattr(self, '_unstable_idx_cache_iter', None)
    if (sparse_intermediate_bounds and refresh_interval > 1
            and current_iter is not None):
        cache_iter = getattr(node, '_cached_unstable_idx_iter', None)
        if cache_iter is not None and current_iter - cache_iter < refresh_interval:
            cached_C = getattr(node, '_cached_sparse_C', None)
            if (cached_C is not None
                    and getattr(node, '_cached_sparse_batch_size', None) == batch_size
                    and getattr(node, '_cached_sparse_dim', None) == dim
                    and getattr(node, '_cached_sparse_crown_batch_size', None) == crown_batch_size):
                return (cached_C,
                        node._cached_sparse_reduced_dim,
                        node._cached_unstable_idx,
                        node._cached_unstable_size)

    # NOTE: batched CROWN is so far only supported for some of the cases below

    # FIXME: C matrix shape incorrect for BoundParams.
    if (isinstance(node, BoundLinear) or isinstance(node, BoundMatMul)) and int(
            os.environ.get('AUTOLIRPA_USE_FULL_C', 0)) == 0:
        if sparse_intermediate_bounds:
            # If we are doing bound refinement and reference bounds are given,
            # we only refine unstable neurons.
            # Also, if we are checking against LP solver we will refine all
            # neurons and do not use this optimization.
            # For each batch element, we find the unstable neurons.
            unstable_idx, unstable_size = self.get_unstable_locations(
                ref_intermediate_lb, ref_intermediate_ub)
            if unstable_size == 0:
                # Do nothing, no bounds will be computed.
                reduced_dim = True
                unstable_idx = []
            elif unstable_size > crown_batch_size:
                # Create C in batched CROWN
                newC = BatchedCrownC('OneHot')
                reduced_dim = True
            elif (((0 < unstable_size <= minimum_sparsity * dim and alpha_is_sparse is None) or
                   alpha_is_sparse) and
                   len(node.output_shape) <= 2):
                # When we already have sparse alpha for this layer, we always
                # use sparse C. Otherwise we determine it by sparsity.
                # Create an abstract C matrix, the unstable_idx are the non-zero
                # elements in specifications for all batches.
                # Shouldn't use OneHotC if the output is not a 1-d tensor.
                newC = OneHotC(
                    [batch_size, unstable_size, *node.output_shape[1:]],
                    self.device, unstable_idx, None)
                reduced_dim = True
            else:
                unstable_idx = None
                del ref_intermediate_lb, ref_intermediate_ub
        if not reduced_dim:
            if dim > crown_batch_size:
                newC = BatchedCrownC('eye')
            else:
                newC = eyeC([batch_size, dim, *node.output_shape[1:]], self.device)
    elif node.patches_start and node.mode == "patches":
        if sparse_intermediate_bounds:
            unstable_idx, unstable_size = self.get_unstable_locations(
                ref_intermediate_lb, ref_intermediate_ub, conv=True)
            if unstable_size == 0:
                # Do nothing, no bounds will be computed.
                reduced_dim = True
                unstable_idx = []
            elif unstable_size > crown_batch_size:
                # Create C in batched CROWN
                newC = BatchedCrownC('Patches')
                reduced_dim = True
            # We sum over the channel direction, so need to multiply that.
            elif (sparse_conv_intermediate_bounds
                  and unstable_size <= minimum_sparsity * dim
                  and alpha_is_sparse is None) or alpha_is_sparse:
                # When we already have sparse alpha for this layer, we always
                # use sparse C. Otherwise we determine it by sparsity.
                # Create an abstract C matrix, the unstable_idx are the non-zero
                # elements in specifications for all batches.
                # The shape of patches is [unstable_size, batch, C, H, W].
                newC = Patches(
                    shape=[unstable_size, batch_size, *node.output_shape[1:-2],
                           1, 1],
                    identity=1, unstable_idx=unstable_idx,
                    output_shape=[batch_size, *node.output_shape[1:]])
                reduced_dim = True
            else:
                unstable_idx = None
                del ref_intermediate_lb, ref_intermediate_ub
        # Here we create an Identity Patches object
        if not reduced_dim:
            newC = Patches(
                None, 1, 0, [node.output_shape[1], batch_size, *node.output_shape[2:],
                *node.output_shape[1:-2], 1, 1], 1,
                output_shape=[batch_size, *node.output_shape[1:]])
    elif (isinstance(node, (BoundAdd, BoundSub)) and node.mode == "patches"
        and len(node.output_shape) >= 4):
        # FIXME: BoundAdd does not always have patches. Need to use a better way
        # to determine patches mode.
        # FIXME: We should not hardcode BoundAdd here!
        if sparse_intermediate_bounds:
            if crown_batch_size < 1e9:
                warnings.warn('Batched CROWN is not supported in this case')
            unstable_idx, unstable_size = self.get_unstable_locations(
                ref_intermediate_lb, ref_intermediate_ub, conv=True)
            if unstable_size == 0:
                # Do nothing, no bounds will be computed.
                reduced_dim = True
                unstable_idx = []
            elif (sparse_conv_intermediate_bounds
                  and unstable_size <= minimum_sparsity * dim
                  and alpha_is_sparse is None) or alpha_is_sparse:
                # When we already have sparse alpha for this layer, we always
                # use sparse C. Otherwise we determine it by sparsity.
                num_channel = node.output_shape[-3]
                # Identity patch size: (ouc_c, 1, 1, 1, out_c, 1, 1).
                patches = torch.eye(
                    num_channel,
                    device=self.device,
                    dtype=specification_dtype,
                ).view(
                        num_channel, 1, 1, 1, num_channel, 1, 1)
                # Expand to (out_c, 1, unstable_size, out_c, 1, 1).
                patches = patches.expand(-1, 1, node.output_shape[-2],
                                         node.output_shape[-1], -1, 1, 1)
                patches = patches[unstable_idx[0], :,
                                  unstable_idx[1], unstable_idx[2]]
                # Expand with the batch dimension. Final shape
                # (unstable_size, batch_size, out_c, 1, 1).
                patches = patches.expand(-1, batch_size, -1, -1, -1)
                newC = Patches(
                    patches, 1, 0, patches.shape, unstable_idx=unstable_idx,
                    output_shape=[batch_size, *node.output_shape[1:]])
                reduced_dim = True
            else:
                unstable_idx = None
                del ref_intermediate_lb, ref_intermediate_ub
        if not reduced_dim:
            num_channel = node.output_shape[-3]
            # Identity patch size: (ouc_c, 1, 1, 1, out_c, 1, 1).
            patches = torch.eye(
                num_channel,
                device=self.device,
                dtype=specification_dtype,
            ).view(
                    num_channel, 1, 1, 1, num_channel, 1, 1)
            # Expand to (out_c, batch, out_h, out_w, out_c, 1, 1).
            patches = patches.expand(-1, batch_size, node.output_shape[-2],
                                     node.output_shape[-1], -1, 1, 1)
            newC = Patches(patches, 1, 0, patches.shape, output_shape=[
                batch_size, *node.output_shape[1:]])
    else:
        if sparse_intermediate_bounds:
            unstable_idx, unstable_size = self.get_unstable_locations(
                ref_intermediate_lb, ref_intermediate_ub)
            if unstable_size == 0:
                # Do nothing, no bounds will be computed.
                reduced_dim = True
                unstable_idx = []
            elif unstable_size > crown_batch_size:
                # Create in C in batched CROWN
                newC = BatchedCrownC('eye')
                reduced_dim = True
            elif (unstable_size <= minimum_sparsity * dim
                  and alpha_is_sparse is None) or alpha_is_sparse:
                newC = torch.zeros(
                    [1, unstable_size, dim],
                    dtype=specification_dtype,
                    device=self.device,
                )
                # Fill the corresponding elements to 1.0
                newC[0, torch.arange(unstable_size), unstable_idx] = 1.0
                newC = newC.expand(batch_size, -1, -1).view(
                    batch_size, unstable_size, *node.output_shape[1:])
                reduced_dim = True
            else:
                unstable_idx = None
                del ref_intermediate_lb, ref_intermediate_ub
        if not reduced_dim:
            if dim > 1000:
                warnings.warn(
                    f"Creating an identity matrix with size {dim}x{dim} for node {node}. "
                    "This may indicate poor performance for bound computation. "
                    "If you see this message on a small network please submit "
                    "a bug report.", stacklevel=2)
            if dim > crown_batch_size:
                newC = BatchedCrownC('eye')
            else:
                newC = torch.eye(
                    dim,
                    dtype=specification_dtype,
                    device=self.device,
                ).unsqueeze(0).expand(
                    batch_size, -1, -1
                ).view(batch_size, dim, *node.output_shape[1:])
    # New: Cache the sparse C and unstable indices for future use.
    if (sparse_intermediate_bounds and refresh_interval > 1
            and current_iter is not None and newC is not None):
        node._cached_unstable_idx_iter = current_iter
        node._cached_sparse_C = newC
        node._cached_sparse_reduced_dim = reduced_dim
        node._cached_unstable_idx = unstable_idx
        node._cached_unstable_size = unstable_size
        node._cached_sparse_batch_size = batch_size
        node._cached_sparse_dim = dim
        node._cached_sparse_crown_batch_size = crown_batch_size

    return newC, reduced_dim, unstable_idx, unstable_size


def restore_sparse_bounds(self: 'BoundedModule', node, unstable_idx,
                          unstable_size, ref_intermediate,
                          new_lower=None, new_upper=None):
    ref_intermediate_lb, ref_intermediate_ub = ref_intermediate[1:]
    batch_size = self.batch_size
    if unstable_size == 0:
        # No unstable neurons. Skip the update.
        node.lower = ref_intermediate_lb.detach().clone()
        node.upper = ref_intermediate_ub.detach().clone()
    else:
        if new_lower is None:
            new_lower = node.lower
        if new_upper is None:
            new_upper = node.upper
        # If we only calculated unstable neurons, we need to scatter the results back based on reference bounds.
        if isinstance(unstable_idx, tuple):
            lower = ref_intermediate_lb.detach().clone()
            upper = ref_intermediate_ub.detach().clone()
            # Conv layer with patches, the unstable_idx is a 3-element tuple for 3 indices (C, H,W) of unstable neurons.
            if len(unstable_idx) == 3:
                lower[:, unstable_idx[0], unstable_idx[1], unstable_idx[2]] = new_lower
                upper[:, unstable_idx[0], unstable_idx[1], unstable_idx[2]] = new_upper
            elif len(unstable_idx) == 4:
                lower[:, unstable_idx[0], unstable_idx[1], unstable_idx[2], unstable_idx[3]] = new_lower
                upper[:, unstable_idx[0], unstable_idx[1], unstable_idx[2], unstable_idx[3]] = new_upper
        else:
            # Other layers.
            lower = ref_intermediate_lb.detach().clone().reshape(batch_size, -1)
            upper = ref_intermediate_ub.detach().clone().reshape(batch_size, -1)
            lower[:, unstable_idx] = new_lower.view(batch_size, -1)
            upper[:, unstable_idx] = new_upper.view(batch_size, -1)
        node.lower = lower.view(batch_size, *node.output_shape[1:])
        node.upper = upper.view(batch_size, *node.output_shape[1:])


def get_degrees(node_start):
    if not isinstance(node_start, list):
        node_start = [node_start]
    degrees = {}
    added = {}
    queue = deque()
    for node in node_start:
        queue.append(node)
        added[node.name] = True
    while len(queue) > 0:
        l = queue.popleft()
        for l_pre in l.inputs:
            degrees[l_pre.name] = degrees.get(l_pre.name, 0) + 1
            if not added.get(l_pre.name, False):
                queue.append(l_pre)
                added[l_pre.name] = True
    return degrees


def _preprocess_C(self: 'BoundedModule', C, node):
    if isinstance(C, Patches):
        if C.unstable_idx is None:
            # Patches have size (out_c, batch, out_h, out_w, c, h, w).
            if len(C.shape) == 7:
                out_c, batch_size, out_h, out_w = C.shape[:4]
                output_dim = out_c * out_h * out_w
            else:
                out_dim, batch_size, out_c, out_h, out_w = C.shape[:5]
                output_dim = out_dim * out_c * out_h * out_w
        else:
            # Patches have size (unstable_size, batch, c, h, w).
            output_dim, batch_size = C.shape[:2]
    else:
        batch_size, output_dim = C.shape[:2]

    # The C matrix specified by the user has shape (batch, spec)
    # but internally we have (spec, batch) format.
    if not isinstance(C, (eyeC, Patches, OneHotC)):
        C = C.transpose(0, 1).reshape(
            output_dim, batch_size, *node.output_shape[1:])
    elif isinstance(C, eyeC):
        C = C._replace(shape=(C.shape[1], C.shape[0], *C.shape[2:]))
    elif isinstance(C, OneHotC):
        C = C._replace(
            shape=(C.shape[1], C.shape[0], *C.shape[2:]),
            index=C.index.transpose(0,-1),
            coeffs=None if C.coeffs is None else C.coeffs.transpose(0,-1))

    if isinstance(C, Patches) and C.unstable_idx is not None:
        # Sparse patches; the output shape is (unstable_size, ).
        output_shape = [C.shape[0]]
    elif prod(node.output_shape[1:]) != output_dim and not isinstance(C, Patches):
        # For the output node, the shape of the bound follows C
        # instead of the original output shape
        #
        # TODO Maybe don't set node.lower and node.upper in this case?
        # Currently some codes still depend on node.lower and node.upper
        output_shape = [-1]
    else:
        # Generally, the shape of the bounds match the output shape of the node
        output_shape = node.output_shape[1:]

    return C, batch_size, output_dim, output_shape


def addA(A1, A2):
    """ Add two A (each of them is either Tensor or Patches) """
    if type(A1) == type(A2):
        return A1 + A2
    elif type(A1) == Patches:
        return A1 + A2
    elif type(A2) == Patches:
        return A2 + A1
    else:
        raise NotImplementedError(f'Unsupported types for A1 ({type(A1)}) and A2 ({type(A2)}')


def add_bound(node, node_pre, lA=None, uA=None):
    """
    Propagate lA and uA to a preceding node.
    @param node:        The current bounded node
    @param node_pre:    An input of the current bounded node that needs lA, lbias ,etc. back propagated to it
    @param lA:          lA matrix associated with the current bounded node
    @param uA:          uA matrix associated with the current bounded node
    @return:
    """

    if lA is not None:
        if node_pre.lA is None:
            # First A added to this node.
            node_pre.zero_lA_mtx = node.zero_backward_coeffs_l
            node_pre.lA = lA
        else:
            node_pre.zero_lA_mtx = node_pre.zero_lA_mtx and node.zero_backward_coeffs_l
            new_node_lA = addA(node_pre.lA, lA)
            node_pre.lA = new_node_lA
    if uA is not None:
        if node_pre.uA is None:
            # First A added to this node.
            node_pre.zero_uA_mtx = node_pre.zero_backward_coeffs_u
            node_pre.uA = uA
        else:
            node_pre.zero_uA_mtx = node_pre.zero_uA_mtx and node.zero_backward_coeffs_u
            node_pre.uA = addA(node_pre.uA, uA)


def add_constant_node(lb, ub, node):
    def contribution(A, sign):
        bound = node.get_bias(A, node.forward_value)
        if not isinstance(bound, Tensor):
            return bound
        if isinstance(A, Tensor):
            abs_A = finite_abs(A)
        elif isinstance(A, Patches) and not A.identity:
            abs_A = A.create_similar(patches=finite_abs(A.patches))
        else:
            abs_A = A
        magnitude = node.get_bias(abs_A, finite_abs(node.forward_value))
        return widen_bound(bound, rounding_error(magnitude, coefficient_terms(A)), sign)

    new_lb = contribution(node.lA, -1)
    new_ub = contribution(node.uA, 1)
    if isinstance(lb, Tensor) and isinstance(new_lb, Tensor) and lb.ndim > 0 and lb.ndim != new_lb.ndim:
        new_lb = new_lb.reshape(lb.shape)
    if isinstance(ub, Tensor) and isinstance(new_ub, Tensor) and ub.ndim > 0 and ub.ndim != new_ub.ndim:
        new_ub = new_ub.reshape(ub.shape)
    lb = add_enclosed_bound(lb, new_lb, -1) # FIXME (09/16): shape for the bias of BoundConstant.
    ub = add_enclosed_bound(ub, new_ub, 1)
    return lb, ub


def save_root_A(node, A_record, A_dict, roots, needed_A_dict, lb, ub,
                unstable_idx):
    root_A_record = {}
    for i in range(len(roots)):
        if roots[i].lA is None and roots[i].uA is None:
            continue
        if roots[i].name in needed_A_dict:
            if roots[i].lA is not None:
                if isinstance(roots[i].lA, Patches):
                    _lA = roots[i].lA.detach()
                else:
                    _lA = roots[i].lA.transpose(0, 1).detach()
            else:
                _lA = None
            if roots[i].uA is not None:
                if isinstance(roots[i].uA, Patches):
                    _uA = roots[i].uA.detach()
                else:
                    _uA = roots[i].uA.transpose(0, 1).detach()
            else:
                _uA = None

            # Include all the bias terms except the one concretized from the
            # current root node.
            lb_ = lb - roots[i].lb if (roots[i].lb is not None) else lb
            ub_ = ub - roots[i].ub if (roots[i].ub is not None) else ub

            root_A_record.update({roots[i].name: {
                "lA": _lA,
                "uA": _uA,
                # When not used, lb or ub is tensor(0). They have been transposed above.
                "lbias": lb_.detach() if lb_.ndim > 1 else None,
                "ubias": ub_.detach() if ub_.ndim > 1 else None,
                "unstable_idx": unstable_idx
            }})

    root_A_record.update(A_record)  # merge to existing A_record
    A_dict.update({node.name: root_A_record})


def select_unstable_idx(ref_intermediate_lb, ref_intermediate_ub, unstable_locs, max_crown_size):
    """When there are too many unstable neurons, only bound those
    with the loosest reference bounds."""
    gap = (
        ref_intermediate_ub[:, unstable_locs]
        - ref_intermediate_lb[:, unstable_locs]).sum(dim=0)
    indices = torch.argsort(gap, descending=True)
    indices_selected = indices[:max_crown_size]
    indices_selected, _ = torch.sort(indices_selected)
    print(f'{len(indices_selected)}/{len(indices)} unstable neurons selected for CROWN')
    return indices_selected


def get_unstable_locations(self: 'BoundedModule', ref_intermediate_lb,
                           ref_intermediate_ub, conv=False, channel_only=False):
    # FIXME (2023): This function should be a member class of the Bound object, since the
    # definition of unstable neurons depends on the activation function.
    max_crown_size = self.bound_opts.get('max_crown_size', int(1e9))
    # For conv layer we only check the case where all neurons are active/inactive.
    unstable_masks = torch.logical_and(ref_intermediate_lb < 0, ref_intermediate_ub > 0)
    # For simplicity, merge unstable locations for all elements in this batch. TODO: use individual unstable mask.
    # It has shape (H, W) indicating if a neuron is unstable/stable.
    # TODO: so far we merge over the batch dimension to allow easier implementation.
    if channel_only:
        # Only keep channels with unstable neurons. Used for initializing alpha.
        unstable_locs = unstable_masks.sum(dim=(0,2,3)).bool()
        # Shape is consistent with linear layers: a list of unstable neuron channels (no batch dim).
        unstable_idx = unstable_locs.nonzero().squeeze(1)
    else:
        if not conv and unstable_masks.ndim > 2:
            # Flatten the conv layer shape.
            unstable_masks = unstable_masks.reshape(unstable_masks.size(0), -1)
            ref_intermediate_lb = ref_intermediate_lb.reshape(ref_intermediate_lb.size(0), -1)
            ref_intermediate_ub = ref_intermediate_ub.reshape(ref_intermediate_ub.size(0), -1)
        unstable_locs = unstable_masks.sum(dim=0).bool()
        if conv:
            # Now converting it to indices for these unstable nuerons.
            # These are locations (i,j) of unstable neurons.
            unstable_idx = unstable_locs.nonzero(as_tuple=True)
        else:
            unstable_idx = unstable_locs.nonzero().squeeze(1)

    unstable_size = get_unstable_size(unstable_idx)
    if unstable_size > max_crown_size:
        indices_seleted = select_unstable_idx(
            ref_intermediate_lb, ref_intermediate_ub, unstable_locs, max_crown_size)
        if isinstance(unstable_idx, tuple):
            unstable_idx = tuple(u[indices_seleted] for u in unstable_idx)
        else:
            unstable_idx = unstable_idx[indices_seleted]
    unstable_size = get_unstable_size(unstable_idx)

    return unstable_idx, unstable_size


def get_alpha_crown_start_nodes(
        self: 'BoundedModule',
        node,
        c=None,
        share_alphas=False,
        final_node_name=None,
    ):
    """
    Given a layer "node", return a list of following nodes after this node whose bounds
    will propagate through this node. Each element in the list is a tuple with 3 elements:
    (following_node_name, following_node_shape, unstable_idx)
    """
    # When use_full_conv_alpha is True, conv layers do not share alpha.
    sparse_intermediate_bounds = self.bound_opts.get('sparse_intermediate_bounds', False)
    use_full_conv_alpha_thresh = self.bound_opts.get('use_full_conv_alpha_thresh', 512)

    start_nodes = []

    for nj in self.backward_from[node.name]:  # Pre-activation layers.
        unstable_idx = None
        use_sparse_conv = None  # Whether a sparse-spec alpha is used for a conv output node. None for non-conv output node.
        use_full_conv_alpha = self.bound_opts.get('use_full_conv_alpha', False)

        # Find the indices of unstable neuron, used for create sparse-feature alpha.
        if (sparse_intermediate_bounds
                and isinstance(node, BoundOptimizableActivation)
                and nj.name != final_node_name and not share_alphas):
            # Create sparse optimization variables for intermediate neurons.
            # These are called "sparse-spec" alpha because we only create alpha only for
            # the intermediate of final output nodes whose bounds are needed.
            # "sparse-spec" alpha makes sense only for piece-wise linear functions.
            # For other intermediate nodes, there is no "unstable" or "stable" neuron.
            # FIXME: whether an layer has unstable/stable neurons should be in Bound obj.
            # FIXME: get_unstable_locations should be a member class of ReLU.
            if len(nj.output_name) == 1 and isinstance(self[nj.output_name[0]], (BoundRelu, BoundSignMerge, BoundMaxPool)):
                if ((isinstance(nj, (BoundLinear, BoundMatMul)))
                        and int(os.environ.get('AUTOLIRPA_USE_FULL_C', 0)) == 0):
                    # unstable_idx has shape [neuron_size_of_nj]. Batch dimension is reduced.
                    unstable_idx, _ = self.get_unstable_locations(nj.lower, nj.upper)
                elif isinstance(nj, (BoundConv, BoundAdd, BoundSub, BoundBatchNormalization)) and nj.mode == 'patches':
                    if nj.name in node.patch_size:
                        # unstable_idx has shape [channel_size_of_nj]. Batch and spatial dimensions are reduced.
                        unstable_idx, _ = self.get_unstable_locations(
                            nj.lower, nj.upper, channel_only=not use_full_conv_alpha, conv=True)
                        use_sparse_conv = False  # alpha is shared among channels. Sparse-spec alpha in hw dimension not used.
                        if use_full_conv_alpha and unstable_idx[0].size(0) > use_full_conv_alpha_thresh:
                            # Too many unstable neurons. Using shared alpha per channel.
                            unstable_idx, _ = self.get_unstable_locations(
                                nj.lower, nj.upper, channel_only=True, conv=True)
                            use_full_conv_alpha = False
                    else:
                        # Matrix mode for conv layers. Although the bound propagation started with patches mode,
                        # when A matrix is propagated to this layer, it might become a dense matrix since patches
                        # can be come very large after many layers. In this case,
                        # unstable_idx has shape [c_out * h_out * w_out]. Batch dimension is reduced.
                        unstable_idx, _ = self.get_unstable_locations(nj.lower, nj.upper)
                        use_sparse_conv = True  # alpha is not shared among channels, and is sparse in spec dimension.
            else:
                # FIXME: we should not check for fixed names here. Need to enable patches mode more generally.
                if isinstance(nj, (BoundConv, BoundAdd, BoundSub, BoundBatchNormalization)) and nj.mode == 'patches':
                    use_sparse_conv = False  # Sparse-spec alpha can never be used, because it is not a ReLU activation.

        if nj.name == final_node_name:
            # Final layer, always the number of specs as the shape.
            size_final = self[final_node_name].output_shape[1:] if c is None else c.size(1)
            # The 4-th element indicates that this start node is the final node,
            # which may be utilized by operators that do not know the name of
            # the final node.
            start_nodes.append((final_node_name, size_final, None, True))
            continue

        if share_alphas:
            # all intermediate neurons from the same layer share the same set of alphas.
            output_shape = 1
        elif isinstance(node, BoundOptimizableActivation) and node.patch_size and nj.name in node.patch_size:
            # Patches mode. Use output channel size as the spec size. This still shares some alpha, but better than no sharing.
            if use_full_conv_alpha:
                # alphas not shared among channels, so the spec dim shape is c,h,w
                # The patch size is [out_ch, batch, out_h, out_w, in_ch, H, W]. We use out_ch as the output shape.
                output_shape = node.patch_size[nj.name][0], node.patch_size[nj.name][2], node.patch_size[nj.name][3]
            else:
                # The spec dim is c only, and is shared among h, w.
                output_shape = node.patch_size[nj.name][0]
            assert not sparse_intermediate_bounds or use_sparse_conv is False  # Double check our assumption holds. If this fails, then we created wrong shapes for alpha.
        else:
            # Output is linear layer (use_sparse_conv = None), or patch converted to matrix (use_sparse_conv = True).
            assert not sparse_intermediate_bounds or use_sparse_conv is not False  # Double check our assumption holds. If this fails, then we created wrong shapes for alpha.
            output_shape = nj.lower.shape[1:]  # FIXME: for non-relu activations it's still expecting a prod.
        start_nodes.append((nj.name, output_shape, unstable_idx, False))

    return start_nodes


def merge_A(node, batch_A, ret_A):
    for key0 in batch_A:
        if key0 not in ret_A: ret_A[key0] = {}
        for key1 in batch_A[key0]:
            value = batch_A[key0][key1]
            if key1 not in ret_A[key0]:
                # create:
                ret_A[key0].update({
                    key1: {
                        "lA": value["lA"],
                        "uA": value["uA"],
                        "lbias": value["lbias"],
                        "ubias": value["ubias"],
                        "unstable_idx": value["unstable_idx"]
                    }
                })
            elif key0 == node.name:
                # merge:
                # the batch splitting only happens for current node, i.e.,
                # for other nodes the returned lA should be the same across different batches
                # so no need to repeatly merge them
                exist = ret_A[key0][key1]

                if exist["unstable_idx"] is not None:
                    if isinstance(exist["unstable_idx"], torch.Tensor):
                        merged_unstable = torch.cat([
                            exist["unstable_idx"],
                            value['unstable_idx']], dim=0)
                    elif isinstance(exist["unstable_idx"], tuple):
                        if exist["unstable_idx"]:
                            merged_unstable = tuple([
                                torch.cat([exist["unstable_idx"][idx],
                                           value['unstable_idx'][idx]], dim=0)
                                for idx in range(len(exist['unstable_idx']))]
                            )
                        else:
                            merged_unstable = None
                    else:
                        raise NotImplementedError(
                            f'Unsupported type {type(exist["unstable_idx"])}')
                else:
                    merged_unstable = None
                merge_dict = {"unstable_idx": merged_unstable}
                for name in ["lA", "uA"]:
                    if exist[name] is not None:
                        if isinstance(exist[name], torch.Tensor):
                            # for matrix the spec dim is 1
                            merge_dict[name] = torch.cat([exist[name], value[name]], dim=1)
                        else:
                            assert isinstance(exist[name], Patches)
                            # for patches the spec dim`is 0
                            merge_dict[name] = exist[name].create_similar(
                                torch.cat([exist[name].patches, value[name].patches], dim=0),
                                unstable_idx=merged_unstable
                            )
                    else:
                        merge_dict[name] = None
                for name in ["lbias", "ubias"]:
                    if exist[name] is not None:
                        # for bias the spec dim in 1
                        merge_dict[name] = torch.cat([exist[name], value[name]], dim=1)
                    else:
                        merge_dict[name] = None
                ret_A[key0][key1] = merge_dict
    return ret_A
