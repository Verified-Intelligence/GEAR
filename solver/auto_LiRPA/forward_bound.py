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
import torch
import warnings
from .bound_ops import *
from .utils import *
from .linear_bound import LinearBound
from .perturbations import PerturbationL0Norm, PerturbationLpNorm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule

import sys
sys.setrecursionlimit(1000000)


def _save_forward_A(self: 'BoundedModule', node, lw, uw, lower, upper):
    """Save target-node forward affine bounds for perturbed model inputs.

    Forward coefficients use the concatenated perturbation coordinates of all
    roots. Each returned entry is mapped back to one model input. Its bias
    includes the concretized contributions of all other perturbed roots, as in
    the backward return_A contract. ``lw`` and ``uw`` have shape
    ``[batch, perturbation_dim, *target_shape]``; returned coefficients have
    shape ``[batch, specification, *input_shape]``.
    """
    roots = self.roots()
    lw_flat = lw.reshape(self.batch_size, self.dim_in, -1)
    uw_flat = uw.reshape(self.batch_size, self.dim_in, -1)
    lower = lower.reshape(self.batch_size, -1)
    upper = upper.reshape(self.batch_size, -1)
    entries = {}
    requested_inputs = set(self.needed_A_dict.get(node.name, ()))
    offset = 0
    for root in roots:
        if root.perturbation is None:
            continue
        next_offset = offset + root.dim
        if type(root) is not BoundInput or root.name not in requested_inputs:
            offset = next_offset
            continue
        # Lp and L0 initialize forward variables as input coordinates or a
        # selected subset, so multiplying by this map produces input-space A.
        if not isinstance(
                root.perturbation, (PerturbationL0Norm, PerturbationLpNorm)):
            raise NotImplementedError(
                'forward return_A supports Lp and L0 input perturbations, got '
                f'{type(root.perturbation).__name__}')
        if root.lb is None or root.ub is None:
            raise RuntimeError(
                f'Missing forward contribution for root {root.name}')

        input_lw = root.linear.lw[:, offset:next_offset].reshape(
            self.batch_size, root.dim, -1)
        input_uw = root.linear.uw[:, offset:next_offset].reshape(
            self.batch_size, root.dim, -1)
        coordinate_lA = lw_flat[:, offset:next_offset].transpose(1, 2)
        coordinate_uA = uw_flat[:, offset:next_offset].transpose(1, 2)
        # BMM requires one dtype; retain the precision of every coefficient
        # tensor participating in the coordinate transformation.
        calculation_dtype = torch.promote_types(
            torch.promote_types(coordinate_lA.dtype, coordinate_uA.dtype),
            torch.promote_types(input_lw.dtype, input_uw.dtype))
        # Expand identity/selector coordinates into model-input coordinates.
        lA = coordinate_lA.to(dtype=calculation_dtype).bmm(
            input_lw.to(dtype=calculation_dtype))
        uA = coordinate_uA.to(dtype=calculation_dtype).bmm(
            input_uw.to(dtype=calculation_dtype))
        input_shape = self.get_forward_value(root).shape[1:]
        # forward_concretize temporarily records this root's contribution.
        # Removing it leaves a bias that independently concretizes all other
        # roots and the affine constant term.
        lbias = lower - root.lb.reshape(self.batch_size, -1)
        ubias = upper - root.ub.reshape(self.batch_size, -1)
        entries[root.name] = {
            'lA': lA.reshape(self.batch_size, -1, *input_shape).detach(),
            'uA': uA.reshape(self.batch_size, -1, *input_shape).detach(),
            'lbias': lbias.detach(),
            'ubias': ubias.detach(),
            'unstable_idx': None,
        }
        offset = next_offset
    self.A_dict[node.name] = entries


def forward_general(
        self: 'BoundedModule', C=None, node:'Bound'=None, concretize=False,
        offset=0, from_node=False, return_A=False):
    """Propagate forward affine bounds and optionally save input-space A."""

    if self.dynamic:
        return self.forward_general_dynamic(C=C, node=node, concretize=concretize, offset=offset)
    if C is None:
        if (hasattr(node, 'linear') and
                node.linear.lower is not None
                and node.linear.upper is not None
                and not return_A):
            return node.linear.lower, node.linear.upper
        # Exactness follows the active perturbations, including parameters.
        if not node.perturbed:
            value = self.get_forward_value(node)
            node.lower = node.upper = value
            if node.batch_dim == -1:
                node.linear = LinearBound(
                    None, value, None, value, value, value)
                return value, value
            # Batch-shaped fixed branches propagate zero coefficients for
            # return_A, preserving the batch and variable axes when merged.
        if node.is_lower_bound_current() and not return_A:
            node.linear = LinearBound(None, node.lower, None, node.upper, node.lower, node.upper)
            return node.lower, node.upper

    for l_pre in node.inputs:
        if not hasattr(l_pre, 'linear'):
            self.forward_general(
                node=l_pre, offset=offset, from_node=from_node,
                return_A=return_A)
        elif (return_A and l_pre.perturbed
              and (getattr(l_pre.linear, 'lw', None) is None
                   or getattr(l_pre.linear, 'uw', None) is None)):
            self.forward_general(
                node=l_pre, offset=offset, from_node=from_node,
                return_A=True)
    inp = [l_pre.linear for l_pre in node.inputs]
    node._start = '_forward'
    if (C is not None and type(node) is BoundLinear and
            not node.is_input_perturbed(1)
            and not node.is_input_perturbed(2)):
        linear = node.bound_forward(self.dim_in, *inp, C=C)
        C_merged = True
    else:
        linear = node.linear = node.bound_forward(self.dim_in, *inp)
        C_merged = False

    lw, uw = linear.lw, linear.uw
    lower, upper = linear.lb, linear.ub

    # Combine linear bounds with C matrix
    if C is not None and not C_merged:
        # FIXME use bound_forward of BoundLinear
        C_pos, C_neg = C.clamp(min=0), C.clamp(max=0)
        # Flatten lw, uw for matrix multiplication
        lw = lw.reshape(self.batch_size, self.dim_in, -1)
        uw = uw.reshape(self.batch_size, self.dim_in, -1)
        _lw = torch.matmul(lw, C_pos.transpose(-1, -2)) + torch.matmul(uw, C_neg.transpose(-1, -2))
        _uw = torch.matmul(uw, C_pos.transpose(-1, -2)) + torch.matmul(lw, C_neg.transpose(-1, -2))
        lw, uw = _lw, _uw
        # Flatten lower, upper for matrix multiplication
        lower = lower.reshape(self.batch_size, -1)
        upper = upper.reshape(self.batch_size, -1)
        _lower = ( torch.matmul(lower.unsqueeze(1), C_pos.transpose(-1, -2)) 
                    + torch.matmul(upper.unsqueeze(1), C_neg.transpose(-1, -2)) )
        _upper = ( torch.matmul(upper.unsqueeze(1), C_pos.transpose(-1, -2))
                    + torch.matmul(lower.unsqueeze(1), C_neg.transpose(-1, -2)) )
        lower, upper = _lower.squeeze(1), _upper.squeeze(1)

    logger.debug(f'Forward bounds to {node}')

    if concretize:
        # return_A exposes temporary per-root concretization contributions;
        # clear them even if input-coordinate mapping raises an exception.
        try:
            if lw is not None or uw is not None:
                lower, upper = self.concretize_bounds(
                    node=node,
                    lower=lower,
                    upper=upper,
                    concretize_mode='forward',
                    lw=lw,
                    uw=uw,
                    return_A=return_A,
                    clip_neuron_selection_value=self.clip_neuron_selection_value,
                    clip_neuron_selection_type=self.clip_neuron_selection_type
                )

            linear.lower, linear.upper = lower, upper

            if C is None:
                node.linear = linear
                node.lower, node.upper = lower, upper

            if return_A:
                requested_inputs = self.needed_A_dict.get(node.name, ())
                if requested_inputs and (lw is None or uw is None):
                    raise RuntimeError(
                        f'Forward bounds for {node.name} do not contain both '
                        'lower and upper affine coefficients')
                if requested_inputs:
                    _save_forward_A(self, node, lw, uw, lower, upper)
        finally:
            if return_A:
                for root in self.roots():
                    root.lb = root.ub = None

        if self.bound_opts['forward_refinement']:
            need_refinement = False
            for out in node.output_name:
                out_node = self[out]
                for i in getattr(out_node, 'requires_input_bounds', []):
                    if out_node.inputs[i] == node:
                        need_refinement = True
                        break
            if need_refinement:
                self.forward_refinement(node)
        return lower, upper


def forward_general_dynamic(self: 'BoundedModule', C=None, node:'Bound'=None,
                            concretize=False, offset=0, variable_values=None):
    """Propagate the zonotope.

    Shapes carried on a returned LinearBound, for a node of value shape
    `*shape` and `numel = prod(shape)`:

        lw, uw  [batch, n_cols, *shape]   one column per bounded variable
        lb, ub  [batch, *shape]           the value at e = 0
        tot_dim                           n_cols, when every column is held

    `variable_values` supplies the variables as numbers instead of columns: a
    node's block is read off it and folded into the bias, so the pass carries
    no coefficients and its bias channel is the relaxed network at that point.
    A variable is a column or a number there, never both. Left None -- the
    usual case -- the variables stay free and the coefficients are their
    Jacobian.
    """
    if concretize or variable_values is not None:
        # Concretizing sums the windows itself; supplying values leaves no
        # columns to window.
        return self._dynamic_window(
            C=C, node=node, concretize=concretize, offset=offset,
            variable_values=variable_values)

    # A caller reading the coefficients wants all of them, and one window
    # carries at most `forward_max_dim`. Hand back the concatenation, or the
    # caller silently gets a bound missing every column past the first window
    # and takes it for the whole one.
    linear = self._dynamic_window(C=C, node=node, offset=offset)
    if linear.lw is None or offset + linear.lw.shape[1] >= linear.tot_dim:
        return linear
    # Each window is [batch, n_cols_i, *shape], differing only in n_cols_i, so
    # they concatenate on dim 1 into [batch, tot_dim - offset, *shape].
    lw = [linear.lw]
    got = offset + linear.lw.shape[1]
    while got < linear.tot_dim:
        nxt = self._dynamic_window(C=C, node=node, offset=got)
        # The windows tile the columns exactly, so an empty one means the loop
        # cannot finish rather than that it is done.
        assert nxt.lw is not None and nxt.lw.shape[1] > 0, (
            f'{node}: the window at offset {got} of {linear.tot_dim} is empty')
        lw.append(nxt.lw)
        got += nxt.lw.shape[1]
    assert got == linear.tot_dim, (
        f'{node}: the windows covered {got} columns, not {linear.tot_dim}')
    lw = torch.cat(lw, dim=1)
    # Only the first window carries the bias; the rest were zeroed.
    out = LinearBound(
        lw, linear.lb, lw, linear.lb, lower=linear.lower, upper=linear.upper,
        offset=offset, tot_dim=linear.tot_dim)
    out.dynamic_variable_ranges = linear.dynamic_variable_ranges
    return out


def _dynamic_window(self: 'BoundedModule', C=None, node:'Bound'=None,
                    concretize=False, offset=0, variable_values=None,
                    traversal_cache=None):
    """One window of forward_general_dynamic, or the whole of a concretization.

    Carries at most `forward_max_dim` columns, those at
    `[offset, offset + forward_max_dim)` of the propagation's layout.
    """
    max_dim = 0 if variable_values is not None else self.bound_opts['forward_max_dim']

    if C is None:
        if hasattr(node, 'linear'):
            assert not concretize

            linear = node.linear
            if offset == 0:
                if linear.lw is None:
                    return linear
                elif linear.lw.shape[1] <= max_dim:
                    return linear
            if linear.lw is not None and variable_values is not None:
                # This bound's columns are the global ones it reaches, so its
                # value at `variable_values` is lb + lw . variable_values over exactly those.
                #
                #   linear.lw  [batch, width, *shape]
                #   variable_values      [batch, n_gens]        n_gens >= width
                #   reshaped   [batch, width, 1...]   one 1 per axis of *shape
                #   product    [batch, width, *shape] -> sum(1) -> [batch, *shape]
                #
                # The trailing 1s are what make the column axis line up with
                # dim 1 instead of broadcasting onto the last axis of *shape.
                width = linear.lw.shape[1]
                lb = linear.lb + (linear.lw * variable_values[:, :width].reshape(
                    *variable_values.shape[:1], width,
                    *([1] * (linear.lw.dim() - 2)))).sum(1)
                # No columns left to carry: every one of them is now in lb.
                # Kept as an empty [batch, 0, *shape] rather than None so
                # downstream operators still see a coefficient tensor.
                lw = linear.lw[:, :0]
                tot_dim = linear.tot_dim
            elif linear.lw is not None:
                lw = linear.lw[:, offset:offset+max_dim]
                tot_dim = linear.tot_dim
                if offset == 0:
                    lb = linear.lb
                else:
                    lb = torch.zeros_like(linear.lb)
            else:
                lw = None
                tot_dim = 0
                lb = linear.lb
            out = LinearBound(
                lw, lb, lw, lb, offset=offset, tot_dim=tot_dim)
            out.dynamic_variable_ranges = linear.dynamic_variable_ranges
            out.variable_values = variable_values
            return out

        # Fixed operands contribute their bias in the first coefficient window.
        if not node.perturbed:
            if not node.is_lower_bound_current():
                node.lower = node.upper = self.get_forward_value(node)
            if concretize:
                return node.lower, node.upper
            else:
                if offset > 0:
                    lb = torch.zeros_like(node.lower)
                else:
                    lb = node.lower
                node.linear = LinearBound(None, lb, None, lb, node.lower, node.upper)
                return node.linear

    if offset == 0:
        logger.debug(f'forward_general_dynamic: node={node}')

    # One traversal must not visit a shared ancestor once per path that reaches
    # it. On a chain there is one path and this is bookkeeping; on a residual
    # net the paths double at every block, so without it the early layers are
    # recomputed exponentially often -- measured at 530 evaluations of a single
    # ReLU on a tinyimagenet ResNet. Nothing a node depends on changes within
    # one traversal, so remembering the answer is just not asking twice.
    if traversal_cache is None:
        traversal_cache = {}
    inp = []
    for l_pre in node.inputs:
        if l_pre.name not in traversal_cache:
            traversal_cache[l_pre.name] = self._dynamic_window(
                node=l_pre, offset=offset, variable_values=variable_values,
                traversal_cache=traversal_cache)
        linear_inp = traversal_cache[l_pre.name]
        linear_inp.lower = l_pre.lower
        linear_inp.upper = l_pre.upper
        inp.append(linear_inp)
    node._start = '_forward'
    if (C is not None and isinstance(node, BoundLinear) and
            not node.is_input_perturbed(1) and not node.is_input_perturbed(2)):
        linear = node.bound_dynamic_forward(
            *inp, C=C, max_dim=max_dim, offset=offset)
        C_merged = True
    else:
        linear = node.bound_dynamic_forward(
            *inp, max_dim=max_dim, offset=offset)
        C_merged = False
    # One book serves the whole propagation, so the columns an operator
    # returns mean what its inputs' did.
    linear.dynamic_variable_ranges = next(
        (b.dynamic_variable_ranges for b in inp if b.dynamic_variable_ranges is not None), None)
    linear.variable_values = variable_values
    if offset > 0:
        linear.lb = linear.ub = torch.zeros_like(linear.lb)

    lw, lb, tot_dim = linear.lw, linear.lb, linear.tot_dim
    #logger.debug(f'forward_general_dynamic: node={node}, w_size={lw.shape[1]}, tot_dim={tot_dim}')

    if C is not None and not C_merged:
        # FIXME use bound_forward of BoundLinear
        C = C.to(lw)                 # as BoundLinear.bound_dynamic_forward does
        lw = torch.matmul(lw, C.transpose(-1, -2))
        lb = torch.matmul(lb.unsqueeze(1), C.transpose(-1, -2)).squeeze(1)

    if concretize:
        lower = upper = lb
        if lw is not None:
            batch_size = lw.shape[0]
            assert (lw.ndim > 1)
            if lw.shape[1] > 0:
                # every generator is on [-1, 1], so this is the row's L1 norm:
                #   lw       [batch, n_cols, *shape]
                #   reshape  [batch, n_cols, numel]
                #   sum(1)   [batch, numel] -> view -> [batch, *shape]
                reach = lw.reshape(
                    batch_size, lw.shape[1], -1).abs().sum(1).view(lb.shape)
                lower, upper = lower - reach, upper + reach
            offset_next = offset + max_dim
            more = offset_next < tot_dim
        else:
            more = False

        if C is None and offset == 0 and not more:
            node.linear = linear

        if more:
            if lw is not None and lw.shape[1] > 0:
                del reach
                del lw
                del linear
            del inp
            # The next window traverses the graph again and builds its own
            # cache, so release this one rather than holding every node's
            # coefficients for the rest of the recursion.
            traversal_cache.clear()
            # TODO make it non-recursive
            lower_next, upper_next = self._dynamic_window(
                C, node, concretize=True, offset=offset_next)
            lower = lower + lower_next
            upper = upper + upper_next

        # Adding separately concretized coefficient chunks can produce NaN
        # even when each chunk has defined endpoints.
        if offset == 0:
            node.check_bound_invariants(
                lower, upper, context='dynamic forward concretization')

        if C is None:
            node.lower, node.upper = lower, upper

        return lower, upper
    else:
        return linear


def _zonotope_root(self: 'BoundedModule'):
    """The perturbed root's bound, which is what says there is a zonotope here.

    Only init_forward's dynamic branch lays out a column layout, and
    compute_bounds rebuilds the roots' bounds on every call, so carrying one is
    exactly "the state this module holds is a dynamic-forward pass". An
    ordinary forward leaves LinearBounds too, but a pair of one-sided lines --
    read as a zonotope that is the lower line alone, which need not contain the
    network. Every reader asks here first.
    """
    root = next((r for r in self.roots()
                 if getattr(r, 'perturbation', None) is not None), None)
    linear = getattr(root, 'linear', None)
    if linear is None or linear.dynamic_variable_ranges is None:
        raise RuntimeError(
            'there is no zonotope to read: the last bound this module computed '
            'was not a dynamic-forward propagation')
    return linear


def drop_dynamic_linear(self: 'BoundedModule', keep=()):
    """Forget every cached linear bound except the ones in `keep`.

    A cached bound answers from the layout and the relaxation it was computed
    under, and records neither, so it has to go whenever either changes: a new
    propagation lays the columns out again, and imposing a line relaxes the
    same node differently.
    """
    keep = {id(bound) for bound in keep}
    for node in self.nodes():
        if hasattr(node, 'linear') and id(node.linear) not in keep:
            delattr(node, 'linear')


def zonotope_variable_ranges(self: 'BoundedModule'):
    """Which node owns which coefficient columns, as `name -> (start, count)`.

    The propagation's own record, so a caller laying its columns out beside the
    pass reads the order the pass actually used rather than re-deriving one.
    """
    return self._zonotope_root().dynamic_variable_ranges


def zonotope_rows(self: 'BoundedModule', node, select=None):
    """The zonotope a dynamic-forward pass computed for `node`.

    Returns `(rows, centre)` for the affine form below, over the flattened
    output elements, or over just the ones `select` picks out:

        rows    [numel, W]          [len(select), W]      with `select`
        centre  [numel]             [len(select)]         with `select`

    where `numel = prod(node.output_shape[1:])` and `W` is the node's own
    column prefix -- the end of the last range it reaches, `<= n_gens`. Column
    `j < W` is column `j` of every other node's rows; columns `>= W` are zero
    for this node and are not returned.

        row . e + centre,      every variable in [-1, 1]

    Nothing is reconstructed: a node's LinearBound already is its zonotope, so
    this only reshapes it.

    Pass a list of nodes to read several from one propagation: returns
    `{name: (rows, centre)}`, takes `select` as `{name: idx}`, and leaves no
    bound cached. Put the deepest node first so the rest are read off it.
    """
    if isinstance(node, (list, tuple)):
        picks = select or {}
        try:
            return {n.name: zonotope_rows(self, n, picks.get(n.name))
                    for n in node}
        finally:
            self.drop_dynamic_linear()

    root = self._zonotope_root()
    linear = getattr(node, 'linear', None)
    if linear is None or linear.lw is None:
        linear = node.linear = self.forward_general_dynamic(node=node)
    assert linear.lw is not None, (
        f'{node.name} has no coefficients: it does not depend on the input')
    if linear.dynamic_variable_ranges is not root.dynamic_variable_ranges:
        raise RuntimeError(
            f'{node.name} carries coefficients from a propagation this module '
            f'is no longer holding: they are numbered in a column space that '
            f'no longer exists')
    if linear.lw.shape[0] != 1:
        raise ValueError(
            f'{node.name}: a zonotope describes one input box, but the pass is '
            f'carrying {linear.lw.shape[0]}; run it on one box at a time')
    # linear.lw is [1, n_gens, *shape]; drop the batch axis of size 1, flatten
    # the value axes, and transpose so a row is one output element:
    #   [1, n_gens, *shape] -> [n_gens, numel] -> [numel, n_gens]
    lw = linear.lw.reshape(linear.lw.shape[1], -1).t()          # [numel, n_gens]
    lb = linear.lb.reshape(-1)                                  # [numel]
    if select is not None:
        # a caller usually wants a handful of neurons out of a layer, and the
        # row matrix is the expensive part
        lw, lb = lw[select], lb[select]     # [len(select), n_gens], [len(select)]
    return lw, lb


def zonotope_bound(self: 'BoundedModule', node: 'Bound', C):
    """`node`'s zonotope concretized under `C`, without building its rows.

    The dual of zonotope_rows over the same zonotope. Rows cost a column per
    variable, which is the right price when a caller wants most of them; this
    wants the single row `C . G`, which is what one reverse pass gives.

    Supplying the variables as values makes the pass's bias channel the relaxed
    network at this node, so the row is `d(C.out)/de` and the bound is
    `C.out(0) - |row|_1`. It stays differentiable in whatever the relaxation
    depends on, which is what lets a caller search over slopes without
    materializing G.

    `C` is read as `[n_spec, numel]` (a `[numel]` vector counts as one spec)
    and the result is `[n_spec]`, one lower bound per row of `C`.
    """
    root = self._zonotope_root()
    if root.lb.shape[0] != 1:
        raise ValueError(
            f'{node.name}: a zonotope describes one input box, but the pass is '
            f'carrying {root.lb.shape[0]}; run it on one box at a time')
    n_gens = max(start + count
                 for start, count in root.dynamic_variable_ranges.values())
    # [1, n_gens], all zeros: the centre of the box every variable lives on.
    # The pass is affine in e, so where the gradient is taken does not matter;
    # 0 is the point lb already means.
    e = torch.zeros(root.lb.shape[0], n_gens, dtype=root.lb.dtype,
                    device=root.lb.device, requires_grad=True)
    # The lines may have moved since the last read, so no cached bound may
    # answer for them.
    self.drop_dynamic_linear(keep=(root,))
    # With e supplied as values the pass carries no columns, so lb alone is
    # the relaxed network at this node: [1, *shape] -> [numel].
    out = self.forward_general_dynamic(
        node=node, variable_values=e).lb.reshape(-1)
    C = C.reshape(-1, out.shape[0]).to(out)             # [n_spec, numel]
    centres = C.matmul(out)                             # [n_spec]
    # d(centres[k])/de is spec k's row of C.G, shaped like e. One reverse pass
    # gives one such row, so ask for all of them at once rather than walking
    # the specs; [n_spec] seeds -> [n_spec, *e.shape] rows.
    rows, = torch.autograd.grad(
        centres, e, grad_outputs=torch.eye(
            centres.shape[0], dtype=out.dtype, device=out.device),
        is_grads_batched=True, create_graph=True)       # [n_spec, 1, n_gens]
    # every variable is on [-1, 1], so the reach is the row's L1 norm
    return centres - rows.flatten(1).abs().sum(-1)      # [n_spec]


def clean_memory(self: 'BoundedModule', node):
    """ Remove linear bounds that are no longer needed. """
    # TODO add an option to retain these bounds
    for inp in node.inputs:
        if hasattr(inp, 'linear') and inp.linear is not None:
            clean = True
            for out in inp.output_name:
                out_node = self[out]
                if not (hasattr(out_node, 'linear') and out_node.linear is not None):
                    clean = False
            if clean:
                if isinstance(inp.linear, tuple):
                    for item in inp.linear:
                        del item
                delattr(inp, 'linear')


def forward_refinement(self: 'BoundedModule', node):
    """ Refine forward bounds with backward bound propagation
    (only refine unstable positions). """
    unstable_size_before = torch.logical_and(node.lower < 0, node.upper > 0).sum()
    if unstable_size_before == 0:
        return
    unstable_idx, unstable_size = self.get_unstable_locations(
        node.lower, node.upper, conv=isinstance(node, BoundConv))
    logger.debug(f'Forward refinement for {node}')
    batch_size = node.lower.shape[0]
    ret = self.batched_backward(
        node, C=None, unstable_idx=unstable_idx, batch_size=batch_size)
    self.restore_sparse_bounds(
        node, unstable_idx, unstable_size, node.lower, node.upper,
        new_lower=ret[0], new_upper=ret[1])
    unstable_size_after = torch.logical_and(node.lower < 0, node.upper > 0).sum()
    logger.debug(f'  Unstable neurons: {unstable_size_before} -> {unstable_size_after}')
    # TODO also update linear bounds?


def init_forward(self: 'BoundedModule', roots, dim_in):
    if dim_in == 0:
        raise ValueError("At least one node should have a specified perturbation")
    if self.dynamic:
        # The layout below is rebuilt here, so bounds laid out over the previous
        # one have to go. compute_bounds(x=...) already cleared them;
        # compute_bounds(x=None) reuses the input and does not.
        self.drop_dynamic_linear(keep=[root.linear for root in roots
                                       if hasattr(root, 'linear')])
    prev_dim_in = 0
    # Assumption: roots[0] is the input node which implies batch_size
    batch_size = self.get_forward_value(roots[0]).shape[0]
    for i in range(len(roots)):
        if hasattr(roots[i], 'perturbation') and roots[i].perturbation is not None:
            shape = roots[i].linear.lw.shape
            if self.dynamic:
                if shape[1] != dim_in:
                    raise NotImplementedError('Dynamic forward bound is not supported yet when there are multiple perturbed inputs.')
                ptb = roots[i].perturbation
                if (type(ptb) != PerturbationLpNorm or ptb.norm < np.inf
                        or ptb.x_L is None or ptb.x_U is None):
                    raise NotImplementedError(
                        'For dynamic forward bounds, only Linf (box) perturbations are supported, and x_L and x_U must be explicitly provided.')
                x_L = (ptb.x_L_sparse.view(batch_size, -1) if ptb.sparse
                       else ptb.x_L.view(batch_size, -1))
                x_U = (ptb.x_U_sparse.view(batch_size, -1) if ptb.sparse
                       else ptb.x_U.view(batch_size, -1))
                # Put the input's variables on [-1, 1], where a relaxation's
                # already live, so one statement holds everywhere: the bias is
                # the value at e = 0 and a column is a deviation from it. Left
                # alone the input's columns carry the box, centre included, and
                # every reader has to know to add that centre back.
                #   lw          [batch, dim_in, *shape]
                #   x_L, x_U     [batch, dim_in]
                #   spread       one 1 per axis of *shape, so the box lines up
                #                with dim 1 rather than the last value axis
                # centre: lb + sum over dim_in of lw * midpoint -> [batch, *shape]
                # radii:  lw scaled column-wise -> [batch, dim_in, *shape]
                lw = roots[i].linear.lw
                spread = [1] * (lw.dim() - 2)
                roots[i].linear.lb = roots[i].linear.ub = (
                    roots[i].linear.lb
                    + (lw * ((x_U + x_L) / 2).reshape(
                        batch_size, -1, *spread)).sum(1))
                roots[i].linear.lw = roots[i].linear.uw = lw * (
                    (x_U - x_L) / 2).reshape(batch_size, -1, *spread)

                # Which node owns which coefficient columns, seeded with the
                # input's own range so a node takes its range after it. It
                # rides on these bounds, so it lasts exactly one propagation.
                roots[i].linear.dynamic_variable_ranges = {
                    roots[i].name: (0, dim_in)}
            else:
                lw = torch.zeros(shape[0], dim_in, *shape[2:]).to(roots[i].linear.lw)
                lw[:, prev_dim_in:(prev_dim_in+shape[1])] = roots[i].linear.lw
                if roots[i].linear.lw.data_ptr() == roots[i].linear.uw.data_ptr():
                    uw = lw
                else:
                    uw = torch.zeros(shape[0], dim_in, *shape[2:]).to(roots[i].linear.uw)
                    uw[:, prev_dim_in:(prev_dim_in+shape[1])] = roots[i].linear.uw
                roots[i].linear.lw = lw
                roots[i].linear.uw = uw
            if i >= self.num_global_inputs:
                roots[i].forward_value = roots[i].forward_value.unsqueeze(0).repeat(
                    *([batch_size] + [1] * self.forward_value.ndim))
            prev_dim_in += shape[1]
        else:
            b = fv = roots[i].forward_value
            shape = fv.shape
            if roots[i].batch_dim != -1:
                w = torch.zeros(
                    shape[0], dim_in, *shape[1:], dtype=fv.dtype,
                    device=self.device)
                warnings.warn(f'Creating a LinearBound with zero weights with shape {w.shape}')
            else:
                w = None
            roots[i].linear = LinearBound(w, b, w, b, b, b)
            roots[i].lower = roots[i].upper = b
