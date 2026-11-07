"""Build a shared-slope zonotope generator state for Domain Clipper BaB.

The zonotope is the one a dynamic-forward pass already propagates: BoundRelu
relaxes an unstable neuron as one line plus a symmetric band and opens a
coefficient column for it, and BoundedModule.zonotope_rows reads the result off
any node. So nothing here builds a zonotope. What is left is the part that is
the clipper's own: choosing the lines its query wants, and laying the rows out
the way its BaB and dual kernel expect.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from auto_LiRPA import BoundedTensor
from auto_LiRPA.operators.relu import BoundRelu
from auto_LiRPA.perturbations import PerturbationLpNorm


@contextmanager
def _borrowed_relaxation(relus):
    """Restore each ReLU's alpha and opt_stage on exit, whatever the block did.

    The clipper imposes its slopes under the '_forward' alpha key on
    alpha-CROWN's own module, which is handed to beta-CROWN when the clipper
    does not close the property -- so an alpha it left replaced would be gone
    from under the solver that comes next.
    """
    saved = [(r.alpha, r.opt_stage) for r in relus]
    try:
        yield
    finally:
        for r, (alpha, opt_stage) in zip(relus, saved):
            r.alpha, r.opt_stage = alpha, opt_stage


def _triangle_slopes(relus, interm):
    """Install the band-minimising line u/(u-l) on every ReLU as an optimizable
    alpha, and return ``{name: that tensor}`` for the optimizer to step.

    The line is alpha, the parameter auto_LiRPA already optimizes for the
    two-line forward relaxation, so the operator reads it where it always does.
    alpha covers the whole layer because that is its shape -- a stable neuron
    is held at its exact line whatever alpha says.

      triangle  [batch=1, *node_shape]
      alpha     [alpha_size=2, spec=1, batch=1, *node_shape], alpha's layout;
                the spec axis is 1 because the search wants one slope per
                neuron across every row of the state, and the operator reads
                alpha[0, 0].
    """
    slopes = {}
    for r in relus:
        lo, hi = interm[r.inputs[0].name]
        triangle = (hi / (hi - lo).clamp_min(1e-12)).clamp(0., 1.)
        alpha = triangle.unsqueeze(0).expand(
            r.alpha_size, 1, *triangle.shape).clone().requires_grad_(True)
        r.alpha = {'_forward': alpha}
        r.opt_start()
        slopes[r.name] = alpha
    return slopes


def build_gen_state(net, ret, x, device, dtype=torch.float64,
                    opt_slope_qw=None, opt_iters=20, opt_lr=0.03,
                    bidx=0, verbose=True):
    """
    Build the zonotope generator state. Returns the state dict.

    Everything in the returned state is over one input box and one flat column
    space of `n_gens = n_input + (unstable ReLU neurons)` columns, each on
    [-1, 1]:

        output_gen_coeffs  [n_out, n_gens]   the output rows, G
        output_center      [n_out]           the output at e = 0
        e_lb, e_hi         [n_gens]          the box the columns live on
        unstable_list      one dict per unstable neuron, its pre-activation row
                           stored sparse as (row_indices, row_values)
        relu_error_col     (relu name, flat neuron index) -> its column

    Raises NotImplementedError if the graph has an operator dynamic forward has
    no construction for; the caller treats that as "no gen state".
    """
    bm = net.net                                 # the auto_LiRPA BoundedModule
    lower_bounds = ret['lower_bounds']           # dict layer_name -> pre-activation lower bd
    upper_bounds = ret['upper_bounds']

    # The pass puts the input's generators on [-1, 1] itself, so there is no box
    # arithmetic to do here. `bidx` selects ONE input box from a (possibly
    # batched) spec -- the gen state is built for a single box, and the caller
    # must ensure any other batch elements share it.
    ptb = x.ptb
    x_L = ptb.x_L[bidx:bidx + 1]
    x_U = ptb.x_U[bidx:bidx + 1]
    if not (torch.equal(ptb.x_L, x_L.expand_as(ptb.x_L))
            and torch.equal(ptb.x_U, x_U.expand_as(ptb.x_U))):
        raise NotImplementedError(
            f'the gen state describes the single box at bidx={bidx}, but the '
            f'batch holds {ptb.x_L.shape[0]} different ones')
    compute_dtype = x_L.dtype                    # the dtype the pass runs in
    n_input = int(x_L[0].numel())
    x1 = BoundedTensor((x_L + x_U) / 2.0, PerturbationLpNorm(
        eps=ptb.eps, norm=ptb.norm, x_L=x_L, x_U=x_U))

    final_name = net.final_name
    final_node = bm[final_name]
    n_out = int(np.prod([int(d) for d in final_node.output_shape[1:]]))
    relu_names = [r.name for r in bm.relus]

    # A row is an equality only because every relaxation gap on its path owns a
    # column. ReLU is the only inexact operator dynamic forward has a
    # construction for, so that holds today, but nothing enforces it.
    #
    # Only the upstream case matters. Downstream of every ReLU a gap reaches
    # the output row alone, which the bound minimizes, so it is merely loose.
    # Upstream of one it reaches that neuron's pre-activation row, which a
    # split reads as an equality: the active child imposes `row(e) + c_in >= 0`
    # where a real point only gives `z_true >= 0` and `row(e) + c_in <=
    # z_true`, so real points get cut. Refuse that; the caller falls through to
    # beta-CROWN, which is sound on its own.
    preact_ancestors, stack = set(), [r.inputs[0] for r in bm.relus]
    while stack:
        node = stack.pop()
        if node.name not in preact_ancestors:
            preact_ancestors.add(node.name)
            stack.extend(node.inputs)
    # `node.used` became membership in bm.bound_ancestors upstream (#847).
    unrepresented = [name for name in sorted(preact_ancestors)
                     if name not in set(relu_names)
                     and bm[name].perturbed and bm[name] in bm.bound_ancestors
                     and getattr(bm[name], 'requires_input_bounds', None)]
    if unrepresented:
        raise NotImplementedError(
            f'{unrepresented} relax upstream of a ReLU without a generator '
            f'column, so a split would read their gap as part of an equality')

    # The pre-activation bounds every relaxation is read against. Frozen, so the
    # pass uses alpha-CROWN's rather than recomputing looser ones from its own
    # zonotope -- and so the slope search below moves the lines and nothing else.
    interm = {
        r.inputs[0].name: (
            lower_bounds[r.inputs[0].name][bidx:bidx + 1].to(
                device=device, dtype=compute_dtype),
            upper_bounds[r.inputs[0].name][bidx:bidx + 1].to(
                device=device, dtype=compute_dtype),
        )
        for r in bm.relus
    }

    # Per ReLU: which neurons are unstable, and the slope to relax them with,
    # started at the band-minimizing u/(u-l). Only the unstable ones get a
    # parameter: a stable neuron is exact at slope 1 or 0, and moving it off
    # that would only open a band where there is none.
    geo = {}
    for r in bm.relus:
        # lo, hi   [numel]        the layer's pre-activation bounds, flattened
        # u_idx    [n_unstable]   flat indices of the unstable neurons, ascending
        #                         -- the same order the pass hands out columns in
        lo, hi = (t.flatten() for t in interm[r.inputs[0].name])
        geo[r.name] = dict(
            lo=lo.to(dtype), hi=hi.to(dtype),
            u_idx=((lo < 0) & (hi > 0)).nonzero(as_tuple=True)[0],
            shape=[int(d) for d in r.output_shape[1:]])

    def forward_pass():
        """One dynamic-forward pass, propagated to the final node."""
        return bm.compute_bounds(
            x=(x1,), method='dynamic-forward', final_node_name=final_name,
            interm_bounds=interm, bound_upper=False)[0]

    # ---- Phase 1 (optional): per-neuron slope search = zonotope CROWN.
    # Concretizing the pass IS the worst query's root bound, so this maximizes
    # the pass's own output over the lines -- which alpha-CROWN cannot do, its
    # alpha being per start node.
    searching = (opt_slope_qw is not None and opt_iters > 0
                 and any(geo[r.name]['u_idx'].numel() for r in bm.relus))

    # The lines go in through alpha, which belongs to the caller; restore it
    # whatever happens. The cached pass is dropped by the reader that makes it.
    with _borrowed_relaxation(bm.relus):
        if searching:
            spec_output_coeffs = torch.as_tensor(
                opt_slope_qw, device=device, dtype=compute_dtype)
            if spec_output_coeffs.dim() == 1:
                spec_output_coeffs = spec_output_coeffs.unsqueeze(0)
            params = list(_triangle_slopes(bm.relus, interm).values())
            forward_pass()                    # prime the module for zonotope_bound
            opt = torch.optim.Adam(params, lr=opt_lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opt_iters)
            best_bound, best_slopes = -float('inf'), None
            for it in range(opt_iters):
                opt.zero_grad()
                bound = bm.zonotope_bound(final_node, spec_output_coeffs).min()
                bound_value = float(bound.detach())
                if bound_value > best_bound:
                    best_bound = bound_value
                    best_slopes = [p.detach().clone() for p in params]
                (-bound).backward()
                opt.step()
                sched.step()
                for r in bm.relus:
                    r.clip_alpha()
                if verbose and (it % 10 == 0 or it == opt_iters - 1):
                    print(f'[gen-state] slope-opt it={it} '
                          f'root_bound={bound_value:.4f} best={best_bound:.4f}')
            if best_slopes is not None:
                with torch.no_grad():
                    for p, best in zip(params, best_slopes):
                        p.copy_(best)

        # ---- Phase 2: the output row and every unstable pre-activation row,
        # from one pass with the chosen lines. zonotope_rows leaves nothing
        # cached, so this is the last read; the variable layout is captured
        # first, since dropping the pass drops it too.
        preacts = {r.name: bm[r.inputs[0].name] for r in bm.relus}
        with torch.no_grad():
            forward_pass()
            variable_ranges = dict(bm.zonotope_variable_ranges())
            rows = bm.zonotope_rows(
                [final_node, *preacts.values()],
                select={n.name: geo[name]['u_idx']
                        for name, n in preacts.items()})
        output_gen_coeffs, output_center = (t.to(dtype) for t in rows[final_name])
        n_gens = int(output_gen_coeffs.shape[1])

        # ---- Phase 3: lay the pre-activation rows out the way BaB expects.
        # A pre-activation's zonotope cannot contain its own ReLU's error
        # column -- the pass opens that downstream of it -- so unlike the
        # backward direction there is nothing to exclude by hand.
        relu_error_col = {}
        unstable_list = []
        for r in bm.relus:
            g = geo[r.name]
            u_idx = g['u_idx']
            col, count = variable_ranges[r.name]
            # Who is unstable, though, both sides have to agree on: they read it
            # off the same frozen bounds, and the rows would land in the wrong
            # columns if they ever disagreed.
            assert count == int(u_idx.numel()), (
                f'{r.name}: the pass opened {count} generators for '
                f'{int(u_idx.numel())} unstable neurons')
            if not count:
                continue
            # The same function the pass relaxed with, called on the same
            # arguments, read the same way -- not a second way of writing the
            # relaxation.
            # line, half   [1, *shape] -> slope, band [numel], indexed by the
            #              flat neuron index j, so they cover the whole layer
            line, half = BoundRelu.zonotope_line(
                *interm[r.inputs[0].name], r.imposed_zonotope_line())
            slope, band = line.reshape(-1).to(dtype), half.reshape(-1).to(dtype)
            # layer_rows [n_unstable, n_gens], preact_center [n_unstable] --
            # both indexed by position i within u_idx, not by neuron index j
            layer_rows, preact_center = rows[r.inputs[0].name]
            preact_center = preact_center.to(dtype)
            rows_cpu = layer_rows.to(dtype).cpu()
            # Column offsets from Python ints: indexing these element-wise costs one
            # GPU sync per unstable neuron, which would be most of this loop.
            slope_cpu, band_cpu = slope.tolist(), band.tolist()
            lo_cpu, hi_cpu = g['lo'].tolist(), g['hi'].tolist()
            center_cpu = preact_center.tolist()
            # i indexes u_idx (rows, centres, columns); j is the flat neuron
            # index into the layer (slope, band, lo, hi). Keeping the two apart
            # is the whole bookkeeping of this loop.
            for i, j in enumerate(u_idx.tolist()):
                # nz  [nnz]  columns this pre-activation actually reaches; the
                #            row is stored sparse because a neuron early in the
                #            network reaches almost none of n_gens
                nz = rows_cpu[i].nonzero(as_tuple=True)[0]
                relu_error_col[(r.name, j)] = col + i
                unstable_list.append(dict(
                    layer_idx=r.name, neuron_idx=int(j),
                    preact_center=center_cpu[i], lo=lo_cpu[j], hi=hi_cpu[j],
                    relu_slope=slope_cpu[j], band_half_width=band_cpu[j],
                    relu_error_col=col + i,
                    row_indices=nz.numpy().astype(np.int64),
                    row_values=rows_cpu[i, nz].numpy().astype(np.float64)))
        assert sum(count for _, count in variable_ranges.values()) == n_gens, (
            variable_ranges, n_gens)

        if verbose:
            print(f'[Clipper Gen-State] n_input={n_input} '
                  f'n_unstable={n_gens - n_input} n_gens={n_gens}')

        return dict(
            n_gens=n_gens, n_input=n_input, n_out=n_out,
            unstable_list=unstable_list,
            output_gen_coeffs=output_gen_coeffs, output_center=output_center,
            e_lb=torch.full((n_gens,), -1.0, device=device, dtype=dtype),
            e_hi=torch.full((n_gens,), 1.0, device=device, dtype=dtype),
            relu_error_col=relu_error_col, relu_order=relu_names,
        )


def root_bound(state, spec_output_coeffs, spec_bias):
    """
    Concretized root LP bound min_{e in box} (spec_output_coeffs @ output_gen_coeffs) e + (spec_output_coeffs @ output_center + spec_bias).
    spec_output_coeffs: [n_out] or [n_spec, n_out].  Returns the box minimum (scalar / [n_spec]).
    """
    single = torch.as_tensor(spec_output_coeffs).dim() == 1
    lower_bound = root_decomposition(state, spec_output_coeffs, spec_bias)['lower_bound']
    return lower_bound[0] if single else lower_bound


def root_decomposition(state, spec_output_coeffs, spec_bias):
    """
    Root-level coefficient diagnostics for one or more output specs.
    """
    output_gen_coeffs = state['output_gen_coeffs']
    output_center = state['output_center']
    e_hi = state['e_hi']
    spec_output_coeffs = torch.as_tensor(spec_output_coeffs, device=output_gen_coeffs.device, dtype=output_gen_coeffs.dtype)
    if spec_output_coeffs.dim() == 1:
        spec_output_coeffs = spec_output_coeffs.unsqueeze(0)
    # [n_spec, n_out] @ [n_out, n_gens] -> [n_spec, n_gens]
    spec_gen_coeffs = spec_output_coeffs @ output_gen_coeffs                      # [n_spec, n_gens]
    spec_center = spec_output_coeffs @ output_center + torch.as_tensor(spec_bias, device=output_gen_coeffs.device, dtype=output_gen_coeffs.dtype)
    # [n_spec, n_gens] * [n_gens] -> [n_spec, n_gens]; e_hi is the box radius,
    # so this is each column's worst-case contribution to the spec
    abs_contribs = spec_gen_coeffs.abs() * e_hi
    n_input = int(state['n_input'])
    unstable_error_cols = [int(u['relu_error_col']) for u in state['unstable_list']]
    input_abs_contrib = abs_contribs[:, :n_input].sum(-1)
    if unstable_error_cols:
        unstable_error_cols_t = torch.tensor(
            unstable_error_cols, device=output_gen_coeffs.device, dtype=torch.long)
        unstable_error_coeffs = spec_gen_coeffs.index_select(1, unstable_error_cols_t)
        relu_error_abs_contrib = abs_contribs.index_select(1, unstable_error_cols_t).sum(-1)
    else:
        unstable_error_coeffs = spec_gen_coeffs.new_zeros((spec_gen_coeffs.shape[0], 0))
        relu_error_abs_contrib = spec_gen_coeffs.new_zeros((spec_gen_coeffs.shape[0],))
    total_abs_contrib = abs_contribs.sum(-1)
    lower_bound = spec_center - total_abs_contrib
    return {
        'spec_center': spec_center,
        'spec_gen_coeffs': spec_gen_coeffs,
        'abs_contribs': abs_contribs,
        'input_abs_contrib': input_abs_contrib,
        'relu_error_abs_contrib': relu_error_abs_contrib,
        'total_abs_contrib': total_abs_contrib,
        'lower_bound': lower_bound,
        'unstable_error_cols': unstable_error_cols,
        'unstable_error_coeffs': unstable_error_coeffs,
    }


def intermediate_bounds_table(state):
    """
    Return one diagnostic row per unstable ReLU neuron.
    """
    rows = []
    for unstable_neuron in state['unstable_list']:
        rows.append({
            'layer_idx': unstable_neuron['layer_idx'],
            'neuron_idx': int(unstable_neuron['neuron_idx']),
            'lo': float(unstable_neuron['lo']),
            'hi': float(unstable_neuron['hi']),
            'preact_center': float(unstable_neuron['preact_center']),
            'relu_slope': float(unstable_neuron['relu_slope']),
            'band_half_width': float(unstable_neuron['band_half_width']),
            'relu_error_col': int(unstable_neuron['relu_error_col']),
        })
    return rows
