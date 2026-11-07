"""A fast per-node lower bound for the zonotope substitution BaB.

Everything here is a pure function of the gen state (from
``clipper_gen_state.build_gen_state``) and the branch decisions so far -- no
class, no module state beyond a ``torch.compile`` cache.

Notation used throughout:

    n_gens   number of zonotope columns  = n_input + (unstable ReLU neurons)
    n_out    output dimension of the specification
    S        number of unstable neurons, i.e. splits available
    D        depth: how many neurons a branch has fixed so far (D <= S)
    B        number of open domains solved together in one BaB round
    Kb       line-search bucket count

The spec is one objective row ``c`` over the network output; this kernel bounds
``c . output`` and BaB drives every branch's bound above ``rhs``.

The zonotope is ``centre + G e`` with every ``e_i in [-1, 1]``. Fixing a ReLU
to inactive (side 0) or active (side 1) *substitutes* its error column: the
column is dropped from ``G`` and its objective weight is folded, via the
pre-activation's own row, back into the surviving columns and the constant.
``build_problem`` precomputes those substitution constants; ``node_bound_warm_fast``
runs a small dual ascent over the two half-space multipliers the split adds.
"""
from __future__ import annotations
import time
import numpy as np
import torch

_TOL = 1e-9
_FAST_KERNEL_CACHE = {}


def _fast_line_search_logbucket(rc, da, width, slope0, Kb, upper=None):
    """Step length for one dual-ascent iteration, per domain, in parallel.

    The dual objective along the ascent direction is concave and piecewise
    linear in the step ``eta``: as ``eta`` grows, each coordinate of
    ``rc + eta * da`` crosses zero at a breakpoint ``eta_j = -rc_j / da_j``, and
    each crossing drops the slope by ``|da_j| * width``. This finds the ``eta``
    at which the running slope first reaches ``slope0`` (the slope at eta = 0),
    bucketing the breakpoints in log space so the whole thing is a scatter-add
    plus a cumsum rather than a sort.

    Shapes (n = n_gens):
        rc, da        [B, n]
        width         scalar (2.0: the width of [-1, 1])
        slope0        [B]
        upper         [B] or None   -- a cap on eta (feasibility bound)
    Returns:
        eta           [B]
        emax          [B]           -- the cap actually used
        crossed       [B, Kb]       -- which buckets the slope crossed in
    """
    # positive[b, j]: coordinate j has a breakpoint at some eta > 0
    positive = ((da > 0) & (rc < 0)) | ((da < 0) & (rc > 0))
    eta_j = torch.where(positive, -rc / da, torch.full_like(rc, float('inf')))   # [B, n]
    max_eta = torch.where(positive, eta_j, torch.full_like(eta_j, -float('inf'))).max(-1).values  # [B]
    has_breakpoint = positive.any(-1)                                            # [B]
    if upper is None:
        emax = torch.where(has_breakpoint, max_eta, torch.zeros(rc.shape[0], device=rc.device, dtype=rc.dtype))
    else:
        finite_upper = torch.isfinite(upper)
        fallback = torch.where(finite_upper, upper, torch.zeros_like(upper))
        emax = torch.where(has_breakpoint, max_eta, fallback)
        emax = torch.where(finite_upper, torch.minimum(emax, upper), emax)       # [B]
    # active[b, j]: breakpoint j lies in (0, emax]
    active = positive & (eta_j <= emax.unsqueeze(-1)) & (eta_j > 0)              # [B, n]
    has_active = active.any(-1)
    emin = torch.where(has_active, torch.where(active, eta_j, torch.full_like(eta_j, float('inf'))).min(-1).values,
                       torch.zeros(rc.shape[0], device=rc.device, dtype=rc.dtype))  # [B]
    # log-space buckets spanning [emin, emax]; Kb of them, width lw in log units
    safe_emin = emin.clamp_min(1e-30)
    safe_emax = torch.maximum(emax, safe_emin).clamp_min(1e-30)
    le = torch.log(safe_emin)                                                    # [B]
    lw = (torch.log(safe_emax) - le).clamp_min(1e-30) / Kb                       # [B]
    eta_for_index = torch.where(active, eta_j.clamp_min(1e-30), safe_emin.unsqueeze(-1))
    bidx = (((torch.log(eta_for_index) - le.unsqueeze(-1)) / lw.unsqueeze(-1)).floor().long()).clamp(0, Kb - 1)  # [B, n]
    bidx = torch.where(active, bidx, torch.full_like(bidx, Kb))                  # inactive -> spill bucket Kb
    decr = torch.where(active, da.abs() * width, torch.zeros_like(da))          # [B, n] slope drop per breakpoint
    buckets = torch.zeros(rc.shape[0], Kb + 1, device=rc.device, dtype=rc.dtype)  # [B, Kb+1]
    buckets.scatter_add_(1, bidx, decr)
    # crossed[b, k]: cumulative slope drop through bucket k has reached slope0
    crossed = buckets[:, :Kb].cumsum(-1) >= slope0.unsqueeze(-1)                 # [B, Kb]
    eta = torch.exp(le + (crossed.float().argmax(-1).to(rc.dtype) + 0.5) * lw)   # midpoint of the first crossed bucket
    eta = torch.where(crossed.any(-1), eta, emax)                               # never crossed -> go to the cap
    return torch.where(slope0 > _TOL, eta, torch.zeros_like(eta)).clamp_min(0.0), emax, crossed


def node_bound_warm_fast(F: dict, sides: torch.Tensor, phase_dual: torch.Tensor,
                         box_dual: torch.Tensor, K: int, Kb: int = 256):
    """Dual lower bound for a batch of branch nodes, warm-started from the parent.

    The node's primal is: minimise ``c0_path + path_row . e`` over ``e in
    [-1, 1]^n_gens`` subject to, for each fixed split, one *phase* half-space
    (the ReLU sits on the chosen side) and one *box* half-space (the
    pre-activation stays in the range the relaxation is valid over). Its dual
    maximises over two non-negative multiplier vectors, one entry per split.
    Dual feasibility is free, so any multipliers give a valid lower bound; this
    does ``K`` steps of ascent to tighten it.

    Shapes (D = number of fixed splits, n = n_gens):
        F['a_g']         [D, n]   pre-activation generator rows of the fixed splits
        F['a_l1']        [D]      row L1 norms = max |a_g . e| over the box
        F['d_base']      [n]      objective row with the fixed splits' own
                                  error columns already zeroed
        F['c0']          scalar   objective constant (spec centre)
        F['ratio_off/on'][D]      substitution scale for side 0 / side 1
        F['c0_off/on']   [D]      substitution constant shift for side 0 / side 1
        F['c_in']        [D]      pre-activation centre of each fixed split
        F['z_lo/z_hi']   [D]      the box half-space bound for side 0 / side 1
        sides            [B, D]   0 (inactive) or 1 (active) per split per domain
        phase_dual       [B, D]   phase multipliers, warm-started
        box_dual         [B, D]   box multipliers, warm-started
    Returns:
        best             [B]      the tightened lower bound (+inf if infeasible)
        phase_dual       [B, D]   multipliers after ascent (fed to the children)
        box_dual         [B, D]
    """
    a_g = F['a_g']                                                              # [D, n]
    B = sides.shape[0]
    D = sides.shape[1]
    dtype = a_g.dtype
    # sign = +1 on the inactive side, -1 on the active side: it flips the
    # pre-activation half-space between "z <= 0" and "z >= 0".
    sign = torch.where(sides == 0, torch.ones_like(sides), -torch.ones_like(sides)).to(dtype)  # [B, D]
    # rho: the substitution scale for the chosen side of each split. path_base
    # below adds rho_k * a_g[k] to the objective row -- the eliminated error
    # column, re-expressed through the pre-activation.
    rho = torch.where(sides == 0, F['ratio_off'], F['ratio_on'])                # [B, D]
    c0_path = F['c0'] + torch.where(sides == 0, F['c0_off'], F['c0_on']).sum(1)  # [B]
    cin, zlo, zhi = F['c_in'], F['z_lo'], F['z_hi']                             # [D] each
    # b0: phase half-space constant. Inactive wants z <= 0 i.e. a_g.e + c_in <= 0;
    # active wants z >= 0 i.e. -(a_g.e) - c_in <= 0, hence -c_in.
    b0 = torch.where(sides == 0, cin.expand(B, D), (-cin).expand(B, D))         # [B, D]
    # b1: box half-space constant, keeping the pre-activation where the band is
    # valid -- one-sided per chosen side.
    b1 = torch.where(sides == 0, (-zlo).expand(B, D), (-zhi).expand(B, D))      # [B, D]
    a_l1 = F['a_l1']                                                            # [D]
    # A half-space is unsatisfiable over the box iff its constant exceeds the
    # row's reach; such a node is infeasible and certified (+inf) below.
    infeasible = ((b0 > a_l1.unsqueeze(0) + 1e-6)
                  | (b1 > a_l1.unsqueeze(0) + 1e-6)).any(1)                     # [B]

    # path_base: objective generator row after substituting every fixed split.
    path_base = F['d_base'].unsqueeze(0) + rho @ a_g                           # [B, n] = [B, D] @ [D, n]
    # rc: reduced-cost row, objective plus the dual-weighted phase/box rows.
    rc = path_base + (sign * (phase_dual - box_dual)) @ a_g                    # [B, n]
    # dual bound = c0_path + min_e (rc . e) + dual-constant terms
    best = c0_path - rc.abs().sum(-1) + (phase_dual * b0 + box_dual * b1).sum(-1)  # [B]
    # also take the zero-multiplier bound (just the substituted objective)
    best = torch.maximum(best, c0_path - path_base.abs().sum(-1))              # [B]

    for _ in range(K):
        # box minimiser of the current reduced cost
        x = torch.where(rc < 0, torch.ones_like(rc), -torch.ones_like(rc))     # [B, n]
        p = x @ a_g.t()                                                        # [B, D] -- fixed rows at x
        s0 = b0 + sign * p                                                     # [B, D] phase slack at x
        s1 = b1 - sign * p                                                     # [B, D] box slack at x
        # complementarity: an inactive multiplier with a satisfied constraint
        # contributes no ascent direction.
        sp0 = torch.where((phase_dual <= _TOL) & (s0 < 0), torch.zeros_like(s0), s0)  # [B, D]
        sp1 = torch.where((box_dual <= _TOL) & (s1 < 0), torch.zeros_like(s1), s1)    # [B, D]
        slope0 = (sp0 * s0 + sp1 * s1).sum(-1)                                 # [B] ascent slope at eta = 0
        da = (sign * (sp0 - sp1)) @ a_g                                        # [B, n] how rc moves with eta
        # step before a multiplier would go negative
        w0 = torch.where(sp0 < 0, -phase_dual / sp0, torch.full_like(phase_dual, float('inf')))  # [B, D]
        w1 = torch.where(sp1 < 0, -box_dual / sp1, torch.full_like(box_dual, float('inf')))      # [B, D]
        ebnd = torch.minimum(w0.min(-1).values, w1.min(-1).values)            # [B]
        eta, _, _ = _fast_line_search_logbucket(rc, da, 2.0, slope0, Kb, upper=ebnd)  # [B]
        eta = torch.minimum(eta, torch.where(torch.isfinite(ebnd), ebnd, eta)).clamp_min(0.0)
        phase_dual = (phase_dual + eta.unsqueeze(-1) * sp0).clamp_min(0.0)     # [B, D]
        box_dual = (box_dual + eta.unsqueeze(-1) * sp1).clamp_min(0.0)         # [B, D]
        rc = path_base + (sign * (phase_dual - box_dual)) @ a_g               # [B, n]
        best = torch.maximum(
            best,
            c0_path - rc.abs().sum(-1) + (phase_dual * b0 + box_dual * b1).sum(-1))  # [B]

    best = torch.where(infeasible, torch.full_like(best, float('inf')), best)  # [B]
    return best, phase_dual, box_dual


def _fast_problem_at_depth(problem, depth):
    """Slice ``problem`` (split-indexed, in branch order) to its first ``depth``
    splits and shape it the way ``node_bound_warm_fast`` reads it.

    ``d_base`` is the objective row with those splits' error columns zeroed --
    they are substituted, not free. Shapes as in ``node_bound_warm_fast``'s F.
    """
    d_base = problem['spec_gen_coeffs'].clone()                                # [n_gens]
    d_base[problem['relu_error_col'][:depth]] = 0.0
    a_g = problem['preact_gen_coeffs'][:depth].contiguous()                    # [depth, n_gens]
    return {
        'a_g': a_g,
        'a_l1': a_g.abs().sum(-1),                                            # [depth]
        'd_base': d_base,
        'c0': torch.as_tensor(problem['spec_center'], device=a_g.device, dtype=a_g.dtype),
        'ratio_off': problem['subst_gen_scale'][:depth, 0].contiguous(),      # [depth]
        'ratio_on': problem['subst_gen_scale'][:depth, 1].contiguous(),
        'c0_off': problem['subst_const_shift'][:depth, 0].contiguous(),
        'c0_on': problem['subst_const_shift'][:depth, 1].contiguous(),
        'c_in': problem['preact_center'][:depth].contiguous(),
        'z_lo': problem['inactive_lower_shift'][:depth].contiguous(),
        'z_hi': problem['active_upper_shift'][:depth].contiguous(),
    }


def _get_warm_fast_kernel(K, Kb):
    """A ``torch.compile``d ``node_bound_warm_fast`` bound to (K, Kb), memoised
    so the compile happens once per configuration."""
    key = ('warm', int(K), int(Kb))
    if key not in _FAST_KERNEL_CACHE:
        def kern(F, sides, phase_dual, box_dual):
            return node_bound_warm_fast(F, sides, phase_dual, box_dual, K=K, Kb=Kb)
        _FAST_KERNEL_CACHE[key] = torch.compile(kern, dynamic=True)
    return _FAST_KERNEL_CACHE[key]


def build_problem(state, spec_output_coeffs, spec_bias, order, device, dtype=torch.float32):
    """Precompute the per-split substitution constants, in branch/scoring order.

    ``spec_output_coeffs`` is one objective row over the output (shape [n_out]);
    ``spec_gen_coeffs = spec_output_coeffs @ output_gen_coeffs`` is that row over
    the zonotope columns ([n_gens]), ``spec_center`` its constant (scalar).

    For each split the returned tensors are indexed by position in ``order``
    (branch order), length ``S = len(order)``:

        preact_gen_coeffs     [S, n_gens]  dense pre-activation row, own col zeroed
        subst_gen_scale       [S, 2]       [inactive, active] scale on that row
        subst_const_shift     [S, 2]       [inactive, active] constant shift
        preact_center         [S]          pre-activation value at e = 0
        inactive_lower_shift  [S]          box bound for the inactive side
        active_upper_shift    [S]          box bound for the active side
        relu_error_col        [S]          global column the split eliminates

    Substitution derivation. The band-form relaxation of an unstable ReLU is
    ``relu(z) = slope*z + band + band*e_new`` with ``e_new in [-1, 1]``. Fixing
    the side pins ``relu(z)`` and solves for ``e_new`` in terms of the
    pre-activation ``z = preact_center + preact_row . e``:

        inactive (relu(z) = 0):  e_new = -(slope/band)*z - 1
        active   (relu(z) = z):  e_new =  ((1-slope)/band)*z - 1

    The objective's term ``eliminated_error_coeff * e_new`` (with
    ``eliminated_error_coeff = spec_gen_coeffs[relu_error_col]``) then becomes a
    scale on ``preact_row`` plus a constant -- exactly the two arrays above.
    With no band the ReLU is exact and only the constant ``-eliminated_error_coeff``
    survives.
    """
    n_gens = state['n_gens']
    unstable_neurons = state['unstable_list']
    spec_output_coeffs_t = torch.as_tensor(
        spec_output_coeffs, device=device, dtype=state['output_gen_coeffs'].dtype)
    spec_gen_coeffs = (spec_output_coeffs_t @ state['output_gen_coeffs']).to(dtype)   # [n_gens]
    spec_center = float(spec_output_coeffs_t @ state['output_center'] + spec_bias)    # scalar
    spec_gen_coeffs_np = spec_gen_coeffs.detach().cpu().numpy()

    num_splits = len(order)                                                     # S
    preact_gen_coeffs = torch.zeros(num_splits, n_gens, device=device, dtype=dtype)   # [S, n_gens]
    subst_gen_scale = torch.empty(num_splits, 2, device=device, dtype=dtype)          # [S, 2]
    subst_const_shift = torch.empty(num_splits, 2, device=device, dtype=dtype)        # [S, 2]
    preact_center = torch.empty(num_splits, device=device, dtype=dtype)               # [S]
    inactive_lower_shift = torch.empty(num_splits, device=device, dtype=dtype)        # [S]
    active_upper_shift = torch.empty(num_splits, device=device, dtype=dtype)          # [S]
    relu_error_col = torch.empty(num_splits, device=device, dtype=torch.long)         # [S]

    for split_index, unstable_index in enumerate(order):
        unstable_neuron = unstable_neurons[unstable_index]
        relu_slope = float(unstable_neuron['relu_slope'])
        band_half_width = float(unstable_neuron['band_half_width'])
        preact_center_value = float(unstable_neuron['preact_center'])
        relu_error_col_idx = int(unstable_neuron['relu_error_col'])
        row_indices = np.asarray(unstable_neuron['row_indices'], np.int64)     # [nnz]
        row_values = np.asarray(unstable_neuron['row_values'], np.float64)     # [nnz]

        # scatter the sparse pre-activation row into a dense [n_gens] vector;
        # its own error column carries nothing (it is what gets substituted).
        preact_gen_row = torch.zeros(n_gens, dtype=dtype)                      # [n_gens]
        preact_gen_row[torch.from_numpy(row_indices)] = torch.from_numpy(row_values).to(dtype)
        preact_gen_row[relu_error_col_idx] = 0.0
        preact_gen_coeffs[split_index] = preact_gen_row.to(device)

        has_band = band_half_width > 1e-12
        inv_band_half_width = 1.0 / (band_half_width if has_band else 1.0)
        eliminated_error_coeff = float(spec_gen_coeffs_np[relu_error_col_idx])  # objective weight on e_new

        subst_gen_scale[split_index, 0] = (
            -(relu_slope * inv_band_half_width) * eliminated_error_coeff
        ) if has_band else 0.0
        subst_const_shift[split_index, 0] = (
            -(1 + relu_slope * preact_center_value * inv_band_half_width)
            * eliminated_error_coeff
        ) if has_band else -eliminated_error_coeff

        subst_gen_scale[split_index, 1] = (
            ((1 - relu_slope) * inv_band_half_width) * eliminated_error_coeff
        ) if has_band else 0.0
        subst_const_shift[split_index, 1] = (
            ((1 - relu_slope) * preact_center_value * inv_band_half_width - 1)
            * eliminated_error_coeff
        ) if has_band else -eliminated_error_coeff

        preact_center[split_index] = preact_center_value
        # 2*band/slope = -l when the inactive branch is the binding one, so this
        # is the pre-activation past which the band stops enclosing relu on the
        # inactive side; the kernel uses it as the box half-space bound b1.
        inactive_lower_shift[split_index] = (
            2 * band_half_width / max(relu_slope, 1e-30) + preact_center_value
        ) if (has_band and relu_slope > 1e-12) else 1e9
        active_upper_shift[split_index] = (
            2 * band_half_width / max(1 - relu_slope, 1e-30) - preact_center_value
        ) if (has_band and (1 - relu_slope) > 1e-12) else 1e9
        relu_error_col[split_index] = relu_error_col_idx

    return dict(
        n_gens=n_gens,
        preact_gen_coeffs=preact_gen_coeffs,
        spec_gen_coeffs=spec_gen_coeffs,
        spec_center=spec_center,
        subst_gen_scale=subst_gen_scale,
        subst_const_shift=subst_const_shift,
        preact_center=preact_center,
        inactive_lower_shift=inactive_lower_shift,
        active_upper_shift=active_upper_shift,
        relu_error_col=relu_error_col,
        S=num_splits,
        dtype=dtype,
    )


def reorder_problem(problem, order):
    """Permute the split-indexed tensors into ``order`` in place (the dense
    generator rows are just index-selected, not rebuilt).

    ``order`` is a length-S permutation; every [S, ...] tensor is reindexed on
    dim 0.
    """
    order_t = torch.as_tensor(order, device=problem['preact_gen_coeffs'].device, dtype=torch.long)
    for key in (
            'preact_gen_coeffs', 'subst_gen_scale', 'subst_const_shift',
            'preact_center', 'inactive_lower_shift', 'active_upper_shift',
            'relu_error_col'):
        problem[key] = problem[key].index_select(0, order_t)
    return problem


def box_halfspace_delta_lb_scores(state, spec_output_coeffs, spec_bias):
    """Branch-ordering heuristic: for each unstable neuron, how much would
    splitting it raise the worst child's lower bound, estimated with the box
    plus that neuron's single half-space.

    ``spec_output_coeffs`` is the objective row over the output ([n_out]);
    ``d_obj_base`` is it over the columns ([n_gens]), ``baseline_lb`` the
    unsplit box bound. Returns one score per unstable neuron ([n_unstable]),
    already relative to ``baseline_lb`` so larger = more promising.
    """
    spec_output_coeffs_t = torch.as_tensor(
        spec_output_coeffs, device=state['output_gen_coeffs'].device,
        dtype=state['output_gen_coeffs'].dtype)
    d_obj_base = (spec_output_coeffs_t @ state['output_gen_coeffs']).detach().cpu().double().numpy().reshape(-1)  # [n_gens]
    c0_obj_base = float((spec_output_coeffs_t @ state['output_center']).detach().cpu().item() + spec_bias)        # scalar
    base_abs = np.abs(d_obj_base)                                              # [n_gens]
    base_abs_sum = float(base_abs.sum())
    baseline_lb = c0_obj_base - base_abs_sum

    def _lmin_sparse(supp, a_supp, d_supp, d_enc, enc, c0, beta):
        """Minimise ``c0 + d.e`` over ``e in [-1,1]`` subject to one half-space
        ``a.e + beta <= 0``, working only on the sparse support ``supp``.

        supp, a_supp, d_supp, d_enc all align with the neuron's nonzero
        pre-activation columns; ``enc`` is the eliminated column. The unconstrained
        optimum is the box vertex; if it violates the half-space, walk the
        active constraint by increasing multiplier -- the objective is piecewise
        linear in it, with a breakpoint each time a coordinate saturates.
        Returns the constrained minimum (``+inf`` if the half-space is
        infeasible over the box).
        """
        off_abs = (
            base_abs_sum
            - float(base_abs[supp].sum())
            - float(base_abs[enc])
            + float(np.abs(d_supp).sum())
            + abs(float(d_enc))
        )
        g0 = c0 - off_abs                       # box optimum with the substituted row
        eff = np.where(d_supp != 0, np.sign(d_supp), np.sign(a_supp))
        gp0 = -beta - float((a_supp * eff).sum())   # half-space slack at that vertex
        if gp0 <= 0:
            return g0                          # vertex already feasible
        valid = (a_supp != 0) & (d_supp != 0)
        if not valid.any():
            return float('inf')
        lam_v = -d_supp[valid] / a_supp[valid]     # per-coord breakpoint multiplier
        a_v = a_supp[valid]
        pos = lam_v > 0
        if not pos.any():
            return float('inf')
        lam_pos = lam_v[pos]                       # [m] breakpoints > 0
        decr = 2.0 * np.abs(a_v[pos])             # [m] slope drop at each breakpoint
        order = np.argsort(lam_pos)
        lam_s = lam_pos[order]                     # sorted breakpoints
        decr_s = decr[order]
        cum_before = np.concatenate(([0.0], np.cumsum(decr_s)))   # [m+1]
        gaps = np.diff(np.concatenate(([0.0], lam_s)))            # [m] spacing between breakpoints
        slope_j = gp0 - cum_before[:-1]                           # [m] slack slope on each segment
        contrib = np.cumsum(slope_j * gaps)                       # [m] accumulated slack change
        m = int(np.searchsorted(cum_before, gp0, side='left'))    # segment where slack hits 0
        if m > lam_s.size:
            return float('inf')
        return g0 + float(contrib[m - 1])

    scores = np.zeros(len(state['unstable_list']), dtype=np.float64)           # [n_unstable]
    for idx, u in enumerate(state['unstable_list']):
        lam = float(u['relu_slope'])
        mu = float(u['band_half_width'])
        c_in = float(u['preact_center'])
        row_idx = np.asarray(u['row_indices'], dtype=np.int64)                 # [nnz]
        row_val = np.asarray(u['row_values'], dtype=np.float64)               # [nnz]
        e_new_col = int(u['relu_error_col'])
        if row_idx.size == 0:
            continue
        if mu > 1e-12:
            ew_k = float(d_obj_base[e_new_col]) / mu       # objective weight on e_new, per band unit
        else:
            continue
        db_supp = d_obj_base[row_idx]                                          # [nnz]
        d_enc = float(d_obj_base[e_new_col]) - ew_k * mu

        # inactive child: substitute e_new, constraint is preact <= 0 (beta = -c_in)
        d_off_supp = db_supp - ew_k * lam * row_val
        c0_off = c0_obj_base - ew_k * (lam * c_in + mu)
        lb_off = _lmin_sparse(row_idx, row_val, d_off_supp, d_enc, e_new_col,
                              c0_off, -c_in)

        # active child: preact >= 0, so the half-space row flips sign (beta = c_in)
        d_on_supp = db_supp + ew_k * (1.0 - lam) * row_val
        c0_on = c0_obj_base + ew_k * ((1.0 - lam) * c_in - mu)
        lb_on = _lmin_sparse(row_idx, -row_val, d_on_supp, d_enc, e_new_col,
                             c0_on, c_in)
        scores[idx] = float(min(lb_off, lb_on)) - baseline_lb
    return scores


def _print_bab_round(info):
    print(f"BaB round {info['round']}")
    print(f"batch: {info['batch']}")
    print('splitting decisions:')
    print(f"split level 0: {info['decision']}")
    print(f"best_l after optimization: {info['best_l']}")
    print(f"beta sum per layer: {info['beta_per_layer']}")
    print(f"alpha/beta optimization time: {info['bound_time']}")
    print('pruning_in_iteration open status: True')
    print(f"ratio of positive domain = {info['n_pruned']} / {info['batch']} = {info['ratio']}")
    print(f"pruning-in-iteration extra time: {info['prune_time']}")
    print('Time: ' + '    '.join(f'{k} {v:.4f}' for k, v in info['round_time'].items()))
    print('Accumulated time: ' + '    '.join(f'{k} {v:.4f}' for k, v in info['acc_time'].items()))
    print(f"max lb {info['max_lb']} min lb {info['min_lb']}")
    print(f"Number of Verified Splits: {info['n_pruned']} of {info['batch']}")
    print('Current worst splitting domains lb-rhs (depth):')
    print(', '.join(f'{b:.5f} ({d})' for b, d in info['worst_domains']))
    print(f"Current (lb-rhs): {info['worst']}")
    print(f"{info['nodes']} domains visited")
    print(f"Length of domains: {info['n_open']}")
    print(f"Cumulative time: {info['cum_time']}")
    print()


def verify_spec(state, spec_output_coeffs, spec_bias, device, time_limit=None, K=1, chunk=16384,
                frontier_cap=None, bab_dtype=torch.float32, verbose=True,
                fast_Kb=256):
    """Zonotope substitution BaB for one specification row.

    ``spec_output_coeffs`` [n_out], ``spec_bias`` scalar; the goal is
    ``spec_output_coeffs . output + spec_bias > 0`` for the whole box. Returns
    ``(verdict, info)`` with verdict in {'safe', 'unknown'}.

    A branch is a path of 0/1 decisions over the first ``depth`` splits (in the
    score order). Each round solves every open domain's dual bound, prunes the
    ones already above 0, and expands the rest by appending both sides of split
    ``depth``.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    # branch order: use the fast box+halfspace worst-child score.
    unordered_problem = build_problem(
        state, spec_output_coeffs, spec_bias, list(range(len(state['unstable_list']))),
        device, bab_dtype)
    scores = box_halfspace_delta_lb_scores(state, spec_output_coeffs, spec_bias)   # [S]
    order = np.argsort(-scores).tolist()                                          # best split first
    S = len(order)
    problem = reorder_problem(unordered_problem, order)

    # root box bound: concretise the objective row over [-1, 1]^n_gens
    box_minimizer = torch.where(
        problem['spec_gen_coeffs'] < 0,
        torch.ones_like(problem['spec_gen_coeffs']),
        -torch.ones_like(problem['spec_gen_coeffs']))                            # [n_gens]
    g_root = float((problem['spec_gen_coeffs'] * box_minimizer).sum()) + problem['spec_center']
    if verbose:
        print(f'initial alpha-zono substitution BaB: {S} unstable neurons, K={K}, '
              f'root (lb-rhs)={g_root:.4f}')
    if g_root > _TOL:
        return 'safe', dict(nodes=1, depth=0, peak=0, wall=0.0)
    if S == 0:
        # Nothing to split and the root did not clear: the root bound is all
        # this kernel can say.
        return 'unknown', dict(nodes=1, depth=0, peak=0, wall=0.0,
                               reason='no_splits')

    # per-split (in branch order) relu layer name / neuron, for the round log:
    # the neuron split that produced round d's batch is order[d-1].  layer_idx is the
    # relu's graph node name (e.g. '/input-72'), matching the native [/input-72, 28] style.
    unstable_list = state['unstable_list']
    split_layer = [str(unstable_list[order[k]]['layer_idx']) for k in range(S)]   # [S]
    split_neuron = [int(unstable_list[order[k]]['neuron_idx']) for k in range(S)]  # [S]
    uniq_layers = sorted(set(split_layer))
    layer_id = {name: i for i, name in enumerate(uniq_layers)}

    # torch.compile warm-up at a ladder of depths so the first real round is not
    # paying a compile.
    warm_fast_kernel = _get_warm_fast_kernel(K, fast_Kb)
    for warm_depth in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
        if warm_depth <= S:
            Fw = _fast_problem_at_depth(problem, warm_depth)
            zw = torch.zeros(8, warm_depth, device=device, dtype=bab_dtype)
            sw = torch.zeros(8, warm_depth, device=device, dtype=torch.long)
            warm_fast_kernel(Fw, sw, zw, zw.clone())
    if torch.cuda.is_available() and torch.device(device).type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    # frontier at depth 1: the two children of the first split.
    split_sides = torch.tensor([[0], [1]], device=device, dtype=torch.int8)      # [B, depth]
    phase_dual = torch.zeros(2, 1, device=device, dtype=bab_dtype)               # [B, depth]
    box_dual = torch.zeros(2, 1, device=device, dtype=bab_dtype)                 # [B, depth]
    nodes = 1; depth = 1; peak = 2
    round_no = 0
    acc = {'solve': 0.0, 'prune': 0.0, 'add': 0.0, 'bab_iteration': 0.0}
    while split_sides.shape[0] > 0 and depth <= S:
        if time_limit is not None and time.perf_counter() - t0 > time_limit:
            return 'unknown', dict(nodes=nodes, depth=depth, peak=peak,
                                   open=int(split_sides.shape[0]),
                                   reason='time_limit',
                                   wall=time.perf_counter() - t0)
        B = split_sides.shape[0]; nodes += B; peak = max(peak, B); round_no += 1
        best_bound = torch.empty(B, device=device, dtype=bab_dtype)             # [B]
        next_phase_dual = torch.empty(B, depth, device=device, dtype=bab_dtype)  # [B, depth]
        next_box_dual = torch.empty(B, depth, device=device, dtype=bab_dtype)    # [B, depth]
        t_solve = time.perf_counter()
        # solve the round in chunks; halve the chunk on OOM down to 128.
        i = 0; cs = chunk
        while i < B:
            s = min(cs, B - i)
            while True:
                try:
                    bound_chunk, phase_dual_chunk, box_dual_chunk = warm_fast_kernel(
                        _fast_problem_at_depth(problem, depth),
                        split_sides[i:i+s].to(torch.long).contiguous(),
                        phase_dual[i:i+s],
                        box_dual[i:i+s])
                    break
                except torch.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if s <= 128:
                        raise
                    s //= 2; cs = max(128, min(cs, s))
            best_bound[i:i+s] = bound_chunk
            next_phase_dual[i:i+s] = phase_dual_chunk
            next_box_dual[i:i+s] = box_dual_chunk
            i += s
        round_solve_time = time.perf_counter() - t_solve

        t_prune = time.perf_counter()
        # a domain is verified once its bound clears 0; keep only the rest.
        keep = best_bound <= _TOL                                              # [B]
        n_pruned = int((~keep).sum())
        open_split_sides = split_sides[keep]                                   # [n_open, depth]
        kept_phase_dual = next_phase_dual[keep]
        kept_box_dual = next_box_dual[keep]
        n_open = open_split_sides.shape[0]
        round_prune_time = time.perf_counter() - t_prune

        if frontier_cap and frontier_cap > 0 and n_open > 0 and depth < S and 2 * n_open > frontier_cap:
            return 'unknown', dict(nodes=nodes, depth=depth, peak=peak,
                                   open=int(2 * n_open), reason='frontier_cap',
                                   wall=time.perf_counter() - t0)

        # build next-round frontier now (so its cost shows in this round's 'add' time)
        t_add = time.perf_counter()
        will_expand = n_open > 0 and depth < S
        next_split_sides = next_phase = next_box = None
        if will_expand:
            # each surviving domain spawns two: append side 0 and side 1 of
            # split `depth`, carrying its warm-started multipliers.
            next_rows = 2 * n_open
            next_depth = depth + 1
            next_split_sides = torch.empty(next_rows, next_depth, device=device, dtype=torch.int8)   # [2*n_open, depth+1]
            next_phase = torch.empty(next_rows, next_depth, device=device, dtype=bab_dtype)
            next_box = torch.empty(next_rows, next_depth, device=device, dtype=bab_dtype)

            next_split_sides[:n_open, :depth] = open_split_sides
            next_split_sides[n_open:, :depth] = open_split_sides
            next_split_sides[:n_open, depth] = 0
            next_split_sides[n_open:, depth] = 1

            next_phase[:n_open, :depth] = kept_phase_dual
            next_phase[n_open:, :depth] = kept_phase_dual
            next_phase[:, depth] = 0

            next_box[:n_open, :depth] = kept_box_dual
            next_box[n_open:, :depth] = kept_box_dual
            next_box[:, depth] = 0
        round_add_time = time.perf_counter() - t_add

        cum = time.perf_counter() - t0
        acc['solve'] += round_solve_time; acc['prune'] += round_prune_time
        acc['add'] += round_add_time; acc['bab_iteration'] = cum

        if verbose:
            # beta sum per layer = dual mass (phase_dual+box_dual) grouped by relu layer
            beta_per_layer = [0.0 for _ in uniq_layers]
            for col, layer_name in enumerate(split_layer[:depth]):
                beta_per_layer[layer_id[layer_name]] += float(
                    next_phase_dual[:, col].abs().sum()
                    + next_box_dual[:, col].abs().sum())
            beta_per_layer = [round(v, 4) for v in beta_per_layer]
            finite = best_bound[torch.isfinite(best_bound)]
            max_lb = float(finite.max()) if finite.numel() else float('inf')
            best_l = float(finite.sum()) if finite.numel() else float('inf')
            open_bounds = best_bound[keep]
            k = min(20, n_open)
            worst_vals = (torch.topk(open_bounds, k, largest=False).values.tolist()
                          if k > 0 else [])
            # worst still-open (lb-rhs); at the closing round (none open) fall back
            # to the smallest now-verified bound so the line stays meaningful.
            worst = float(open_bounds.min()) if n_open else float(best_bound.min())
            _print_bab_round(dict(
                round=round_no, batch=B,
                decision=f"[layer {split_layer[depth-1]}, neuron {split_neuron[depth-1]}]"
                         f"  (uniform: all {B} domains, global branch order)",
                best_l=best_l, beta_per_layer=beta_per_layer,
                bound_time=round_solve_time, n_pruned=n_pruned, ratio=n_pruned / B,
                prune_time=round_prune_time,
                round_time={'solve': round_solve_time, 'prune': round_prune_time,
                            'add': round_add_time,
                            'bab_iteration': round_solve_time + round_prune_time + round_add_time},
                acc_time=acc,
                max_lb=max_lb, min_lb=float(best_bound.min()),
                worst_domains=[(v, depth) for v in worst_vals], worst=worst,
                nodes=nodes, n_open=n_open, cum_time=cum))

        if n_open == 0:
            return 'safe', dict(nodes=nodes, depth=depth, peak=peak,
                                wall=time.perf_counter() - t0)
        if depth >= S:
            return 'unknown', dict(nodes=nodes, depth=depth, peak=peak,
                                   open=n_open, reason='splits_exhausted',
                                   wall=time.perf_counter() - t0)
        split_sides, phase_dual, box_dual = next_split_sides, next_phase, next_box
        depth += 1
    # Every exit above returns; falling out of the loop means no round ran, and
    # that is never a proof.
    return 'unknown', dict(nodes=nodes, depth=depth, peak=peak,
                           reason='no_round', wall=time.perf_counter() - t0)
