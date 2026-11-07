"""The backward generator-state builder, frozen as a reference oracle.

This is `clipper_gen_state.build_gen_state` as it stood before the Domain
Clipper read its state off a dynamic-forward pass, kept verbatim so the forward
builder has a ruler to be measured against. It reconstructs each row from CROWN
traversals -- one per pre-activation, with `BoundRelu.bound_backward` patched at
class level to return the zonotope's line and to capture the traversal's A
matrix -- and it optimizes slopes on a hand-written forward evaluation of the
relaxed network. Given the same relaxation it must produce the same zonotope,
and `test_forward_gen_state_matches_the_backward_reference` holds it to 1e-9.

It is a measuring stick, not a fallback. Do not import it from production code
and do not tidy it: its value is that it is an independent implementation of
the same object, written before the one it checks. If it has to change to keep
passing, the interesting question is which of the two moved.
"""
from __future__ import annotations
import contextlib
import numpy as np
import torch
from auto_LiRPA.operators.relu import BoundRelu
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

def _classify(node):
    """
    auto_LiRPA op class name, e.g. 'BoundConv' / 'BoundRelu'.
    """
    return node.__class__.__name__

@contextlib.contextmanager
def _zono_relu_backward_patch(geo):
    """
    Temporarily install the zonotope relu's generator-side backward on BoundRelu.
    """
    cap = {}
    orig_backward = BoundRelu.bound_backward

    def _dense(A, out_shape):
        # the hook returns dense A, so A stays dense; densify defensively if Patches.
        return A if isinstance(A, torch.Tensor) else A.to_matrix(out_shape)

    def _zono_relu_backward(self, last_lA, last_uA, *args, start_node=None, **kw):
        last_lA = _dense(last_lA, geo[self.name]['out_shape']) if last_lA is not None else None
        last_uA = _dense(last_uA, geo[self.name]['out_shape']) if last_uA is not None else None
        if start_node is not None and last_lA is not None:
            cap[(start_node.name, self.name)] = last_lA.detach()
        cc = geo[self.name]['c'].to(last_lA.dtype if last_lA is not None else last_uA.dtype)
        cc = cc.view(1, 1, *geo[self.name]['shape'])
        return [(None if last_lA is None else last_lA * cc,
                 None if last_uA is None else last_uA * cc)], 0, 0

    BoundRelu.bound_backward = _zono_relu_backward
    try:
        yield cap
    finally:
        BoundRelu.bound_backward = orig_backward

def build_gen_state(net, ret, x, device, dtype=torch.float64,
                    opt_slope_qw=None, opt_iters=20, opt_lr=0.03, reuse_alpha=False,
                    bidx=0, verbose=True):
    """
    Build the zonotope generator state. Returns the state dict.
    """
    bm = net.net                                 # the auto_LiRPA BoundedModule
    lower_bounds = ret['lower_bounds']           # dict layer_name -> pre-activation lower bd
    upper_bounds = ret['upper_bounds']           # dict layer_name -> pre-activation upper bd

    # Input box -> center + radii.  We fold the radii into the input generators so
    # that e_input in [-1,1] maps to the real input box [x_L, x_U].  `bidx` selects
    # ONE input box from a (possibly batched) spec -- the gen-state is built for a
    # single box; the caller must ensure any other batch elements share this box.
    x_L = x.ptb.x_L[bidx:bidx + 1].to(device=device, dtype=dtype)
    x_U = x.ptb.x_U[bidx:bidx + 1].to(device=device, dtype=dtype)
    center0 = ((x_L + x_U) / 2.0)
    radii0 = ((x_U - x_L) / 2.0)
    in_shape = tuple(center0.shape)              # [1, C, H, W]
    n_input = int(center0.numel())

    # ---- extraction setup for auto_LiRPA CROWN backward ----
    final_name = net.final_name
    final_node = [n for n in bm.nodes() if n.name == final_name][0]
    n_out = int(np.prod([int(d) for d in final_node.output_shape[1:]]))
    relu_names = [r.name for r in bm.relus]
    compute_dtype = x.ptb.x_L.dtype
    # the incoming x may be batched (open OR-disjuncts sharing a box) -> build a batch-1 input.
    _ptb = x.ptb
    _xL1 = _ptb.x_L[bidx:bidx + 1]; _xU1 = _ptb.x_U[bidx:bidx + 1]
    x1 = BoundedTensor((_xL1 + _xU1) / 2.0,
                    PerturbationLpNorm(eps=_ptb.eps, norm=_ptb.norm, x_L=_xL1, x_U=_xU1))

    # ---- Phase 1: per-relu geometry (lower/upper, unstable mask) + slope params -------
    # Allocate one generator column per unstable neuron, after the n_input input
    # columns.  `relu_error_col[(relu, j)]` is neuron j's column index in the e vector.
    geo = {}                # relu_name -> dict(lo,hi,unstable,u_idx,active,n)
    relu_error_col = {}     # (relu_name, neuron_idx) -> generator column index
    col = n_input           # running column counter (input dims come first)
    relu_slope_param = {}   # relu_name -> trainable slope over its unstable neurons in [0,1]
    for r in bm.relus:
        pre = r.inputs[0].name
        # lower/upper = per-neuron pre-activation bounds l_i, u_i,
        # taken from alpha-CROWN's intermediate bounds (ret['lower/upper_bounds']).
        lo = lower_bounds[pre][bidx].flatten().to(device=device, dtype=dtype)
        hi = upper_bounds[pre][bidx].flatten().to(device=device, dtype=dtype)
        N = lo.numel()
        active = lo >= 0          # l_i >= 0  -> always-on, relu(z)=z (no relaxation)
        dead = hi <= 0            # u_i <= 0  -> always-off, relu(z)=0
        unstable = (~active) & (~dead)   # l_i < 0 < u_i  -> needs a band + e_new
        u_idx = unstable.nonzero(as_tuple=True)[0]
        denom = (hi - lo).clamp_min(1e-12)
        # classical zonotope slope c = u/(u-l)
        # the minimal-band initialization; later refined by slope-opt (Phase 2).
        slope = (hi / denom)[u_idx].clamp(0.0, 1.0)
        geo[r.name] = dict(lo=lo, hi=hi, active=active, unstable=unstable,
                           u_idx=u_idx, n=N,
                           shape=tuple(int(d) for d in r.output_shape[1:]),
                           out_shape=tuple(int(d) for d in r.output_shape))
        relu_slope_param[r.name] = slope.clone().detach().requires_grad_(True)
        for j in u_idx.tolist():
            relu_error_col[(r.name, j)] = col
            col += 1
    n_gens = col
    if verbose:
        print(f'[Clipper Gen-State] n_input={n_input} n_unstable={n_gens - n_input} '
              f'n_gens={n_gens}')

    inp_name = [n.name for n in bm.nodes() if _classify(n) == 'BoundInput'][0]

    def _relu_band_params(rname):
        """Per-neuron slope/band for relu `rname` from the current relu_slope_param.

        Returns (relu_slope_full [N], band_half_width_full [N], unstable_band_half_width [n_u]) where N is the
        layer width.  Differentiable w.r.t. relu_slope_param (used by slope-opt).
          * active (lo>=0): relu_slope=1, band_half_width=0      (out = z exactly)
          * dead   (hi<=0): relu_slope=0, band_half_width=0      (out = 0 exactly)
          * unstable      : relu_slope in (0,1), band_half_width = max((1-relu_slope)hi/2, -relu_slope*lo/2)
        """
        g = geo[rname]
        N = g['n']; u_idx = g['u_idx']
        lo = g['lo']; hi = g['hi']
        unstable_relu_slope = relu_slope_param[rname].clamp(1e-6, 1 - 1e-6)     # slope c in (0,1)
        relu_slope_full = torch.zeros(N, device=device, dtype=dtype)
        relu_slope_full = relu_slope_full + g['active'].to(dtype)        # active -> c=1, else 0
        relu_slope_full = relu_slope_full.index_put((u_idx,), unstable_relu_slope)     # unstable -> trainable c
        # Upper bias  dbar = max{(1-c)u, -c l}, and
        # band half-width  band_half_width = q = dbar/2 = max((1-c)u/2, -c l/2).
        hi_u = hi[u_idx]; lo_u = lo[u_idx]                 # u, l on unstable neurons
        unstable_band_half_width = torch.maximum((1 - unstable_relu_slope) * hi_u / 2.0, -unstable_relu_slope * lo_u / 2.0)
        band_half_width_full = torch.zeros(N, device=device, dtype=dtype)
        band_half_width_full = band_half_width_full.index_put((u_idx,), unstable_band_half_width)
        return relu_slope_full, band_half_width_full, unstable_band_half_width

    def _forward(e_in, e_new):
        """
        Differentiable zonotope forward; returns (out_flat, pre_act, band_half_width_cache).
        """
        # Seed leaf values: the perturbed input (radii folded in) + every constant
        # param/buffer (so node.forward can consume weights, biases, shapes, ...).
        vals = {}
        for node in bm.nodes():
            cls = _classify(node)
            if cls == 'BoundInput':
                vals[node.name] = (center0.flatten() + radii0.flatten() * e_in).reshape(in_shape)
            elif cls in ('BoundParams', 'BoundBuffers'):
                v = bm.get_forward_value(node)
                vals[node.name] = v.to(device=device, dtype=dtype) if v.is_floating_point() else v
        pre_act = {}
        band_half_width_cache = {}
        for node in bm.nodes():
            cls = _classify(node); nm = node.name
            if cls in ('BoundInput', 'BoundParams', 'BoundBuffers'):
                continue
            if cls == 'BoundRelu':
                # out = relu_slope*z + band_half_width*(1 + e_new).
                z = vals[node.inputs[0].name]
                zf = z.flatten()
                pre_act[nm] = zf                           # keep z for row seeding
                relu_slope_full, band_half_width_full, unstable_band_half_width = _relu_band_params(nm)
                band_half_width_cache[nm] = unstable_band_half_width
                a_center = relu_slope_full * zf + band_half_width_full         # relu_slope*z + band_half_width (e_new at 0)
                ui = geo[nm]['u_idx']
                if ui.numel() > 0:
                    # add the band_half_width*e_new generator term ONLY at unstable positions
                    contrib = torch.zeros_like(a_center).index_add(
                        0, ui, unstable_band_half_width * e_new[nm])
                    a_center = a_center + contrib
                vals[nm] = a_center.reshape(z.shape)
            else:
                vals[nm] = node.forward(*[vals[i.name] for i in node.inputs])
        return vals[net.final_name].flatten(), pre_act, band_half_width_cache

    def _make_leaves():
        e_in = torch.zeros(n_input, device=device, dtype=dtype, requires_grad=True)
        e_new = {r.name: torch.zeros(geo[r.name]['u_idx'].numel(), device=device,
                                     dtype=dtype, requires_grad=True)
                 for r in bm.relus}
        return e_in, e_new

    # ---- Phase 2 (optional): per-neuron slope optimization = zonotope CROWN
    # Maximize the spec's root lower bound over the per-neuron slopes relu_slope.
    if opt_slope_qw is not None and opt_iters > 0:
        # opt_slope_qw may be a single spec [n_out] or several queries [n_q, n_out].
        spec_output_coeffs = torch.as_tensor(opt_slope_qw, device=device, dtype=dtype)
        if spec_output_coeffs.dim() == 1:
            spec_output_coeffs = spec_output_coeffs.unsqueeze(0)                           # [n_q, n_out]
        params = [relu_slope_param[r.name] for r in bm.relus if relu_slope_param[r.name].numel() > 0]
        opt = torch.optim.Adam(params, lr=opt_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opt_iters)
        best_bound = -float('inf')
        best_relu_slope = None
        for it in range(opt_iters):
            opt.zero_grad()
            e_in, e_new = _make_leaves()
            out_val, _, _ = _forward(e_in, e_new)
            leaves = [e_in] + [e_new[r.name] for r in bm.relus]
            qbounds = []
            for q in range(spec_output_coeffs.shape[0]):
                scal = (spec_output_coeffs[q] * out_val).sum()
                d = torch.autograd.grad(scal, leaves, create_graph=True,
                                        retain_graph=True, allow_unused=True)
                absum = sum((g.abs().sum() for g in d if g is not None))
                qbounds.append(scal - absum)
            bound = torch.stack(qbounds).min()             # worst query
            (-bound).backward()                            # maximize the worst root
            opt.step()
            sched.step()
            with torch.no_grad():
                for p in params:
                    p.clamp_(1e-6, 1 - 1e-6)
                if float(bound) > best_bound:
                    best_bound = float(bound)
                    best_relu_slope = [p.detach().clone() for p in params]
            if verbose and (it % 10 == 0 or it == opt_iters - 1):
                print(f'[gen-state] slope-opt it={it} root_bound={float(bound):.4f} '
                      f'best={best_bound:.4f}')
        # restore best slopes
        if best_relu_slope is not None:
            with torch.no_grad():
                for p, b in zip(params, best_relu_slope):
                    p.copy_(b)

    # finalize the chosen per-neuron slope c and band (full [N] vectors) for extraction.
    with torch.no_grad():
        for r in bm.relus:
            slope_full, band_full, _ = _relu_band_params(r.name)
            geo[r.name]['c'] = slope_full.detach()
            geo[r.name]['band'] = band_full.detach()

    # ---- Phase 3: extract generators AND centers via auto_LiRPA's CROWN backward ----
    interm = {r.inputs[0].name: (lower_bounds[r.inputs[0].name][bidx:bidx + 1].to(device=device, dtype=compute_dtype),
                                 upper_bounds[r.inputs[0].name][bidx:bidx + 1].to(device=device, dtype=compute_dtype))
              for r in bm.relus}
    def _rows_from(cap, target_name, sel_idx, target_width, skip_relu=None):
        """
        Affine form of target[sel_idx] over e: returns (rows [m,n_gens], center [m]).
        """
        m = sel_idx.numel()
        C = torch.zeros(m, target_width, device=device, dtype=compute_dtype)
        C[torch.arange(m, device=device), sel_idx] = 1.0
        cap.clear()
        interm_use = {k2: v for k2, v in interm.items() if k2 != target_name}
        _, _, A = bm.compute_bounds(
            x=(x1,), method='backward', C=C.unsqueeze(0),
            final_node_name=target_name, interm_bounds=interm_use,
            return_A=True, needed_A_dict={target_name: {inp_name: None}})
        a_in = A[target_name][inp_name]['lA'].reshape(m, -1).to(dtype)         # [m, n_input]
        lbias = A[target_name][inp_name]['lbias'].reshape(m).to(dtype)         # [m]
        rows = torch.zeros(m, n_gens, device=device, dtype=dtype)
        rows[:, :n_input] = a_in * radii0.flatten()
        for rn in relu_names:
            if rn == skip_relu:
                continue
            key = (target_name, rn)
            if key not in cap:
                continue
            ui = geo[rn]['u_idx']
            if ui.numel() == 0:
                continue
            lastlA = cap[key].reshape(m, -1).to(dtype)                         # [m, width]
            cols = torch.tensor([relu_error_col[(rn, int(j))] for j in ui.tolist()],
                                device=device, dtype=torch.long)
            rows[:, cols] = lastlA[:, ui] * geo[rn]['band'][ui]
        center = (a_in * center0.flatten()).sum(1) + lbias + rows[:, n_input:].sum(1)
        return rows, center

    with _zono_relu_backward_patch(geo) as cap:
        output_gen_coeffs, output_center = _rows_from(
            cap, final_name, torch.arange(n_out, device=device), n_out)
        unstable_list = []
        for r in bm.relus:
            ui = geo[r.name]['u_idx']
            if ui.numel() == 0:
                continue
            # row of z_k does NOT depend on its own post-activation error -> skip self.
            rowG, preact_center = _rows_from(cap, r.inputs[0].name, ui, geo[r.name]['n'], skip_relu=r.name)
            lo = geo[r.name]['lo'][ui]; hi = geo[r.name]['hi'][ui]
            c = geo[r.name]['c']; band = geo[r.name]['band']
            rowG_cpu = rowG.detach().cpu()
            for i, j in enumerate(ui.tolist()):
                nz = rowG_cpu[i].nonzero(as_tuple=True)[0]
                unstable_list.append(dict(
                    layer_idx=r.name, neuron_idx=int(j),
                    preact_center=float(preact_center[i]), lo=float(lo[i]), hi=float(hi[i]),
                    relu_slope=float(c[j]), band_half_width=float(band[j]),
                    relu_error_col=relu_error_col[(r.name, int(j))],
                    row_indices=nz.numpy().astype(np.int64),
                    row_values=rowG_cpu[i, nz].numpy().astype(np.float64)))

    e_lb = torch.full((n_gens,), -1.0, device=device, dtype=dtype)
    e_hi = torch.full((n_gens,), 1.0, device=device, dtype=dtype)
    return dict(
        n_gens=n_gens, n_input=n_input, n_out=n_out,
        unstable_list=unstable_list,
        output_gen_coeffs=output_gen_coeffs.detach(), output_center=output_center.detach(),
        e_lb=e_lb, e_hi=e_hi,
        relu_error_col=relu_error_col, relu_order=relu_names,
    )
