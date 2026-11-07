"""
When `--clipper_only_bab_timeout` is enabled, general_bab()
calls `try_clipper_only_bab` BEFORE the beta-CROWN BaB loop:
    1. build the zonotope generator state (clipper_gen_state.build_gen_state):
    2. run the factored substitution-form dual BaB (clipper_dual_kernel.verify_spec)
    3. if every open spec certifies, the property is verified and we skip
       beta-CROWN; otherwise we fall through unchanged.
"""
from __future__ import annotations

import os
import time

import torch
import arguments
from clipper_gen_state import build_gen_state, root_bound
from clipper_dual_kernel import verify_spec


def _gen_state_bab(state, spec_output_coeffs, spec_bias, device, time_limit=None,
                   frontier_cap=None):
    """
    Run the clipper-only dual BaB for every open spec.
    Args:
        state: zonotope generator state.
        spec_output_coeffs: [n_open, n_out] spec coefficients over the network output.
        spec_bias: [n_open] spec biases (= -rhs).
        device: device to run the dual BaB on.
    Returns ('safe', info) only if ALL open specs certify; else ('unknown', info).
    """
    bab_args = arguments.Config["bab"]
    K = bab_args.get("clipper_only_K", 1)
    fast_Kb = bab_args.get("clipper_only_fast_Kb", 256)
    n_open = spec_output_coeffs.shape[0]
    infos = []
    for spec_idx in range(n_open):
        if time_limit is not None and time_limit <= 0:
            return 'unknown', {'specs': infos, 'failed': spec_idx, 'reason': 'time_limit'}
        verdict, info = verify_spec(
            state, spec_output_coeffs[spec_idx], float(spec_bias[spec_idx]), device,
            time_limit=time_limit, K=K, frontier_cap=frontier_cap,
            verbose=True, fast_Kb=fast_Kb)
        infos.append((spec_idx, verdict, info))
        print(f'[Domain Clipper] spec {spec_idx}: {verdict} nodes={info.get("nodes")} '
              f'depth={info.get("depth")} peak={info.get("peak")} '
              f'reason={info.get("reason", "done")} wall={info.get("wall", 0):.2f}s')
        if verdict != 'safe':
            return 'unknown', {'specs': infos, 'failed': spec_idx}
    return 'safe', {'specs': infos}


def _dump_graph(net, ret):
    """
    One-time structural dump of the BoundedModule graph.
    """
    print('==== Domain Clipper Graph Dump ====')
    bm = net.net
    relu_names = {r.name for r in bm.relus}
    lower_bounds = ret.get('lower_bounds', {})
    upper_bounds = ret.get('upper_bounds', {})
    for node in bm.nodes():
        cls = node.__class__.__name__
        ins = [i.name for i in node.inputs]
        osh = tuple(node.output_shape) if node.output_shape is not None else None
        extra = ''
        for attr in ('weight', 'bias'):
            w = getattr(node, attr, None)
            if w is not None and hasattr(w, 'shape'):
                extra += f' {attr}={tuple(w.shape)}'
        for attr in ('stride', 'padding', 'dilation', 'groups'):
            v = getattr(node, attr, None)
            if v is not None:
                extra += f' {attr}={v}'
        print(f'{node.name:>10} {cls:18} relu={int(node.name in relu_names)} '
              f'bnd={int(node.name in lower_bounds)} out={osh} in={ins}{extra}')
    print('relus order:', [r.name for r in bm.relus])
    print('final_name:', net.final_name, ' input_name:', bm.input_name)
    for r in bm.relus:
        pre = r.inputs[0].name
        if pre in lower_bounds:
            print(f'  relu {r.name} pre={pre} lower_bounds{tuple(lower_bounds[pre].shape)} '
                  f'upper_bounds{tuple(upper_bounds[pre].shape)}')
    print('==== END Domain Clipper Graph Dump ====')


def try_clipper_only_bab(net, ret, rhs, x, c=None, timeout=None):
    """
    Pre-beta-CROWN attempt via the lifted zonotope substitution BaB.

    Builds the zonotope generator state from the frozen
    CROWN data + intermediate bounds, then runs the clipper-only dual BaB.
    """
    bab_args = arguments.Config["bab"]
    # needs the zonotope intermediate-domain clipping data and the spec matrix c.
    if not bab_args["clip_n_verify"]["clip_interm_domain"]["enabled"]:
        return False, ret["global_lb"]
    if getattr(net, 'A_saved', None) is None:
        return False, ret["global_lb"]
    if c is None:
        return False, ret["global_lb"]
    if net.final_name not in net.A_saved:
        return False, ret["global_lb"]

    if os.environ.get('DUMP_GRAPH', ''):
        _dump_graph(net, ret)

    device = net.device
    global_lower_bound = ret["global_lb"]                   # [B, n_spec]
    rhs_2d = rhs[:, :, 0] if rhs.dim() == 3 else rhs        # [B, n_spec]
    batch_size = x.ptb.x_L.shape[0]

    if batch_size > 1:
        same_input_box = bool(torch.allclose(x.ptb.x_L, x.ptb.x_L[0:1])) and \
                         bool(torch.allclose(x.ptb.x_U, x.ptb.x_U[0:1]))
        if not same_input_box:
            print(f'[Domain Clipper] skip: batched spec with DISTINCT input boxes '
                  f'(B={batch_size}) -> fall through to beta-CROWN')
            return False, ret["global_lb"]

    # open (batch, spec) pairs = sub-problems alpha-CROWN has NOT verified.
    open_pairs = (global_lower_bound - rhs_2d <= 0).nonzero(as_tuple=False)   # [K, 2] -> (batch, spec)
    if open_pairs.shape[0] == 0:
        return True, global_lower_bound

    # each open pair becomes a query (spec_output_coeffs over the network output, spec_bias = -rhs).
    open_batch_idx = open_pairs[:, 0]
    open_spec_idx = open_pairs[:, 1]
    c_batch_idx = open_batch_idx.to(c.device)
    c_spec_idx = open_spec_idx.to(c.device)
    spec_output_coeffs = c[c_batch_idx, c_spec_idx].to(device)   # [K, n_out]
    spec_bias = (-rhs_2d[open_batch_idx, open_spec_idx]).to(device)  # [K]
    alpha_crown_margin = (
        global_lower_bound[open_batch_idx, open_spec_idx]
        - rhs_2d[open_batch_idx, open_spec_idx]).to(device)

    # ---- build one shared gen-state for all open queries ----
    # Slope choices are relaxation parameters, so sharing them across queries is
    # sound.  build_gen_state already accepts a [n_query, n_out] opt_slope_qw and
    # optimizes the worst root; this avoids rebuilding the same generator table
    # once per query (idx34 has 10+ open queries).
    slope_iters = bab_args.get("clipper_only_slope_iters", 80)
    num_queries = spec_output_coeffs.shape[0]
    shared_slope_iters = bab_args.get("clipper_only_shared_slope_iters", -1)
    if shared_slope_iters < 0:
        shared_slope_iters = 0 if num_queries > 1 else slope_iters
    shared_frontier_cap = bab_args.get("clipper_only_shared_frontier_cap", 250000)
    clipper_start = time.time()
    slope_opt_spec_coeffs = spec_output_coeffs.detach().cpu().double().numpy()
    remaining = None if timeout is None else timeout - (time.time() - clipper_start)
    if remaining is not None and remaining <= 0:
        print('[Domain Clipper] timeout before shared gen-state build; fall through to beta-CROWN')
        return False, global_lower_bound
    t_state = time.time()
    try:
        state = build_gen_state(net, ret, x, device, bidx=0,
                                opt_slope_qw=slope_opt_spec_coeffs,
                                opt_iters=shared_slope_iters, verbose=False)
    except NotImplementedError as e:
        # The gen state is read off a dynamic-forward pass, which has no
        # construction for every operator. Stopping here costs nothing: the
        # clipper is an accelerator and beta-CROWN alone is sound.
        print(f'[Domain Clipper] skip: dynamic forward cannot propagate this '
              f'graph ({e}) -> fall through to beta-CROWN')
        return False, global_lower_bound
    print(f'[Domain Clipper GenState] shared build for {num_queries} queries '
          f'{time.time()-t_state:.2f}s n_gens={state["n_gens"]} '
          f'unstable={len(state["unstable_list"])} slope_iters={shared_slope_iters}')

    # ---- verify each open query ----
    for query_idx in range(num_queries):
        remaining = None if timeout is None else timeout - (time.time() - clipper_start)
        if remaining is not None and remaining <= 0:
            print('[Domain Clipper] timeout before query build; fall through to beta-CROWN')
            return False, global_lower_bound
        query_spec_output_coeffs = spec_output_coeffs[query_idx:query_idx + 1].to(
            dtype=state["output_gen_coeffs"].dtype)
        query_spec_bias = spec_bias[query_idx:query_idx + 1].to(
            dtype=state["output_gen_coeffs"].dtype)
        root_bound_value = float(root_bound(state, query_spec_output_coeffs, query_spec_bias).item())
        print(f'[Domain Clipper GenState] query {query_idx+1}/{num_queries} '
              f'root={root_bound_value:.4f} alpha-CROWN-margin={float(alpha_crown_margin[query_idx]):.4f}')
        remaining = None if timeout is None else timeout - (time.time() - clipper_start)
        verdict, info = _gen_state_bab(
            state, query_spec_output_coeffs, query_spec_bias, device,
            time_limit=remaining,
            frontier_cap=(shared_frontier_cap if num_queries > 1 else None))
        if verdict != 'safe':
            first_info = info.get('specs', [(None, None, {})])[-1][2] if isinstance(info, dict) else {}
            if first_info.get('reason') == 'frontier_cap' and slope_iters > shared_slope_iters:
                remaining = None if timeout is None else timeout - (time.time() - clipper_start)
                if remaining is not None and remaining <= 0:
                    print('[Domain Clipper] timeout before per-query fallback; fall through to beta-CROWN')
                    return False, global_lower_bound
                t_fallback = time.time()
                fallback_spec = spec_output_coeffs[query_idx:query_idx + 1].detach().cpu().double().numpy()
                fallback_state = build_gen_state(
                    net, ret, x, device, bidx=0,
                    opt_slope_qw=fallback_spec, opt_iters=slope_iters,
                    verbose=False)
                fallback_root = float(root_bound(
                    fallback_state,
                    spec_output_coeffs[query_idx:query_idx + 1].to(
                        dtype=fallback_state["output_gen_coeffs"].dtype),
                    spec_bias[query_idx:query_idx + 1].to(
                        dtype=fallback_state["output_gen_coeffs"].dtype)).item())
                print(f'[Domain Clipper GenState] query {query_idx+1}/{num_queries} '
                      f'fallback build {time.time()-t_fallback:.2f}s '
                      f'root={fallback_root:.4f} slope_iters={slope_iters}')
                remaining = None if timeout is None else timeout - (time.time() - clipper_start)
                verdict, info = _gen_state_bab(
                    fallback_state, query_spec_output_coeffs, query_spec_bias,
                    device, time_limit=remaining, frontier_cap=None)
        print(f'[Domain Clipper GenState Bab] query {query_idx+1}/{num_queries}: {verdict}  {info.get("specs", info)}')
        if verdict != 'safe':
            return False, global_lower_bound         # one query unproven -> beta-CROWN handles it

    updated_global_lower_bound = global_lower_bound.clone()
    updated_global_lower_bound[:, :] = rhs_2d + 1.0                  # every query verified -> property safe
    return True, updated_global_lower_bound
