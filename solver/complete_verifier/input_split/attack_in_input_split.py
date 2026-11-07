#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
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
import arguments
from attack import (
            pgd_attack_with_general_specs, PGDAttackResult,
            check_and_save_cex)
from utils import unpad_to_list_of_tensors


def _format_alm_log_value(value, max_items=8):
    """Format scalar/tensor ALM log values without dumping huge batches."""
    if not isinstance(value, torch.Tensor):
        return str(value)
    value = value.detach()
    flat = value.reshape(-1)
    if flat.numel() <= max_items:
        return str(value)
    preview = flat[:max_items].detach().cpu().tolist()
    return (
        f"tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
        f"device={value.device}, preview={preview}, "
        f"... {flat.numel() - max_items} more)"
    )


def _extract_best_feasible_objectives(result, rhs_obj, device, dtype, feas_tol=None):
    """Extract best feasible shifted/raw objective, x, and selected domain index."""
    inf_obj = torch.tensor([float('inf')], device=device, dtype=dtype)
    if feas_tol is None:
        feas_tol = float(arguments.Config["solver"]["nlp"].get("constr_viol_tol", 1e-8))

    def _apply_feasibility_mask(success_mask):
        viol = getattr(result, "constraint_violation", None)
        if viol is None:
            return success_mask
        if not isinstance(viol, torch.Tensor):
            viol_tensor = torch.tensor([float(viol)], device=success_mask.device, dtype=torch.float32)
        else:
            viol_tensor = viol.to(device=success_mask.device)
        return success_mask & (viol_tensor <= feas_tol)

    if not isinstance(result.objective_value, torch.Tensor):
        if bool(result.success):
            shifted = torch.tensor([float(result.objective_value)], device=device, dtype=dtype)
            raw = shifted + rhs_obj[0, 0:1]
            x_sol = result.x_solution.unsqueeze(0) if result.x_solution is not None else None
            return shifted, raw, x_sol, 0
        return inf_obj, inf_obj.clone(), None, None

    rhs_flat = rhs_obj.reshape(-1).to(device=result.objective_value.device, dtype=result.objective_value.dtype)

    if not isinstance(result.success, torch.Tensor):
        if bool(result.success):
            success_mask = _apply_feasibility_mask(torch.tensor([True], device=rhs_flat.device))
            if not bool(success_mask.item()):
                return inf_obj, inf_obj.clone(), None, None
            raw_obj = result.objective_value + rhs_flat
            best_idx = raw_obj.argmin()
            shifted = result.objective_value[best_idx:best_idx+1]
            raw = raw_obj[best_idx:best_idx+1]
            x_sol = result.x_solution[best_idx:best_idx+1] if result.x_solution is not None else None
            return shifted, raw, x_sol, int(best_idx.item())
        return inf_obj, inf_obj.clone(), None, None

    raw_obj = result.objective_value + rhs_flat
    success_mask = _apply_feasibility_mask(result.success)
    masked_raw = torch.where(
        success_mask,
        raw_obj,
        torch.full_like(raw_obj, float('inf')),
    )
    best_idx = masked_raw.argmin()
    best_raw = masked_raw[best_idx:best_idx+1]
    if not torch.isfinite(best_raw).all():
        return inf_obj, inf_obj.clone(), None, None

    best_shifted = result.objective_value[best_idx:best_idx+1]
    best_x = result.x_solution[best_idx:best_idx+1] if result.x_solution is not None else None
    return best_shifted, best_raw, best_x, int(best_idx.item())


def _format_bab_obj_semantics(
    num_domains,
    success_count,
    selected_domain,
    shifted_obj,
    raw_obj,
    rhs_obj,
):
    """Build explicit BaB objective semantics log line."""
    return (
        "  [ALM-BaB] Objective semantics: shifted=(raw-rhs_obj), "
        f"selected_domain={selected_domain}/{num_domains}, "
        f"success={success_count}/{num_domains}, "
        f"shifted_obj={shifted_obj:.6f}, raw_obj={raw_obj:.6f}, rhs_obj={rhs_obj:.6f}"
    )


def _allocate_bab_restarts(num_domains, nlp_config, default_restarts):
    """Allocate restarts from a shared B*R budget in BaB."""
    if num_domains <= 0:
        return 1

    # Treat default_restarts as the TOTAL requested restarts for the batch 
    # to avoid blowing up B*R when num_domains is small.
    allocated = max(1, int(default_restarts) // int(num_domains))

    br_budget = int(nlp_config.get("bab_br_budget", 0))
    if br_budget > 0:
        max_possible = br_budget // int(num_domains)
        return max(1, min(allocated, max_possible))

    return allocated


# Restart cap set after the in-BaB ALM ran out of GPU memory: the model is too
# large for the configured B*R budget, and every later call in this process
# keeps under the cap.
_OOM_RESTART_CAP = None


from nlp_solver.continuation import retained, invoke


@retained
def _solve_within_memory(solver, problem, num_restarts, _sliced=False, **kwargs):
    """solver.solve_with_restarts, halving the restarts on CUDA OOM.

    The ALM is a primal heuristic: fewer restarts only narrow the search. A
    single restart that does not fit still raises (the BaB batch retry takes it).
    """
    global _OOM_RESTART_CAP
    while True:
        restarts = int(num_restarts)
        if _OOM_RESTART_CAP is not None:
            restarts = max(1, min(restarts, _OOM_RESTART_CAP))
        try:
            return (yield from invoke(solver.solve_with_restarts, problem, num_restarts=restarts,
                                      sliced=_sliced, **kwargs))
        except RuntimeError as error:
            from nlp_solver.utils import is_cuda_oom
            if not is_cuda_oom(error) or restarts <= 1:
                raise
            _OOM_RESTART_CAP = restarts // 2
            torch.cuda.empty_cache()
            print(f"  [ALM-BaB] {restarts} restarts ran out of GPU memory; "
                  f"capping at {_OOM_RESTART_CAP}")


def _resolve_alm_batch_budget(num_domains, num_restarts, nlp_config):
    """Resolve per-call ALM chunking so each call stays within the shared B*R budget."""
    br_budget = int(nlp_config.get("bab_br_budget", 0))
    if not bool(nlp_config.get("bab_cap_batch_by_br_budget", False)) or br_budget <= 0:
        return int(num_restarts), int(num_domains), br_budget

    effective_restarts = max(1, int(num_restarts))
    if effective_restarts > br_budget:
        effective_restarts = br_budget

    max_domains_per_call = max(1, br_budget // effective_restarts)
    max_domains_per_call = min(int(num_domains), max_domains_per_call)
    return effective_restarts, max_domains_per_call, br_budget


def _merge_input_linear_constraints(base_constraints, extra_constraints):
    """Concatenate two `(A, b)` input-constraint tuples along the row dimension."""
    if base_constraints is None:
        return extra_constraints
    if extra_constraints is None:
        return base_constraints
    return (
        torch.cat([base_constraints[0], extra_constraints[0]], dim=1),
        torch.cat([base_constraints[1], extra_constraints[1]], dim=1),
    )


def pgd_attack_on_domains(model_ori, domains, stage, vnnlib):
    # dm_l, dm_u: [num_domains, *input_shape]
    # c: [num_domains, max_num_and, output_dim], rhs: [num_domains, max_num_and]
    # spec_size: [num_and_1, num_and_2, num_and_3, ...], with the length of num_domains
    if stage == 'init':
        # Use all domains
        attack_config = arguments.Config["attack"]["input_split"]
        _, dm_l, dm_u, c, rhs, spec_size, _ = domains[:len(domains)]
    elif stage == 'bab':
        # Use only the worst domains
        max_domains = arguments.Config["attack"]["input_split_check_adv"]["max_num_domains"]
        worst_indices = domains.get_topk_indices(k=min(max_domains, len(domains)))
        indices = torch.cat([worst_indices, torch.tensor([len(domains) - 1], device=worst_indices.device)])
        attack_config = arguments.Config["attack"]["input_split_check_adv"]
        _, dm_l, dm_u, c, rhs, spec_size, _ = domains[indices]
    else:
        raise ValueError(f"Invalid stage '{stage}' for PGD attack")

    # [1, num_domains, input_shape]
    dm_l = dm_l.unsqueeze(0)
    dm_u = dm_u.unsqueeze(0)
    center_x = (dm_l + dm_u) / 2

    same_spec_size = (spec_size == spec_size[0]).all()
    if same_spec_size:
        # All domains have the same number of AND specifications
        # c: [1, num_and * num_domains, output_dim], rhs: [1, num_and * num_domains]
        c = c.view(1, -1, c.shape[-1])
        rhs = rhs.view(1, -1)
    else:
        # Different domains may have different number of AND specifications
        # c: [1, num_and_1 + num_and_2 + ..., output_dim], rhs: [1, num_and_1 + num_and_2 + ...]
        c = torch.cat(unpad_to_list_of_tensors(c, 0, 1, spec_size, True), dim=1)
        rhs = torch.cat(unpad_to_list_of_tensors(rhs, 0, 1, spec_size, True), dim=1)

    alpha = (dm_u - dm_l).max() / 8 if attack_config["pgd_alpha"] == "auto" else float(attack_config["pgd_alpha"])
    pgd_steps = attack_config["pgd_steps"]
    num_restarts = attack_config["pgd_restarts"]

    ret: PGDAttackResult = pgd_attack_with_general_specs(
        model_ori, center_x, dm_l, dm_u, c, rhs, spec_size,
        alpha=alpha, pgd_steps=pgd_steps, num_restarts=num_restarts,
        vnnlib=vnnlib
    )

    adv_input, adv_output = ret.adv_input_per_or, ret.adv_output_per_or
    # [1, num_domains, *input_shape], [1, num_domains, output_dim]
    attack_success, best_or_idx = ret.attack_success, ret.best_or_idx
    # [1], [1]

    if attack_success.all():
        print(f"pgd attack succeed in stage {stage}, with index:", best_or_idx)
        _, verified_success = check_and_save_cex(
            adv_input[:, best_or_idx].squeeze(1),
            adv_output[:, best_or_idx].squeeze(1),
            vnnlib, arguments.Config["attack"]["cex_path"], "unsafe"
        )
        return verified_success

    return False


@retained
def alm_attack_on_batch(
    model_ori,
    dm_l,
    dm_u,
    c,
    rhs,
    spec_size,
    *,
    history=None,
    nlp_config=None,
    x_warm=None,
    base_input_linear_constraints=None,
    dual_cut_linear_constraints=None,
    incumbent=None,
):
    """
    Run ALM-PGD on a picked batch of domains.

    The caller is responsible for choosing which domains to run ALM on.
    """
    from nlp_solver import ALMConfig, ALMPGDSolver, NNProblemSingleDomain

    if nlp_config is None:
        nlp_config = arguments.Config["solver"]["nlp"]

    num_domains = dm_l.shape[0]
    device = dm_l.device
    dtype = dm_l.dtype
    
    # Check if all domains have the same spec size
    same_spec_size = (spec_size == spec_size[0]).all()
    if not same_spec_size:
        # For simplicity, only support same spec size for now
        print("  [ALM-BaB] Different spec sizes not supported, skipping")
        inf_obj = torch.tensor([float('inf')], device=device)
        return inf_obj, inf_obj.clone(), None
    
    # Build ALM config
    alm_cfg = ALMConfig.from_dict(nlp_config)
    # Some solving-mode paths use the current primal incumbent as objective rhs.
    # If incumbent is non-finite, rhs[:, 0] can become inf/nan and break ALM.
    num_domains_total = num_domains
    domain_indices = torch.arange(num_domains_total, device=device)
    finite_obj_mask = torch.isfinite(rhs[:, 0])
    if not finite_obj_mask.all():
        num_finite = int(finite_obj_mask.sum().item())
        num_skipped = num_domains - num_finite
        if alm_cfg.verbose:
            print(
                f"  [ALM-BaB] Skipping {num_skipped}/{num_domains} domains "
                "with non-finite objective rhs"
            )
        if num_finite == 0:
            inf_obj = torch.tensor([float('inf')], device=device, dtype=dtype)
            return inf_obj, inf_obj.clone(), None
        domain_indices = domain_indices[finite_obj_mask]
        dm_l = dm_l[finite_obj_mask]
        dm_u = dm_u[finite_obj_mask]
        c = c[finite_obj_mask]
        rhs = rhs[finite_obj_mask]
        spec_size = spec_size[finite_obj_mask]
        num_domains = int(dm_l.shape[0])
        if base_input_linear_constraints is not None:
            base_input_linear_constraints = (
                base_input_linear_constraints[0][finite_obj_mask],
                base_input_linear_constraints[1][finite_obj_mask],
            )
        if dual_cut_linear_constraints is not None:
            dual_cut_linear_constraints = (
                dual_cut_linear_constraints[0][finite_obj_mask],
                dual_cut_linear_constraints[1][finite_obj_mask],
            )

    # Use a shared B*R budget in BaB, with fallback to configured restarts.
    default_restarts = nlp_config.get("bab_num_restarts", max(1, alm_cfg.num_restarts // 10))
    num_restarts = _allocate_bab_restarts(
        num_domains=num_domains,
        nlp_config=nlp_config,
        default_restarts=default_restarts,
    )
    requested_restarts = num_restarts
    num_restarts, max_domains_per_call, br_budget = _resolve_alm_batch_budget(
        num_domains=num_domains,
        num_restarts=num_restarts,
        nlp_config=nlp_config,
    )
    chunked_by_budget = (
        max_domains_per_call < num_domains or num_restarts != requested_restarts
    )
    alm_cfg.num_restarts = num_restarts
    # Use reduced timeout
    alm_cfg.timeout = nlp_config.get("bab_timeout", min(5.0, alm_cfg.timeout))
    # Reduce outer iterations for speed
    alm_cfg.outer_max_iter = nlp_config.get("bab_outer_max_iter", min(10, alm_cfg.outer_max_iter))
    # Reduce inner steps for speed (BaB has warm starts, needs fewer steps)
    alm_cfg.inner_steps = int(nlp_config.get("bab_inner_steps", min(50, alm_cfg.inner_steps)))
    # Optimality polish in BaB has its own size and time cap (negative: keep the root values).
    if int(nlp_config.get("bab_alm_opt_polish_candidates", -1)) >= 0:
        alm_cfg.opt_polish_candidates = int(nlp_config["bab_alm_opt_polish_candidates"])
    if float(nlp_config.get("bab_alm_opt_polish_time", -1.0)) >= 0:
        alm_cfg.opt_polish_time = float(nlp_config["bab_alm_opt_polish_time"])

    # Domain-width adaptive outer iterations: when domains are small (deep BaB),
    # constraints are nearly trivially satisfied, so skip most outer iterations.
    skip_outer_width = float(nlp_config.get("bab_skip_outer_width", 0.0))
    if skip_outer_width > 0:
        # Average max-dimension width across all domains
        avg_max_width = (dm_u - dm_l).flatten(1).max(dim=1).values.mean().item()
        if avg_max_width < skip_outer_width:
            alm_cfg.outer_max_iter = 1
            if alm_cfg.verbose:
                print(f"  [ALM-BaB] Small domain (avg_max_width={avg_max_width:.4e} "
                      f"< {skip_outer_width:.4e}), outer_max_iter=1")

    solver = ALMPGDSolver(alm_cfg)
    
    best_obj = torch.tensor([float('inf')], device=device, dtype=dtype)
    best_raw_obj = torch.tensor([float('inf')], device=device, dtype=dtype)
    best_x = None
    
    # For each domain, we have c: [num_and, output_dim], rhs: [num_and]
    # The first row is objective, remaining rows are constraints
    # We process all domains as a batch for each OR clause
    
    # c: [num_domains, num_and, output_dim], rhs: [num_domains, num_and]
    # Extract objective (row 0) and constraints (row 1:)
    C_obj = c[:, 0:1, :]  # [num_domains, 1, output_dim]
    rhs_obj = rhs[:, 0:1]  # [num_domains, 1]
    C_constr = c[:, 1:, :]  # [num_domains, num_and-1, output_dim]
    rhs_constr = rhs[:, 1:]  # [num_domains, num_and-1]

    original_base_input_count = (
        int(base_input_linear_constraints[0].shape[1])
        if base_input_linear_constraints is not None
        else 0
    )
    deduplicate_base_input = bool(
        original_base_input_count
        and nlp_config.get("expand_input_constraints_as_outputs", False)
        and nlp_config.get("alm_deduplicate_expanded_input_constraints", False)
    )
    base_input_count = 0 if deduplicate_base_input else original_base_input_count
    dual_cut_count = (
        int(dual_cut_linear_constraints[0].shape[1])
        if dual_cut_linear_constraints is not None
        else 0
    )
    shared_mu_prefix_count = int(C_constr.shape[1]) + base_input_count
    total_alm_constraint_count = shared_mu_prefix_count + dual_cut_count
    
    # Prepare warm start: for each domain, check if x_warm is inside bounds
    x_init = None
    inside_bounds = None
    if x_warm is not None:
        x_warm = x_warm.to(device=device, dtype=dtype)
        if x_warm.ndim == len(dm_l.shape) - 1:
            x_warm = x_warm.unsqueeze(0)
        
        x_warm_expanded = x_warm.expand(num_domains, *x_warm.shape[1:])
        inside_bounds = ((x_warm_expanded >= dm_l) & (x_warm_expanded <= dm_u)).flatten(1).all(dim=1)
        x_init = torch.clamp(x_warm_expanded, dm_l, dm_u)
        
        if alm_cfg.verbose:
            n_inside = inside_bounds.sum().item()
            print(f"  [ALM-BaB] Warm start: {n_inside}/{num_domains} domains contain x_warm")
    
    # Warm-start ALM mu_constr from dual ascent lambda* if available
    mu_init = None
    dual_to_alm_mu_warmstart = bool(nlp_config.get("dual_to_alm_mu_warmstart", True))
    if (dual_to_alm_mu_warmstart
            and history is not None
            and getattr(history, 'dual_lam_star', None) is not None):
        dual_lam_star = history.dual_lam_star.to(device=device, dtype=dtype)
        legacy_shared_mu_count = int(C_constr.shape[1]) + original_base_input_count
        legacy_total_mu_count = legacy_shared_mu_count + dual_cut_count
        if deduplicate_base_input and dual_lam_star.shape == (legacy_shared_mu_count,):
            # The dual state was built for [output, repeated base-input].
            # Keep only the output prefix; temporary dual cuts start at zero.
            mu_init = dual_lam_star[:int(C_constr.shape[1])]
            if dual_cut_count > 0:
                mu_init = torch.cat([
                    mu_init,
                    torch.zeros(dual_cut_count, device=device, dtype=dtype),
                ])
        elif deduplicate_base_input and dual_lam_star.shape == (legacy_total_mu_count,):
            # Remap [output, repeated base-input, dual cuts] to
            # [output, dual cuts].
            mu_init = torch.cat([
                dual_lam_star[:int(C_constr.shape[1])],
                dual_lam_star[legacy_shared_mu_count:],
            ])
        elif dual_lam_star.shape == (shared_mu_prefix_count,):
            if dual_cut_count > 0:
                mu_init = torch.cat([
                    dual_lam_star,
                    torch.zeros(dual_cut_count, device=device, dtype=dtype),
                ])
            else:
                mu_init = dual_lam_star
        elif dual_lam_star.shape == (total_alm_constraint_count,):
            mu_init = dual_lam_star
        if mu_init is not None:
            if alm_cfg.verbose:
                print(f"  [ALM-BaB] Warm-starting mu from dual lambda* "
                      f"(norm={mu_init.norm().item():.4f})")

    # Enhance x_init with dual-guided x* for domains where x_warm was out of bounds or unavailable
    if history is not None and getattr(history, 'dual_x_star', None) is not None:
        x_star_dual = history.dual_x_star.to(device=device, dtype=dtype)
        if x_star_dual.shape[0] >= num_domains_total and num_domains != num_domains_total:
            x_star_dual = x_star_dual[domain_indices]
        
        n_avail = min(x_star_dual.shape[0], num_domains)
        if n_avail > 0:
            x_star_partial = x_star_dual[:n_avail].reshape(n_avail, *dm_l.shape[1:])
            
            if x_init is None:
                # If x_warm was totally absent, just use dual_x_star
                x_center = (dm_l[n_avail:] + dm_u[n_avail:]) / 2
                x_init = torch.cat([x_star_partial, x_center], dim=0)
                x_init = torch.clamp(x_init, dm_l, dm_u)
                if alm_cfg.verbose:
                    print(f"  [ALM-BaB] Warm-starting x from dual-guided x* "
                          f"(avail={n_avail}/{num_domains})")
            else:
                # If x_warm was provided, replace bounds where it failed
                replace_mask = ~inside_bounds[:n_avail]
                if replace_mask.any():
                    x_init[:n_avail][replace_mask] = torch.clamp(x_star_partial[replace_mask], dm_l[:n_avail][replace_mask], dm_u[:n_avail][replace_mask])
                    if alm_cfg.verbose:
                        print(f"  [ALM-BaB] Replaced {replace_mask.sum().item()} invalid x_warm with dual_x_star")

    if alm_cfg.verbose and br_budget > 0:
        if chunked_by_budget:
            print(
                "  [ALM-BaB] Shared budget: "
                f"total_domains={num_domains}, requested_restarts={requested_restarts}, "
                f"effective_restarts={num_restarts}, chunk_domains={max_domains_per_call}, "
                f"chunk_B*R={max_domains_per_call * num_restarts}/{br_budget}"
            )
        else:
            print(
                f"  [ALM-BaB] Shared budget: domains={num_domains}, restarts={num_restarts}, "
                f"B*R={num_domains * num_restarts}/{br_budget}"
            )

    num_chunks = (num_domains + max_domains_per_call - 1) // max_domains_per_call
    mu_sum = None
    mu_weight = 0

    for chunk_idx, chunk_start in enumerate(range(0, num_domains, max_domains_per_call)):
        chunk_end = min(chunk_start + max_domains_per_call, num_domains)
        chunk_slice = slice(chunk_start, chunk_end)
        chunk_dm_l = dm_l[chunk_slice]
        chunk_dm_u = dm_u[chunk_slice]
        chunk_C_obj = C_obj[chunk_slice]
        chunk_rhs_obj = rhs_obj[chunk_slice]
        chunk_C_constr = C_constr[chunk_slice]
        chunk_rhs_constr = rhs_constr[chunk_slice]
        chunk_domain_indices = domain_indices[chunk_slice]
        chunk_x_init = x_init[chunk_slice] if x_init is not None else None
        chunk_input_constr_A = None
        chunk_input_constr_b = None
        chunk_input_constraints = None
        if base_input_linear_constraints is not None and not deduplicate_base_input:
            chunk_input_constraints = (
                base_input_linear_constraints[0][chunk_slice],
                base_input_linear_constraints[1][chunk_slice],
            )
        if dual_cut_linear_constraints is not None:
            chunk_input_constraints = _merge_input_linear_constraints(
                chunk_input_constraints,
                (
                    dual_cut_linear_constraints[0][chunk_slice],
                    dual_cut_linear_constraints[1][chunk_slice],
                ),
            )
        if chunk_input_constraints is not None:
            chunk_input_constr_A = chunk_input_constraints[0]
            chunk_input_constr_b = chunk_input_constraints[1]

        problem = NNProblemSingleDomain(
            model=model_ori,
            dm_l=chunk_dm_l,
            dm_u=chunk_dm_u,
            C_obj=chunk_C_obj,
            rhs_obj=chunk_rhs_obj,
            C_constr=chunk_C_constr,
            rhs_constr=chunk_rhs_constr,
            input_constr_A=chunk_input_constr_A,
            input_constr_b=chunk_input_constr_b,
            initialization_constr_A=(
                None if base_input_linear_constraints is None
                else base_input_linear_constraints[0][chunk_slice]
            ),
            initialization_constr_b=(
                None if base_input_linear_constraints is None
                else base_input_linear_constraints[1][chunk_slice]
            ),
            device=device,
            dtype=dtype,
        )

        result = yield from invoke(_solve_within_memory,
            solver,
            problem,
            num_restarts=num_restarts,
            x_init=chunk_x_init,
            mu_init=mu_init,
            incumbent=incumbent, sliced=alm_cfg.controller, _sliced=alm_cfg.controller,
        )

        if result.final_mu is not None:
            final_mu = result.final_mu
            if final_mu.ndim == 1:
                chunk_mu_mean = final_mu.detach().cpu()
            else:
                chunk_mu_mean = final_mu.mean(0).detach().cpu()
            chunk_weight = chunk_end - chunk_start
            if mu_sum is None:
                mu_sum = chunk_mu_mean * chunk_weight
            else:
                mu_sum += chunk_mu_mean * chunk_weight
            mu_weight += chunk_weight

        if alm_cfg.verbose:
            prefix = f"  [ALM-BaB] Chunk {chunk_idx + 1}/{num_chunks}: " if num_chunks > 1 else "  [ALM-BaB] "
            print(
                f"{prefix}Solved {chunk_end - chunk_start} domains: "
                f"success={_format_alm_log_value(result.success)}, "
                f"obj={_format_alm_log_value(result.objective_value)}, "
                f"time={result.solve_time:.2f}s"
            )

        if isinstance(result.success, torch.Tensor):
            any_success = result.success.any().item()
        else:
            any_success = result.success

        if not any_success:
            continue

        # Allow looser feasibility tolerance in BaB
        bab_feas_tol = float(nlp_config.get("bab_feas_tol", 1e-4))
        chunk_best_obj, chunk_best_raw_obj, chunk_best_x, chunk_best_idx = _extract_best_feasible_objectives(
            result, chunk_rhs_obj, device=device, dtype=dtype, feas_tol=bab_feas_tol
        )

        if chunk_best_idx is None or not torch.isfinite(chunk_best_raw_obj).all():
            continue

        if chunk_best_raw_obj < best_raw_obj:
            best_obj = chunk_best_obj
            best_raw_obj = chunk_best_raw_obj
            best_x = chunk_best_x

            if alm_cfg.verbose:
                success_count = (
                    int(result.success.sum().item())
                    if isinstance(result.success, torch.Tensor)
                    else int(bool(result.success))
                )
                rhs_selected = chunk_rhs_obj[chunk_best_idx, 0].item()
                selected_domain = int(chunk_domain_indices[chunk_best_idx].item())
                print(_format_bab_obj_semantics(
                    num_domains=num_domains_total,
                    success_count=success_count,
                    selected_domain=selected_domain,
                    shifted_obj=float(best_obj.item()),
                    raw_obj=float(best_raw_obj.item()),
                    rhs_obj=float(rhs_selected),
                ))

    # Store mean ALM mu_constr in history for warm-starting dual ascent lambda.
    # ALM mu and dual ascent lambda are Lagrange multipliers for the same constraints
    # (C_constr @ f(x) <= rhs_constr), so they have the same semantics and scale.
    # We store the mean across all solved chunks as a (n_constr,) vector that can
    # broadcast to any batch size, avoiding the ALM(N domains) vs dual-ascent(B batch) mismatch.
    if (mu_sum is not None
            and mu_weight > 0
            and history is not None):
        history.alm_mean_mu = (mu_sum / mu_weight)[:shared_mu_prefix_count]

    return best_obj, best_raw_obj, best_x


def update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model_ori):
    device = x_L.device
    max_num_domains = arguments.Config['attack']['input_split_check_adv']['max_num_domains']
    num_domains = min(max_num_domains, x_L.shape[0])
    print(f'Running PGD attack on {num_domains} domains')
    x_L = x_L[:num_domains]
    x_U = x_U[:num_domains]
    cs = cs[:num_domains]
    rhs = thresholds[:num_domains]
    dm_lb = dm_lb[:num_domains]

    data_max = x_U.unsqueeze(0)
    data_min = x_L.unsqueeze(0)
    x = (data_min + data_max) / 2
    # [1, num_domains, *input_shape]

    spec_size = torch.full([cs.shape[0]], cs.shape[1], dtype=torch.int64, device=device)
    C_mat = cs.reshape(1, -1, cs.shape[-1])
    # [1, num_domains * num_and, output_dim]
    rhs_mat = rhs.reshape(1, -1)
    # [1, num_domains * num_and]

    alpha = (data_max - data_min).max() / 8

    pgd_steps = arguments.Config["attack"]["input_split_check_adv"]["pgd_steps"]
    ret: PGDAttackResult = pgd_attack_with_general_specs(
        model_ori, x, data_min, data_max, C_mat, rhs_mat,
        spec_size, alpha=alpha, pgd_steps=pgd_steps
        )

    adv_input, adv_output = ret.adv_input_per_or, ret.adv_output_per_or
    # [1, num_domains, *input_shape], [1, num_domains, output_dim]
    adv_output = adv_output.view(cs.shape[0], -1, 1)
    # [num_domains, output_dim, 1]
    upper_bound = cs.matmul(adv_output).squeeze(-1)
    # [num_domains, num_spec], 

    print('Trying to update RHS with attack')
    print(f'  Current RHS: mean {rhs.mean().item()}')
    print(f'  New upper bound: mean {upper_bound.mean().item()}')
    print(f'  Number of updated RHS: {(upper_bound < rhs).sum()}/{rhs.numel()}')
    rhs = torch.min(rhs, upper_bound)
    thresholds[:num_domains] = rhs
    gap = rhs - dm_lb
    min_gap = gap.min()
    print('  Gap between lower/upper bounds: '
          f'mean {gap.mean().item()}, min {min_gap.item()}')
    assert min_gap >= -1e-3, 'Gap between lower and upper bounds is negative'

    return thresholds
