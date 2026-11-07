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
"""
Entry points for ReLU BaB verification.
ReLU BaB is isolated into three independent stages: preprocess, solve and postprocess.

Note:
Each stage will only rely on explicit argument.
No shared LiRPANet is present. Only solve stage own a LiRPANet.
bab attack features are completely removed here.
"""

import time
import torch
import copy
from branching_domains import (
    BatchedDomainList,
    ShallowFirstBatchedDomainList,
    get_worst_objective_bound,
)
from beta_CROWN_solver import LiRPANet
from auto_LiRPA.utils import (
    stop_criterion_batch_any,
    AutoBatchSize,
)
from domain_clipper import DomainClipScorer
from heuristics import get_branching_heuristic
from state import WorkingIntermBoundsInfo, IntermBoundsFactory
from cuts.cut_verification import cut_verification
from cuts.infered_cuts import BICCOS
from utils import (
    Stats,
    get_unstable_neurons,
    check_auto_enlarge_batch_size,
)
from prune import prune_alphas
import arguments
from bab import multi_tree_bab
from clipper_bab import try_clipper_only_bab
from activation_split.decision_precompute import compute_first_iteration_decision
from activation_split.stage_arguments import (
    PreprocessConstArguments,
    PreprocessMutableArguments,
    SolveConstArguments,
    SolveMutableArguments,
    PostprocessConstArguments,
    PostprocessMutableArguments,
)
from activation_split.protocols import PreprocessPacket, SolvePacket, PostprocessPacket
from activation_split.stage_preprocess import branch_and_bound_preprocess
from activation_split.stage_solve import branch_and_bound_solve
from activation_split.stage_postprocess import branch_and_bound_postprocess
from solving_utils import fixed_primal_dual_enabled, is_primal_dual_converged


def _fixed_primal_dual_converged(primal, dual) -> bool:
    """Apply the configured gap tolerances in internal minimization space."""
    nlp_args = arguments.Config["solver"]["nlp"]
    abs_tol = float(nlp_args.get("bab_primal_dual_abs_tol", 1e-5))
    rel_tol = float(nlp_args.get("bab_primal_dual_rel_tol", 1e-4))
    return is_primal_dual_converged(
        primal, dual, abs_tol=abs_tol, rel_tol=rel_tol
    )


def _report_fixed_dual(primal, dual) -> None:
    """Print a parser-compatible fixed-incumbent primal/dual progress record."""
    primal = torch.as_tensor(primal).reshape(1)
    dual = torch.as_tensor(
        dual, device=primal.device, dtype=primal.dtype
    ).reshape(-1)[0]
    gap = float((primal.reshape(-1)[0] - dual).item())
    if arguments.Config["solving"]["is_maximization"]:
        primal_print, dual_print = -primal, -dual
    else:
        primal_print, dual_print = primal, dual
    print(
        "[GenBaB dual] "
        f"primal={float(primal_print.reshape(-1)[0].item()):.12g} "
        f"dual={float(dual_print.item()):.12g} gap={gap:.12g}"
    )
    print(f"Primal Solution {primal_print}, Dual Solution {dual_print}")
    print(f"Solution gap: {gap}")


def general_bab(
    net: LiRPANet,
    x,
    c,
    rhs,
    *,
    reference_dict,
    timeout=None,
    max_iterations=None,
):
    """Entry point for ReLU BaB."""
    start_time = time.time()
    stats = Stats()

    # ============ Retrieve Options and Arguments ============
    solver_args = arguments.Config["solver"]
    bab_args = arguments.Config["bab"]
    branch_args = bab_args["branching"]
    timeout = timeout or bab_args["timeout"]
    max_domains = bab_args["max_domains"]
    batch = solver_args["batch_size"]
    cut_enabled = bab_args["cut"]["enabled"]
    biccos_args = bab_args["cut"]["biccos"]
    enable_clip_domains = (
        bab_args["clip_n_verify"]["clip_input_domain"]["enabled"]
        or bab_args["clip_n_verify"]["clip_interm_domain"]["enabled"]
    )
    max_iterations = max_iterations or bab_args["max_iterations"]
    MTS_enabled = (
        biccos_args["enabled"] and biccos_args["multi_tree_branching"]["enabled"]
    )
    fixed_dual_eval = fixed_primal_dual_enabled()

    stop_criterion = stop_criterion_batch_any
    stop_criterion_func = stop_criterion(rhs)

    refined_lower_bounds = reference_dict.get("lower_bounds", None)
    refined_upper_bounds = reference_dict.get("upper_bounds", None)
    reference_lA = reference_dict.get("lA", None)
    reference_alphas = reference_dict.get("alphas", None)
    refined_betas = reference_dict.get("refined_betas", None)

    # ============ Initialize LiRPANet for ReLU BaB iteration ============
    # Initialize BaB with bounds produced by incomplete verification.
    ret = net.build_with_refined_bounds(
        x,
        c,
        rhs,
        stop_criterion,
        refined_lower_bounds,
        refined_upper_bounds,
        reference_lA,
        reference_alphas,
        refined_betas,
    )

    (global_ub, global_lb, updated_mask, lA, alpha) = (
        ret["global_ub"],
        ret["global_lb"],
        ret["mask"],
        ret["lA"],
        ret["alphas"],
    )
    if fixed_dual_eval:
        fixed_primal = rhs.reshape(rhs.shape[0], -1)[0, 0]
        objective_dual = global_lb.reshape(global_lb.shape[0], -1)[:, 0].min()
        _report_fixed_dual(fixed_primal, objective_dual)
        if _fixed_primal_dual_converged(fixed_primal, objective_dual):
            return (objective_dual - fixed_primal).item(), 0, "safe"

    initial_bs_ratio = None
    if cut_enabled:
        # Always reset the cut module if it exists
        net.net.cut_timestamp = -1
        net.net.cut_module = None
        # All intermediate bounds are set during the incomplete verification phase.
        # We only need to set the final layer bounds here.
        net.net[net.net.final_name].lower = global_lb
        net.net[net.net.final_name].upper = global_ub
        net.set_cuts()
        if biccos_args["enabled"]:
            net.biccos = BICCOS(ret, rhs, net.final_name)
            initial_bs_ratio, MTS_enabled = net.biccos.set_auto_params()

    net.interm_transfer = bab_args["interm_transfer"]

    all_label_global_lb = torch.min(global_lb - rhs).item()
    all_label_global_ub = torch.max(global_ub - rhs).item()

    if arguments.Config["debug"]["lp_test"] in ["LP", "MIP"]:
        return all_label_global_lb, 0, "unknown"

    if stop_criterion_func(global_lb).all():
        return all_label_global_lb, 0, "safe"

    # ---- Clipper BaB with zonotope ----
    # Builds an same-slope CROWN (zonotope) generator state
    # one e_new per unstable neuron + per-neuron slope-opt
    clipper_only_enabled = bab_args["clipper_only_bab_timeout"]
    if clipper_only_enabled:
        try:
            clipper_closed, global_lb = try_clipper_only_bab(net, ret, rhs, x, c=c, timeout=timeout)
        except torch.OutOfMemoryError as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            oom_msg = str(e).splitlines()[0]
            print(f"[Domain Clipper] CUDA OOM; fall through to beta-CROWN ({oom_msg})")
            clipper_closed = False
        all_label_global_lb = torch.min(global_lb - rhs).item()
        if clipper_closed:
            print("[Domain Clipper] all specs verified by zonotope substitution BaB")
            return all_label_global_lb, 0, "safe"

    # If we are not optimizing intermediate layer bounds, we do not need to
    # save all the intermediate alpha.
    # We only keep the alpha for the last layer.
    if not solver_args["beta-crown"]["enable_opt_interm_bounds"]:
        # new_alpha shape:
        # [dict[relu_layer_name, {final_layer: torch.tensor storing alpha}]
        # for each sample in batch]
        alpha = prune_alphas(alpha, net.alpha_start_nodes)

    if MTS_enabled:
        DomainClass = ShallowFirstBatchedDomainList
    else:
        DomainClass = BatchedDomainList

    # ============ Initialize DomainList and its company ============
    domains = DomainClass(
        ret,
        c,
        lA,
        global_lb,
        global_ub,
        alpha,
        copy.deepcopy(ret["history"]),
        rhs,
        net=net,
        x=x,
        branching_input_and_activation=branch_args["branching_input_and_activation"],
        timer=stats.timer,
    )
    net.domain_interm_factory = IntermBoundsFactory.from_interm(
        interm=WorkingIntermBoundsInfo.from_two_dicts(
            ret["lower_bounds"], ret["upper_bounds"]
        ),
        final_name=net.final_name,
        device=net.device,
    )
    num_domains = len(domains)

    # after domains are added, we replace global_lb, global_ub with the multiple
    # targets 'real' global lb and ub to make them scalars
    global_lb, global_ub = all_label_global_lb, all_label_global_ub
    updated_mask, tot_ambi_nodes = get_unstable_neurons(updated_mask, net)
    net.tot_ambi_nodes = tot_ambi_nodes
    domains.update_unstable_mask(updated_mask)
    net.unstable_mask = domains.unstable_mask

    # ============ Initialize clip and branching decision ============
    if enable_clip_domains:
        assert net.domain_clipper is not None
        # init domain_clipper.update_unstable_idx
        net.domain_clipper.update_unstable_idx(updated_mask, net)
        # create domain scorer
        domain_clip_scorer = DomainClipScorer(
            final_name=net.final_name,
            true_indices=copy.deepcopy(net.domain_clipper.true_indices),
            topk_objective=net.domain_clipper.topk_objective,
        )
    else:
        domain_clip_scorer = None

    if cut_enabled:
        cut_verification(net, domains, recorder=net.recorder)

    branching_heuristic = get_branching_heuristic(net)

    # ============ multi-tree search as a pre-solve for BICCOS ============
    # If we are using shallow branching, we need to do the multi-tree search
    # as the pre-solve part for BICCOS.
    if isinstance(domains, ShallowFirstBatchedDomainList):
        global_lb = multi_tree_bab(
            net,
            domains,
            batch,
            stop_criterion_func,
            biccos_args,
            stats,
            start_time,
            initial_bs_ratio,  # pylint: disable=used-before-assignment
        )
    # ============= Prepare direct variables for BaB iterations ======
    num_domains = len(domains)
    vram_ratio = 0.85 if cut_enabled else 0.9
    auto_batch_size = AutoBatchSize(
        batch,
        net.device,
        vram_ratio,
        enable=arguments.Config["solver"]["auto_enlarge_batch_size"],
    )

    total_round = 0
    result = None
    timer = stats.timer
    net.timer = timer

    compute_first_iteration_decision(
        net=net,
        domains=domains,
        domain_interm_factory=net.domain_interm_factory,
        domain_clip_scorer=domain_clip_scorer,
        branching_heuristic=branching_heuristic,
        drop_lA=True,
        enable_clip_domains=enable_clip_domains,
        device=net.device,
        timer=timer,
    )

    pre_const_args = PreprocessConstArguments.from_net(net)
    solve_const_args = SolveConstArguments(
        domain_clip_scorer=domain_clip_scorer,
        branching_heuristic=branching_heuristic,
    )
    post_const_args = PostprocessConstArguments.from_net(net)

    # =================== BaB iterations ===================
    while num_domains > 0 and (max_iterations == -1 or total_round < max_iterations):
        timer.start("bab_iteration")
        total_round += 1
        print(f"BaB round {total_round}")

        auto_batch_size.record_actual_batch_size(min(batch, len(domains)))
        pre_mutable_args = PreprocessMutableArguments(
            iter_idx=total_round,
            device_batch_limit=batch,
            stats=stats,
            domain_interm_factory=net.domain_interm_factory
        )
        pre_packet: PreprocessPacket = branch_and_bound_preprocess(
            domains,
            pre_const_args,
            pre_mutable_args,
        )
        if pre_packet.exit_status != "normal":
            raise RuntimeError(
                f"Preprocess stage exited with status: {pre_packet.exit_status}"
            )
        solve_mutable_args = SolveMutableArguments(stats=stats)
        solve_packet: SolvePacket = branch_and_bound_solve(
            net=net,
            prePacket=pre_packet,
            const_args=solve_const_args,
            mutable_args=solve_mutable_args,
        )
        if solve_packet.early_exit_status != "normal_exit":
            if solve_packet.early_exit_status == "all_node_split_LP_unsafe":
                stats.all_node_split = False
                result = "unsafe_bab"
                break
            elif solve_packet.early_exit_status == "all_node_split_unknown":
                stats.all_node_split = False
                result = "unknown"
                break
            else:
                raise RuntimeError(
                    f"Solve stage exited with status: {solve_packet.early_exit_status}"
                )
        post_mutable_args = PostprocessMutableArguments(
            iter_idx=total_round,
            stats=stats,
        )
        post_packet: PostprocessPacket = branch_and_bound_postprocess(
            domains,
            solve_packet,
            post_const_args,
            post_mutable_args,
        )
        if post_packet.exit_status != "normal":
            raise RuntimeError(
                f"Postprocess stage exited with status: {post_packet.exit_status}"
            )

        if fixed_dual_eval:
            objective_state = get_worst_objective_bound(domains)
            if objective_state is None:
                global_lb = torch.tensor(0.0, device=net.device)
                result = "safe"
            else:
                objective_dual, fixed_primal = objective_state
                global_lb = objective_dual - fixed_primal
                _report_fixed_dual(fixed_primal, objective_dual)
                if _fixed_primal_dual_converged(
                    fixed_primal, objective_dual
                ):
                    result = "safe"
        else:
            global_lb = post_packet.global_lb.max()

        # clean packets
        del pre_packet
        del solve_packet
        del post_packet

        # auto enlarge
        batch = check_auto_enlarge_batch_size(auto_batch_size)
        num_domains = len(domains)
        if result is None and num_domains > max_domains:
            print("Maximum number of visited domains has reached.")
            result = "unknown"
        elif result is None and time.time() - start_time > timeout:
            print("Time out!!!!!!!!")
            result = "unknown"
        timer.add("bab_iteration")
        if result:
            break
        print(f"Cumulative time: {time.time() - start_time}\n")

    if not result:
        # No domains left and not timed out.
        result = "safe"

    return global_lb, stats.visited, result, stats
