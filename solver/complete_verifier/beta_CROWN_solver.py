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
import multiprocessing
import os
import ast
import copy
from collections import defaultdict

import torch
import arguments
import warnings

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import (
        stop_criterion_all,
        stop_criterion_batch_any, stop_criterion_general,
        reduction_str2func)

from attack import attack
from utils import (
    Timer,
    take_batch,
    expand_batch,
    transfer_obj,
    merge_splits_per_example
)
from load_model import Customized
from state import (
    AlphaValueData,
    AlphaFullInfoData,
    BetaFullData,
    NumsEffectiveBetasPerDomain,
    BatchedlA,
    IntermBoundsFactory,
)
from state.alpha import resolve_alpha_storage_dtype
from prune import PruneAfterCROWN
from crown_spec_batching import compute_bounds_in_spec_batches
from state import WorkingIntermBoundsInfo


def _collect_ibp_reference_bounds(net, batch_x, batch_c):
    reference_interm_bounds = {}
    try:
        net.compute_bounds(
            x=(batch_x,), C=batch_c, method="ibp", bound_upper=False)
    except (AssertionError, ValueError) as exc:
        warnings.warn(
            "IBP enhancement failed; continuing without IBP reference "
            f"bounds: {exc}"
        )
        return reference_interm_bounds

    for node in net.nodes():
        if (node.perturbed
            and isinstance(node.lower, torch.Tensor)
            and isinstance(node.upper, torch.Tensor)):
            reference_interm_bounds[node.name] = (node.lower, node.upper)
    return reference_interm_bounds


class LiRPANet:
    def __init__(self, model_ori, in_size, c=None, device=None):
        """
        convert pytorch model to auto_LiRPA module
        """
        general_args = arguments.Config['general']
        solver_args = arguments.Config['solver']
        model_args = arguments.Config['model']
        bab_args = arguments.Config['bab']
        # c is not None only when LiRPANet is initialized from construct_mip_with_model()
        self.c = c
        self.model_ori = model_ori
        self.input_shape = in_size
        self.device = device or general_args['device']
        model_ori_state_dict = copy.deepcopy(model_ori.state_dict())
        bound_opts = {
            'deterministic': general_args['deterministic'],
            'conv_mode': general_args['conv_mode'],
            'sparse_features_alpha': general_args['sparse_alpha'],
            'sparse_spec_alpha': general_args['sparse_alpha'],
            'sparse_intermediate_bounds': general_args['sparse_interm'],
            'batched_crown_max_vram_ratio': solver_args['batched_crown_auto_enlarge_max_vram_ratio'],
            'crown_batch_size': solver_args['crown']['batch_size'],
            'max_crown_size': solver_args['crown']['max_crown_size'],
            'forward_refinement': solver_args['forward']['refine'],
            'forward_max_dim': solver_args['forward']['max_dim'],
            'use_full_conv_alpha': solver_args['alpha-crown']['full_conv_alpha'],
            'disable_optimization': solver_args['alpha-crown']['disable_optimization'],
            'fixed_reducemax_index': True,
            'matmul': {'share_alphas': solver_args['alpha-crown']['matmul_share_alphas']},
            'tanh': {'loose_threshold': bab_args['branching']['nonlinear_split']['loose_tanh_threshold']},
            'activation_bound_option': solver_args['crown']['activation_bound_option'],
            'buffers': {'has_batchdim': general_args['buffer_has_batchdim']},
            'optimize_bound_args': {
                'apply_output_constraints_to': solver_args['invprop']['apply_output_constraints_to'],
                'tighten_input_bounds': solver_args['invprop']['tighten_input_bounds'],
                'best_of_oc_and_no_oc': solver_args['invprop']['best_of_oc_and_no_oc'],
                'directly_optimize': solver_args['invprop']['directly_optimize'],
                'oc_lr': solver_args['invprop']['oc_lr'],
                'share_gammas': solver_args['invprop']['share_gammas'],
            },
            "optimize_graph": {
                "optimizer": eval(model_args['optimize_graph']) if model_args['optimize_graph'] else None,
            },
            "compare_crown_with_ibp": solver_args['crown']['compare_crown_with_ibp'],
            "forward_before_compute_bounds": solver_args['forward_before_compute_bounds'],
            'op_budget': dict(model_args['op_budget']),
        }
        # Update with user-provided bound_opts if specified
        user_bound_opts_str = model_args['bound_opts']
        if user_bound_opts_str:
            try:
                user_bound_opts = ast.literal_eval(user_bound_opts_str)
            except Exception as e:
                raise ValueError(
                    f"Invalid --bound_opts argument: must be a valid Python dictionary literal.\n"
                    f"Error: {e}"
                )
            if 'op_budget' in user_bound_opts:
                raise ValueError("op_budget is set only through model.op_budget.* in the config, not --bound_opts")
            # Recursively update nested dictionaries
            def update_nested_dict(base, updates):
                for key, value in updates.items():
                    if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                        update_nested_dict(base[key], value)
                    else:
                        base[key] = value
            update_nested_dict(bound_opts, user_bound_opts)
        def build(opts=bound_opts):
            return BoundedModule(model_ori, torch.zeros(in_size, device=self.device), bound_opts=opts,
                                 device=self.device)
        # GEAR facade (G5 DESIGN sections 2 Reuse, 4.1 commit modes): a handed-over model may supply the build
        # (template reuse, the whole-commit rebuild, the op-budget record); the frozen path builds as before.
        gear_build = getattr(model_ori, '__gear_build__', None)
        self.net = build() if gear_build is None else gear_build(build, bound_opts)
        eval(general_args['graph_optimizer'])(self.net)
        self.root = self.net[self.net.root_names[0]]

        self.net.eval()
        self.return_A = False
        self.needed_A_dict = None
        self.cutter = None # class for generating and optimizing cuts
        self.biccos = None # class for BICCOS
        self.timer = Timer()

        # for fetching cplex in parallel
        self.mip_building_proc: None | multiprocessing.Process = None
        self.processes: None | dict = None
        self.get_cuts_watch_dog_proc: None | multiprocessing.Process = None
        self.pool = self.pool_result = self.pool_termination_flag = None # For multi-process.

        # for recording whether we need to return intermediate bounds
        # after initial intermediate bounds fetching, this switch will be
        # aligned with arg.bab.interm_transfer
        self.interm_transfer = True

        self.domain_clipper = None # class for domains clipper

        self.final_name = self.net.final_name
        self.alpha_start_nodes = [self.final_name] + list(filter(
            lambda x: len(x.strip()) > 0,
            bab_args['optimized_interm'].split(',')))

        # If input split is enabled, we do not split nodes, self.nonlinear_split is False
        self.nonlinear_split = (
            not bab_args['branching']['input_split']['enable']
            and bab_args['branching']['method'] == 'nonlinear'
        )

        if arguments.Config['model']['with_jacobian']:
            print('Not checking the conversion correctness for this model with JacobianOP')
        else:
            # check conversion correctness
            dummy = torch.randn(in_size, device=self.device)
            try:
                assert torch.allclose(model_ori(dummy), self.net(dummy), atol=1e-7, rtol=5e-4)
            except AssertionError:
                print('torch allclose failed: norm '
                    f'{torch.norm(model_ori(dummy) - self.net(dummy))}')
        model_ori.load_state_dict(model_ori_state_dict, strict=False)

        self.unstable_mask = {}
        self.domain_interm_factory: IntermBoundsFactory | None = None

        self.tot_ambi_nodes = 0  # total number of ambiguous nodes, predefined here.
        self.solver_model_initialized = False

    # FIXME: should not pass lb and ub into function, they should be from self.net
    def get_mask(self):
        masks = {}
        if arguments.Config['bab']['branching']['input_split']['enable']:
            # input bab does not need to fetch split masks.
            return masks
        for node in self.net.get_splittable_activations():
            mask = []
            for idx, input_node in enumerate(node.inputs):
                if not input_node.perturbed:
                    continue
                if input_node.lower is None and input_node.upper is None:
                    mask.append(None)
                else:
                    mask.append(node.get_split_mask(
                        input_node.lower, input_node.upper, idx))
            masks[node.name] = mask
        return masks

    def get_split_nodes(self, verbose=False):
        if arguments.Config['bab']['branching']['nonlinear_split']['relu_only']:
            self.net.disabled_split_nodes.update(
                node for node in self.net.nodes()
                if not isinstance(node, BoundRelu))
        self.net.get_split_nodes()
        # self.net.get_output_nodes()
        self.split_activations = self.net.split_activations
        if verbose:
            print('Split layers:')
            for layer in self.net.split_nodes:
                print(f'  {layer}: {self.split_activations[layer.name]}')
            print('Nonlinear functions:')
            for node in self.net.nodes():
                if node.perturbed and len(node.requires_input_bounds):
                    print('  ', node)
        if not self.nonlinear_split:
            self.nonlinear_split = any(
                not isinstance(node, BoundRelu)
                for node in self.net.get_splittable_activations())

    def build(self, x, c, rhs,
              stop_criterion,
              bounding_method=None, vnnlib_handler=None,
              interm_bounds=None, return_A=False,
              or_spec_size=None,
              full_alpha_info=False,
              ):
        # TODO merge with build_with_refined_bounds()
        solver_args = arguments.Config['solver']
        bab_args = arguments.Config['bab']
        branching_args = bab_args['branching']
        share_alphas = solver_args['alpha-crown']['share_alphas']
        bounding_method = bounding_method or solver_args['bound_prop_method']
        batch_size_target = solver_args['build_batch_size']
        input_split_args = branching_args["input_split"]
        branching_input_and_activation = branching_args['branching_input_and_activation']
        enable_clip_domains = (bab_args["clip_n_verify"]["clip_interm_domain"]['enabled'])
        clip_in_alpha_crown = (
            bab_args["clip_n_verify"]["clip_interm_domain"]["clip_in_alpha_crown"])
        enable_opt_interm_bounds = arguments.Config["solver"]["beta-crown"]["enable_opt_interm_bounds"]
        enable_input_split = branching_args["input_split"]['enable']
        need_alphas = bounding_method in ['alpha-crown', 'alpha-forward']
        # get_all_alphas: we need both intermediate and final alphas
        get_all_alphas = enable_opt_interm_bounds or enable_input_split
        device = self.device

        solving_mode = arguments.Config["solving"]["solving_mode"]
        obj_index = arguments.Config["solving"]["obj_index"]
        if solving_mode:
            assert c.shape[0] == 1, "For now, solving mode does not support multiple OR."
            # Model output dimension
            output_dim = c.shape[2]
            # TODO: Maybe the shape checking here is unecessary?
            assert (rhs.shape[0] == 1) and (rhs.shape[1] == c.shape[1]), "C and rhs shape do not match!"
            assert obj_index < output_dim, "Objective index exceeds range!"
            new_one_hot = torch.zeros((1, 1, output_dim), device=c.device)
            new_one_hot[0, 0, obj_index] = 1
            c = torch.cat((new_one_hot, c), dim=1)
            new_rhs = torch.zeros((1, 1), device=rhs.device)
            # The constraint rows arrive relaxed by the verify predicate's tolerance (incomplete_verifier), so the stops
            # and prunes below never call a row infeasible while a point within the tolerance satisfies it.
            rhs = torch.cat((new_rhs, rhs), dim=1)
            # For solving mode, we do not want to stop verification when the objective (index 0)
            # is > 0. We only stop when all constraints are satisfied.
            # So we set the stopping threshold for objective to inf.
            # We keep a copy of rhs with 0 for optimization (decision_thresh) and attack.
            rhs_opt = rhs.clone()
            rhs[:, 0] = float('inf')
        else:
            rhs_opt = rhs

        # When full_alpha_info is True, we get all items related to alpha,
        #   including alpha values, some indices (like alpha_lookup_idx if sparse alpha is enabled)...
        #   the alpha dict is in the form of
        #       {node_name: {'alpha': {start_node_name: alpha_value},
        #                    'alpha_lookup_idx': {start_node_name: alpha_lookup_idx}, ...}}
        # When full_alpha_info is False, we only get alpha values.
        #   the alpha dict is in the form of
        #       {node_name: {start_node_name: alpha_value}}
        #   Compared with the dict if full_alpha_info is True, we reduce one level of nesting.
        batch_handler = BatchHandler(
            x, c, rhs, stop_criterion, or_spec_size,
            interm_bounds, batch_size_target, self.final_name,
            need_alphas, full_alpha_info, None, clip_in_alpha_crown
        )

        self.net.set_bound_opts({'verbosity': 1})
        self.set_crown_bound_opts('alpha')
        # If input split is enabled, we do not need to get split nodes.
        if not branching_args['input_split']['enable']:
            self.get_split_nodes(verbose=True)

        self._set_A_options(return_A=return_A or enable_clip_domains or clip_in_alpha_crown)
        if clip_in_alpha_crown:
            self.net.set_bound_opts({
                'clip_in_alpha_crown': True,
                'clip_in_alpha_crown_disjuncts_separated': (
                    batch_handler.optimize_disjuncts_separately),
            })

        total_batches = batch_handler.total_batches
        for now_batch in range(total_batches):
            print(f'build batch [{now_batch+1}/{total_batches}]')
            torch.cuda.empty_cache()

            # Get batch input.
            batch_x, batch_c, batch_rhs, stop_criterion_func, batch_interm_bounds, _, batch_or_spec_size = (
                batch_handler.get_batch_input(now_batch, self.device)
            )
            self.net.set_bound_opts({'optimize_bound_args': {'stop_criterion_func': stop_criterion_func}})
            self.x = batch_x
            self.c = batch_c

            if solving_mode:
                batch_rhs_opt = batch_handler.take_batch(rhs_opt, now_batch, device)
            else:
                batch_rhs_opt = batch_rhs

            # ---------- Compute bounds with different methods START ----------
            prune_after_crown = None
            if bounding_method == 'alpha-crown':
                # NOTE: There are some fast paths for alpha-crown.
                # 1. If we have only one batch, and we can verify it, we can return directly.
                # 2. If we enable attack with middle order and the attack is successful, we can return directly.
                # Besides, we can prune the specifications verified by initial CROWN bounds
                # to save the computational cost of alpha-crown.

                # first get CROWN bounds
                # Reference bounds are intermediate layer bounds from initial CROWN bounds.
                final_layer_only = True if input_split_args["beta"] else False
                lb, ub, aux_reference_bounds = self.net.init_alpha(
                    (batch_x,), share_alphas=share_alphas, c=batch_c, bound_upper=False,
                        final_layer_alpha_only=final_layer_only)
                print('initial CROWN bounds (first 10 items):', lb.flatten()[:10].tolist())

                if arguments.Config['general']['save_output']:
                    assert total_batches == 1
                    arguments.Globals['out']['init_crown_bounds'] = lb.cpu()

                if stop_criterion_func(lb).all().item():
                    # Fast path. Initial CROWN bound can verify the network.
                    print('Verified with initial CROWN!')
                    ret = (lb, None)
                    if total_batches == 1:
                        # If we only have one batch, we can return the result directly.
                        return lb, {}
                else:
                    # Prune the specifications that can be verified by initial CROWN bounds.
                    if solver_args['prune_after_crown']:
                        prune_after_crown = PruneAfterCROWN(
                            self.net, batch_x, batch_c, batch_rhs, lb,
                            aux_reference_bounds,
                            stop_criterion,
                            or_spec_size=batch_or_spec_size)
                        # Update batched data after pruning.
                        batch_x, batch_c, batch_rhs, batch_or_spec_size, stop_criterion_func = prune_after_crown.get_pruned_data()
                        self.x = batch_x
                        self.c = batch_c
                        self.net.set_bound_opts({
                            'optimize_bound_args': {'stop_criterion_func': stop_criterion_func},
                        })
                    if arguments.Config['attack']['pgd_order'] == 'middle' and vnnlib_handler is not None:                        
                        _, verified_success, attack_examples, _, _ = attack(
                            self.model_ori, batch_x, batch_c, batch_rhs_opt, batch_or_spec_size,
                            vnnlib_handler.vnnlib,
                            verified_status="unknown", verified_success=False)

                        if verified_success:
                            print("pgd attack succeed in middle order")
                            return None, {'attack_examples': attack_examples}

                    if enable_clip_domains or clip_in_alpha_crown:
                        print('Using alpha-CROWN with output constraints to initialize bounds.')
                    else:
                        print('Using alpha-CROWN to initialize bounds.')
                    ret = self.net.compute_bounds(
                        x=(batch_x,), C=batch_c, method='CROWN-Optimized',
                        return_A=self.return_A, needed_A_dict=self.needed_A_dict,
                        bound_upper=False, aux_reference_bounds=aux_reference_bounds,
                        cutter=self.cutter, interm_bounds=batch_interm_bounds,
                        decision_thresh=rhs_opt)
            elif bounding_method == 'alpha-forward':
                warnings.warn('alpha-forward can only be used with input split for now')
                self.net.bound_opts['optimize_bound_args']['init_alpha'] = True
                ret = self.net.compute_bounds(
                    x=(batch_x,), C=batch_c, method='forward-optimized', bound_upper=False)
            else:
                with torch.no_grad():
                    if bounding_method == 'init-crown':
                        assert not self.return_A
                        lb, ub, _ = self.net.init_alpha(
                            (batch_x,), share_alphas=share_alphas, c=batch_c,
                            bound_upper=False
                            )
                        ret = lb, ub
                    else:
                        reference_interm_bounds = {}

                        input_split_configs = arguments.Config['bab']['branching']['input_split']
                        if (input_split_configs['enable']
                            and input_split_configs["ibp_enhancement"]):
                            reference_interm_bounds = _collect_ibp_reference_bounds(
                                self.net, batch_x, batch_c)

                        ret = compute_bounds_in_spec_batches(
                            self.net,
                            spec_batch_size=solver_args['crown'].get(
                                'spec_batch_size', solver_args['crown']['batch_size']),
                            oom_retry=True,
                            x=(batch_x,), C=batch_c, method=bounding_method,
                            bound_upper=False, return_A=self.return_A,
                            needed_A_dict=self.needed_A_dict,
                            reference_bounds=reference_interm_bounds)
            # ----------- Compute bounds with different methods END -----------

            lb, ub = ret[0], ret[1]
            A = ret[-1]

            if branching_input_and_activation:
                assert total_batches == 1
                # Use A matrix of the input, the find best neuron to branch in input space.
                from auto_LiRPA.not_in_release import not_in_release
                not_in_release("bab.branching.branching_input_and_activation")
            else:
                batch_input_split_idx = {}

            print(f'initial {bounding_method} bounds (first 10 items):', lb.flatten()[:10].tolist())
            global_lb = lb.min().item()
            print(f'Global lower bound: {global_lb}')

            # DEBUG: check loose bounds
            if os.environ.get('ABCROWN_VIEW_INTERM', False):
                print('Intermediate bounds after initial alpha-CROWN:')
                self._print_interm_bounds()
                import pdb; pdb.set_trace()

            # Get batch results.
            if need_alphas:
                batch_alpha = self.net.get_alpha(get_all=get_all_alphas,
                    half=resolve_alpha_storage_dtype(
                        arguments.Config["solver"]["alpha-crown"]["alpha_dtype"],
                        reproduce_alpha_bounds=arguments.Config[
                            "solver"]["reproduce_alpha_bounds"],
                    ) is torch.float16,
                    full_info=full_alpha_info, drop_unused=True
                )
            else:
                batch_alpha = None
            batch_lb, batch_ub = WorkingIntermBoundsInfo.from_net(self, move=False).to_complete_init_bounds(
                self, lb, None
            )  # primals are better upper bounds
            batch_mask = self.get_mask()
            batch_lA = BatchedlA.from_net(self)
            batch_mask = self.get_mask()
            if prune_after_crown:
                prune_after_crown.recover(batch_lb, batch_ub, batch_lA, batch_alpha, 
                                        batch_mask, batch_input_split_idx, full_alpha_info)

            # save initial alpha-crown for tests
            if arguments.Config['general']['save_output'] and bounding_method == 'alpha-crown':
                assert total_batches == 1
                arguments.Globals['out']['init_alpha_crown'] = batch_lb[self.final_name].cpu()

            batch_handler.add_batch_result(batch_lb, batch_ub, batch_lA, batch_alpha,
                batch_mask, batch_input_split_idx, A)


        # FIXME There is not only A here. There are also biases.
        # Need to rename.
        self.A_saved = A

        ####### Clip and Verify Domains Initialization #######
        if enable_clip_domains:
            from auto_LiRPA.not_in_release import not_in_release
            not_in_release("bab.clip_n_verify (domain clipping)")
        ###### Clip and Verify Domains Initialization END #######

        result = batch_handler.get_results(self.empty_history())

        return result['global_lb'], result

    def _set_A_options(self, bab=False, return_A=False):
        branching_args = arguments.Config['bab']['branching']
        input_and_act = branching_args['branching_input_and_activation']
        get_upper_bound = bab and arguments.Config['bab']['get_upper_bound']
        if get_upper_bound or input_and_act:
            self.needed_A_dict = defaultdict(set)
            self.needed_A_dict[self.net.output_name[0]].add(
                self.net.input_name[0])
        if self.nonlinear_split or return_A:
            self.needed_A_dict = defaultdict(set)
            for node in self.net.nodes():
                if node != self.net.final_name:
                    self.needed_A_dict[node.name].add(self.net.input_name[0])
        # FIXME just use "self.needed_A_dict is not None" without the extra "self.return_A"
        if self.needed_A_dict is not None:
            self.return_A = True
        else:
            self.return_A = False

    def empty_history(self):
        '''
        history: a tuple of tensors
            history[0]: relu_idx
            history[1]: relu_status
            history[2]: relu_bias
            history[3]: relu_score
            history[4]: depths
        '''
        if arguments.Config['bab']['branching']['input_split']['enable']:
            # For input split, we do not need to track the history.
            return None
        return {layer.name: ([], [], [], [], []) for layer in self.net.split_nodes}

    def set_crown_bound_opts(self, crown_name):
        solver_args = arguments.Config['solver']
        bab_args = arguments.Config['bab']
        crown_args = solver_args[f'{crown_name}-crown']
        opt_bound_args = {
            'reproduce_alpha_bounds': (
                solver_args['reproduce_alpha_bounds']),
            'lr_alpha': crown_args['lr_alpha'],
            'iteration': crown_args['iteration'],
            'lr_decay': crown_args['lr_decay'],
            'start_save_best': solver_args['start_save_best'],
            'loss_reduction_func': reduction_str2func(
                arguments.Config['general']['loss_reduction_func']),
        }
        if crown_name == 'alpha':
            opt_bound_args.update({
                'enable_alpha_crown': True, 'enable_beta_crown': False,
                'init_alpha': False, 'fix_interm_bounds': True,
                'use_shared_alpha': crown_args['share_alphas'],
                'early_stop_patience': solver_args['early_stop_patience'],
                'pruning_in_iteration': False,
                'max_time': crown_args['max_time'] * bab_args['timeout'],
            })
        elif crown_name == 'beta':
            opt_bound_args.update({
                'optimizer': crown_args['optimizer'],
                'lr_beta': crown_args['lr_beta'],
                'pruning_in_iteration': bab_args['pruning_in_iteration'],
                'pruning_in_iteration_threshold': bab_args['pruning_in_iteration_ratio'],
                'lr_cut_beta': bab_args['cut']['lr_beta'],
                'apply_output_constraints_to': [],
                'tighten_input_bounds': False,
                'directly_optimize': [],
                'share_gammas': False,
            })
        self.net.set_bound_opts({'optimize_bound_args': opt_bound_args})

    def alpha_drop_unused(self):
        optimizable_activations = self.net.get_enabled_opt_act()
        keep_nodes = self.alpha_start_nodes
        for m in optimizable_activations:
            m.drop_unused_alpha(keep_nodes)
    def _print_interm_bounds(self, lb=None, ub=None):
        if lb is not None:
            for k in lb:
                print(self.net[k])
                for out in self.net[k].output_name:
                    print('  ', self.net[out])
                print('  lower:', lb[k])
                if ub is not None and k in ub:
                    print('  upper:', ub[k])
                unstable = torch.logical_and(lb[k] < 0, ub[k] > 0).float()
                unstable = unstable.reshape(unstable.size(0), -1).sum(dim=-1)
                print('  unstable:', unstable, unstable.mean(), unstable.max())
        else:
            for node in self.net.nodes():
                if not node.perturbed:
                    continue
                if isinstance(node.lower, torch.Tensor):
                    print(node)
                    print('  lower:', node.lower.reshape(-1)[:10])
                    if isinstance(node.upper, torch.Tensor):
                        print('  upper:', node.upper.reshape(-1)[:10])
                        print(' Average gap:', (node.upper-node.lower).mean())

    from input_split.bounding import get_lower_bound_naive

class BatchHandler:
    """
    A helper class to handle batching of inputs and results in build() of LiRPANet.
    """
    def __init__(self, x: BoundedTensor, c, rhs, stop_criterion, or_spec_size, interm_bounds, 
                 batch_size_target, final_name, need_alphas, full_alpha_info,
                 reference_alphas=None, reference_lA=None, clip_in_alpha_crown=False):

        batch_size_ori = c.shape[0]
        same_x_range = (x[0:1] == x).all().item()
        # the batch size of x should be the same as c and rhs or 1 when the input range is the same.
        assert same_x_range or (x.shape[0] == batch_size_ori)

        if arguments.Config['general']['store_all_specs_on_cpu']:
            self.device = "cpu"
        else:
            self.device = arguments.Config['general']['device']

        self.x = x
        self.c = c
        self.rhs = rhs
        self.stop_criterion = stop_criterion
        self.or_spec_size = or_spec_size
        self.need_alphas = need_alphas
        self.full_alpha_info = full_alpha_info
        self.interm_bounds = interm_bounds
        self.batch_size_ori = batch_size_ori
        self.batch_size_target = batch_size_target
        self.final_name = final_name
        self.total_batches = (batch_size_ori + batch_size_target - 1) // batch_size_target
        self.reference_alphas = reference_alphas
        self.reference_lA = reference_lA

        # stop_criterion_general and stop_criterion_all are only used when
        # different OR specs are optimized together,
        # i.e., the original batch size should be exactly 1.
        if stop_criterion is stop_criterion_general or stop_criterion is stop_criterion_all:
            self.optimize_disjuncts_separately = False
            assert batch_size_ori == 1
        elif stop_criterion is stop_criterion_batch_any:
            self.optimize_disjuncts_separately = True

        if clip_in_alpha_crown:
            assert self.optimize_disjuncts_separately, (
                "clip_in_alpha_crown requires optimize_disjuncts_separately "
                "before batching disjunctive specifications. If this causes "
                "OOM, reduce build_batch_size to a smaller value."
            )

        # when the x range is not the same, the alpha can be different for different x or batches.
        # example: BoundRelu. alpha shape, alpha_lookup_idx and alpha_indices can be different,
        # when sparse alpha is enabled, making merging batches infeasible.
        # so we only support one batch when the x range is not the same.
        if need_alphas:
            assert same_x_range or self.total_batches == 1

        self.batch_lb, self.batch_ub = [], []
        self.batch_lA, self.batch_alpha = [], []
        self.batch_mask, self.batch_input_split_idx = [], []
        self.batch_A = []

    def take_batch(self, data, batch_idx, device=None, batch_dim=0):
        return take_batch(data, self.batch_size_target, batch_idx, device, batch_dim)

    def get_batch_input(self, batch_idx, device):
        batch_x = self.take_batch(self.x, batch_idx, device=device)
        batch_c = self.take_batch(self.c, batch_idx, device)
        batch_rhs = self.take_batch(self.rhs, batch_idx, device)
        actual_batch_size = batch_c.shape[0]


        # update decision_thresh used in stop_criterion_func to the current batch
        stop_criterion = self.stop_criterion
        batch_or_spec_size = None
        if self.or_spec_size is not None:
            if self.optimize_disjuncts_separately:
                batch_or_spec_size = self.take_batch(self.or_spec_size, batch_idx, device)
            else:
                batch_or_spec_size = self.or_spec_size
        if stop_criterion is stop_criterion_general:
            assert self.or_spec_size is not None
            batch_or_spec_size = self.or_spec_size
            stop_criterion_func = stop_criterion(batch_or_spec_size, batch_rhs)
        elif (stop_criterion is stop_criterion_batch_any) or (stop_criterion is stop_criterion_all):
            stop_criterion_func = stop_criterion(batch_rhs)
        else:
            raise ValueError(f'Unknown stop criterion function: {self.stop_criterion_name}')

        batch_interm_bounds = None
        if self.interm_bounds is not None:
            batch_interm_bounds = {}
            for k in self.interm_bounds:
                if k != self.final_name:
                    interm_bound_batch_size = self.interm_bounds[k][0].shape[0]
                    if interm_bound_batch_size == 1:
                        # if the interm bounds are shared before, we expand them to the batch size.
                        batch_interm_bounds[k] = [
                            expand_batch(self.interm_bounds[k][0], actual_batch_size, device),
                            expand_batch(self.interm_bounds[k][1], actual_batch_size, device)
                        ]
                    else:
                        # otherwise, the batch size of the interm bounds should be the same as the batch size of c and rhs,
                        # and we should take the batch size from the interm bounds.
                        assert interm_bound_batch_size == self.batch_size_ori
                        batch_interm_bounds[k] = [
                            self.take_batch(self.interm_bounds[k][0], batch_idx, device),
                            self.take_batch(self.interm_bounds[k][1], batch_idx, device)
                        ]

        batch_alphas = None
        if self.reference_alphas is not None:
            batch_alphas = {}
            # since we assume when alphas are needed,
            # the x range is the same, or the total batches is 1,
            # here we should take batch alpha values from the reference alphas
            # and just copy other items like alpha_lookup_idx, alpha_indices... if exist.
            for node_name in self.reference_alphas:
                batch_alphas[node_name] = {}
                for kk in self.reference_alphas[node_name]:
                    if kk == 'alpha':
                        # alpha values are different for different batches.
                        batch_alphas[node_name][kk] = {}
                        for start_node_name, alpha_value in self.reference_alphas[node_name][kk].items():
                            # [alpha_size, prod(start_node_shape), batch_size, *node_shape]
                            alpha_batch_dim = 2
                            alpha_batch_size = alpha_value.shape[alpha_batch_dim]
                            if alpha_batch_size == 1:
                                # if the alpha is shared before, we expand it to the batch size.
                                batch_alphas[node_name][kk][start_node_name] = expand_batch(
                                    alpha_value, actual_batch_size, device, alpha_batch_dim
                                )
                            else:
                                # otherwise, the batch size of the alpha should be the same as the batch size of c and rhs,
                                # and we should take the batch size from the alpha.
                                assert alpha_batch_size == self.batch_size_ori
                                batch_alphas[node_name][kk][start_node_name] = self.take_batch(
                                    alpha_value, batch_idx, device, alpha_batch_dim
                                )
                    else:
                        # other items are the same for all batches.
                        batch_alphas[node_name][kk] = self.reference_alphas[node_name][kk]

        # rhs will be used to prune the verified or specs.
        self.curr_rhs = batch_rhs

        return batch_x, batch_c, batch_rhs, stop_criterion_func, batch_interm_bounds, batch_alphas, batch_or_spec_size

    def add_batch_result(self, lb, ub, lA, alphas, mask, input_split_idx, A):
        device = self.device
        self.batch_lb.append(transfer_obj(lb, device))
        self.batch_ub.append(transfer_obj(ub, device))
        self.batch_lA.append(transfer_obj(lA, device))
        self.batch_alpha.append(transfer_obj(alphas, device))
        self.batch_mask.append(transfer_obj(mask, device))
        self.batch_input_split_idx.append(transfer_obj(input_split_idx, device))
        if A is not None:
            self.batch_A.append(transfer_obj(A, device))

    def get_results(self, history, ret_betas=None):
        lb = {k: torch.cat([item_lb[k] for item_lb in self.batch_lb])
              for k in self.batch_lb[0]}
        ub = {k: torch.cat([item_ub[k] for item_ub in self.batch_ub])
              for k in self.batch_ub[0]}
        lA = None
        if self.batch_lA[0] is not None:
            lA = {k: torch.cat([item_lA[k] for item_lA in self.batch_lA])
                for k in self.batch_lA[0]}

        if not self.need_alphas:
            alphas = None
        elif self.full_alpha_info:
            alphas = {}
            for k in self.batch_alpha[0]:
                alphas[k] = {}
                alphas[k]['alpha'] = {}
                for kk in self.batch_alpha[0][k]['alpha']:
                    alphas[k]['alpha'][kk] = torch.cat([item_alpha[k]['alpha'][kk] for item_alpha in self.batch_alpha], dim=2)
                for opt_key in self.batch_alpha[0][k]:
                    if opt_key != 'alpha':
                        # NOTE: assume all other keys are the same for all batches
                        # FIXME: assertion needed
                        alphas[k][opt_key] = self.batch_alpha[0][k][opt_key]
        else:
            alphas = {k: {kk: torch.cat([item_alpha[k][kk] for item_alpha in self.batch_alpha], dim=2)
                    for kk in self.batch_alpha[0][k]} for k in self.batch_alpha[0]}

        mask = {}
        for k in self.batch_mask[0]:
            for i in range(len(self.batch_mask[0][k])):
                if self.batch_mask[0][k][i] is None:
                    assert all(item_mask[k][i] is None for item_mask in self.batch_mask)
                    mask[k] = [None] * len(self.batch_mask[0][k])
                else:
                    mask[k] = [torch.cat([item_mask[k][i] for item_mask in self.batch_mask])
                            for i in range(len(self.batch_mask[0][k]))]

        input_split_idx = {k: torch.cat([item_input_split_idx[k] for item_input_split_idx in self.batch_input_split_idx])
                for k in self.batch_input_split_idx[0]}

        ret = {
            'mask': mask, 'lA': lA, 'lower_bounds': lb, 'upper_bounds': ub,
            'alphas': alphas, 'history': history,
            'input_split_idx': input_split_idx,
            'global_lb': lb[self.final_name],
            'global_ub': ub[self.final_name],
        }
        if ret['lA'] is None:
            ret['lA'] = self.reference_lA

        ret['betas'] = ret_betas

        return ret
