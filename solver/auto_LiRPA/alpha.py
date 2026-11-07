#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################


"""
# This file contains functions to handle alphas in the LiRPANet.
# It includes functions to drop unused alphas, get alphas, and set alphas.
"""

import torch
from .utils import transfer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule

def drop_unused_alpha(self: 'BoundedModule'):
    optimizable_activations = self.get_enabled_opt_act()
    keep_nodes = self.alpha_start_nodes
    for m in optimizable_activations:
        m.drop_unused_alpha(keep_nodes)

def get_alpha(self: 'BoundedModule', get_all=True, half=False, device=None, full_info=False, drop_unused=True):
    """
    Saves alphas in a dictionary.

    :param get_all: if True, saves all alphas, otherwise only those which will be optimized.
    :param half: if True, saves alphas in half precision.
    :param device: device to transfer alphas to.
    :param full_info: if True, saves full alpha information as a dict, otherwise only the alpha values.
    :param drop_unused: if True, drops unused alphas before saving.
    :return: dictionary with alphas.
    :rtype: dict
    :note: the dictionary is structured as follows:
            When `full_info` is True:
                {node_name: {'alpha': {start_node_name: alpha_value},
                             'alpha_lookup_idx': {start_node_name: alpha_lookup_idx}, ...}}
            When `full_info` is False:
                {node_name: {start_node_name: alpha_value}}
    """
    if drop_unused and not get_all:
        self.drop_unused_alpha()

    # alpha has size (2, spec, batch, *shape). When we save it,
    # we make batch dimension the first.
    # spec is some intermediate layer neurons, or output spec size.
    ret = {}
    dtype = None if not half else torch.float16
    optimizable_activations = self.get_enabled_opt_act()
    for m in optimizable_activations:
        if full_info:
            ret[m.name] = m.dump_alpha(device=device, dtype=dtype)
        else:
            ret[m.name] = {}
            for spec_name, alpha in m.alpha.items():
                if get_all or spec_name in self.alpha_start_nodes:
                    ret[m.name][spec_name] = transfer(alpha, device, dtype=dtype)
    return ret

def set_alpha(self: 'BoundedModule', alpha, set_all=False, full_info=False, device=None, dtype=None):
    """
    Sets alphas from a dictionary. It is a reverse operation of `get_alpha`.

    :param alpha: dictionary with alphas.
    :param set_all: if True, sets all alphas, otherwise only those which will be optimized.
    :param full_info: if True, alpha is a dict with full information, otherwise it is a tensor.
    :param device: device to transfer alphas to.
    :param dtype: dtype to transfer alphas to.
    """
    # If alpha is empty, we do nothing.
    if len(alpha) == 0:
        return

    # if set_all is True, we set all alphas from input alpha,
    # otherwise we only set alphas which will be further optimized.

    optimizable_activations = self.get_enabled_opt_act()
    keep_nodes = self.alpha_start_nodes
    for m in optimizable_activations:
        if full_info:
            m.restore_alpha(alpha[m.name], device=device, dtype=dtype)
        else:
            # Domain storage may reduce alpha precision (float16); optimization
            # resumes with the dtype owned by the computation graph, on the
            # graph device (upstream #816 restores alpha with the output dtype).
            restore_dtype = dtype if dtype is not None else m.output_dtype
            restore_device = device if device is not None else self.device
            for spec_name in list(m.alpha.keys()):
                if (spec_name in alpha[m.name]) and (spec_name in keep_nodes or set_all):
                    m.alpha[spec_name] = alpha[m.name][spec_name].to(
                        device=restore_device, dtype=restore_dtype)
                    # Duplicate for the second half of the batch.
                    m.alpha[spec_name] = m.alpha[spec_name].detach().requires_grad_()

    if not set_all:
        # If we are not setting all alphas, we need to drop unused alphas.
        # in most cases, set_all is False and dropping should be done before, here just in case.
        self.drop_unused_alpha()

def init_alpha(self: 'BoundedModule', x, share_alphas=False, method='backward',
                c=None, bound_lower=True, bound_upper=True, final_node_name=None,
                interm_bounds=None, reference_alphas=None,
                skip_bound_compute=False, final_layer_alpha_only=False):
    self(*x) # Do a forward pass to set perturbed nodes
    final = (self.final_node() if final_node_name is None
                else self[final_node_name])
    self._update_bound_ancestors(final)

    optimizable_activations = self.get_enabled_opt_act()
    for node in optimizable_activations:
        # TODO(7/6/2023) In the future, we may need to enable alpha sharing
        # automatically by consider the size of all the optimizable nodes in the
        # graph. For now, only an adhoc check in MatMul is added.
        node._all_optimizable_activations = optimizable_activations

        # initialize the parameters
        node.opt_init()

    apply_output_constraints_to = (
        self.bound_opts['optimize_bound_args']['apply_output_constraints_to']
    )
    if (not skip_bound_compute or interm_bounds is None or
            reference_alphas is None or not all(
                [act.name in reference_alphas
                    for act in optimizable_activations])):
        skipped = False
        # if new interval is None, then CROWN interval is not present
        # in this case, we still need to redo a CROWN pass to initialize
        # lower/upper
        with torch.no_grad():
            # We temporarilly deactivate output constraints
            self.bound_opts['optimize_bound_args']['apply_output_constraints_to'] = []
            l, u = self.compute_bounds(
                x=x, C=c, method=method, bound_lower=bound_lower,
                bound_upper=bound_upper, final_node_name=final_node_name,
                interm_bounds=interm_bounds)
            self.bound_opts['optimize_bound_args']['apply_output_constraints_to'] = (
                apply_output_constraints_to
            )
            if len(apply_output_constraints_to) > 0:
                # Some layers, such as the BoundTanh layer, do some of their initialization
                # in the forward pass. We need to call the forward pass again to ensure
                # that they are initialized for the output constraints, too.
                l, u = self.compute_bounds(
                    x=x, C=c, method=method, bound_lower=bound_lower,
                    bound_upper=bound_upper, final_node_name=final_node_name,
                    interm_bounds=interm_bounds, cache_bounds=True)
    else:
        # we skip, but we still would like to figure out the "used",
        # "perturbed", "backward_from" of each note in the graph
        skipped = True
        # this set the "perturbed" property
        self.set_input(*x, interm_bounds=interm_bounds)
        self.backward_from = {node: [final] for node in self._modules}
        l = u = None

    final_node_name = final_node_name or self.final_name

    init_intermediate_bounds = {}
    splittable_activations = set(self.get_splittable_activations())
    for node in optimizable_activations:
        start_nodes = []
        if method in ['forward', 'forward+backward']:
            start_nodes.append(('_forward', 1, None, False))
        if method in ['backward', 'forward+backward']:
            start_nodes += self.get_alpha_crown_start_nodes(
                node,
                c=c,
                share_alphas=share_alphas,
                final_node_name=final_node_name,
            )
        # When final_layer_alpha_only is True (e.g., when enable_beta and
        # bound_prop_method is alpha-crown), only initialize alpha for the
        # final layer to save memory and computation.
        if final_layer_alpha_only:
            start_nodes = [s for s in start_nodes if s[0] == final_node_name]
        if not start_nodes:
            continue
        if skipped:
            node.restore_alpha(reference_alphas[node.name], device=x[0].device, dtype=x[0].dtype)

        else:
            node.init_opt_parameters(start_nodes)
        if node in splittable_activations:
            for i, input_node in enumerate(node.inputs):
                if (not input_node.perturbed
                        or node.inputs[i].lower is None
                        and node.inputs[i].upper is None):
                    continue
                init_intermediate_bounds[node.inputs[i].name] = (
                    [node.inputs[i].lower.detach(),
                    node.inputs[i].upper.detach()])
    if (
        apply_output_constraints_to is not None
        and len(apply_output_constraints_to) > 0
        and hasattr(self, 'constraints')
    ):
        # self.constraints.shape = (batch_size, num_constraints, num_output_neurons)
        # For abCROWN we know that:
        # If the output constraints are a conjunction, the shape is (1, num_constraints, *)
        # If the output constraints are a disjunction, the shape is (num_constraints, 1, *)
        # Checking which entry is 1 allows to discern both cases.
        # If auto_LiRPA is used directly, we could have batches of inputs with more than one
        # constraint. This is currently not supported.
        if self.constraints.size(0) == 1:
            num_gammas = self.constraints.size(1)
        elif self.constraints.size(1) == 1:
            num_gammas = self.constraints.size(0)
        else:
            raise NotImplementedError(
                'To use output constraints, either have a batch size of 1 or use only one '
                'output constraint'
            )
        for node in self.nodes():
            if not node.are_output_constraints_activated_for_layer(
                    apply_output_constraints_to):
                continue
            node.init_gammas(
                num_gammas,
                dtype=self.get_output_dtype(node),
                device=self.device,
            )

    if self.bound_opts['verbosity'] >= 1:
        print('Optimizable variables initialized.')
    if skip_bound_compute:
        return init_intermediate_bounds
    else:
        return l, u, init_intermediate_bounds
