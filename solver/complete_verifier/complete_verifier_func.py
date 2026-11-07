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
"""complete verifier main interface."""

import time
import gc
import torch

import arguments
from utils import Logger
from specifications import BatchedSpecs
from input_split.batch_branch_and_bound import input_bab_parallel

from typing import TYPE_CHECKING

from solving_utils import load_fixed_primal

if TYPE_CHECKING:
    from abcrown import ABCROWN


def _share_spec_matrix(lirpa_model, c):
    """The incomplete verifier's bound model built this same matrix (objective row on top
    of the specs) for its own bound computations; hold one copy of the (spec rows x
    outputs) matrix, 3.3 GiB on pooling_sppc1stp, not two."""
    shared = getattr(lirpa_model, "c", None)
    if (torch.is_tensor(shared) and shared.shape == c.shape and shared.dtype == c.dtype
            and shared.device == c.device and torch.equal(shared, c)):
        return shared
    return c


def bab(self: 'ABCROWN',
        reference_dict=None,
        model=None,
        timeout=None,
        return_domains=False,
        max_iterations=None,
        index=None
        ):

    if arguments.Config['general']['store_all_specs_on_cpu']:
        device = "cpu"
    else:
        device = arguments.Config['general']['device']
    all_specs: BatchedSpecs = self.vnnlib_handler.all_specs
    x, c, rhs, or_spec_size, _, _ = all_specs.get(device)

    input_split = arguments.Config['bab']['branching']['input_split']['enable']
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
        new_rhs = torch.zeros((1, 1), device=rhs.device, dtype=rhs.dtype)
        if not input_split:
            fixed_primal = load_fixed_primal(rhs)
            if fixed_primal is not None:
                new_rhs[0, 0] = fixed_primal
                print(
                    "[GenBaB dual] Fixed primal objective: "
                    f"{fixed_primal.item():.12g}"
                )
        rhs = torch.cat((new_rhs, rhs), dim=1)

    vnnlib = self.vnnlib_handler.vnnlib

    # initialize the model.
    # Reuse the model produced by incomplete verification.
    assert reference_dict
    lirpa_model = reference_dict.get('model', None)
    assert lirpa_model is not None
    if solving_mode:
        c = _share_spec_matrix(lirpa_model, c)

    # save the output of the model from the last input
    # it is just for comparing with reference in tests
    output = lirpa_model.net(x[-1:].to(lirpa_model.device)).flatten()
    print('Model prediction is:', output)
    if arguments.Config['general']['save_output']:
        arguments.Globals['out']['pred'] = output.cpu()

    if input_split:
        result = input_bab_parallel(
            lirpa_model, x, c, rhs, or_spec_size,
            reference_dict=reference_dict,
            timeout=timeout, max_iterations=max_iterations,
            vnnlib=vnnlib,
            return_domains=return_domains,
            index=index
        )
        if return_domains:
            return result
    else:
        from auto_LiRPA.not_in_release import not_in_release
        not_in_release("activation-space BaB (only input-space splitting is part of this release)")

    # return global lb, number of domains visited, and status
    return result

def complete_verifier(
        self: 'ABCROWN', model_ori,
        index, timeout_threshold, bab_ret=None,
        reference_dict=None
):
    start_time = time.time()

    torch.cuda.empty_cache()
    gc.collect()

    # FIXME: double check the implementation of enable_opt_interm_bounds
    # the current version does not imporve the result.
    assert not arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']

    all_specs: BatchedSpecs = self.vnnlib_handler.all_specs

    print(f'###### Verifying property Instance {index} ######')
    all_specs.print_stats()
    timeout = timeout_threshold - (time.time() - start_time)
    print(f'Remaining timeout: {timeout}')
    start_time_bab = time.time()

    # Complete verification (BaB, BaB with refine, or MIP).
    l, num_domains_visited, ret = self.bab(
        reference_dict=reference_dict,
        model=model_ori,
        timeout=timeout,
        index=index,
    )
    bab_ret.append([index, l, num_domains_visited, time.time() - start_time_bab])

    timeout = timeout_threshold - (time.time() - start_time)

    if ret == 'unsafe':
        return 'unsafe-bab'
    elif ret == 'unknown' or timeout < 0:
        return 'unknown'
    elif ret != 'safe':
        raise ValueError(f'Unknown return value of bab: {ret}')
    return 'safe'
