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
import math
import torch
from torch import Tensor
from typing import Tuple, Optional

import arguments

_BASE_CONSTRAINTS_UNSET = object()


@torch.no_grad()
def input_split_parallel(
        x_L: Tensor,
        x_U: Tensor,
        shape: tuple = None,
        split_depth: int = 1,
        i_idx: Optional[Tensor] = None,
        split_partitions: int = 2,
        split_hint: Optional[list[float]] = None,
) -> Tuple[Tensor, Tensor, int]:
    """
    Split the x_L and x_U given split_idx and split_depth.

    :param x_L:                 The lower bound on the inputs of the subdomains
    :param x_U:                 The upper bound on the inputs of the subdomains
    :param shape:
    :param split_depth:         How many splits we wish to consider for all subdomains where split_depth <= input_dim
    :param i_idx:               Input indices to split on for each batch
    :param split_partitions:    Partitions per node. split_partition=2 simply creates a binary tree.
    :param split_hint:          Only valid when split_partitions=2, the domains get split at split_hint rather than the
                                midpoint
                                (lb + ub)/2. This is beneficial when clipping domains.

    :return new_x_L:            The lower bound on the inputs of the new subdomains
    :return new_x_U:            The upper bound on the inputs of the new subdomains
    :return split_depth:        The depth of the split
    """
    # FIXME: this function should not be in this file.
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)

    x_L_cp = x_L.clone()
    x_U_cp = x_U.clone()

    split_depth = min(split_depth, i_idx.size(1))
    remaining_depth = split_depth
    input_dim = x_L.shape[1]

    if split_hint is not None:
        assert split_partitions == 2, "Can only handle split_hint with split_partitions==2"
        # convert split_hint to a tensor with length input_dim
        if len(split_hint) == 1:
            split_hint = torch.tensor(split_hint, device=x_L_cp.device, dtype=x_L_cp.dtype).expand(input_dim)
        else:
            assert len(split_hint) == input_dim, f"split_dim has dimension {len(split_hint)} when input_dim is {input_dim}"
            split_hint = torch.tensor(split_hint, device=x_L_cp.device, dtype=x_L_cp.dtype)

    while remaining_depth > 0:
        for i in range(min(input_dim, remaining_depth)):
            indices = torch.arange(x_L_cp.shape[0])
            copy_num = x_L_cp.shape[0]//x_L.shape[0]
            idx = i_idx[:,i].repeat(copy_num).long()

            has_crossing = None
            if split_hint is not None:
                # find dimensions that have a crossing at split_hint
                has_crossing = torch.logical_and(x_L_cp[indices, idx] < split_hint[idx], x_U_cp[indices, idx] > split_hint[idx])

            x_L_cp_list, x_U_cp_list = [], []
            for partition in range(split_partitions):
                x_L_cp_tmp = x_L_cp.clone()
                x_U_cp_tmp = x_U_cp.clone()

                lrange = ((partition + 1) * x_L_cp[indices, idx] +
                          (split_partitions - partition - 1) * x_U_cp[indices, idx]) / split_partitions
                urange = (partition * x_L_cp[indices, idx] +
                          (split_partitions - partition) * x_U_cp[indices, idx]) / split_partitions

                if split_hint is not None:
                    if partition == 0:
                        # creating upper subdomains with split_hint
                        new_l = torch.where(has_crossing, split_hint[idx], lrange)
                        new_u = urange
                    else:
                        # creating lower subdomains with split_hint
                        new_l = lrange
                        new_u = torch.where(has_crossing, split_hint[idx], urange)
                else:
                    new_l = lrange
                    new_u = urange

                x_L_cp_tmp[indices, idx] = new_l
                x_U_cp_tmp[indices, idx] = new_u

                x_L_cp_list.append(x_L_cp_tmp)
                x_U_cp_list.append(x_U_cp_tmp)

            x_L_cp = torch.cat(x_L_cp_list)
            x_U_cp = torch.cat(x_U_cp_list)

        remaining_depth -= min(input_dim, remaining_depth)

    split_depth = split_depth - remaining_depth

    new_x_L = x_L_cp.reshape(-1, *shape[1:])
    new_x_U = x_U_cp.reshape(-1, *shape[1:])

    return new_x_L, new_x_U, split_depth


def get_split_depth(current_domain_num, split_partitions=2):
    """Calculates the recursive split depth to meet a minimum batch size.

    This function determines the necessary depth of splitting required for a
    collection of a given size (`x_L`) to reach a target minimum size. The
    target size is configured externally as a ratio of a standard batch size.

    If the initial size `x_L` is already larger than the minimum target,
    the function returns a default depth of 1. Otherwise, it calculates
    the depth `d` such that `x_L * (split_partitions ** d)` is approximately
    equal to the minimum target batch size.

    Args:
        x_L: The initial size of the domain.
        split_partitions (int, optional): The number of partitions created at
            each split step. Defaults to 2.

    Returns:
        int: The computed split depth, guaranteed to be at least 1.
    """
    if current_domain_num == 0:
        return 1

    split_depth = 1
    min_batch_size_ratio = arguments.Config["solver"]["min_batch_size_ratio"]
    batch_size = arguments.Config["solver"]["batch_size"]
    min_batch_size = min_batch_size_ratio * batch_size

    if current_domain_num < min_batch_size:
        # This calculation determines the depth needed for the collection to grow
        # to the minimum batch size.
        # Note: The use of integer division `//` makes this an approximation.
        split_depth = int(math.log(min_batch_size // current_domain_num) // math.log(split_partitions))
        split_depth = max(split_depth, 1) # Ensure depth is at least 1

    return split_depth


def repeat_data_after_split(split_depth: int,
                            split_partitions: int,
                            cs: Optional[Tensor] = None,
                            thresholds: Optional[Tensor] = None,
                            dm_lb: Optional[Tensor] = None,
                            history = None,
                            spec_size: Optional[Tensor] = None,
                            lA: Optional[Tensor] = None,
                            lbias: Optional[Tensor] = None,
                            constraints: Optional[tuple] = None,
                            base_constraints: Optional[tuple] = _BASE_CONSTRAINTS_UNSET,
                            attack_counter: Optional[Tensor] = None,
                            split_counter: Optional[Tensor] = None,
)->Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[dict], Optional[Tensor],  Optional[Tensor], Optional[Tensor],  Optional[tuple], Optional[tuple], Optional[Tensor], Optional[Tensor]]:
    """
    Repeat the data after splitting the input domain.

    :param split_depth:         The depth of the split
    :param split_partitions:    Partitions per node. split_partition=2 simply creates a binary tree.
    :param cs:                  The C matrices of input domains
    :param thresholds:          The thresholds of input domains
    :param dm_lb:               The lower bounds of input domains
    :param history:             The history of the input domains
    :param spec_size:           The number of ANDs in the spec of input domains
    :param lA:                  The lA matrices of the input domains
    :param lbias:               The lbias values of the input domains
    :param lbias:               The lbias values of the input domains
    :param constraints:         The constraints tuple of the input domains
    :param attack_counter:      The attack_counter tensor
    :param split_counter:       The split_counter tensor
    
    :return repeated_cs
    :return repeated_thresholds
    :return repeated_dm_lb
    :return repeated_spec_size
    :return repeated_lA
    :return repeated_lbias
    :return r_constraints       The repeated constraints
    :return r_attack_counter    The repeated attack_counter
    :return r_split_counter     The repeated split_counter
    """
    new_batch_size = split_partitions ** split_depth
    def repeat_one_dim(data: Optional[Tensor], target_dim, repeat_count, shared=False):
        if data is None:
            return None
        # A tensor that is already a stride-0 broadcast along this axis (the spec
        # matrix is: one row shared by every domain) can be re-broadcast to the larger
        # size for free. Copying it instead cost 4.7 GiB of identical matrices per
        # split on fhn_10step at batch 449, which is what forced the high-output
        # instances down to batch 8-127.
        # `shared` data (the spec matrix, which no consumer writes into) with a single
        # row along the axis is the same case: the root has one copy, and repeating it
        # for the first split cost batch x (spec rows x outputs) -- 6.5 GiB on
        # pooling_sppc1stp -- when every consumer reads the broadcast anyway.
        if data.stride(target_dim) == 0 or (shared and data.shape[target_dim] == 1):
            # stride 0 means every slice along this axis is the same memory, so take
            # one and re-broadcast it to the larger size. `expand` only widens a
            # dimension of size 1, hence the narrow first.
            size = list(data.shape)
            size[target_dim] = data.shape[target_dim] * repeat_count
            one = data.narrow(target_dim, 0, 1)
            return one.expand(*size)
        return data.repeat(*(1 if i != target_dim else repeat_count for i in range(data.dim())))

    repeated_cs = repeat_one_dim(cs, 0, new_batch_size, shared=True)
    repeated_thresholds = repeat_one_dim(thresholds, 0, new_batch_size)
    repeated_dm_lb = repeat_one_dim(dm_lb, 0, new_batch_size)
    repeated_spec_size = repeat_one_dim(spec_size, 0, new_batch_size)
    if history is not None:
        history.repeat_for_split(new_batch_size, repeat_one_dim)
    repeated_lA = repeat_one_dim(lA, 0, new_batch_size)
    repeated_lbias = repeat_one_dim(lbias, 0, new_batch_size)
    r_constraints = None
    if constraints is not None:
        constraints_A, constraints_b = constraints
        repeated_cA = repeat_one_dim(constraints_A, 0, new_batch_size)
        repeated_cb = repeat_one_dim(constraints_b, 0, new_batch_size)
        r_constraints = (repeated_cA, repeated_cb)

    r_base_constraints = None
    include_base_constraints = base_constraints is not _BASE_CONSTRAINTS_UNSET
    if include_base_constraints and base_constraints is not None:
        base_A, base_b = base_constraints
        repeated_base_A = repeat_one_dim(base_A, 0, new_batch_size)
        repeated_base_b = repeat_one_dim(base_b, 0, new_batch_size)
        r_base_constraints = (repeated_base_A, repeated_base_b)

    r_attack_counter = None
    if attack_counter is not None:
        r_attack_counter = repeat_one_dim(attack_counter, 0, new_batch_size)

    r_split_counter = None
    if split_counter is not None:
        r_split_counter = repeat_one_dim(split_counter, 0, new_batch_size)
        r_split_counter += 1

    repeated = (
        repeated_cs,
        repeated_thresholds,
        repeated_dm_lb,
        repeated_spec_size,
        repeated_lA,
        repeated_lbias,
        r_constraints,
    )
    if include_base_constraints:
        return repeated + (
            r_base_constraints,
            r_attack_counter,
            r_split_counter,
        )
    return repeated + (
        r_attack_counter,
        r_split_counter,
    )

def input_split_and_repeat(
        x_L: Tensor,
        x_U: Tensor,
        current_domain_num: int,
        shape: tuple = None,
        i_idx: Optional[Tensor] = None,
        split_partitions: int = 2,
        split_hint: Optional[list[float]] = None,
        cs: Optional[Tensor] = None,
        thresholds: Optional[Tensor] = None,
        dm_lb: Optional[Tensor] = None,
        history = None,
        spec_size: Optional[Tensor] = None,
        lA: Optional[Tensor] = None,
        lbias: Optional[Tensor] = None,
        constraints: Optional[tuple] = None,
        base_constraints: Optional[tuple] = _BASE_CONSTRAINTS_UNSET,
        attack_counter: Optional[Tensor] = None,
        split_counter: Optional[Tensor] = None,
)->Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[dict], Optional[Tensor],
Optional[Tensor], Optional[Tensor], Optional[tuple], Optional[tuple], Optional[Tensor], Optional[Tensor]]:
    """
    Split the input domain and repeat the data.
    This function is the wrapped version of input_split_parallel and repeat_data_after_split.

    :param x_L:                 The lower bound on the inputs of the subdomains
    :param x_U:                 The upper bound on the inputs of the subdomains
    :param shape:
    :param i_idx:               Input indices to split on for each batch
    :param split_partitions:    Partitions per node. split_partition=2 simply creates a binary tree.
    :param split_hint:          Only valid when split_partitions=2, the domains get split at split_hint rather than the
                                midpoint
                                (lb + ub)/2. This is beneficial when clipping domains.
    :param cs:                  The C matrices of input domains
    :param thresholds:          The thresholds of input domains
    :param dm_lb:               The lower bounds of input domains
    :param spec_size:           The number of ANDs in the spec of input domains
    :param lA:                  The lA matrices of the input domains
    :param lbias:               The lbias values of the input domains
    :param constraints:         The constraints tuple of the input domains
    :param attack_counter:      The attack_counter tensor
    :param split_counter:       The split_counter tensor
                                
    :return new_x_L:            The lower bound on the inputs of the new subdomains
    :return new_x_U:            The upper bound on the inputs of the new subdomains
    :return repeated_cs:        The repeated cs
    :return repeated_thresholds: The repeated thresholds
    :return repeated_dm_lb:     The repeated dm_lb
    :return repeared_spec_sice: The repeated spec size
    :return repeated_lA:        The repeated lA
    :return repeated_lbias:     The repeated lbias
    :return r_constraints:      The repeated constraints
    :return r_attack_counter:   The repeated attack_counter
    :return r_split_counter:    The repeated split_counter
    """
    split_depth = get_split_depth(current_domain_num, split_partitions=split_partitions)
    new_x_L, new_x_U, split_depth = input_split_parallel(
        x_L, x_U, shape, split_depth, i_idx, split_partitions, split_hint)

    include_extended = (
        base_constraints is not _BASE_CONSTRAINTS_UNSET
        or attack_counter is not None
        or split_counter is not None
    )
    repeat_kwargs = {}
    if base_constraints is not _BASE_CONSTRAINTS_UNSET:
        repeat_kwargs["base_constraints"] = base_constraints
    (
        repeated_cs,
        repeated_thresholds,
        repeated_dm_lb,
        repeated_spec_size,
        repeated_lA,
        repeated_lbias,
        r_constraints,
        *extended,
    ) = repeat_data_after_split(
        split_depth,
        split_partitions,
        cs,
        thresholds,
        dm_lb,
        history,
        spec_size,
        lA,
        lbias,
        constraints,
        attack_counter=attack_counter,
        split_counter=split_counter,
        **repeat_kwargs,
    )
    result = (
        new_x_L,
        new_x_U,
        repeated_cs,
        repeated_thresholds,
        repeated_dm_lb,
        repeated_spec_size,
        repeated_lA,
        repeated_lbias,
        r_constraints,
    )
    if include_extended:
        if base_constraints is not _BASE_CONSTRAINTS_UNSET:
            r_base_constraints, r_attack_counter, r_split_counter = extended
        else:
            r_attack_counter, r_split_counter = extended
            r_base_constraints = None
        return result + (
            r_base_constraints,
            r_attack_counter,
            r_split_counter,
        )
    return result
