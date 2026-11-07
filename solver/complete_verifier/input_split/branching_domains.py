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
import torch.nn.functional as F
import arguments
from torch import Tensor
from typing import Optional, Union, Tuple, List
from tensor_storage import get_tensor_storage, StackTensorStorage
from abc import ABC, abstractmethod
import psutil, os

from auto_LiRPA.concretize_func import drop_redundant_constraints
from auto_LiRPA.utils import select_rows
from history import SplitHistory, ensure_split_history
from input_constraints import affine_constraints_domain_lower_bound, drop_prefix_constraints
from input_split.utils import bound_thresholds
from utils import pad_list_of_input_to_tensor


class InputDomainList(ABC):
    """
    Abstract class that maintains a list of domains for input split.
    """
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        """Number of domains remaining in the list"""
        ...

    @abstractmethod
    def __getitem__(self, *args, **kwargs):
        """
        get lb, dm_l, dm_u, cs, threshold
        (and possibly more element) for idx
        """
        ...

    @abstractmethod
    def add(self, *args, **kwargs):
        """Add domains to the list"""
        ...

    @abstractmethod
    def pick_out_batch(self, batch_size, device=None):
        """Pick out a batch of subdomains from the domain list."""
        ...

    @staticmethod
    def get_topk_indices(self, k=1, largest=False):
        """get the topk indices, by default worst k"""
        ...


class UnsortedInputDomainList(InputDomainList):
    """Unsorted domain list for input split."""

    def __init__(
        self,
        storage_depth: int,
        output_device: str,
        history=None,
        sort_index: Optional[int]=None,
        sort_heuristic: Optional[str]="lb_rhs",
        sort_descending: bool=True,
        use_split_idx: bool=True,
        use_alpha: Optional[bool] = None,
        use_beta: Optional[bool] = None,
    ):
        """
        The initialization method for the UnsortedInputDomainList class.
        
        :param storage_depth:       The maximum number of splits we could use in input BaB
        :param output_device:       The default device storing accessed data. Complete domain data is always stored on 'cpu'.
        :param use_alpha:           True if we must also store alpha parameters
        :param sort_index:          The index along which to sort the domains
        :param sort_descending:     If True, domains will get sorted in descending order w.r.t. their lower bounds
                                    whenever the 'sort' method is called.
        :param use_split_idx:       If True, we also store the split indices for each domain
        """
        super(UnsortedInputDomainList, self).__init__()
        self.output_device = output_device
        self.lb = None
        self.dm_l = None
        self.dm_u = None
        self.alpha = {}
        self.beta = []
        use_alpha = bool(use_alpha) if use_alpha is not None else bool(getattr(history, "use_alpha", False))
        use_beta = bool(use_beta) if use_beta is not None else bool(getattr(history, "use_beta", False))
        self.history = ensure_split_history(history, use_alpha=use_alpha, use_beta=use_beta)
        self.use_alpha = self.history.use_alpha
        self.use_beta = self.history.use_beta
        self._legacy_pickout_api = history is None
        self.sort_index = sort_index
        self.sort_heuristic = sort_heuristic
        # `cs` is the specification matrix. In input split it is the same matrix
        # for every domain -- splitting the input box does not change the spec --
        # and it is the largest per-domain tensor there is: 1113x1113 on fhn_10step,
        # 2047x2047 on ch100d. Storing one copy per domain asked for 274 GB of host
        # memory. Keep a single shared row and broadcast it in `pick_out_batch`;
        # `UnsortedMultiSpecInputDomainList` is the class for genuinely varying specs.
        self.cs = None
        self.threshold = None
        self.constraint_A = None
        self.constraint_b = None
        # The base (problem-level) input constraints are identical for every
        # domain: they are created once at the root and only ever broadcast over
        # the batch by `repeat_one_dim`. Storing one copy per domain made the
        # domain list grow at O(#domains x #rows x #inputs) and asked for 274 GB
        # of host memory on the 100-dimensional instance. Keep a single shared
        # row instead and broadcast it back out in `pick_out_batch`.
        self.base_constraint_A = None
        self.base_constraint_b = None
        self.split_idx = None
        self.certificate_scope_refs = None
        self.storage_depth = storage_depth
        self.sort_descending = sort_descending
        self.volume = self.all_volume = None
        self.use_split_idx = use_split_idx
        self.spec_size = None
        # attack_counter records the total frequencies that a domain is selected to be attacked, designed specially for solving mode
        self.attack_counter = None
        # split_counter records, per domain, the splits on its path since the last split of its widest input (solving mode;
        # input_split.branching_heuristics._exhaustive_subdivision_guard). It is not the depth from the root.
        self.split_counter = None
        # x_star stores the relaxation minimizer for warm-starting PGD attacks
        self.x_star = None
        # Lazy sort (bab.lazy_sort_domains): rows [0, _lazy_head) are in the order a full
        # sort would give them, but only on paper; pick_out_batch puts the rows it is
        # about to pop in place. _lazy_key freezes the sort parameters of that sort.
        try:
            self.lazy_sort = bool(arguments.Config["bab"].get("lazy_sort_domains", False))
        except Exception:  # pylint: disable=broad-except
            self.lazy_sort = False
        self._lazy_head = None
        self._lazy_key = None
        # Contiguous copy of the objective column lb[:, 0] when lb has several columns
        # (solving mode: objective + constraint rows). The per-iteration worst-domain
        # topk and the sort keys read it instead of a strided column of lb: 11M domains
        # x 213 columns is 700 MB of cache lines for 88 MB of keys.
        self.lb_obj = None
        # bab.domain_storage_gpu_mb: keep the per-domain storages on the GPU while they
        # take less than this many MB (pick_out_batch and add then move nothing across
        # PCIe); past it they move to host memory for good. 0 keeps them on the host.
        self._storage_device = "cpu"
        self._gpu_budget = 0
        try:
            budget_mb = int(arguments.Config["bab"].get("domain_storage_gpu_mb", 0) or 0)
            if (budget_mb > 0 and torch.device(output_device).type == "cuda"
                    and arguments.Config["bab"]["tree_traversal"] == "depth_first"
                    and not self.use_alpha and not self.use_beta):
                self._storage_device = output_device
                self._gpu_budget = budget_mb * 1024 ** 2
        except Exception:  # pylint: disable=broad-except
            pass

    def _enforce_gpu_budget(self, growth=1):
        """Move every per-domain storage to host memory once they outgrow the budget.

        `growth=2` before an append: a full storage doubles its capacity when it grows,
        so moving then keeps even the transient reallocation within the budget.
        """
        if self._storage_device == "cpu":
            return
        stores = self._per_domain_storages()
        allocated = sum(st._storage.numel() * st._storage.element_size() for st in stores)
        if allocated * growth <= self._gpu_budget:
            return
        for st in stores:
            st.move_to("cpu")
        if self.x_star is not None and self.x_star not in stores:
            self.x_star.move_to("cpu")
        self._storage_device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _take(self, st, batch_size, device):
        """Pop `batch_size` rows as a tensor on `device` that owns its memory."""
        rows = st.pop(batch_size)
        if rows.device == torch.device(device) or (rows.is_cuda and torch.device(device).type == "cuda"):
            # A view of GPU storage: the next append would overwrite it.
            return rows.clone()
        return rows.to(device=device, non_blocking=True)

    def __len__(self):
        if self.dm_l is None:
            return 0
        return self.dm_l.num_used

    def _broadcast_cs(self, batch_size, device):
        """Broadcast the shared spec matrix to `batch_size` domains, without copying.

        `expand` gives a stride-0 view, so the whole batch costs one row of storage.
        Move to the target device first: `.to()` on an expanded tensor would
        materialise every copy. Nothing writes into `cs` -- every consumer indexes or
        slices it -- so sharing the storage is safe.
        """
        shared = self.cs.to(device, non_blocking=True)
        return shared.expand(batch_size, *shared.shape[1:])

    def __getitem__(self, idx):
        # convert idx to tensor on cpu for slicing.
        if isinstance(idx, slice):
            idx = torch.arange(len(self), device="cpu")[idx]
        else:
            idx = torch.as_tensor(idx, device="cpu").view(-1)
        assert idx.shape[0] > 0, "Empty index"
        output_device = self.output_device
        # Only return x_star if it exists and has matching size (may not exist for initial domains)
        x_star_out = None
        if self.x_star is not None and self.x_star.num_used == self.lb.num_used:
            x_star_out = self.x_star._storage[idx].to(output_device)
        return (
            self.lb._storage[idx].to(output_device),
            self.dm_l._storage[idx].to(output_device),
            self.dm_u._storage[idx].to(output_device),
            self._broadcast_cs(idx.shape[0], output_device),
            self.threshold._storage[idx].to(output_device),
            torch.tensor(self.spec_size, device=output_device, dtype=torch.int32).expand(idx.shape[0]),
            x_star_out,
        )

    @staticmethod
    def filter_verified_domains(
            batch: int,
            lb: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            cs: Tensor,
            threshold: Tensor,
            history=None,
            lA: Optional[Tensor] = None,
            lbias: Optional[Tensor] = None,
            constraints: Optional[tuple] = None,
            base_constraints: Optional[tuple] = None,
            split_idx: Optional[Tensor] = None,
            spec_sizes: Optional[Tensor] = None,
            check_dm_lbs: bool = True,
            check_input_boxes: bool = True,
            remaining_index: Optional[Tensor] = None,
            attack_counter: Tensor = None,
            split_counter: Tensor = None,
            x_star: Optional[Tensor] = None
    ) -> Tuple[int, Tensor, Tensor, Tensor, dict,
    Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[tuple], Optional[tuple], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Filters out the domains that are verified and only returns unverified domains

        :param batch:                                   Batch size of domains.
        :param lb: (batch, spec_dim)                    Domain lower bound output
        :param dm_l: (batch, dim_in)                    Input domain lower input bound
        :param dm_u: (batch, dim_in)                    Input domain upper input bound
        :param alpha:                                   CROWN alpha parameters for domains
        :param cs: (batch, spec_dim, lA_rows)           specification matrix
        :param threshold: (batch, spec_dim)             Threshold to verify specification with
        :param lA: (batch, lA_rows or spec_dim, dim_in) CROWN lA coefficient matrix
        :param lbias: (batch, spec_dim)                 CROWN lbias coefficient matrix
        :param constraints:                             A tuple of linear constraints (constr_A, constr_b)
        :param split_idx: (batch, num of splits)        Specifies along which dimensions to split
        :param spec_sizes: (batch)                      The number of ANDs in the spec of each domain
        :param check_dm_lbs:                            If true, filters out domains that have been verified
                                                        by lb > threshold
        :param check_input_boxes:                       If true, filters out domains that have been verified
                                                        by dm_l > dm_u
        :param remaining_index:                         If not None, user is specifying which domains are unverified
        :param attack_counter:                          The attack_counter tensor
        :param split_counter:                           The split_counter tensor
        :param x_star: (batch, dim_in)                  Relaxation minimizer for warm-starting PGD attacks

        :return batch_filt:                             Number of filtered domains.
        :return lb_filt:                                Filtered domain lower bound output
        :return dm_l_filt:                              Filtered input domain lower input bound
        :return dm_u_filt:                              Filtered input domain upper input bound
        :return alpha_filt:                             Filtered CROWN alpha parameters for domains
        :return cs_filt:                                Filtered specification matrix
        :return threshold_filt:                         Filtered threshold to verify specification with
        :return lA_filt:                                Filtered CROWN lA coefficient matrix
        :return lbias_filt:                             Filtered CROWN lbias coefficient matrix
        :return constraints_filt:                       Filtered linear constraints tuple (constr_A, constr_b)
        :return split_idx_filt:                         Filtered specifies along which dimensions to split
        :return spec_size_filt:                         Filtered the number of ANDs in the spec of each domain
        :return attack_counter_filt:                    Filtered attack_counter
        :return split_counter_filt:                     Filtered split_counter
        :return x_star_filt:                            Filtered relaxation minimizer
        """
        legacy_mode = False
        if isinstance(cs, dict) and torch.is_tensor(threshold) and torch.is_tensor(history):
            legacy_alpha = cs
            cs = threshold
            threshold = history
            history = ensure_split_history(
                None,
                use_alpha=bool(legacy_alpha),
                use_beta=False,
                alphas=legacy_alpha,
            )
            legacy_mode = True

        if remaining_index is None:
            remaining_index = UnsortedInputDomainList.get_remaining_index(
                batch,
                lb,
                threshold,
                dm_l,
                dm_u,
                check_dm_lbs,
                check_input_boxes,
                base_constraints=base_constraints,
                constraints=constraints,
                rowpool=getattr(history, 'rowpool', None),
            )
        lb_filt = lb[remaining_index]
        dm_l_filt = dm_l[remaining_index]
        dm_u_filt = dm_u[remaining_index]
        cs_filt = cs[remaining_index]
        batch_filt = len(dm_l_filt)
        if history is not None and hasattr(history, "filter_opt_variables"):
            history.filter_opt_variables(remaining_index, batch_filt)

        threshold_filt = threshold[remaining_index]
        lA_filt = lA[remaining_index] if lA is not None else None
        lbias_filt = lbias[remaining_index] if lbias is not None else None

        constraints_filt = None
        if constraints is not None:
            c_A, c_b = constraints
            if isinstance(remaining_index, slice):
                c_A_filt = c_A[remaining_index]
                c_b_filt = c_b[remaining_index]
            else:
                c_A_filt = select_rows(c_A, remaining_index)
                c_b_filt = select_rows(c_b, remaining_index)
            constraints_filt = (c_A_filt, c_b_filt)

        base_constraints_filt = None
        if base_constraints is not None:
            base_A, base_b = base_constraints
            if isinstance(remaining_index, slice):
                base_constraints_filt = (
                    base_A[remaining_index],
                    base_b[remaining_index],
                )
            else:
                base_constraints_filt = (
                    select_rows(base_A, remaining_index),
                    select_rows(base_b, remaining_index),
                )

        attack_counter_filt = None
        if attack_counter is not None:
            attack_counter_filt = attack_counter[remaining_index]

        split_counter_filt = None
        if split_counter is not None:
            split_counter_filt = split_counter[remaining_index]

        split_idx_filt = split_idx[remaining_index] if split_idx is not None else None
        spec_sizes_filt = spec_sizes[remaining_index] if spec_sizes is not None else None

        x_star_filt = x_star[remaining_index] if x_star is not None else None

        if legacy_mode:
            alpha_filt = history.alphas if history is not None else {}
            return (
                batch_filt,
                lb_filt,
                dm_l_filt,
                dm_u_filt,
                alpha_filt,
                cs_filt,
                threshold_filt,
                lA_filt,
                lbias_filt,
                constraints_filt,
                split_idx_filt,
                spec_sizes_filt,
            )

        return (
            batch_filt,
            lb_filt,
            dm_l_filt,
            dm_u_filt,
            cs_filt,
            threshold_filt,
            lA_filt,
            lbias_filt,
            constraints_filt,
            base_constraints_filt,
            split_idx_filt,
            spec_sizes_filt,
            attack_counter_filt,
            split_counter_filt,
            x_star_filt,
        )

    @staticmethod
    def get_remaining_index(
            batch: int,
            lb: Tensor,
            threshold: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            check_dm_lbs=True,
            check_input_boxes=True,
            base_constraints: Optional[tuple] = None,
            constraints: Optional[tuple] = None,
            rowpool=None,
    ) -> Union[slice, Tensor]:
        """
        Gets the indices of the batch instances that are not verified. Verification conditions are specified by
        the check_dm_lbs and check_input_boxes flags. If both are None, all indicies are returned.

        @param batch:                       Batch size of domains
        @param lb: (batch, spec_dim)        Domain lower bound output
        @param threshold: (batch, spec_dim) Threshold to verify specification with
        @param dm_l: (batch, dim_in)        Input domain lower input bound
        @param dm_u: (batch, dim_in)        Input domain upper input bound
        @param check_dm_lbs:                If true, filters out domains that have been verified by lb > threshold
        @param check_input_boxes:           If true, filters out domains that have been verified by dm_l > dm_u
        @param base_constraints:            Original standard-form input constraints A x + b <= 0
        @param constraints:                 Current standard-form domain constraints for concretization
        @param rowpool:                     The instance's RowPool (solving mode): the constraint rows are compared
                                            at its relaxed thresholds

        @return:                            The indices of the batch instances that are left unverified
        """
        solving_mode = arguments.Config["solving"]["solving_mode"]
        if solving_mode:
            # TODO: We assume that the lb has shape (batch_size, 1+n_constr)
            # Be careful for keeping this statistics
            batch_size = lb.shape[0]
        remaining_mask = torch.ones(batch, dtype=torch.bool, device=lb.device)
        if check_dm_lbs:
            if solving_mode and lb.ndim >= 2 and lb.shape[1] > 1:
                # Equality constraints arrive as paired inequalities whose true
                # optimum sits exactly on the threshold, so FP rounding (~1e-16)
                # can push a valid lower bound just above it. Require a margin
                # consistent with the primal feasibility tolerance before
                # declaring a domain constraint-infeasible.
                threshold = bound_thresholds(rowpool, threshold)
            # A pruning margin was tried here so the domain holding the optimum would
            # survive and the per-domain check could keep observing it. It was reverted:
            # keeping those near-degenerate domains alive let them be split further until
            # the bound computation produced NaN, and it destroyed the one genuinely
            # clean verification (ns_cavity went from witness present 44/44 and the cap
            # never binding, to crashing with the witness lost in 31 of 75 iterations).
            # The limit is real rather than incidental: once the incumbent equals the
            # optimum, the domain containing it sits exactly on the pruning boundary.
            lb_gt = torch.isfinite(lb) & torch.isfinite(threshold) & (lb > threshold)
            if solving_mode and lb.ndim >= 2 and lb.shape[1] > 0:
                # A verified incumbent already witnesses objective == P. Search
                # only for strict improvements: cutoff-scoped auxiliary bounds
                # are capped at their proving P and can never exceed it. Requiring
                # lb > P would keep a resolved domain alive until another primal
                # improvement, making a better incumbent increase proof work.
                lb_gt[:, 0] |= (torch.isfinite(lb[:, 0]) & torch.isfinite(threshold[:, 0])
                                & (lb[:, 0] >= threshold[:, 0]))
            lb_verified_mask = lb_gt.any(1)
            lb_nonfinite_mask = (~torch.isfinite(lb)).any(1) | (~torch.isfinite(threshold)).any(1)
            obj_posinf_verified = torch.zeros(batch, dtype=torch.bool, device=lb.device)
            if solving_mode and lb.ndim >= 2 and lb.shape[1] > 0:
                obj_lb = lb[:, 0]
                obj_threshold = threshold[:, 0]
                # In solving mode, +inf on the objective row is the explicit sentinel for
                # an infeasible / fully-resolved domain under constrained concretization.
                obj_posinf_verified = torch.isposinf(obj_lb) & torch.isfinite(obj_threshold)
                lb_verified_mask = lb_verified_mask | obj_posinf_verified
            # Every discarded objective domain has minimum >= the incumbent;
            # discarded constraint domains satisfy lb > their relaxed threshold. Keeping the tightest such threshold lets the
            # reported global bound account for the regions that were thrown away --
            # without it, discarding the domain that holds the optimum (which happens at
            # a one-ulp margin once the incumbent is exact) leaves the survivors' minimum
            # as the reported dual, and that sits ABOVE the true optimum.
            if bool(lb_verified_mask.any()) and lb.ndim >= 2 and lb.shape[1] > 0:
                _disc = threshold[lb_verified_mask][..., 0]
                _disc = _disc[torch.isfinite(_disc)]
                if _disc.numel():
                    _cur = float(_disc.min())
                    _prev = arguments.Globals.get("discarded_threshold_min")
                    if _prev is None or _cur < _prev:
                        arguments.Globals["discarded_threshold_min"] = _cur
            remaining_mask = remaining_mask & ~lb_verified_mask
            if solving_mode:
                obj_filtered = lb_gt[..., 0].sum() + obj_posinf_verified.sum()
                constr_filtered = lb_verified_mask.sum() - obj_filtered
                nonfinite_kept = (remaining_mask & lb_nonfinite_mask).sum()
                print(
                    f"\tCheck lb filtering: batch size: {batch_size}, "
                    f"filter from obj: {obj_filtered}, filter from constr: {constr_filtered}, "
                    f"non-finite lbs kept: {nonfinite_kept}"
                )
        if base_constraints is not None:
            derived_constraints = constraints
            if constraints is not None and base_constraints is not None:
                prefix_count = base_constraints[0].shape[1]
                merged_A, merged_b = constraints
                base_A, base_b = base_constraints
                if (
                    merged_A.shape[1] >= prefix_count
                    and torch.equal(merged_A[:, :prefix_count], base_A)
                    and torch.equal(merged_b[:, :prefix_count], base_b)
                ):
                    derived_constraints = drop_prefix_constraints(
                        constraints,
                        prefix_count=prefix_count,
                    )
            base_lb, base_infeasible_mask = affine_constraints_domain_lower_bound(
                base_constraints,
                dm_l,
                dm_u,
                domain_constraints=derived_constraints,
                rearrange_constraints=arguments.Config["bab"]["clip_n_verify"]["rearrange_constraints"],
                return_infeasible=True,
            )
            base_lb_gt = torch.isfinite(base_lb) & (base_lb > 0)
            base_verified_mask = base_lb_gt.any(1) | base_infeasible_mask
            base_nonfinite_mask = (~torch.isfinite(base_lb)).any(1)
            newly_filtered_mask = remaining_mask & base_verified_mask
            remaining_mask = remaining_mask & ~base_verified_mask
            if solving_mode:
                print(
                    f"\tCheck input constraint filtering: batch size: {batch_size}, "
                    f"filter from input constr: {newly_filtered_mask.sum()}, "
                    f"non-finite input constr kept: {(remaining_mask & base_nonfinite_mask).sum()}"
                )
        if check_input_boxes:
            dm_l_flat = dm_l.view(batch, -1)
            dm_u_flat = dm_u.view(batch, -1)
            nonfinite_box_mask = (~torch.isfinite(dm_l_flat)).any(1) | (~torch.isfinite(dm_u_flat)).any(1)
            invalid_box_mask = (~nonfinite_box_mask) & (dm_l_flat > dm_u_flat).any(1)
            remaining_mask = remaining_mask & ~invalid_box_mask
            if solving_mode:
                # TODO: For now, check_input boxes and check_dm_lbs will not be True at the same time (you can check in batch_branch_and_bound.py)
                # If in the future they will be enabled together, this statistics calculation should be changed.
                input_filtered = invalid_box_mask.sum()
                nonfinite_kept = nonfinite_box_mask.sum()
                print(
                    f"\tCheck input filtering: batch size: {batch_size}, "
                    f"filter from input: {input_filtered}, non-finite boxes kept: {nonfinite_kept}"
                )
        if remaining_mask.all():
            return slice(None)
        return torch.where(remaining_mask)[0]

    def add(
            self,
            lb: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            cs: Tensor,
            threshold: Optional[Tensor] = None,
            *legacy_args,
            constraints: tuple = None,
            base_constraints: tuple = None,
            split_idx: Optional[Tensor] = None,
            remaining_index: Optional[Tensor] = None,
            check_dm_lbs: bool=True,
            check_input_boxes: bool=True,
            attack_counter: Tensor = None,
            split_counter: Tensor = None,
            x_star: Optional[Tensor] = None,
            **kwargs
    ) -> None:
        """
        Takes verified and unverified subdomains and only adds the unverified subdomains

        @param lb: Shape (batch, num_spec)                  Lower bound on domain outputs
        @param dm_l: Shape (batch, *input_shape)               Lower bound on domain inputs
        @param dm_u: Shape (batch, *input_shape)               Upper bound on domain inputs
        @param alpha:                                       alpha parameters
        @param cs: Shape (batch, num_spec, lA rows)         The C transformation matrix
        @param threshold: Shape (batch, num_spec)           The specification thresholds
        @param constraints:                                 constraints parameters
        @param split_idx: Shape (batch, num of splits)      Specifies along which dimensions to split
        @param remaining_index:                             If not None, user is specifying which domains are unverified
        @param check_dm_lbs:                                If true, filters out domains that have been verified
                                                            by lb > threshold
        @param check_input_boxes:                           If true, filters out domains that have been verified
                                                            by dm_l > dm_u
        @param attack_counter:                              Counter of whether chosen by pgd_attack
        @param split_counter:                               Total split depth from the root domain
        @param x_star: Shape (batch, dim_in)                Relaxation minimizer for warm-starting PGD attacks
        """
        legacy_alpha = None
        if isinstance(cs, dict):
            legacy_alpha = cs
            cs = threshold
            if legacy_args:
                threshold = legacy_args[0]
            if legacy_alpha:
                self.history = ensure_split_history(
                    self.history,
                    use_alpha=True,
                    use_beta=self.use_beta,
                    alphas=legacy_alpha,
                )
                self.use_alpha = self.history.use_alpha
        if threshold is None:
            raise ValueError("threshold must be provided when adding domains.")

        self._enforce_gpu_budget(growth=2)
        # check shape correctness
        batch = len(lb)
        if batch == 0:
            return
        if self.use_split_idx:
            assert split_idx is not None, "Cannot accept split_idx"
            assert len(split_idx) == batch
            assert split_idx.shape[1] == self.storage_depth
        else:
            assert split_idx is None, "Expected to receive split_idx"
        assert len(dm_l) == len(dm_u) == len(cs) == len(threshold) == batch
        if self.use_alpha:
            if self.history.alphas is None:
                raise ValueError("alpha should not be None in alpha-crown.")
        # initialize attributes using input shapes and types
        if self.lb is None:
            self.lb = get_tensor_storage(lb.shape, dtype=lb.dtype, device=self._storage_device)
        if self.dm_l is None:
            self.dm_l = get_tensor_storage(dm_l.shape, dtype=dm_l.dtype, device=self._storage_device)
        if self.dm_u is None:
            self.dm_u = get_tensor_storage(dm_u.shape, dtype=dm_u.dtype, device=self._storage_device)
        if self.use_alpha:
            if not self.alpha:
                for key0 in self.history.alphas.keys():
                    self.alpha[key0] = {}
                    for key1 in self.history.alphas[key0].keys():
                        self.alpha[key0][key1] = get_tensor_storage(
                            self.history.alphas[key0][key1].shape, concat_dim=2,
                            dtype=self.history.alphas[key0][key1].dtype, device="cpu"
                        )
            else:
                # Remove alpha layers that are no longer in history.alphas.
                # This happens when some layers are disabled for optimization.
                for key0 in list(self.alpha.keys()):
                    if key0 not in self.history.alphas:
                        del self.alpha[key0]
                    else:
                        for key1 in list(self.alpha[key0].keys()):
                            if key1 not in self.history.alphas[key0]:
                                del self.alpha[key0][key1]
        if self.threshold is None:
            self.threshold = get_tensor_storage(threshold.shape, dtype=threshold.dtype, device=self._storage_device)
        if self.split_idx is None and self.use_split_idx:
            self.split_idx = get_tensor_storage(split_idx.shape, dtype=split_idx.dtype, device=self._storage_device)
        if self.spec_size is None:
            self.spec_size = cs.shape[1]
        if attack_counter is not None:
            if self.attack_counter is None:
                self.attack_counter = get_tensor_storage(attack_counter.shape, dtype=attack_counter.dtype, device=self._storage_device)
        if split_counter is not None:
            if self.split_counter is None:
                self.split_counter = get_tensor_storage(split_counter.shape, dtype=split_counter.dtype, device=self._storage_device)
        if x_star is not None:
            if self.x_star is None:
                self.x_star = get_tensor_storage(x_star.shape, dtype=x_star.dtype, device=self._storage_device)
        else:
            # If x_star is None but storage exists, clear it to avoid size mismatch
            # (we're adding domains without x_star, so storage would become out of sync)
            if self.x_star is not None:
                self.x_star = None
        # compute unverified indices
        if remaining_index is None:
            remaining_index = UnsortedInputDomainList.get_remaining_index(
                batch,
                lb,
                threshold,
                dm_l,
                dm_u,
                check_dm_lbs,
                check_input_boxes,
                base_constraints=base_constraints,
                constraints=constraints,
                rowpool=getattr(self.history, 'rowpool', None),
            )
        # append the tensors
        lb_kept = lb[remaining_index]
        self.lb.append(lb_kept.to(self.lb.device))
        if lb.ndim == 2 and lb.shape[1] > 1:
            if self.lb_obj is None:
                if self.lb.num_used != lb_kept.shape[0]:
                    raise RuntimeError("lb_obj must be created with the first domains")
                self.lb_obj = get_tensor_storage((lb.shape[0], 1), dtype=lb.dtype, device=self._storage_device)
            self.lb_obj.append(lb_kept[:, :1].to(self.lb_obj.device))

        dm_l = dm_l[remaining_index]
        dm_u = dm_u[remaining_index]
        registry = getattr(self.history, 'certificate_registry', None)
        if registry is not None:
            refs = registry.select_scopes(dm_l, dm_u, threshold[remaining_index, 0])
            if self.certificate_scope_refs is None:
                self.certificate_scope_refs = get_tensor_storage(
                    refs.shape, dtype=torch.long, device=self._storage_device)
            self.certificate_scope_refs.append(refs.to(self.certificate_scope_refs.device))
        measure = getattr(self, 'controller_measure', None)
        if measure is not None:
            measure.add(dm_l, dm_u, lb_kept[:, 0])
        self._add_volume(dm_l, dm_u)
        self.dm_l.append(dm_l.to(self.dm_l.device))
        self.dm_u.append(dm_u.to(self.dm_u.device))
        if self.use_alpha:
            for key0 in self.history.alphas.keys():
                for key1 in self.history.alphas[key0].keys():
                    self.alpha[key0][key1].append(
                        self.history.alphas[key0][key1][:, :, remaining_index]
                        .to(self.alpha[key0][key1].device)
                    )
        if self.use_beta:
            idx = remaining_index
            if isinstance(idx, slice):
                self.beta.extend(self.history.betas[idx])
            else:
                for i in idx.tolist():
                    self.beta.append(self.history.betas[i])
        # Indexing a stride-0 broadcast with a tensor materialises the whole batch --
        # 9.36 GiB on ns_vorticity at batch 533 -- and the only reason to do it here is
        # the all-rows-equal assertion, which is vacuous precisely when the stride is 0
        # because every row is then the same memory. Narrow instead.
        cs_in = cs.narrow(0, 0, 1) if cs.stride(0) == 0 else cs[remaining_index]
        if cs_in.shape[0] > 0:
            if self.cs is None:
                self.cs = cs_in[:1].detach().clone().cpu()
            else:
                assert torch.equal(
                    cs_in, self.cs.to(cs_in.device, cs_in.dtype).expand_as(cs_in)), (
                    "cs differs between domains; UnsortedInputDomainList stores one "
                    "shared spec matrix. Use UnsortedMultiSpecInputDomainList instead.")
        self.threshold.append(
            threshold[remaining_index]
            .to(self.threshold.device)
        )

        if constraints is not None:
            if isinstance(remaining_index, slice):
                constraint_A = constraints[0][remaining_index]
                constraint_b = constraints[1][remaining_index]
            else:
                constraint_A = select_rows(constraints[0], remaining_index)
                constraint_b = select_rows(constraints[1], remaining_index)
            self._append_constraints(constraint_A, constraint_b, dm_l, dm_u)
        if base_constraints is not None:
            base_constraint_A, base_constraint_b = base_constraints
            if isinstance(remaining_index, slice):
                base_constraint_A = base_constraint_A[remaining_index]
                base_constraint_b = base_constraint_b[remaining_index]
            else:
                base_constraint_A = select_rows(base_constraint_A, remaining_index)
                base_constraint_b = select_rows(base_constraint_b, remaining_index)
            for name, incoming in (("base_constraint_A", base_constraint_A),
                                   ("base_constraint_b", base_constraint_b)):
                if incoming.shape[0] == 0:
                    continue
                shared = getattr(self, name)
                if shared is None:
                    setattr(self, name, incoming[:1].detach().clone().cpu())
                else:
                    ref = shared.to(incoming.device, incoming.dtype)
                    assert torch.equal(incoming, ref.expand_as(incoming)), (
                        f"{name} differs between domains; the shared-row "
                        "representation assumes the base constraints are "
                        "problem-level and never specialised per domain.")
        if self.use_split_idx:
            self.split_idx.append(
                split_idx[remaining_index]
                .to(self.split_idx.device)
            )
        if attack_counter is not None:
            self.attack_counter.append(
                attack_counter[remaining_index].
                to(self.attack_counter.device)
            )
        if split_counter is not None:
            self.split_counter.append(
                split_counter[remaining_index].
                to(self.split_counter.device)
            )
        if x_star is not None:
            self.x_star.append(
                x_star[remaining_index].
                to(self.x_star.device)
            )
        self._enforce_gpu_budget()

    def _append_constraints(self, constraint_A, constraint_b, dm_l, dm_u):
        """Store each domain's cut rows, minus those redundant on its box.

        A row that holds on the whole box holds on every sub-box, so dropping it is exact
        for the domain and all its descendants. With many output specs nearly every row
        goes (KS50: 1676 of 1677), and keeping them dominated memory and the domain sort.
        The stored width grows on demand; narrower batches are padded with the no-op row
        0 x - 1 <= 0.
        """
        batch = constraint_A.shape[0]
        if batch == 0:
            return
        x0 = ((dm_l + dm_u) / 2).reshape(batch, -1, 1).to(constraint_A.dtype)
        eps = ((dm_u - dm_l) / 2).reshape(batch, -1, 1).to(constraint_A.dtype)
        constraint_A, constraint_b = drop_redundant_constraints(constraint_A, constraint_b, eps, x0)
        if self.constraint_A is None:
            self.constraint_A = get_tensor_storage(constraint_A.shape, dtype=constraint_A.dtype, device=self._storage_device)
            self.constraint_b = get_tensor_storage(constraint_b.shape, dtype=constraint_b.dtype, device=self._storage_device)
        # Bring both storages to one width: an out-of-memory error between the two
        # widen calls can leave A wider than b (A90).
        width = constraint_A.shape[1]
        stored = max(width, self.constraint_A.shape[1], self.constraint_b.shape[1])
        self.constraint_A.widen(1, stored, 0.)
        self.constraint_b.widen(1, stored, -1.)
        if width < stored:
            constraint_A = F.pad(constraint_A, (0, 0, 0, stored - width))
            constraint_b = F.pad(constraint_b, (0, stored - width), value=-1.)
        # append() copies across devices itself; a .to(storage device) first would stage
        # the rows in a temporary host buffer and copy them twice.
        self.constraint_A.append(constraint_A.type(self.constraint_A.dtype))
        self.constraint_b.append(constraint_b.type(self.constraint_b.dtype))

    def pick_out_batch(self, batch_size: int, device=None
                       )->Tuple[dict, Tensor, Tensor, Tensor, Tensor, Tensor, Optional[tuple], Optional[tuple], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Picks out a batch of subdomains from the domain list.

        :param batch_size:  The maximum number of domains we should pick out
        :param device:      The device all Tensors should be sent to

        :return alphas:         If supported, contains alpha parameters for the batch
        :return lb:             Output lower bounds
        :return dm_l:           Domain input lower bounds
        :return dm_u:           Domain input upper bounds
        :return cs:             Specification matrices
        :return threshold:      Thresholds
        :return constraints:    The constraints tuple.
        :return spec_sizes:     The number of ANDs in the spec of each domain
        :return split_idx:      If supported, the input dimensions we should split along
        :return attack_counter: The attack_counter tensor
        :return split_counter:  The split_counter tensor
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_size = min(len(self), batch_size)
        assert batch_size > 0, "List of InputDomain is empty; pop failed."
        self._materialize_lazy_head(batch_size)
        if device is None:
            device = self.output_device
        lb = self._take(self.lb, batch_size, device)
        if self.lb_obj is not None:
            self.lb_obj.pop(batch_size)
        dm_l = self._take(self.dm_l, batch_size, device)
        dm_u = self._take(self.dm_u, batch_size, device)
        measure = getattr(self, 'controller_measure', None)
        if measure is not None:
            measure.add(dm_l, dm_u, lb[:, 0], sign=-1)
        if self.certificate_scope_refs is not None:
            self.history.certificate_scope_refs = self._take(self.certificate_scope_refs, batch_size, device)
        alpha = {}
        if self.use_alpha:
            for key0, val0 in self.alpha.items():
                alpha[key0] = {}
                for key1, val1 in val0.items():
                    alpha[key0][key1] = val1.pop(batch_size).to(
                        device=device, non_blocking=True)
        beta = self.beta[len(self) - batch_size:len(self)] if self.use_beta else None
        if self.history is not None and hasattr(self.history, "update_bounds_optimizers"):
            self.history.update_bounds_optimizers(alpha, beta)
        cs = self._broadcast_cs(batch_size, device)
        threshold = self._take(self.threshold, batch_size, device)
        constraints = None
        if self.constraint_A is not None or self.constraint_b is not None:
            constraint_A = self._take(self.constraint_A, batch_size, device)
            constraint_b = self._take(self.constraint_b, batch_size, device)
            constraints = (constraint_A, constraint_b)
        base_constraints = None
        if self.base_constraint_A is not None and self.base_constraint_b is not None:
            def _broadcast(shared):
                # One row to the device, then the copies there: repeating on the host
                # first built and transferred batch_size copies (140 ms per two
                # iterations on wastewater04m2).
                reps = [batch_size] + [1] * (shared.dim() - 1)
                return shared.to(device, non_blocking=True).repeat(reps)
            base_constraints = (_broadcast(self.base_constraint_A),
                                _broadcast(self.base_constraint_b))
        spec_sizes = torch.tensor(cs.shape[1], device=device).repeat(batch_size)
        attack_counter = None
        if self.attack_counter is not None:
            attack_counter = self._take(self.attack_counter, batch_size, device)
        split_counter = None
        if self.split_counter is not None:
            max_split_counter = self.split_counter._storage[:self.split_counter.num_used].max()
            split_counter = self._take(self.split_counter, batch_size, device)

        if self.use_split_idx:
            split_idx = self._take(self.split_idx, batch_size, device)
        else:
            split_idx = None

        # Pop x_star if it exists and is synchronized with other storages
        if self.x_star is not None:
            if self.x_star.num_used == self.lb.num_used + batch_size:
                # x_star was synchronized before pop, so pop from it too
                self.x_star.pop(batch_size)
            else:
                # x_star is out of sync, clear it
                self.x_star = None

        solution_pt = arguments.Config["solving"]["solution_point"]
        enable_solution_pt = arguments.Config["solving"]["enable_solution_point"]
        if enable_solution_pt:
            assert solution_pt != ""
            preset_input = torch.load(solution_pt)
            preset_input = torch.tensor(preset_input, device=dm_l.device, dtype=dm_l.dtype)
            preset_in_domain_mask = torch.logical_and(
                (preset_input >= dm_l).all(dim=-1),
                (preset_input <= dm_u).all(dim=-1)
            )

            print("================== Dual Hit Report ==================")
            print(f"\tCurrent max splits since a widest-input split: {max_split_counter}")
            if torch.any(preset_in_domain_mask):
                preset_domain_idx = torch.where(preset_in_domain_mask)[-1][-1].item()
                print(f"\tPreset input is in domain! Splits since a widest-input split: {self.split_counter._storage[preset_domain_idx].item()}")
            else:
                print("\tDomain not presented!")
        
        self._add_volume(dm_l, dm_u, sign=-1)
        if self._legacy_pickout_api:
            return (
                alpha,
                lb,
                dm_l,
                dm_u,
                cs,
                threshold,
                constraints,
                spec_sizes,
                split_idx,
            )
        return (
            lb,
            dm_l,
            dm_u,
            cs,
            threshold,
            constraints,
            base_constraints,
            spec_sizes,
            split_idx,
            attack_counter,
            split_counter,
        )

    def _add_volume(self, dm_l, dm_u, sign=1):
        volume = torch.prod(dm_u - dm_l, dim=-1).sum().item()
        if self.all_volume is None:
            self.all_volume = volume
            self.volume = 0
        self.volume = self.volume + sign * volume

    def get_progess(self):
        if self.all_volume is None or self.all_volume == 0:
            return 0.
        else:
            return 1 - self.volume / self.all_volume

    def _get_sort_margin(self, margin):
        if self.sort_index is not None:
            return margin[..., self.sort_index]
        else:
            return margin.max(dim=1).values

    def _objective_key(self, lo, hi, heuristic_name, sort_index):
        """lb[lo:hi, 0] from the contiguous copy when that is the sort key, else None."""
        if (self.lb_obj is not None and heuristic_name == "lb_only" and sort_index == 0
                and self.lb_obj.num_used == self.lb.num_used):
            return self.lb_obj._storage[lo:hi, 0]
        return None

    def get_topk_indices(self, k=1, largest=False, return_margin=False, enable_attack_counter=False):
        assert k <= len(self), print("Asked indices more than domain length.")
        key = None if enable_attack_counter else self._objective_key(
            0, self.lb.num_used, self.sort_heuristic, self.sort_index)
        if key is not None:
            margins, indices = key.topk(k, largest=largest)
            return (indices, margins) if return_margin else indices
        lb = self.lb._storage[: self.lb.num_used]
        threshold = self.threshold._storage[: self.threshold.num_used]
        if self.sort_heuristic == "lb_rhs":
            heuristic = lb - threshold
        elif self.sort_heuristic == "lb_only":
            # No need to clone here, we will not perform inplace-modification to heuristic.
            # But you should pay attention to it.
            heuristic = lb
        else:
            raise ValueError(f"Sort heuristic: {self.sort_heuristic} not implemented!")

        # Infuse attack counter to heuristic, we will sort primarily according to the counter
        # This will make domain selection a BFS.
        if not enable_attack_counter:
            margins, indices = self._get_sort_margin(heuristic).topk(k, largest=largest)
        else:
            h_min = heuristic.min()
            h_max = heuristic.max()
            heuristic_acnt = (heuristic - h_min) / (h_max - h_min) + self.attack_counter._storage[:self.attack_counter.num_used]
            margins, indices = self._get_sort_margin(heuristic_acnt).topk(k, largest=largest)

        if return_margin:
            return indices, margins
        return indices

    def _per_domain_storages(self):
        """Every storage holding one row per domain (the order of the domain list)."""
        stores = [self.lb, self.lb_obj, self.dm_l, self.dm_u, self.threshold,
                  self.constraint_A, self.constraint_b, self.attack_counter,
                  self.split_counter, self.certificate_scope_refs]
        if self.use_split_idx:
            stores.append(self.split_idx)
        if self.x_star is not None and self.x_star.num_used == self.lb.num_used:
            stores.append(self.x_star)
        return [st for st in stores if st is not None]

    def storage_sizes(self):
        """(storage, rows) for every per-domain storage; see truncate_storages."""
        stores = [self.lb, self.lb_obj, self.dm_l, self.dm_u, self.threshold,
                  self.constraint_A, self.constraint_b, self.attack_counter,
                  self.split_counter, self.split_idx, self.x_star, self.certificate_scope_refs]
        for v0 in self.alpha.values():
            stores.extend(v0.values())
        return [(st, st.num_used) for st in stores if st is not None]

    def intersect_root(self, lower, upper):
        """Intersect every open box and compact all aligned storage together.

        Only strict empty intersections are removed. Old lower bounds and
        affine rows remain certificates on the smaller boxes. Compute on the
        root's device, even if native domain storage has spilled to the host.
        Called only between batches, with no checked-out domains.
        """
        if self.use_alpha or self.use_beta:
            raise ValueError('M2 domain intersection requires input-only GEAR history')
        from nlp_solver.rigorous_dual import round_down
        n = len(self)
        if not n: return 0
        stores = self._per_domain_storages()
        if not all(isinstance(st,StackTensorStorage) for st in stores):
            raise ValueError('M2 root intersection requires contiguous stack storage')
        old_head = self._lazy_head
        # If a chunk fails, all remaining rows still participate in selection.
        if old_head is not None: self._lazy_head = n
        self.controller_measure = None
        write = new_head = 0
        # Fixed memory capacity, independent of frontier size.
        for start in range(0,n,65536):
            end = min(n,start+65536)
            lo = self.dm_l._storage[start:end].to(device=lower.device,dtype=torch.float64)
            hi = self.dm_u._storage[start:end].to(device=upper.device,dtype=torch.float64)
            lo = torch.maximum(lo,lower.double())
            hi = torch.minimum(hi,upper.double())
            keep = ~((lo > hi).flatten(1).any(1))
            count = int(keep.sum())
            if old_head is not None:
                new_head += int(keep[:max(0,min(end,old_head)-start)].sum())
            if write == start and count == end-start:
                # Nothing dropped so far: rows stay in place, only the boxes change.
                rl = round_down(lo,self.dm_l._storage.dtype).to(self.dm_l.device)
                ru = (-round_down(-hi,self.dm_u._storage.dtype)).to(self.dm_u.device)
                self.dm_l._storage[start:end].copy_(rl); self.dm_u._storage[start:end].copy_(ru)
                write = end
                continue
            # Gather before overwriting; every source chunk starts at or
            # after its compacted destination, so later chunks stay intact.
            selected = []
            for st in stores:
                if st is self.dm_l: rows = round_down(lo[keep],st._storage.dtype)
                elif st is self.dm_u: rows = -round_down(-hi[keep],st._storage.dtype)
                else: rows = st._storage[start:end][keep.to(st.device)]
                selected.append(rows.to(st.device))
            # Finish every allocating gather before committing this chunk.
            # An allocation failure leaves a cover (possibly with duplicates),
            # with all metadata aligned, for the native tree to resume.
            for st,rows in zip(stores,selected):
                st._storage[write:write+count].copy_(rows)
            write += count
        for st in stores: st.num_used = write
        if old_head is not None: self._lazy_head = new_head
        self.volume = 0.
        for start in range(0,write,65536):
            self._add_volume(self.dm_l._storage[start:min(write,start+65536)],
                             self.dm_u._storage[start:min(write,start+65536)])
        return n-write

    def truncate_storages(self, sizes):
        """Undo appends made after `sizes = storage_sizes()` (StackTensorStorage appends
        at the end, so dropping the extra rows restores the recorded state). Storages
        created after the snapshot are emptied."""
        recorded = {id(st): n for st, n in sizes}
        for st, _ in self.storage_sizes():
            n = recorded.get(id(st), 0)
            if st.num_used > n:
                st.num_used = n
        if self.use_beta and self.beta is not None and self.lb is not None:
            del self.beta[self.lb.num_used:]

    def _lazy_sort_ok(self, enable_attack_counter):
        return (self.lazy_sort and not enable_attack_counter and not self.use_alpha
                and not self.use_beta and isinstance(self.lb, StackTensorStorage))

    def _materialize_lazy_head(self, batch_size):
        """Put in place the rows a full sort would have left where pick_out_batch pops.

        After a lazy sort, rows [0, h) stand for a sorted list and rows [h, n) were
        appended since. Popping `batch_size` rows takes all n - h appended rows and the
        last r = batch_size - (n - h) rows of the sorted list: the r smallest margins
        (for a descending sort), in descending order. Only those r rows are selected
        and moved, instead of permuting every storage. The popped domains are the ones
        a full sort gives, up to the order of domains with equal sort keys.
        """
        h = self._lazy_head
        if h is None:
            return
        n = self.lb.num_used
        h = min(h, n)
        r = min(batch_size - (n - h), h)
        if r <= 0:
            self._lazy_head = h
            return
        heuristic_name, sort_index, descending = self._lazy_key

        def margin(lo, hi):
            key = self._objective_key(lo, hi, heuristic_name, sort_index)
            if key is not None:
                return key
            lb = self.lb._storage[lo:hi]
            heuristic = lb if heuristic_name == "lb_only" else lb - self.threshold._storage[lo:hi]
            return heuristic[..., sort_index] if sort_index is not None else heuristic.max(dim=1).values

        key = margin(0, h)
        if not key.is_cuda and torch.cuda.is_available():
            # Host storage: select on the GPU (one contiguous key column across).
            key_dev = key.to("cuda")
            vals, sel = key_dev.topk(r, largest=not descending)
            sel = sel[vals.argsort(descending=descending)].cpu()
        else:
            vals, sel = key.topk(r, largest=not descending)
            sel = sel[vals.argsort(descending=descending)]
        # sel: the rows a full sort leaves at [h - r, h), in their sorted order. Rows of
        # that range that are not selected move to the slots the selected ones leave.
        in_tail = torch.zeros(h, dtype=torch.bool, device=sel.device)
        in_tail[sel] = True
        tail = torch.arange(h - r, h, device=sel.device)
        outside, holes = sel[sel < h - r], tail[~in_tail[tail]]
        for st in self._per_domain_storages():
            buf = st._storage
            rows = buf[sel]                      # gathered before anything is overwritten
            if outside.numel():
                buf[outside] = buf[holes]
            buf[h - r:h] = rows
        self._lazy_head = h - r

    def sort(self, enable_attack_counter=False):
        if self._lazy_sort_ok(enable_attack_counter):
            self._lazy_head = self.lb.num_used
            self._lazy_key = (self.sort_heuristic, self.sort_index, self.sort_descending)
            return
        self._lazy_head = None
        lb = self.lb._storage[: self.lb.num_used]
        threshold = self.threshold._storage[: self.threshold.num_used]
        if self.sort_heuristic == "lb_rhs":
            heuristic = lb - threshold
        elif self.sort_heuristic == "lb_only":
            # No need to clone here, we will not perform inplace-modification to heuristic.
            # But you should pay attention to it.
            heuristic = lb
        else:
            raise ValueError(f"Sort heuristic: {self.sort_heuristic} not implemented!")

        key = None if enable_attack_counter else self._objective_key(
            0, self.lb.num_used, self.sort_heuristic, self.sort_index)
        if key is not None:
            indices = key.argsort(descending=self.sort_descending)
        elif not enable_attack_counter:
            indices = self._get_sort_margin(heuristic).argsort(
                descending=self.sort_descending)
        else:
            h_min = heuristic.min()
            h_max = heuristic.max()
            heuristic_acnt = (heuristic - h_min) / (h_max - h_min) + self.attack_counter._storage[:self.attack_counter.num_used]
            indices = self._get_sort_margin(heuristic_acnt).argsort(
                descending=self.sort_descending)

        # sort the storage
        self.lb._storage[: self.lb.num_used] = self.lb._storage[indices]
        if self.lb_obj is not None:
            self.lb_obj._storage[: self.lb_obj.num_used] = self.lb_obj._storage[indices]
        self.dm_l._storage[: self.dm_l.num_used] = self.dm_l._storage[indices]
        if self.certificate_scope_refs is not None:
            refs = self.certificate_scope_refs
            refs._storage[:refs.num_used] = refs._storage[indices]
        self.dm_u._storage[: self.dm_u.num_used] = self.dm_u._storage[indices]
        if self.use_alpha:
            for val0 in self.alpha.values():
                for val1 in val0.values():
                    val1._storage[
                    :, :, :val1.num_used] = val1._storage[:, :, indices]
        self.threshold._storage[: self.threshold.num_used] = self.threshold._storage[indices]
        if self.use_beta:
            indices_list = indices.tolist()
            if len(indices_list) == 1:
                self.beta = [self.beta[indices_list[0]]]
            else:
                self.beta = [self.beta[i] for i in indices_list]


        if self.constraint_A is not None:
            self.constraint_A._storage[: self.constraint_A.num_used] = self.constraint_A._storage[indices]
        if self.constraint_b is not None:
            self.constraint_b._storage[: self.constraint_b.num_used] = self.constraint_b._storage[indices]

        if self.attack_counter is not None:
            self.attack_counter._storage[: self.attack_counter.num_used] = self.attack_counter._storage[indices]
        if self.split_counter is not None:
            self.split_counter._storage[: self.split_counter.num_used] = self.split_counter._storage[indices]

        if self.use_split_idx:
            self.split_idx._storage[: self.split_idx.num_used] = self.split_idx._storage[indices]

        # Only sort x_star if it exists and has matching size (may not exist for initial domains)
        if self.x_star is not None and self.x_star.num_used == self.lb.num_used:
            self.x_star._storage[: self.x_star.num_used] = self.x_star._storage[indices]

    def report_memory(self):
        def _report_memory(attr_name, allocated_in_MB, used_in_MB):
            print(f"[{attr_name}] allocated: {allocated_in_MB:.2f} MB, used: {used_in_MB:.2f} MB")

        if self.lb is not None:
            _report_memory("lb", *self.lb.calculate_memory())
        if self.dm_l is not None:
            _report_memory("dm_l", *self.dm_l.calculate_memory())
        if self.dm_u is not None:
            _report_memory("dm_u", *self.dm_u.calculate_memory())
        if self.use_alpha:
            alpha_allocated_in_MB, alpha_used_in_MB = 0, 0
            for val0 in self.alpha.values():
                for val1 in val0.values():
                    curr_alpha_allocated, curr_alpha_used = val1.calculate_memory()
                    alpha_allocated_in_MB += curr_alpha_allocated
                    alpha_used_in_MB += curr_alpha_used
            _report_memory("alpha", alpha_allocated_in_MB, alpha_used_in_MB)
        if self.cs is not None:
            cs_MB = self.cs.numel() * self.cs.element_size() / 1024 ** 2
            _report_memory("cs (shared)", cs_MB, cs_MB)
        if self.threshold is not None:
            _report_memory("threshold", *self.threshold.calculate_memory())
        if self.use_split_idx and self.split_idx is not None:
            _report_memory("split_idx", *self.split_idx.calculate_memory())

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_used_MB = memory_info.rss / 1024 / 1024
        print(f"Total memory used: {memory_used_MB:.2f} MB")

        return


class UnsortedMultiSpecInputDomainList(InputDomainList):
    """Unsorted domain list for input split with arbitrary specifications."""

    def __init__(
        self,
        or_spec_size: Tensor,
        input_shape: Tuple,
        output_dim: int,
        storage_depth: int,
        output_device: str,
        history=None,
        sort_index: Optional[int] = None,
        sort_heuristic: Optional[str] = "lb_rhs",
        sort_descending: bool = True,
        use_split_idx: bool = True,
        alpha_final_name: Optional[str] = None,
        use_alpha: Optional[bool] = None,
    ):
        """

        The initialization method for the UnsortedMultiSpecInputDomainList class.

        :param or_spec_size:        A tensor containing the unique specification dimensions we may support
        :param input_shape:         The shape of the input to the network
        :param output_dim:          The output dimension of the network before applying C
        :param storage_depth:       The maximum number of splits we could use in input BaB
        :param output_device:       The default device storing accessed data. Complete domain data is always stored on 'cpu'.
        :param use_alpha:           True if we must also store alpha parameters
        :param sort_index:          The output index to use for sorting domains.
        :param sort_heuristic:      The heuristic type for sorting domains.
        :param sort_descending:     If True, domains will get sorted in descending order w.r.t. their lower bounds
                                    whenever the 'sort' method is called.
        :param use_split_idx:       If True, we also store the split indices for each domain
        :param alpha_final_name:    The name of the final node in LiRPANet.
                                    alpha of the final node has the spec_size.
        """
        super().__init__()

        self.output_device = output_device
        or_spec_size = or_spec_size.unique()
        # we store the spec with largest number of ANDs first
        self.or_spec_size = or_spec_size.to(output_device).sort(descending=True).values

        self.input_shape = input_shape
        self.output_dim = output_dim
        use_alpha = bool(use_alpha) if use_alpha is not None else bool(getattr(history, "use_alpha", False))
        self.history = ensure_split_history(history, use_alpha=use_alpha, use_beta=False)
        self.use_alpha = self.history.use_alpha
        self.use_split_idx = use_split_idx
        self.storage_depth = storage_depth
        if self.use_alpha:
            assert alpha_final_name is not None, "Expected to receive alpha_final_name"
            self.alpha_final_name = alpha_final_name

        # Initialize all input domain lists
        unsorted_input_domain_list_args = {
            'output_device': output_device,
            'storage_depth': storage_depth,
            'sort_index': sort_index,
            'sort_heuristic': sort_heuristic,
            'sort_descending': sort_descending,
            'use_split_idx': use_split_idx,
        }
        spec_domain_lists = []
        for _ in self.or_spec_size:
            spec_domain_lists.append(
                UnsortedInputDomainList(
                    history=ensure_split_history(None, use_alpha=self.use_alpha, use_beta=False),
                    use_alpha=self.use_alpha,
                    **unsorted_input_domain_list_args,
                )
            )

        self.spec_domain_lists: List[UnsortedInputDomainList] = spec_domain_lists

    def pick_out_batch(self, batch_size, device=None):
        """
        Pick out a batch of subdomains from the domain list. Zero-padding is applied when necessary.

        :param batch_size:      The maximum number of domains we should pick out
        :param device:          The device all Tensors should be sent to

        :return alphas:         If supported, contains alpha parameters for the batch
        :return lb:             Output lower bounds
        :return dm_l:           Domain input lower bounds
        :return dm_u:           Domain input upper bounds
        :return cs:             Specification matrices
        :return threshold:      Thresholds
        :return spec_sizes:     The number of ANDs in the spec of each domain
        :return split_idx:      If supported, the input dimensions we should split along
        """
        if device is None:
            device = self.output_device

        # determine which domain lists we should pick out from and how many domains we should pick out.
        pickout_dl_idx, pickout_num_per_dl = self._get_pickout_decision(batch_size)

        # if we only need to pick out from one domain list, we can directly return the pickout 
        if len(pickout_dl_idx) == 1:
            curr_dl = self.spec_domain_lists[pickout_dl_idx[0]]
            return curr_dl.pick_out_batch(pickout_num_per_dl[0], device=device)

        assert not self.use_alpha, "UnsortedMultiSpecInputDomainList does not support alpha-aware pickout."
        lb, dm_l, dm_u, cs, threshold, split_idx, spec_sizes = [], [], [], [], [], [], []
        if not self.use_split_idx:
            split_idx = None
        for i, dl_idx in enumerate(pickout_dl_idx):
            curr_dl = self.spec_domain_lists[dl_idx]
            curr_pickout_num = pickout_num_per_dl[i]
            curr_lb, curr_dm_l, curr_dm_u, curr_cs, curr_threshold, curr_constraints, curr_base_constraints, curr_spec_sizes, curr_split_idx, _, _ = \
                curr_dl.pick_out_batch(curr_pickout_num, device=device)
            lb.append(curr_lb)
            dm_l.append(curr_dm_l)
            dm_u.append(curr_dm_u)
            cs.append(curr_cs)
            threshold.append(curr_threshold)
            assert curr_constraints is None, "Constraints are not supported in UnsortedMultiSpecInputDomainList"
            assert curr_base_constraints is None, "Base input constraints are not supported in UnsortedMultiSpecInputDomainList"
            if self.use_split_idx:
                split_idx.append(curr_split_idx)

            spec_sizes.append(curr_spec_sizes)
        
        # concatenate the lists to tensors
        dm_l = torch.cat(dm_l, dim=0)
        dm_u = torch.cat(dm_u, dim=0)
        spec_sizes = torch.cat(spec_sizes, dim=0)
        if self.use_split_idx:
            split_idx = torch.cat(split_idx, dim=0)

        # zero-pad the lists to tensors
        lb = pad_list_of_input_to_tensor(
                lb, pad_value=float('-inf'), pad_dim=1, batch_dim=0, is_orginal_tensor=True, device=device
            )
        cs = pad_list_of_input_to_tensor(
                cs, pad_value=0, pad_dim=1, batch_dim=0, is_orginal_tensor=True, device=device
            )
        threshold = pad_list_of_input_to_tensor(
                threshold, pad_value=float('inf'), pad_dim=1, batch_dim=0, is_orginal_tensor=True, device=device
            )

        constraints = None
        return lb, dm_l, dm_u, cs, threshold, constraints, None, spec_sizes, split_idx, None, None

    def _get_pickout_decision(self, batch_size: int) -> Tuple[List[int], List[int]]:
        """
        Returns the indices of the input domain lists that we should pick out and corresponding
        number of domains we should pick out from each input domain list.

        :param batch_size:   The number of domains we wish to pick out in this iteration.
        :return:        A list of indices of the input domain lists we should pick from.
                        A list of the respective numbers of batches we should pickout from each input domain list.
        """
        # get the number of domains per domain list
        num_domains_per_dl = self._get_num_domains_per_dl()
        # mask of non-emtpy domain lists
        non_empty_mask = num_domains_per_dl > 0

        # If we do not have enough total subdomains to fill out this pickout batch,
        # we pick out from all domain lists that are not empty
        if num_domains_per_dl.sum() < batch_size:
            # indices of non-empty domain lists
            non_empty_dl_idx = torch.where(non_empty_mask)[0].tolist()
            pickout_num_per_dl = num_domains_per_dl[non_empty_mask].tolist()
            return non_empty_dl_idx, pickout_num_per_dl

        # If we have at least one input domain list that can fulfill the pickout batch,
        # we return the first one.
        individual_list_mask = num_domains_per_dl >= batch_size
        if individual_list_mask.any():
            return [torch.where(individual_list_mask)[0][0].item()], [batch_size]

        # If there are enough domains in total, while none of the individual domain lists
        # can fulfill the pickout batch, we pick out from multiple domain lists.
        # In this case, we pick out domain lists in the descending order of the number of ANDs.
        # Since we have sorted the domain lists in this way, we can simply pick out from the front.
        global_offsets_per_dl = self._get_global_offsets_per_dl()
        pickout_dl_mask = (global_offsets_per_dl < batch_size) & non_empty_mask
        pickout_dl_idx = torch.where(pickout_dl_mask)[0].tolist()
        pickout_num_per_dl = num_domains_per_dl[pickout_dl_mask]
        pickout_num_per_dl[-1] = batch_size - global_offsets_per_dl[pickout_dl_mask][-1]
        return pickout_dl_idx, pickout_num_per_dl.tolist()

    def get_progress(self):
        # TODO: Test this method
        progress_list = [d.get_progess() for d in self.spec_domain_lists]
        return (torch.sum(torch.tensor(progress_list, self.output_device)) / len(progress_list)).item()

    def add(self,
            lb: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            alpha,
            cs: Tensor,
            threshold: Optional[Tensor] = None,
            spec_sizes: Optional[Tensor] = None,
            split_idx: Optional[Tensor] = None,
            remaining_index: Optional[Tensor] = None,
            check_dm_lbs: bool=True,
            check_input_boxes: bool=True,
            **kwargs
    ) -> None:
        """
        Takes verified and unverified subdomains and only adds the unverified subdomains

        @param lb: Shape (batch, num_spec)                  Lower bound on domain outputs
        @param dm_l: Shape (batch, *input_shape)            Lower bound on domain inputs
        @param dm_u: Shape (batch, *input_shape)            Upper bound on domain inputs
        @param alpha:                                       alpha parameters
        @param cs: Shape (batch, num_spec, lA rows)         The C transformation matrix
        @param threshold: Shape (batch, num_spec)           The specification thresholds
        @param spec_sizes: Shape (batch,)                   The number of ANDs in the spec of each domain
        @param split_idx: Shape (batch, num of splits)      Specifies along which dimensions to split
        @param remaining_index:                             If not None, user is specifying which domains are unverified
        @param check_dm_lbs:                                If true, filters out domains that have been verified by lb > threshold
        @param check_input_boxes:                           If true, filters out domains that have been verified by dm_l > dm_u
        """
        if torch.is_tensor(alpha):
            if threshold is not None:
                raise ValueError("Ambiguous UnsortedMultiSpecInputDomainList.add call.")
            alpha, cs, threshold = {}, alpha, cs

        if threshold is None:
            raise ValueError("threshold must be provided.")
        if spec_sizes is None:
            raise ValueError("spec_sizes must be provided for multi-spec domain lists.")
        if kwargs.get("constraints") is not None or kwargs.get("base_constraints") is not None:
            raise AssertionError("Constraints are not supported in UnsortedMultiSpecInputDomainList.")

        batch = len(lb)
        if batch == 0:
            return

        # filter out the verified domains. No RowPool reaches the filter: only solving mode has one, and it asserts a
        # single OR spec (complete_verifier_func.bab), so input_bab_parallel never builds this list there.
        history = None
        if self.use_alpha and alpha:
            history = ensure_split_history(None, use_alpha=True, use_beta=False, alphas=alpha)
        batch, lb, dm_l, dm_u, cs, threshold, _, _, _, _, split_idx, spec_sizes, _, _, _ = \
            UnsortedInputDomainList.filter_verified_domains(
                batch, lb, dm_l, dm_u, cs, threshold,
                history=history,
                split_idx=split_idx,
                spec_sizes=spec_sizes,
                check_dm_lbs=check_dm_lbs,
                check_input_boxes=check_input_boxes,
                remaining_index=remaining_index,
            )
        filtered_alpha = history.alphas if history is not None else {}

        for i, this_spec_size in enumerate(self.or_spec_size):
            this_dl_mask = spec_sizes == this_spec_size
            if this_dl_mask.any():
                domain_idx_in_this_dl = torch.where(this_dl_mask)[0]
                # get the data for this domain list
                temp_history = None
                if self.use_alpha and filtered_alpha:
                    temp_history = ensure_split_history(
                        None,
                        use_alpha=True,
                        use_beta=False,
                        alphas=filtered_alpha,
                    )
                _, curr_lb, curr_dm_l, curr_dm_u, curr_cs, curr_threshold, _, _, _, _, curr_split_idx, _, _, _, _ = \
                    UnsortedInputDomainList.filter_verified_domains(
                        batch, lb, dm_l, dm_u, cs, threshold,
                        history=temp_history,
                        split_idx=split_idx,
                        remaining_index=domain_idx_in_this_dl,
                        check_dm_lbs=False,
                        check_input_boxes=False,
                    )
                curr_alpha = temp_history.alphas if temp_history is not None else {}
                # add the data to the domain list
                # since we have filtered out the verified domains, we do not need to check them again.
                # slice to remove the potential padding and align the added data with that in domain list
                curr_lb = curr_lb[:, :this_spec_size]
                curr_cs = curr_cs[:, :this_spec_size]
                curr_threshold = curr_threshold[:, :this_spec_size]
                for key0 in curr_alpha.keys():
                    for key1 in curr_alpha[key0].keys():
                        if key1 == self.alpha_final_name:
                            curr_alpha[key0][key1] = curr_alpha[key0][key1][:, :this_spec_size]
                self.spec_domain_lists[i].add(
                    curr_lb, curr_dm_l, curr_dm_u, curr_alpha, curr_cs, curr_threshold,
                    split_idx=curr_split_idx,
                    remaining_index=None,
                    check_dm_lbs=False,
                    check_input_boxes=False,
                )


    def sort(self):
        """
        Sorts all non-empty input domain lists
        """
        for d in self.spec_domain_lists:
            if len(d) > 0:
                d.sort()

    def __len__(self) -> int:
        """
        :return: Total domains among all input domain lists.
        """
        return self._get_num_domains_per_dl().sum().item()

    def __getitem__(self, idx):
        output_device = self.output_device
        # convert idx to tensor on the specified device for slicing.
        # it is for efficient inner computation only, when calling domain list's __getitem__,
        # we transfer the idx to cpu (the device of the domain list).
        if isinstance(idx, slice):
            idx = torch.arange(len(self), device=output_device)[idx]
        else:
            idx = torch.as_tensor(idx, device=output_device).view(-1)
        assert idx.shape[0] > 0, "Empty index"
        global_offsets_per_dl = self._get_global_offsets_per_dl()
        spec_idx = torch.searchsorted(global_offsets_per_dl, idx, right=True) - 1
        inner_idx = idx - global_offsets_per_dl[spec_idx]

        # collect data from every domain list
        lb, dm_l, dm_u, cs, threshold = [], [], [], [], []
        # record the number of ANDs and number of domains for every domain list we visit
        spec_sizes, num_domains_per_dl = [], []
        # since we access the domain lists in order, instead of the order of idx,
        # we need to keep track of the order of elements we get in the orginal idx.
        order_in_original_idx = []
        for i, domain_list in enumerate(self.spec_domain_lists):
            mask = spec_idx == i
            if mask.any():
                curr_idx = mask.nonzero(as_tuple=True)[0]
                spec_sizes.append(self.or_spec_size[i].item())
                num_domains_per_dl.append(curr_idx.shape[0])
                order_in_original_idx.append(curr_idx)
                # TensorStorage is on CPU, so we need to move the idx to CPU if needed
                curr_lb, curr_dm_l, curr_dm_u, curr_cs, curr_threshold, *_ = domain_list[
                    inner_idx[mask].cpu()
                ]
                lb.append(curr_lb)
                dm_l.append(curr_dm_l)
                dm_u.append(curr_dm_u)
                cs.append(curr_cs)
                threshold.append(curr_threshold)

        order_in_original_idx = torch.cat(order_in_original_idx, dim=0)
        inverted_order = torch.argsort(order_in_original_idx)

        # batch the tensors and adjust the order
        # [batch, *input_shape]
        dm_l = torch.cat(dm_l, dim=0)[inverted_order]
        dm_u = torch.cat(dm_u, dim=0)[inverted_order]

        # lb, cs, threshold have the dim of num_and
        # so we need to pad them if domains are from different domain lists with different num_and.
        if (spec_idx == spec_idx[0]).all():
            # if all domains are from the same domain list, just concatenate them
            lb = torch.cat(lb, dim=0)[inverted_order]
            cs = torch.cat(cs, dim=0)[inverted_order]
            threshold = torch.cat(threshold, dim=0)[inverted_order]
        else:
            lb = pad_list_of_input_to_tensor(
                lb, pad_value=float('-inf'), pad_dim=1, batch_dim=0, is_orginal_tensor=True, device=output_device
            )[inverted_order]
            cs = pad_list_of_input_to_tensor(
                cs, pad_value=0, pad_dim=1, batch_dim=0, is_orginal_tensor=True, device=output_device
            )[inverted_order]
            threshold = pad_list_of_input_to_tensor(
                threshold, pad_value=float('inf'), pad_dim=1, batch_dim=0, is_orginal_tensor=True, device=output_device
            )[inverted_order]

        # indicate the number of ANDs for every domain
        # it can be used to remove the padding if needed
        spec_sizes = torch.repeat_interleave(
            torch.tensor(spec_sizes, device=output_device),
            torch.tensor(num_domains_per_dl, device=output_device),
        )[inverted_order]

        return lb, dm_l, dm_u, cs, threshold, spec_sizes

    def get_topk_indices(
        self, k: int = 1, largest: bool = False, return_margin: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        This function returns the indices (w/ or w/o margins)of top k domains
        with worst(smallest) / best(largest) decision margins among all domain lists.

        :param k:                       The indices for the top k worst lower bounds
        :param return_margin:           If True, returns the margins along with the indices.
        :return global_topk_indices:    The global top k indices and margins (if return_margin is True)
        """
        assert k <= len(self), print("Asked indices more than domain length.")
        output_device = self.output_device
        # get the top k margins and indices for each domain list
        local_topk_inner_indices, local_topk_margins, local_topk_num = [], [], []
        for i, domain_list in enumerate(self.spec_domain_lists):
            if len(domain_list) > 0:
                curr_indices, curr_margins = domain_list.get_topk_indices(
                    min(k, len(domain_list)), largest=largest, return_margin=True
                )
                local_topk_inner_indices.append(curr_indices)
                local_topk_margins.append(curr_margins)
                local_topk_num.append(curr_indices.shape[0])

        # local_topk_inner_indices: inner indices of the top k margins for each domain list
        # local_topk_margins: top k margins for each domain list
        # local_topk_dl_indices: dl indices of the top k margins for each domain list
        local_topk_inner_indices = torch.cat(local_topk_inner_indices, dim=0).to(output_device)
        local_topk_margins = torch.cat(local_topk_margins, dim=0).to(output_device)
        local_topk_dl_indices = torch.repeat_interleave(
            torch.arange(len(local_topk_num), device=output_device),
            torch.tensor(local_topk_num, device=output_device),
        )

        # get top k margins and indices among top k margins of all domain lists
        sorted_topk_margins, sorted_topk_indices = local_topk_margins.topk(k, largest=largest)

        topk_indices_inner = local_topk_inner_indices[sorted_topk_indices]
        topk_indices_dl = local_topk_dl_indices[sorted_topk_indices]

        global_offsets_per_dl = self._get_global_offsets_per_dl()
        global_topk_indices = global_offsets_per_dl[topk_indices_dl] + topk_indices_inner

        if return_margin:
            return global_topk_indices, sorted_topk_margins
        return global_topk_indices

    def _get_num_domains_per_dl(self):
        """
        Returns the number of domains in each input domain list.
        """
        return torch.tensor([len(d) for d in self.spec_domain_lists], device=self.output_device)

    def _get_global_offsets_per_dl(self):
        """
        Returns the cumulative sum of the number of domains in each input domain list.
        """
        num_domains_per_dl = torch.cat([torch.tensor([0], device=self.output_device), self._get_num_domains_per_dl()[:-1]])
        return num_domains_per_dl.cumsum(0)
