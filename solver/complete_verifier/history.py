import torch
import copy
from prune import prune_alphas
from typing import List, Dict, Tuple, Optional

class SplitHistory:
    """
    Stores:
        - ReLU split records (for ReLU-BaB; can be empty under input-BaB)
        - Fixed-alphas to use for CROWN
        - betas (β-CROWN dual variables for output constraints)
    """

    def __init__(
        self,
        initial_alpha: Optional[Dict[str, torch.Tensor]] = None,
        initial_beta: Optional[Dict[str, torch.Tensor]] = None,
        split_history: Optional[Dict[str, List[Tuple]]] = None,
        final_name: Optional[str] = None,
    ):
        # Fixed α (used by CROWN)
        self.alphas: Dict[str, torch.Tensor] = initial_alpha if initial_alpha is not None else {}
        # optimizable β (used by Beta-CROWN)
        self.betas: Dict[str, torch.Tensor] = initial_beta if initial_beta is not None else []
        # Per-layer ReLU split history (unused for input-BaB)
        self.split_history: Dict[str, List[Tuple]] = split_history if split_history is not None else {}
        self.final_name = final_name
        self.use_alpha = False
        self.use_beta = False
        # Mean ALM mu_constr across domains (for warm-starting dual ascent lambda)
        # Shape: (n_constr,) = (num_and - 1,)
        self.alm_mean_mu: Optional[torch.Tensor] = None
        # Dual ascent optimal lambda* mean (for warm-starting ALM mu_constr)
        # Shape: (n_constr,) = (num_and - 1,)
        self.dual_lam_star: Optional[torch.Tensor] = None
        # Dual-guided relaxation minimizer x* (for warm-starting ALM x_init)
        # Shape: (B, d_in) from dual-guided lA_eff
        self.dual_x_star: Optional[torch.Tensor] = None
        # Best ALM primal solution x* found during BaB (for gradient-guided branching)
        # Shape: (*input_shape) or (1, *input_shape); set by run_bab_alm_step
        self.alm_best_primal_x: Optional[torch.Tensor] = None
        # Per-mechanism state, written only by the mechanism that owns it (None = off).
        # Declared here so the state is visible; defaults equal the old getattr defaults.
        self.qtb = None                      # quadratic/convex tangent bound (nlp_in_bab)
        self.convex_row_cuts = None          # nlp_solver.convex_row_cuts.ConvexRowCuts
        self.root_width = None               # (n,) root box widths, for width guards
        self.angle_pairs = None              # (P, 2) long: variable pairs of x_i^2 + x_j^2 = const equalities
        self.qtb_mean_y = None               # QTB dual warm start
        self.lp_mean_y = None                # node-LP dual warm start
        self.lifted = None                   # nlp_solver.lifted_lp.LiftedLP
        self.lifted_mean_y = None
        self.certificate_registry = None     # shared scope bank, never copied per domain
        self.certificate_scope_refs = None   # (B, 1) stable IDs into that bank
        self.lifted_box = None               # OBBT box (lower, upper)
        self.lifted_root_box = None
        self.lifted_obbt_rows = (None, None, None)
        self.lifted_cutoff = None
        self.lifted_obbt_pending = False      # paying OBBT stopped by a work limit
        self.lifted_branch_score = None
        self.lifted_split_point = None
        self.lifted_bab_cutoff = None
        self.lifted_bab_leaves = None
        self.lifted_bab_open = None
        self.lifted_bab_new_incumbent = None
        self.lifted_bab_spent = 0.0
        self.lifted_reobbt_time = 0.0
        self.rowpool = None                  # nlp_solver.structure.RowPool (solving mode): tau applied once, read by every bound

    def set_usage_flags(self, use_alpha: bool, use_beta: bool):
        self.use_alpha = use_alpha
        self.use_beta = use_beta
    
    def get_prune_alphas(self, nodes):
        return prune_alphas(self.alphas, nodes)

    def update_bounds_optimizers(
        self,
        alphas: Dict[str, torch.Tensor],
        betas: Optional[Dict[str, torch.Tensor]] = None
    ):
        if alphas is not None:
            self.alphas = alphas
        if betas is not None:
            self.betas = betas

    def filter_opt_variables(self, remaining_index: torch.Tensor, batch_filt: int) -> Dict[str, torch.Tensor]:
        alpha_filt = {}
        beta_filt = None
        if self.use_alpha and batch_filt > 0:
            with torch.no_grad():
                # alpha may have different device from other tensors.
                # In get_lower_bound_naive() we transfer alpha to cpu and float16.
                alpha_device = next(iter(next(iter(self.alphas.values())).values())).device
                if isinstance(remaining_index, torch.Tensor):
                    remaining_index_for_alpha = remaining_index.to(alpha_device)
                else:
                    remaining_index_for_alpha = remaining_index
                for key0 in self.alphas.keys():
                    alpha_filt[key0] = {}
                    for key1 in self.alphas[key0].keys():
                        # alpha[key0][key1] has shape (alpha_size, prod(start_node_shape), batch_size, node_size)
                        alpha_filt[key0][key1] = self.alphas[key0][key1][:, :, remaining_index_for_alpha]
        if self.use_beta and batch_filt > 0 and self.betas is not None and torch.is_tensor(remaining_index):
            # remaining_index: LongTensor like tensor([0, 3, 5], device=..., dtype=torch.long)
            beta_filt = [self.betas[i] for i in remaining_index.tolist()]

        for _name in ("lifted_branch_score", "lifted_split_point", "certificate_scope_refs"):
            _t = getattr(self, _name, None)
            if torch.is_tensor(_t) and torch.is_tensor(remaining_index):
                setattr(self, _name, _t[remaining_index.to(_t.device)])

        self.update_bounds_optimizers(alpha_filt, beta_filt)

    def repeat_for_split(self, new_batch_size: int, repeat_func: callable):
        """
        Repeat alphas/betas for each split branch.
        """
        repeated_alphas = None
        repeated_betas = None
        if self.alphas is not None:
            repeated_alphas = {}
            for m in self.alphas:
                repeated_alphas[m] = {}
                for spec_name in self.alphas[m]:
                    repeated_alphas[m][spec_name] = repeat_func(self.alphas[m][spec_name], 2, new_batch_size)
        if self.betas is not None:
            repeated_betas = []
            for _ in range(new_batch_size):
                for d in self.betas:
                    new_d = {}
                    for k, v in d.items():
                        if isinstance(v, torch.Tensor):
                            new_d[k] = v.clone()
                        else:
                            # Fallback for nested dicts if necessary, though betas are usually flat dicts of tensors
                            new_d[k] = copy.deepcopy(v)
                    repeated_betas.append(new_d)
        if torch.is_tensor(self.certificate_scope_refs):
            self.certificate_scope_refs = repeat_func(self.certificate_scope_refs, 0, new_batch_size)
        self.update_bounds_optimizers(repeated_alphas, repeated_betas)

    # -------- ReLU split recording --------
    def add_split(self, node_name: str, relu_idx, relu_status, relu_bias, relu_score, depth):
        if node_name not in self.split_history:
            self.split_history[node_name] = []
        self.split_history[node_name].append((relu_idx, relu_status, relu_bias, relu_score, depth))

    # -------------- Accessors --------------
    def __getitem__(self, node_name: str):
        if isinstance(node_name, str):
            return self.split_history.get(node_name, [])
        raise TypeError("History can only be indexed by node name (string).")

    def __repr__(self):
        return (f"SplitHistory(nodes={list(self.split_history.keys())}, "
                f"alpha_keys={list(self.alphas.keys())}, "
                f"beta_keys={list(self.betas.keys())})")

    def __len__(self):
        return len(self.split_history)


def ensure_split_history(history=None, *, use_alpha=None, use_beta=None,
                         alphas=None, betas=None):
    """Return a SplitHistory, building one when the caller only has raw alphas.

    Fields are overwritten only when explicitly supplied, so passing an existing
    history through leaves its alphas and usage flags untouched.
    """
    if history is None:
        history = SplitHistory(initial_alpha=alphas, initial_beta=betas)
    else:
        if alphas is not None:
            history.alphas = alphas
        if betas is not None:
            history.betas = betas
    if use_alpha is not None or use_beta is not None:
        history.set_usage_flags(bool(use_alpha), bool(use_beta))
    return history
