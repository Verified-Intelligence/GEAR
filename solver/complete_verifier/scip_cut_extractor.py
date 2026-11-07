#!/usr/bin/env python3
"""Extract linear cutting planes from SCIP and save them for use by the verifier.

Usage (standalone):
    python scip_cut_extractor.py problem.nl --output cuts.json [--timelimit 90] [--settings scip_technical.set]

The script solves the NLP with SCIP (honouring the given settings file).
During the solve, an event handler captures LP rows at each node.
After solve, the script also captures cut-pool rows.

Only cuts that are purely over the *original* variables (i.e. no auxiliary
variables from presolve/reformulation) are kept.  Each valid cut is stored as
a sparse row: variable indices, coefficients, lhs, rhs, constant.

Output format (JSON):
    {
        "problem_name": "<stem>",
        "nl_file": "<path>",
        "n_original_vars": 46,
        "original_var_names": ["x0", "x1", ...],
        "scip_status": "optimal",
        "primal_bound": 229.70,
        "dual_bound": 229.10,
        "n_total_lp_rows": 311,
        "n_extracted_cuts": 42,
        "cuts": [
            {
                "name": "rlt_cut_...",
                "var_indices": [0, 3, 7],
                "coefficients": [1.0, -0.5, 2.3],
                "lhs": null,
                "rhs": 5.0,
                "constant": 0.0
            },
            ...
        ]
    }
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# SCIP renames original variables during presolve.
# For .nl files with variables x0, x1, ..., SCIP typically creates
# transformed copies named t_x0, t_x1, etc.  We match both patterns.
_TRANSFORMED_VAR_RE = re.compile(r'^t_(x\d+)$')


def _extract_row(row, name_to_idx: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """Convert one LP Row into a cut dict, or None if it uses auxiliary vars."""
    cols = row.getCols()
    vals = row.getVals()
    lhs = row.getLhs()
    rhs = row.getRhs()
    constant = row.getConstant()
    cut_name = row.name

    if len(cols) == 0:
        return None

    var_indices: List[int] = []
    coefficients: List[float] = []

    for col, val in zip(cols, vals):
        var_name = col.getVar().name
        if var_name in name_to_idx:
            var_indices.append(name_to_idx[var_name])
            coefficients.append(float(val))
        else:
            # References an auxiliary variable — skip
            return None

    lhs_out = float(lhs) if not math.isinf(lhs) else None
    rhs_out = float(rhs) if not math.isinf(rhs) else None

    return {
        "name": cut_name,
        "var_indices": var_indices,
        "coefficients": coefficients,
        "lhs": lhs_out,
        "rhs": rhs_out,
        "constant": float(constant),
    }


def _build_name_to_idx(orig_var_names: List[str]) -> Dict[str, int]:
    """Build a var-name → original-index map covering both raw and transformed names.

    SCIP renames ``x0`` → ``t_x0`` during presolve.  We build a map that
    resolves *both* ``x0`` and ``t_x0`` to index 0, etc.
    """
    name_to_idx: Dict[str, int] = {}
    orig_set = set(orig_var_names)
    for i, name in enumerate(orig_var_names):
        name_to_idx[name] = i
        # Also register the transformed name
        name_to_idx[f"t_{name}"] = i
    return name_to_idx


def _solve_and_extract(
    nl_path: str,
    *,
    timelimit: float = 90.0,
    settings_file: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve *nl_path* with SCIP and return a dict of extracted cuts."""
    from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE  # type: ignore

    m = Model()
    m.setParam("display/verblevel", 1 if verbose else 0)
    m.readProblem(nl_path)
    m.setParam("limits/time", timelimit)

    # Apply user settings (e.g. scip_technical.set)
    if settings_file and os.path.isfile(settings_file):
        m.readParams(settings_file)

    # Enable RLT cuts (matching SCIPModel defaults)
    m.setParam("separating/rlt/priority", -10000)
    m.setParam("separating/rlt/maxunknownterms", -1)
    m.setParam("separating/rlt/maxusedvars", -1)
    m.setParam("separating/rlt/maxroundsroot", -1)
    m.setParam("separating/rlt/onlyoriginal", False)
    m.setParam("separating/rlt/goodscore", 0.5)
    m.setParam("separating/rlt/hiddenrlt", True)

    # ---- Collect original variable names before solve ----
    orig_vars = m.getVars()
    orig_var_names: List[str] = [v.name for v in orig_vars]
    name_to_idx = _build_name_to_idx(orig_var_names)
    n_orig = len(orig_var_names)

    if verbose:
        print(f"[scip_cut_extractor] {n_orig} original variables: {orig_var_names[:10]}...")

    # ---- Event handler to capture LP rows during solve ----
    class LPRowCapture(Eventhdlr):
        """Capture LP rows after each LP solve (root + nodes)."""
        def __init__(self):
            self.all_cuts: Dict[str, Dict[str, Any]] = {}  # keyed by cut name for dedup
            self.n_total_rows_seen = 0

        def eventinit(self):
            self.model.catchEvent(SCIP_EVENTTYPE.ROWADDEDSEPA, self)
            self.model.catchEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)
            self.model.catchEvent(SCIP_EVENTTYPE.LPSOLVED, self)

        def eventexit(self):
            self.model.dropEvent(SCIP_EVENTTYPE.ROWADDEDSEPA, self)
            self.model.dropEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)
            self.model.dropEvent(SCIP_EVENTTYPE.LPSOLVED, self)

        def eventexec(self, event):
            try:
                rows = self.model.getLPRowsData()
            except Exception:
                return

            self.n_total_rows_seen = max(self.n_total_rows_seen, len(rows))

            for row in rows:
                cut_name = row.name
                if cut_name in self.all_cuts:
                    continue  # Already captured
                cut_dict = _extract_row(row, name_to_idx)
                if cut_dict is not None:
                    self.all_cuts[cut_name] = cut_dict

    capture = LPRowCapture()
    m.includeEventhdlr(capture, "LPRowCapture",
                        "Capture LP rows (cuts) during SCIP solve")

    m.optimize()

    status = m.getStatus()
    primal_bound = m.getPrimalbound() if m.getNSols() > 0 else None
    dual_bound = m.getDualbound()

    if verbose:
        print(f"[scip_cut_extractor] status={status}  primal={primal_bound}  dual={dual_bound}")

    # ---- Collect final results ----
    cuts = list(capture.all_cuts.values())
    n_total_rows = capture.n_total_rows_seen

    # Also try to get LP rows right now (may still be available for some statuses)
    try:
        final_rows = m.getLPRowsData()
        for row in final_rows:
            cut_name = row.name
            if cut_name not in capture.all_cuts:
                cut_dict = _extract_row(row, name_to_idx)
                if cut_dict is not None:
                    cuts.append(cut_dict)
                    capture.all_cuts[cut_name] = cut_dict
        n_total_rows = max(n_total_rows, len(final_rows))
    except Exception:
        pass

    skipped_aux = n_total_rows - len(cuts)

    if verbose:
        print(
            f"[scip_cut_extractor] Extracted {len(cuts)} original-variable cuts "
            f"from {n_total_rows} total LP rows "
            f"(skipped {skipped_aux} with auxiliary vars)"
        )

    problem_name = Path(nl_path).stem
    return {
        "problem_name": problem_name,
        "nl_file": str(nl_path),
        "n_original_vars": n_orig,
        "original_var_names": orig_var_names,
        "scip_status": status,
        "primal_bound": primal_bound,
        "dual_bound": float(dual_bound) if dual_bound is not None else None,
        "n_total_lp_rows": n_total_rows,
        "n_extracted_cuts": len(cuts),
        "cuts": cuts,
    }


def load_cuts_as_constraints(
    json_path: str,
    n_vars: Optional[int] = None,
    *,
    dm_l: Optional[Any] = None,
    dm_u: Optional[Any] = None,
    max_cuts: int = 20,
    slack_tol: float = 0.0,
    cosine_dedup_threshold: float = 0.99,
    verbose: bool = True,
):
    """Load a SCIP cuts JSON file and return (A, b) tensors in ``A x <= b`` form.

    Each cut ``lhs <= sum(coef_i * x[idx_i]) + constant <= rhs`` is split into
    up to two rows:
      *  ``sum(coef_i * x[idx_i]) + constant <= rhs``  →  ``A_row x <= rhs - constant``
      *  ``lhs <= sum(coef_i * x[idx_i]) + constant``  →  ``-A_row x <= -(lhs - constant)``

    When ``dm_l`` and ``dm_u`` (box bounds) are provided, the loaded cuts are
    filtered and ranked to keep only the most useful ones:

      1. **Slack filter** – computes ``slack = b_i - max_{x in [l,u]} A_i x``.
         Cuts with ``slack > slack_tol`` are inactive over the entire domain and
         discarded (they can never tighten the relaxation).
      2. **Direction dedup** – among cuts whose normed direction vectors have
         cosine similarity > ``cosine_dedup_threshold``, only the tightest
         (smallest slack) is kept.
      3. **Top-K cap** – at most ``max_cuts`` are retained, ranked by slack
         (tightest first).

    Parameters
    ----------
    json_path : str
        Path to the JSON file produced by ``scip_cut_extractor.py``.
    n_vars : int, optional
        Expected number of variables.  Inferred from the JSON if not given.
    dm_l, dm_u : array-like, optional
        Box lower/upper bounds of shape ``(n_vars,)`` or ``(1, n_vars)``.
        Required for filtering.  If not provided, all cuts are returned (no
        filtering).
    max_cuts : int
        Maximum number of cuts to keep.  Default 20.
    slack_tol : float
        Slack threshold for inactive-cut removal.  Default 0.0 (keep only
        cuts that are active or violated somewhere in the box).
    cosine_dedup_threshold : float
        Cosine similarity threshold for treating two cuts as near-parallel.
        Default 0.99.
    verbose : bool
        Print filtering statistics.

    Returns
    -------
    (A, b) : tuple of torch.Tensor
        A has shape (m, n_vars), b has shape (m,).
        Returns None if no cuts were loaded.
    """
    import torch

    with open(json_path, "r") as f:
        data = json.load(f)

    if n_vars is None:
        n_vars = data["n_original_vars"]

    A_rows: List[List[float]] = []
    b_vals: List[float] = []

    for cut in data["cuts"]:
        indices = cut["var_indices"]
        coefficients = cut["coefficients"]
        constant = cut["constant"]
        lhs = cut["lhs"]
        rhs = cut["rhs"]

        # Build dense row
        row = [0.0] * n_vars
        for idx, coef in zip(indices, coefficients):
            if idx < n_vars:
                row[idx] = coef

        # Upper bound constraint: sum(coef*x) + constant <= rhs  →  A x <= rhs - constant
        if rhs is not None:
            A_rows.append(list(row))
            b_vals.append(rhs - constant)

        # Lower bound constraint: lhs <= sum(coef*x) + constant  →  -A x <= -(lhs - constant)
        if lhs is not None:
            neg_row = [-c for c in row]
            A_rows.append(neg_row)
            b_vals.append(-(lhs - constant))

    if len(A_rows) == 0:
        return None

    A = torch.tensor(A_rows, dtype=torch.float32)
    b = torch.tensor(b_vals, dtype=torch.float32)

    n_raw = A.shape[0]

    # --- Filtering pipeline ---
    if dm_l is not None and dm_u is not None:
        A, b = filter_cuts(
            A, b,
            dm_l=dm_l,
            dm_u=dm_u,
            max_cuts=max_cuts,
            slack_tol=slack_tol,
            cosine_dedup_threshold=cosine_dedup_threshold,
            verbose=verbose,
        )
        if A is None:
            return None

    if verbose and dm_l is not None:
        print(f"[scip_cut_extractor] {n_raw} raw rows -> {A.shape[0]} after filtering")

    return A, b


def filter_cuts(
    A: 'torch.Tensor',
    b: 'torch.Tensor',
    *,
    dm_l: Any,
    dm_u: Any,
    max_cuts: int = 20,
    slack_tol: float = 0.0,
    cosine_dedup_threshold: float = 0.99,
    verbose: bool = False,
) -> Optional[Tuple['torch.Tensor', 'torch.Tensor']]:
    """Filter ``A x <= b`` constraints to keep only the most useful cuts.

    Steps:
      1. Slack filter – remove cuts inactive over the entire box ``[dm_l, dm_u]``.
      2. Direction dedup – cluster near-parallel cuts; keep tightest per cluster.
      3. Top-K – keep at most ``max_cuts``, ranked by tightness (smallest slack).

    Parameters
    ----------
    A : Tensor (m, d)
    b : Tensor (m,)
    dm_l, dm_u : array-like (d,) or (1, d)
    max_cuts : int
    slack_tol : float
    cosine_dedup_threshold : float
    verbose : bool

    Returns
    -------
    (A_filtered, b_filtered) or None if all cuts are discarded.
    """
    import torch

    if A.shape[0] == 0:
        return None

    dm_l_t = torch.as_tensor(dm_l, dtype=A.dtype).reshape(-1)
    dm_u_t = torch.as_tensor(dm_u, dtype=A.dtype).reshape(-1)

    m, d = A.shape
    assert dm_l_t.shape[0] == d and dm_u_t.shape[0] == d, \
        f"Bound dim mismatch: A has {d} cols, bounds have {dm_l_t.shape[0]}"

    # -----------------------------------------------------------------
    # Step 1: Slack filter
    #   For A_i x <= b_i over box [l, u]:
    #     max_{x in box} A_i x = sum( max(A_ij * l_j, A_ij * u_j) )
    #   slack = b_i - max_{x in box} A_i x
    #   slack <= 0  means the cut is active/violated somewhere → useful
    #   slack > slack_tol means the cut is inactive everywhere → discard
    # -----------------------------------------------------------------
    Ax_at_upper = torch.where(A >= 0, A * dm_u_t.unsqueeze(0), A * dm_l_t.unsqueeze(0))
    max_Ax = Ax_at_upper.sum(dim=1)  # (m,)
    slack = b - max_Ax  # (m,)  negative or zero = active/violated

    active_mask = slack <= slack_tol
    n_active = int(active_mask.sum().item())

    if verbose:
        print(f"[filter_cuts] Slack filter: {m} -> {n_active} "
              f"(removed {m - n_active} inactive, tol={slack_tol})")

    if n_active == 0:
        return None

    A = A[active_mask]
    b = b[active_mask]
    slack = slack[active_mask]
    m = A.shape[0]

    # -----------------------------------------------------------------
    # Step 2: Direction-based deduplication
    #   Normalize rows and compute pairwise cosine similarity.
    #   Among near-parallel cuts (cosine > threshold), keep only the
    #   tightest (smallest slack).
    # -----------------------------------------------------------------
    if m > 1 and cosine_dedup_threshold < 1.0:
        norms = A.norm(dim=1, keepdim=True).clamp(min=1e-12)
        A_normed = A / norms  # (m, d)

        # Sort by slack (tightest first) so when we encounter duplicates,
        # the first one is always the best.
        order = slack.argsort()
        A_normed = A_normed[order]
        A = A[order]
        b = b[order]
        slack = slack[order]

        keep = torch.ones(m, dtype=torch.bool)
        # Greedy: for each cut (from tightest), mark near-parallel successors
        for i in range(m):
            if not keep[i]:
                continue
            # Compute cosine similarity with all remaining cuts
            if i + 1 < m:
                cos_sim = (A_normed[i].unsqueeze(0) * A_normed[i+1:]).sum(dim=1)
                # Mark near-parallel cuts as duplicates
                dup_mask = cos_sim.abs() > cosine_dedup_threshold
                if dup_mask.any():
                    dup_indices = torch.where(dup_mask)[0] + (i + 1)
                    keep[dup_indices] = False

        n_dedup = int(keep.sum().item())
        if verbose:
            print(f"[filter_cuts] Direction dedup: {m} -> {n_dedup} "
                  f"(removed {m - n_dedup} near-parallel, cos_thresh={cosine_dedup_threshold})")

        A = A[keep]
        b = b[keep]
        slack = slack[keep]
        m = A.shape[0]

    # -----------------------------------------------------------------
    # Step 3: Top-K by tightness (smallest slack first)
    # -----------------------------------------------------------------
    if m > max_cuts:
        topk_idx = slack.argsort()[:max_cuts]
        A = A[topk_idx]
        b = b[topk_idx]
        if verbose:
            print(f"[filter_cuts] Top-K cap: {m} -> {max_cuts}")
        m = max_cuts

    if m == 0:
        return None

    return A, b


def main():
    parser = argparse.ArgumentParser(
        description="Extract SCIP cutting planes from a .nl file."
    )
    parser.add_argument("nl_file", help="Path to the .nl problem file.")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path.  Default: <nl_stem>_scip_cuts.json next to the .nl file.",
    )
    parser.add_argument("--timelimit", type=float, default=90.0, help="SCIP time limit (s).")
    parser.add_argument(
        "--settings", "-s",
        default=None,
        help="SCIP settings file (e.g. scip_technical.set).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    args = parser.parse_args()

    nl_path = os.path.abspath(args.nl_file)
    if not os.path.isfile(nl_path):
        print(f"ERROR: {nl_path} not found.", file=sys.stderr)
        return 1

    result = _solve_and_extract(
        nl_path,
        timelimit=args.timelimit,
        settings_file=args.settings,
        verbose=not args.quiet,
    )

    out_path = args.output
    if out_path is None:
        stem = Path(nl_path).stem
        out_path = str(Path(nl_path).parent / f"{stem}_scip_cuts.json")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved {result['n_extracted_cuts']} cuts to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
