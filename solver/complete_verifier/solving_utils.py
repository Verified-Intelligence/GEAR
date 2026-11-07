#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
#########################################################################
"""Shared utilities for primal-dual solving mode."""

from pathlib import Path

import torch
from torch import Tensor

import arguments


def is_primal_dual_converged(
    primal_solution: Tensor,
    dual_solution: Tensor,
    abs_tol: float = 1e-5,
    rel_tol: float = 0.0,
) -> bool:
    """Return whether a minimization primal/dual gap is certified closed.

    Both values must use the verifier's internal minimization convention.
    A dual above the primal by more than ``abs_tol`` is inconsistent and is
    never accepted. Relative tolerance uses
    ``max(1, abs(primal), abs(dual))`` as its scale.
    """
    primal = torch.as_tensor(primal_solution).reshape(-1)
    dual = torch.as_tensor(
        dual_solution, device=primal.device, dtype=primal.dtype
    ).reshape(-1)
    if primal.numel() != 1 or dual.numel() != 1:
        raise ValueError("primal and dual solutions must both be scalar")
    primal, dual = primal[0], dual[0]

    if not (torch.isfinite(primal) and torch.isfinite(dual)):
        return False

    gap = float((primal - dual).item())
    if gap < -abs_tol:
        return False
    if gap <= abs_tol:
        return True

    scale = max(1.0, abs(float(primal.item())), abs(float(dual.item())))
    return gap / scale <= rel_tol


def load_fixed_primal(reference: Tensor) -> Tensor | None:
    """Load the configured finite scalar incumbent on ``reference``'s device.

    ``None`` is returned when external solver results are disabled. The result
    file is expected to have been written with :func:`torch.save`.
    """
    solving_args = arguments.Config["solving"]
    if not solving_args["enable_solver_res"]:
        return None

    result_path = solving_args["solver_res"]
    if not result_path:
        raise ValueError(
            "solving.solver_res is required when enable_solver_res is true"
        )
    if not Path(result_path).is_file():
        raise FileNotFoundError(
            f"solving.solver_res does not exist: {result_path}"
        )

    loaded = torch.load(
        result_path,
        map_location=reference.device,
        weights_only=False,
    )
    primal = torch.as_tensor(
        loaded, device=reference.device, dtype=reference.dtype
    ).reshape(-1)
    if primal.numel() != 1:
        raise ValueError(
            "solving.solver_res must torch.save a scalar or 1-element tensor; "
            f"got shape {tuple(primal.shape)}"
        )
    if not torch.isfinite(primal[0]):
        raise ValueError("solving.solver_res must contain a finite scalar")
    return primal[0]


def fixed_primal_dual_enabled() -> bool:
    """Return whether BaB is certifying an externally supplied incumbent."""
    solving_args = arguments.Config["solving"]
    return bool(
        solving_args["solving_mode"]
        and solving_args["enable_solver_res"]
    )
