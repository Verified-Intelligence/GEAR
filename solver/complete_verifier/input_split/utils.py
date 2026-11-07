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
import time
import sys

from dataclasses import dataclass
from contextlib import contextmanager

import arguments
from auto_LiRPA.concretize_func import construct_constraints


def relax_constraint_rhs(rhs):
    """The verify predicate's tolerance on the constraint-row thresholds, in solving mode. Only the RowPool reads it
    (rowpool.spec_tau); every bound consumer reads the pool's thresholds. Rows 1: get
    `solver.nlp.constr_viol_tol` added, row 0 (the objective / incumbent) is untouched.

    The verifier's feasibility semantic is "f_k <= rhs_k + constr_viol_tol": the filter
    (branching_domains.filter_verified_domains) adds the tolerance before declaring a
    domain constraint-infeasible, and the ALM/local solvers accept an incumbent whose
    violation is within it. A cut built at the raw rhs -- a clip, a constrained-concretize
    plane `plane_k(x) <= rhs_k`, a dual-ascent rhs -- removes points the filter and the
    primal both treat as feasible. On Burgers G1 the reduced vnnlib keeps the boundary pair
    Y_99 = sum_k sin(k pi) x_k (mathematically 0, numerically 1.2e-16 (x0 - 2 x1 + 3 x2)),
    Y_100 = -Y_99, both <= 0: at zero tolerance the root clip shrank the root box to the
    slab of the hyperplane x0 - 2 x1 + 3 x2 = 0 and discarded the certified optimum (which
    violates Y_99 <= 0 by 2.4e-16) before BaB began, so the dual collapsed onto the
    incumbent. Relaxing a cut only enlarges the feasible set it encodes, so every bound
    derived from it stays a valid lower bound. No-op outside solving mode and for a spec
    with no constraint rows."""
    if rhs is None or not torch.is_tensor(rhs) or rhs.ndim < 2 or rhs.shape[-1] < 2:
        return rhs
    try:
        if not arguments.Config["solving"]["solving_mode"]:
            return rhs
        tol = float(arguments.Config["solver"]["nlp"]["constr_viol_tol"])
    except (KeyError, TypeError):
        return rhs
    if tol <= 0:
        return rhs
    rhs = rhs.clone()
    rhs[..., 1:] = rhs[..., 1:] + tol
    return rhs


def relax_input_constraint_rhs(b):
    """Right-hand sides of the base linear constraints A x <= b as a dual bound must see
    them in solving mode: `solver.nlp.constr_viol_tol` added, as relax_constraint_rhs does
    for the spec rows (the RowPool reads it through rowpool.base_tau, and the spec handler
    through relax_input_constraints). The primal accepts A x <= b + constr_viol_tol, so an exact row
    excludes points it accepts (pooling_bental5tp: a point violating a linear row by
    8.6e-8 sat 1.4e-6 below the reported bound). No-op outside solving mode."""
    if b is None or not torch.is_tensor(b):
        return b
    try:
        if not arguments.Config["solving"]["solving_mode"]:
            return b
        tol = float(arguments.Config["solver"]["nlp"]["constr_viol_tol"])
    except (KeyError, TypeError):
        return b
    return b + tol if tol > 0 else b


def relax_input_constraints(standard_constraints):
    """The base rows A x + c <= 0 (standard form, see construct_constraints) as every dual
    consumer must see them in solving mode: c - constr_viol_tol, i.e. relax_input_constraint_rhs
    on the A x <= b form. Applied once, where the rows are built from the problem's A x <= b
    (vnnlibHandler); the root CROWN pass, clipping, constrained concretization, the domain
    filter, the dual-ascent LP and the lifted OBBT all read the relaxed rows as they are.

    The primal and the verifier accept A x <= b + constr_viol_tol, so a bound that enforces
    a row at its raw rhs excludes points they accept. ex7_2_1: the optimum with the rows at
    +1e-4 is 1222.50869, with the input rows exact 1222.68643 (its second input row is active
    at +1e-4 there); clipping, concretization and the filter used b, and D rose to 1222.654
    above the optimum 1222.509 once the incumbent was suboptimal. Rows appended later are
    not relaxed again: the square-row and convex cuts carry their own tolerance, the symmetry
    ordering rows need none. No-op outside solving mode."""
    if standard_constraints is None:
        return None
    A, c = standard_constraints
    return A, -relax_input_constraint_rhs(-c)


def bound_thresholds(rowpool, thresholds):
    """The thresholds a bound must see: the constraint rows relaxed by the verify predicate's tolerance, which the
    instance's RowPool (solving mode) applies once. `thresholds` as given without a pool (NN verification). In solving
    mode a missing pool is a caller's bug and raises: the raw rhs would cut off points the verify predicate accepts."""
    if rowpool is not None:
        return rowpool.thresholds(thresholds)
    try:
        solving_mode = arguments.Config["solving"]["solving_mode"]
    except KeyError:                     # partial configs, e.g. in unit tests
        solving_mode = False
    if solving_mode:
        raise ValueError("a bound in solving mode needs the instance's RowPool (history.rowpool): "
                         "the raw thresholds lack the verify predicate's tolerance")
    return thresholds


def construct_cut_constraints(lA, lbias, thresholds, batchsize, x_dim, rowpool):
    """`construct_constraints` (standard form A x + b <= 0 from `lA x + lbias <= thresholds`)
    with the constraint rows' thresholds relaxed by the pool (`rowpool`, None without one). Every
    place that turns lower planes into cuts for concretization must go through this, so the
    cuts agree with the filter's feasibility tolerance.

    A +inf threshold (the objective row before any incumbent exists) is no cut at all;
    it becomes the redundant row `0 x - 1 <= 0` so no -inf offset reaches the
    constrained concretization."""
    thresholds = bound_thresholds(rowpool, thresholds)
    if thresholds is not None and torch.is_tensor(thresholds):
        inactive = torch.isposinf(thresholds.reshape(batchsize, -1))
        if inactive.any():
            lA = lA.reshape(batchsize, -1, x_dim).masked_fill(inactive.unsqueeze(-1), 0.0)
            lbias = lbias.reshape(batchsize, -1).masked_fill(inactive, -1.0)
            thresholds = thresholds.reshape(batchsize, -1).masked_fill(inactive, 0.0)
    return construct_constraints(lA, lbias, thresholds, batchsize, x_dim)


def initial_verify_criterion(lbs, rhs):
    """check whether verify successful"""
    # lbs: b, n_bounds (already multiplied with c in compute_bounds())
    verified_idx = torch.any(
        (lbs - rhs) > 0, dim=-1
    )  # return boolean results in x's batch-wise
    if verified_idx.all():  # check whether all x verified
        print("Verified by initial bound!")
        return True, torch.where(verified_idx == 0)[0]
    else:
        return False, torch.where(verified_idx == 0)[0]

@dataclass
class Timer:
    total_func_time: float = 0.0
    total_prepare_time: float = 0.0
    total_bound_time: float = 0.0
    total_beta_bound_time: float = 0.0
    total_transfer_time: float = 0.0
    total_finalize_time: float = 0.0

    def __init__(self):
        self.time_start = {}
        self.time_last = {}
        self.time_sum = {}

    def start(self, name):
        self.time_start[name] = time.time()
        if name not in self.time_sum:
            self.time_sum[name] = 0

    def add(self, name):
        self.time_last[name] = time.time() - self.time_start[name]
        self.time_sum[name] += self.time_last[name]

    def print(self):
        print('Time: ', end='')
        for k, v in self.time_last.items():
            print(f'{k} {v:.4f}', end='    ')
        print()
        print('Accumulated time: ', end='')
        for k, v in self.time_sum.items():
            print(f'{k} {v:.4f}', end='    ')
        print()

class Stats:
    def __init__(self):
        self.visited = 0  # number of domains bounded in BaB
        self.storage_depth = 0  # maximum possible splits per BaB round
        self.timer = Timer()

class PrefixedStdout:
    def __init__(self, prefix, original_stdout):
        self.prefix = prefix
        self.original_stdout = original_stdout
        self.at_start = True

    def write(self, text):
        """
        This will be called by print() to write data. 
        """
        if not text:
            self.original_stdout.write(text)
            return

        lines = text.split('\n')
        
        if self.at_start and lines[0]:
            self.original_stdout.write(self.prefix + lines[0])
            self.at_start = False
        elif lines[0]:
            self.original_stdout.write(lines[0])
            
        for line in lines[1:-1]:
            self.original_stdout.write('\n')
            self.original_stdout.write(self.prefix + line)
        
        if len(lines) > 1:
            self.original_stdout.write('\n')
            if lines[-1]:
                self.original_stdout.write(self.prefix + lines[-1])
                self.at_start = False
            else:
                self.at_start = True
        
        elif not text.endswith('\n'):
            self.at_start = False
            
    def flush(self):
        """
        Print() might call flush(). Just call the flush() from original stdout stream is enough.
        """
        self.original_stdout.flush()

@contextmanager
def prefix_print_output(prefix="\t"):
    original_stdout = sys.stdout
    
    sys.stdout = PrefixedStdout(prefix, original_stdout)
    
    try:
        yield
    finally:
        sys.stdout = original_stdout

@contextmanager
def no_op_context():
    try:
        yield
    finally:
        pass