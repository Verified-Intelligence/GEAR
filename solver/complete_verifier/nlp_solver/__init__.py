"""NLP-based falsification solver for neural network verification.

This module provides NLP solver integration (ALM-PGD and IPOPT)
to solve the root primal problem for finding counterexamples.
"""

# Utilities
from .utils import (
    _infer_device_dtype,
    _eval_mode,
    _split_or_clauses,
)

# Problem definitions
from .problem import (
    NLPResult,
    NNProblemSingleDomain,
)

# ALM-PGD solver
from .alm import (
    ALMConfig,
    ALMResult,
    ALMPGDSolver,
    resolve_alm_restart_count,
)

# Statistics and verification
from .statistics import (
    verify_counterexample,
)

# Falsification entry points
from .falsification import (
    ComparisonResult,
    compare_solvers,
)

# IPOPT solver (cyipopt when available, SciPy local fallback otherwise)
from .ipopt_solver import (
    IPOPTResult,
    NNProblemIPOPT,
    IPOPTSolver,
    ipopt_falsification,
    IPOPT_AVAILABLE,
    CYIPOPT_AVAILABLE,
    SCIPY_AVAILABLE,
)

__all__ = [
    # Utilities
    '_infer_device_dtype',
    '_eval_mode',
    '_split_or_clauses',
    # Problem definitions
    'NLPResult',
    'NNProblemSingleDomain',
    # ALM-PGD solver
    'ALMConfig',
    'ALMResult',
    'ALMPGDSolver',
    'resolve_alm_restart_count',
    # Statistics
    'verify_counterexample',
    # Falsification
    'ComparisonResult',
    'compare_solvers',
    # IPOPT solver
    'IPOPTResult',
    'NNProblemIPOPT',
    'IPOPTSolver',
    'ipopt_falsification',
    'IPOPT_AVAILABLE',
    'CYIPOPT_AVAILABLE',
    'SCIPY_AVAILABLE',
]
