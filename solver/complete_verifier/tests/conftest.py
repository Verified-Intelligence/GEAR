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
"""
Pytest configuration for complete_verifier tests.
"""
import warnings

import pytest
import torch


# Reusable skip marker for tests requiring CUDA
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


def pytest_configure(config):
    """Configure pytest warning filters."""
    config.addinivalue_line("markers", "slow: long-running test (> 30 s)")
    # Suppress onnx2pytorch warnings about non-writable numpy arrays
    # This is a known issue in onnx2pytorch and doesn't affect functionality
    warnings.filterwarnings(
        'ignore',
        message='The given NumPy array is not writable',
        category=UserWarning
    )
    # Suppress onnx2pytorch experimental batch_size warning
    warnings.filterwarnings(
        'ignore',
        message='Using experimental implementation that allows',
        category=UserWarning
    )
    # Suppress load_model warning about conversion correctness check
    warnings.filterwarnings(
        'ignore',
        message="Not able to check model's conversion correctness",
        category=UserWarning
    )


@pytest.fixture(autouse=True)
def _restore_torch_default_dtype():
    """Keep the global default dtype from leaking between tests.

    Solver entry points (api.py, nlp_example.py, abcrown.py) call
    torch.set_default_dtype() and do not restore it. Without this, whether
    later modules pass depends on collection order.
    """
    default_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(default_dtype)
