"""GEAR library API, public v1 without the builder.

    import gear as gr
    c = gr.compile(gr.read_legacy("hs071", root=ROOT, family="gams_world"))
    r = gr.solve(c, gr.Options(preset="gear-f0", timing="legacy"))

Torch sources (``gr.compile(module, box=..., objective=..., le=..., eq=...)``) take WP12a's direct path.
"""
from .compile import (COMPILE_KEYS, Compiled, IntegerReceipt, Legacy, compile, read_legacy)  # noqa: F401
from .errors import (GearError, InternalError, LegacyArtifactError, ModelError, OptionError, ParseError,  # noqa: F401
                     UnsupportedModelError, UnsupportedOperatorError, UnsupportedProblemError)
from .options import Batch, DEFAULT_PRESET, Gap, Options, UNSET  # noqa: F401
from .result import MAX, MIN, Provenance, Replay, Result, Sense, Stats, Status, Termination  # noqa: F401
from .solver import Solver, solve  # noqa: F401
from .dynamics import (DynamicsOptions, DynamicsProblem, DynamicsResult, DynamicsRound,  # noqa: F401
                       VAPINNOptions, VAPINNTrainer, solve_dynamics)

inf = float("inf")


def __getattr__(name):
    """``TorchSource`` and ``OutputAssignment`` are WP12a's (``gear.io.torch_direct``)."""
    if name in ("TorchSource", "OutputAssignment"):
        from .io import torch_direct
        return getattr(torch_direct, name)
    raise AttributeError(name)
