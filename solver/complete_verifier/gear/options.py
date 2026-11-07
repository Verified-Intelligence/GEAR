"""Options of a solve (G5 DESIGN section 1). The resolver lives in ``gear/config.py`` (WP2); this module is the
public name of its types."""
from .config import Batch, DEFAULT_PRESET, Gap, Options, Resolved, UNSET, resolve  # noqa: F401

__all__ = ["Batch", "DEFAULT_PRESET", "Gap", "Options", "UNSET"]
