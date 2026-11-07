"""Errors of the GEAR library (G5 DESIGN section 6). They are raised only before search starts; after search starts a
failure becomes a ``Result``. ``gear.config.OptionError`` and ``gear.spec.LegacyArtifactError`` are these classes."""


class GearError(Exception):
    """Base class of every error the library raises before search."""


class ParseError(GearError):
    """A source file (or a ``.gear`` file) cannot be read."""

    def __init__(self, message, path=None, line=None, col=None):
        super().__init__(message)
        self.path, self.line, self.col = path, line, col


class UnsupportedModelError(GearError):
    """A reader construct, an export failure or an unmapped op."""


class ModelError(GearError):
    """Builder or argument misuse: a missing or malformed box, an output assignment, a mutated ``Compiled``."""


class UnsupportedProblemError(GearError):
    """Integer variables or an infinite box without an enabled route (F2)."""


class UnsupportedOperatorError(GearError):
    """An operator with no lowering rule."""


class OptionError(GearError, ValueError):
    """An option is unknown, has the wrong type, conflicts with another, or violates a preset rule."""


class LegacyArtifactError(GearError):
    """A legacy artifact is missing, altered (hash mismatch) or outside the supported subset."""


class InternalError(GearError):
    """A broken invariant: a value-check mismatch, a handoff check, a program that differs from its record."""
