"""Option resolver and presets (G5 WP2, DESIGN section 1 'Options rules' and 'Presets').

``resolve(options)`` turns an ``Options`` into the full effective legacy config: the preset (a fully resolved config
stored in ``gear/presets/<name>.yaml``) laid over the schema defaults, then the fields that are not ``UNSET``, then
the ``advanced`` dotted keys. It validates against the ``arguments.py`` schema and returns a deep-frozen mapping.
``thaw`` turns that back into the plain nested dict that ``ConfigBuilder.to_dict()`` gives.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import dataclass, field, replace as _replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional

import yaml

from .errors import OptionError  # (re-exported: gear.config.OptionError is gear.errors.OptionError)

PRESET_DIR = Path(__file__).resolve().parent / "presets"
DEFAULT_PRESET = "gear-f0"


class _Unset:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "UNSET"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


UNSET = _Unset()


@dataclass(frozen=True)
class Gap:
    abs: float | _Unset = UNSET
    rel: float | _Unset = UNSET


@dataclass(frozen=True)
class Batch:
    domains: int | Literal["auto"] | _Unset = UNSET
    gpu_domain_mb: int | None | _Unset = UNSET


def freeze(value, where="value"):
    """Deep freeze: Mapping -> MappingProxyType (str keys), list/tuple -> tuple, set -> frozenset."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise OptionError(f"{where}: mapping keys must be strings")
        return MappingProxyType({k: freeze(v, f"{where}.{k}") for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v, where) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(v, where) for v in value)
    raise OptionError(f"{where}: unsupported value type {type(value).__name__}")


def thaw(value):
    """Inverse of ``freeze`` for the config: MappingProxyType -> dict, tuple -> list."""
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class Options:
    preset: str = DEFAULT_PRESET
    time_limit: float | _Unset = UNSET
    timing: Literal["solve", "legacy"] = "solve"
    gap: Gap = Gap()
    feas_tol: float | _Unset = UNSET
    device: str | _Unset = UNSET
    dtype: str | _Unset = UNSET
    batch: Batch = Batch()
    seed: int | _Unset = UNSET
    advanced: Mapping[str, Any] = field(default_factory=dict)
    certificate: bool = False
    log: Optional[str] = None
    verbosity: int = 1

    def __post_init__(self):
        object.__setattr__(self, "advanced", freeze(dict(self.advanced), "advanced"))

    def replace(self, **fields) -> "Options":
        return _replace(self, **fields)

    def resolved(self) -> Mapping[str, Any]:
        return resolve(self).config


@dataclass(frozen=True)
class Resolved:
    config: Mapping[str, Any]            # the full effective legacy config, deep-frozen
    preset: str
    preset_sha256: str
    historical: bool
    cpu_solver_free: bool
    anomalies: tuple


def _path(config, dotted):
    node = config
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.get(p) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            return None, parts[-1]
    return node, parts[-1]


def _get(config, dotted):
    node, leaf = _path(config, dotted)
    return None if node is None else node.get(leaf)


def _set(config, dotted, value):
    parts = dotted.split(".")
    node = config
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


_schema_cache: dict = {}


def schema_defaults() -> dict:
    """The arguments.py defaults as the nested dict ``ConfigBuilder`` starts from (a private handler)."""
    if not _schema_cache:
        import arguments
        h = arguments.ConfigHandler()
        h.construct_config_dict(h.default_args)
        _schema_cache["d"] = copy.deepcopy(h.all_args)
    return copy.deepcopy(_schema_cache["d"])


def load_preset(name: str) -> tuple[dict, dict, str]:
    """(config, metadata, sha256 of the file). ``name`` is a preset in gear/presets or a path to a preset file."""
    path = Path(name)
    if not path.suffix:
        path = PRESET_DIR / f"{name}.yaml"
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise OptionError(f"unknown preset {name!r}: {error}") from None
    doc = yaml.safe_load(raw)
    if not isinstance(doc, dict) or "config" not in doc:
        raise OptionError(f"preset {name!r} has no 'config' section")
    meta = dict(doc.get("preset") or {})
    meta.setdefault("historical", False)
    meta.setdefault("cpu_solver_free", False)
    return doc["config"], meta, hashlib.sha256(raw).hexdigest()


# Field -> legacy keys (DESIGN section 1). dtype, batch.domains and gap are handled in _field_edits.
FEAS_KEYS = ("solver.nlp.constr_viol_tol", "solver.nlp.bab_feas_tol", "solver.nlp.ipopt_constr_viol_tol")
GAP_KEYS = {"abs": "solver.nlp.bab_primal_dual_abs_tol", "rel": "solver.nlp.bab_primal_dual_rel_tol"}


def _check_type(key, value, default):
    """A value for a schema leaf must have the default's type (int is accepted for float and becomes float)."""
    if default is None:
        return value
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, int):
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(default, float):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if ok:
            value = float(value)
    elif isinstance(default, str):
        ok = isinstance(value, str)
    elif isinstance(default, (list, tuple)):
        ok = isinstance(value, (list, tuple))
        if ok:
            value = list(value)
    else:
        ok = type(value) is type(default)
    if not ok:
        raise OptionError(f"{key}: expected {type(default).__name__}, got {type(value).__name__} {value!r}")
    return value


def _field_edits(o: Options) -> dict[str, Any]:
    edits: dict[str, Any] = {}
    if o.time_limit is not UNSET:
        edits["bab.override_timeout"] = o.time_limit
    for name, key in GAP_KEYS.items():
        v = getattr(o.gap, name)
        if v is not UNSET:
            edits[key] = v
    if o.feas_tol is not UNSET:
        for key in FEAS_KEYS:
            edits[key] = o.feas_tol
    if o.device is not UNSET:
        if not isinstance(o.device, str):
            raise OptionError("device must be a string: no multi-GPU in v1")
        edits["general.device"] = o.device
    if o.dtype is not UNSET:
        if o.dtype != "float64":
            raise OptionError(f"dtype {o.dtype!r}: only 'float64' is supported")
        edits["general.double_fp"] = True
    if o.seed is not UNSET:
        edits["general.seed"] = o.seed
    d = o.batch.domains
    if d is not UNSET and d != "auto":
        if isinstance(d, bool) or not isinstance(d, int) or d < 1:
            raise OptionError(f"batch.domains must be a positive int or 'auto', got {d!r}")
        edits["solver.batch_size"] = d
        edits["solver.auto_enlarge_batch_size"] = False
    if o.batch.gpu_domain_mb is not UNSET:
        edits["bab.domain_storage_gpu_mb"] = o.batch.gpu_domain_mb
    return edits


def _check_a67(config, meta):
    if meta["historical"]:
        return
    text = _get(config, "model.bound_opts")
    try:
        opts = ast.literal_eval(text) if isinstance(text, str) and text else {}
    except (ValueError, SyntaxError):
        raise OptionError("model.bound_opts is not a Python dict literal") from None
    if not (isinstance(opts, dict) and isinstance(opts.get("fp_enclosure"), dict)
            and opts["fp_enclosure"].get("affine_backward") is True):
        raise OptionError("preset requires model.bound_opts fp_enclosure.affine_backward = True (A67)")


def resolve(options: Options) -> Resolved:
    if options.timing not in ("solve", "legacy"):
        raise OptionError(f"timing must be 'solve' or 'legacy', got {options.timing!r}")
    preset_cfg, meta, digest = load_preset(options.preset)
    schema = schema_defaults()
    config = schema
    _deep_merge(config, preset_cfg)
    edits = _field_edits(options)
    advanced = {k: thaw(v) for k, v in options.advanced.items()}
    clash = sorted(set(edits) & set(advanced))
    if clash:
        raise OptionError(f"field and advanced key set the same legacy key: {', '.join(clash)}")
    defaults = schema_defaults()
    for key, value in {**edits, **advanced}.items():
        parent, leaf = _path(defaults, key)
        if parent is None or leaf not in parent or isinstance(parent[leaf], dict):
            raise OptionError(f"unknown option key {key!r}")
        _set(config, key, _check_type(key, value, parent[leaf]))
    feas = {_get(config, k) for k in FEAS_KEYS}
    if len(feas) != 1:
        raise OptionError(f"the feas_tol keys disagree: { {k: _get(config, k) for k in FEAS_KEYS} }")
    _check_a67(config, meta)
    anomalies = ("historical_preset:ref505:a67_off",) if meta.get("name") == "ref505" else ()
    return Resolved(freeze(config), options.preset, digest, bool(meta["historical"]), bool(meta["cpu_solver_free"]),
                    anomalies)


def _deep_merge(base: dict, updates: Mapping):
    for key, value in updates.items():
        if isinstance(value, Mapping):
            node = base.setdefault(key, {})
            _deep_merge(node, value)
        else:
            base[key] = copy.deepcopy(value)
