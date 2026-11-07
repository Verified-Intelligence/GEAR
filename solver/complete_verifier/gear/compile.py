"""``gear.compile``: dispatch by source form, fail-closed (G5 DESIGN section 2).

v1 has two direct paths. ``direct:legacy`` reads a frozen artifact triple (rules L1-L4: file hashes in the shipped
manifest, the module vocabulary, the vnnlib subset, program identity with ``gear/data/legacy505.program.sha256``).
``direct:torch`` is WP12a's ``gear.io.torch_direct.compile_torch``. Text formats, ``.gear`` files and ``Problem``s
need the IR stack (WP7-WP11) and raise ``UnsupportedModelError`` until it lands.

A ``Compiled`` is immutable: its arrays are private read-only copies, its program's tensors belong to it alone, and
``solve`` recomputes ``program_hash`` and ``hash`` before any model is built (``check_unchanged``).
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np
import torch

from .config import Options, UNSET, resolve
from .errors import LegacyArtifactError, ModelError, OptionError, UnsupportedModelError, InternalError
from .result import MAX, MIN, Sense
from .spec import RowSpec

# Frozen-artifact families: (model directory, vnnlib directory) under a benchmark root.
FAMILIES = {"gams_world": ("gams_world/mod_files_gams", "gams_world/vnnlib_gams"),
            "minlplib_linear": ("minlplib_linear/py_minlplib_linear", "minlplib_linear/vnnlib_minlplib_linear")}
# L2: the SHA-256 of the two modules.py files that define the 21 legacy classes (the gams_world and minlplib copies
# are identical files).
LEGACY_VOCAB_SHA256 = frozenset({"d637c4ee61149dad9f3921f73c236b9e5279cf8bf91d21e80b8c1b5bebdef5b0",
                                 "a71f2b829801baf9c7c9c18f39b9490be93849cf204f85a6b44a99b1a4528369"})
LEGACY_CONTAINERS = (torch.nn.ModuleList, torch.nn.ModuleDict)
# Option values that compile reads or that a compiled fact depends on (section 2 COMPILE_KEYS). The three feas_tol
# keys are equal after resolution, so one of them stands for feas_tol. A key the schema lacks resolves to None.
COMPILE_KEYS = ("solver.nlp.constr_viol_tol",
                "model.op_budget.hidden_squares", "model.op_budget.complete_squares", "model.op_budget.prove_rows",
                "model.op_budget.commit",
                "solver.nlp.open_bounds", "solver.nlp.minlp.enabled",
                "solver.nlp.expand_input_constraints_as_outputs")


def cfg_get(config: Mapping, dotted: str):
    """The value at a dotted key of a (frozen) nested config; None if absent."""
    node = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def compile_keys(config: Mapping) -> dict:
    keys = {k: cfg_get(config, k) for k in COMPILE_KEYS}
    passes = cfg_get(config, "gear.passes")
    for name in sorted(passes or {}):
        keys[f"gear.passes.{name}.op_budget"] = passes[name].get("op_budget")
    return keys


@dataclass(frozen=True)
class Legacy:
    """A frozen-artifact triple: ``<stem>_model.py``, ``<stem>_model_state.pt``, ``<stem>.vnnlib``."""
    model_py: Path
    state_pt: Path
    vnnlib: Path
    name: Optional[str] = None     # the instance name (keys the program manifest); default: the vnnlib stem

    def __post_init__(self):
        for f in ("model_py", "state_pt", "vnnlib"):
            object.__setattr__(self, f, Path(getattr(self, f)).resolve())
        if self.name is None:
            object.__setattr__(self, "name", self.vnnlib.stem)
        if not self.state_pt.name.endswith("_model_state.pt"):
            raise LegacyArtifactError(f"{self.state_pt}: not a <stem>_model_state.pt file")

    @property
    def stem(self) -> str:
        """The file stem of the model files (``-`` of the name may be ``_`` there)."""
        return self.state_pt.name[: -len("_model_state.pt")]


def read_legacy(name: str, *, root, family: str) -> Legacy:
    """The triple of benchmark instance ``name`` of ``family`` under ``root`` (stem variants as the CLI resolves them)."""
    if family not in FAMILIES:
        raise LegacyArtifactError(f"unknown family {family!r}; expected one of {sorted(FAMILIES)}")
    from .io.legacy import resolve_paths
    md, vb = (Path(root) / d for d in FAMILIES[family])
    pt, vnnlib, md, stem = resolve_paths(name, model_dir=md, vnnlib_base=vb)
    return Legacy(md / f"{stem}_model.py", pt, vnnlib, name)


@dataclass(frozen=True)
class IntegerReceipt:
    kind: str                  # "c0b" | "declared"
    box_sha256: str
    vtype_sha256: str
    fields: Mapping[str, str]


@dataclass(frozen=True)
class Compiled:
    path: str                                  # "direct:legacy" | "direct:torch"
    original: Any                              # Legacy | TorchSource: what Replay and every solve re-read
    problem: Any                               # the lowered IR Problem; None on the direct paths
    program: tuple                             # (nodes_op, nodes_in, nodes_out, template), parse_module format
    rowspec: RowSpec                           # C (CSR over the source outputs) and rhs, before the expansion
    linear: Optional[tuple]                    # (A [m, n], b [m]) float64 read-only, or None
    structural_outputs: bool                   # the wrapper variant (section 2 'Wrapper variant')
    box: tuple                                 # (lo, hi) float64 [n], read-only
    vtype: Optional[np.ndarray]
    sense: Sense
    assignment: Any                            # torch: OutputAssignment; legacy: None (rowspec holds the rows)
    compile_options: Mapping[str, Any]         # deep-frozen: COMPILE_KEYS + optimize, passes, constants
    pipeline: tuple
    proofs: tuple
    integer_receipt: Optional[IntegerReceipt]
    reasons: tuple
    source: Mapping[str, str]                  # identity of the source: file -> sha256, or {"export": sha256}
    names: tuple                               # variable names ("X_i")
    program_hash: str
    hash: str
    torch_compiled: Any = None                 # WP12a's TorchCompiled behind a direct:torch Compiled (not hashed)


# ---- canonical hashing (section 2 'Hash') ------------------------------------------------------------------------
def _arr(a) -> list:
    a = np.ascontiguousarray(a)
    return ["A", str(a.dtype), list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()]


def _canon(v):
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return ["i", int(v)]
    if isinstance(v, float):
        return ["f", v.hex()]
    if isinstance(v, np.ndarray):
        return _arr(v)
    if isinstance(v, torch.Tensor):
        return _arr(v.detach().cpu().numpy())
    if isinstance(v, Mapping):
        return ["d", [[str(k), _canon(v[k])] for k in sorted(v, key=str)]]
    if isinstance(v, (list, tuple)):
        return ["l", [_canon(x) for x in v]]
    if hasattr(v, "__dataclass_fields__"):
        return [type(v).__name__, [[f.name, _canon(getattr(v, f.name))] for f in fields(v)]]
    raise InternalError(f"cannot hash a {type(v).__name__}")


def compiled_hash(c: Compiled) -> str:
    rs = c.rowspec
    doc = {"path": c.path, "program_hash": c.program_hash, "structural_outputs": c.structural_outputs,
           "rowspec": [rs.num_outputs, _arr(rs.indptr), _arr(rs.indices), _arr(rs.data), _arr(rs.rhs)],
           "linear": None if c.linear is None else [_arr(c.linear[0]), _arr(c.linear[1])],
           "box": [_arr(c.box[0]), _arr(c.box[1])], "vtype": "none" if c.vtype is None else _arr(c.vtype),
           "sense": int(c.sense), "assignment": _canon(c.assignment), "compile_options": _canon(c.compile_options),
           "pipeline": _canon(c.pipeline), "proofs": _canon(c.proofs), "integer_receipt": _canon(c.integer_receipt),
           "source": _canon(c.source), "names": list(c.names)}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _program_hash(program) -> str:
    from .io.program_manifest import program_hash
    return program_hash(*program)


def check_unchanged(c: Compiled) -> None:
    """Recompute ``program_hash`` and ``hash`` (section 2 'Ownership'); ModelError if the Compiled was mutated."""
    if c.torch_compiled is not None:
        from .io.torch_direct import check_integrity
        check_integrity(c.torch_compiled)
    if _program_hash(c.program) != c.program_hash or compiled_hash(c) != c.hash:
        raise ModelError("Compiled was mutated after compile")


def _ro(a, dtype=None) -> np.ndarray:
    out = np.array(a, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _frozen_rowspec(rs: RowSpec) -> RowSpec:
    return RowSpec(rs.num_outputs, _ro(rs.indptr), _ro(rs.indices), _ro(rs.data), _ro(rs.rhs))


def _finish(**kw) -> Compiled:
    kw.setdefault("torch_compiled", None)
    c = Compiled(hash="", **kw)
    object.__setattr__(c, "hash", compiled_hash(c))
    return c


# ---- dispatch ----------------------------------------------------------------------------------------------------
def _check_passes(optimize, passes):
    if not (passes in ("default", "none") or (isinstance(passes, tuple) and all(isinstance(p, str) for p in passes))):
        raise OptionError(f"passes must be 'default', 'none' or a tuple of pass names, got {passes!r}")
    if isinstance(passes, tuple) and optimize is False:
        raise OptionError("a passes tuple implies optimize=True; optimize=False forbids it")
    if optimize not in (None, True, False):
        raise OptionError(f"optimize must be None, True or False, got {optimize!r}")


def compile(src, *, box=None, vtype=None, sense=MIN, objective=UNSET, le=(), eq=(), linear=None, optimize=None,
            passes="default", constants=UNSET, options: Optional[Options] = None) -> Compiled:
    """Compile ``src`` for ``solve`` (section 2). ``options`` are the Options the solve will use; compile reads only
    their COMPILE_KEYS. Every failure is raised here, before any solve."""
    _check_passes(optimize, passes)
    if constants is not UNSET and constants not in ("decimal", "dyadic"):
        raise OptionError(f"constants must be 'decimal' or 'dyadic', got {constants!r}")
    resolved = resolve(options if options is not None else Options())
    keys = compile_keys(resolved.config)
    frozen_opts = {"keys": keys, "optimize": optimize, "passes": passes,
                   "constants": None if constants is UNSET else constants}
    torch_args = dict(box=box, vtype=vtype, objective=objective, le=le, eq=eq, linear=linear)
    defaults = dict(box=None, vtype=None, objective=UNSET, le=(), eq=(), linear=None)
    is_torch = isinstance(src, torch.nn.Module) or (callable(src) and not isinstance(src, (Legacy, str, Path)))
    if not is_torch:
        given = [k for k, v in torch_args.items() if v is not defaults[k] and not (k in ("le", "eq") and v == ())]
        if sense != MIN:
            given.append("sense")
        if given:
            raise ModelError(f"torch-only arguments given for a {type(src).__name__} source: {', '.join(given)}")
    if isinstance(src, Legacy):
        if optimize is True or isinstance(passes, tuple):
            raise UnsupportedModelError("a legacy artifact is never rebuilt; rebuild from its source .mod instead")
        if constants == "decimal":
            raise OptionError("legacy constants are float64 tensors; constants='decimal' does not apply")
        return _compile_legacy(src, frozen_opts)
    if is_torch:
        try:
            from .io.torch_direct import compile_torch
        except ImportError as error:
            raise UnsupportedModelError(f"the direct torch path is not available: {error}") from None
        tc = compile_torch(src, sense=int(sense), optimize=optimize, passes=passes, constants=constants,
                           compile_keys=keys, **torch_args)
        return _from_torch(tc, frozen_opts)
    if isinstance(src, (str, Path)):
        raise UnsupportedModelError(f"{Path(src).suffix or src!s}: readers and the IR stack are not in v1 "
                                    "(WP7-WP11); use gear.read_legacy for a frozen artifact")
    raise UnsupportedModelError(f"unsupported source type {type(src).__name__}")


def _freeze_options(d):
    from .config import freeze
    return freeze(d, "compile_options")


def _from_torch(tc, frozen_opts) -> Compiled:
    n = len(tc.box[0])
    return _finish(path=tc.path, original=tc.original, problem=None, program=tuple(tc.program),
                   rowspec=_frozen_rowspec(tc.rowspec), linear=tc.linear, structural_outputs=bool(tc.structural_outputs),
                   box=tc.box, vtype=tc.vtype, sense=Sense(tc.sense), assignment=tc.assignment,
                   compile_options=_freeze_options(frozen_opts), pipeline=(), proofs=(), integer_receipt=None,
                   reasons=tuple(tc.reasons), source=MappingProxyType({"export": tc.original.export_sha256}),
                   names=tuple(f"X_{i}" for i in range(n)), program_hash=tc.program_hash, torch_compiled=tc)


# ---- direct:legacy -----------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _float64_default():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def _sha256_file(path: Path, cache: dict) -> str:
    if path not in cache:
        cache[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cache[path]


def read_artifact(src: Legacy):
    """Read the triple with the WP3 loader and check rule L1 strictly: every file it read (the triple and the
    ``modules.py`` files) must be listed in ``gear/data/legacy505.sha256`` with its SHA-256, keyed by its path
    relative to the one benchmark root the triple sits in (any root: identity is the relative path and the content).
    Returns ``(artifact, {relative path: sha256})``."""
    from .io.legacy import benchmark_root, load_manifest, manifest_key, read_legacy as read_files
    for p in (src.model_py, src.state_pt, src.vnnlib):
        if not p.is_file():
            raise LegacyArtifactError(f"{p}: missing")
    if len({benchmark_root(p) for p in (src.model_py, src.state_pt, src.vnnlib)}) != 1:
        raise LegacyArtifactError(f"{src}: the model files and the vnnlib are not under one benchmark root")
    art = read_files(src.name, model_dir=src.state_pt.parent, vnnlib_base=src.vnnlib.parent, verify=True)
    md = art.model_dir
    if (art.pt_path, art.vnnlib_path, md / f"{src.stem}_model.py") != (src.state_pt, src.vnnlib, src.model_py):
        raise LegacyArtifactError(f"{src}: the files do not form one artifact triple")
    _, hashes = load_manifest()
    files = [(art.vnnlib_path, art.vnnlib_path.name), (art.pt_path, art.pt_path.name),
             (src.model_py, src.model_py.name), (md / "modules.py", "modules.py")]
    if "minlplib/mod_files/modules.py" in art.sha256:            # the base file minlplib_linear's modules.py loads
        files.append((md.parent.parent / "minlplib" / "mod_files" / "modules.py", "minlplib/mod_files/modules.py"))
    source = {}
    for p, key in files:
        rel = manifest_key(p)
        if rel not in hashes:
            raise LegacyArtifactError(f"{p}: not a frozen artifact; rebuild from its .mod")
        if art.sha256[key] != hashes[rel]:                        # the loader hashed what it read
            raise LegacyArtifactError(f"{p}: sha256 {art.sha256[key]} differs from the frozen manifest ({hashes[rel]})")
        source[rel] = art.sha256[key]
    return art, MappingProxyType(source)


def _check_vocab(model: torch.nn.Module, model_py: Path) -> None:
    """L2: the root class comes from the artifact's own model file, every other module class from one of the two
    ``modules.py`` files (by SHA-256 of the defining file). torch's ``ModuleList``/``ModuleDict`` hold submodules
    and compute nothing; the artifacts use them as containers."""
    cache = {}
    for name, m in model.named_modules():
        cls = type(m)
        if cls in LEGACY_CONTAINERS:
            continue
        # The file the class's own functions were compiled from (minlplib_linear's modules.py execs the base
        # modules.py under a module name that is not in sys.modules, so inspect.getsourcefile cannot find it).
        files = {Path(v.__code__.co_filename).resolve() for v in vars(cls).values() if inspect.isfunction(v)}
        if len(files) != 1:
            raise LegacyArtifactError(f"module {name or '<root>'}: class {cls.__qualname__} has no single defining file")
        where = files.pop()
        ok = where == model_py if name == "" else where.is_file() and _sha256_file(where, cache) in LEGACY_VOCAB_SHA256
        if not ok:
            raise LegacyArtifactError(f"module {name or '<root>'}: class {cls.__qualname__} ({where}) is outside the "
                                      "legacy vocabulary")


def build_model(art, device="cpu"):
    """A fresh model of the artifact from its state (read once), as the CLI builds it."""
    from .io.legacy import _model_class, _state_once
    cls = _model_class(art)
    with _state_once(art) as calls:
        model = cls(pt_path=str(art.pt_path)).to(device)
    if len(calls) != 1:
        raise LegacyArtifactError(f"{art.pt_path}: the model constructor loaded the state {len(calls)} times")
    return model


def _model_dtype(model) -> torch.dtype:
    for t in list(model.parameters()) + list(model.buffers()):
        if t.is_floating_point() or t.is_complex():
            return t.dtype
    return torch.float64


def solver_view(model, linear, structural_outputs: bool):
    """The model as ``LiRPANet`` sees it (``_prepare_model`` then ``_maybe_augment_model_with_structural_constraints``)
    on the CPU."""
    model = model.to(device="cpu", dtype=torch.float64).eval()
    if structural_outputs:
        from api import _StructuralConstraintOutputWrapper
        dt = _model_dtype(model)
        A, b = linear
        model = _StructuralConstraintOutputWrapper(model, A.to(device="cpu", dtype=dt), b.to(device="cpu", dtype=dt))
        model = model.to(device="cpu", dtype=dt).eval()
    return model


def _compile_legacy(src: Legacy, frozen_opts) -> Compiled:
    from .io.legacy_translate import translate_module
    from .io.program_manifest import check_program
    art, source = read_artifact(src)
    lin = art.input_linear_constraints
    if lin is not None and (lin[0].ndim != 2 or lin[0].is_sparse):
        raise LegacyArtifactError(f"{src.state_pt}: linear_constr A must be a dense [m, n] matrix")
    structural = bool(lin is not None and frozen_opts["keys"]["solver.nlp.expand_input_constraints_as_outputs"])
    n = int(art.lower.shape[0])
    with _float64_default():
        model = build_model(art)
        _check_vocab(model, src.model_py)
        sense = MAX if bool(getattr(model, "is_maximization", False)) else MIN
        view = solver_view(model, lin, structural)
        x = torch.zeros(1, n, dtype=torch.float64)
        with torch.no_grad():
            width = int(view(x).shape[1])
        nodes_op, nodes_in, nodes_out, template = translate_module(view, x)
    program = (tuple(nodes_op), tuple(nodes_in), tuple(nodes_out), template)
    phash = _program_hash(program)
    check_program(src.name, structural, phash)                                 # L4
    s = int(lin[0].shape[0]) if structural else 0
    if width != art.rowspec.num_outputs + s:
        raise InternalError(f"{src.name}: the program has {width} outputs, the expanded rows {art.rowspec.num_outputs + s}")
    linear = None if lin is None else (_ro(lin[0].detach().cpu().numpy(), np.float64),
                                       _ro(lin[1].detach().cpu().reshape(-1).numpy(), np.float64))
    return _finish(path="direct:legacy", original=src, problem=None, program=program,
                   rowspec=_frozen_rowspec(art.rowspec), linear=linear, structural_outputs=structural,
                   box=(_ro(art.lower, np.float64), _ro(art.upper, np.float64)), vtype=None, sense=sense,
                   assignment=None, compile_options=_freeze_options(frozen_opts), pipeline=(), proofs=(),
                   integer_receipt=None, reasons=(), source=source, names=tuple(f"X_{i}" for i in range(n)),
                   program_hash=phash)


def __getattr__(name):
    """``TorchSource``, ``OutputAssignment`` and ``ExportedSourceModule`` are WP12a's (gear.io.torch_direct)."""
    if name in ("TorchSource", "OutputAssignment", "ExportedSourceModule"):
        from .io import torch_direct
        return getattr(torch_direct, name)
    raise AttributeError(name)
