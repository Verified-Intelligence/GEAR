"""Fast loader for the frozen legacy artifacts (G5 WP3, anomaly A208).

``read_legacy`` reads the three files of an artifact (``X_model.py``, ``X_model_state.pt``, ``X.vnnlib``)
once each, verifies each file's sha256 against its entry in ``gear/data/legacy505.sha256`` (rule L1, F10;
the entry is keyed by the file's path relative to the benchmark root it sits in, so a byte-identical copy under
any root is the same artifact, and an unlisted or altered file raises), and parses the property with the strict
subset parser. ``load_legacy_fast`` then
hands the solver what ``nlp_example.run_problem`` builds today: the same ``VerificationSpec`` and the
same model, bit for bit. Differences are host-side only:

- no ``.vnnlib.compiled`` is read or written (nor the dense 2.9 GB pickle on the largest instance);
- the output rows stay sparse (``RowSpec``) until the dense ``C`` is made, once, for the solver;
- ``state.pt`` is read once (today: twice) and only the one model file is imported (today: the package
  ``__init__``, which imports every model of the benchmark).

Not covered (WP5 falls back to the vnnlib loader): ``--scip_cuts_file``.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import torch

from ..spec import LegacyArtifactError, RowSpec, parse_vnnlib_subset

SCRIPT_DIR = Path(__file__).resolve().parents[2]  # complete_verifier/
MANIFEST = Path(__file__).resolve().parents[1] / "data" / "legacy505.sha256"
_ROOT_PREFIX = "# root "


class LoadedProblem(NamedTuple):
    """What ``nlp_example.run_problem`` holds before it builds the solver."""

    pt_path: Path
    vnnlib_path: Path
    input_linear_constraints: Optional[tuple]
    spec: object   # api.VerificationSpec
    model: torch.nn.Module


@dataclass
class LegacyArtifact:
    name: str
    pt_path: Path
    vnnlib_path: Path
    model_dir: Path
    lower: np.ndarray
    upper: np.ndarray
    rowspec: RowSpec
    input_linear_constraints: Optional[tuple]
    state: object  # the loaded state.pt, handed once to the model constructor
    sha256: dict = field(default_factory=dict)  # file name -> hex digest (provenance)


# ---- manifest (L1) ----------------------------------------------------------------------------------
_manifest_cache: Optional[tuple[Path, dict[str, str]]] = None


def load_manifest(path: Path = MANIFEST) -> tuple[Optional[Path], dict[str, str]]:
    """``(root the manifest was generated from, {relative_path: sha256})``; ``(None, {})`` if the file is absent.
    The root is provenance only: identity is ``manifest_key(path)`` and the digest (``_check_hash``)."""
    global _manifest_cache
    if _manifest_cache is not None and path == MANIFEST:
        return _manifest_cache
    if not path.is_file():
        return None, {}
    root, hashes = None, {}
    for line in path.read_text().splitlines():
        if line.startswith(_ROOT_PREFIX):
            root = Path(line[len(_ROOT_PREFIX):].strip())
        elif line and not line.startswith("#"):
            digest, rel = line.split("  ", 1)
            hashes[rel] = digest
    if path == MANIFEST:
        _manifest_cache = (root, hashes)
    return root, hashes


def benchmark_root(path: Path) -> Path:
    """The benchmark root an artifact file sits in: every family keeps its files in ``<root>/<family>/<dir>/``."""
    return Path(path).resolve().parents[2]


def manifest_key(path: Path) -> str:
    """The manifest key of an artifact file: its path relative to its benchmark root (``<family>/<dir>/<file>``)."""
    return Path(path).resolve().relative_to(benchmark_root(path)).as_posix()


def _check_hash(path: Path, data: bytes, verify: bool) -> str:
    digest = hashlib.sha256(data).hexdigest()
    if verify:
        _, hashes = load_manifest()
        listed = hashes.get(manifest_key(path))
        if listed is None:
            raise LegacyArtifactError(f"{path}: not a frozen artifact; rebuild from its .mod")
        if listed != digest:
            raise LegacyArtifactError(f"{path}: sha256 {digest} differs from the frozen manifest ({listed})")
    return digest


# ---- paths (same rules as nlp_example._resolve_problem_paths) ---------------------------------------
def _resolve(path) -> Path:
    p = Path(path)
    return (p if p.is_absolute() else SCRIPT_DIR / p).resolve()


def resolve_paths(problem_name, *, model_dir, vnnlib_base, benchmark=None, benchmark_root=None):
    if benchmark and benchmark_root:
        br = Path(benchmark_root).resolve()
        if benchmark in ("minlplib", "minnlp"):
            md, vb = br / "minlplib" / "mod_files", br / "minlplib" / "vnnlib"
        elif benchmark == "minlplib_linear":
            md, vb = br / "minlplib_linear" / "py_minlplib_linear", br / "minlplib_linear" / "vnnlib_minlplib_linear"
        elif benchmark == "gams_world":
            md, vb = br / "gams_world" / "mod_files_gams", br / "gams_world" / "vnnlib_gams"
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
    else:
        md, vb = _resolve(model_dir), _resolve(vnnlib_base)

    def first(stem_variants, base, suffix):
        for cand in dict.fromkeys(stem_variants):
            p = base / f"{cand}{suffix}"
            if p.exists():
                return p.resolve(), cand
        raise FileNotFoundError(f"Could not find artifact for '{problem_name}' under {base} with suffix '{suffix}'.")

    pt_path, stem = first((problem_name, problem_name.replace("-", "_")), md, "_model_state.pt")
    vnnlib_path, _ = first((problem_name, problem_name.replace("_", "-")), vb, ".vnnlib")
    return pt_path, vnnlib_path, md, stem


# ---- reader -----------------------------------------------------------------------------------------
def read_legacy(problem_name, *, model_dir=None, vnnlib_base=None, benchmark=None, benchmark_root=None,
                verify=True) -> LegacyArtifact:
    """Read one artifact: three file reads, one ``torch.load``, no cache files, nothing written."""
    pt_path, vnnlib_path, md, stem = resolve_paths(
        problem_name, model_dir=model_dir, vnnlib_base=vnnlib_base,
        benchmark=benchmark, benchmark_root=benchmark_root)
    py_path = md / f"{stem}_model.py"
    sha = {}

    vnn_bytes = vnnlib_path.read_bytes()
    sha[vnnlib_path.name] = _check_hash(vnnlib_path, vnn_bytes, verify)
    try:
        text = vnn_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise LegacyArtifactError(f"{vnnlib_path}: not ASCII ({exc})") from None
    lower, upper, rowspec = parse_vnnlib_subset(text, str(vnnlib_path))
    del vnn_bytes, text

    pt_bytes = pt_path.read_bytes()
    sha[pt_path.name] = _check_hash(pt_path, pt_bytes, verify)
    state = torch.load(io.BytesIO(pt_bytes), map_location="cpu", weights_only=False)
    del pt_bytes
    sha[py_path.name] = _check_hash(py_path, py_path.read_bytes(), verify)
    mod_path = md / "modules.py"
    if not mod_path.is_file():
        raise LegacyArtifactError(f"{md}: modules.py not found")
    sha[mod_path.name] = _check_hash(mod_path, mod_path.read_bytes(), verify)
    if md.parent.name == "minlplib_linear":  # its modules.py loads the base file; no other family reads it
        base = md.parent.parent / "minlplib" / "mod_files" / "modules.py"
        if not base.is_file():
            raise LegacyArtifactError(f"{base}: not found (minlplib_linear's modules.py loads it)")
        sha["minlplib/mod_files/modules.py"] = _check_hash(base, base.read_bytes(), verify)

    return LegacyArtifact(problem_name, pt_path, vnnlib_path, md, lower, upper, rowspec,
                          _input_linear_constraints(state), state, sha)


def _input_linear_constraints(state):
    """As ``nlp_example._load_input_linear_constraints`` (A and b stay the tensors of the file)."""
    linear_constrs = state.get("linear_constr")
    if linear_constrs is None:
        return None
    A, b = linear_constrs.get("A"), linear_constrs.get("b")
    if A is None and b is None:
        return None
    if A is None or b is None:
        raise ValueError("linear_constr must provide both A and b, or neither.")
    return A, b


# ---- model (single-file import, state handed over) --------------------------------------------------
def _stem_to_class_name(stem: str) -> str:
    s = stem.replace("-", "_").replace(".", "_")
    return "".join(p.capitalize() for p in s.split("_") if p) + "Model"


def _model_class(art: LegacyArtifact):
    """Import ``<dir>/<stem>_model.py`` without running the package ``__init__``.

    A bare parent package is registered under the directory name (as the real import would), so the
    file's ``from .modules import ...`` resolves. A package that is already imported is reused.
    """
    md = art.model_dir
    pkg = md.name
    # Same sys.path edits as nlp_example._load_models_module: the minlplib_linear models import the
    # top-level ``modules`` (found through ``md``), the gams_world ones ``.modules``.
    for d in (md.parent, md):
        if str(d) in sys.path:
            sys.path.remove(str(d))
        sys.path.insert(0, str(d))
    top = sys.modules.get("modules")
    if top is not None and Path(getattr(top, "__file__", "") or "").resolve() != md / "modules.py":
        del sys.modules["modules"]  # left over from another benchmark directory
    loaded = sys.modules.get(pkg)
    if loaded is not None and [Path(p).resolve() for p in getattr(loaded, "__path__", ())] != [md]:
        # the same directory name under another root: drop that package and its modules, or the import below
        # would look in (or reuse a model module from) the other directory
        for key in [k for k in sys.modules if k == pkg or k.startswith(pkg + ".")]:
            del sys.modules[key]
    if pkg not in sys.modules:
        stub = types.ModuleType(pkg)
        stub.__path__ = [str(md)]
        sys.modules[pkg] = stub
    stem = art.pt_path.name[: -len("_state.pt")]  # "<stem>_model"
    mod = importlib.import_module(f"{pkg}.{stem}")
    return getattr(mod, _stem_to_class_name(art.name))


@contextlib.contextmanager
def _state_once(art: LegacyArtifact):
    """Serve the constructor's ``torch.load(pt_path, ...)`` from the state already in memory."""
    real_load, calls = torch.load, []

    def load(f, *args, **kwargs):
        if Path(str(f)).resolve() != art.pt_path:
            return real_load(f, *args, **kwargs)
        calls.append(f)
        return art.state

    torch.load = load
    try:
        yield calls
    finally:
        torch.load = real_load


# ---- the solver boundary ----------------------------------------------------------------------------
def _build_spec(art: LegacyArtifact):
    """``VerificationSpec.build_from_vnnlib`` minus ``read_vnnlib``, with the dense ``C`` made once.

    The handler is run on the input box and a one-row stand-in for ``C``, so the box (x_L, x_U) goes through
    exactly the code of today's path. ``C`` and ``rhs`` become tensors the way the handler makes them
    (float64 array -> default dtype), without the three intermediate copies of the handler.
    With ``specification.shrink_eps`` set the handler also rewrites the rows, so the whole path is used.
    """
    import arguments
    from api import VerificationSpec, _config_context
    from specifications import vnnlibHandler

    rs = art.rowspec
    box = list(zip(art.lower.tolist(), art.upper.tolist()))
    shape = [-1, len(box)]
    whole = arguments.Config["specification"]["shrink_eps"] is not None
    if whole:
        mat = rs.to_dense() if rs.nrows else np.zeros((0, rs.num_outputs), dtype=float)
        vnnlib = [(box, [(mat, rs.rhs)])]
    else:
        vnnlib = [(box, [(np.zeros((1, rs.num_outputs)), np.zeros(1))])]
    with _config_context({"general": {"store_all_specs_on_cpu": True}}):
        handler = vnnlibHandler(vnnlib, shape)
    x, c, rhs, or_spec_size, _, _ = handler.all_specs.get("cpu")
    lower = x.ptb.x_L.detach().cpu()
    upper = x.ptb.x_U.detach().cpu()
    if whole:
        c_cpu, rhs_cpu = c.detach().cpu(), rhs.detach().cpu()
        clauses = [[(c_cpu[0, :rs.nrows].clone(), rhs_cpu[0, :rs.nrows].clone())]] if rs.nrows else [[]]
        num_output = handler.num_output
        del vnnlib, handler, c, c_cpu
    else:
        dtype = torch.get_default_dtype()
        if rs.nrows:
            C = torch.from_numpy(rs.to_dense()).to(dtype=dtype)
            clauses = [[(C, torch.from_numpy(rs.rhs).to(dtype=dtype, copy=True))]]
        else:
            clauses = [[]]
        num_output = handler.num_output
    input_spec = VerificationSpec.InputSpec(
        lower, upper, input_linear_constraints=art.input_linear_constraints)
    output_spec = VerificationSpec.OutputSpec(clauses, num_outputs=num_output)
    return VerificationSpec(input_spec=input_spec, output_spec=output_spec)


def load_legacy_fast(problem_name, *, model_dir=None, vnnlib_base=None, benchmark=None,
                     benchmark_root=None, device="cpu", verify=True) -> LoadedProblem:
    """Drop-in for ``_resolve_problem_paths`` + ``_load_input_linear_constraints`` + ``build_spec`` +
    model construction in ``nlp_example.run_problem``; the caller supplies the global config and default
    dtype exactly as there (call after ``_set_runtime_environment`` and ``Config.parse_config``)."""
    art = read_legacy(problem_name, model_dir=model_dir, vnnlib_base=vnnlib_base, benchmark=benchmark,
                      benchmark_root=benchmark_root, verify=verify)
    spec = _build_spec(art)
    model_cls = _model_class(art)
    with _state_once(art) as calls:
        model = model_cls(pt_path=str(art.pt_path)).to("cuda:0" if device == "cuda" else "cpu")
    if len(calls) != 1:
        raise LegacyArtifactError(f"{art.pt_path}: the model constructor loaded the state {len(calls)} times")
    return LoadedProblem(art.pt_path, art.vnnlib_path, art.input_linear_constraints, spec, model)
