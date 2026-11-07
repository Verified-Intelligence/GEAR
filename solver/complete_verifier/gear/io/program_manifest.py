"""Program identity of the frozen legacy artifacts (G5 WP6, DESIGN section 2 L4 and section 5 'Program hash').

``program_hash`` is the sha256 of a canonical encoding of the node tuples that ``parse_module`` returns (or the
translator in ``legacy_translate`` produces) for the model as ``LiRPANet`` sees it. ``gear/data/legacy505.program.sha256``
holds one hash per (stem, structural_outputs) pair: ``False`` for all 505 stems, ``True`` (the model wrapped in
``_StructuralConstraintOutputWrapper``) for the 216 stems with linear rows, 721 entries in all.
``check_program`` refuses a hash that differs from the entry, or a pair with no entry, with ``LegacyArtifactError``.
The file is generated offline by the tool that traces with ``parse_module``; nothing here uses ``torch.jit``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ..spec import LegacyArtifactError

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "legacy505.program.sha256"


def _tensor(t: torch.Tensor):
    t = t.detach().to("cpu").contiguous()
    return ["T", str(t.dtype), list(t.shape), hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()]


def _enc(v):
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return _tensor(v)
    if isinstance(v, bool):
        return ["b", v]
    if isinstance(v, int):
        return ["i", v]
    if isinstance(v, float):
        return ["f", v.hex()]
    if isinstance(v, str):
        return ["s", v]
    if isinstance(v, (bytes, bytearray)):
        return ["y", hashlib.sha256(bytes(v)).hexdigest()]
    if isinstance(v, (list, tuple)):
        return [type(v).__name__, [_enc(x) for x in v]]
    if isinstance(v, dict):
        return ["d", [[_enc(k), _enc(x)] for k, x in v.items()]]
    return ["o", type(v).__name__, repr(v)]


def _perturbation(p):
    if p is None:
        return None
    return [type(p).__name__, [[k, _tensor(x)] for k, x in sorted(vars(p).items()) if isinstance(x, torch.Tensor)]]


def _node(n):
    param = n.param
    if isinstance(param, dict):  # op nodes: ordered name -> tensor (or None)
        param = ["d", [[k, _enc(x)] for k, x in param.items()]]
    else:
        param = _enc(param)
    return [n.name, n.ori_name, list(n.inputs), _enc(n.attr), n.op, param, n.model_input_index,
            getattr(n, "output_index", None), _perturbation(n.perturbation)]


def program_hash(nodes_op, nodes_in, nodes_out, template) -> str:
    """sha256 of the canonical encoding of the program: nodesOP, nodesIn, nodesOut, then the template's structure."""
    doc = [[_node(n) for n in nodes_op], [_node(n) for n in nodes_in], list(nodes_out), repr(template)]
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_program_manifest(path: Path = MANIFEST) -> tuple[dict, dict[tuple[str, bool], str]]:
    """(header, {(stem, structural_outputs): sha256}). Lines: ``<sha256>  <stem>  <0|1>``; ``# key: value`` is the header."""
    header, entries = {}, {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            k, _, v = line[1:].partition(":")
            header[k.strip()] = v.strip()
            continue
        digest, stem, flag = line.split()
        entries[(stem, flag == "1")] = digest
    return header, entries


def check_program(stem: str, structural_outputs: bool, digest: str, path: Path = MANIFEST) -> None:
    _, entries = load_program_manifest(path)
    want = entries.get((stem, bool(structural_outputs)))
    if want is None:
        raise LegacyArtifactError(f"no program manifest entry for ({stem!r}, structural_outputs={bool(structural_outputs)})")
    if want != digest:
        raise LegacyArtifactError(
            f"program hash of {stem!r} (structural_outputs={bool(structural_outputs)}) is {digest[:12]}..., "
            f"manifest has {want[:12]}...: the environment or the translator no longer reproduces the oracle")
