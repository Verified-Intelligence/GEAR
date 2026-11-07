"""Program manifest gear/data/legacy505.program.sha256 (G5 WP6, DESIGN section 2 L4). CPU only."""
import pytest
import torch

from gear.io import program_manifest as pm
from gear.spec import LegacyArtifactError
from tests.test_legacy_translate import BENCH, SAMPLE, _build_real, f64, mods  # noqa: F401 (fixture: float64 default)


def test_manifest_has_721_entries_keyed_by_stem_and_variant():
    header, entries = pm.load_program_manifest()
    assert len(entries) == 721
    assert sum(1 for (_, wrapped) in entries if not wrapped) == 505 and sum(1 for (_, wrapped) in entries if wrapped) == 216
    assert {s for s, w in entries if w} <= {s for s, w in entries if not w}
    assert header["torch"] == torch.__version__ or header["torch"].startswith("2.")
    assert header["base"].startswith("62094fd6e")
    assert all(len(h) == 64 for h in entries.values())


def test_check_program_refuses_a_missing_entry_and_a_mismatch():
    _, entries = pm.load_program_manifest()
    (stem, wrapped), digest = next(iter(entries.items()))
    pm.check_program(stem, wrapped, digest)
    with pytest.raises(LegacyArtifactError, match="no program manifest entry"):
        pm.check_program("no_such_stem", False, digest)
    with pytest.raises(LegacyArtifactError, match="no longer reproduces"):
        pm.check_program(stem, wrapped, "0" * 64)
    unwrapped_only = next(s for s, w in entries if not w and (s, True) not in entries)
    with pytest.raises(LegacyArtifactError, match="no program manifest entry"):
        pm.check_program(unwrapped_only, True, digest)


@pytest.mark.parametrize("fam,stem", SAMPLE)
def test_translator_program_hash_equals_the_manifest_and_parse_module(mods, fam, stem):
    if not (BENCH / fam).exists():
        pytest.skip("benchmark root missing")
    from auto_LiRPA.parse_graph import parse_module
    from gear.io.legacy_translate import translate_module
    model, n = _build_real(fam, stem)
    model = model.eval()
    wrapped = type(model).__name__ == "_StructuralConstraintOutputWrapper"
    x = torch.zeros(1, n, dtype=f64)
    got = pm.program_hash(*translate_module(model, x))
    assert got == pm.program_hash(*parse_module(model, x))
    pm.check_program(stem, wrapped, got)


def test_program_hash_is_sensitive_to_tensors_and_names():
    from collections import OrderedDict
    from auto_LiRPA.parse_graph import Node
    def prog(value, name="/0"):
        node = Node(name=name, ori_name="", inputs=[], attr={"value": torch.tensor([value], dtype=torch.float64)},
                    op="onnx::Constant", param=OrderedDict(), output_index=0)
        return [node], [], ["/0"], None
    base = pm.program_hash(*prog(1.0))
    assert base == pm.program_hash(*prog(1.0))
    assert base != pm.program_hash(*prog(1.0000000000000002))
    assert base != pm.program_hash(*prog(-0.0 if False else 2.0))
    assert base != pm.program_hash(*prog(1.0, name="/1"))
    assert pm.program_hash(*prog(0.0)) != pm.program_hash(*prog(-0.0))
