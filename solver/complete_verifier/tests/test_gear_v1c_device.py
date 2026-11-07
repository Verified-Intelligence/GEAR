"""G5 v1c (CUDA kit run 2 follow-up, A294; v1 interface decision), CPU only: gr.compile refuses a torch source with
tensors on a non-CPU device with a ModelError that names the device and asks for a CPU module; nothing is copied or
moved. 'meta' stands in for CUDA (any non-CPU device takes the same check)."""
import pytest
import torch

import gear as gr
from tests import test_gear_wp12a_torch_direct as tdt


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.mark.parametrize("where", ["parameter", "buffer", "tensor attribute"])
def test_a_source_on_another_device_is_refused_by_name(where):
    if where == "tensor attribute":
        src = tdt.Lifted()
        src.k = src.k.to("meta")
        box, kw, key = tdt.BOX2, {}, "k"
    else:
        src = tdt.MLP()
        if where == "buffer":
            src.register_buffer("scale", torch.ones(2, device="meta"))
            key = "scale"
        else:
            src.l2.to("meta")
            key = "l2.weight"
        box, kw = tdt.BOX3, dict(le=[1])
    before = {k: v.device for k, v in list(src.named_parameters()) + list(src.named_buffers())}
    with pytest.raises(gr.ModelError, match=r"tensors on meta \(first: '" + key + r"'\).*CPU module.*model\.cpu\(\)"):
        gr.compile(src, box=box, objective=0, **kw)
    assert before == {k: v.device for k, v in list(src.named_parameters()) + list(src.named_buffers())}  # no copy, no move


def test_a_cpu_source_still_compiles():
    assert gr.compile(tdt.MLP(), box=tdt.BOX3, objective=0, le=[1]).path == "direct:torch"
