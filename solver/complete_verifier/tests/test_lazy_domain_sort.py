"""bab.lazy_sort_domains pops the same domains, in the same order, as a full sort."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arguments  # noqa: E402,F401
from input_split.branching_domains import UnsortedInputDomainList  # noqa: E402


def _domains(lazy, objective_copy=True):
    dl = UnsortedInputDomainList(storage_depth=1, output_device="cpu", use_split_idx=False,
                                 sort_index=0, sort_heuristic="lb_only")
    dl.lazy_sort = lazy
    if not objective_copy:      # the strided lb[:, 0] path, as before lb_obj existed
        dl._objective_key = lambda *a, **k: None
    return dl


def _add(dl, lb, tag):
    """Domains identified by `tag`: dm_l[:, 0] and the constraint offset carry it."""
    n = lb.shape[0]
    dm_l = torch.zeros(n, 3)
    dm_l[:, 0] = tag
    dm_u = dm_l + 1
    cs = torch.eye(2).unsqueeze(0).expand(n, 2, 2)
    thr = torch.full((n, 2), float("inf"))
    cA = torch.ones(n, 1, 3)
    cb = (tag - 1e6).unsqueeze(1)          # never redundant... kept verbatim
    dl.add(torch.stack([lb, lb], 1), dm_l, dm_u, cs, thr, constraints=(cA, cb),
           check_dm_lbs=False, check_input_boxes=False)


def _pop(dl, b):
    ret = dl.pick_out_batch(b)
    lb, dm_l, constraints = ret[1], ret[2], ret[6]
    assert torch.equal(dm_l[:, 0], constraints[1][:, 0] + 1e6)   # rows moved together
    return dm_l[:, 0].clone(), lb[:, 0].clone()


def test_lazy_sort_pops_like_full_sort():
    g = torch.Generator().manual_seed(0)
    full, lazy, ref = _domains(False), _domains(True), _domains(False, objective_copy=False)
    next_tag = [0.0]

    def add_both(n):
        lb = torch.rand(n, generator=g) * 10 - 5          # distinct keys: no ties
        tag = torch.arange(next_tag[0], next_tag[0] + n)
        next_tag[0] += n
        _add(full, lb, tag)
        _add(lazy, lb, tag)
        _add(ref, lb, tag)

    add_both(500)
    script = [("sort",), ("pop", 64), ("add", 30), ("pop", 64), ("pop", 128), ("add", 300),
              ("pop", 16), ("sort",), ("pop", 1), ("add", 5), ("pop", 200), ("pop", 50),
              ("sort",), ("sort",), ("pop", 400), ("add", 10), ("pop", 10000)]
    for op in script:
        if op[0] == "sort":
            full.sort()
            lazy.sort()
            ref.sort()
        elif op[0] == "add":
            add_both(op[1])
        else:
            if len(full) == 0:
                break
            tr, lr = _pop(ref, op[1])
            tf, lf = _pop(full, op[1])
            tl, ll = _pop(lazy, op[1])
            assert torch.equal(tf, tr) and torch.equal(lf, lr), op
            assert torch.equal(tf, tl), op
            assert torch.equal(lf, ll), op
        assert len(full) == len(lazy) == len(ref)
        if len(ref):
            # the worst-domain lookup of every iteration
            assert torch.equal(ref[ref.get_topk_indices()][0], lazy[lazy.get_topk_indices()][0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("lazy,budget", [(False, 1 << 30), (True, 1 << 30), (True, 20000), (False, 1)])
def test_gpu_storage_pops_like_host_storage(lazy, budget):
    # bab.domain_storage_gpu_mb: storages on the GPU (moving to the host past `budget`
    # bytes) must pop exactly what the host storage pops.
    g = torch.Generator().manual_seed(1)
    ref = _domains(False)
    gpu = UnsortedInputDomainList(storage_depth=1, output_device="cuda", use_split_idx=False,
                                  sort_index=0, sort_heuristic="lb_only")
    gpu.lazy_sort = lazy
    gpu._storage_device, gpu._gpu_budget = "cuda", budget
    next_tag = [0.0]

    def add_both(n):
        lb = torch.rand(n, generator=g) * 10 - 5
        tag = torch.arange(next_tag[0], next_tag[0] + n)
        next_tag[0] += n
        _add(ref, lb, tag)
        _add(gpu, lb, tag)

    add_both(400)
    for op in [("sort",), ("pop", 64), ("add", 300), ("pop", 128), ("sort",), ("pop", 500),
               ("add", 50), ("pop", 32), ("pop", 10000)]:
        if op[0] == "sort":
            ref.sort()
            gpu.sort()
        elif op[0] == "add":
            add_both(op[1])
        else:
            if len(ref) == 0:
                break
            tr, lr = _pop(ref, op[1])
            tg, lg = _pop(gpu, op[1])
            assert torch.equal(tr, tg.cpu()) and torch.equal(lr, lg.cpu()), op
        assert len(ref) == len(gpu)
    if budget < 1000:
        assert gpu._storage_device == "cpu"
