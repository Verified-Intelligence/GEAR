"""Feasible witnesses survive tightening even when some bound proposals fail."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import auto_LiRPA.concretize_func as cf


@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('no_return_inf', [False, True])
@pytest.mark.parametrize('bad_value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('backend', ['cpu','cuda','fused'])
def test_random_boxes_keep_feasible_points_with_failed_rows(monkeypatch, sign, no_return_inf, bad_value, backend):
    if backend!='cpu' and not torch.cuda.is_available():
        pytest.skip('needs CUDA')
    monkeypatch.setattr(cf,'FUSED_SWEEP',backend=='fused')
    g = torch.Generator().manual_seed(75)
    B, X, K = 12, 5, 8
    lo = -torch.rand(B, X, dtype=torch.float64, generator=g)
    hi = torch.rand(B, X, dtype=torch.float64, generator=g)
    x = lo + (hi-lo)*torch.rand(B, X, dtype=torch.float64, generator=g)
    A = torch.randn(B, K, X, dtype=torch.float64, generator=g)
    b = -(A*x[:, None]).sum(-1)-torch.rand(B, K, dtype=torch.float64, generator=g)*.1
    # Row 0 can be the objective cutoff; the others can come from any operator.
    A[::2, 0, 0] = bad_value
    b[1::2, 3] = bad_value
    obj = torch.cat([torch.eye(X, dtype=torch.float64),
                     torch.randn(3, X, dtype=torch.float64, generator=g)]).unsqueeze(0).expand(B, -1, -1)
    if backend!='cpu':
        lo,hi,x,A,b,obj=(v.cuda() for v in (lo,hi,x,A,b,obj))
    processed, masks = cf.sort_out_constr_batches(lo, hi, (A,b), no_return_inf=no_return_inf)
    result = cf.constraints_solving(lo, hi, obj, processed, sign, sorted_out_batches=masks,
                                    no_return_inf=no_return_inf, use_grad=False)
    if no_return_inf:
        result, infeasible = result
        assert infeasible is None or not infeasible.any()
    values = (obj*x[:, None]).sum(-1)
    assert torch.isfinite(result).all()
    assert (sign*(result.squeeze(-1)-values) >= -1e-10).all()
    # Identity objectives are exactly the coordinate-tightening path.
    tightened = result[:, :X, 0]
    assert (sign*(tightened-x) >= -1e-10).all()


@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('no_return_inf', [False, True])
@pytest.mark.parametrize('bad_value', [float('nan'), float('inf'), -float('inf')])
def test_failed_sweep_is_a_box_fallback_not_an_infeasibility_proof(monkeypatch, sign, no_return_inf, bad_value):
    lo, hi = torch.zeros(2,2,dtype=torch.float64), torch.ones(2,2,dtype=torch.float64)
    objective = torch.eye(2,dtype=torch.float64).unsqueeze(0).expand(2,-1,-1)
    constraints = (torch.tensor([[[1.,1.]],[[1.,-1.]]],dtype=torch.float64),
                   torch.tensor([[-.75],[-.25]],dtype=torch.float64))
    processed,masks=cf.sort_out_constr_batches(lo,hi,constraints,no_return_inf=no_return_inf)
    monkeypatch.setattr(cf,'FUSED_SWEEP',False)
    monkeypatch.setattr(cf,'_solve_dual_var',lambda a,c,d,e:torch.full(c.shape[:2],bad_value,dtype=c.dtype,device=c.device))
    result=cf.constraints_solving(lo,hi,objective,processed,sign,sorted_out_batches=masks,
                                  no_return_inf=no_return_inf,use_grad=False)
    if no_return_inf:
        result,infeasible=result
        assert infeasible is None or not infeasible.any()
    assert torch.isfinite(result).all()
    assert (result.squeeze(-1)<=lo).all() if sign<0 else (result.squeeze(-1)>=hi).all()


def test_preprocessed_infeasibility_remains_a_proof():
    lo,hi=torch.zeros(2,1,dtype=torch.float64),torch.ones(2,1,dtype=torch.float64)
    # Batch 0 is impossible, batch 1 contains x=.25.
    constraints=(torch.ones(2,1,1,dtype=torch.float64),torch.tensor([[1.],[-.5]],dtype=torch.float64))
    processed,masks=cf.sort_out_constr_batches(lo,hi,constraints,no_return_inf=True)
    _,infeasible=cf.constraints_solving(lo,hi,torch.ones(2,1,1,dtype=torch.float64),processed,
                                       sorted_out_batches=masks,no_return_inf=True,use_grad=False)
    assert infeasible.tolist()==[True,False]


@pytest.mark.parametrize('sign',[-1.,1.])
def test_nonfinite_objectives_are_vacuous_without_claiming_infeasibility(sign):
    lo,hi=torch.zeros(2,2,dtype=torch.float64),torch.ones(2,2,dtype=torch.float64)
    objective=torch.tensor([[[float('nan'),1.]],[[float('inf'),0.]]],dtype=torch.float64)
    for constraints in (None,(torch.ones(2,1,2,dtype=torch.float64),-torch.ones(2,1,dtype=torch.float64))):
        result,bad=cf.constraints_solving(lo,hi,objective,constraints,sign,no_return_inf=True,use_grad=False)
        assert (result==sign*torch.inf).all()
        assert bad is None or not bad.any()


@pytest.mark.parametrize('sign',[-1.,1.])
def test_direct_centred_rows_and_selected_objectives_preserve_witnesses(sign):
    lo,hi=torch.zeros(2,2,dtype=torch.float64),torch.ones(2,2,dtype=torch.float64)
    A=torch.tensor([[[1.,1.],[float('nan'),0.]],[[1.,1.],[0.,1.]]],dtype=torch.float64)
    d=torch.tensor([[0.,0.],[0.,float('inf')]],dtype=torch.float64)
    objective=torch.eye(2,dtype=torch.float64).unsqueeze(0).expand(2,-1,-1)
    result,bad=cf.constraints_solving(lo,hi,objective,(A,d),sign,no_return_inf=True,use_grad=False,
                                     objective_indices=torch.tensor([[0],[1]]))
    assert result.shape==(2,2,1)
    assert (sign*(result.squeeze(-1)-.25)>=-1e-10).all()
    assert bad is None or not bad.any()
