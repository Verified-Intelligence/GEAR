"""Bit-exact golden records of the solver's core stages on small benchmark instances (CPU, fp64, one thread).

record(bench, name) runs, in this order, the structure report, CROWN bounds, domain clipping, the lifted LP, the
root ALM and the split decision of the production configuration (exp_configs/NLP_solver/gear.yaml) and returns
every result as tensors or strings. compare(old, new) names the keys that differ. Not wall-clock dependent: every
time budget is set to 1e9 so that a stage stops on its own iteration counts.
"""
import ast
import dataclasses
import hashlib
import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, CV)
sys.path.insert(0, os.path.dirname(CV))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import structure_instances as si  # noqa: E402

PROFILE = os.path.join(CV, 'exp_configs', 'NLP_solver', 'gear.yaml')
INSTANCES = os.path.join(HERE, 'parity_instances.tsv')
GOLDEN = os.path.join(HERE, 'parity_golden')
N_BOXES = 8
N_LIFTED_BOXES = 4
ALM_OUTER, ALM_RESTARTS = 3, 8
NO_BUDGET = 1e9


def read_list():
    with open(INSTANCES) as f:
        return [tuple(l.split('\t')[:2]) for l in f.read().splitlines() if l.strip()]


def golden_path(name):
    return os.path.join(GOLDEN, name + '.pt')


def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v


def _profile():
    """(bound_opts, nlp options) of the production profile; arguments.Config holds the profile's values too, as
    the runner leaves it"""
    import yaml
    import arguments
    from api import ConfigBuilder
    with open(PROFILE) as f:
        bound_opts = ast.literal_eval(yaml.safe_load(f)['model']['bound_opts'])
    cfg = ConfigBuilder.from_yaml(PROFILE).to_dict()
    arguments.Config.parse_config([], verbose=False)
    for section, values in cfg.items():
        if section != 'model':
            _merge(arguments.Config[section], values)
    arguments.Config['general']['device'] = 'cpu'
    return bound_opts, dict(arguments.Config['solver']['nlp'])


def _scalar(v):
    return torch.tensor(float(v), dtype=torch.float64)


def _jsonable(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=repr)
    return repr(o)


def _structure(rep):
    rows = [{f.name: getattr(r, f.name) for f in dataclasses.fields(r)} for r in rep.rows]
    return json.dumps(dict(n=rep.n, skipped=rep.skipped, spent=rep.spent, poles=sorted(rep.poles, key=repr),
                           rows=rows), sort_keys=True, default=_jsonable)


def _boxes(L, U):
    """the root box and N_BOXES sub-boxes: centre uniform in the root, half-width 0.5 (u - l) U^3, clipped"""
    n = L.numel()
    gen = torch.Generator().manual_seed(0)
    c = L + (U - L) * torch.rand(N_BOXES, n, generator=gen, dtype=torch.float64)
    w = 0.5 * (U - L) * torch.rand(N_BOXES, n, generator=gen, dtype=torch.float64) ** 3
    lo = torch.cat([L.reshape(1, n), torch.maximum(L, c - w)])
    hi = torch.cat([U.reshape(1, n), torch.minimum(U, c + w)])
    return lo, hi


def _crown(bm, C, lo, hi):
    from auto_LiRPA import BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    x = BoundedTensor((lo + hi) / 2, PerturbationLpNorm(x_L=lo, x_U=hi))
    out, inp = bm.output_name[0], bm.input_name[0]
    with torch.no_grad():
        ret = bm.compute_bounds(x=(x,), C=C.expand(lo.shape[0], -1, -1), method='CROWN', bound_upper=False,
                                return_A=True, needed_A_dict={out: {inp}})
    A = ret[-1][out][inp]
    return ret[0].detach(), A['lA'].detach().flatten(2), A['lbias'].detach()


def _lifted(rec, bm, model, C, rhs, lo, hi, nlp, pool):
    from input_split.lifted_bound import setup_lifted_lp
    history = SimpleNamespace(qtb=None, rowpool=pool)
    setup_lifted_lp(SimpleNamespace(net=bm, model_ori=model), history, lo[:1], hi[:1], C, rhs, None, nlp,
                    allow_obbt=False)
    lp = history.lifted
    rec['lifted/built'] = 'yes' if lp is not None else 'no'
    if lp is None:
        return
    rows = pool.thresholds(rhs.reshape(1, -1))[:, 1:]
    for k in range(N_LIFTED_BOXES + 1):
        data = lp.sparse_lp(lo[k].numpy(), hi[k].numpy(), rows[0].numpy())
        if k == 0:
            c, c0, G, h, lz, uz = data
            digest = hashlib.sha256()
            for a in (c, np.float64(c0), G.indptr, G.indices, G.data, h, lz, uz):
                digest.update(np.ascontiguousarray(a).tobytes())
            rec['lifted/digest'] = digest.hexdigest()
        bound, _ = lp.solve_node(*data)
        rec['lifted/root' if k == 0 else f'lifted/sub{k - 1}'] = _scalar(bound)


def _alm(rec, model, C, rhs, L, U, nlp):
    from input_split.nlp_in_bab import apply_root_alm_overrides
    from nlp_solver import ALMConfig, ALMPGDSolver, NNProblemSingleDomain
    cfg = ALMConfig.from_dict(dict(apply_root_alm_overrides(nlp), alm_outer_max_iter=ALM_OUTER,
                                   alm_opt_polish_time=NO_BUDGET, alm_batched_gn_max_time=NO_BUDGET,
                                   timeout=NO_BUDGET))
    cfg.verbose = False
    cfg.obj_patience = -1                        # no early stop at the root, as falsification._run_alm
    cfg.inner_plateau_check = 0
    problem = NNProblemSingleDomain(model=model, dm_l=L, dm_u=U, C_obj=C[0, :1], rhs_obj=torch.zeros(1, dtype=L.dtype),
                                    C_constr=C[0, 1:], rhs_constr=rhs[0, 1:], device=torch.device('cpu'))
    torch.manual_seed(0)
    res = ALMPGDSolver(cfg).solve_with_restarts(problem, num_restarts=ALM_RESTARTS, timeout=NO_BUDGET)
    x = res.x_solution
    rec['alm/x'] = torch.empty(0, dtype=torch.float64) if x is None else x.detach().flatten().double().clone()
    rec['alm/obj'] = torch.as_tensor(res.objective_value).detach().double().flatten().clone()
    rec['alm/viol'] = torch.as_tensor(res.constraint_violation).detach().double().flatten().clone()


def load_instance(bench, name):
    """(the instance as structure_instances.load builds it with the production profile, the profile's nlp options)"""
    bound_opts, nlp = _profile()
    # other test modules put their own `modules.py` on sys.path; the benchmark models must import their own
    for m in [m for m in sys.modules if m.split('.')[0] in ('modules', 'mod_files_gams', 'py_minlplib_linear')]:
        del sys.modules[m]
    return si.load(bench, name, bound_opts), nlp


def record(bench, name):
    from input_split.branching_heuristics import input_split_heuristic_sb
    from input_split.clip import clip_domains
    from nlp_solver.structure import build_structure
    torch.set_num_threads(1)
    torch.manual_seed(0)
    d, nlp = load_instance(bench, name)
    bm, model, x0, C, L, U, n = d['bm'], d['model'], d['x0'], d['C'], d['L'], d['U'], d['n']
    m = C.shape[1]
    rec = {}
    root_rhs = torch.zeros(1, m, dtype=torch.float64)
    bm(x0)
    report = build_structure(bm, x0, C, root_rhs, L.reshape(1, n), U.reshape(1, n))
    rec['structure'] = _structure(report)

    lo, hi = _boxes(L, U)
    lb, lA, lbias = _crown(bm, C, lo, hi)
    rec['crown_lb/0'], rec['crown_lA/0'] = lb[0], lA[0]
    for k in range(1, N_BOXES + 1):
        rec[f'crown_lb/{k}'], rec[f'crown_lA/{k}'] = lb[k], lA[k]

    # objective threshold: the median sub-box bound, so that about half of the sub-boxes can be clipped
    thr = torch.zeros(N_BOXES, m, dtype=torch.float64)
    thr[:, 0] = lb[1:, 0].nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).median()
    cl, cu = clip_domains(lo[1:].clone(), hi[1:].clone(), thr, lA[1:], lbias[1:], num_iters=2, rowpool=report.pool)
    for k in range(N_BOXES):
        rec[f'clip/{k}'] = torch.stack([cl[k].flatten(), cu[k].flatten()]).detach().clone()

    lifted_rhs = root_rhs.clone()
    lifted_rhs[0, 0] = float('inf')
    _lifted(rec, bm, model, C, lifted_rhs, lo, hi, nlp, report.pool)
    _alm(rec, model, C, root_rhs, L, U, nlp)

    history = SimpleNamespace(root_width=(U - L).reshape(-1))
    circles = sorted({r.circle for r in report.rows if r.circle})
    history.angle_pairs = torch.tensor(circles, dtype=torch.long) if circles else None
    pick = input_split_heuristic_sb(lo[1:], hi[1:], lb[1:], thr, lA[1:], history=history)
    for k in range(N_BOXES):
        rec[f'branch/{k}'] = pick[k].detach().clone()
    return rec


def _same(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return (a.shape == b.shape and a.dtype == b.dtype
                and bool(((a == b) | (a.isnan() & b.isnan())).all()))
    return type(a) is type(b) and a == b


def compare(old, new):
    """keys that are missing from either side, or whose values differ bit for bit"""
    return [k for k in sorted(set(old) | set(new)) if k not in old or k not in new or not _same(old[k], new[k])]
