"""Benchmark-instance loading shared by the offline structure scripts (structure_parity.py, structure_audit.py).
Not a pytest module."""
import ast
import os
import re
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.dirname(HERE)
sys.path.insert(0, CV)
sys.path.insert(0, os.path.dirname(CV))
import torch  # noqa: E402
import yaml  # noqa: E402
from arguments import ConfigHandler  # noqa: E402

torch.set_num_threads(1)
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "benchmarks")
LAYOUT = {'gams_world': ('gams_world/mod_files_gams', 'gams_world/vnnlib_gams'),
          'minlplib_linear': ('minlplib_linear/py_minlplib_linear', 'minlplib_linear/vnnlib_minlplib_linear')}
CONFIG = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "exp_configs" / "NLP_solver" / "f3.yaml")


def add_arguments(ap):
    """--list, --config, --benchmark-root, --limit and --only, shared by the scripts"""
    ap.add_argument('--list', required=True, help='tsv of benchmark<TAB>instance')
    ap.add_argument('--config', default=CONFIG, help='yaml whose model.bound_opts the instances are built with')
    ap.add_argument('--benchmark-root', default=ROOT)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--only', help='comma-separated instance names')


def read_instances(a):
    """(bound_opts, [(benchmark, instance)]) selected by the arguments of add_arguments"""
    with open(a.config) as f:
        bound_opts = ast.literal_eval(yaml.safe_load(f)['model']['bound_opts'])
    with open(a.list) as f:
        inst = [tuple(l.split('\t')[:2]) for l in f.read().splitlines() if l.strip()]
    if a.only:
        inst = [i for i in inst if i[1] in set(a.only.split(','))]
    return bound_opts, inst[:a.limit] if a.limit else inst


class Wrapper(torch.nn.Module):
    """The solver's wrapper: objective, the linear constraint rows, then the remaining outputs."""

    def __init__(self, base, A, b):
        super().__init__()
        self.base_model = base
        self.register_buffer('constraint_A', A.clone())
        self.register_buffer('constraint_b', b.clone())

    def forward(self, x):
        o = self.base_model(x)
        s = torch.nn.functional.linear(x.reshape(x.shape[0], -1), self.constraint_A, -self.constraint_b)
        return torch.cat([o[:, :1], s, o[:, 1:]], dim=1)


def load(bench, name, bound_opts, root=ROOT):
    import nlp_example as ne
    from auto_LiRPA import BoundedModule
    mdir_rel, vdir_rel = LAYOUT[bench]
    mdir, vdir = f'{root}/{mdir_rel}', f'{root}/{vdir_rel}'
    pt = f'{mdir}/{name.replace("-", "_")}_model_state.pt'
    mod = ne._load_models_module(mdir)
    model = getattr(mod, ne._stem_to_class_name(name))(pt_path=pt).double().eval()
    vp = f'{vdir}/{name}.vnnlib'
    if not os.path.exists(vp):
        vp = f'{vdir}/{name.replace("_", "-")}.vnnlib'
    lo, hi = {}, {}
    with open(vp) as f:
        lines = f.read().splitlines()
    for line in lines:
        m = re.match(r'\(assert \((<=|>=) X_(\d+) ([-+0-9.eE]+)\)\)', line.strip())
        if m:
            (hi if m.group(1) == '<=' else lo)[int(m.group(2))] = float(m.group(3))
    n = max(max(lo), max(hi)) + 1
    L = torch.tensor([lo[i] for i in range(n)], dtype=torch.float64)
    U = torch.tensor([hi[i] for i in range(n)], dtype=torch.float64)
    sd = torch.load(pt, map_location='cpu', weights_only=False)
    lc = sd.get('linear_constr') if isinstance(sd, dict) else None
    if lc is not None and lc.get('A') is not None:
        model = Wrapper(model, lc['A'].double().reshape(-1, n), lc['b'].double().reshape(-1)).double().eval()
    x0 = ((L + U) / 2).reshape(1, n)
    with torch.no_grad():
        m_out = model(x0).numel()
    C = torch.eye(m_out, dtype=torch.float64).unsqueeze(0)      # every output is a spec row
    rhs = torch.zeros(1, m_out, dtype=torch.float64)
    bm = BoundedModule(model, torch.zeros(1, n, dtype=torch.float64),
                       bound_opts=dict(bound_opts, op_budget=dict(ConfigHandler.OP_BUDGET_DEFAULTS)), device='cpu')
    bm.eval()
    return dict(model=model, bm=bm, L=L, U=U, x0=x0, C=C, rhs=rhs, n=n)
