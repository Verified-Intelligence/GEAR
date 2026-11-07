"""G5 WP6: the legacy node translator must reproduce ``parse_module`` exactly (names included).

Per-class tuples on synthetic modules of the 21 classes of the two ``modules.py`` files plus the two ``api.py``
wrappers, a sample of real instances, and the opt-in seam in ``BoundedModule._convert_nodes``.
The generated artifacts live outside the repository; the tests skip when they are absent.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BENCH = Path(os.environ.get('GEAR_BENCH', str(__import__("pathlib").Path(__file__).resolve().parents[3] / "benchmarks")))
GAMS = BENCH / 'gams_world' / 'mod_files_gams' / 'modules.py'
LINEAR = BENCH / 'minlplib_linear' / 'py_minlplib_linear' / 'modules.py'
pytestmark = pytest.mark.skipif(not (GAMS.exists() and LINEAR.exists()), reason='legacy artifacts not available')

f64 = torch.float64


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def mods():
    old = torch.get_default_dtype()
    torch.set_default_dtype(f64)
    G = _load(GAMS, 'wp6_gams_modules')
    L = _load(LINEAR, 'wp6_linear_modules')
    yield G, L
    torch.set_default_dtype(old)


def _canon_val(v):
    if isinstance(v, torch.Tensor):
        return ('T', str(v.dtype), tuple(v.shape), v.flatten().tolist())
    if isinstance(v, (list, tuple)):
        return (type(v).__name__, [_canon_val(i) for i in v])
    if isinstance(v, dict):
        return ('d', sorted((k, _canon_val(x)) for k, x in v.items()))
    return (type(v).__name__, v)


def _canon(prog):
    op, inn, out, tmpl = prog
    c_in = [(n.name, n.ori_name, n.inputs, n.attr, n.op, n.model_input_index, n.output_index, n.perturbation,
             None if n.param is None else (n.param.dtype, tuple(n.param.shape), n.param.numpy().tobytes()))
            for n in inn]
    c_op = [(n.name, n.op, list(n.inputs), _canon_val(n.attr), n.output_index, n.ori_name, n.model_input_index,
             [(k, None if v is None else (tuple(v.shape), v.dtype)) for k, v in n.param.items()]) for n in op]
    return c_in, c_op, list(out), tmpl


def assert_same(model, n):
    from auto_LiRPA.parse_graph import parse_module
    from gear.io.legacy_translate import translate_module
    model = model.eval()
    x = torch.zeros(1, n, dtype=f64)
    ref = _canon(parse_module(model, x))
    got = _canon(translate_module(model, x))
    assert got[2:] == ref[2:]
    assert got[0] == ref[0]
    assert len(got[1]) == len(ref[1])
    for i, (a, b) in enumerate(zip(got[1], ref[1])):
        assert a == b, f'node {i}: {a} != {b}'


def _simple_cases(G):
    gat = lambda idx: G.VariableVectorGatherModule(idx)
    cv = lambda v: G.ConstantVectorModule(v, dtype=f64)
    sc = lambda v: G.ScalarConstantModule(v, dtype=f64)
    bop = lambda a, b, o: G.VectorBinaryOpModule(a, b, o)
    red = lambda m: G.VectorReduceModule(m, 'sum')
    lin = lambda: G.SharedLinearFormModule([0, 1], [[1., 2.], [3., 4.]], [0., 0.], dtype=f64)
    out = {
        'gather': lambda: gat([0, 2, 1]),
        'const_vec': lambda: bop(cv([1., 2., 3.]), gat([0, 2, 1]), 'mul'),
        'scalar_const': lambda: bop(sc(2.), gat([0, 2, 1]), 'mul'),
        'bin_add_const_left': lambda: bop(cv([1., 2.]), gat([0, 1]), 'add'),
        'slice': lambda: G.LastDimSliceModule(1, 3),
        'scale': lambda: G.VectorScaleModule(gat([0, 1]), 2.5, dtype=f64),
        'vsum3': lambda: G.VectorSumModule([red(gat([0, 1])), red(gat([1, 2])), red(gat([0, 2]))]),
        'shared_lin': lin,
        'shared_lin_bias': lambda: G.SharedLinearFormModule([0, 1], [[1., 2.], [3., 4.]], [0.5, 0.], dtype=f64),
        'shared_lin_straight': lambda: G.SharedLinearFormModule(None, [[1., 2., 0.], [3., 4., 1.]], [0.5, 0.],
                                                                 dtype=f64),
        'batched_affine_w1': lambda: G.BatchedAffineGatherModule([[0], [1]], [[2.], [3.]], [0., 0.], dtype=f64),
        'batched_affine_w2': lambda: G.BatchedAffineGatherModule([[0, 1], [1, 2]], [[2., 1.], [3., 1.]], [0.5, 0.],
                                                                  dtype=f64),
        'single_core': lambda: G.SharedLinearSingleCoreModule(lin(), red(G.LastDimSliceModule(0, 2))),
        'augment': lambda: G.SharedLinearAugmentModule(lin(), G.LastDimSliceModule(1, 4)),
        'min_module': lambda: G.MinModule([red(gat([0, 1])), red(gat([1, 2]))]),
        'concat_modules': lambda: G.ConcatModules([red(gat([0, 1])), gat([0, 2])], 3),
        # shared children: the same module object appears twice, so the graph merges the common subexpression
        'shared_dag': lambda: (lambda g: bop(g, bop(g, cv([1., 2.]), 'mul'), 'add'))(gat([0, 1])),
    }
    for o in ('add', 'sub', 'mul', 'div', 'pow'):
        out['bin_' + o] = (lambda o=o: bop(gat([0, 1]), cv([1., 2.]) if o != 'pow' else sc(2.), o))
    for o in ('log', 'sin', 'cos', 'tan', 'exp', 'sqrt', 'tanh', 'abs', 'neg'):
        out['un_' + o] = (lambda o=o: G.VectorUnaryOpModule(gat([0, 1]), o))
    for o in ('sum', 'min', 'max', 'mean'):
        out['reduce_' + o] = (lambda o=o: G.VectorReduceModule(gat([0, 1]), o))
    return out


def _composite_cases(G, L):
    gat = lambda idx: G.VariableVectorGatherModule(idx)
    cv = lambda v: G.ConstantVectorModule(v, dtype=f64)
    sc = lambda v: G.ScalarConstantModule(v, dtype=f64)
    bop = lambda a, b, o: G.VectorBinaryOpModule(a, b, o)
    lin2 = lambda: G.SharedLinearFormModule([0, 1], [[1., 2.], [3., 4.]], [0., 0.], dtype=f64)
    ast = lambda mods, layout, total: G.CombinedASTModule(mods, layout, total)
    bucket = lambda b, coef: G.ResidualBucketScatterAddModule(b, coef, dtype=f64)
    exp3 = lambda: G.VectorUnaryOpModule(gat([0, 1, 0]), 'exp')
    cross = lambda lin, res, idx, total: G.CrossFormulaLinearResidualVectorModule(lin, res, idx, total, dtype=f64)
    two = lambda: ast([gat([0, 1])], [('vector_pick', 0, 0, 0), ('vector_pick', 0, 1, 1)], 2)
    out = {
        'ast_pick': lambda: ast([gat([0, 1, 2])], [('vector_pick', 0, 0, 0), ('vector_pick', 0, 2, 1)], 2),
        'ast_scalar0d': lambda: ast([sc(2.0)], [('scalar_at', 0, 0)], 1),
        'ast_pad': lambda: ast([G.VectorReduceModule(gat([0, 1]), 'sum')], [('vector_pick', 0, 2, 0)], 1),
        'ast_slice': lambda: ast([gat([0, 1, 2])], [('scalar_at', 0, 0)], 1),
        'ast_missing': lambda: ast([gat([0, 1])], [('vector_pick', 0, 0, 0), ('vector_pick', 0, 1, 2)], 3),
        'ast_two_modules': lambda: ast([gat([0, 1]), G.VectorReduceModule(gat([1, 2]), 'sum')],
                                       [('scalar_at', 1, 0), ('vector_pick', 0, 1, 1)], 2),
        'bucket_vector': lambda: bucket(exp3(), [1., 1., 2.]),
        'bucket_ones': lambda: bucket(exp3(), [1., 1., 1.]),
        'bucket_scalar': lambda: bucket(exp3(), [2.5, 2.5, 2.5]),
        'bucket_const_vec': lambda: bucket(cv([1., 2., 3.]), [1., 2., 3.]),
        'bucket_scalar0d': lambda: bucket(sc(2.0), [1., 2., 3.]),
        'bucket_width1': lambda: bucket(G.VectorReduceModule(gat([0, 1]), 'sum'), [1., 2., 3.]),
        'cross_scatter': lambda: cross(lin2(), [bucket(exp3(), [1., 1., 2.])], [[0, 1, 1]], 2),
        'cross_noscatter': lambda: cross(lin2(), [two()], None, 2),
        'cross_noscatter_nolin': lambda: cross(None, [two(), two()], None, 2),
        'coef_scale': lambda: L.ResidualCoefScaleModule(bop(sc(2.), gat([0, 1, 0]), 'mul'), [1., 2., 3.], dtype=f64),
        'coef_scale_const_vec': lambda: L.ResidualCoefScaleModule(cv([1., 2., 3.]), [1., 2., 3.], dtype=f64),
        'coef_scale_scalar0d': lambda: L.ResidualCoefScaleModule(sc(2.), [1., 2., 3.], dtype=f64),
        'coef_scale_width1': lambda: L.ResidualCoefScaleModule(G.VectorReduceModule(gat([0, 1]), 'sum'),
                                                               [1., 2., 3.], dtype=f64),
        'lift_compact': lambda: L.LiftCompactToFullModule(
            L.ResidualCoefScaleModule(bop(sc(2.), gat([0, 1, 0]), 'mul'), [1., 2., 3.], dtype=f64),
            [0, 1, 0], 2, dtype=f64),
        'general_residual': lambda: L.GeneralResidualModule(
            [L.LiftCompactToFullModule(
                L.ResidualCoefScaleModule(bop(sc(2.), gat([0, 1, 0]), 'mul'), [1., 2., 3.], dtype=f64),
                [0, 1, 0], 2, dtype=f64), lin2()], [[], []], 2, dtype=f64),
        'general_residual_compact': lambda: L.GeneralResidualModule(
            [L.ResidualCoefScaleModule(bop(sc(2.), gat([0, 1, 0]), 'mul'), [1., 2., 3.], dtype=f64),
             L.ResidualCoefScaleModule(cv([1., 2.]), [1., 2.], dtype=f64)], [[0, 1, 0], [1, 0]], 2, dtype=f64),
    }
    return out


class _DemoModel(nn.Module):
    """Same shape as the generated ``<Name>Model`` classes."""

    def __init__(self, root):
        super().__init__()
        self._root = root

    def forward(self, x):
        return self._root(x)


_DemoModel.__name__ = 'DemoModel'


def _all_cases(mods):
    G, L = mods
    d = {k: (f, 3) for k, f in _simple_cases(G).items()}
    d.update({k: (f, 3) for k, f in _composite_cases(G, L).items()})
    return d


def test_cases_cover_all_classes(mods):
    G, L = mods
    names = set()
    for fac, _ in _all_cases(mods).values():
        for m in fac().modules():
            names.add(type(m).__name__)
    classes = {n for n, c in vars(G).items() if isinstance(c, type) and issubclass(c, nn.Module) and c.__module__ == G.__name__}
    classes |= {n for n in ('ResidualCoefScaleModule', 'LiftCompactToFullModule', 'GeneralResidualModule')}
    classes -= {'Module'}
    assert len(classes) == 21
    assert classes <= names, classes - names


def _case_ids():
    return []


@pytest.mark.parametrize('name', [
    'gather', 'const_vec', 'scalar_const', 'bin_add_const_left', 'slice', 'scale', 'vsum3', 'shared_lin',
    'shared_lin_bias', 'shared_lin_straight', 'batched_affine_w1', 'batched_affine_w2', 'single_core', 'augment',
    'min_module', 'concat_modules', 'shared_dag',
    'bin_add', 'bin_sub', 'bin_mul', 'bin_div', 'bin_pow',
    'un_log', 'un_sin', 'un_cos', 'un_tan', 'un_exp', 'un_sqrt', 'un_tanh', 'un_abs', 'un_neg',
    'reduce_sum', 'reduce_min', 'reduce_max', 'reduce_mean',
    'ast_pick', 'ast_scalar0d', 'ast_pad', 'ast_slice', 'ast_missing', 'ast_two_modules',
    'bucket_vector', 'bucket_ones', 'bucket_scalar', 'bucket_const_vec', 'bucket_scalar0d', 'bucket_width1',
    'cross_scatter', 'cross_noscatter', 'cross_noscatter_nolin',
    'coef_scale', 'coef_scale_const_vec', 'coef_scale_scalar0d', 'coef_scale_width1',
    'lift_compact', 'general_residual', 'general_residual_compact',
])
def test_class_tuples_equal_parse_module(mods, name):
    fac, n = _all_cases(mods)[name]
    assert_same(fac(), n)


def _two_naming_modules(G, L):
    """Several named values: helper-function names (``v``) and ``torch.unique`` names (``input``)."""
    gat = lambda idx: G.VariableVectorGatherModule(idx)
    sc = lambda v: G.ScalarConstantModule(v, dtype=f64)
    cs = lambda: L.ResidualCoefScaleModule(G.VectorBinaryOpModule(sc(2.), gat([0, 1, 0]), 'mul'), [1., 2., 3.],
                                           dtype=f64)
    lift = lambda: L.LiftCompactToFullModule(cs(), [0, 1, 0], 2, dtype=f64)
    return L.GeneralResidualModule([lift(), cs(), lift()], [[], [0, 1, 0], []], 2, dtype=f64)


def test_names_with_collisions(mods):
    G, L = mods
    assert_same(_two_naming_modules(G, L), 3)
    two = lambda: G.ResidualBucketScatterAddModule(G.VectorUnaryOpModule(G.VariableVectorGatherModule([0, 1, 0]),
                                                                       'exp'), [1., 3., 2.], dtype=f64)
    root = G.VectorSumModule([G.VectorReduceModule(two(), 'sum'), G.VectorReduceModule(two(), 'sum'),
                              G.VectorReduceModule(two(), 'sum')])
    assert_same(root, 3)


def test_wrappers(mods):
    from api import _ObjectiveOutputWrapper, _StructuralConstraintOutputWrapper
    G, L = mods
    lin = G.SharedLinearFormModule([0, 1], [[1., 2.], [3., 4.]], [0., 0.], dtype=f64)
    base = _DemoModel(lin)
    A = torch.tensor([[1., 0., -1.], [0., 2., 1.]], dtype=f64)
    b = torch.tensor([0.5, 1.5], dtype=f64)
    assert_same(_StructuralConstraintOutputWrapper(base, A, b), 3)
    assert_same(_ObjectiveOutputWrapper(base, torch.tensor([1., -2.], dtype=f64)), 3)
    assert_same(_ObjectiveOutputWrapper(_StructuralConstraintOutputWrapper(base, A, b), torch.ones(4, dtype=f64)), 3)
    assert_same(base, 3)


def test_out_of_vocabulary_raises(mods):
    from gear.io.legacy_translate import LegacyTranslateError, translate_module
    with pytest.raises(LegacyTranslateError):
        translate_module(nn.ReLU(), torch.zeros(1, 3, dtype=f64))


SAMPLE = [  # smallest instances covering the 19 classes the 505 use, plus bigger ones with renamed values
    ('gams_world', 'ex005'), ('gams_world', 'elena_schwefel'), ('gams_world', 'cute_bqp1var'),
    ('gams_world', 'hs034'), ('gams_world', 'hs030'), ('minlplib_linear', 'prob06'),
    ('gams_world', 'cute_explin'), ('gams_world', 'schitt_s229'), ('minlplib_linear', 'st_e22'),
    ('gams_world', 'hs005'), ('minlplib_linear', 'alkyl'), ('minlplib_linear', 'st_e08'),
    ('gams_world', 'cute_himmelp3'), ('gams_world', 'LM2n10'), ('gams_world', 'fekete2'),
    ('minlplib_linear', 'ex3_1_2'), ('gams_world', 'cute_hs101'),
]


def _build_real(fam, stem):
    from importlib import import_module
    d = {'gams_world': ('gams_world/mod_files_gams', 'gams_world/vnnlib_gams'),
         'minlplib_linear': ('minlplib_linear/py_minlplib_linear', 'minlplib_linear/vnnlib_minlplib_linear')}[fam]
    md = BENCH / d[0]
    for p in (md.parent.parent / 'minlplib' / 'mod_files', md.parent, md):
        if str(p) in sys.path:
            sys.path.remove(str(p))
        sys.path.insert(0, str(p))
    for k in [k for k in sys.modules if k == 'modules' or k == md.name or k.startswith(md.name + '.')]:
        del sys.modules[k]
    pkg = import_module(md.name)
    cls = ''.join(p.capitalize() for p in stem.replace('-', '_').split('_') if p) + 'Model'
    pt = md / f'{stem}_model_state.pt'
    model = getattr(pkg, cls)(pt_path=str(pt))
    import re
    txt = next((BENCH / d[1] / f'{c}.vnnlib' for c in (stem, stem.replace('_', '-')) if (BENCH / d[1] / f'{c}.vnnlib').exists())).read_text()
    n = len(set(re.findall(r'\(declare-const (X_\d+) Real\)', txt)))
    st = torch.load(pt, map_location='cpu', weights_only=False)
    lc = st.get('linear_constr') if isinstance(st, dict) else None
    if lc is not None and lc.get('A') is not None:
        from api import _StructuralConstraintOutputWrapper
        model = _StructuralConstraintOutputWrapper(model, lc['A'].to(f64), lc['b'].to(f64)).to(f64)
    return model, n


@pytest.mark.parametrize('fam,stem', SAMPLE)
def test_real_instances_equal_parse_module(mods, fam, stem):
    if not (BENCH / fam).exists():
        pytest.skip('benchmark root missing')
    model, n = _build_real(fam, stem)
    assert_same(model, n)


def _boundedmodule_nodes(model, n, hook):
    from auto_LiRPA import BoundedModule
    if hook:
        from gear.io.legacy_translate import use_translator
        use_translator(model)
    bm = BoundedModule(model, torch.zeros(1, n, dtype=f64), device='cpu')
    return bm


def test_hook_off_is_identical(mods, monkeypatch):
    """Without ``__gear_program__`` the seam is the old code path: parse_module is called and nothing else."""
    import auto_LiRPA.bound_general as bg
    G, L = mods
    fac, n = _all_cases(mods)['cross_scatter']
    calls = []
    orig = bg.parse_module
    monkeypatch.setattr(bg, 'parse_module', lambda *a, **k: calls.append(1) or orig(*a, **k))
    model = fac()
    assert not hasattr(model, '__gear_program__')
    bm = _boundedmodule_nodes(model, n, hook=False)
    assert len(calls) == 1
    x = torch.tensor([[0.3, -0.7, 1.1]], dtype=f64)
    assert torch.equal(bm(x), model(x))


@pytest.mark.parametrize('name', ['cross_scatter', 'general_residual', 'ast_missing', 'batched_affine_w2'])
def test_hook_on_builds_the_same_bound_graph(mods, monkeypatch, name):
    """With the hook on, parse_module is never called, the BoundedModule graph is node-for-node the old one and
    forward is bitwise equal."""
    import auto_LiRPA.bound_general as bg
    G, L = mods
    fac, n = _all_cases(mods)[name]
    ref_model = fac()
    ref = _boundedmodule_nodes(ref_model, n, hook=False)

    def boom(*a, **k):
        raise AssertionError('parse_module called with the hook on')
    monkeypatch.setattr(bg, 'parse_module', boom)
    model = fac()
    model.load_state_dict(ref_model.state_dict())
    new = _boundedmodule_nodes(model, n, hook=True)
    assert [(k, type(v).__name__, [i.name for i in v.inputs]) for k, v in ref._modules.items()] == \
           [(k, type(v).__name__, [i.name for i in v.inputs]) for k, v in new._modules.items()]
    x = torch.tensor([[0.3, -0.7, 1.1]], dtype=f64)
    assert torch.equal(new(x), ref(x))
    assert torch.equal(new(x), model(x))
