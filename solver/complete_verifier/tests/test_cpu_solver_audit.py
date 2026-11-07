"""CPU solver audit behavior, with isolated hooks and a real-library smoke test."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


KINDS = ('linprog', 'highs', 'ipopt', 'lanes', 'native')
VERIFIER = Path(__file__).resolve().parents[1]


@pytest.fixture
def hooks(monkeypatch):
    # Fresh state and a private atexit registry keep process-wide hooks out of
    # other tests. Backends record forwarding without solving or launching jobs.
    spec = importlib.util.spec_from_file_location('_test_cpu_audit', VERIFIER / 'cpu_solver_audit.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    backend_calls, exit_hooks = [], []
    result = object()

    def backend(kind):
        def solve(*args, **kwargs):
            backend_calls.append((kind, args, kwargs))
            return result
        return solve

    highs = type('Highs', (), {'run': backend('highs')})
    ipopt = type('IPOPTSolver', (), {'_solve_single_cyipopt': backend('ipopt')})
    native = type('NativeExecutor', (), {'submit': backend('native')})
    modules = {
        'scipy.optimize': SimpleNamespace(linprog=backend('linprog')),
        'highspy': SimpleNamespace(Highs=highs),
        'nlp_solver.ipopt_solver': SimpleNamespace(IPOPTSolver=ipopt),
        'nlp_solver.lp_alternation': SimpleNamespace(start=backend('lanes')),
        'input_split.native_lane': SimpleNamespace(NativeExecutor=native),
    }
    targets = {
        'linprog': (modules['scipy.optimize'], 'linprog'),
        'highs': (highs, 'run'),
        'ipopt': (ipopt, '_solve_single_cyipopt'),
        'lanes': (modules['nlp_solver.lp_alternation'], 'start'),
        'native': (native, 'submit'),
    }
    instances = dict(highs=highs(), ipopt=ipopt(), native=native())

    def invoke(kind, *args, **kwargs):
        owner, name = targets[kind]
        return getattr(instances.get(kind, owner), name)(*args, **kwargs)

    monkeypatch.setattr(audit, 'import_module', modules.__getitem__)
    monkeypatch.setattr(audit, 'atexit', SimpleNamespace(register=exit_hooks.append))
    return SimpleNamespace(audit=audit, modules=modules, targets=targets,
                           calls=backend_calls, exits=exit_hooks, result=result,
                           invoke=invoke, instances=instances)


@pytest.mark.parametrize('kind', KINDS)
def test_entry_counts_and_preserves_arguments_and_result(hooks, kind):
    hooks.audit.install()
    assert hooks.invoke(kind, 'payload', timeout=3) is hooks.result
    assert hooks.audit.COUNTS == {key: int(key == kind) for key in KINDS}
    args = (hooks.instances[kind], 'payload') if kind in hooks.instances else ('payload',)
    assert hooks.calls == [(kind, args, {'timeout': 3})]


@pytest.mark.parametrize('kind', KINDS)
def test_strict_blocks_first_call_in_worker_thread(hooks, kind):
    hooks.audit.install(strict=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(hooks.invoke, kind)
        with pytest.raises(RuntimeError, match=f'strict mode blocked {kind}'):
            future.result(timeout=5)
    assert hooks.calls == []
    assert hooks.audit.COUNTS == {key: int(key == kind) for key in KINDS}


def test_counts_are_shared_across_worker_threads(hooks):
    hooks.audit.install()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(hooks.invoke, KINDS * 200))
    assert all(result is hooks.result for result in results)
    assert hooks.audit.COUNTS == dict.fromkeys(KINDS, 200)
    assert len(hooks.calls) == 1000


def test_install_is_idempotent_and_strict_cannot_be_downgraded(hooks, capsys):
    hooks.audit.install()
    wrappers = {kind: getattr(owner, name) for kind, (owner, name) in hooks.targets.items()}
    hooks.invoke('linprog')
    hooks.audit.install()
    hooks.audit.install(strict=True)
    hooks.audit.install(strict=False)
    assert all(getattr(owner, name) is wrappers[kind]
               for kind, (owner, name) in hooks.targets.items())
    assert len(hooks.exits) == 1
    for _ in range(2):
        with pytest.raises(RuntimeError, match='linprog'):
            hooks.invoke('linprog')
    assert hooks.audit.COUNTS['linprog'] == 3
    assert len(hooks.calls) == 1
    assert capsys.readouterr().out.count('[cpu-solver-audit] enabled') == 1


def test_preimported_module_aliases_are_patched(monkeypatch, hooks):
    modules = []
    # Include every GEAR linprog consumer, plus renamed imports and aliases of
    # the other entry points. Imports after install see the source attributes.
    for name in ('input_split.lifted_bound', 'input_split.nlp_in_bab',
                 'nlp_solver.dual_ascent', 'nlp_solver.exp_certificate',
                 'nlp_solver.gp_certificate', 'nlp_solver.lifted_lp'):
        module = ModuleType(name)
        module.linprog = hooks.modules['scipy.optimize'].linprog
        module.lp = module.linprog
        monkeypatch.setitem(sys.modules, name, module)
        modules.append(module)
    consumer = ModuleType('_cpu_audit_preimported')
    for kind, (owner, name) in hooks.targets.items():
        setattr(consumer, kind, getattr(owner, name))
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)

    hooks.audit.install()
    for module in modules:
        assert module.linprog is hooks.modules['scipy.optimize'].linprog
        module.lp()
    for kind, (owner, name) in hooks.targets.items():
        assert getattr(consumer, kind) is getattr(owner, name)
        # Unbound method aliases still forward self normally.
        args = (hooks.instances[kind],) if kind in hooks.instances else ()
        getattr(consumer, kind)(*args)
    assert hooks.audit.COUNTS == dict(linprog=7, highs=1, ipopt=1, lanes=1, native=1)


def test_exit_report_has_exact_fields_and_flushes(hooks, monkeypatch):
    hooks.audit.install()
    hooks.invoke('native')
    printer = Mock()
    monkeypatch.setattr(hooks.audit, 'print', printer, raising=False)
    hooks.exits[0]()
    printer.assert_called_once_with(
        '[cpu-solver] linprog=0 highs=0 ipopt=0 lanes=0 native=1', flush=True)


def test_import_has_no_side_effects(hooks, capsys):
    assert hooks.exits == []
    assert hooks.audit.COUNTS == dict.fromkeys(KINDS, 0)
    assert capsys.readouterr().out == ''
    hooks.invoke('linprog')
    assert hooks.audit.COUNTS['linprog'] == 0


@pytest.mark.parametrize('missing', ['scipy.optimize', 'highspy'])
def test_missing_optional_backend_does_not_disable_other_hooks(hooks, monkeypatch, missing):
    def optional_import(name):
        if name == missing:
            raise ModuleNotFoundError(name)
        return hooks.modules[name]
    monkeypatch.setattr(hooks.audit, 'import_module', optional_import)
    hooks.audit.install(strict=True)
    for kind in ('ipopt', 'lanes', 'native'):
        with pytest.raises(RuntimeError, match=kind):
            hooks.invoke(kind)
    assert len(hooks.exits) == 1


def test_real_libraries_preimports_strict_and_atexit():
    # A fresh interpreter verifies real imports, extension method patchability,
    # existing instances, a real LP/IPOPT solve, and the actual exit registration.
    pytest.importorskip('scipy.optimize')
    pytest.importorskip('highspy')
    script = r'''
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linprog
import highspy
from nlp_solver.ipopt_solver import IPOPTSolver, CYIPOPT_AVAILABLE
from nlp_solver.lp_alternation import start
from nlp_solver import lp_alternation
from input_split.native_lane import NativeExecutor
import cpu_solver_audit as audit

solver = IPOPTSolver({'ipopt_max_iter': 30, 'constr_viol_tol': 1e-6})
h = highspy.Highs()
h.setOptionValue('output_flag', False)
executor = object.__new__(NativeExecutor)
executor.root = SimpleNamespace()
executor._submit = lambda *args: 'submitted'
lp_alternation.Lane = lambda *args: 'lane'

class Problem:
    n_vars, n_constraints = 1, 0
    lb, ub = np.array([0.]), np.array([1.])
    def objective(self, x): return float((x[0] - .25) ** 2)
    def constraints(self, x): return np.empty(0)
    def gradient(self, x): return np.array([2 * (x[0] - .25)])
    def jacobian(self, x): return np.empty((0, 1))

audit.install()
audit.install()
with ThreadPoolExecutor(max_workers=1) as pool:
    result = pool.submit(linprog, [1.], bounds=[(0., 1.)], method='highs').result()
assert result.success and result.fun == 0.
h.run()
assert start(None, None) == 'lane'
assert executor.submit(None) == 'submitted'
if CYIPOPT_AVAILABLE:
    result = solver._solve_single_cyipopt(Problem(), np.array([.5]), 3.)
    assert result.x_solution is not None, result.status
    assert abs(result.x_solution[0] - .25) < 1e-4, result
before = audit.COUNTS.copy()
assert before == dict(linprog=1, highs=1, ipopt=int(CYIPOPT_AVAILABLE), lanes=1, native=1), before

audit.install(strict=True)
audit.install(strict=False)
for kind, call in [
    ('linprog', lambda: linprog([1.])),
    ('highs', h.run),
    ('ipopt', lambda: solver._solve_single_cyipopt(None, None, None)),
    ('lanes', lambda: start(None, None)),
    ('native', lambda: executor.submit(None)),
]:
    try:
        call()
    except RuntimeError as error:
        assert kind in str(error)
    else:
        raise AssertionError(kind + ' was not blocked')
assert audit.COUNTS == {kind: count + 1 for kind, count in before.items()}
print('real hooks passed')
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=os.pathsep.join((str(VERIFIER), str(VERIFIER.parent))))
    result = subprocess.run([sys.executable, '-c', script], cwd=VERIFIER.parent,
                            env=env, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'real hooks passed' in result.stdout
    summaries = [line for line in result.stdout.splitlines() if line.startswith('[cpu-solver] ')]
    assert len(summaries) == 1
    assert summaries[0] in (
        '[cpu-solver] linprog=2 highs=2 ipopt=2 lanes=2 native=2',
        '[cpu-solver] linprog=2 highs=2 ipopt=1 lanes=2 native=2',
    )
