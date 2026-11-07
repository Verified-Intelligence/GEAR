"""Opt-in, process-wide audit of GEAR's CPU solver entry points.

Patched attributes:
* scipy.optimize.linprog
* highspy.Highs.run (when highspy is installed)
* nlp_solver.ipopt_solver.IPOPTSolver._solve_single_cyipopt
* nlp_solver.lp_alternation.start
* input_split.native_lane.NativeExecutor.submit

Loaded module attributes bound to these functions are replaced by identity,
including aliases from imports made before install(). GEAR's linprog sites are
input_split.{lifted_bound,nlp_in_bab} and
nlp_solver.{dual_ascent,exp_certificate,gp_certificate,lifted_lp}; their local
imports also resolve the patched scipy attribute on subsequent calls.

cyipopt.Problem.solve is immutable in the supported cyipopt build. The GEAR
entry counts each attempted cyipopt solve (including retries), before setup and
outside its exception handler, so strict errors propagate to its caller.
Child interpreters do not share the counters: lanes/native count the parent's
dispatches, which detect CPU work even in already-prewarmed native workers.
Install before starting solver work; retained local/closure references and
previously captured bound methods cannot be replaced through module attributes.
"""

import atexit
from functools import wraps
from importlib import import_module
import sys
import threading


COUNTS = dict(linprog=0, highs=0, ipopt=0, lanes=0, native=0)
_lock = threading.Lock()
_install_lock = threading.Lock()
_installed = False
_strict = False


def _counted(fn, kind):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with _lock:
            COUNTS[kind] += 1
            strict = _strict
        if strict:
            raise RuntimeError(f'[cpu-solver] strict mode blocked {kind}')
        return fn(*args, **kwargs)

    return wrapper


def _patch(owner, name, kind):
    original = getattr(owner, name)
    wrapper = _counted(original, kind)
    setattr(owner, name, wrapper)
    # A source-module patch alone misses `from ... import fn` done earlier.
    # Match identity, not names, so renamed imports are covered as well.
    for module in tuple(sys.modules.values()):
        namespace = vars(module) if module is not None else {}
        for alias, value in tuple(namespace.items()):
            if value is original:
                namespace[alias] = wrapper


def _report():
    with _lock:
        counts = COUNTS.copy()
    print('[cpu-solver] ' + ' '.join(f'{kind}={count}' for kind, count in counts.items()), flush=True)


def install(strict: bool = False):
    """Count CPU calls and print totals at exit; strict mode blocks each attempt.

    Repeated installation preserves counts, wrappers and the single exit hook.
    Passing strict=True upgrades an existing audit; later calls cannot disable it.
    Importing this module alone has no effect. Callers opt in explicitly (the
    GEAR_CPU_SOLVER_AUDIT environment switch belongs to the caller).
    """
    global _installed, _strict
    with _install_lock:
        with _lock:
            _strict = _strict or bool(strict)
        if _installed:
            return

        # Resolve GEAR modules before patching, including their import-time aliases.
        ipopt = import_module('nlp_solver.ipopt_solver')
        alternation = import_module('nlp_solver.lp_alternation')
        native = import_module('input_split.native_lane')
        targets = [(ipopt.IPOPTSolver, '_solve_single_cyipopt', 'ipopt'),
                   (alternation, 'start', 'lanes'),
                   (native.NativeExecutor, 'submit', 'native')]
        for module_name, attr, kind in [('scipy.optimize', 'linprog', 'linprog'),
                                        ('highspy', 'Highs', 'highs')]:
            try:
                module = import_module(module_name)
            except ImportError:
                continue
            targets.append((module.Highs, 'run', kind) if kind == 'highs'
                           else (module, attr, kind))
        for owner, name, kind in targets:
            _patch(owner, name, kind)

        atexit.register(_report)
        _installed = True
        print(f'[cpu-solver-audit] enabled strict={_strict}', flush=True)
