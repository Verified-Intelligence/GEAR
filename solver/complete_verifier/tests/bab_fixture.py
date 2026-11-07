"""Runs one benchmark instance through the production pipeline (nlp_example.run_problem, profile gear.yaml) on CPU with an
injected incumbent that nothing can replace, and parses the BaB log. Not a pytest module.

The BaB-loop tests drive the loop through here and not through production options. Every primal channel that could move the
incumbent is off (in-BaB ALM and IPOPT, the lifted B&B and its leaf IPOPT, the IPOPT solve at the QTB anchor) and so is the one
stage that stops on the clock (lifted OBBT), so the iteration counts and bounds depend on the code alone.

    python tests/bab_fixture.py BENCHMARK INSTANCE INCUMBENT [MAX_ITERATIONS]      (from complete_verifier; the log goes to stdout)
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.dirname(HERE)
BENCHMARKS = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "benchmarks")
PROFILE = os.path.join(CV, 'exp_configs', 'NLP_solver', 'gear.yaml')
OVERRIDES = {'solver__nlp__in_bab': False, 'solver__nlp__enable_ipopt_in_bab': False, 'solver__nlp__bab_interval': 10 ** 9,
             'solver__nlp__lifted_lp_bab_time': 0.0, 'solver__nlp__lifted_lp_leaf_ipopt': False,
             'solver__nlp__lifted_lp_obbt': False}


def _drive(bench, name, incumbent, max_iterations):
    sys.path.insert(0, CV)
    import torch
    import nlp_example
    from input_split import batch_branch_and_bound, nlp_in_bab

    build_config = nlp_example._build_config
    overrides = dict(OVERRIDES, **({'bab__max_iterations': max_iterations} if max_iterations else {}))
    nlp_example._build_config = lambda args: build_config(args).set(**overrides)
    # the solve at the QTB anchor offers its point as an incumbent; only the tangent point is wanted
    tangent_point = nlp_in_bab.compute_qtb_tangent_point
    nlp_in_bab.compute_qtb_tangent_point = lambda *a, **k: (tangent_point(*a, **k)[0], None)
    # the children whose key _raise_child_keys moved above the parent's bound
    raise_keys, raised = batch_branch_and_bound._raise_child_keys, [0]

    def counting_raise_keys(dm_lb, *args, **kwargs):
        keys = raise_keys(dm_lb, *args, **kwargs)
        raised[0] += int((keys != dm_lb).sum())
        return keys

    batch_branch_and_bound._raise_child_keys = counting_raise_keys
    with tempfile.NamedTemporaryFile(suffix='.pt') as pt:
        torch.save(torch.tensor(incumbent, dtype=torch.float64), pt.name)
        args = nlp_example._build_arg_parser().parse_args([
            '--problem_name', name, '--benchmark', bench, '--benchmark_root', BENCHMARKS, '--device', 'cpu',
            '--use_nlp', 'true', '--clip_type', 'complete', '--branching_method', 'sb', '--use_beta', 'False',
            '--override_timeout', '1200', '--batch_size', '1', '--enable_ipopt_in_bab', 'false', '--config', PROFILE,
            '--inject_primal_pt', pt.name])
        result = nlp_example.run_problem(args)
    print(f'key_raised={raised[0]}')
    print(f'problem_name={name} status={result.status}')


ITERATION = re.compile(r'^Iteration (\d+)\s*$')
BOUNDS = re.compile(r'Primal Solution tensor\(\[([^\]]+)\][^)]*\), Dual Solution (\S+)')
STATUS = re.compile(r'^problem_name=\S+ status=(\S+)')
RAISED = re.compile(r'^key_raised=(\d+)')


def run_full(bench, name, incumbent, max_iterations=None):
    """(status, [(iteration, P, D)], number of children whose key was raised) of the BaB loop at the injected incumbent: P and D
    as printed at the end of each iteration."""
    cmd = [sys.executable, os.path.join(HERE, 'bab_fixture.py'), bench, name, repr(float(incumbent))]
    out = subprocess.run(cmd + ([str(max_iterations)] if max_iterations else []), cwd=CV, capture_output=True, text=True,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
    assert out.returncode == 0, out.stderr[-2000:]
    status, records, raised, it = None, [], None, None
    for line in out.stdout.splitlines():
        if (m := ITERATION.match(line)):
            it = int(m.group(1))
        elif (m := BOUNDS.search(line)) and it is not None:
            records.append((it, float(m.group(1).split(',')[0]), float(m.group(2))))
        elif (m := RAISED.match(line)):
            raised = int(m.group(1))
        elif (m := STATUS.match(line)):
            status = m.group(1)
    return status, records, raised


def run(bench, name, incumbent, max_iterations=None):
    """(status, [(iteration, P, D)]) of run_full"""
    return run_full(bench, name, incumbent, max_iterations)[:2]


if __name__ == '__main__':
    _drive(sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else None)
