"""The native public API and real solver, for both branches of the cascade."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('force_surrogate,level', [(False, 1), (True, 2)])
def test_discrete_control_example_end_to_end(force_surrogate, level):
    solver_root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(solver_root / 'examples/dynamics.py'), '--device', 'cpu']
    if force_surrogate:
        command.append('--force-surrogate')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=os.pathsep.join((str(solver_root), str(solver_root / 'complete_verifier'))))
    process = subprocess.run(command, cwd=solver_root.parent, env=env, capture_output=True, text=True, timeout=120)
    lines = [line for line in process.stdout.splitlines() if line.startswith('DYNAMICS_RESULT ')]
    assert process.returncode == 0 and len(lines) == 1, process.stdout[-6000:] + process.stderr[-3000:]
    result = json.loads(lines[0].split(' ', 1)[1])
    assert result['accepted'] and result['level'] == level
    assert result['certificate_scope'] == ('original' if level == 1 else 'surrogate')
    if level == 1:
        assert result['original']['status'] == 'OPTIMAL' and not result['rounds']
        assert result['original_validation']['passed']
    else:
        assert result['original']['status'] == 'NO_SOLUTION'
        last = result['rounds'][-1]
        assert last['optimization']['passed'] and all(m['passed'] for m in last['validation'])
        assert last['optimization']['reference_max_violation'] <= 1e-6
        assert last['optimization']['gap'] <= 1e-4
