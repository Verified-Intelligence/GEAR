# GEAR solver

GEAR: A GPU-Accelerated Global Solver for Bounded Continuous Nonlinear Programs via Linear Bound Propagation.

GEAR computes a feasible point (primal bound P) and a certified lower bound (dual bound D) for a bounded nonlinear program. It reports a problem as solved when the returned point satisfies every constraint and bound to absolute tolerance 1e-6 and (P - D) / max(1, |P|, |D|) <= 1e-4. The bounds come from linear bound propagation (CROWN) through the model's computation graph on the GPU, combined with branch-and-bound on the input box.

This directory is a reduced copy of the alpha-beta-CROWN / auto_LiRPA tree that contains the code GEAR runs; the neural-network verification parts are removed. See `NOTICE` for licenses.

## Install

Python 3.11, PyTorch 2.8.0 with CUDA 12.8 wheels, an NVIDIA GPU with a driver for CUDA 12.8 (the solver also runs on the CPU with `--device cpu`). IPOPT is used for the local solves and comes with `cyipopt` from conda-forge.

```bash
conda create -y -p ./env -c conda-forge python=3.11 cyipopt
./env/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
./env/bin/python -m pip install -r solver/requirements.txt
export PYTHONPATH=$PWD/solver:$PWD/solver/complete_verifier PYTHONNOUSERSITE=1
```

(Paths are relative to the repository root.)

## Benchmarks

`benchmarks/gear-benchmarks.tar.xz` holds the 505 converted models; see `benchmarks/README.md`.

```bash
(cd benchmarks && sha256sum -c SHA256SUMS && tar -xJf gear-benchmarks.tar.xz)
```

The models were converted from the MINLPLib and GAMS World sources; the conversion tools are not included.

## Run one instance

CLI, from `solver/complete_verifier`:

```bash
python nlp_example.py --problem_name hs030 --benchmark gams_world --benchmark_root ../../benchmarks \
  --config exp_configs/NLP_solver/f3.yaml --device cuda --use_beta False --override_timeout 180 --batch_size 1
```

The last line is `problem_name=hs030 status=verified success=True`. Add `--result_json out.json` to write status, P, D, the incumbent and its strict re-check.

API:

```python
from pathlib import Path
import gear as gr
c = gr.compile(gr.read_legacy("hs071", root=Path("benchmarks").resolve(), family="gams_world"))
r = gr.solve(c, gr.Options(preset="gear-f0", device="cuda", time_limit=180))
print(r.status, r.primal_bound, r.dual_bound, r.replay.ok)
```

`gr.compile` also takes a `torch.nn.Module` with `box=`, `objective=`, `le=`, `eq=` (output indices); a worked example is in the tutorial on the project page.

For discrete dynamics, `gr.solve_dynamics` first solves the original graph and,
if needed, trains and refines a VA-PINN with reference-model checks and two
independent test sets. Original and surrogate certificates have separate scopes.
See [the dynamics API](docs/dynamics.md) and `examples/dynamics.py` for both paths.

## Default configuration

The default is the configuration `f3` (`complete_verifier/exp_configs/NLP_solver/f3.yaml`, preset `gear-f0`), the configuration frozen after the NeurIPS 2026 submission. `nlp_example.py` takes it with `--config`; the API preset `gear-f0` is the same configuration.

## Tests

```bash
cd solver/complete_verifier/tests && python -m pytest -m "not slow" -q
```

Some tests read the benchmarks from `benchmarks/` (or `$GEAR_BENCH`) and are skipped when it is missing.

## Citation

```bibtex
@inproceedings{zhou2026gear,
  title     = {{GEAR}: A {GPU}-Accelerated Global Solver for Bounded Continuous Nonlinear Programs via Linear Bound Propagation},
  author    = {Zhou, Duo and Chen, Hesun and Zhong, Xiangru and Hanasusanto, Grani A. and Zhang, Huan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
