# GEAR benchmark: 505 bounded continuous nonlinear programs

This directory holds the 505 instances on which the paper evaluates GEAR and 12 AMPL solvers.

## Contents

| Family | Instances | Source collection |
|---|---|---|
| `gams_world` | 265 | GAMS World |
| `minlplib_linear` | 240 | MINLPLib |
| Total | 505 | |

Every instance has continuous variables only (`discrete_vars` is 0 for all 505 in `baselines/benchmark/manifest.tsv`) and a finite box. 501 instances are minimizations and 4 are maximizations. The number of variables ranges from 1 to 8991 and the number of constraints from 0 to 11330.

## Files

| File | Content |
|---|---|
| `gear-benchmarks.tar.xz` | the models: 1521 files, 987 MiB unpacked, 9.3 MB packed |
| `MANIFEST.tsv` | columns `family`, `path`, `bytes`, `sha256`: one row for each unpacked file |
| `instances.tsv` | columns `family`, `instance`: the 505 instances |
| `SHA256SUMS` | checksums of the archive, `MANIFEST.tsv` and `instances.tsv` |

```bash
(cd benchmarks && sha256sum -c SHA256SUMS && tar -xJf gear-benchmarks.tar.xz)
```

The archive unpacks to:

```
gams_world/
  mod_files_gams/            <name>_model.py, <name>_model_state.pt, __init__.py, modules.py
  vnnlib_gams/               <name>.vnnlib
minlplib_linear/
  py_minlplib_linear/        <name>_model.py, <name>_model_state.pt, __init__.py, modules.py
  vnnlib_minlplib_linear/    <name>.vnnlib
minlplib/
  mod_files/modules.py       the shared module library
```

One instance is three files:

| File | Content |
|---|---|
| `<name>_model.py` | A PyTorch module that maps a batch of points `x` to the objective and the constraint outputs |
| `<name>_model_state.pt` | The constants of the module |
| `<name>.vnnlib` | The variable bounds and the specification of the output rows |

Pass the `benchmarks/` directory as `--benchmark_root` to `nlp_example.py`, as `--root` to `gear.cli`, or as `root` to `gear.read_legacy`. Running an instance with `nlp_example.py` writes a `.vnnlib.compiled` cache next to its `.vnnlib` file.

The archive compresses 106:1 because the large `.pt` files store sparse matrices densely as float64 (for example `arki0006`: a 13.3M-entry constraint matrix with 1.4% nonzero entries). No two of the 505 `.pt` files are identical.

## Conversion from the AMPL models

The AMPL models are in `baselines/benchmark/<family>/<name>.mod`. The `.mod` files were written by GAMS Convert (the header of each file states the Convert run). The 12 AMPL baseline solvers ran these files.

The models were generated from the original AMPL and GAMS sources. The conversion follows the paper (appendix, dataset):

- Each instance is brought to the form: minimize `f_0(x)` subject to `f_i(x) <= 0`, `x in [l, u]`. Maximization objectives are negated in this form.
- Two-sided inequalities are split into paired one-sided inequalities. Equalities are encoded as paired inequalities with a fixed tolerance.
- Objective offsets, source variable bounds and model scaling are preserved.
- The module takes one vector input and returns the objective and all shifted constraints as one vector output; a feasibility residual `max_i f_i(x)` is appended as an additional output. Common subexpressions are shared when the parser exposes them.
- Solutions are reported in the sense of the original model. Every returned point is re-evaluated on the original model.

The conversion tools are not part of this repository. Running GEAR on the converted models needs no AMPL, GAMS or other solver license, and the files contain no license data.

## Per-instance metadata

`baselines/benchmark/manifest.tsv` has one row per instance:

| Column | Meaning |
|---|---|
| `family` | `gams_world` or `minlplib_linear` |
| `instance` | Instance name |
| `relative_model_path`, `sha256` | The AMPL model file and its checksum |
| `nvars`, `ncons` | Number of variables and constraints |
| `discrete_vars` | Number of discrete variables (0 for all instances) |
| `sense` | `minimize` or `maximize` |
| `functions` | Nonlinear functions that occur (for example `cos,exp,sqrt`; empty if none) |

`benchmarks/MANIFEST.tsv` lists the converted files with sizes and checksums (columns above); `benchmarks/instances.tsv` lists the instance keys.

Reference values. The repository does not contain a separate table of best-known objective values. For each instance, the objective value P and the certified bound D of the GEAR run in the paper are in `logs/gear_results.tsv` (columns `P`, `D`, `relative_gap`, in the sense of the model), and the objective value of each AMPL solver's returned point is in the column `P_ampl` of `baselines/results/per_instance_long.tsv`. 427 instances are solved by at least one global solver and 421 by GEAR under the scoring rule of the paper.

## Source collections and licenses

| Collection | Link | License |
|---|---|---|
| MINLPLib | https://www.minlplib.org | CC BY 4.0. Cite the MINLPLib authors when using the `minlplib_linear` family. |
| GAMS World | https://github.com/GAMS-dev/gamsworld | MIT License; Copyright (c) 2021 GAMS Software GmbH, GAMS Development Corp. and Andre Savitsky (ZabinskyEtAlLib: additionally Montaz Ali, Zelda Zabinsky, Charoenchai Khompatraporn). The MIT notice must stay with copies of the `gams_world` models. |

The code in `solver/` is BSD-3-Clause (`solver/NOTICE`) and the rest of the repository is Apache-2.0.

## Checksums

- `SHA256SUMS` in this directory covers the archive, `MANIFEST.tsv` and `instances.tsv`; verify with the command above.
- `MANIFEST.tsv` has the SHA-256 of every unpacked file. The `.vnnlib.compiled` caches written by runs are not listed.
- `baselines/benchmark/manifest.tsv` has the SHA-256 of every AMPL model.

## Citing

Cite the GEAR paper (NeurIPS 2026, https://neurips.cc/virtual/2026/loc/atlanta/poster/155708; BibTeX in the [wiki](https://github.com/Verified-Intelligence/GEAR/wiki/Citing)) and the source collections: MINLPLib (https://www.minlplib.org) and GAMS World (https://github.com/GAMS-dev/gamsworld).
