# GEAR
Repo for GEAR: A GPU-Accelerated Global Solver for Bounded Continuous Nonlinear Programs via Linear Bound Propagation

Project page: https://verified-intelligence.github.io/GEAR/ (results, analysis and how to reproduce every number), with the gap atlas at
https://verified-intelligence.github.io/GEAR/atlas.html (all 505 instances for GEAR and the 12 AMPL solvers, any subset selectable:
certification under the paper's scoring, gap-vs-time curves, formulations, commands and logs). Source of both in `site/`.

## Contents

| Path | Content |
|---|---|
| `solver/` | the GEAR solver: command line (`complete_verifier/nlp_example.py`) and Python API (`complete_verifier/gear/`); BSD 3-Clause, see `solver/NOTICE` |
| `benchmarks/` | the 505 benchmark instances in the format GEAR reads, with manifest and checksums; see `benchmarks/README.md` |
| `logs/` | solver logs and per-instance results behind the paper's tables |
| `baselines/` | the 12 AMPL baseline solvers: models, configurations, harness and results |
| `site/` | source of the project page |

Installation, a quick start, the Python API and how to reproduce the paper's run are in the
[wiki](https://github.com/Verified-Intelligence/GEAR/wiki) and in `solver/README.md`.

## Logs

`logs/` holds the solver output behind the reported results: one archive per solver, one log per instance (505 each).

| Path | Content |
|---|---|
| `logs/<solver>.tar.xz` | final 180 s runs of the 12 AMPL baselines in `baselines/results/` (SCIP: `final_scip_pow1`) |
| `logs/gear.tar.xz` | GEAR, the final version evaluated in the paper, 180 s, one NVIDIA RTX 5090 |
| `logs/gear_results.tsv` | GEAR per instance: certified, status, P and D in the model's sense, relative gap, time |
| `logs/configs/int3p_crownmerge.yaml` | the GEAR configuration of these runs |
| `logs/MANIFEST.tsv` | solver, family, instance, bytes and sha256 of every log |
| `logs/SHA256SUMS` | checksums of the archives, the configuration, `gear_results.tsv` and the manifest |

- Archives unpack to `<solver>/<family>/<instance>.log`.
- Local paths are shortened: `benchmarks/` is the benchmark root, `runs/` the baseline run directory, `<gear>/` the GEAR source tree, `<env>/` the Python environment, `<root>` the working directory, `~` the home directory, `<host>` the machine name.
- The `Configurations:` block at the top of a GEAR log lists the argument defaults, not the settings in effect; those are in `logs/configs/int3p_crownmerge.yaml`.

## Scoring

A run is solved when its returned point satisfies every constraint and bound of the original model to absolute tolerance 1e-6 and its
primal value P and certified lower bound D satisfy (P - D) / max(1, |P|, |D|) <= 1e-4. The AMPL baselines are scored by
`baselines/harness/report.py` (see `baselines/README.md`); GEAR's points are replayed on the original model before they are returned.

```bash
# Table 1 of the paper from the scored results
awk -F'\t' 'NR>1 && (($2=="global") ? $6 : $10)==1 {n[$1]++} END {for (s in n) print s, n[s]}' baselines/results/per_instance_long.tsv
awk -F'\t' 'NR>1 && $4==1 {n++} END {print "gear", n}' logs/gear_results.tsv
```
