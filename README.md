# GEAR
Repo for GEAR: A GPU-Accelerated Global Solver for Nonlinear Programs via Linear Bound Propagation

Gap atlas (all 505 instances for GEAR and the 12 AMPL solvers, any subset selectable: certification under the paper's scoring, gap-vs-time curves, formulations, commands and logs; source in `site/`): https://verified-intelligence.github.io/GEAR/

## Logs

`logs/` holds the solver output behind the reported results: one archive per solver, one log per instance (505 each).

| Path | Content |
|---|---|
| `logs/<solver>.tar.xz` | final 180 s runs of the 12 AMPL baselines in `baselines/results/` |
| `logs/gear.tar.xz` | GEAR r7 (v406 release, commit 7b2e88ea0), 180 s, one NVIDIA RTX 5090 |
| `logs/configs/r7_full406.yaml` | the GEAR configuration of these runs |
| `logs/MANIFEST.tsv` | solver, family, instance, bytes and sha256 of every log |
| `logs/SHA256SUMS` | checksums of the archives, the configuration and the manifest |

- Archives unpack to `<solver>/<family>/<instance>.log`.
- Local paths are shortened: `benchmarks/` is the benchmark root, `runs/` the baseline run directory, `<gear>/` the GEAR source tree, `<env>/` the Python environment, `<root>` the working directory, `~` the home directory, `<host>` the machine name.
- The `Configurations:` block at the top of a GEAR log lists the argument defaults, not the settings in effect; those are in `logs/configs/r7_full406.yaml`.
