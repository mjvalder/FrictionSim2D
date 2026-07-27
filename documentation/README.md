# FrictionSim2D documentation

User documentation for FrictionSim2D, a package that generates LAMMPS-ready friction simulation inputs for 2D materials.

Read in this order if you are new; each document also stands alone.

| # | Document | Covers |
|---|---|---|
| 1 | [installation.md](installation.md) | Environment setup, optional extras, verification, troubleshooting |
| 2 | [overview.md](overview.md) | What the package does, the five simulation models, output layout, quick start |
| 3 | [configuration.md](configuration.md) | Writing `.ini` config files, sweep behaviour, worked examples |
| 4 | [commands.md](commands.md) | Complete CLI reference for every command group |
| 5 | [gui.md](gui.md) | Desktop application: configure, preview in 3D, generate |
| 6 | [examples.md](examples.md) | End-to-end workflows from generation to plots |
| 7 | [settings.md](settings.md) | Every `settings.yaml` section and its defaults |
| 8 | [pes_scan.md](pes_scan.md) | Static potential-energy-surface (corrugation-barrier) scans |
| 9 | [indent.md](indent.md) | Indentation runs: penetration on hold, adhesion on retract |
| 10 | [aiida.md](aiida.md) | AiiDA setup, submission, import, query, archive exchange |
| 11 | [hpc_jobs.md](hpc_jobs.md) | Scheduler script generation and two-phase submission |
| 12 | [provenance.md](provenance.md) | Provenance manifest model and file tracking internals |
| 13 | [python_api.md](python_api.md) | Driving the package from Python instead of the CLI |

If your goal is simply to generate and run simulations, documents 1 through 6 are
sufficient. Documents 12 and 13 matter for development and reproducibility work.

## Publication dataset

[`publication/`](publication/) holds the data and configuration files accompanying the
published dataset, DOI [10.82186/gs4dc-90a11](https://doi.org/10.82186/gs4dc-90a11):

- [publication/README.md](publication/README.md) — dataset overview, import instructions, and how to reproduce every publication figure
- [publication/plots_publication.json](publication/plots_publication.json) — the master plot configuration
