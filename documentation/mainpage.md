# FrictionSim2D

FrictionSim2D generates structured simulation folders and LAMMPS input files for atomistic friction studies on 2D materials.

Supported simulation models:

- AFM: tip-substrate-sheet geometry
- Sheet-on-sheet: N-layer stack (minimum 3 layers)

## What FrictionSim2D Produces

For each generated case, FrictionSim2D creates:

- `lammps/`: LAMMPS input scripts (`system.in`, `slide*.in`, and settings include files)
- `data/`: structure and combined data files
- `results/`: destination for simulation outputs and postprocessing
- `visuals/`: trajectory and visual outputs
- `provenance/`: copied input assets and `manifest.json`

Optional extras:

- `hpc/` submission scripts for PBS or SLURM
- AiiDA registration/submission/import workflows

## Quick Start (CLI)

Generate AFM simulations:

```bash
FrictionSim2D run afm afm_config.ini --output-dir ./simulation_output
```

Generate sheet-on-sheet simulations:

```bash
FrictionSim2D run sheetonsheet sheet_config.ini --output-dir ./simulation_output
```

Generate scheduler scripts for an existing simulation root:

```bash
FrictionSim2D hpc generate ./simulation_output/simulation_YYYYMMDD_HHMMSS --scheduler pbs
```

## Typical Workflow

1. Write a config file (`afm_config.ini` or `sheet_config.ini`).
2. Run `FrictionSim2D run ...` to generate simulation trees.
3. Optionally generate or use bundled HPC scripts.
4. Run LAMMPS jobs locally or on cluster.
5. Optionally import/query results with AiiDA commands.

## Command Families

- `run`: generate AFM or sheet-on-sheet simulations
- `settings`: inspect/init/reset global settings
- `hpc`: build scheduler scripts from generated simulations
- `aiida`: setup, submit, import, query, export archives
- `postprocess`: read and plot simulation results
- `db`: setup keys, upload/query shared database records, and run staged validation workflow
- `api`: serve REST API on top of the database

## Related Docs

- [installation.md](installation.md)
- [configuration_guide.md](configuration_guide.md)
- [commands.md](commands.md)
- [examples.md](examples.md)

For day-to-day runs, start with [configuration_guide.md](configuration_guide.md), then use [commands.md](commands.md) and [examples.md](examples.md).
