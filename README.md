# FrictionSim2D

FrictionSim2D generates and manages LAMMPS friction simulation workflows for 2D materials.
It supports AFM and sheet-on-sheet setups, HPC script generation, and optional AiiDA provenance/submission workflows.

## Installation (Conda-only)

This project is intended to run from a Conda environment so that LAMMPS, Atomsk, and AiiDA service dependencies are consistent.

> pip installation is not the supported path in this repository documentation.

Two environment files are provided depending on whether you need AiiDA:

| File | Use case |
|------|----------|
| `conda/environment.yml` | Standard use – LAMMPS + all Python deps, **no AiiDA** |
| `conda/environment-aiida.yml` | AiiDA provenance + HPC daemon, requires Python 3.11 |

### Option A – Without AiiDA (recommended for most users)

```bash
conda env create -f conda/environment.yml
conda activate frictionsim2d
```

### Option B – With AiiDA (provenance tracking, HPC submission)

> **Requirements:** Python 3.11, RabbitMQ, PostgreSQL  
> (all bundled via `aiida-core.services` – no separate install needed)

```bash
conda env create -f conda/environment-aiida.yml
conda activate frictionsim2d-aiida
```

Then run the one-time setup script to start RabbitMQ and create the AiiDA profile:

```bash
export PYTHONPATH=$PWD
bash src/aiida/start_aiida.sh
```

### Run from source

From the repository root (either environment):

```bash
export PYTHONPATH=$PWD
python -m src.cli --help
```

Optional alias:

```bash
alias FrictionSim2D='python -m src.cli'
```

### Verify installation

```bash
# Both environments
FrictionSim2D --help
lmp -h
atomsk --version

# AiiDA environment only
verdi --version
verdi daemon status
```

## Quick start

Generate AFM simulations:

```bash
FrictionSim2D run afm examples/afm_config.ini --output-dir ./simulation_output
```

Generate sheet-on-sheet simulations:

```bash
FrictionSim2D run sheetonsheet examples/sheet_config.ini --output-dir ./simulation_output
```

Generate with AiiDA registration enabled:

```bash
FrictionSim2D run afm examples/afm_config.ini --aiida --output-dir ./simulation_output
```

Generate HPC scripts:

```bash
FrictionSim2D hpc generate ./simulation_output/simulation_YYYYMMDD_HHMMSS --scheduler pbs
```

## CLI overview

- `FrictionSim2D run afm ...`
- `FrictionSim2D run sheetonsheet ...`
- `FrictionSim2D hpc generate ...`
- `FrictionSim2D settings show|init|reset`
- `FrictionSim2D aiida status|setup|submit|import|query|export|import-archive|package`
- `FrictionSim2D db upload|query|stats|delete`

## Shared database

FrictionSim2D can upload results to and query from a shared PostgreSQL database, allowing users to share and compare simulation results.

### Configure connection

Set the connection via environment variables (or pass `--host`, `--user`, `--password` flags):

```bash
export FRICTION_DB_HOST=db.example.com
export FRICTION_DB_PORT=5432
export FRICTION_DB_NAME=frictionsim2d
export FRICTION_DB_USER=myuser
export FRICTION_DB_PASSWORD=secret
```

### Upload results

```bash
FrictionSim2D db upload ./simulation_output/afm_run/results --uploader alice
```

### Query results

```bash
# Print table to terminal
FrictionSim2D db query --material h-MoS2 --layers 1

# Save to CSV
FrictionSim2D db query --material h-WS2 --csv output.csv
```

### Database statistics

```bash
FrictionSim2D db stats
```

### Delete your own rows

```bash
FrictionSim2D db delete --uploader alice
```

## AiiDA notes

- Use `FrictionSim2D aiida setup` for first-time profile/computer/code setup.
- For secure HPC clusters, running AiiDA natively on the HPC environment is often more robust than local SSH transport.
- Use archive transfer when needed: `aiida export` on source environment and `aiida import-archive` on target.

## Output structure

Generated simulation roots typically contain:

- `afm/` or `sheetonsheet/` runs
- per-run `lammps/`, `data/`, `results/`, `visuals/`, `provenance/`
- optional `hpc/` submission scripts

## Documentation

- [documentation/mainpage.md](documentation/mainpage.md)
- [documentation/installation.md](documentation/installation.md)
- [documentation/essentials.md](documentation/essentials.md)
- [documentation/commands.md](documentation/commands.md)
- [documentation/examples.md](documentation/examples.md)
- [documentation/PROVENANCE_ARCHITECTURE.md](documentation/PROVENANCE_ARCHITECTURE.md)

## Development

Run tests:

```bash
pytest -v
```

Run lint (example):

```bash
pylint src/aiida
```
