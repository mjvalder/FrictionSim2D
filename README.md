# FrictionSim2D

FrictionSim2D generates and manages LAMMPS friction simulation workflows for 2D materials.
It supports AFM and sheet-on-sheet setups, HPC script generation, and optional AiiDA provenance/submission workflows.

## Installation

FrictionSim2D is distributed as a Conda package for maximum reproducibility and minimal dependency conflicts.

### Option A – Base package (recommended for most users)

Install LAMMPS, Atomsk, and core dependencies:

```bash
conda install -c conda-forge frictionsim2d
conda activate frictionsim2d
```

Verify:
```bash
FrictionSim2D --help
lmp -h
atomsk --version
```

### Option B – With AiiDA (provenance tracking, HPC submission)

Install base package + AiiDA with bundled PostgreSQL and RabbitMQ:

```bash
conda install -c conda-forge frictionsim2d-aiida
conda activate frictionsim2d-aiida
```

Then bootstrap AiiDA and start services (one-time setup):

```bash
frictionsim2d-start-aiida
```

This:
- Starts RabbitMQ with automatic `consumer_timeout` patching (36000000 ms default)
- Starts PostgreSQL
- Creates your first AiiDA profile

Override RabbitMQ timeout if needed:
```bash
export RABBITMQ_CONSUMER_TIMEOUT_MS=72000000
frictionsim2d-start-aiida
```

Optional: install conda activate/deactivate hooks for automatic service lifecycle:
```bash
frictionsim2d-install-hooks
```

After this, PostgreSQL and RabbitMQ start/stop automatically with `conda activate/deactivate`.

Verify AiiDA setup:
```bash
verdi --version
verdi profile show
verdi daemon status
```

### Run from source

After installation, the `FrictionSim2D` command is available in your environment.

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
